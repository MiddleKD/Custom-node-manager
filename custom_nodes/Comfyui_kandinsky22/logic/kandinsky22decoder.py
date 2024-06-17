import dataclasses
import logging
import numpy as np
import torch
import torchvision.transforms.functional as F

from pathlib import Path
from PIL import Image
from typing import List, Optional, Callable, Dict, Tuple, Union

from comfy import model_management
from diffusers import KandinskyV22Pipeline, DDPMScheduler
from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2 import downscale_height_and_width
from diffusers.utils import numpy_to_pil
from diffusers.utils.torch_utils import randn_tensor

from .utils import get_vanilla_callback


logger = logging.getLogger()


@dataclasses.dataclass
class ImageLatents:
    movq_scale_factor: int
    init_latents: torch.Tensor = None
    noise_latents: torch.Tensor = None
    hint: torch.Tensor = None


'''
TOOD: add literals

NEAREST: Literal[0]
BOX: Literal[4]
BILINEAR: Literal[2]
LINEAR: Literal[2]
HAMMING: Literal[5]
BICUBIC: Literal[3]
CUBIC: Literal[3]
LANCZOS: Literal[1]
ANTIALIAS: Literal[1]
'''


def prepare_image(image: torch.Tensor, width: int = 512, height: int = 512):
    pil_image = numpy_to_pil(image.numpy())[0]
    pil_image = pil_image.resize((width, height), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr).unsqueeze(0)
    return image


def prepare_latents_on_img(image, movq, shape, decoder_info, seed):
    device: torch.device = model_management.get_torch_device()
    offload_device: torch.device = model_management.intermediate_device()
    movq.to(device)

    batch_size, height, width = shape
    movq_scale_factor, num_channels_latents, dtype = decoder_info
    generator = torch.Generator().manual_seed(seed)

    image = prepare_image(image, width, height)
    image = image.to(dtype=dtype, device=movq.device)

    latents = movq.encode(image)["latents"].repeat_interleave(batch_size, dim=0)
    noise = randn_tensor(latents.shape, generator=generator, dtype=dtype, device=latents.device)

    movq.to(offload_device)

    return ImageLatents(
        init_latents=latents,
        noise_latents=noise,
        movq_scale_factor=movq_scale_factor,
    )


def get_timesteps(scheduler, num_inference_steps, strength):
    t_start = max(int(num_inference_steps * (1.0 - strength)), 0)
    timesteps = scheduler.timesteps[t_start:]
    return timesteps, num_inference_steps - t_start


def prepare_latents(shape, decoder_info, seed):
    generator = torch.Generator().manual_seed(seed)

    movq_scale_factor, num_channels_latents, dtype = decoder_info
    batch_size, height, width = shape
    height, width = downscale_height_and_width(height, width, movq_scale_factor)
    shape = batch_size, num_channels_latents, height, width

    noise = randn_tensor(shape, generator=generator, dtype=dtype)
    return ImageLatents(
        noise_latents=noise,
        movq_scale_factor=movq_scale_factor,
    )


def combine_hint_latents(
        hint: torch.Tensor,
        latents: Union[torch.Tensor, ImageLatents]):
    if isinstance(latents, torch.Tensor):
        latents = ImageLatents(noise_latents=latents)
    img_size = latents.noise_latents.shape[-2:]
    img_size = np.array(img_size, dtype=np.uint32) * latents.movq_scale_factor
    img_size = img_size.tolist()

    if hint.shape[-2:] != img_size:
        hint = hint.permute((0, 3, 1, 2))
        hint = F.resize(hint, img_size)
        hint = hint.permute((0, 2, 3, 1))

    latents.hint = hint
    return latents


def prepare_added_cond_kwargs(
        unet,
        image_embeds,
        negative_image_embeds,
        hint,
        num_images_per_prompt,
        do_classifier_free_guidance):
    result = {}

    image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    negative_image_embeds = negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    
    if do_classifier_free_guidance:
        image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)
    result["image_embeds"] = image_embeds.to(dtype=unet.dtype, device=unet.device)

    if hint is not None:
        if hint.shape[1] > 3:
            hint = hint.permute((0, 3, 1, 2))
        hint = hint.repeat_interleave(num_images_per_prompt, dim=0)
        if do_classifier_free_guidance:
            hint = torch.cat([hint, hint], dim=0)
        result["hint"] = hint.to(dtype=unet.dtype, device=unet.device)

    return result

def decode(
        device: torch.device,
        decoder: Tuple,
        image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        negative_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        latents: torch.Tensor = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        init_latents: Optional[torch.Tensor] = None,
        strength: float = 1.0,
        hint: Optional[torch.Tensor] = None,
    ):
    num_images_per_prompt = latents.shape[0]
    do_classifier_free_guidance = guidance_scale > 1.0
    scheduler, unet = decoder

    added_cond_kwargs = prepare_added_cond_kwargs(
        unet, image_embeds, negative_image_embeds, hint, num_images_per_prompt, do_classifier_free_guidance
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    if init_latents is None:
        timesteps = scheduler.timesteps

        # create initial latent
        latents = latents * scheduler.init_noise_sigma
        latents = latents.to(device)
    elif strength > 0.0:
        timesteps, num_inference_steps = get_timesteps(scheduler, num_inference_steps, strength)
        latent_timestep = timesteps[:1]
        init_latents.to(device)
        noise_latents = latents.to(device)

        scheduler: DDPMScheduler
        latents = scheduler.add_noise(init_latents, noise_latents, latent_timestep)
    else:
        return init_latents

    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        noise_pred = unet(
            sample=latent_model_input,
            timestep=t,
            encoder_hidden_states=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            _, variance_pred_text = variance_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

        if not (
                hasattr(scheduler.config, "variance_type")
                and scheduler.config.variance_type in ["learned", "learned_range"]
        ):
            noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(
            noise_pred,
            t,
            latents,
            generator=generator,
        )[0]

        if callback_on_step_end is not None:
            callback_on_step_end(i, num_inference_steps)

    return latents


def unet_decode(
        decoder: Tuple,
        image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        negative_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        latents: Union[torch.FloatTensor, ImageLatents],
        seed: int,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        strength: float = 1.0,
    ):
    generator = torch.Generator().manual_seed(seed)
    device: torch.device = model_management.get_torch_device()
    offload_device: torch.device = model_management.intermediate_device()

    scheduler, unet = decoder
    unet.to(device)

    init_latents = None
    hint         = None

    if isinstance(latents, ImageLatents):
        init_latents = latents.init_latents
        noise_latents = latents.noise_latents
        hint = latents.hint
    else:
        noise_latents = latents

    result = decode(
        device,
        decoder,
        image_embeds,
        negative_image_embeds,
        noise_latents,
        num_inference_steps,
        guidance_scale,
        generator,
        callback_on_step_end=get_vanilla_callback(num_inference_steps),
        init_latents=init_latents,
        hint=hint,
        strength=strength
    )
    # TODO: offload effectively
    unet.to(offload_device)

    return result


def movq_decode(latents, movq):
    device: torch.device = model_management.get_torch_device()
    offload_device: torch.device = model_management.intermediate_device()
    movq.to(device)

    image = movq.decode(latents, force_not_quantize=True)["sample"]
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float()  # .numpy()
    # image = numpy_to_pil(image)

    movq.to(offload_device)

    return image


def load_decoder_kandinsky22(path: Path):
    pipeline: KandinskyV22Pipeline = KandinskyV22Pipeline.from_pretrained(
        path, torch_dtype=torch.float16
    )

    scheduler = pipeline.components['scheduler']
    unet = pipeline.components['unet']
    movq = pipeline.components['movq']

    num_channels_latents = movq.config.latent_channels  # unet.config.in_channels
    movq_scale_factor = 2 ** (len(movq.config.block_out_channels) - 1)

    return \
        movq, \
        (scheduler, unet), \
        (movq_scale_factor, num_channels_latents, unet.dtype)

