import dataclasses
import torch
import torch.nn.functional as F

from Favorfit_kandinsky.model.controlnet import ControlNetModel
from typing import List, Optional, Callable, Dict, Tuple, Union
from diffusers import DDPMScheduler
from comfy import model_management
from custom_nodes.Comfyui_kandinsky22.logic.kandinsky22decoder import (
    prepare_added_cond_kwargs,
    get_timesteps,   
)


@dataclasses.dataclass
class ImageLatentsWithAdvanced:
    movq_scale_factor: int
    init_latents: torch.Tensor = None
    noise_latents: torch.Tensor = None
    hint: torch.Tensor = None
    control_image: torch.Tensor = None
    controlnet_model: ControlNetModel = None
    inject_image: torch.Tensor = None
    inject_mask: torch.Tensor = None
    strength: float = None

def movq_encode(images, movq):
    device: torch.device = model_management.get_torch_device()
    offload_device: torch.device = model_management.intermediate_device()
    movq.to(device)

    images = (images*2-1).permute(0,3,1,2).to(device=movq.device, dtype=torch.float16)
    latents = movq.encode(images)["latents"]
    movq.to(offload_device)

    return latents


def decode(
        device: torch.device,
        decoder: Tuple,
        image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        negative_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        latents: Union[torch.Tensor, ImageLatentsWithAdvanced] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        init_latents: Optional[torch.Tensor] = None,
        strength: float = 1.0,
        hint: Optional[torch.Tensor] = None,
    ):

    is_control = False; is_image_inject = False
    if isinstance(latents, ImageLatentsWithAdvanced):
        init_latents = latents.init_latents
        hint = latents.hint
        noise = latents.noise_latents.to(device)

        if getattr(latents, "control_image", None) is not None:
            control_images = [(cur*2-1).permute(0,3,1,2).to(device) for cur in latents.control_image]
            control_models = [cur.to(device) for cur in latents.controlnet_model]
            control_strengths = latents.strength
            is_control = True
        
        if getattr(latents, "inject_image", None) is not None:
            inject_image = latents.inject_image
            inject_mask = latents.inject_mask
            is_image_inject = True
        
        latents = noise

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
    
    if is_control == True and do_classifier_free_guidance:
        control_images = [cur.repeat(2,1,1,1) for cur in control_images]

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16):
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            down_block_res_samples = None
            mid_block_res_sample = None
            if is_control == True:
                for control_model, control_image, control_strength in zip(control_models, control_images, control_strengths):
                    down_block_res_samples, mid_block_res_sample = control_model(
                        latent_model_input,
                        timestep=t,
                        encoder_hidden_states=None,
                        controlnet_cond=control_image,
                        conditioning_scale=control_strength,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )

            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
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

            if is_image_inject == True:
                if i < len(timesteps)-1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = scheduler.add_noise(
                        inject_image, noise, torch.tensor([noise_timestep])
                    ).to(dtype=latents.dtype)
                    latents = inject_mask * init_latents_proper + (1 - inject_mask) * latents

            if callback_on_step_end is not None:
                callback_on_step_end(i, num_inference_steps)

    return latents
