import os
import folder_paths
from pathlib import Path
from comfy import model_management
import custom_nodes.Comfyui_kandinsky22.logic.kandinsky22decoder as original_kandinsky_decoder
from .kandinsky_controlnet import load_kandinsky_controlnet
from .kandinsky_advanced import (decode as hijack_decode,
                                 movq_encode,
                                 ImageLatentsWithAdvanced)
from ..Favorfit_utils.utils import resize


class KandinskyControlnetLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet_name": ([os.path.dirname(cur) for cur in folder_paths.get_filename_list("controlnet") if "kandinsky" in cur], ),
            }
        }

    RETURN_TYPES = ("CONTROL_NET_KANDINSKY",)

    FUNCTION = "load_controlnet"
    CATEGORY = "Favorfit_custom"

    def load_controlnet(self, controlnet_name):
        ckpt = os.path.join(folder_paths.get_folder_paths("controlnet")[0], controlnet_name)
        ckpt_pth = Path(ckpt)
        return (load_kandinsky_controlnet(ckpt_pth), )

class KandinskyControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latents": ("LATENT", ),
                             "control_net": ("CONTROL_NET_KANDINSKY", ),
                             "control_image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_kandinsky_controlnet"

    CATEGORY = "Favorfit_custom"

    def apply_kandinsky_controlnet(self, latents, control_net, control_image, strength):
        
        if strength == 0:
            return (latents, )
        if isinstance(latents, ImageLatentsWithAdvanced):
            latents.control_image.append(control_image)
            latents.controlnet_model.apprnd(control_net)
            latents.strength.append(strength)
        else:
            latents = ImageLatentsWithAdvanced(**vars(latents), 
                                              control_image = [control_image],
                                              controlnet_model = [control_net],
                                              strength = [strength])

        return (latents, )
    
class KandinskyControlUnetDecoder:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder": ("DECODER", ),
                "latents": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "image_embeds": ("PRIOR_LATENT", ),
                "negative_image_embeds": ("PRIOR_LATENT", ),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT", )

    FUNCTION = "run"
    CATEGORY = "Favorfit_custom"

    def run(self, decoder, image_embeds, negative_image_embeds, latents, num_inference_steps, guidance_scale, seed, strength):
        
        original_kandinsky_decoder.decode = hijack_decode
        latents = original_kandinsky_decoder.unet_decode(
            decoder, image_embeds, negative_image_embeds, latents,
            seed, num_inference_steps, guidance_scale, strength
        )
        return (latents,)
    
class KandinskyImageInject:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latents": ("LATENT", ),
                             "inject_image_embed": ("LATENT", ),
                             "inject_mask": ("MASK",),
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_kandinsky_image_inject"

    CATEGORY = "Favorfit_custom"

    def apply_kandinsky_image_inject(self, latents, inject_image_embed, inject_mask):
        b, c, h, w = inject_image_embed.shape
        if len(inject_mask.shape) != 4:
            inject_mask = inject_mask.unsqueeze(0)
        inject_mask = resize(inject_mask, h, w).to(device=model_management.get_torch_device(), dtype=inject_image_embed.dtype)

        if isinstance(latents, ImageLatentsWithAdvanced):
            latents.inject_image = inject_image_embed
            latents.inject_mask = inject_mask
        else:
            latents = ImageLatentsWithAdvanced(**vars(latents), 
                                              inject_image = inject_image_embed,
                                              inject_mask = inject_mask)
        
        return (latents, )

class Kandinsky22MovqEncoder:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "movq": ("MOVQ",),
                "images": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("LATENT", )

    FUNCTION = "encode"
    CATEGORY = "Favorfit_custom"

    def encode(self, images, movq):
        latents = movq_encode(images, movq)
        return (latents,)
