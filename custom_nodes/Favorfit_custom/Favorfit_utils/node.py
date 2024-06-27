import os
from comfy.model_patcher import ModelPatcher
from comfy import model_management
from Favorfit_utils.utils import (make_outpaint_condition, 
                                  get_embed_ts_file,
                                  resize_diffusion_available_size,
                                  embed_ts_dir,
                                  resize)

class FavorfitMakeOutpaintCondition:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "make_outpaint_cond"

    OUTPUT_NODE = False

    CATEGORY = "Favorfit_custom"

    def make_outpaint_cond(self, image, mask):
        if len(mask.shape) != 4:
            mask = mask.unsqueeze(0)
        outpaint_condition = make_outpaint_condition(image, mask)
        return outpaint_condition

class FavorfitLoadEmbedTensor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_name": ([cur for cur in os.listdir(embed_ts_dir)],),
            },
        }

    RETURN_TYPES = ("PRIOR_LATENT",)
    RETURN_NAMES = ("embed",)

    FUNCTION = "load_embed_ts"

    OUTPUT_NODE = False

    CATEGORY = "Favorfit_custom"

    def load_embed_ts(self, file_name):
        embed_ts = get_embed_ts_file(os.path.join(embed_ts_dir, file_name))
        return (embed_ts,)

class FavorfitResizeDiffusionAvailable:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_size": ("INT", {"default": -1}),
                "max_size": ("INT", {"default": 2048}),
                "min_size": ("INT", {"default": 512}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "height", "width")

    FUNCTION = "resize_diffusion_available_size"

    OUTPUT_NODE = False

    CATEGORY = "Favorfit_custom"

    def resize_diffusion_available_size(self, image, target_size, max_size, min_size):
        if target_size == -1: target_size = None
        resized_image_tensor = resize_diffusion_available_size(image, target_size, max_size, min_size)
        b, height, width, c = resized_image_tensor.shape
        return (resized_image_tensor, height, width)

class ApplyImageInject:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model":("MODEL", ),
                             "latents": ("LATENT", ),
                             "inject_image_embed": ("LATENT", ),
                             "inject_mask": ("MASK",),
                             "start_sigma": ("FLOAT", {"default": 15.0}),
                             "end_sigma": ("FLOAT", {"default": 0.0})
                             }}
    RETURN_TYPES = ("MODEL", "LATENT",)
    FUNCTION = "apply_image_inject"

    CATEGORY = "Favorfit_custom"

    def apply_image_inject(self, model, latents, inject_image_embed, inject_mask, start_sigma, end_sigma):
        if isinstance(inject_image_embed, dict):
            inject_image_embed = inject_image_embed["samples"]
        b, c, h, w = inject_image_embed.shape
        if len(inject_mask.shape) != 4:
            inject_mask = inject_mask.unsqueeze(0)
        
        inject_image_embed = inject_image_embed.to(device=model_management.get_torch_device(), dtype=inject_image_embed.dtype)
        inject_mask = resize(inject_mask, h, w).to(device=model_management.get_torch_device(), dtype=inject_image_embed.dtype)
        
        latents["samples"] = inject_image_embed
        latents["noise_mask"] = inject_mask
        if hasattr(model, "model_options"):
            model.model_options["is_image_inject"] = {"start_sigma":start_sigma, "end_sigma":end_sigma}

        return (model, latents, )

class ResetModelPatcherCalculateWeight:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model":("MODEL", ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "reset_moodelpatcher_weight"

    CATEGORY = "Favorfit_custom"

    def reset_moodelpatcher_weight(self, model:ModelPatcher):

        if hasattr(model, "original_calculate_weight"):
            model.calculate_weight = ModelPatcher.original_calculate_weight
            ModelPatcher.calculate_weight = ModelPatcher.original_calculate_weight
        
        return (model, )