import os
from Favorfit_utils.utils import (make_outpaint_condition, 
                                  get_embed_ts_file,
                                  resize_diffusion_available_size,
                                  embed_ts_dir)

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
