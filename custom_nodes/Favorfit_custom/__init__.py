import sys, os
sys.path.append(os.path.dirname(__file__))
from .Favorfit_remove_bg.node import FavorfitRemoveBg
from .Favorfit_image_to_text.node import FavorfitImageToText
from .Favorfit_utils.node import (FavorfitMakeOutpaintCondition,
                                  FavorfitLoadEmbedTensor,
                                  FavorfitResizeDiffusionAvailable,
                                  ApplyImageInject)
from .Favorfit_kandinsky.nodes import (KandinskyControlnetLoader,
                                       KandinskyControlNetApply,
                                       KandinskyControlUnetDecoder,
                                       KandinskyImageInject,
                                       Kandinsky22MovqEncoder)

NODE_CLASS_MAPPINGS = {
    "FavorfitRemoveBg": FavorfitRemoveBg,
    "FavorfitImageToText": FavorfitImageToText,
    "FavorfitMakeOutpaintCondition": FavorfitMakeOutpaintCondition,
    "KandinskyControlnetLoader": KandinskyControlnetLoader,
    "KandinskyControlNetApply": KandinskyControlNetApply,
    "KandinskyControlUnetDecoder": KandinskyControlUnetDecoder,
    "KandinskyImageInject":KandinskyImageInject,
    "Kandinsky22MovqEncoder":Kandinsky22MovqEncoder,
    "FavorfitLoadEmbedTensor": FavorfitLoadEmbedTensor,
    "FavorfitResizeDiffusionAvailable": FavorfitResizeDiffusionAvailable,
    "ApplyImageInject": ApplyImageInject,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FavorfitRemoveBg": "Favorfit Remove Background",
    "FavorfitImageToText": "Favorfit Image To Text",
    "FavorfitMakeOutpaintCondition": "Favorfit Make Outpaint Condition",
    "KandinskyControlnetLoader": "Kandinsky Controlnet Loader",
    "KandinskyControlNetApply": "Kandinsky Controlnet Apply",
    "KandinskyControlUnetDecoder": "Kandinsky Control Unet Decoder",
    "KandinskyImageInject":"Kandinsky Image Inject Apply",
    "Kandinsky22MovqEncoder":"Kandinsky Movq Encoder",
    "FavorfitLoadEmbedTensor": "Favorfit Load Embed Tensor",
    "FavorfitResizeDiffusionAvailable": "Favorfit Resize Dif Avail",
    "ApplyImageInject": "Favorfit Image Inject Ksampler",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
