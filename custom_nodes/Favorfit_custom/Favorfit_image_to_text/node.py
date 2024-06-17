import os, torch
from glob import glob
from torchvision import transforms
from Favorfit_image_to_text import clip_image_to_text as image_to_text
from Favorfit_utils.utils import get_model_list
from folder_paths import models_dir

class FavorfitImageToText:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "dir_name": (get_model_list("Favorfit_custom/image_to_text", get_dir=True),)
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "image_to_text"

    OUTPUT_NODE = False

    CATEGORY = "Favorfit_custom"

    def image_to_text(self, image, dir_name):
        dir_name = os.path.join(models_dir, "Favorfit_custom/image_to_text", dir_name)

        # image: [batch, height, width, channel]
        batch_size = image.shape[0]
        processed_images = []

        for i in range(batch_size):
            image_tensor = image[i]
            image_tensor = torch.clamp((image_tensor + 1.0) / 2.0, min=0.0, max=1.0)

            # 텐서를 [높이, 너비, 채널] 형태로 변경하고 정규화 역변환 적용
            image_tensor = image_tensor.permute(2, 0, 1)  # [height, width, channel] -> [channel, height, width]

            # 텐서를 PIL 이미지로 변환
            image_pil = transforms.ToPILImage()(image_tensor)
            
            # 모델 호출
            image_to_text_model = image_to_text.load_interrogator(dir_name, device="cuda")
            caption = image_to_text.inference(image_pil, image_to_text_model)
        
        del image_to_text_model

        return (caption,)
