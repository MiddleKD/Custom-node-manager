import os, torch
from torchvision import transforms
from Favorfit_remove_bg import inference as remove_bg
from Favorfit_utils.utils import get_model_list
from folder_paths import models_dir


class FavorfitRemoveBg:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "ckpt": (get_model_list("Favorfit_custom"),)
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    FUNCTION = "remove_background"

    OUTPUT_NODE = False

    CATEGORY = "Favorfit_custom"

    def remove_background(self, image, ckpt):
        ckpt = os.path.join(models_dir, "Favorfit_custom", ckpt)

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
            rmbg_model = remove_bg.call_model(ckpt, device="cuda")
            mask = remove_bg.inference(image_pil, rmbg_model).convert("L")
            
            # PIL 이미지를 다시 텐서로 변환
            mask_tensor = transforms.ToTensor()(mask)

            # 텐서 형태를 [channel, height, width] -> [height, width, channel]로 변경
            # mask_tensor = mask_tensor.permute(1, 2, 0)

            processed_images.append(mask_tensor)
        
        del rmbg_model

        # 리스트를 다시 텐서로 결합
        mask = torch.stack(processed_images, dim=0)
        return (mask[0],)
