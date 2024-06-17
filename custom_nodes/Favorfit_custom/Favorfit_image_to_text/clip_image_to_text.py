from PIL import Image
from torch import nn
from my_clip_interrogator import Config, Interrogator
from keyword_utils import post_process_text, remove_color_from_text

class InterrogatorWrapper(nn.Module):
    def __init__(self, model_path, caption_model_name="blip-base", device="cpu"):
        super().__init__()
        self.clip_interrogator = Interrogator(
        Config(caption_model_name=caption_model_name, 
               clip_model_name="ViT-L-14/openai", 
               cache_path=model_path, 
               clip_model_path=model_path,
               quiet=True,
               device=device)
        )
    
    def forward(self, img_pil, mode="fast"):
        if mode=="fast":
            result = self.clip_interrogator.interrogate_fast(img_pil)
        elif mode=="classic":
            result = self.clip_interrogator.interrogate_classic(img_pil)
        elif mode=="negative":
            result = self.clip_interrogator.interrogate_negative(img_pil)
        elif mode == "simple":
            result = self.clip_interrogator.interrogate_simple(img_pil)
            result = post_process_text(result)
        else:
            result = self.clip_interrogator.interrogate(img_pil)
        
        return result


def load_interrogator(model_path, caption_model_name="blip-base", device="cpu"):
    clip_intrerrogator = InterrogatorWrapper(
        model_path=model_path,
        caption_model_name=caption_model_name, 
        device=device
    )
    return clip_intrerrogator


def inference(img_pil, model, mode="fast", remove_color=False):
    result = model(img_pil, mode=mode)

    if remove_color == True:
        result = remove_color_from_text(result)

    return result
    

if __name__ == "__main__":
    img_pil = Image.open("/media/mlfavorfit/sdb/contolnet_dataset/val/158_Chane.jpg").convert('RGB')
    
    model = load_interrogator("/home/mlfavorfit/lib/favorfit/kjg/0_model_weights/image_to_text/clip", caption_model_name="blip-base", device="cuda") 
    result = inference(img_pil=img_pil, model=model, mode="simple", remove_color=True)
    print(result)
