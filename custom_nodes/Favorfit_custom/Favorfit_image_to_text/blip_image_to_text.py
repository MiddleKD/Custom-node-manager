import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder


def transform_pil(pil_img, image_size, device):
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(pil_img).unsqueeze(0).to(device)
    return image


def load_interrogator(model_path, vit="base", device="cpu"):
    model = blip_decoder(pretrained=model_path, image_size=384, vit=vit)
    model.eval()
    model = model.to(device)
    return model


def inference(img_pil, model):
    img_ts = transform_pil(img_pil, image_size=384, device=next(model.parameters()).device)
    with torch.no_grad():
        caption = model.generate(img_ts, sample=True, top_p=0.9, max_length=20, min_length=5)[0]
    return caption


def main_call(model_path, root_dir, save_jsonl_path, device="cpu"):
    import os
    import json
    from PIL import Image
    from glob import glob
    from tqdm import tqdm
    
    model = load_interrogator(model_path, device=device)

    fns = glob(os.path.join(root_dir, "*"))
    
    result_list = []
    for fn in tqdm(fns, total=len(fns)):
        img_pil = Image.open(fn)
        caption = inference(img_pil, model)
        result_list.append({"image":os.path.basename(fn), "text":caption})

    with open(save_jsonl_path, 'w') as jsonl_file:
        for data in result_list:
            json_line = json.dumps(data)
            jsonl_file.write(json_line + '\n')


if __name__ == "__main__":
    # from PIL import Image
    # img_pil = Image.open("/home/mlfavorfit/lib/favorfit/kjg/Favorfit_backoffice/Favorfit_diffusion/images/tv.png").convert('RGB')
    
    # model = load_interrogator("./models/ckpt/model_base_caption_capfilt_large.pth", device="cuda")
    # result = inference(img_pil, model)
    # print(result)

    main_call("./models/ckpt/model_base_caption_capfilt_large.pth", "/media/mlfavorfit/sdb/contolnet_dataset/control_net_train_shuffle/images", "./temp.jsonl", device="cuda")
