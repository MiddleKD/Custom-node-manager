import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
import numpy as np

from model.inspyrenet import InSPyReNet_SwinB, InSPyReNet_Res2Net50

class dynamic_resize:
    def __init__(self, L=1280): 
        self.L = L
                    
    def __call__(self, img):
        size = list(img.size)
        if (size[0] >= size[1]) and size[1] > self.L: 
            size[0] = size[0] / (size[1] / self.L)
            size[1] = self.L
        elif (size[1] > size[0]) and size[0] > self.L:
            size[1] = size[1] / (size[0] / self.L)
            size[0] = self.L
        size = (int(round(size[0] / 32)) * 32, int(round(size[1] / 32)) * 32)
    
        return img.resize(size, Image.BILINEAR)

def call_model(ckpt, device):
    class RemoveBackGround(nn.Module):
        def __init__(self, model_path, device=None, types="map"):
            super().__init__()
            backbone = "swinB"
            self.meta = {'base_size': [1024, 1024],
                        'threshold': None,
                        'ckpt_name': model_path,
                        'resize': dynamic_resize(L=1280)}
            if device is not None:
                self.device = device
            else:
                self.device = "cpu"

            if backbone == "swinB":
                self.model = InSPyReNet_SwinB(depth=64, **self.meta)
            elif backbone == "resnet":
                self.model = InSPyReNet_Res2Net50(depth=64, pretrained=False, **self.meta)
            else:
                print("No such model backbone")
                raise AssertionError

            self.model.eval()
            self.model.load_state_dict(torch.load(self.meta["ckpt_name"], map_location="cpu"), strict=True)
            self.model = self.model.to(self.device)

            self.transform = transforms.Compose([dynamic_resize(),
                                                transforms.ToTensor(),])
            
            self.types = types

        def forward(self, img):
            shape = img.size[::-1]
            x = self.transform(img)
            x = x.unsqueeze(0)
            x = x.to(self.device)

            with torch.no_grad():
                pred = self.model(x)
            
            pred = F.interpolate(pred, shape, mode="bilinear", align_corners=True)
            pred = pred.data.cpu()
            pred = pred.numpy().squeeze()

            img = np.array(img)

            if self.types == 'map':
                img = (np.stack([pred] * 3, axis=-1) * 255)

            return img.astype(np.uint8)
    remover = RemoveBackGround(model_path=ckpt, device=device)
    return remover


def inference(img_pil, model):
    result = model(img_pil)
    return Image.fromarray(result)


def main_call(model_path, root_dir, save_dir, device="cpu"):
    from glob import glob
    from tqdm import tqdm
    
    model = call_model(model_path, device)

    fns = glob(os.path.join(root_dir, "*"))
    
    for fn in tqdm(fns, total=len(fns)):
        img_pil = Image.open(fn)
        result = inference(img_pil, model)
        result.save(os.path.join(save_dir, os.path.basename(fn)))


if __name__ == "__main__":
    main_call(model_path="/home/mlfavorfit/Desktop/lib_link/favorfit/kjg/0_model_weights/remove_bg/remove_bg.pth",
              root_dir="/media/mlfavorfit/sdb/cat_toy/images", 
              save_dir="/media/mlfavorfit/sdb/cat_toy/images2", 
              device="cuda")
