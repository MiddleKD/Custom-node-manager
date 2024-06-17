# from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random
from PIL import Image

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler



input_image = Image.open("/home/mlfavorfit/Downloads/PART_03/02_product/sample_img/1c317974a59844b0b9844217a59621bb.jpg")
num_samples = 2
seed = -1



apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_v11p_sd15_softedge.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

with torch.no_grad():
    img = resize_image(HWC3(input_image))
    H, W, C = img.shape

    detected_map = apply_canny(img, 100, 200)
    detected_map = HWC3(detected_map)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusion=True)
    
    print(model)