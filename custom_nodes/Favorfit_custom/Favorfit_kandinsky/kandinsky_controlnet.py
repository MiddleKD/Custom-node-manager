import torch
import torch.nn.functional as F
from pathlib import Path
from Favorfit_kandinsky.model.controlnet import ControlNetModel

def load_kandinsky_controlnet(path: Path):
    
    controlnet = ControlNetModel.from_pretrained(
        path, torch_dtype=torch.float16
    ).eval()

    return controlnet
