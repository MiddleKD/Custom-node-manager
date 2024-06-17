import os
import folder_paths
from typing import List

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from comfy import model_management

def make_outpaint_condition(image, mask):
    mask = mask.permute(0,2,3,1)
    black_image = torch.zeros_like(image)
    composed_output = black_image * (1-mask) + image * (mask[:,None])

    return composed_output

def tensor_to_pil_image(tensor):

    if len(tensor.shape) == 4:
        tensor = tensor[0]

    # Convert the tensor to a numpy array
    array = tensor.permute(2,1,0).numpy()
    # Scale the array values to the range [0, 255]
    array = (array * 255).astype(np.uint8)

    # Convert the numpy array to a PIL Image
    image = Image.fromarray(array)
    # image.show()

    return image

embed_ts_dir = os.path.join(folder_paths.models_dir, "embed_ts")
def get_embed_ts_file(path):
    embed_ts = torch.tensor(np.load(path), device=model_management.get_torch_device())
    
    return embed_ts

def resize(image_tensor:torch.Tensor, height, width):
    transform_resize = transforms.Resize((height, width))
    resized_tensor = transform_resize(image_tensor)
    return resized_tensor

def resize_store_ratio(image_tensor:torch.Tensor, target_size:int=None, max_size:int=1920, min_size:int=512):
    # Check if the input tensor is a batch of images
    if len(image_tensor.shape) == 4:  # batch_size x channels x height x width
        batch_size, channels, height, width = image_tensor.shape
    else:  # channels x height x width
        batch_size = None
        channels, height, width = image_tensor.shape

    if target_size is None:
        if max(width, height) > max_size:
            target_size = max_size
        else:
            target_size = min_size
    
    if width > height:
        new_width = target_size
        new_height = int((height / width) * new_width)
    else:
        new_height = target_size
        new_width = int((width / height) * new_height)

    # Resizing the image tensor
    transform_resize = transforms.Resize((new_height, new_width))
    resized_tensor = transform_resize(image_tensor)

    return resized_tensor

def resize_diffusion_available_size(image_tensor, target_size=None, max_size=1920, min_size=512):
    # Check if the input tensor is a batch of images
    if image_tensor.shape[-1] == 3:
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.permute(0,3,1,2)
        else:
            image_tensor = image_tensor.permute(2,0,1)

    if len(image_tensor.shape) == 4:  # batch_size x channels x height x width
        batch_size, channels, height, width = image_tensor.shape
    else:  # channels x height x width
        batch_size = None
        channels, height, width = image_tensor.shape

    # Resize maintaining aspect ratio
    if target_size is not None or max(width, height) > max_size or max(width, height) < min_size:
        image_tensor = resize_store_ratio(image_tensor, 
                                          target_size=target_size, 
                                          max_size=max_size, 
                                          min_size=min_size)
    
    # Calculate new dimensions as multiples of 64
    if batch_size:
        new_width = (image_tensor.size(3) // 64) * 64
        new_height = (image_tensor.size(2) // 64) * 64
    else:
        new_width = (image_tensor.size(2) // 64) * 64
        new_height = (image_tensor.size(1) // 64) * 64
    
    # Resize tensor to new dimensions
    transform_resize = transforms.Resize((new_height, new_width))
    resized_tensor = transform_resize(image_tensor)
    
    if len(resized_tensor.shape) == 4:
        resized_tensor = resized_tensor.permute(0,2,3,1)
    else:
        resized_tensor = resized_tensor.permute(1,2,0)

    return resized_tensor

def center_crop_and_resize(input_image, target_size=(512, 512), output_type='ts'):

    if isinstance(input_image, Image.Image):
        # Convert PIL image to tensor
        input_image = transforms.ToTensor()(input_image)
        input_image = input_image.unsqueeze(0)  # Add batch dimension

    if input_image.ndimension() == 4:
        if input_image.size(1) != 3:
            # If input is in format [batch, height, width, channels], permute to [batch, channels, height, width]
            input_image = input_image.permute(0, 3, 1, 2)
    elif input_image.ndimension() == 3:
        if input_image.size(-1) == 3:
            # If input is in format [height, width, channels], permute to [channels, height, width]
            input_image = input_image.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
        else:
            # Add batch dimension if missing and input is [channels, height, width]
            input_image = input_image.unsqueeze(0)
    elif input_image.ndimension() == 2:
        # Add channel and batch dimension if input is [height, width]
        input_image = input_image.unsqueeze(0).unsqueeze(0)


    batch_size, _, height, width = input_image.size()
    crop_size = min(height, width)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # Center crop
    cropped_image = input_image[:, :, top:bottom, left:right]

    # Resize
    resized_image = F.interpolate(cropped_image, size=target_size, mode='bilinear', align_corners=False)

    if output_type == 'pil':
        # Convert tensor to PIL image
        resized_image = resized_image.squeeze(0)  # Remove batch dimension
        resized_image = transforms.ToPILImage()(resized_image)

    return resized_image

def get_model_list(dir_name, sub_token=None, get_dir=False):
    target_dir = os.path.join(folder_paths.models_dir, dir_name)

    if get_dir == True:
        final_list = [cur for cur in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, cur))]
    else:
        final_list = [cur for cur in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, cur)) or os.path.islink(os.path.join(target_dir, cur))]

    if sub_token is not None:
        final_list = [cur for cur in final_list if sub_token in cur]
    
    return final_list