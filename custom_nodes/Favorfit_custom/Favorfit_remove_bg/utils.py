from PIL import Image
import requests
from io import BytesIO
import base64

def load_pil_from_url(url):
    img_data = requests.get(url, stream=True).raw
    img_pil = Image.open(img_data)
    return img_pil

def load_bs64_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    image_data = response.content
    return base64.b64encode(image_data).decode()

def bs64_to_pil(img_bs64):
    img_data = base64.b64decode(img_bs64)
    img_pil = Image.open(BytesIO(img_data))
    return img_pil

def img_box_crop(img_pil, box):
    x1, y1, x2, y2 = box.values()
    return img_pil.crop((x1, y1, x2, y2))

def np_to_bs64(img_np):
    img = Image.fromarray(img_np.astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="png")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

def pil_to_bs64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="png")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

def save_image_to_local(img):
    Image.fromarray(img).save("./test_img.png")

def convert_to_rgb(img_pil):
    if img_pil.mode == 'RGBA':
        img_rgb = Image.new('RGB', img_pil.size, (255, 255, 255))
        img_rgb.paste(img_pil, mask=img_pil.split()[3])
        return img_rgb
    elif img_pil.mode == 'RGB':
        return img_pil
    else:
        print("Image is not in RGBA or RGB mode.")
        return img_pil

def padding_mask_img(img_pil, mask_img, box):
    mask_img = Image.fromarray(mask_img)
    
    if box == None:
        return mask_img
    
    x1, y1, x2, y2 = box.values()
    black_img = Image.new("RGB", img_pil.size)
    black_img.paste(mask_img, (x1, y1, x2, y2))
    return black_img

