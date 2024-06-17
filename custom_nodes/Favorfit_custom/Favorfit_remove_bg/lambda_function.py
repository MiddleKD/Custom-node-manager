from inference import RemoveBackGround
import json
from utils import bs64_to_pil, img_box_crop, pil_to_bs64, convert_to_rgb, padding_mask_img

def respond(err, res):
    respond_msg = {'statusCode': 502 if err is not None else 200, 'body': json.dumps(res)}
    # print(f'Respond Message: {respond_msg}')
    return respond_msg

def lambda_handler(event, context):

    print("Loading arguments")
    # print(f"Event: {event}")
    
    args = event["body"]
    if isinstance(args, str):
        args = json.loads(args)

    #파이팅 하세요!! 국민대 경영학부 비대위원장 silver stella의 흔적
    
    if "image_b64" in args:
        img_bs64 = args["image_b64"]
    else:
        print("Can not found image base64 format")
        raise AssertionError
    
    print(type(args), list(args.keys()), img_bs64[:50])
    if not img_bs64.startswith("/"):
        img_bs64 = img_bs64.split(",", 1)[1]

    print("Check gpu")
    if "avail_gpu" in args:
        if args["avail_gpu"] == True:
            target_device = "cuda"
        else:
            target_device = "cpu"
    else:
        target_device = "cpu"
    
    print("Loading image")
    img_pil = bs64_to_pil(img_bs64)

    print("Cropping image")
    if "bbox" in args:
        box = args["bbox"]
        img_pil_croped = img_box_crop(img_pil, box)
    else:
        box = None
        img_pil_croped = img_pil

    print("Loading model")
    remover = RemoveBackGround(backbone="swinB", device=target_device)

    print("Processing model")
    img_pil_croped = convert_to_rgb(img_pil_croped)
    output = remover.process(img_pil_croped)

    print("Padding mask")
    padded_mask = padding_mask_img(img_pil=img_pil, mask_img=output, box=box)

    print("Respond")
    output_b64 = pil_to_bs64(padded_mask)
    result = {"mask": "data:application/octet-stream;base64," + output_b64}

    return respond(None, result)
