from transformers import CLIPTokenizer
import os

cur_dir = os.path.dirname(__file__)
with open(os.path.join(cur_dir, 'data', 'stopwords.txt'),'r') as f:
    stopwords = f.readlines()
stopwords= [x.replace('\n','') for x in stopwords]

with open(os.path.join(cur_dir, 'data', 'junkwords.txt'),'r') as f:
    junkwords = f.readlines()
junkwords= [x.replace('\n','') for x in junkwords] 

with open(os.path.join(cur_dir, "data", 'color_list.txt'),'r') as f:
    colorwords = f.readlines()
colorwords= [x.replace('\n','') for x in colorwords] 

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir=os.path.join(cur_dir, "data/clip_tokenizer"))

def tokenize_text(text):
    tokens = tokenizer.tokenize(text)
    tokens = [cur.replace("</w>","") for cur in tokens]
    
    processed_tokens = []
    for token in tokens:
        if token in stopwords:
            continue
        processed_tokens.append(token)
    
    return processed_tokens

def remove_color_from_text(text):
    for colorword in colorwords:
        text = text.replace(colorword, "")
    text = text.replace("  ", " ")
    return text

def post_process_text(text):
    
    for junkword in junkwords:
        text = text.replace(junkword, "")
    
    tokens = tokenize_text(text)

    return " ".join(tokens)


if __name__ == "__main__":
    text = "the product is in a white box with a pink background, high detail product photo, close-up product photo, official product photo, product picture, product photo, product image, detailed product photo, official product image, detailed product image, miniature product photo, product advertisement, product introduction photo, product photo studio lighting, professional product photo, product shoot, product photograph"
    result = post_process_text(text)
    print(result)
