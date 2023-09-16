import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json
from collections import OrderedDict
from utils import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument('image_path', type=str, help='path to the image file for prediction')
    parser.add_argument('checkpoint', type=str, help='path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False, help='use GPU for inference if available')
    return parser.parse_args()

def process_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img = img.crop((0, 0, 224, 224))
    img = np.array(img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))
    return img

def predict(image_path, model, topk, gpu):
    model.eval()
    if gpu and torch.cuda.is_available():
        model.to('cuda')
    else:
        model.to('cpu')

    image = process_image(image_path)
    image = torch.from_numpy(image).float().unsqueeze(0)
    if gpu and torch.cuda.is_available():
        image = image.to('cuda')
    else:
        image = image.to('cpu')

    with torch.no_grad():
        output = model.forward(image)

    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)

    if gpu and torch.cuda.is_available():
        top_p, top_class = top_p.cpu(), top_class.cpu()

    top_p = top_p.numpy()[0]
    top_class = top_class.numpy()[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]

    return top_p, top_class

def load_category_names(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def main():
    args = parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    model = load_checkpoint(checkpoint)
    cat_to_name = load_category_names(category_names)

    top_p, top_class = predict(image_path, model, top_k, gpu)

    class_names = [cat_to_name[class_ ] for class_ in top_class]

    for i in range(top_k):
        print(f"Top class: {class_names[i]}, Probability: {top_p[i]:.3f}")

if __name__ == "__main__":
    main()
