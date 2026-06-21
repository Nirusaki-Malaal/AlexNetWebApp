import torch
from torchvision.transforms import v2
import torchvision.io as io
import os
from app import model

RELATIVE_ROOT = "../static/models"

def process_directory(directory):
    if os.path.exists(directory):
        print("YES")
    image_tensor = io.read_image(directory) ## image
    
    transform = v2.Compose([
        v2.Resize(size=(227,227)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalizing data with mean of alexnet trained on imagenet
    ])
    image_tensor = transform(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)
    return model.prediction(image_tensor)
