import torch
from torchvision.transforms import v2
import torchvision.io as io
from .model import Model, DEVICE
import os
from pathlib import Path

RELATIVE_ROOT = "../static/models"

def process_directory(directory):
    if os.path.exists(directory):
        print("YES")
    image_tensor = io.read_image(directory) ## image
    
    transform = v2.Compose([
        v2.Resize(size=(227,227)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # --> root of error
    ])
    image_tensor = transform(image_tensor) #--> error
    image_tensor = image_tensor.unsqueeze(0)
    base_dir = Path(__file__).resolve().parent.parent
    
    model = Model(alpha=1e-4).to(DEVICE)
    if os.path.exists(str(Path(base_dir, 'static', 'models', 'model.pth'))):
        pass

    
    model.load_state_dict(torch.load(str(Path(base_dir, 'static', 'models', 'model.pth')), weights_only=True))
    return model.prediction(image_tensor)
