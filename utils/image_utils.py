from PIL import Image
import torchvision.transforms as T
import torch
from torchvision.utils import save_image
import os

def load_target_image(path, size=128):
    """加载并缩放目标图像为张量 (1, 3, H, W)"""
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)

def save_tensor_image(tensor, path):
    """保存张量图像 (3, H, W 或 1, 3, H, W) 到 PNG 文件"""
    if tensor.ndim == 4:
        tensor = tensor[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(tensor.clamp(0, 1), path)
