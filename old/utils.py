from PIL import Image
import torchvision.transforms as T
import torch
from torchvision.utils import save_image

def load_target_image(path, size=128):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # (1, 3, H, W)

def save_image_grid(images, path):
    imgs = torch.stack(images)
    save_image(imgs, path, nrow=len(images))
