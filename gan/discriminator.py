import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels * 2, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * (img_size // 8) * (img_size // 8), 1)
        )

    def forward(self, canvas, target):
        
        x = torch.cat([canvas, target], dim=1)
        return self.model(x)