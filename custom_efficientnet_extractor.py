import torch
import torch.nn as nn
import timm

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, observation_space, output_dim=256, in_channels=9):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_lite0',
            pretrained=False,
            features_only=True,
            in_chans=in_channels
        )
        self.fc = nn.Linear(320, output_dim)
        self.features_dim = output_dim  # 必须加这一行

    def forward(self, x):
        feats = self.backbone(x)[-1]
        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        return self.fc(pooled)