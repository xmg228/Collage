import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy

class HybridActor(nn.Module):
    def __init__(self, feature_dim, num_shape_types, num_continuous_params):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 256)
        self.shape_logits = nn.Linear(256, num_shape_types)
        self.continuous_head = nn.Linear(256, num_continuous_params)

    def forward(self, features):
        x = F.relu(self.fc(features))
        shape_logits = self.shape_logits(x)
        params = torch.tanh(self.continuous_head(x))
        return torch.cat([shape_logits, params], dim=-1)

class HybridActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, num_shape_types=5, num_continuous_params=8, **kwargs):
        super().__init__(*args, **kwargs)
        feature_dim = self.mlp_extractor.latent_dim_pi
        # 用自定义 HybridActor 替换 self.actor
        self.actor = HybridActor(feature_dim, num_shape_types, num_continuous_params)
        # 不要重写 forward，保持父类逻辑