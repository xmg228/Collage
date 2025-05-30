import os
import glob
import torch
import numpy as np
from torch import nn
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.optim import Adam
from envs.collage_env import CollageEnv
from torch.utils.tensorboard import SummaryWriter

class SimpleCNN(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        c, h, w = state_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(),
            nn.Flatten()
        )
        # 计算展平后的特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, *state_shape)
            n_flatten = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, np.prod(action_shape))
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(next(self.parameters()).device)
        batch = obs.shape[0]
        features = self.conv(obs)
        logits = self.fc(features)
        return logits, state

def make_env():
    target_image_paths = glob.glob("assets/targets/*.*")
    return CollageEnv(target_image_paths=target_image_paths, num_shapes=40, canvas_size=128)

if __name__ == "__main__":
    train_envs = DummyVectorEnv([make_env for _ in range(1)])
    test_envs = DummyVectorEnv([make_env for _ in range(1)])

    obs_shape = (6, 128, 128)
    action_shape = 9
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 使用自定义CNN
    actor_net = SimpleCNN(obs_shape, action_shape).to(device)
    actor = ActorProb(actor_net, action_shape, max_action=1.0, device=device).to(device)
    actor_optim = Adam(actor.parameters(), lr=3e-4)

    critic1_net = SimpleCNN(obs_shape, action_shape).to(device)
    critic1 = Critic(critic1_net, action_shape, device=device).to(device)
    critic1_optim = Adam(critic1.parameters(), lr=3e-4)

    critic2_net = SimpleCNN(obs_shape, action_shape).to(device)
    critic2 = Critic(critic2_net, action_shape, device=device).to(device)
    critic2_optim = Adam(critic2.parameters(), lr=3e-4)

    policy = SACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        tau=0.005, gamma=0.99, alpha=0.2,
        estimation_step=1, action_space=train_envs.action_space
    )

    train_collector = Collector(policy, train_envs, ReplayBuffer(size=5000), exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    writer = SummaryWriter(log_dir="runs/tianshou")

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10,
        step_per_epoch=10000,
        step_per_collect=100,
        episode_per_test=5,
        batch_size=64,
        update_per_step=1,
        save_fn=lambda p: torch.save(p.state_dict(), "models/tianshou_sac.pth"),
        writer=writer
    )

    writer.close()
    print("训练完成，结果：", result)