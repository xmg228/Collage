import gymnasium as gym
import torch
import numpy as np
import random
from render.collage_renderer import CollageRenderer
from loss.perceptual import PerceptualLoss
from utils.image_utils import load_target_image
import torch.nn.functional as F

class CollageEnv(gym.Env):
    def __init__(self, target_image_paths, num_shapes, canvas_size, discriminator=None, gan_weight=0.1, gan_buffer=None, reward_type="ssim"):
        super().__init__()
        self.canvas_size = canvas_size
        self.num_shapes = num_shapes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reward_type = reward_type

        # 支持传入多个目标图像路径
        if isinstance(target_image_paths, str):
            self.target_image_paths = [target_image_paths]
        else:
            self.target_image_paths = target_image_paths

        # 初始化时先加载第一个目标
        self.target = load_target_image(self.target_image_paths[0], size=canvas_size).to(self.device)
        self.loss_fn = PerceptualLoss().to(self.device)
        self.renderer = CollageRenderer(canvas_size).to(self.device)

        self.max_steps = num_shapes
        self.step_count = 0

        # 动作空间: [type (0~1), x, y, sx, sy, angle, r, g, b]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        # 观察空间: 当前画布 + 目标图像 (6通道)，float32，[0,1]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(9, canvas_size, canvas_size), dtype=np.uint8)

        self.discriminator = discriminator
        self.gan_weight = gan_weight
        self.gan_buffer = gan_buffer


        # 奖励归一化相关
        self.reward_running_mean = 0
        self.reward_running_var = 1
        self.reward_count = 1

        # 生成归一化坐标通道 (2, H, W)
        xs = torch.linspace(0, 1, self.canvas_size)
        ys = torch.linspace(0, 1, self.canvas_size)
        coord_x = xs.view(1, 1, -1).expand(1, self.canvas_size, self.canvas_size)
        coord_y = ys.view(1, -1, 1).expand(1, self.canvas_size, self.canvas_size)
        self.coord = torch.cat([coord_x, coord_y], dim=0).to(self.device)  # (2, H, W)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.canvas = torch.ones(3, self.canvas_size, self.canvas_size, device=self.device)
        self.step_count = 0
        # 每次 reset 随机选一个目标图像
        target_path = random.choice(self.target_image_paths)
        self.target = load_target_image(target_path, size=self.canvas_size).to(self.device)
        return self._get_obs(), {}

    def _get_obs(self):
        # 计算目标图像边缘
        target_gray = self.target[0].mean(0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=target_gray.dtype, device=target_gray.device).view(1,1,3,3)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=target_gray.dtype, device=target_gray.device).view(1,1,3,3)
        edge = F.conv2d(target_gray, sobel_x, padding=1) + F.conv2d(target_gray, sobel_y, padding=1)
        edge = edge.squeeze(0)  # [1,H,W]
        # 拼接到obs
        obs = torch.cat([
            self.canvas,          # (3, H, W)
            self.target[0],       # (3, H, W)
            self.coord,           # (2, H, W)
            edge                  # (1, H, W)
        ], dim=0).cpu().numpy()   # (9, H, W)
        obs = np.clip(obs * 255, 0, 255).astype(np.uint8)
        return obs

    def step(self, action):
        action = torch.tensor(action, device=self.device)
        action = (action + 1) / 2  # [-1,1] -> [0,1]
        canvas_before = self.canvas.clone()
        self.canvas = self.renderer.render_single(action, self.canvas)
        self.step_count += 1

        terminated = self.step_count >= self.max_steps
        truncated = False

        reward = 0.0
        if self.reward_type == "gan" and self.discriminator is not None:
            with torch.no_grad():
                d_before = self.discriminator(canvas_before.unsqueeze(0), self.target)
                d_after = self.discriminator(self.canvas.unsqueeze(0), self.target)
                reward = (d_after - d_before).item() * self.gan_weight
        elif self.reward_type == "ssim":
            before = self.loss_fn(canvas_before.unsqueeze(0), self.target).item()
            after = self.loss_fn(self.canvas.unsqueeze(0), self.target).item()
            reward = (after - before) 
            #reward = -self.loss_fn(self.canvas.unsqueeze(0), self.target).item()

        # 你也可以支持混合reward
        elif self.reward_type == "both" and self.discriminator is not None:
            with torch.no_grad():
                d_before = self.discriminator(canvas_before.unsqueeze(0), self.target)
                d_after = self.discriminator(self.canvas.unsqueeze(0), self.target)
                gan_reward = (d_after - d_before).item() * self.gan_weight
            mse_reward = -self.loss_fn(self.canvas.unsqueeze(0), self.target).item()
            reward = gan_reward + mse_reward

        # --- 归一化 ---
        # 更新均值和方差（滑动平均）
        self.reward_count += 1
        alpha = 0.01  # 平滑系数，可调
        self.reward_running_mean = (1 - alpha) * self.reward_running_mean + alpha * reward
        self.reward_running_var = (1 - alpha) * self.reward_running_var + alpha * ((reward - self.reward_running_mean) ** 2)
        normed_reward = (reward - self.reward_running_mean) / (np.sqrt(self.reward_running_var) + 1e-8)


        # 存入GAN经验池
        if hasattr(self, "gan_buffer") and self.gan_buffer is not None:
            self.gan_buffer.push(self.canvas.detach().cpu(), self.target.detach().cpu())

        return self._get_obs(), normed_reward, terminated, truncated, {}
