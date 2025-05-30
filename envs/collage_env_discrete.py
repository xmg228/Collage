import gymnasium as gym
import torch
import numpy as np
import random
from render.collage_renderer import CollageRenderer
from loss.perceptual import PerceptualLoss
from utils.image_utils import load_target_image

class CollageEnv(gym.Env):
    def __init__(self, target_image_paths, num_shapes, canvas_size, discriminator=None, gan_weight=0.1, gan_buffer=None, reward_type="mse"):
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
        self.n_actions = 100000  # 预设动作数量  
        # 预设动作参数表
        self.action_table = self._build_action_table(self.n_actions)
        self.action_space = gym.spaces.Discrete(self.n_actions)  # ←加上这一行

        # 观察空间: 当前画布 + 目标图像 (6通道)，float32，[0,1]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(8, canvas_size, canvas_size), dtype=np.uint8
        )

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

    def _build_action_table(self, n):
        # 生成 n 个动作，每个动作是 9 维参数
        table = []
        for i in range(n):
            param = np.random.uniform(0, 1, size=(9,))
            table.append(param)
        return np.array(table)

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
        obs = torch.cat([
            self.canvas,          # (3, H, W)
            self.target[0],       # (3, H, W)
            self.coord            # (2, H, W)
        ], dim=0).cpu().numpy()   # (8, H, W)
        obs = np.clip(obs * 255, 0, 255).astype(np.uint8)
        return obs

    def step(self, action):
        # 查表法：action 是一个整数索引
        action = int(action)
        action_param = self.action_table[action]  # shape: (9,)
        action_tensor = torch.tensor(action_param, device=self.device, dtype=torch.float32)


        canvas_before = self.canvas.clone()
        self.canvas = self.renderer.render_single(action_tensor, self.canvas)
        self.step_count += 1

        terminated = self.step_count >= self.max_steps
        truncated = False

        # 计算 reward（与原来一致）
        reward = 0.0
        if self.reward_type == "gan" and self.discriminator is not None:
            with torch.no_grad():
                d_before = self.discriminator(canvas_before.unsqueeze(0), self.target)
                d_after = self.discriminator(self.canvas.unsqueeze(0), self.target)
                reward = (d_after - d_before).item() * self.gan_weight
        elif self.reward_type == "mse":
            before = self.loss_fn(canvas_before.unsqueeze(0), self.target).item()
            after = self.loss_fn(self.canvas.unsqueeze(0), self.target).item()
            reward = (before - after) 
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

        # ----------------

        # 存入GAN经验池
        if hasattr(self, "gan_buffer") and self.gan_buffer is not None:
            self.gan_buffer.push(self.canvas.detach().cpu(), self.target.detach().cpu())

        # 返回归一化后的 reward
        return self._get_obs(), normed_reward, terminated, truncated, {}
