from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.collage_env import CollageEnv
from torchvision.utils import make_grid, save_image
import os
import glob
import torch
from gan.discriminator import Discriminator
from gan.wgangp_utils import compute_gradient_penalty
from utils.gan_replay_buffer import GANReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

IMAGE_SIZE = 64
NUM_SHAPES = 10

gan_buffer = GANReplayBuffer(capacity=10000)
device = "cuda" if torch.cuda.is_available() else "cpu"
discriminator = Discriminator(img_channels=3, img_size=IMAGE_SIZE).to(device)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))


def make_env():
    def _init():
        return CollageEnv(
            target_image_paths=glob.glob("assets/targets/*.*"),
            num_shapes=NUM_SHAPES,
            canvas_size=IMAGE_SIZE,
            discriminator=discriminator,
            gan_weight=0.1,
            gan_buffer=gan_buffer
        )
    return _init


def train_discriminator(discriminator, optimizer, real_canvas, fake_canvas, target, lambda_gp=10):
    # real_canvas, fake_canvas, target: [B, 3, H, W]
    d_real = discriminator(real_canvas, target)
    d_fake = discriminator(fake_canvas, target)
    gp = compute_gradient_penalty(discriminator, real_canvas, fake_canvas, target)
    d_loss = -d_real.mean() + d_fake.mean() + lambda_gp * gp
    optimizer.zero_grad()
    d_loss.backward()
    optimizer.step()
    return d_loss.item()


def main():
    # 使用并行环境进行训练
    num_envs = 12
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    eval_env = CollageEnv(
        target_image_paths=glob.glob("assets/targets/*.*"),
        num_shapes=NUM_SHAPES,
        canvas_size=IMAGE_SIZE,
        discriminator=discriminator,
        gan_weight=0.1,
        gan_buffer=gan_buffer
    )

    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = SAC(
        "CnnPolicy",
        env,
        action_noise=action_noise,  # 添加动作噪声
        verbose=1,
        tensorboard_log="./runs",
        buffer_size=10_000,
        learning_rate=1e-4,
        ent_coef='auto_1.0',
        batch_size=64
    )
    total_steps = 10_000_000
    log_steps = 1_000
    save_freq = 10
    writer = SummaryWriter(log_dir="./runs")

    for step in range(0, total_steps, log_steps*save_freq):

        for save_cnt in range(save_freq):
            model.learn(total_timesteps=log_steps, reset_num_timesteps=False)
            
            # 判别器训练
            batch_size = 32
            for _ in range(5):  # 判别器训练步数
                if len(gan_buffer) < batch_size:
                    continue  # 经验池不够，跳过
                fake_canvas, target = gan_buffer.sample(batch_size)
                fake_canvas = fake_canvas.to(device)
                target = target.to(device)
                real_canvas = target  # 真实图像直接用目标图像
                d_loss = train_discriminator(discriminator, d_optimizer, real_canvas, fake_canvas, target, lambda_gp=10)
            
            # 评估当前策略
            obs, _ = eval_env.reset()
            total_reward = 0
            for i in range(NUM_SHAPES):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            writer.add_scalar("eval/total_reward", total_reward, step + save_cnt * log_steps)
            

        # 可视化当前画布
        canvas = eval_env.canvas.detach().cpu()
        target = eval_env.target.detach().cpu()
        imgs = torch.stack([canvas, target.squeeze(0)], dim=0)
        grid = make_grid(imgs, nrow=2, normalize=True, value_range=(0, 1))
        save_image(grid, f"results/compare_{step+log_steps*save_freq}.png")

        # 保存模型
        model.save(f"models/collage_sac_model_{step+log_steps*save_freq}")


    writer.close()


if __name__ == "__main__":
    main()
