from sbx import CrossQ,SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.collage_env_hybrid import CollageEnv
from torchvision.utils import make_grid, save_image
import os
import glob
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from custom_efficientnet_extractor import EfficientNetFeatureExtractor

IMAGE_SIZE = 64
NUM_SHAPES = 10

def make_env():
    def _init():
        return CollageEnv(
            target_image_paths=glob.glob("assets/targets/*.*"),
            num_shapes=NUM_SHAPES,
            canvas_size=IMAGE_SIZE,
        )
    return _init

def main():
    # 使用并行环境进行训练
    num_envs = 8
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    eval_env = CollageEnv(
        target_image_paths=glob.glob("assets/targets/*.*"),
        num_shapes=NUM_SHAPES,
        canvas_size=IMAGE_SIZE,
    )

    run_name = f"hybrid_{int(time.time())}"
    writer = SummaryWriter(log_dir=f"./runs/{run_name}")

    policy_kwargs = {
        "features_extractor_class": EfficientNetFeatureExtractor,
        "features_extractor_kwargs": {"output_dim": 256, "in_channels": 9},  # 你的obs通道数
        "net_arch": {"pi": [], "qf": [256, 256]},
        "dropout_rate": 0.01,
        "layer_norm": True,
    }

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./runs/{run_name}",
        buffer_size=50_000,
        learning_rate=3e-4,
        ent_coef='auto_1.0',
        batch_size=64,
        learning_starts=10000,
        gradient_steps=20,
        policy_delay=20,
        policy_kwargs=policy_kwargs,
    )
    #model.set_parameters("models/collage_sac_model_1700000")  # 加载训练模型

    total_steps = 10_000_000
    log_steps = 1_000
    save_freq = 100

    for step in range(0, total_steps, log_steps*save_freq):

        for save_cnt in range(save_freq):
            model.learn(total_timesteps=log_steps, reset_num_timesteps=False)

            # 评估当前策略
            obs, _ = eval_env.reset()
            total_reward = 0
            step_imgs = []
            step_rewards = []
            for i in range(NUM_SHAPES):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                step_imgs.append(eval_env.canvas.detach().cpu().clone())
                step_rewards.append(reward)
                if terminated or truncated:
                    break

            writer.add_scalar("eval/total_reward", total_reward, step + save_cnt * log_steps)

            REWARD_THRESHOLD = 20
            if total_reward > REWARD_THRESHOLD:
                os.makedirs("results/high_reward_steps", exist_ok=True)
                for idx, (img, r) in enumerate(zip(step_imgs, step_rewards)):
                    save_image(img, f"results/high_reward_steps/step_{step+save_cnt*log_steps}_{idx}_reward{r:.2f}.png")
                with open(f"results/high_reward_steps/rewards_{step+save_cnt*log_steps}.txt", "w") as f:
                    for idx, r in enumerate(step_rewards):
                        f.write(f"step {idx}: reward={r}\n")

                canvas = eval_env.canvas.detach().cpu()
                target = eval_env.target.detach().cpu()
                imgs = torch.stack([canvas, target.squeeze(0)], dim=0)
                grid = make_grid(imgs, nrow=2, normalize=True, value_range=(0, 1))
                save_image(grid, f"results/high_reward_steps/compare_{step+save_cnt*log_steps}_reward{total_reward:.2f}.png")

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
