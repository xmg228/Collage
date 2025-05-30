from stable_baselines3 import PPO
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
NUM_SHAPES =20

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
    }

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./runs/{run_name}",
        batch_size=128,
        n_steps=2048,  # 可根据显存和任务调整
        ent_coef=0.01, # 可调大以增强探索
        learning_rate=3e-4,
        #policy_kwargs=policy_kwargs,
    )
    #model.set_parameters("models/collage_sac_model_400000")  # 加载训练模型

    total_steps = 10_000_000
    log_steps = 1_000
    save_freq = 100
    EVAL_EPISODES = 5  # 每次评估采样次数

    for step in range(0, total_steps, log_steps*save_freq):

        for save_cnt in range(save_freq):
            model.learn(total_timesteps=log_steps, reset_num_timesteps=False)

            # 多次评估当前策略
            eval_rewards = []
            best_reward = -float('inf')
            best_imgs = None
            best_step_rewards = None

            for _ in range(EVAL_EPISODES):
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
                eval_rewards.append(total_reward)
                # 记录最高分那一轮
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_imgs = step_imgs
                    best_step_rewards = step_rewards

            avg_reward = np.mean(eval_rewards)
            writer.add_scalar("eval/total_reward", avg_reward, step + save_cnt * log_steps)

            REWARD_THRESHOLD = 20
            if avg_reward > REWARD_THRESHOLD and best_imgs is not None:
                os.makedirs("results/high_reward_steps", exist_ok=True)
                for idx, (img, r) in enumerate(zip(best_imgs, best_step_rewards)):
                    save_image(img, f"results/high_reward_steps/step_{step+save_cnt*log_steps}_{idx}_reward{r:.2f}.png")
                with open(f"results/high_reward_steps/rewards_{step+save_cnt*log_steps}.txt", "w") as f:
                    for idx, r in enumerate(best_step_rewards):
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
