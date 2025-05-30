from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from envs.collage_env_hybrid import CollageEnv
from torchvision.utils import make_grid, save_image
from torchvision.datasets import CIFAR100
from torchvision import transforms
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

IMAGE_SIZE = 64
NUM_SHAPES = 10

# 只传递 dataset
def make_env():
    def _init():
        return CollageEnv(
            target_images=dataset,
            num_shapes=NUM_SHAPES,
            canvas_size=IMAGE_SIZE,
            training=True,
        )
    return _init


# 定义预处理
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # 调整分辨率
    transforms.ToTensor()
])

# 加载数据集
dataset = CIFAR100(
    root="assets/dataset",  # 指向 assets 目录
    train=True,
    download=False,  # 已经有本地文件
    transform=transform
)




def main():
    # 使用并行环境进行训练
    num_envs = 6
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    eval_env = CollageEnv(
        target_images=dataset,
        num_shapes=NUM_SHAPES,
        canvas_size=IMAGE_SIZE,
    )

    run_name = f"hybrid_{int(time.time())}"
    writer = SummaryWriter(log_dir=f"./runs/{run_name}")


    action_dim = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim),
        sigma=0.1 * np.ones(action_dim)
    )

    model = SAC(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./runs/{run_name}",
        buffer_size=50_000,
        learning_rate=3e-4,
        ent_coef='auto_1.0',
        action_noise=action_noise,  # 添加噪声
        batch_size=64,
        learning_starts=10_000,
        #policy_kwargs=policy_kwargs,
    )
    #model.set_parameters("models/collage_sac_model_10000")  # 加载训练模型

    total_steps = 10_000_000
    log_steps = 1000
    save_freq = 50
    EVAL_EPISODES = 5  # 每次评估采样次数

    for step in range(0, total_steps, log_steps*save_freq):

        for save_cnt in range(save_freq):
            model.learn(total_timesteps=log_steps, reset_num_timesteps=False)

            # 多次评估当前策略
            eval_rewards = []
            best_reward = -float('inf')
            best_imgs = None
            best_step_rewards = None

            # 用于统计各奖励项
            all_basic_reward = []
            all_color_reward = []
            all_new_cover_reward = []
            all_overlap_penalty = []
            all_param_penalty = []
            all_region_diversity_reward = []
            all_final_reward = []

            for _ in range(EVAL_EPISODES):
                obs, _ = eval_env.reset()
                total_reward = 0
                step_imgs = []
                step_rewards = []

                # 本episode奖励项累加器
                episode_basic_reward = 0
                episode_color_reward = 0
                episode_new_cover_reward = 0
                episode_overlap_penalty = 0
                episode_param_penalty = 0
                episode_region_diversity_reward = 0
                episode_final_reward = 0

                for i in range(NUM_SHAPES):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    total_reward += reward
                    step_imgs.append(eval_env.canvas.detach().cpu().clone())
                    step_rewards.append(reward)
                    # 累加每步奖励项
                    episode_basic_reward += info.get("basic_reward", 0)
                    episode_color_reward += info.get("color_reward", 0)
                    episode_new_cover_reward += info.get("new_cover_reward", 0)
                    episode_overlap_penalty += info.get("overlap_penalty", 0)
                    episode_param_penalty += info.get("param_penalty", 0)
                    episode_region_diversity_reward += info.get("region_diversity_reward", 0)
                    if terminated or truncated:
                        episode_final_reward = info.get("final_reward", 0)
                        break

                eval_rewards.append(total_reward)
                all_basic_reward.append(episode_basic_reward)
                all_color_reward.append(episode_color_reward)
                all_new_cover_reward.append(episode_new_cover_reward)
                all_overlap_penalty.append(episode_overlap_penalty)
                all_param_penalty.append(episode_param_penalty)
                all_region_diversity_reward.append(episode_region_diversity_reward)
                all_final_reward.append(episode_final_reward)

                # 记录最高分那一轮
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_imgs = step_imgs
                    best_step_rewards = step_rewards

            avg_reward = np.mean(eval_rewards)
            # 写入各奖励项的平均值
            writer.add_scalar("eval/total_reward", avg_reward, step + save_cnt * log_steps)
            writer.add_scalar("eval/basic_reward", np.mean(all_basic_reward), step + save_cnt * log_steps)
            writer.add_scalar("eval/color_reward", np.mean(all_color_reward), step + save_cnt * log_steps)
            writer.add_scalar("eval/new_cover_reward", np.mean(all_new_cover_reward), step + save_cnt * log_steps)
            writer.add_scalar("eval/overlap_penalty", np.mean(all_overlap_penalty), step + save_cnt * log_steps)
            writer.add_scalar("eval/param_penalty", np.mean(all_param_penalty), step + save_cnt * log_steps)
            writer.add_scalar("eval/region_diversity_reward", np.mean(all_region_diversity_reward), step + save_cnt * log_steps)
            writer.add_scalar("eval/final_reward", np.mean(all_final_reward), step + save_cnt * log_steps)
            
            REWARD_THRESHOLD = 30
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
