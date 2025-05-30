from sbx import CrossQ,SAC
from envs.collage_env_hybrid import CollageEnv
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import os
import glob
import optax   
import time
import torch

IMAGE_SIZE = 64
NUM_SHAPES = 10
def main():
    # 创建环境
    env = CollageEnv(
        target_image_paths=glob.glob("assets/targets/*.*"),
        num_shapes=NUM_SHAPES,
        canvas_size=IMAGE_SIZE
    )
    
    
    run_name = f"hybrid_{int(time.time())}"
    writer = SummaryWriter(log_dir=f"./runs/{run_name}")

    model = SAC(
        env=env,
        policy="SimbaPolicy",
        learning_rate=3e-4,
        # qf_learning_rate=1e-3,
        tensorboard_log=f"./runs/{run_name}",
        policy_kwargs={
            "optimizer_class": optax.adamw,
            # "optimizer_kwargs": {"weight_decay": 0.01},
            # Note: here [128] represent a residual block, not just a single layer
            "net_arch": {"pi": [128], "qf": [256, 256]},
            "n_critics": 2,
            "dropout_rate": 0.01,
            "layer_norm": False,
        },
        # Important: input normalization using VecNormalize
        #normalize={"norm_obs": True, "norm_reward": False},
        batch_size=64,
        buffer_size=50_000,
        learning_starts=1000,
        gradient_steps=20,
        policy_delay=20,
    )
    #model.learn(total_timesteps=1000000)

    total_steps = 10_000_000
    log_steps = 1_000
    save_freq = 100

    for step in range(0, total_steps, log_steps*save_freq):

        for save_cnt in range(save_freq):
            model.learn(total_timesteps=log_steps, reset_num_timesteps=False)
            
            
            # 评估当前策略
            obs, _ = env.reset()
            total_reward = 0
            step_imgs = []
            step_rewards = []
            for i in range(NUM_SHAPES):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                step_imgs.append(env.canvas.detach().cpu().clone())
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

                canvas = env.canvas.detach().cpu()
                target = env.target.detach().cpu()
                imgs = torch.stack([canvas, target.squeeze(0)], dim=0)
                grid = make_grid(imgs, nrow=2, normalize=True, value_range=(0, 1))
                save_image(grid, f"results/high_reward_steps/compare_{step+save_cnt*log_steps}_reward{total_reward:.2f}.png")


        # 可视化当前画布
        canvas = env.canvas.detach().cpu()
        target = env.target.detach().cpu()
        imgs = torch.stack([canvas, target.squeeze(0)], dim=0)
        grid = make_grid(imgs, nrow=2, normalize=True, value_range=(0, 1))
        save_image(grid, f"results/compare_{step+log_steps*save_freq}.png")

        # 保存模型
        model.save(f"models/collage_droq_model_{step+log_steps*save_freq}")


    writer.close()

if __name__ == "__main__":
    main()