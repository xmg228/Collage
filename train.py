from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from envs.collage_env import CollageEnv
from torchvision.utils import save_image
import os
import glob


def main():
    
    # 批量获取目标图像路径
    target_image_paths = glob.glob("assets/targets/*.*")  # 或 *.jpg
    env = CollageEnv(target_image_paths=target_image_paths, num_shapes=40, canvas_size=128)
    check_env(env)

    model = SAC("CnnPolicy", env, verbose=1, tensorboard_log="./runs", buffer_size=50_000,learning_rate=3e-4)
    total_steps = 1000000
    log_freq = 10000

    for step in range(0, total_steps, log_freq):
        model.learn(total_timesteps=log_freq, reset_num_timesteps=False)
        # 保存模型
        model.save(f"models/collage_sac_model_{step+log_freq}")
        # 可视化当前画布
        obs, _ = env.reset()
        for i in range(40):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            # 每步保存一次画布
            save_image(env.canvas, f"results/step_{i:03d}.png")
            if terminated or truncated:
                break



if __name__ == "__main__":
    main()
