from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.collage_env import CollageEnv
from torchvision.utils import save_image
import os
import glob


def make_env():
    def _init():
        return CollageEnv(target_image_paths=glob.glob("assets/targets/*.*"), num_shapes=40, canvas_size=128)
    return _init


def main():
    # 使用并行环境进行训练
    num_envs = 4
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    eval_env = CollageEnv(target_image_paths=glob.glob("assets/targets/*.*"), num_shapes=40, canvas_size=128)

    model = SAC("CnnPolicy", env, verbose=1, tensorboard_log="./runs", buffer_size=100_000, learning_rate=3e-4, ent_coef='auto_0.5')
    total_steps = 1000000
    log_freq = 10000

    for step in range(0, total_steps, log_freq):
        model.learn(total_timesteps=log_freq, reset_num_timesteps=False)
        model.save(f"models/collage_sac_model_{step+log_freq}")

        # 评估当前策略
        obs, _ = eval_env.reset()
        total_reward = 0
        for i in range(40):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Step {step+log_freq}: Eval reward = {total_reward}")

        # 可视化当前画布
        save_image(eval_env.canvas, f"results/step_{step+log_freq}.png")


if __name__ == "__main__":
    main()
