import torch
import pickle
from envs.collage_env import CollageEnv
import glob

def collect_seq_targets(num_samples=1000, num_steps=40, save_path="random_seq_targets.pkl"):
    env = CollageEnv(
        target_image_paths=glob.glob("assets/targets/*.*"),
        num_shapes=num_steps,
        canvas_size=128,
        discriminator=None,
        gan_weight=0.0,
        gan_buffer=None
    )
    data = []
    for i in range(num_samples):
        obs, _ = env.reset()
        traj = []
        for t in range(num_steps):
            canvas = env.canvas.detach().cpu().clone()  # 当前画布
            target = env.target.detach().cpu().clone()  # 目标图片
            action = env.action_space.sample()
            traj.append({
                "canvas": canvas,
                "target": target,
                "action": action
            })
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        data.append(traj)
        if (i+1) % 100 == 0:
            print(f"Collected {i+1}/{num_samples}")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {num_samples} trajectories to {save_path}")

if __name__ == "__main__":
    collect_seq_targets()