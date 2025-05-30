import torch
import pickle
import glob
import numpy as np
from stable_baselines3 import SAC
from envs.collage_env import CollageEnv

def collect_seq_targets(env, num_samples=100, num_steps=40):
    data = []
    for i in range(num_samples):
        obs, _ = env.reset()
        traj = []
        for t in range(num_steps):
            canvas = env.canvas.detach().cpu().clone()
            target = env.target.detach().cpu().clone()
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
    return data

def bc_seq_pretrain(model, samples, epochs=5, batch_size=32, device="cuda"):
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)
    model.policy.train()
    flat_samples = [step for traj in samples for step in traj]
    for epoch in range(epochs):
        np.random.shuffle(flat_samples)
        losses = []
        for i in range(0, len(flat_samples), batch_size):
            batch = flat_samples[i:i+batch_size]
            obs_batch = []
            action_batch = []
            for sample in batch:
                canvas = sample["canvas"]
                target = sample["target"]
                if target.dim() == 4:
                    target = target.squeeze(0)
                obs = torch.cat([canvas, target], dim=0).unsqueeze(0)
                obs = (obs * 255).clip(0,255).to(torch.uint8)
                obs_batch.append(obs)
                action_batch.append(torch.tensor(sample["action"]).float())
            obs_batch = torch.cat(obs_batch, dim=0).to(device)
            action_batch = torch.stack(action_batch).to(device)
            mean_action = model.policy._predict(obs_batch, deterministic=True)
            loss = torch.nn.functional.mse_loss(mean_action, action_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"BC Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.4f}")

def main_loop(total_cycles=5, bc_samples=200, bc_epochs=5, rl_steps=10000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = CollageEnv(
        target_image_paths=glob.glob("assets/targets/*.*"),
        num_shapes=40,
        canvas_size=128,
        discriminator=None,
        gan_weight=0.0,
        gan_buffer=None
    )
    model = SAC("CnnPolicy", env, verbose=1, buffer_size=10000, learning_rate=1e-4, device=device)
    for cycle in range(total_cycles):
        print(f"\n=== Cycle {cycle+1}/{total_cycles} ===")
        # 1. 采集数据
        samples = collect_seq_targets(env, num_samples=bc_samples, num_steps=40)
        # 2. 行为克隆预训练
        bc_seq_pretrain(model, samples, epochs=bc_epochs, batch_size=32, device=device)
        # 3. RL训练
        #print("Start RL training...")
        #model.learn(total_timesteps=rl_steps, reset_num_timesteps=False)
        model.save(f"models/collage_sac_cycle{cycle+1}")
    print("训练完成！")

if __name__ == "__main__":
    main_loop(total_cycles=5, bc_samples=500, bc_epochs=10, rl_steps=10000)