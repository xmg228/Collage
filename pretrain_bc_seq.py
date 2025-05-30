import torch
import pickle
from stable_baselines3 import SAC
from envs.collage_env import CollageEnv
import glob
import numpy as np

def bc_seq_pretrain(model, data_path="random_seq_targets.pkl", epochs=10, batch_size=32, device="cuda"):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} trajectories.")

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)
    model.policy.train()

    # 展平成单步样本
    samples = []
    for traj in data:
        for step in traj:
            samples.append(step)
    print(f"Total steps: {len(samples)}")

    for epoch in range(epochs):
        np.random.shuffle(samples)
        losses = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            obs_batch = []
            action_batch = []
            for sample in batch:
                canvas = sample["canvas"]
                target = sample["target"]
                target = target.squeeze(0) 
                obs = torch.cat([canvas, target], dim=0).unsqueeze(0)  # [1,6,128,128]
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.4f}")

if __name__ == "__main__":
    env = CollageEnv(
        target_image_paths=glob.glob("assets/targets/*.*"),
        num_shapes=40,
        canvas_size=128,
        discriminator=None,
        gan_weight=0.0,
        gan_buffer=None
    )
    model = SAC("CnnPolicy", env, verbose=1, buffer_size=10000, learning_rate=1e-4)
    bc_seq_pretrain(model)
    model.save("models/collage_sac_bc_seq_pretrain")