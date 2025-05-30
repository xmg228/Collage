from stable_baselines3 import SAC
from envs.collage_env_hybrid import CollageEnv
from torchvision.utils import save_image
import os

IMAGE_SIZE = 64
NUM_SHAPES = 10

def test():
    # 加载训练好的模型
    model = SAC.load("models/collage_sac_model_1700000")

    # 构建环境
    env = CollageEnv(target_images="assets/targets/Square.png", num_shapes=NUM_SHAPES, canvas_size=IMAGE_SIZE)
    obs, _ = env.reset()

    # 生成拼贴图像
    for i in range(NUM_SHAPES):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        # 每步保存一次画布
        save_image(env.canvas, f"results/step_{i:03d}.png")
        if terminated or truncated:
            break

    # 保存输出图像
    os.makedirs("results", exist_ok=True)
    save_image(env.canvas, "results/test_collage.png")
    print("拼贴图像已保存至 results/test_collage.png")

if __name__ == "__main__":
    test()
