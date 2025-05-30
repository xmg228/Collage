import torch
import torch.nn as nn
from collage_renderer import CollageRenderer
from loss import PerceptualLoss
from utils import load_target_image, save_image_grid

# 参数设置
num_shapes = 40
canvas_size = 128
steps = 1000
lr = 1e-2

# 目标图像加载
target = load_target_image("assets/target.png", size=canvas_size).to("cuda")

# 渲染器和损失函数
renderer = CollageRenderer(num_shapes, canvas_size).to("cuda")
loss_fn = PerceptualLoss().to("cuda")

# 优化器
optimizer = torch.optim.Adam(renderer.parameters(), lr=lr)

# 训练循环
for step in range(steps):
    optimizer.zero_grad()
    
    collage = renderer()  # 输出拼贴图像
    loss = loss_fn(collage, target)

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"[{step}] Loss: {loss.item():.4f}")
        save_image_grid([target.squeeze(0), collage], f"results/{step}.png")
