import torchvision.transforms.functional as TF
from PIL import Image
import torch.nn as nn
import torch

class CollageRenderer(nn.Module):
    def __init__(self, canvas_size):
        super().__init__()
        self.canvas_size = canvas_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载图形素材（白色图形，透明背景）
        self.shape_textures = [
            #TF.to_tensor(Image.open("assets/shapes/shared/Star-74dcc2c6.png").convert("RGBA")).to(self.device),
            #TF.to_tensor(Image.open("assets/shapes/shared/Heart-f5973f34.png").convert("RGBA")).to(self.device),
            TF.to_tensor(Image.open("assets/shapes/shared/Circle-a1eeb3c1.png").convert("RGBA")).to(self.device),
            TF.to_tensor(Image.open("assets/shapes/shared/Square-bb33f5a5.png").convert("RGBA")).to(self.device),
            #TF.to_tensor(Image.open("assets/shapes/shared/Shield-a8b6511b.png").convert("RGBA")).to(self.device),
            TF.to_tensor(Image.open("assets/shapes/shared/Triangle-b21afb96.png").convert("RGBA")).to(self.device),
            TF.to_tensor(Image.open("assets/shapes/shared/StrokeBent-d1364135.png").convert("RGBA")).to(self.device),
        ]

    def render_single(self, action, canvas=None):
        # 解析动作参数
        shape_type = int((action[0] * len(self.shape_textures)).clamp(0, len(self.shape_textures) - 1))
        tx, ty, sx, sy, angle, r, g, b = action[1:]

        tx *= self.canvas_size
        ty *= self.canvas_size
        sx = 0.05 + sx * 0.95
        sy = 0.05 + sy * 0.95
        angle = angle * 360

        # 选择形状并变换（包含alpha通道）
        shape_rgba = self.shape_textures[shape_type]
        shape_rgb = shape_rgba[:3]
        shape_alpha = shape_rgba[3:4]

        shape_rgb = TF.resize(shape_rgb, (int(self.canvas_size * sy), int(self.canvas_size * sx)))
        shape_alpha = TF.resize(shape_alpha, (int(self.canvas_size * sy), int(self.canvas_size * sx)))
        shape_rgb = TF.rotate(shape_rgb, angle.item(), expand=True)
        shape_alpha = TF.rotate(shape_alpha, angle.item(), expand=True)
        h, w = shape_rgb.shape[1:]

        # 计算粘贴位置
        x0 = int(max(0, tx - w // 2))
        y0 = int(max(0, ty - h // 2))
        x1 = min(self.canvas_size, x0 + w)
        y1 = min(self.canvas_size, y0 + h)
        sx0, sy0 = 0, 0
        sx1, sy1 = x1 - x0, y1 - y0
        if x0 < 0:
            sx0 = -x0
            x0 = 0
        if y0 < 0:
            sy0 = -y0
            y0 = 0

        shape_cropped = shape_rgb[:, sy0:sy1, sx0:sx1]
        alpha_cropped = shape_alpha[:, sy0:sy1, sx0:sx1]

        # 着色
        color = torch.tensor([r, g, b], device=self.device).view(3, 1, 1)
        colored_shape = shape_cropped * color

        # 初始化或使用已有画布
        if canvas is None:
            canvas = torch.ones(3, self.canvas_size, self.canvas_size, device=self.device)

        # alpha混合累积贴图
        alpha = alpha_cropped.expand_as(colored_shape)
        canvas_slice = canvas[:, y0:y1, x0:x1]
        canvas[:, y0:y1, x0:x1] = canvas_slice * (1 - alpha) + colored_shape * alpha

        return canvas
