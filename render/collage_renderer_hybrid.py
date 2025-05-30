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
            TF.to_tensor(Image.open("assets/shapes/shared/Shield-a8b6511b.png").convert("RGBA")).to(self.device),
            TF.to_tensor(Image.open("assets/shapes/shared/Triangle-b21afb96.png").convert("RGBA")).to(self.device),
            #TF.to_tensor(Image.open("assets/shapes/shared/StrokeBent-d1364135.png").convert("RGBA")).to(self.device),
        ]

        for i, tex in enumerate(self.shape_textures):
            rgb = tex[:3]
            alpha = tex[3:4]
            mask = (alpha < 0.05)
            rgb[:, mask[0]] = 1.0  # 将透明区RGB设为白色
            self.shape_textures[i][:3] = rgb

    def render_single(self, shape_type, params, canvas=None):
        # shape_type: int, params: tensor of shape (8,)
        tx, ty, sx, sy, angle_sin, angle_cos, r, g, b = params

        tx *= self.canvas_size
        ty *= self.canvas_size
        sx = 0.05 + sx * 0.95
        sy = 0.05 + sy * 0.95


        # 保证为 torch.Tensor
        angle_sin = torch.tensor(angle_sin, device=self.device)
        angle_cos = torch.tensor(angle_cos, device=self.device)
        # 还原角度（弧度）
        angle = torch.atan2(angle_sin, angle_cos)  # [-pi, pi]
        angle_deg = angle * 180 / torch.pi         # [-180, 180]
        # 若你希望范围为 [0, 360]，可加 angle_deg = (angle_deg + 360) % 360

        # 选择形状并变换（包含alpha通道）
        shape_rgba = self.shape_textures[shape_type]
        shape_rgb = shape_rgba[:3]
        shape_alpha = shape_rgba[3:4]

        # 着色应在 resize/rotate 之前
        color = torch.tensor([r, g, b], device=self.device).view(3, 1, 1)
        shape_rgb = shape_rgb * color

        shape_rgb = TF.resize(shape_rgb, (int(self.canvas_size * sy), int(self.canvas_size * sx)))
        shape_alpha = TF.resize(shape_alpha, (int(self.canvas_size * sy), int(self.canvas_size * sx)))
        shape_rgb = TF.rotate(shape_rgb, angle_deg.item(), expand=True, fill=[1.0, 1.0, 1.0])
        shape_alpha = TF.rotate(shape_alpha, angle_deg.item(), expand=True, fill=[0.0])
        h, w = shape_rgb.shape[1:]

        # 计算粘贴位置（允许超出边界）
        x0 = int(tx - w // 2)
        y0 = int(ty - h // 2)
        x1 = x0 + w
        y1 = y0 + h

        # 计算画布和素材的有效重叠区域
        canvas_x0 = max(0, x0)
        canvas_y0 = max(0, y0)
        canvas_x1 = min(self.canvas_size, x1)
        canvas_y1 = min(self.canvas_size, y1)

        shape_x0 = max(0, -x0)
        shape_y0 = max(0, -y0)
        shape_x1 = shape_x0 + (canvas_x1 - canvas_x0)
        shape_y1 = shape_y0 + (canvas_y1 - canvas_y0)

        shape_cropped = shape_rgb[:, shape_y0:shape_y1, shape_x0:shape_x1]
        alpha_cropped = shape_alpha[:, shape_y0:shape_y1, shape_x0:shape_x1]

        # 初始化或使用已有画布
        if canvas is None:
            canvas = torch.ones(3, self.canvas_size, self.canvas_size, device=self.device)

        # alpha混合累积贴图
        alpha = alpha_cropped.expand_as(shape_cropped)
        canvas_slice = canvas[:, canvas_y0:canvas_y1, canvas_x0:canvas_x1]
        canvas[:, canvas_y0:canvas_y1, canvas_x0:canvas_x1] = canvas_slice * (1 - alpha) + shape_cropped * alpha

        return canvas
