import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from PIL import Image
import torchvision.transforms.functional as TF

class CollageRenderer(nn.Module):
    def __init__(self, num_shapes, canvas_size):
        super().__init__()
        self.num_shapes = num_shapes
        self.canvas_size = canvas_size

        self.shape_texture = torch.tensor(
            TF.to_tensor(Image.open("shapes/shared/Circle-a1eeb3c1.png").convert("RGBA")),
            device="cuda"
        ).unsqueeze(0)

        # 更广的初始化
        self.x_raw = nn.Parameter(torch.randn(num_shapes))
        self.y_raw = nn.Parameter(torch.randn(num_shapes))
        self.scale_x_raw = nn.Parameter(torch.rand(num_shapes))
        self.scale_y_raw = nn.Parameter(torch.rand(num_shapes))
        self.angle = nn.Parameter(torch.zeros(num_shapes))
        self.color = nn.Parameter(torch.rand(num_shapes, 3))

    def forward(self):
        min_scale, max_scale = 0.2, 0.8  # 让形状更大
        canvas = torch.ones(3, self.canvas_size, self.canvas_size, device=self.x_raw.device)
        for i in range(self.num_shapes):
            tx = torch.sigmoid(self.x_raw[i]) * (self.canvas_size - 1)
            ty = torch.sigmoid(self.y_raw[i]) * (self.canvas_size - 1)
            scx = torch.sigmoid(self.scale_x_raw[i]) * (max_scale - min_scale) + min_scale
            scy = torch.sigmoid(self.scale_y_raw[i]) * (max_scale - min_scale) + min_scale
            angle = self.angle[i]
            color = self.color[i]

            # 取 shape 的 RGB 和 alpha 通道
            shape = self.shape_texture[0].unsqueeze(0)  # [1, 4, H, W]
            shape_rgb = shape[:, :3]
            shape_alpha = shape[:, 3:]

            # 用 F.interpolate 缩放（支持 autograd）
            out_h = torch.clamp(self.canvas_size * scy, min=1.0)
            out_w = torch.clamp(self.canvas_size * scx, min=1.0)
            size = (torch.round(out_h).long().item(), torch.round(out_w).long().item())
            shape_rgb = F.interpolate(shape_rgb, size=size, mode='bilinear', align_corners=False)
            shape_alpha = F.interpolate(shape_alpha, size=size, mode='bilinear', align_corners=False)

            # 用 kornia 旋转
            shape_rgb = kornia.geometry.transform.rotate(shape_rgb, angle.unsqueeze(0), align_corners=False)
            shape_alpha = kornia.geometry.transform.rotate(shape_alpha, angle.unsqueeze(0), align_corners=False)

            shape_rgb = shape_rgb[0]
            shape_alpha = shape_alpha[0]

            h, w = shape_rgb.shape[1:]
            # 用 float 计算中心点，最后索引时用 int
            x0 = int(torch.round(tx - w / 2).item())
            y0 = int(torch.round(ty - h / 2).item())
            x1 = x0 + w
            y1 = y0 + h

            # 边界裁剪
            x0_clip, y0_clip = max(0, x0), max(0, y0)
            x1_clip, y1_clip = min(self.canvas_size, x1), min(self.canvas_size, y1)
            sx0, sy0 = x0_clip - x0, y0_clip - y0
            sx1, sy1 = sx0 + (x1_clip - x0_clip), sy0 + (y1_clip - y0_clip)

            if x1_clip > x0_clip and y1_clip > y0_clip:
                shape_cropped = shape_rgb[:, sy0:sy1, sx0:sx1]
                alpha_cropped = shape_alpha[:, sy0:sy1, sx0:sx1]
                colored_shape = shape_cropped * color.view(3, 1, 1)
                alpha = alpha_cropped.expand_as(colored_shape)
                canvas_slice = canvas[:, y0_clip:y1_clip, x0_clip:x1_clip]
                blended = canvas_slice * (1 - alpha) + colored_shape * alpha
                canvas = canvas.clone()
                canvas[:, y0_clip:y1_clip, x0_clip:x1_clip] = blended

        return canvas.clamp(0, 1)