import gymnasium as gym
import torch
import numpy as np
import random
from render.collage_renderer_hybrid import CollageRenderer
from loss.perceptual import PerceptualLoss
from utils.image_utils import load_target_image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

class CollageEnv(gym.Env):
    def __init__(self, target_images, num_shapes, canvas_size, training=False):
        super().__init__()
        self.canvas_size = canvas_size
        self.num_shapes = num_shapes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 新增：支持dataset对象
        if hasattr(target_images, "__getitem__") and hasattr(target_images, "__len__"):
            # 传入的是dataset对象
            self.target_dataset = target_images
            self.targets = None
        elif isinstance(target_images, list) and isinstance(target_images[0], torch.Tensor):
            # 直接传入图片tensor
            self.targets = [img.to(self.device) for img in target_images]
            self.target_dataset = None
        elif isinstance(target_images, str) or (isinstance(target_images, list) and isinstance(target_images[0], str)):
            # 路径方式，兼容原有逻辑
            if isinstance(target_images, str):
                self.target_image_paths = [target_images]
            else:
                self.target_image_paths = target_images
            self.targets = [load_target_image(p, size=canvas_size).to(self.device) for p in self.target_image_paths]
            self.target_dataset = None
        else:
            raise ValueError("target_images must be a dataset, a list of tensors, or a list of paths.")

        # 初始化target
        if self.target_dataset is not None:
            self.target = None  # 每次reset时动态采样
        else:
            self.target = self.targets[0]

        self.shape_textures = [
            #TF.to_tensor(Image.open("assets/shapes/shared/Star-74dcc2c6.png").convert("RGBA")).to(self.device),
            #TF.to_tensor(Image.open("assets/shapes/shared/Heart-f5973f34.png").convert("RGBA")).to(self.device),
            TF.to_tensor(Image.open("assets/shapes/shared/Circle-a1eeb3c1.png").convert("RGBA")).to(self.device),
            TF.to_tensor(Image.open("assets/shapes/shared/Square-bb33f5a5.png").convert("RGBA")).to(self.device),
            TF.to_tensor(Image.open("assets/shapes/shared/Shield-a8b6511b.png").convert("RGBA")).to(self.device),
            TF.to_tensor(Image.open("assets/shapes/shared/Triangle-b21afb96.png").convert("RGBA")).to(self.device),
            #TF.to_tensor(Image.open("assets/shapes/shared/StrokeBent-d1364135.png").convert("RGBA")).to(self.device),
        ]

        self.loss_fn = PerceptualLoss().to(self.device)
        self.renderer = CollageRenderer(canvas_size).to(self.device)

        self.max_steps = num_shapes
        self.step_count = 0

        # 动作空间: [type (0~1), x, y, sx, sy, sin, cos, r, g, b]
        self.num_shape_types = len(self.renderer.shape_textures)  # 获取素材类型数量
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_shape_types + 9,), dtype=np.float32)
        # 观察空间: 当前画布 + 目标图像 + 坐标 + 目标边缘 + 任务时间 (10通道)+覆盖次数，float32，[0,1]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(11, canvas_size, canvas_size), dtype=np.uint8)

  

        # 生成归一化坐标通道 (2, H, W)
        xs = torch.linspace(0, 1, self.canvas_size)
        ys = torch.linspace(0, 1, self.canvas_size)
        coord_x = xs.view(1, 1, -1).expand(1, self.canvas_size, self.canvas_size)
        coord_y = ys.view(1, -1, 1).expand(1, self.canvas_size, self.canvas_size)
        self.coord = torch.cat([coord_x, coord_y], dim=0).to(self.device)  # (2, H, W)

        self.sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=self.device).view(1,1,3,3)
        self.sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=self.device).view(1,1,3,3)

        self.training = training # 默认为False，训练时外部可设为True

        # 覆盖计数
        self.cover_count = np.zeros((self.canvas_size, self.canvas_size), dtype=np.int32)

        # 区域多样性
        self.covered_centers = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.step_count = 0
        # 每次 reset 随机选一个目标图像
        if self.target_dataset is not None:
            idx = random.randint(0, len(self.target_dataset)-1)
            sample = self.target_dataset[idx]
            # 判断是Dataset还是路径
            if isinstance(sample, str):
                # 路径，需加载图片
                self.target = load_target_image(sample, size=self.canvas_size).to(self.device)
            elif isinstance(sample, tuple) or isinstance(sample, list):
                # Dataset返回 (img, label)
                self.target = sample[0].to(self.device)
            elif isinstance(sample, torch.Tensor):
                self.target = sample.to(self.device)
            else:
                raise ValueError("Unknown dataset sample type: {}".format(type(sample)))
            # 保证 self.target 是 [3, H, W]
            if self.target.dim() == 4:
                self.target = self.target.squeeze(0)
        else:
            idx = random.randint(0, len(self.targets)-1)
            self.target = self.targets[idx]
        
        
        
        #self.canvas = torch.ones(3, self.canvas_size, self.canvas_size, device=self.device)
        #渲染背景色
        main_color = self.get_weighted_main_color(self.target)  # shape: [3]
        self.canvas = main_color.view(3, 1, 1).expand(3, self.canvas_size, self.canvas_size).clone()
        


        # 重置覆盖计数
        self.cover_count = np.zeros((self.canvas_size, self.canvas_size), dtype=np.int32)

        # 清空区域多样性记录
        self.covered_centers = []

        return self._get_obs(), {}

    def _get_obs(self):
        # 计算目标图像边缘
        target_gray = self.target.mean(0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
        edge = F.conv2d(target_gray, self.sobel_x, padding=1) + F.conv2d(target_gray, self.sobel_y, padding=1)
        edge = edge.squeeze(0)  # [1,H,W]
        # 时间通道，归一化到[0,1]
        time_channel = torch.full(
            (1, self.canvas_size, self.canvas_size),
            fill_value=self.step_count / self.max_steps,
            device=self.device
        )
        # 拼接到obs
        threshold = 5  # 覆盖计数阈值
        cover_channel = torch.from_numpy(self.cover_count).float().unsqueeze(0).to(self.device) / threshold  # 归一化
        obs = torch.cat([
            self.canvas,
            self.target,      # 直接用 self.target
            self.coord,
            edge,
            time_channel,
            cover_channel
        ], dim=0).cpu().numpy()
        obs = np.clip(obs * 255, 0, 255).astype(np.uint8)
        #self.show_obs_layers(obs) #观测空间监测
        return obs

    def step(self, action):
        # 动作空间为 [shape_logits..., tx, ty, sx, sy, angle_sin, angle_cos, r, g, b]
        action = np.asarray(action)
        shape_logits = action[:self.num_shape_types]
        params = action[self.num_shape_types:]

        # 只缩放 tx, ty, sx, sy, r, g, b
        tx, ty, sx, sy, angle_sin, angle_cos, r, g, b = params
        tx = (tx + 1) / 2
        ty = (ty + 1) / 2
        sx = (sx + 1) / 2
        sy = (sy + 1) / 2
        # angle_sin, angle_cos 保持原样（[-1,1]）
        r = (r + 1) / 2
        g = (g + 1) / 2
        b = (b + 1) / 2

        params = [tx, ty, sx, sy, angle_sin, angle_cos, r, g, b]

        canvas_before = self.canvas.clone()

        logits = torch.tensor(shape_logits, device=self.device)
        logits = logits * 10
        tau = 1  # 温度参数，可调
        if self.training:  # 训练时用Gumbel-Softmax
            shape_probs = F.gumbel_softmax(logits, tau=tau, hard=False)  # [num_shape_types]
            self.canvas = self.render_single_soft(shape_probs, params, self.canvas)
        else:  # 推理/评估时仍用argmax
            shape_type = int(torch.argmax(logits).item())
            self.canvas = self.renderer.render_single(shape_type, params, self.canvas)



   
        reward = 0.0

        before = self.loss_fn(canvas_before.unsqueeze(0), self.target.unsqueeze(0)).item()
        after = self.loss_fn(self.canvas.unsqueeze(0), self.target.unsqueeze(0)).item()
        basic_reward = (after - before) * 50  # 基础奖励，提升越多奖励越高
        reward += basic_reward




        # 获取当前素材类型
        if self.training:
            chosen_type = int(torch.argmax(shape_probs).item())
        else:
            chosen_type = int(torch.argmax(logits).item())

        # 获取素材的 alpha 通道
        alpha_tensor = self.shape_textures[chosen_type][3:4]  # [1, H, W]



         # 仿射变换 alpha 通道，得到本次动作影响区域的 mask
        alpha_mask = self.get_mask_like_render(alpha_tensor, tx, ty, sx, sy, angle_sin, angle_cos, self.canvas_size)
        alpha_mask = (alpha_mask > 0.05).float()  # 可调阈值
        

        # 计算局部区域
        canvas_crop = self.canvas * alpha_mask
        target_crop = self.target * alpha_mask
        canvas_crop_before = canvas_before * alpha_mask
        mask_area = (alpha_mask > 0.05)
        # 展平空间维度，只在空间上用mask
        mask_area = (alpha_mask > 0.05)  # [H, W]
        #if mask_area.sum() > 0:
            # [3, H, W] -> [3, N]，N为mask内像素数
        #   canvas_crop_before_masked = canvas_crop_before[:, mask_area]  # [3, N]
        #    target_crop_masked = target_crop[:, mask_area]                # [3, N]
        #    mse_before = F.mse_loss(canvas_crop_before_masked, target_crop_masked)
        #    canvas_crop_masked = canvas_crop[:, mask_area]
        #    mse_after = F.mse_loss(canvas_crop_masked, target_crop_masked)
        #else:
        #    mse_before = mse_after = torch.tensor(0.0, device=self.device)

        # 计算中心权重
        center_x, center_y = 0.5, 0.5
        sigma = 0.25  # 可调
        dist2 = (tx - center_x) ** 2 + (ty - center_y) ** 2
        center_weight = np.exp(-dist2 / (2 * sigma ** 2))


        # 局部颜色一致性奖励
        if mask_area.sum() > 0:
            canvas_mean = canvas_crop[:, mask_area].mean(dim=1)
            canvas_before_mean = canvas_crop_before[:, mask_area].mean(dim=1)
            target_mean = target_crop[:, mask_area].mean(dim=1)
            color_diff = torch.norm(canvas_mean - target_mean, p=2)
            color_diff_before = torch.norm(canvas_before_mean - target_mean, p=2)
            # 根据区域大小调整颜色奖励
            area = mask_area.sum() /(mask_area.numel() + 1e-8)  # 计算覆盖区域占比
            size_weight = np.exp(-((area.cpu() - 0.05) ** 2) / (2 * 0.02 ** 2))  # 面积在5%左右时奖励最大
            
            min_area = 0.01 # 最小面积阈值
            max_aspect_ratio = 5

            # 面积权重
            if area < min_area:
                size_weight = 0.0
            else:
                size_weight = np.exp(-((area.cpu() - 0.04) ** 2) / (2 * 0.02 ** 2))

            # 细长惩罚
            aspect_ratio = max(sx, sy) / (min(sx, sy) + 1e-6)
            if aspect_ratio > max_aspect_ratio:
                size_weight *= np.exp(-((aspect_ratio - max_aspect_ratio) ** 2) / 2)

            # 计算mask区域内canvas颜色方差，方差越小越均匀
            num_pixels = mask_area.sum().item()
            if num_pixels <= 1:
                target_var = torch.tensor(0.0, device=self.device)
            else:
                target_var = target_crop[:, mask_area].var(dim=1, unbiased=False).mean()
            target_var = torch.clamp(target_var, min=0.0, max=1.0)
            uniformity_reward = float(np.exp(-target_var.item() * 10))  

            color_reward = (color_diff_before - color_diff) *(1 - color_diff) * size_weight * center_weight *uniformity_reward 
            color_reward = color_reward.cpu().item() * 5 # 系数可调

        else:
            color_reward = 0.0
        #print(f"Color Reward: {color_reward:.4f},  Basic Reward: {basic_reward:.4f}")

        # 加入reward
        reward += color_reward  

        # 显示 alpha_mask
        #plt.imshow(alpha_mask.cpu().numpy())
        # plt.show()
        



        # 添加参数变化惩罚
        penalty = 0.0
        if hasattr(self, "last_params") and self.step_count > 0:
            diff = np.linalg.norm(np.array(params) - np.array(self.last_params))
            if diff < 0.01:  # 只有非常小的变化才惩罚
                penalty = 1.0
            else:
                penalty = 0.0
            reward -= penalty
        self.last_params = params


 
        # 更新覆盖计数
        mask_np = alpha_mask.cpu().numpy().astype(bool)

        # 分阶段奖励
        new_cover = 0
        if self.step_count < self.max_steps * 0.3:
            new_cover = ((self.cover_count == 0) & mask_np).sum() *0.001 
            reward += new_cover  # 前期鼓励新覆盖
        else:
            # 后期提升细节相关reward权重
            reward += color_reward  # 再次奖励细节


        # 统计本轮有多少像素被多次覆盖  
        prev_cover_count = self.cover_count.copy()
        self.cover_count[mask_np] += 1
        overlap_pixels = ((prev_cover_count[mask_np] >= 5)).sum() # 覆盖阈值5次
        overlap_penalty = overlap_pixels * 0.001 # 每个像素扣0.001分
        reward -= overlap_penalty

        #print(f"Step {self.step_count}, Overlap Penalty: {overlap_penalty:.4f}, Total Reward: {reward:.4f}")

        center = (tx, ty)
        self.covered_centers.append(center)
        # 与历史中心点距离较远则奖励
        region_diversity_reward = 0.0
        if len(self.covered_centers) > 1:
            dists = [np.linalg.norm(np.array(center) - np.array(c)) for c in self.covered_centers[:-1]]
            min_dist = min(dists)
            region_diversity_reward = min(1.0, min_dist * 2.0) 
            reward += region_diversity_reward

        self.step_count += 1

        terminated =  (self.step_count >= self.max_steps) or (after > 0.85)  # 达到目标相似度
        truncated = False

        final_reward = 0
        if terminated or truncated:
            final_reward = (after-0.5) * 50 # 结束奖励
        reward += final_reward






        return self._get_obs(), reward, terminated, truncated, { 
            "basic_reward": basic_reward,
            "color_reward": color_reward,
            "new_cover_reward": new_cover,
            "overlap_penalty": -overlap_penalty,
            "param_penalty": -penalty,
            "region_diversity_reward": region_diversity_reward,
            "final_reward": final_reward,}

    def render_single_soft(self, shape_probs, params, canvas=None):
        # shape_probs: [num_shape_types]，每个素材的权重
        # params: [tx, ty, sx, sy, angle_sin, angle_cos, r, g, b]
        # 对每个素材分别渲染，然后加权求和
        canvases = []
        for i, prob in enumerate(shape_probs):
            if prob.item() < 1e-4:
                continue  # 跳过权重极小的
            canvases.append(prob * self.renderer.render_single(i, params, canvas.clone() if canvas is not None else None))
        if canvases:
            return sum(canvases)
        else:
            return canvas



    def get_mask_like_render(self, alpha_tensor, tx, ty, sx, sy, angle_sin, angle_cos, canvas_size):
        # alpha_tensor: [1, H, W]，原素材alpha
        # sx, sy: [0,1]，需做线性映射
        sx = 0.05 + float(sx) * 0.95
        sy = 0.05 + float(sy) * 0.95
        angle = np.arctan2(float(angle_sin), float(angle_cos)) * 180 / np.pi

        # resize
        h = int(canvas_size * sy)
        w = int(canvas_size * sx)
        alpha_resized = TF.resize(alpha_tensor, (h, w))

        # rotate
        alpha_rot = TF.rotate(alpha_resized, angle, expand=True)

        # 计算粘贴位置
        tx = float(tx) * canvas_size
        ty = float(ty) * canvas_size
        h2, w2 = alpha_rot.shape[1:]
        x0 = int(tx - w2 // 2)
        y0 = int(ty - h2 // 2)
        x1 = x0 + w2
        y1 = y0 + h2

        # 画布
        mask = torch.zeros(1, canvas_size, canvas_size, device=alpha_tensor.device)
        # 计算有效区域
        canvas_x0 = max(0, x0)
        canvas_y0 = max(0, y0)
        canvas_x1 = min(canvas_size, x1)
        canvas_y1 = min(canvas_size, y1)
        shape_x0 = max(0, -x0)
        shape_y0 = max(0, -y0)
        shape_x1 = shape_x0 + (canvas_x1 - canvas_x0)
        shape_y1 = shape_y0 + (canvas_y1 - canvas_y0)
        mask[:, canvas_y0:canvas_y1, canvas_x0:canvas_x1] = alpha_rot[:, shape_y0:shape_y1, shape_x0:shape_x1]
        return mask[0]

    # 假设 obs 是 (10, H, W) 的 numpy 数组
    import matplotlib.pyplot as plt

    def show_obs_layers(self, obs):
        num_layers = obs.shape[0]
        if not hasattr(self, "_fig") or self._fig is None:
            self._fig, self._axes = plt.subplots(1, num_layers, figsize=(3*num_layers, 3))
            if num_layers == 1:
                self._axes = [self._axes]
            plt.ion()
            plt.show()
        for i in range(num_layers):
            img = obs[i]
            ax = self._axes[i]
            ax.clear()
            if img.ndim == 2:
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                ax.imshow(img.transpose(1,2,0))
            ax.set_title(f'Layer {i}')
            ax.axis('off')
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def get_weighted_main_color(self,target_img):
        # target_img: [3, H, W], float, 0~1
        H, W = target_img.shape[1:]
        # 生成高斯权重（中心高，边缘低）
        y, x = np.ogrid[:H, :W]
        cy, cx = H / 2, W / 2
        sigma = min(H, W) / 3
        weight = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        weight = weight / weight.sum()
        # 展平
        pixels = target_img.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        weights = weight.reshape(-1)
        # 量化颜色
        quant = (pixels * 100).round() / 100
        # 统计加权众数
        from collections import Counter
        color_str = (quant * 255).astype(np.uint8).astype(str)
        color_str = np.char.add(np.char.add(color_str[:,0], ','), np.char.add(color_str[:,1], ',')) + color_str[:,2]
        # 用权重加权计数
        color_weight = {}
        for c, w in zip(color_str, weights):
            color_weight[c] = color_weight.get(c, 0) + w
        main_color_str = max(color_weight, key=color_weight.get)
        r, g, b = [int(x) for x in main_color_str.split(',')]
        main_color = torch.tensor([r, g, b], dtype=torch.float32, device=target_img.device) / 255.0
        return main_color




