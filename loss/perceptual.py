import torch
import torch.nn as nn
import torchmetrics
#import lpips

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual_reward = perceptual_reward
        #self.lpips_fn = lpips.LPIPS(net='vgg').eval()

    def forward(self, pred, target):
        reward = self.perceptual_reward(
        pred, target,
        #pips_fn=self.lpips_fn,
        )
        return reward



def perceptual_reward(
    pred, target,
    ssim_weight=0.25, edge_weight=0.05, lpips_weight=0.00,color_weight=0.25, color_distance_weight=0.45,
    lpips_fn=None
):
    # SSIM
    ssim = torchmetrics.functional.structural_similarity_index_measure(pred, target)
    # 边缘SSIM
    pred_gray = pred.mean(1, keepdim=True)
    target_gray = target.mean(1, keepdim=True)
    sobel_x = pred.new_tensor([[1,0,-1],[2,0,-2],[1,0,-1]]).view(1,1,3,3)
    sobel_y = pred.new_tensor([[1,2,1],[0,0,0],[-1,-2,-1]]).view(1,1,3,3)
    pred_edge = nn.functional.conv2d(pred_gray, sobel_x, padding=1) + nn.functional.conv2d(pred_gray, sobel_y, padding=1)
    target_edge = nn.functional.conv2d(target_gray, sobel_x, padding=1) + nn.functional.conv2d(target_gray, sobel_y, padding=1)
    edge_ssim = torchmetrics.functional.structural_similarity_index_measure(pred_edge, target_edge)
    # LPIPS
    lpips_score = 0
    if lpips_fn is not None:
        pred_lpips = pred * 2 - 1
        target_lpips = target * 2 - 1
        lpips_score = lpips_fn(pred_lpips, target_lpips).mean()
    # 颜色分布距离
    color_reward = 0
    pred_hist = color_histogram(pred)
    target_hist = color_histogram(target)
    #print(pred_hist, target_hist)
    hist_dist = torch.norm(pred_hist - target_hist, p=2)
    color_reward = 1 - hist_dist  # 距离越小reward越高

    # 均方误差
    color_distance_reward = blockwise_mean_color_distance(pred, target)
   
    # 组合
    reward = (
        ssim_weight * ssim +
        edge_weight * edge_ssim +
        lpips_weight * (1 - lpips_score) +
        color_weight * color_reward +
        color_distance_weight * color_distance_reward
    )
     # 打印各项指标
    #print(f"ssim: {ssim_weight * ssim:.4f}, edge_ssim: {edge_weight * edge_ssim:.4f}, lpips: {lpips_weight * (1 - lpips_score):.4f}, color: {color_weight * color_reward:.4f},mse: {mse_weight * (1 - mse):.4f}, total: {reward:.4f}")
    return reward

def color_histogram(x, bins=8):
    # x: [B, C, H, W], [0,1]
    B, C, H, W = x.shape
    hists = []
    for c in range(C):
        # 合并 batch 统计全局直方图
        hist = torch.histc(x[:,c,:,:], bins=bins, min=0, max=1)
        hist = hist / (hist.sum() + 1e-8)
        hists.append(hist)
    return torch.cat(hists)  # [C*bins]

def blockwise_mean_color_distance(pred, target, num_blocks=8):
    # pred, target: [B, C, H, W], [0,1]
    B, C, H, W = pred.shape
    block_h = H // num_blocks
    block_w = W // num_blocks
    dist_sum = 0.0
    for i in range(num_blocks):
        for j in range(num_blocks):
            pred_block = pred[:, :, i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            target_block = target[:, :, i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            pred_mean = pred_block.mean(dim=[2,3])  # [B, C]
            target_mean = target_block.mean(dim=[2,3])
            dist = torch.norm(pred_mean - target_mean, p=2, dim=1)  # [B]
            dist_sum += dist
    dist_avg = dist_sum / (num_blocks * num_blocks)
    # 奖励可设为 1 - dist_avg（归一化后），或直接用 -dist_avg
    return 1 - dist_avg.mean()



