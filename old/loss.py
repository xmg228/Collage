import lpips
import torch.nn as nn
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self, use_lpips=True):
        super().__init__()
        self.use_lpips = use_lpips
        if use_lpips:
            self.loss_fn = lpips.LPIPS(net='vgg').cuda()
        else:
            self.loss_fn = lambda a, b: F.mse_loss(a, b)

    def forward(self, pred, target):
        return self.loss_fn(pred, target)
