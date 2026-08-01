import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        intersection = torch.sum(predict * target)
        union = torch.sum(predict) + torch.sum(target)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

class SoftSkeletonize(nn.Module):
    def __init__(self, iters=4):
        super().__init__()
        self.iters = iters

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        return img

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        return img

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img):
        skel = torch.zeros_like(img)
        curr = img
        for _ in range(self.iters):
            eroded = self.soft_erode(curr)
            opened = self.soft_open(eroded)
            delta = F.relu(eroded - opened)
            skel = torch.max(skel, delta)
            curr = eroded
        return skel

class SoftclDiceLoss(nn.Module):
    def __init__(self, iters=4, smooth=1e-5):
        super().__init__()
        self.skel = SoftSkeletonize(iters=iters)
        self.smooth = smooth

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        
        s_prec = self.skel(predict)
        s_targ = self.skel(target)
        
        t_prec = (torch.sum(predict * s_targ) + self.smooth) / (torch.sum(s_targ) + self.smooth)
        t_sens = (torch.sum(s_prec * target) + self.smooth) / (torch.sum(s_prec) + self.smooth)
        
        cldice = (2.0 * t_prec * t_sens + self.smooth) / (t_prec + t_sens + self.smooth)
        return 1.0 - cldice

class BoundaryWeightedBCELoss(nn.Module):
    def __init__(self, alpha=2.0, sigma=3.0):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def _compute_weight_map(self, target_np):
        # target_np: (B, H, W)
        B, H, W = target_np.shape
        weight_maps = np.ones((B, H, W), dtype=np.float32)
        for i in range(B):
            mask = target_np[i] > 0.5
            if np.any(mask) and not np.all(mask):
                dist_ext = distance_transform_edt(~mask)
                dist_int = distance_transform_edt(mask)
                dist = dist_ext + dist_int
                w = 1.0 + self.alpha * np.exp(-(dist ** 2) / (2.0 * (self.sigma ** 2)))
                weight_maps[i] = w
        return weight_maps

    def forward(self, predict, target):
        # predict: logits (B, 1, H, W), target: (B, 1, H, W)
        target_np = target.squeeze(1).detach().cpu().numpy()
        w_np = self._compute_weight_map(target_np)
        weight_map = torch.from_numpy(w_np).to(predict.device).unsqueeze(1)
        
        bce = F.binary_cross_entropy_with_logits(predict, target, reduction='none')
        weighted_bce = torch.mean(bce * weight_map)
        return weighted_bce

class CombinedWSLoss(nn.Module):
    def __init__(self, use_boundary=False, use_cldice=False, lambda_cldice=0.5):
        super().__init__()
        self.bce_std = nn.BCEWithLogitsLoss()
        self.bce_bnd = BoundaryWeightedBCELoss(alpha=2.0, sigma=3.0) if use_boundary else None
        self.dice = DiceLoss()
        self.cldice = SoftclDiceLoss(iters=4) if use_cldice else None
        self.lambda_cldice = lambda_cldice

    def forward(self, predict, target):
        loss_bce = self.bce_bnd(predict, target) if self.bce_bnd is not None else self.bce_std(predict, target)
        loss_dice = self.dice(predict, target)
        total_loss = loss_bce + loss_dice
        
        if self.cldice is not None:
            loss_cl = self.cldice(predict, target)
            total_loss = total_loss + self.lambda_cldice * loss_cl
            
        return total_loss
