"""
Cascaded Multi-scale Module (CMM) Decoder for WS-DBNet
======================================================
This module provides the modular implementation of Section 4:
4.1: Full 5-Stage CMM Coverage (Multi-scale feature aggregation across all 5 decoder stages).
4.2: Efficient Channel Attention (ECA) 1D Conv gating in CMM Path Selection.
4.3: Progressive CMM (Heavy multi-scale state-space fusion at low resolution,
     lightweight depthwise convolution at high resolution).

Authors: Glacial Lake AI Research Team
Dataset: Remote Sensing Glacial Lake Segmentation (512x512)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA) Layer.
    Uses 1D convolution with adaptive kernel size k determined by channel dimension.
    Replaces parameter-heavy 2-layer MLPs to avoid channel dimensionality bottlenecks.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x) # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class SS2D_Approximation(nn.Module):
    """
    2D State-Space (SS2D) Spatial Mixing Module.
    Approximates multi-directional continuous state-space scanning with linear complexity.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv2d = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        # x shape: (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        x_proj = x_proj.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        x_proj = self.act(self.conv2d(x_proj))
        x_proj = x_proj.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)
        x_proj = x_proj * self.act(z)
        out = self.out_proj(x_proj)
        return out

class CMMBlockModular(nn.Module):
    """
    Modular Cascaded Multi-scale Module (CMM) Block.
    
    Parameters:
    -----------
    channels : int
        Channel dimension of the current decoder stage.
    variant : str
        Ablation configuration switch:
        - '4.1': 5-stage standard CMM coverage.
        - '4.2': CMM with ECA 1D conv gating in path selection.
        - '4.3': Progressive CMM (heavy at low-res, lightweight depthwise at high-res).
        - '4.1_4.3': Proposed Optimal Decoder (5-stage progressive CMM).
        - '4.2_4.3': Progressive CMM with ECA gating.
        - '4all': All decoder modifications enabled simultaneously.
    """
    def __init__(self, channels, variant='4.1_4.3'):
        super().__init__()
        self.variant = variant
        self.ln = nn.LayerNorm(channels)
        self.ss2d = SS2D_Approximation(channels)
        self.dropout = nn.Dropout(0.1)

        is_prog = any(k in variant for k in ['4b', '4.3', '4all', '4.1_4.3', '4.2_4.3'])
        is_eca = any(k in variant for k in ['4.2', '4all', '4.2_4.3'])

        if is_prog and channels <= 64:
            # High-resolution stages (<= 64 channels, e.g. Stage 1 & 2): Lightweight Depthwise Convolution
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                ECALayer(channels) if is_eca else nn.Identity()
            )
        elif is_eca:
            # Low/mid-resolution stages with ECA gating
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(32, channels), channels),
                nn.ReLU(inplace=True),
                ECALayer(channels)
            )
        else:
            # Standard CMM convolution block
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(32, channels), channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        res = x
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_ss2d = self.dropout(self.ss2d(self.ln(x_perm))).permute(0, 3, 1, 2).contiguous()
        x1 = res + x_ss2d
        out = x1 + self.conv(x1)
        return out

class ConvBlock(nn.Module):
    """Basic Convolution-BatchNorm-ReLU Block."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class WS_DBNet_Decoder(nn.Module):
    """
    Complete 5-Stage Progressive CMM Decoder Module.
    Takes multi-level encoder features (f3, f4, f5) and skip connections (c1, c2, c6)
    and reconstructs the full-resolution (512x512) binary segmentation mask.
    """
    def __init__(self, base_c=16, out_channels=1, variant='4.1_4.3'):
        super().__init__()
        self.variant = variant
        
        # Stage 5 (16x16 -> 32x32)
        self.up5 = nn.ConvTranspose2d(base_c*32, base_c*16, kernel_size=2, stride=2)
        self.dec5 = CMMBlockModular(base_c*16, variant=variant)

        # Stage 4 (32x32 -> 64x64)
        self.up4 = nn.ConvTranspose2d(base_c*16, base_c*8, kernel_size=2, stride=2)
        self.dec4 = CMMBlockModular(base_c*8, variant=variant)

        # Stage 3 (64x64 -> 128x128)
        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.dec3 = CMMBlockModular(base_c*4, variant=variant)

        # Stage 2 (128x128 -> 256x256)
        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, kernel_size=2, stride=2)
        if any(k in variant for k in ['4a', '4.1', '4.2', '4.3', '4b', '4all', '4.1_4.3', '4.2_4.3']):
            self.dec2 = CMMBlockModular(base_c*2, variant=variant)
        else:
            self.dec2 = ConvBlock(base_c*2, base_c*2)

        # Stage 1 (256x256 -> 512x512)
        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, kernel_size=2, stride=2)
        if any(k in variant for k in ['4a', '4.1', '4.2', '4.3', '4b', '4all', '4.1_4.3', '4.2_4.3']):
            self.dec1 = CMMBlockModular(base_c, variant=variant)
        else:
            self.dec1 = ConvBlock(base_c, base_c)

        # Final 1x1 projection
        self.final_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, c6, f5, f4, f3, c2, c1):
        d5 = self.up5(c6) + f5
        d5 = self.dec5(d5)

        d4 = self.up4(d5) + f4
        d4 = self.dec4(d4)

        d3 = self.up3(d4) + f3
        d3 = self.dec3(d3)

        d2 = self.up2(d3) + c2
        d2 = self.dec2(d2)

        d1 = self.up1(d2) + c1
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return out

if __name__ == '__main__':
    # Unit test decoder standalone
    base_c = 16
    decoder = WS_DBNet_Decoder(base_c=base_c, out_channels=1, variant='4.1_4.3')
    
    # Mock encoder inputs
    c6 = torch.randn(2, base_c*32, 16, 16)
    f5 = torch.randn(2, base_c*16, 32, 32)
    f4 = torch.randn(2, base_c*8, 64, 64)
    f3 = torch.randn(2, base_c*4, 128, 128)
    c2 = torch.randn(2, base_c*2, 256, 256)
    c1 = torch.randn(2, base_c, 512, 512)

    output = decoder(c6, f5, f4, f3, c2, c1)
    print(f"Decoder Test Successful! Output Shape: {output.shape}")
    assert output.shape == (2, 1, 512, 512), "Decoder output shape mismatch!"
