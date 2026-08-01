import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Efficient Channel Attention (ECA) ---
class ECALayer(nn.Module):
    """Efficient Channel Attention module using 1D convolution"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k_size = t if t % 2 != 0 else t + 1
        k_size = max(3, k_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv1d(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)

# --- 2D Haar Discrete Wavelet Transform (DWT) ---
class HaarDWT2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (B, C, H, W)
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) / 2.0
        lh = (-x00 - x01 + x10 + x11) / 2.0
        hl = (-x00 + x01 - x10 + x11) / 2.0
        hh = (x00 - x01 - x10 + x11) / 2.0
        
        hf_energy = torch.mean(torch.abs(lh) + torch.abs(hl) + torch.abs(hh), dim=1, keepdim=True)
        return ll, hf_energy

# --- Modular CrossBlock (Spatial Branch) ---
class CrossBlockModular(nn.Module):
    def __init__(self, in_channels, out_channels, variant='v1'):
        super().__init__()
        self.variant = variant
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Convolutions
        if '1a' in variant or '1ab' in variant:
            # Multi-scale parallel strip convolutions (n=5, 9, 13)
            self.conv_m1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), padding=(0, 2))
            self.conv_m2 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0))
            self.conv_n1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 9), padding=(0, 4))
            self.conv_n2 = nn.Conv2d(in_channels, out_channels, kernel_size=(9, 1), padding=(4, 0))
            self.conv_k1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 13), padding=(0, 6))
            self.conv_k2 = nn.Conv2d(in_channels, out_channels, kernel_size=(13, 1), padding=(6, 0))
            self.fuse_conv = nn.Conv2d(out_channels * 6, out_channels, kernel_size=3, padding=1)
        else:
            # Original DBCNet single-scale (3x3, 1x9, 9x1)
            self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv1xk = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 9), padding=(0, 4))
            self.convkx1 = nn.Conv2d(in_channels, out_channels, kernel_size=(9, 1), padding=(4, 0))
            self.fuse_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)
            
        self.gn_fuse = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # ECA or Gate
        if '1b' in variant or '1ab' in variant:
            self.eca = ECALayer(out_channels)
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(out_channels, 3 if '1a' not in variant else 6, kernel_size=1),
                nn.Softmax(dim=1)
            )
        else:
            self.eca = None
            
        # Residual projection if channel count changes
        if ('1a' in variant or '1ab' in variant) and in_channels != out_channels:
            self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_proj = None

    def forward(self, x):
        if '1a' in self.variant or '1ab' in self.variant:
            f1 = self.relu(self.conv_m1(x))
            f2 = self.relu(self.conv_m2(x))
            f3 = self.relu(self.conv_n1(x))
            f4 = self.relu(self.conv_n2(x))
            f5 = self.relu(self.conv_k1(x))
            f6 = self.relu(self.conv_k2(x))
            feats = [f1, f2, f3, f4, f5, f6]
        else:
            f1 = self.relu(self.conv3x3(x))
            f2 = self.relu(self.conv1xk(x))
            f3 = self.relu(self.convkx1(x))
            feats = [f1, f2, f3]
            
        cat_feat = torch.cat(feats, dim=1)
        out = self.relu(self.gn_fuse(self.fuse_conv(cat_feat)))
        
        if self.eca is not None:
            out = self.eca(out)
            
        if '1a' in self.variant or '1ab' in self.variant:
            residual = self.res_proj(x) if self.res_proj is not None else x
            out = out + residual
            
        return out

# --- Pure PyTorch SS2D Block (Mamba Context Branch) ---
class SS2D_Approximation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv2d = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        x_proj = x_proj.permute(0, 3, 1, 2).contiguous()
        x_proj = self.act(self.conv2d(x_proj))
        x_proj = x_proj.permute(0, 2, 3, 1).contiguous()
        x_proj = x_proj * self.act(z)
        out = self.out_proj(x_proj)
        return out

class VSSBlockModular(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ss2d = SS2D_Approximation(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.ln(x_perm)
        out = self.ss2d(x_norm)
        out = self.dropout(out)
        out = out.permute(0, 3, 1, 2).contiguous()
        return x + out

# --- Modular Feature Fusion Module (FFM) ---
class FFMModular(nn.Module):
    def __init__(self, channels, variant='v1'):
        super().__init__()
        self.variant = variant
        if '3a' in variant or '3ab' in variant:
            self.atten_spatial = ECALayer(channels)
            self.atten_context = ECALayer(channels)
            self.hybrid_conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, groups=channels), # depthwise
                nn.Conv2d(channels, channels, kernel_size=1), # pointwise
                nn.GELU(),
                ECALayer(channels)
            )
        else:
            # Baseline SE attention
            self.atten_spatial = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(channels // 4, channels),
                nn.Sigmoid()
            )
            self.atten_context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(channels // 4, channels),
                nn.Sigmoid()
            )
            self.hybrid_conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
                nn.GELU()
            )

        if '3b' in variant or '3ab' in variant:
            self.cross_spatial_gate = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.cross_spatial_gate = None

    def forward(self, spatial_feat, context_feat):
        if '3a' in self.variant or '3ab' in self.variant:
            out_s = self.atten_spatial(spatial_feat)
            out_c = self.atten_context(context_feat)
        else:
            b, c, _, _ = spatial_feat.shape
            w_s = self.atten_spatial(spatial_feat).view(b, c, 1, 1)
            w_c = self.atten_context(context_feat).view(b, c, 1, 1)
            out_s = spatial_feat * w_s
            out_c = context_feat * w_c

        if self.cross_spatial_gate is not None:
            gate_s = self.cross_spatial_gate(spatial_feat)
            gate_c = self.cross_spatial_gate(context_feat)
            out_s = out_s * gate_c
            out_c = out_c * gate_s

        cat_feat = torch.cat([spatial_feat, context_feat], dim=1)
        out_h = self.hybrid_conv(cat_feat)
        return out_s + out_c + out_h

# --- Modular CMM Decoder Block ---
class CMMBlockModular(nn.Module):
    def __init__(self, channels, variant='v1'):
        super().__init__()
        self.variant = variant
        self.ln = nn.LayerNorm(channels)
        self.ss2d = SS2D_Approximation(channels)
        self.dropout = nn.Dropout(0.1)

        if ('4b' in variant or '4.3' in variant or '4all' in variant) and channels <= 64:
            # Lightweight depthwise CMM for high resolution
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                ECALayer(channels)
            )
        elif ('4.2' in variant or '4all' in variant):
            # CMM with ECA gating
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(min(32, channels), channels),
                nn.ReLU(inplace=True),
                ECALayer(channels)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
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
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# --- Full Modular WS-DBNet ---
class WS_DBNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_c=16,
                 phase1='v1', phase2='v1', phase3='v1', phase4='v1'):
        super().__init__()
        self.phase1 = phase1
        self.phase2 = phase2
        self.phase3 = phase3
        self.phase4 = phase4
        
        # Haar Wavelet module if Phase 2 enabled
        if '2a' in phase2 or '2ab' in phase2:
            self.dwt = HaarDWT2D()
            self.dwt_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        else:
            self.dwt = None

        # Spatial Branch (CrossNet+)
        self.s_layer1 = CrossBlockModular(in_channels, base_c, variant=phase1)
        self.s_down1 = nn.Sequential(nn.Conv2d(base_c, base_c*2, 3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*2), base_c*2))
        
        self.s_layer2 = CrossBlockModular(base_c*2, base_c*2, variant=phase1)
        self.s_down2 = nn.Sequential(nn.Conv2d(base_c*2, base_c*4, 3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*4), base_c*4))
        
        self.s_layer3 = CrossBlockModular(base_c*4, base_c*4, variant=phase1)
        self.s_down3 = nn.Sequential(nn.Conv2d(base_c*4, base_c*8, 3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*8), base_c*8))
        
        self.s_layer4 = CrossBlockModular(base_c*8, base_c*8, variant=phase1)
        self.s_down4 = nn.Sequential(nn.Conv2d(base_c*8, base_c*16, 3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*16), base_c*16))
        
        self.s_layer5 = CrossBlockModular(base_c*16, base_c*16, variant=phase1)

        # Context Branch (Wavelet-Mamba)
        self.c_layer1 = nn.Sequential(ConvBlock(in_channels, base_c), ConvBlock(base_c, base_c))
        self.c_down1 = nn.Sequential(nn.Conv2d(base_c, base_c*2, 3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*2), base_c*2))
        
        self.c_layer2 = nn.Sequential(ConvBlock(base_c*2, base_c*2), ConvBlock(base_c*2, base_c*2))
        self.c_down2 = nn.Sequential(nn.Conv2d(base_c*2, base_c*4, 3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*4), base_c*4))
        
        self.c_layer3 = nn.Sequential(*[VSSBlockModular(base_c*4) for _ in range(2)])
        self.c_down3 = nn.Sequential(nn.Conv2d(base_c*4, base_c*8, 3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*8), base_c*8))
        
        self.c_layer4 = nn.Sequential(*[VSSBlockModular(base_c*8) for _ in range(2)])
        self.c_down4 = nn.Sequential(nn.Conv2d(base_c*8, base_c*16, 3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*16), base_c*16))
        
        self.c_layer5 = nn.Sequential(*[VSSBlockModular(base_c*16) for _ in range(4)]) # scaled down for efficiency
        self.c_down5 = nn.Sequential(nn.Conv2d(base_c*16, base_c*32, 3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*32), base_c*32))
        
        self.c_layer6 = nn.Sequential(*[VSSBlockModular(base_c*32) for _ in range(2)])

        # Feature Fusion Modules (FFM+)
        self.ffm3 = FFMModular(base_c*4, variant=phase3)
        self.ffm4 = FFMModular(base_c*8, variant=phase3)
        self.ffm5 = FFMModular(base_c*16, variant=phase3)

        # Decoder (CMM+)
        self.up5 = nn.ConvTranspose2d(base_c*32, base_c*16, kernel_size=2, stride=2)
        self.dec5 = CMMBlockModular(base_c*16, variant=phase4)

        self.up4 = nn.ConvTranspose2d(base_c*16, base_c*8, kernel_size=2, stride=2)
        self.dec4 = CMMBlockModular(base_c*8, variant=phase4)

        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.dec3 = CMMBlockModular(base_c*4, variant=phase4)

        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, kernel_size=2, stride=2)
        if '4a' in phase4 or '4.1' in phase4 or '4.2' in phase4 or '4.3' in phase4 or '4b' in phase4 or '4all' in phase4:
            self.dec2 = CMMBlockModular(base_c*2, variant=phase4)
        else:
            self.dec2 = ConvBlock(base_c*2, base_c*2)

        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, kernel_size=2, stride=2)
        if '4a' in phase4 or '4.1' in phase4 or '4.2' in phase4 or '4.3' in phase4 or '4b' in phase4 or '4all' in phase4:
            self.dec1 = CMMBlockModular(base_c, variant=phase4)
        else:
            self.dec1 = ConvBlock(base_c, base_c)

        self.final_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        # Wavelet DWT preprocessing if enabled
        if self.dwt is not None:
            ll, hf_energy = self.dwt(x)
            # Re-scale LL back to 512x512 for fusion
            ll_up = F.interpolate(ll, size=x.shape[2:], mode='bilinear', align_corners=False)
            x_in = x + self.dwt_conv(ll_up)
        else:
            x_in = x

        # Spatial Branch
        s1 = self.s_layer1(x_in)
        s2 = self.s_layer2(self.s_down1(s1))
        s3 = self.s_layer3(self.s_down2(s2))
        s4 = self.s_layer4(self.s_down3(s3))
        s5 = self.s_layer5(self.s_down4(s4))

        # Context Branch
        c1 = self.c_layer1(x_in)
        c2 = self.c_layer2(self.c_down1(c1))
        c3 = self.c_layer3(self.c_down2(c2))
        c4 = self.c_layer4(self.c_down3(c3))
        c5 = self.c_layer5(self.c_down4(c4))
        c6 = self.c_layer6(self.c_down5(c5))

        # Fusion
        f3 = self.ffm3(s3, c3)
        f4 = self.ffm4(s4, c4)
        f5 = self.ffm5(s5, c5)

        # Decoder
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
    # Forward pass test for default and modified variants
    x = torch.randn(2, 3, 512, 512)
    model = WS_DBNet(phase1='1ab', phase2='2ab', phase3='3ab', phase4='4b')
    y = model(x)
    print("WS-DBNet Forward Test Successful! Output Shape:", y.shape)
    assert y.shape == (2, 1, 512, 512)
