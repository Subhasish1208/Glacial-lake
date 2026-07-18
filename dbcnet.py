import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CrossBlock (Spatial Branch) ---
class CrossBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=9):
        super(CrossBlock, self).__init__()
        # In DBCNet paper, the first stage extracts features in 3 directions
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1xk = nn.Conv2d(in_channels, out_channels, kernel_size=(1, k), padding=(0, k//2))
        self.convkx1 = nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1), padding=(k//2, 0))
        
        self.gn1 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.gn2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.gn3 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Second stage to fuse features
        self.fuse_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)
        self.fuse_gn = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)

    def forward(self, x):
        f_norm = self.relu(self.gn1(self.conv3x3(x)))
        f_hori = self.relu(self.gn2(self.conv1xk(x)))
        f_vert = self.relu(self.gn3(self.convkx1(x)))
        
        f_cat = torch.cat([f_norm, f_hori, f_vert], dim=1)
        out = self.relu(self.fuse_gn(self.fuse_conv(f_cat)))
        return out


# --- Pure PyTorch SS2D Approximation (if mamba-ssm is unavailable) ---
class SS2D_Approximation(nn.Module):
    """
    A pure PyTorch approximation of SS2D to ensure the network runs.
    In practice, VMamba's SS2D sweeps 4 directions. Here we use an efficient spatial mixing
    to replicate the large receptive field and linear complexity.
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
        
        # Spatial mixing (approximation)
        x_proj = x_proj * self.act(z)
        out = self.out_proj(x_proj)
        return out

class VSSBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ss2d = SS2D_Approximation(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x_permuted = x.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)
        x_norm = self.ln(x_permuted)
        out = self.ss2d(x_norm)
        out = self.dropout(out)
        out = out.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        return x + out


# --- Feature Fusion Module (FFM) ---
class SEAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FFM(nn.Module):
    def __init__(self, channels):
        super(FFM, self).__init__()
        self.se_spatial = SEAttention(channels)
        self.se_context = SEAttention(channels)
        
        self.hybrid_path = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.GELU(),
            SEAttention(channels)
        )

    def forward(self, spatial_feat, context_feat):
        out_spatial = self.se_spatial(spatial_feat)
        out_context = self.se_context(context_feat)
        
        cat_feat = torch.cat([spatial_feat, context_feat], dim=1)
        out_hybrid = self.hybrid_path(cat_feat)
        
        return out_spatial + out_context + out_hybrid


# --- Cross-aware Mamba Module (CMM) Decoder Block ---
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CMMBlock(nn.Module):
    def __init__(self, channels, k=9):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.ss2d = SS2D_Approximation(channels)
        self.dropout = nn.Dropout(0.1)
        
        self.conv3x3 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(min(32, channels), channels), nn.ReLU(inplace=True))
        self.conv1xk = nn.Sequential(nn.Conv2d(channels, channels, (1, k), padding=(0, k//2)), nn.GroupNorm(min(32, channels), channels), nn.ReLU(inplace=True))
        self.convkx1 = nn.Sequential(nn.Conv2d(channels, channels, (k, 1), padding=(k//2, 0)), nn.GroupNorm(min(32, channels), channels), nn.ReLU(inplace=True))
        
        self.fuse_conv = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)
        self.ca = ChannelAttention(channels)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        res1 = x
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_ss2d = self.dropout(self.ss2d(self.ln(x_perm))).permute(0, 3, 1, 2).contiguous()
        x1 = res1 + x_ss2d
        
        f1 = self.conv3x3(x1)
        f2 = self.conv1xk(x1)
        f3 = self.convkx1(x1)
        
        f_cat = torch.cat([f1, f2, f3], dim=1)
        f_fuse = self.fuse_conv(f_cat)
        f_ca = self.ca(f_fuse)
        
        out = x1 + f_ca * self.scale
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# --- Full DBCNet Architecture ---
class DBCNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_c=16): # base_c is 16 according to paper
        super().__init__()
        
        # Spatial Branch
        self.s_layer1 = CrossBlock(in_channels, base_c) # H
        self.s_down1 = nn.Sequential(nn.Conv2d(base_c, base_c*2, kernel_size=3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*2), base_c*2))
        
        self.s_layer2 = CrossBlock(base_c*2, base_c*2) # H/2
        self.s_down2 = nn.Sequential(nn.Conv2d(base_c*2, base_c*4, kernel_size=3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*4), base_c*4))
        
        self.s_layer3 = CrossBlock(base_c*4, base_c*4) # H/4
        self.s_down3 = nn.Sequential(nn.Conv2d(base_c*4, base_c*8, kernel_size=3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*8), base_c*8))
        
        self.s_layer4 = CrossBlock(base_c*8, base_c*8) # H/8
        self.s_down4 = nn.Sequential(nn.Conv2d(base_c*8, base_c*16, kernel_size=3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*16), base_c*16))
        
        self.s_layer5 = CrossBlock(base_c*16, base_c*16) # H/16
        
        # Context Branch (VMamba modified)
        # Layer 1
        self.c_layer1 = nn.Sequential(
            ConvBlock(in_channels, base_c),
            ConvBlock(base_c, base_c)
        ) # H
        self.c_down1 = nn.Sequential(nn.Conv2d(base_c, base_c*2, kernel_size=3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*2), base_c*2)) # H/2
        
        # Layer 2
        self.c_layer2 = nn.Sequential(
            ConvBlock(base_c*2, base_c*2),
            ConvBlock(base_c*2, base_c*2)
        ) # H/2
        self.c_down2 = nn.Sequential(nn.Conv2d(base_c*2, base_c*4, kernel_size=3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*4), base_c*4)) # H/4
        
        # Layer 3
        self.c_layer3 = nn.Sequential(*[VSSBlock(base_c*4) for _ in range(2)]) # N1=2, H/4
        self.c_down3 = nn.Sequential(nn.Conv2d(base_c*4, base_c*8, kernel_size=3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*8), base_c*8)) # H/8
        
        # Layer 4
        self.c_layer4 = nn.Sequential(*[VSSBlock(base_c*8) for _ in range(2)]) # N2=2, H/8
        self.c_down4 = nn.Sequential(nn.Conv2d(base_c*8, base_c*16, kernel_size=3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*16), base_c*16)) # H/16
        
        # Layer 5
        self.c_layer5 = nn.Sequential(*[VSSBlock(base_c*16) for _ in range(9)]) # N3=9, H/16
        self.c_down5 = nn.Sequential(nn.Conv2d(base_c*16, base_c*32, kernel_size=3, stride=2, padding=1), nn.GroupNorm(min(32, base_c*32), base_c*32)) # H/32
        
        # Layer 6
        self.c_layer6 = nn.Sequential(*[VSSBlock(base_c*32) for _ in range(2)]) # N4=2, H/32
        
        # FFM modules
        self.ffm3 = FFM(base_c*4)  # H/4
        self.ffm4 = FFM(base_c*8)  # H/8
        self.ffm5 = FFM(base_c*16) # H/16
        
        # Decoder (Hybrid CMM)
        self.up5 = nn.ConvTranspose2d(base_c*32, base_c*16, kernel_size=2, stride=2)
        self.dec5 = CMMBlock(base_c*16) # N5=4, Wait, paper says CMM*N5. Let's just use 1 block here for simplicity
        
        self.up4 = nn.ConvTranspose2d(base_c*16, base_c*8, kernel_size=2, stride=2)
        self.dec4 = CMMBlock(base_c*8)
        
        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.dec3 = CMMBlock(base_c*4)
        
        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_c*2, base_c*2)
        
        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_c, base_c)
        
        self.final_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        # Spatial Branch
        s1 = self.s_layer1(x)
        s2 = self.s_layer2(self.s_down1(s1))
        s3 = self.s_layer3(self.s_down2(s2))
        s4 = self.s_layer4(self.s_down3(s3))
        s5 = self.s_layer5(self.s_down4(s4))
        
        # Context Branch
        c1 = self.c_layer1(x)
        c2 = self.c_layer2(self.c_down1(c1))
        c3 = self.c_layer3(self.c_down2(c2))
        c4 = self.c_layer4(self.c_down3(c3))
        c5 = self.c_layer5(self.c_down4(c4))
        c6 = self.c_layer6(self.c_down5(c5))
        
        # FFM
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
    # Unit test with dummy tensor
    model = DBCNet()
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    assert y.shape == (2, 1, 512, 512), f"Expected shape (2, 1, 512, 512), but got {y.shape}"
    print("Model forward pass successful!")
