import torch
import torch.nn as nn
import torch.nn.functional as F

class SS2D_Approximation(nn.Module):
    """
    A pure PyTorch approximation of SS2D to ensure the network runs out-of-the-box on Windows.
    Uses linear projections, depthwise 2D convolutions, and SiLU activations to approximate
    the 2D state-space scanning mechanism.
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

class VSSBlock(nn.Module):
    """
    Visual State Space Block.
    """
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

class PatchEmbed(nn.Module):
    """
    Stem block to project input image to feature map.
    """
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.GroupNorm(num_groups=min(32, embed_dim), num_channels=embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class DownsampleBlock(nn.Module):
    """
    Downsampling block to reduce resolution by 2 and increase channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.down(x)

class DoubleConv(nn.Module):
    """
    Helper block for decoder convolutions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class VMambaEncoder(nn.Module):
    """
    Hierarchical VMamba (VSSM) Encoder producing multi-scale feature maps.
    """
    def __init__(self, in_channels=3, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512]):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=in_channels, embed_dim=dims[0], patch_size=4)
        
        # Stage 1
        self.stage1 = nn.Sequential(*[VSSBlock(dims[0]) for _ in range(depths[0])])
        
        # Stage 2
        self.down1 = DownsampleBlock(dims[0], dims[1])
        self.stage2 = nn.Sequential(*[VSSBlock(dims[1]) for _ in range(depths[1])])
        
        # Stage 3
        self.down2 = DownsampleBlock(dims[1], dims[2])
        self.stage3 = nn.Sequential(*[VSSBlock(dims[2]) for _ in range(depths[2])])
        
        # Stage 4
        self.down3 = DownsampleBlock(dims[2], dims[3])
        self.stage4 = nn.Sequential(*[VSSBlock(dims[3]) for _ in range(depths[3])])

    def forward(self, x):
        # input: B, 3, 512, 512
        x1 = self.patch_embed(x)      # B, 64, 128, 128
        x1 = self.stage1(x1)
        
        x2 = self.down1(x1)           # B, 128, 64, 64
        x2 = self.stage2(x2)
        
        x3 = self.down2(x2)           # B, 256, 32, 32
        x3 = self.stage3(x3)
        
        x4 = self.down3(x3)           # B, 512, 16, 16
        x4 = self.stage4(x4)
        
        return [x1, x2, x3, x4]

class VMambaSeg(nn.Module):
    """
    VMamba Segmentation Model (Encoder-Decoder architecture).
    """
    def __init__(self, in_channels=3, out_channels=1, dims=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = VMambaEncoder(in_channels=in_channels, dims=dims)
        
        # Decoder blocks
        # Bottleneck to Stage 3 Skip Connection
        self.up3 = nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(dims[2] + dims[2], dims[2]) # 256 + 256 -> 256
        
        # Stage 3 to Stage 2 Skip Connection
        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(dims[1] + dims[1], dims[1]) # 128 + 128 -> 128
        
        # Stage 2 to Stage 1 Skip Connection
        self.up1 = nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(dims[0] + dims[0], dims[0]) # 64 + 64 -> 64
        
        # Upsample back to original image resolution (512x512)
        self.up_final1 = nn.ConvTranspose2d(dims[0], dims[0] // 2, kernel_size=2, stride=2) # 64 -> 32 (256x256)
        self.dec_final1 = DoubleConv(dims[0] // 2, dims[0] // 2)
        
        self.up_final2 = nn.ConvTranspose2d(dims[0] // 2, dims[0] // 4, kernel_size=2, stride=2) # 32 -> 16 (512x512)
        self.dec_final2 = DoubleConv(dims[0] // 4, dims[0] // 4)
        
        # Final classifier head
        self.final_conv = nn.Conv2d(dims[0] // 4, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        features = self.encoder(x)
        x1, x2, x3, x4 = features # Resolutions: 128x128, 64x64, 32x32, 16x16
        
        # Decoder forward pass with skip connections
        d3 = self.up3(x4) # 16x16 -> 32x32
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3) # 32x32 -> 64x64
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2) # 64x64 -> 128x128
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)
        
        # Final upsampling to original input size
        f1 = self.up_final1(d1) # 128x128 -> 256x256
        f1 = self.dec_final1(f1)
        
        f2 = self.up_final2(f1) # 256x256 -> 512x512
        f2 = self.dec_final2(f2)
        
        out = self.final_conv(f2)
        return out

if __name__ == '__main__':
    # Shape verification test
    model = VMambaSeg()
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    assert y.shape == (2, 1, 512, 512), f"Expected shape (2, 1, 512, 512), but got {y.shape}"
    print("VMambaSeg model forward pass successful!")
