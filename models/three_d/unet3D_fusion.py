import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Basic convolutional block: (Conv3d -> BN -> ReLU) x 2
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    """
    Encoder block: pooling + conv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class DecoderBlock(nn.Module):
    """
    Decoder block:
    - upconv: ConvTranspose3d(up_in, out_ch)
    - conv: ConvBlock(skip_in + out_ch, out_ch)
    """
    def __init__(self, up_in, skip_in, out_ch):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(up_in, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(skip_in + out_ch, out_ch)

    def forward(self, x, skip):
        x = self.upconv(x)
        # align spatial dims
        if x.shape[2:] != skip.shape[2:]:
            diffs = [skip.size(i+2) - x.size(i+2) for i in range(3)]
            pad = []
            for diff in diffs[::-1]: pad += [diff//2, diff - diff//2]
            x = F.pad(x, pad)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class FusionUNet3(nn.Module):
    """
    3-class 3D U-Net with multi-scale encoder fusion in skip connections
    """
    def __init__(self, in_ch=1, base_f=32, num_classes=3):
        super().__init__()
        # Encoder
        self.init_conv = ConvBlock(in_ch, base_f)
        self.enc1 = EncoderBlock(base_f, base_f*2)
        self.enc2 = EncoderBlock(base_f*2, base_f*4)
        self.enc3 = EncoderBlock(base_f*4, base_f*8)
        self.enc4 = EncoderBlock(base_f*8, base_f*16)
        # Bottleneck
        self.bottleneck = ConvBlock(base_f*16, base_f*32)
        # Decoder: define skip input channels after fusion
        # fuse4: x4 (16f) + x2_ds (4f) => 20f
        self.dec4 = DecoderBlock(up_in=base_f*32, skip_in=base_f*16+base_f*4, out_ch=base_f*16)
        # fuse3: d4 (16f) + x3 (8f) + x1_ds (2f) => 26f
        self.dec3 = DecoderBlock(up_in=base_f*16, skip_in=base_f*16+base_f*8+base_f*2, out_ch=base_f*8)
        # fuse2: d3 (8f) + x2 (4f) + x0_ds (1f) => 13f
        self.dec2 = DecoderBlock(up_in=base_f*8, skip_in=base_f*8+base_f*4+base_f, out_ch=base_f*4)
        # fuse1: d2 (4f) + x1 (2f) => 6f
        self.dec1 = DecoderBlock(up_in=base_f*4, skip_in=base_f*4+base_f*2, out_ch=base_f*2)
        # Final
        self.final_conv = nn.Conv3d(base_f*2, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x0 = self.init_conv(x)       # f
        x1 = self.enc1(x0)           # 2f
        x2 = self.enc2(x1)           # 4f
        x3 = self.enc3(x2)           # 8f
        x4 = self.enc4(x3)           # 16f
        # Bottleneck
        xb = self.bottleneck(x4)     # 32f
        # Prepare multi-scale skips
        x2_ds = F.interpolate(x2, size=x4.shape[2:], mode='trilinear', align_corners=False)
        fuse4 = torch.cat([x4, x2_ds], dim=1)
        d4 = self.dec4(xb, fuse4)    # out 16f

        x1_ds = F.interpolate(x1, size=d4.shape[2:], mode='trilinear', align_corners=False)
        fuse3 = torch.cat([d4, x3, x1_ds], dim=1)
        d3 = self.dec3(d4, fuse3)    # out 8f

        x0_ds = F.interpolate(x0, size=d3.shape[2:], mode='trilinear', align_corners=False)
        fuse2 = torch.cat([d3, x2, x0_ds], dim=1)
        d2 = self.dec2(d3, fuse2)    # out 4f

        fuse1 = torch.cat([d2, x1], dim=1)
        d1 = self.dec1(d2, fuse1)    # out 2f

        return self.final_conv(d1)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionUNet3(in_ch=1, base_f=16, num_classes=3).to(device)
    x = torch.randn(1, 1, 128, 128, 128).to(device)
    y = model(x)
    print('Output shape:', y.shape)  # [1,3,128,128,128]
