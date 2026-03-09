import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock3D(nn.Module):
    """3D Attention Gate (AG) for skip connections"""

    def __init__(self, in_channels, gate_channels):
        super().__init__()
        self.conv_theta = nn.Conv3d(in_channels, gate_channels, kernel_size=1)
        self.conv_phi = nn.Conv3d(gate_channels, gate_channels, kernel_size=1)
        self.conv_psi = nn.Conv3d(gate_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # x: skip connection features (e.g., enc4)
        # g: gating signal from decoder (e.g., upconv4 output)
        theta = self.conv_theta(x)  # (B, gate_channels, D, H, W)
        phi = self.conv_phi(g)  # (B, gate_channels, D, H, W)
        psi = self.relu(theta + phi)
        attn = self.sigmoid(self.conv_psi(psi))  # (B, 1, D, H, W)
        return x * attn  # 注意力加权后的特征


class ResidualBlock3D(nn.Module):
    """Basic 3D Residual Block with optional channel adjustment"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                   bias=False) if in_channels != out_channels else None
        self.skip_bn = nn.BatchNorm3d(out_channels) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
            identity = self.skip_bn(identity)

        out += identity
        out = self.relu(out)
        return out


class ResidualUNet3D(nn.Module):
    """3D Residual U-Net with Attention Gates"""

    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        super().__init__()
        features = init_features

        # Encoder Path
        self.encoder1 = ResidualBlock3D(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = ResidualBlock3D(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = ResidualBlock3D(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.encoder4 = ResidualBlock3D(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(features * 8, features * 16)

        # Decoder Path with Attention Gates
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.attn4 = AttentionBlock3D(features * 8, features * 8)  # 添加注意力门
        self.decoder4 = ResidualBlock3D(features * 16, features * 8)  # 输入通道翻倍（拼接后）

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.attn3 = AttentionBlock3D(features * 4, features * 4)
        self.decoder3 = ResidualBlock3D(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.attn2 = AttentionBlock3D(features * 2, features * 2)
        self.decoder2 = ResidualBlock3D(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.attn1 = AttentionBlock3D(features, features)
        self.decoder1 = ResidualBlock3D(features * 2, features)

        # Output
        self.conv_final = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # (B, f, D, H, W)
        enc2 = self.encoder2(self.pool1(enc1))  # (B, 2f, D/2, H/2, W/2)
        enc3 = self.encoder3(self.pool2(enc2))  # (B, 4f, D/4, H/4, W/4)
        enc4 = self.encoder4(self.pool3(enc3))  # (B, 8f, D/8, H/8, W/8)

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))  # (B, 16f, D/16, H/16, W/16)

        # Decoder with Attention
        dec4 = self.upconv4(bottleneck)  # (B, 8f, D/8, H/8, W/8)
        enc4_attn = self.attn4(enc4, dec4)  # 注意力加权编码器特征
        dec4 = torch.cat((dec4, enc4_attn), dim=1)  # (B, 16f, D/8, H/8, W/8)
        dec4 = self.decoder4(dec4)  # (B, 8f, D/8, H/8, W/8)

        dec3 = self.upconv3(dec4)  # (B, 4f, D/4, H/4, W/4)
        enc3_attn = self.attn3(enc3, dec3)
        dec3 = torch.cat((dec3, enc3_attn), dim=1)  # (B, 8f, D/4, H/4, W/4)
        dec3 = self.decoder3(dec3)  # (B, 4f, D/4, H/4, W/4)

        dec2 = self.upconv2(dec3)  # (B, 2f, D/2, H/2, W/2)
        enc2_attn = self.attn2(enc2, dec2)
        dec2 = torch.cat((dec2, enc2_attn), dim=1)  # (B, 4f, D/2, H/2, W/2)
        dec2 = self.decoder2(dec2)  # (B, 2f, D/2, H/2, W/2)

        dec1 = self.upconv1(dec2)  # (B, f, D, H, W)
        enc1_attn = self.attn1(enc1, dec1)
        dec1 = torch.cat((dec1, enc1_attn), dim=1)  # (B, 2f, D, H, W)
        dec1 = self.decoder1(dec1)  # (B, f, D, H, W)

        # Output
        out = self.conv_final(dec1)  # (B, out_channels, D, H, W)
        return out


# 使用示例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet3D(in_channels=1, out_channels=3).to(device)
    x = torch.randn(1, 1, 64, 64, 64).to(device)  # 模拟输入数据
    y = model(x)
    print(f"Output shape: {y.shape}")  # 应输出: torch.Size([1, 3, 64, 64, 64])