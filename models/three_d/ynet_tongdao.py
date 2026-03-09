import torch
import torch.nn as nn
from collections import OrderedDict

# =====通道注意力，带步长的卷积
# v2 SE3D 模块保持不变  带步长的卷积
class SE3D(nn.Module):
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        hidden = max(channels // r, 1)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg(x)
        w = self.fc(w)
        return x * w


# _block 方法保持不变
def _block(in_channels, features, name, se_reduction=8):
    return nn.Sequential(
        OrderedDict(
            [
                (name + "conv1",
                 nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=True)),
                (name + "norm1", nn.BatchNorm3d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (name + "conv2",
                 nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)),
                (name + "norm2", nn.BatchNorm3d(num_features=features)),
                (name + "se", SE3D(features, r=se_reduction)),
                (name + "relu2", nn.ReLU(inplace=True)),
            ]
        )
    )


class UNet3DDualDecoder(nn.Module):  # 新的类名
    def __init__(self, in_channels=1, out_channels=3, init_features=64, se_reduction=8):
        super(UNet3DDualDecoder, self).__init__()

        features = init_features

        # 编码器（共享）
        self.encoder1 = _block(in_channels, features, name="enc1", se_reduction=se_reduction)
        # === 更改点 1: MaxPool3d -> Conv3d with stride=2 ===
        self.down1 = nn.Conv3d(features, features, kernel_size=3, stride=2, padding=1, bias=True)

        self.encoder2 = _block(features, features * 2, name="enc2", se_reduction=se_reduction)
        # === 更改点 2: MaxPool3d -> Conv3d with stride=2 ===
        self.down2 = nn.Conv3d(features * 2, features * 2, kernel_size=3, stride=2, padding=1, bias=True)

        self.encoder3 = _block(features * 2, features * 4, name="enc3", se_reduction=se_reduction)
        # === 更改点 3: MaxPool3d -> Conv3d with stride=2 ===
        self.down3 = nn.Conv3d(features * 4, features * 4, kernel_size=3, stride=2, padding=1, bias=True)

        self.encoder4 = _block(features * 4, features * 8, name="enc4", se_reduction=se_reduction)
        # === 更改点 4: MaxPool3d -> Conv3d with stride=2 ===
        self.down4 = nn.Conv3d(features * 8, features * 8, kernel_size=3, stride=2, padding=1, bias=True)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck", se_reduction=se_reduction)

        # 解码器部分完全不变
        # 解码器 1
        self.upconv4_1 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4_1 = _block((features * 8) * 2, features * 8, name="dec4_1", se_reduction=se_reduction)
        self.upconv3_1 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3_1 = _block((features * 4) * 2, features * 4, name="dec3_1", se_reduction=se_reduction)
        self.upconv2_1 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2_1 = _block((features * 2) * 2, features * 2, name="dec2_1", se_reduction=se_reduction)
        self.upconv1_1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1_1 = _block(features * 2, features, name="dec1_1", se_reduction=se_reduction)
        self.conv_1 = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # 解码器 2 (为了简洁，省略了重复代码，结构与解码器1相同)
        # ... (与原代码相同)
        self.upconv4_2 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4_2 = _block((features * 8) * 2, features * 8, name="dec4_2", se_reduction=se_reduction)
        self.upconv3_2 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3_2 = _block((features * 4) * 2, features * 4, name="dec3_2", se_reduction=se_reduction)
        self.upconv2_2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2_2 = _block((features * 2) * 2, features * 2, name="dec2_2", se_reduction=se_reduction)
        self.upconv1_2 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1_2 = _block(features * 2, features, name="dec1_2", se_reduction=se_reduction)
        self.conv_2 = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # 编码
        enc1 = self.encoder1(x)
        # === 更改点 5: 调用新的下采样层 ===
        enc2 = self.encoder2(self.down1(enc1))
        enc3 = self.encoder3(self.down2(enc2))
        enc4 = self.encoder4(self.down3(enc3))
        bottleneck = self.bottleneck(self.down4(enc4))

        # 解码器 1 (forward pass 不变)
        dec4_1 = self.upconv4_1(bottleneck)
        dec4_1 = torch.cat((dec4_1, enc4), dim=1)
        dec4_1 = self.decoder4_1(dec4_1)
        dec3_1 = self.upconv3_1(dec4_1)
        dec3_1 = torch.cat((dec3_1, enc3), dim=1)
        dec3_1 = self.decoder3_1(dec3_1)
        dec2_1 = self.upconv2_1(dec3_1)
        dec2_1 = torch.cat((dec2_1, enc2), dim=1)
        dec2_1 = self.decoder2_1(dec2_1)
        dec1_1 = self.upconv1_1(dec2_1)
        dec1_1 = torch.cat((dec1_1, enc1), dim=1)
        dec1_1 = self.decoder1_1(dec1_1)
        output1 = self.conv_1(dec1_1)

        # 解码器 2 (forward pass 不变)
        dec4_2 = self.upconv4_2(bottleneck)
        dec4_2 = torch.cat((dec4_2, enc4), dim=1)
        dec4_2 = self.decoder4_2(dec4_2)
        dec3_2 = self.upconv3_2(dec4_2)
        dec3_2 = torch.cat((dec3_2, enc3), dim=1)
        dec3_2 = self.decoder3_2(dec3_2)
        dec2_2 = self.upconv2_2(dec3_2)
        dec2_2 = torch.cat((dec2_2, enc2), dim=1)
        dec2_2 = self.decoder2_2(dec2_2)
        dec1_2 = self.upconv1_2(dec2_2)
        dec1_2 = torch.cat((dec1_2, enc1), dim=1)
        dec1_2 = self.decoder1_2(dec1_2)
        output2 = self.conv_2(dec1_2)

        return output1, output2
#
#
# # ---------------------------
# # 简单自检
# # ---------------------------
# if __name__ == "__main__":
#     x = torch.randn(1, 1, 64, 128, 128)
#     # 使用新的模型进行测试
#     net = UNet3DDualDecoder(in_channels=1, out_channels=3, init_features=32, se_reduction=8)
#     y1, y2 = net(x)
#     print("out1:", y1.shape, "out2:", y2.shape)
#     # 检查参数量
#     total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     print(f"Total trainable parameters: {total_params:,}")


# === v3 ReLU → LeakyReLU 通道注意力，带步长的卷积
# import torch
# import torch.nn as nn
# from collections import OrderedDict
#
# class SE3D(nn.Module):
#     def __init__(self, channels: int, r: int = 8, negative_slope: float = 0.01):
#         super().__init__()
#         hidden = max(channels // r, 1)
#         self.avg = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Sequential(
#             nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
#             nn.LeakyReLU(negative_slope=negative_slope, inplace=True),  # ← 替换
#             nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         w = self.avg(x)
#         w = self.fc(w)
#         return x * w
#
# def _block(in_channels, features, name, se_reduction=8, negative_slope: float = 0.01):
#     return nn.Sequential(
#         OrderedDict(
#             [
#                 (name + "conv1",
#                  nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=True)),
#                 (name + "norm1", nn.BatchNorm3d(num_features=features)),
#                 (name + "relu1", nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),  # ← 替换
#                 (name + "conv2",
#                  nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)),
#                 (name + "norm2", nn.BatchNorm3d(num_features=features)),
#                 (name + "se", SE3D(features, r=se_reduction, negative_slope=negative_slope)),
#                 (name + "relu2", nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),  # ← 替换
#             ]
#         )
#     )
#
# class UNet3DDualDecoder(nn.Module):
#     def __init__(self, in_channels=1, out_channels=3, init_features=64, se_reduction=8, negative_slope: float = 0.01):
#         super(UNet3DDualDecoder, self).__init__()
#
#         features = init_features
#
#         # 编码器（共享）
#         self.encoder1 = _block(in_channels, features, name="enc1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down1 = nn.Conv3d(features, features, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.encoder2 = _block(features, features * 2, name="enc2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down2 = nn.Conv3d(features * 2, features * 2, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.encoder3 = _block(features * 2, features * 4, name="enc3", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down3 = nn.Conv3d(features * 4, features * 4, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.encoder4 = _block(features * 4, features * 8, name="enc4", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down4 = nn.Conv3d(features * 8, features * 8, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.bottleneck = _block(features * 8, features * 16, name="bottleneck", se_reduction=se_reduction, negative_slope=negative_slope)
#
#         # 解码器 1
#         self.upconv4_1 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4_1 = _block((features * 8) * 2, features * 8, name="dec4_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv3_1 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3_1 = _block((features * 4) * 2, features * 4, name="dec3_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv2_1 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2_1 = _block((features * 2) * 2, features * 2, name="dec2_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv1_1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
#         self.decoder1_1 = _block(features * 2, features, name="dec1_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.conv_1 = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
#
#         # 解码器 2
#         self.upconv4_2 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4_2 = _block((features * 8) * 2, features * 8, name="dec4_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv3_2 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3_2 = _block((features * 4) * 2, features * 4, name="dec3_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv2_2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2_2 = _block((features * 2) * 2, features * 2, name="dec2_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv1_2 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
#         self.decoder1_2 = _block(features * 2, features, name="dec1_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.conv_2 = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
#
#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.down1(enc1))
#         enc3 = self.encoder3(self.down2(enc2))
#         enc4 = self.encoder4(self.down3(enc3))
#         bottleneck = self.bottleneck(self.down4(enc4))
#
#         dec4_1 = self.upconv4_1(bottleneck)
#         dec4_1 = torch.cat((dec4_1, enc4), dim=1)
#         dec4_1 = self.decoder4_1(dec4_1)
#         dec3_1 = self.upconv3_1(dec4_1)
#         dec3_1 = torch.cat((dec3_1, enc3), dim=1)
#         dec3_1 = self.decoder3_1(dec3_1)
#         dec2_1 = self.upconv2_1(dec3_1)
#         dec2_1 = torch.cat((dec2_1, enc2), dim=1)
#         dec2_1 = self.decoder2_1(dec2_1)
#         dec1_1 = self.upconv1_1(dec2_1)
#         dec1_1 = torch.cat((dec1_1, enc1), dim=1)
#         dec1_1 = self.decoder1_1(dec1_1)
#         output1 = self.conv_1(dec1_1)
#
#         dec4_2 = self.upconv4_2(bottleneck)
#         dec4_2 = torch.cat((dec4_2, enc4), dim=1)
#         dec4_2 = self.decoder4_2(dec4_2)
#         dec3_2 = self.upconv3_2(dec4_2)
#         dec3_2 = torch.cat((dec3_2, enc3), dim=1)
#         dec3_2 = self.decoder3_2(dec3_2)
#         dec2_2 = self.upconv2_2(dec3_2)
#         dec2_2 = torch.cat((dec2_2, enc2), dim=1)
#         dec2_2 = self.decoder2_2(dec2_2)
#         dec1_2 = self.upconv1_2(dec2_2)
#         dec1_2 = torch.cat((dec1_2, enc1), dim=1)
#         dec1_2 = self.decoder1_2(dec1_2)
#         output2 = self.conv_2(dec1_2)
#
#         return output1, output2

# # ==== v4  v3基础上把BatchNorm3d换成InstanceNorm3d  ----290轮因为100轮没有提升失败
# import torch
# import torch.nn as nn
# from collections import OrderedDict
#
# class SE3D(nn.Module):
#     def __init__(self, channels: int, r: int = 8, negative_slope: float = 0.01):
#         super().__init__()
#         hidden = max(channels // r, 1)
#         self.avg = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Sequential(
#             nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
#             nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
#             nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         w = self.avg(x)
#         w = self.fc(w)
#         return x * w
#
# def _block(in_channels, features, name, se_reduction=8, negative_slope: float = 0.01):
#     return nn.Sequential(
#         OrderedDict(
#             [
#                 (name + "conv1",
#                  nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=True)),
#                 # (name + "norm1", nn.BatchNorm3d(num_features=features)), # <-- 原代码
#                 (name + "norm1", nn.InstanceNorm3d(num_features=features, affine=True)), # <-- 修改
#                 (name + "relu1", nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
#                 (name + "conv2",
#                  nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)),
#                 # (name + "norm2", nn.BatchNorm3d(num_features=features)), # <-- 原代码
#                 (name + "norm2", nn.InstanceNorm3d(num_features=features, affine=True)), # <-- 修改
#                 (name + "se", SE3D(features, r=se_reduction, negative_slope=negative_slope)),
#                 (name + "relu2", nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
#             ]
#         )
#     )
#
# class UNet3DDualDecoder(nn.Module):
#     def __init__(self, in_channels=1, out_channels=3, init_features=64, se_reduction=8, negative_slope: float = 0.01):
#         super(UNet3DDualDecoder, self).__init__()
#
#         features = init_features
#
#         # 编码器（共享）
#         self.encoder1 = _block(in_channels, features, name="enc1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down1 = nn.Conv3d(features, features, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.encoder2 = _block(features, features * 2, name="enc2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down2 = nn.Conv3d(features * 2, features * 2, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.encoder3 = _block(features * 2, features * 4, name="enc3", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down3 = nn.Conv3d(features * 4, features * 4, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.encoder4 = _block(features * 4, features * 8, name="enc4", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down4 = nn.Conv3d(features * 8, features * 8, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.bottleneck = _block(features * 8, features * 16, name="bottleneck", se_reduction=se_reduction, negative_slope=negative_slope)
#
#         # 解码器 1
#         self.upconv4_1 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4_1 = _block((features * 8) * 2, features * 8, name="dec4_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv3_1 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3_1 = _block((features * 4) * 2, features * 4, name="dec3_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv2_1 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2_1 = _block((features * 2) * 2, features * 2, name="dec2_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv1_1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
#         self.decoder1_1 = _block(features * 2, features, name="dec1_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.conv_1 = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
#
#         # 解码器 2
#         self.upconv4_2 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4_2 = _block((features * 8) * 2, features * 8, name="dec4_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv3_2 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3_2 = _block((features * 4) * 2, features * 4, name="dec3_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv2_2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2_2 = _block((features * 2) * 2, features * 2, name="dec2_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv1_2 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
#         self.decoder1_2 = _block(features * 2, features, name="dec1_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.conv_2 = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
#
#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.down1(enc1))
#         enc3 = self.encoder3(self.down2(enc2))
#         enc4 = self.encoder4(self.down3(enc3))
#         bottleneck = self.bottleneck(self.down4(enc4))
#
#         # 解码器 1 分支
#         dec4_1 = self.upconv4_1(bottleneck)
#         dec4_1 = torch.cat((dec4_1, enc4), dim=1)
#         dec4_1 = self.decoder4_1(dec4_1)
#         dec3_1 = self.upconv3_1(dec4_1)
#         dec3_1 = torch.cat((dec3_1, enc3), dim=1)
#         dec3_1 = self.decoder3_1(dec3_1)
#         dec2_1 = self.upconv2_1(dec3_1)
#         dec2_1 = torch.cat((dec2_1, enc2), dim=1)
#         dec2_1 = self.decoder2_1(dec2_1)
#         dec1_1 = self.upconv1_1(dec2_1)
#         dec1_1 = torch.cat((dec1_1, enc1), dim=1)
#         dec1_1 = self.decoder1_1(dec1_1)
#         output1 = self.conv_1(dec1_1)
#
#         # 解码器 2 分支
#         dec4_2 = self.upconv4_2(bottleneck)
#         dec4_2 = torch.cat((dec4_2, enc4), dim=1)
#         dec4_2 = self.decoder4_2(dec4_2)
#         dec3_2 = self.upconv3_2(dec4_2)
#         dec3_2 = torch.cat((dec3_2, enc3), dim=1)
#         dec3_2 = self.decoder3_2(dec3_2)
#         dec2_2 = self.upconv2_2(dec3_2)
#         dec2_2 = torch.cat((dec2_2, enc2), dim=1)
#         dec2_2 = self.decoder2_2(dec2_2)
#         dec1_2 = self.upconv1_2(dec2_2)
#         dec1_2 = torch.cat((dec1_2, enc1), dim=1)
#         dec1_2 = self.decoder1_2(dec1_2)
#         output2 = self.conv_2(dec1_2)
#
#         return output1, output2


# v5 ================================== v3基础 加了残差

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import OrderedDict
#
#
# class SE3D(nn.Module):
#     def __init__(self, channels: int, r: int = 8, negative_slope: float = 0.01):
#         super().__init__()
#         hidden = max(channels // r, 1)
#         self.avg = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Sequential(
#             nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
#             nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
#             nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         w = self.avg(x)
#         w = self.fc(w)
#         return x * w
#
#
# class ResSEBlock3D(nn.Module):
#     """
#     两层 3x3x3 卷积 + BN + LeakyReLU + SE 的残差块。
#     当 in_channels != out_channels 时，使用 1x1x1 conv(+BN) 做“投影”以对齐通道后再相加。
#     """
#     def __init__(self, in_channels, out_channels, se_reduction=8, negative_slope: float = 0.01):
#         super().__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
#         self.bn1   = nn.BatchNorm3d(out_channels)
#         self.act1  = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
#
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
#         self.bn2   = nn.BatchNorm3d(out_channels)
#
#         self.se    = SE3D(out_channels, r=se_reduction, negative_slope=negative_slope)
#
#         # 投影分支：通道不一致时对齐；空间尺寸本块不变（stride=1），无需对齐 D/H/W
#         if in_channels != out_channels:
#             self.proj = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
#                 nn.BatchNorm3d(out_channels)
#             )
#         else:
#             self.proj = nn.Identity()
#
#         self.act2 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
#
#     def forward(self, x):
#         identity = self.proj(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.act1(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out = self.se(out)          # 注意力放在第二层卷积后更稳
#         out = out + identity        # 残差相加
#         out = self.act2(out)
#         return out
#
#
# def _block(in_channels, features, name, se_reduction=8, negative_slope: float = 0.01):
#     # 维持原函数签名，内部改为残差块
#     return ResSEBlock3D(
#         in_channels=in_channels,
#         out_channels=features,
#         se_reduction=se_reduction,
#         negative_slope=negative_slope
#     )
#
#
# class UNet3DDualDecoder(nn.Module):
#     def __init__(self, in_channels=1, out_channels=3, init_features=64, se_reduction=8, negative_slope: float = 0.01):
#         super(UNet3DDualDecoder, self).__init__()
#
#         features = init_features
#
#         # 编码器（共享）
#         self.encoder1 = _block(in_channels, features, name="enc1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down1 = nn.Conv3d(features, features, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.encoder2 = _block(features, features * 2, name="enc2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down2 = nn.Conv3d(features * 2, features * 2, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.encoder3 = _block(features * 2, features * 4, name="enc3", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down3 = nn.Conv3d(features * 4, features * 4, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.encoder4 = _block(features * 4, features * 8, name="enc4", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.down4 = nn.Conv3d(features * 8, features * 8, kernel_size=3, stride=2, padding=1, bias=True)
#
#         self.bottleneck = _block(features * 8, features * 16, name="bottleneck", se_reduction=se_reduction, negative_slope=negative_slope)
#
#         # 解码器 1
#         self.upconv4_1 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4_1 = _block((features * 8) * 2, features * 8, name="dec4_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv3_1 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3_1 = _block((features * 4) * 2, features * 4, name="dec3_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv2_1 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2_1 = _block((features * 2) * 2, features * 2, name="dec2_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv1_1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
#         self.decoder1_1 = _block(features * 2, features, name="dec1_1", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.conv_1 = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
#
#         # 解码器 2
#         self.upconv4_2 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4_2 = _block((features * 8) * 2, features * 8, name="dec4_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv3_2 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3_2 = _block((features * 4) * 2, features * 4, name="dec3_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv2_2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2_2 = _block((features * 2) * 2, features * 2, name="dec2_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.upconv1_2 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
#         self.decoder1_2 = _block(features * 2, features, name="dec1_2", se_reduction=se_reduction, negative_slope=negative_slope)
#         self.conv_2 = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
#
#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.down1(enc1))
#         enc3 = self.encoder3(self.down2(enc2))
#         enc4 = self.encoder4(self.down3(enc3))
#         bottleneck = self.bottleneck(self.down4(enc4))
#
#         dec4_1 = self.upconv4_1(bottleneck)
#         dec4_1 = torch.cat((dec4_1, enc4), dim=1)
#         dec4_1 = self.decoder4_1(dec4_1)
#         dec3_1 = self.upconv3_1(dec4_1)
#         dec3_1 = torch.cat((dec3_1, enc3), dim=1)
#         dec3_1 = self.decoder3_1(dec3_1)
#         dec2_1 = self.upconv2_1(dec3_1)
#         dec2_1 = torch.cat((dec2_1, enc2), dim=1)
#         dec2_1 = self.decoder2_1(dec2_1)
#         dec1_1 = self.upconv1_1(dec2_1)
#         dec1_1 = torch.cat((dec1_1, enc1), dim=1)
#         dec1_1 = self.decoder1_1(dec1_1)
#         output1 = self.conv_1(dec1_1)
#
#         dec4_2 = self.upconv4_2(bottleneck)
#         dec4_2 = torch.cat((dec4_2, enc4), dim=1)
#         dec4_2 = self.decoder4_2(dec4_2)
#         dec3_2 = self.upconv3_2(dec4_2)
#         dec3_2 = torch.cat((dec3_2, enc3), dim=1)
#         dec3_2 = self.decoder3_2(dec3_2)
#         dec2_2 = self.upconv2_2(dec3_2)
#         dec2_2 = torch.cat((dec2_2, enc2), dim=1)
#         dec2_2 = self.decoder2_2(dec2_2)
#         dec1_2 = self.upconv1_2(dec2_2)
#         dec1_2 = torch.cat((dec1_2, enc1), dim=1)
#         dec1_2 = self.decoder1_2(dec1_2)
#         output2 = self.conv_2(dec1_2)
#
#         return output1, output2

