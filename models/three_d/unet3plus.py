# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------- 基础模块 -------------------
def get_norm(norm: str, num_features: int):
    norm = (norm or "bn").lower()
    if norm in ["bn", "batch", "batchnorm", "batchnorm3d"]:
        return nn.BatchNorm3d(num_features)
    elif norm in ["in", "instance", "instancenorm", "instancenorm3d"]:
        return nn.InstanceNorm3d(num_features, affine=True)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

class ConvBlock3d(nn.Module):
    """ 3D 两层卷积块：Conv3d -> Norm -> ReLU -> Conv3d -> Norm -> ReLU """
    def __init__(self, in_ch, out_ch, norm="bn"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            get_norm(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            get_norm(norm, out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

def kaiming_init(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# ------------------- 3D UNet 3+（logits 输出） -------------------
class UNet3Plus3D(nn.Module):
    """
    3D 版 UNet 3+（Full-Scale Skip Connection）。
    - 返回 logits（不做 Sigmoid/Softmax）
    - 建议输入体素大小 D,H,W 能被 16 整除（网络下采样 4 次）
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_classes:   int = 1,
        base_filters=(64, 128, 256, 512, 1024),
        feature_scale: int = 4,     # 通道缩放，默认 /4 以控显存
        norm: str = "bn",
        align_corners: bool = True,
    ):
        super().__init__()
        self.in_channels   = in_channels
        self.n_classes     = n_classes
        self.align_corners = align_corners

        # 缩放通道数以控制显存
        filters = [max(int(c // feature_scale), 8) for c in base_filters]  # 最少 8
        f1, f2, f3, f4, f5 = filters

        # ---------------- Encoder ----------------
        self.conv1 = ConvBlock3d(in_channels, f1, norm)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2 = ConvBlock3d(f1, f2, norm)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3 = ConvBlock3d(f2, f3, norm)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4 = ConvBlock3d(f3, f4, norm)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv5 = ConvBlock3d(f4, f5, norm)   # 最底层

        # ---------------- Decoder: 统一 Cat 通道 ----------------
        self.CatChannels = f1
        self.CatBlocks   = 5
        self.UpChannels  = self.CatChannels * self.CatBlocks

        # —— stage hd4（1/8）
        self.h1_PT_hd4 = nn.MaxPool3d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv3d(f1, self.CatChannels, 3, padding=1, bias=False)
        self.h1_PT_hd4_bn   = get_norm(norm, self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd4 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv3d(f2, self.CatChannels, 3, padding=1, bias=False)
        self.h2_PT_hd4_bn   = get_norm(norm, self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h3_PT_hd4 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv3d(f3, self.CatChannels, 3, padding=1, bias=False)
        self.h3_PT_hd4_bn   = get_norm(norm, self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h4_Cat_hd4_conv = nn.Conv3d(f4, self.CatChannels, 3, padding=1, bias=False)
        self.h4_Cat_hd4_bn   = get_norm(norm, self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd4_conv = nn.Conv3d(f5, self.CatChannels, 3, padding=1, bias=False)
        self.hd5_UT_hd4_bn   = get_norm(norm, self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        self.conv4d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1, bias=False)
        self.bn4d_1   = get_norm(norm, self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        # —— stage hd3（1/4）
        self.h1_PT_hd3 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv3d(f1, self.CatChannels, 3, padding=1, bias=False)
        self.h1_PT_hd3_bn   = get_norm(norm, self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd3 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv3d(f2, self.CatChannels, 3, padding=1, bias=False)
        self.h2_PT_hd3_bn   = get_norm(norm, self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h3_Cat_hd3_conv = nn.Conv3d(f3, self.CatChannels, 3, padding=1, bias=False)
        self.h3_Cat_hd3_bn   = get_norm(norm, self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd3_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd4_UT_hd3_bn   = get_norm(norm, self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd3_conv = nn.Conv3d(f5, self.CatChannels, 3, padding=1, bias=False)
        self.hd5_UT_hd3_bn   = get_norm(norm, self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        self.conv3d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1, bias=False)
        self.bn3d_1   = get_norm(norm, self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        # —— stage hd2（1/2）
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv3d(f1, self.CatChannels, 3, padding=1, bias=False)
        self.h1_PT_hd2_bn   = get_norm(norm, self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        self.h2_Cat_hd2_conv = nn.Conv3d(f2, self.CatChannels, 3, padding=1, bias=False)
        self.h2_Cat_hd2_bn   = get_norm(norm, self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd3_UT_hd2_bn   = get_norm(norm, self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd4_UT_hd2_bn   = get_norm(norm, self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd2_conv = nn.Conv3d(f5, self.CatChannels, 3, padding=1, bias=False)
        self.hd5_UT_hd2_bn   = get_norm(norm, self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1, bias=False)
        self.bn2d_1   = get_norm(norm, self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        # —— stage hd1（1/1）
        self.h1_Cat_hd1_conv = nn.Conv3d(f1, self.CatChannels, 3, padding=1, bias=False)
        self.h1_Cat_hd1_bn   = get_norm(norm, self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        self.hd2_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd2_UT_hd1_bn   = get_norm(norm, self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd3_UT_hd1_bn   = get_norm(norm, self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd4_UT_hd1_bn   = get_norm(norm, self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd1_conv = nn.Conv3d(f5, self.CatChannels, 3, padding=1, bias=False)
        self.hd5_UT_hd1_bn   = get_norm(norm, self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        self.conv1d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1, bias=False)
        self.bn1d_1   = get_norm(norm, self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # 输出头
        self.outconv1 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)

        # 初始化
        self.apply(kaiming_init)

    # ---- 一致性的上采样/下采样封装 ----
    def _up(self, x, scale_factor):
        return F.interpolate(x, scale_factor=scale_factor, mode="trilinear", align_corners=self.align_corners)

    def forward(self, x):
        # ---------------- Encoder ----------------
        h1 = self.conv1(x)               # 1/1
        h2 = self.conv2(self.pool1(h1))  # 1/2
        h3 = self.conv3(self.pool2(h2))  # 1/4
        h4 = self.conv4(self.pool3(h3))  # 1/8
        hd5 = self.conv5(self.pool4(h4)) # 1/16

        # ---------------- Decoder ----------------
        # ---- hd4 (1/8) ----
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4= self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4= self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self._up(hd5, 2))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4], dim=1))))

        # ---- hd3 (1/4) ----
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3= self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3= self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self._up(hd4, 2))))
        hd5_UT_hd3= self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self._up(hd5, 4))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3], dim=1))))

        # ---- hd2 (1/2) ----
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2= self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2= self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self._up(hd3, 2))))
        hd4_UT_hd2= self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self._up(hd4, 4))))
        hd5_UT_hd2= self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self._up(hd5, 8))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat([h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2], dim=1))))

        # ---- hd1 (1/1) ----
        h1_Cat_hd1= self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1= self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self._up(hd2, 2))))
        hd3_UT_hd1= self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self._up(hd3, 4))))
        hd4_UT_hd1= self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self._up(hd4, 8))))
        hd5_UT_hd1= self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self._up(hd5, 16))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat([h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1], dim=1))))

        # 输出 logits（不做激活）
        d1 = self.outconv1(hd1)
        return d1

# ------------------- Deep Supervision 版本（5 个尺度 logits） -------------------
class UNet3Plus3D_DeepSup(UNet3Plus3D):
    """
    Deep Supervision：返回 (d1, d2, d3, d4, d5) 五个尺度的 logits，
    其中 d{2..5} 会插值到与 d1 相同的空间尺寸（便于直接计算损失）。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        C = self.UpChannels
        f5 = max(int((kwargs.get("base_filters", (64,128,256,512,1024))[4]) // kwargs.get("feature_scale", 4)), 8)

        self.outconv1 = nn.Conv3d(C, self.n_classes, 3, padding=1)
        self.outconv2 = nn.Conv3d(C, self.n_classes, 3, padding=1)
        self.outconv3 = nn.Conv3d(C, self.n_classes, 3, padding=1)
        self.outconv4 = nn.Conv3d(C, self.n_classes, 3, padding=1)
        self.outconv5 = nn.Conv3d(f5, self.n_classes, 3, padding=1)

        self.apply(kaiming_init)

    def forward(self, x):
        # ------- 复用父类结构，但需要拿到中间变量 -------
        # Encoder
        h1 = self.conv1(x)
        h2 = self.conv2(self.pool1(h1))
        h3 = self.conv3(self.pool2(h2))
        h4 = self.conv4(self.pool3(h3))
        hd5 = self.conv5(self.pool4(h4))

        # Decoder hd4
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4= self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4= self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(F.interpolate(hd5, scale_factor=2, mode="trilinear", align_corners=self.align_corners))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4], dim=1))))

        # Decoder hd3
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3= self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3= self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(F.interpolate(hd4, scale_factor=2, mode="trilinear", align_corners=self.align_corners))))
        hd5_UT_hd3= self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(F.interpolate(hd5, scale_factor=4, mode="trilinear", align_corners=self.align_corners))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3], dim=1))))

        # Decoder hd2
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2= self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2= self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(F.interpolate(hd3, scale_factor=2, mode="trilinear", align_corners=self.align_corners))))
        hd4_UT_hd2= self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(F.interpolate(hd4, scale_factor=4, mode="trilinear", align_corners=self.align_corners))))
        hd5_UT_hd2= self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(F.interpolate(hd5, scale_factor=8, mode="trilinear", align_corners=self.align_corners))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat([h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2], dim=1))))

        # Decoder hd1
        h1_Cat_hd1= self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1= self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(F.interpolate(hd2, scale_factor=2, mode="trilinear", align_corners=self.align_corners))))
        hd3_UT_hd1= self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(F.interpolate(hd3, scale_factor=4, mode="trilinear", align_corners=self.align_corners))))
        hd4_UT_hd1= self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(F.interpolate(hd4, scale_factor=8, mode="trilinear", align_corners=self.align_corners))))
        hd5_UT_hd1= self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(F.interpolate(hd5, scale_factor=16, mode="trilinear", align_corners=self.align_corners))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat([h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1], dim=1))))

        # 多尺度输出（均插值到 hd1 尺寸）
        d1 = self.outconv1(hd1)
        d2 = F.interpolate(self.outconv2(hd2), size=d1.shape[2:], mode="trilinear", align_corners=self.align_corners)
        d3 = F.interpolate(self.outconv3(hd3), size=d1.shape[2:], mode="trilinear", align_corners=self.align_corners)
        d4 = F.interpolate(self.outconv4(hd4), size=d1.shape[2:], mode="trilinear", align_corners=self.align_corners)
        d5 = F.interpolate(self.outconv5(hd5), size=d1.shape[2:], mode="trilinear", align_corners=self.align_corners)
        return d1, d2, d3, d4, d5

# ------------------- 快速用法 -------------------
if __name__ == "__main__":
    # 假设 3D Patch: [B, C, D, H, W]，建议可被 16 整除（例如 64×128×128）
    x = torch.randn(2, 1, 64, 128, 128)  # 2 个样本，单通道 CT 体块
    net = UNet3Plus3D(in_channels=1, n_classes=3, feature_scale=4, norm="in")  # 三分类 A/V 示例
    y = net(x)  # logits: [2, 3, 64, 128, 128]
    print("UNet3Plus3D:", y.shape)

    net_ds = UNet3Plus3D_DeepSup(in_channels=1, n_classes=1, feature_scale=4, norm="in")
    y1, y2, y3, y4, y5 = net_ds(x)  # 都是 logits，且尺寸与 y1 相同
    print("DeepSup:", y1.shape, y2.shape, y3.shape, y4.shape, y5.shape)
