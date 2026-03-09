import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- 基础模块 --------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.conv2 = nn.Conv3d(in_channels, mid_channels, 1)
        self.conv3 = nn.Conv3d(mid_channels, in_channels, 1)
        self.conv4 = nn.Conv3d(2 * in_channels, out_channels, 3, 1, 1)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = torch.cat([res, x], dim=1)
        x = self.act(x)
        x = self.conv4(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, 3, 2, 1, groups=in_channels)
        self.conv2 = nn.Conv3d(in_channels, mid_channels, 1)
        self.conv3 = nn.Conv3d(mid_channels, in_channels, 1)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.act = nn.GELU()
        self.res = nn.Conv3d(in_channels, in_channels, 1, 2)

    def forward(self, x):
        res = self.res(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = torch.cat([res, x], dim=1)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, in_channels, 2, 2, groups=in_channels)
        self.conv2 = nn.Conv3d(in_channels, mid_channels, 1)
        self.conv3 = nn.Conv3d(mid_channels, in_channels // 2, 1)
        self.conv4 = nn.Conv3d(in_channels, in_channels // 2, 3, 1, 1)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.act = nn.GELU()
        self.res = nn.ConvTranspose3d(in_channels, in_channels // 2, 2, 2)

    def forward(self, x):
        res = self.res(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = torch.cat([res, x], dim=1)
        x = self.conv4(x)
        return x


# -------------------- 主干网络 --------------------
class HiPaS_STS_Single(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, mid_channels=16, r=4, prior_channels=0):
        """
        自适应版本：支持有/无先验图输入
        Args:
            in_channels: 输入 CT 通道数
            out_channels: 输出类别数（背景/动脉/静脉）
            mid_channels: 基础特征通道数
            r: 扩张倍率
            prior_channels: 先验图通道数（可为0）
        """
        super().__init__()
        self.use_prior = prior_channels > 0
        self.pconv = nn.Conv3d(prior_channels, prior_channels, 3, 1, 1) if self.use_prior else None

        self.trans = nn.Sequential(
            nn.Conv3d(in_channels + (prior_channels if self.use_prior else 0), in_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(in_channels, in_channels, 3, 1, 1),
        )

        self.stem = nn.Conv3d(in_channels, mid_channels, 3, 1, 1)

        self.enc_block_0 = ResBlock(mid_channels, r * mid_channels, mid_channels)
        self.down_0 = DownSample(mid_channels, r * mid_channels)

        self.enc_block_1 = ResBlock(2 * mid_channels, r * 2 * mid_channels, 2 * mid_channels)
        self.down_1 = DownSample(2 * mid_channels, r * 2 * mid_channels)

        self.enc_block_2 = ResBlock(4 * mid_channels, r * 4 * mid_channels, 4 * mid_channels)
        self.down_2 = DownSample(4 * mid_channels, r * 4 * mid_channels)

        self.enc_block_3 = ResBlock(8 * mid_channels, r * 8 * mid_channels, 8 * mid_channels)
        self.down_3 = DownSample(8 * mid_channels, r * 8 * mid_channels)

        self.bottom = nn.Conv3d(16 * mid_channels, 16 * mid_channels, 1)

        self.up_3 = UpSample(16 * mid_channels, r * 16 * mid_channels)
        self.dec_block_3 = ResBlock(16 * mid_channels, r * 8 * mid_channels, 8 * mid_channels)

        self.up_2 = UpSample(8 * mid_channels, r * 8 * mid_channels)
        self.dec_block_2 = ResBlock(8 * mid_channels, r * 4 * mid_channels, 4 * mid_channels)

        self.up_1 = UpSample(4 * mid_channels, r * 4 * mid_channels)
        self.dec_block_1 = ResBlock(4 * mid_channels, r * 2 * mid_channels, 2 * mid_channels)

        self.up_0 = UpSample(2 * mid_channels, r * 2 * mid_channels)
        self.dec_block_0 = ResBlock(2 * mid_channels, r * mid_channels, mid_channels)

        self.out = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, 3, 1, 1),
            # nn.Sigmoid()
        )

    def forward(self, x, p=None):
        # 自适应先验融合
        if self.use_prior and (p is not None):
            x = torch.cat((x, self.pconv(p)), dim=1)
        x = self.trans(x)

        # Encoder
        x = self.stem(x)
        r0 = self.enc_block_0(x); x = self.down_0(r0)
        r1 = self.enc_block_1(x); x = self.down_1(r1)
        r2 = self.enc_block_2(x); x = self.down_2(r2)
        r3 = self.enc_block_3(x); x = self.down_3(r3)

        # Bottom
        x = self.bottom(x)

        # Decoder
        x = self.up_3(x); x = torch.cat([x, r3], dim=1); x = self.dec_block_3(x)
        x = self.up_2(x); x = torch.cat([x, r2], dim=1); x = self.dec_block_2(x)
        x = self.up_1(x); x = torch.cat([x, r1], dim=1); x = self.dec_block_1(x)
        x = self.up_0(x); x = torch.cat([x, r0], dim=1); x = self.dec_block_0(x)

        return self.out(x)
