# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

# -------------------- 3D 高斯窗 --------------------
def gaussian_1d(window_size, sigma=1.5, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    return g / (g.sum() + 1e-12)

def create_window_3d(window_size, channel=1, sigma=1.5, device=None, dtype=None):
    """
    返回形状 [channel, 1, K, K, K] 的 3D 高斯核，用于分组卷积。
    """
    if window_size % 2 == 0:
        window_size -= 1  # 建议奇数窗
    g = gaussian_1d(window_size, sigma=sigma, device=device, dtype=dtype)
    # 外积构成 3D 高斯核
    g3 = (g[:, None, None] * g[None, :, None] * g[None, None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,K,K,K]
    window = g3.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

# -------------------- 3D SSIM 基元 --------------------
def _val_range_from(img):
    # 与原 2D 版本相近的启发式
    if torch.max(img) > 128:
        max_val = 255.
    else:
        max_val = 1.
    if torch.min(img) < -0.5:
        min_val = -1.
    else:
        min_val = 0.
    return max_val - min_val

def ssim3d(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    """
    img1, img2: [B, C, D, H, W]，建议先归一化到 [0,1]，或传入 val_range。
    返回：标量（size_average=True）或按 B 聚合的张量（False 时保留 B 维；C 在内部已分组）
    """
    assert img1.shape == img2.shape, "img1/img2 must have the same shape [B,C,D,H,W]"
    _, channel, D, H, W = img1.size()

    if val_range is None:
        L = _val_range_from(img1)
    else:
        L = float(val_range)

    # 注意：这里与原 2D 代码一致，使用 padding=0（valid 卷积），仅取均值不关心边界尺寸缩小
    padd = 0

    if window is None:
        real_size = int(min(window_size, D, H, W))
        if real_size % 2 == 0:
            real_size -= 1
        window = create_window_3d(real_size, channel=channel, device=img1.device, dtype=img1.dtype)

    mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padd, groups=channel)

    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12   = F.conv3d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = (v1 / v2).mean()  # contrast sensitivity（与原实现一致）

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        # 保留 batch 维；通道已做分组卷积，这里对 C、D、H、W 求均值
        ret = ssim_map.mean(dim=(1, 2, 3, 4))

    if full:
        return ret, cs
    return ret

def msssim3d(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    """
    3D 多尺度 SSIM。默认使用经典权重。
    注意：每次尺度都会 avg_pool3d(/2)，请保证 D/H/W 足够大（或缩小 levels/窗口）。
    """
    device = img1.device
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device, dtype=img1.dtype)
    levels = weights.numel()

    mssim = []
    mcs   = []
    cur1, cur2 = img1, img2
    for _ in range(levels):
        sim, cs = ssim3d(cur1, cur2, window_size=window_size, size_average=size_average, full=True, val_range=val_range), \
                  None  # 占位，稍后重算 cs 以避免重复卷积
        # 为减少重复计算，可以写一个返回 (ssim, cs) 的版本；这里直接再算一次 cs：
        s, cs = ssim3d(cur1, cur2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(s)
        mcs.append(cs)

        # 体素下采样（各向同性 /2）
        cur1 = F.avg_pool3d(cur1, kernel_size=2, stride=2, ceil_mode=False)
        cur2 = F.avg_pool3d(cur2, kernel_size=2, stride=2, ceil_mode=False)

    mssim = torch.stack(mssim)
    mcs   = torch.stack(mcs)

    if normalize:
        mssim = (mssim + 1) / 2
        mcs   = (mcs   + 1) / 2

    # 与经典 MS-SSIM 公式一致：前 (L-1) 层用 mcs，最后一层用 mssim
    output = torch.prod((mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1]))
    return output

# -------------------- 3D 类封装（缓存窗口） --------------------
class SSIM3D(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super().__init__()
        self.window_size  = window_size
        self.size_average = size_average
        self.val_range    = val_range
        self.register_buffer("window", torch.Tensor())  # 动态按需创建
        self.channel = None

    def forward(self, img1, img2):
        assert img1.shape == img2.shape, "img1/img2 must be same shape [B,C,D,H,W]"
        _, c, d, h, w = img1.shape
        # 若通道或 dtype 变化，则重建窗口
        if (self.channel != c) or (self.window.dtype != img1.dtype) or (self.window.device != img1.device):
            self.window = create_window_3d(
                min(self.window_size, d, h, w),
                channel=c, device=img1.device, dtype=img1.dtype
            )
            self.channel = c

        return ssim3d(img1, img2, window_size=self.window.shape[-1], window=self.window,
                      size_average=self.size_average, val_range=self.val_range)

class MSSSIM3D(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size  = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return msssim3d(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)
