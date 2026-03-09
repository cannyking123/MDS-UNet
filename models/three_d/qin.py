import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

"""
Tubule-Sensitive CNNs (TS-CNN) — PyTorch 3D implementation
Reproduces the key components described by Qin et al.,
"Learning Tubule-Sensitive CNNs for Pulmonary Airway and Artery-Vein Segmentation in CT" (2021):
  • 3D U-Net backbone (5 scales)
  • Feature Recalibration (FR): spatial-priority integration → channel-wise reweighting
  • Decoder-side Attention Distillation (AD): high-res attention supervises low-res
  • Anatomy priors as extra input channels (lung context + airway-wall distance)
  • Multi-head outputs: (A/V/background) + Vessel/Non-vessel auxiliary head
  • Optional coordinate map concatenation at the highest decoder level

Notes
-----
- Forward returns logits only (no thresholding). Use your losses externally.
- For AD, we also return attention feature maps per decoder stage so you can
  compute distillation loss outside the module.
- InstanceNorm3d + LeakyReLU are used as in many strong 3D U-Net baselines.
- Lightweight and faithful; you can swap blocks/norms as needed.
"""

# -----------------------------
# Utility blocks
# -----------------------------

class ConvINLRelu3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act  = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b1 = ConvINLRelu3D(in_ch, out_ch)
        self.b2 = ConvINLRelu3D(out_ch, out_ch)
    def forward(self, x):
        return self.b2(self.b1(x))

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3D(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # handle size mismatch due to odd dims
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        x = F.pad(x, (0, dx, 0, dy, 0, dz))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# -----------------------------
# Feature Recalibration (FR)
# -----------------------------

class FeatureRecalibration3D(nn.Module):
    """
    Spatial-priority integration + channel recalibration.
    - Spatial integration: depthwise 3x3x3 conv (captures local tubular context)
    - Channel recalibration: squeeze (GAP) → 1x1x1 MLP → sigmoid weights
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.dw = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()
    def forward(self, x):
        s = self.dw(x)
        z = s.mean(dim=(2,3,4), keepdim=True)  # GAP
        w = self.gate(self.fc2(self.act(self.fc1(z))))
        return x * w

# -----------------------------
# Attention map helper (for AD loss)
# -----------------------------

def attention_map(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = feat.pow(2).mean(dim=1, keepdim=True)   # [B,1,D,H,W]
    return torch.sqrt(a + eps)                  # 避免 clamp_，无就地操作


# -----------------------------
# TS-CNN Network
# -----------------------------

class TSCNN3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes_av: int = 3,   # [bg, artery, vein]
        use_coord_map: bool = True,
        fr_on_encoder: bool = True,
        fr_on_decoder: bool = True,
        base_ch: int = 32,
    ):
        super().__init__()
        self.use_coord_map = use_coord_map
        # Encoder
        self.enc1 = DoubleConv3D(in_channels, base_ch)
        self.fr1  = FeatureRecalibration3D(base_ch) if fr_on_encoder else nn.Identity()

        self.enc2 = Down3D(base_ch, base_ch*2)
        self.fr2  = FeatureRecalibration3D(base_ch*2) if fr_on_encoder else nn.Identity()

        self.enc3 = Down3D(base_ch*2, base_ch*4)
        self.fr3  = FeatureRecalibration3D(base_ch*4) if fr_on_encoder else nn.Identity()

        self.enc4 = Down3D(base_ch*4, base_ch*8)
        self.fr4  = FeatureRecalibration3D(base_ch*8) if fr_on_encoder else nn.Identity()

        self.enc5 = Down3D(base_ch*8, base_ch*16)
        self.fr5  = FeatureRecalibration3D(base_ch*16) if fr_on_encoder else nn.Identity()

        # Decoder
        self.up4 = Up3D(base_ch*16, base_ch*8)
        self.fr_d4 = FeatureRecalibration3D(base_ch*8) if fr_on_decoder else nn.Identity()

        self.up3 = Up3D(base_ch*8, base_ch*4)
        self.fr_d3 = FeatureRecalibration3D(base_ch*4) if fr_on_decoder else nn.Identity()

        self.up2 = Up3D(base_ch*4, base_ch*2)
        self.fr_d2 = FeatureRecalibration3D(base_ch*2) if fr_on_decoder else nn.Identity()

        self.up1 = Up3D(base_ch*2, base_ch)
        self.fr_d1 = FeatureRecalibration3D(base_ch) if fr_on_decoder else nn.Identity()

        # Optional coordinate map concatenation at highest decoder level (D4)
        cat_ch = base_ch*8 + (3 if use_coord_map else 0)
        self.coord_cat = ConvINLRelu3D(cat_ch, base_ch*8) if use_coord_map else nn.Identity()

        # Heads
        self.head_av = nn.Conv3d(base_ch, num_classes_av, kernel_size=1)   # logits for [bg, A, V]
        self.head_v  = nn.Conv3d(base_ch, 1, kernel_size=1)                # logits for vessel/non-vessel

    @staticmethod
    def _coord_map_like(x: torch.Tensor) -> torch.Tensor:
        """Normalized coordinate map in [-1,1], shape [B,3,D,H,W]."""
        B, _, D, H, W = x.shape
        z = torch.linspace(-1, 1, steps=D, device=x.device)
        y = torch.linspace(-1, 1, steps=H, device=x.device)
        xg = torch.linspace(-1, 1, steps=W, device=x.device)
        zz, yy, xx = torch.meshgrid(z, y, xg, indexing='ij')
        cm = torch.stack([zz, yy, xx], dim=0).unsqueeze(0).repeat(B,1,1,1,1)
        return cm

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
          x: [B, C_in, D, H, W] where C_in may include anatomy priors.
        Returns:
          dict with keys:
            'logits_av': [B, 3, D, H, W]
            'logits_v' : [B, 1, D, H, W]
            'attn_d{1..4}': attention maps for decoder stages (for AD loss)
        """
        # Encoder path
        e1 = self.fr1(self.enc1(x))          # [B, C, D, H, W]
        e2 = self.fr2(self.enc2(e1))         # [B, 2C]
        e3 = self.fr3(self.enc3(e2))         # [B, 4C]
        e4 = self.fr4(self.enc4(e3))         # [B, 8C]
        e5 = self.fr5(self.enc5(e4))         # [B,16C]

        # Decoder path
        d4 = self.up4(e5, e4)                # [B, 8C]
        if self.use_coord_map:
            cm = self._coord_map_like(d4)
            d4 = torch.cat([d4, cm], dim=1)
            d4 = self.coord_cat(d4)
        d4 = self.fr_d4(d4)
        attn_d4 = attention_map(d4)

        d3 = self.fr_d3(self.up3(d4, e3))    # [B, 4C]
        attn_d3 = attention_map(d3)

        d2 = self.fr_d2(self.up2(d3, e2))    # [B, 2C]
        attn_d2 = attention_map(d2)

        d1 = self.fr_d1(self.up1(d2, e1))    # [B, 1C]
        attn_d1 = attention_map(d1)

        logits_av = self.head_av(d1)
        logits_v  = self.head_v(d1)

        return {
            'logits_av': logits_av,
            'logits_v' : logits_v,
            'attn_d1': attn_d1,
            'attn_d2': attn_d2,
            'attn_d3': attn_d3,
            'attn_d4': attn_d4,
        }

# -----------------------------
# Loss helpers (Dice+Focal, Distillation)
# -----------------------------

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, probs: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        # probs: [B,C,...], target_onehot: [B,C,...]
        dims = tuple(range(2, probs.ndim))
        inter = (probs * target_onehot).sum(dims)
        denom = probs.sum(dims) + target_onehot.sum(dims) + self.eps
        dice = (2 * inter + self.eps) / denom
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
    def forward(self, probs: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        # probs: [B,C,...], target_onehot: [B,C,...]
        p = (probs * target_onehot).sum(dim=1).clamp_min(self.eps)  # [B,...] matched-class prob
        return (-((1 - p) ** self.gamma) * p.log()).mean()

import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss3D_W(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean', ignore_index=None,
                 clip_min=0.1, clip_max=10.0, normalize='mean_valid'):
        """
        h-loss 输入是logits, targets
        normalize:
          - 'mean_valid' -> (loss*w).sum() / w.sum()  （推荐）
          - 'mean_mask'  -> (loss*w).sum() / valid.sum()（若希望和未加权时相同分母）
          - None         -> 按 reduction='mean'/'sum' 走（不建议）
        """
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None

        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.normalize = normalize

    def forward(self, logits, targets, weight_map=None):
        """
        logits:     [B, C, D, H, W]
        targets:    [B, D, H, W]  (int64)
        weight_map: [B, D, H, W] or [B, 1, D, H, W] (float), optional
        """
        B, C, D, H, W = logits.shape

        # ----- ignore_index -> valid_mask & 安全的 targets_temp -----
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            targets_temp = targets.clone()
            targets_temp[~valid_mask] = 0
        else:
            valid_mask = None
            targets_temp = targets

        # ----- softmax 概率与 log 概率 -----
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # ----- one-hot -----
        tgt_1h = F.one_hot(targets_temp, num_classes=C).permute(0,4,1,2,3).float()  # [B,C,D,H,W]

        # ----- focal 部分 -----
        pt = (probs * tgt_1h).sum(dim=1).clamp(min=1e-6)   # [B,D,H,W]
        log_pt = (log_probs * tgt_1h).sum(dim=1)           # [B,D,H,W]
        focal_weight = (1.0 - pt) ** self.gamma

        # ----- alpha（类权重） -----
        if self.alpha is not None:
            # NOTE: 如果 weight_map 已经承担了强的类别/层级平衡作用，考虑先把 alpha=None
            alpha_factor = self.alpha[targets_temp]  # [B,D,H,W]
        else:
            alpha_factor = 1.0

        base_loss = -alpha_factor * focal_weight * log_pt  # [B,D,H,W]

        # ----- valid mask 应用到 loss -----
        if valid_mask is not None:
            base_loss = base_loss * valid_mask

        # ----- 处理 weight_map -----
        if weight_map is not None:
            if weight_map.dim() == 5 and weight_map.size(1) == 1:
                weight_map = weight_map[:,0]  # squeeze 到 [B,D,H,W]
            assert weight_map.shape == targets.shape, \
                f"weight_map shape {weight_map.shape} must be [B,D,H,W] to match targets {targets.shape}"

            w = weight_map.detach().float()
            # 裁剪，避免极端值毁掉训练
            if self.clip_min is not None or self.clip_max is not None:
                w = torch.clamp(w, min=self.clip_min, max=self.clip_max)

            if valid_mask is not None:
                w = w * valid_mask

            # 归一化策略
            if self.normalize == 'mean_valid':
                # 把分母设为权重在有效体素的总和 -> 平均权重 ~= 1
                denom = w.sum().clamp(min=1.0)
                loss = (base_loss * w).sum() / denom
                return loss
            elif self.normalize == 'mean_mask':
                # 维持与未加权时相同的分母（有效体素数），权重只影响分子
                denom = (valid_mask.sum() if valid_mask is not None else torch.tensor(base_loss.numel(), device=base_loss.device))
                denom = denom.float().clamp(min=1.0)
                loss = (base_loss * w).sum() / denom
                return loss
            else:
                # 不做归一化，走 reduction
                base_loss = base_loss * w

        # ----- 没有 weight_map 或不归一化时，按 reduction 输出 -----
        if self.reduction == 'mean':
            if valid_mask is not None:
                denom = valid_mask.sum().clamp(min=1)
                return base_loss.sum() / denom
            return base_loss.mean()
        elif self.reduction == 'sum':
            return base_loss.sum()
        else:
            return base_loss  # [B,D,H,W]
class FocalLossWeighted(nn.Module):
    """
    Focal Loss on PROBABILITIES with ONE-HOT targets, with optional voxel-wise weights and per-class alpha.
    Works for N-D (2D/3D) shapes.

    Args:
      alpha: Optional[List/ndarray/1D-Tensor of length C] per-class weights.
      gamma: focusing parameter (default 2.0)
      reduction: 'mean' | 'sum' | 'none'   (used ONLY when no custom normalization is requested)
      normalize: 'mean_valid' | 'mean_mask' | None
          - 'mean_valid': (loss*w).sum() / (w.sum() or valid_count)   <-- 推荐
          - 'mean_mask' : (loss*w).sum() / valid_count
          - None        : follow `reduction`
      clip_min/max: clamp weight_map to avoid extreme weights
    """
    def __init__(
        self,
        alpha=None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        normalize: str = 'mean_valid',
        clip_min: float | None = 0.1,
        clip_max: float | None = 10.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction
        self.normalize = normalize
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps

    def forward(
        self,
        probs: torch.Tensor,         # [B, C, *spatial] softmax/sigmoid outputs
        target_onehot: torch.Tensor, # [B, C, *spatial] one-hot
        weight_map: torch.Tensor | None = None,  # [B, 1, *spatial] or [B, *spatial]
        valid_mask: torch.Tensor | None = None,  # [B, 1, *spatial] or [B, *spatial]
    ):
        assert probs.shape == target_onehot.shape, f"probs {probs.shape} vs onehot {target_onehot.shape}"
        B, C = probs.shape[:2]
        spatial = probs.shape[2:]

        # matched-class prob p
        p = (probs * target_onehot).sum(dim=1).clamp_min(self.eps)  # [B, *spatial]

        # base focal term: -(1-p)^gamma * log(p)
        loss = -((1.0 - p) ** self.gamma) * p.log()                 # [B, *spatial]

        # per-class alpha -> voxel alpha via onehot
        if self.alpha is not None:
            alpha = self.alpha.reshape(1, C, *([1] * len(spatial))) # [1,C,*,*,*]
            alpha_voxel = (alpha * target_onehot).sum(dim=1)        # [B, *spatial]
            loss = loss * alpha_voxel

        # valid mask
        if valid_mask is not None:
            if valid_mask.dim() == probs.dim() - 1:  # [B,*spatial] -> ok
                vm = valid_mask
            elif valid_mask.dim() == probs.dim():
                vm = valid_mask.squeeze(1)           # [B,1,*] -> [B,*]
            else:
                raise ValueError("valid_mask dims mismatch")
            loss = loss * vm
        else:
            # 默认所有体素有效
            vm = torch.ones_like(loss, dtype=loss.dtype, device=loss.device)

        # weight_map
        if weight_map is not None:
            if weight_map.dim() == probs.dim():      # [B,1,*]
                w = weight_map.squeeze(1)
            elif weight_map.dim() == probs.dim() - 1:# [B,*]
                w = weight_map
            else:
                raise ValueError("weight_map dims mismatch")
            if (self.clip_min is not None) or (self.clip_max is not None):
                w = torch.clamp(w, min=self.clip_min if self.clip_min is not None else -float('inf'),
                                   max=self.clip_max if self.clip_max is not None else float('inf'))
            loss = loss * w
        else:
            w = torch.ones_like(loss, dtype=loss.dtype, device=loss.device)

        # normalization / reduction
        if self.normalize == 'mean_valid':
            denom = (w * vm).sum().clamp_min(1.0)
            return (loss).sum() / denom
        elif self.normalize == 'mean_mask':
            denom = vm.sum().float().clamp_min(1.0)
            return (loss).sum() / denom
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss  # [B,*spatial]

class BinaryFocalLogitsWeighted(nn.Module):
    """
    二类 Focal（输入 logits），支持体素权重/有效掩膜/归一化。
    logits: [B,1,*], targets: [B,1,*] or [B,*] in {0,1}
    """
    def __init__(self, gamma=2.0, reduction='mean', normalize='mean_valid',
                 clip_min=0.1, clip_max=10.0, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.normalize = normalize
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps

    def forward(self, logits, targets, weight_map=None, valid_mask=None):
        # logits -> prob
        p1 = torch.sigmoid(logits)                  # [B,1,*]
        p0 = 1.0 - p1
        # 统一形状
        if targets.dim() == logits.dim() - 1:       # [B,*] -> [B,1,*]
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # focal term
        # 针对 y=1: -(1-p)^g * log(p); 针对 y=0: -p^g * log(1-p)
        p_t = targets * p1.clamp_min(self.eps) + (1-targets) * p0.clamp_min(self.eps)
        mod  = ((1.0 - p_t) ** self.gamma)
        loss = - mod * p_t.log()                    # [B,1,*]

        # 有效掩膜
        if valid_mask is not None:
            if valid_mask.dim() == logits.dim() - 1:
                valid_mask = valid_mask.unsqueeze(1)
            loss = loss * valid_mask
        else:
            valid_mask = torch.ones_like(loss)

        # 体素权重
        if weight_map is not None:
            if weight_map.dim() == logits.dim():
                weight_map = weight_map.squeeze(1)
            if weight_map.dim() == logits.dim() - 1:
                weight_map = weight_map.unsqueeze(1)
            w = torch.clamp(weight_map, min=self.clip_min if self.clip_min is not None else -float('inf'),
                                        max=self.clip_max if self.clip_max is not None else  float('inf'))
            loss = loss * w
        else:
            w = torch.ones_like(loss)

        # 归一化
        if self.normalize == 'mean_valid':
            denom = (w * valid_mask).sum().clamp_min(1.0)
            return loss.sum() / denom
        elif self.normalize == 'mean_mask':
            denom = valid_mask.sum().float().clamp_min(1.0)
            return loss.sum() / denom
        else:
            return loss.mean() if self.reduction=='mean' else (loss.sum() if self.reduction=='sum' else loss)
def distill_loss(attn_low: torch.Tensor, attn_high: torch.Tensor) -> torch.Tensor:
    """L2 distillation between low-res map and upsampled stop-grad high-res map."""
    if attn_low.shape[2:] != attn_high.shape[2:]:
        attn_high = F.interpolate(attn_high, size=attn_low.shape[2:], mode='trilinear', align_corners=True)
    return F.mse_loss(attn_low, attn_high.detach())


# -----------------------------
# Example: total loss computation
# -----------------------------

import torch.nn.functional as F


import torch
import torch.nn.functional as F

def _ensure_5d(x):
    if x is None:
        return None
    if x.ndim == 4:
        x = x.unsqueeze(1)       # [B,D,H,W] -> [B,1,D,H,W]
    return x

def _resize_like(x, size, mode='nearest'):
    if x is None:
        return None
    if x.shape[2:] != size:
        x = F.interpolate(x.float(), size=size, mode=mode)
    return x

def compute_total_loss(outputs, av_gt, vessel_gt, w=None, alpha: float = 0.1):
    logits_av = outputs['logits_av']   # [B,3,D,H,W]
    logits_v  = outputs['logits_v']    # [B,1,D,H,W]
    size = logits_av.shape[2:]         # (D,H,W)

    # --- 标签：补维 + 对齐 ---
    av_gt     = _ensure_5d(av_gt)
    vessel_gt = _ensure_5d(vessel_gt)
    av_gt     = _resize_like(av_gt,     size, mode='nearest').long()     # [B,1,D,H,W]
    vessel_gt = _resize_like(vessel_gt, size, mode='nearest').long()     # [B,1,D,H,W]

    # --- 权重图（可选）：补维 + 对齐 ---
    w = _ensure_5d(w)
    w = _resize_like(w, size, mode='nearest')  # 语义权重图用 nearest

    # --- AV（三类）：直接用 logits 版 focal（你写的），注意 targets 要 4D ---
    focal_av = FocalLoss3D_W(gamma=0.0, normalize='mean_valid')(
        logits_av, av_gt.squeeze(1), w  # <- 这里一定要 squeeze 掉通道维
    )

    # --- Vessel（二类）：单通道 logits 用二分类 logits 版 focal ---
    focal_v  = BinaryFocalLogitsWeighted(gamma=0.0, normalize='mean_valid')(
        logits_v, vessel_gt, w          # 这里的 targets 用 [B,1,D,H,W] 或 [B,D,H,W] 都可
    )

    # --- 解码器侧蒸馏 ---
    Ld = distill_loss(outputs['attn_d1'], outputs['attn_d2']) \
       + distill_loss(outputs['attn_d2'], outputs['attn_d3']) \
       + distill_loss(outputs['attn_d3'], outputs['attn_d4'])

    total = focal_av + focal_v + alpha * Ld
    logs = {
        'loss_total': float(total.detach().cpu()),
        'focal_av': float(focal_av.detach().cpu()),
        'focal_v' : float(focal_v.detach().cpu()),
        'distill' : float(Ld.detach().cpu()),
    }
    return total, logs





if __name__ == "__main__":
    # Sanity check run
    x = torch.randn(2, 1, 64, 96, 96)  # e.g., CT + 2 priors
    av_gt = torch.randint(0, 3, (2,1,64,96,96))
    v_gt  = (av_gt > 0).long()  # vessel mask from AV

    model = TSCNN3D(in_channels=3, base_ch=24)
    out = model(x)
    loss, logs = compute_total_loss(out, av_gt, v_gt, alpha=0.1)
    print('OK:', {k: round(v, 4) for k,v in logs.items()})
