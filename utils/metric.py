import torchio as tio
from pathlib import Path
import torch
import numpy as np
import copy
from monai.metrics import compute_hausdorff_distance


def all_metric(gt, wt_pred, et_pred, tc_pred):
    wt_dice, wt_recall, wt_specificity, wt_hs95 = metric(gt[0], wt_pred)
    et_dice, et_recall, et_specificity, et_hs95 = metric(gt[1], et_pred)
    tc_dice, tc_recall, tc_specificity, tc_hs95 = metric(gt[2], tc_pred)
    return (
        [wt_dice, wt_recall, wt_specificity, wt_hs95],
        [et_dice, et_recall, et_specificity, et_hs95],
        [tc_dice, tc_recall, tc_specificity, tc_hs95],
    )


def metric(gt, pred, spacing=None):
    # * input shape: (batch, channel, height, width)

    preds = pred.detach().numpy()
    gts = gt.detach().numpy()

    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)  # float data does not support bit_and and bit_or

    if spacing:
        pred = pred[None, :, :, :, :]
        gdth = gdth[None, :, :, :, :]
        hs95 = compute_hausdorff_distance(pred, gdth, percentile=95, spacing=spacing).numpy()[0][0]

    gdth = gdth.squeeze()  # (240,240) 去除 size 为 1 的维度
    pred = pred.squeeze()  # (240,240) 去除 size 为 1 的维度
    fp_array = copy.deepcopy(pred)  # keep pred unchanged  # 拷贝预测结果，用于生成 FP 掩码
    fn_array = copy.deepcopy(gdth)  # 拷贝真实标签，用于生成 FN 掩码
    gdth_sum = np.sum(gdth)  # 正样本数（P）
    pred_sum = np.sum(pred)  # 预测为正的像素数（P_pred）
    intersection = gdth & pred  # 按位与，得到 TP 掩码
    union = gdth | pred  # 按位或，得到 P ∪ P_pred
    intersection_sum = np.count_nonzero(intersection)  # TP 数
    union_sum = np.count_nonzero(union)  # 联集数，用于 Jaccard

    tp_array = intersection  # 真阳性区域

    tmp = pred - gdth  # 预测为1而GT为0的位置 → FP
    fp_array[tmp < 1] = 0  # 只保留 FP 部分

    tmp2 = gdth - pred  # GT为1而预测为0的位置 → FN
    fn_array[tmp2 < 1] = 0  # 只保留 FN 部分

    tn_array = np.ones(gdth.shape) - union  # 非预测非GT区域 = 1 - 联集

    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

    smooth = 0.001
    precision = tp / (pred_sum + smooth)
    recall = tp / (gdth_sum + smooth)
    specificity = tn / (tn + fp + smooth)

    false_positive_rate = fp / (fp + tn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)

    jaccard = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

    # hs95 = 0
    # hs95 = hausdorff_95(gdth, pred, (1, 1))
    # hs95 = hausdorff_95(preds, gts, (1, 1, 1))

    if spacing:
        return precision, recall, jaccard, dice, hs95
    else:
        return jaccard, dice


# def hausdorff_95(gt_array, pred_array, spacing):
#     '''
#     :params gt_array: ground true mask
#     :params pred_array: the result of segmentation
#     :params num_class: label number
#     :params spacing: spacing of the image
#     '''
#     Hausdorff_95_score = []
#     gt_array = gt_array.astype(bool)
#     pred_array = pred_array.astype(bool)

#     # compute Hausdorff_95 score
#     surface_distances = surface_distance.compute_surface_distances(pred_array, gt_array, spacing)
#     hs95 = surface_distance.compute_robust_hausdorff(surface_distances, 95)
#     return hs95
import torch
import torch.nn.functional as F


def compute_multiclass_dice(
    predict,
    target,
    num_classes,
    epsilon=1e-6,
    ignore_background=True,
    background_class=0,
):
    """
    计算多分类 Dice 系数（增强版）

    参数:
        predict: 模型预测的 logits，shape [B, C, D, H, W]
        target: 真实标签，shape [B, 1, D, H, W] 或 [B, D, H, W]
        num_classes: 总类别数
        epsilon: 防止除零
        ignore_background: 是否忽略背景类别（默认 True）
        background_class: 背景类别的索引（默认 0）

    返回:
        dice_per_class: tensor[C]，每类 Dice 分数
        dice_mean: 标量，平均 Dice 分数（仅对有效类别）
    """
    if target.ndim == predict.ndim:
        target = target.squeeze(1)  # [B, D, H, W]

    predict_prob = F.softmax(predict, dim=1)  # [B, C, D, H, W]

    # one-hot 编码 target: [B, C, D, H, W]
    target_onehot = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()

    B, C = predict_prob.shape[:2]
    predict_flat = predict_prob.view(B, C, -1)
    target_flat = target_onehot.view(B, C, -1)

    # 计算交并比
    intersection = (predict_flat * target_flat).sum(-1)  # [B, C]
    union = predict_flat.sum(-1) + target_flat.sum(-1)  # [B, C]
    dice = (2 * intersection + epsilon) / (union + epsilon)  # [B, C]

    # 每类 Dice（batch 平均）
    dice_per_class = dice.mean(dim=0)  # [C]

    # 仅对“在 target 中出现的类别”统计平均 Dice
    class_present = target_flat.sum(dim=2).sum(dim=0) > 0  # [C]

    if ignore_background and background_class < num_classes:
        class_present[background_class] = False  # 忽略背景类

    # 平均 Dice，仅对出现的前景类
    valid_dice = dice_per_class[class_present]
    dice_mean = valid_dice.mean() if valid_dice.numel() > 0 else torch.tensor(0.0, device=predict.device)

    return dice_per_class, dice_mean


# =======


def dice_coefficient(pred_mask, true_mask, num_classes=None, smooth=1e-6):
    """
    计算 Dice 系数（支持多分类和二分类）

    参数:
        pred_mask (torch.Tensor): 预测的类别标签，形状 [bs, (1,) h, w, d] 或 [bs, h, w, d]，值为整数 0 ~ c-1
        true_mask (torch.Tensor): 真实的类别标签，形状需与 pred_mask 相同
        num_classes (int): 类别数（若为 None，则自动从 true_mask 推断）
        smooth (float): 平滑因子，防止除以零

    返回:
        dice (torch.Tensor): Dice 系数（标量或各类别 Dice 的均值）
    """
    # 确保输入是整数标签
    # assert pred_mask.dtype == torch.long, "pred_mask 应为整数标签（torch.long）"
    # assert true_mask.dtype == torch.long, "true_mask 应为整数标签（torch.long）"
    # print(pred_mask.shape)
    # print(true_mask.shape)
    # print(pred_mask.shape)

    # 统一维度为 [bs, h, w, d]
    if pred_mask.dim() == 5:
        pred_mask = pred_mask.squeeze(1)  # 去除可能的单通道维度 [bs,1,h,w,d] -> [bs,h,w,d]
    if true_mask.dim() == 5:
        true_mask = true_mask.squeeze(1)

    # 自动推断类别数（如果未指定）
    if num_classes is None:
        num_classes = max(pred_mask.max().item(), true_mask.max().item()) + 1

    # 初始化 Dice 存储
    dice_per_class = torch.zeros(num_classes, device=pred_mask.device)

    # 逐类别计算 Dice
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).float()  # 当前类别的二值掩膜
        true_cls = (true_mask == cls).float()

        intersection = (pred_cls * true_cls).sum()
        union = pred_cls.sum() + true_cls.sum()

        dice_per_class[cls] = (2. * intersection + smooth) / (union + smooth)

    # 返回各类别 Dice 的平均值（忽略背景类需手动处理）
    return dice_per_class.mean()  # 或 return dice_per_class[1:].mean() 忽略背景类

# ========wk

import torch
import torch.nn.functional as F
import numpy as np
from monai.metrics import compute_hausdorff_distance

def multiclass_metrics(gt, pred, num_classes, spacing=None, smooth=1e-5):
    """
    多分类医学图像评估函数：返回平均 Dice、Jaccard、Precision、Recall（+ 可选 Hausdorff95）

    参数:
        gt: [B, 1, H, W] or [B, 1, H, W, D] → 整数标签图（非 one-hot）
        pred: same as gt → 预测标签图（整数索引）
        num_classes: int → 类别数
        spacing: tuple or None → 体素间距（用于 Hausdorff95）
        smooth: float → 平滑因子

    返回:
        precision_mean, recall_mean, jaccard_mean, dice_mean, (可选) hausdorff95_mean
    """
    assert gt.shape == pred.shape, "gt and pred must have the same shape"
    gt = gt.squeeze(1)
    pred = pred.squeeze(1)

    spatial_dims = list(range(1, gt.dim()))

    gt_onehot = F.one_hot(gt.long(), num_classes).permute(0, -1, *range(1, gt.dim())).float()
    pred_onehot = F.one_hot(pred.long(), num_classes).permute(0, -1, *range(1, pred.dim())).float()

    dice_total = 0.0
    jaccard_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    hausdorff_total = 0.0 if spacing else None

    for c in range(num_classes):
        gtc = gt_onehot[:, c]
        predc = pred_onehot[:, c]

        intersection = (gtc * predc).sum(dim=spatial_dims)
        pred_sum = predc.sum(dim=spatial_dims)
        gt_sum = gtc.sum(dim=spatial_dims)
        union = pred_sum + gt_sum

        dice = ((2 * intersection + smooth) / (union + smooth)).mean().item()
        jaccard = ((intersection + smooth) / (union - intersection + smooth)).mean().item()
        precision = (intersection / (pred_sum + smooth)).mean().item()
        recall = (intersection / (gt_sum + smooth)).mean().item()

        dice_total += dice
        jaccard_total += jaccard
        precision_total += precision
        recall_total += recall

        if spacing:
            gtc_bin = gtc.detach().cpu().numpy().astype(np.uint8)
            predc_bin = predc.detach().cpu().numpy().astype(np.uint8)
            hd = compute_hausdorff_distance(
                torch.tensor(predc_bin[None]), torch.tensor(gtc_bin[None]),
                percentile=95.0, spacing=spacing
            )[0][0].item()
            hausdorff_total += hd

    dice_mean = dice_total / num_classes
    jaccard_mean = jaccard_total / num_classes
    precision_mean = precision_total / num_classes
    recall_mean = recall_total / num_classes

    if spacing:
        hausdorff_mean = hausdorff_total / num_classes
        return precision_mean, recall_mean, jaccard_mean, dice_mean, hausdorff_mean
    else:
        return jaccard_mean, dice_mean





