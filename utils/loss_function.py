from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.metric import dice_coefficient


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class Binary_Loss(nn.Module):

    def __init__(self):
        super(Binary_Loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, model_output, targets):
        #targets[targets == 0] = -1

        # torch.empty(3, dtype=torch.long)
        # model_output = model_output.long()
        # targets = targets.long()
        # print(model_output)
        # print(F.sigmoid(model_output))
        # print(targets)
        # print('kkk')
        # model_output =torch.LongTensor(model_output.cpu())
        # targets =torch.LongTensor(targets.cpu())
        # model_output = model_output.type(torch.LongTensor)
        # targets = targets.type(torch.LongTensor)
        loss = self.criterion(model_output, targets)

        return loss


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.eplison = 1e-5
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        # num = predict.size(0)
        # pre = torch.sigmoid(predict).view(num, -1)
        # tar = target.view(num, -1)
        #
        # intersection = (pre * tar).sum(-1).sum()
        # union = (pre + tar).sum(-1).sum()
        # dice_loss = 1 - 2 * (intersection + self.eplison) / (union + self.eplison)





        return 1.0- dice_coefficient(predict, target, 2)

        # dice = BinaryDiceLoss(**self.kwargs)
        # total_loss = 0
        # predict = F.softmax(predict, dim=1)
        #
        # for i in range(target.shape[1]):
        #     if i != self.ignore_index:
        #         dice_loss = dice(predict[:, i], target[:, i])
        #         if self.weight is not None:
        #             assert self.weight.shape[0] == target.shape[1], \
        #                 'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
        #             dice_loss *= self.weights[i]
        #         total_loss += dice_loss
        #
        # return total_loss/target.shape[1]



class DiceLossss(nn.Module):

    def __init__(self, n_classes):
        super(DiceLossss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, epsilon=1e-6):
        """
        多分类Dice损失函数，支持忽略背景或任意类别

        参数:
            num_classes: 类别数（包含背景）
            ignore_index: 忽略的类别索引（默认0为背景）
            epsilon: 防止除0
        """
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, predict, target):
        """
        参数:
            predict: 网络输出，形状为 [B, C, ...]，建议已 softmax（也可 raw logits）
            target: 真实标签，形状为 [B, ...]，为整型类别图（不需要 one-hot）

        返回:
            Dice loss scalar
        """
        B, C = predict.shape[:2]

        # 如果是 raw logits，转成概率
        if predict.max() > 1.0:
            predict = F.softmax(predict, dim=1)

        # 去掉 target 的单通道维度（如果有）
        if target.ndim == predict.ndim:
            target = target.squeeze(1)

        # 转 one-hot：形状 [B, ..., C]
        target_onehot = F.one_hot(target.long(), num_classes=self.num_classes)

        # 变成 [B, C, ...]
        dims = list(range(target_onehot.ndim))
        target_onehot = target_onehot.permute(0, -1, *dims[1:-1]).float()

        assert target_onehot.shape == predict.shape, f"shape mismatch: {target_onehot.shape} vs {predict.shape}"

        # 展平：[B, C, -1]
        predict_flat = predict.view(B, C, -1)
        target_flat = target_onehot.view(B, C, -1)

        # 交集 + 平方和
        intersection = (predict_flat * target_flat).sum(-1)
        union = predict_flat.sum(-1) + target_flat.sum(-1)

        # 哪些类别在该 batch 中有效（target 非零）
        valid_mask = (target_flat.sum(-1) > 0).float()

        # 忽略背景类别
        valid_mask[:, self.ignore_index] = 0

        # 加权 dice score
        dice_score = (2 * intersection + self.epsilon) / (union + self.epsilon)
        dice_score = dice_score * valid_mask

        # 最终 loss = 1 - 有效前景类别的平均 dice
        total_valid = valid_mask.sum(dim=1) + self.epsilon  # 每个 batch 有效类别数
        loss = (1 - dice_score.sum(dim=1) / total_valid).mean()

        return loss

# =====wk
import torch
import torch.nn.functional as F


class Dice_Loss(nn.Module):
    """
    计算多分类dice,输入应该是模型输出 logit（predict）和真实标签target
    """
    def __init__(self):
        super(Dice_Loss, self).__init__()

    def forward(self, predict, target, num_class=2):
        total_loss = 0.0
        epsilon = 1e-6
        predict = torch.softmax(predict, dim=1)

        # print(f"target===={target.shape}")
        for i in range(0, num_class):
            predict_i = predict[:,i].float().reshape(-1)
            # print(f"predict_i===={predict_i.shape}")
            target = (target == i).float().reshape(-1)
            # print(f"target_i===={target.shape}")
            intersection = (predict_i * target).sum()
            union = predict_i.pow(2).sum() + target.pow(2).sum()

            dice = (2 * intersection + epsilon) / (union + epsilon)
            dice_loss = 1 - dice.mean()
            total_loss += dice_loss

        return total_loss/(num_class)


# =====  cvpr 2020 保持管状结构的损失
def soft_dice(y_true, y_pred):
    """计算dice损失

    Args:
        y_true: 真实标签 [B, C, H, W] 或 [B, C, H, W, D]
        y_pred: 预测概率 [B, C, H, W] 或 [B, C, H, W, D]

    Returns:
        dice损失值
    """
    smooth = 1.0
    intersection = torch.sum(y_true * y_pred)
    coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return 1.0 - coeff


class SoftSkeletonize(nn.Module):
    """软骨架化模块"""

    def __init__(self, num_iter=30):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape) == 4:  # 2D: [B, C, H, W]
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:  # 3D: [B, C, H, W, D]
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape) == 4:  # 2D
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:  # 3D
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):
        return self.soft_skel(img)


class soft_dice_cldice(nn.Module):
    """通用的Dice-cLDice损失函数，支持二分类和多分类"""

    def __init__(self, iter_=3, alpha=1, smooth=1., exclude_background=False, num_classes=3):
        """
        Args:
            iter_: 骨架化迭代次数
            alpha: cLDice权重 (0-1)，0表示只用Dice，1表示只用cLDice
            smooth: 平滑因子
            exclude_background: 是否排除背景类
            num_classes: 类别数量
        """
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.exclude_background = exclude_background
        self.num_classes = num_classes
        self.soft_skeletonize = SoftSkeletonize(num_iter=self.iter)

        # 参数验证
        assert 0 <= alpha <= 1, "alpha应该在0-1之间"
        assert num_classes >= 2, "类别数至少为2"

    def forward(self, y_true, y_pred):
        """
        Args:
            y_true: 真实标签，形状为 [B, H, W] 或 [B, H, W, D] (索引格式)
                    或 [B, C, H, W] 或 [B, C, H, W, D] (one-hot格式)
            y_pred: 网络输出的logits，形状为 [B, C, H, W] 或 [B, C, H, W, D]

        Returns:
            组合损失值
        """
        # 将预测logits转换为概率图
        y_pred = torch.softmax(y_pred, dim=1)

        # 处理真实标签：如果是索引格式，转换为one-hot
        if y_true.dim() == y_pred.dim() - 1:  # 缺少通道维度
            # 确保标签是整数类型
            y_true = y_true.long()
            # 创建one-hot编码
            y_true = F.one_hot(y_true, num_classes=self.num_classes)

            # 调整维度顺序: [B, H, W, D, C] -> [B, C, H, W, D]
            if y_pred.dim() == 5:  # 3D情况
                y_true = y_true.permute(0, 4, 1, 2, 3).float()
            else:  # 2D情况
                y_true = y_true.permute(0, 3, 1, 2).float()

        # 确保数据类型一致
        y_true = y_true.float()

        # 如果排除背景，则移除第0通道
        if self.exclude_background:
            y_true = y_true[:, 1:, ...]
            y_pred = y_pred[:, 1:, ...]
            num_eval_classes = self.num_classes - 1
        else:
            num_eval_classes = self.num_classes

        # 如果没有需要评估的类别，返回0
        if num_eval_classes == 0:
            return torch.tensor(0.0, device=y_pred.device)

        # 计算每个类别的Dice损失
        dice_loss = 0.0
        for i in range(num_eval_classes):
            # 保持通道维度 [B, 1, H, W] 或 [B, 1, H, W, D]
            dice_loss += soft_dice(
                y_true[:, i:i + 1],
                y_pred[:, i:i + 1]
            )
        dice_loss /= num_eval_classes

        # 计算每个类别的clDice损失
        cl_dice_loss = 0.0
        for i in range(num_eval_classes):
            # 对每个类别单独进行骨架化
            skel_pred = self.soft_skeletonize(y_pred[:, i:i + 1])  # [B, 1, H, W] 或 [B, 1, H, W, D]
            skel_true = self.soft_skeletonize(y_true[:, i:i + 1])

            # 计算拓扑精度和敏感度
            tprec_numerator = torch.sum(skel_pred * y_true[:, i:i + 1]) + self.smooth
            tprec_denominator = torch.sum(skel_pred) + self.smooth
            tprec = tprec_numerator / tprec_denominator

            tsens_numerator = torch.sum(skel_true * y_pred[:, i:i + 1]) + self.smooth
            tsens_denominator = torch.sum(skel_true) + self.smooth
            tsens = tsens_numerator / tsens_denominator

            # 计算clDice，防止除零
            cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + 1e-8)
            cl_dice_loss += cl_dice

        cl_dice_loss /= num_eval_classes

        # 组合损失
        total_loss = (1.0 - self.alpha) * dice_loss + self.alpha * cl_dice_loss

        return total_loss
# =====
# ====HiPaS用的

import torch
import torch.nn as nn
import torch.nn.functional as F

class Configurable3DSegmentationLoss(nn.Module):
    def __init__(self,
                 weight_dice: float = 1.0,
                 weight_penalty: float = 0.0,
                 ignore_background_dice: bool = False,
                 eps: float = 1e-6):
        super(Configurable3DSegmentationLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_penalty = weight_penalty
        self.ignore_background_dice = ignore_background_dice
        self.eps = eps

        self.class_1_idx = 1  # 动脉
        self.class_2_idx = 2  # 静脉

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # --- 维度检查 ---
        if logits.dim() != 5:
            raise ValueError(f"Expected 5D logits [B, C, D, H, W], but got {logits.dim()} dimensions.")
        if targets.dim() != 4:
            raise ValueError(f"Expected 4D targets [B, D, H, W], but got {targets.dim()} dimensions.")

        # 1. 输入预处理
        num_classes = logits.shape[1]
        probas = F.softmax(logits, dim=1)
        targets = targets.long()

        # ==================================================================
        # 2. 计算 Dice Loss (保持原有实现)
        # ==================================================================
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        probas_flat = probas.contiguous().view(probas.shape[0], num_classes, -1)
        targets_one_hot_flat = targets_one_hot.contiguous().view(targets_one_hot.shape[0], num_classes, -1)

        intersection = torch.sum(probas_flat * targets_one_hot_flat, dim=2)
        cardinality = torch.sum(probas_flat + targets_one_hot_flat, dim=2)
        dice_score = (2. * intersection + self.eps) / (cardinality + self.eps)

        if self.ignore_background_dice:
            dice_loss = 1 - dice_score[:, 1:].mean()
        else:
            dice_loss = 1 - dice_score.mean()

        # ==================================================================
        # 3. 实现原文的重叠交叉损失 (Overlap-Cross Loss)
        # ==================================================================

        # 获取动脉和静脉的预测概率和真实标签
        # probas: [B, C, D, H, W]
        P_A = probas[:, self.class_1_idx, :, :, :]  # 动脉预测概率
        P_V = probas[:, self.class_2_idx, :, :, :]  # 静脉预测概率

        # targets: [B, D, H, W]
        T_A = (targets == self.class_1_idx).float()  # 动脉真实标签
        T_V = (targets == self.class_2_idx).float()  # 静脉真实标签

        # 计算分子：动脉预测与静脉真实的点积 + 静脉预测与动脉真实的点积
        numerator = (P_A * T_V).sum() + (P_V * T_A).sum()

        # 计算分母：所有预测概率和真实标签的总和
        # 注意：这里使用所有类别的预测概率和真实标签，而不仅仅是动脉和静脉
        denominator = probas.sum() + targets_one_hot.sum()

        # 原文公式：L_overlap = (P_A · T_V + P_V · T_A) / (P + T)
        overlap_loss = numerator / (denominator + self.eps)

        # ==================================================================
        # 4. 合并总损失
        # ==================================================================
        total_loss = self.weight_dice * dice_loss + self.weight_penalty * overlap_loss

        return total_loss


# ====
class FocalLoss3D(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean', ignore_index=None):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None

        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits:  [B, C, D, H, W]
        targets: [B, D, H, W]
        """
        # 提前处理忽略索引
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            # 创建临时targets避免one-hot错误
            targets_temp = targets.clone()
            targets_temp[~valid_mask] = 0  # 设置为有效类别0
        else:
            valid_mask = None
            targets_temp = targets

        # 计算log-probs和probs
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # 更高效的pt和log_pt计算
        B, C, D, H, W = logits.shape
        targets_onehot = F.one_hot(targets_temp, num_classes=C)  # [B, D, H, W, C]
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]

        # 添加数值安全措施
        pt = (probs * targets_onehot).sum(dim=1).clamp(min=1e-6)  # 防止0
        log_pt = (log_probs * targets_onehot).sum(dim=1)  # 已从log_softmax来，数值安全

        # Focal权重
        focal_weight = (1.0 - pt) ** self.gamma

        # Alpha权重（更高效实现）
        if self.alpha is not None:
            # 直接根据targets索引alpha值
            alpha_factor = self.alpha[targets_temp]  # [B, D, H, W]
        else:
            alpha_factor = 1.0

        # 损失计算
        loss = -alpha_factor * focal_weight * log_pt

        # 应用忽略位置mask
        if self.ignore_index is not None:
            loss = loss * valid_mask

        # Reduction处理
        if self.reduction == 'mean':
            if valid_mask is not None:
                return loss.sum() / valid_mask.sum().clamp(min=1)
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # [B, D, H, W]
# =======ynet loss
import torch
import torch.nn as nn
import torch.nn.functional as F
# h-loss
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


import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlapMarginLossPair(nn.Module):
    """
    互斥重叠惩罚 + 间隔分离损失（双分支版本，改良）
    - logits_a, logits_v: [B, C, D, H, W]（未激活）
    - targets_a, targets_v: [B, D, H, W] 或 [B, 1, D, H, W]，二值{0,1}或含 ignore_index
    """
    def __init__(
        self,
        lam_overlap: float = 0.5,
        lam_margin: float  = 0.5,
        margin: float      = 0.25,
        use_uncertainty_mask: bool = True,
        delta: float = 0.20,
        ignore_index: int | None = None,
        foreground_channel: int = 1,
        restrict_overlap_to_fg: bool = True,  # ⭐ 新增：仅在前景邻域施加重叠惩罚
        pred_gate: float | None = None,       # 可选：再用预测门限进一步约束重叠区域，如 0.1~0.2
    ):
        super().__init__()
        self.lam_overlap = lam_overlap
        self.lam_margin  = lam_margin
        self.margin      = margin
        self.use_uncertainty_mask = use_uncertainty_mask
        self.delta = delta
        self.ignore_index = ignore_index
        self.fg_ch = foreground_channel
        self.restrict_overlap_to_fg = restrict_overlap_to_fg
        self.pred_gate = pred_gate

    @staticmethod
    def _ensure_4d_labels(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 5 and t.shape[1] == 1:
            t = t[:, 0]
        return t

    def _foreground_prob(self, logits: torch.Tensor) -> torch.Tensor:
        C = logits.shape[1]
        if C == 1:
            p = torch.sigmoid(logits)
        else:
            assert 0 <= self.fg_ch < C, f"foreground_channel={self.fg_ch} out of range for C={C}"
            probs = torch.softmax(logits, dim=1)
            p = probs[:, self.fg_ch:self.fg_ch+1]
        return p  # [B,1,D,H,W]

    def forward(self, logits_a, targets_a, logits_v, targets_v):
        device = logits_a.device

        # --- 概率 ---
        pA = self._foreground_prob(logits_a)  # [B,1,D,H,W]
        pV = self._foreground_prob(logits_v)  # [B,1,D,H,W]

        # --- 标签 ---
        ta = self._ensure_4d_labels(targets_a).to(device)  # [B,D,H,W]
        tv = self._ensure_4d_labels(targets_v).to(device)

        if self.ignore_index is not None:
            valid_a = (ta != self.ignore_index)
            valid_v = (tv != self.ignore_index)
        else:
            valid_a = torch.ones_like(ta, dtype=torch.bool)
            valid_v = torch.ones_like(tv, dtype=torch.bool)

        # ===========================
        # 1) 互斥重叠惩罚  L_overlap
        # ===========================
        overlap_core = (pA * pV).squeeze(1)  # [B,D,H,W]

        # 基础有效掩膜：至少一个标签非 ignore
        valid_mask = (valid_a | valid_v)

        # （可选）限制在前景邻域：仅对有标注前景邻域惩罚，更聚焦边界/混淆区
        if self.restrict_overlap_to_fg:
            fg_sup_mask = ((ta == 1) & valid_a) | ((tv == 1) & valid_v)
            valid_mask = valid_mask & fg_sup_mask

        # 不确定掩膜：|pA-pV| < delta
        if self.use_uncertainty_mask:
            uncertain = (torch.abs(pA - pV).squeeze(1) < self.delta)
            valid_mask = valid_mask & uncertain

        # 预测门限（进一步过滤两支都几乎为0的背景，如 pred_gate=0.05~0.2）
        if self.pred_gate is not None:
            pred_union = ((pA.squeeze(1) > self.pred_gate) | (pV.squeeze(1) > self.pred_gate))
            valid_mask = valid_mask & pred_union

        denom_overlap = valid_mask.sum().clamp_min(1)
        L_overlap = (overlap_core * valid_mask).sum() / denom_overlap

        # ===========================
        # 2) 间隔分离损失  L_margin
        # ===========================
        posA = (ta == 1) & valid_a
        posV = (tv == 1) & valid_v

        if posA.any():
            hinge_A = F.relu(self.margin - (pA - pV).squeeze(1))
            L_margin_A = (hinge_A * posA).sum() / posA.sum().clamp_min(1)
        else:
            L_margin_A = torch.tensor(0.0, device=device)

        if posV.any():
            hinge_V = F.relu(self.margin - (pV - pA).squeeze(1))
            L_margin_V = (hinge_V * posV).sum() / posV.sum().clamp_min(1)
        else:
            L_margin_V = torch.tensor(0.0, device=device)

        # 用平均而不是相加，尺度更稳
        L_margin = 0.5 * (L_margin_A + L_margin_V)

        total = self.lam_overlap * L_overlap + self.lam_margin * L_margin

        # # 便于监控
        # logs = {
        #     "L_overlap": L_overlap.detach(),
        #     "L_margin":  L_margin.detach(),
        #     "L_margin_A": L_margin_A.detach(),
        #     "L_margin_V": L_margin_V.detach(),
        #     "overlap_vox": denom_overlap.detach(),
        # }
        return total






