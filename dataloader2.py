# -*- coding: utf-8 -*-
# dataloader_single.py —— 单标签任务版（适用于单编码器单解码器模型）

from pathlib import Path
import torch
import torch.utils.data
import torchio as tio
from torchio.transforms import (
    Pad, RandomFlip, RandomAffine, RandomElasticDeformation, RandomNoise,
    RandomBiasField, ZNormalization, OneOf, Compose, RandomGamma
)
from torchio.data import UniformSampler, LabelSampler
from torchio import SubjectsDataset, Queue


# ---------- 工具函数 ----------
from typing import List

def _sorted_nii(path_like, prefer_uncompressed: bool = True) -> List[Path]:
    """
    返回按文件名排序后的 NIfTI 列表，支持 .nii 与 .nii.gz
    - prefer_uncompressed=True 时，若同名既有 .nii 又有 .nii.gz，优先保留 .nii
    """
    if not path_like:
        return []
    p = Path(path_like)
    nii = list(p.glob("*.nii"))
    niigz = list(p.glob("*.nii.gz"))

    # 处理潜在的“同名重复”（xxx.nii 与 xxx.nii.gz 都存在）
    if prefer_uncompressed:
        stem_set_nii = {f.stem for f in nii}  # .nii 的 stem
        # .nii.gz 的 stem 要去掉两层后缀
        dedup_niigz = [f for f in niigz if f.name[:-7] not in stem_set_nii]
        files = nii + dedup_niigz
    else:
        stem_set_niigz = {f.name[:-7] for f in niigz}
        dedup_nii = [f for f in nii if f.stem not in stem_set_niigz]
        files = niigz + dedup_nii

    return sorted(files, key=lambda x: x.name)
def _base_stem(p: Path) -> str:
    """
    统一得到文件基名（去除 .nii 和 .nii.gz 后缀），用于稳健配对 image/gt/weight。
    例：
      a.nii.gz -> a
      b.nii    -> b
    """
    n = p.name
    if n.endswith(".nii.gz"):
        return n[:-7]
    if n.endswith(".nii"):
        return n[:-4]
    return p.stem

# ---------- 构造 Subjects ----------
def get_subjects_single(config, split: str = "train"):
    """
    支持 split ∈ {'train', 'val', 'test', 'predict'}
    - predict 阶段若提供 pred_gt_path，则一并加载 gt；
      若 use_pred_weights=True 且提供 pred_weights_path，则加载 weight。
    """
    if split == "predict":
        img_list = _sorted_nii(getattr(config, "pred_data_path", ""))
        gt_list  = _sorted_nii(getattr(config, "pred_gt_path", ""))  # 可为空，但你的predict()需要gt，建议提供
        use_weights = bool(getattr(config, "use_pred_weights", False))
        w_list  = _sorted_nii(getattr(config, "pred_weights_path", "")) if use_weights else []
    elif split == "test":
        img_list = _sorted_nii(getattr(config, "pred_data_path", ""))
        gt_list  = _sorted_nii(getattr(config, "pred_gt_path", ""))
        use_weights = bool(getattr(config, "use_test_weights", False))
        w_list  = _sorted_nii(getattr(config, "test_weights_path", "")) if use_weights else []
    elif split == "val":
        img_list = _sorted_nii(getattr(config, "val_data_path", ""))
        gt_list  = _sorted_nii(getattr(config, "val_gt_path", ""))
        use_weights = bool(getattr(config, "use_val_weights", False))
        w_list  = _sorted_nii(getattr(config, "val_weights_path", "")) if use_weights else []
    else:  # train
        img_list = _sorted_nii(getattr(config, "data_path", ""))
        gt_list  = _sorted_nii(getattr(config, "gt_path", ""))
        use_weights = True
        w_list  = _sorted_nii(getattr(config, "weights", ""))

    if not img_list:
        raise FileNotFoundError(f"[{split}] 未找到影像，请检查路径。")

    # 映射成 {基名: Path}，用交集稳健配对
    img_map = { _base_stem(p): p for p in img_list }
    gt_map  = { _base_stem(p): p for p in gt_list } if gt_list else {}
    w_map   = { _base_stem(p): p for p in w_list } if (use_weights and w_list) else {}

    need_gt = split in {"train", "val", "test", "predict"}  # 你的 predict() 会取 gt，因此默认需要
    keys = sorted(set(img_map)) if not need_gt else sorted(set(img_map) & set(gt_map))

    # 告警提示（不终止）：哪些样本没有匹配到 GT / 权重
    if need_gt:
        miss_gt = sorted(set(img_map) - set(gt_map))
        if miss_gt:
            print(f"[{split}] 警告：{len(miss_gt)} 个样本未匹配到 GT，例如：{miss_gt[:5]}（已按交集构建subjects）")
    if use_weights and w_map:
        miss_w = sorted(set(keys) - set(w_map))
        if miss_w:
            print(f"[{split}] 警告：{len(miss_w)} 个样本未匹配到权重图，例如：{miss_w[:5]}")

    subjects = []
    for k in keys:
        d = {'source': tio.ScalarImage(img_map[k])}
        if need_gt:
            d['gt'] = tio.LabelMap(gt_map[k])
        if use_weights and k in w_map:
            d['weight'] = tio.ScalarImage(w_map[k])
        subjects.append(tio.Subject(d))

    if not subjects:
        raise RuntimeError(f"[{split}] subjects 构造为空，请检查路径或命名是否对应。")
    return subjects



# ---------- 训练集 ----------
class Dataset(torch.utils.data.Dataset):
    """
    单标签训练集：
    - 使用 LabelSampler 控制前景采样比例（默认 0.33）
    - 支持随机增广（config.aug=True）
    - 支持 voxel-wise 权重
    """
    def __init__(self, config):
        super().__init__()
        self.subjects = get_subjects_single(config, split="train")

        patch_size = tuple(getattr(config, 'patch_size', (64, 64, 64)))
        pad = Pad(padding=patch_size, padding_mode=0)

        if bool(getattr(config, "aug", False)):
            transforms = Compose([
                pad,
                RandomBiasField(),
                ZNormalization(include=['source']),
                RandomNoise(),
                RandomFlip(axes=(0,)),
                OneOf({
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }),
            ])
        else:
            transforms = Compose([
                ZNormalization(include=['source']),
                RandomGamma(log_gamma=(-0.3, 0.3)),
                pad,
            ])

        self.training_set = SubjectsDataset(self.subjects, transform=transforms)

        # 前景采样控制
        pos_ratio = float(getattr(config, "pos_ratio", 0.33))
        pos_ratio = max(0.0, min(1.0, pos_ratio))
        label_probs = {0: 1.0 - pos_ratio, 1: pos_ratio}

        self.queue_dataset = Queue(
            self.training_set,
            max_length=int(getattr(config, "queue_length", 10)),
            samples_per_volume=int(getattr(config, "samples_per_volume", 10)),
            sampler=LabelSampler(
                patch_size=patch_size,
                label_name='gt',
                label_probabilities=label_probs
            ),
            num_workers=int(getattr(config, "queue_num_workers", 0)),
        )


# ---------- 验证集 ----------
class DatasetVal(torch.utils.data.Dataset):
    """
    单标签验证集：
    - 无随机增强
    - 使用 UniformSampler
    """
    def __init__(self, config):
        super().__init__()
        self.subjects = get_subjects_single(config, split="val")

        patch_size = tuple(getattr(config, 'patch_size', (64, 64, 64)))
        pad = Pad(padding=patch_size, padding_mode=0)

        transforms = Compose([
            pad,
            ZNormalization(include=['source']),
        ])

        self.dataset = SubjectsDataset(self.subjects, transform=transforms)
        self.queue_dataset = Queue(
            self.dataset,
            max_length=int(getattr(config, "val_queue_length", 64)),
            samples_per_volume=int(getattr(config, "val_samples_per_volume", 64)),
            sampler=UniformSampler(patch_size=patch_size),
            num_workers=int(getattr(config, "val_queue_num_workers", 4)),
        )


# ---------- 测试集 ----------
class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.subjects = get_subjects_single(config, split="test")

        patch_size = tuple(getattr(config, 'patch_size', (64, 64, 64)))
        pad = Pad(padding=patch_size, padding_mode=0)

        transforms = Compose([
            pad,
            ZNormalization(include=['source']),
        ])

        self.dataset = SubjectsDataset(self.subjects, transform=transforms)
        self.queue_dataset = Queue(
            self.dataset,
            max_length=int(getattr(config, "test_queue_length", 64)),
            samples_per_volume=int(getattr(config, "test_samples_per_volume", 64)),
            sampler=UniformSampler(patch_size=patch_size),
            num_workers=int(getattr(config, "test_queue_num_workers", 4)),
        )


# ---------- 预测集 ----------
class DatasetPredict(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.subjects = get_subjects_single(config, split="predict")

        patch_size = tuple(getattr(config, 'patch_size', (64, 64, 64)))
        pad = Pad(padding=patch_size, padding_mode=0)

        transforms = Compose([
            pad,
            ZNormalization(include=['source']),
        ])

        self.dataset = SubjectsDataset(self.subjects, transform=transforms)
        self.queue_dataset = Queue(
            self.dataset,
            max_length=int(getattr(config, "pred_queue_length", 64)),
            samples_per_volume=int(getattr(config, "pred_samples_per_volume", 64)),

            sampler=UniformSampler(patch_size=patch_size),
            num_workers=int(getattr(config, "pred_queue_num_workers", 4))
        )
