import os
import argparse
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['NIBABEL_KEEP_FILE_OPEN'] = '0'  # 关闭 keep-file-open 池 ulimit 1048576   ulimit -n 500000
import time
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
from torchio import SubjectsLoader
from utils.EarlyStopping import EarlyStopping
from utils.loss_function import Dice_Loss, soft_dice_cldice, FocalLoss3D, FocalLoss3D_W, OverlapMarginLossPair
from utils.metric import metric, multiclass_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torchio.data import SubjectsLoader
# from logger import create_logger
from timm.utils import AverageMeter
import torch.fft as fft
from accelerate import Accelerator

# from utils import yaml_read
# from utils.conf_base import Default_Conf
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
import torch.nn as nn
import logging
from rich.logging import RichHandler
import hydra

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ! solve warning
import subprocess, shlex

def df_shm():
    if os.path.exists("/dev/shm"):
        try:
            out = subprocess.check_output(shlex.split("df -h /dev/shm"), text=True)
            print("[DF]\n" + out.strip())
        except Exception as e:
            print(f"[DF] failed: {e}")
    else:
        print("[DF] /dev/shm not found; skip.")

def weights_init_normal(init_type):
    def init_func(m):
        classname = m.__class__.__name__
        gain = 0.02

        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    return init_func


def get_logger(config):
    file_handler = logging.FileHandler(os.path.join(config.hydra_path, f"{config.job_name}.log"))
    rich_handler = RichHandler()

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False
    log.info("Successfully create rich logger")

    return log
def low_pass_torch(input,limit):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1]))<limit
    pass2 = torch.abs(fft.fftfreq(input.shape[-2]))<limit
    kernel = torch.outer(pass2, pass1).to(input)
    fft_input = fft.rfftn(input)
    return fft.irfftn(fft_input*kernel,s=input.shape[-3:])

def high_pass_torch(input,limit):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1]))>limit
    pass2 = torch.abs(fft.fftfreq(input.shape[-2]))>limit
    kernel = torch.outer(pass2, pass1).to(input)
    fft_input = fft.rfftn(input)
    return fft.irfftn(fft_input*kernel,s=input.shape[-3:])

def train(config, model, logger):
    import os, time
    import torch
    import torch.nn as nn
    from torch.utils.tensorboard import SummaryWriter
    from accelerate import Accelerator
    from rich.progress import Progress, TextColumn, MofNCompleteColumn, BarColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.panel import Panel
    from rich.console import Group
    import torchio as tio

    # 训练开始前做一次缓存（放在外面）
    enc_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and (n.startswith("encoder") or "enc" in n)]

    ema_ratio = None
    ema_cos = None
    UPDATE_EVERY = 50
    EPS = 1e-8

    # 兼容不同 TorchIO 版本的 GridSampler / GridAggregator 导入
    try:
        from torchio.inference import GridSampler, GridAggregator  # 新版
    except Exception:
        try:
            from torchio.data.inference import GridSampler, GridAggregator  # 旧版
        except Exception:
            GridSampler = getattr(tio, 'GridSampler')
            GridAggregator = getattr(tio, 'GridAggregator')

    DATA = tio.DATA
    LOCATION = tio.LOCATION

    # ================= 验证函数（整卷滑窗 + 进度条复用 + 只算Dice） =================
    def evaluate_epoch_full(accelerator, model, val_dataset, focal_criterion1, focal_criterion2, config):
        model.eval()

        # meters（只算 Dice）
        val_dice_1_meter = AverageMeter()
        val_dice_2_meter = AverageMeter()

        # ===== 验证参数 =====
        patch_size = tuple(getattr(config, 'patch_size', (64, 64, 64)))
        overlap_cfg = getattr(config, 'val_patch_overlap', 0.10)
        if isinstance(overlap_cfg, (int, float)):
            patch_overlap = tuple(max(1, int(p * float(overlap_cfg))) for p in patch_size)
        else:
            patch_overlap = tuple(overlap_cfg)
        infer_bs = int(getattr(config, 'val_infer_bs', 4))
        overlap_mode = getattr(config, 'val_overlap_mode', 'average')  # 'hann' 更平滑，'average' 更快

        # 进度条（用外层 progress，不调用 start）
        try:
            total_vols = len(getattr(val_dataset, 'subjects', []))
        except Exception:
            total_vols = 0
        val_vol_task = progress.add_task("[magenta]val volumes", total=total_vols)
        val_patch_task = None

        # 可选：一次性把验证集读入内存（若内存允许），减少 I/O
        for s in getattr(val_dataset, 'subjects', []):
            try:
                s.load()
            except Exception:
                pass

        # Dice（背景+前景取均值；若某类 GT 与 pred 都空，记为 1）
        def dice_for_label(gt_3d, pred_3d, label, eps=1e-6):
            gt_mask = (gt_3d == label).float()
            pd_mask = (pred_3d == label).float()
            inter = (gt_mask * pd_mask).sum()
            denom = gt_mask.sum() + pd_mask.sum()
            if denom == 0:
                return 1.0
            return float((2 * inter + eps) / (denom + eps))

        def dice_bg_fg_mean(gt_3d, pred_3d, labels=(0, 1)):
            return sum(dice_for_label(gt_3d, pred_3d, lb) for lb in labels) / len(labels)

        # DataLoader 参数
        val_workers = int(getattr(config, 'val_num_workers', 4))
        prefetch = int(getattr(config, 'val_prefetch_factor', 2))
        use_pin_memory = bool(getattr(config, 'val_pin_memory', True))

        with torch.inference_mode():  # 比 no_grad 更快
            for vi, subject in enumerate(val_dataset.subjects):
                sampler = GridSampler(subject, patch_size=patch_size, patch_overlap=patch_overlap)
                num_patches = len(sampler)

                # 首次建立任务行；后续 reset，不新增新行，避免累积
                if val_patch_task is None:
                    val_patch_task = progress.add_task(
                        f"[cyan]val patches (vol {vi + 1}/{total_vols})", total=num_patches
                    )
                else:
                    progress.reset(
                        val_patch_task, total=num_patches,
                        description=f"[cyan]val patches (vol {vi + 1}/{total_vols})"
                    )

                loader = torch.utils.data.DataLoader(
                    sampler,
                    batch_size=infer_bs,
                    num_workers=val_workers,
                    pin_memory=use_pin_memory,
                    prefetch_factor=prefetch if val_workers > 0 else None,
                    persistent_workers=False
                )

                agg1 = GridAggregator(sampler, overlap_mode=overlap_mode)
                agg2 = GridAggregator(sampler, overlap_mode=overlap_mode)

                for patches in loader:
                    x = patches['source'][DATA].to(accelerator.device, non_blocking=True).float()
                    locs = patches[LOCATION]
                    if not torch.is_tensor(locs):
                        locs = torch.as_tensor(locs)

                    # 半精度推理（AMP）
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pred1, pred2 = model(x)

                    # 减少 H2D 往返：先在 GPU 做 softmax，再搬 CPU 给 aggregator
                    prob1 = torch.softmax(pred1, dim=1).to(dtype=torch.float16).detach()
                    prob2 = torch.softmax(pred2, dim=1).to(dtype=torch.float16).detach()
                    agg1.add_batch(prob1.cpu(), locs)
                    agg2.add_batch(prob2.cpu(), locs)

                    progress.update(val_patch_task, advance=prob1.size(0))

                vol_prob1 = agg1.get_output_tensor()  # [C, D,H,W] on CPU
                vol_prob2 = agg2.get_output_tensor()
                vol_pred1 = vol_prob1.argmax(0).long()
                vol_pred2 = vol_prob2.argmax(0).long()

                gt1_full = subject['gt_1'][DATA].long().squeeze(0)
                gt2_full = subject['gt_2'][DATA].long().squeeze(0)

                val_dice_1_meter.update(dice_bg_fg_mean(gt1_full, vol_pred1), n=1)
                val_dice_2_meter.update(dice_bg_fg_mean(gt2_full, vol_pred2), n=1)

                progress.update(val_patch_task, completed=num_patches)
                progress.update(val_vol_task, advance=1)

        # 清理验证的 task，避免下一轮继续占位
        if val_patch_task is not None:
            progress.remove_task(val_patch_task)
        progress.remove_task(val_vol_task)

        model.train()
        return {
            "val_loss": 0.0,  # 只算 Dice
            "val_dice_1": val_dice_1_meter.avg,
            "val_dice_2": val_dice_2_meter.avg,
            "val_dice_mean": 0.5 * (val_dice_1_meter.avg + val_dice_2_meter.avg),
        }

    # ================= cuDNN =================
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    # * init averageMeter
    loss_meter = AverageMeter()
    dice_meter_1 = AverageMeter()
    dice_meter_2 = AverageMeter()

    # progress（训练期常驻；注意：不调用 progress.start()，交给 Live 管理）
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )

    # * set optimizer（保持 Adam，不改）
    optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr)

    # * set loss function（不改）
    class_weights = torch.tensor([0.99, 1.1]).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    dice_criterion = Dice_Loss().cuda()

    focal_criterion = FocalLoss3D_W(alpha=class_weights).cuda()
    anti_loss_fn = OverlapMarginLossPair(
        lam_overlap=0.5, lam_margin=0.5, margin=0.25,
        use_uncertainty_mask=True, delta=0.20,
        ignore_index=255, foreground_channel=1
    )
    focal_criterion1 = FocalLoss3D_W(gamma=3.0).cuda()
    focal_criterion2 = FocalLoss3D_W(gamma=3.0).cuda()
    cldice_dice_criterion1 = soft_dice_cldice()
    cldice_dice_criterion2 = soft_dice_cldice()

    # 动态多分支权重初始化
    lambda_a, lambda_v = 1.0, 1.0

    # 调度器占位
    scheduler = None
    ckpt_sched_state = None

    # * load model（支持断点续训，恢复 best_val 与 best_epoch）
    if config.load_mode == 1:
        logger.info(f"load model from: {os.path.join(config.ckpt)}")
        ckpt = torch.load(os.path.join(config.ckpt), map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        ckpt_sched_state = ckpt.get("scheduler", None)
        elapsed_epochs = ckpt["epoch"]
        best_val = ckpt.get("best_val", -float("inf"))
        best_epoch = ckpt.get("best_epoch", 0)  # <<< 新增：恢复历史最优出现的 epoch
    else:
        elapsed_epochs = 0
        best_val = -float("inf")
        best_epoch = 0

    model.train()

    # * tensorboard writer
    writer = SummaryWriter(config.hydra_path)

    # * load dataset
    from dataloader import Dataset, DatasetVal
    train_dataset = Dataset(config)
    train_loader = SubjectsLoader(
        dataset=train_dataset.queue_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False
    )
    val_dataset = DatasetVal(config)
    val_loader = SubjectsLoader(
        dataset=val_dataset.queue_dataset,
        batch_size=getattr(config, "val_batch_size", 2),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False
    )

    # 训练进度信息
    epochs = int(config.epochs) - int(elapsed_epochs)
    iteration = elapsed_epochs * len(train_loader)

    # ========= PolyLR 创建 =========
    if getattr(config, "use_scheduler", True):
        total_epochs = int(config.epochs)
        power = float(getattr(config, "poly_power", 0.9))
        min_lr = float(getattr(config, "poly_min_lr", 0.0))

        is_resume = (elapsed_epochs > 0)
        if is_resume:
            for g in optimizer.param_groups:
                if "initial_lr" not in g:
                    g["initial_lr"] = float(config.init_lr)
            last_epoch = int(iteration)
        else:
            last_epoch = -1
            try:
                scheduler = torch.optim.lr_scheduler.PolynomialLR(
                    optimizer, total_iters=total_epochs, power=power, last_epoch=last_epoch
                )
            except Exception:
                from torch.optim.lr_scheduler import _LRScheduler
                class _PolyLR(_LRScheduler):
                    def __init__(self, optimizer, total_iters, power=0.9, min_lr=0.0, last_epoch=-1):
                        self.total_iters = int(total_iters)
                        self.power = float(power)
                        self.min_lr = float(min_lr)
                        super().__init__(optimizer, last_epoch)
                    def get_lr(self):
                        t = min(self.last_epoch, self.total_iters)
                        factor = (1.0 - t / self.total_iters) ** self.power if self.total_iters > 0 else 0.0
                        return [self.min_lr + (base_lr - self.min_lr) * factor for base_lr in self.base_lrs]
                scheduler = _PolyLR(optimizer, total_iters=total_epochs, power=power, min_lr=min_lr, last_epoch=last_epoch)

        if ckpt_sched_state is not None and scheduler is not None:
            try:
                scheduler.load_state_dict(ckpt_sched_state)
                logger.info("Loaded scheduler state from checkpoint.")
            except Exception as e:
                logger.warning(f"Skip loading old scheduler state: {e}")

    # 进度条 & accelerate（注意：不调用 progress.start()）
    epoch_tqdm = progress.add_task("[red]epoch progress", total=epochs)
    batch_tqdm = progress.add_task("[blue]batch progress", total=len(train_loader))
    accelerator = Accelerator()

    # 统一交给 accelerate
    train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, val_loader, model, optimizer, scheduler
    )

    # ===== 早停控制参数 =====
    patience = int(getattr(config, "early_stop_patience", 0))
    min_delta = float(getattr(config, "early_stop_min_delta", 0.0))
    validate_every = int(getattr(config, "validate_every", 1))
    no_improve = 0

    # ====== Best 模型面板（常驻底部，显示历史最佳）======
    if best_val != -float("inf") and best_epoch > 0:
        best_text = f"[bold green]Best: dice_mean={best_val:.4f} @ epoch {best_epoch}[/]"
    else:
        best_text = "Best: —"
    best_panel = Panel(best_text, title="Best Model", border_style="green")

    # 用 Live 把进度条 + 面板固定在屏幕（底部常驻 Best）
    with Live(Group(progress, best_panel), refresh_per_second=4, transient=False):
        for epoch in range(1, epochs + 1):
            progress.update(epoch_tqdm, completed=epoch)
            epoch += elapsed_epochs

            num_iters = 0
            load_meter = AverageMeter()
            train_time = AverageMeter()
            load_start = time.time()
            ema_ratio = None

            for i, batch in enumerate(train_loader):
                if i % 50 == 0:
                    df_shm()
                with torch.autograd.set_detect_anomaly(True):
                    progress.update(batch_tqdm, completed=i + 1)
                    train_start = time.time()
                    load_time = time.time() - load_start
                    optimizer.zero_grad()

                    x = batch["source"]["data"]
                    gt_1 = batch["gt_1"]["data"].long()
                    gt_2 = batch["gt_2"]["data"].long()
                    w_1 = batch.get("weight_1", {}).get("data", None)
                    w_2 = batch.get("weight_2", {}).get("data", None)

                    x = x.type(torch.FloatTensor).to(accelerator.device)
                    gt_1 = gt_1.to(accelerator.device).squeeze(1)
                    gt_2 = gt_2.to(accelerator.device).squeeze(1)
                    w_1 = batch["weight_1"]["data"].float().squeeze(1).to(accelerator.device)
                    w_2 = batch["weight_2"]["data"].float().squeeze(1).to(accelerator.device)

                    # ==== 这里开始插入：监控 LabelSampler 前景采样是否≈33% ====
                    # fg = batch.get('fg', {}).get('data',
                    #                              None)  # [B, 1, Dz, Dy, Dx]（dataloader 里 Lambda(_make_union_fg) 生成）
                    # if fg is not None:
                    #     # 用中心±1邻域判断，避免偶数尺寸的中心对齐偏差
                    #     ps = tuple(getattr(config, 'patch_size', (64, 64, 64)))
                    #     cz, cy, cx = ps[0] // 2, ps[1] // 2, ps[2] // 2
                    #     r = 1
                    #     center_block = (fg[:, 0,
                    #                     max(0, cz - r):cz + r + 1,
                    #                     max(0, cy - r):cy + r + 1,
                    #                     max(0, cx - r):cx + r + 1] > 0)
                    #     center_is_pos = center_block.flatten(1).any(dim=1).float()  # [B], 1=该patch中心邻域有前景
                    #     center_pos_ratio = center_is_pos.mean().item()  # 本批次“前景中心”比例（期望≈config.pos_ratio）
                    #
                    #     # 可选：整个 patch 是否含前景（通常 ≥ center_pos_ratio）
                    #     patch_has_pos = (fg > 0).flatten(1).any(dim=1).float().mean().item()
                    #
                    #     if i % 50 == 0:  # 每50个batch打印一次
                    #         logger.info(f"[batch {i}] center_pos_ratio≈{center_pos_ratio:.2f} "
                    #                     f"(patch_has_pos≈{patch_has_pos:.2f}, target={getattr(config, 'pos_ratio', 0.33):.2f})")

                    # ======
                    pred_1, pred_2 = model(x)
                    mask_1 = pred_1.argmax(dim=1, keepdim=True)
                    mask_2 = pred_2.argmax(dim=1, keepdim=True)

                    loss_a = focal_criterion1(pred_1, gt_1,w_1)
                    loss_v = focal_criterion2(pred_2, gt_2,w_2)

                    if i % UPDATE_EVERY == 0:
                        # 仅对共享编码器参数求导，显著省时省显存
                        ga_list = torch.autograd.grad(
                            loss_a, enc_params, retain_graph=True, allow_unused=True, create_graph=False
                        )
                        gv_list = torch.autograd.grad(
                            loss_v, enc_params, retain_graph=True, allow_unused=True, create_graph=False
                        )

                        # 展平并拼接（只收集非 None）
                        with torch.no_grad():
                            ga_vec = [g.detach().reshape(-1) for g in ga_list if g is not None]
                            gv_vec = [g.detach().reshape(-1) for g in gv_list if g is not None]

                            if len(ga_vec) == 0 or len(gv_vec) == 0:
                                print(f"[step {i}] cos=N/A (no shared grads)")
                            else:
                                ga_v = torch.cat(ga_vec)
                                gv_v = torch.cat(gv_vec)

                                Ga = torch.linalg.vector_norm(ga_v).item()
                                Gv = torch.linalg.vector_norm(gv_v).item()

                                if Ga < EPS or Gv < EPS:
                                    print(f"[step {i}] cos=N/A (tiny grads) |ga|={Ga:.3e} |gv|={Gv:.3e}")
                                else:
                                    # 更稳的余弦
                                    cos = torch.nn.functional.cosine_similarity(
                                        ga_v.unsqueeze(0), gv_v.unsqueeze(0), dim=1, eps=EPS
                                    ).item()

                                    ratio = Ga / (Gv + EPS)

                                    # EMA 平滑
                                    ema_ratio = ratio if (ema_ratio is None) else (0.9 * ema_ratio + 0.1 * ratio)
                                    ema_cos = cos if (ema_cos is None) else (0.9 * ema_cos + 0.1 * cos)

                                    # —— 动态加权（保留你的形式，但更稳健）——
                                    # 用平滑后的 ratio 做平衡；alpha 调高会更“用力”对齐梯度范数
                                    alpha = 1.0
                                    bal = (ema_ratio ** alpha)
                                    lambda_a = 2.0 * (1.0 / (1.0 + bal))
                                    lambda_v = 2.0 - lambda_a

                                    # 冲突时（ema_cos<0）对大的那一侧再略收一点，缓和拉扯
                                    if ema_cos < 0:
                                        shrink = 0.10  # 10% 轻度抑制
                                        if ema_ratio >= 1.0:  # |ga|>|gv|
                                            lambda_a *= (1.0 - shrink)
                                        else:
                                            lambda_v *= (1.0 - shrink)

                                    # 合理裁剪
                                    lambda_a = float(min(max(lambda_a, 0.4), 1.6))
                                    lambda_v = 2-lambda_a

                                    print(f"[step {i}] cos={cos:+.3f}  |ga|/|gv|={ratio:.2f}  "
                                          f"ema_cos={ema_cos:+.3f}  ema_ratio={ema_ratio:.2f}  "
                                          f"lambda_a={lambda_a:.2f}, lambda_v={lambda_v:.2f}")
                    loss = lambda_a * loss_a + lambda_v * loss_v
                    accelerator.backward(loss)
                optimizer.step()

                num_iters += 1
                iteration += 1

                # metrics
                _, dice_1 = multiclass_metrics(gt_1.unsqueeze(1).cpu(), mask_1.cpu(), num_classes=config.out_classes)
                _, dice_2 = multiclass_metrics(gt_2.unsqueeze(1).cpu(), mask_2.cpu(), num_classes=config.out_classes)

                writer.add_scalar("Training/Loss", loss.item(), iteration)
                writer.add_scalar("Training/dice_1", dice_1, iteration)
                writer.add_scalar("Training/dice_2", dice_2, iteration)

                loss_meter.update(loss.item(), x.size(0))
                dice_meter_1.update(dice_1, x.size(0))
                dice_meter_2.update(dice_2, x.size(0))
                train_time.update(time.time() - train_start)
                load_meter.update(load_time)

                cur_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"\nEpoch: {epoch} Batch: {i}, data load time: {load_meter.val:.3f}s , train time: {train_time.val:.3f}s\n"
                    f"LR: {cur_lr:.6e}\n"
                    f"Loss: {loss_meter.val}\n"
                    f"Dice_1: {dice_meter_1.val}\n"
                    f"Dice_2: {dice_meter_2.val}\n"
                )
                load_start = time.time()

            # * one epoch logger（训练均值）
            logger.info(
                f"\nEpoch {epoch} used time:  {load_meter.sum+train_time.sum:.3f} s\n"
                f"Loss Avg:  {loss_meter.avg}\n"
                f"Dice_1 Avg:  {dice_meter_1.avg}\n"
                f"Dice_2 Avg:  {dice_meter_2.avg}\n"
            )

            # === Epoch-end scheduler step（按 epoch 衰减）===
            if getattr(config, "use_scheduler", True) and (scheduler is not None):
                scheduler.step()
                if 'min_lr' in locals() and min_lr > 0:
                    for g in optimizer.param_groups:
                        g['lr'] = max(g['lr'], min_lr)

            # 验证频率控制
            do_validate = (epoch % validate_every == 0) or \
                          (epoch == elapsed_epochs + 1) or (epoch == elapsed_epochs + epochs)

            # Save latest（每个 epoch 都存一次最新状态，含 best_epoch）
            scheduler_dict = scheduler.state_dict() if (getattr(config, "use_scheduler", True) and scheduler is not None) else None
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler_dict,
                    "epoch": epoch,
                    "best_val": best_val,
                    "best_epoch": best_epoch,  # <<< 新增
                },
                os.path.join(config.hydra_path, config.latest_checkpoint_file),
            )

            if do_validate:
                # ===== 验证（整卷）=====
                val_stats = evaluate_epoch_full(accelerator, model, val_dataset, focal_criterion1, focal_criterion2, config)

                writer.add_scalar("Val/Loss",      val_stats["val_loss"],      epoch)
                writer.add_scalar("Val/Dice_1",    val_stats["val_dice_1"],    epoch)
                writer.add_scalar("Val/Dice_2",    val_stats["val_dice_2"],    epoch)
                writer.add_scalar("Val/Dice_Mean", val_stats["val_dice_mean"], epoch)

                logger.info(
                    f"[Val] Epoch {epoch}\n"
                    f"  Val_Loss:      {val_stats['val_loss']:.6f}\n"
                    f"  Val_Dice_1:    {val_stats['val_dice_1']:.6f}\n"
                    f"  Val_Dice_2:    {val_stats['val_dice_2']:.6f}\n"
                    f"  Val_Dice_Mean: {val_stats['val_dice_mean']:.6f}\n"
                )

                # ===== best 保存 & 早停 =====
                cur_val = val_stats["val_dice_mean"]
                improved = (cur_val - best_val) > min_delta
                if improved:
                    best_val = cur_val
                    best_epoch = epoch  # <<< 新增

                    # 更新底部面板（历史最佳）
                    best_panel.renderable = f"[bold green]Best: dice_mean={best_val:.4f} @ epoch {best_epoch}[/]"

                    # 保存最佳（含 best_epoch）
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optim": optimizer.state_dict(),
                            "scheduler": scheduler_dict,
                            "epoch": epoch,
                            "best_val": best_val,
                            "best_epoch": best_epoch,  # <<< 新增
                        },
                        os.path.join(config.hydra_path, "checkpoint_best.pt"),
                    )
                    logger.info(f"✅ New best model saved at epoch {epoch} (Val_Dice_Mean={best_val:.6f})")
                    no_improve = 0
                else:
                    no_improve += 1

                # 分段保存（同样写入 best_epoch）
                if epoch % config.epochs_per_checkpoint == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optim": optimizer.state_dict(),
                            "scheduler": scheduler_dict,
                            "epoch": epoch,
                            "best_val": best_val,
                            "best_epoch": best_epoch,  # <<< 新增
                        },
                        os.path.join(config.hydra_path, f"checkpoint_{epoch:04d}.pt"),
                    )

                # 早停
                if patience > 0 and no_improve >= patience:
                    logger.info(f"⏹ Early stopping at epoch {epoch} "
                                f"(no improvement for {patience} validation windows; "
                                f"min_delta={min_delta}).")
                    return  # 直接退出 train

            # 每个 epoch 末重置训练期指标（不影响 TensorBoard）
            loss_meter.reset(); dice_meter_1.reset(); dice_meter_2.reset()





@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    if isinstance(config.patch_size, str):
        assert (
            len(config.patch_size.split(",")) <= 3
        ), f'patch size can only be one str or three str but got {len(config.patch_size.split(","))}'
        if len(config.patch_size.split(",")) == 3:
            config.patch_size = tuple(map(int, config.patch_size.split(",")))
        else:
            config.patch_size = int(config.patch_size)
    # os["CUDA_AVAILABLE_DEVICES"] = config.gpu

    # * model selection
    if config.network == "res_unet":
        from models.three_d.residual_unet3d import UNet

        model = UNet(in_channels=config.in_classes, n_classes=config.out_classes, base_n_filter=32)
    elif config.network == "ynet":
        from models.three_d.ynet3d import UNet3DDualDecoder # * 3d unet
        print("ynet")
        model = UNet3DDualDecoder(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == "ynet1":
        from models.three_d.ynet_tongdao import UNet3DDualDecoder # * 3d unet
        print("ynet1")
        model = UNet3DDualDecoder(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == "unet":
        from models.three_d.unet3d import UNet3D  # * 3d unet

        model = UNet3D(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == "er_net":
        from models.three_d.ER_net import ER_Net

        model = ER_Net(classes=config.out_classes, channels=config.in_classes)
    elif config.network == "re_net":
        from models.three_d.RE_net import RE_Net

        model = RE_Net()
    elif config.network == "IS":
        from models.three_d.IS import UNet3D

        model = UNet3D(in_channels=config.in_classes, out_channels=config.out_classes,init_features=32)

    elif config.network == "unetr":
        from models.three_d.unetr import UNETR

        model = UNETR()
    elif config.network == "densenet":
        from models.three_d.densenet3d import SkipDenseNet3D

        model = SkipDenseNet3D(in_channels=config.in_classes, classes=config.out_classes)

    elif config.network == "vtnet":
        from models.three_d.vtnet import VTUNet

        model = VTUNet(num_classes=config.out_classes, input_dim=config.in_classes)
    elif config.network == "vnet":
        from models.three_d.vnet3d import VNet

        model = VNet(in_channels=config.in_classes, classes=config.out_classes)
    elif config.network == "densevoxelnet":
        from models.three_d.densevoxelnet3d import DenseVoxelNet

        model = DenseVoxelNet(in_channels=config.in_classes, classes=config.out_classes)
    elif config.network == "csrnet":
        from models.three_d.csrnet import CSRNet

        model = CSRNet(in_channels=config.in_classes, out_channels=config.out_classes)
    elif config.network == "dunet":
        from models.three_d.Double_Unet import Double_Unet

        model = Double_Unet(in_channels=config.in_classes, out_channels=config.out_classes)
    model.apply(weights_init_normal(config.init_type))

    # * create logger
    logger = get_logger(config)
    info = "\nParameter Settings:\n"
    for k, v in config.items():
        info += f"{k}: {v}\n"
    logger.info(info)

    train(config, model, logger)
    logger.info(f"tensorboard file saved in:{config.hydra_path}")


if __name__ == "__main__":
    main()