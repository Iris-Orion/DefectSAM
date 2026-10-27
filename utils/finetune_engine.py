
import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytz
import monai
import numpy as np
import cv2
import os
import time
import swanlab
from tqdm import tqdm
from torch.optim import AdamW
from datetime import datetime
from transformers import SamModel
from peft import PeftModel, PeftConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from utils.helper_function import get_lr_scheduler, set_seed
from utils.utils import (compute_dice_score,
                        compute_iou_score,
                        save_model,
                        save_training_logs,
                        print_trainable_parameters)
from utils.mfu import SAMMFUEstimator, MFUTracker

from utils.loratask import prepare_sam_qkv_for_qv_peft


def _preview_keys(keys, limit: int = 8) -> str:
    keys = list(keys)
    suffix = " ..." if len(keys) > limit else ""
    return f"{keys[:limit]}{suffix}"


def load_lora_state_dict_for_model(
    model: torch.nn.Module,
    lora_state_dict: dict,
    strict: bool = False,
    validate_trainable: bool = True,
):
    """
    Load LoRA-only checkpoints into either normal or torch.compile-wrapped models.

    save_lora_parameters() strips "_orig_mod." before saving. When the current
    model still has a compiled vision_encoder, its state_dict expects
    "vision_encoder._orig_mod.*" keys, so adapt those keys before loading.
    """
    model_keys = set(model.state_dict().keys())
    trainable_keys = {name for name, param in model.named_parameters() if param.requires_grad}
    state_dict_to_load = {}

    for checkpoint_key, value in lora_state_dict.items():
        key = checkpoint_key.replace("_orig_mod.", "")
        if key not in model_keys and key.startswith("vision_encoder."):
            compiled_key = key.replace("vision_encoder.", "vision_encoder._orig_mod.", 1)
            if compiled_key in model_keys:
                key = compiled_key
        state_dict_to_load[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(state_dict_to_load, strict=False)

    # Newer transformers versions make shared_image_embedding trainable, but older
    # checkpoints never saved it. That is a version gap, not a LoRA mismatch, so only warn.
    compat_missing = sorted(
        key for key in missing_keys
        if key in trainable_keys and "shared_image_embedding" in key
    )
    if compat_missing:
        print(
            f"[WARN] load_lora_state_dict: checkpoint 缺少以下参数（transformers 版本差异，已跳过）: {compat_missing}"
        )
    missing_trainable = sorted(
        key for key in missing_keys
        if key in trainable_keys and key not in compat_missing
    )
    unexpected_trainable = sorted(
        key for key in unexpected_keys
        if "lora" in key or "shared_image_embedding" in key or "gate" in key
    )

    if validate_trainable and (missing_trainable or unexpected_trainable):
        message = ["LoRA checkpoint 加载未严格对齐。"]
        if missing_trainable:
            message.append(
                f"缺失 {len(missing_trainable)} 个可训练参数: {_preview_keys(missing_trainable)}"
            )
        if unexpected_trainable:
            message.append(
                f"checkpoint 中有 {len(unexpected_trainable)} 个模型无法匹配的 LoRA 相关参数: {_preview_keys(unexpected_trainable)}"
            )
        raise RuntimeError(" ".join(message))

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            "LoRA checkpoint strict=True 加载失败。"
            f"missing={_preview_keys(missing_keys)}, "
            f"unexpected={_preview_keys(unexpected_keys)}"
        )
    return missing_keys, unexpected_keys


def debug_print_optimizer_param_groups(optimizer: torch.optim.Optimizer) -> None:
    """Print lr/wd/size per param group to verify that LoRA+ grouping took effect."""
    print("\n========== Optimizer Param Groups ==========")
    total_params = 0
    for group_idx, group in enumerate(optimizer.param_groups):
        params = group.get("params", [])
        param_count = sum(param.numel() for param in params)
        tensor_count = len(params)
        group_lr = group.get("lr", None)
        group_wd = group.get("weight_decay", None)
        total_params += param_count
        print(
            f"[Group {group_idx}] lr={group_lr:.6e} | wd={group_wd} | "
            f"tensors={tensor_count} | params={param_count:,}"
        )
    print(f"Total params in optimizer groups: {total_params:,}")
    print("===========================================\n")

def prepare_base_model_for_hf_adapter_loading(base_model: SamModel, ft_type: str):
    if ft_type in ['dora_qv_encoder', 'lokr_qv_encoder', 'loha_qv_encoder']:
        return prepare_sam_qkv_for_qv_peft(base_model, target_part='vision_encoder')
    return base_model

def severstal_get_offset():
    """
    Return (y0, x0, y1, x1) of the letterboxed 256x1600 Severstal image inside the
    256x256 low-res mask space (image letterboxed into 1024x1024).
    """
    orig_h, orig_w = (256, 1600)
    target_h, target_w = (1024, 1024)
    pred_h, pred_w = (256, 256)

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    x_offset, y_offset = (target_w - new_w) // 2, (target_h - new_h) // 2

    crop_y_start = int(y_offset * (pred_h / target_h))
    crop_y_end = int((y_offset + new_h) * (pred_h / target_h))
    crop_x_start = int(x_offset * (pred_w / target_w))
    crop_x_end = int((x_offset + new_w) * (pred_w / target_w))
    return (crop_y_start, crop_x_start, crop_y_end, crop_x_end)


def _select_best_mask(pred_masks, iou_scores):
    """Pick the multimask output with the highest predicted IoU.
    Args:
        pred_masks: [B, 1, 3, H, W] (before squeezing point_batch) or [B, 3, H, W]
        iou_scores: [B, 1, 3] or [B, 3]
    Returns:
        selected: [B, 1, H, W]
    """
    if pred_masks.dim() == 5:
        pred_masks = pred_masks.squeeze(1)   # [B, 3, H, W]
    if iou_scores.dim() == 3:
        iou_scores = iou_scores.squeeze(1)   # [B, 3]
    best_idx = iou_scores.argmax(dim=1)      # [B]
    B = pred_masks.shape[0]
    selected = pred_masks[torch.arange(B, device=pred_masks.device), best_idx]  # [B, H, W]
    return selected.unsqueeze(1)  # [B, 1, H, W]


def _process_batch_severstal(batch, model, loss_fn, device, use_amp, auto_seg = False, offset_info = None, multimask=False):
    """
    Crop the letterboxed region (offset_info) from the 256x256 logits and upsample it to
    256x1600 before computing the loss against the original-size GT.
    """
    images = batch["image"].to(device)
    ground_truth_masks = batch["mask"].unsqueeze(1).float().to(device)      # [B, 1, 256, 1600]
    if auto_seg:
        bboxes = None    # TODO: auto_seg should use a grid point prompt
    else:
        bboxes = batch["bbox"].unsqueeze(1).to(device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        outputs = model(pixel_values=images, input_boxes=bboxes, multimask_output=multimask)
        if multimask:
            predicted_masks = _select_best_mask(outputs.pred_masks, outputs.iou_scores)  # [B, 1, 256, 256]
        else:
            predicted_masks = outputs.pred_masks.squeeze(1)  # [B, 1, 256, 256]

        orig_h, orig_w = (256, 1600)
        crop_y_start, crop_x_start, crop_y_end, crop_x_end = offset_info

        masks_cropped_small = predicted_masks[:, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        predicted_masks_256_1600 = F.interpolate(
            masks_cropped_small,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        ) # [B, 1, 256, 1600]

        loss = loss_fn(predicted_masks_256_1600, ground_truth_masks)
    return loss, predicted_masks_256_1600, ground_truth_masks


def _process_batch_severstal_resize_infer(batch, model, loss_fn, device, use_amp, auto_seg=False, offset_info=None, multimask=False):
    """
    Severstal resize-mode evaluation: images are resized straight to 1024x1024 without
    letterbox padding, so the 256x256 logits are interpolated directly back to 256x1600
    and scored against the original GT.
    """
    images = batch["image"].to(device)
    ground_truth_masks = batch["mask"].unsqueeze(1).float().to(device)
    if auto_seg:
        bboxes = None
    else:
        bboxes = batch["bbox"].unsqueeze(1).to(device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        outputs = model(pixel_values=images, input_boxes=bboxes, multimask_output=multimask)
        if multimask:
            predicted_masks = _select_best_mask(outputs.pred_masks, outputs.iou_scores)
        else:
            predicted_masks = outputs.pred_masks.squeeze(1)

        predicted_masks_256_1600 = F.interpolate(
            predicted_masks,
            size=(256, 1600),
            mode="bilinear",
            align_corners=False,
        )
        loss = loss_fn(predicted_masks_256_1600, ground_truth_masks)
    return loss, predicted_masks_256_1600, ground_truth_masks


def _process_batch(batch, model, loss_fn, device, use_amp, auto_seg = False, offset_info = None, multimask=False):
    """
    Batch step for SD900, magnetic tile and NEU (train and eval). The GT is downsampled
    to 256x256 and the loss is computed at the logit resolution.
    """
    images = batch["image"].to(device)
    ground_truth_masks = batch["mask"].unsqueeze(1).float().to(device)
    if auto_seg:
        bboxes = None    # TODO: auto_seg should use a grid point prompt
    else:
        bboxes = batch["bbox"].unsqueeze(1).to(device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        outputs = model(pixel_values=images, input_boxes=bboxes, multimask_output=multimask)
        if multimask:
            predicted_masks = _select_best_mask(outputs.pred_masks, outputs.iou_scores)  # [B, 1, 256, 256]
        else:
            predicted_masks = outputs.pred_masks.squeeze(1)  # [B, 1, 256, 256]

        gt_downsampled = F.interpolate(ground_truth_masks, size=(256, 256), mode="nearest")
        loss = loss_fn(predicted_masks, gt_downsampled)

    return loss, predicted_masks, gt_downsampled


def _sample_points_from_mask(gt_mask, num_points=1, sample_from='foreground'):
    """
    Sample point prompts from a binary GT mask, as in SAM's original training.
    Args:
        gt_mask: [H, W] (or [1, H, W]) numpy array or tensor
        sample_from: 'foreground' samples foreground points; any other value, or an
            empty foreground, samples background points instead
    Returns:
        coords: [num_points, 2] float32, (x, y) order, padded with (0, 0)
        labels: [num_points] int64, 1 = foreground, 0 = background
    """
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]

    fg_indices = np.argwhere(gt_mask > 0.5)  # [N, 2] -> (y, x)
    bg_indices = np.argwhere(gt_mask <= 0.5)

    coords = []
    labels = []

    if sample_from == 'foreground' and len(fg_indices) > 0:
        chosen = fg_indices[np.random.choice(len(fg_indices), size=min(num_points, len(fg_indices)), replace=False)]
        for yx in chosen:
            coords.append([yx[1], yx[0]])  # (x, y)
            labels.append(1)
    elif len(bg_indices) > 0:
        chosen = bg_indices[np.random.choice(len(bg_indices), size=min(num_points, len(bg_indices)), replace=False)]
        for yx in chosen:
            coords.append([yx[1], yx[0]])
            labels.append(0)

    while len(coords) < num_points:
        coords.append([0, 0])
        labels.append(0)

    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int64)

def _sample_correction_points(pred_mask, gt_mask, num_points=1):
    """
    Sample correction points from the larger error region of the previous prediction
    (pred_mask is logits): false negatives give foreground points (label 1), false
    positives give background points (label 0).
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = (torch.sigmoid(pred_mask) > 0.5).float().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]

    false_negative = (gt_mask > 0.5) & (pred_mask < 0.5)
    false_positive = (gt_mask < 0.5) & (pred_mask > 0.5)

    fn_indices = np.argwhere(false_negative)
    fp_indices = np.argwhere(false_positive)

    coords = []
    labels = []

    if len(fn_indices) >= len(fp_indices) and len(fn_indices) > 0:
        chosen = fn_indices[np.random.choice(len(fn_indices), size=min(num_points, len(fn_indices)), replace=False)]
        for yx in chosen:
            coords.append([yx[1], yx[0]])
            labels.append(1)
    elif len(fp_indices) > 0:
        chosen = fp_indices[np.random.choice(len(fp_indices), size=min(num_points, len(fp_indices)), replace=False)]
        for yx in chosen:
            coords.append([yx[1], yx[0]])
            labels.append(0)

    while len(coords) < num_points:
        coords.append([0, 0])
        labels.append(0)

    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int64)


def _process_batch_sam_style(batch, model, loss_fn, device, use_amp,
                             auto_seg=False, offset_info=None,
                             prompt_probs=(0.5, 0.3, 0.2),
                             num_iter_rounds=3,
                             num_points=1,
                             multimask=False):
    """
    SAM-style iterative training step.

      1. Round 0 draws one prompt type for the whole batch: box (bbox jitter is done in
         the dataset), foreground points sampled from the GT, or no prompt.
      2. Later rounds use only point prompts sampled from the error regions of the
         previous round's logits. The logits are NOT fed back as a mask prompt.
      3. Only the last round keeps gradients and returns the loss; earlier rounds run
         under no_grad to save memory.

    Args:
        prompt_probs: sampling probabilities of (box, point, no_prompt) for round 0
        num_iter_rounds: total number of rounds
        num_points: points sampled per image per round
    """
    images = batch["image"].to(device)
    ground_truth_masks = batch["mask"].unsqueeze(1).float().to(device)  # [B, 1, H, W]
    batch_size = images.shape[0]

    gt_downsampled = F.interpolate(ground_truth_masks, size=(256, 256), mode="nearest")  # [B, 1, 256, 256]

    prompt_type = np.random.choice(['box', 'point', 'no_prompt'], p=prompt_probs)

    prev_masks_logits = None
    is_last_round = (num_iter_rounds == 1)

    for round_idx in range(num_iter_rounds):
        is_last_round = (round_idx == num_iter_rounds - 1)
        input_boxes = None
        input_points = None
        input_labels = None

        if round_idx == 0:
            if prompt_type == 'box':
                input_boxes = batch["bbox"].unsqueeze(1).to(device)  # [B, 1, 4]
            elif prompt_type == 'point':
                all_coords = []
                all_labels = []
                for b in range(batch_size):
                    gt_1024 = ground_truth_masks[b, 0].cpu().numpy()
                    coords, labs = _sample_points_from_mask(gt_1024, num_points=num_points, sample_from='foreground')
                    all_coords.append(coords)
                    all_labels.append(labs)
                input_points = torch.tensor(np.stack(all_coords), dtype=torch.float32, device=device).unsqueeze(1)
                input_labels = torch.tensor(np.stack(all_labels), dtype=torch.long, device=device).unsqueeze(1)
        else:
            all_coords = []
            all_labels = []
            for b in range(batch_size):
                coords, labs = _sample_correction_points(
                    prev_masks_logits[b], gt_downsampled[b],
                    num_points=num_points
                )
                coords = coords * 4.0  # 256 -> 1024
                all_coords.append(coords)
                all_labels.append(labs)
            input_points = torch.tensor(np.stack(all_coords), dtype=torch.float32, device=device).unsqueeze(1)
            input_labels = torch.tensor(np.stack(all_labels), dtype=torch.long, device=device).unsqueeze(1)

        if not is_last_round:
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(
                        pixel_values=images,
                        input_boxes=input_boxes,
                        input_points=input_points,
                        input_labels=input_labels,
                        multimask_output=False,
                    )
                    prev_masks_logits = outputs.pred_masks.squeeze(1).detach()  # [B, 1, 256, 256]
        else:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(
                    pixel_values=images,
                    input_boxes=input_boxes,
                    input_points=input_points,
                    input_labels=input_labels,
                    multimask_output=multimask,
                )
                if multimask:
                    predicted_masks = _select_best_mask(outputs.pred_masks, outputs.iou_scores)
                else:
                    predicted_masks = outputs.pred_masks.squeeze(1)  # [B, 1, 256, 256]
                loss = loss_fn(predicted_masks, gt_downsampled)

    return loss, predicted_masks, gt_downsampled

class CUDAPrefetcher:
    """
    nanoGPT-style one-batch-ahead prefetch. Relies on DataLoader(pin_memory=True) plus
    non_blocking=True so CUDA overlaps the H2D copy with compute; no manual streams.
    """
    def __init__(self, dataloader, device: torch.device):
        self.loader = dataloader
        self.device = device

    def __len__(self):
        return len(self.loader)

    def _to_device(self, batch):
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def __iter__(self):
        it = iter(self.loader)
        try:
            nxt = self._to_device(next(it))
        except StopIteration:
            return
        for raw in it:
            batch = nxt
            nxt = self._to_device(raw)
            yield batch
        yield nxt


def _train_one_epoch(model,
                    dataloader,
                    optimizer,
                    scheduler,
                    loss_fn,
                    process_batch_fn,
                    scaler, device,
                    auto_seg=False,
                    offset_info=None,
                    mfu_tracker: MFUTracker = None,
                    master_process: bool = True,
                    multimask: bool = False,
                    global_step: int = 0,
                    compute_hd95: bool = False,
                    grad_clip: float = 1.0):
    """
    Args:
        mfu_tracker: optional; when given, MFU is sampled every MFU_SAMPLE_INTERVAL steps
            and shown in the tqdm postfix.
        master_process: only the DDP master shows the progress bar.
        global_step: global step count passed to AdaLoRA's update_and_allocate.
    """
    model.train()
    # Accumulate on GPU and call .item() once per epoch; a per-step .item() forces a
    # CPU-GPU sync and stalls the pipeline.
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    total_dice = torch.zeros((), dtype=torch.float32, device=device)
    total_iou = torch.zeros((), dtype=torch.float32, device=device)
    # HD95 is off during training by default (OOM risk); enable with --train_hd95.
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0) if compute_hd95 else None
    use_amp = scaler is not None
    # MFU timing needs a torch.cuda.synchronize() before and after the step, which would
    # stall the pipeline every step, so only sample every MFU_SAMPLE_INTERVAL steps.
    MFU_SAMPLE_INTERVAL = 50

    pbar = tqdm(CUDAPrefetcher(dataloader, device), desc="Training", total=len(dataloader), disable=not master_process)
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        mfu_sample_this_step = mfu_tracker is not None and batch_idx % MFU_SAMPLE_INTERVAL == 0
        if mfu_sample_this_step:
            torch.cuda.synchronize()
            t_step_start = time.time()

        loss, pred_masks, gt = process_batch_fn(batch,
                                                model,
                                                loss_fn,
                                                device,
                                                use_amp,
                                                auto_seg=auto_seg,
                                                offset_info=offset_info,
                                                multimask=multimask)

        _raw = getattr(model, 'module', model)  # DDP unwrap

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
                )
            optimizer.step()

        # AdaLoRA rank allocation must run after optimizer.step() and before scheduler.step().
        if hasattr(_raw, 'update_and_allocate'):
            _raw.update_and_allocate(global_step)
        global_step += 1

        if mfu_sample_this_step:
            torch.cuda.synchronize()
            mfu_tracker.update(time.time() - t_step_start)
            pbar.set_postfix_str(mfu_tracker.status())

        scheduler.step()  # stepped per batch, not per epoch

        total_loss += loss.detach()
        with torch.no_grad():
            total_dice += compute_dice_score(pred_masks, gt, as_tensor=True)
            total_iou += compute_iou_score(pred_masks, gt, as_tensor=True)

            if hd95_metric is not None:
                hd95_metric(y_pred=(torch.sigmoid(pred_masks) > 0.5).float().detach().cpu(), y=gt.detach().cpu())

    avg_loss = total_loss.item() / len(dataloader)
    avg_dice = total_dice.item() / len(dataloader)
    avg_iou = total_iou.item() / len(dataloader)
    if hd95_metric is not None:
        avg_hd95 = hd95_metric.aggregate().item()
        hd95_metric.reset()
    else:
        avg_hd95 = float('nan')

    avg_mfu = mfu_tracker.mfu if mfu_tracker is not None else -1.0
    return avg_loss, avg_dice, avg_iou, avg_hd95, avg_mfu, global_step

def _evaluate( model,
            dataloader,
            loss_fn,
            process_batch_fn,
            device, scaler,
            auto_seg = False,
            offset_info = None,
            master_process: bool = True,
            multimask: bool = False,
            ddp: bool = False):
    """
    Args:
        master_process: only the DDP master shows the progress bar.
        ddp: all_reduce loss/dice/iou/hd95 across ranks into a global mean. Requires
            DistributedSampler(shuffle=False) so every rank sees the same batch count.
    """
    model.eval()
    total_loss = torch.zeros(1, dtype=torch.float32, device=device)
    total_dice = torch.zeros(1, dtype=torch.float32, device=device)
    total_iou = torch.zeros(1, dtype=torch.float32, device=device)
    num_batches = 0
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0)
    use_amp = scaler is not None

    with torch.no_grad():
        for batch in tqdm(CUDAPrefetcher(dataloader, device), desc="Evaluating", total=len(dataloader), disable=not master_process):
            loss, pred_masks, gt = process_batch_fn(batch,
                                                    model,
                                                    loss_fn,
                                                    device,
                                                    use_amp,
                                                    auto_seg = auto_seg,
                                                    offset_info = offset_info,
                                                    multimask = multimask)

            total_loss += loss.detach().float()
            total_dice += float(compute_dice_score(pred_masks, gt))
            total_iou += float(compute_iou_score(pred_masks, gt))
            num_batches += 1
            hd95_metric(y_pred=(torch.sigmoid(pred_masks) > 0.5).float().detach().cpu(), y=gt.detach().cpu())

    local_hd95 = hd95_metric.aggregate().item()
    hd95_metric.reset()

    stats = torch.tensor(
        [
            total_loss.item(),
            total_dice.item(),
            total_iou.item(),
            float(num_batches),
            local_hd95 * num_batches,
        ],
        dtype=torch.float64,
        device=device,
    )

    if ddp and dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    loss_sum, dice_sum, iou_sum, n_batches_global, hd95_weighted_sum = stats.tolist()

    if n_batches_global <= 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_loss = loss_sum / n_batches_global
    avg_dice = dice_sum / n_batches_global
    avg_iou = iou_sum / n_batches_global
    avg_hd95 = hd95_weighted_sum / n_batches_global

    return avg_loss, avg_dice, avg_iou, avg_hd95

def run_finetune_engine(train_dataloader,
                        val_dataloader,
                        test_dataloader,
                        model,
                        device,
                        hyperparameters,
                        process_batch_fn,
                        loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean'),
                        save_dir = "./new_weights/unspec_output", auto_seg=False,
                        ddp_info=None,
                        train_sampler=None):
    """
    Fine-tuning loop shared by all datasets, single GPU or DDP. DDP is detected from the
    RANK environment variable, not from ddp_info.
    Args:
        process_batch_fn: one of the `_process_batch*` functions. Letterbox offsets are
            computed only for `_process_batch_severstal`.
        ddp_info (dict, optional): only its 'world_size' is read, for the MFU batch size.
        train_sampler (DistributedSampler, optional): set_epoch is called on it every epoch.
    """
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        print(f"using: world_size={ddp_world_size}, rank={ddp_rank}, local_rank={ddp_local_rank}")
    else:
        print("running on a single gpu, and one process")
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    set_seed(hyperparameters['seed'], seed_offset=seed_offset)

    shanghai_tz = pytz.timezone('Asia/Shanghai')
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")

    save_hf_format = hyperparameters['save_hf_format']
    save_lora_only = hyperparameters['save_custom_lora']
    use_swanlab = hyperparameters.get('use_swanlab', False) and master_process
    log_name = hyperparameters["ft_type"] + "_rank_" + str(hyperparameters["lora_rank"]) if (
        save_lora_only or save_hf_format
    ) else hyperparameters["ft_type"]

    lr = hyperparameters['learning_rate']
    wd = hyperparameters['weight_decay']
    num_epochs = hyperparameters['num_epochs']
    warmup_ratio = hyperparameters['warmup_ratio']
    patience = hyperparameters['patience']
    min_delta = hyperparameters['min_delta']
    use_loraplus_optim = hyperparameters.get('use_loraplus_optim', False)
    lora_plus_lr_ratio = hyperparameters.get('lora_plus_lr_ratio', 16.0)
    should_use_early_stop = not hyperparameters.get('disable_early_stop', False)
    grad_clip = hyperparameters.get('grad_clip', 1.0)
    use_multimask = hyperparameters.get('multimask', False)
    train_compute_hd95 = hyperparameters.get('train_hd95', False)

    model.to(device)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]

    optimizer = None
    loraplus_status_msg = None
    if use_loraplus_optim:
        param_groups = []
        for module in model.modules():
            if hasattr(module, 'get_loraplus_param_groups') and callable(module.get_loraplus_param_groups):
                param_groups.extend(
                    module.get_loraplus_param_groups(
                        base_lr=lr,
                        lora_plus_lr_ratio=lora_plus_lr_ratio,
                        weight_decay=wd,
                    )
                )
        if param_groups:
            optimizer = AdamW(param_groups)
            loraplus_status_msg = f"Using LoRA+ optimizer param groups (ratio={lora_plus_lr_ratio})."
        else:
            loraplus_status_msg = "LoRA+ optimizer requested but no compatible modules found; fallback to standard AdamW."

    if optimizer is None:
        optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=wd)

    _use_bf16 = torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda', enabled=not _use_bf16)

    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    if hyperparameters.get('no_scheduler', False):
        cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        cosine_scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    sam_type = hyperparameters.get('sam_type', 'sam_base')
    mfu_estimator = SAMMFUEstimator(sam_type=sam_type, gpu_type='rtx5090')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    _ft_type = hyperparameters.get('ft_type', '')
    _is_adalora = _ft_type == 'adalora_encoder'
    needs_find_unused = _ft_type == 'sam_fully'
    should_use_compile = not hyperparameters.get('no_compile', False) and not _is_adalora
    compile_status_msg = None

    if ddp:
        model = DDP(
            model,
            device_ids=[ddp_local_rank],
            find_unused_parameters=needs_find_unused,
        )

    raw_model = model.module if ddp else model
    _compile_is_peft = save_hf_format and hasattr(raw_model, 'base_model')
    _compiled_ve = None
    _orig_ve = None
    if should_use_compile:
        try:
            if _compile_is_peft:
                _peft_inner = raw_model.base_model.model
                _orig_ve = _peft_inner.vision_encoder
                _compiled_ve = torch.compile(_orig_ve, mode='default')
                _peft_inner.vision_encoder = _compiled_ve
            else:
                _orig_ve = raw_model.vision_encoder
                _compiled_ve = torch.compile(_orig_ve, mode='default')
                raw_model.vision_encoder = _compiled_ve
            compile_status_msg = f"torch.compile 已启用 ({'DDP + ' if ddp else ''}仅 vision_encoder, mode='default')"
        except Exception as e:
            compile_status_msg = f"torch.compile 不可用，跳过: {e}"
            should_use_compile = False
    else:
        compile_status_msg = "torch.compile 已禁用 (AdaLoRA 与静态图不兼容)" if _is_adalora else "torch.compile 已禁用 (--no_compile)"

    model.train()

    swanlab_run = None
    if use_swanlab:
        swanlab_run = swanlab.init(
            project=hyperparameters.get('swanlab_project', 'please name your swanlab project'),
            experiment_name=f"{hyperparameters.get('task_name', 'please name your experiment name')}_{start_timestamp}",
            config=hyperparameters,
        )

    offset_info = None
    if process_batch_fn == _process_batch_severstal:
        offset_info = severstal_get_offset()

    if master_process:
        print_trainable_parameters(raw_model)
        print(f"using device {device}")
        for name, param in raw_model.named_parameters():
            if 'lora' in name and param.requires_grad:
                print(name, param.shape)

        print(f"权重保存格式: hugging face格式:{save_hf_format} || lora格式:{save_lora_only}")
        if loraplus_status_msg is not None:
            print(loraplus_status_msg)
        debug_print_optimizer_param_groups(optimizer)
        print(mfu_estimator.summary())
        if should_use_early_stop:
            print(f"早停已启用 (patience={patience}, min_delta={min_delta}).")
        else:
            print("早停已关闭：将运行完整训练轮次，同时仍跟踪最佳指标以保存模型。")
        print(compile_status_msg)
        print(f"梯度裁剪: {'max_norm=' + str(grad_clip) if grad_clip > 0 else '已禁用 (grad_clip=0)'}")
        if use_multimask:
            print("multimask_output=True + best IoU selection 已启用")
        print(f"训练阶段 HD95 计算: {'启用' if train_compute_hd95 else '关闭 (默认)'}")
        if offset_info is not None:
            print("Severstal-specific processing enabled. Calculating offset info.")

    history = {
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss": [], "val_dice": [], "val_iou": [], "val_hd95": [],
        "train_mfu": [],
    }
    best_val_dicescore = 0.0
    no_improve_epochs = 0
    best_epoch = -1
    best_model_path = None
    global_step = 0
    last_completed_epoch = 0

    for epoch in range(num_epochs):
        if master_process:
            print(f"--- Epoch {epoch + 1}/{num_epochs} ---")

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        single_batch_size = hyperparameters.get('batch_size', 4)
        world_size = ddp_info['world_size'] if ddp and ddp_info is not None else ddp_world_size
        global_batch_size = single_batch_size * world_size
        mfu_tracker = MFUTracker(mfu_estimator, batch_size=global_batch_size, ema_alpha=0.9)

        train_loss, train_dice, train_iou, train_hd95, train_mfu, global_step = _train_one_epoch(
            model, train_dataloader, optimizer,
            cosine_scheduler, loss_fn, process_batch_fn, scaler, device,
            auto_seg=auto_seg, offset_info=offset_info, mfu_tracker=mfu_tracker,
            master_process=master_process, multimask=use_multimask,
            global_step=global_step, compute_hd95=train_compute_hd95,
            grad_clip=grad_clip)

        val_loss, val_dice, val_iou, val_hd95 = _evaluate(
            model, val_dataloader, loss_fn, process_batch_fn,
            device, scaler, auto_seg=auto_seg, offset_info=offset_info,
            master_process=master_process, multimask=use_multimask,
            ddp=ddp)

        last_completed_epoch = epoch + 1
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["train_mfu"].append(round(train_mfu * 100, 4) if train_mfu >= 0 else None)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_hd95"].append(val_hd95)

        if master_process:
            mfu_str = f"{train_mfu*100:.2f}%" if train_mfu >= 0 else "N/A"
            print(f"Training loss: {train_loss:.4f}, train dice: {train_dice:.4f}, "
                  f"train iou: {train_iou:.4f}, MFU: {mfu_str}")
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"Current learning rate (param_group {i}): {param_group['lr']:.6e}")
            print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val HD95: {val_hd95:.4f}')

        if swanlab_run:
            log_dict = {
                "train/loss": train_loss,
                "train/dice": train_dice,
                "train/iou": train_iou,
                "val/loss": val_loss,
                "val/dice": val_dice,
                "val/iou": val_iou,
                "val/hd95": val_hd95,
                "learning_rate": optimizer.param_groups[0]['lr'],
            }
            if train_mfu >= 0:
                log_dict["train/mfu_pct"] = train_mfu * 100
            swanlab.log(log_dict, step=epoch + 1)

        improved = val_dice - best_val_dicescore > min_delta
        if improved:
            best_val_dicescore = val_dice
            no_improve_epochs = 0
            best_epoch = epoch + 1
            history['best_epoch'] = best_epoch
            history['best_metrics'] = {
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "val_hd95": val_hd95,
            }
            if master_process:
                print(f"验证 dice 改善到 {best_val_dicescore:.4f}, 保存模型...")
                if _compiled_ve is not None:
                    if _compile_is_peft:
                        _peft_inner.vision_encoder = _orig_ve
                    else:
                        raw_model.vision_encoder = _orig_ve
                best_model_path = save_model(
                    hyperparameters=hyperparameters,
                    start_timestamp=start_timestamp,
                    model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch + 1,
                    model_name=log_name,
                    target_dir=save_dir,
                    SAVE_HUGGINGFACE_PRETRAINED_MODEL=save_hf_format,
                    save_lora_only=save_lora_only,
                )
                if _compiled_ve is not None:
                    if _compile_is_peft:
                        _peft_inner.vision_encoder = _compiled_ve
                    else:
                        raw_model.vision_encoder = _compiled_ve
            if swanlab_run:
                swanlab.log({
                    "best_epoch": best_epoch,
                    "best/train_loss": train_loss,
                    "best/train_dice": train_dice,
                    "best/train_iou": train_iou,
                    "best/val_loss": val_loss,
                    "best/val_dice": val_dice,
                    "best/val_iou": val_iou,
                    "best/val_hd95": val_hd95,
                })
        else:
            if should_use_early_stop:
                no_improve_epochs += 1
                if master_process:
                    print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
            elif master_process:
                print("验证 dice 未改善（未启用早停），继续完整训练。")

        if master_process:
            save_training_logs(
                hyperparameters=hyperparameters,
                results=history,
                epoch=last_completed_epoch,
                start_timestamp=start_timestamp,
                result_name=log_name,
                target_dir=save_dir,
            )

        if should_use_early_stop and no_improve_epochs >= patience:
            if master_process:
                print(f"验证指标连续 {patience} 个 epoch 无改善，提前停止训练。")
            break

    if master_process:
        print("\n--- Training finished. Starting final evaluation with the best model. ---")

    if ddp and dist.is_available() and dist.is_initialized():
        dist.barrier()
        best_model_path_holder = [best_model_path if master_process else None]
        dist.broadcast_object_list(best_model_path_holder, src=0)
        best_model_path = best_model_path_holder[0]

    # Restore the uncompiled vision_encoder before loading so state_dict keys match.
    if _compiled_ve is not None:
        if _compile_is_peft:
            _peft_inner.vision_encoder = _orig_ve
        else:
            raw_model.vision_encoder = _orig_ve
        if master_process:
            print("加载权重前已恢复原始 vision_encoder（解除 torch.compile）。")

    if not best_model_path:
        if master_process:
            print("Warning: No best model was saved. Final evaluation will be on the last state of the model.")
        loaded_model = raw_model
    else:
        if master_process:
            print(f"Loading best model from: {best_model_path}")
        if save_hf_format:
            try:
                config = PeftConfig.from_pretrained(best_model_path)
                base_model_path = config.base_model_name_or_path
                fresh_base = SamModel.from_pretrained(base_model_path)
                fresh_base = prepare_base_model_for_hf_adapter_loading(fresh_base, _ft_type)
                loaded_model = PeftModel.from_pretrained(fresh_base, best_model_path)
                loaded_model.to(device)
            except Exception as e:
                raise RuntimeError(f"Failed to load Hugging Face PEFT model from {best_model_path}") from e
        elif save_lora_only:
            lora_state_dict = torch.load(best_model_path, map_location=device)
            load_lora_state_dict_for_model(raw_model, lora_state_dict, strict=False)
            loaded_model = raw_model
        else:
            checkpoint = torch.load(best_model_path, map_location=device)
            try:
                raw_model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to load full checkpoint from {best_model_path}. "
                    f"If the checkpoint was saved without torch.compile but the current model has a compiled vision_encoder, "
                    f"this is likely due to state_dict key mismatch (missing vision_encoder._orig_mod. prefix). "
                    f"Original error: {e}"
                ) from e
            loaded_model = raw_model

    loaded_model.eval()
    final_test_loss, final_test_dice, final_test_iou, final_test_hd95 = _evaluate(
        loaded_model, test_dataloader, loss_fn, process_batch_fn, device, scaler,
        auto_seg=auto_seg, offset_info=offset_info, master_process=master_process,
        multimask=use_multimask, ddp=ddp)

    if master_process:
        print(f'Final Test Set Evaluation: Loss: {final_test_loss:.4f}, Dice: {final_test_dice:.4f}, IoU: {final_test_iou:.4f}, HD95: {final_test_hd95:.4f}')
        history["final_test_metrics"] = {
            "loss": final_test_loss,
            "dice": final_test_dice,
            "iou": final_test_iou,
            "hd95": final_test_hd95,
        }

        if swanlab_run:
            swanlab.log({
                "test/test_dice": final_test_dice,
                "test/test_iou": final_test_iou,
                "test/test_hd95": final_test_hd95,
            })
            swanlab.finish()

        save_training_logs(
            hyperparameters=hyperparameters,
            results=history,
            epoch=last_completed_epoch,
            start_timestamp=start_timestamp,
            result_name=log_name,
            target_dir=save_dir,
        )
        print("--- Training finished ---")

    return history, model

def evaluate_all_metrics(model, dataloader, loss_fn, process_batch_fn, device, auto_seg=False, offset_info=None):
    """
    Final evaluation: mean IoU, Dice and HD95. HD95 only aggregates batches where both
    the GT and the prediction are non-empty.
    """
    model.eval()
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0)

    total_dice, total_iou = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Final Evaluation"):
            _, pred_masks_logits, gt_masks = process_batch_fn(
                batch, model, loss_fn=loss_fn, device=device, use_amp=False, auto_seg=auto_seg, offset_info=offset_info)

            total_dice += compute_dice_score(pred_masks_logits, gt_masks)
            total_iou += compute_iou_score(pred_masks_logits, gt_masks)

            pred_masks_binary = (torch.sigmoid(pred_masks_logits) > 0.5).float()

            if gt_masks.sum() > 0 and pred_masks_binary.sum() > 0:
                hd95_metric(y_pred=pred_masks_binary.detach().cpu(), y=gt_masks.detach().cpu())


    mean_iou = total_iou / len(dataloader)
    mean_dice = total_dice / len(dataloader)
    mean_hd95 = hd95_metric.aggregate().item()

    hd95_metric.reset()
    
    print(f"Evaluation Results:")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"  95% Hausdorff Distance (HD95): {mean_hd95:.4f}")

    return {
        "iou": mean_iou,
        "dice": mean_dice,
        "hd95": mean_hd95,
    }


def inference_engine(model, args, best_model_path, 
                     train_dataloader, val_dataloader, test_dataloader, 
                     process_batch_fn, loss_fn, scaler, device, 
                     results_filename="evaluation_results.txt",
                     auto_seg: bool = False, eval_traindataset = False,
                     eval_valdataset: bool = True):
    SAVE_HUGGINGFACE_PRETRAINED_MODEL = args.save_hf_format
    save_lora_only = args.save_custom_lora
    zero_shot_infer = args.zero_shot
    if SAVE_HUGGINGFACE_PRETRAINED_MODEL:
            try:
                config = PeftConfig.from_pretrained(best_model_path)
                base_model_path = config.base_model_name_or_path
                base_model = SamModel.from_pretrained(base_model_path)
                base_model = prepare_base_model_for_hf_adapter_loading(base_model, args.ft_type).to(device)
                
                loaded_model = PeftModel.from_pretrained(base_model, best_model_path)
                print("Successfully loaded model in Hugging Face PEFT format.")
            except Exception as e:
                raise RuntimeError(f"Failed to load Hugging Face PEFT model from {best_model_path}") from e
    elif save_lora_only:
        lora_state_dict = torch.load(best_model_path, map_location=device)
        load_lora_state_dict_for_model(model, lora_state_dict, strict=False)
        loaded_model = model.to(device)
        loaded_model.eval()

    else:
        checkpoint = torch.load(best_model_path, map_location=device)
        _compiled_ve_infer = None
        if hasattr(model.vision_encoder, '_orig_mod'):
            _compiled_ve_infer = model.vision_encoder
            model.vision_encoder = model.vision_encoder._orig_mod
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            if _compiled_ve_infer is not None:
                model.vision_encoder = _compiled_ve_infer
            raise RuntimeError(
                f"Failed to load full checkpoint from {best_model_path}. "
                f"If the checkpoint was saved without torch.compile but the current model has a compiled vision_encoder, "
                f"this is likely due to state_dict key mismatch (missing vision_encoder._orig_mod. prefix). "
                f"Original error: {e}"
            ) from e
        loaded_model = model
        loaded_model.to(device)
        print("Successfully loaded model from standard PyTorch checkpoint.")
    
    offset_info = None
    if process_batch_fn == _process_batch_severstal:
        print("Severstal-specific processing enabled. Calculating offset info.")
        offset_info = severstal_get_offset()

    print("\n--- Evaluating on Test Set ---")
    test_results = evaluate_all_metrics(loaded_model, test_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    val_results = None
    if eval_valdataset:
        print("\n--- Evaluating on Validation Set ---")
        val_results = evaluate_all_metrics(loaded_model, val_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    train_results = None
    if eval_traindataset:
        print("\n--- Evaluating on Training Set (optional) ---")
        train_results = evaluate_all_metrics(loaded_model, train_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    print("\n--- Summary of Results ---")
    if eval_traindataset:
        print(f"Training Set:   Dice={train_results['dice']:.4f}, IoU={train_results['iou']:.4f}, HD95={train_results['hd95']:.4f}")
    if eval_valdataset:
        print(f"Validation Set: Dice={val_results['dice']:.4f},  IoU={val_results['iou']:.4f}, HD95={val_results['hd95']:.4f}")
    print(f"Test Set:     Dice={test_results['dice']:.4f}, IoU={test_results['iou']:.4f}, HD95={test_results['hd95']:.4f}")

    print(f"\n--- Saving results to {results_filename} ---")
    try:
        with open(results_filename, "a", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint Path: {best_model_path}\n")
            f.write("\n--- Evaluation Metrics ---\n")
            if train_results:
                f.write(f"Training Set:   Dice={train_results['dice']:.4f}, IoU={train_results['iou']:.4f}, HD95={train_results['hd95']:.4f}\n")
            if val_results:
                f.write(f"Validation Set: Dice={val_results['dice']:.4f},  IoU={val_results['iou']:.4f}, HD95={val_results['hd95']:.4f}\n")
            f.write(f"Test Set:     Dice={test_results['dice']:.4f}, IoU={test_results['iou']:.4f}, HD95={test_results['hd95']:.4f}\n")
            f.write("="*50 + "\n\n")
        print("Results successfully saved.")
    except Exception as e:
        print(f"Error saving results to file: {e}")
    
    del loaded_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def zero_shot(model_path, 
                     train_dataloader, val_dataloader, test_dataloader, 
                     process_batch_fn, loss_fn, device, 
                     results_filename="evaluation_results.txt",
                     auto_seg: bool = False, eval_traindataset = False,
                     eval_valdataset: bool = True):
    loaded_model = SamModel.from_pretrained(model_path).to(device)

    offset_info = None
    if process_batch_fn == _process_batch_severstal:
        print("Severstal-specific processing enabled. Calculating offset info.")
        offset_info = severstal_get_offset()

    print("\n--- Evaluating on Test Set ---")
    test_results = evaluate_all_metrics(loaded_model, test_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    val_results = None
    if eval_valdataset:
        print("\n--- Evaluating on Validation Set ---")
        val_results = evaluate_all_metrics(loaded_model, val_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    train_results = None
    if eval_traindataset:
        print("\n--- Evaluating on Training Set (optional) ---")
        train_results = evaluate_all_metrics(loaded_model, train_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    print("\n--- Summary of Results ---")
    if eval_traindataset:
        print(f"Training Set:   Dice={train_results['dice']:.4f}, IoU={train_results['iou']:.4f}, HD95={train_results['hd95']:.4f}")
    if eval_valdataset:
        print(f"Validation Set: Dice={val_results['dice']:.4f},  IoU={val_results['iou']:.4f}, HD95={val_results['hd95']:.4f}")
    print(f"Test Set:     Dice={test_results['dice']:.4f}, IoU={test_results['iou']:.4f}, HD95={test_results['hd95']:.4f}")

    print(f"\n--- Saving results to {results_filename} ---")
    try:
        with open(results_filename, "a", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint Path: {model_path}\n")
            f.write("\n--- Evaluation Metrics ---\n")
            if train_results:
                f.write(f"Training Set:   Dice={train_results['dice']:.4f}, IoU={train_results['iou']:.4f}, HD95={train_results['hd95']:.4f}\n")
            if val_results:
                f.write(f"Validation Set: Dice={val_results['dice']:.4f},  IoU={val_results['iou']:.4f}, HD95={val_results['hd95']:.4f}\n")
            f.write(f"Test Set:     Dice={test_results['dice']:.4f}, IoU={test_results['iou']:.4f}, HD95={test_results['hd95']:.4f}\n")
            f.write("="*50 + "\n\n")
        print("Results successfully saved.")
    except Exception as e:
        print(f"Error saving results to file: {e}")
    
    del loaded_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
