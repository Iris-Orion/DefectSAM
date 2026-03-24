# 适配 severstal neu sd900 magnetic_tile flood数据集
import torch
import torch.nn.functional as F
import pytz
import monai
import numpy as np
import cv2
import time
import swanlab
import argparse
from tqdm import tqdm
from torch.optim import AdamW
from datetime import datetime
from transformers import SamModel
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import get_cosine_schedule_with_warmup
from utils.utils import (compute_dice_score, 
                         compute_iou_score, 
                         save_model, 
                         save_training_logs, 
                         print_trainable_parameters)

from utils.sam_arch import (get_loradsc_model, 
                            get_loradsc_gated_model, 
                            get_sam_loraDSC_qv_vision_encoder, 
                            create_model_for_inference, 
                            loraConv_attnqkv)
from utils.loratask import get_hf_lora_model, get_hf_adalora_model


def debug_print_optimizer_param_groups(optimizer: torch.optim.Optimizer) -> None:
    """
    打印优化器中每个 param group 的关键信息，便于确认 LoRA+ 分组是否生效。
    """
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

def create_model_from_type(args: argparse.Namespace, train_dataloader: DataLoader = None):
    """
    根据给定的模型类型字符串和参数创建一个模型实例。

    Args:
        model_type (str): 模型的类型标识符 (e.g., 'loradsc_qv', 'lora_encoder').
        args (argparse.Namespace): 包含所有超参数的args对象。
        train_dataloader (DataLoader, optional): 仅在需要计算总步数时(如AdaLora)提供。

    Returns:
        torch.nn.Module: 创建好的模型实例。
    """
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    model_type = args.ft_type
    sam_type = args.sam_type

    print(f"--- Creating model of type: {model_type} with rank: {lora_rank} ---")

    if model_type == 'loradsc_qv':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True, sam_type=sam_type)

    elif model_type == 'loradsc_qv_gated':
        return get_loradsc_gated_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
                                       ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True,
                                       gate_init=1e-3, sam_type=sam_type)
    
    elif model_type == 'loradsc_q':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=False, ft_v=False, add_dsc_conv=True)
    
    elif model_type == 'loradsc_qk':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=True, ft_v=False, add_dsc_conv=True)
    
    elif model_type == 'loradsc_qkv':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=True, ft_v=True, add_dsc_conv=True)
    
    elif model_type == 'lora_attn':
        return loraConv_attnqkv(lora_rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, add_std_conv=False)
    
    elif model_type == 'lora_attn_qv':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=False)

    elif model_type in ['lora_encoder', 'lora_decoder', 'adalora_encoder', 'sam_fully', 'sam_decoder']:
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")

        if model_type == 'lora_encoder':
            return get_hf_lora_model(model=hgsam_model, target_part='vision_encoder')
        
        elif model_type == 'lora_decoder':
            return get_hf_lora_model(model=hgsam_model, target_part='mask_decoder')

        elif model_type == 'adalora_encoder':
            if train_dataloader is None:
                raise ValueError("train_dataloader must be provided for 'adalora_encoder' type.")
            ada_lora_rank = 8
            ada_init_r = 12
            total_step = args.num_epochs * len(train_dataloader)
            return get_hf_adalora_model(model=hgsam_model, total_step=total_step, target_part='vision_encoder', lora_rank=ada_lora_rank, init_r=ada_init_r)
        
        elif model_type == 'sam_fully':
            return hgsam_model
        
        elif model_type == 'sam_decoder':
            for name, param in hgsam_model.named_parameters():
                if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                    param.requires_grad_(False)
            return hgsam_model
            
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Please check your configuration.")

def reverse_letterbox_1ch(input: np.ndarray, orig_size: tuple, target_size: tuple = (1024, 1024)) -> np.ndarray:
    """
    参数:
        mask_letterbox: np.ndarray, shape = (target_height, target_width) 例如 (1024, 1024)
        orig_size: 原始尺寸 (orig_height, orig_width) 例如 (256, 1600)
        target_size: letterbox处理时的目标尺寸 (target_width, target_height) 例如 (1024, 1024)
    返回:
        restored_mask: np.ndarray, shape = (orig_height, orig_width)
    """
    orig_h, orig_w = orig_size
    target_w, target_h = target_size

    # 计算原始letterbox处理时的参数
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # 提取letterbox中的有效区域
    mask_cropped = input[y_offset : y_offset + new_h, x_offset : x_offset + new_w]

    # 使用最近邻插值缩放到原始尺寸
    # 注意OpenCV的resize参数是(width, height)
    restored_mask = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    return restored_mask

def severstal_get_offset():
    """
    对每张 mask 应用 reverse letterbox (PyTorch 版本)
    """
    orig_h, orig_w = (256, 1600)
    target_h, target_w = (1024, 1024)
    pred_h, pred_w = (256, 256)         #256 x 256

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    x_offset, y_offset = (target_w - new_w) // 2, (target_h - new_h) // 2

    #将 letterbox 空间的裁剪坐标映射到低分辨率 (256x256) 预测掩码空间
    crop_y_start = int(y_offset * (pred_h / target_h))
    crop_y_end = int((y_offset + new_h) * (pred_h / target_h))
    crop_x_start = int(x_offset * (pred_w / target_w))
    crop_x_end = int((x_offset + new_w) * (pred_w / target_w))
    return (crop_y_start, crop_x_start, crop_y_end, crop_x_end)

def _process_batch_severstal(batch, model, loss_fn, device, use_amp, auto_seg = False, offset_info = None):
    """
    处理severstal数据集单个批次的数据，执行前向传播和损失计算。
    """
    images = batch["image"].to(device)
    ground_truth_masks = batch["mask"].unsqueeze(1).float().to(device)      # [b, 256, 1600] ----unsqueeze---> [b, 1, 256, 1600]
    if auto_seg:
        bboxes = None    #自动分割
    else:
        bboxes = batch["bbox"].unsqueeze(1).to(device)   #box prompt

    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(pixel_values=images, input_boxes=bboxes, multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)  # predicted_masks: [B, 1, 256, 256]

        orig_h, orig_w = (256, 1600)
        crop_y_start, crop_x_start, crop_y_end, crop_x_end = offset_info

        # print(f"crop_y_start, crop_x_start, crop_y_end, crop_x_end: {crop_y_start, crop_x_start, crop_y_end, crop_x_end}")

        # 在256，256的尺寸内提取有效区域 (使用 Tensor slicing，这会保留计算图)
        masks_cropped_small = predicted_masks[:, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        
        # 使用 PyTorch 的 interpolate 缩放到原始尺寸
        predicted_masks_256_1600 = F.interpolate(
            masks_cropped_small,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        ) # [B, 1, 256, 1600]

        # 计算损失
        loss = loss_fn(predicted_masks_256_1600, ground_truth_masks)
    return loss, predicted_masks_256_1600, ground_truth_masks

def _process_batch(batch, model, loss_fn, device, use_amp, auto_seg = False, offset_info = None):
    """
    处理单个批次的数据，执行前向传播和损失计算。
    此函数可用于sd900, magnetic, neu,训练和验证。
    """
    images = batch["image"].to(device)
    ground_truth_masks = batch["mask"].unsqueeze(1).float().to(device)
    if auto_seg:
        bboxes = None    #TODO 有待完善，直接使用None其实不符合sam的默认输入要求
    else:
        bboxes = batch["bbox"].unsqueeze(1).to(device)   #box prompt

    with torch.amp.autocast(device_type="cuda"):
        outputs = model(pixel_values=images, input_boxes=bboxes, multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)  # [B, 1, 256, 256]

        ## 第一种选择，将预测结果进行上采样  # [B, 1, 256, 256]  ---interpolate---> [B, 1, 1024, 1024]
        # ori_res_masks = F.interpolate(predicted_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
        # assert ori_res_masks.shape == ground_truth_masks.shape, \
        # f"Shape mismatch: ori_res_masks shape is {ori_res_masks.shape}, " \
        # f"but ground_truth_masks shape is {ground_truth_masks.shape}."
        # loss = loss_fn(ori_res_masks, ground_truth_masks)

        # 第二种选择，将gt进行下采样到(256, 256)，在256x256的低分辨率空间计算损失，效率更高
        gt_downsampled = F.interpolate(ground_truth_masks, size=(256, 256), mode="nearest")
        loss = loss_fn(predicted_masks, gt_downsampled)      # [B, 1, 256, 256]

    return loss, predicted_masks, gt_downsampled


def _sample_points_from_mask(gt_mask, num_points=1, sample_from='foreground'):
    """
    从GT mask中采样点坐标，模拟SAM原始训练中的point prompt。
    Args:
        gt_mask: [H, W] numpy array 或 tensor, 二值mask
        num_points: 采样点数
        sample_from: 'foreground' 从前景采样, 'error_region' 从预测错误区域采样
    Returns:
        coords: [num_points, 2] numpy array, (x, y) 格式
        labels: [num_points] numpy array, 1=前景, 0=背景
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

    # 如果采样不足（极端情况），用零填充
    while len(coords) < num_points:
        coords.append([0, 0])
        labels.append(0)

    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int64)


def _sample_correction_points(pred_mask, gt_mask, num_points=1):
    """
    从上一轮预测的错误区域采样纠正点（SAM多轮迭代的核心）。
    假阳性区域采样背景点(label=0)，假阴性区域采样前景点(label=1)。
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = (torch.sigmoid(pred_mask) > 0.5).float().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]

    false_negative = (gt_mask > 0.5) & (pred_mask < 0.5)  # 漏检
    false_positive = (gt_mask < 0.5) & (pred_mask > 0.5)  # 误检

    fn_indices = np.argwhere(false_negative)
    fp_indices = np.argwhere(false_positive)

    coords = []
    labels = []

    # 优先从较大的错误区域采样
    if len(fn_indices) >= len(fp_indices) and len(fn_indices) > 0:
        chosen = fn_indices[np.random.choice(len(fn_indices), size=min(num_points, len(fn_indices)), replace=False)]
        for yx in chosen:
            coords.append([yx[1], yx[0]])
            labels.append(1)  # 前景纠正点
    elif len(fp_indices) > 0:
        chosen = fp_indices[np.random.choice(len(fp_indices), size=min(num_points, len(fp_indices)), replace=False)]
        for yx in chosen:
            coords.append([yx[1], yx[0]])
            labels.append(0)  # 背景纠正点

    while len(coords) < num_points:
        coords.append([0, 0])
        labels.append(0)

    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int64)


def _process_batch_sam_style(batch, model, loss_fn, device, use_amp,
                             auto_seg=False, offset_info=None,
                             prompt_probs=(0.5, 0.3, 0.2),
                             num_iter_rounds=3,
                             num_points=1):
    """
    复刻SAM原始训练流程的process_batch分支。

    核心逻辑:
      1. 第一轮: 随机选择 prompt 类型 (box / point / no_prompt)
         - box: 从GT mask提取bbox (扰动已在dataset中完成)
         - point: 从GT前景区域随机采样点
         - no_prompt: 不提供任何prompt (概率较低)
      2. 后续轮次: 将上一轮的pred mask logits作为mask prompt送回，
         同时从预测错误区域采样纠正点
      3. 只对最后一轮计算loss并反向传播，前面的轮次用no_grad节省显存

    Args:
        prompt_probs: (box_prob, point_prob, no_prompt_prob) 三种prompt的采样概率
        num_iter_rounds: 迭代轮数 (SAM论文中默认用了多轮)
        num_points: 每轮采样的点数
    """
    images = batch["image"].to(device)
    ground_truth_masks = batch["mask"].unsqueeze(1).float().to(device)  # [B, 1, H, W]
    batch_size = images.shape[0]

    gt_downsampled = F.interpolate(ground_truth_masks, size=(256, 256), mode="nearest")  # [B, 1, 256, 256]

    # 随机选择第一轮的prompt类型
    prompt_type = np.random.choice(['box', 'point', 'no_prompt'], p=prompt_probs)

    prev_masks_logits = None  # 上一轮的预测logits，用作mask prompt
    is_last_round = (num_iter_rounds == 1)

    for round_idx in range(num_iter_rounds):
        is_last_round = (round_idx == num_iter_rounds - 1)
        input_boxes = None
        input_points = None
        input_labels = None

        # 构造当前轮的prompt
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
            # else: no_prompt -> 全部为None
        else:
            # 后续轮次：从错误区域采样纠正点
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
            # 前面的轮次不需要梯度，只生成mask prompt，大幅节省显存
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(
                        pixel_values=images,
                        input_boxes=input_boxes,
                        input_points=input_points,
                        input_labels=input_labels,
                        multimask_output=False,
                    )
                    prev_masks_logits = outputs.pred_masks.squeeze(1).detach()  # [B, 1, 256, 256]
        else:
            # 最后一轮：保留梯度，计算loss
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(
                    pixel_values=images,
                    input_boxes=input_boxes,
                    input_points=input_points,
                    input_labels=input_labels,
                    multimask_output=False,
                )
                predicted_masks = outputs.pred_masks.squeeze(1)  # [B, 1, 256, 256]
                loss = loss_fn(predicted_masks, gt_downsampled)

    return loss, predicted_masks, gt_downsampled


def _process_batch_with_point_grid(batch, model, loss_fn, device, use_amp,
                                   auto_seg=True, points_per_side=16, offset_info = None):
    """
    使用密集点网格作为提示进行边缘检测微调
    """
    images = batch["image"].to(device)
    ground_truth_masks = batch["mask"].unsqueeze(1).float().to(device)
    batch_size, _, h, w = images.shape
    
    if auto_seg:
        # 生成规则网格点
        grid_size = points_per_side
        y_coords = torch.linspace(0, h-1, grid_size, device=device)
        x_coords = torch.linspace(0, w-1, grid_size, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # [grid_size*grid_size, 2]
        point_coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        
        # ✅ 修复：添加 point_batch_size 维度
        # [B, 1, N, 2] - 每张图生成 1 组掩码，使用 N 个点
        point_coords = point_coords.unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, -1, -1
        )  # [B, 1, N, 2]
        
        # ✅ labels 也需要匹配形状 [B, 1, N]
        point_labels = torch.ones(
            batch_size, 1, point_coords.shape[2], 
            dtype=torch.long,
            device=device
        )
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(
                            pixel_values=images,
                            input_points=point_coords,  # 使用点提示
                            input_labels=point_labels,
                            multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            
            print(f"predicted_masks shape: {predicted_masks.shape}")
            print(f"ground_truth_masks shape: {ground_truth_masks.shape}")
            gt_downsampled = F.interpolate(ground_truth_masks, size=(256, 256), mode="nearest")
            loss = loss_fn(predicted_masks, gt_downsampled)
    else:
        raise NotImplementedError("Point grid processing is only implemented for auto_seg=True.")

    return loss, predicted_masks, gt_downsampled

def _train_one_epoch(model, 
                     dataloader, 
                     optimizer, 
                     scheduler, 
                     loss_fn, 
                     procees_batch_fn, 
                     scaler, device, 
                     auto_seg = False, 
                     offset_info=None):
    """
    执行一个完整的训练 epoch。
    """
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0)
    use_amp = scaler is not None

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        loss, pred_masks, gt = procees_batch_fn(batch, 
                                                model, 
                                                loss_fn, 
                                                device, 
                                                use_amp, 
                                                auto_seg = auto_seg, 
                                                offset_info = offset_info)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        scheduler.step()  # 学习率调度，根据选择的scheduler需要判断scheduler在每个batch中更新还是在一轮epoch之后再更新

        total_loss += loss.item()
        total_dice += compute_dice_score(pred_masks, gt)
        total_iou += compute_iou_score(pred_masks, gt)
        hd95_metric(y_pred=(torch.sigmoid(pred_masks) > 0.5).float().cpu(), y=gt.cpu())

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_hd95 = hd95_metric.aggregate().item()
    hd95_metric.reset()

    return avg_loss, avg_dice, avg_iou, avg_hd95

def _evaluate(model, 
              dataloader, 
              loss_fn, 
              procees_batch_fn, 
              device, scaler, 
              auto_seg = False, 
              offset_info = None):
    """
    评估模型。
    """
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0)
    use_amp = scaler is not None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            loss, pred_masks, gt = procees_batch_fn(batch, 
                                                    model, 
                                                    loss_fn, 
                                                    device, 
                                                    use_amp, 
                                                    auto_seg = auto_seg, 
                                                    offset_info = offset_info)
            
            total_loss += loss.item()
            total_dice += compute_dice_score(pred_masks, gt)
            total_iou += compute_iou_score(pred_masks, gt)
            hd95_metric(y_pred=(torch.sigmoid(pred_masks) > 0.5).float().cpu(), y=gt.cpu())

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_hd95 = hd95_metric.aggregate().item()
    hd95_metric.reset()

    return avg_loss, avg_dice, avg_iou, avg_hd95

def run_finetune_engine(train_dataloader, 
                        val_dataloader, 
                        test_dataloader, 
                        model, 
                        device,
                        hyperparameters, 
                        process_batch_fn,
                        loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean'), 
                        save_dir = "./new_weights/unspec_output", auto_seg=False):
    """
    一个通用的微调训练引擎，适用于所有数据集。
    Args:
        process_batch_fn (function): 用于处理数据批次的函数。应为 `_process_batch` 或 `_process_batch_severstal`。
    """

    print_trainable_parameters(model)
    print(f"using devcie {device}")
    
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            print(name, param.shape)                # lora系列模型的插入位置信息debug

    shanghai_tz = pytz.timezone('Asia/Shanghai')                                # 设置时区为亚洲/上海
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")       # 获取当前时间戳

    # 2. 初始化 WandB (如果开启)
    swanlab_run = None
    if hyperparameters.get('use_swanlab', False):
        swanlab_run = swanlab.init(
            project=hyperparameters.get('swanlab_project', 'retina_project'),
            experiment_name=f"{hyperparameters.get('task_name')}_{start_timestamp}",
            config=hyperparameters,     # 自动记录所有超参数
        )

    SAVE_HUGGINGFACE_PRETRAINED_MODEL = hyperparameters['save_hf_format']
    save_lora_only = hyperparameters['save_custom_lora']
    if save_lora_only or SAVE_HUGGINGFACE_PRETRAINED_MODEL:
        log_name = hyperparameters["ft_type"] + "_rank_" + str(hyperparameters["lora_rank"])
    else:
        log_name = hyperparameters["ft_type"]

    print(f"权重保存格式: hugging face格式:{SAVE_HUGGINGFACE_PRETRAINED_MODEL} || lora格式:{save_lora_only}")

    # 从超参数中获取值
    lr = hyperparameters['learning_rate']
    wd = hyperparameters['weight_decay']
    num_epochs = hyperparameters['num_epochs']
    warmup_ratio = hyperparameters['warmup_ratio']
    patience = hyperparameters['patience']
    min_delta = hyperparameters['min_delta']
    use_loraplus_optim = hyperparameters.get('use_loraplus_optim', False)
    lora_plus_lr_ratio = hyperparameters.get('lora_plus_lr_ratio', 16.0)
    use_early_stop = not hyperparameters.get('disable_early_stop', False)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]      # 收集所有可训练的参数

    optimizer = None
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
            print(f"Using LoRA+ optimizer param groups (ratio={lora_plus_lr_ratio}).")
        else:
            print("LoRA+ optimizer requested but no compatible modules found; fallback to standard AdamW.")

    if optimizer is None:
        optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=wd)

    debug_print_optimizer_param_groups(optimizer)
    # loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    scaler = torch.cuda.amp.GradScaler(enabled=True)  # 混合精度训练

    # 学习率调度策略 
    total_steps = len(train_dataloader) * num_epochs
    warmup_ratio = 0.1
    warmup_steps = int(warmup_ratio * total_steps)

    cosine_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles = 0.5
    )

    history = {
        "train_loss": [], "train_dice": [], "train_iou": [], "train_hd95": [],
        "val_loss": [], "val_dice": [], "val_iou": [], "val_hd95": [],
    }
    best_val_dicescore = 0.0
    no_improve_epochs = 0
    best_epoch = -1
    best_model_path = None

    if use_early_stop:
        print(f"早停已启用 (patience={patience}, min_delta={min_delta}).")
    else:
        print("早停已关闭：将运行完整训练轮次，同时仍跟踪最佳指标以保存模型。")

    model.to(device)
    model.train()

    # CHANGE: 仅在需要时计算 offset_info
    offset_info = None
    if process_batch_fn == _process_batch_severstal:
        print("Severstal-specific processing enabled. Calculating offset info.")
        offset_info = severstal_get_offset()

    # 将 process_batch_fn 传递给训练周期函数
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")

        train_loss, train_dice, train_iou, train_hd95 = _train_one_epoch(model, train_dataloader, optimizer,
                                                                   cosine_scheduler, loss_fn, process_batch_fn, scaler, device, auto_seg = auto_seg, offset_info = offset_info)
        print(f"Training loss: {train_loss:.4f}, train dice: {train_dice:.4f}, train iou: {train_iou:.4f}, train hd95: {train_hd95:.4f}")

        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Current learning rate (param_group {i}): {param_group['lr']:.6e}")

        val_loss, val_dice, val_iou, val_hd95 = _evaluate(model, val_dataloader, loss_fn, process_batch_fn,
                                                      device, scaler, auto_seg = auto_seg, offset_info=offset_info)
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val HD95: {val_hd95:.4f}')

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["train_hd95"].append(train_hd95)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_hd95"].append(val_hd95)

        if swanlab_run:
            swanlab.log({
                "train/loss": train_loss,
                "train/dice": train_dice,
                "train/iou": train_iou,
                "train/hd95": train_hd95,
                "val/loss": val_loss,
                "val/dice": val_dice,
                "val/iou": val_iou,
                "val/hd95": val_hd95,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch+1)

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
                "train_hd95": train_hd95,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "val_hd95": val_hd95,
            }
            print(f"验证 dice 改善到 {best_val_dicescore:.4f}, 保存模型...")
            best_model_path = save_model( hyperparameters=hyperparameters,
                                            start_timestamp = start_timestamp,
                                            model=model,
                                            optimizer=optimizer,
                                            scaler=scaler,
                                            epoch = epoch+1,
                                            model_name= log_name,
                                            target_dir= save_dir,
                                            SAVE_HUGGINGFACE_PRETRAINED_MODEL = SAVE_HUGGINGFACE_PRETRAINED_MODEL,
                                            save_lora_only=save_lora_only)
            if swanlab_run:
                swanlab.log({
                    "best_epoch": best_epoch,
                    "best/train_loss": train_loss,
                    "best/train_dice": train_dice,
                    "best/train_iou": train_iou,
                    "best/train_hd95": train_hd95,
                    "best/val_loss": val_loss,
                    "best/val_dice": val_dice,
                    "best/val_iou": val_iou,
                    "best/val_hd95": val_hd95,
                    })
        else:
            if use_early_stop:
                no_improve_epochs += 1
                print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
            else:
                print("验证 dice 未改善（未启用早停），继续完整训练。")

        output_data = save_training_logs( hyperparameters = hyperparameters,
                                            results = history,
                                            epoch = epoch+1,
                                            start_timestamp = start_timestamp,
                                            result_name = log_name,
                                            target_dir = save_dir)
        if use_early_stop and no_improve_epochs >= patience:
            print(f"验证指标连续 {patience} 个 epoch 无改善，提前停止训练。")
            break

    print("\n--- Training finished. Starting final evaluation with the best model. ---")
    if not best_model_path:
        print("Warning: No best model was saved. Final evaluation will be on the last state of the model.")
        loaded_model = model
    else:
        print(f"Loading best model from: {best_model_path}")
        if SAVE_HUGGINGFACE_PRETRAINED_MODEL:
            try:
                config = PeftConfig.from_pretrained(best_model_path)
                base_model_path = config.base_model_name_or_path
                base_model = SamModel.from_pretrained(base_model_path)
                
                loaded_model = PeftModel.from_pretrained(base_model, best_model_path)       # 从保存的路径加载 PeftModel
                loaded_model.to(device)
                print("Successfully loaded model in Hugging Face PEFT format.")
            except Exception as e:
                print(f"Error loading Hugging Face PEFT model: {e}")
                loaded_model = None            # 加载失败
        elif save_lora_only:
            loaded_model = create_model_for_inference(model=model, lora_weights_path=best_model_path, device=device)
        else:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            loaded_model = model
            print("Successfully loaded model from standard PyTorch checkpoint.")

    # 在测试集上评估
    final_test_loss, final_test_dice, final_test_iou, final_test_hd95 = _evaluate(loaded_model, test_dataloader, loss_fn, process_batch_fn, device, scaler, auto_seg=auto_seg, offset_info = offset_info)
    print(f'Final Test Set Evaluation: Loss: {final_test_loss:.4f}, Dice: {final_test_dice:.4f}, IoU: {final_test_iou:.4f}, HD95: {final_test_hd95:.4f}')
    history["final_test_metrics"]={"loss": final_test_loss, "dice": final_test_dice, "iou": final_test_iou, "hd95": final_test_hd95}

    if swanlab_run:
        swanlab.log({"test/test_dice": final_test_dice,
                     "test/test_iou": final_test_iou,
                     "test/test_hd95": final_test_hd95})
        swanlab.finish()

    # 将最终结果保存到日志文件中
    save_training_logs(
        hyperparameters=hyperparameters, results=history, epoch=output_data["last_completed_epoch"],
        start_timestamp=start_timestamp, result_name=log_name, target_dir=save_dir,
    )
    print("--- Training finished ---")
    return history, model


def evaluate_all_metrics_profiler(model, dataloader, loss_fn, process_batch_fn, device, auto_seg=False, offset_info=None):
    """
    在给定的数据集上评估模型，并使用 PROFILER 分析性能。
    """
    model.eval()

    # --- 您的原始指标初始化代码保持不变 ---
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0)
    total_dice, total_iou = 0, 0
    
    # CHANGE 2: 设置分析参数 (预热1个批次，记录接下来的3个批次)
    warmup_steps = 1
    active_steps = 3
    total_steps_to_profile = warmup_steps + active_steps

    # CHANGE 3: 初始化 Profiler 上下文管理器
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], # 同时记录 CPU 和 GPU 活动
        record_shapes=True, # 记录张量的形状
        profile_memory=True, # 记录显存使用情况
        with_stack=True, # 记录调用堆栈，便于追溯源头
        schedule=torch.profiler.schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=1) # 使用schedule来自动控制
    ) as prof:
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader, desc="Profiling & Final Evaluation")):
                
                # CHANGE 4: 在循环内部使用 record_function 标记关键代码块，使报告更清晰
                with record_function("process_batch_and_metrics"):
                    _, pred_masks_logits, gt_masks = process_batch_fn(
                        batch, model, loss_fn=loss_fn, device=device, use_amp=False, auto_seg=auto_seg, offset_info=offset_info
                    )

                    # 手动计算的指标
                    total_dice += compute_dice_score(pred_masks_logits, gt_masks)
                    total_iou += compute_iou_score(pred_masks_logits, gt_masks)
                
                    # MONAI 指标
                    pred_masks_binary = (torch.sigmoid(pred_masks_logits) > 0.5).float()
                    hd95_metric(y_pred=pred_masks_binary, y=gt_masks)

                # CHANGE 5: 通知 profiler 已完成一步
                prof.step()

                # CHANGE 6: 如果只想分析几个批次，可以提前中断循环
                if step >= total_steps_to_profile -1 :
                     # 注意：为了得到完整的评估结果，这里我们不中断。
                     # Profiler在达到active_steps后会自动停止记录。
                     # 如果你只想快速分析，可以取消下面这行注释:
                     # break 
                     pass

    # CHANGE 7: 在循环结束后，打印分析结果
    # 按照 CUDA 总耗时降序排列，显示前 15 个最耗时的操作
    print("\n" + "="*80)
    print("PYTORCH PROFILER RESULTS:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print("="*80 + "\n")

    # --- 您的原始结果计算和打印代码保持不变 ---
    mean_iou = total_iou / len(dataloader)
    mean_dice = total_dice / len(dataloader)
    mean_hd95 = hd95_metric.aggregate().item()
    
    # 重置计算器状态
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

def evaluate_all_metrics(model, dataloader, loss_fn, process_batch_fn, device, auto_seg=False, offset_info=None):
    """
    在给定的数据集上评估模型，并计算所有需要的指标 (IoU, Dice, BF1, HD)。
    此函数用于训练完成后的最终评估。
    """
    model.eval()

    # 1. 初始化所有指标计算器
    # IoU Metric
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    # Dice Metric
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    # Hausdorff Distance Metric (95th percentile)
    # 使用95%分位数HD更稳健，能有效避免离群点的极端影响
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0)

    total_dice, total_iou = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Final Evaluation"):
            # neu, sd900, magnetic用这个
            _, pred_masks_logits, gt_masks = process_batch_fn(
                batch, model, loss_fn=loss_fn, device=device, use_amp=False, auto_seg=auto_seg, offset_info=offset_info
            )

            total_dice += compute_dice_score(pred_masks_logits, gt_masks)
            total_iou += compute_iou_score(pred_masks_logits, gt_masks)

            # # 2. 将模型输出的logits转换为二进制掩码 (Binarize predictions)
            # # MONAI的指标函数需要 (B, C, H, W) 格式的二进制输入
            pred_masks_binary = (torch.sigmoid(pred_masks_logits) > 0.5).float()

            # # 3. 将当前批次的结果喂给指标计算器
            # # MONAI会自动处理累加过程
            # iou_metric(y_pred=pred_masks_binary, y=gt_masks)
            # dice_metric(y_pred=pred_masks_binary, y=gt_masks)
            # 避免result in nan/inf distance.
            if gt_masks.sum() > 0 and pred_masks_binary.sum() > 0:
                hd95_metric(y_pred=pred_masks_binary, y=gt_masks)

    # 4. 在所有数据都处理完后，计算最终的平均指标
    # mean_iou = iou_metric.aggregate().item()
    # mean_dice = dice_metric.aggregate().item()

    mean_iou = total_iou / len(dataloader)
    mean_dice = total_dice / len(dataloader)
    mean_hd95 = hd95_metric.aggregate().item()

    # 重置计算器状态，以便下次使用
    # iou_metric.reset()
    # dice_metric.reset()
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
                     auto_seg: bool = False, eval_traindataset = False):
    """
    纯推理, 加bbox和不加bbox?
    """
    SAVE_HUGGINGFACE_PRETRAINED_MODEL = args.save_hf_format
    save_lora_only = args.save_custom_lora
    zero_shot_infer = args.zero_shot
    if SAVE_HUGGINGFACE_PRETRAINED_MODEL:
            try:
                config = PeftConfig.from_pretrained(best_model_path)
                base_model_path = config.base_model_name_or_path
                base_model = SamModel.from_pretrained(base_model_path).to(device)
                
                loaded_model = PeftModel.from_pretrained(base_model, best_model_path)       # 从保存的路径加载 PeftModel
                print("Successfully loaded model in Hugging Face PEFT format.")
            except Exception as e:
                print(f"Error loading Hugging Face PEFT model: {e}")
                loaded_model = None            # 加载失败
    elif save_lora_only:
        loaded_model = create_model_for_inference(model=model, lora_weights_path=best_model_path, device=device)
    
    else:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model = model
        loaded_model.to(device)
        print("Successfully loaded model from standard PyTorch checkpoint.")
    
    offset_info = None
    if process_batch_fn == _process_batch_severstal:
        print("Severstal-specific processing enabled. Calculating offset info.")
        offset_info = severstal_get_offset()

    # --- 执行评估 ---
    print("\n--- Evaluating on Test Set ---")
    # test_results = evaluate_all_metrics_profiler(loaded_model, test_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)
    test_results = evaluate_all_metrics(loaded_model, test_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    print("\n--- Evaluating on Validation Set ---")
    # val_results = evaluate_all_metrics_profiler(loaded_model, val_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)
    val_results = evaluate_all_metrics(loaded_model, val_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    train_results = None
    if eval_traindataset:
        print("\n--- Evaluating on Training Set (optional) ---")
        # train_results = evaluate_all_metrics_profiler(loaded_model, train_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)
        train_results = evaluate_all_metrics(loaded_model, train_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    # --- 整理结果 ---
    print("\n--- Summary of Results ---")
    if eval_traindataset:
        print(f"Training Set:   Dice={train_results['dice']:.4f}, IoU={train_results['iou']:.4f}, HD95={train_results['hd95']:.4f}")
    print(f"Validation Set: Dice={val_results['dice']:.4f},  IoU={val_results['iou']:.4f}, HD95={val_results['hd95']:.4f}")
    print(f"Test Set:     Dice={test_results['dice']:.4f}, IoU={test_results['iou']:.4f}, HD95={test_results['hd95']:.4f}")

    # --- 将路径和结果保存到TXT文件 ---
    print(f"\n--- Saving results to {results_filename} ---") # <-- Use the parameter here
    try:
        with open(results_filename, "a", encoding="utf-8") as f: # <-- And here
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint Path: {best_model_path}\n")
            f.write("\n--- Evaluation Metrics ---\n")
            if train_results:
                f.write(f"Training Set:   Dice={train_results['dice']:.4f}, IoU={train_results['iou']:.4f}, HD95={train_results['hd95']:.4f}\n")
            f.write(f"Validation Set: Dice={val_results['dice']:.4f},  IoU={val_results['iou']:.4f}, HD95={val_results['hd95']:.4f}\n")
            f.write(f"Test Set:     Dice={test_results['dice']:.4f}, IoU={test_results['iou']:.4f}, HD95={test_results['hd95']:.4f}\n")
            f.write("="*50 + "\n\n")
        print("Results successfully saved.")
    except Exception as e:
        print(f"Error saving results to file: {e}")
    
    # 推理完毕，手动释放模型显存
    del loaded_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def zero_shot(model_path, 
                     train_dataloader, val_dataloader, test_dataloader, 
                     process_batch_fn, loss_fn, device, 
                     results_filename="evaluation_results.txt",
                     auto_seg: bool = False, eval_traindataset = False):
    loaded_model = SamModel.from_pretrained(model_path).to(device)

    offset_info = None
    if process_batch_fn == _process_batch_severstal:
        print("Severstal-specific processing enabled. Calculating offset info.")
        offset_info = severstal_get_offset()

    # --- 执行评估 ---
    print("\n--- Evaluating on Test Set ---")
    # test_results = evaluate_all_metrics_profiler(loaded_model, test_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)
    test_results = evaluate_all_metrics(loaded_model, test_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    print("\n--- Evaluating on Validation Set ---")
    # val_results = evaluate_all_metrics_profiler(loaded_model, val_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)
    val_results = evaluate_all_metrics(loaded_model, val_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    train_results = None
    if eval_traindataset:
        print("\n--- Evaluating on Training Set (optional) ---")
        # train_results = evaluate_all_metrics_profiler(loaded_model, train_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)
        train_results = evaluate_all_metrics(loaded_model, train_dataloader, loss_fn, process_batch_fn, device=device, auto_seg=auto_seg, offset_info=offset_info)

    # --- 整理结果 ---
    print("\n--- Summary of Results ---")
    if eval_traindataset:
        print(f"Training Set:   Dice={train_results['dice']:.4f}, IoU={train_results['iou']:.4f}, HD95={train_results['hd95']:.4f}")
    print(f"Validation Set: Dice={val_results['dice']:.4f},  IoU={val_results['iou']:.4f}, HD95={val_results['hd95']:.4f}")
    print(f"Test Set:     Dice={test_results['dice']:.4f}, IoU={test_results['iou']:.4f}, HD95={test_results['hd95']:.4f}")

    # --- 将路径和结果保存到TXT文件 ---
    print(f"\n--- Saving results to {results_filename} ---") # <-- Use the parameter here
    try:
        with open(results_filename, "a", encoding="utf-8") as f: # <-- And here
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint Path: {model_path}\n")
            f.write("\n--- Evaluation Metrics ---\n")
            if train_results:
                f.write(f"Training Set:   Dice={train_results['dice']:.4f}, IoU={train_results['iou']:.4f}, HD95={train_results['hd95']:.4f}\n")
            f.write(f"Validation Set: Dice={val_results['dice']:.4f},  IoU={val_results['iou']:.4f}, HD95={val_results['hd95']:.4f}\n")
            f.write(f"Test Set:     Dice={test_results['dice']:.4f}, IoU={test_results['iou']:.4f}, HD95={test_results['hd95']:.4f}\n")
            f.write("="*50 + "\n\n")
        print("Results successfully saved.")
    except Exception as e:
        print(f"Error saving results to file: {e}")
    
    # 推理完毕，手动释放模型显存
    del loaded_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def neu_seg_inference_engine(model, train_dataloader, test_dataloader, checkpoint_path: str, hyperparameters, device, use_bbox: bool = False, eval_traindataset = False):
    """
    旧代码存放
    纯推理, 加bbox和不加bbox?
    """
    if hyperparameters["task_type"] == 1:
        # 使用我们封装好的函数一步创建模型
        final_model = create_model_for_inference(
            model,
            lora_rank=hyperparameters["lora_rank"],
            lora_add_conv=True,
            lora_weights_path=checkpoint_path,
            device=device
        )
        model = final_model

    elif hyperparameters["task_type"] == 2:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()  

def neu_seg_finetune_engine(model, device, train_dataloader, test_dataloader, optimizer, loss_fn, scaler, scheduler, hyperparameters):
    """
    旧代码存放位置
    """
    model.to(device)
    
    print_trainable_parameters(model)
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            print(name, param.shape)
    print(f"using devcie {device}")

    shanghai_tz = pytz.timezone('Asia/Shanghai')        # 设置时区为亚洲/上海
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳

    SAVE_HUGGINGFACE_PRETRAINED_MODEL = (hyperparameters["task_type"] == 0)
    save_lora_only = (hyperparameters["task_type"] == 1)
    print(f"权重保存格式: hugging face格式:{SAVE_HUGGINGFACE_PRETRAINED_MODEL} || lora格式:{save_lora_only}")

    best_test_dicescore = 0.0  # 记录最好的test_dicescore  
    no_improve_epochs = 0   # 连续无改进的 epoch 计数
    
    results = {"train_loss": [], "train_dicescore": [], "train_ious": [], "test_loss": [], "test_dicescore": [], "test_ious": []}

    num_epochs = hyperparameters['num_epochs']
    earlystop_PATIENCE = hyperparameters['patience']
    earlystop_MIN_DELTA = hyperparameters['min_delta']
    task_name = hyperparameters['task_name']
    output_dir = hyperparameters['output_dir']
    use_amp = scaler is not None

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_dicescore, train_ious = 0, 0, 0
        for batch in tqdm(train_dataloader):
            ground_truth_masks = batch["mask"].float().to(device)      # gt
            ground_truth_masks = ground_truth_masks.unsqueeze(1)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(pixel_values = batch["image"].to(device),
                                    input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                    multimask_output=False)
                    predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]

                    gt_down_256 = F.interpolate(
                        ground_truth_masks, 
                        size=(256, 256),
                        mode="nearest"                                      # 下采样用最近邻，保证mask是二值/多值分割不被插值破坏
                    )
                    loss = loss_fn(predicted_masks, gt_down_256)            # 直接在 256x256 空间计算 Loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                outputs = model(pixel_values = batch["image"].to(device),
                                input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                multimask_output=False)
                predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]
                gt_down_256 = F.interpolate(
                        ground_truth_masks, 
                        size=(256, 256),
                        mode="nearest"                                      # 下采样用最近邻，保证mask是二值/多值分割不被插值破坏
                    )
                loss = loss_fn(predicted_masks, gt_down_256)            # 直接在 256x256 空间计算 Loss
                loss.backward()         # 反向传播
                optimizer.step()        # optimize
                scheduler.step()

            train_loss += loss.item()
            train_dicescore += compute_dice_score(predicted_masks, gt_down_256)
            train_ious += compute_iou_score(predicted_masks, gt_down_256)
    
        train_loss = train_loss / len(train_dataloader)
        train_dicescore = train_dicescore / len(train_dataloader)
        train_ious = train_ious / len(train_dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}] \ntrain loss: {train_loss:.4f}, train dice score: {train_dicescore:.4f}, train iou: {train_ious:.4f}")

        # 验证
        model.eval()
        with torch.no_grad():
            test_loss, test_dicescore, test_ious = 0, 0, 0
            for batch in tqdm(test_dataloader):
                ground_truth_masks = batch["mask"].float().to(device)      # gt
                ground_truth_masks = ground_truth_masks.unsqueeze(1)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(pixel_values = batch["image"].to(device),
                                    input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                    multimask_output=False)
                    predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]
                    gt_down_256 = F.interpolate(ground_truth_masks, size=(256, 256), mode="nearest")
                    loss = loss_fn(predicted_masks, gt_down_256) 

                test_loss += loss.item()
                test_dicescore += compute_dice_score(predicted_masks, gt_down_256)
                test_ious += compute_iou_score(predicted_masks, gt_down_256)   
            test_loss = test_loss / len(test_dataloader)
            test_dicescore = test_dicescore / len(test_dataloader)
            test_ious = test_ious / len(test_dataloader)
            print(f'Validation Loss: {test_loss:.4f}, Val Dice: {test_dicescore:.4f}, Val IoU: {test_ious:.4f}')

        results["train_loss"].append(train_loss)
        results["train_dicescore"].append(train_dicescore)
        results["train_ious"].append(train_ious)
        results["test_loss"].append(test_loss)
        results["test_dicescore"].append(test_dicescore)
        results["test_ious"].append(test_ious)

        save_training_logs( hyperparameters = hyperparameters, 
                            results = results, 
                            epoch = epoch+1,    
                            start_timestamp = start_timestamp, 
                            result_name = task_name,
                            target_dir= output_dir)

        if test_dicescore - best_test_dicescore > earlystop_MIN_DELTA:
            best_test_dicescore = test_dicescore
            best_epoch = epoch + 1  # 更新最佳epoch
            no_improve_epochs = 0
            print(f"验证 dice 改善到 {best_test_dicescore:.5f}, 保存模型...")
            save_model( hyperparameters=hyperparameters,
                        start_timestamp = start_timestamp, 
                        model = model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch = epoch+1,
                        model_name= task_name,
                        target_dir= output_dir,
                        SAVE_HUGGINGFACE_PRETRAINED_MODEL = SAVE_HUGGINGFACE_PRETRAINED_MODEL,
                        save_lora_only=save_lora_only)
        else:
            no_improve_epochs += 1
            print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
        if no_improve_epochs >= earlystop_PATIENCE:
            print(f"验证指标连续 {earlystop_PATIENCE} 个 epoch 无改善，提前停止训练。")
            break
    return results, model

def sd_900_finetune_engine(train_dataloader, val_dataloader, test_dataloader, model, device, hyperparameters, save_dir = "./new_weights/sd900_output"):
    """
    sd900微调代码存档
    """
    # ----- debug information ----- #
    print_trainable_parameters(model)
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            print(name, param.shape)
    print(f"using devcie {device}")
    
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")

    SAVE_HUGGINGFACE_PRETRAINED_MODEL = hyperparameters['save_hf_format']
    save_lora_only = hyperparameters['save_custom_lora']
    if save_lora_only:
        log_name = hyperparameters["ft_type"] + "_rank_" + str(hyperparameters["lora_rank"])
    if SAVE_HUGGINGFACE_PRETRAINED_MODEL:
        log_name =  hyperparameters["ft_type"] + "_rank_" + str(hyperparameters["lora_rank"])
    if (not save_lora_only) and (not SAVE_HUGGINGFACE_PRETRAINED_MODEL):
        log_name = hyperparameters["ft_type"]

    print(f"权重保存格式: hugging face格式:{SAVE_HUGGINGFACE_PRETRAINED_MODEL} || lora格式:{save_lora_only}")
    # ----- debug information ----- #

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]      # 收集所有可训练的参数

    # 从超参数中获取值
    lr = hyperparameters['learning_rate']
    wd = hyperparameters['weight_decay']
    num_epochs = hyperparameters['num_epochs']
    warmup_ratio = hyperparameters['warmup_ratio']
    patience = hyperparameters['patience']
    min_delta = hyperparameters['min_delta']
    
    optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=wd)
    loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # scaler = None
    # loss_fn = seg_loss
    
    # 学习率调度策略 
    total_steps = len(train_dataloader) * hyperparameters['num_epochs']
    warmup_ratio = 0.1
    warmup_steps = int(warmup_ratio * total_steps)

    cosine_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles = 0.5
    )

    history = {"train_loss": [], "train_dicescore": [], "train_ious": [],"val_loss": [], "val_dicescore": [], "val_ious": []}
    best_val_dicescore = 0.0  # 记录最好的test_dicescore   TODO: 思考是用最小的loss的那一轮还是最大的dice score的那一轮
    no_improve_epochs = 0   # 连续无改进的 epoch 计数
    
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")

        train_loss, train_dicescore, train_ious = _train_one_epoch(
            model, train_dataloader, optimizer, cosine_scheduler, loss_fn, scaler, device)
        print(f"Training loss: {train_loss:.4f}, train dice score: {train_dicescore:.4f}, train iou: {train_ious:.4f}")

        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Current learning rate (param_group {i}): {param_group['lr']:.6e}")

        val_loss, val_dicescore, val_ious = _evaluate(model, val_dataloader, loss_fn, device, scaler)
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dicescore:.4f}, Val IoU: {val_ious:.4f}')

        history["train_loss"].append(train_loss)
        history["train_dicescore"].append(train_dicescore)
        history["train_ious"].append(train_ious)
        history["val_loss"].append(val_loss)
        history["val_dicescore"].append(val_dicescore)
        history["val_ious"].append(val_ious)

        if val_dicescore - best_val_dicescore > min_delta:
            best_val_dicescore = val_dicescore
            # best_model_wts = model.state_dict()
            no_improve_epochs = 0
            print(f"验证 dice 改善到 {best_val_dicescore:.4f}, 保存模型...")
            best_model_path = save_model( hyperparameters=hyperparameters,
                                            start_timestamp = start_timestamp, 
                                            model=model, 
                                            optimizer=optimizer,
                                            scaler=scaler,
                                            epoch = epoch+1,
                                            model_name= log_name,
                                            target_dir= save_dir,
                                            SAVE_HUGGINGFACE_PRETRAINED_MODEL = SAVE_HUGGINGFACE_PRETRAINED_MODEL,
                                            save_lora_only=save_lora_only)
        else:
            no_improve_epochs += 1
            print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
        
        output_data = save_training_logs( hyperparameters = hyperparameters,
                                results = history,
                                epoch = epoch+1,
                                start_timestamp = start_timestamp,
                                result_name = log_name,
                                target_dir = save_dir)
        if no_improve_epochs >= patience:
            print(f"验证指标连续 {patience} 个 epoch 无改善，提前停止训练。")
            break
        # --- 训练结束后，使用最佳模型进行最终评估 ---
    print("\n--- Training finished. Starting final evaluation with the best model. ---")
    if not best_model_path:
        print("Warning: No best model was saved. Final evaluation will be on the last state of the model.")
    else:
        print(f"Loading best model from: {best_model_path}")
        if SAVE_HUGGINGFACE_PRETRAINED_MODEL:
            try:
                config = PeftConfig.from_pretrained(best_model_path)
                base_model_path = config.base_model_name_or_path
                base_model = SamModel.from_pretrained(base_model_path)
                
                
                loaded_model = PeftModel.from_pretrained(base_model, best_model_path)# 从保存的路径加载 PeftModel
                loaded_model.to(device)
                print("Successfully loaded model in Hugging Face PEFT format.")
            except Exception as e:
                print(f"Error loading Hugging Face PEFT model: {e}")
                loaded_model_for_eval = None # 加载失败
            # lora_decoder_hf_path = best_model_path
            # config = PeftConfig.from_pretrained(lora_decoder_hf_path)
            # hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
            # lora_decoder_hf_model = PeftModel.from_pretrained(hgsam_model, lora_decoder_hf_path)
        elif save_lora_only:
            loaded_model = create_model_for_inference(
                                                model=model,
                                                lora_rank=hyperparameters["lora_rank"],
                                                lora_add_conv=True,
                                                lora_weights_path=best_model_path,
                                                device=device
                                            )
        else:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            loaded_model = model
            print("Successfully loaded model from standard PyTorch checkpoint.")

    # 在测试集上评估
    final_test_loss, final_test_dice, final_test_iou = _evaluate(loaded_model, test_dataloader, loss_fn, device, scaler)
    print(f'Final Test Set Evaluation: Loss: {final_test_loss:.4f}, Dice: {final_test_dice:.4f}, IoU: {final_test_iou:.4f}')
    history["final_test_metrics"]={"loss": final_test_loss, "dice": final_test_dice, "iou": final_test_iou}

    # 将最终结果保存到日志文件中
    save_training_logs(
        hyperparameters=hyperparameters, results=history, epoch=output_data["last_completed_epoch"],
        start_timestamp=start_timestamp, result_name=log_name, target_dir=save_dir,
    )
    print("--- Training finished ---")
    return history, loaded_model



def _process_batch_severstal_archieve(batch, model, loss_fn, device, use_amp, auto_seg = False, offset_info = None):
    """
    处理severstal数据集单个批次的数据，执行前向传播和损失计算。
    """
    images = batch["image"].to(device)
    ground_truth_masks = batch["mask"].unsqueeze(1).float().to(device)      # [b, 256, 1600] ----unsqueeze---> [b, 1, 256, 1600]
    if auto_seg:
        bboxes = None    #自动分割
    else:
        bboxes = batch["bbox"].unsqueeze(1).to(device)   #box prompt

    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(pixel_values=images, input_boxes=bboxes, multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)  # predicted_masks: [B, 1, 256, 256]

        # 1. 上采样到 letterbox 输入尺寸
        predicted_masks_1024 = F.interpolate(predicted_masks, size=(1024, 1024), mode="nearest")  # [B, 1, 1024, 1024]

        orig_h, orig_w = (256, 1600)
        target_h, target_w = (1024, 1024)
        
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # 提取有效区域 (使用 Tensor slicing，这会保留计算图)
        masks_cropped = predicted_masks_1024[:, :, y_offset:y_offset + new_h, x_offset:x_offset + new_w]
        
        # 使用 PyTorch 的 interpolate 缩放到原始尺寸
        predicted_masks_256_1600 = F.interpolate(
            masks_cropped,
            size=(256, 1600),
            mode="bilinear",
            align_corners=False
        ) # [B, 1, 256, 1600]

        # 计算损失
        loss = loss_fn(predicted_masks_256_1600, ground_truth_masks)
    return loss, predicted_masks_256_1600, ground_truth_masks
