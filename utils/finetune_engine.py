# 适配 severstal neu sd900 magnetic_tile flood数据集
import torch
import torch._dynamo
import torch.distributed as dist
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.suppress_errors = True   # 符号形状推导失败时静默回退到 eager，不打印警告

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
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from torch.profiler import profile, record_function, ProfilerActivity
from utils.helper_function import get_lr_scheduler
from utils.utils import (compute_dice_score,
                         compute_iou_score,
                         save_model,
                         save_training_logs,
                         print_trainable_parameters)
from utils.mfu import SAMMFUEstimator, MFUTracker

from utils.sam_arch import (get_loradsc_model,
                            get_loradsc_residual_model,
                            get_loradsc_gated_model,
                            get_loradsc_residual_gated_model,
                            create_model_for_inference,
                            get_loraplus_model,
                            get_loraga_model,
                            get_lorapro_model,
                            get_moelora_model,
                            get_moeloraplus_model)
from utils.loratask import (get_hf_adalora_model,
                            get_hf_dora_qv_model,
                            get_hf_lokr_qv_model,
                            get_hf_lora_model,
                            prepare_sam_qkv_for_qv_peft)
from utils.sam_arch import LoRA_Moe_DepwiseConv_Samqv


def collect_moe_gate_stats(model):
    """收集所有 MoE LoRA 层的 gate weights 分布统计，用于诊断门控坍塌。
    返回 dict: {
        'gate_q/layer_{i}/expert_{j}_mean': float,
        'gate_v/layer_{i}/expert_{j}_mean': float,
        'gate_q/layer_{i}/entropy': float,   # 越高=越均匀，log(3)≈1.099 为最大值
        'gate_v/layer_{i}/entropy': float,
        'gate_q/layer_{i}/max_prob': float,  # 最大专家概率均值，越接近1=越坍塌
        'gate_v/layer_{i}/max_prob': float,
    }
    如果模型中没有 MoE 层或没有缓存的 gate probs，返回空 dict。
    """
    raw_model = model.module if hasattr(model, 'module') else model

    # 收集所有子模块，包括 torch.compile 包装的子模块内部
    all_modules = []
    def _collect(m):
        all_modules.append(m)
        # torch.compile 包装后原始模块在 _orig_mod 中，需要递归进入
        if hasattr(m, '_orig_mod'):
            for sub in m._orig_mod.modules():
                all_modules.append(sub)
        else:
            for child in m.children():
                _collect(child)
    _collect(raw_model)

    stats = {}
    layer_idx = 0
    for module in all_modules:
        if isinstance(module, LoRA_Moe_DepwiseConv_Samqv):
            for prefix, attr in [('gate_q', '_last_gate_probs_q'), ('gate_v', '_last_gate_probs_v')]:
                probs = getattr(module, attr, None)
                if probs is None:
                    continue
                # probs: [batch, num_experts], 取 batch 均值
                mean_probs = probs.mean(dim=0)  # [num_experts]
                for j, p in enumerate(mean_probs):
                    stats[f'{prefix}/layer_{layer_idx}/expert_{j}_mean'] = p.item()
                # 熵: -sum(p * log(p))，衡量分布均匀度
                entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum().item()
                stats[f'{prefix}/layer_{layer_idx}/entropy'] = entropy
                # 最大概率均值: 每个 batch 样本的 max prob 的均值
                max_prob = probs.max(dim=-1).values.mean().item()
                stats[f'{prefix}/layer_{layer_idx}/max_prob'] = max_prob
            layer_idx += 1
    return stats


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


def prepare_base_model_for_hf_adapter_loading(base_model: SamModel, ft_type: str):
    """在加载 HF PEFT adapter 前，对需要的基座结构做与训练期一致的预处理。"""
    if ft_type in ['dora_qv_encoder', 'lokr_qv_encoder']:
        return prepare_sam_qkv_for_qv_peft(base_model, target_part='vision_encoder')
    return base_model

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
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, 
        ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True, sam_type=sam_type)
    
    elif model_type == 'lora_attn_qv':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=False)

    # 只注入global attn的部分没什么区别
    
    elif model_type == 'loradsc_qv_residual':
        return get_loradsc_residual_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
                                          ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True, sam_type=sam_type)

    elif model_type == 'loradsc_qv_gated':
        return get_loradsc_gated_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
                                       ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True,
                                       gate_init=1e-3, sam_type=sam_type)

    elif model_type == 'loradsc_qv_residual_gated':
        return get_loradsc_residual_gated_model(
            rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
            ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True,
            gate_init=0.0,
            use_symmetric_init=args.use_residual_gated_symmetric_init,
            symmetric_init_std=args.residual_gated_symmetric_init_std,
            sam_type=sam_type)

    elif model_type == 'loradsc_qkv_residual_gated':
        return get_loradsc_residual_gated_model(
            rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
            ft_q=True, ft_k=True, ft_v=True, add_dsc_conv=True,
            gate_init=0.0,
            use_symmetric_init=args.use_residual_gated_symmetric_init,
            symmetric_init_std=args.residual_gated_symmetric_init_std,
            sam_type=sam_type)

    elif model_type == 'loradsc_q_residual_gated':
        return get_loradsc_residual_gated_model(
            rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
            ft_q=True, ft_k=False, ft_v=False, add_dsc_conv=True,
            gate_init=0.0,
            use_symmetric_init=args.use_residual_gated_symmetric_init,
            symmetric_init_std=args.residual_gated_symmetric_init_std,
            sam_type=sam_type)

    elif model_type == 'loradsc_qv_adaptive':
        args.use_loraplus_optim = True
        from utils.sam_arch import get_loradsc_adaptive_gated_model
        return get_loradsc_adaptive_gated_model(
            rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
            ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True, sam_type=sam_type)

    elif model_type == 'loradsc_qkv_adaptive':
        args.use_loraplus_optim = True
        from utils.sam_arch import get_loradsc_adaptive_gated_model
        return get_loradsc_adaptive_gated_model(
            rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
            ft_q=True, ft_k=True, ft_v=True, add_dsc_conv=True, sam_type=sam_type)

    elif model_type == 'loradsc_q':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=False, ft_v=False, add_dsc_conv=True)
    
    elif model_type == 'loradsc_qk':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=True, ft_v=False, add_dsc_conv=True)
    
    elif model_type == 'loradsc_qkv':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=True, ft_v=True, add_dsc_conv=True)

    elif model_type == 'loraplus_qv':
        args.use_loraplus_optim = True  # 强制启用 LoRA+ 优化器
        return get_loraplus_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
                                  ft_q=True, ft_k=False, ft_v=True, sam_type=sam_type)

    elif model_type == 'loraga_qv':
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        return get_loraga_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
                                train_dataloader=train_dataloader, device=device, sam_type=sam_type)

    elif model_type == 'lorapro_qv':
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        model, pro_hook = get_lorapro_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
                                            train_dataloader=train_dataloader, device=device, sam_type=sam_type)
        model._lorapro_hook = pro_hook  # 挂到模型上，训练循环中调用
        return model

    elif model_type == 'moelora_qv':
        expert_type = getattr(args, 'moe_expert_type', 'conv')
        return get_moelora_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
                                 num_experts=3, kernel_sizes=[3, 5, 7], sam_type=sam_type,
                                 expert_type=expert_type)

    elif model_type == 'moeloraplus_qv':
        args.use_loraplus_optim = True  # 强制启用 LoRA+ 优化器
        expert_type = getattr(args, 'moe_expert_type', 'conv')
        return get_moeloraplus_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
                                     num_experts=3, kernel_sizes=[3, 5, 7], sam_type=sam_type,
                                     expert_type=expert_type)

    elif model_type in ['lora_encoder', 'lora_decoder', 'adalora_encoder', 'dora_qv_encoder', 'lokr_qv_encoder', 'sam_fully', 'sam_decoder']:
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")

        if model_type == 'adalora_encoder':
            if train_dataloader is None:
                raise ValueError("train_dataloader must be provided for 'adalora_encoder' type.")
            ada_target_r = lora_rank
            ada_init_r = max(int(ada_target_r * 1.5), ada_target_r)
            total_step = args.num_epochs * len(train_dataloader)
            return get_hf_adalora_model(
                model=hgsam_model,
                total_step=total_step,
                target_part='vision_encoder',
                target_r=ada_target_r,
                init_r=ada_init_r,
                lora_alpha=lora_alpha,
            )

        elif model_type == 'lora_encoder':
            return get_hf_lora_model(hgsam_model, lora_rank=lora_rank, lora_alpha=lora_alpha,
                                     lora_dropout=lora_dropout, target_part='vision_encoder')

        elif model_type == 'lora_decoder':
            return get_hf_lora_model(hgsam_model, lora_rank=lora_rank, lora_alpha=lora_alpha,
                                     lora_dropout=lora_dropout, target_part='mask_decoder')

        elif model_type == 'dora_qv_encoder':
            if getattr(args, 'save_custom_lora', False):
                raise ValueError(
                    "dora_qv_encoder only supports Hugging Face PEFT save/load; "
                    "please disable --save_custom_lora."
                )
            args.save_hf_format = True
            return get_hf_dora_qv_model(
                hgsam_model,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_part='vision_encoder',
            )

        elif model_type == 'lokr_qv_encoder':
            if getattr(args, 'save_custom_lora', False):
                raise ValueError(
                    "lokr_qv_encoder only supports Hugging Face PEFT save/load; "
                    "please disable --save_custom_lora."
                )
            args.save_hf_format = True
            return get_hf_lokr_qv_model(
                hgsam_model,
                lokr_rank=lora_rank,
                lokr_alpha=lora_alpha,
                rank_dropout=0.0,
                module_dropout=0.0,
                target_part='vision_encoder',
            )

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

def _select_best_mask(pred_masks, iou_scores):
    """从 multimask 输出中选取 model 预测 IoU 最高的 mask。
    Args:
        pred_masks: [B, 1, 3, H, W] (squeeze point_batch 前) 或 [B, 3, H, W]
        iou_scores: [B, 1, 3] 或 [B, 3]
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
    处理severstal数据集单个批次的数据，执行前向传播和损失计算。
    """
    images = batch["image"].to(device)
    ground_truth_masks = batch["mask"].unsqueeze(1).float().to(device)      # [b, 256, 1600] ----unsqueeze---> [b, 1, 256, 1600]
    if auto_seg:
        bboxes = None    #自动分割 TODO auto_seg应该是用grid prompt
    else:
        bboxes = batch["bbox"].unsqueeze(1).to(device)   #box prompt

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        outputs = model(pixel_values=images, input_boxes=bboxes, multimask_output=multimask)
        if multimask:
            predicted_masks = _select_best_mask(outputs.pred_masks, outputs.iou_scores)  # [B, 1, 256, 256]
        else:
            predicted_masks = outputs.pred_masks.squeeze(1)  # [B, 1, 256, 256]

        orig_h, orig_w = (256, 1600)
        crop_y_start, crop_x_start, crop_y_end, crop_x_end = offset_info

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

def _process_batch(batch, model, loss_fn, device, use_amp, auto_seg = False, offset_info = None, multimask=False):
    """
    处理单个批次的数据，执行前向传播和损失计算。
    此函数可用于sd900, magnetic, neu,训练和验证。
    """
    images = batch["image"].to(device)
    ground_truth_masks = batch["mask"].unsqueeze(1).float().to(device)
    if auto_seg:
        bboxes = None    #TODO 有待完善，auto_seg 应该是 grid point prompt
    else:
        bboxes = batch["bbox"].unsqueeze(1).to(device)   #box prompt

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        outputs = model(pixel_values=images, input_boxes=bboxes, multimask_output=multimask)
        if multimask:
            predicted_masks = _select_best_mask(outputs.pred_masks, outputs.iou_scores)  # [B, 1, 256, 256]
        else:
            predicted_masks = outputs.pred_masks.squeeze(1)  # [B, 1, 256, 256]

        # 将gt进行下采样到(256, 256)，在256x256的低分辨率空间计算损失，效率更高
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
                             num_points=1,
                             multimask=False):
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
            # 最后一轮：保留梯度，计算loss
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
    nanoGPT 风格的数据预取：依赖 DataLoader pin_memory=True + non_blocking=True，
    由 CUDA 内部 copy stream 完成 H2D 传输与计算的重叠，无需手动管理 stream / record_stream。
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
                     procees_batch_fn,
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
    执行一个完整的训练 epoch。
    mfu_tracker: 可选，传入后每个 batch 会更新 MFU 并在 tqdm 后缀中显示。
    master_process: DDP 模式下只在主进程显示进度条。
    global_step: 当前全局训练步数，用于 AdaLoRA 的 update_and_allocate。
    """
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0
    # 训练阶段默认不计算 HD95：CPU 上的 EDT 既慢又占内存，
    # 容易把 DataLoader worker 拖到 OOM 被 SIGKILL。需要时可通过 --train_hd95 打开。
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0) if compute_hd95 else None
    use_amp = scaler is not None

    pbar = tqdm(CUDAPrefetcher(dataloader, device), desc="Training", total=len(dataloader), disable=not master_process)
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # MFU 计时起点：仅当 MFU 启用时 sync 一次确保上一步 GPU 工作已完成
        if mfu_tracker is not None:
            torch.cuda.synchronize()
            t_step_start = time.time()

        loss, pred_masks, gt = procees_batch_fn(batch,
                                                model,
                                                loss_fn,
                                                device,
                                                use_amp,
                                                auto_seg=auto_seg,
                                                offset_info=offset_info,
                                                multimask=multimask)

        # 获取 LoRA-Pro hook（如果存在）
        _raw = getattr(model, 'module', model)  # DDP unwrap
        _pro_hook = getattr(_raw, '_lorapro_hook', None)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
                )
            elif _pro_hook is not None:
                scaler.unscale_(optimizer)
            if _pro_hook is not None:
                _pro_hook.replace_gradients()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
                )
            if _pro_hook is not None:
                _pro_hook.replace_gradients()
            optimizer.step()

        # AdaLoRA rank 分配：必须在 optimizer.step() 之后、scheduler.step() 之前调用
        if hasattr(_raw, 'update_and_allocate'):
            _raw.update_and_allocate(global_step)
        global_step += 1

        # MFU 更新：sync 一次取 step 结束时间，dt 覆盖完整的 forward+backward+optimizer
        if mfu_tracker is not None:
            torch.cuda.synchronize()
            mfu_tracker.update(time.time() - t_step_start)
            pbar.set_postfix_str(mfu_tracker.status())

        scheduler.step()  # 学习率调度，默认根据每个batch更新一次，而不是一个epoch完之后更新一次

        total_loss += loss.item()
        with torch.no_grad():
            total_dice += compute_dice_score(pred_masks, gt)
            total_iou += compute_iou_score(pred_masks, gt)

            if hd95_metric is not None:
                hd95_metric(y_pred=(torch.sigmoid(pred_masks) > 0.5).float().detach().cpu(), y=gt.detach().cpu())

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    if hd95_metric is not None:
        avg_hd95 = hd95_metric.aggregate().item()
        hd95_metric.reset()
    else:
        avg_hd95 = float('nan')

    avg_mfu = mfu_tracker.mfu if mfu_tracker is not None else -1.0
    return avg_loss, avg_dice, avg_iou, avg_hd95, avg_mfu, global_step

def _evaluate(model,
              dataloader,
              loss_fn,
              procees_batch_fn,
              device, scaler,
              auto_seg = False,
              offset_info = None,
              master_process: bool = True,
              multimask: bool = False,
              ddp: bool = False):
    """
    评估模型。

    Args:
        master_process: DDP 模式下只在主进程显示进度条。
        ddp: 是否在 DDP 模式下运行。开启后会在结尾用 all_reduce 把各 rank 的
             loss/dice/iou/hd95 汇总到全局平均。要求 dataloader 使用
             DistributedSampler(shuffle=False)，保证每 rank batch 数相同。
    """
    model.eval()
    # 用 GPU tensor 累积，便于结尾做一次 all_reduce
    total_loss = torch.zeros(1, dtype=torch.float32, device=device)
    total_dice = torch.zeros(1, dtype=torch.float32, device=device)
    total_iou = torch.zeros(1, dtype=torch.float32, device=device)
    num_batches = 0
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0)
    use_amp = scaler is not None

    with torch.no_grad():
        for batch in tqdm(CUDAPrefetcher(dataloader, device), desc="Evaluating", total=len(dataloader), disable=not master_process):
            loss, pred_masks, gt = procees_batch_fn(batch,
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

    # HD95：先在本 rank 内 aggregate，得到本地样本均值；
    # 乘以本 rank batch 数后汇总，再除以全局 batch 数，等价于跨 rank 的加权平均。
    # 前提：val/test 的 DistributedSampler 必须使用 drop_last=True，
    # 否则末尾 padding 重复样本会被双重计入 → 跨 rank 指标与单卡不一致。
    # （drop_last=True 时每 rank 都丢掉相同数量的尾部样本，batch 数严格相等。）
    local_hd95 = hd95_metric.aggregate().item()
    hd95_metric.reset()

    # 打包所有标量到一个 tensor，单次 all_reduce 最省通信
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
    一个通用的微调训练引擎，适用于所有数据集。支持单卡和 DDP 多卡训练。
    Args:
        process_batch_fn (function): 用于处理数据批次的函数。应为 `_process_batch` 或 `_process_batch_severstal`。
        ddp_info (dict, optional): DDP 配置字典，由 setup_ddp() 返回。None 时退化为单卡训练。
        train_sampler (DistributedSampler, optional): DDP 模式下的训练数据采样器，每个 epoch 需调用 set_epoch。
    """
    # ---------- DDP 参数解析（仿照 nanoGPT）----------
    ddp = ddp_info is not None and ddp_info.get('ddp', False)
    master_process = ddp_info['master_process'] if ddp else True
    ddp_local_rank = ddp_info['local_rank'] if ddp else 0

    if master_process:
        print_trainable_parameters(model)
        print(f"using device {device}")
        if ddp:
            print(f"DDP 已启用: world_size={ddp_info['world_size']}, rank={ddp_info['rank']}, local_rank={ddp_local_rank}")

    if master_process:
        for name, param in model.named_parameters():
            if 'lora' in name and param.requires_grad:
                print(name, param.shape)                # lora系列模型的插入位置信息debug

    shanghai_tz = pytz.timezone('Asia/Shanghai')                                # 设置时区为亚洲/上海
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")       # 获取当前时间戳

    swanlab_run = None
    if hyperparameters.get('use_swanlab', False) and master_process:  # DDP: 只在主进程初始化 swanlab
        swanlab_run = swanlab.init(
            project=hyperparameters.get('swanlab_project', 'please name your swanlab project'),
            experiment_name=f"{hyperparameters.get('task_name', 'please name your experiment name')}_{start_timestamp}",
            config=hyperparameters,     # 自动记录所有超参数
        )

    SAVE_HUGGINGFACE_PRETRAINED_MODEL = hyperparameters['save_hf_format']
    save_lora_only = hyperparameters['save_custom_lora']
    if save_lora_only or SAVE_HUGGINGFACE_PRETRAINED_MODEL:
        log_name = hyperparameters["ft_type"] + "_rank_" + str(hyperparameters["lora_rank"])
    else:
        log_name = hyperparameters["ft_type"]

    if master_process:
        print(f"权重保存格式: hugging face格式:{SAVE_HUGGINGFACE_PRETRAINED_MODEL} || lora格式:{save_lora_only}")

    # 从超参数中获取值
    # 使用用户指定的全局 LR，不做自动缩放；用户应根据等效 batch size 自行设定
    lr = hyperparameters['learning_rate']
    wd = hyperparameters['weight_decay']
    num_epochs = hyperparameters['num_epochs']
    warmup_ratio = hyperparameters['warmup_ratio']
    patience = hyperparameters['patience']
    min_delta = hyperparameters['min_delta']
    use_loraplus_optim = hyperparameters.get('use_loraplus_optim', False)
    lora_plus_lr_ratio = hyperparameters.get('lora_plus_lr_ratio', 16.0)
    use_early_stop = not hyperparameters.get('disable_early_stop', False)

    model.to(device)

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
            if master_process:
                print(f"Using LoRA+ optimizer param groups (ratio={lora_plus_lr_ratio}).")
        else:
            if master_process:
                print("LoRA+ optimizer requested but no compatible modules found; fallback to standard AdamW.")

    if optimizer is None:
        optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=wd)

    if master_process:
        debug_print_optimizer_param_groups(optimizer)
    # loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    # BF16 指数范围与 FP32 相同，不会产生梯度上溢，无需 GradScaler
    _use_bf16 = torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda', enabled=not _use_bf16)  # BF16 时禁用，FP16 时启用

    # 学习率调度策略 
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)

    cosine_scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    # MFU 估算器：基于 SAM ViT-B，RTX 5090 BF16 峰值算力
    sam_type = hyperparameters.get('sam_type', 'sam_base')
    mfu_estimator = SAMMFUEstimator(sam_type=sam_type, gpu_type='rtx5090')
    if master_process:
        print(mfu_estimator.summary())

    history = {
        "train_loss": [], "train_dice": [], "train_iou": [], "train_hd95": [],
        "val_loss": [], "val_dice": [], "val_iou": [], "val_hd95": [],
        "train_mfu": [],
    }
    best_val_dicescore = 0.0
    no_improve_epochs = 0
    best_epoch = -1
    best_model_path = None
    global_step = 0  # 用于 AdaLoRA update_and_allocate

    if master_process:
        if use_early_stop:
            print(f"早停已启用 (patience={patience}, min_delta={min_delta}).")
        else:
            print("早停已关闭：将运行完整训练轮次，同时仍跟踪最佳指标以保存模型。")

    # TF32：在 matmul 和 cudnn 上允许 TF32，在几乎不损失精度的前提下提升吞吐
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # AdaLoRA 在 update_and_allocate 时对参数做 in-place SVD 截断，与静态图不兼容，必须跳过
    _ft_type = hyperparameters.get('ft_type', '')
    _is_adalora = _ft_type == 'adalora_encoder'
    # 只有 sam_fully 的 prompt_encoder 存在 requires_grad=True 但 forward 不触达的参数
    # （box-only prompt 路径下 point/mask embedding 未被激活），需要 find_unused_parameters=True。
    needs_find_unused = _ft_type == 'sam_fully'
    use_compile = (
        not hyperparameters.get('no_compile', False)
        and not _is_adalora
    )

    # ---------- DDP 包装 ----------
    if ddp:
        model = DDP(
            model,
            device_ids=[ddp_local_rank],
            find_unused_parameters=needs_find_unused,
        )
    # raw_model 在 compile 之前捕获：torch.compile 只包装 forward，不影响底层参数和 state_dict，
    # 因此保存权重时直接用 raw_model.state_dict()，无需在保存前还原到未编译版本。
    raw_model = model.module if ddp else model

    # ---------- torch.compile（DDP 之后，仅 vision_encoder）----------
    # mask_decoder 含条件分支，compile 会导致图断裂；vision_encoder 占 ~95% 计算量，收益最大。
    # mode='default'：避免 reduce-overhead 开启 CUDA Graph 带来的严格静态形状限制。
    # raw_model 已在上方固定，子模块替换不影响 state_dict，保存权重无需还原。
    if use_compile:
        try:
            raw_model.vision_encoder = torch.compile(raw_model.vision_encoder, mode='default')
            if master_process:
                print(f"torch.compile 已启用 ({'DDP + ' if ddp else ''}仅 vision_encoder, mode='default')")
        except Exception as e:
            if master_process:
                print(f"torch.compile 不可用，跳过: {e}")
            use_compile = False
    else:
        if master_process:
            if _is_adalora:
                print("torch.compile 已禁用 (AdaLoRA 与静态图不兼容)")
            else:
                print("torch.compile 已禁用 (--no_compile)")

    model.train()

    grad_clip = hyperparameters.get('grad_clip', 1.0)
    if master_process:
        print(f"梯度裁剪: {'max_norm=' + str(grad_clip) if grad_clip > 0 else '已禁用 (grad_clip=0)'}")

    use_multimask = hyperparameters.get('multimask', False)
    if master_process and use_multimask:
        print("multimask_output=True + best IoU selection 已启用")

    train_compute_hd95 = hyperparameters.get('train_hd95', False)
    if master_process:
        print(f"训练阶段 HD95 计算: {'启用' if train_compute_hd95 else '关闭 (默认)'}")

    # severstal需要计算 offset_info
    offset_info = None
    if process_batch_fn == _process_batch_severstal:
        if master_process:
            print("Severstal-specific processing enabled. Calculating offset info.")
        offset_info = severstal_get_offset()

    # 将 process_batch_fn 传递给训练周期函数
    for epoch in range(num_epochs):
        if master_process:
            print(f"--- Epoch {epoch + 1}/{num_epochs} ---")

        # DDP: 每个 epoch 设置 sampler 的 epoch，保证各进程数据的 shuffle 不同
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 每个 epoch 重新创建 tracker，EMA 从头累积
        single_batch_size = hyperparameters.get('batch_size', 4)
        world_size = ddp_info['world_size'] if ddp else 1
        global_batch_size = single_batch_size * world_size
        mfu_tracker = MFUTracker(mfu_estimator, batch_size=global_batch_size, ema_alpha=0.9)

        train_loss, train_dice, train_iou, train_hd95, train_mfu, global_step = _train_one_epoch(
            model, train_dataloader, optimizer,
            cosine_scheduler, loss_fn, process_batch_fn, scaler, device,
            auto_seg=auto_seg, offset_info=offset_info, mfu_tracker=mfu_tracker,
            master_process=master_process, multimask=use_multimask,
            global_step=global_step, compute_hd95=train_compute_hd95,
            grad_clip=grad_clip)

        if master_process:
            mfu_str = f"{train_mfu*100:.2f}%" if train_mfu >= 0 else "N/A"
            print(f"Training loss: {train_loss:.4f}, train dice: {train_dice:.4f}, "
                  f"train iou: {train_iou:.4f}, train hd95: {train_hd95:.4f}, MFU: {mfu_str}")

            for i, param_group in enumerate(optimizer.param_groups):
                print(f"Current learning rate (param_group {i}): {param_group['lr']:.6e}")

        val_loss, val_dice, val_iou, val_hd95 = _evaluate(model, val_dataloader, loss_fn, process_batch_fn,
                                                      device, scaler, auto_seg = auto_seg, offset_info=offset_info,
                                                      master_process=master_process, multimask=use_multimask,
                                                      ddp=ddp)
        if master_process:
            print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val HD95: {val_hd95:.4f}')

        # MoE gate 诊断：在终端打印门控分布摘要
        if master_process:
            moe_gate_stats = collect_moe_gate_stats(model)
            if moe_gate_stats:
                entropy_vals = [v for k, v in moe_gate_stats.items() if 'entropy' in k]
                max_prob_vals = [v for k, v in moe_gate_stats.items() if 'max_prob' in k]
                if entropy_vals:
                    print(f'MoE Gate: avg_entropy={sum(entropy_vals)/len(entropy_vals):.4f} (max=1.099), '
                          f'avg_max_prob={sum(max_prob_vals)/len(max_prob_vals):.4f}')

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["train_hd95"].append(train_hd95)
        history["train_mfu"].append(round(train_mfu * 100, 4) if train_mfu >= 0 else None)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_hd95"].append(val_hd95)

        if swanlab_run:
            log_dict = {
                "train/loss": train_loss,
                "train/dice": train_dice,
                "train/iou": train_iou,
                "train/hd95": train_hd95,
                "val/loss": val_loss,
                "val/dice": val_dice,
                "val/iou": val_iou,
                "val/hd95": val_hd95,
                "learning_rate": optimizer.param_groups[0]['lr'],
            }
            if train_mfu >= 0:
                log_dict["train/mfu_pct"] = train_mfu * 100
            # MoE gate 诊断：收集门控权重分布并记录到 swanlab
            moe_stats = collect_moe_gate_stats(model)
            if moe_stats:
                log_dict.update({f"moe/{k}": v for k, v in moe_stats.items()})
            swanlab.log(log_dict, step=epoch+1)

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
            # DDP: 只在主进程保存模型，避免多进程同时写文件冲突
            if master_process:
                print(f"验证 dice 改善到 {best_val_dicescore:.4f}, 保存模型...")
                # raw_model 在 compile 之前捕获，state_dict() 直接可用，无需还原。
                best_model_path = save_model( hyperparameters=hyperparameters,
                                                start_timestamp = start_timestamp,
                                                model=raw_model,
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
                if master_process:
                    print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
            else:
                if master_process:
                    print("验证 dice 未改善（未启用早停），继续完整训练。")

        if master_process:
            output_data = save_training_logs( hyperparameters = hyperparameters,
                                                results = history,
                                                epoch = epoch+1,
                                                start_timestamp = start_timestamp,
                                                result_name = log_name,
                                                target_dir = save_dir)
        if use_early_stop and no_improve_epochs >= patience:
            if master_process:
                print(f"验证指标连续 {patience} 个 epoch 无改善，提前停止训练。")
            break

    # ---------- 最终评估：所有 rank 都执行（torchrun 要求所有进程一起退出）----------
    if master_process:
        print("\n--- Training finished. Starting final evaluation with the best model. ---")

    if not best_model_path:
        if master_process:
            print("Warning: No best model was saved. Final evaluation will be on the last state of the model.")
        loaded_model = raw_model
    else:
        if master_process:
            print(f"Loading best model from: {best_model_path}")
        if SAVE_HUGGINGFACE_PRETRAINED_MODEL:
            try:
                config = PeftConfig.from_pretrained(best_model_path)
                base_model_path = config.base_model_name_or_path
                fresh_base = SamModel.from_pretrained(base_model_path)
                fresh_base = prepare_base_model_for_hf_adapter_loading(fresh_base, _ft_type)

                loaded_model = PeftModel.from_pretrained(fresh_base, best_model_path)
                loaded_model.to(device)
            except Exception as e:
                if master_process:
                    print(f"Error loading Hugging Face PEFT model: {e}")
                loaded_model = None
        elif save_lora_only:
            # 需要从最佳 epoch 的保存文件中读取 lora 参数，覆盖掉当前的过拟合参数
            lora_state_dict = torch.load(best_model_path, map_location=device)
            raw_model.load_state_dict(lora_state_dict, strict=False)

            loaded_model = raw_model
            loaded_model.eval()
        else:
            checkpoint = torch.load(best_model_path, map_location=device)
            raw_model.load_state_dict(checkpoint['model_state_dict'])
            loaded_model = raw_model

    # 所有 rank 都跑评估（只有 master 显示进度条和打印结果）
    final_test_loss, final_test_dice, final_test_iou, final_test_hd95 = _evaluate(
        loaded_model, test_dataloader, loss_fn, process_batch_fn, device, scaler,
        auto_seg=auto_seg, offset_info=offset_info, master_process=master_process,
        multimask=use_multimask, ddp=ddp)

    if master_process:
        print(f'Final Test Set Evaluation: Loss: {final_test_loss:.4f}, Dice: {final_test_dice:.4f}, IoU: {final_test_iou:.4f}, HD95: {final_test_hd95:.4f}')
        history["final_test_metrics"]={"loss": final_test_loss, "dice": final_test_dice, "iou": final_test_iou, "hd95": final_test_hd95}

        if swanlab_run:
            swanlab.log({"test/test_dice": final_test_dice,
                         "test/test_iou": final_test_iou,
                         "test/test_hd95": final_test_hd95})
            swanlab.finish()

        save_training_logs(
            hyperparameters=hyperparameters, results=history, epoch=output_data["last_completed_epoch"],
            start_timestamp=start_timestamp, result_name=log_name, target_dir=save_dir,
        )
        print("--- Training finished ---")

    return history, model

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
                hd95_metric(y_pred=pred_masks_binary.detach().cpu(), y=gt_masks.detach().cpu())

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
                base_model = SamModel.from_pretrained(base_model_path)
                base_model = prepare_base_model_for_hf_adapter_loading(base_model, args.ft_type).to(device)
                
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
