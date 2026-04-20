import math
import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import matplotlib.pyplot as plt
import numpy as np
import random


def set_device(gpu_idx: int=0):
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Set device: {device}")
    return device

def cleanup_ddp():
    """销毁 DDP 进程组，在训练结束时调用。"""
    if int(os.environ.get('RANK', -1)) != -1:
        destroy_process_group()

def set_seed(seed: int = 42, seed_offset: int = 0):
    """
    设置随机种子以确保实验可重复性。
    DDP 模式下每个进程使用不同的 seed_offset，保证各卡采样数据不同。

    参数:
        seed (int): 随机种子值。默认为42。
        seed_offset (int): 种子偏移量，DDP 模式下传入 rank 值。默认为0。
    """
    actual_seed = seed + seed_offset
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)  # 多 GPU 时同步设置

def get_lr_scheduler(optimizer, warmup_steps, total_steps, eta_min_ratio=0.1):
    """Cosine warmup scheduler，最终学习率衰减到 base_lr * eta_min_ratio。

    Args:
        eta_min_ratio: 最终学习率 = 初始学习率 * eta_min_ratio，默认 0.1
    """
    def lr_lambda(current_step):
        # 1) linear warmup
        if current_step < warmup_steps:
            return (current_step + 1) / (warmup_steps + 1)
        # 2) past total_steps, clamp to min lr
        if current_step > total_steps:
            return eta_min_ratio
        # 3) cosine decay down to eta_min_ratio
        decay_ratio = (current_step - warmup_steps) / (total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return eta_min_ratio + coeff * (1 - eta_min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_bounding_box(ground_truth_map, perturb=True, perturb_range=20):
    """
    从ground_truth中获得bbox。
    输入:ground_truth_map 是一个合并的掩码 : np.array
    参数:
        perturb: 是否对bbox添加随机扰动（训练时True，验证/测试时False）
        perturb_range: 随机扰动的最大像素数，与SAM原始训练策略一致（默认±20px）
    """
    assert type(ground_truth_map) == np.ndarray, "check type"
    if np.sum(ground_truth_map) == 0:
        bbox = [0, 0, 0, 0]
    else:
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        if perturb:
            H, W = ground_truth_map.shape
            x_min = max(0, x_min - np.random.randint(0, perturb_range))
            x_max = min(W, x_max + np.random.randint(0, perturb_range))
            y_min = max(0, y_min - np.random.randint(0, perturb_range))
            y_max = min(H, y_max + np.random.randint(0, perturb_range))

        bbox = [x_min, y_min, x_max, y_max]
    return bbox

