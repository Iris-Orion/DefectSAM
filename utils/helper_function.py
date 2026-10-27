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
    if int(os.environ.get('RANK', -1)) != -1:
        destroy_process_group()

def set_seed(seed: int = 42, seed_offset: int = 0):
    """Seed all RNGs with seed + seed_offset. Under DDP pass the rank as seed_offset so each process samples different data."""
    actual_seed = seed + seed_offset
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)

def get_lr_scheduler(optimizer, warmup_steps, total_steps, eta_min_ratio=0.1):
    """Linear warmup, then cosine decay to base_lr * eta_min_ratio (held there after total_steps)."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return (current_step + 1) / (warmup_steps + 1)
        if current_step > total_steps:
            return eta_min_ratio
        decay_ratio = (current_step - warmup_steps) / (total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return eta_min_ratio + coeff * (1 - eta_min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def use_eager_attn_for_compat():
    """Inference patch for checkpoints trained under transformers 4.47.0.

    From 5.5.0, SamModel.from_pretrained defaults to sdpa attention, which shifts Dice by ~0.002
    versus the 4.47.0 eager implementation. Call at the top of inference scripts that evaluate
    4.47.0 checkpoints so attention matches training. Idempotent.
    """
    from transformers import SamModel
    if getattr(SamModel.from_pretrained, '_eager_compat_patched', False):
        return
    _orig = SamModel.from_pretrained
    def _patched(pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("attn_implementation", "eager")
        return _orig(pretrained_model_name_or_path, *args, **kwargs)
    _patched._eager_compat_patched = True
    SamModel.from_pretrained = _patched


def get_bounding_box(ground_truth_map, perturb=True, perturb_range=20):
    """Return [x_min, y_min, x_max, y_max] of a binary mask, or [0, 0, 0, 0] if it is empty.

    perturb: grow each side outward by a random 0..perturb_range-1 px, mimicking SAM's box noise
    (True for training, False for val/test).
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

