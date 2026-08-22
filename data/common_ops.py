import os
from typing import List, Sequence, Tuple

import random
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


def letterbox_image_np(image: np.ndarray, size: Sequence[int]) -> np.ndarray:
    """
    对 RGB 图像进行等比例缩放并居中填充。

    参数:
        image: np.ndarray, shape=(H, W, C)
        size: (target_width, target_height)
    """
    ih, iw, ic = image.shape
    target_w, target_h = size
    scale = min(target_w / iw, target_h / ih)
    new_w = int(iw * scale)
    new_h = int(ih * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    letterboxed_image = np.full((target_h, target_w, ic), 128, dtype=image.dtype)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    letterboxed_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return letterboxed_image


def letterbox_mask_np(mask: np.ndarray, size: Sequence[int]) -> np.ndarray:
    """
    对单通道 mask 进行等比例缩放并居中填充。

    参数:
        mask: np.ndarray, shape=(H, W)
        size: (target_width, target_height)
    """
    h, w = mask.shape
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    letterbox_result = np.zeros((target_h, target_w), dtype=mask.dtype)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    letterbox_result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = mask_resized
    return letterbox_result


def build_binary_mask(mask: np.ndarray, threshold: float = 0) -> np.ndarray:
    """
    将 mask 二值化并统一为 float32。
    """
    return (mask > threshold).astype(np.float32)


def build_stratify_keys(mask_dir: str, mask_files: List[str]) -> List[str]:
    """
    扫描 mask 文件，按“非背景标签组合”构建分层键。
    """
    stratify_keys = []
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask_np = np.array(Image.open(mask_path))
        present_labels = sorted(label for label in np.unique(mask_np) if label > 0)
        key = "_".join(map(str, present_labels)) if present_labels else "background"
        stratify_keys.append(key)
    return stratify_keys


def split_stratified_6_2_2(labels: Sequence[int], seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对标签做 6:2:2 的分层划分，返回 train/val/test 的绝对索引。
    """
    labels_np = np.asarray(labels)
    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_indices, tmp_indices = next(split_1.split(np.zeros(len(labels_np)), labels_np))

    tmp_labels = labels_np[tmp_indices]
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel_indices, test_rel_indices = next(split_2.split(np.zeros(len(tmp_labels)), tmp_labels))

    val_indices = tmp_indices[val_rel_indices]
    test_indices = tmp_indices[test_rel_indices]
    return train_indices, val_indices, test_indices

def debug_dataset_info(train_dataset, val_dataset, test_dataset):
    """
    通用数据集调试信息打印，本项目所有 Dataset 均可直接传入。

    不硬编码任何字段名，而是从随机样本自省结构：
        - dict 样本（SAM 微调）按键遍历，字段多寡不限，
          因此 create_neu_dataset_stratified / sd900_finetune_create_dataset /
          create_magnetic_dataset / create_dataset_with_prompt 四者通用，
          severstal 多出来的 letterboxed_mask 也会自动打印；
        - tuple/list 样本（baseline）按位置编号 [0] [1] [2] 遍历。

    参数:
        train_dataset / val_dataset / test_dataset: 实现了 __len__ 和 __getitem__ 的 Dataset 或 Subset
    """
    print("-" * 100)
    print(f"{'[数据统计] 训练集图片:':<22}{len(train_dataset)} 张")
    print(f"{'[数据统计] 验证集图片:':<22}{len(val_dataset)} 张")
    print(f"{'[数据统计] 测试集图片:':<22}{len(test_dataset)} 张")

    idx = random.randint(0, len(train_dataset) - 1)
    sample = train_dataset[idx]
    fields = sample.items() if isinstance(sample, dict) else enumerate(sample)

    print("-" * 100)
    print(f"随机样本 idx={idx}    类型={type(sample).__name__}    字段数={len(sample)}")
    print(f"{'field':<18}{'shape':<20}{'dtype':<12}value")
    for key, value in fields:
        # dict 用键名，元组用位置编号
        name = key if isinstance(key, str) else f"[{key}]"
        # image_id / label 这类标量或字符串没有 shape，直接打印类型和值
        if not hasattr(value, "shape"):
            print(f"{name + ':':<18}{type(value).__name__:<20}{value}")
            continue
        # torch CPU tensor 与 np.ndarray 统一按 numpy 处理，避免写两套分支
        arr = np.asarray(value)
        uniques = np.unique(arr)
        # 取值种类少（mask 的 0/1、bbox 的 4 个坐标）时列出全部取值，
        # 种类多（image 这类连续值）时只报范围，避免刷屏
        detail = (f"unique={np.array2string(uniques, precision=1)}" if uniques.size <= 8
                  else f"range=[{arr.min():.4f}, {arr.max():.4f}]    n_unique={uniques.size}")
        print(f"{name + ':':<18}{str(tuple(arr.shape)):<20}{str(arr.dtype):<12}{detail}")
    print("-" * 100)
