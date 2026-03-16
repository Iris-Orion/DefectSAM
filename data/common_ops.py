import os
from typing import List, Sequence, Tuple

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
