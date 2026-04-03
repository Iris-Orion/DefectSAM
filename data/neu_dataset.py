import os
import random
from collections import Counter

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from data.common_ops import build_binary_mask, build_stratify_keys, letterbox_image_np, letterbox_mask_np
from utils.helper_function import get_bounding_box


class NEU_SEG_Dataset_BSL(Dataset):
    """
    无需box prompt的baseline使用
    """

    def __init__(self, images_dir, annotations_dir, transforms: A.Compose):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.annotations_dir, self.mask_files[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image_id = self.image_files[idx]

        image_np = np.array(image)
        mask_np = np.array(mask)

        # 原数据集将不同缺陷的掩码值分别设为了1-3，但是整个图片是单通道的，所以这里二值化为全为1进行处理
        mask_np = build_binary_mask(mask_np)
        if self.transforms:
            augmented = self.transforms(image=image_np, mask=mask_np)
            image = augmented['image'] / 255.0  # [3, 256, 256]
            mask = (augmented['mask']).unsqueeze(0)  # [1, 256, 256]
        return image, mask, image_id


class NEU_SEG_Dataset(Dataset):
    """SAM finetune 使用"""

    def __init__(self, images_dir, annotations_dir, transforms: A.Compose, is_train=True):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.is_train = is_train
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.png')])

    def letterbox_imagenp(self, image: np.ndarray, size):
        return letterbox_image_np(image, size)

    def letterbox_mask_1ch(self, mask: np.ndarray, size: tuple) -> np.ndarray:
        return letterbox_mask_np(mask, size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.annotations_dir, self.mask_files[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image_id = self.image_files[idx]

        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_np = build_binary_mask(mask_np)

        img_1024 = self.letterbox_imagenp(image_np, [1024, 1024])
        mask_1024 = self.letterbox_mask_1ch(mask_np, [1024, 1024])

        if self.transforms:
            augmented = self.transforms(image=img_1024, mask=mask_1024)
            image_tensor = augmented['image'] / 255.0
            mask_tensor = augmented['mask'].float()

        # bbox 必须在增强之后从变换后的 mask 上计算，否则翻转后 bbox 与 mask 不一致
        mask_np_for_bbox = mask_tensor.squeeze().numpy().astype(np.uint8)
        bbox = get_bounding_box(mask_np_for_bbox, perturb=self.is_train)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return {
            "image": image_tensor.float(),
            "mask": mask_tensor.float(),
            "bbox": bbox_tensor.float(),
            "image_id": image_id,
        }


# 定义变换
def get_neu_albumentation_transforms_baseline():
    train_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ])
    test_transforms = A.Compose([
        A.Resize(height=256, width=256),
        ToTensorV2(),
    ])
    return train_transforms, test_transforms


def get_neu_albumentation_transforms_ft():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ])
    test_transforms = A.Compose([
        ToTensorV2(),
    ])
    return train_transforms, test_transforms


def neu_bsl_create_dataset():
    train_images_dir = 'data/NEU_Seg-main/images/training'
    train_annotations_dir = 'data/NEU_Seg-main/annotations/training'
    test_images_dir = 'data/NEU_Seg-main/images/test'
    test_annotations_dir = 'data/NEU_Seg-main/annotations/test'

    train_transforms, test_transforms = get_neu_albumentation_transforms_baseline()

    all_mask_files = sorted([f for f in os.listdir(train_annotations_dir) if f.endswith('.png')])
    print("正在为分层抽样生成策略键...")
    stratify_keys = build_stratify_keys(train_annotations_dir, all_mask_files)

    print("策略键生成完毕。")
    print(f"原始训练集中的类别组合分布: \n{Counter(stratify_keys)}")

    indices = list(range(len(all_mask_files)))
    val_split_ratio = 0.25
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split_ratio,
        stratify=stratify_keys,
        random_state=42,
    )

    train_keys = [stratify_keys[i] for i in train_indices]
    val_keys = [stratify_keys[i] for i in val_indices]
    print(f"\n划分后训练集的分布: \n{Counter(train_keys)}")
    print(f"\n划分后验证集的分布: \n{Counter(val_keys)}")

    full_dataset_for_train = NEU_SEG_Dataset_BSL(train_images_dir, train_annotations_dir, transforms=train_transforms)
    full_dataset_for_val = NEU_SEG_Dataset_BSL(train_images_dir, train_annotations_dir, transforms=test_transforms)
    test_dataset = NEU_SEG_Dataset_BSL(test_images_dir, test_annotations_dir, transforms=test_transforms)

    train_dataset = Subset(full_dataset_for_train, train_indices)
    val_dataset = Subset(full_dataset_for_val, val_indices)
    return train_dataset, val_dataset, test_dataset


def generate_neu_keys(mask_dir, mask_files):
    return build_stratify_keys(mask_dir, mask_files)


def create_neu_dataset_stratified():
    train_images_dir = 'data/NEU_Seg-main/images/training'
    train_annotations_dir = 'data/NEU_Seg-main/annotations/training'
    test_images_dir = 'data/NEU_Seg-main/images/test'
    test_annotations_dir = 'data/NEU_Seg-main/annotations/test'

    train_transforms, test_transforms = get_neu_albumentation_transforms_ft()

    all_train_mask_files = sorted([f for f in os.listdir(train_annotations_dir) if f.endswith('.png')])
    all_test_mask_files = sorted([f for f in os.listdir(test_annotations_dir) if f.endswith('.png')])

    stratify_keys_train = generate_neu_keys(train_annotations_dir, all_train_mask_files)
    stratify_keys_test = generate_neu_keys(test_annotations_dir, all_test_mask_files)

    print("策略键生成完毕。")
    print(f"原始训练集中的类别组合分布: \n{Counter(stratify_keys_train)}")
    print(f"原始测试集中的类别组合分布: \n{Counter(stratify_keys_test)}")

    full_train_dataset = NEU_SEG_Dataset(train_images_dir, train_annotations_dir, transforms=train_transforms)
    indices = list(range(len(full_train_dataset)))

    val_split_ratio = 0.25
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split_ratio,
        stratify=stratify_keys_train,
        random_state=42,
    )

    print("\n验证划分结果")
    train_keys = [stratify_keys_train[i] for i in train_indices]
    val_keys = [stratify_keys_train[i] for i in val_indices]
    print(f"\n划分后训练集的分布: \n{Counter(train_keys)}")
    print(f"\n划分后验证集的分布: \n{Counter(val_keys)}")

    full_train_dataset = NEU_SEG_Dataset(train_images_dir, train_annotations_dir, transforms=train_transforms, is_train=True)
    full_val_dataset = NEU_SEG_Dataset(train_images_dir, train_annotations_dir, transforms=test_transforms, is_train=False)
    test_dataset = NEU_SEG_Dataset(test_images_dir, test_annotations_dir, transforms=test_transforms, is_train=False)

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)
    return train_dataset, val_dataset, test_dataset


def debug_neu_dataset_info(train_dataset, val_dataset, test_dataset):
    print("=" * 40)
    print(f"[数据统计] 训练集图片: {len(train_dataset)} 张")
    print(f"[数据统计] 验证集图片: {len(val_dataset)} 张")
    print(f"[数据统计] 测试集图片: {len(test_dataset)} 张")
    print("=" * 40)

    idx = random.randint(0, len(train_dataset) - 1)
    sample = train_dataset[idx]
    img, mask, bbox, image_id = sample['image'], sample['mask'], sample['bbox'], sample['image_id']

    print("=" * 40)
    print("debug shapes")
    print(f"img: {img.shape}, mask: {mask.shape}, bbox: {bbox}, image_id: {image_id}")
    print("=" * 40)

    print("=" * 40)
    print("debug data types and data ranges")
    print(f"img: {img.dtype}, mask: {mask.dtype}, bbox: {bbox.dtype}")
    print(f"img: {img.min()} ~ {img.max()}, mask: {mask.unique()}")
    print("=" * 40)
