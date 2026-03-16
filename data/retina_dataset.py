import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, Subset

from data.common_ops import build_binary_mask
from data.sam_dataset_base import SegDatasetForFinetune
from utils.helper_function import get_bounding_box


class Retina_Dataset_Bsl(Dataset):
    """
    Retina dataset for baseline assesment
    """

    def __init__(self, images_dir, annotations_dir, transforms: A.Compose):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.png')])

        if len(self.image_files) != len(self.mask_files):
            raise RuntimeError(f"图像数量({len(self.image_files)})与掩码数量({len(self.mask_files)})不一致！")

        for img_f, mask_f in zip(self.image_files, self.mask_files):
            if os.path.splitext(img_f)[0] != os.path.splitext(mask_f)[0]:
                raise ValueError(f"检测到不匹配的文件对：\nImg: {img_f}\nMask: {mask_f}\n请检查数据集排序。")

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
        if self.transforms:
            augmented = self.transforms(image=image_np, mask=mask_np)
            image = augmented['image'] / 255.0
            mask = (augmented['mask']).unsqueeze(0)
        return image, mask, image_id


class Retina_Dataset_ft(SegDatasetForFinetune):
    """Retina dataset for fine-tuning"""

    def __init__(self, images_dir, annotations_dir, transforms: A.Compose):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.png')])

        if len(self.image_files) != len(self.mask_files):
            raise RuntimeError(f"图像数量({len(self.image_files)})与掩码数量({len(self.mask_files)})不一致！")

        for img_f, mask_f in zip(self.image_files, self.mask_files):
            if os.path.splitext(img_f)[0] != os.path.splitext(mask_f)[0]:
                raise ValueError(f"检测到不匹配的文件对：\nImg: {img_f}\nMask: {mask_f}\n请检查数据集排序。")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.annotations_dir, self.mask_files[idx])

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image_id = self.image_files[idx]
        mask_id = self.mask_files[idx]
        if image_id != mask_id:
            raise ValueError(f"图像ID和掩码ID不匹配: {image_id} vs {mask_id}")

        resized_img = self.letterbox_imagenp(image, [1024, 1024])
        resized_mask = self.letterbox_mask_1ch(mask, [1024, 1024])

        bbox = get_bounding_box(resized_mask)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        if self.transforms:
            augmented = self.transforms(image=resized_img, mask=resized_mask)
            image_tensor = augmented['image'] / 255.0
            mask_tensor = (augmented['mask']) / 255.0
            mask_tensor = (mask_tensor > 0).float()

        return {
            "image": image_tensor.float(),
            "mask": mask_tensor.float(),
            "bbox": bbox_tensor.float(),
            "image_id": image_id,
        }


def general_albumentation_transforms():
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


def general_albumentation_transforms_for_finetune():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        ToTensorV2(),
    ])
    return train_transforms, val_transforms


def create_retina_dataset_baseline():
    train_transforms, test_transforms = general_albumentation_transforms()

    retina_train_images_dir = 'data/Retina_Blood_Vessel/train/image'
    retina_train_annotations_dir = 'data/Retina_Blood_Vessel/train/mask'

    retina_test_images_dir = 'data/Retina_Blood_Vessel/test/image'
    retina_test_annotations_dir = 'data/Retina_Blood_Vessel/test/mask'

    train_dataset_with_aug = Retina_Dataset_Bsl(
        retina_train_images_dir,
        retina_train_annotations_dir,
        transforms=train_transforms,
    )

    val_dataset_no_aug = Retina_Dataset_Bsl(
        retina_train_images_dir,
        retina_train_annotations_dir,
        transforms=test_transforms,
    )

    dataset_size = len(train_dataset_with_aug)
    indices = list(range(dataset_size))
    split = 20

    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    retina_train_subset = Subset(train_dataset_with_aug, train_indices)
    retina_val_subset = Subset(val_dataset_no_aug, val_indices)

    retina_test_set = Retina_Dataset_Bsl(
        retina_test_images_dir,
        retina_test_annotations_dir,
        transforms=test_transforms,
    )

    print(f"训练集大小: {len(retina_train_subset)}")
    print(f"验证集大小: {len(retina_val_subset)}")
    print(f"测试集大小: {len(retina_test_set)}")

    return retina_train_subset, retina_val_subset, retina_test_set


def create_retina_dataset_ft():
    train_transforms, test_transforms = general_albumentation_transforms_for_finetune()

    retina_train_images_dir = 'data/Retina_Blood_Vessel/train/image'
    retina_train_annotations_dir = 'data/Retina_Blood_Vessel/train/mask'

    retina_test_images_dir = 'data/Retina_Blood_Vessel/test/image'
    retina_test_annotations_dir = 'data/Retina_Blood_Vessel/test/mask'

    train_dataset_with_aug = Retina_Dataset_ft(
        retina_train_images_dir,
        retina_train_annotations_dir,
        transforms=train_transforms,
    )

    val_dataset_no_aug = Retina_Dataset_ft(
        retina_train_images_dir,
        retina_train_annotations_dir,
        transforms=test_transforms,
    )

    dataset_size = len(train_dataset_with_aug)
    indices = list(range(dataset_size))
    split = 20

    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    retina_train_subset = Subset(train_dataset_with_aug, train_indices)
    retina_val_subset = Subset(val_dataset_no_aug, val_indices)

    retina_test_set = Retina_Dataset_ft(
        retina_test_images_dir,
        retina_test_annotations_dir,
        transforms=test_transforms,
    )

    print(f"训练集大小: {len(retina_train_subset)}")
    print(f"验证集大小: {len(retina_val_subset)}")
    print(f"测试集大小: {len(retina_test_set)}")

    return retina_train_subset, retina_val_subset, retina_test_set
