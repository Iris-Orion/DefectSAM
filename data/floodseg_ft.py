import os

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from data.common_ops import letterbox_image_np, letterbox_mask_np


class FloodSegDataset(Dataset):
    def __init__(self, root_dir, image_folder="Image", mask_folder="Mask", transforms: A.Compose = None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)
        self.transforms = transforms

        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

        assert len(self.image_files) == len(self.mask_files), "图片和掩码数量不匹配"

    def letterbox_imagenp(self, image, size=(1024, 1024)):
        return letterbox_image_np(image, size)

    def letterbox_mask_1ch(self, mask, size=(1024, 1024)):
        return letterbox_mask_np(mask, size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {img_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法加载掩码: {mask_path}")

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask)

        image_tensor = image_tensor.float() / 255.0
        mask_tensor = (mask_tensor > 0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": 0,
        }


# 定义变换
def get_floodseg_albumentation_transforms():
    train_transforms = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.Resize(1024, 1024),
        ToTensorV2(),
    ])
    return train_transforms, val_transforms


def floodseg_create_dataset(flood_root_dir='./data/FloodSeg', split_seed=42):
    train_transforms, val_test_transforms = get_floodseg_albumentation_transforms()

    train_full_dataset = FloodSegDataset(root_dir=flood_root_dir, transforms=train_transforms)
    val_test_full_dataset = FloodSegDataset(root_dir=flood_root_dir, transforms=val_test_transforms)

    dataset_size = len(train_full_dataset)
    indices = list(range(dataset_size))

    train_size = int(0.6 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(split_seed)
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        indices,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_test_full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(val_test_full_dataset, test_indices)

    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test.")

    return train_dataset, val_dataset, test_dataset
