import logging
import os
from collections import Counter

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.common_ops import letterbox_image_np, letterbox_mask_np, split_stratified_6_2_2
from utils.helper_function import get_bounding_box


class SDsaliency900Dataset_BSL(Dataset):
    """无需box prompt的baseline使用"""

    def __init__(self, source_dir, ground_truth_dir, transforms: A.Compose):
        self.source_dir = source_dir
        self.ground_truth_dir = ground_truth_dir
        self.transforms = transforms

        self.image_names = sorted([f for f in os.listdir(source_dir) if f.endswith('.bmp')])
        self.labels = []

        for name in self.image_names:
            if name.startswith('In_'):
                self.labels.append(0)
            elif name.startswith('Pa_'):
                self.labels.append(1)
            elif name.startswith('Sc_'):
                self.labels.append(2)
            else:
                raise ValueError(f'未知的文件命名格式: {name}')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.source_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.ground_truth_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image_np = np.array(image)
        mask_np = np.array(mask)

        if self.transforms:
            augmented = self.transforms(image=image_np, mask=mask_np)
            image = augmented['image'] / 255.0
            mask = (augmented['mask'] / 255.0).unsqueeze(0)
        return image.float(), mask.float(), label


class SDsaliency900Dataset_FT(Dataset):
    """SAM finetune 使用"""

    def __init__(self, source_dir, ground_truth_dir, transforms: A.Compose, is_train=True):
        self.source_dir = source_dir
        self.ground_truth_dir = ground_truth_dir
        self.transforms = transforms
        self.is_train = is_train

        self.image_names = sorted([f for f in os.listdir(source_dir) if f.endswith('.bmp')])
        self.labels = []

        for name in self.image_names:
            if name.startswith('In_'):
                self.labels.append(0)
            elif name.startswith('Pa_'):
                self.labels.append(1)
            elif name.startswith('Sc_'):
                self.labels.append(2)
            else:
                raise ValueError(f'未知的文件命名格式: {name}')

    def letterbox_imagenp(self, image: np.ndarray, size):
        return letterbox_image_np(image, size)

    def letterbox_mask_1ch(self, mask: np.ndarray, size: tuple) -> np.ndarray:
        return letterbox_mask_np(mask, size)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.source_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.ground_truth_dir, mask_name)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        resized_img = self.letterbox_imagenp(image, [1024, 1024])
        resized_mask = self.letterbox_mask_1ch(mask, [1024, 1024])

        if self.transforms:
            augmented = self.transforms(image=resized_img, mask=resized_mask)
            image_tensor = augmented['image'] / 255.0
            mask_tensor = augmented['mask'] / 255.0
        else:
            image_tensor = torch.from_numpy(resized_img.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(resized_mask).float() / 255.0

        # bbox 在数据增强之后从变换后的 mask 上计算，确保与增强后的空间位置一致
        mask_np_for_bbox = (mask_tensor.numpy() * 255).astype(np.uint8)
        if mask_np_for_bbox.ndim == 3:
            mask_np_for_bbox = mask_np_for_bbox[0]
        bbox = get_bounding_box(mask_np_for_bbox, perturb=self.is_train)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return {
            "image": image_tensor.float(),
            "mask": mask_tensor.float(),
            "bbox": bbox_tensor.float(),
            "label": self.labels[idx],
        }


# 定义变换
def get_sd900_albumentation_transforms_baseline():
    train_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.Resize(height=256, width=256),
        ToTensorV2(),
    ])
    return train_transforms, val_transforms


def get_sd900_albumentation_transforms_ft():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        ToTensorV2(),
    ])
    return train_transforms, val_transforms


# 打印类别分布
def get_label_distribution(dataset):
    labels_subset = [dataset.dataset.labels[idx] for idx in dataset.indices]
    label_counts = Counter(labels_subset)
    sorted_counts = dict(sorted(label_counts.items()))
    return sorted_counts


def test_sd900_info(train_dataset: Dataset):
    image, mask, label = train_dataset[0]
    print(f"debug信息: image_names:{train_dataset.dataset.image_names[0]}")
    print(f"debug信息: image.shape: {image.shape}, mask.shape: {mask.shape}, label: {label}")
    print(f"debug信息: image dtype: {image.dtype} || mask dtype: {mask.dtype}")
    print(f"debug信息: image的值的类型: {torch.unique(image)}")
    print(f"debug信息: mask的值的类型: {torch.unique(mask)}")


def get_image_names_from_subset(subset):
    base_dataset = subset.dataset
    indices = subset.indices
    return [base_dataset.image_names[i] for i in indices]


def sd900_bsl_create_dataset():
    sd900_src_dir = './data/sd900/Source Images'
    sd900_gt_dir = './data/sd900/Ground truth'

    full_dataset = SDsaliency900Dataset_BSL(
        source_dir=sd900_src_dir,
        ground_truth_dir=sd900_gt_dir,
        transforms=None,
    )
    labels = np.array(full_dataset.labels)

    train_indices, val_indices, test_indices = split_stratified_6_2_2(labels, seed=42)

    train_transforms, val_test_transforms = get_sd900_albumentation_transforms_baseline()
    train_full_dataset = SDsaliency900Dataset_BSL(source_dir=sd900_src_dir, ground_truth_dir=sd900_gt_dir, transforms=train_transforms)
    val_test_full_dataset = SDsaliency900Dataset_BSL(source_dir=sd900_src_dir, ground_truth_dir=sd900_gt_dir, transforms=val_test_transforms)

    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_test_full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(val_test_full_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset, labels


def sd900_finetune_create_dataset(
    sd900_src_dir='./data/sd900/Source Images',
    sd900_gt_dir='./data/sd900/Ground truth',
    split_seed=42,
):
    full_dataset = SDsaliency900Dataset_FT(source_dir=sd900_src_dir, ground_truth_dir=sd900_gt_dir, transforms=None)

    labels = np.array(full_dataset.labels)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"sd900全数据集的Labels: {labels.tolist()}")

    train_indices, val_indices, test_indices = split_stratified_6_2_2(labels, seed=split_seed)

    train_transforms, val_test_transforms = get_sd900_albumentation_transforms_ft()
    train_full_dataset = SDsaliency900Dataset_FT(source_dir=sd900_src_dir, ground_truth_dir=sd900_gt_dir, transforms=train_transforms, is_train=True)
    val_test_full_dataset = SDsaliency900Dataset_FT(source_dir=sd900_src_dir, ground_truth_dir=sd900_gt_dir, transforms=val_test_transforms, is_train=False)

    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_test_full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(val_test_full_dataset, test_indices)
    return train_dataset, val_dataset, test_dataset


def sd900_finetune_create_dataloader(args, train_dataset, val_dataset, test_dataset):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    return train_loader, val_loader, test_loader


def debug_sd900_img_and_mask():
    img = cv2.cvtColor(cv2.imread('data/sd900/Source Images/In_1.bmp'), cv2.COLOR_BGR2RGB)
    gt = cv2.imread('data/sd900/Ground truth/In_1.png', cv2.IMREAD_GRAYSCALE)
    print(
        f"---------cv2读取img和mask---------\n"
        f"img.shape:{img.shape}, gt.shape:{gt.shape}\n"
        f"img.type:{type(img)}, gt.type:{type(gt)}\n"
        f"img.dtype:{img.dtype}, gt.dtype:{gt.dtype}\n"
        f"img data range: {img.min()} ~ {img.max()}\n"
        f"gt data unique values: {np.unique(gt)}\n"
        f"-----------------------------------"
    )

    img_tr = transforms.ToTensor()(img)
    gt_tr = transforms.ToTensor()(gt)
    print(
        f"---------torchvision转换--------\n"
        f"img_tr.shape:{img_tr.shape}, gt_tr.shape:{gt_tr.shape}\n"
        f"img_tr.type:{type(img_tr)}, gt_tr.type:{type(gt_tr)}\n"
        f"img_tr.dtype:{img_tr.dtype}, gt_tr.dtype:{gt_tr.dtype}\n"
        f"img_tr data range: {img_tr.min()} ~ {img_tr.max()}\n"
        f"gt_tr data unique: {torch.unique(gt_tr)}\n"
        f"-----------------------------------"
    )

    train_aug, _ = get_sd900_albumentation_transforms_ft()
    augmented = train_aug(image=img, mask=gt)
    img_aug = augmented['image']
    mask_aug = augmented['mask']

    print(
        f"---------albumentation 转换 totensorv2并不会归一化需要自己除以255.0并转为float32--------\n"
        f"img_aug.shape:{img_aug.shape}, gt_tr.shape:{mask_aug.shape}\n"
        f"img_aug.type:{type(img_aug)}, gt_tr.type:{type(mask_aug)}\n"
        f"img_aug.dtype:{img_aug.dtype}, gt_tr.dtype:{mask_aug.dtype}\n"
        f"img_aug data range: {img_aug.min()} ~ {img_aug.max()}\n"
        f"mask_aug data unique: {torch.unique(mask_aug)}\n"
        f"-----------------------------------"
    )

    img_aug_float32 = (img_aug / 255.0).float()
    mask_aug_float32 = (mask_aug / 255.0).float()
    print(f"img_aug.shape: {img_aug_float32.shape}, img_aug_float32.dtype: {img_aug_float32.dtype}")
    print(f"mask_aug.shape: {mask_aug_float32.shape}, mask_aug.dtype: {mask_aug_float32.dtype}")
if __name__ == "__main__":
    debug_sd900_img_and_mask()
