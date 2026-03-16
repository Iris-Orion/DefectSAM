import logging
import os
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class BSDS500BoundaryDataset(Dataset):
    """
    用于BSDS500数据集的PyTorch Dataset类，用于边缘检测任务。
    """

    def __init__(self, root_dir, split='train', transforms: A.Compose = None, consensus_threshold=0.5):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.consensus_threshold = consensus_threshold

        self.image_dir = os.path.join(root_dir, 'images', split)
        self.gt_dir = os.path.join(root_dir, 'ground_truth', split)

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.gt_files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.mat')])

        assert len(self.image_files) == len(self.gt_files), "图片和.mat文件数量不匹配"

        for img_file, gt_file in zip(self.image_files, self.gt_files):
            img_base = os.path.splitext(img_file)[0]
            gt_base = os.path.splitext(gt_file)[0]
            assert img_base == gt_base, f"文件名不匹配: {img_file} vs {gt_file}"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_id = self.image_files[idx]
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        mat_data = scipy.io.loadmat(gt_path)

        ground_truths = mat_data['groundTruth'][0]
        num_annotators = 5
        h, w = ground_truths[0]['Boundaries'][0, 0].shape
        summed_boundaries = np.zeros((h, w), dtype=np.float32)

        for i in range(num_annotators):
            annotator_gt = ground_truths[i]['Boundaries'][0, 0].astype(np.float32)
            summed_boundaries += annotator_gt

        ground_truth_prob = ((summed_boundaries / num_annotators) >= 0.2).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image, masks=[ground_truth_prob])
            image_tensor = augmented['image'].to(torch.float32) / 255.0
            transformed_bound_mask = augmented['masks'][0]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
            transformed_bound_mask = torch.from_numpy(ground_truth_prob)

        boundary_tensor = transformed_bound_mask.unsqueeze(0).float()

        return {
            "img_id": img_id,
            "image": image_tensor,
            "mask": boundary_tensor,
        }


def get_bsd500_albumentation_transforms():
    """定义用于BSDS500数据集的Albumentations变换"""
    train_transforms = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ])
    val_test_transforms = A.Compose([
        A.Resize(1024, 1024),
        ToTensorV2(),
    ])
    return train_transforms, val_test_transforms


def bsd500_create_dataset(bsd_root_dir='./data/BSD500'):
    train_transforms, val_test_transforms = get_bsd500_albumentation_transforms()

    train_dataset = BSDS500BoundaryDataset(root_dir=bsd_root_dir, split='train', transforms=train_transforms)
    val_dataset = BSDS500BoundaryDataset(root_dir=bsd_root_dir, split='val', transforms=val_test_transforms)
    test_dataset = BSDS500BoundaryDataset(root_dir=bsd_root_dir, split='test', transforms=val_test_transforms)

    print(f"BSDS500 Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test images.")

    return train_dataset, val_dataset, test_dataset


def log_info_bsd500_dataset(train_dataset, val_dataset, test_dataset):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"BSDS500 Dataset Sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    sample = train_dataset[0]
    sample_id, img, mask = sample['img_id'], sample['image'], sample['mask']
    logging.info(f"Sample from Train Dataset - ID: {sample_id}, Image shape: {img.shape}, Mask shape: {mask.shape}, Mask unique values: {torch.unique(mask)}")


def bsd500_debug_and_visulize(dataset):
    """
    调试和可视化函数，适用于任何返回同样字典结构的Dataset
    """
    idx = random.randint(0, len(dataset) - 1)
    if isinstance(dataset, torch.utils.data.Subset):
        sample = dataset.dataset[dataset.indices[idx]]
    else:
        sample = dataset[idx]

    img_id, img, bd_mask = sample['img_id'], sample['image'], sample['mask']

    print(f"Image tensor shape: {img.shape}, dtype: {img.dtype}, range: {img.min():.2f} - {img.max():.2f}")
    print(f"boundary_mask tensor shape: {bd_mask.shape}, dtype: {bd_mask.dtype}, unique values: {torch.unique(bd_mask)}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if bd_mask.dim() == 3:
        bd_mask = bd_mask.squeeze(0)
    plt.imshow(bd_mask, cmap='gray')
    plt.title('Ground Truth Boundary')
    plt.axis('off')

    plt.show()
