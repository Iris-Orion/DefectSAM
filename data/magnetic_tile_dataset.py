import os
import random
from collections import Counter
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.common_ops import letterbox_image_np, letterbox_mask_np, split_stratified_6_2_2
from utils.helper_function import get_bounding_box


class MagneticTileDataset_Baseline(Dataset):
    def __init__(self, image_paths, mask_paths, labels, transforms: A.Compose):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.labels = labels

    def letterbox_image(self, image, size):
        return letterbox_image_np(image, size)

    def letterbox_mask_1ch(self, mask: np.ndarray, size: tuple) -> np.ndarray:
        return letterbox_mask_np(mask, size)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        image_np = np.array(image)
        mask_np = np.array(mask)

        augmented = self.transforms(image=image_np, mask=mask_np)
        aug_img_np = augmented["image"]
        aug_mask_1ch_np = augmented["mask"]

        resize_img_np = self.letterbox_image(aug_img_np, [256, 256])
        resize_mask_np = self.letterbox_mask_1ch(aug_mask_1ch_np, [256, 256])

        resize_img_tensor = torch.from_numpy(resize_img_np).permute(2, 0, 1).float() / 255.0
        resize_mask_tensor = (torch.from_numpy(resize_mask_np) > 0.5).unsqueeze(0).float()

        label = self.labels[idx]
        return resize_img_tensor, resize_mask_tensor, label

    def __len__(self):
        return len(self.image_paths)


class MagneticTileDatasetWithBoxPrompt(Dataset):
    def __init__(self, image_paths, mask_paths, labels, transforms: A.Compose):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.labels = labels

    def letterbox_image(self, image, size):
        return letterbox_image_np(image, size)

    def letterbox_mask_1ch(self, mask: np.ndarray, size: tuple) -> np.ndarray:
        return letterbox_mask_np(mask, size)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        image_np = np.array(image)
        mask_np = np.array(mask)

        augmented = self.transforms(image=image_np, mask=mask_np)
        aug_img_np = augmented["image"]
        aug_mask_1ch_np = augmented["mask"]

        resize_img_np = self.letterbox_image(aug_img_np, [1024, 1024])
        resize_mask_np = self.letterbox_mask_1ch(aug_mask_1ch_np, [1024, 1024])
        bbox = get_bounding_box(resize_mask_np)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        resize_img_tensor = torch.from_numpy(resize_img_np).permute(2, 0, 1).float() / 255.0
        resize_mask_tensor = (torch.from_numpy(resize_mask_np) > 0.5).float()

        return {
            "image": resize_img_tensor,
            "mask": resize_mask_tensor,
            "bbox": bbox_tensor.float(),
            "label": self.labels[idx],
        }

    def __len__(self):
        return len(self.image_paths)


# Make function to find classes in target directory
def magtile_find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def magtile_get_all_imgmsk_paths(root_dir):
    classes, class_to_idx = magtile_find_classes(root_dir)
    image_paths = []
    mask_paths = []
    labels = []

    for cls in classes:
        cls_dir = os.path.join(root_dir, cls, 'Imgs')
        if not os.path.isdir(cls_dir):
            print(f"Warning: Directory {cls_dir} does not exist. Skipping class {cls}.")
            continue
        for file in sorted(os.listdir(cls_dir)):
            if file.endswith('.jpg'):
                img_path = os.path.join(cls_dir, file)
                mask_filename = os.path.splitext(file)[0] + '.png'
                mask_path = os.path.join(cls_dir, mask_filename)
                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    labels.append(class_to_idx[cls])
                else:
                    print(f"Warning: Mask {mask_path} does not exist for image {img_path}. Skipping.")

    assert len(image_paths) == len(mask_paths) == len(labels), "Mismatch in number of images, masks, and labels."
    return image_paths, mask_paths, labels


def mag_albumentation_transforms():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])
    val_transforms = A.Compose([])
    return train_transforms, val_transforms


def debug_dataset_datatype(dataset: MagneticTileDataset_Baseline):
    idx = random.randint(0, len(dataset) - 1)
    image, mask, label = dataset[idx]
    print(f"Image type: {image.dtype}, Mask type: {mask.dtype}, Label type: {type(label)}")
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Label: {label}")
    print(f"Image 值的范围: {torch.min(image)} ~ {torch.max(image)}, Mask unique: {torch.unique(mask)}")


def get_mag_label_distribution(labels, classes):
    counts = Counter(labels)
    print(f"--- ({len(labels)} samples) ---")
    for class_idx, count in sorted(counts.items()):
        class_name = classes[class_idx]
        percentage = (count / len(labels)) * 100
        print(f"  - Class '{class_name}' (ID: {class_idx}): {count} samples ({percentage:.2f}%)")


def create_mag_dataset_baseline():
    magtile_root_dir = 'data/Magnetic-Tile'
    image_paths, mask_paths, labels = magtile_get_all_imgmsk_paths(magtile_root_dir)
    classes, classes_to_idx = magtile_find_classes(magtile_root_dir)

    train_idx, val_idx, test_idx = split_stratified_6_2_2(labels, seed=42)

    train_image_paths = [image_paths[i] for i in train_idx]
    train_mask_paths = [mask_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_image_paths = [image_paths[i] for i in val_idx]
    val_mask_paths = [mask_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_image_paths = [image_paths[i] for i in test_idx]
    test_mask_paths = [mask_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_transforms, val_transforms = mag_albumentation_transforms()

    train_dataset = MagneticTileDataset_Baseline(train_image_paths, train_mask_paths, train_labels, transforms=train_transforms)
    val_dataset = MagneticTileDataset_Baseline(val_image_paths, val_mask_paths, val_labels, transforms=val_transforms)
    test_dataset = MagneticTileDataset_Baseline(test_image_paths, test_mask_paths, test_labels, transforms=val_transforms)

    print("Full dataset")
    get_mag_label_distribution(labels, classes)
    print("training dataset")
    get_mag_label_distribution(train_labels, classes)
    print("Val dataset")
    get_mag_label_distribution(val_labels, classes)
    print("Test dataset")
    get_mag_label_distribution(test_labels, classes)

    return train_dataset, val_dataset, test_dataset


def create_magnetic_dataset():
    magtile_root_dir = 'data/Magnetic-Tile'
    image_paths, mask_paths, labels = magtile_get_all_imgmsk_paths(magtile_root_dir)
    classes, classes_to_idx = magtile_find_classes(magtile_root_dir)

    train_idx, val_idx, test_idx = split_stratified_6_2_2(labels, seed=42)

    train_image_paths = [image_paths[i] for i in train_idx]
    train_mask_paths = [mask_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_image_paths = [image_paths[i] for i in val_idx]
    val_mask_paths = [mask_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_image_paths = [image_paths[i] for i in test_idx]
    test_mask_paths = [mask_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_transforms, val_transforms = mag_albumentation_transforms()

    print("\n" + "=" * 50)
    print("           Dataset Class Distribution           ")
    print("=" * 50)

    total_counts = Counter(labels)
    print(f"\n--- Original Full Dataset ({len(labels)} samples) ---")
    for class_idx, count in sorted(total_counts.items()):
        class_name = classes[class_idx]
        print(f"  - Class '{class_name}' (ID: {class_idx}): {count} samples")

    train_counts = Counter(train_labels)
    print(f"\n--- Training Set ({len(train_labels)} samples) ---")
    for class_idx, count in sorted(train_counts.items()):
        class_name = classes[class_idx]
        percentage = (count / len(train_labels)) * 100
        print(f"  - Class '{class_name}' (ID: {class_idx}): {count} samples ({percentage:.2f}%)")

    val_counts = Counter(val_labels)
    print(f"\n--- Validation Set ({len(val_labels)} samples) ---")
    for class_idx, count in sorted(val_counts.items()):
        class_name = classes[class_idx]
        percentage = (count / len(val_labels)) * 100
        print(f"  - Class '{class_name}' (ID: {class_idx}): {count} samples ({percentage:.2f}%)")

    test_counts = Counter(test_labels)
    print(f"\n--- Test Set ({len(test_labels)} samples) ---")
    for class_idx, count in sorted(test_counts.items()):
        class_name = classes[class_idx]
        percentage = (count / len(test_labels)) * 100
        print(f"  - Class '{class_name}' (ID: {class_idx}): {count} samples ({percentage:.2f}%)")

    print("\n" + "=" * 50 + "\n")

    train_dataset = MagneticTileDatasetWithBoxPrompt(train_image_paths, train_mask_paths, train_labels, transforms=train_transforms)
    val_dataset = MagneticTileDatasetWithBoxPrompt(val_image_paths, val_mask_paths, val_labels, transforms=val_transforms)
    test_dataset = MagneticTileDatasetWithBoxPrompt(test_image_paths, test_mask_paths, test_labels, transforms=val_transforms)

    return train_dataset, val_dataset, test_dataset
