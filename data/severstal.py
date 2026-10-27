import os
import cv2
import torch
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from data.common_ops import debug_dataset_info


def _empty_mask() -> np.ndarray:
    return np.zeros((256, 1600, 4), dtype=np.uint8)


def _is_no_defect_image(image_df: pd.DataFrame) -> bool:
    return image_df.empty or (image_df['ClassId'] == 0).all()


def build_mask_cache(df, cache_dir="./data/severstal_steel_defect_detection/mask_cache"):
    """Decode every defect image's RLE into a compressed .npz once; existing cache files are skipped."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fnames = df['ImageId'].unique()
    new_count = 0
    skipped_no_defect_count = 0
    existing_count = 0
    for image_id in fnames:
        image_df = df[df['ImageId'] == image_id]
        if _is_no_defect_image(image_df):
            skipped_no_defect_count += 1
            continue

        npz_path = cache_dir / f"{Path(image_id).stem}.npz"
        if npz_path.exists():
            existing_count += 1
            continue
        mask = _rle2mask_raw(image_id, image_df)
        np.savez_compressed(npz_path, mask=mask)
        new_count += 1

    print(
        f"[mask_cache] 新生成 {new_count} 个 uint8 压缩缓存文件，"
        f"跳过无缺陷 {skipped_no_defect_count} 张，"
        f"已存在 {existing_count} 个 → {cache_dir}"
    )


def _rle2mask_raw(image_id, df):
    mask = _empty_mask()

    image_df = df[df['ImageId'] == image_id]
    if _is_no_defect_image(image_df):
        return mask

    for _, row in image_df.iterrows():
        class_id = int(row['ClassId']) - 1  # ClassId 1-4 -> channel 0-3
        rle = row['EncodedPixels']

        if pd.isna(rle):
            continue

        label = rle.split(" ")
        positions = map(int, label[0::2])
        lengths = map(int, label[1::2])

        mask_flat = np.zeros(256 * 1600, dtype=np.uint8)
        for pos, le in zip(positions, lengths):
            mask_flat[pos:(pos+le)] = 1

        mask[:, :, class_id] = mask_flat.reshape((256, 1600), order='F')

    return mask


def rle2mask(image_id, df, cache_dir="./data/severstal_steel_defect_detection/mask_cache"):
    """Return the 4-channel mask of `image_id`, shape (256, 1600, 4), from the .npz cache or by RLE decoding.

    No in-process memory cache: with persistent_workers=True worker memory would grow every epoch until OOM SIGKILL.
    """
    npz_path = Path(cache_dir) / f"{Path(image_id).stem}.npz"
    if npz_path.exists():
        return np.load(npz_path)['mask']

    image_df = df[df['ImageId'] == image_id]
    if _is_no_defect_image(image_df):
        return _empty_mask()

    return _rle2mask_raw(image_id, image_df)

def traindf_preprocess( split_seed: int = 42, 
                        csv_path: str = "./data/severstal_steel_defect_detection/train.csv",
                        data_path: str = "./data/severstal_steel_defect_detection/train_images",
                        train_ratio: float = 0.60,
                        val_ratio: float = 0.20,
                        test_ratio: float = 0.20,
                        include_no_defect: bool = True,
                        create_mini_dataset: bool = False, mini_size: int = 1024):
    """Split train.csv by image into (train_df, val_df, test_df).

    With include_no_defect, images absent from train.csv are added as ClassId 0 rows with empty EncodedPixels.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("训练、验证和测试的比例总和必须为1.0")
    
    trainfolder_df = pd.read_csv(csv_path)

    if include_no_defect:
        all_train_images = os.listdir(data_path)

        defect_image_ids = trainfolder_df['ImageId'].unique()
        no_defect_image_ids = [img for img in all_train_images if img not in defect_image_ids]

        no_defect_df = pd.DataFrame({
            'ImageId': no_defect_image_ids,
            'ClassId': [0] * len(no_defect_image_ids),
            'EncodedPixels': [None] * len(no_defect_image_ids)
        })

        full_df = pd.concat([trainfolder_df, no_defect_df], ignore_index=True)
    else:
        full_df = trainfolder_df

    train_val_ids, test_ids = train_test_split(full_df['ImageId'].unique(), test_size=test_ratio, random_state=split_seed)
    val_size_relative = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_relative, random_state=split_seed)
    
    if create_mini_dataset:
        train_ids = random.sample(list(train_ids), mini_size)
        val_ids = random.sample(list(val_ids), mini_size // int(train_ratio / val_ratio))
        test_ids =random.sample(list(test_ids), mini_size // int(train_ratio / test_ratio)) 

    train_df = full_df[full_df['ImageId'].isin(train_ids)]
    val_df = full_df[full_df['ImageId'].isin(val_ids)]
    test_df = full_df[full_df['ImageId'].isin(test_ids)]

    print("数据集划分完成。")
    print("-" * 30)
    print(f"总计独立图片数量: {len(full_df['ImageId'].unique())}")
    print(f"训练集图片数量: {len(train_ids)} (目标: {train_ratio:.0%}, 实际: {len(train_ids)/len(full_df['ImageId'].unique()):.1%})")
    print(f"验证集图片数量: {len(val_ids)} (目标: {val_ratio:.0%}, 实际: {len(val_ids)/len(full_df['ImageId'].unique()):.1%})")
    print(f"测试集图片数量: {len(test_ids)} (目标: {test_ratio:.0%}, 实际: {len(test_ids)/len(full_df['ImageId'].unique()):.1%})")
    print("-" * 30)
    print(f"训练集总行数: {len(train_df)}")
    print(f"验证集总行数: {len(val_df)}")
    print(f"测试集总行数: {len(test_df)}")
    print("-" * 30)
    
    train_class_counts = train_df['ClassId'].value_counts().sort_index()
    val_class_counts = val_df['ClassId'].value_counts().sort_index()
    test_class_counts = test_df['ClassId'].value_counts().sort_index()

    print("Train Class Distribution:")
    print(train_class_counts)

    print("\nValidation Class Distribution:")
    print(val_class_counts)

    print("\nTest Class Distribution:")
    print(test_class_counts)

    build_mask_cache(full_df)

    return train_df, val_df, test_df

def get_bounding_box(ground_truth_map, perturb=True, perturb_px=None):
    """Return [x_min, y_min, x_max, y_max] of the mask foreground, or [0, 0, 0, 0] for an empty mask.

    perturb: expand each side by a random 0-19 px (training).
    perturb_px: expand each side by this fixed amount (val/test); overrides `perturb` when not None.
    """
    if torch.sum(ground_truth_map)== 0:
        bbox = [0, 0, 0, 0]
    else:
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = ground_truth_map.shape
        if perturb_px is not None:
            x_min = max(0, x_min - perturb_px)
            x_max = min(W, x_max + perturb_px)
            y_min = max(0, y_min - perturb_px)
            y_max = min(H, y_max + perturb_px)
        elif perturb:
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]
    return bbox

def show_box(box, ax):
    """Copied from the official segment-anything repo."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

class SteelDataset(Dataset):
    def __init__(self, df, data_path, transforms: A.Compose):
        self.df = df
        self.root = data_path
        self.transforms = transforms
        self.fnames = self.df['ImageId'].unique()
 
    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        mask = rle2mask(image_id, self.df)
        image_path = os.path.join(self.root, "train_images", image_id)
        img = np.array(Image.open(image_path).convert("RGB"))

        augmented = self.transforms(image=img, mask=mask)
        img = augmented["image"].float() / 255.0
        mask = augmented["mask"].permute(2, 0, 1)
        mask = torch.sum(mask, dim=0, keepdim=True).float()
        return img, mask, image_id
    
    def __len__(self):
        return len(self.fnames)

class SteelDataset_WithBoxPrompt(Dataset):
    """Severstal with box prompts for SAM fine-tuning.

    Augments image and merged binary mask at native (256, 1600), letterboxes the image to (1024, 1024) as model
    input, keeps the augmented (256, 1600) mask as ground truth, and computes the box prompt on the letterboxed mask.
    """
    def __init__(self, df, data_path, transforms: transforms.Compose, is_train=True, perturb_px=None):
        self.df = df
        self.root = data_path
        self.transforms = transforms
        self.is_train = is_train
        self.perturb_px = perturb_px
        self.fnames = self.df['ImageId'].unique()
    
    def preprocess(self, original_image: torch.Tensor):
        h, w = original_image.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        original_image = F.pad(original_image, (0, padw, 0, padh))
        return original_image
    
    def letterbox_img_np(self, img_np, target_size):
        """Aspect-preserving resize with gray padding; target_size is (width, height).

        Adapted from https://blog.csdn.net/weixin_44791964/article/details/102940768
        """
        original_h, original_w = img_np.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w/original_w, target_h/original_h)
        nw = int(original_w*scale)
        nh = int(original_h*scale)

        if img_np.ndim == 3 and img_np.shape[2] > 1:
            interpolation = cv2.INTER_CUBIC
            fill_value = [128, 128, 128]

        resized_img = cv2.resize(img_np, (nw, nh), interpolation=interpolation)

        letterboxed_img = np.full((target_h, target_w, img_np.shape[2]), fill_value, dtype=img_np.dtype)

        x_offset = (target_w - nw) // 2
        y_offset = (target_h - nh) // 2

        letterboxed_img[y_offset:y_offset+nh, x_offset:x_offset+nw, :] = resized_img

        return letterboxed_img
    
    def letterbox_mask(self, mask: np.ndarray, size: tuple) -> np.ndarray:
        """Nearest-neighbour letterbox of an (H, W, C) mask to (target_height, target_width, C).

        size: (target_width, target_height).
        """
        H, W, C = mask.shape

        target_w, target_h = size

        scale = min(target_w / W, target_h / H)
        new_w = int(W * scale)
        new_h = int(H * scale)

        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # cv2.resize drops the channel axis when C == 1.
        if mask_resized.ndim == 2:
            mask_resized = np.expand_dims(mask_resized, axis=-1)

        letterbox_result = np.zeros((target_h, target_w, C), dtype=mask.dtype)

        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        letterbox_result[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = mask_resized

        return letterbox_result

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        original_mask_4ch = rle2mask(image_id, self.df)                         # (256, 1600, 4)
        original_mask_1ch = np.sum(original_mask_4ch, axis=2, keepdims=True)    # (256, 1600, 1)

        image_path = os.path.join(self.root, "train_images", image_id)
        original_img_np = np.array(Image.open(image_path).convert("RGB"))       # (256, 1600, 3)
        
        augmented = self.transforms(image=original_img_np, mask=original_mask_1ch)
        aug_img_np = augmented["image"]                                                 # (256, 1600, 3)
        aug_mask_1ch_np = augmented["mask"]                                             # (256, 1600, 1)

        letterboxed_img_np = self.letterbox_img_np(aug_img_np, [1024, 1024])                                # (1024, 1024, 3)
        model_input_tensor = torch.from_numpy(letterboxed_img_np).permute(2, 0, 1).float() / 255.0          # (3, 1024, 1024)
        gt_mask_1ch_tensor = torch.from_numpy(aug_mask_1ch_np).permute(2, 0, 1).float()                     # (1, 256, 1600)

        # The box prompt lives in the letterboxed 1024x1024 frame.
        letterboxed_mask_np = self.letterbox_mask(aug_mask_1ch_np, [1024, 1024])          # (1024, 1024, 1)

        letterboxed_mask_tensor = torch.from_numpy(letterboxed_mask_np).permute(2, 0, 1).float()    # (1, 1024, 1024)
    

        bbox = get_bounding_box(letterboxed_mask_tensor.squeeze(), perturb=self.is_train, perturb_px=self.perturb_px)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)           # (4,): [x_min, y_min, x_max, y_max]

        return {"image": model_input_tensor,
                "mask": gt_mask_1ch_tensor.squeeze(),  # [256, 1600]
                "letterboxed_mask": letterboxed_mask_tensor.squeeze(),  # [1024, 1024]
                "image_id": image_id, 
                "bbox": bbox_tensor
                }
    
    def __len__(self):
        return len(self.fnames)


class SteelDataset_WithBoxPromptResizeInfer(SteelDataset_WithBoxPrompt):
    """Severstal inference variant: plain resize (no letterbox) of image and merged binary mask to (1024, 1024).

    The box prompt is computed on the resized mask; the (256, 1600) mask is still returned as ground truth.
    """
    def __init__(self, df, data_path, transforms: transforms.Compose = None, is_train=False, perturb_px=None):
        super().__init__(df=df, data_path=data_path, transforms=transforms, is_train=is_train, perturb_px=perturb_px)
        self.resize_transform = A.Compose([
            A.Resize(
                height=1024,
                width=1024,
                interpolation=cv2.INTER_CUBIC,
            )
        ])

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        original_mask_4ch = rle2mask(image_id, self.df)
        original_mask_1ch = np.sum(original_mask_4ch, axis=2, keepdims=True)
        original_mask_1ch = (original_mask_1ch > 0).astype(np.uint8)

        image_path = os.path.join(self.root, "train_images", image_id)
        original_img_np = np.array(Image.open(image_path).convert("RGB"))

        if self.transforms is not None:
            augmented = self.transforms(image=original_img_np, mask=original_mask_1ch)
            original_img_np = augmented["image"]
            original_mask_1ch = augmented["mask"]

        resized = self.resize_transform(image=original_img_np, mask=original_mask_1ch)
        resized_img_np = resized["image"]
        resized_mask_np = resized["mask"]
        if resized_mask_np.ndim == 2:
            resized_mask_np = np.expand_dims(resized_mask_np, axis=-1)

        model_input_tensor = torch.from_numpy(resized_img_np).permute(2, 0, 1).float() / 255.0
        gt_mask_tensor = torch.from_numpy(original_mask_1ch).permute(2, 0, 1).float()
        resized_mask_tensor = torch.from_numpy(resized_mask_np).permute(2, 0, 1).float()

        bbox = get_bounding_box(resized_mask_tensor.squeeze(), perturb=self.is_train, perturb_px=self.perturb_px)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return {
            "image": model_input_tensor,
            "mask": gt_mask_tensor.squeeze(),
            "resized_mask": resized_mask_tensor.squeeze(),
            "image_id": image_id,
            "bbox": bbox_tensor,
        }

def create_datasets_no_prompt(train_df, val_df, test_df, data_path, train_transform, val_transform):
    train_dataset = SteelDataset(train_df, data_path, transforms=train_transform)
    val_dataset = SteelDataset(val_df, data_path, transforms=val_transform)
    test_datset = SteelDataset(test_df, data_path, transforms=val_transform)
    return train_dataset, val_dataset, test_datset

    
def create_dataloaders_no_prompt(train_df, val_df, test_df,data_path, train_transform, val_transform, batch_size, num_workers):
    train_dataset, val_dataset, test_dataset = create_datasets_no_prompt(train_df, val_df, test_df, data_path, train_transform, val_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    return train_dataloader, val_dataloader, test_dataloader
    
def create_dataset_with_prompt(include_no_defect=True):
    train_df, val_df, test_df = traindf_preprocess(create_mini_dataset=False, mini_size=256, include_no_defect=include_no_defect)
    data_path = "./data/severstal_steel_defect_detection"
    train_transforms, val_transforms = get_severstal_ft_albumentations_transforms()

    print('#'*20 + ' test_dataset_with_prompt ' + '#'*20)
    train_dataset = SteelDataset_WithBoxPrompt(train_df, data_path=data_path, transforms=train_transforms, is_train=True)
    val_dataset = SteelDataset_WithBoxPrompt(val_df, data_path=data_path, transforms=val_transforms, is_train=False)
    test_dataset = SteelDataset_WithBoxPrompt(test_df, data_path=data_path, transforms=val_transforms, is_train=False)
    return train_dataset, val_dataset, test_dataset

def get_albumentations_transforms():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        ToTensorV2()
    ])
    return train_transforms, val_transforms

def get_severstal_ft_albumentations_transforms():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])

    val_transforms = A.Compose([
    ])
    return train_transforms, val_transforms

def mask2rle(img):
    '''Encode a binary (H, W) mask as a Kaggle RLE string ("start length ..."); empty mask -> "".'''
    if img.sum() == 0:
        return ""
    pixels = img.T.flatten()  # column-major, as Kaggle's RLE expects
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return ' '.join(str(x) for x in runs)

def random_visualize(dataset, num_pics):
    random_samples_idx = random.sample(range(len(dataset)), k=num_pics)
    fig, ax = plt.subplots(nrows=num_pics, ncols=5, figsize=(16, 8))
    for i, targ_sample in enumerate(random_samples_idx):
        image, masks = dataset[targ_sample][0], dataset[targ_sample][1]
        image = image.permute(1, 2, 0)    
        
        ax[i, 0].imshow(image)
        ax[i, 0].set_title("Original image")
        ax[i, 0].axis("off")
        for j in range(4):
            ax[i, j+1].imshow(masks[j], cmap='gray')
            ax[i, j+1].set_title(f"mask chanel {j+1}")
            ax[i, j+1].axis("off")
    plt.tight_layout()
    plt.show()


def visualize_dataset_and_groundtruth_mask(dataset, img_count=1):
    train_df, val_df = traindf_preprocess()
    print(train_df.head())
    print(val_df.head())
    random_samples_idx = random.sample(range(len(dataset)), k=img_count)
    fig, ax = plt.subplots(nrows=img_count, ncols=1, figsize=(16, 8))
    for i, targ_sample in enumerate(random_samples_idx):
        image, masks = dataset[targ_sample][0], dataset[targ_sample][1]
        image = image.permute(1, 2, 0)
        
        ax[i, 0].imshow(image)
        ax[i, 0].set_title("Original image")
        ax[i, 0].axis("off")
        for j in range(4):
            ax[i,1].imshow(masks[j], alpha=0.5)
            ax[i,1].axis("off")
    plt.tight_layout()
    plt.show()

def visualize_bbox_prompt(steel_dataset_withboxPrompt):
    idx = random.randint(0, len(steel_dataset_withboxPrompt))
    sample = steel_dataset_withboxPrompt[idx]
    img0, mask0, imgid0, bbox = sample['image'], sample['mask'], sample['image_id'], sample['bbox']
    print(f"img0 type: {type(img0)}, img0 shape: {img0.shape}")
    print(f"mask type : {type(mask0)}, mask shape: {mask0.shape}")
    img0_plt_format = img0.permute(1, 2, 0)

    cmaps = ['Reds', 'Greens', 'Blues', 'Purples']

    mask = torch.sum(mask0, dim=0, keepdim=True)
    print(f"comined mask (keep dim) shape: {mask.shape}")

    print(f"img get bbox position: {bbox}")

    fig, ax = plt.subplots()
    ax.imshow(img0_plt_format)
    ax.imshow(mask0, cmap=cmaps[3], alpha = 0.8 * (mask0 > 0))
    ax.set_title(f"{imgid0} with mask")

    plt.figure()
    plt.imshow(img0_plt_format)
    plt.imshow(mask.permute(1, 2, 0).cpu().numpy(), cmap='Oranges', alpha=(0.8 * (mask > 0).float()).squeeze(0).cpu().numpy())
    show_box(bbox, ax)

def reverse_letterbox_mask_1ch(mask_letterbox: np.ndarray, orig_size: tuple, target_size: tuple = (1024, 1024)) -> np.ndarray:
    """Crop the letterbox padding and resize back to (orig_height, orig_width).

    mask_letterbox: (target_height, target_width).
    orig_size: (orig_height, orig_width), e.g. (256, 1600).
    target_size: (target_width, target_height) used for the letterbox, e.g. (1024, 1024).
    """
    orig_h, orig_w = orig_size
    target_w, target_h = target_size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    mask_cropped = mask_letterbox[y_offset : y_offset + new_h, x_offset : x_offset + new_w]

    restored_mask = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    return restored_mask

if __name__ == "__main__":
    a, b, c = create_dataset_with_prompt(include_no_defect=False)
    debug_dataset_info(a, b, c)