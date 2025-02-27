import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import monai
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from datetime import datetime
import pytz
import random
import cv2
from sklearn.model_selection import StratifiedShuffleSplit
import logging
from helper_function import get_bounding_box
from data_setup import magtile_get_all_imgmsk_paths, find_classes 


class NEU_SEG_Dataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms: A.Compose):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.png')])
    
    def letterbox_imagenp(self, image: np.ndarray, size):
        """
        对图片进行resize, 使图片不失真。在空缺的地方进行padding
        inspired by https://blog.csdn.net/weixin_44791964/article/details/102940768 but we modify it to np.ndarray
        """
        ih, iw, ic = image.shape
        w, h = size                 # 目标尺寸
        scale = min(w/iw, h/ih)     # 计算缩放比例，保持纵横比
        nw = int(iw*scale)
        nh = int(ih*scale)

        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)  # 双三次插值

        # 创建一个填充背景的图像，这里使用RGB(128, 128, 128)作为填充色
        new_image = np.full((h, w, ic), 128, dtype=image.dtype)

        # 计算填充的偏移量，将缩放后的图像居中放置
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        new_image[dy:dy+nh, dx:dx+nw,:] = image_resized
        return new_image

    def letterbox_mask_1ch(self, mask: np.ndarray, size: tuple) -> np.ndarray:
        """
        参数:
            mask: np.ndarray, shape = (H, W)
            size: (target_width, target_height), 例如 (1024, 1024)
        返回:
            result: np.ndarray, shape = (target_height, target_width)
        """
        H, W = mask.shape
        target_w, target_h = size
        scale = min(target_w / W, target_h / H)     # 计算等比例缩放
        new_w = int(W * scale)
        new_h = int(H * scale)

        # 最近邻插值缩放
        # 注意 OpenCV 的 resize 第二个参数是 (width, height)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # 最近邻插值
        letterbox_result = np.zeros((target_h, target_w), dtype=mask.dtype)      # 创建空白画布
        x_offset = (target_w - new_w) // 2                                          # 计算粘贴位置，使其居中
        y_offset = (target_h - new_h) // 2
        letterbox_result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_resized        # 将缩放后的掩码贴到画布中
        return letterbox_result
    
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
        label = np.unique(mask_np)[1]

        # 原数据集将不同缺陷的掩码值分别设为了1-3，但是整个图片是单通道的，所以这里二值化为全为1进行处理
        # 将所有大于0的掩码值设为1
        mask_np = (mask_np > 0).astype(np.float32)

        resized_img = self.letterbox_imagenp(image_np, [1024, 1024])
        resized_mask = self.letterbox_mask_1ch(mask_np, [1024, 1024])
        bbox = get_bounding_box(resized_mask)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        if self.transforms:
            augmented = self.transforms(image=resized_img, mask=resized_mask)
            image_tensor = augmented['image'] / 255.0
            mask_tensor = augmented['mask']
            mask_tensor = mask_tensor.float()
        return {
            "image": image_tensor.float(),
            "mask": mask_tensor.float(),
            "bbox": bbox_tensor.float(),
            "image_id": image_id,
            "label": label
        }

# 定义变换
def get_neu_albumentation_transforms():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),    
        ToTensorV2()
    ])
    test_transforms = A.Compose([
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transforms, test_transforms

def create_neu_dataset():
    # 数据集路径
    train_images_dir = 'data/NEU_Seg-main/images/training'
    train_annotations_dir = 'data/NEU_Seg-main/annotations/training'

    test_images_dir = 'data/NEU_Seg-main/images/test'
    test_annotations_dir = 'data/NEU_Seg-main/annotations/test'

    train_transforms, test_transforms = get_neu_albumentation_transforms()

    # 创建Dataset实例
    train_dataset = NEU_SEG_Dataset(train_images_dir, train_annotations_dir, transforms=train_transforms)
    test_dataset = NEU_SEG_Dataset(test_images_dir, test_annotations_dir, transforms=test_transforms)
    return train_dataset, test_dataset




# 定义 Dataset
class SDsaliency900Dataset(Dataset):
    def __init__(self, source_dir, ground_truth_dir, transforms: A.Compose):
        self.source_dir = source_dir
        self.ground_truth_dir = ground_truth_dir
        self.transforms = transforms

        self.image_names = sorted([f for f in os.listdir(source_dir) if f.endswith('.bmp')])
        self.labels = []

        for name in self.image_names:
            if name.startswith('In_'):
                self.labels.append(0)  # Inclusion
            elif name.startswith('Pa_'):
                self.labels.append(1)  # Patches
            elif name.startswith('Sc_'):
                self.labels.append(2)  # Scratches
            else:
                raise ValueError(f'未知的文件命名格式: {name}')
    
    def letterbox_imagenp(self, image: np.ndarray, size):
        """
        对图片进行resize, 使图片不失真。在空缺的地方进行padding
        inspired by https://blog.csdn.net/weixin_44791964/article/details/102940768 but we modify it to np.ndarray
        """
        ih, iw, ic = image.shape
        w, h = size                 # 目标尺寸
        scale = min(w/iw, h/ih)     # 计算缩放比例，保持纵横比
        nw = int(iw*scale)
        nh = int(ih*scale)

        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)  # 双三次插值

        # 创建一个填充背景的图像，这里使用RGB(128, 128, 128)作为填充色
        new_image = np.full((h, w, ic), 128, dtype=image.dtype)

        # 计算填充的偏移量，将缩放后的图像居中放置
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        new_image[dy:dy+nh, dx:dx+nw,:] = image_resized
        return new_image

    def letterbox_mask_1ch(self, mask: np.ndarray, size: tuple) -> np.ndarray:
        """
        参数:
            mask: np.ndarray, shape = (H, W)
            size: (target_width, target_height), 例如 (1024, 1024)
        返回:
            result: np.ndarray, shape = (target_height, target_width)
        """
        H, W = mask.shape
        target_w, target_h = size
        scale = min(target_w / W, target_h / H)     # 计算等比例缩放
        new_w = int(W * scale)
        new_h = int(H * scale)

        # 最近邻插值缩放
        # 注意 OpenCV 的 resize 第二个参数是 (width, height)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # 最近邻插值
        letterbox_result = np.zeros((target_h, target_w), dtype=mask.dtype)      # 创建空白画布
        x_offset = (target_w - new_w) // 2                                          # 计算粘贴位置，使其居中
        y_offset = (target_h - new_h) // 2
        letterbox_result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_resized        # 将缩放后的掩码贴到画布中
        return letterbox_result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.source_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.ground_truth_dir, mask_name)
        label = self.labels[idx]

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)   # (200, 200, 3)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)              # (200, 200)
        
        resized_img = self.letterbox_imagenp(image, [1024, 1024])      # (1024, 1024, 3)
        resized_mask = self.letterbox_mask_1ch(mask, [1024, 1024])     # (256, 256)
        bbox = get_bounding_box(resized_mask)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32) 
        
        if self.transforms:
            augmented = self.transforms(image=resized_img, mask=resized_mask)
            image_tensor = augmented['image'] / 255.0
            mask_tensor = augmented['mask'] / 255.0  # 将mask归一化到0-1之间, albumentation不会自动归一化
        
        return {"image": image_tensor.float(),
                "mask": mask_tensor.float(),
                "bbox": bbox_tensor.float(),
                "label": self.labels[idx]
                }

# 定义变换
def get_sd900_albumentation_transforms():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),    ##原先mask的是0, 255， image的值在0-255范围之间
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transforms, val_transforms

def sd_900_finetune_create_dataset():
    # 设置目录路径
    sd900_src_dir = './data/sd900/Source Images'
    sd900_gt_dir = './data/sd900/Ground truth'

    # 创建数据集
    full_dataset = SDsaliency900Dataset(
        source_dir=sd900_src_dir,
        ground_truth_dir=sd900_gt_dir,
        transforms = None
    )

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 分层划分
    labels = full_dataset.labels
    logging.info(f"sd900全数据集的Labels: {labels}")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # n_splits = 1表示只拆分一次 >1表示可以拆分多次次形成不同的数据集
    train_idx, test_idx = next(split.split(full_dataset.image_names, labels))

    train_transforms, val_transforms = get_sd900_albumentation_transforms()
    train_fulldataset = SDsaliency900Dataset(source_dir=sd900_src_dir, ground_truth_dir=sd900_gt_dir, transforms = train_transforms)
    test_fulldataset = SDsaliency900Dataset(source_dir=sd900_src_dir, ground_truth_dir=sd900_gt_dir, transforms = val_transforms)

    train_dataset = torch.utils.data.Subset(train_fulldataset, train_idx)
    test_dataset = torch.utils.data.Subset(test_fulldataset, test_idx)

    # 创建 DataLoader
    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_dataset, test_dataset


class MagneticTileDatasetWithBoxPrompt(Dataset):
    def __init__(self, image_paths, mask_paths, labels, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.labels = labels
    
    def letterbox_image(self, image, size):
        """
        对图片进行resize, 使图片不失真。在空缺的地方进行padding
        cited from https://blog.csdn.net/weixin_44791964/article/details/102940768
        """
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image
    
    def letterbox_mask_1ch(self, mask: np.ndarray, size: tuple) -> np.ndarray:
        """
        参数:
            mask: np.ndarray, shape = (H, W)
            size: (target_width, target_height), 例如 (1024, 1024)
        返回:
            result: np.ndarray, shape = (target_height, target_width)
        """
        H, W = mask.shape
        target_w, target_h = size
        scale = min(target_w / W, target_h / H)     # 计算等比例缩放
        new_w = int(W * scale)
        new_h = int(H * scale)

        # 最近邻插值缩放
        # 注意 OpenCV 的 resize 第二个参数是 (width, height)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        letterbox_result = np.zeros((target_h, target_w), dtype=mask.dtype)      # 创建空白画布
        x_offset = (target_w - new_w) // 2                                          # 计算粘贴位置，使其居中
        y_offset = (target_h - new_h) // 2
        letterbox_result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_resized        # 将缩放后的掩码贴到画布中
        return letterbox_result

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        mask_np = np.array(mask)

        resize_img = self.letterbox_image(image, [1024, 1024]) # sam必须以1024x1024输入
        resize_mask = self.letterbox_mask_1ch(mask_np, [256, 256]) # sam的low res输出是256x256，该数据集大小本身就比较小
        bbox = get_bounding_box(resize_mask)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        if self.transform:
            resize_img_tensor = self.transform(resize_img)
            resize_mask_tensor = self.transform(resize_mask)
            resize_mask_tensor = (resize_mask_tensor > 0.5).float()
        label = self.labels[idx]
        return {"image": resize_img_tensor.float(),
                "mask": resize_mask_tensor,
                "bbox": bbox_tensor.float(),
                "label": self.labels[idx]
                }
    
    def __len__(self):
        return len(self.image_paths)

def create_MagneticTile_Dataset():
    magtile_root_dir = 'data/Magnetic-Tile'
    image_paths, mask_paths, labels = magtile_get_all_imgmsk_paths(magtile_root_dir)
    classes, classes_to_idx = find_classes(magtile_root_dir)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)      # 定义分层抽样器（确保训练集和测试集中各类别的比例相同）
    train_idx, test_idx = next(split.split(np.zeros(len(labels)), labels))

    # Map indices to paths and labels
    train_image_paths = [image_paths[i] for i in train_idx]
    train_mask_paths = [mask_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    test_image_paths = [image_paths[i] for i in test_idx]
    test_mask_paths = [mask_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_transforms = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor()
        ]
    )

    val_transforms = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor()
        ]
    )
    train_dataset = MagneticTileDatasetWithBoxPrompt(train_image_paths, train_mask_paths, train_labels, transform=train_transforms)
    test_dataset = MagneticTileDatasetWithBoxPrompt(test_image_paths, test_mask_paths, test_labels, transform=val_transforms)
    return train_dataset, test_dataset