import os
import torch
import cv2
import random
import logging
import scipy.io 
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from data.data_utils_baseline import magtile_find_classes, magtile_get_all_imgmsk_paths, general_albumentation_transforms
from utils.helper_function import get_bounding_box

class SegDatasetForFinetune(Dataset):
    """
    base class for dataset for finetune
    """
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

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

class SDsaliency900Dataset(Dataset):
    def __init__(self, source_dir, ground_truth_dir, transforms: A.Compose):
        self.source_dir = source_dir
        self.ground_truth_dir = ground_truth_dir
        self.transforms = transforms

        # 确保在不同的系统上，文件列表的顺序是一致的
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
        resized_mask = self.letterbox_mask_1ch(mask, [1024, 1024])     # (1024, 1024)
        bbox = get_bounding_box(resized_mask)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32) 
        
        if self.transforms:
            augmented = self.transforms(image=resized_img, mask=resized_mask)
            image_tensor = augmented['image'] / 255.0
            mask_tensor = augmented['mask'] / 255.0  # 将mask归一化到0-1之间, albumentation不会自动归一化
        
        return {"image": image_tensor.float(),  "mask": mask_tensor.float(),
                "bbox": bbox_tensor.float(),    "label": self.labels[idx]}

def general_albumentation_transforms_for_finetune():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),    #原先mask的是0, 255， image的值在0-255范围之间
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transforms, val_transforms


# 定义变换
def get_sd900_albumentation_transforms():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),    #原先mask的是0, 255， image的值在0-255范围之间
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transforms, val_transforms

def sd_900_finetune_create_dataset( sd900_src_dir = './data/sd900/Source Images', 
                                    sd900_gt_dir = './data/sd900/Ground truth',
                                    split_seed = 42):

    full_dataset = SDsaliency900Dataset(source_dir=sd900_src_dir, ground_truth_dir=sd900_gt_dir, transforms = None)

    # 分层划分
    labels = full_dataset.labels
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"sd900全数据集的Labels: {labels}")

    labels = np.array(full_dataset.labels)
    # print(full_dataset.image_names)

    # 第一次划分：分出60%作为训练集，剩下40%作为临时集
    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=split_seed)
    train_indices, tmp_indices = next(split_1.split(np.zeros(len(labels)), labels))

    # 从原始标签中获取临时集的标签
    tmp_labels = labels[tmp_indices]

    # 第二次划分：将40%的临时集对半分为验证集和测试集 (各占总数的20%)
    # test_size=0.5 表示将 tmp_indices 对半划分
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=split_seed)
    val_relative_indices, test_relative_indices = next(split_2.split(np.zeros(len(tmp_labels)), tmp_labels))

    # 将相对索引映射回原始数据集的绝对索引
    val_indices = tmp_indices[val_relative_indices]
    test_indices = tmp_indices[test_relative_indices]

    # 获取带不同变换的数据集实例
    train_transforms, val_test_transforms = get_sd900_albumentation_transforms()
    train_full_dataset = SDsaliency900Dataset(source_dir=sd900_src_dir, ground_truth_dir=sd900_gt_dir, transforms=train_transforms)
    val_test_full_dataset = SDsaliency900Dataset(source_dir=sd900_src_dir, ground_truth_dir=sd900_gt_dir, transforms=val_test_transforms)

    # 使用最终的索引列表创建 Subset，避免嵌套
    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_test_full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(val_test_full_dataset, test_indices)
    return train_dataset, val_dataset, test_dataset

def sd_900_finetune_create_dataloader(args, train_dataset, val_dataset, test_dataset):

    batch_size = args.batch_size
    num_workers = args.num_workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def debug_sd900_img_and_mask():
    img = cv2.cvtColor(cv2.imread('data/sd900/Source Images/In_1.bmp'), cv2.COLOR_BGR2RGB)   # (200, 200, 3)
    gt = cv2.imread('data/sd900/Ground truth/In_1.png', cv2.IMREAD_GRAYSCALE)     #  原来是(200, 200, 3) --> (200, 200)
    print(f"---------cv2读取img和mask---------\n" 
            f"img.shape:{img.shape}, gt.shape:{gt.shape}\n"
            f"img.type:{type(img)}, gt.type:{type(gt)}\n"
            f"img.dtype:{img.dtype}, gt.dtype:{gt.dtype}\n"
            f"img data range: {img.min()} ~ {img.max()}\n"
            f"gt data unique values: {np.unique(gt)}\n"
            f"-----------------------------------")

    img_tr = transforms.ToTensor()(img)  #  (3, 200, 200)
    gt_tr = transforms.ToTensor()(gt)    #  (1, 200, 200)
    print(f"---------torchvision转换--------\n"
            f"img_tr.shape:{img_tr.shape}, gt_tr.shape:{gt_tr.shape}\n"
            f"img_tr.type:{type(img_tr)}, gt_tr.type:{type(gt_tr)}\n"
            f"img_tr.dtype:{img_tr.dtype}, gt_tr.dtype:{gt_tr.dtype}\n"
            f"img_tr data range: {img_tr.min()} ~ {img_tr.max()}\n"
            f"gt_tr data unique: {torch.unique(gt_tr)}\n"
            f"-----------------------------------")

    train_aug, test_aug = get_sd900_albumentation_transforms()
    augmented = train_aug(image=img, mask=gt)
    img_aug = augmented['image']  # (3, 200, 200)   C, H ,W
    mask_aug = augmented['mask']  # (200, 200)   H, W

    print(f"---------albumentation 转换 totensorv2并不会归一化需要自己除以255.0并转为float32--------\n"
            f"img_aug.shape:{img_aug.shape}, gt_tr.shape:{mask_aug.shape}\n"
            f"img_aug.type:{type(img_aug)}, gt_tr.type:{type(mask_aug)}\n"
            f"img_aug.dtype:{img_aug.dtype}, gt_tr.dtype:{mask_aug.dtype}\n"
            f"img_aug data range: {img_aug.min()} ~ {img_aug.max()}\n"
            f"mask_aug data unique: {torch.unique(mask_aug)}\n"
            f"-----------------------------------")


class MagneticTileDatasetWithBoxPrompt(Dataset):
    def __init__(self, image_paths, mask_paths, labels, transforms: A.Compose):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.labels = labels
    
    def letterbox_image(self, image, size):
        """
        对图片进行resize, 使图片不失真。在空缺的地方进行padding
        cited from https://blog.csdn.net/weixin_44791964/article/details/102940768
        """
        ih, iw, c = image.shape
        target_w, target_h = size
        scale = min(target_w/iw, target_h/ih)
        new_w = int(iw*scale)
        new_h = int(ih*scale)

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        letterboxed_image = np.full((target_h, target_w, image.shape[2]), 128, dtype=image.dtype)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        letterboxed_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

        # image = image.resize((nw,nh), Image.BICUBIC)
        # new_image = Image.new('RGB', size, (128,128,128))
        # new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return letterboxed_image
    
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
        image_np = np.array(image)
        mask_np = np.array(mask)

        # 1. 初始化变量，避免 UnboundLocalError
        # 无论是否有 transform，都从原始的 numpy 数组开始
        # aug_img_np = image_np
        # aug_mask_1ch_np = mask_np

        # if self.transforms:
        augmented = self.transforms(image=image_np, mask=mask_np)
        aug_img_np = augmented["image"]        
        aug_mask_1ch_np = augmented["mask"]    

        # print(f"aug_img_np dtype: {aug_img_np.dtype}")
        # print(f"aug_mask_1ch_np dtype: {aug_mask_1ch_np.dtype}")

        resize_img_np = self.letterbox_image(aug_img_np, [1024, 1024])             # sam必须以1024x1024输入
        resize_mask_np = self.letterbox_mask_1ch(aug_mask_1ch_np, [1024, 1024])      # sam的low res输出是256x256，该数据集大小本身就比较小
        # print(f"resize_mask_np unique: {np.unique(resize_mask_np)}")   # (0-255)
        bbox = get_bounding_box(resize_mask_np)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        
        resize_img_tensor = torch.from_numpy(resize_img_np).permute(2, 0, 1).float() / 255.0
        resize_mask_tensor = (torch.from_numpy(resize_mask_np) > 0.5).float()

        label = self.labels[idx]
        return {"image": resize_img_tensor,
                "mask": resize_mask_tensor,
                "bbox": bbox_tensor.float(),
                "label": self.labels[idx]
                }
    
    def __len__(self):
        return len(self.image_paths)

# 定义变换
def mag_albumentation_transforms():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])
    val_transforms = A.Compose([
    ])
    return train_transforms, val_transforms

def create_magnetic_dataset():
    magtile_root_dir = 'data/Magnetic-Tile'
    image_paths, mask_paths, labels = magtile_get_all_imgmsk_paths(magtile_root_dir)
    classes, classes_to_idx = magtile_find_classes(magtile_root_dir)

    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)      # 定义分层抽样器（确保训练集和测试集中各类别的比例相同）
    train_val_idx, test_idx = next(split_1.split(np.zeros(len(labels)), labels))

    test_image_paths = [image_paths[i] for i in test_idx]
    test_mask_paths = [mask_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_val_image_paths = [image_paths[i] for i in train_val_idx]
    train_val_mask_paths = [mask_paths[i] for i in train_val_idx]
    train_val_labels = [labels[i] for i in train_val_idx]

    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)      # 定义分层抽样器（确保训练集和测试集中各类别的比例相同）
    train_idx, val_idx = next(split_2.split(np.zeros(len(train_val_labels)), train_val_labels))

    # Map indices to paths and labels
    train_image_paths = [train_val_image_paths[i] for i in train_idx]
    train_mask_paths = [train_val_mask_paths[i] for i in train_idx]
    train_labels = [train_val_labels[i] for i in train_idx]

    # 提取验证集数据
    val_image_paths = [train_val_image_paths[i] for i in val_idx]
    val_mask_paths = [train_val_mask_paths[i] for i in val_idx]
    val_labels = [train_val_labels[i] for i in val_idx]

    train_transforms, val_transforms = mag_albumentation_transforms()

    # ------------------ 新增：统计每个类别的数量 ------------------
    print("\n" + "="*50)
    print("           Dataset Class Distribution           ")
    print("="*50)

    # 1. 统计原始完整数据集
    total_counts = Counter(labels)
    print(f"\n--- Original Full Dataset ({len(labels)} samples) ---")
    for class_idx, count in sorted(total_counts.items()):
        class_name = classes[class_idx]
        print(f"  - Class '{class_name}' (ID: {class_idx}): {count} samples")

    # 2. 统计训练集
    train_counts = Counter(train_labels)
    print(f"\n--- Training Set ({len(train_labels)} samples) ---")
    for class_idx, count in sorted(train_counts.items()):
        class_name = classes[class_idx]
        percentage = (count / len(train_labels)) * 100
        print(f"  - Class '{class_name}' (ID: {class_idx}): {count} samples ({percentage:.2f}%)")

    # 3. 统计验证集
    val_counts = Counter(val_labels)
    print(f"\n--- Validation Set ({len(val_labels)} samples) ---")
    for class_idx, count in sorted(val_counts.items()):
        class_name = classes[class_idx]
        percentage = (count / len(val_labels)) * 100
        print(f"  - Class '{class_name}' (ID: {class_idx}): {count} samples ({percentage:.2f}%)")

    # 4. 统计测试集
    test_counts = Counter(test_labels)
    print(f"\n--- Test Set ({len(test_labels)} samples) ---")
    for class_idx, count in sorted(test_counts.items()):
        class_name = classes[class_idx]
        percentage = (count / len(test_labels)) * 100
        print(f"  - Class '{class_name}' (ID: {class_idx}): {count} samples ({percentage:.2f}%)")
    
    print("\n" + "="*50 + "\n")
    # ------------------ 统计代码结束 ------------------

    train_dataset = MagneticTileDatasetWithBoxPrompt(train_image_paths, train_mask_paths, train_labels, transforms=train_transforms)
    val_dataset = MagneticTileDatasetWithBoxPrompt(val_image_paths, val_mask_paths, val_labels, transforms=val_transforms)
    test_dataset = MagneticTileDatasetWithBoxPrompt(test_image_paths, test_mask_paths, test_labels, transforms=val_transforms)

    return train_dataset, val_dataset, test_dataset


class FloodSegDataset(Dataset):
    def __init__(self, root_dir, image_folder="Image", mask_folder="Mask", transforms: A.Compose = None):
        """
        Args:
            root_dir (string): 数据集根目录, e.g., './data/FloodSeg'
            image_folder (string): 存放原始图片的文件夹名
            mask_folder (string): 存放掩码图片的文件夹名
            transforms (A.Compose, optional): Albumentations的变换操作.
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)
        self.transforms = transforms
        
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))
        
        assert len(self.image_files) == len(self.mask_files), "图片和掩码数量不匹配"

    def letterbox_imagenp(self, image: np.ndarray, size=(1024, 1024)):
        ih, iw, ic = image.shape
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.full((h, w, ic), 128, dtype=image.dtype)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        new_image[dy:dy+nh, dx:dx+nw,:] = image_resized
        return new_image

    def letterbox_mask_1ch(self, mask: np.ndarray, size=(1024, 1024)):
        H, W = mask.shape
        target_w, target_h = size
        scale = min(target_w / W, target_h / H)
        new_w = int(W * scale)
        new_h = int(H * scale)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        letterbox_result = np.zeros((target_h, target_w), dtype=mask.dtype)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        letterbox_result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_resized
        return letterbox_result
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)    # 0-255
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {img_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)              # 0-255
        if mask is None:
            raise FileNotFoundError(f"无法加载掩码: {mask_path}")

        # # 1. Letterbox resizing to 1024x1024
        # resized_img = self.letterbox_imagenp(image, size=(1024, 1024))              # (1024, 1024, 3) 255 0 uint8
        # resized_mask = self.letterbox_mask_1ch(mask, size=(1024, 1024))             # (1024, 1024) 255 0 uint8

        # print(resized_img.shape, resized_img.max(), resized_img.min(), np.unique(resized_img), resized_img.dtype)              
        # print(resized_mask.shape, resized_mask.max(), resized_mask.min(), np.unique(resized_mask), resized_mask.dtype)           
        
        # 2. Apply albumentations transforms
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image_tensor = augmented['image']                               
            mask_tensor = augmented['mask']          
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)    # (3, 1024, 1024)
            mask_tensor = torch.from_numpy(mask)   # (1024, 1024)

        # 3. Normalize and ensure correct data types
        image_tensor = image_tensor.float() / 255.0
        mask_tensor = (mask_tensor > 0).float() # Ensure mask is binary (0.0 or 1.0)

        # 4. Return the dictionary (WITHOUT bbox)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            # "bbox" key is now removed
            "label": 0  # Dummy label for structural consistency
        }

# 定义变换
def get_floodseg_albumentation_transforms():
    """定义用于FloodSeg数据集的Albumentations变换"""
    train_transforms = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        ToTensorV2()  # 将图像和掩码转换为PyTorch张量 (H,W,C) -> (C,H,W) for image, (H,W) -> (H,W) for mask
    ])
    val_transforms = A.Compose([
        # A.Normalize similar to above.
        A.Resize(1024, 1024),
        ToTensorV2()
    ])
    return train_transforms, val_transforms

def floodseg_create_dataset(flood_root_dir='./data/FloodSeg', split_seed=42):
    """为FloodSeg数据集创建训练、验证和测试Dataloader"""

    train_transforms, val_test_transforms = get_floodseg_albumentation_transforms()

    # 创建带有不同变换的完整数据集实例
    train_full_dataset = FloodSegDataset(root_dir=flood_root_dir, transforms=train_transforms)
    val_test_full_dataset = FloodSegDataset(root_dir=flood_root_dir, transforms=val_test_transforms)
    
    # 随机划分数据集
    dataset_size = len(train_full_dataset)
    indices = list(range(dataset_size))
    # 60% 训练, 20% 验证, 20% 测试
    train_size = int(0.6 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(split_seed)
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        indices, [train_size, val_size, test_size], generator=generator
    )

    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_test_full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(val_test_full_dataset, test_indices)
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test.")
    
    return train_dataset, val_dataset, test_dataset

class BSDS500BoundaryDataset(Dataset):
    """
    用于BSDS500数据集的PyTorch Dataset类，用于边缘检测任务。
    """
    def __init__(self, root_dir, split='train', transforms: A.Compose = None, consensus_threshold=0.5):
        """
        Args:
            root_dir (string): BSDS500/data 目录的路径。
            split (string): 'train', 'val', or 'test'.
            transforms (A.Compose, optional): Albumentations的变换操作。
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.consensus_threshold = consensus_threshold
        
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.gt_dir = os.path.join(root_dir, 'ground_truth', split)

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.gt_files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.mat')])
        
        assert len(self.image_files) == len(self.gt_files), "图片和.mat文件数量不匹配"

        # 验证文件名对应关系
        for img_file, gt_file in zip(self.image_files, self.gt_files):
            img_base = os.path.splitext(img_file)[0]
            gt_base = os.path.splitext(gt_file)[0]
            assert img_base == gt_base, f"文件名不匹配: {img_file} vs {gt_file}"

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_id = self.image_files[idx]
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)       # (H, W, C), 0-255, uint8

        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        mat_data = scipy.io.loadmat(gt_path)

        ground_truths = mat_data['groundTruth'][0]                          # 所有人的标注
        num_annotators = 5                                 # 每张图片有 4 - 8个人的标注
        # num_annotators = len(ground_truths)                                 # 每张图片有 4 - 8个人的标注
        h, w = ground_truths[0]['Boundaries'][0, 0].shape                   # 获取第一个标注图的形状以初始化累加器
        summed_boundaries = np.zeros((h, w), dtype=np.float32)              # 初始化

        # 累加所有标注者的边缘图
        for i in range(num_annotators):
            annotator_gt = ground_truths[i]['Boundaries'][0, 0].astype(np.float32)
            summed_boundaries += annotator_gt

        # 取平均值，得到 [0, 1] 范围的概率图
        ground_truth_prob = ((summed_boundaries / num_annotators) >= 0.2).astype(np.float32)
        
        # 5. 应用Albumentations变换
        if self.transforms:
            # 将两个掩码放入一个列表中传递
            augmented = self.transforms(image=image, masks= [ground_truth_prob])
            image_tensor = augmented['image'].to(torch.float32) / 255.0
            # 从返回的'masks'列表中解包
            transformed_bound_mask = augmented['masks'][0]
        else:
            # 手动转换
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
            transformed_bound_mask = torch.from_numpy(ground_truth_prob)
            
        # 6. 确认数据类型
        # 'Boundary' mask 最好是浮点型 (Float), 用于二元损失函数 (如BCE, Dice)
        boundary_tensor = transformed_bound_mask.unsqueeze(0).float()

        # 7. 返回包含两个掩码的字典
        return {
            "img_id": img_id,
            "image": image_tensor,              # [3, 1024, 1024]
            "mask": boundary_tensor,            # [1, 1024, 1024]
        }

def get_bsd500_albumentation_transforms():
    """定义用于BSDS500数据集的Albumentations变换"""
    train_transforms = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()  # 将图像和掩码转换为PyTorch张量
    ])
    # 验证和测试集通常不做数据增强，只做尺寸调整
    val_test_transforms = A.Compose([
        A.Resize(1024, 1024),
        ToTensorV2()
    ])
    return train_transforms, val_test_transforms

def bsd500_create_dataset(bsd_root_dir='./data/BSD500'):
    """
    为BSDS500数据集创建训练、验证和测试Dataset实例。
    BSDS500有预定义的划分，我们直接使用。
    """
    train_transforms, val_test_transforms = get_bsd500_albumentation_transforms()

    train_dataset = BSDS500BoundaryDataset(root_dir=bsd_root_dir, split='train', transforms=train_transforms)
    val_dataset = BSDS500BoundaryDataset(root_dir=bsd_root_dir, split='val', transforms=val_test_transforms)
    test_dataset = BSDS500BoundaryDataset(root_dir=bsd_root_dir, split='test', transforms=val_test_transforms)
    
    print(f"BSDS500 Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test images.")
    
    return train_dataset, val_dataset, test_dataset

def log_info_bsd500_dataset(train_dataset, val_dataset, test_dataset):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"BSDS500 Dataset Sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    id, img, mask = train_dataset[0]['img_id'], train_dataset[0]['image'], train_dataset[0]['mask']
    logging.info(f"Sample from Train Dataset - ID: {id}, Image shape: {img.shape}, Mask shape: {mask.shape}, Mask unique values: {torch.unique(mask)}")

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
        # label = np.unique(mask_np)[1]   # label不能这样定义，所以在create_neu_dataset_stratified中重新手动分离

        # 原数据集将不同缺陷的掩码值分别设为了1-3，但是整个图片是单通道的，所以这里二值化为全为1进行处理
        # 将所有大于0的掩码值设为1
        mask_np = (mask_np > 0).astype(np.float32)

        img_1024 = self.letterbox_imagenp(image_np, [1024, 1024])
        mask_1024 = self.letterbox_mask_1ch(mask_np, [1024, 1024])
        bbox = get_bounding_box(mask_1024)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        if self.transforms:
            augmented = self.transforms(image=img_1024, mask=mask_1024)
            image_tensor = augmented['image'] / 255.0
            mask_tensor = augmented['mask']
            mask_tensor = mask_tensor.float()
        return {
            "image": image_tensor.float(), "mask": mask_tensor.float(),
            "bbox": bbox_tensor.float(), "image_id": image_id, 
            # "label": label
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

def generate_neu_keys(mask_dir, mask_files):
    stratify_keys = []
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask_np = np.array(Image.open(mask_path))
        present_labels = sorted([label for label in np.unique(mask_np) if label > 0])
        key = "_".join(map(str, present_labels)) if present_labels else "background"
        stratify_keys.append(key)
    return stratify_keys

def create_neu_dataset_stratified(): 
    """
    修改完bug
    """
    train_images_dir = 'data/NEU_Seg-main/images/training'
    train_annotations_dir = 'data/NEU_Seg-main/annotations/training'
    test_images_dir = 'data/NEU_Seg-main/images/test'
    test_annotations_dir = 'data/NEU_Seg-main/annotations/test'

    train_transforms, test_transforms = get_neu_albumentation_transforms()

    # 手动扫描掩码，为分层抽样创建标签
    all_train_mask_files = sorted([f for f in os.listdir(train_annotations_dir) if f.endswith('.png')])
    all_test_mask_files = sorted([f for f in os.listdir(test_annotations_dir) if f.endswith('.png')])

    stratify_keys_train = generate_neu_keys(train_annotations_dir, all_train_mask_files)
    stratify_keys_test = generate_neu_keys(test_annotations_dir, all_test_mask_files)
    
    print("策略键生成完毕。")
    print(f"原始训练集中的类别组合分布: \n{Counter(stratify_keys_train)}")
    print(f"原始测试集中的类别组合分布: \n{Counter(stratify_keys_test)}")

    # 创建完整数据集和索引
    full_train_dataset = NEU_SEG_Dataset(train_images_dir, train_annotations_dir, transforms=train_transforms)
    test_dataset = NEU_SEG_Dataset(test_images_dir, test_annotations_dir, transforms=test_transforms)
    indices = list(range(len(full_train_dataset)))


    # 使用 scikit-learn 进行分层抽样，得到索引
    val_split_ratio = 0.25
    # test_size 参数指定了验证集的大小， stratify 参数传入标签列表，函数会根据此列表进行分层， random_state 相当于 torch.Generator().manual_seed()，用于复现结果
    train_indices, val_indices = train_test_split(indices, test_size=val_split_ratio, stratify=stratify_keys_train, random_state=42)

    # 验证划分结果 (可选但推荐) ---
    print(f"\n验证划分结果")
    train_keys = [stratify_keys_train[i] for i in train_indices]
    val_keys = [stratify_keys_train[i] for i in val_indices]
    print(f"\n划分后训练集的分布: \n{Counter(train_keys)}")
    print(f"\n划分后验证集的分布: \n{Counter(val_keys)}")

    # 创建两个数据集实例，分别绑定不同的 transforms
    full_train_dataset = NEU_SEG_Dataset(train_images_dir, train_annotations_dir, transforms=train_transforms)
    full_val_dataset   = NEU_SEG_Dataset(train_images_dir, train_annotations_dir, transforms=test_transforms)
    test_dataset       = NEU_SEG_Dataset(test_images_dir, test_annotations_dir, transforms=test_transforms)

    # 使用索引构建 Subset
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset   = Subset(full_val_dataset, val_indices)
    return train_dataset, val_dataset, test_dataset


def debug_neu_dataset_info(train_dataset, val_dataset, test_dataset):
    print("="*40)
    print(f"[数据统计] 训练集图片: {len(train_dataset)} 张")
    print(f"[数据统计] 验证集图片: {len(val_dataset)} 张")
    print(f"[数据统计] 测试集图片: {len(test_dataset)} 张")
    print("="*40)

    idx = random.randint(0, len(train_dataset) - 1)
    sample = train_dataset[idx]
    img, mask, bbox, image_id = sample['image'], sample['mask'], sample['bbox'], sample['image_id']
    # img, mask, bbox, image_id, label = sample['image'], sample['mask'], sample['bbox'], sample['image_id'], sample['label']  # 原来的
    print("="*40)
    print(f"debug shapes")
    print(f"img: {img.shape}, mask: {mask.shape}, bbox: {bbox}, image_id: {image_id}")
    print("="*40)

    print("="*40)
    print(f"debug data types and data ranges")
    print(f"img: {img.dtype}, mask: {mask.dtype}, bbox: {bbox.dtype}")
    print(f"img: {img.min()} ~ {img.max()}, mask: {mask.unique()}")
    print("="*40)

def bsd500_debug_and_visulize(dataset):
    """
    调试和可视化函数，适用于任何返回同样字典结构的Dataset
    """
    # 从数据集中随机选择一个样本
    idx = random.randint(0, len(dataset) - 1)
    # Subset对象需要先访问其内部dataset和indices
    if isinstance(dataset, torch.utils.data.Subset):
        sample = dataset.dataset[dataset.indices[idx]]
    else:
        sample = dataset[idx]
        
    img_id, img, bd_mask =  sample['img_id'], sample['image'], sample['mask']

    print(f"Image tensor shape: {img.shape}, dtype: {img.dtype}, range: {img.min():.2f} - {img.max():.2f}")
    print(f"boundary_mask tensor shape: {bd_mask.shape}, dtype: {bd_mask.dtype}, unique values: {torch.unique(bd_mask)}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0)) # C,H,W -> H,W,C
    plt.title('Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # 如果mask有通道维度，则去掉
    if bd_mask.dim() == 3:
        bd_mask = bd_mask.squeeze(0)
    plt.imshow(bd_mask, cmap='gray')
    plt.title('Ground Truth Boundary')
    plt.axis('off')
    
    plt.show()

class Retina_Dataset_ft(SegDatasetForFinetune):
    """
    Retina dataset for fine-tuning
    """
    def __init__(self, images_dir, annotations_dir, transforms: A.Compose):
        ...
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.png')])

        if len(self.image_files) != len(self.mask_files):
            raise RuntimeError(f"图像数量({len(self.image_files)})与掩码数量({len(self.mask_files)})不一致！")

        # 检查文件名是否一一对应 
        for i, (img_f, mask_f) in enumerate(zip(self.image_files, self.mask_files)):
            if os.path.splitext(img_f)[0] != os.path.splitext(mask_f)[0]:
                raise ValueError(f"检测到不匹配的文件对：\nImg: {img_f}\nMask: {mask_f}\n请检查数据集排序。")
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):

        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.annotations_dir, self.mask_files[idx])

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)   # (h, w, 3)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)              # (h, w)    cv2 returns numpy.ndarray with dtype uint8

        image_id = self.image_files[idx]
        mask_id = self.mask_files[idx]
        if image_id != mask_id:
            raise ValueError(f"图像ID和掩码ID不匹配: {image_id} vs {mask_id}")

        resized_img = self.letterbox_imagenp(image, [1024, 1024])      # (1024, 1024, 3)
        resized_mask = self.letterbox_mask_1ch(mask, [1024, 1024])     # (1024, 1024)
        
        bbox = get_bounding_box(resized_mask)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32) 

        if self.transforms:
            augmented = self.transforms(image=resized_img, mask=resized_mask)
            image_tensor = augmented['image'] / 255.0          
            mask_tensor = (augmented['mask']) / 255.0
            mask_tensor = (mask_tensor > 0).float()   # binary mask
        return {"image": image_tensor.float(),  "mask": mask_tensor.float(), "bbox": bbox_tensor.float(), "image_id": image_id}

def create_retina_dataset_ft():
    train_transforms, test_transforms = general_albumentation_transforms_for_finetune()

    retina_train_images_dir = 'data/Retina_Blood_Vessel/train/image'           
    retina_train_annotations_dir = 'data/Retina_Blood_Vessel/train/mask'

    retina_test_images_dir = 'data/Retina_Blood_Vessel/test/image'
    retina_test_annotations_dir = 'data/Retina_Blood_Vessel/test/mask'

    train_dataset_with_aug = Retina_Dataset_ft(retina_train_images_dir, 
                                                retina_train_annotations_dir, 
                                                transforms=train_transforms)
    
    # no augmentation
    val_dataset_no_aug = Retina_Dataset_ft(retina_train_images_dir, 
                                            retina_train_annotations_dir, 
                                            transforms=test_transforms)

    # split
    dataset_size = len(train_dataset_with_aug)  # 80
    indices = list(range(dataset_size))
    split = 20                                  # 20

    # 打乱索引
    np.random.seed(42)          # seed
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    retina_train_subset = Subset(train_dataset_with_aug, train_indices)
    retina_val_subset = Subset(val_dataset_no_aug, val_indices)

    retina_test_set = Retina_Dataset_ft(  retina_test_images_dir, 
                                            retina_test_annotations_dir, 
                                            transforms=test_transforms)

    print(f"训练集大小: {len(retina_train_subset)}")                   # 60
    print(f"验证集大小: {len(retina_val_subset)}")                     # 20
    print(f"测试集大小: {len(retina_test_set)}")                       # 20

    return retina_train_subset, retina_val_subset, retina_test_set

if __name__ == "__main__":
    neu_train_dataset, neu_val_dataset, neu_test_dataset = create_neu_dataset_stratified()
