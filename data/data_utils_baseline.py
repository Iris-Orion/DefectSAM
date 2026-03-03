### --- 非lora相关的baseline数据集相关代码 --- ###
import os
import torch
import cv2
import random
import numpy as np
import albumentations as A

from PIL import Image
from typing import Tuple, List, Dict
from collections import Counter
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset, Subset

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
        # 将所有大于0的掩码值设为1
        mask_np = (mask_np > 0).astype(np.float32)
        if self.transforms:
            augmented = self.transforms(image=image_np, mask=mask_np)
            image = augmented['image'] / 255.0       # [3, 256, 256]
            mask = (augmented['mask']).unsqueeze(0)  # [1, 256, 256]
        return image, mask, image_id

# 定义变换
def get_neu_albumentation_transforms():
    train_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),    ##原先mask的是0, 255， image的值在0-255范围之间
        ToTensorV2()
    ])
    test_transforms = A.Compose([
        A.Resize(height=256, width=256),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transforms, test_transforms

def neu_bsl_create_dataset():
    train_images_dir = 'data/NEU_Seg-main/images/training'                  # 数据集路径
    train_annotations_dir = 'data/NEU_Seg-main/annotations/training'
    test_images_dir = 'data/NEU_Seg-main/images/test'
    test_annotations_dir = 'data/NEU_Seg-main/annotations/test'

    train_transforms, test_transforms = get_neu_albumentation_transforms()
    
    # 手动扫描掩码，为分层抽样创建标签
    all_mask_files = sorted([f for f in os.listdir(train_annotations_dir) if f.endswith('.png')])
    stratify_keys = []
    print("正在为分层抽样生成策略键...")
    for mask_file in all_mask_files:
        mask_path = os.path.join(train_annotations_dir, mask_file)
        mask_np = np.array(Image.open(mask_path))
        present_labels = sorted([label for label in np.unique(mask_np) if label > 0])
        key = "_".join(map(str, present_labels)) if present_labels else "background"
        stratify_keys.append(key)
    
    print("策略键生成完毕。")
    print(f"原始训练集中的类别组合分布: \n{Counter(stratify_keys)}")
        
    indices = list(range(len(all_mask_files)))
    val_split_ratio = 0.25     # 这里修复了之前的bug, 之前是0.2, 和finetune创建数据集时不一致  
    train_indices, val_indices = train_test_split(indices, test_size=val_split_ratio, stratify=stratify_keys, random_state=42)# 保证结果可复现
    
    # (可选但推荐) 验证划分结果
    train_keys = [stratify_keys[i] for i in train_indices]
    val_keys = [stratify_keys[i] for i in val_indices]
    print(f"\n划分后训练集的分布: \n{Counter(train_keys)}")
    print(f"\n划分后验证集的分布: \n{Counter(val_keys)}")

    full_dataset_for_train = NEU_SEG_Dataset_BSL(train_images_dir, train_annotations_dir, transforms=train_transforms)
    full_dataset_for_val = NEU_SEG_Dataset_BSL(train_images_dir, train_annotations_dir, transforms=test_transforms)
    test_dataset = NEU_SEG_Dataset_BSL(test_images_dir, test_annotations_dir, transforms=test_transforms)
    
    train_dataset = Subset(full_dataset_for_train, train_indices)
    val_dataset   = Subset(full_dataset_for_val, val_indices)
    return train_dataset, val_dataset, test_dataset

class SDsaliency900Dataset(Dataset):
    """
    无需box prompt的baseline使用
    """
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

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = self.labels[idx]
  
        img_path = os.path.join(self.source_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.ground_truth_dir, mask_name)

        image = Image.open(img_path).convert('RGB')     ## 原图像是灰度图 ---> 显式的转为rgb  原图的size是(200 * 200)
        mask = Image.open(mask_path).convert('L')       ## mask是灰度图

        image_np = np.array(image)
        mask_np = np.array(mask)

        if self.transforms:
            augmented = self.transforms(image=image_np, mask=mask_np)
            image = augmented['image'] / 255.0          # [3, 256, 256]
            mask = (augmented['mask'] / 255.0).unsqueeze(0)  # 将mask归一化到0-1之间, albumentation不会自动归一化  [1, 256, 256]
        return image.float(), mask.float(), label


# 定义变换
def get_sd900_albumentation_transforms():
    train_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),    ##原先mask的是0, 255， image的值在0-255范围之间
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Resize(height=256, width=256),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transforms, val_transforms

# 打印类别分布
def get_label_distribution(dataset):
    labels_subset = [dataset.dataset.labels[idx] for idx in dataset.indices]
    label_counts = Counter(labels_subset)
    sorted_counts = dict(sorted(label_counts.items()))
    return sorted_counts


def test_sd900_info(train_dataset:Dataset):
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
    # 设置目录路径
    sd900_src_dir = './data/sd900/Source Images'
    sd900_gt_dir = './data/sd900/Ground truth'

    # 创建数据集
    full_dataset = SDsaliency900Dataset(
        source_dir=sd900_src_dir,
        ground_truth_dir=sd900_gt_dir,
        transforms = None
    )
    labels = np.array(full_dataset.labels)
    # print(full_dataset.image_names)

    # 第一次划分：分出60%作为训练集，剩下40%作为临时集
    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_indices, tmp_indices = next(split_1.split(np.zeros(len(labels)), labels))

    # 从原始标签中获取临时集的标签
    tmp_labels = labels[tmp_indices]

    # 第二次划分：将40%的临时集对半分为验证集和测试集 (各占总数的20%)
    # test_size=0.5 表示将 tmp_indices 对半划分
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
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

    return train_dataset, val_dataset, test_dataset, labels

class MagneticTileDataset_Baseline(Dataset):
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

        augmented = self.transforms(image=image_np, mask=mask_np)
        aug_img_np = augmented["image"]        
        aug_mask_1ch_np = augmented["mask"]    

        resize_img_np = self.letterbox_image(aug_img_np, [256, 256])             # sam必须以1024x1024输入
        resize_mask_np = self.letterbox_mask_1ch(aug_mask_1ch_np, [256, 256])      # sam的low res输出是256x256，该数据集大小本身就比较小
        
        resize_img_tensor = torch.from_numpy(resize_img_np).permute(2, 0, 1).float() / 255.0   # [3, 256, 256]
        resize_mask_tensor = (torch.from_numpy(resize_mask_np) > 0.5).unsqueeze(0).float()     # [1, 256, 256]

        label = self.labels[idx]
        return resize_img_tensor, resize_mask_tensor, label
                
    
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
    classes, class_to_idx = magtile_find_classes(root_dir)# 使用 find_classes 函数获取类别名和标签映射
    image_paths = []                                   # 初始化图像、掩码路径和标签列表
    mask_paths = []
    labels = []
    
    # 遍历所有类别，收集图像和掩码路径
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls, 'Imgs')
        if not os.path.isdir(cls_dir):
            print(f"Warning: Directory {cls_dir} does not exist. Skipping class {cls}.")
            continue
        # for file in os.listdir(cls_dir):
        for file in sorted(os.listdir(cls_dir)): # 添加排序 确保不同环境下的一致性
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
    # 检查数据一致性
    assert len(image_paths) == len(mask_paths) == len(labels), \
        "Mismatch in number of images, masks, and labels."
    return image_paths, mask_paths, labels

def mag_albumentation_transforms():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])
    val_transforms = A.Compose([])
    return train_transforms, val_transforms

def debug_dataset_datatype(dataset: MagneticTileDataset_Baseline):
    idx = random.randint(0, len(dataset)-1)
    image, mask, label = dataset[idx]
    print(f"Image type: {image.dtype}, Mask type: {mask.dtype}, Label type: {type(label)}")
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Label: {label}")
    print(f"Image 值的范围: {torch.min(image)} ~ {torch.max(image)}, Mask unique: {torch.unique(mask)}")

# 打印类别分布以验证

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

    train_dataset = MagneticTileDataset_Baseline(train_image_paths, train_mask_paths, train_labels, transforms=train_transforms)
    val_dataset = MagneticTileDataset_Baseline(val_image_paths, val_mask_paths, val_labels, transforms=val_transforms)
    test_dataset = MagneticTileDataset_Baseline(test_image_paths, test_mask_paths, test_labels, transforms=val_transforms)

    print(f"Full dataset")
    get_mag_label_distribution(labels, classes)
    print(f"training dataset")
    get_mag_label_distribution(train_labels, classes)
    print(f"Val dataset")
    get_mag_label_distribution(val_labels, classes)
    print(f"Test dataset")
    get_mag_label_distribution(test_labels, classes)

    return train_dataset, val_dataset, test_dataset

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

        # 检查文件名是否一一对应 
        for i, (img_f, mask_f) in enumerate(zip(self.image_files, self.mask_files)):
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
        mask_id = self.mask_files[idx]

        image_np = np.array(image)
        mask_np = np.array(mask)

        mask_np = (mask_np > 0).astype(np.float32)
        if self.transforms:
            augmented = self.transforms(image=image_np, mask=mask_np)
            image = augmented['image'] / 255.0          # [3, 256, 256]
            mask = (augmented['mask']).unsqueeze(0)     # [1, 256, 256]
        return image, mask, image_id

def general_albumentation_transforms():
    train_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),    ##原先mask的是0, 255， image的值在0-255范围之间
        ToTensorV2()
    ])
    test_transforms = A.Compose([
        A.Resize(height=256, width=256),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transforms, test_transforms

def create_retina_dataset_baseline():
    train_transforms, test_transforms = general_albumentation_transforms()

    retina_train_images_dir = 'data/Retina_Blood_Vessel/train/image'           
    retina_train_annotations_dir = 'data/Retina_Blood_Vessel/train/mask'

    retina_test_images_dir = 'data/Retina_Blood_Vessel/test/image'
    retina_test_annotations_dir = 'data/Retina_Blood_Vessel/test/mask'

    train_dataset_with_aug = Retina_Dataset_Bsl(retina_train_images_dir, 
                                                retina_train_annotations_dir, 
                                                transforms=train_transforms)
    
    # no augmentation
    val_dataset_no_aug = Retina_Dataset_Bsl(retina_train_images_dir, 
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

    retina_test_set = Retina_Dataset_Bsl(  retina_test_images_dir, 
                                            retina_test_annotations_dir, 
                                            transforms=test_transforms)

    print(f"训练集大小: {len(retina_train_subset)}")                   # 60
    print(f"验证集大小: {len(retina_val_subset)}")                     # 20
    print(f"测试集大小: {len(retina_test_set)}")                       # 20

    return retina_train_subset, retina_val_subset, retina_test_set

if __name__ == "__main__":
    # neu_bsl_create_dataset()
    create_retina_dataset_baseline()