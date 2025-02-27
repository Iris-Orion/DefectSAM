import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F
from typing import List, Tuple, Dict

# random.seed(23)

def rle2mask(image_id, df):
    """
    输入: img_id   xxxxxxx.jpg
    输出: jpg对应的四通道的掩码
    输出的np.shape: [256, 1600, 4]   [H, W, C]
    """
    # 4 : 对应四种缺陷
    # RLE 的数据编码方式一般是“列优先”格式（Fortran-style）,即先填充列
    mask = np.zeros((256, 1600, 4), dtype=np.float32)
    
    image_df = df[df['ImageId'] == image_id]
    if (image_df['ClassId'] == 0).all():
        return mask 

    for _, row in image_df.iterrows():
        class_id = int(row['ClassId']) - 1  # numpy channel: 0 - 3
        rle = row['EncodedPixels']

        # 处理EncodedPixels为空，但是若不修改原始的train.csv，是不会存在nan情况的
        if pd.isna(rle):
            continue

        label = rle.split(" ")     # 这会return一个list
        positions = map(int, label[0::2])    # 起始位置
        lengths = map(int, label[1::2])      # 长度

        mask_flat = np.zeros(256 * 1600, dtype=np.float32)
        for pos, le in zip(positions, lengths):
            mask_flat[pos:(pos+le)] = 1

        mask[:, :, class_id] = mask_flat.reshape((256, 1600), order='F')
    
    return mask

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

def get_torchvision_transforms():
    train_transforms = transforms.Compose([
        # transforms.Resize()    ?? 是否需要resize
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.ColorJitter(brightness=[0, 0.2], contrast=[0, 0.2]),  ## 这个参数设置的不对，不要用
        transforms.ToTensor(),   # totensor 需要在 normalize之前
        # 来自imagenet的计算结果
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),   # totensor 需要在 normalize之前
        # 来自imagenet的计算结果
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, val_transforms

class SteelDataset(Dataset):
    def __init__(self, df, data_path, transforms: A.Compose):
        self.df = df
        self.root = data_path
        self.transforms = transforms
        self.fnames = self.df['ImageId'].unique()
 
    def __getitem__(self, idx):
        image_id = self.fnames[idx]   # 图片名字
        mask = rle2mask(image_id, self.df)  # 掩码
        image_path = os.path.join(self.root, "train_images", image_id)
        img = np.array(Image.open(image_path).convert("RGB"))  # 保证为rgb

        augmented = self.transforms(image=img, mask=mask)   # 调用的是albumentations库
        img = augmented["image"].float() / 255.0            # 归一化了
        mask = augmented["mask"].permute(2, 0, 1)   # [h, w, c] ---> [c, h, w]
        return img, mask, image_id
    
    def __len__(self):
        return len(self.fnames)

class SteelDataset_WithBoxPrompt(Dataset):
    """
    利用hugging 
    返回带有box提示的sevestal数据集用于微调sam
    img: torch tensor : [C, H, W]
    mask_tensor: [4, H, W]
    image_id: "str"
    bbox_tensor: tensor
    """
    def __init__(self, df, data_path, transforms: transforms.Compose):
        self.df = df
        self.root = data_path
        self.transforms = transforms
        self.fnames = self.df['ImageId'].unique()
    
    def preprocess(self, original_image: torch.Tensor):
        """
        Normalize pixel values and pad to a square input.
        这个来自segment anything的源码
        """
        # Normalize colors
        # batched_x = (batched_x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = original_image.shape[-2:]   # (h, w, c)
        padh = 1024 - h
        padw = 1024 - w
        original_image = F.pad(original_image, (0, padw, 0, padh))
        return original_image
    
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
    
    def letterbox_mask_4ch(self, mask: np.ndarray, size: tuple) -> np.ndarray:
        """
        对形状为 (H, W, 4) 的四通道掩码做 letterbox 操作，
        等比例缩放并居中填充到指定 size = (target_width, target_height)。
        
        参数:
            mask: np.ndarray, shape = (H, W, 4)
            size: (target_width, target_height), 例如 (1024, 1024)

        返回:
            result: np.ndarray, shape = (target_height, target_width, 4)
        """
        # 原图尺寸
        H, W, C = mask.shape
        assert C == 4, f"输入的掩码必须是4通道, 但拿到的是 {C} 通道。"

        # 目标尺寸
        target_w, target_h = size

        # 计算等比例缩放
        scale = min(target_w / W, target_h / H)
        new_w = int(W * scale)
        new_h = int(H * scale)

        # 最近邻插值缩放
        # 注意 OpenCV 的 resize 第二个参数是 (width, height)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 创建空白画布 (target_h, target_w, 4)，填充值设为0（通常表示背景）
        letterbox_result = np.zeros((target_h, target_w, 4), dtype=mask.dtype)

        # 计算粘贴位置，使其居中
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # 将缩放后的掩码贴到画布中
        letterbox_result[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = mask_resized

        return letterbox_result

    def __getitem__(self, idx):
        image_id = self.fnames[idx]                                         # 图片名字: 返回字符串
        mask = rle2mask(image_id, self.df)                                  # 四通道掩码shape: (256, 1600, 4)
        image_path = os.path.join(self.root, "train_images", image_id)
        img = Image.open(image_path).convert("RGB")                         # 保证为rgb: (256, 1600, 3)

        # 对图像进行transform
        resize_img = self.letterbox_image(img, [1024, 1024])
        resize_img_tensor = self.transforms(resize_img)                 # (C, H, W) tensor 这里调用的是torchvision transforms的形式

        mask = self.letterbox_mask_4ch(mask, [1024, 1024])              # (4, 1024, 1024)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)           # (256, 1600, 4) ---> (4, 256, 1600)
        combined_masks = torch.sum(mask_tensor, dim=0)                  # 按dim=0 sum, 不保留维度，通道C进行相加, [H, W]

        bbox = get_bounding_box(combined_masks)                         # torch.tensor([x_min, y_min, x_max, y_max]): size=(4)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        # TODO: 使用全部severstal数据集时，将无缺陷的掩码图像的bbox设为[0, 0, 0, 0]
        # 返回图像, mask, image_id，以及boxes作为prompt
        return {"image": resize_img_tensor.float(),
                "mask_4ch": mask_tensor,
                "combined_mask": combined_masks.float(),  
                "image_id": image_id, 
                "bbox": bbox_tensor.float()
                }
    
    def __len__(self):
        return len(self.fnames)

def create_datasets(train_df, 
                    val_df, 
                    data_path, 
                    train_transform,
                    val_transform):
    '''
    创建不返回box prompt的severstal数据集
    '''
    train_dataset = SteelDataset(train_df, data_path, transforms=train_transform)
    val_dataset = SteelDataset(val_df, data_path, transforms=val_transform)

    return train_dataset, val_dataset

    
def create_dataloaders(train_df, 
                       val_df, 
                       data_path, 
                       train_transform,
                       val_transform,
                       batch_size, 
                       num_workers):
    '''
    创建不返回box prompt的severstal数据集的dataloader
    '''
    train_dataset, val_dataset = create_datasets(train_df, val_df, data_path, train_transform, val_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_dataloader, val_dataloader
    
def traindf_preprocess(split_seed: int = 42, 
                       csv_path: str = "./data/severstal_steel_defect_detection/train.csv",
                       data_path: str = "./data/severstal_steel_defect_detection/train_images",
                       create_mini_dataset: bool = False,
                       mini_size: int = 1024):
    """
    处理原始的train.csv, 返回train_df和val_df dataframe
    并且将未有缺陷的图片也加入dataframe中
    """
    trainfolder_df = pd.read_csv(csv_path)

    # 所有的训练集图片（包含无缺陷的）
    all_train_images = os.listdir(data_path)

    # 首先获取唯一的 缺陷图像ImageId 列表
    defect_image_ids = trainfolder_df['ImageId'].unique()
    no_defect_image_ids = [img for img in all_train_images if img not in defect_image_ids]

    # 为无缺陷图像生成一个对应的DataFrame (class_id为0，EncodedPixels为空)
    no_defect_df = pd.DataFrame({
        'ImageId': no_defect_image_ids,
        'ClassId': [0] * len(no_defect_image_ids),           # 考虑用NaN代替
        'EncodedPixels': [None] * len(no_defect_image_ids)
    })

    full_df = pd.concat([trainfolder_df, no_defect_df], ignore_index=True)

    # 拆分数据集
    train_ids, val_ids = train_test_split(full_df['ImageId'].unique(), test_size=0.2, random_state=split_seed)
    
    # 创建mini数据集
    if create_mini_dataset:
        train_ids = random.sample(list(train_ids), mini_size)
        val_ids = random.sample(list(val_ids), mini_size // 4)

    train_df = full_df[full_df['ImageId'].isin(train_ids)]
    val_df = full_df[full_df['ImageId'].isin(val_ids)]

    # 统计 train_df 中每个 ClassId 的数量
    train_class_counts = train_df['ClassId'].value_counts().sort_index()

    # 统计 val_df 中每个 ClassId 的数量
    val_class_counts = val_df['ClassId'].value_counts().sort_index()

    print("Train Class Distribution:")
    print(train_class_counts)

    print("\nValidation Class Distribution:")
    print(val_class_counts)
    return train_df, val_df

def traindf_preprocess_onlydefect(split_seed: int=42, 
                       csv_path: str = "./data/severstal_steel_defect_detection/train.csv",
                       data_path: str = "./data/severstal_steel_defect_detection/train_images",
                       create_mini_dataset: bool = False,
                       mini_size: int = 1024):
    """
    处理原始的train.csv, 返回train_df和val_df dataframe
    只处理原来的带有缺陷的图像照片
    """
    trainfolder_df = pd.read_csv(csv_path)

    # 首先获取唯一的 缺陷图像ImageId 列表
    defect_image_ids = trainfolder_df['ImageId'].unique()

    # 拆分数据集
    train_ids, val_ids = train_test_split(defect_image_ids, test_size=0.2, random_state=split_seed)
    
    # 创建mini数据集
    if create_mini_dataset:
        train_ids = random.sample(list(train_ids), mini_size)
        val_ids = random.sample(list(val_ids), mini_size // 4)

    train_df = trainfolder_df[trainfolder_df['ImageId'].isin(train_ids)]
    val_df = trainfolder_df[trainfolder_df['ImageId'].isin(val_ids)]

    # 统计 train_df 中每个 ClassId 的数量
    train_class_counts = train_df['ClassId'].value_counts().sort_index()

    # 统计 val_df 中每个 ClassId 的数量
    val_class_counts = val_df['ClassId'].value_counts().sort_index()

    print("Train Class Distribution:")
    print(train_class_counts)

    print("\nValidation Class Distribution:")
    print(val_class_counts)
    return train_df, val_df


def mask2rle(img):
    '''
    
    '''

def random_visualize(dataset, num_pics):
    random_samples_idx = random.sample(range(len(dataset)), k=num_pics)
    fig, ax = plt.subplots(nrows=num_pics, ncols=5, figsize=(16, 8))
    for i, targ_sample in enumerate(random_samples_idx):
        image, masks = dataset[targ_sample][0], dataset[targ_sample][1]
        # mask的shape是[4, 256, 1600]，所以每个通道的shape就是[256, 1600]，所以mask不需要permute
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

def test_create_dataset():
    p = Path("data/severstal_steel_defect_detection")
    train_df, val_df = traindf_preprocess_onlydefect()
    print(train_df.head())
    print(val_df.head())
    train_transform, val_transform = get_albumentations_transforms()
    train_dataset, val_dataset = create_datasets(train_df, 
                                                 val_df, 
                                                 data_path=p, 
                                                 train_transform=train_transform,
                                                 val_transform=val_transform)
    train_dataloader, val_dataloader = create_dataloaders(  train_df, 
                                                            val_df,
                                                            data_path=p, 
                                                            train_transform=train_transform,
                                                            val_transform=val_transform, 
                                                            batch_size=4,
                                                            num_workers=4)
    print("------test data type and shape of dataset------")
    img0, mask0, imgid0 = train_dataset[0]
    print(type(img0), type(mask0), type(imgid0))
    print(f"imgid: {imgid0}")
    print(f"img0.dtype: {img0.dtype} ||| mask0.dtype: {mask0.dtype}")
    print(f"img0.shape: {img0.shape} ||| mask0.shape: {mask0.shape}")
    print(f"img0.data.range: {img0.min()} ~ {img0.max()} ||| mask0.data.range: {mask0.unique()}")
    print("------test dataset return data type end------")

def visualize_dataset_and_groundtruth_mask(img_count=1):
    p = Path("data/severstal_steel_defect_detection")
    train_df, val_df = traindf_preprocess_onlydefect()
    print(train_df.head())
    print(val_df.head())
    train_transform, val_transform = get_albumentations_transforms()
    train_dataset, val_dataset = create_datasets(train_df, 
                                                 val_df, 
                                                 data_path=p, 
                                                 train_transform=train_transform,
                                                 val_transform=val_transform)
    random_samples_idx = random.sample(range(len(train_dataset)), k=img_count)
    fig, ax = plt.subplots(nrows=img_count, ncols=1, figsize=(16, 8))
    for i, targ_sample in enumerate(random_samples_idx):
        image, masks = train_dataset[targ_sample][0], train_dataset[targ_sample][1]
        # mask的shape是[4, 256, 1600]，所以每个通道的shape就是[256, 1600]，所以mask不需要permute
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
    """
    input: torch.dataset
    """
    # TODO: steel_dataset_withboxPrompt的逻辑返回了一个字典，如果要使用这个函数，修改下面的逻辑
    img0, mask0, imgid0, bbox = steel_dataset_withboxPrompt[random.randint(0, len(steel_dataset_withboxPrompt))]
    print(f"img0 type: {type(img0)}, img0 shape: {img0.shape}")
    print(f"mask type : {type(mask0)}, mask shape: {mask0.shape}")
    img0_plt_format = img0.permute(1, 2, 0)

    cmaps = ['Reds', 'Greens', 'Blues', 'Purples']

    mask = torch.sum(mask0, dim=0, keepdim=True)
    print(f"comined mask (keep dim) shape: {mask.shape}")

    # bbox = get_bounding_box(mask)
    print(f"img get bbox position: {bbox}")

    fig, ax = plt.subplots()
    ax.imshow(img0_plt_format)
    ax.imshow(mask0[0], cmap=cmaps[0], alpha = 0.8 * (mask0[0] > 0))
    ax.imshow(mask0[1], cmap=cmaps[1], alpha = 0.8 * (mask0[1] > 0))
    ax.imshow(mask0[2], cmap=cmaps[2], alpha = 0.8 * (mask0[2] > 0))
    ax.imshow(mask0[3], cmap=cmaps[3], alpha = 0.8 * (mask0[3] > 0))
    ax.set_title(f"{imgid0} with mask")

    plt.figure()
    plt.imshow(img0_plt_format)
    plt.imshow(mask.permute(1, 2, 0).cpu().numpy(), cmap='Oranges', alpha=(0.8 * (mask > 0).float()).squeeze(0).cpu().numpy())
    show_box(bbox, ax)

def get_bounding_box(ground_truth_map):
    """
    从ground_truth中获得bbox
    输入:ground_truth_map 是一个合并的掩码 (256, 1600)   tensor格式
    TODO: severstal数据集的ground_truth_map不是合并的, 根据(4, 256, 1600)判断每个channel中的box
    """
    # 先转为np格式
    # ground_truth_map_np = ground_truth_map.cpu().numpy()
    # assert type(ground_truth_map_np) == np.ndarray, 'convert to np plz'
    if torch.sum(ground_truth_map)== 0:
        bbox = [0, 0, 0, 0]
    # 从掩码中得到bounding box
    else:
        y_indices, x_indices = np.where(ground_truth_map > 0)   # np.where 支持  ground_truth_map是tensor输入
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]
    return bbox

def show_box(box, ax):
    """
    copy from segment-anything official repo
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

def prepare_image(image, transform, device):
    """
    已经弃用的函数
    resize输入的图像到(1024, 1024), 此函数的逻辑调用ResizeLongestSide函数, 
    但是resize到指定大小的逻辑不佳
    """
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()

# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def magtile_get_all_imgmsk_paths(root_dir):
    classes, class_to_idx = find_classes(root_dir)# 使用 find_classes 函数获取类别名和标签映射
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

#  simple test
def old_test():
    data_path = Path("data/")
    image_path = data_path / "severstal_steel_defect_detection"

    train_dir = image_path / "train_images"
    test_dir = image_path / "test_images"

    train_df = pd.read_csv("./data/severstal_steel_defect_detection/train.csv")
    submission_df = pd.read_csv("./data/severstal_steel_defect_detection/sample_submission.csv")

    image_path_list = list(image_path.glob("*/*.jpg"))

    random_image_path = random.choice(image_path_list)
    img = Image.open(random_image_path)

    chosen_image_path = train_dir / "0002cc93b.jpg"
    img2 = Image.open(chosen_image_path)


    # mask = rle2mask('008ef3d74.jpg', train_df.head(20))
    # image_id = '0002cc93b.jpg'
    idx = random.choice(train_df.index)
    image_id = train_df.iloc[idx]['ImageId']
    mask = rle2mask(image_id, train_df)

    img_testmask = Image.open(f'data/severstal_steel_defect_detection/train_images/{image_id}')

    # 可视化
    fig, axes = plt.subplots(5, 1, figsize=(30, 15))
    axes[0].imshow(img_testmask)
    axes[0].set_title(img_testmask)
    axes[0].axis('off')
    for i in range(4):
        axes[i+1].imshow(mask[:, :, i], cmap='gray')
        axes[i+1].set_title(f"Class {i+1} Mask")
        axes[i+1].axis('off')

    plt.suptitle(f"Masks for Image ID: {image_id}")
    plt.show()

    manual_transforms = transforms.Compose([
        # transforms.Resize()    ?? 是否需要resize
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=[0, 0.2], contrast=[0, 0.2]),
        transforms.ToTensor(),   # totensor 需要在 normalize之前
        # 来自imagenet的计算结果
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


if __name__ == '__main__':
    # old_test()
    test_create_dataset()

    train_df, val_df = traindf_preprocess(create_mini_dataset=False, mini_size=256)
    data_path = "./data/severstal_steel_defect_detection"
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = SteelDataset_WithBoxPrompt(train_df, data_path=data_path, transforms=train_transforms)
    val_dataset = SteelDataset_WithBoxPrompt(val_df, data_path=data_path, transforms=val_transforms)

    img0, mask0, imgid0 = val_dataset[0]["image"], train_dataset[0]["combined_mask"], train_dataset[0]["image_id"]
    print(type(img0), type(mask0), type(imgid0))
    print(f"imgid: {imgid0}")
    print(f"img0.dtype: {img0.dtype} ||| mask0.dtype: {mask0.dtype}")
    print(f"img0.shape: {img0.shape} ||| mask0.shape: {mask0.shape}")
    print(f"img0.data.range: {img0.min()} ~ {img0.max()} ||| mask0.data.range: {mask0.unique()}")
    print("------test dataset return data type end------")
