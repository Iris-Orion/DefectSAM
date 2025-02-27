### 制作sd900数据集的dataset 和 dataloader
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import logging
import modelLib
from utils import compute_dice_score, compute_iou_score
import argparse
import monai
from hf_finetune_engine import print_trainable_parameters
from helper_function import set_seed
from datetime import datetime
import pytz
from utils import save_model

# 定义 Dataset
class SDsaliency900Dataset(Dataset):
    def __init__(self, source_dir, ground_truth_dir, transforms: A.Compose):
        self.source_dir = source_dir
        self.ground_truth_dir = ground_truth_dir
        self.transforms = transforms

        self.image_names = [f for f in os.listdir(source_dir) if f.endswith('.bmp')]
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

        image = Image.open(img_path).convert('RGB')   ## 原图像是灰度图 ---> 显式的转为rgb  原图的size是(200 * 200)
        mask = Image.open(mask_path).convert('L')       ## mask是灰度图

        image_np = np.array(image)
        mask_np = np.array(mask)

        if self.transforms:
            augmented = self.transforms(image=image_np, mask=mask_np)
            image = augmented['image']
            mask = augmented['mask'] / 255.0  # 将mask归一化到0-1之间, albumentation不会自动归一化
        return image.float(), mask.float(), label

# 定义变换
def get_sd900_albumentation_transforms():
    train_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),    ##原先mask的是0, 255， image的值在0-255范围之间
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transforms, val_transforms

# 打印类别分布
def get_label_distribution(dataset):
    labels_subset = [dataset.dataset.labels[idx] for idx in dataset.indices]
    label_counts = Counter(labels_subset)
    sorted_counts = dict(sorted(label_counts.items()))
    return sorted_counts

def test_sdd900dataset_size():
    image, mask, label = train_dataset[0]
    print(f"debug信息: image.shape: {image.shape}, mask.shape: {mask.shape}, label: {label}")
    print(f"debug信息: image的值的类型: {torch.unique(image)}")
    print(f"debug信息: mask的值的类型: {torch.unique(mask)}")

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='训练模型的args选择')

# 添加参数
parser.add_argument('--batch_size', type=int, default=24, help='批量大小')
parser.add_argument('--num_epochs', type=int, default=25, help='训练轮数')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
parser.add_argument('--use_amp', action='store_true', help='是否使用自动混合精度')

parser.add_argument('--patience', type=int, default=10, help='早停容忍的epoch数')
parser.add_argument('--min_delta', type=float, default=0.001, help='验证指标提升的最小阈值')
parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')

# 解析参数
args = parser.parse_args()
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
PATIENCE = args.patience
MIN_DELTA = args.min_delta
WEIGHT_DECAY = args.weight_decay

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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 替换 print 语句
logging.info(f"训练集类别分布: {get_label_distribution(train_dataset)}")
logging.info(f"测试集类别分布: {get_label_distribution(test_dataset)}")
test_sdd900dataset_size()

# model = modelLib.build_segformer_model(
#     encoder_name="mit_b0",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1,
#     activation=None
# )

# model = modelLib.get_smp_deeplabv3plus(
#     encoder_name="efficientnet-b0",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1,
#     activation=None
# )

# model = modelLib.get_smp_unet(
#     encoder_name="resnet34",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1,
#     activation=None
# )
model = modelLib.get_vanilla_unet(n_channels=3, num_classes=1)
# meddiff_model = modelLib.get_medsegdiff_model(img_size=256)  # medsegdiff模型
# model = meddiff_model

# 定义损失函数和优化器
# criterion = nn.BCEWithLogitsLoss()
criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练循环
num_epochs = NUM_EPOCHS
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def baseline_engine(model, device):
    shanghai_tz = pytz.timezone('Asia/Shanghai')        # 设置时区为亚洲/上海
    model.to(device)
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳
    best_val_dice = -1.0    # 初始化最佳验证 dice 分数
    no_improve_epochs = 0   # 连续无改进的 epoch 计数

    results = {
        "train_loss": [],
        "train_dicescore": [],
        "train_ious": [],
        "test_loss": [],
        "test_dicescore": [],
        "test_ious": []
    }
    model.train()
    print_trainable_parameters(model)
    for epoch in range(num_epochs):
        epoch_loss, train_dicescore, train_ious = 0, 0, 0
        for images, masks, labels in train_loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)

            outputs = model(images)
            # logging.info(f"outputs.shape: {outputs.shape}, masks.shape: {masks.shape}")
            loss = criterion(outputs, masks)
            ## 计算dice 和 miou
            train_dicescore += compute_dice_score(outputs, masks)
            train_ious += compute_iou_score(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_dicescore = train_dicescore / len(train_loader)
        train_ious = train_ious / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] \nTrain Loss: {train_loss:.4f}, Train Dice: {train_dicescore:.4f}, Train IoU: {train_ious:.4f}')

        # 验证
        model.eval()
        val_loss, val_dicescore, val_ious = 0, 0, 0
        with torch.no_grad():
            for images, masks, labels in test_loader:
                images = images.to(device)
                masks = masks.unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                ## 计算dice 和 miou
                val_dicescore += compute_dice_score(outputs, masks)
                val_ious += compute_iou_score(outputs, masks)
            avg_val_loss = val_loss / len(test_loader)
            val_dicescore = val_dicescore / len(test_loader)
            val_ious = val_ious / len(test_loader)
            print(f'Validation Loss: {avg_val_loss:.4f}, Val Dice: {val_dicescore:.4f}, Val IoU: {val_ious:.4f}')
        results["train_loss"].append(train_loss)
        results["train_dicescore"].append(train_dicescore)
        results["train_ious"].append(train_ious)
        results["test_loss"].append(avg_val_loss)
        results["test_dicescore"].append(val_dicescore)
        results["test_ious"].append(val_ious)

        # early stopping逻辑：这里以验证 dice 为监控指标，只有当提升大于 min_delta 时才认为有改进
        if val_dicescore - best_val_dice > MIN_DELTA:
            best_val_dice = val_dicescore
            no_improve_epochs = 0
            print(f"验证 dice 改善到 {best_val_dice:.4f}, 保存模型...")
            # 当验证指标改善时保存模型
            save_model(
                hyperparameters=hyperparameters, 
                start_timestamp=start_timestamp,
                results=results, 
                model=model, 
                optimizer=optimizer,
                scaler=None,
                epoch=epoch+1,  # 记录epoch数，从 1 开始
                model_name="sd900_baseline_vanillaunet_",
                result_name="sd900_baseline_vanillaunet_",
                target_dir="./model_output/sd900_output",
                SAVE_HUGGINGFACE_PRETRAINED_MODEL=False
            )
        else:
            no_improve_epochs += 1
            print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
        
        # 若连续无改进达到 patience 数, 则提前停止训练
        if no_improve_epochs >= PATIENCE:
            print(f"验证指标连续 {PATIENCE} 个 epoch 无改善，提前停止训练。")
            break
        model.train()

def medsegdiff_baseline_engine(diffusion_model = model):
    diffusion_model.train()  # 将模型设置为训练模式
    print_trainable_parameters(model)  # 打印模型参数（或diffusion的内部模型参数）

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # 累计指标（如Dice、IoU）可根据需要添加，这里重点示范前向传播的修改
        for images, masks, labels in train_loader:
            # 将数据放到同一设备，注意：masks的通道数需与diffusion初始化时mask_channels匹配
            images = images.to(device)  # 原始未分割图，形状 (B, 3, H, W)
            masks = masks.to(device)    # 分割标签，形状应为 (B, 4, H, W)；如果你的 masks 只包含单通道，请转换至4通道

            # forward: 使用 diffusion 对象，将已知的分割结果（masks）和原始图像(images)同时输入，
            # loss = diffusion(segmented_imgs, input_imgs)
            loss = diffusion_model(masks, images)  # 这一步内部会进行扩散训练的处理逻辑

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f}')

        # 验证阶段
        diffusion_model.eval()
        val_loss = 0.0
        # 记录其它指标（例如 Dice、IoU）时，需适应扩散模型的输出和真实标签格式
        with torch.no_grad():
            for images, masks, labels in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                loss = diffusion_model(masks, images)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(test_loader)
            print(f'Validation Loss: {avg_val_loss:.4f}')
    return diffusion_model


def sam_zero_shot_engine():
    pass

if __name__ == '__main__':
    set_seed(42)

    hyperparameters = {
        # "task type": TASK_TYPE, 
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "optimizer": "Adam",
        "learing_rate": LEARNING_RATE,
        "loss function": "monai.DiceCELoss",
        "weight decay": WEIGHT_DECAY,
        # "best epoch": BEST_EPOCH,
        "Task name": "neu_seg_baseline_vanilla_unet",
        # "USE_AMP": USE_AMP,
        # "Mini dataset": create_mini_dataset,
        # 添加其他超参数...
    }
    baseline_engine(model=model, device=device)
    # diffusion_model = medsegdiff_baseline_engine()