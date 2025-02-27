import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import modelLib
import monai
import torch.optim as optim
from utils import compute_dice_score, compute_iou_score
from hf_finetune_engine import print_trainable_parameters
from tqdm import tqdm
import argparse
from datetime import datetime
import pytz
from utils import save_model
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

parser = argparse.ArgumentParser(description='训练模型的args选择')
parser.add_argument('--batch_size', type=int, default=4, help='训练的batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='训练的学习率')
parser.add_argument('--num_epochs', type=int, default=5, help='epochs')
# 新增 early stopping 参数
parser.add_argument('--patience', type=int, default=5, help='早停容忍的epoch数')
parser.add_argument('--min_delta', type=float, default=0.001, help='验证指标提升的最小阈值')
parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
args = parser.parse_args()
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
PATIENCE = args.patience
MIN_DELTA = args.min_delta
WEIGHT_DECAY = args.weight_decay

shanghai_tz = pytz.timezone('Asia/Shanghai')        # 设置时区为亚洲/上海

class NEU_SEG_Dataset(Dataset):
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
            image = augmented['image'] / 255.0
            mask = augmented['mask']  # 将mask归一化到0-1之间, albumentation不会自动归一化
            mask = mask.float()
        return image, mask, image_id

# 定义变换
def get_neu_albumentation_transforms():
    train_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),    ##原先mask的是0, 255， image的值在0-255范围之间s
        ToTensorV2()
    ])
    test_transforms = A.Compose([
        A.Resize(height=256, width=256),
        # A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transforms, test_transforms

def neu_seg_baseline_engine(model, device):
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳s
    results = {
        "train_loss": [],
        "train_dicescore": [],
        "train_ious": [],
        "test_loss": [],
        "test_dicescore": [],
        "test_ious": []
    }
    model.to(device)
    best_val_dice = -1.0    # 初始化最佳验证 dice 分数
    no_improve_epochs = 0   # 连续无改进的 epoch 计数

    model.train()
    print_trainable_parameters(model)
    for epoch in range(NUM_EPOCHS):
        epoch_loss, train_dicescore, train_ious = 0, 0, 0
        for images, masks, labels in tqdm(train_loader):
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)

            outputs = model(images)
            # print(f"Output shape: {outputs.shape}, Output range: {outputs.min().item()} ~ {outputs.max().item()}")
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
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] \nTrain Loss: {train_loss:.4f}, Train Dice: {train_dicescore:.4f}, Train IoU: {train_ious:.4f}')

        # 验证
        model.eval()
        test_loss, test_dicescore, test_ious = 0, 0, 0
        with torch.no_grad():
            for images, masks, labels in tqdm(test_loader):
                images = images.to(device)
                masks = masks.unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                test_loss += loss.item()
                
                ## 计算dice 和 miou
                test_dicescore += compute_dice_score(outputs, masks)
                test_ious += compute_iou_score(outputs, masks)
            test_loss = test_loss / len(test_loader)
            test_dicescore = test_dicescore / len(test_loader)
            test_ious = test_ious / len(test_loader)
            print(f'Validation Loss: {test_loss:.4f}, Val Dice: {test_dicescore:.4f}, Val IoU: {test_ious:.4f}')
        results["train_loss"].append(train_loss)
        results["train_dicescore"].append(train_dicescore)
        results["train_ious"].append(train_ious)
        results["test_loss"].append(test_loss)
        results["test_dicescore"].append(test_dicescore)
        results["test_ious"].append(test_ious)

        # early stopping逻辑：这里以验证 dice 为监控指标，只有当提升大于 min_delta 时才认为有改进
        if test_dicescore - best_val_dice > MIN_DELTA:
            best_val_dice = test_dicescore
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
                model_name="neu_seg_baseline_vanillaUnet_",
                result_name="neu_seg_baseline_vanillaUnet_",
                target_dir="./model_output/neu_seg_output",
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
    return results, model


if __name__ == "__main__":
    # NUM_EPOCHS = 10
    # LEARNING_RATE = 1e-4
    # BATCH_SIZE = 4
    from hf_finetune_engine import print_trainable_parameters

    # 数据集路径
    train_images_dir = 'data/NEU_Seg-main/images/training'
    train_annotations_dir = 'data/NEU_Seg-main/annotations/training'

    test_images_dir = 'data/NEU_Seg-main/images/test'
    test_annotations_dir = 'data/NEU_Seg-main/annotations/test'

    train_transforms, test_transforms = get_neu_albumentation_transforms()

    # 创建Dataset实例
    train_dataset = NEU_SEG_Dataset(train_images_dir, train_annotations_dir, transforms=train_transforms)
    test_dataset = NEU_SEG_Dataset(test_images_dir, test_annotations_dir, transforms=test_transforms)

    print("="*40)
    print(f"[数据统计] 训练集图片: {len(train_dataset)} 张")
    print(f"[数据统计] 测试集图片: {len(test_dataset)} 张")
    print("="*40)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    unique_vals = set()
    for i in range(len(train_dataset)):
        _, mask, image_id = train_dataset[i]
        # 假设 mask 是 torch.Tensor（调用 ToTensorV2 后）
        # 如果 mask 是浮点型数据，可以先转为 numpy 数组
        mask_np = mask.numpy()
        unique_vals.update(np.unique(mask_np).tolist())

    print("所有mask的unique值：", unique_vals)

    image, mask, _ = train_dataset[0]
    print(f"-----------Debug Test-------------")
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"image dtype: {image.dtype} || mask dtype: {mask.dtype}")
    print(f"image data range: {image.min()} ~ {image.max()} || mask data unique: {mask.unique()}")

    model = modelLib.build_segformer_model(
    encoder_name="mit_b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
    )
    # model = modelLib.get_smp_unet(encoder_name = "resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    # model = modelLib.get_smp_deeplabv3plus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    # model = modelLib.get_vanilla_unet(n_channels = 3, num_classes = 1)
    print_trainable_parameters(model)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 定义损失函数和优化器
    # criterion = nn.BCEWithLogitsLoss()
    criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    hyperparameters = {
        # "task type": TASK_TYPE, 
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "optimizer": "Adam",
        "learing_rate": LEARNING_RATE,
        "loss function": "monai.DiceCELoss",
        "weight decay": WEIGHT_DECAY,
        # "best epoch": BEST_EPOCH,
        "Task name": "neu_seg_baseline_vanillaUnet",
        # "USE_AMP": USE_AMP,
        # "Mini dataset": create_mini_dataset,
        # 添加其他超参数...
    }
    
    neu_seg_baseline_engine(model=model, device=device)