from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random
from collections import Counter
import monai
from torch import optim
import logging
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
import pytz
import cv2

from utils import save_model
import modelLib
from hf_finetune_engine import print_trainable_parameters
from utils import compute_dice_score, compute_iou_score
from data_setup import magtile_get_all_imgmsk_paths, find_classes

parser = argparse.ArgumentParser(description='训练模型的args选择')
parser.add_argument('--batch_size', type=int, default=4, help='训练的batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='训练的学习率')
parser.add_argument('--num_epochs', type=int, default=5, help='epochs')
# 新增 early stopping 参数
parser.add_argument('--patience', type=int, default=10, help='早停容忍的epoch数')
parser.add_argument('--min_delta', type=float, default=0.0001, help='验证指标提升的最小阈值')

args = parser.parse_args()
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
PATIENCE = args.patience
MIN_DELTA = args.min_delta

####--- seed ---### 
def set_seed(seed: int = 42):
    """
    设置随机种子以确保实验可重复性。

    参数:
        seed (int): 随机种子值。默认为42。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # # 确保在每次操作时使用相同的种子
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_seed(42)
####--- seed ---### 

shanghai_tz = pytz.timezone('Asia/Shanghai')        # 设置时区为亚洲/上海

class MagneticTileDataset(Dataset):
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        mask_np = np.array(mask)

        resize_img = self.letterbox_image(image, [256, 256])        # 
        resize_mask = self.letterbox_mask_1ch(mask_np, [256, 256])  # 
        if self.transform:
            resize_img_tensor = self.transform(resize_img)
            resize_mask_tensor = self.transform(resize_mask)
            resize_mask_tensor = (resize_mask_tensor > 0.5).float()
        label = self.labels[idx]
        return resize_img_tensor, resize_mask_tensor, label

def debug_dataset_datatype(dataset: MagneticTileDataset):
    idx = random.randint(0, len(dataset)-1)
    image, mask, label = dataset[idx]
    print(f"Image type: {image.dtype}, Mask type: {mask.dtype}, Label type: {type(label)}")
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Label: {label}")
    print(f"Image 值的范围: {torch.min(image)} ~ {torch.max(image)}, Mask unique: {torch.unique(mask)}")

# 打印类别分布以验证
def get_label_distribution(dataset):
    labels_subset = [dataset.labels[idx] for idx in range(len(dataset))]
    return Counter(labels_subset)


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
train_dataset = MagneticTileDataset(train_image_paths, train_mask_paths, train_labels, transform=train_transforms)
test_dataset = MagneticTileDataset(test_image_paths, test_mask_paths, test_labels, transform=val_transforms)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# debug信息
logging.basicConfig(level=logging.INFO, format=None)
logger = logging.getLogger(__name__)
debug_idx = random.randint(0, len(train_idx)-1)

logger.info(f"classes_to_idx: {classes_to_idx}")
logger.info(f"Number of training samples: {len(train_image_paths)}")
logger.info(f"Number of testing samples: {len(test_image_paths)}")
logger.info(f"debug信息：{image_paths[train_idx[debug_idx]]}, {mask_paths[train_idx[debug_idx]]}, {labels[train_idx[debug_idx]]}")
debug_dataset_datatype(train_dataset)
logger.info(f"训练集长度：{len(train_dataset)}, 测试集长度：{len(test_dataset)}")
logger.info(f"label分布：{get_label_distribution(train_dataset)}, {get_label_distribution(test_dataset)}")


# model = modelLib.get_smp_deeplabv3plus(
#     encoder_name="efficientnet-b0", 
#     encoder_weights="imagenet", 
#     in_channels=3, 
#     classes=1, 
#     activation=None)

# model = modelLib.build_segformer_model(
#     encoder_name="mit_b0",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1,
#     activation=None)

model = modelLib.get_vanilla_unet(n_channels=3, num_classes=1)

# model = modelLib.get_smp_unet(
#     encoder_name="resnet34",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1,
#     activation=None
# )

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
# criterion = nn.BCEWithLogitsLoss()
criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# 训练循环
num_epochs = NUM_EPOCHS

def magtile_baseline_engine():
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳s
    results = {
        "train_loss": [],
        "train_dicescore": [],
        "train_ious": [],
        "test_loss": [],
        "test_dicescore": [],
        "test_ious": []
    }

    best_val_dice = -1.0    # 初始化最佳验证 dice 分数
    no_improve_epochs = 0   # 连续无改进的 epoch 计数

    model.train()
    print_trainable_parameters(model)
    for epoch in range(num_epochs):
        train_loss, train_dicescore, train_ious = 0, 0, 0
        for images, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            # logging.info(f"outputs.shape: {outputs.shape}, masks.shape: {masks.shape}")
            loss = criterion(outputs, masks)
            ## 计算dice 和 miou
            train_dicescore += compute_dice_score(outputs, masks)
            train_ious += compute_iou_score(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_dicescore = train_dicescore / len(train_loader)
        train_ious = train_ious / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] \nTrain Loss: {train_loss:.4f}, Train Dice: {train_dicescore:.4f}, Train IoU: {train_ious:.4f}')

        # 验证
        model.eval()
        test_loss, test_dicescore, test_ious = 0, 0, 0
        with torch.no_grad():
            for images, masks, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Testing", leave=False):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                test_loss += loss.item()
                
                ## 计算dice 和 miou
                test_dicescore += compute_dice_score(outputs, masks)
                test_ious += compute_iou_score(outputs, masks)
            test_loss = test_loss / len(test_loader)
            test_dicescore = test_dicescore / len(test_loader)
            test_ious = test_ious / len(test_loader)
            print(f'test Loss: {test_loss:.4f}, test Dice: {test_dicescore:.4f}, test IoU: {test_ious:.4f}')
        
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
            save_model(
                        hyperparameters=hyperparameters, 
                        start_timestamp=start_timestamp,
                        results=results, 
                        model=model, 
                        optimizer=optimizer,
                        scaler = None,
                        epoch=epoch+1,  # add 1 because from zero
                        model_name="magtile_baseline_vanillaunet",
                        result_name="magtile_baseline_vanillaunet_",
                        target_dir= "./model_output/mag_output",
                        SAVE_HUGGINGFACE_PRETRAINED_MODEL = False)
        else:
            no_improve_epochs += 1
            print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
        
        # 若连续无改进达到 patience 数, 则提前停止训练
        if no_improve_epochs >= PATIENCE:
            print(f"验证指标连续 {PATIENCE} 个 epoch 无改善，提前停止训练。")
            break
        model.train()
    return results, model

hyperparameters = {
    # "task type": TASK_TYPE, 
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "loss function": "monai.DiceCELoss",
    # "best epoch": BEST_EPOCH,
    "Task name": "magtile_baseline_vanillaunet",
    # "USE_AMP": USE_AMP,
    # "Mini dataset": create_mini_dataset,
    # 添加其他超参数...
}

if __name__ == "__main__":
    results, fintuned_model = magtile_baseline_engine()
