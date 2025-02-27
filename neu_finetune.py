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
from transformers import SamModel
from utils import compute_dice_score, compute_iou_score
from hf_finetune_engine import print_trainable_parameters
from utils import save_model
from helper_function import get_bounding_box
from sam_arch import get_LoRA_DepWiseConv_Samqv_vision_encoder, loraConv_attnqkv, get_sam_lora_qv_encoder
from loratask import get_hf_lora_model, get_hf_adalora_model

parser = argparse.ArgumentParser(description='训练模型的args选择')
parser.add_argument('--batch_size', type=int, default=2, help='训练的batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='训练的学习率')
parser.add_argument('--num_epochs', type=int, default=50, help='epochs')
# 新增 early stopping 参数
parser.add_argument('--patience', type=int, default=10, help='早停容忍的epoch数')
parser.add_argument('--min_delta', type=float, default=0.0001, help='验证指标提升的最小阈值')
parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
parser.add_argument('--lora_rank', type=int, default=16, help='lora rank')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
PATIENCE = args.patience
MIN_DELTA = args.min_delta
WEIGHT_DECAY = args.weight_decay
LORA_RANK = args.lora_rank

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

def debug_dataset_info(train_dataset, test_dataset):
    print("="*40)
    print(f"[数据统计] 训练集图片: {len(train_dataset)} 张")
    print(f"[数据统计] 测试集图片: {len(test_dataset)} 张")
    print("="*40)

    idx = random.randint(0, len(train_dataset) - 1)
    sample = train_dataset[idx]
    img, mask, bbox, image_id, label = sample['image'], sample['mask'], sample['bbox'], sample['image_id'], sample['label']
    print("="*40)
    print(f"debug shapes")
    print(f"img: {img.shape}, mask: {mask.shape}, bbox: {bbox}, image_id: {image_id}, defect type: {label}")
    print("="*40)

    print("="*40)
    print(f"debug data types and data ranges")
    print(f"img: {img.dtype}, mask: {mask.dtype}, bbox: {bbox.dtype}")
    print(f"img: {img.min()} ~ {img.max()}, mask: {mask.unique()}")
    print("="*40)

def neu_seg_finetune_engine(model, device, train_dataloader, test_dataloader, hyperparameters, save_tgt_dir = './model_output/neu_seg_output'):
    model.to(device)

    print_trainable_parameters(model)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]            # 收集所有可训练的 LoRA 参数

    optimizer = AdamW(trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loss_fn = seg_loss

    shanghai_tz = pytz.timezone('Asia/Shanghai')        # 设置时区为亚洲/上海
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳

    best_test_dicescore = 0.0  # 记录最好的test_dicescore  
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
    for epoch in range(NUM_EPOCHS):
        train_loss, train_dicescore, train_ious = 0, 0, 0
        for batch in tqdm(train_dataloader):
            ground_truth_masks = batch["mask"].float().to(device)      # gt
            ground_truth_masks = ground_truth_masks.unsqueeze(1)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values = batch["image"].to(device),
                                    input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                    multimask_output=False)
                    predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]

                    gt_down_256 = F.interpolate(
                        ground_truth_masks, 
                        size=(256, 256),
                        mode="nearest"                                      # 下采样用最近邻，保证mask是二值/多值分割不被插值破坏
                    )
                    loss = loss_fn(predicted_masks, gt_down_256)            # 直接在 256x256 空间计算 Loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values = batch["image"].to(device),
                                input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                multimask_output=False)
                predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]
                gt_down_256 = F.interpolate(
                        ground_truth_masks, 
                        size=(256, 256),
                        mode="nearest"                                      # 下采样用最近邻，保证mask是二值/多值分割不被插值破坏
                    )
                loss = loss_fn(predicted_masks, gt_down_256)            # 直接在 256x256 空间计算 Loss
                loss.backward()         # 反向传播
                optimizer.step()        # optimize

            train_loss += loss.item()
            train_dicescore += compute_dice_score(predicted_masks, gt_down_256)
            train_ious += compute_iou_score(predicted_masks, gt_down_256)
    
        train_loss = train_loss / len(train_dataloader)
        train_dicescore = train_dicescore / len(train_dataloader)
        train_ious = train_ious / len(train_dataloader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] \ntrain loss: {train_loss:.4f}, train dice score: {train_dicescore:.4f}, train iou: {train_ious:.4f}")

        # 验证
        model.eval()
        with torch.no_grad():
            test_loss, test_dicescore, test_ious = 0, 0, 0
            for batch in tqdm(test_dataloader):
                ground_truth_masks = batch["mask"].float().to(device)      # gt
                ground_truth_masks = ground_truth_masks.unsqueeze(1)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(pixel_values = batch["image"].to(device),
                                        input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                        multimask_output=False)
                        predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]
                        gt_down_256 = F.interpolate(
                            ground_truth_masks, 
                            size=(256, 256),
                            mode="nearest"                             
                        )
                        loss = loss_fn(predicted_masks, gt_down_256) 

                        test_loss += loss.item()
                        test_dicescore += compute_dice_score(predicted_masks, gt_down_256)
                        test_ious += compute_iou_score(predicted_masks, gt_down_256)   
            test_loss = test_loss / len(test_dataloader)
            test_dicescore = test_dicescore / len(test_dataloader)
            test_ious = test_ious / len(test_dataloader)
            print(f'Validation Loss: {test_loss:.4f}, Val Dice: {test_dicescore:.4f}, Val IoU: {test_ious:.4f}')

        results["train_loss"].append(train_loss)
        results["train_dicescore"].append(train_dicescore)
        results["train_ious"].append(train_ious)
        results["test_loss"].append(test_loss)
        results["test_dicescore"].append(test_dicescore)
        results["test_ious"].append(test_ious)

        if test_dicescore - best_test_dicescore > MIN_DELTA:
            best_test_dicescore = test_dicescore
            # best_model_wts = model.state_dict()
            best_epoch = epoch + 1  # 更新最佳epoch
            no_improve_epochs = 0
            print(f"验证 dice 改善到 {best_test_dicescore:.4f}, 保存模型...")
            save_model( hyperparameters=hyperparameters,
                        start_timestamp = start_timestamp, 
                        results=results, 
                        model=model, 
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch = epoch+1,
                        model_name="our_method_"  + str(LORA_RANK) + "_",
                        result_name="our_method_" + str(LORA_RANK) + "_",
                        target_dir= save_tgt_dir,
                        SAVE_HUGGINGFACE_PRETRAINED_MODEL = True)
        else:
            no_improve_epochs += 1
            print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
                # 若连续无改进达到 patience 数, 则提前停止训练
        if no_improve_epochs >= PATIENCE:
            print(f"验证指标连续 {PATIENCE} 个 epoch 无改善，提前停止训练。")
            break
        model.train()
    return results, model

def neu_zeroshot(model, device, dataloader, scaler):
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_dicescore, test_ious = 0, 0
        for batch in tqdm(dataloader):
            ground_truth_masks = batch["mask"].float().to(device)      # gt
            ground_truth_masks = ground_truth_masks.unsqueeze(1)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values = batch["image"].to(device),
                                    input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                    multimask_output=False)
                    predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]
                    gt_down_256 = F.interpolate(
                        ground_truth_masks, 
                        size=(256, 256),
                        mode="nearest"                             
                    )
                    test_dicescore += compute_dice_score(predicted_masks, gt_down_256)
                    test_ious += compute_iou_score(predicted_masks, gt_down_256)   
        test_dicescore = test_dicescore / len(dataloader)
        test_ious = test_ious / len(dataloader)
        # print(f'Dice: {test_dicescore:.4f}, IoU: {test_ious:.4f}')
    return test_dicescore, test_ious



if __name__ == '__main__':

    train_dataset, test_dataset = create_neu_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    debug_dataset_info(train_dataset, test_dataset)
    sam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    medsam_model = SamModel.from_pretrained("./HuggingfaceModel/wanglab/medsam-vit-base/model")

    # for name, param in sam_model.named_parameters():
    # # 冻结某些层
    #     if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    #         param.requires_grad_(False)

    model = get_LoRA_DepWiseConv_Samqv_vision_encoder(rank=LORA_RANK, dropout=0.05)               # 深度可分离卷积 conv q,v
    # model = get_hf_lora_model(model = sam_model, target_part='mask_decoder')                    # hugging face lora 编码器 qkv
    # model = loraConv_attnqkv()                                                                    # 和hugging face lora 编码器 qkv 最后的可训练参数是一样的
    # model = get_sam_lora_qv_encoder(rank=16, dropout=0.05)
    # model = get_hf_adalora_model(model=sam_model, target_part='vision_encoder')
    # model = sam_model

    print_trainable_parameters(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hyperparameters = {
                        # "task type": TASK_TYPE, 
                        "batch_size": BATCH_SIZE,
                        "epochs": NUM_EPOCHS,
                        "lora_rank": LORA_RANK,
                        "min_delta": MIN_DELTA,
                        "optimizer": "AdamW",
                        "learning_rate": LEARNING_RATE,
                        "loss function": "monai.DiceCELoss",
                        # "best epoch": BEST_EPOCH,
                        "Task name": "回学校之后测试，our method，rank=16，微调neu-seg",
                        # "USE_AMP": USE_AMP,
                        # "Mini dataset": create_mini_dataset,
                        # 添加其他超参数...
                        }
    # 微调任务
    neu_seg_finetune_engine(model, device, train_dataloader, test_dataloader, hyperparameters)

    # zero-shot 任务
    # train_dice, train_ious = neu_zeroshot(model=medsam_model,
    #                                       device=device,
    #                                       dataloader=train_dataloader,
    #                                       scaler = torch.cuda.amp.GradScaler(enabled=True))

    # test_dice, test_ious = neu_zeroshot(model=medsam_model,
    #                                       device=device,
    #                                       dataloader=test_dataloader,
    #                                       scaler = torch.cuda.amp.GradScaler(enabled=True))
    # print(f'train_dice: {train_dice:.4f}, train_ious: {train_ious:.4f} || test_dice: {test_dice:.4f}, test_ious: {test_ious:.4f}')