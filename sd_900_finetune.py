import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import logging
import modelLib
from utils import compute_dice_score, compute_iou_score
import argparse
import monai
from hf_finetune_engine import print_trainable_parameters
import cv2
from transformers import SamModel
from datetime import datetime
import pytz
from utils import save_model
from tqdm import tqdm
import torch.nn.functional as F
import loratask
from sam_arch import get_LoRA_DepWiseConv_Samqv_vision_encoder, get_LoRA_Moe_DepwiseConv_Samqv_vison_encoder, loraConv_attnqkv, get_sam_hg_model, get_sam_lora_qv_encoder  #重新写的lora conv替换qkv
import random
from helper_function import set_seed
###--------------------------args ---------------------------------###
parser = argparse.ArgumentParser(description='训练模型的args选择')
parser.add_argument('--batch_size', type=int, default=2, help='训练的batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='训练的学习率')
parser.add_argument('--num_epochs', type=int, default=50, help='epochs')
parser.add_argument('--patience', type=int, default=10, help='early stop patience')
parser.add_argument('--lora_rank', type=int, default=16, help='lora rank')

args = parser.parse_args()
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
PATIENCE = args.patience
LORA_RANK = args.lora_rank

# BATCH_SIZE = 4
# LEARNING_RATE = 1e-4
# NUM_EPOCHS = 2
###--------------------------args ---------------------------------###


set_seed(42)        ####--- seed ---### 

hyperparameters = {
    # "task type": TASK_TYPE, 
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "lora_rank": LORA_RANK,
    "optimizer": "AdamW",
    "learing_rate": LEARNING_RATE,
    "loss function": "monai.DiceCELoss",
    # "best epoch": BEST_EPOCH,
    "Task name": "adalora微调sd900",
    # "USE_AMP": USE_AMP,
    # "Mini dataset": create_mini_dataset,
    # 添加其他超参数...
}

def get_bounding_box(ground_truth_map):
    """
    从ground_truth中获得bbox
    """
    assert type(ground_truth_map) == np.ndarray, "check ground truth type"
    if np.sum(ground_truth_map)== 0:
        bbox = [0, 0, 0, 0]
    else:
        # 从非空掩码中得到bounding box
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]
    return bbox

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
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_dataloader, test_dataloader

def sd_900_finetune_engine():
    train_dataloader, test_dataloader = sd_900_finetune_create_dataset()
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    # for name, param in hgsam_model.named_parameters():
    #     # 冻结某些层
    #     if name.startswith("vision_encoder"):
    #         param.requires_grad_(False)

    #----------------------- model 选择 --------------------------#
    # model = loratask.get_hf_lora_model(model = hgsam_model, target_part = 'mask_decoder')
    # model = loratask.get_hf_lora_model(model = hgsam_model, target_part = 'vision_encoder')
    # model = hgsam_model
    model = loratask.get_hf_adalora_model(model = hgsam_model, target_part='vision_encoder')
    # model = get_sam_lora_qkv_vision_encoder()
    # model = get_sam_lora_conv_qkv_vision_encoder()
    # model = get_LoRA_Moe_DepwiseConv_Samqv_vison_encoder()
    # model = get_LoRA_DepWiseConv_Samqv_vision_encoder(rank=LORA_RANK, dropout=0.05)   # 深度可分离卷积 conv q,v
    # model = get_sam_lora_conv_qkv_vision_encoder()  # 非深度可分离卷积 conv q, v
    # model = loraConv_attnqkv(rank=16, dropout=0.1) # 标准卷积lora qkv
    # model = get_sam_lora_qv_encoder(rank=16, dropout=0.05)
    #----------------------- model 选择 --------------------------#


    print_trainable_parameters(model)
    print(model.vision_encoder)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"using devcie {device}")
    model.to(device)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]            # 收集所有可训练的 LoRA 参数

    optimizer = AdamW(trainable_parameters, lr=LEARNING_RATE, weight_decay=0.01)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # scaler = None
    loss_fn = seg_loss
    shanghai_tz = pytz.timezone('Asia/Shanghai')        # 设置时区为亚洲/上海

    results = {
        "train_loss": [],
        "train_dicescore": [],
        "train_ious": [],
        "test_loss": [],
        "test_dicescore": [],
        "test_ious": []
    }

    best_test_dicescore = 0.0  # 记录最好的test_dicescore   TODO: 思考是用最小的loss的那一轮还是最大的dice score的那一轮
    best_epoch = 0
    no_improve_epochs = 0   # 连续无改进的 epoch 计数

    model.train()
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳
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

                    ## 第一种选择，将预测结果进行上采样
                    # ori_res_masks = F.interpolate(
                    #                             predicted_masks,              # [B, 1, 256, 256]  ---interpolate---> [B, 1, 1024, 1024]
                    #                             size=(1024, 1024),  
                    #                             mode="bilinear",
                    #                             align_corners=False
                    #                             )
                    # assert ori_res_masks.shape == ground_truth_masks.shape, \
                    # f"Shape mismatch: ori_res_masks shape is {ori_res_masks.shape}, " \
                    # f"but ground_truth_masks shape is {ground_truth_masks.shape}."
                    # loss = loss_fn(ori_res_masks, ground_truth_masks)

                    ## 第二种选择，将gt进行下采样到?(256, 256)
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
                
                ## 第一种选择，将预测结果进行上采样
                # ori_res_masks = F.interpolate(
                #                                 predicted_masks,              # [B, 1, 256, 256]  ---interpolate---> [B, 1, 1024, 1024]
                #                                 size=(1024, 1024),  
                #                                 mode="bilinear",
                #                                 align_corners=False
                #                                 )
                # assert ori_res_masks.shape == ground_truth_masks.shape, \
                # f"Shape mismatch: ori_res_masks shape is {ori_res_masks.shape}, " \
                # f"but ground_truth_masks shape is {ground_truth_masks.shape}."

                # loss = loss_fn(ori_res_masks, ground_truth_masks)

                ## 第二种选择，将gt进行下采样到?(256, 256)
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

                        ## 第一种选择，将预测结果进行上采样
                        # ori_res_masks = F.interpolate(
                        #                         predicted_masks,              # [B, 1, 256, 256]  ---interpolate---> [B, 1, 1024, 1024]
                        #                         size=(1024, 1024),  
                        #                         mode="bilinear",
                        #                         align_corners=False
                        #                         )
                        # loss = loss_fn(ori_res_masks, ground_truth_masks)

                        ## 第二种选择，将gt进行下采样到?(256, 256)
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

        if test_dicescore - best_test_dicescore > 0.0001:
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
                        model_name="sd900_adalora_",  # + str(LORA_RANK) + "_",
                        result_name="sd900_adalora_", # + str(LORA_RANK) + "_",
                        target_dir= "./model_output/sd900_output",
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

def sd_900_inference_engine(checkpoint_path: str, use_bbox: bool = False, eval_traindataset = False):
    """
    纯推理, 加bbox和不加bbox?
    """
    train_dataloader, test_dataloader = sd_900_finetune_create_dataset()  
    checkpoint = torch.load(checkpoint_path)
    model = get_LoRA_DepWiseConv_Samqv_vision_encoder(rank=LORA_RANK, dropout=0.05)
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  

    os.makedirs("./inference_results/sd900infer", exist_ok=True)
    
    with torch.no_grad():
        if eval_traindataset:
            train_dice, train_iou = inference_step(dataloader=train_dataloader, model=model, device=device, use_bbox=use_bbox)
        test_dice, test_iou = inference_step(dataloader=test_dataloader, model=model, device=device, use_bbox=use_bbox)

    print(f"\nFinal Evaluation Metrics:")
    print(f"Train Dice / Train IoU: {train_dice:.4f} / {train_iou:.4f}")
    print(f"Test Dice / Test IoU: {test_dice:.4f} / {test_iou:.4f}")


def inference_step(dataloader, model, device, use_bbox):
    total_dice, total_iou = 0.0, 0.0
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        images = batch["image"].to(device)
        gt_masks = batch["mask"].float().to(device)
        gt_masks = gt_masks.unsqueeze(1)
        if use_bbox:
            bboxes = batch["bbox"].unsqueeze(1).to(device)
        else:
            bboxes = None
    
        outputs = model(
            pixel_values=images,
            input_boxes=bboxes,
            multimask_output=False
        )
        
        pred_masks = outputs.pred_masks.squeeze(1)  # [B, 1, 256, 256]
        gt_downsampled = F.interpolate(
            gt_masks,
            size=(256, 256),
            mode="nearest"
        )  # 下采样GT到256x256
        
        # print(f"pred_masks shape: {pred_masks.shape}, gt_downsampled shape: {gt_downsampled.shape}")

        # 计算指标
        batch_dice = compute_dice_score(pred_masks, gt_downsampled)
        batch_iou = compute_iou_score(pred_masks, gt_downsampled)
        
        total_dice += batch_dice
        total_iou += batch_iou
    mean_dice = total_dice / len(dataloader)
    mean_iou = total_iou / len(dataloader)
    return mean_dice, mean_iou

if __name__ == '__main__':
    debug_sd900_img_and_mask()

    # 训练
    # results, fintuned_model = sd_900_finetune_engine()

    # zero shot
    train_dataloader, test_dataloader = sd_900_finetune_create_dataset()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"using devcie {device}")
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    medsam_model = SamModel.from_pretrained("./HuggingfaceModel/wanglab/medsam-vit-base/model")

    model = medsam_model
    model.to(device)
    model.eval()
    with torch.no_grad():
        train_dice, train_iou = inference_step(dataloader=train_dataloader,
                                            model=model,
                                            device=device,
                                            use_bbox=True)
        
        test_dice, test_iou = inference_step(dataloader=test_dataloader,
                                            model=model,
                                            device=device,
                                            use_bbox=True)
    print(f"train dice: {train_dice:.4f}  train_iou: {train_iou:.4f} || test dice: {test_dice:.4f} test_iou: {test_iou:.4f}")

    # 测试
    # sd_900_inference_engine(checkpoint_path='/workspace/DefectDetection/model_output/sd900_output/sd900_depwiseConv_qv_encoder_16_20250124_180057.pth',
    #                         use_bbox=False,
    #                         eval_traindataset=True)