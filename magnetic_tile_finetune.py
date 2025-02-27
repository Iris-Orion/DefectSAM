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
import cv2
import torch.nn.functional as F
from datetime import datetime
import pytz

import modelLib
from hf_finetune_engine import print_trainable_parameters
from utils import compute_dice_score, compute_iou_score
from data_setup import magtile_get_all_imgmsk_paths, find_classes
from transformers import SamModel
from torch.optim import AdamW
from hf_finetune_engine import hfsam_finetune
from utils import save_model
import loratask
from sam_arch import get_sam_loraconv_qv_vision_encoder, get_LoRA_DepWiseConv_Samqv_vision_encoder, get_sam_lora_qv_encoder

parser = argparse.ArgumentParser(description='训练模型的args选择')
parser.add_argument('--batch_size', type=int, default=2, help='训练的batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='训练的学习率')
parser.add_argument('--num_epochs', type=int, default=50, help='epochs')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='set weight decay')
parser.add_argument('--patience', type=int, default=10, help='早停容忍的epoch数')        # 新增 early stopping 参数
parser.add_argument('--lora_rank', type=int, default=16, help='lora rank')

args = parser.parse_args()
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
WEIGHT_DECAY = args.weight_decay
PATIENCE = args.patience
LORA_RANK = args.lora_rank

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

def get_bounding_box(ground_truth_map):
    """
    从ground_truth中获得bbox
    输入:ground_truth_map 是一个合并的掩码 (256, 1600)   tensor格式
    TODO: ground_truth_map不是合并的, 根据(4, 256, 1600)判断每个channel中的box
    """
    # 先转为np格式
    # print(type(ground_truth_map))
    assert type(ground_truth_map) == np.ndarray, "check type"
    if ground_truth_map.sum() == 0:
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

train_transforms = transforms.Compose([
    transforms.ToTensor()
    ]
)

val_transforms = transforms.Compose([
    transforms.ToTensor()
    ]
)

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

train_dataset = MagneticTileDatasetWithBoxPrompt(train_image_paths, train_mask_paths, train_labels, transform=train_transforms)
test_dataset = MagneticTileDatasetWithBoxPrompt(test_image_paths, test_mask_paths, test_labels, transform=val_transforms)

# 创建 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

hyperparameters = {
    # "task type": TASK_TYPE, 
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "loss function": "monai.DiceCELoss",
    "lora rank": LORA_RANK,
    # "best epoch": BEST_EPOCH,
    "Task name": "magnetic tile 微调 sam maskdecoder",
    # "USE_AMP": USE_AMP,
    # "Mini dataset": create_mini_dataset,
    # 添加其他超参数...
}

def finetune_task():
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    for name, param in hgsam_model.named_parameters():
        # 冻结某些层
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    # model = loratask.get_hf_lora_model(model = hgsam_model, target_part = 'mask_decoder')
    # model = get_sam_loraconv_qv_vision_encoder()  
    # model = get_LoRA_DepWiseConv_Samqv_vision_encoder(rank=LORA_RANK, dropout=0.05)
    # model = get_sam_lora_qv_encoder(rank=LORA_RANK, dropout=0.05)
    # model = loratask.get_hf_adalora_model(model=hgsam_model, target_part='vision_encoder')
    model = hgsam_model
    print_trainable_parameters(model)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"using devcie {device}")
    model.to(device)
    model.train()

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]            # 收集所有可训练的 LoRA 参数
    optimizer = AdamW(trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # scaler = None
    loss_fn = seg_loss

    results = {
        "train_loss": [],
        "train_dicescore": [],
        "train_ious": [],
        "test_loss": [],
        "test_dicescore": [],
        "test_ious": []
    }

    best_test_dicescore = 0.0  # 记录最好的test_dicescore   TODO: 思考是用最小的loss的那一轮还是最大的dice score的那一轮
    best_model_wts = None
    best_epoch = 0
    no_improve_epochs = 0   # 连续无改进的 epoch 计数
    patience = PATIENCE
    
    shanghai_tz = pytz.timezone('Asia/Shanghai')        # 设置时区为亚洲/上海
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳

    for epoch in range(NUM_EPOCHS):
        train_loss, train_dicescore, train_ious = 0, 0, 0
        for batch in tqdm(train_dataloader):
            ground_truth_masks = batch["mask"].float().to(device)      # gt
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values = batch["image"].to(device),
                                    input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                    multimask_output=False)
                    predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]

                    assert predicted_masks.shape == ground_truth_masks.shape, \
                    f"Shape mismatch: predicted_masks shape is {predicted_masks.shape}, " \
                    f"but ground_truth_masks shape is {ground_truth_masks.shape}."

                    loss = loss_fn(predicted_masks, ground_truth_masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values = batch["image"].to(device),
                                input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                multimask_output=False)
                predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]
                assert predicted_masks.shape == ground_truth_masks.shape, \
                f"Shape mismatch: predicted_masks shape is {predicted_masks.shape}, " \
                f"but ground_truth_masks shape is {ground_truth_masks.shape}."

                loss = loss_fn(predicted_masks, ground_truth_masks)
                loss.backward()         # 反向传播
                optimizer.step()        # optimize

            train_loss += loss.item()
            train_dicescore += compute_dice_score(predicted_masks, ground_truth_masks)
            train_ious += compute_iou_score(predicted_masks, ground_truth_masks)
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
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(pixel_values = batch["image"].to(device),
                                        input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                        multimask_output=False)
                        predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]
                        loss = loss_fn(predicted_masks, ground_truth_masks)

                        test_loss += loss.item()
                        test_dicescore += compute_dice_score(predicted_masks, ground_truth_masks)
                        test_ious += compute_iou_score(predicted_masks, ground_truth_masks)   
            test_loss = test_loss / len(test_dataloader)
            test_dicescore = test_dicescore / len(test_dataloader)
            test_ious = test_ious / len(test_dataloader)
            print(f'Validation Loss: {test_loss:.4f}, Val Dice: {test_dicescore:.4f}, Val IoU: {test_ious:.4f}')
        
        model.train()

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
                        model_name="magtile_ft_maskdecoder_", # + str(LORA_RANK) + "_",
                        result_name="magtile_ft_maskdecoder_", # + str(LORA_RANK) + "_",
                        target_dir= "./model_output/mag_output",
                        SAVE_HUGGINGFACE_PRETRAINED_MODEL = True)
        else:
            no_improve_epochs += 1
            print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
                # 若连续无改进达到 patience 数, 则提前停止训练
        if no_improve_epochs >= patience:
            print(f"验证指标连续 {patience} 个 epoch 无改善，提前停止训练。")
            break
    return results, model

def mag_inference(model, device, dataloader, scaler):
    model.to(device) 
    model.eval()                # 验证
    
    with torch.no_grad():
        test_dicescore, test_ious = 0, 0
        for batch in tqdm(dataloader):
            ground_truth_masks = batch["mask"].float().to(device)      # gt
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values = batch["image"].to(device),
                                    input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                    multimask_output=False)
                    predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]

                    test_dicescore += compute_dice_score(predicted_masks, ground_truth_masks)
                    test_ious += compute_iou_score(predicted_masks, ground_truth_masks)   
        test_dicescore = test_dicescore / len(dataloader)
        test_ious = test_ious / len(dataloader)
        # print(f'Validation Loss: {test_loss:.4f}, Val Dice: {test_dicescore:.4f}, Val IoU: {test_ious:.4f}')
    return test_dicescore, test_ious

if __name__ == "__main__":

    # 微调任务
    results, fintuned_model = finetune_task()

    ################ zero-shot####################
    # hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    # medsam_model = SamModel.from_pretrained("./HuggingfaceModel/wanglab/medsam-vit-base/model")

    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(device)
    # print(f"using devcie {device}")

    # seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    # scaler = torch.cuda.amp.GradScaler(enabled=True)

    # train_dice, train_ious = mag_inference(model=hgsam_model,
    #                                        device=device,
    #                                        dataloader=train_dataloader,
    #                                        scaler=scaler,
    #                                        )
    # test_dice, test_ious = mag_inference(model=hgsam_model,
    #                                        device=device,
    #                                        dataloader=test_dataloader,
    #                                        scaler=scaler,
    #                                        )
    # print(f'train_dice: {train_dice:.4f}, train_ious: {train_ious:.4f} || test_dice: {test_dice:.4f}, test_ious: {test_ious:.4f}')
    #################### zero-shot ###########


    # save_model( hyperparameters=hyperparameters, 
    #             results=results, 
    #             model=model, 
    #             optimizer=optimizer,
    #             model_name="magtile_finetune_fully_finetune_",
    #             result_name="magtile_finetune_fully_finetune_",
    #             target_dir= "./model_output/mag_output",
    #             SAVE_HUGGINGFACE_PRETRAINED_MODEL = True)
    
