import monai.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import os
import pytz
import json
from datetime import datetime


class DiceLoss(nn.Module):
    """
    在计算 Dice Loss 时，不应对预测值进行二值化，因为这会导致梯度为零，
    影响模型的训练。应该直接使用模型的输出(通常是概率或 logits)
    """
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        # 如果模型最后有sigmoid，则注释掉下面这一行
        # pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # 如果乘以一个黑色的通道，交集就直接变为了0，因此这个通道的loss也就变成了0
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        dice_loss = 1 - dice
        
        return dice_loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class Combine_DiceCrossEntropy_Loss(nn.Module):
    """
    inspired by  https://github.com/bowang-lab/MedSAM
    """
    def __init__(self, weight_ce = 1.0, weight_dice = 1.0):
        super(Combine_DiceCrossEntropy_Loss, self).__init__()
        # 两个loss
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

        # loss分别的权重
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice 

    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        total_loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss
        
        return total_loss
    
class MultiClassDiceLoss(nn.Module):
    """
    多标签多分类，可以针对每个通道计算 Dice Loss，然后求平均。
    """
    def __init__(self, epsilon=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        dice_loss = 0
        for i in range(target.shape[1]):
            pred_flat = pred[:, i, :, :].contiguous().view(-1)
            target_flat = target[:, i, :, :].contiguous().view(-1)
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss += 1 - dice
        return dice_loss / target.shape[1]

def compute_dice_score(pred, target, epsilon=1e-6, threshold = 0.5):
    """
    计算dice score
    """
    assert pred.shape == target.shape
    batch_size = pred.shape[0]
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).to(torch.float32)  # 加一个阈值
    # pred_flat = pred.view(batch_size, -1)
    pred_flat = pred.contiguous().view(batch_size, -1)  # segformer用
    target_flat = target.view(batch_size, -1)

    intersection = (pred_flat * target_flat).sum()   # 交集
    union = pred_flat.sum() + target_flat.sum()      # 并集

    # if union == 0:
    #     return 1.0 if intersection == 0 else 0.0

    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.item()

def compute_iou_score(pred, target, epsilon=1e-6, threshold=0.5):
    """
    计算IOU (Intersection over Union) score
    """
    assert pred.shape == target.shape
    batch_size = pred.shape[0]
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).to(torch.float32)  # 加一个阈值
    pred_flat = pred.contiguous().view(batch_size, -1)  # 展平预测结果
    target_flat = target.view(batch_size, -1)  # 展平真实标签

    intersection = (pred_flat * target_flat).sum()  # 计算交集
    union = pred_flat.sum() + target_flat.sum() - intersection  # 计算并集

    iou = (intersection + epsilon) / (union + epsilon)  # 计算IOU
    return iou.item()

def save_lora_parameters(model: torch.nn.Module, filename: str) -> None:
    """
    保存模型中所有需要训练的(LoRA)参数。
    """
    # 1. 创建一个字典，只包含需要梯度的参数
    lora_state_dict = {
        name.replace('_orig_mod.', ''): param
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    
    # 2. 检查是否找到了任何可训练的参数
    if not lora_state_dict:
        print("警告：在模型中没有找到任何可训练的(LoRA)参数。")
        return

    # 3. 保存这个只包含LoRA参数的state_dict
    torch.save(lora_state_dict, filename)
    print(f"成功将 {len(lora_state_dict)} 个LoRA层对应的参数张量保存到 {filename}")

def save_model(hyperparameters,
               start_timestamp,             # 训练开始时间
               model: torch.nn.Module,
               optimizer,
               scaler,
               epoch,                       # 当前 epoch，用于恢复训练时知道从哪个 epoch 开始
               model_name: str,
               target_dir: str = "./model_output",
               SAVE_HUGGINGFACE_PRETRAINED_MODEL: bool = False,
               save_lora_only: bool = False ):               # 新增只保存 LoRA参数
    """
    根据指定策略保存模型检查点、优化器状态和scaler状态。

    Args:
        model (torch.nn.Module): 要保存的模型。
        optimizer: 优化器。
        scaler: AMP scaler (可以为 None)。
        hyperparameters (dict): 模型的超参数，用于保存完整检查点。
        epoch (int): 当前的 epoch。
        start_timestamp (str): 训练开始的时间戳，用于生成唯一文件名。
        model_name (str): 模型的基础名称。
        target_dir (str, optional): 保存目录。默认为 "./model_output"。
        save_lora_only (bool, optional): 如果为 True，仅保存 LoRA 参数。默认为 False。
        SAVE_HUGGINGFACE_PRETRAINED_MODEL (bool, optional): 如果为 True，使用Hugging Face的 `save_pretrained` 方法保存。默认为 False。
    """
    save_dir = target_dir
    os.makedirs(save_dir, exist_ok=True)

    # 使用 Hugging Face 格式保存
    if SAVE_HUGGINGFACE_PRETRAINED_MODEL:
        saved_model_str = model_name + start_timestamp + ".pth"
        model_path = os.path.join(save_dir, saved_model_str)
        model.save_pretrained(model_path)   # hugging face保存方式
        print(f"模型已保存至:{model_path}")
        return model_path

        # optimizer_path = os.path.join(save_dir, f"optimizer_{start_timestamp}.pth")
        # torch.save(optimizer.state_dict(), optimizer_path) # 保存优化器
        # print(f"优化器已保存至:{optimizer_path}", )
    
    # 自定义 save_lora_only 保存策略
    elif save_lora_only:
        lora_file_name = f"{model_name}_{start_timestamp}.pth"
        model_path = os.path.join(save_dir, lora_file_name)
        save_lora_parameters(model, model_path)
        return model_path

    # 保存完整的 PyTorch 检查点
    else:
        model_filename = f"{model_name}_{start_timestamp}.pth"
        model_path = os.path.join(save_dir, model_filename)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'hyperparameters': hyperparameters,
            'last_epoch': epoch
        }
        # if scaler is not None:
        #     checkpoint['scaler'] = scaler.state_dict()
        torch.save(checkpoint, model_path) # 正常保存模型
        print(f"模型已保存至:{model_path}")
        return model_path
###--------------- 保存模型 ------------------###


def save_training_logs(
                        hyperparameters,
                        results,
                        epoch: int,
                        start_timestamp: str,
                        result_name: str,
                        target_dir: str = "./model_output"):
    """
    将超参数和训练结果保存到JSON日志文件中。

    Args:
        hyperparameters (dict): 训练使用的超参数。
        results (dict): 包含训练和测试指标的字典。
        epoch (int): 训练结束时的 epoch。
        start_timestamp (str): 训练开始的时间戳，用于生成唯一文件名。
        result_name (str): 日志文件的基础名称。
        target_dir (str, optional): 保存目录。默认为 "./model_output"。
    """
    save_dir = target_dir
    os.makedirs(save_dir, exist_ok=True)

    # 定义日志文件路径
    log_filename = f"{result_name}_{start_timestamp}.json"
    log_path = os.path.join(save_dir, log_filename)
    
    # 获取结束时间戳
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    end_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")

    # 准备要输出的数据
    output_data = {
        "hyperparameters": hyperparameters,
        "training_start_timestamp": start_timestamp,
        "training_end_timestamp": end_timestamp,
        "last_completed_epoch": epoch,
        "training_results": results
    }

    # 写入JSON文件
    with open(log_path, "w", encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"训练日志已保存至: {log_path}")

    return output_data


def print_trainable_parameters(model):
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")


#  -----loss ------#
def get_loss_dict():
    """
    返回一个包含多个不同 Loss 对象的字典，方便管理和调用。
    """
    loss_dict = {
        "dice_calculator": DiceLoss(),
        "bce_logit": torch.nn.BCEWithLogitsLoss(),
        "dice_bce": DiceBCELoss(),
        "seg_monai_dice": monai.losses.DiceLoss(softmax=True, squared_pred=True, reduction='mean'),
        "combine_dice_ce": Combine_DiceCrossEntropy_Loss(),
        "cross_entropy": torch.nn.CrossEntropyLoss(reduction='mean'),
        "bce": torch.nn.BCELoss(),
        "monai_diceCEloss":monai.losses.DiceCELoss(softmax=True, squared_pred=True, reduction='mean')  # 多分类任务
    }
    return loss_dict

def get_loss_fn(loss_name: str = "bce_logit"):
    """
    根据指定的名称返回对应的 Loss 对象。
    默认为 'bce_logit'，即 BCEWithLogitsLoss。
    """
    losses = get_loss_dict()
    if loss_name not in losses:
        raise ValueError(f"Unknown loss name '{loss_name}'. "
                         f"Available names: {list(losses.keys())}")
    return losses[loss_name]
