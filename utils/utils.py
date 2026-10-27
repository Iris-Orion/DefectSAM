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
    """Soft Dice loss on raw predictions; no sigmoid is applied, so pass probabilities. Never binarize: thresholding zeroes the gradient."""
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        dice_loss = 1 - dice
        
        return dice_loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs)       
        
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
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice 

    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        total_loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss
        
        return total_loss
    
class MultiClassDiceLoss(nn.Module):
    """Multi-label soft Dice: sigmoid per channel, Dice per channel, averaged over channels."""
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

def compute_dice_score(pred, target, epsilon=1e-6, threshold = 0.5, as_tensor=False):
    """Batch-pooled Dice: intersection and union are summed over the whole batch, not averaged per image.

    as_tensor=True returns a device scalar tensor (no .item(), so no per-batch host sync); call .item()
    once per epoch. Default returns a Python float.
    """
    assert pred.shape == target.shape
    batch_size = pred.shape[0]
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).to(torch.float32)
    pred_flat = pred.contiguous().view(batch_size, -1)  # contiguous() is needed for SegFormer outputs
    target_flat = target.view(batch_size, -1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()


    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice if as_tensor else dice.item()

def compute_iou_score(pred, target, epsilon=1e-6, threshold=0.5, as_tensor=False):
    """Batch-pooled IoU; as_tensor behaves as in compute_dice_score."""
    assert pred.shape == target.shape
    batch_size = pred.shape[0]
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).to(torch.float32)
    pred_flat = pred.contiguous().view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    iou = (intersection + epsilon) / (union + epsilon)
    return iou if as_tensor else iou.item()

def save_lora_parameters(model: torch.nn.Module, filename: str) -> None:
    lora_state_dict = {
        name.replace('_orig_mod.', ''): param
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    
    if not lora_state_dict:
        print("警告：在模型中没有找到任何可训练的(LoRA)参数。")
        return

    torch.save(lora_state_dict, filename)
    print(f"成功将 {len(lora_state_dict)} 个LoRA层对应的参数张量保存到 {filename}")

def save_model(hyperparameters,
               start_timestamp,
               model: torch.nn.Module,
               optimizer,
               scaler,
               epoch,
               model_name: str,
               target_dir: str = "./model_output",
               SAVE_HUGGINGFACE_PRETRAINED_MODEL: bool = False,
               save_lora_only: bool = False ):
    """Save a checkpoint and return its path.

    SAVE_HUGGINGFACE_PRETRAINED_MODEL uses model.save_pretrained; else save_lora_only saves only
    trainable parameters; else a full dict with model_state_dict, hyperparameters and last_epoch.
    optimizer and scaler are accepted but never saved.
    """
    save_dir = target_dir
    os.makedirs(save_dir, exist_ok=True)

    if SAVE_HUGGINGFACE_PRETRAINED_MODEL:
        saved_model_str = model_name + start_timestamp + ".pth"
        model_path = os.path.join(save_dir, saved_model_str)
        model.save_pretrained(model_path)
        print(f"模型已保存至:{model_path}")
        return model_path

    
    elif save_lora_only:
        lora_file_name = f"{model_name}_{start_timestamp}.pth"
        model_path = os.path.join(save_dir, lora_file_name)
        save_lora_parameters(model, model_path)
        return model_path

    else:
        model_filename = f"{model_name}_{start_timestamp}.pth"
        model_path = os.path.join(save_dir, model_filename)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'hyperparameters': hyperparameters,
            'last_epoch': epoch
        }
        torch.save(checkpoint, model_path)
        print(f"模型已保存至:{model_path}")
        return model_path


def save_training_logs(
                        hyperparameters,
                        results,
                        epoch: int,
                        start_timestamp: str,
                        result_name: str,
                        target_dir: str = "./model_output"):
    """Write hyperparameters and results to <target_dir>/<result_name>_<start_timestamp>.json; return the written dict."""
    save_dir = target_dir
    os.makedirs(save_dir, exist_ok=True)

    log_filename = f"{result_name}_{start_timestamp}.json"
    log_path = os.path.join(save_dir, log_filename)
    
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    end_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")

    output_data = {
        "hyperparameters": hyperparameters,
        "training_start_timestamp": start_timestamp,
        "training_end_timestamp": end_timestamp,
        "last_completed_epoch": epoch,
        "training_results": results
    }

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


def get_loss_dict():
    loss_dict = {
        "dice_calculator": DiceLoss(),
        "bce_logit": torch.nn.BCEWithLogitsLoss(),
        "dice_bce": DiceBCELoss(),
        "seg_monai_dice": monai.losses.DiceLoss(softmax=True, squared_pred=True, reduction='mean'),
        "combine_dice_ce": Combine_DiceCrossEntropy_Loss(),
        "cross_entropy": torch.nn.CrossEntropyLoss(reduction='mean'),
        "bce": torch.nn.BCELoss(),
        "monai_diceCEloss":monai.losses.DiceCELoss(softmax=True, squared_pred=True, reduction='mean')
    }
    return loss_dict

def get_loss_fn(loss_name: str = "bce_logit"):
    losses = get_loss_dict()
    if loss_name not in losses:
        raise ValueError(f"Unknown loss name '{loss_name}'. "
                         f"Available names: {list(losses.keys())}")
    return losses[loss_name]
