import monai.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import os
from datetime import datetime
import pytz
import json

###--------------- 自定义Loss ------------------###
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

#PyTorch
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

###--------------- 自定义Loss ------------------###


###--------------- 自定义评估指标 ------------------###
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
###--------------- 自定义评估指标 ------------------###


###--------------- 自定义评估指标 ------------------###
def compute_dice_score_onSam(pred, target, epsilon=1e-6):
    """
    该函数已经弃用
    计算直接sam输出的mask与ground truth的 dice score, 
    pred, target 均为 0/1 或 bool 类型的二值掩码
    """
    # 确保都是 float 类型
    pred = pred.view(-1).float()
    target = target.view(-1).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.item()
###--------------- 自定义评估指标 ------------------###


###--------------- 保存模型 ------------------###
def save_model(
               hyperparameters,
               start_timestamp, # 训练开始时间
               results,
               model: torch.nn.Module,
               optimizer,
               scaler,
               epoch,               # 当前 epoch，用于恢复训练时知道从哪个 epoch 开始
               model_name: str,
               result_name: str,
               target_dir: str = "./model_output",
               SAVE_HUGGINGFACE_PRETRAINED_MODEL: bool = False):
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    end_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")

    save_dir = target_dir
    os.makedirs(save_dir, exist_ok=True)

    saved_model_str = model_name + start_timestamp + ".pth"
    result_path_str = result_name + start_timestamp + ".json"

    model_path = os.path.join(save_dir, saved_model_str)
    results_path = os.path.join(save_dir, result_path_str)

    # 合并超参数和训练结果
    output_results = {
        "hyperparameters": hyperparameters,
        "end_timestap": end_timestamp,
        "training_results": results,
        "last_epoch": epoch,
    }

    if SAVE_HUGGINGFACE_PRETRAINED_MODEL:
        model.save_pretrained(model_path)   # hugging face保存方式
        optimizer_path = os.path.join(save_dir, f"optimizer_{start_timestamp}.pth")
        torch.save(optimizer.state_dict(), optimizer_path) # 保存优化器
        print(f"模型已保存至:{model_path}", )
        print(f"优化器已保存至:{optimizer_path}", )
    else:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparameters': hyperparameters,
            'results': results,
            'end_timestamp': end_timestamp,
            'last_epoch': epoch
        }
        if scaler is not None:
            checkpoint['scaler'] = scaler.state_dict()
        torch.save(checkpoint, model_path) # 正常保存模型
        print(f"模型和优化器状态已保存至:{model_path}")

    # 保存结果和超参数
    with open(results_path, "w") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=4)
    print(f"训练结果和超参数已保存至:{results_path}")

    # 打印最终的训练和测试结果
    # print("\n最终训练结果:")
    # print(f"训练损失: {results['train_loss'][-1]:.4f}")
    # print(f"训练Dice分数: {results['train_dicescore'][-1]:.4f}")
    # print(f"训练iou: {results['train_ious'][-1]:.4f}")
    # print(f"测试损失: {results['test_loss'][-1]:.4f}")
    # print(f"测试Dice分数: {results['test_dicescore'][-1]:.4f}")
    # print(f"测试iou: {results['test_ious'][-1]:.4f}")

    # 打印最好的训练和测试结果（对应最小测试损失的 epoch）
    # best_epoch = results["test_loss"].index(min(results["test_loss"]))  # 最小测试损失对应的 epoch
    # best_epoch = hyperparameters["best epoch"]
    # print("\n最好的训练结果(对应最小验证损失的 epoch):")
    # print(f"Epoch {best_epoch}:")
    # print(f"训练损失: {results['train_loss'][best_epoch-1]:.4f}")
    # print(f"训练Dice分数: {results['train_dicescore'][best_epoch-1]:.4f}")
    # print(f"训练iou分数: {results['train_ious'][best_epoch-1]:.4f}")
    # print(f"测试损失: {results['test_loss'][best_epoch-1]:.4f}")
    # print(f"测试Dice分数: {results['test_dicescore'][best_epoch-1]:.4f}")
    # print(f"测试iou分数: {results['test_ious'][best_epoch-1]:.4f}")
    return
###--------------- 保存模型 ------------------###



# 实例化并使用 DiceCoefficient 类
# dice_calculator = DiceLoss(epsilon=1)
# mean_dice_loss = dice_calculator(testoutput.detach().cpu().squeeze(), idx_mask)
# # 输出结果
# print("平均 Dice 系数: {:.4f}".format(mean_dice_loss))
# # for i, score in enumerate(dice_loss_per_channel):
# #     print("通道 {} 的 Dice 系数: {:.4f}".format(i, score))