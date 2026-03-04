import os
import pytz
import torch
import monai
import torch.nn.functional as F

from torch import nn
from transformers import SamModel
from peft import PeftModel, PeftConfig
from torch.optim import AdamW
from datetime import datetime
from torch.utils.data import DataLoader
from monai.losses import DiceLoss, FocalLoss
from transformers import get_cosine_schedule_with_warmup

import utils.finetune_engine as ut_ft
from data.data_utils_ft import bsd500_create_dataset, log_info_bsd500_dataset
from utils.utils import save_model, save_training_logs, print_trainable_parameters
from utils.config import get_common_ft_args
from utils.finetune_engine import inference_engine, create_model_from_type, _process_batch_with_point_grid, severstal_get_offset, _train_one_epoch, _evaluate, create_model_for_inference
from utils.helper_function import set_seed

class BalancedBCELoss(nn.Module):
    """
    类别平衡的二元交叉熵损失函数。
    专为解决边缘检测等任务中正负样本极度不平衡问题设计。
    """
    def __init__(self, epsilon=1e-6):
        super(BalancedBCELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算损失。
        :param pred_logits: 模型的原始输出 (logits), 在 sigmoid 激活之前。
        :param target: 真实标签 (ground truth), 值为 0 或 1。
        """
        target = target.float()
        if pred_logits.device != target.device:
            target = target.to(pred_logits.device)

        num_pos = torch.sum(target == 1).float().detach()
        num_neg = torch.sum(target == 0).float().detach()
        
        pos_weight = num_neg / (num_pos + self.epsilon)

        # 使用 F.binary_cross_entropy_with_logits，它需要 logits 作为输入
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, 
            target, 
            pos_weight=pos_weight
        )
        return loss

def bsd500_finetune_engine( train_dataloader, val_dataloader, test_dataloader, 
                            model, device, hyperparameters, 
                            process_batch_fn,
                            loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean'), 
                            save_dir = "./new_weights/unspec_output", auto_seg=False):
    """
    一个通用的微调训练引擎，适用于所有数据集。
    Args:
        process_batch_fn (function): 用于处理数据批次的函数。应为 `_process_batch` 或 `_process_batch_severstal`。
    """

    print_trainable_parameters(model)
    print(f"using devcie {device}")
    
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            print(name, param.shape)                # lora系列模型的插入位置信息debug

    shanghai_tz = pytz.timezone('Asia/Shanghai')                                # 设置时区为亚洲/上海
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")       # 获取当前时间戳

    SAVE_HUGGINGFACE_PRETRAINED_MODEL = hyperparameters['save_hf_format']
    save_lora_only = hyperparameters['save_custom_lora']
    if save_lora_only or SAVE_HUGGINGFACE_PRETRAINED_MODEL:
        log_name = hyperparameters["ft_type"] + "_rank_" + str(hyperparameters["lora_rank"])
    else:
        log_name = hyperparameters["ft_type"]

    print(f"权重保存格式: hugging face格式:{SAVE_HUGGINGFACE_PRETRAINED_MODEL} || lora格式:{save_lora_only}")

    # 从超参数中获取值
    lr = hyperparameters['learning_rate']
    wd = hyperparameters['weight_decay']
    num_epochs = hyperparameters['num_epochs']
    warmup_ratio = hyperparameters['warmup_ratio']
    patience = hyperparameters['patience']
    min_delta = hyperparameters['min_delta']
    
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]      # 收集所有可训练的参数
    optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=wd)
    # loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    scaler = torch.cuda.amp.GradScaler(enabled=True)  # 混合精度训练

    # 学习率调度策略 
    total_steps = len(train_dataloader) * num_epochs
    warmup_ratio = 0.1
    warmup_steps = int(warmup_ratio * total_steps)

    cosine_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles = 0.5
    )

    history = {"train_loss": [],"val_loss": []}
    no_improve_epochs = 0       # 连续无改进的 epoch 计数
    best_epoch = -1
    best_val_loss = float('inf') 

    model.to(device)
    model.train()

    offset_info = None
    if process_batch_fn == ut_ft._process_batch_severstal:
        print("Severstal-specific processing enabled. Calculating offset info.")
        offset_info = severstal_get_offset()
    
    best_model_path = None

    # 将 process_batch_fn 传递给训练周期函数
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")

        train_loss, _, _ = _train_one_epoch(model, train_dataloader, optimizer, 
                                                                   cosine_scheduler, loss_fn, process_batch_fn, scaler, device, auto_seg = auto_seg, offset_info = offset_info)
        print(f"Training loss: {train_loss:.4f}")         # training

        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Current learning rate (param_group {i}): {param_group['lr']:.6e}")  # 打印一个batch之后此时的学习率

        val_loss, _, _ = _evaluate(model, val_dataloader, loss_fn, process_batch_fn,
                                                      device, scaler, auto_seg = auto_seg, offset_info=offset_info)
        print(f'Val Loss: {val_loss:.4f}')          #validation

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - min_delta:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            no_improve_epochs = 0
            best_epoch = epoch + 1
            history['best_epoch'] = best_epoch
            best_model_path = save_model( hyperparameters=hyperparameters,
                                            start_timestamp = start_timestamp, 
                                            model=model, 
                                            optimizer=optimizer,
                                            scaler=scaler,
                                            epoch = epoch+1,
                                            model_name= log_name,
                                            target_dir= save_dir,
                                            SAVE_HUGGINGFACE_PRETRAINED_MODEL = SAVE_HUGGINGFACE_PRETRAINED_MODEL,
                                            save_lora_only=save_lora_only)
        else:
            no_improve_epochs += 1
            print(f"Val Loss 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")

        output_data = save_training_logs( hyperparameters = hyperparameters,
                                            results = history,
                                            epoch = epoch+1,
                                            start_timestamp = start_timestamp,
                                            result_name = log_name,
                                            target_dir = save_dir)
        if no_improve_epochs >= patience:
            print(f"验证指标连续 {patience} 个 epoch 无改善，提前停止训练。")
            break
        # --- 训练结束后，使用最佳模型进行最终评估 ---
    print("\n--- Training finished. Starting final evaluation with the best model. ---")
    if not best_model_path:
        print("Warning: No best model was saved. Final evaluation will be on the last state of the model.")
        loaded_model = model
    else:
        print(f"Loading best model from: {best_model_path}")
        if SAVE_HUGGINGFACE_PRETRAINED_MODEL:
            try:
                config = PeftConfig.from_pretrained(best_model_path)
                base_model_path = config.base_model_name_or_path
                base_model = SamModel.from_pretrained(base_model_path)
                
                loaded_model = PeftModel.from_pretrained(base_model, best_model_path)       # 从保存的路径加载 PeftModel
                loaded_model.to(device)
                print("Successfully loaded model in Hugging Face PEFT format.")
            except Exception as e:
                print(f"Error loading Hugging Face PEFT model: {e}")
                loaded_model = None            # 加载失败
        elif save_lora_only:
            loaded_model = create_model_for_inference(model=model, lora_weights_path=best_model_path, device=device)
        else:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            loaded_model = model
            print("Successfully loaded model from standard PyTorch checkpoint.")

    # 在测试集上评估
    final_test_loss, final_test_dice, final_test_iou = _evaluate(loaded_model, test_dataloader, loss_fn, process_batch_fn, device, scaler, auto_seg=auto_seg, offset_info = offset_info)
    print(f'Final Test Set Evaluation: Loss: {final_test_loss:.4f}, Dice: {final_test_dice:.4f}, IoU: {final_test_iou:.4f}')
    history["final_test_metrics"]={"loss": final_test_loss, "dice": final_test_dice, "iou": final_test_iou}

    # 将最终结果保存到日志文件中
    save_training_logs(
        hyperparameters=hyperparameters, results=history, epoch=output_data["last_completed_epoch"],
        start_timestamp=start_timestamp, result_name=log_name, target_dir=save_dir,
    )
    print("--- Training finished ---")
    return history, model

if __name__ == '__main__':
    args = get_common_ft_args()
    set_seed(42)
    hyperparameters = vars(args)

    hyperparameters['optimizer'] = "AdamW"
    hyperparameters['scheduler'] = "cosine_scheduler"
    hyperparameters['loss_function'] = "BalancedBCELoss"
    hyperparameters['output_dir'] = './new_weights/bsd500_output'

    train_dataset, val_dataset, test_dataset = bsd500_create_dataset(bsd_root_dir='./data/BSD500')
    log_info_bsd500_dataset(train_dataset, val_dataset, test_dataset)

    print("Hyperparameters:", hyperparameters)

    batch_size = args.batch_size
    num_workers = args.num_workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout

    device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")
    loss_fn = BalancedBCELoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    model = create_model_from_type(args=args, train_dataloader=train_loader)
    if not args.infer_mode:
        results, fintuned_model = bsd500_finetune_engine(train_dataloader=train_loader,
                                                        val_dataloader = val_loader,
                                                        test_dataloader=test_loader,
                                                        model=model,
                                                        device=device,
                                                        loss_fn = loss_fn,
                                                        process_batch_fn = _process_batch_with_point_grid,
                                                        hyperparameters=hyperparameters,
                                                        save_dir = os.path.join(hyperparameters['output_dir'], hyperparameters['ft_type']),
                                                        auto_seg = args.auto_seg)

    else:
        checkpoint_path='/workspace/DefectDetection/new_weights/sd900_output/loradsc_qv_rank_16_20250720_080218.pth'
        scaler = torch.cuda.amp.GradScaler(enabled=True) 
        loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        inference_engine (  model, args, best_model_path = checkpoint_path, 
                            train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader,
                            process_batch_fn = _process_batch_with_point_grid, loss_fn=loss_fn, scaler=scaler,
                            device=device, auto_seg = args.auto_seg, eval_traindataset=True)