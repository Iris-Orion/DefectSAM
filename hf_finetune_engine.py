import torch
import torch.nn.functional as F
from utils import compute_dice_score, compute_iou_score
from tqdm import tqdm
from utils import save_model
from datetime import datetime
import pytz

###########-----finetune 代码封装-----##############
def hfsam_finetune_train_step(model, 
                              dataloader: torch.utils.data.DataLoader, 
                              loss_fn: torch.nn.Module, 
                              optimizer: torch.optim.Optimizer, 
                              device: torch.device,
                              scaler: torch.cuda.amp.GradScaler = None,
                              use_amp: bool = True,):   ## 默认使用混合精度
    """
    Finetune hugging-face sam model for one epoch
    """
    model.train()

    train_loss, train_dicescore, train_ious = 0, 0, 0

    for batch in tqdm(dataloader):
        ground_truth_masks = batch["combined_mask"].float().to(device)      # gt
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            # print("正在使用amp")
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values = batch["image"].to(device),
                                input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                multimask_output=False)
                # 计算loss
                predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]
                ori_res_masks = F.interpolate(
                    predicted_masks,              # [B, 1, 256, 256]  ---interpolate---> [B, 1, 1024, 1024]
                    size=(1024, 1024),  
                    mode="bilinear",
                    align_corners=False
                )
                loss = loss_fn(ori_res_masks, ground_truth_masks.unsqueeze(1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            #  前向传播
            outputs = model(pixel_values = batch["image"].to(device),
                            input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                            multimask_output=False)
            
            # 计算loss
            predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]
            ori_res_masks = F.interpolate(
                predicted_masks,              # [B, 1, 256, 256]  ---interpolate---> [B, 1, 1024, 1024]
                size=(1024, 1024),  
                mode="bilinear",
                align_corners=False
            )
            loss = loss_fn(ori_res_masks, ground_truth_masks.unsqueeze(1))
            loss.backward()         # 反向传播
            optimizer.step()        # optimize

        train_loss += loss.item()
        train_dicescore += compute_dice_score(ori_res_masks, ground_truth_masks.unsqueeze(1))
        train_ious += compute_iou_score(ori_res_masks, ground_truth_masks.unsqueeze(1))
        
    train_loss = train_loss / len(dataloader)
    train_dicescore = train_dicescore / len(dataloader)
    train_ious = train_ious / len(dataloader)
    return train_loss, train_dicescore, train_ious

def hfsam_finetune_test_step(model, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device: torch.device):
    model.eval()
    test_loss, test_dicescore, test_ious = 0, 0, 0

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            #  前向传播
            outputs = model(pixel_values = batch["image"].to(device),
                            input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]   如果input_box为空，训练和测试的时候都要为空
                            multimask_output=False)
            
            # 计算loss
            predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]
            ground_truth_masks = batch["combined_mask"].float().to(device)      # gt
            ori_res_masks = F.interpolate(
                predicted_masks,              # [B, 1, 256, 256]  ---interpolate---> [B, 1, 1024, 1024]
                size=(1024, 1024),  
                mode="bilinear",
                align_corners=False
            )
            loss = loss_fn(ori_res_masks, ground_truth_masks.unsqueeze(1))
            test_loss += loss.item()
            test_dicescore += compute_dice_score(ori_res_masks, ground_truth_masks.unsqueeze(1))
            test_ious += compute_iou_score(ori_res_masks, ground_truth_masks.unsqueeze(1))
    test_loss = test_loss / len(dataloader)
    test_dicescore = test_dicescore / len(dataloader)
    test_ious= test_ious / len(dataloader)
    return test_loss, test_dicescore, test_ious

def hfsam_finetune(model, 
                   hyperparameters,
                   train_dataloader: torch.utils.data.DataLoader, 
                   test_dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   epochs: int,
                   device: torch.device,
                   patience: int = 7,
                   use_amp: bool = True,
                   lora_rank: int = 16):    ## 默认使用混合精度
    results = {
        "train_loss": [],
        "train_dicescore": [],
        "train_ious": [],
        "test_loss": [],
        "test_dicescore": [],
        "test_ious": []
    }
    model.to(device)
    
    best_test_dicescore = 0.0  # 记录最好的test_dicescore   TODO: 思考是用最小的loss的那一轮还是最大的dice score的那一轮
    best_model_wts = None
    best_epoch = 0
    no_improve_epochs = 0   # 连续无改进的 epoch 计数

    # 初始化GradScaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    shanghai_tz = pytz.timezone('Asia/Shanghai')        # 设置时区为亚洲/上海
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳

    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss, train_dicescore, train_ious = hfsam_finetune_train_step(model=model,
                                                 dataloader=train_dataloader,
                                                 loss_fn=loss_fn,
                                                 optimizer=optimizer,
                                                 device=device,
                                                 scaler=scaler,
                                                 use_amp=use_amp)
        # 考虑一下如果不早停会怎样
        test_loss, test_dicescore, test_ious = hfsam_finetune_test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss: 4f} | "
            f"train_dicescore: {train_dicescore: 4f} | "
            f"train_ious: {train_ious: 4f} | "
            f"test_loss: {test_loss: 4f} | "
            f"test_dicescore: {test_dicescore: 4f} | "
            f"test_ious: {test_ious: 4f} | "
            # f"cur_learningrate: {scheduler.get_last_lr()}"
        )
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
        
            save_model (hyperparameters=hyperparameters,
                        start_timestamp=start_timestamp, 
                        results=results, 
                        model=model, 
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        model_name="alldataset_depwiseconv_fixbug_" + str(lora_rank) + "_",
                        result_name="alldataset_depwiseconv_fixbug_" + str(lora_rank) + "_",
                        target_dir= "./model_output/severstal_output",
                        SAVE_HUGGINGFACE_PRETRAINED_MODEL = False)
        else:
            no_improve_epochs += 1
            print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
                # 若连续无改进达到 patience 数, 则提前停止训练
        if no_improve_epochs >= patience:
            print(f"验证指标连续 {patience} 个 epoch 无改善，提前停止训练。")
            break
    return results, model, best_epoch
    

###########-----finetune 代码封装-----##############

###########-----zero shot 代码封装-----##############
def hfsam_zeroshot(model, 
                   hyperparameters,
                   train_dataloader: torch.utils.data.DataLoader, 
                   test_dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   epochs: int,
                   device: torch.device,
                   patience: int = 7,
                   use_amp: bool = True,
                   lora_rank: int = 16):    ## 默认使用混合精度
    results = {
        "train_loss": [],
        "train_dicescore": [],
        "train_ious": [],
        "test_loss": [],
        "test_dicescore": [],
        "test_ious": []
    }
    model.to(device)

    # 初始化GradScaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    train_loss, train_dicescore, train_ious = hfsam_finetune_test_step(model=model,
                                                dataloader=train_dataloader,
                                                loss_fn=loss_fn,
                                                device=device)
    # 考虑一下如果不早停会怎样
    test_loss, test_dicescore, test_ious = hfsam_finetune_test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
    print(
        f"train_loss: {train_loss: 4f} | "
        f"train_dicescore: {train_dicescore: 4f} | "
        f"train_ious: {train_ious: 4f} | "
        f"test_loss: {test_loss: 4f} | "
        f"test_dicescore: {test_dicescore: 4f} | "
        f"test_ious: {test_ious: 4f} | "
        # f"cur_learningrate: {scheduler.get_last_lr()}"
    )

    return results, model

###########-----zero shot 代码封装-----##############


# ----打印训练参数情况----#
def print_trainable_parameters(model):
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")
# ----打印训练参数情况  ----#