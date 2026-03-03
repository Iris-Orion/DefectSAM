import pytz
import torch
import time
import wandb
import swanlab
import baseline.modelLib as modelLib
import argparse
from tqdm import tqdm
from datetime import datetime
from monai.metrics import HausdorffDistanceMetric

from utils.utils import compute_dice_score, compute_iou_score, print_trainable_parameters, save_model, save_training_logs

def create_bsl_model_from_type(args: argparse.Namespace):
    model_choice = args.bse_model

    print(f"--- Creating model of type: {model_choice}")

    model_map = {
        "vanilla_unet": lambda: modelLib.get_vanilla_unet(n_channels=3, num_classes=1),
        "unet_res34": lambda: modelLib.get_smp_unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None),
        "deeplabv3plus_effb0": lambda: modelLib.get_smp_deeplabv3plus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=1, activation=None),
        "deeplabv3plus_effb3": lambda: modelLib.get_smp_deeplabv3plus(encoder_name="efficientnet-b3", encoder_weights="imagenet", in_channels=3, classes=1, activation=None),
        "segformer_b0": lambda: modelLib.build_segformer_model(encoder_name="mit_b0", encoder_weights="imagenet", in_channels=3, classes=1, activation=None),
        "segformer_b2": lambda: modelLib.build_segformer_model(encoder_name="mit_b2", encoder_weights="imagenet", in_channels=3, classes=1, activation=None),
    }

    if model_choice not in model_map:
        raise ValueError(f"Unknown model choice: {model_choice}")

    return model_map[model_choice]()

def train_one_epoch(model, data_loader, criterion, optimizer, scheduler, device):
    """
    Returns: tuple: (平均损失, 平均Dice分数, 平均IoU分数)
    """
    model.train()  # 设置模型为训练模式
    total_loss, total_dicescore, total_ious = 0, 0, 0
    for images, masks, _ in tqdm(data_loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        # masks = masks.unsqueeze(1).to(device) # 如果需要，取消注释

        outputs = model(images)
        loss = criterion(outputs, masks)                            # 前向传播
        
        total_dicescore += compute_dice_score(outputs, masks)       # 为日志记录计算指标
        total_ious += compute_iou_score(outputs, masks)
        
        optimizer.zero_grad()   # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 若是 torch.optim.lr_scheduler.StepLR等类型，则应该按照epoch 调整学习率， TODO: 判断调度器类型：
        scheduler.step()        # 更新学习率
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    avg_dicescore = total_dicescore / len(data_loader)
    avg_ious = total_ious / len(data_loader)

    return avg_loss, avg_dicescore, avg_ious

def evaluate(model, data_loader, criterion, device):
    """
    Returns:  tuple: (平均损失, 平均Dice分数, 平均IoU分数)
    """
    model.eval()  # 设置模型为评估模式
    total_loss, total_dicescore, total_ious = 0, 0, 0

    # 使用95%分位数HD更稳健，能有效避免离群点的极端影响
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0)
    mean_hd95 = 0
    with torch.no_grad():  # 在此模式下，不计算梯度
        for images, masks, _ in tqdm(data_loader, desc=f"Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            # masks = masks.unsqueeze(1).to(device) # 如果需要，取消注释

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # 计算dice和iou
            total_dicescore += compute_dice_score(outputs, masks)
            total_ious += compute_iou_score(outputs, masks)

            # 1. 在GPU上完成模型的数学运算，然后将结果的副本移至CPU
            pred_masks_binary_cpu = (torch.sigmoid(outputs) > 0.5).float().cpu()

            # 2. 将对应的真实标签的副本也移至CPU
            masks_cpu = masks.cpu()

            # 3. 将CPU上的张量喂给metric对象进行累积
            hd95_metric(y_pred=pred_masks_binary_cpu, y=masks_cpu)

    avg_loss = total_loss / len(data_loader)
    avg_dicescore = total_dicescore / len(data_loader)
    avg_ious = total_ious / len(data_loader)
    mean_hd95 = hd95_metric.aggregate().item()

    hd95_metric.reset()
    return avg_loss, avg_dicescore, avg_ious, mean_hd95

def baseline_experiment(model, device, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, hyperparameters, save_best_model=True):
    print("Hyperparameters:", hyperparameters)
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")

    # 2. 初始化 WandB (如果开启)
    wandb_run = None
    if hyperparameters.get('use_wandb', False):
        wandb_run = wandb.init(
            project=hyperparameters.get('wandb_project', 'retina_project'),
            config=hyperparameters, # 自动记录所有超参数
            name=f"{hyperparameters.get('task_name')}_{start_timestamp}"
        )

    # Initialize SwanLab
    swanlab_run = None
    if hyperparameters.get('use_swanlab', False):
        swanlab_run = swanlab.init(
            project=hyperparameters.get('swanlab_project', 'retina_project'),
            experiment_name=f"{hyperparameters.get('task_name')}_{start_timestamp}",
            config=hyperparameters,
        )

    num_epochs = hyperparameters.get("num_epochs", 100)
    earlystop_PATIENCE = hyperparameters.get("patience", 10)
    earlystop_MIN_DELTA = hyperparameters.get("min_delta", 0.0001)
    task_name = hyperparameters.get("task_name", "unnamed_task")
    output_dir = hyperparameters.get('output_dir', "unset_output_dir")

    # 用于记录每个epoch的过程性结果
    history = {
        "train_loss": [], "train_dicescore": [], "train_ious": [],
        "val_loss": [], "val_dicescore": [], "val_ious": []
    }

    model.to(device)
    best_val_dice = 0.0
    best_model_path = ""
    no_improve_epochs = 0
    best_epoch = -1
    print_trainable_parameters(model)

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        avg_train_loss, avg_train_dicescore, avg_train_ious = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        print(f'Train Results: Loss: {avg_train_loss:.4f}, Dice: {avg_train_dicescore:.4f}, IoU: {avg_train_ious:.4f}')

        avg_val_loss, avg_val_dicescore, avg_val_ious, avg_val_hd95 = evaluate(
            model, val_loader, criterion, device
        )
        print(f'Validation Results: Loss: {avg_val_loss:.4f}, Dice: {avg_val_dicescore:.4f}, IoU: {avg_val_ious:.4f}')

        # if scheduler is not None:
        #     scheduler.step()
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Current learning rate (param_group {i}): {param_group['lr']:.6e}")

        history["train_loss"].append(avg_train_loss)
        history["train_dicescore"].append(avg_train_dicescore)
        history["train_ious"].append(avg_train_ious)
        history["val_loss"].append(avg_val_loss)
        history["val_dicescore"].append(avg_val_dicescore)
        history["val_ious"].append(avg_val_ious)

        # 3. 将结果同步到 WandB
        if wandb_run:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/dice": avg_train_dicescore,
                "train/iou": avg_train_ious,
                "val/loss": avg_val_loss,
                "val/dice": avg_val_dicescore,
                "val/iou": avg_val_ious,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        if swanlab_run:
            swanlab.log({
                "train/loss": avg_train_loss,
                "train/dice": avg_train_dicescore,
                "train/iou": avg_train_ious,
                "val/loss": avg_val_loss,
                "val/dice": avg_val_dicescore,
                "val/iou": avg_val_ious,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch+1)

        # 早停逻辑和模型保存
        if avg_val_dicescore - best_val_dice > earlystop_MIN_DELTA:
            best_val_dice = avg_val_dicescore
            no_improve_epochs = 0
            best_epoch = epoch + 1
            history['best_epoch'] = best_epoch
            
            print(f"Validation dice 改善到 {best_val_dice:.4f}. Saving model...")
            if save_best_model:
                best_model_path = save_model(
                    hyperparameters=hyperparameters, start_timestamp=start_timestamp, model=model,
                    optimizer=optimizer, scaler=None, epoch=epoch + 1, model_name=task_name,
                    target_dir=output_dir,
                    SAVE_HUGGINGFACE_PRETRAINED_MODEL=False,
                    save_lora_only=False
                )
            
            if swanlab_run:
                swanlab.log({
                    "best_val_dicescore": best_val_dice,
                    "best_epoch": best_epoch
                    })
        else:
            no_improve_epochs += 1
            print(f"Validation dice 未改善. No improvement for {no_improve_epochs} consecutive epochs.")
        
        # 保存每个epoch的日志
        output_data = save_training_logs(
            hyperparameters=hyperparameters, results=history, epoch=epoch+1,
            start_timestamp=start_timestamp, result_name=task_name, target_dir=output_dir
        )

        if no_improve_epochs >= earlystop_PATIENCE:
            print(f"Early stopping triggered after {earlystop_PATIENCE} epochs without improvement.")
            break
            
    # --- 训练结束后，使用最佳模型进行最终评估 ---
    print("\n--- Training finished. Starting final evaluation with the best model. ---")
    if not best_model_path:
        print("Warning: No best model was saved. Final evaluation will be on the last state of the model.")
    else:
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 在测试集上评估
    final_test_loss, final_test_dice, final_test_iou, final_test_hd95 = evaluate(model, test_loader, criterion, device)
    print(f'Final Test Set Evaluation: Loss: {final_test_loss:.4f}, Dice: {final_test_dice:.4f}, IoU: {final_test_iou:.4f}')
    history["final_test_metrics"]={"loss": final_test_loss, "dice": final_test_dice, "iou": final_test_iou}
    
    # 将最终结果保存到日志文件中
    save_training_logs(
        hyperparameters=hyperparameters, results=history, epoch=output_data["last_completed_epoch"],
        start_timestamp=start_timestamp, result_name=task_name, target_dir=output_dir,
    )

    # 4. 训练结束，记录最终测试结果并关闭
    if wandb_run:
        wandb.run.summary["best_val_dice"] = best_val_dice
        wandb.run.summary["final_test_dice"] = final_test_dice
        wandb.finish()
    
    if swanlab_run:
        swanlab.log({"final_test_dice": final_test_dice, 
                     "final_test_iou": final_test_iou})
        swanlab.finish()

    return history, model


def bsl_inference_engine(model, best_model_path, 
                        train_dataloader, val_dataloader, test_dataloader, 
                        loss_fn,  device, 
                        results_filename="evaluation_results.txt",
                        eval_traindataset = False):

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model = model
    loaded_model.to(device)
    print("Successfully loaded model from standard PyTorch checkpoint.")
    
    print("\n--- Evaluating on Test Set ---")
    _, test_dicescore, test_ious, test_hd95 =  evaluate(loaded_model, test_dataloader, loss_fn, device=device)

    print("\n--- Evaluating on Validation Set ---")
    _, val_dicescore, val_ious, val_hd95 =  evaluate(loaded_model, val_dataloader, loss_fn, device=device)

    train_results = None
    if eval_traindataset:
        print("\n--- Evaluating on Training Set (optional) ---")
        _, train_dicescore, train_ious, train_hd95 = evaluate(loaded_model, train_dataloader, loss_fn, device=device)

    # --- 整理结果 ---      
    print("\n--- Summary of Results ---")
    if eval_traindataset:
        print(f"Training Set:   Dice={train_dicescore:.4f}, IoU={train_ious:.4f}, HD95={train_hd95:.4f}")
    print(f"Validation Set: Dice={val_dicescore:.4f},  IoU={val_ious:.4f}, HD95={val_hd95:.4f}")
    print(f"Test Set:     Dice={test_dicescore:.4f}, IoU={test_ious:.4f}, HD95={test_hd95:.4f}")

    # --- 将路径和结果保存到TXT文件 ---
    print(f"\n--- Saving results to {results_filename} ---") # <-- Use the parameter here
    try:
        with open(results_filename, "a", encoding="utf-8") as f: # <-- And here
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint Path: {best_model_path}\n")
            f.write("\n--- Evaluation Metrics ---\n")
            if eval_traindataset:
                f.write(f"Training Set:   Dice={train_dicescore:.4f}, IoU={train_ious:.4f}, HD95={train_hd95:.4f}\n")
            f.write(f"Validation Set: Dice={val_dicescore:.4f},  IoU={val_ious:.4f}, HD95={val_hd95:.4f}\n")
            f.write(f"Test Set:     Dice={test_dicescore:.4f}, IoU={test_ious:.4f}, HD95={test_hd95:.4f}\n")
            f.write("="*50 + "\n\n")
        print("Results successfully saved.")
    except Exception as e:
        print(f"Error saving results to file: {e}")
    
    # 推理完毕，手动释放模型显存
    del loaded_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def baseline_experiment_archieve_v2(model, device, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, hyperparameters, save_best_model=True):
    print(hyperparameters)
    shanghai_tz = pytz.timezone('Asia/Shanghai')                              # 设置时区为亚洲/上海
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳

    num_epochs = hyperparameters.get("num_epochs", 100)
    earlystop_PATIENCE = hyperparameters.get("patience", 10)
    earlystop_MIN_DELTA = hyperparameters.get("min_delta", 0.0001)
    task_name = hyperparameters.get("task_name", "unnamed_task")
    output_dir = hyperparameters.get('output_dir', "unset_output_dir")

    results = {
        "train_loss": [], "train_dicescore": [], "train_ious": [],
        "val_loss": [], "val_dicescore": [], "val_ious": []
    }

    model.to(device)
    best_val_dice = 0.0     # 初始化最佳验证 dice 分数
    no_improve_epochs = 0   # 连续无改进的 epoch 计数
    print_trainable_parameters(model)

    for epoch in range(num_epochs):
        model.train()        # 训练
        train_loss, train_dicescore, train_ious = 0, 0, 0
        for images, masks, _ in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            # masks = masks.unsqueeze(1).to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)
            train_dicescore += compute_dice_score(outputs, masks)   ## 计算dice 和 iou
            train_ious += compute_iou_score(outputs, masks)
            
            # print(f"images.shape: {images.shape}, masks.shape: {masks.shape}, outputs.shape: {outputs.shape}")
            # print(f"image.type: {images.type()}, mask.type: {masks.type()}, outputs.type: {outputs.type()}")
            # print(f"Unique values in images: {torch.unique(images)}")
            # print(f"Unique values in masks: {torch.unique(masks)}")
            # print(f"Unique values in outputs: {torch.unique(outputs)}")
            # print(f"loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dicescore = train_dicescore / len(train_loader)
        avg_train_ious = train_ious / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] \nTrain Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dicescore:.4f}, Train IoU: {avg_train_ious:.4f}')

        if scheduler is not None:
            scheduler.step()
        
        for i, param_group in enumerate(optimizer.param_groups):
                current_lr = param_group['lr']
                print(f"当前学习率 (param_group {i}): {current_lr:.6e}")

        model.eval()          # 验证
        val_loss, val_dicescore, val_ious = 0, 0, 0
        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                # masks = masks.unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                ## 计算dice 和 miou
                val_dicescore += compute_dice_score(outputs, masks)
                val_ious += compute_iou_score(outputs, masks)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_dicescore = val_dicescore / len(test_loader)
            avg_val_ious = val_ious / len(test_loader)
            print(f'Validation Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dicescore:.4f}, Val IoU: {avg_val_ious:.4f}')
        
        results["train_loss"].append(avg_train_loss)
        results["train_dicescore"].append(avg_train_dicescore)
        results["train_ious"].append(avg_train_ious)
        results["val_loss"].append(avg_val_loss)
        results["val_dicescore"].append(avg_val_dicescore)
        results["val_ious"].append(avg_val_ious)

        save_training_logs( hyperparameters = hyperparameters, 
                            results = results, 
                            epoch = epoch+1,    
                            start_timestamp = start_timestamp, 
                            result_name = task_name,
                            target_dir= output_dir)

        # early stopping逻辑：这里以验证 dice 为监控指标，只有当提升大于 min_delta 时才认为有改进
        if avg_val_dicescore - best_val_dice > earlystop_MIN_DELTA:
            best_val_dice = avg_val_dicescore
            no_improve_epochs = 0
            print(f"验证 dice 改善到 {best_val_dice:.4f}")
            # 当验证指标改善时保存模型
            if save_best_model:
                model_path = save_model(
                                        hyperparameters=hyperparameters, 
                                        start_timestamp=start_timestamp,
                                        model=model, 
                                        optimizer=optimizer,
                                        scaler=None,
                                        epoch=epoch+1,  # 记录epoch数，从 1 开始
                                        model_name= task_name ,
                                        target_dir= output_dir,
                                        SAVE_HUGGINGFACE_PRETRAINED_MODEL=False,
                                        save_lora_only = False 
                                        )
        else:
            no_improve_epochs += 1
            print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")

        # 若连续无改进达到 patience 数, 则提前停止训练
        if no_improve_epochs >= earlystop_PATIENCE:
            print(f"验证指标连续 {earlystop_PATIENCE} 个 epoch 无改善，提前停止训练。")
            break
    return results, model

def baseline_experiment_archieve(model, device, train_loader, test_loader, criterion, optimizer, scheduler, hyperparameters, save_best_model=True):
    """
    存档baseline engine的实现
    """
    print(hyperparameters)
    shanghai_tz = pytz.timezone('Asia/Shanghai')                              # 设置时区为亚洲/上海
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳

    num_epochs = hyperparameters.get("num_epochs", 100)
    earlystop_PATIENCE = hyperparameters.get("patience", 10)
    earlystop_MIN_DELTA = hyperparameters.get("min_delta", 0.0001)
    task_name = hyperparameters.get("task_name", "unnamed_task")
    output_dir = hyperparameters.get('output_dir', "unset_output_dir")

    results = {
        "train_loss": [], "train_dicescore": [], "train_ious": [],
        "test_loss": [], "test_dicescore": [], "test_ious": []
    }

    model.to(device)
    best_val_dice = 0.0     # 初始化最佳验证 dice 分数
    no_improve_epochs = 0   # 连续无改进的 epoch 计数
    print_trainable_parameters(model)

    for epoch in range(num_epochs):
        model.train()        # 训练
        train_loss, train_dicescore, train_ious = 0, 0, 0
        for images, masks, _ in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            # masks = masks.unsqueeze(1).to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)
            train_dicescore += compute_dice_score(outputs, masks)   ## 计算dice 和 iou
            train_ious += compute_iou_score(outputs, masks)
            
            # print(f"images.shape: {images.shape}, masks.shape: {masks.shape}, outputs.shape: {outputs.shape}")
            # print(f"image.type: {images.type()}, mask.type: {masks.type()}, outputs.type: {outputs.type()}")
            # print(f"Unique values in images: {torch.unique(images)}")
            # print(f"Unique values in masks: {torch.unique(masks)}")
            # print(f"Unique values in outputs: {torch.unique(outputs)}")
            # print(f"loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dicescore = train_dicescore / len(train_loader)
        avg_train_ious = train_ious / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] \nTrain Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dicescore:.4f}, Train IoU: {avg_train_ious:.4f}')

        if scheduler is not None:
            scheduler.step()
        
        for i, param_group in enumerate(optimizer.param_groups):
                current_lr = param_group['lr']
                print(f"当前学习率 (param_group {i}): {current_lr:.6e}")

        model.eval()          # 验证
        test_loss, test_dicescore, test_ious = 0, 0, 0
        with torch.no_grad():
            for images, masks, _ in tqdm(test_loader):
                images = images.to(device)
                masks = masks.to(device)
                # masks = masks.unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                test_loss += loss.item()
                
                ## 计算dice 和 miou
                test_dicescore += compute_dice_score(outputs, masks)
                test_ious += compute_iou_score(outputs, masks)
            avg_test_loss = test_loss / len(test_loader)
            avg_test_dicescore = test_dicescore / len(test_loader)
            avg_test_ious = test_ious / len(test_loader)
            print(f'Validation Loss: {avg_test_loss:.4f}, Val Dice: {avg_test_dicescore:.4f}, Val IoU: {avg_test_ious:.4f}')
        
        results["train_loss"].append(avg_train_loss)
        results["train_dicescore"].append(avg_train_dicescore)
        results["train_ious"].append(avg_train_ious)
        results["test_loss"].append(avg_test_loss)
        results["test_dicescore"].append(avg_test_dicescore)
        results["test_ious"].append(avg_test_ious)

        save_training_logs( hyperparameters = hyperparameters, 
                            results = results, 
                            epoch = epoch+1,    
                            start_timestamp = start_timestamp, 
                            result_name = task_name,
                            target_dir= output_dir)

        # early stopping逻辑：这里以验证 dice 为监控指标，只有当提升大于 min_delta 时才认为有改进
        if avg_test_dicescore - best_val_dice > earlystop_MIN_DELTA:
            best_val_dice = avg_test_dicescore
            no_improve_epochs = 0
            print(f"验证 dice 改善到 {best_val_dice:.4f}")
            # 当验证指标改善时保存模型
            if save_best_model:
                save_model(
                    hyperparameters=hyperparameters, 
                    start_timestamp=start_timestamp,
                    model=model, 
                    optimizer=optimizer,
                    scaler=None,
                    epoch=epoch+1,  # 记录epoch数，从 1 开始
                    model_name= task_name ,
                    target_dir= output_dir,
                    SAVE_HUGGINGFACE_PRETRAINED_MODEL=False,
                    save_lora_only = False
                )
        else:
            no_improve_epochs += 1
            print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")

        # 若连续无改进达到 patience 数, 则提前停止训练
        if no_improve_epochs >= earlystop_PATIENCE:
            print(f"验证指标连续 {earlystop_PATIENCE} 个 epoch 无改善，提前停止训练。")
            break
    return results, model


def sd900_baseline_engine(model, device, train_loader, test_loader, criterion, optimizer, scheduler, hyperparameters):
    shanghai_tz = pytz.timezone('Asia/Shanghai')        # 设置时区为亚洲/上海
    model.to(device)
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳
    best_val_dice = -1.0    # 初始化最佳验证 dice 分数
    no_improve_epochs = 0   # 连续无改进的 epoch 计数

    num_epochs = hyperparameters.get("num_epoch", 100)
    earlystop_PATIENCE = hyperparameters.get("patience", 10)
    earlystop_MIN_DELTA = hyperparameters.get("min_delta", 0.0001)

    results = {
        "train_loss": [],
        "train_dicescore": [],
        "train_ious": [],
        "test_loss": [],
        "test_dicescore": [],
        "test_ious": []
    }
    model.train()
    print_trainable_parameters(model)
    for epoch in range(num_epochs):
        epoch_loss, train_dicescore, train_ious = 0, 0, 0
        for images, masks, labels in train_loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)

            outputs = model(images)
            # logging.info(f"outputs.shape: {outputs.shape}, masks.shape: {masks.shape}")
            loss = criterion(outputs, masks)
            ## 计算dice 和 miou
            train_dicescore += compute_dice_score(outputs, masks)
            train_ious += compute_iou_score(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_dicescore = train_dicescore / len(train_loader)
        train_ious = train_ious / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] \nTrain Loss: {train_loss:.4f}, Train Dice: {train_dicescore:.4f}, Train IoU: {train_ious:.4f}')

        # 验证
        model.eval()
        val_loss, val_dicescore, val_ious = 0, 0, 0
        with torch.no_grad():
            for images, masks, labels in test_loader:
                images = images.to(device)
                masks = masks.unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                ## 计算dice 和 miou
                val_dicescore += compute_dice_score(outputs, masks)
                val_ious += compute_iou_score(outputs, masks)
            avg_val_loss = val_loss / len(test_loader)
            val_dicescore = val_dicescore / len(test_loader)
            val_ious = val_ious / len(test_loader)
            print(f'Validation Loss: {avg_val_loss:.4f}, Val Dice: {val_dicescore:.4f}, Val IoU: {val_ious:.4f}')
        results["train_loss"].append(train_loss)
        results["train_dicescore"].append(train_dicescore)
        results["train_ious"].append(train_ious)
        results["test_loss"].append(avg_val_loss)
        results["test_dicescore"].append(val_dicescore)
        results["test_ious"].append(val_ious)

        save_training_logs(hyperparameters = hyperparameters, 
                           results = results, 
                           epoch = epoch+1, 
                           start_timestamp = start_timestamp, 
                           result_name = hyperparameters['task_name'] + '_',
                           target_dir="./new_weights/sd900_output")

        # early stopping逻辑：这里以验证 dice 为监控指标，只有当提升大于 min_delta 时才认为有改进
        if val_dicescore - best_val_dice > earlystop_MIN_DELTA:
            best_val_dice = val_dicescore
            no_improve_epochs = 0
            print(f"验证 dice 改善到 {best_val_dice:.4f}, 保存模型...")
            # 当验证指标改善时保存模型
            save_model(
                hyperparameters=hyperparameters, 
                start_timestamp=start_timestamp,
                model=model, 
                optimizer=optimizer,
                scaler=None,
                epoch=epoch+1,  # 记录epoch数，从 1 开始
                model_name= hyperparameters['task_name'] + '_',
                target_dir="./new_weights/sd900_output",
                SAVE_HUGGINGFACE_PRETRAINED_MODEL=False,
                save_lora_only = False
            )
        else:
            no_improve_epochs += 1
            print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
        
        # 若连续无改进达到 patience 数, 则提前停止训练
        if no_improve_epochs >= earlystop_PATIENCE:
            print(f"验证指标连续 {earlystop_PATIENCE} 个 epoch 无改善，提前停止训练。")
            break
        model.train()

def neu_seg_baseline_engine(model, device, train_loader, test_loader, criterion, optimizer, scheduler, hyperparameters):
    shanghai_tz = pytz.timezone('Asia/Shanghai')                              # 设置时区为亚洲/上海
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳

    num_epochs = hyperparameters.get("num_epochs", 100)
    earlystop_PATIENCE = hyperparameters.get("patience", 10)
    earlystop_MIN_DELTA = hyperparameters.get("min_delta", 0.0001)

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
    for epoch in range(num_epochs):
        epoch_loss, train_dicescore, train_ious = 0, 0, 0
        for images, masks, image_ids in tqdm(train_loader):
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
            scheduler.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_dicescore = train_dicescore / len(train_loader)
        train_ious = train_ious / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] \nTrain Loss: {train_loss:.4f}, Train Dice: {train_dicescore:.4f}, Train IoU: {train_ious:.4f}')

        # 验证
        model.eval()
        test_loss, test_dicescore, test_ious = 0, 0, 0
        with torch.no_grad():
            for images, masks, image_ids in tqdm(test_loader):
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

        save_training_logs( hyperparameters = hyperparameters, 
                            results = results, 
                            epoch = epoch+1,    
                            start_timestamp = start_timestamp, 
                            result_name = hyperparameters['task_name'] + '_',
                            target_dir="./new_weights/neu_seg_output")

        # early stopping逻辑：这里以验证 dice 为监控指标，只有当提升大于 min_delta 时才认为有改进
        if test_dicescore - best_val_dice > earlystop_MIN_DELTA:
            best_val_dice = test_dicescore
            no_improve_epochs = 0
            print(f"验证 dice 改善到 {best_val_dice:.4f}, 保存模型...")
            # 当验证指标改善时保存模型
            save_model(
                hyperparameters=hyperparameters, 
                start_timestamp=start_timestamp,
                model=model, 
                optimizer=optimizer,
                scaler=None,
                epoch=epoch+1,  # 记录epoch数，从 1 开始
                model_name= hyperparameters['task_name'] + '_',
                target_dir="./new_weights/neu_seg_output",
                SAVE_HUGGINGFACE_PRETRAINED_MODEL=False,
                save_lora_only = False
            )
        else:
            no_improve_epochs += 1
            print(f"验证 dice 未改善, 已连续 {no_improve_epochs} 个 epoch 没有提升.")
        
        # 若连续无改进达到 patience 数, 则提前停止训练
        if no_improve_epochs >= earlystop_PATIENCE:
            print(f"验证指标连续 {earlystop_PATIENCE} 个 epoch 无改善，提前停止训练。")
            break

    return results, model

def magtile_baseline_engine(model, device, train_loader, test_loader, criterion, optimizer, scheduler, hyperparameters):
    shanghai_tz = pytz.timezone('Asia/Shanghai')        # 设置时区为亚洲/上海
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳s
    results = {
        "train_loss": [],
        "train_dicescore": [],
        "train_ious": [],
        "test_loss": [],
        "test_dicescore": [],
        "test_ious": []
    }
    num_epochs = hyperparameters.get("num_epochs", 100)
    earlystop_PATIENCE = hyperparameters.get("patience", 10)
    earlystop_MIN_DELTA = hyperparameters.get("min_delta", 0.0001)

    best_val_dice = -1.0    # 初始化最佳验证 dice 分数
    no_improve_epochs = 0   # 连续无改进的 epoch 计数

    print_trainable_parameters(model)
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_dicescore, train_ious = 0, 0, 0
        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
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
            scheduler.step()

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
        if test_dicescore - best_val_dice > earlystop_MIN_DELTA:
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
        if no_improve_epochs >= earlystop_PATIENCE:
            print(f"验证指标连续 {earlystop_PATIENCE} 个 epoch 无改善，提前停止训练。")
            break
    return results, model