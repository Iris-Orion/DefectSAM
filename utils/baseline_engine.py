import os
import pytz
import torch
import time
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
        "unet_res34": lambda: modelLib.get_smp_unet(encoder_name="resnet34", 
                                                    encoder_weights="imagenet", 
                                                    in_channels=3, classes=1, 
                                                    activation=None),
        "deeplabv3plus_effb0": lambda: modelLib.get_smp_deeplabv3plus(encoder_name="efficientnet-b0", 
                                                                      encoder_weights="imagenet", 
                                                                      in_channels=3, 
                                                                      classes=1, 
                                                                      activation=None),
        "deeplabv3plus_effb3": lambda: modelLib.get_smp_deeplabv3plus(encoder_name="efficientnet-b3", 
                                                                      encoder_weights="imagenet", 
                                                                      in_channels=3, 
                                                                      classes=1, 
                                                                      activation=None),
        "segformer_b0": lambda: modelLib.build_segformer_model(encoder_name="mit_b0", 
                                                               encoder_weights="imagenet", 
                                                               in_channels=3, 
                                                               classes=1, 
                                                               activation=None),
        "segformer_b2": lambda: modelLib.build_segformer_model(encoder_name="mit_b2", 
                                                               encoder_weights="imagenet", 
                                                               in_channels=3, 
                                                               classes=1, 
                                                               activation=None),
    }

    if model_choice not in model_map:
        raise ValueError(f"Unknown model choice: {model_choice}")

    return model_map[model_choice]()

def train_one_epoch(model, data_loader, criterion, optimizer, scheduler, device, scaler,
                    scheduler_per_batch=False, compute_hd95: bool = False):
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0
    # 训练阶段默认不计算 HD95：CPU 上的 EDT 既慢又占内存，
    # 容易把 DataLoader worker 拖到 OOM 被 SIGKILL。需要时可通过 --train_hd95 打开。
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0) if compute_hd95 else None
    use_amp = scaler is not None

    # nanoGPT 风格：pin_memory + non_blocking，由 CUDA 内部 copy stream 重叠传输与计算
    loader_iter = iter(data_loader)
    try:
        n_img, n_msk, n_meta = next(loader_iter)
        n_img = n_img.to(device, non_blocking=True)
        n_msk = n_msk.to(device, non_blocking=True)
    except StopIteration:
        return 0, 0, 0, 0

    pbar = tqdm(total=len(data_loader), desc="Training")

    while True:
        images, masks = n_img, n_msk

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)

        # prefetch next batch: async H2D transfer overlaps with backward below
        try:
            n_img, n_msk, n_meta = next(loader_iter)
            n_img = n_img.to(device, non_blocking=True)
            n_msk = n_msk.to(device, non_blocking=True)
            has_next = True
        except StopIteration:
            has_next = False

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler_per_batch and scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        with torch.no_grad():
            total_dice += compute_dice_score(outputs, masks)
            total_iou += compute_iou_score(outputs, masks)
            if hd95_metric is not None:
                hd95_metric(y_pred=(torch.sigmoid(outputs) > 0.5).float().cpu(), y=masks.cpu())
        pbar.update(1)
        if not has_next:
            break

    pbar.close()
    avg_loss = total_loss / len(data_loader)
    avg_dice = total_dice / len(data_loader)
    avg_iou = total_iou / len(data_loader)
    if hd95_metric is not None:
        avg_hd95 = hd95_metric.aggregate().item()
        hd95_metric.reset()
    else:
        avg_hd95 = float('nan')
    return avg_loss, avg_dice, avg_iou, avg_hd95

def evaluate(model, data_loader, criterion, device, scaler=None):
    """
    Returns:  tuple: (平均损失, 平均Dice分数, 平均IoU分数, 平均HD95)
    """
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95.0)
    use_amp = scaler is not None

    with torch.no_grad():
        for images, masks, _ in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks)
            total_loss += loss.item()

            total_dice += compute_dice_score(outputs, masks)
            total_iou += compute_iou_score(outputs, masks)

            hd95_metric(y_pred=(torch.sigmoid(outputs) > 0.5).float().cpu(), y=masks.cpu())

    avg_loss = total_loss / len(data_loader)
    avg_dice = total_dice / len(data_loader)
    avg_iou = total_iou / len(data_loader)
    avg_hd95 = hd95_metric.aggregate().item()
    hd95_metric.reset()

    return avg_loss, avg_dice, avg_iou, avg_hd95

def baseline_experiment(model, device, 
                        train_loader, val_loader, test_loader, 
                        criterion, optimizer, scheduler, 
                        hyperparameters, save_best_model=True, scheduler_per_batch=False):
    print("Hyperparameters:", hyperparameters)
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")

    swanlab_run = None
    if hyperparameters.get('use_swanlab', False):
        swanlab_run = swanlab.init(
            project=hyperparameters.get('swanlab_project', 'please name your swanlab experiment'),
            experiment_name=f"{hyperparameters.get('task_name')}_{start_timestamp}",
            config=hyperparameters,
        )

    num_epochs =            hyperparameters.get("num_epochs", 100)
    earlystop_PATIENCE =    hyperparameters.get("patience", 10)
    earlystop_MIN_DELTA =   hyperparameters.get("min_delta", 0.0001)
    disable_early_stop =    hyperparameters.get("disable_early_stop", False)
    task_name =             hyperparameters.get("task_name", "unnamed_task")
    output_dir =            hyperparameters.get('output_dir', "<dataset_dir>/<baseline_model>/")
    model_choice =          hyperparameters.get('bse_model', None)

    # 统一保存目录结构: <dataset_dir>/<baseline_model>/
    if model_choice:
        normalized_dir = os.path.normpath(output_dir)
        if os.path.basename(normalized_dir) != model_choice:
            output_dir = os.path.join(output_dir, model_choice)
            hyperparameters['output_dir'] = output_dir

    history = {
        "train_loss": [], "train_dice": [], "train_iou": [], "train_hd95": [],
        "val_loss": [], "val_dice": [], "val_iou": [], "val_hd95": [],
    }

    # TF32：几乎零精度损失，提升 matmul / cudnn 吞吐
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model.to(device)

    # torch.compile：静态图优化，减少 kernel launch overhead
    if hyperparameters.get('no_compile', False):
        print("torch.compile 已被 --no_compile 禁用")
    else:
        try:
            model = torch.compile(model, mode='default')
            print("torch.compile 已启用 (mode='default')")
        except Exception as e:
            print(f"torch.compile 不可用，跳过: {e}")

    # BF16 GradScaler：BF16 指数范围与 FP32 相同，无需梯度缩放
    _use_bf16 = torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda', enabled=not _use_bf16)
    print(f"AMP dtype: {'bfloat16' if _use_bf16 else 'float16'} | GradScaler enabled: {not _use_bf16}")

    best_val_dice = 0.0
    best_model_path = ""
    no_improve_epochs = 0
    best_epoch = -1
    print_trainable_parameters(model)

    train_compute_hd95 = hyperparameters.get('train_hd95', False)
    print(f"训练阶段 HD95 计算: {'启用' if train_compute_hd95 else '关闭 (默认)'}")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        avg_train_loss, avg_train_dice, avg_train_iou, avg_train_hd95 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler,
            scheduler_per_batch, compute_hd95=train_compute_hd95
        )
        print(f'Train Results: Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}, IoU: {avg_train_iou:.4f}, HD95: {avg_train_hd95:.4f}')

        avg_val_loss, avg_val_dice, avg_val_iou, avg_val_hd95 = evaluate(
            model, val_loader, criterion, device, scaler
        )
        print(f'Validation Results: Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}, HD95: {avg_val_hd95:.4f}')

        # 若未设定scheduler_per_batch = True进入下面这个分支，按照epoch来更新？
        if not scheduler_per_batch and scheduler is not None:
            scheduler.step()
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Current learning rate (param_group {i}): {param_group['lr']:.6e}")

        # 手动log每一轮的训练训练数据
        history["train_loss"].append(avg_train_loss)
        history["train_dice"].append(avg_train_dice)
        history["train_iou"].append(avg_train_iou)
        history["train_hd95"].append(avg_train_hd95)

        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)
        history["val_iou"].append(avg_val_iou)
        history["val_hd95"].append(avg_val_hd95)

        if swanlab_run:
            # swanlab log每一轮的训练训练数据
            swanlab.log({
                "train/loss": avg_train_loss,
                "train/dice": avg_train_dice,
                "train/iou": avg_train_iou,
                "train/hd95": avg_train_hd95,
                "val/loss": avg_val_loss,
                "val/dice": avg_val_dice,
                "val/iou": avg_val_iou,
                "val/hd95": avg_val_hd95,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch+1)

        # 早停逻辑和模型保存
        if avg_val_dice - best_val_dice > earlystop_MIN_DELTA:
            best_val_dice = avg_val_dice
            no_improve_epochs = 0
            best_epoch = epoch + 1
            history['best_epoch'] = best_epoch

            # 手动额外log验证集指标最好的那一轮的训练数据
            history['best_metrics'] = {
                "train_loss": avg_train_loss,
                "train_dice": avg_train_dice,
                "train_iou": avg_train_iou,
                "train_hd95": avg_train_hd95,
                "val_loss": avg_val_loss,
                "val_dice": avg_val_dice,
                "val_iou": avg_val_iou,
                "val_hd95": avg_val_hd95,
            }
            
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
                # log最好的轮次和最好的轮次对应的训练集和验证集上的指标
                swanlab.log({
                    "best_epoch": best_epoch,
                    "best/train_loss": avg_train_loss,
                    "best/train_dice": avg_train_dice,
                    "best/train_iou": avg_train_iou,
                    "best/train_hd95": avg_train_hd95,
                    "best/val_loss": avg_val_loss,
                    "best/val_dice": avg_val_dice,
                    "best/val_iou": avg_val_iou,
                    "best/val_hd95": avg_val_hd95,
                    })
        else:
            no_improve_epochs += 1
            print(f"Validation dice 未改善. No improvement for {no_improve_epochs} consecutive epochs.")
        
        # 保存每个epoch的日志
        output_data = save_training_logs(
            hyperparameters=hyperparameters, results=history, epoch=epoch+1,
            start_timestamp=start_timestamp, result_name=task_name, target_dir=output_dir
        )

        if (not disable_early_stop) and (no_improve_epochs >= earlystop_PATIENCE):
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
    test_loss, test_dice, test_iou, test_hd95 = evaluate(model, test_loader, criterion, device)
    print(f'Final Test Set Evaluation: Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, IoU: {test_iou:.4f}, HD95: {test_hd95:.4f}')
    history["final_test_metrics"]={"loss": test_loss, "dice": test_dice, "iou": test_iou, "hd95": test_hd95}
    
    # 将最终结果保存到日志文件中
    save_training_logs(
        hyperparameters=hyperparameters, results=history, epoch=output_data["last_completed_epoch"],
        start_timestamp=start_timestamp, result_name=task_name, target_dir=output_dir,
    )
    
    if swanlab_run:
        swanlab.log({"test/test_dice": test_dice, 
                     "test/test_iou": test_iou,
                     "test/test_hd95": test_hd95})
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

