import json
import os
import torch
import monai
import copy
import numpy as np
from torch.utils.data import DataLoader
from transformers import SamModel
from utils.finetune_engine import run_finetune_engine, inference_engine, _process_batch, zero_shot, create_model_from_type
from utils.helper_function import set_seed
from utils.config import get_common_ft_args
from data.retina_dataset import create_retina_dataset_ft, create_retina_dataset_ft_kfold

from weights.sd900_wts import sd900_dict


def build_retina_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    return train_dataloader, val_dataloader, test_dataloader


def summarize_kfold_results(fold_summaries, save_dir, ft_type, num_folds):
    """
    汇总 K 折结果，保存 mean/std，便于论文中直接引用。
    """
    if not fold_summaries:
        return None

    def _metric_stats(metric_name, section):
        values = [fold[section][metric_name] for fold in fold_summaries if fold[section].get(metric_name) is not None]
        if not values:
            return {"mean": None, "std": None}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    summary = {
        "ft_type": ft_type,
        "num_folds": num_folds,
        "fold_results": fold_summaries,
        "aggregate": {
            "best_val": {
                "dice": _metric_stats("dice", "best_val_metrics"),
                "iou": _metric_stats("iou", "best_val_metrics"),
                "hd95": _metric_stats("hd95", "best_val_metrics"),
            },
            "final_test": {
                "dice": _metric_stats("dice", "final_test_metrics"),
                "iou": _metric_stats("iou", "final_test_metrics"),
                "hd95": _metric_stats("hd95", "final_test_metrics"),
            },
        },
    }

    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, f"{ft_type}_{num_folds}fold_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    def _format_stats(stats):
        if stats["mean"] is None:
            return "N/A"
        return f"{stats['mean']:.4f} ± {stats['std']:.4f}"

    print("\n================ K-Fold Summary ================")
    print(f"Best Val Dice: {_format_stats(summary['aggregate']['best_val']['dice'])}")
    print(f"Best Val IoU:  {_format_stats(summary['aggregate']['best_val']['iou'])}")
    print(f"Best Val HD95: {_format_stats(summary['aggregate']['best_val']['hd95'])}")
    print(f"Test Dice:     {_format_stats(summary['aggregate']['final_test']['dice'])}")
    print(f"Test IoU:      {_format_stats(summary['aggregate']['final_test']['iou'])}")
    print(f"Test HD95:     {_format_stats(summary['aggregate']['final_test']['hd95'])}")
    print(f"K 折汇总已保存至: {summary_path}")
    print("================================================\n")

    return summary


def extract_best_val_metrics(results):
    """
    优先读取训练引擎记录的 best_metrics；若不存在，则从历史曲线中回退提取。
    """
    best_metrics = results.get('best_metrics')
    if best_metrics:
        return {
            "dice": best_metrics.get('val_dice'),
            "iou": best_metrics.get('val_iou'),
            "hd95": best_metrics.get('val_hd95'),
        }

    val_dice_history = results.get('val_dice', [])
    val_iou_history = results.get('val_iou', [])
    val_hd95_history = results.get('val_hd95', [])
    if not val_dice_history:
        return {"dice": None, "iou": None, "hd95": None}

    best_idx = int(np.argmax(val_dice_history))
    return {
        "dice": val_dice_history[best_idx],
        "iou": val_iou_history[best_idx] if best_idx < len(val_iou_history) else None,
        "hd95": val_hd95_history[best_idx] if best_idx < len(val_hd95_history) else None,
    }


if __name__ == '__main__':
    set_seed(42)
    args = get_common_ft_args()
    hyperparameters = vars(args)
    hyperparameters['optimizer'] = 'AdamW'
    hyperparameters['loss_function'] = 'monai.DiceCELoss'
    hyperparameters['scheduler'] = 'cosine'
    hyperparameters['task_name'] = "retina_" + hyperparameters['ft_type']

    print(hyperparameters)

    batch_size = args.batch_size
    num_workers = args.num_workers
    base_save_dir = "./new_weights/finetune/retina"

    # hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    # for name, param in hgsam_model.named_parameters():
    #     # 冻结某些层
    #     if name.startswith("vision_encoder"):
    #         param.requires_grad_(False)

    #----------------------- model 选择 --------------------------#
    # model = loratask.get_hf_lora_model(model = hgsam_model, target_part = 'mask_decoder')
    # model = loratask.get_hf_lora_model(model = hgsam_model, target_part = 'vision_encoder')
    # model = hgsam_model
    # model = loratask.get_hf_adalora_model(model = hgsam_model, target_part='vision_encoder', lora_rank=LORA_RANK, total_step=NUM_EPOCHS * len(train_dataloader))

    #----------------------- lora-dsc --------------------------#
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout

    device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

    if not args.infer_mode:
        if args.use_kfold:
            fold_indices = range(args.num_folds) if args.fold_index == -1 else [args.fold_index]
            fold_summaries = []

            for fold_index in fold_indices:
                print(f"\n================ Fold {fold_index + 1}/{args.num_folds} ================\n")
                train_dataset, val_dataset, test_dataset = create_retina_dataset_ft_kfold(
                    num_folds=args.num_folds,
                    fold_index=fold_index,
                    random_state=42,
                    shuffle=True,
                )
                train_dataloader, val_dataloader, test_dataloader = build_retina_dataloaders(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )

                fold_args = copy.deepcopy(args)
                fold_hyperparameters = vars(fold_args).copy()
                fold_hyperparameters['optimizer'] = 'AdamW'
                fold_hyperparameters['loss_function'] = 'monai.DiceCELoss'
                fold_hyperparameters['scheduler'] = 'cosine'
                fold_hyperparameters['task_name'] = f"retina_{fold_hyperparameters['ft_type']}_fold{fold_index + 1}"
                fold_hyperparameters['use_kfold'] = True
                fold_hyperparameters['num_folds'] = args.num_folds
                fold_hyperparameters['fold_index'] = fold_index

                model = create_model_from_type(args=fold_args, train_dataloader=train_dataloader)
                results, _ = run_finetune_engine(
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                    model=model,
                    device=device,
                    process_batch_fn=_process_batch,
                    hyperparameters=fold_hyperparameters,
                    save_dir=os.path.join(base_save_dir, f"fold_{fold_index + 1}"),
                    auto_seg=args.auto_seg,
                )

                best_val_metrics = extract_best_val_metrics(results)
                fold_summary = {
                    "fold_index": fold_index,
                    "best_epoch": results.get('best_epoch'),
                    "best_val_metrics": best_val_metrics,
                    "final_test_metrics": results.get('final_test_metrics', {}),
                }
                fold_summaries.append(fold_summary)

            summarize_kfold_results(
                fold_summaries=fold_summaries,
                save_dir=base_save_dir,
                ft_type=args.ft_type,
                num_folds=args.num_folds,
            )
        else:
            train_dataset, val_dataset, test_dataset = create_retina_dataset_ft()
            train_dataloader, val_dataloader, test_dataloader = build_retina_dataloaders(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            model = create_model_from_type(args=args, train_dataloader=train_dataloader)
            results, fintuned_model = run_finetune_engine(train_dataloader=train_dataloader,
                                                            val_dataloader = val_dataloader,
                                                            test_dataloader=test_dataloader,
                                                            model=model,
                                                            device=device,
                                                            process_batch_fn = _process_batch,
                                                            hyperparameters=hyperparameters,
                                                            save_dir = base_save_dir,
                                                            auto_seg = args.auto_seg)
    
    elif args.zero_shot:
        train_dataset, val_dataset, test_dataset = create_retina_dataset_ft()
        train_dataloader, val_dataloader, test_dataloader = build_retina_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

        SambPath = "./HuggingfaceModel/sam_vit_base/model"
        MedSamPath = "./HuggingfaceModel/wanglab/medsam-vit-base/model"
        zero_shot(model_path = SambPath,
                  train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                    loss_fn=seg_loss,
                    process_batch_fn=_process_batch,
                    device=device,
                    results_filename="sd900_zeroshot_evaluation_results.txt",
                    auto_seg=False,
                    eval_traindataset=True)
    else:
        train_dataset, val_dataset, test_dataset = create_retina_dataset_ft()
        train_dataloader, val_dataloader, test_dataloader = build_retina_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        scaler = torch.amp.GradScaler('cuda', enabled=True) 
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        # checkpoints_to_evaluate = sd900_dict()
        # checkpoints_to_evaluate = scale_sd900_dict()
        # checkpoints_to_evaluate = new_sd900_dict()
        checkpoints_to_evaluate = sd900_dict()

        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")
        for checkpoint_info in checkpoints_to_evaluate:
            checkpoint_path = checkpoint_info["path"]
            loading_type = checkpoint_info["type"]

            current_model = None # 初始化    

            # description = checkpoint_info["description"]
            print(f"================================================================")
            # print(f"==> [STARTING INFERENCE] for: {description}")
            print(f"==> Path: {checkpoint_path}")
            print(f"==> Loading Type: {loading_type}")
            print(f"================================================================")

            current_args = copy.deepcopy(args)          # 创建 args 的一个深拷贝，以防止循环间的副作用

            current_args.ft_type = loading_type
            # current_args.save_custom_lora = checkpoint_info["save_custom_lora"]
            # current_args.save_hf_format = checkpoint_info["save_hf_format"]
            # 如果 checkpoint_info 中没有 "lora_rank"，则使用 current_args 中已有的值（即默认值）
            current_args.lora_rank = checkpoint_info.get("lora_rank", current_args.lora_rank)
            current_args.lora_alpha = checkpoint_info.get("lora_alpha", current_args.lora_alpha)

            current_model = create_model_from_type(args = current_args, train_dataloader=train_dataloader)

            # checkpoint_path='/workspace/DefectDetection/new_weights/sd900_output/loradsc_qv_rank_16_20250720_080218.pth'
            inference_engine (  model=current_model,
                                args=current_args,  # 使用为本次循环特地配置的 args
                                best_model_path=checkpoint_path,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                test_dataloader=test_dataloader,
                                loss_fn=seg_loss,
                                process_batch_fn=_process_batch,
                                scaler=scaler,
                                device=device,
                                results_filename="sd900_eval_results.txt",
                                auto_seg=current_args.auto_seg,
                                eval_traindataset=True)

    # zero shot
    # train_dataloader, test_dataloader = sd_900_finetune_create_dataset()
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(device)
    # print(f"using devcie {device}")
    # hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    # medsam_model = SamModel.from_pretrained("./HuggingfaceModel/wanglab/medsam-vit-base/model")

    # model = medsam_model
    # model.to(device)
    # model.eval()
    # with torch.no_grad():
    #     train_dice, train_iou = inference_step(dataloader=train_dataloader,
    #                                         model=model,
    #                                         device=device,
    #                                         use_bbox=True)
        
    #     test_dice, test_iou = inference_step(dataloader=test_dataloader,
    #                                         model=model,
    #                                         device=device,
    #                                         use_bbox=True)
    # print(f"train dice: {train_dice:.4f}  train_iou: {train_iou:.4f} || test dice: {test_dice:.4f} test_iou: {test_iou:.4f}")

    
