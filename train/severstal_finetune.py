"""
支持单卡和 DDP 多卡训练

单卡训练:
    python -m train.severstal_finetune --batch_size 4 --device_id 1 --num_epoch 2 --save_custom_lora

多卡训练 (指定 GPU):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m train.severstal_finetune --batch_size 2 --num_epochs 2
"""
import os
import json
import monai
import torch
import copy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import SamModel

from data import severstal
from data.severstal import SteelDataset_WithBoxPrompt, SteelDataset_WithBoxPromptResizeInfer
from utils.config import get_severstal_ft_args
from utils.helper_function import set_seed, cleanup_ddp
from utils.finetune_engine import (
    run_finetune_engine,
    inference_engine,
    _process_batch_severstal,
    _process_batch_severstal_resize_infer,
    zero_shot,
)
from weights.severstal_wts import severstal_dict
from utils.loratask import create_model_from_type


def _load_single_weight_hyperparameters(json_path):
    if not json_path:
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for key in ("hyperparameters", "config", "args"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload if isinstance(payload, dict) else {}


def _build_single_checkpoint_info(args):
    hyperparameters = _load_single_weight_hyperparameters(args.single_weight_json)
    ft_type = hyperparameters.get("ft_type", args.ft_type)
    return {
        "path": args.single_weight_path,
        "type": ft_type,
        "save_custom_lora": hyperparameters.get("save_custom_lora", args.save_custom_lora),
        "save_hf_format": hyperparameters.get("save_hf_format", args.save_hf_format),
        "lora_rank": hyperparameters.get("lora_rank", args.lora_rank),
        "lora_alpha": hyperparameters.get("lora_alpha", args.lora_alpha),
    }


def main():
    ddp = int(os.environ.get('RANK', -1)) != -1
    ddp_rank = int(os.environ.get('RANK', 0)) if ddp else 0
    ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0)) if ddp else 0
    ddp_world_size = int(os.environ.get('WORLD_SIZE', 1)) if ddp else 1
    master_process = ddp_rank == 0

    args = get_severstal_ft_args()
    # 模型初始化前所有 rank 使用相同 seed，保证各进程初始权重一致（DDP 正确性前提）
    set_seed(args.seed, seed_offset=0)
    hyperparameters = vars(args)

    mini_dataset = args.mini_dataset            # for debugging purposes
    include_no_defect = args.include_no_defect

    hyperparameters['optimizer'] = "AdamW"
    hyperparameters['scheduler'] = "cosine_scheduler"
    hyperparameters['loss_function'] = "monai.DiceCELoss"
    hyperparameters['output_dir'] = './new_weights/finetune/severstal_output'
    hyperparameters['task_name'] = "severstal_" + hyperparameters['ft_type']
    if master_process:
        print(hyperparameters)

    data_path = "./data/severstal_steel_defect_detection"
    train_df, val_df, test_df  = severstal.traindf_preprocess(split_seed = 42,
                                                                train_ratio=0.6,
                                                                val_ratio = 0.2,
                                                                test_ratio = 0.2,
                                                                include_no_defect=include_no_defect,
                                                                create_mini_dataset=args.mini_dataset,
                                                                mini_size=256)
    train_transforms, val_transforms = severstal.get_severstal_ft_albumentations_transforms()

    use_resize_infer = (args.infer_mode or args.zero_shot) and args.severstal_infer_preprocess == "resize"
    process_batch_fn = _process_batch_severstal_resize_infer if use_resize_infer else _process_batch_severstal

    if use_resize_infer and master_process:
        print("Severstal inference preprocess: albumentations.Resize(1024, 1024)")

    if use_resize_infer:
        train_dataset = SteelDataset_WithBoxPromptResizeInfer(
            train_df, data_path=data_path, transforms=val_transforms, is_train=False, perturb_px=5
        )
        val_dataset = SteelDataset_WithBoxPromptResizeInfer(
            val_df, data_path=data_path, transforms=val_transforms, is_train=False, perturb_px=5
        )
        test_dataset = SteelDataset_WithBoxPromptResizeInfer(
            test_df, data_path=data_path, transforms=val_transforms, is_train=False, perturb_px=5
        )
    else:
        train_dataset = SteelDataset_WithBoxPrompt(train_df, data_path=data_path, transforms=train_transforms, is_train=True)
        val_dataset = SteelDataset_WithBoxPrompt(val_df, data_path=data_path, transforms=val_transforms, is_train=False, perturb_px=5)
        test_dataset = SteelDataset_WithBoxPrompt(test_df, data_path=data_path, transforms=val_transforms, is_train=False, perturb_px=5)

    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, drop_last=True) if ddp else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, drop_last=True) if ddp else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0)
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    # ---------- 设备选择 ----------
    if ddp:
        device = torch.device(f"cuda:{ddp_local_rank}")
    else:
        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

    if not args.infer_mode and not args.zero_shot:
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)

        run_finetune_engine(train_dataloader, val_dataloader, test_dataloader,
                            model, device, hyperparameters,
                            process_batch_fn=process_batch_fn,
                            save_dir = "./new_weights/finetune/severstal_output/" + hyperparameters['ft_type'],
                            auto_seg=args.auto_seg,
                            train_sampler=train_sampler)

    elif args.zero_shot:
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        if not ddp:
            device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

        SambPath = "./HuggingfaceModel/sam_vit_base/model"
        MedSamPath = "./HuggingfaceModel/wanglab/medsam-vit-base/model"
        zero_shot(model_path = MedSamPath,
                  train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                    loss_fn=seg_loss,
                    process_batch_fn=process_batch_fn,
                    device=device,
                    results_filename="evaluation_results.txt",
                    auto_seg=False,
                    eval_traindataset=args.severstal_eval_train,
                    eval_valdataset=args.severstal_eval_val)

    else:
        if args.single_weight_path:
            checkpoints_to_evaluate = [_build_single_checkpoint_info(args)]
        elif args.include_no_defect:
            checkpoints_to_evaluate = severstal_dict()
        else:
            checkpoints_to_evaluate = None
        if checkpoints_to_evaluate is None:
            raise ValueError("infer_mode 需要 --single_weight_path，或启用 include_no_defect 以使用 severstal_dict() 批量评估。")
        scaler = torch.amp.GradScaler(enabled=True)
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        if not ddp:
            device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")
        results_filename = (
            "severstal_resize_infer_evaluation_results.txt"
            if use_resize_infer else
            "severstal_evaluation_results.txt"
        )
        for checkpoint_info in checkpoints_to_evaluate:
            checkpoint_path = checkpoint_info["path"]
            loading_type = checkpoint_info["type"]

            current_model = None # 初始化

            print(f"================================================================")
            print(f"==> Path: {checkpoint_path}")
            print(f"==> Loading Type: {loading_type}")
            print(f"================================================================")

            current_args = copy.deepcopy(args)

            current_args.ft_type = loading_type
            current_args.save_custom_lora = checkpoint_info["save_custom_lora"]
            current_args.save_hf_format = checkpoint_info["save_hf_format"]
            current_args.lora_rank = checkpoint_info.get("lora_rank", current_args.lora_rank)
            current_args.lora_alpha = checkpoint_info.get("lora_alpha", current_args.lora_alpha)

            current_model = create_model_from_type(args = current_args, train_dataloader=train_dataloader)
            inference_engine(
                                model=current_model,
                                args=current_args,
                                best_model_path=checkpoint_path,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                test_dataloader=test_dataloader,
                                loss_fn=seg_loss,
                                process_batch_fn=process_batch_fn,
                                scaler=scaler,
                                device=device,
                                results_filename=results_filename,
                                auto_seg=False,
                                eval_traindataset=args.severstal_eval_train,
                                eval_valdataset=args.severstal_eval_val
                                                        )
            print(f"\n==> [INFERENCE COMPLETE] for: {checkpoint_path}\n\n")



if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup_ddp()
