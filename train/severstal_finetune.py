"""
支持单卡和 DDP 多卡训练（仿照 nanoGPT 写法）。

单卡训练:
    python train/severstal_finetune.py --batch_size 2
    python -m train.severstal_finetune --batch_size 4 --device_id 0 --num_epoch 2

多卡训练 (指定 GPU):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m train.severstal_finetune --batch_size 2 --num_epochs 2
"""
import os
import monai
import torch
import copy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import SamModel

from data import severstal
from data.severstal import SteelDataset_WithBoxPrompt
from utils.config import get_severstal_ft_args
from utils.helper_function import set_seed, setup_ddp, cleanup_ddp
from utils.finetune_engine import create_model_from_type, run_finetune_engine, inference_engine, _process_batch_severstal, zero_shot
from weights.severstal_wts import severstal_dict

def main():
    # ---------- DDP 初始化 ----------
    ddp_info = setup_ddp()
    ddp = ddp_info['ddp']
    master_process = ddp_info['master_process']

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

    train_dataset = SteelDataset_WithBoxPrompt(train_df, data_path=data_path, transforms=train_transforms, is_train=True)
    val_dataset = SteelDataset_WithBoxPrompt(val_df, data_path=data_path, transforms=val_transforms, is_train=False)
    test_dataset = SteelDataset_WithBoxPrompt(test_df, data_path=data_path, transforms=val_transforms, is_train=False)

    # ---------- DataLoader：DDP 模式下所有 split 都使用 DistributedSampler ----------
    # val/test 用 drop_last=True：DistributedSampler 默认 drop_last=False 时，
    # 末尾不能整除 world_size 的样本会被**重复填充**到其他 rank。
    # 这些重复样本在 _evaluate 的 all_reduce 加权汇总里会被双重计入，
    # 导致 DDP 下 val/test 指标与单卡有偏差，影响早停判断和 best checkpoint 选择。
    # 代价：每 rank 末尾最多丢 (world_size - 1) 个样本，对 ~1000+ 验证集 < 0.3%。
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True) if ddp else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=True) if ddp else None

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
        device = ddp_info['device']
    else:
        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

    if not args.infer_mode and not args.zero_shot:
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)
        # 模型创建完成后切换为 per-rank seed，保证各 rank 数据增强多样性
        set_seed(args.seed, seed_offset=ddp_info['rank'])

        run_finetune_engine(train_dataloader, val_dataloader, test_dataloader,
                            model, device, hyperparameters,
                            process_batch_fn=_process_batch_severstal,
                            save_dir = "./new_weights/finetune/severstal_output/" + hyperparameters['ft_type'],
                            auto_seg=args.auto_seg,
                            ddp_info=ddp_info,
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
                    process_batch_fn=_process_batch_severstal,
                    device=device,
                    results_filename="evaluation_results.txt",
                    auto_seg=False,
                    eval_traindataset=True)

    else:
        if args.include_no_defect:
            checkpoints_to_evaluate = severstal_dict()
        else:
            checkpoints_to_evaluate = None
        scaler = torch.amp.GradScaler(enabled=True)
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        if not ddp:
            device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")
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
                                process_batch_fn=_process_batch_severstal,
                                scaler=scaler,
                                device=device,
                                results_filename="severstal_evaluation_results.txt",
                                auto_seg=False,
                                eval_traindataset=True
                                                        )
            print(f"\n==> [INFERENCE COMPLETE] for: {checkpoint_path}\n\n")



if __name__ == '__main__':
    try:
        main()
    finally:
        # ---------- DDP 清理 ----------
        # 放在 finally 里：训练异常退出（如 worker SIGKILL）时也要 destroy_process_group，
        # 否则 torchrun 会报 "destroy_process_group() was not called before program exit"。
        cleanup_ddp()
