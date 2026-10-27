"""
Supports single-GPU and DDP multi-GPU training.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m train.sd900_finetune --batch_size 4 --ft_type loradsc_qkv_residual_gated
"""
import os
import torch
import monai
import copy
from transformers import SamModel
from torch.utils.data import DataLoader, DistributedSampler

from utils.helper_function import set_seed, cleanup_ddp
from utils.config import get_common_ft_args
from utils.finetune_engine import (run_finetune_engine,
                                   inference_engine,
                                   _process_batch, _process_batch_sam_style,
                                   zero_shot,
                                   )
from data.sd900_dataset import sd900_finetune_create_dataset

from weights.sd900_wts import sd900_dict
from utils.loratask import create_model_from_type


def main():
    ddp = int(os.environ.get('RANK', -1)) != -1
    ddp_rank = int(os.environ.get('RANK', 0)) if ddp else 0
    ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0)) if ddp else 0
    ddp_world_size = int(os.environ.get('WORLD_SIZE', 1)) if ddp else 1
    master_process = ddp_rank == 0

    args = get_common_ft_args()
    # Every rank uses the same seed before model init so initial weights match across processes (required for DDP).
    set_seed(args.seed, seed_offset=0)
    
    hyperparameters = vars(args)
    hyperparameters['optimizer'] = 'AdamW'
    hyperparameters['loss_function'] = 'monai.DiceCELoss'
    hyperparameters['scheduler'] = 'cosine'
    hyperparameters['task_name'] = "sd900_" + hyperparameters['ft_type'] + "_" +  hyperparameters['sam_type']

    if master_process:
        print(hyperparameters)

    train_dataset, val_dataset, test_dataset = sd900_finetune_create_dataset()
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, drop_last=True) if ddp else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, drop_last=True) if ddp else None
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=hyperparameters['num_workers'],
        pin_memory=True,
        persistent_workers=(hyperparameters['num_workers'] > 0)
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=hyperparameters['num_workers'],
        pin_memory=True,
        persistent_workers=(hyperparameters['num_workers'] > 0)
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        sampler=test_sampler,
        num_workers=hyperparameters['num_workers'],
        pin_memory=True,
        persistent_workers=(hyperparameters['num_workers'] > 0)
    )
    
    if ddp:
        device = torch.device(f"cuda:{ddp_local_rank}")
    else:
        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")


    if not args.infer_mode and not args.zero_shot:
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)

        # sam_style_train: SAM's original multi-prompt iterative strategy; otherwise a single box prompt.
        if args.sam_style_train:
            batch_fn = _process_batch_sam_style
        else:
            batch_fn = _process_batch

        results, fintuned_model = run_finetune_engine(train_dataloader=train_dataloader,
                                                        val_dataloader = val_dataloader,
                                                        test_dataloader=test_dataloader,
                                                        model=model,
                                                        device=device,
                                                        process_batch_fn = batch_fn,
                                                        hyperparameters=hyperparameters,
                                                        save_dir = "./new_weights/finetune/sd900_output/" + hyperparameters['ft_type'],
                                                        auto_seg = args.auto_seg,
                                                        train_sampler=train_sampler)
    
    elif args.zero_shot:
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        if not ddp:
            device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

        SambPath = "./HuggingfaceModel/sam_vit_base/model"
        MedSamPath = "./HuggingfaceModel/wanglab/medsam-vit-base/model"

        zs_path = SambPath if args.zero_shot_model == 'sam_base' else MedSamPath
        zero_shot(  model_path = zs_path,
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
        scaler = torch.amp.GradScaler('cuda', enabled=True) 
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        checkpoints_to_evaluate = sd900_dict()

        if not ddp:
            device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")
        
        for checkpoint_info in checkpoints_to_evaluate:
            checkpoint_path = checkpoint_info["path"]
            loading_type = checkpoint_info["type"]

            current_model = None

            if master_process:
                print(f"================================================================")
                print(f"==> Path: {checkpoint_path}")
                print(f"==> Loading Type: {loading_type}")
                print(f"================================================================")

            current_args = copy.deepcopy(args)

            current_args.ft_type = loading_type
            current_args.lora_rank = checkpoint_info.get("lora_rank", current_args.lora_rank)
            current_args.lora_alpha = checkpoint_info.get("lora_alpha", current_args.lora_alpha)

            current_model = create_model_from_type(args = current_args, train_dataloader=train_dataloader)

            inference_engine (  model=current_model,
                                args=current_args,
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


if __name__ == '__main__':
    try:
        main()
    finally:
        # In finally so destroy_process_group still runs when training raises.
        cleanup_ddp()
