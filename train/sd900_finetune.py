"""
支持单卡和 DDP 多卡训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m train.sd900_finetune --batch_size 4 --ft_type loradsc_qkv_residual_gated
"""
import torch
import monai
import copy
from transformers import SamModel
from torch.utils.data import DataLoader, DistributedSampler

from utils.helper_function import set_seed, setup_ddp, cleanup_ddp
from utils.config import get_common_ft_args
from utils.finetune_engine import (run_finetune_engine,
                                   inference_engine,
                                   _process_batch, _process_batch_sam_style,
                                   zero_shot,
                                   create_model_from_type)
from data.sd900_dataset import sd900_finetune_create_dataset

from weights.sd900_wts import sd900_dict


def main():
    ddp_info = setup_ddp()
    ddp = ddp_info['ddp']
    master_process = ddp_info['master_process']

    args = get_common_ft_args()
    # 模型初始化前所有 rank 使用相同 seed，保证各进程初始权重一致（DDP 正确性前提）
    set_seed(args.seed, seed_offset=0)
    
    hyperparameters = vars(args)
    hyperparameters['optimizer'] = 'AdamW'
    hyperparameters['loss_function'] = 'monai.DiceCELoss'
    hyperparameters['scheduler'] = 'cosine'
    hyperparameters['task_name'] = "sd900_" + hyperparameters['ft_type'] + "_" +  hyperparameters['sam_type']

    if master_process:
        print(hyperparameters)

    train_dataset, val_dataset, test_dataset = sd900_finetune_create_dataset()
    
    # ---------- DataLoader：DDP 模式使用 DistributedSampler ----------
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True) if ddp else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=True) if ddp else None
    
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
        device = ddp_info['device']
    else:
        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")


    if not args.infer_mode and not args.zero_shot:
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)
        # 模型创建完成后切换为 per-rank seed，保证各 rank 数据增强多样性
        set_seed(args.seed, seed_offset=ddp_info['rank'])

        # 选择 process_batch 函数：SAM原始多prompt多轮策略 or 单一box prompt
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
                                                        ddp_info=ddp_info,
                                                        train_sampler=train_sampler)
    
    elif args.zero_shot:
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        if not ddp:
            device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

        SambPath = "./HuggingfaceModel/sam_vit_base/model"
        MedSamPath = "./HuggingfaceModel/wanglab/medsam-vit-base/model"

        zero_shot(  model_path = SambPath,
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

        # checkpoints_to_evaluate = sd900_dict()
        # checkpoints_to_evaluate = scale_sd900_dict()
        # checkpoints_to_evaluate = new_sd900_dict()
        checkpoints_to_evaluate = sd900_dict()

        if not ddp:
            device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")
        
        for checkpoint_info in checkpoints_to_evaluate:
            checkpoint_path = checkpoint_info["path"]
            loading_type = checkpoint_info["type"]

            current_model = None # 初始化    

            # description = checkpoint_info["description"]
            if master_process:
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


if __name__ == '__main__':
    try:
        main()
    finally:
        # 放在 finally 里：训练异常退出时也要 destroy_process_group，
        cleanup_ddp()
