"""
Supports single-GPU and DDP multi-GPU training.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m train.magnetic_tile_finetune --batch_size 4 --ft_type loradsc_qkv_residual_gated
"""
import os
import copy
import torch
import monai
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler

from utils.config import get_common_ft_args
from utils.utils import compute_dice_score, compute_iou_score
from data.magnetic_tile_dataset import create_magnetic_dataset
from utils.helper_function import set_seed, cleanup_ddp
from utils.finetune_engine import run_finetune_engine, _process_batch, inference_engine, zero_shot
from weights.magnetic_wts import magnetic_dict
from utils.loratask import create_model_from_type

def mag_inference(model, device, dataloader, scaler):
    model.to(device) 
    model.eval()
    
    with torch.no_grad():
        test_dicescore, test_ious = 0, 0
        for batch in tqdm(dataloader):
            ground_truth_masks = batch["mask"].float().to(device)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values = batch["image"].to(device),
                                    input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                    multimask_output=False)
                    predicted_masks = outputs.pred_masks.squeeze(1)                     # [B, 1, 256, 256]

                    test_dicescore += compute_dice_score(predicted_masks, ground_truth_masks)
                    test_ious += compute_iou_score(predicted_masks, ground_truth_masks)   
        test_dicescore = test_dicescore / len(dataloader)
        test_ious = test_ious / len(dataloader)
    return test_dicescore, test_ious

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

    train_dataset, val_dataset, test_dataset = create_magnetic_dataset()

    if master_process:
        sample = train_dataset[0]
        img, mask, bbox, label = sample['image'], sample['mask'], sample['bbox'], sample['label']

        print(f"img shape: {img.shape}, mask.shape: {mask.shape}")
        print(f"img dtype: {img.dtype}, mask dtype: {mask.dtype}")
        print(f"img min: {img.min()}, img max: {img.max()}")
        print(f"img unique: {img.unique()}")
        print(f"mask min: {mask.min()}, mask max: {mask.max()}")
        print(f"label: {label}")

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

    hyperparameters['optimizer'] = 'AdamW'
    hyperparameters['loss_function'] = 'monai.DiceCELoss'
    hyperparameters['scheduler'] = 'cosine'
    hyperparameters['task_name'] = "magnetic_" + hyperparameters['ft_type']

    if master_process:
        print(hyperparameters)

    if ddp:
        device = torch.device(f"cuda:{ddp_local_rank}")
    else:
        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

    if not args.infer_mode and not args.zero_shot:
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)
        
        results, fintuned_model = run_finetune_engine(train_dataloader, val_dataloader, test_dataloader, 
                                                      model, device, hyperparameters, 
                                                      process_batch_fn = _process_batch,
                                                      save_dir = "./new_weights/finetune/mag_output/"+hyperparameters['ft_type'],
                                                      auto_seg = args.auto_seg,
                                                      train_sampler=train_sampler)
    
    elif args.zero_shot:
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        if not ddp:
            device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

        SambPath = "./HuggingfaceModel/sam_vit_base/model"
        MedSamPath = "./HuggingfaceModel/wanglab/medsam-vit-base/model"

        zs_path = SambPath if args.zero_shot_model == 'sam_base' else MedSamPath
        zero_shot(model_path = zs_path,
                  train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                    loss_fn=seg_loss,
                    process_batch_fn=_process_batch,
                    device=device,
                    results_filename="mag_zeroshot_evaluation_results.txt",
                    auto_seg=False,
                    eval_traindataset=True)
    
    else:
        scaler = torch.amp.GradScaler('cuda', enabled=True) 
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        checkpoints_to_evaluate = magnetic_dict()
        
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
                                process_batch_fn=_process_batch,
                                scaler=scaler,
                                device=device,
                                results_filename="magnetic_eval_results.txt",
                                auto_seg=current_args.auto_seg,
                                eval_traindataset=False
                                                        )
            if master_process:
                print(f"\n==> [INFERENCE COMPLETE] for: {checkpoint_path}\n\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        # In finally so destroy_process_group still runs when training raises.
        cleanup_ddp()
