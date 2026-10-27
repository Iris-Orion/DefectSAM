"""
Supports single-GPU and DDP multi-GPU training.
"""
import os
import torch
import monai
from transformers import SamModel
from torch.utils.data import DataLoader, DistributedSampler

from data.neu_dataset import create_neu_dataset_stratified
from utils.config import get_common_ft_args
from utils.finetune_engine import run_finetune_engine, _process_batch, inference_engine, zero_shot
from utils.helper_function import set_seed, cleanup_ddp
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

    hyperparameters['optimizer'] = "AdamW"
    hyperparameters['scheduler'] = "cosine_scheduler"
    hyperparameters['loss_function'] = "monai.DiceCELoss"
    hyperparameters['output_dir'] = './new_weights/neu_seg_output/finetune'
    hyperparameters['task_name'] = 'neu_' + hyperparameters['ft_type'] + "_" +  hyperparameters['sam_type']

    if master_process:
        print(hyperparameters)

    train_dataset, val_dataset, test_dataset = create_neu_dataset_stratified()

    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, drop_last=True) if ddp else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, drop_last=True) if ddp else None
    train_dataloader = DataLoader(
                                    train_dataset,
                                    batch_size=hyperparameters['batch_size'],
                                    shuffle=(train_sampler is None),  # DataLoader rejects shuffle=True together with a sampler; under DDP the sampler shuffles.
                                    sampler=train_sampler,
                                    num_workers=hyperparameters['num_workers'],
                                    pin_memory=True,
                                    persistent_workers=(hyperparameters['num_workers'] > 0)
                                )
    val_dataloader = DataLoader( val_dataset, 
                                batch_size=hyperparameters['batch_size'], 
                                shuffle=False, 
                                sampler=val_sampler, 
                                num_workers=hyperparameters['num_workers'], 
                                pin_memory=True, 
                                persistent_workers=(hyperparameters['num_workers'] > 0))
    test_dataloader = DataLoader( test_dataset, 
                                 batch_size=hyperparameters['batch_size'], 
                                 shuffle=False, sampler=test_sampler, 
                                 num_workers=hyperparameters['num_workers'], 
                                 pin_memory=True, 
                                 persistent_workers=(hyperparameters['num_workers'] > 0))

    if ddp:
        device = torch.device(f"cuda:{ddp_local_rank}")
    else:
        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

    if not args.infer_mode and not args.zero_shot:
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)
        results, fintuned_model = run_finetune_engine(train_dataloader, val_dataloader, test_dataloader,
                                                      model, device, hyperparameters,
                                                      process_batch_fn = _process_batch,
                                                      save_dir = "./new_weights/finetune/neu_seg_output/"+hyperparameters['ft_type'],
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
                    results_filename="neu_zeroshot_evaluation_results.txt",
                    auto_seg=False,
                    eval_traindataset=True)



if __name__ == '__main__':
    try:
        main()
    finally:
        # In finally so destroy_process_group still runs when training raises.
        cleanup_ddp()
