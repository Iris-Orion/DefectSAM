"""
支持单卡和 DDP 多卡训练（仿照 nanoGPT 写法）。

单卡训练:
    python train/neu_finetune.py --batch_size 2

多卡训练 (单机 2 卡):
    torchrun --standalone --nproc_per_node=2 train/neu_finetune.py --batch_size 2
"""
import torch
import monai
from transformers import SamModel
from torch.utils.data import DataLoader, DistributedSampler

from data.neu_dataset import create_neu_dataset_stratified, debug_neu_dataset_info
from utils.config import get_common_ft_args
from utils.finetune_engine import run_finetune_engine, _process_batch, inference_engine, create_model_from_type, zero_shot
from utils.helper_function import set_seed, setup_ddp, cleanup_ddp

def main():
    ddp_info = setup_ddp()
    ddp = ddp_info['ddp']
    master_process = ddp_info['master_process']

    args = get_common_ft_args()
    # 模型初始化前所有 rank 使用相同 seed，保证各进程初始权重一致（DDP 正确性前提）
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

    # ---------- DataLoader：DDP 模式使用 DistributedSampler ----------
    # val/test 用 drop_last=True：DistributedSampler 默认 drop_last=False 时，
    # 末尾不能整除 world_size 的样本会被重复填充到其他 rank。
    # 这些重复样本在 _evaluate 的 all_reduce 加权汇总里会被双重计入，
    # 导致 DDP 下 val/test 指标偏高，影响早停判断和 best checkpoint 选择。
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True) if ddp else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=True) if ddp else None
    train_dataloader = DataLoader(
                                    train_dataset,
                                    batch_size=hyperparameters['batch_size'],
                                    shuffle=(train_sampler is None),  # DDP 时由 sampler 控制 shuffle，不在 DataLoader 层设置
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
        device = ddp_info['device']
    else:
        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

    if master_process:
        debug_neu_dataset_info(train_dataset, val_dataset, test_dataset)

    if not args.infer_mode and not args.zero_shot:
        # Training and finetune
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)
        # 模型创建完成后切换为 per-rank seed，保证各 rank 数据增强多样性
        set_seed(args.seed, seed_offset=ddp_info['rank'])
        results, fintuned_model = run_finetune_engine(train_dataloader, val_dataloader, test_dataloader,
                                                      model, device, hyperparameters,
                                                      process_batch_fn = _process_batch,
                                                      save_dir = "./new_weights/finetune/neu_seg_output/"+hyperparameters['ft_type'],
                                                      auto_seg = args.auto_seg,
                                                      ddp_info=ddp_info,
                                                      train_sampler=train_sampler)

    elif args.zero_shot:
        # Zero-shot evaluation
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
                    process_batch_fn=_process_batch,
                    device=device,
                    results_filename="infer_results/neu_eval/neu_abla_difRank.txt",
                    auto_seg=False,
                    eval_traindataset=True)



if __name__ == '__main__':
    try:
        main()
    finally:
        # 放在 finally 里：训练异常退出时也要 destroy_process_group，
        cleanup_ddp()
