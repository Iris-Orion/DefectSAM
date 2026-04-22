"""
SD900 数据集上使用 32×32=1024 个网格点作为 prompt 的 SAM 微调训练脚本

显存估算 (batch_size=4):
- 图像编码器 (ViT-B): ~6-8 GB
- Prompt Encoder (1024点): ~0.5-1 GB
- Mask Decoder: ~0.5 GB
- 模型参数/梯度/优化器: ~1.6 GB
- 激活值缓存: ~4-6 GB
- 总计: ~14-20 GB (RTX 5090 32GB 安全)

使用方法:
    python -m train.sd900_finetune_grid32 \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --ft_type lora_attn_qv \
        --num_epochs 50 \
        --swanlab \
        --pj_name sd900_grid32 \
        --device_id 0
"""

import torch
import monai
import copy
from functools import partial
from transformers import SamModel
from utils.helper_function import set_seed
from utils.config import get_common_ft_args
from utils.finetune_engine import (run_finetune_engine,
                                   inference_engine,
                                   _process_batch_with_point_grid,
                                   zero_shot,
                                   create_model_from_type)
from data.sd900_dataset import (sd900_finetune_create_dataset, 
                                sd900_finetune_create_dataloader)

from weights.sd900_wts import sd900_dict


if __name__ == '__main__':
    args = get_common_ft_args()
    set_seed(args.seed)
    hyperparameters = vars(args)
    hyperparameters['optimizer'] = 'AdamW'
    hyperparameters['loss_function'] = 'monai.DiceCELoss'
    hyperparameters['scheduler'] = 'cosine'
    hyperparameters['task_name'] = "sd900_grid32_" + hyperparameters['ft_type'] + "_" + hyperparameters['sam_type']
    hyperparameters['output_dir'] = './new_weights/sd900_grid32_output'

    print("=" * 60)
    print("SD900 32×32 Grid Points Training")
    print("=" * 60)
    print(f"Hyperparameters: {hyperparameters}")
    print("=" * 60)

    # 创建数据集
    train_dataset, val_dataset, test_dataset = sd900_finetune_create_dataset()
    train_dataloader, val_dataloader, test_dataloader = sd900_finetune_create_dataloader(
        args=args, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        test_dataset=test_dataset
    )
    
    device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

    # 创建 32×32 网格点的 batch 处理函数
    # 1024 个点覆盖整张 512×512 图像，点间距约 16 像素
    process_batch_grid32 = partial(
        _process_batch_with_point_grid,
        points_per_side=32,      # 32×32 = 1024 个网格点
        auto_seg=True,           # 必须启用 auto_seg 才能使用点网格
        offset_info=None,        # SD900 不需要 offset_info
        multimask=False          # 单 mask 输出
    )

    if not args.infer_mode:
        print("\n>>> 启动训练模式 (32×32 Grid Points)")
        print(f">>> 网格点数: 32×32 = 1024")
        print(f">>> Auto Seg: True")
        print(f">>> Batch Size: {args.batch_size}")
        print(f">>> 预估显存: 14-20 GB\n")
        
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)

        results, fintuned_model = run_finetune_engine(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            model=model,
            device=device,
            process_batch_fn=process_batch_grid32,  # 使用 32×32 网格点
            hyperparameters=hyperparameters,
            save_dir="./new_weights/finetune/sd900_grid32_output/" + hyperparameters['ft_type'],
            auto_seg=True  # 关键：必须启用 auto_seg
        )
    
    elif args.zero_shot:
        print("\n>>> 启动 Zero-Shot 评估模式")
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        
        SambPath = "./HuggingfaceModel/sam_vit_base/model"
        MedSamPath = "./HuggingfaceModel/wanglab/medsam-vit-base/model"
        
        zero_shot(
            model_path=SambPath,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=seg_loss,
            process_batch_fn=process_batch_grid32,  # Zero-shot 也用 grid prompt
            device=device,
            results_filename="sd900_grid32_zeroshot_evaluation_results.txt",
            auto_seg=True,  # Zero-shot 使用 auto_seg
            eval_traindataset=True
        )
    
    else:
        print("\n>>> 启动推理模式")
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        # 加载待评估的 checkpoint
        checkpoints_to_evaluate = sd900_dict()

        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")
        
        for checkpoint_info in checkpoints_to_evaluate:
            checkpoint_path = checkpoint_info["path"]
            loading_type = checkpoint_info["type"]

            print(f"\n{'='*60}")
            print(f"==> Path: {checkpoint_path}")
            print(f"==> Loading Type: {loading_type}")
            print(f"{'='*60}")

            current_args = copy.deepcopy(args)
            current_args.ft_type = loading_type
            current_args.lora_rank = checkpoint_info.get("lora_rank", current_args.lora_rank)
            current_args.lora_alpha = checkpoint_info.get("lora_alpha", current_args.lora_alpha)

            current_model = create_model_from_type(args=current_args, train_dataloader=train_dataloader)

            inference_engine(
                model=current_model,
                args=current_args,
                best_model_path=checkpoint_path,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=seg_loss,
                process_batch_fn=process_batch_grid32,  # 推理使用 32×32 网格
                scaler=scaler,
                device=device,
                results_filename="sd900_grid32_eval_results.txt",
                auto_seg=True,  # 推理也使用 auto_seg
                eval_traindataset=True
            )
