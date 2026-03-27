import torch
import monai
import copy
from transformers import SamModel
from utils.helper_function import set_seed
from utils.config import get_common_ft_args
from utils.finetune_engine import (run_finetune_engine,
                                   inference_engine,
                                   _process_batch, _process_batch_sam_style,
                                   zero_shot,
                                   create_model_from_type)
from data.sd900_dataset import (sd900_finetune_create_dataset, 
                                sd900_finetune_create_dataloader)

from weights.weights_dict_dhs_sd900 import sd900_dict, scale_sd900_dict, new_sd900_dict, sam_sd900_dict


if __name__ == '__main__':
    set_seed(42)
    args = get_common_ft_args()
    hyperparameters = vars(args)
    hyperparameters['optimizer'] = 'AdamW'
    hyperparameters['loss_function'] = 'monai.DiceCELoss'
    hyperparameters['scheduler'] = 'cosine'
    hyperparameters['task_name'] = "sd900_" + hyperparameters['ft_type'] + "_" +  hyperparameters['sam_type']

    print(hyperparameters)

    train_dataset, val_dataset, test_dataset = sd900_finetune_create_dataset()
    train_dataloader, val_dataloader, test_dataloader = sd900_finetune_create_dataloader(args=args, 
                                                                                          train_dataset=train_dataset, 
                                                                                          val_dataset=val_dataset, 
                                                                                          test_dataset=test_dataset)
    
    device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")


    if not args.infer_mode:
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)

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
                                                        save_dir = "./new_weights/finetune/sd900_output/"+hyperparameters['ft_type'],
                                                        auto_seg = args.auto_seg)
    
    elif args.zero_shot:
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
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
        checkpoints_to_evaluate = sam_sd900_dict()

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

    