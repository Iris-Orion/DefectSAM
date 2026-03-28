import os
import torch
import monai
import copy
from transformers import SamModel
from torch.utils.data import DataLoader

from data.neu_dataset import create_neu_dataset_stratified, debug_neu_dataset_info
from utils.config import get_common_ft_args
from utils.finetune_engine import run_finetune_engine, _process_batch, inference_engine, create_model_from_type, zero_shot
from utils.helper_function import set_seed
from weights.neu_weights import ft_neu_dict, dif_rank_neu_dict, alpha_scale_neu_dict, autoseg_neu_dict

if __name__ == '__main__':
    set_seed(42)
    args = get_common_ft_args()

    hyperparameters = vars(args)

    hyperparameters['optimizer'] = "AdamW"
    hyperparameters['scheduler'] = "cosine_scheduler"
    hyperparameters['loss_function'] = "monai.DiceCELoss"
    hyperparameters['output_dir'] = './new_weights/neu_seg_output/finetune'
    hyperparameters['task_name'] = 'neu_' + hyperparameters['ft_type'] + "_" +  hyperparameters['sam_type']

    print(hyperparameters)
    
    train_dataset, val_dataset, test_dataset = create_neu_dataset_stratified()
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=hyperparameters['num_workers'], pin_memory=True, persistent_workers=(hyperparameters['num_workers'] > 0))
    val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters['batch_size'], shuffle=False, num_workers=hyperparameters['num_workers'], pin_memory=True, persistent_workers=(hyperparameters['num_workers'] > 0))
    test_dataloader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False, num_workers=hyperparameters['num_workers'], pin_memory=True, persistent_workers=(hyperparameters['num_workers'] > 0))

    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout

    # model = get_sam_loraDSC_qv_vision_encoder(rank=lora_rank, lora_alpha = lora_alpha, dropout=lora_dropout, add_dsc_conv=True)              # 最初的稳定实现版本，深度可分离卷积 conv q,v
    # model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")                 # hugging face api SAM
    # model = SamModel.from_pretrained("./HuggingfaceModel/wanglab/medsam-vit-base/model")      # hugging face api MedSAm

    device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")
    debug_neu_dataset_info(train_dataset, val_dataset,test_dataset)

    if not args.infer_mode and not args.zero_shot:
        # Training and finetune
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)
        results, fintuned_model = run_finetune_engine(train_dataloader, val_dataloader, test_dataloader, 
                                                      model, device, hyperparameters, 
                                                      process_batch_fn = _process_batch,
                                                      save_dir = os.path.join(hyperparameters['output_dir'], hyperparameters['ft_type']),
                                                      auto_seg = args.auto_seg)
    
    elif args.zero_shot:
        # Zero-shot evaluation
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
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

    else:
        # infer evaluation
        checkpoints_to_evaluate = dif_rank_neu_dict()
        scaler = torch.amp.GradScaler('cuda', enabled=True) 
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")
        for checkpoint_info in checkpoints_to_evaluate:
            checkpoint_path = checkpoint_info["path"]
            loading_type = checkpoint_info["type"]

            current_model = None # 初始化    

            print(f"================================================================")
            print(f"==> [STARTING INFERENCE]")
            print(f"==> Path: {checkpoint_path}")
            print(f"==> Loading Type: {loading_type}")
            print(f"================================================================")

            current_args = copy.deepcopy(args)          # 创建 args 的一个深拷贝，以防止循环间的副作用

            current_args.ft_type = loading_type
            current_args.save_custom_lora = checkpoint_info["save_custom_lora"]
            current_args.save_hf_format = checkpoint_info["save_hf_format"]
            # 如果 checkpoint_info 中没有 "lora_rank"，则使用 current_args 中已有的值（即默认值）
            current_args.lora_rank = checkpoint_info.get("lora_rank", current_args.lora_rank)
            current_args.lora_alpha = checkpoint_info.get("lora_alpha", current_args.lora_alpha)
            current_args.auto_seg = checkpoint_info.get("auto_seg", current_args.auto_seg)

            current_model = create_model_from_type(args = current_args, train_dataloader=train_dataloader)
            # 4. 使用配置好的 current_args 和对应的路径调用推理函数
            inference_engine(
                                model=current_model,
                                args=current_args,  # 使用为本次循环特地配置的 args
                                best_model_path=checkpoint_path,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                test_dataloader=test_dataloader,
                                loss_fn=seg_loss,
                                process_batch_fn=_process_batch,
                                scaler=scaler,
                                device=device,
                                results_filename="infer_results/neu_eval/neu_dif_rank.txt",
                                auto_seg=current_args.auto_seg,
                                eval_traindataset=True
                                                        )
            print(f"\n==> [INFERENCE COMPLETE] for: {checkpoint_path}\n\n")



    # 推理
    # neu_seg_inference_engine(model, train_dataloader, test_dataloader,  
    #                          checkpoint_path = '/workspace/DefectDetection/new_weights/neu_seg_output/neu_seg_finetue_loradsc_qv_encoder_20250712_044024.pth',
    #                          hyperparameters = hyperparameters, device = device, use_bbox =False, eval_traindataset = False,)