import os
import monai
import torch
import copy
from torch.utils.data import DataLoader
from transformers import SamModel

from data import severstal
from data.severstal import SteelDataset_WithBoxPrompt
from utils.config import get_severstal_ft_args
from utils.helper_function import set_seed
from utils.finetune_engine import create_model_from_type, run_finetune_engine, inference_engine, _process_batch_severstal, zero_shot
from weights.weights_dict_dhs_sever import difRank_sever_dict, scale_sever_dict

if __name__ == '__main__':
    set_seed(42)
    args = get_severstal_ft_args()
    hyperparameters = vars(args)

    mini_dataset = args.mini_dataset            # for debugging purposes
    include_no_defect = args.include_no_defect

    hyperparameters['optimizer'] = "AdamW"
    hyperparameters['scheduler'] = "cosine_scheduler"
    hyperparameters['loss_function'] = "monai.DiceCELoss"
    hyperparameters['output_dir'] = './new_weights/finetune/severstal_output'
    hyperparameters['task_name'] = "severstal_" + hyperparameters['ft_type']
    print(hyperparameters)

    data_path = "./data/severstal_steel_defect_detection"
    # train_df, val_df = data_setup.traindf_preprocess_onlydefect(create_mini_dataset=create_mini_dataset, mini_size=256)
    train_df, val_df, test_df  = severstal.traindf_preprocess(split_seed = 42,
                                                                train_ratio=0.6, 
                                                                val_ratio = 0.2, 
                                                                test_ratio = 0.2,
                                                                include_no_defect=include_no_defect,
                                                                create_mini_dataset=args.mini_dataset, 
                                                                mini_size=256)
    train_transforms, val_transforms = severstal.get_severstal_ft_albumentations_transforms()

    train_dataset = SteelDataset_WithBoxPrompt(train_df, data_path=data_path, transforms=train_transforms)
    val_dataset = SteelDataset_WithBoxPrompt(val_df, data_path=data_path, transforms=val_transforms)
    test_dataset = SteelDataset_WithBoxPrompt(test_df, data_path=data_path, transforms=val_transforms)

    # 小数据集不要使用num_workers避免加负载
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers, persistent_workers=(args.num_workers > 0))    # 不pin memory的话： 1024 张一轮训练1min34s     
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, persistent_workers=(args.num_workers > 0))       # 全部训练一轮大概8min03s
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, persistent_workers=(args.num_workers > 0))       # pin memory的话： 1024 张一轮训练1min05s

    if not args.infer_mode:
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)
        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

        run_finetune_engine(train_dataloader, val_dataloader, test_dataloader, 
                            model, device, hyperparameters, 
                            process_batch_fn=_process_batch_severstal,
                            save_dir = os.path.join(hyperparameters['output_dir'], hyperparameters['ft_type']), auto_seg=args.auto_seg)
    
    elif args.zero_shot:
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
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
            checkpoints_to_evaluate = difRank_sever_dict()
            # checkpoints_to_evaluate = sam_sever_dict()
        else:
            # checkpoints_to_evaluate = only_defect_severstal_dict()
            checkpoints_to_evaluate = None
        scaler = torch.amp.GradScaler(enabled=True) 
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
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
            #infer sam相关权重请不要注释下面四行， 如果 checkpoint_info 中没有 "lora_rank"，则使用 current_args 中已有的值（即默认值）
            current_args.save_custom_lora = checkpoint_info["save_custom_lora"]
            current_args.save_hf_format = checkpoint_info["save_hf_format"]           
            current_args.lora_rank = checkpoint_info.get("lora_rank", current_args.lora_rank)
            current_args.lora_alpha = checkpoint_info.get("lora_alpha", current_args.lora_alpha)

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
                                process_batch_fn=_process_batch_severstal,
                                scaler=scaler,
                                device=device,
                                results_filename="severstal_evaluation_results.txt",
                                auto_seg=False,
                                eval_traindataset=True
                                                        )
            print(f"\n==> [INFERENCE COMPLETE] for: {checkpoint_path}\n\n")

    ##############------------- zero shot-----------------##########
    # results, model = hfsam_zeroshot(model=model,
    #                                 hyperparameters=hyperparameters,
    #                                 train_dataloader=train_dataloader,
    #                                 test_dataloader=val_dataloader,
    #                                 loss_fn=seg_loss,
    #                                 optimizer=optimizer,
    #                                 epochs=NUM_EPOCHS,
    #                                 device=device,
    #                                 patience=PATIENCE, 
    #                                 use_amp=USE_AMP,
    #                                 lora_rank = LORA_RANK)
    ####################------------- zero shot------------####################