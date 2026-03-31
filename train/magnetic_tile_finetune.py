import copy
import torch
import monai
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.config import get_common_ft_args
from utils.utils import compute_dice_score, compute_iou_score
from data.magnetic_tile_dataset import create_magnetic_dataset
from utils.helper_function import set_seed
from utils.finetune_engine import run_finetune_engine, _process_batch, inference_engine, zero_shot, create_model_from_type
from weights.magnetic_wts import magnetic_dict

def mag_inference(model, device, dataloader, scaler):
    model.to(device) 
    model.eval()                # 验证
    
    with torch.no_grad():
        test_dicescore, test_ious = 0, 0
        for batch in tqdm(dataloader):
            ground_truth_masks = batch["mask"].float().to(device)      # gt
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values = batch["image"].to(device),
                                    input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
                                    multimask_output=False)
                    predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]

                    test_dicescore += compute_dice_score(predicted_masks, ground_truth_masks)
                    test_ious += compute_iou_score(predicted_masks, ground_truth_masks)   
        test_dicescore = test_dicescore / len(dataloader)
        test_ious = test_ious / len(dataloader)
        # print(f'Validation Loss: {test_loss:.4f}, Val Dice: {test_dicescore:.4f}, Val IoU: {test_ious:.4f}')
    return test_dicescore, test_ious

if __name__ == "__main__":
    args = get_common_ft_args()

    set_seed(42)
    hyperparameters = vars(args)

    train_dataset, val_dataset, test_dataset = create_magnetic_dataset()

    sample = train_dataset[0]
    img, mask, bbox, label = sample['image'], sample['mask'], sample['bbox'], sample['label']

    print(f"img shape: {img.shape}, mask.shape: {mask.shape}")
    print(f"img dtype: {img.dtype}, mask dtype: {mask.dtype}")
    print(f"img min: {img.min()}, img max: {img.max()}")
    print(f"img unique: {img.unique()}")
    print(f"mask min: {mask.min()}, mask max: {mask.max()}")
    print(f"label: {label}")

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))

    hyperparameters['optimizer'] = 'AdamW'
    hyperparameters['loss_function'] = 'monai.DiceCELoss'
    hyperparameters['scheduler'] = 'cosine'
    hyperparameters['task_name'] = "magnetic_" + hyperparameters['ft_type']

    print(hyperparameters)

    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout

    device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

    # 微调任务
    if not args.infer_mode:
        model = create_model_from_type(args=args, train_dataloader=train_dataloader)
        results, fintuned_model = run_finetune_engine(train_dataloader, val_dataloader, test_dataloader, 
                                                      model, device, hyperparameters, 
                                                      process_batch_fn = _process_batch,
                                                      save_dir = "./new_weights/finetune/mag_output/"+hyperparameters['ft_type'],
                                                      auto_seg = args.auto_seg)
    
    elif args.zero_shot:
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

        SambPath = "./HuggingfaceModel/sam_vit_base/model"
        MedSamPath = "./HuggingfaceModel/wanglab/medsam-vit-base/model"
        zero_shot(model_path = SambPath,
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
        # checkpoint_path='/workspace/DefectDetection/new_weights/magnetic_tile_output/loradsc_qv_rank_16_20250721_200224.pth'
        scaler = torch.amp.GradScaler('cuda', enabled=True) 
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        # checkpoints_to_evaluate = magnetic_tile_dict()
        # checkpoints_to_evaluate = scale_magnetic_tile_dict()
        checkpoints_to_evaluate = magnetic_dict()
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
            current_args.save_custom_lora = checkpoint_info["save_custom_lora"]
            current_args.save_hf_format = checkpoint_info["save_hf_format"]
            # 如果 checkpoint_info 中没有 "lora_rank"，则使用 current_args 中已有的值（即默认值）
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
                                process_batch_fn=_process_batch,
                                scaler=scaler,
                                device=device,
                                results_filename="magnetic_eval_results.txt",
                                auto_seg=current_args.auto_seg,
                                eval_traindataset=False
                                                        )
            print(f"\n==> [INFERENCE COMPLETE] for: {checkpoint_path}\n\n")
        
        # inference_engine (  model, args, best_model_path = checkpoint_path, 
        #                     train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader,
        #                     process_batch_fn = _process_batch, loss_fn=loss_fn, scaler=scaler,
        #                     device=device, auto_seg = args.auto_seg, eval_traindataset=True)

    ################ zero-shot####################
    # hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    # medsam_model = SamModel.from_pretrained("./HuggingfaceModel/wanglab/medsam-vit-base/model")

    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(device)
    # print(f"using devcie {device}")

    # seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    # scaler = torch.amp.GradScaler('cuda', enabled=True)

    # train_dice, train_ious = mag_inference(model=hgsam_model,
    #                                        device=device,
    #                                        dataloader=train_dataloader,
    #                                        scaler=scaler,
    #                                        )
    # test_dice, test_ious = mag_inference(model=hgsam_model,
    #                                        device=device,
    #                                        dataloader=test_dataloader,
    #                                        scaler=scaler,
    #                                        )
    # print(f'train_dice: {train_dice:.4f}, train_ious: {train_ious:.4f} || test_dice: {test_dice:.4f}, test_ious: {test_ious:.4f}')
    #################### zero-shot ###########


    # save_model( hyperparameters=hyperparameters, 
    #             results=results, 
    #             model=model, 
    #             optimizer=optimizer,
    #             model_name="magtile_finetune_fully_finetune_",
    #             result_name="magtile_finetune_fully_finetune_",
    #             target_dir= "./model_output/mag_output",
    #             SAVE_HUGGINGFACE_PRETRAINED_MODEL = True)
    
