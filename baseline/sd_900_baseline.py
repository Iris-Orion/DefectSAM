import torch
import logging
import monai
import copy
import torch.optim as optim
from torch.utils.data import  DataLoader
from transformers import get_cosine_schedule_with_warmup

from utils.helper_function import set_seed
from utils.baseline_engine import baseline_experiment, bsl_inference_engine, create_bsl_model_from_type
from utils.config import get_bse_args
from data.data_utils_baseline import sd900_bsl_create_dataset, get_label_distribution, test_sd900_info

from weights.weights_dict_dhs_sd900 import bsl_sd900_dict

if __name__ == '__main__':
    args = get_bse_args()  
    hyperparameters = vars(args)
    set_seed(42)

    train_dataset, val_dataset, test_dataset, fullset_labels = sd900_bsl_create_dataset()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"sd900全数据集的Labels: {fullset_labels}")
    logging.info(f"训练集类别分布: {get_label_distribution(train_dataset)}")
    logging.info(f"验证集类别分布: {get_label_distribution(val_dataset)}")
    logging.info(f"测试集类别分布: {get_label_distribution(test_dataset)}")
    # logging.info(f"训练集: {get_image_names_from_subset(train_dataset)}")
    # logging.info(f"验证集: {get_image_names_from_subset(val_dataset)}")
    # logging.info(f"测试集: {get_image_names_from_subset(test_dataset)}")

    test_sd900_info()

    hyperparameters['optimizer'] = "AdamW"
    hyperparameters['scheduler'] = "cosine_scheduler"
    hyperparameters['loss_function'] = "monai.DiceCELoss"
    hyperparameters['task_name'] = "sd900_" + hyperparameters['bse_model']
    hyperparameters['output_dir'] = "./new_weights/sd900_output"

    if not args.infer_mode:
        model = create_bsl_model_from_type(args=args)

        # 损失函数，优化器，训练循环，device
        # criterion = nn.BCEWithLogitsLoss()
        criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # adam一般不配合weight decay使用
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # 学习率 warmup 策略
        total_steps = len(train_loader) * args.num_epochs
        warmup_ratio = 0.1
        warmup_steps = int(warmup_ratio * total_steps)     

        cosine_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles = 0.5
        )
        
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

        baseline_experiment( model = model, 
                            device = device,
                            train_loader = train_loader,
                            val_loader = val_loader,
                            test_loader = test_loader,
                            criterion = criterion,
                            optimizer = optimizer,
                            scheduler = cosine_scheduler,
                            hyperparameters = hyperparameters,
                            save_best_model = args.save_bse_model
                            )
    
    else:
        checkpoints_to_evaluate = bsl_sd900_dict()
        for checkpoint_info in checkpoints_to_evaluate:

            checkpoint_path = checkpoint_info["path"]
            loading_type = checkpoint_info["type"]

            current_model = None # 初始化    

            print(f"================================================================")
            print(f"==> Path: {checkpoint_path}")
            print(f"==> Loading Type: {loading_type}")
            print(f"================================================================")
        
            current_args = copy.deepcopy(args)          # 创建 args 的一个深拷贝，以防止循环间的副作用
            current_args.bse_model = loading_type

            current_model = create_bsl_model_from_type(args=current_args)
            device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

            criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
            bsl_inference_engine(model = current_model, best_model_path = checkpoint_path, 
                                train_dataloader = train_loader, val_dataloader = val_loader, test_dataloader = test_loader, 
                                loss_fn = criterion,  device = device, 
                                results_filename="bsl_sd900_evaluation_results.txt",
                                eval_traindataset = True)


    # best_model_path = '/workspace/DefectDetection/new_weights/sd900_output/sd900_unet_res34_20250715_035344.pth'
    # bsl_infer(model, best_model_path, train_loader, val_loader, test_loader, criterion, device)
