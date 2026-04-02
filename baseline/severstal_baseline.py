import copy
import monai
import torch
from pathlib import Path
from torchinfo import summary
from utils.helper_function import get_lr_scheduler

from data import severstal
from utils.config import get_severstal_bsl_args
from utils.helper_function import set_seed
from utils.baseline_engine import baseline_experiment, bsl_inference_engine, create_bsl_model_from_type
from weights.severstal_wts import severstal_dict

if __name__ == '__main__':
    args = get_severstal_bsl_args()
    hyperparameters = vars(args)
    print(hyperparameters)
    set_seed(42)

    batch_size = args.batch_size
    create_mini_dataset = args.mini_dataset
    include_no_defect = args.include_no_defect
    NUM_WORKERS = args.num_workers          

    train_df, val_df, test_df = severstal.traindf_preprocess(split_seed=42, 
                                                             include_no_defect = args.include_no_defect,
                                                             create_mini_dataset=create_mini_dataset, 
                                                             mini_size=256)
    severstal_dataset_path = Path("data/severstal_steel_defect_detection")
    train_transforms, val_transforms = severstal.get_albumentations_transforms()    # albumentations
    # train_transforms, val_transforms = data_setup.get_torchvision_transforms()
    train_dataloader, val_dataloader, test_dataloader = severstal.create_dataloaders_no_prompt(train_df,
                                                                                                val_df,
                                                                                                test_df,
                                                                                                data_path = severstal_dataset_path,
                                                                                                train_transform=train_transforms,
                                                                                                val_transform=val_transforms,
                                                                                                batch_size=batch_size,
                                                                                                num_workers=NUM_WORKERS)
    print(f"train_dataloader长度: {len(train_dataloader)}, val_dataloader长度: {len(val_dataloader)}")

    model_choice = args.bse_model

    hyperparameters['optimizer_name'] = "AdamW"
    hyperparameters['scheduler_name'] = "cosine_scheduler"
    hyperparameters['loss_fn_name'] = "monai.DiceCELoss"
    hyperparameters['task_name']= "severstal_" + args.bse_model
    hyperparameters['output_dir'] = './new_weights'
    if not args.infer_mode:
        model = create_bsl_model_from_type(args=args)

        print('-'*50)
        print("summary model before training")
        summary(model)
        print('-'*50)

        # criterion = torch.nn.BCEWithLogitsLoss()
        criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean') 

        # optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        # 学习率 warmup 策略
        total_steps = len(train_dataloader) * args.num_epochs
        warmup_ratio = 0.1
        warmup_steps = int(warmup_ratio * total_steps)     

        cosine_scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
        
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        baseline_experiment(model = model,
                            device = device,
                            train_loader = train_dataloader, 
                            val_loader = val_dataloader,
                            test_loader = test_dataloader, 
                            criterion = criterion,
                            optimizer=optimizer,
                            scheduler = cosine_scheduler, 
                            hyperparameters = hyperparameters, 
                            save_best_model = args.save_bse_model,
                            scheduler_per_batch = True)
    else:
        # checkpoints_to_evaluate = bsl_sever_dict()
        checkpoints_to_evaluate = severstal_dict()
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
                                train_dataloader = train_dataloader, val_dataloader = val_dataloader, test_dataloader = test_dataloader, 
                                loss_fn = criterion,  device = device, 
                                results_filename="bsl_sever_evaluation_results.txt",
                                eval_traindataset = True)
