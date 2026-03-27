import monai
import torch
import copy
from torch.utils.data import DataLoader
from torch import optim
from utils.helper_function import get_lr_scheduler
from utils.helper_function import set_seed
from utils.baseline_engine import baseline_experiment, bsl_inference_engine, create_bsl_model_from_type
from utils.config import get_bse_args
from data.data_utils_baseline import create_mag_dataset_baseline

from weights.weights_dict_dhs_magnetic import baseline_magnetic_tile_dict

if __name__ == "__main__":
    set_seed(42)

    parser = get_bse_args()
    args = parser.parse_args()  
    hyperparameters = vars(args)

    train_dataset, val_dataset, test_dataset = create_mag_dataset_baseline()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    hyperparameters['optimizer'] = "AdamW"
    hyperparameters['scheduler'] = "cosine_scheduler"
    hyperparameters['loss_function'] = "monai.DiceCELoss",
    hyperparameters['task_name']= "magtile_baseline" + hyperparameters['bse_model']
    hyperparameters['output_dir'] = './new_weights/magnetic_tile_output'

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

        cosine_scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
        
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        results, fintuned_model = baseline_experiment(model, 
                                                    device, 
                                                    train_loader,
                                                    val_loader,
                                                    test_loader, 
                                                    criterion, 
                                                    optimizer, 
                                                    scheduler=cosine_scheduler, 
                                                    hyperparameters=hyperparameters, 
                                                    save_best_model=args.save_bse_model)
    else:
        checkpoints_to_evaluate = baseline_magnetic_tile_dict()
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
                                results_filename="bsl_mag_evaluation_results.txt",
                                eval_traindataset = True)
