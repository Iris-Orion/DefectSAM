import torch
import monai
import copy
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from utils.config import get_bse_args
from utils.helper_function import set_seed
from data.data_utils_baseline import neu_bsl_create_dataset
from utils.baseline_engine import baseline_experiment, bsl_inference_engine, create_bsl_model_from_type
from weights.neu_weights import bsl_neu_dict

if __name__ == "__main__":
    set_seed(42)

    args = get_bse_args()
    hyperparameters = vars(args)

    train_dataset, val_dataset, test_dataset = neu_bsl_create_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)    # 创建 DataLoader
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("="*40)
    print(f"[数据统计] 训练集图片: {len(train_dataset)} 张")
    print(f"[数据统计] 验证集图片: {len(val_dataset)} 张")
    print(f"[数据统计] 测试集图片: {len(test_dataset)} 张")
    print("="*40)

    # unique_vals = set()
    # for i in range(len(train_dataset)):
    #     _, mask, image_id = train_dataset[i]
    #     # 假设 mask 是 torch.Tensor（调用 ToTensorV2 后）
    #     # 如果 mask 是浮点型数据，可以先转为 numpy 数组
    #     mask_np = mask.numpy()
    #     unique_vals.update(np.unique(mask_np).tolist())
    # print("所有mask的unique值：", unique_vals)

    image, mask, _ = train_dataset[0]
    print(f"-"*40 + "Debug Test" + "-"*40)
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"image dtype: {image.dtype} || mask dtype: {mask.dtype}")
    print(f"image data range: {image.min()} ~ {image.max()} || mask data unique: {mask.unique()}")
    print(f"-"*40 + "Debug Test" + "-"*40)
    
    hyperparameters['optimizer'] = "AdamW"
    hyperparameters['scheduler'] = "cosine_scheduler"
    hyperparameters['loss_function'] = "monai.DiceCELoss"
    hyperparameters['task_name'] = 'neu_seg_' + hyperparameters['bse_model']
    hyperparameters['output_dir'] = './pretrained_weights/neu_seg_output' + '/' + hyperparameters['bse_model']

    if not args.infer_mode:
        model = create_bsl_model_from_type(args=args)

        device_id = hyperparameters['device_id']
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

        # 定义损失函数和优化器
        # criterion = nn.BCEWithLogitsLoss()
        criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        # optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learing_rate'])
        optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])

        # 学习率策略
        total_steps = len(train_loader) * args.num_epochs
        warmup_ratio = 0.1
        warmup_steps = int(warmup_ratio * total_steps)
        
        cosine_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles = 0.5
        )

        baseline_experiment(model = model, 
                            device = device,
                            train_loader = train_loader,
                            val_loader = val_loader,
                            test_loader = test_loader,
                            criterion = criterion,
                            optimizer = optimizer,
                            scheduler = cosine_scheduler,
                            hyperparameters = hyperparameters,
                            save_best_model = True)
    else:
        checkpoints_to_evaluate = bsl_neu_dict()
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
                                results_filename="infer_results/neu_eval_bsl.txt",
                                eval_traindataset = True)