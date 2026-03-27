import torch
import logging
import monai
import copy
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.helper_function import get_lr_scheduler

from utils.helper_function import set_seed
from utils.baseline_engine import baseline_experiment, bsl_inference_engine, create_bsl_model_from_type
from utils.config import get_bse_args
from data.retina_dataset import create_retina_dataset_baseline

if __name__ == '__main__':
    args = get_bse_args()
    hyperparameters = vars(args)
    set_seed(42)

    train_dataset, val_dataset, test_dataset = create_retina_dataset_baseline()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"训练集大小: {len(train_dataset)}")
    logging.info(f"验证集大小: {len(val_dataset)}")
    logging.info(f"测试集大小: {len(test_dataset)}")

    hyperparameters['optimizer'] = "AdamW"
    hyperparameters['scheduler'] = "cosine_scheduler"
    hyperparameters['loss_function'] = "monai.DiceCELoss"
    hyperparameters['task_name'] = "retina_" + hyperparameters['bse_model']
    hyperparameters['output_dir'] = "./new_weights/baseline/retina"

    if not args.infer_mode:
        model = create_bsl_model_from_type(args=args)

        criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        total_steps = len(train_loader) * args.num_epochs
        warmup_ratio = 0.1
        warmup_steps = int(warmup_ratio * total_steps)

        cosine_scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

        baseline_experiment(model=model,
                           device=device,
                           train_loader=train_loader,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=cosine_scheduler,
                           hyperparameters=hyperparameters,
                           save_best_model=args.save_bse_model,
                           scheduler_per_batch=True)
    
    else:
        logging.info("推理模式：加载已训练模型进行评估")
        model = create_bsl_model_from_type(args=args)
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        # 需要指定checkpoint路径
        checkpoint_path = "./new_weights/retina_best_model.pth"
        bsl_inference_engine(model=model,
                           best_model_path=checkpoint_path,
                           train_dataloader=train_loader,
                           val_dataloader=val_loader,
                           test_dataloader=test_loader,
                           loss_fn=criterion,
                           device=device,
                           results_filename="bsl_retina_evaluation_results.txt",
                           eval_traindataset=True)
