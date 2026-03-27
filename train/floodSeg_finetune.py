import os
import torch
import monai
from torch.utils.data import  DataLoader

from data.floodseg_ft import floodseg_create_dataset
from utils.helper_function import set_seed
from utils.config import get_common_ft_args
from utils.finetune_engine import create_model_from_type, _process_batch_with_point_grid, run_finetune_engine, inference_engine

if __name__ == '__main__':
    args = get_common_ft_args()
    set_seed(42)
    hyperparameters = vars(args)
    hyperparameters['output_dir'] = './pretrained_weights/flood_seg_output'
    print(hyperparameters)

    train_dataset, val_dataset, test_dataset = floodseg_create_dataset()
    batch_size = args.batch_size
    num_workers = args.num_workers

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout

    device = torch.device(f"cuda:{hyperparameters['device_id']}" if torch.cuda.is_available() else "cpu")

    if not args.infer_mode:
        model = create_model_from_type(args=args, train_dataloader=train_loader)
        results, fintuned_model = run_finetune_engine(train_dataloader=train_loader,
                                                        val_dataloader = val_loader,
                                                        test_dataloader=test_loader,
                                                        model=model,
                                                        device=device,
                                                        hyperparameters=hyperparameters,
                                                        process_batch_fn = _process_batch_with_point_grid,
                                                        save_dir = os.path.join(hyperparameters['output_dir'], hyperparameters['ft_type']),
                                                        auto_seg = args.auto_seg)

    else:
        model = create_model_from_type(args=args, train_dataloader=train_loader)
        checkpoint_path='/workspace/DefectDetection/pretrained_weights/sd900_output/loradsc_qv_rank_16_20250720_080218.pth'
        scaler = torch.amp.GradScaler('cuda', enabled=True) 
        loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        inference_engine (  model, args, best_model_path = checkpoint_path, 
                            train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader,
                            process_batch_fn = _process_batch_with_point_grid, loss_fn=loss_fn, scaler=scaler,
                            device=device, auto_seg = args.auto_seg, eval_traindataset=True)