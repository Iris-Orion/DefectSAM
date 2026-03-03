#### Baseline的engine

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from utils.utils import compute_dice_score, compute_iou_score

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """
    Train a pytorch model for one epoch
    """
    model.train()

    train_loss, train_dicescore, train_ious = 0, 0, 0

    for batch, (img, ground_truth, _) in enumerate(tqdm(dataloader, desc="Training", leave=False)):    # tqdm(train_dataloader, desc="Training", leave=False)
        # print(f'train step {batch}')
        img, ground_truth = img.to(device), ground_truth.to(device)
        #'img.shape: {BATCHSIZE, C, H, W}, ground_truth.shape: {BATCHSIZE, C, H, W}'

        pred_mask = model(img)

        loss = loss_fn(pred_mask, ground_truth)

        # print(f"img.shape: {img.shape}, ground_truth.shape: {ground_truth.shape}, pred_mask.shape: {pred_mask.shape}")
        # print(f"img.type: {img.type()}, ground_truth.type: {ground_truth.type()}, pred_mask.type: {pred_mask.type()}")
        # print(f"img.unique: {img.unique()}, ground_truth.unique: {ground_truth.unique()}, pred_mask.unique: {pred_mask.unique()}")

        train_loss += loss.item()
        # 如果使用diceloss，这里计算dice_score可以直接这么写
        train_dicescore += compute_dice_score(pred_mask, ground_truth)
        train_ious += compute_iou_score(pred_mask, ground_truth)

        optimizer.zero_grad()
        loss.backward()
        # print(loss.requires_grad)
        optimizer.step()
    
    train_loss = train_loss / len(dataloader)
    train_dicescore = train_dicescore / len(dataloader)
    train_ious = train_ious / len(dataloader)
    return train_loss, train_dicescore, train_ious

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    model.eval()
    test_loss, test_dicescore, test_ious = 0, 0, 0

    with torch.inference_mode():
        for batch, (img, ground_truth, _) in enumerate(dataloader):
            # print(f'test step {batch}')
            img, ground_truth = img.to(device), ground_truth.to(device)

            test_pred_mask = model(img)

            loss = loss_fn(test_pred_mask, ground_truth)
            test_loss += loss.item()

            # 如果使用diceloss，这里计算dice_score可以直接这么写
            test_dicescore += compute_dice_score(test_pred_mask, ground_truth)
            test_ious += compute_iou_score(test_pred_mask, ground_truth)
    
    test_loss = test_loss / len(dataloader)
    test_dicescore = test_dicescore / len(dataloader)
    test_ious= test_ious / len(dataloader)

    return test_loss, test_dicescore, test_ious

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
        #   scheduler: torch.optim.lr_scheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
        #   EARLY_STOPPING_FLAG: bool = True,
          early_stopping_patience: int = 10):
    results = {
        "train_loss": [],
        "train_dicescore": [],
        "train_ious": [],
        "test_loss": [],
        "test_dicescore": [],
        "test_ious": []
    }

    model.to(device)
    # best_test_loss = float('inf')  # 记录最佳loss
    best_test_dicescore = 0         # 记录最佳dice score
    best_model_wts = None          # 最佳的model state dict
    patience_counter = 0           # early stopping 参数， 记录连续几个epoch没有改善
    best_epoch = 0                 # 记录最好的epoch

    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss, train_dicescore, train_ious = train_step(model=model,
                                                 dataloader=train_dataloader,
                                                 loss_fn=loss_fn,
                                                 optimizer=optimizer,
                                                 device=device)
        # 考虑一下如果不早停会怎样
        test_loss, test_dicescore, test_ious = test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)
        # if test_loss < best_test_loss:
        #     best_test_loss = test_loss
        #     best_model_wts = model.state_dict()
        #     patience_counter = 0  # 重置计数器
        #     best_epoch = epoch + 1  # 更新最佳epoch
        if test_dicescore - best_test_dicescore  > 0.0001:
            best_test_dicescore = test_dicescore
            best_model_wts = model.state_dict()
            patience_counter = 0  # 重置计数器
            best_epoch = epoch + 1  # 更新最佳epoch
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1} with no improvement in test loss.")
            break

        # 调整学习率
        # scheduler.step(test_loss)
        # torch.cuda.empty_cache()
        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss: 4f} | "
            f"train_dicescore: {train_dicescore: 4f} | "
            f"train_ious: {train_ious: 4f} | "
            f"test_loss: {test_loss: 4f} | "
            f"test_dicescore: {test_dicescore: 4f} | "
            f"test_ious: {test_ious: 4f} | "
            # f"cur_learningrate: {scheduler.get_last_lr()}"
        )
        results["train_loss"].append(train_loss)
        results["train_dicescore"].append(train_dicescore)
        results["train_ious"].append(train_ious)
        results["test_loss"].append(test_loss)
        results["test_dicescore"].append(test_dicescore)
        results["test_ious"].append(test_ious)

    model.load_state_dict(best_model_wts)
    return results, model, best_epoch

def train_multigpu(model: torch.nn.Module,
                   train_dataloader: torch.utils.data.DataLoader,
                   test_dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: torch.nn.Module,
                   epochs: int,
                   devices: List[int],
                   early_stopping_patience: int = 10):
    """
    Train a PyTorch model using multiple GPUs
    """
    results = {
        "train_loss": [],
        "train_dicescore": [],
        "test_loss": [],
        "test_dicescore": []
    }

    # Set up multi-GPU training
    device = torch.device(f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model, device_ids=devices).to(device)

    best_test_loss = float('inf')  # 记录最佳loss
    best_model_wts = None          # 最佳的model state dict
    patience_counter = 0           # early stopping 参数， 记录连续几个epoch没有改善
    best_epoch = 0                 # 记录最好的epoch

    print(f"Using device: {device}")

    for epoch in tqdm(range(epochs)):
        train_loss, train_dicescore = train_step(model=model,
                                                 dataloader=train_dataloader,
                                                 loss_fn=loss_fn,
                                                 optimizer=optimizer,
                                                 device=device)
        test_loss, test_dicescore = test_step(model=model,
                                              dataloader=test_dataloader,
                                              loss_fn=loss_fn,
                                              device=device)
        # torch.cuda.empty_cache()

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_dicescore: {train_dicescore:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_dicescore: {test_dicescore:.4f} | "
        )

        results["train_loss"].append(train_loss)
        results["train_dicescore"].append(train_dicescore)
        results["test_loss"].append(test_loss)
        results["test_dicescore"].append(test_dicescore)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_wts = model.state_dict()
            patience_counter = 0  # 重置计数器
            best_epoch = epoch + 1  # 更新最佳epoch
        else:
            patience_counter += 1

        # 如果连续patience个epoch没有改进，则停止训练
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1} with no improvement in test loss.")
            break

    model.load_state_dict(best_model_wts)

    return results, model, best_epoch