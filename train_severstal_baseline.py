import data_setup
import modelLib
from engine import train, train_multigpu
from utils import DiceLoss, DiceBCELoss, Combine_DiceCrossEntropy_Loss, save_model
from helper_function import set_device

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import pytz

import torch
from pathlib import Path
import monai
from torchinfo import summary
import segmentation_models_pytorch as smp
import os
import json
from datetime import datetime
import argparse
import numpy as np
import random
from helper_function import set_seed

# ---- reproduction : set flags / seeds ----#
set_seed(42)
shanghai_tz = pytz.timezone('Asia/Shanghai')        # 设置时区为亚洲/上海
start_timestamp = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")     # 获取当前时间戳
# torch.backends.cudnn.benchmark = True
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# random.seed(42)

parser = argparse.ArgumentParser(description="Get some hyperparameters.")         # Create a parser
parser.add_argument("--num_epochs", 
                     default=1, 
                     type=int, 
                     help="the number of epochs to train for")
parser.add_argument("--learning_rate", 
                     default=1e-4, 
                     type=float, 
                     help="config learning rate")
parser.add_argument("--TRAIN_WITH_MUITLGPUS",
                    action='store_true',                            # 设置此参数时值为 True
                    help="whether to train with multiple GPUs")
parser.add_argument("--no-multigpus",
                    action='store_false',                           # 设置此参数时值为 False
                    dest='TRAIN_WITH_MUITLGPUS',                    # 目标变量与 --TRAIN_WITH_MUITLGPUS 共用
                    help="whether to train with a single GPU")
parser.add_argument("--batch_size",
                     default=16, 
                     type=int, 
                     help="batch_size")
args = parser.parse_args()

#####--------------------- 超参数----------------------#####
TRAIN_WITH_MUITLGPUS = args.TRAIN_WITH_MUITLGPUS
SAVE_MODEL_STATE = True        # 是否保存状态
CREATE_MINI_DATASET = False     # 是否创建迷你数据集
USE_DEFECT_AND_NORMAL_IMG = True  # 是否使用有缺陷和无缺陷的所有图片
BEST_EPOCH = 0
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
print(f"TRAIN_WITH_MUITLGPUS: {args.TRAIN_WITH_MUITLGPUS}")


if TRAIN_WITH_MUITLGPUS:
    # BATCH_SIZE = 64     # 单卡16 多卡6    autodl: 24s
    NUM_WORKERS = 4   # 根据服务器的负载来选，最高32  autodl:32  # 3 张 GPU 运行时，num_workers=4 意味着总共会有 3 * 4 = 12 个加载进程
else:
    # BATCH_SIZE = 24     # vanilla_unet: < 16  segformer: 32      单卡16 多卡6    autodl: 24
    NUM_WORKERS = 4   #  4  根据服务器的负载来选，最高32  autodl:32  # 3 张 GPU 运行时，num_workers=4 意味着总共会有 3 * 4 = 12 个加载进程


# panda dataframe
if USE_DEFECT_AND_NORMAL_IMG:
    train_df, val_df = data_setup.traindf_preprocess(split_seed=42, create_mini_dataset=CREATE_MINI_DATASET)
else:
    train_df, val_df = data_setup.traindf_preprocess_onlydefect(split_seed=42, create_mini_dataset=CREATE_MINI_DATASET)

p = Path("data/severstal_steel_defect_detection")


# 数据变换
train_transforms, val_transforms = data_setup.get_albumentations_transforms()    # albumentations
# train_transforms, val_transforms = data_setup.get_torchvision_transforms()

train_dataloader, val_dataloader = data_setup.create_dataloaders(train_df,
                                                                 val_df,
                                                                 data_path = p,
                                                                 train_transform=train_transforms,
                                                                 val_transform=val_transforms,
                                                                 batch_size=BATCH_SIZE,
                                                                 num_workers=NUM_WORKERS)
print(f"train_dataloader长度: {len(train_dataloader)}, val_dataloader长度: {len(val_dataloader)}")

#  -----loss ------#
def get_loss_dict():
    """
    返回一个包含多个不同 Loss 对象的字典，方便管理和调用。
    """
    loss_dict = {
        "dice_calculator": DiceLoss(),
        "bce_logit": torch.nn.BCEWithLogitsLoss(),
        "dice_bce": DiceBCELoss(),
        "seg_monai_dice": monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean'),
        "combine_dice_ce": Combine_DiceCrossEntropy_Loss(),
        "cross_entropy": torch.nn.CrossEntropyLoss(reduction='mean'),
        "bce": torch.nn.BCELoss(),
        "monai_diceCEloss":monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    }
    return loss_dict

def get_loss_fn(loss_name: str = "bce_logit"):
    """
    根据指定的名称返回对应的 Loss 对象。
    默认为 'bce_logit'，即 BCEWithLogitsLoss。
    """
    losses = get_loss_dict()
    if loss_name not in losses:
        raise ValueError(f"Unknown loss name '{loss_name}'. "
                         f"Available names: {list(losses.keys())}")
    return losses[loss_name]


###--- 选择模型 --- ##
# net = modelLib.get_vanilla_unet()
# net = modelLib.get_smp_deeplabv3plus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=4, activation=None) # 不使用imagenet权重
net = modelLib.get_smp_unet(encoder_name = "resnet34", encoder_weights="imagenet", in_channels=3, classes=4) # 不使用imagenet权重
# net = modelLib.build_segformer_model(encoder_name="mit_b0", encoder_weights="imagenet", in_channels=3, classes=4, activation=None)

# create losses (criterion in pytorch)
loss_fn = get_loss_fn("monai_diceCEloss")
print(loss_fn)          # 这是 BCEWithLogitsLoss 对象
scaler = torch.cuda.amp.GradScaler()

# if running on GPU and we want to use cuda move model there
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = set_device(gpu_idx=0)
    net = net.to(device)

print("summary model before training")
summary(net)
print("-----------------------------")

# optimizer
# optimizer = torch.optim.Adam(smp_resnet34_encoder_unet_model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# 定义学习率调度器
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

if TRAIN_WITH_MUITLGPUS:
    # 多卡训练
    devices_idxs = [0, 1, 2, 3]  # 使用第0, 1, 3块GPU
    # 在train_multigpu（）中设置了gpu的idx
    results, model, BEST_EPOCH = train_multigpu(model=net,
                            train_dataloader=train_dataloader,
                            test_dataloader=val_dataloader,
                            optimizer=optimizer,
                            loss_fn= loss_fn,
                            epochs=NUM_EPOCHS, 
                            devices=devices_idxs,
                            early_stopping_patience=10)
else:    
    # 单卡训练
    results, model, BEST_EPOCH = train(model=net,
                    train_dataloader=train_dataloader,
                    test_dataloader=val_dataloader,
                    optimizer = optimizer,
                    # scheduler=scheduler,
                    loss_fn= loss_fn,
                    epochs=NUM_EPOCHS,
                    device=device
                    )


# 假设超参数设置在字典 `hyperparameters` 中，例如：
hyperparameters = {
    "batch_size": BATCH_SIZE,
    "TASK USE_DEFECT_AND_NORMAL_IMG": USE_DEFECT_AND_NORMAL_IMG, 
    "epochs": NUM_EPOCHS,
    "optimizer": "Adamw",
    "learing_rate": LEARNING_RATE,
    "loss function": "monai losses.DiceCELoss",
    "best epoch": BEST_EPOCH,
    "Task name": "全数据集vanilla unet",
    # 添加其他超参数...
}

# 保存模型
save_model(hyperparameters=hyperparameters,
           start_timestamp=start_timestamp,
           results=results,
           model=model,
           optimizer=optimizer,
           scaler=scaler,
           epoch = NUM_EPOCHS,
           model_name = "alldataset_vanilla_unet_",
           result_name = "alldataset_vanilla_unet_results_",
           target_dir= "./model_output/severstal_output",
           SAVE_HUGGINGFACE_PRETRAINED_MODEL= False)

