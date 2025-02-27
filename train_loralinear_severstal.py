from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import monai
import torch.nn as nn
import random
import numpy as np
import torch
from transformers import SamProcessor, SamModel
from torch.optim import Adam, AdamW
import data_setup
import argparse

from data_setup import SteelDataset_WithBoxPrompt
from utils import save_model
from hf_finetune_engine import hfsam_finetune, print_trainable_parameters, hfsam_zeroshot
from loratask import get_hf_lora_model, get_hf_loha_model, get_hf_lokr_model, get_hf_adalora_model
from lora import LoRALinear, replace_linear_with_lora, replace_linear_with_lora_equi, replace_qkv_with_conv_lora
from sam_arch import get_sam_loraconv_qv_vision_encoder, get_LoRA_DepWiseConv_Samqv_vision_encoder, get_LoRA_Moe_DepwiseConv_Samqv_vison_encoder, loraConv_attnqkv
from peft import LoraConfig, get_peft_model
from new_lora import replace_qkv_with_moelora

####--- seed ---### 
def set_seed(seed: int = 42):
    """
    设置随机种子以确保实验可重复性。

    参数:
        seed (int): 随机种子值。默认为42。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # # 确保在每次操作时使用相同的种子
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_seed(42)
####--- seed ---### 

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='训练模型的args选择')

#### 命令
# 添加参数
parser.add_argument('--batch_size', type=int, default=2, help='批量大小')
parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
parser.add_argument('--best_epoch', type=int, default=0, help='最佳轮数')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
parser.add_argument('--create_mini_dataset', action='store_true', help='是否创建小数据集')
parser.add_argument('--use_amp', action='store_true', help='是否使用自动混合精度')
parser.add_argument('--patience', type=int, default=10, help='早停容忍的epoch数')        # 新增 early stopping 参数
parser.add_argument('--lora_rank', type=int, default=16, help='lora rank')
# 解析参数
args = parser.parse_args()

# ####--------------------------访问超参数--------------------------####
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
BEST_EPOCH = args.best_epoch
LEARNING_RATE = args.learning_rate
create_mini_dataset = args.create_mini_dataset
USE_AMP = args.use_amp
LORA_RANK = args.lora_rank
print(f"USE_AMP: {USE_AMP}")

PATIENCE = args.patience

TASK_TYPE = None
if create_mini_dataset:
    TASK_TYPE = "MINI DATASET"
else:
    TASK_TYPE = "FULL DATASET"
# ####--------------------------访问超参数-------------------------####


##### -----------------   dataset, dataloader ----------------- ####
# train_df, val_df = data_setup.traindf_preprocess_onlydefect(create_mini_dataset=create_mini_dataset, mini_size=256)
train_df, val_df = data_setup.traindf_preprocess(create_mini_dataset=create_mini_dataset, mini_size=256)
data_path = "./data/severstal_steel_defect_detection"
train_transforms = transforms.Compose([
    transforms.ToTensor()
])
val_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = SteelDataset_WithBoxPrompt(train_df, data_path=data_path, transforms=train_transforms)
val_dataset = SteelDataset_WithBoxPrompt(val_df, data_path=data_path, transforms=val_transforms)

# 小数据集不要使用num_workers避免加负载
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)    # 不pin memory的话： 1024 张一轮训练1min34s   全部训练一轮大概8min03s  
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)       # pin memory的话： 1024 张一轮训练1min05s
##### -----------------   dataset, dataloader ----------------- ####

##### ----------------- hyperparameters ----------------- ####
# 保存的超参数
hyperparameters = {
    "task type": TASK_TYPE, 
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "lora_rank": LORA_RANK,
    "optimizer": "AdamW",
    "learing_rate": LEARNING_RATE,
    "loss function": "monai.DiceCELoss",
    "best epoch": BEST_EPOCH,
    "Task name": "包括有缺陷和无缺陷的全数据集进行微调，深度可分离卷积lora, 修正了参数的设置bug， rank=8",
    "USE_AMP": USE_AMP,
    "Mini dataset": create_mini_dataset,
    # 添加其他超参数...
}
##### ----------------- hyperparameters ----------------- ####

#####-------- Lora 微调linear 设置 ------------------------####
# model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
# # # 冻结特定的模型层
# for name, param in model.named_parameters():
    # 冻结 vision_encoder 和 prompt_encoder 的参数
    # if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    #     param.requires_grad_(False)

# # mask_decoder_target_layer_names = []
# # for name, module in model.mask_decoder.named_modules():
# #     if isinstance(module, nn.Linear):
# #         # 获取相对于 mask_decoder 的层名称
# #         mask_decoder_target_layer_names.append(name.split('.')[-1])

# vision_encoder_target_layer_names = []
# for name, module in model.vision_encoder.named_modules():
#     if isinstance(module, nn.Linear):
#         # 获取相对于 mask_decoder 的层名称
#         if any(sub in name for sub in ['q_proj', 'k_proj', 'v_proj', 'qkv']):
#             vision_encoder_target_layer_names.append(name.split('.')[-1])      ### 这样和调用hugging face的参数显示一样了  617472

# # vision_encoder_target_layer_names = []
# # for name, module in model.vision_encoder.named_modules():
# #     if isinstance(module, torch.nn.Linear):
# #         if any(sub in name for sub in ['q_proj', 'k_proj', 'v_proj', 'qkv']):
# #             vision_encoder_target_layer_names.append(name)               

# for name in vision_encoder_target_layer_names:
#     print(name)

# target_keyword = "layers.0.attn.qkv.q."  # 这里请根据实际打印到的名称进行调整

# for name, param in model.named_parameters():
#     # 如果目标层的字符串在完整参数名称中出现，则允许训练
#     if target_keyword in name:
#         print("find q")
#         param.requires_grad = True

# 使用替换函数
# replace_linear_with_lora(model.mask_decoder, target_module_names = mask_decoder_target_layer_names, rank=16, lora_alpha=16, dropout=0.5)
# replace_linear_with_lora(model.vision_encoder, target_module_names = vision_encoder_target_layer_names, rank=16, lora_alpha=16, dropout=0.5)
# replace_linear_with_lora_equi(model.vision_encoder, target_module_names = vision_encoder_target_layer_names, rank=16, lora_alpha=16, dropout=0.5)
# replace_qkv_with_conv_lora(model.vision_encoder, rank=4)
##### ----------------Lora 微调linear Lora 设置 ------------------------####


######-------------------- hugging face 调用 lora qkv设置-----------------#######
hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
medsam_model = SamModel.from_pretrained("./HuggingfaceModel/wanglab/medsam-vit-base/model")
# model = medsam_model
# model = get_hf_lora_model(hgsam_model, target_part='mask_decoder')
# model = get_hf_loha_model(hgsam_model)
# model = get_hf_lokr_model(hgsam_model)
# model = get_hf_adalora_model(hgsam_model, target_part='vision_encoder')

# model = get_sam_lora_conv_qkv_vision_encoder(rank=16, dropout=0.0)   # q, v lora 正常卷积
model = get_LoRA_DepWiseConv_Samqv_vision_encoder(rank=LORA_RANK, dropout=0.05, add_conv=True)
# model = loraConv_attnqkv(rank=16, dropout=0.0)   # q, k, v lora 正常卷积
######----------------- hugging face 调用 lora qkv设置-----------------#######

# model = replace_qkv_with_moelora(model=hgsam_model)

##### ----------------训练 设置 ------------------------####
print_trainable_parameters(model)
print(model.vision_encoder)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
print(f"using devcie {device}")


lora_parameters = [param for param in model.parameters() if param.requires_grad]            # 收集所有可训练的 LoRA 参数
optimizer = AdamW(lora_parameters, lr=LEARNING_RATE, weight_decay=0.01)                     # 定义优化器，只优化 LoRA 参数

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

#########################----------- 微调 -------------########################
results, model, BEST_EPOCH= hfsam_finetune(model=model,
                                           hyperparameters=hyperparameters,
                                           train_dataloader=train_dataloader,
                                           test_dataloader=val_dataloader,
                                           loss_fn=seg_loss,
                                           optimizer=optimizer,
                                           epochs=NUM_EPOCHS,
                                           device=device,
                                           patience=PATIENCE, 
                                           use_amp=USE_AMP,
                                           lora_rank = LORA_RANK)
####################------------ 微调 ---------------------########################

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

##### ----------------训练 设置 ------------------------####

################------------保存模型------------################
# # 保存的超参数
# hyperparameters = {
#     "task type": TASK_TYPE, 
#     "batch_size": BATCH_SIZE,
#     "epochs": NUM_EPOCHS,
#     "optimizer": "AdamW",
#     "learing_rate": LEARNING_RATE,
#     "loss function": "monai.DiceCELoss",
#     "best epoch": BEST_EPOCH,
#     "Task name": "Lora微调vision encoder中所有的q,k,v,qkvproj对应的linear层",
#     "USE_AMP": USE_AMP,
#     "Mini dataset": create_mini_dataset,
#     # 添加其他超参数...
# }

# save_model(hyperparameters=hyperparameters,
#            start_timestamp=start_timestamp, 
#            results=results, 
#            model=model, 
#            optimizer=optimizer,
#            scaler=scaler,
#            model_name="hf_samb_ft_lora_mask_decoder_linear_",
#            result_name="hf_samb_ft_lora_mask_decoder_linear_",
#            target_dir= "./model_output",
#            SAVE_HUGGINGFACE_PRETRAINED_MODEL = True)
###################------------保存模型------------#########################




######----------------------------------------Train debug代码 弃用------------------------------------------------######
# # 训练过程
# model.train()
# for epoch in range(NUM_EPOCHS):
#     epoch_losses = 0.0
#     train_dicescore = 0.0
#     train_ious = 0.0
#     for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
#         #  前向传播
#         outputs = model(pixel_values = batch["image"].to(device),
#                         input_boxes = batch["bbox"].unsqueeze(1).to(device),     # [B, 4] ----unsqueeze----> [B, 1, 4]
#                         multimask_output=False)
        
#         # 计算loss
#         predicted_masks = outputs.pred_masks.squeeze(1)                     # 预测输出  [B, 1, 256, 256]
#         ground_truth_masks = batch["combined_mask"].float().to(device)      # gt
#         ori_res_masks = F.interpolate(
#             predicted_masks,              # [B, 1, 256, 256]  ---interpolate---> [B, 1, 1024, 1024]
#             size=(1024, 1024),  
#             mode="bilinear",
#             align_corners=False
#         )
#         loss = seg_loss(ori_res_masks, ground_truth_masks.unsqueeze(1))

#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()

#         # optimize
#         optimizer.step()
#         epoch_losses += loss.item()
#         train_dicescore += compute_dice_score(ori_res_masks, ground_truth_masks.unsqueeze(1))
#         train_ious += compute_iou_score(ori_res_masks, ground_truth_masks.unsqueeze(1))
        
#     epoch_losses = epoch_losses / len(train_dataloader)
#     train_dicescore = train_dicescore / len(train_dataloader)
#     train_ious = train_ious / len(train_dataloader)
#     print(f"Epoch: {epoch+1} || Epoch loss: {epoch_losses} || Mean dice: {train_dicescore} || Mean iou: {train_ious}")


# 弃用代码
# print("---------")
# for name, param in model.named_parameters():
#     # 冻结 vision_encoder 和 prompt_encoder 的参数
#     print(f"Parameter {name}, requires_grad: {param.requires_grad}")
# print("---------")