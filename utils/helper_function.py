import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import wandb

def set_device(gpu_idx: int=0):
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Set device: {device}")
    return device

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

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    # 学习率调度策略 
    # linear_scheduler = get_linear_schedule_with_warmup( optimizer,
    #                                                     num_warmup_steps=warmup_steps,
    #                                                     num_training_steps=total_steps
    #                                                     )
    
    # 初始化多项式学习率调度器
    # polynomial_scheduler = get_polynomial_decay_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps,
    #     lr_end=1e-7,       # 训练结束时,学习率衰减到何值，默认为1e-7
    #     power=0.9
    # )
    pass

def get_bounding_box(ground_truth_map):
    """
    给neu_finetune用的bbox获取函数
    从ground_truth中获得bbox
    输入:ground_truth_map 是一个合并的掩码 : np.array
    """
    # 先转为np格式
    # print(type(ground_truth_map))
    assert type(ground_truth_map) == np.ndarray, "check type"
    if np.sum(ground_truth_map) == 0:
        bbox = [0, 0, 0, 0]
    else:
        # 从非空掩码中得到bounding box
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]
    return bbox

# def get_bounding_box(ground_truth_map):
#     # 先转为np格式
#     # print(type(ground_truth_map))
#     assert type(ground_truth_map) == np.ndarray, "check type"
#     if ground_truth_map.sum() == 0:
#         bbox = [0, 0, 0, 0]
#     else:
#         # 从非空掩码中得到bounding box
#         y_indices, x_indices = np.where(ground_truth_map > 0)
#         x_min, x_max = np.min(x_indices), np.max(x_indices)
#         y_min, y_max = np.min(y_indices), np.max(y_indices)

#         H, W = ground_truth_map.shape
#         x_min = max(0, x_min - np.random.randint(0, 20))
#         x_max = min(W, x_max + np.random.randint(0, 20))
#         y_min = max(0, y_min - np.random.randint(0, 20))
#         y_max = min(H, y_max + np.random.randint(0, 20))
#         bbox = [x_min, y_min, x_max, y_max]
#     return bbox

# 展示图片
def show_elpv_single_image(img):
    """
    input will be tensor, convert to np.array to show
    """
    img = img.permute(1, 2, 0)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def debug_dataset_info(dataset: torch.utils.data.Dataset):
    idx = random.randint(0, len(dataset)-1)
    print(dataset[idx])


if __name__=="__main__":
    a = set_device(gpu_idx = 1)
    print(a)