### 选择不同的lora微调方法，返回SAM模型
import torch
from peft import LoraConfig, get_peft_model, LoHaConfig, LoKrConfig, AdaLoraConfig
import peft
import copy

def get_hf_lora_model(model, target_part='vision_encoder'):
    target_modules = get_sam_target_modules(model, target_part=target_part)
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="lora_only",
        # modules_to_save=["mask_decoder"]
    )
    model = get_peft_model(model, config)
    return model

def get_hf_loha_model(model):
    target_modules = get_sam_target_modules(model)
    config = LoHaConfig(
        r=16,
        alpha=16,
        target_modules= target_modules,
        module_dropout=0.1,
        # modules_to_save=["classifier"],
    )
    model = get_peft_model(model, config)
    return model

def get_hf_lokr_model(model):
    target_modules = get_sam_target_modules(model)
    config = LoKrConfig(
        r=16,
        alpha=16,
        target_modules=target_modules,
        module_dropout=0.1,
        # modules_to_save=["classifier"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def get_hf_adalora_model(model, target_part='vision_encoder'):
    target_modules = get_sam_target_modules(model, target_part=target_part)
    config = AdaLoraConfig(
        r=8,
        init_r=12,
        tinit=200,
        tfinal=1000,    
        deltaT=10,
        target_modules=target_modules,
        # modules_to_save=["classifier"],
    )
    model = get_peft_model(model, config)
    return model

def filter_target_modules(named_modules, target_substrings):
    """
    筛选出名称中包含指定子字符串的线性层模块名称。
    
    Args:
        named_modules (iterator): 模型的命名模块迭代器。
        target_substrings (list of str): 需要匹配的子字符串列表。
        
    Returns:
        list: 匹配的模块名称列表。
    """
    return [
        name
        for name, module in named_modules
        if isinstance(module, torch.nn.Linear) and any(sub in name for sub in target_substrings)
    ]


def get_sam_target_modules(model, target_part='vision_encoder'):
    """
    动态选择指定部分中所有名为 'q_proj', 'k_proj', 'v_proj', 'qkv' 的线性层作为 LoRA 的目标模块。
    """
    target_substrings = ['q_proj', 'k_proj', 'v_proj', 'qkv']
    
    # 根据 target_part 获取对应的子模块
    submodule_map = {
        'vision_encoder': model.vision_encoder,
        'mask_decoder': model.mask_decoder
    }
    
    if target_part not in submodule_map:
        raise ValueError(f"未知的 target_part: {target_part}. 可选项为 {list(submodule_map.keys())}")
    selected_submodule = submodule_map[target_part]

    target_modules = filter_target_modules(selected_submodule.named_modules(), target_substrings)       # 筛选目标模块
    print("\n选择的目标模块：")
    for module_name in target_modules:
        print(module_name)
    
    if not target_modules:
        raise ValueError("未找到任何匹配的目标模块，请检查模型结构和目标模块名称。")                        # 确保至少选择了一个目标模块
    return target_modules

def simple_task():
    from torch import nn
    class MLP(nn.Module):
        def __init__(self, num_units_hidden=2000):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Linear(20, num_units_hidden),
                nn.ReLU(),
                nn.Linear(num_units_hidden, num_units_hidden),
                nn.ReLU(),
                nn.Linear(num_units_hidden, 2),
                nn.LogSoftmax(dim=-1),
            )

        def forward(self, X):
            return self.seq(X)
    print([(n, type(m)) for n, m in MLP().named_modules()])

    X = torch.rand((1000, 20))
    y = (X.sum(1) > 10).long()

    n_train = 800
    batch_size = 64

    train_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X[:n_train], y[:n_train]),
    batch_size=batch_size,
    shuffle=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X[n_train:], y[n_train:]),
        batch_size=batch_size,
    )

    lr = 0.002
    batch_size = 64
    max_epochs = 30
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    def train(model, optimizer, criterion, train_dataloader, eval_dataloader, epochs):
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for xb, yb in train_dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                train_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            eval_loss = 0
            for xb, yb in eval_dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                with torch.no_grad():
                    outputs = model(xb)
                loss = criterion(outputs, yb)
                eval_loss += loss.detach().float()

            eval_loss_total = (eval_loss / len(eval_dataloader)).item()
            train_loss_total = (train_loss / len(train_dataloader)).item()
            print(f"{epoch=:<2}  {train_loss_total=:.4f}  {eval_loss_total=:.4f}")
    module = MLP().to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train(module, optimizer, criterion, train_dataloader, eval_dataloader, epochs=max_epochs)

    ### lora ###
    config = peft.LoraConfig(
        r=8,
        target_modules=["seq.0", "seq.2"],
        modules_to_save=["seq.4"],
    )

    module = MLP().to(device)
    module_copy = copy.deepcopy(module)  # we keep a copy of the original model for later
    peft_model = peft.get_peft_model(module, config)
    optimizer = torch.optim.Adam(peft_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    peft_model.print_trainable_parameters()

    train(peft_model, optimizer, criterion, train_dataloader, eval_dataloader, epochs=max_epochs)

    for name, param in peft_model.base_model.named_parameters():
        if "lora" not in name:
            continue

        print(f"New parameter {name:<13} | {param.numel():>5} parameters | updated")
    params_before = dict(module_copy.named_parameters())

    for name, param in peft_model.base_model.named_parameters():
        if "lora" in name:
            continue

        name_before = (
            name.partition(".")[-1].replace("original_", "").replace("module.", "").replace("modules_to_save.default.", "")
        )
        param_before = params_before[name_before]
        if torch.allclose(param, param_before):
            print(f"Parameter {name_before:<13} | {param.numel():>7} parameters | not updated")
        else:
            print(f"Parameter {name_before:<13} | {param.numel():>7} parameters | updated")

if __name__ == '__main__':
    simple_task()




####---------------------- archive code ---------------------------------####
# def get_sam_target_modules(model, target_part = 'vision_encoder'):
#     # 动态选择 vision encoder中 所有名为 'q_proj', 'k_proj', 'v_proj', 'qkv' 的线性层作为 LoRA 的目标模块
#     target_modules = []
#     if target_part == 'vision_encoder':
#         for name, module in model.vision_encoder.named_modules():
#             if isinstance(module, torch.nn.Linear):
#                 if any(sub in name for sub in ['q_proj', 'k_proj', 'v_proj', 'qkv']):
#                     target_modules.append(name)
#     elif target_part == 'mask_decoder':
#     # # 选择mask decoder中的qkv
#         for name, module in model.mask_decoder.named_modules():
#             if isinstance(module, torch.nn.Linear):
#                 if any(sub in name for sub in ['q_proj', 'k_proj', 'v_proj', 'qkv']):
#                     target_modules.append(name)

#     print("\n选择的目标模块：")
#     for module_name in target_modules:
#         print(module_name)

#     # 确保至少选择了一个目标模块
#     if not target_modules:
#         raise ValueError("未找到任何匹配的目标模块，请检查模型结构和目标模块名称。")
#     return target_modules