### 选择不同的lora微调方法，返回SAM模型
import copy
import peft
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from peft import AdaLoraConfig, LoHaConfig, LoKrConfig, LoraConfig, get_peft_model
from utils.sam_arch import (get_loradsc_model,
                            get_loradsc_residual_gated_model,
                            get_loraplus_model)
from transformers import SamModel
from utils.utils import print_trainable_parameters

class FusedQKVSplitLinear(nn.Module):
    """将 SAM 的 fused qkv 线性层拆成 q/k/v 三个子层，但保持输出接口不变。"""

    def __init__(self, qkv_layer: nn.Linear):
        super().__init__()
        if not isinstance(qkv_layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear for qkv_layer, got {type(qkv_layer)!r}.")
        if qkv_layer.out_features % 3 != 0:
            raise ValueError(
                f"Expected qkv out_features divisible by 3, got {qkv_layer.out_features}."
            )

        self.in_features = qkv_layer.in_features
        self.out_features = qkv_layer.out_features
        self.has_bias = qkv_layer.bias is not None
        out_per_part = self.out_features // 3

        self.q_proj = nn.Linear(self.in_features, out_per_part, bias=self.has_bias)
        self.k_proj = nn.Linear(self.in_features, out_per_part, bias=self.has_bias)
        self.v_proj = nn.Linear(self.in_features, out_per_part, bias=self.has_bias)

        with torch.no_grad():
            self.q_proj.weight.copy_(qkv_layer.weight[:out_per_part])
            self.k_proj.weight.copy_(qkv_layer.weight[out_per_part:2 * out_per_part])
            self.v_proj.weight.copy_(qkv_layer.weight[2 * out_per_part:])
            if self.has_bias:
                self.q_proj.bias.copy_(qkv_layer.bias[:out_per_part])
                self.k_proj.bias.copy_(qkv_layer.bias[out_per_part:2 * out_per_part])
                self.v_proj.bias.copy_(qkv_layer.bias[2 * out_per_part:])

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return torch.cat((q, k, v), dim=-1)


def prepare_sam_qkv_for_qv_peft(model, target_part='vision_encoder'):
    """
    将 vision encoder 中 fused qkv 替换为暴露 q_proj/k_proj/v_proj 的等价包装层。
    幂等：若已是 FusedQKVSplitLinear，则跳过。
    """
    if target_part != 'vision_encoder':
        raise ValueError("q/v-only PEFT 目前仅支持 target_part='vision_encoder'.")

    for layer in model.vision_encoder.layers:
        qkv_layer = layer.attn.qkv
        if isinstance(qkv_layer, FusedQKVSplitLinear):
            continue
        layer.attn.qkv = FusedQKVSplitLinear(qkv_layer)
    return model


def get_hf_adalora_model(
                        model,
                        total_step,
                        target_part='vision_encoder',
                        target_r=8,
                        init_r=12,
                        lora_alpha=8):
    target_modules = get_sam_target_modules(model, target_part=target_part)
    config = AdaLoraConfig(
        target_r=target_r,
        init_r=init_r,
        lora_alpha=lora_alpha,
        tinit=int(total_step * 0.1),
        tfinal=int(total_step * 0.8),
        deltaT=10,
        target_modules=target_modules,
        total_step=total_step,
        # modules_to_save=["classifier"],
    )
    model = get_peft_model(model, config)
    return model


def get_hf_lora_model(model, lora_rank=16, lora_alpha=16, lora_dropout=0.0, target_part='vision_encoder'):
    target_modules = get_sam_target_modules(model, target_part=target_part)
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        # modules_to_save=["mask_decoder"]  # 如果想同时训练这个部分，取消注释
    )
    model = get_peft_model(model, config)
    return model


def get_hf_dora_qv_model(model, lora_rank=16, lora_alpha=16, lora_dropout=0.0, target_part='vision_encoder'):
    """
    严格 q/v-only DoRA：先将 fused qkv 拆成 q/k/v 子层，再仅对 q_proj/v_proj 注入 DoRA。
    """
    model = prepare_sam_qkv_for_qv_peft(model, target_part=target_part)
    target_modules = get_sam_qv_target_modules_for_peft(model, target_part=target_part)
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        use_dora=True,
    )
    model = get_peft_model(model, config)
    return model


def get_hf_lokr_qv_model(model, lokr_rank=16, lokr_alpha=16, rank_dropout=0.0, module_dropout=0.0, target_part='vision_encoder'):
    """
    严格 q/v-only LoKr：先将 fused qkv 拆成 q/k/v 子层，再仅对 q_proj/v_proj 注入 LoKr。
    """
    model = prepare_sam_qkv_for_qv_peft(model, target_part=target_part)
    target_modules = get_sam_qv_target_modules_for_peft(model, target_part=target_part)
    config = LoKrConfig(
        r=lokr_rank,
        alpha=lokr_alpha,
        target_modules=target_modules,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
    )
    model = get_peft_model(model, config)
    return model


def get_hf_loha_qv_model(model, loha_rank=16, loha_alpha=16, rank_dropout=0.0, module_dropout=0.0, target_part='vision_encoder'):
    """
    严格 q/v-only LoHa：先将 fused qkv 拆成 q/k/v 子层，再仅对 q_proj/v_proj 注入 LoHa。
    """
    model = prepare_sam_qkv_for_qv_peft(model, target_part=target_part)
    target_modules = get_sam_qv_target_modules_for_peft(model, target_part=target_part)
    config = LoHaConfig(
        r=loha_rank,
        alpha=loha_alpha,
        target_modules=target_modules,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
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


def get_sam_qv_target_modules_for_peft(model, target_part='vision_encoder'):
    """
    仅选择 q_proj / v_proj 作为 strict q/v-only PEFT 目标模块，要求模型已经过 qkv 拆分包装。
    """
    if target_part != 'vision_encoder':
        raise ValueError("q/v-only PEFT 目前仅支持 vision_encoder.")

    target_modules = filter_target_modules(
        model.vision_encoder.named_modules(),
        ['q_proj', 'v_proj'],
    )
    print("\n选择的 strict q/v PEFT 目标模块：")
    for module_name in target_modules:
        print(module_name)

    if not target_modules:
        raise ValueError("未找到 q_proj/v_proj 目标模块，请先执行 prepare_sam_qkv_for_qv_peft().")
    return target_modules

def create_model_from_type(args: argparse.Namespace, train_dataloader: DataLoader = None):
    """
    根据给定的模型类型字符串和参数创建一个模型实例。

    Args:
        model_type (str): 模型的类型标识符 (e.g., 'loradsc_qv', 'lora_encoder').
        args (argparse.Namespace): 包含所有超参数的args对象。
        train_dataloader (DataLoader, optional): 仅在需要计算总步数时(如AdaLora)提供。

    Returns:
        torch.nn.Module: 创建好的模型实例。
    """
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    model_type = args.ft_type
    sam_type = args.sam_type

    print(f"--- Creating model of type: {model_type} with rank: {lora_rank} ---")

    if model_type == 'loradsc_qv':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, 
        ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True, sam_type=sam_type)
    
    elif model_type == 'lora_attn_qv':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, 
                                ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=False)

    elif model_type == 'loradsc_qv_residual_gated':
        return get_loradsc_residual_gated_model(
            rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
            ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True,
            gate_init=0.0,
            sam_type=sam_type)

    elif model_type == 'loradsc_qkv_residual_gated':
        return get_loradsc_residual_gated_model(
            rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
            ft_q=True, ft_k=True, ft_v=True, add_dsc_conv=True,
            gate_init=0.0,
            sam_type=sam_type)

    elif model_type == 'loradsc_q_residual_gated':
        return get_loradsc_residual_gated_model(
            rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
            ft_q=True, ft_k=False, ft_v=False, add_dsc_conv=True,
            gate_init=0.0,
            sam_type=sam_type)

    elif model_type == 'loradsc_qv_adaptive':
        args.use_loraplus_optim = True
        from utils.sam_arch import get_loradsc_adaptive_gated_model
        return get_loradsc_adaptive_gated_model(
            rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
            ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True, sam_type=sam_type)

    elif model_type == 'loradsc_qkv_adaptive':
        args.use_loraplus_optim = True
        from utils.sam_arch import get_loradsc_adaptive_gated_model
        return get_loradsc_adaptive_gated_model(
            rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
            ft_q=True, ft_k=True, ft_v=True, add_dsc_conv=True, sam_type=sam_type)

    elif model_type == 'loradsc_q':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=False, ft_v=False, add_dsc_conv=True)
    
    elif model_type == 'loradsc_qk':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=True, ft_v=False, add_dsc_conv=True)
    
    elif model_type == 'loradsc_qkv':
        return get_loradsc_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout, ft_q=True, ft_k=True, ft_v=True, add_dsc_conv=True)

    elif model_type == 'loraplus_qv':
        args.use_loraplus_optim = True  # 强制启用 LoRA+ 优化器
        return get_loraplus_model(rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout,
                                    ft_q=True, ft_k=False, ft_v=True, sam_type=sam_type)

    elif model_type in ['lora_encoder', 'lora_decoder', 'adalora_encoder', 'lokr_encoder',
                        'dora_qv_encoder', 'lokr_qv_encoder', 'loha_qv_encoder', 'sam_fully', 'sam_decoder']:
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")

        if model_type == 'adalora_encoder':
            if train_dataloader is None:
                raise ValueError("train_dataloader must be provided for 'adalora_encoder' type.")
            ada_target_r = lora_rank
            ada_init_r = max(int(ada_target_r * 1.5), ada_target_r)
            total_step = args.num_epochs * len(train_dataloader)
            return get_hf_adalora_model(
                model=hgsam_model,
                total_step=total_step,
                target_part='vision_encoder',
                target_r=ada_target_r,
                init_r=ada_init_r,
                lora_alpha=lora_alpha,
            )

        elif model_type == 'lora_encoder':
            return get_hf_lora_model(hgsam_model, lora_rank=lora_rank, lora_alpha=lora_alpha,
                                        lora_dropout=lora_dropout, target_part='vision_encoder')

        elif model_type == 'lora_decoder':
            return get_hf_lora_model(hgsam_model, lora_rank=lora_rank, lora_alpha=lora_alpha,
                                        lora_dropout=lora_dropout, target_part='mask_decoder')

        
        elif model_type == 'dora_qv_encoder':
            if getattr(args, 'save_custom_lora', False):
                raise ValueError(
                    "dora_qv_encoder only supports Hugging Face PEFT save/load; "
                    "please disable --save_custom_lora."
                )
            args.save_hf_format = True
            return get_hf_dora_qv_model(
                hgsam_model,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_part='vision_encoder',
            )

        elif model_type == 'lokr_qv_encoder':
            if getattr(args, 'save_custom_lora', False):
                raise ValueError(
                    "lokr_qv_encoder only supports Hugging Face PEFT save/load; "
                    "please disable --save_custom_lora."
                )
            args.save_hf_format = True
            return get_hf_lokr_qv_model(
                hgsam_model,
                lokr_rank=lora_rank,
                lokr_alpha=lora_alpha,
                rank_dropout=0.0,
                module_dropout=0.0,
                target_part='vision_encoder',
            )

        elif model_type == 'loha_qv_encoder':
            if getattr(args, 'save_custom_lora', False):
                raise ValueError(
                    "loha_qv_encoder only supports Hugging Face PEFT save/load; "
                    "please disable --save_custom_lora."
                )
            args.save_hf_format = True
            return get_hf_loha_qv_model(
                hgsam_model,
                loha_rank=lora_rank,
                loha_alpha=lora_alpha,
                rank_dropout=0.0,
                module_dropout=0.0,
                target_part='vision_encoder',
            )

        elif model_type == 'lokr_encoder':
            if getattr(args, 'save_custom_lora', False):
                raise ValueError(
                    "loha_qv_encoder only supports Hugging Face PEFT save/load; "
                    "please disable --save_custom_lora."
                )
            args.save_hf_format = True
            return get_hf_lokr_model(model=hgsam_model)

        elif model_type == 'sam_fully':
            return hgsam_model
        

        elif model_type == 'sam_decoder':
            for name, param in hgsam_model.named_parameters():
                if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                    param.requires_grad_(False)
            return hgsam_model
            
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Please check your configuration.")



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
    # simple_task()
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    encoder_layer_0 = hgsam_model.vision_encoder.layers[0]
    decoder_trans_layer_0 = hgsam_model.mask_decoder.transformer.layers[0]
    print(encoder_layer_0)
    # print(decoder_trans_layer_0)
    print_trainable_parameters(get_hf_lora_model(model=hgsam_model, target_part='vision_encoder'))
    # print_trainable_parameters(get_hf_lora_model(model=hgsam_model, target_part='mask_decoder'))
