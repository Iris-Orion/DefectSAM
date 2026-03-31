import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel
from utils.utils import print_trainable_parameters

class LoRACore(nn.Module):
    def __init__(self, qkv_layer, enabled, rank=16, lora_alpha=16, dropout_rate=0):
        super().__init__()
        self.in_features = qkv_layer.in_features
        self.out_features = qkv_layer.out_features
        self.enabled = enabled
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate

        self.linear = qkv_layer
        if rank > 0:
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False
        self.scale = self.lora_alpha / self.rank if self.rank > 0 else 0.0

        # dropout 只作用于 LoRA 分支的输入: lora_b(lora_a(dropout(x))) * scale
        self.lora_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def init_weights(self):
        """计算 LoRA 调整量（子类重写此方法）"""
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError


class LoRASam_Plus(LoRACore):
    """
    LoRA+ for SAM attention qkv projection.

    说明：
    - 结构上仍是标准 LoRA：ΔW = B @ A；
    - LoRA+ 的核心在优化策略：A/B 使用不同学习率（通常 B 更大）；
    - 本类提供 get_loraplus_param_groups 以便优化器直接使用不同 lr。
    """
    def __init__(self, qkv_layer, enabled, rank=16, lora_alpha=16, dropout_rate=0,
                 ft_q=True, ft_k=False, ft_v=True):
        super().__init__(qkv_layer, enabled, rank, lora_alpha, dropout_rate)
        self.ft_q = ft_q
        self.ft_k = ft_k
        self.ft_v = ft_v

        self.lora_a = nn.ModuleDict({})
        self.lora_b = nn.ModuleDict({})

        parts_to_finetune = []
        if self.ft_q:
            parts_to_finetune.append('q')
        if self.ft_k:
            parts_to_finetune.append('k')
        if self.ft_v:
            parts_to_finetune.append('v')

        for part in parts_to_finetune:
            self.lora_a[part] = nn.Linear(self.in_features, rank, bias=False)
            self.lora_b[part] = nn.Linear(rank, self.out_features // 3, bias=False)

        self.init_weights()

    def init_weights(self):
        if self.rank <= 0:
            return
        for part in self.lora_a.keys():
            nn.init.kaiming_uniform_(self.lora_a[part].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b[part].weight)

    def _get_delta(self, x, part: str):
        return self.lora_b[part](self.lora_a[part](self.lora_dropout(x))) * self.scale

    def get_loraplus_param_groups(self, base_lr: float, lora_plus_lr_ratio: float = 16.0, weight_decay: float = 0.0):
        """
        返回 LoRA+ 推荐的参数组：A 用 base_lr，B 用 base_lr * lora_plus_lr_ratio。
        可直接传给 torch.optim.Optimizer。
        """
        if self.rank <= 0:
            return []

        a_params, b_params = [], []
        for part in self.lora_a.keys():
            a_params.extend(self.lora_a[part].parameters())
            b_params.extend(self.lora_b[part].parameters())

        return [
            {"params": a_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": b_params, "lr": base_lr * lora_plus_lr_ratio, "weight_decay": weight_decay},
        ]

    def forward(self, x):
        if self.rank > 0 and self.enabled:
            qkv = self.linear(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

            if self.ft_q:
                q = q + self._get_delta(x, 'q')
            if self.ft_k:
                k = k + self._get_delta(x, 'k')
            if self.ft_v:
                v = v + self._get_delta(x, 'v')

            return torch.cat((q, k, v), dim=-1)
        return self.linear(x)

class LoRASam_DepWiseConv(LoRACore):
    """
    构建LoRA_a -> dsc -> LoRA_B组件
    对sam的encoder的attention部分选中的q, k, v层的各种组合形式进行lora-dsc修改
    当add_dsc_conv=False时退化为标准LoRA
    """
    def __init__(self, qkv_layer, enabled, rank=16, lora_alpha=16, dropout_rate=0,
                 ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True):
        super().__init__(qkv_layer, enabled, rank, lora_alpha, dropout_rate)
        self.ft_q = ft_q
        self.ft_k = ft_k
        self.ft_v = ft_v
        self.add_dsc_conv = add_dsc_conv

        # 使用 ModuleDict 管理动态创建的层
        self.lora_a = nn.ModuleDict({})
        self.lora_b = nn.ModuleDict({})
        if self.add_dsc_conv:
            self.lora_dw_conv = nn.ModuleDict({})
            self.lora_pw_conv = nn.ModuleDict({})

        # 按需创建参数和层
        parts_to_finetune = []
        if self.ft_q: parts_to_finetune.append('q')
        if self.ft_k: parts_to_finetune.append('k')
        if self.ft_v: parts_to_finetune.append('v')

        for part in parts_to_finetune:
            self.lora_a[part] = nn.Linear(self.in_features, rank, bias=False)
            self.lora_b[part] = nn.Linear(rank, self.out_features // 3, bias=False)
            if self.add_dsc_conv:
                self.lora_dw_conv[part] = nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank, bias=False)
                self.lora_pw_conv[part] = nn.Conv2d(rank, rank, kernel_size=1, padding=0, bias=False)
   
        self.init_weights()

    def init_weights(self):
        for part in self.lora_a.keys():
            nn.init.kaiming_uniform_(self.lora_a[part].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b[part].weight)

        if self.add_dsc_conv:
            for part in self.lora_dw_conv.keys():
                nn.init.kaiming_uniform_(self.lora_dw_conv[part].weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lora_pw_conv[part].weight, a=math.sqrt(5))
    
    def _get_delta(self, x, part: str):
        x_drop = self.lora_dropout(x)
        if self.add_dsc_conv:
            delta = self.lora_a[part](x_drop)               # [B, H, W, rank]
            delta = delta.permute(0, 3, 1, 2)               # 转换为卷积格式: [B, rank, H, W]
            delta = self.lora_dw_conv[part](delta)
            delta = self.lora_pw_conv[part](delta)
            delta = delta.permute(0, 2, 3, 1)               # 转换回原格式:[B, H, W, rank]
            delta = self.lora_b[part](delta) * self.scale
        else: # 标准 LoRA
            delta = self.lora_b[part](self.lora_a[part](x_drop)) * self.scale
        return delta

    def forward(self, x):
        if self.rank > 0 and self.enabled:
            qkv = self.linear(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if self.ft_q:
                q = q + self._get_delta(x, 'q')
            if self.ft_k:
                k = k + self._get_delta(x, 'k')
            if self.ft_v:
                v = v + self._get_delta(x, 'v')

            return torch.cat((q, k, v), dim=-1)
        else:
            return self.linear(x)


class LoRASam_DepWiseConv_Residual(LoRASam_DepWiseConv):
    """
    LoRA-DSC with residual connection around the DSC block:
    delta = lora_b( lora_a(x) + DSC(lora_a(x)) ) * scale

    当 DSC 不提供有用信息时，残差路径保留纯 LoRA 信号；
    DSC 仅需学习空间增量修正，降低优化难度。
    """
    def _get_delta(self, x, part: str):
        x_drop = self.lora_dropout(x)
        if self.add_dsc_conv:
            h = self.lora_a[part](x_drop)                   # [B, H, W, rank]
            conv_in = h.permute(0, 3, 1, 2)                 # [B, rank, H, W]
            conv_out = self.lora_pw_conv[part](self.lora_dw_conv[part](conv_in))
            conv_out = conv_out.permute(0, 2, 3, 1)         # [B, H, W, rank]
            delta = self.lora_b[part](h + conv_out) * self.scale   # 残差: identity + DSC
        else:
            delta = self.lora_b[part](self.lora_a[part](x_drop)) * self.scale
        return delta


class LoRASam_DepWiseConv_Gated(LoRASam_DepWiseConv):
    """
    最小改动版 Gated LoRA-DSC：
    - 保持 LoRA 低秩分解不变；
    - 在每个被微调分支(q/k/v)上引入可学习门控 gamma；
    - 提供 LoRA+ 参数组接口（A 与 B/Conv/Gate 分组不同学习率）。
    """
    def __init__(self, qkv_layer, enabled, rank=16, lora_alpha=16, dropout_rate=0,
                 ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True, gate_init: float = 1e-3):
        super().__init__(
            qkv_layer=qkv_layer,
            enabled=enabled,
            rank=rank,
            lora_alpha=lora_alpha,
            dropout_rate=dropout_rate,
            ft_q=ft_q,
            ft_k=ft_k,
            ft_v=ft_v,
            add_dsc_conv=add_dsc_conv,
        )
        self.gate = nn.ParameterDict({
            part: nn.Parameter(torch.tensor(float(gate_init)))
            for part in self.lora_a.keys()
        })

    def _get_delta(self, x, part: str):
        delta = super()._get_delta(x, part)
        return self.gate[part] * delta

    def get_loraplus_param_groups(self, base_lr: float, lora_plus_lr_ratio: float = 16.0, weight_decay: float = 0.0):
        """
        LoRA+ 参数组：
        - A: base_lr
        - B + (DSC conv + gate): base_lr * ratio
        """
        if self.rank <= 0:
            return []

        a_params, b_side_params = [], []
        for part in self.lora_a.keys():
            a_params.extend(self.lora_a[part].parameters())
            b_side_params.extend(self.lora_b[part].parameters())
            if self.add_dsc_conv:
                b_side_params.extend(self.lora_dw_conv[part].parameters())
                b_side_params.extend(self.lora_pw_conv[part].parameters())
            b_side_params.append(self.gate[part])

        return [
            {"params": a_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": b_side_params, "lr": base_lr * lora_plus_lr_ratio, "weight_decay": weight_decay},
        ]


class LoRA_encoder_attn_Conv(nn.Module):
    """
    对sam的encoder的attention部分进行修改, 将qkv的线性层添加LoRA或者LoRA-standard-Conv
    """
    def __init__(self, qkv_layer, merge, rank=16, lora_alpha=16, dropout_rate=0.0, add_std_conv = False):
        super(LoRA_encoder_attn_Conv, self).__init__()
        self.in_features = qkv_layer.in_features
        self.out_features = qkv_layer.out_features
        self.merge = merge
        self.rank = rank
        self.dropout_rate = dropout_rate
        self.lora_alpha = lora_alpha
        self.add_conv = add_std_conv

        self.linear = qkv_layer
        self.linear.bias = qkv_layer.bias

        if rank > 0:
            self.linear.weight.requires_grad = False  # 冻结主线性层权重
            self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
            self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features))
            self.scale = self.lora_alpha / self.rank

            if self.add_conv:
                self.lora_conv = nn.Conv2d(rank, rank, kernel_size=3, padding=1, bias=False)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = nn.Identity()
        self.initial_weights()
    
    def initial_weights(self):
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_b, a=math.sqrt(5))
            nn.init.zeros_(self.lora_a)
            if self.add_conv:
                nn.init.kaiming_uniform_(self.lora_conv.weight, a=math.sqrt(5))
    
    def forward(self, x):
        qkv = self.linear(x)
        if self.rank > 0 and self.merge:
            delta_qkv = x @ self.lora_a.T
            
            # 在LoRA的A矩阵之后应用Dropout
            # delta_qkv = self.dropout(delta_qkv)

            if self.add_conv:
                # print(f"被调用了:{delta_qkv.shape}")    [2, 64, 64, 16]
                delta_qkv = delta_qkv.permute(0, 3, 1, 2)
                delta_qkv = self.lora_conv(delta_qkv)
                delta_qkv = delta_qkv.permute(0, 2, 3, 1)
            
            delta_qkv = (delta_qkv @ self.lora_b.T) * self.scale
            qkv = qkv + delta_qkv 
            return self.dropout(qkv)
        else:
            return qkv

class LoRA_SAM_qvConv(nn.Module):
    """
    对sam的encoder的attention部分的q,v层分开进行修改, 将qkv的线性层添加LoRA或者LoRA-standard-Conv
    """
    def __init__(self, qkv_layer, merge, rank=16, lora_alpha=16, dropout=0.0, add_std_conv = False):
        super(LoRA_SAM_qvConv, self).__init__()
        self.in_features = qkv_layer.in_features
        self.out_features = qkv_layer.out_features
        self.merge = merge
        self.rank = rank
        self.dropout_rate = dropout
        self.lora_alpha = lora_alpha
        self.add_conv = add_std_conv

        self.linear = qkv_layer
        self.linear.bias = qkv_layer.bias
        if rank > 0:
            self.linear.weight.requires_grad = False  # 冻结主线性层权重

            self.lora_a_q  = nn.Parameter(torch.zeros(rank, self.in_features))
            self.lora_b_q = nn.Parameter(torch.zeros(self.out_features // 3, rank))

            self.lora_a_v = nn.Parameter(torch.zeros(rank, self.in_features))
            self.lora_b_v = nn.Parameter(torch.zeros(self.out_features // 3, rank))

            self.scale = self.lora_alpha / self.rank

            if self.add_conv:
                self.lora_conv_q = nn.Conv2d(rank, rank, kernel_size=3, padding=1, bias=False)
                self.lora_conv_v = nn.Conv2d(rank, rank, kernel_size=3, padding=1, bias=False)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()
        self.initial_weights()

    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_a_q, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q)
        nn.init.kaiming_uniform_(self.lora_a_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v)

        if hasattr(self, 'lora_conv_q'):
            nn.init.kaiming_uniform_(self.lora_conv_q.weight, a=math.sqrt(5))
        if hasattr(self, 'lora_conv_v'):
            nn.init.kaiming_uniform_(self.lora_conv_v.weight, a=math.sqrt(5))

    def forward(self, x):
        # print(f"x.shape: {x.shape}")s
        if self.rank > 0 and self.merge:
            qkv = self.linear(x)
            print(f"qkv shape: {qkv.shape}")
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            print(f"q shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
            if self.add_conv:
                delta_q = (x @ self.lora_a_q.T)
                delta_q = delta_q.permute(0, 3, 1, 2)
                delta_q = self.lora_conv_q(delta_q)
                delta_q = delta_q.permute(0, 2, 3, 1)
                delta_q = (delta_q @ self.lora_b_q.T) * self.scale

                delta_v = (x @ self.lora_a_v.T)
                delta_v = delta_v.permute(0, 3, 1, 2)
                delta_v = self.lora_conv_v(delta_v)
                delta_v = delta_v.permute(0, 2, 3, 1)
                delta_v = (delta_v @ self.lora_b_v.T) * self.scale
            else:
                # Compute LoRA adjustments dynamically based on input x
                delta_q = (x @ self.lora_a_q.T) @ self.lora_b_q.T * self.scale  # Shape: [50, 14, 14, 768]
                delta_v = (x @ self.lora_a_v.T) @ self.lora_b_v.T * self.scale  # Shape: [50, 14, 14, 768]
            
            # Add the LoRA adjustments to q and v
            q = q + delta_q
            v = v + delta_v

            qkv_adjusted = torch.cat((q, k, v), dim=-1)
            output = self.dropout(qkv_adjusted)
            print(f"被调用了: {output.shape}")
            print("=" * 20)
            return output
        else:
            return self.dropout(self.linear(x))

class LoRASam_qv_DepWiseConv(nn.Module):
    """
    对sam的encoder的attention部分的q,v层进行lora-dsc修改
    """
    def __init__(self, qkv_layer, merge, rank=16, lora_alpha=16, dropout=0, add_dsc_conv = True):
        # dropout rate: 0.05?
        super(LoRASam_qv_DepWiseConv, self).__init__()
        self.in_features = qkv_layer.in_features
        self.out_features = qkv_layer.out_features
        self.merge = merge
        self.rank = rank
        self.dropout_rate = dropout
        self.lora_alpha = lora_alpha
        self.add_dsc_conv = add_dsc_conv

        self.linear = qkv_layer
        self.linear.bias = qkv_layer.bias
        if rank > 0:
            self.linear.weight.requires_grad = False  # 冻结主线性层权重

            self.lora_a_q  = nn.Parameter(torch.zeros(rank, self.in_features))
            self.lora_b_q = nn.Parameter(torch.zeros(self.out_features // 3, rank))

            self.lora_a_v = nn.Parameter(torch.zeros(rank, self.in_features))
            self.lora_b_v = nn.Parameter(torch.zeros(self.out_features // 3, rank))

            self.scale = self.lora_alpha / self.rank

            if self.add_dsc_conv:
                self.lora_dw_conv_q = nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank, bias=False)
                self.lora_pw_conv_q = nn.Conv2d(rank, rank, kernel_size=1, padding=0, bias=False)

                self.lora_dw_conv_v = nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank, bias=False)
                self.lora_pw_conv_v = nn.Conv2d(rank, rank, kernel_size=1, padding=0, bias=False)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()
        self.initial_weights()

    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_a_q, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q)
        nn.init.kaiming_uniform_(self.lora_a_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v)

        if hasattr(self, 'lora_dw_conv_q'):
            nn.init.kaiming_uniform_(self.lora_dw_conv_q.weight, a=math.sqrt(5))
        if hasattr(self, 'lora_pw_conv_q'):
            nn.init.kaiming_uniform_(self.lora_pw_conv_q.weight, a=math.sqrt(5))
        if hasattr(self, 'lora_dw_conv_v'):
            nn.init.kaiming_uniform_(self.lora_dw_conv_v.weight, a=math.sqrt(5))
        if hasattr(self, 'lora_pw_conv_v'):
            nn.init.kaiming_uniform_(self.lora_pw_conv_v.weight, a=math.sqrt(5))

    def forward(self, x):
        print(f"x.shape: {x.shape}")
        if self.rank > 0 and self.merge:
            qkv = self.linear(x)
            # print("=" * 30)
            # print(f"qkv shape: {qkv.shape}")
            # print("=" * 30)
            q, k, v = torch.chunk(qkv, 3, dim=-1)    # [25, 14, 14, 768]
            # print("=" * 30)
            # print(f"q shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
            # print("=" * 30)
            if self.add_dsc_conv:
                delta_q = (x @ self.lora_a_q.T)
                delta_q = delta_q.permute(0, 3, 1, 2)
                delta_q = self.lora_dw_conv_q(delta_q)
                delta_q = self.lora_pw_conv_q(delta_q)
                delta_q = delta_q.permute(0, 2, 3, 1)
                delta_q = (delta_q @ self.lora_b_q.T) * self.scale

                delta_v = (x @ self.lora_a_v.T)
                delta_v = delta_v.permute(0, 3, 1, 2)
                delta_v = self.lora_dw_conv_v(delta_v)
                delta_v = self.lora_pw_conv_v(delta_v)
                delta_v = delta_v.permute(0, 2, 3, 1)
                delta_v = (delta_v @ self.lora_b_v.T) * self.scale
            else:
                # Compute LoRA adjustments dynamically based on input x
                delta_q = (x @ self.lora_a_q.T) @ self.lora_b_q.T * self.scale  # Shape: [50, 14, 14, 768]
                delta_v = (x @ self.lora_a_v.T) @ self.lora_b_v.T * self.scale  # Shape: [50, 14, 14, 768]
            
            # Add the LoRA adjustments to q and v
            q = q + delta_q
            v = v + delta_v

            qkv_adjusted = torch.cat((q, k, v), dim=-1)
            output = self.dropout(qkv_adjusted)
            print(f"被调用了: {output.shape}")
            print("=" * 20)
            return output
        else:
            return self.dropout(self.linear(x))

class LoRA_Moe_DepwiseConv_Samqv(nn.Module):
    def __init__(self, qkv_layer, merge, rank=16, lora_alpha=16, dropout=0.05, num_experts=3, kernel_sizes=[3, 5, 7]):
        super(LoRA_Moe_DepwiseConv_Samqv, self).__init__()
        self.in_features = qkv_layer.in_features
        self.out_features = qkv_layer.out_features
        self.merge = merge
        self.rank = rank
        self.dropout_rate = dropout
        self.lora_alpha = lora_alpha
        self.num_experts = num_experts
        self.kernel_sizes = kernel_sizes

        self.linear = qkv_layer
        self.linear.bias = qkv_layer.bias
        if rank > 0:
            self.linear.weight.requires_grad = False  # 冻结主线性层权重

            # 定义 Q 的 LoRA 参数
            self.lora_a_q = nn.Parameter(torch.zeros(rank, self.in_features))
            self.lora_b_q = nn.Parameter(torch.zeros(self.out_features // 3, rank))

            # 定义 V 的 LoRA 参数
            self.lora_a_v = nn.Parameter(torch.zeros(rank, self.in_features))
            self.lora_b_v = nn.Parameter(torch.zeros(self.out_features // 3, rank))

            self.scale = self.lora_alpha / self.rank  

            # 定义 Q 和 V 的多个深度可分离卷积专家
            self.q_experts = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(rank, rank, kernel_size=ks, padding=ks//2, groups=rank, bias=False),
                    nn.Conv2d(rank, rank, kernel_size=1, padding=0, bias=False)
                ) for ks in self.kernel_sizes
            ])

            self.v_experts = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(rank, rank, kernel_size=ks, padding=ks//2, groups=rank, bias=False),
                    nn.Conv2d(rank, rank, kernel_size=1, padding=0, bias=False)
                ) for ks in self.kernel_sizes
            ])

            # 定义门控网络
            self.gate_q = nn.Linear(rank, self.num_experts)
            self.gate_v = nn.Linear(rank, self.num_experts)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()
        self.initial_weights()

    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_a_q, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q)
        nn.init.kaiming_uniform_(self.lora_a_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v)

        for conv in self.q_experts:
            for layer in conv:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

        for conv in self.v_experts:
            for layer in conv:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.gate_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.gate_v.weight, a=math.sqrt(5))
        if self.gate_q.bias is not None:
            nn.init.zeros_(self.gate_q.bias)
        if self.gate_v.bias is not None:
            nn.init.zeros_(self.gate_v.bias)

    def forward(self, x):
        # print(f"x.shape: {x.shape}")  # e.g., [batch, height, width, in_features]
        if self.rank > 0 and self.merge:
            qkv = self.linear(x)
            # print(f"qkv shape: {qkv.shape}")  # [batch, height, width, out_features]
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            # print(f"q shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")  # [batch, height, width, out_features//3]

            # Q 的 LoRA 调整
            delta_q = (x @ self.lora_a_q.T)  # [batch, height, width, rank]
            delta_q = delta_q.permute(0, 3, 1, 2)  # [batch, rank, height, width]

            # 通过多个专家进行卷积，并使用门控权重加权
            expert_outputs_q = [expert(delta_q) for expert in self.q_experts]  # List of [batch, rank, height, width]
            expert_outputs_q = torch.stack(expert_outputs_q, dim=1)  # [batch, num_experts, rank, height, width]

            # 计算门控权重
            gate_input_q = delta_q.mean(dim=[2,3])  # [batch, rank]
            gate_probs_q = torch.softmax(self.gate_q(gate_input_q), dim=-1)  # [batch, num_experts]
            self._last_gate_probs_q = gate_probs_q.detach()  # 缓存用于诊断
            gate_weights_q = gate_probs_q.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [batch, num_experts, 1, 1, 1]

            # 加权求和
            weighted_experts_q = (expert_outputs_q * gate_weights_q).sum(dim=1)  # [batch, rank, height, width]

            # 转换回原始形状
            weighted_experts_q = weighted_experts_q.permute(0, 2, 3, 1)  # [batch, height, width, rank]
            delta_q = (weighted_experts_q @ self.lora_b_q.T) * self.scale  # [batch, height, width, out_features//3]

            # V 的 LoRA 调整
            delta_v = (x @ self.lora_a_v.T)  # [batch, height, width, rank]
            delta_v = delta_v.permute(0, 3, 1, 2)  # [batch, rank, height, width]

            # 通过多个专家进行卷积，并使用门控权重加权
            expert_outputs_v = [expert(delta_v) for expert in self.v_experts]  # List of [batch, rank, height, width]
            expert_outputs_v = torch.stack(expert_outputs_v, dim=1)  # [batch, num_experts, rank, height, width]

            # 计算门控权重
            gate_input_v = delta_v.mean(dim=[2,3])  # [batch, rank]
            gate_probs_v = torch.softmax(self.gate_v(gate_input_v), dim=-1)  # [batch, num_experts]
            self._last_gate_probs_v = gate_probs_v.detach()  # 缓存用于诊断
            gate_weights_v = gate_probs_v.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [batch, num_experts, 1, 1, 1]

            # 加权求和
            weighted_experts_v = (expert_outputs_v * gate_weights_v).sum(dim=1)  # [batch, rank, height, width]

            # 转换回原始形状
            weighted_experts_v = weighted_experts_v.permute(0, 2, 3, 1)  # [batch, height, width, rank]
            delta_v = (weighted_experts_v @ self.lora_b_v.T) * self.scale  # [batch, height, width, out_features//3]

            # 将 LoRA 调整添加到 q 和 v
            q = q + delta_q
            v = v + delta_v

            # 拼接调整后的 q, k, v
            qkv_adjusted = torch.cat((q, k, v), dim=-1)  # [batch, height, width, out_features]
            output = self.dropout(qkv_adjusted)
            # print(f"Adjusted qkv shape: {output.shape}")  # [batch, height, width, out_features]
            return output
        else:
            return self.dropout(self.linear(x))

def get_sam_hg_model():
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    # for name, param in hgsam_model.named_parameters():
    # # 冻结某些层
    #     if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    #         param.requires_grad_(False)
    return hgsam_model

def get_loradsc_model(rank, lora_alpha, dropout_rate, ft_q, ft_k, ft_v, add_dsc_conv, sam_type: str = "sam_base"):
    """
    lora-dsc-选中q,k,v的选定组合进行微调,不选中bias, 不选中output的proj层
    """
    if sam_type == "sam_base":
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    elif sam_type == "sam_large":
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_large")

    for name, param in hgsam_model.named_parameters():
    # 冻结某些层
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)
    for layers in hgsam_model.vision_encoder.layers:
        # lora_alpha 默认应该等于 rank
        layers.attn.qkv = LoRASam_DepWiseConv(qkv_layer = layers.attn.qkv, enabled = True,
                                              rank=rank, lora_alpha=lora_alpha, dropout_rate=dropout_rate,
                                              ft_q = ft_q, ft_k = ft_k, ft_v = ft_v,
                                              add_dsc_conv=add_dsc_conv)
    return hgsam_model

def get_loradsc_global_only_model(rank, lora_alpha, dropout_rate, ft_q, ft_k, ft_v, sam_type: str = "sam_base"):
    """
    仅在 global attention block 上注入 DSC，window attention block 使用纯 LoRA。
    SAM ViT-B global_attn_indexes = [2, 5, 8, 11]
    SAM ViT-L global_attn_indexes = [5, 11, 17, 23]
    """
    if sam_type == "sam_base":
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
        global_attn_indexes = {2, 5, 8, 11}
    elif sam_type == "sam_large":
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_large")
        global_attn_indexes = {5, 11, 17, 23}

    for name, param in hgsam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)

    for i, layers in enumerate(hgsam_model.vision_encoder.layers):
        use_dsc = i in global_attn_indexes
        layers.attn.qkv = LoRASam_DepWiseConv(
            qkv_layer=layers.attn.qkv, enabled=True,
            rank=rank, lora_alpha=lora_alpha, dropout_rate=dropout_rate,
            ft_q=ft_q, ft_k=ft_k, ft_v=ft_v,
            add_dsc_conv=use_dsc)

    dsc_count = len(global_attn_indexes)
    total = len(list(hgsam_model.vision_encoder.layers))
    print(f"--- DSC injected in {dsc_count}/{total} layers (global attention only) ---")
    return hgsam_model


def get_loradsc_gated_model(rank, lora_alpha, dropout_rate, ft_q, ft_k, ft_v, add_dsc_conv,
                            gate_init: float = 1e-3, sam_type: str = "sam_base"):
    """
    Gated LoRA-DSC 版本 SAM。
    """
    if sam_type == "sam_base":
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    elif sam_type == "sam_large":
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_large")
    else:
        raise ValueError(f"Unknown sam_type: {sam_type}")

    for name, param in hgsam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)

    for layers in hgsam_model.vision_encoder.layers:
        layers.attn.qkv = LoRASam_DepWiseConv_Gated(
            qkv_layer=layers.attn.qkv,
            enabled=True,
            rank=rank,
            lora_alpha=lora_alpha,
            dropout_rate=dropout_rate,
            ft_q=ft_q,
            ft_k=ft_k,
            ft_v=ft_v,
            add_dsc_conv=add_dsc_conv,
            gate_init=gate_init,
        )
    return hgsam_model

def get_loraplus_model(rank, lora_alpha, dropout_rate, ft_q=True, ft_k=False, ft_v=True, sam_type: str = "sam_base"):
    """
    构建 LoRA+ 版本 SAM。

    说明：LoRA+ 的核心是优化阶段对 A/B 使用不同学习率。
    模型注入后，请在优化器中调用 LoRASam_Plus.get_loraplus_param_groups。
    """
    if sam_type == "sam_base":
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    elif sam_type == "sam_large":
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_large")
    else:
        raise ValueError(f"Unknown sam_type: {sam_type}")

    for name, param in hgsam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)

    for layers in hgsam_model.vision_encoder.layers:
        layers.attn.qkv = LoRASam_Plus(
            qkv_layer=layers.attn.qkv,
            enabled=True,
            rank=rank,
            lora_alpha=lora_alpha,
            dropout_rate=dropout_rate,
            ft_q=ft_q,
            ft_k=ft_k,
            ft_v=ft_v,
        )

    return hgsam_model

def get_sam_loraconv_qv_vision_encoder(rank=16, lora_alpha=16, dropout=0.0, add_std_conv=False):
    """
    对选中的qv层进行lora分解, 根据add_conv参数决定是否加入标准卷积
    """
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    for name, param in hgsam_model.named_parameters():
    # 冻结某些层
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)
    for layers in hgsam_model.vision_encoder.layers:
        layers.attn.qkv = LoRA_SAM_qvConv(qkv_layer = layers.attn.qkv, merge = True, rank=rank, lora_alpha=lora_alpha, dropout=dropout, add_dsc_conv=add_std_conv)
    return hgsam_model

def get_sam_loraDSC_qv_vision_encoder(rank=16, lora_alpha=16, dropout=0.0, add_dsc_conv=True):
    """
    lora qv 使用深度可分离卷积
    """
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    for name, param in hgsam_model.named_parameters():
    # 冻结某些层
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)
    for layers in hgsam_model.vision_encoder.layers:
        # lora_alpha 默认应该等于 rank
        layers.attn.qkv = LoRASam_qv_DepWiseConv(qkv_layer = layers.attn.qkv, merge = True, rank=rank, lora_alpha=lora_alpha, dropout=dropout, add_dsc_conv=add_dsc_conv)
    return hgsam_model

def loraConv_attnqkv(lora_rank=16, lora_alpha=16, dropout_rate=0.0, add_std_conv=False):
    """
    lora qkv
    """
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    for name, param in hgsam_model.named_parameters():
    # 冻结某些层
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)
    for layers in hgsam_model.vision_encoder.layers:
        layers.attn.qkv = LoRA_encoder_attn_Conv(qkv_layer = layers.attn.qkv, merge = True, rank=lora_rank, lora_alpha=lora_alpha, dropout_rate=dropout_rate, add_std_conv=add_std_conv)
    return hgsam_model




# def get_LoRA_Moe_DepwiseConv_Samqv_vison_encoder(rank=16, dropout=0.05):
#     """
#     lora moe机制合并的深度可分离卷积 qv
#     """
#     hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
#     for name, param in hgsam_model.named_parameters():
#     # 冻结某些层
#         if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
#             param.requires_grad_(False)
#     for layers in hgsam_model.vision_encoder.layers:
#         layers.attn.qkv = LoRA_Moe_DepwiseConv_Samqv(qkv_layer = layers.attn.qkv, merge = True, rank=rank, lora_alpha=rank, dropout=dropout, num_experts=3, kernel_sizes=[3, 5, 7])
#     return hgsam_model

def get_moelora_model(rank, lora_alpha, dropout_rate, num_experts=3, kernel_sizes=[3, 5, 7], sam_type="sam_base"):
    """MoE-LoRA: 多专家DSC卷积 + 门控路由"""
    if sam_type == "sam_base":
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    elif sam_type == "sam_large":
        hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_large")
    else:
        raise ValueError(f"Unknown sam_type: {sam_type}")

    for name, param in hgsam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)
    for layers in hgsam_model.vision_encoder.layers:
        layers.attn.qkv = LoRA_Moe_DepwiseConv_Samqv(
            qkv_layer=layers.attn.qkv, merge=True,
            rank=rank, lora_alpha=lora_alpha, dropout=dropout_rate,
            num_experts=num_experts, kernel_sizes=kernel_sizes)
    return hgsam_model


def apply_lora_ga_init(model, train_dataloader, device, rank, lora_alpha):
    """
    LoRA-GA: 对已注入 LoRA 的模型，用首批数据的梯度 SVD 重新初始化 A/B，
    并修正基底权重以维持零阶近似。
    逐层串行计算梯度以避免 OOM。
    """
    import monai
    model.to(device)
    model.eval()  # 避免 dropout 干扰

    loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    batch = next(iter(train_dataloader))
    images = batch["image"].to(device)
    gt_masks = batch["mask"].unsqueeze(1).float().to(device)
    bboxes = batch["bbox"].unsqueeze(1).to(device)

    scale = lora_alpha / rank
    out_per_part = model.vision_encoder.layers[0].attn.qkv.linear.out_features // 3

    # 逐层串行：每次只解冻一个 qkv 层，计算梯度，SVD 初始化，再冻结
    for layer_idx, layer in enumerate(model.vision_encoder.layers):
        qkv_mod = layer.attn.qkv
        # 临时解冻当前层的 qkv 基底权重
        qkv_mod.linear.weight.requires_grad_(True)

        # 前向 + 反向（只用 1 张图，节省显存）
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(pixel_values=images[:1], input_boxes=bboxes[:1], multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            gt_down = F.interpolate(gt_masks[:1], size=(256, 256), mode="nearest")
            loss = loss_fn(predicted_masks, gt_down)
        loss.backward()

        G = qkv_mod.linear.weight.grad  # (3*out, in)
        if G is not None:
            part_slices = {'q': slice(0, out_per_part), 'v': slice(2 * out_per_part, 3 * out_per_part)}
            for part, sl in part_slices.items():
                if part not in qkv_mod.lora_a:
                    continue
                G_part = G[sl, :].float()  # 确保 SVD 在 fp32

                U, S, Vh = torch.linalg.svd(G_part, full_matrices=False)
                A0 = U[:, :rank].contiguous()
                B0 = Vh[rank:2 * rank, :].contiguous()

                with torch.no_grad():
                    qkv_mod.lora_a[part].weight.copy_(B0)
                    qkv_mod.lora_b[part].weight.copy_(A0)
                    qkv_mod.linear.weight.data[sl, :] -= scale * (A0 @ B0)

        qkv_mod.linear.weight.requires_grad_(False)
        model.zero_grad()
        torch.cuda.empty_cache()

    print(f"[LoRA-GA] SVD initialization complete for {len(model.vision_encoder.layers)} layers.")
    return model


def get_loraga_model(rank, lora_alpha, dropout_rate, train_dataloader, device, sam_type="sam_base"):
    """LoRA-GA: 标准LoRA结构 + SVD梯度初始化"""
    model = get_loradsc_model(rank=rank, lora_alpha=lora_alpha, dropout_rate=dropout_rate,
                              ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=False, sam_type=sam_type)
    model = apply_lora_ga_init(model, train_dataloader, device, rank, lora_alpha)
    return model


class LoRAProGradHook:
    """
    LoRA-Pro: 每步训练后替换 LoRA 梯度为 Sylvester 方程的最优解。
    通过 forward hook 保存输入，backward hook 获取输出梯度，
    在 optimizer.step() 前调用 replace_gradients() 修正梯度。
    """
    def __init__(self, model, rank, lora_alpha):
        self.model = model
        self.rank = rank
        self.scale = lora_alpha / rank
        self._hooks = []
        self._saved = {}  # layer_idx -> {'input': ..., 'grad_output': ...}

        for idx, layer in enumerate(model.vision_encoder.layers):
            qkv_mod = layer.attn.qkv
            # Forward hook: 保存输入
            h = qkv_mod.register_forward_hook(self._make_fwd_hook(idx))
            self._hooks.append(h)
            # Backward hook: 保存输出梯度
            h = qkv_mod.register_full_backward_hook(self._make_bwd_hook(idx))
            self._hooks.append(h)

    def _make_fwd_hook(self, idx):
        def hook(module, input, output):
            self._saved.setdefault(idx, {})['input'] = input[0].detach()
        return hook

    def _make_bwd_hook(self, idx):
        def hook(module, grad_input, grad_output):
            self._saved.setdefault(idx, {})['grad_output'] = grad_output[0].detach()
        return hook

    def replace_gradients(self):
        """在 loss.backward() 之后、optimizer.step() 之前调用"""
        for idx, layer in enumerate(self.model.vision_encoder.layers):
            qkv_mod = layer.attn.qkv
            saved = self._saved.get(idx)
            if saved is None or 'input' not in saved or 'grad_output' not in saved:
                continue

            X = saved['input']          # [B, H, W, in_features]
            dY = saved['grad_output']   # [B, H, W, 3*out_per_part]

            out_per_part = qkv_mod.linear.out_features // 3
            # 分拆 dY 为 q/k/v 梯度
            dY_parts = {'q': dY[..., :out_per_part], 'v': dY[..., 2*out_per_part:]}

            for part in ['q', 'v']:
                if part not in qkv_mod.lora_a:
                    continue

                A = qkv_mod.lora_b[part].weight.data  # (out_per_part, rank)
                B = qkv_mod.lora_a[part].weight.data  # (rank, in_features)

                # 重构全量梯度: G_t = dY_part^T @ X  但直接实例化太大
                # 改用投影: 只需计算 H_A 和 H_B
                dY_p = dY_parts[part]  # [B, H, W, out_per_part]

                # Reshape for matmul: (N, out_per_part) and (N, in_features)
                N = X.shape[0] * X.shape[1] * X.shape[2]
                X_flat = X.reshape(N, -1).float()        # (N, in_features)
                dY_flat = dY_p.reshape(N, -1).float()    # (N, out_per_part)

                # G_t = dY_flat^T @ X_flat  → (out_per_part, in_features)
                # 但不要实例化完整 G_t，用分块投影:
                # H_B_raw = A^T @ G_t = A^T @ (dY^T @ X) = (dY @ A)^T @ X
                AtdY = dY_flat @ A           # (N, rank)
                H_B_raw = AtdY.T @ X_flat    # (rank, in_features)

                # H_A_raw = G_t @ B^T = (dY^T @ X) @ B^T = dY^T @ (X @ B^T)
                XBt = X_flat @ B.T           # (N, rank)
                H_A_raw = dY_flat.T @ XBt    # (out_per_part, rank)

                # Sylvester 修正: 从 H_A_raw 中减去 A 空间的投影，分配给 H_B
                # P_A = A @ inv(A^T @ A) @ A^T
                AtA = A.T @ A + 1e-6 * torch.eye(self.rank, device=A.device)  # 正则化
                AtA_inv = torch.linalg.inv(AtA)
                proj_A = A @ AtA_inv @ A.T  # (out_per_part, out_per_part) - 太大!

                # 改用高效计算:
                # proj_component = A @ inv(A^T@A) @ A^T @ H_A_raw = A @ (inv(A^T@A) @ (A^T @ H_A_raw))
                proj_coeff = AtA_inv @ (A.T @ H_A_raw)  # (rank, rank)
                H_A_proj = A @ proj_coeff               # (out_per_part, rank)

                H_A = H_A_raw - H_A_proj  # 正交补: 去掉 A 列空间中的分量

                # H_B: 将 H_A 解掉后的残差分配给 B
                # residual G_t contribution not captured by H_A: G_res = G_t - H_A @ B
                # H_B = inv(A^T A) @ A^T @ G_res = inv(A^T A) @ (A^T @ G_t - A^T @ H_A @ B)
                # = H_B_raw/scale - inv(A^T A) @ A^T @ H_A @ B  (since H_B_raw = A^T @ G_t implicitly)
                # 简化: 直接用伪逆 B 空间
                BBt = B @ B.T + 1e-6 * torch.eye(self.rank, device=B.device)
                BBt_inv = torch.linalg.inv(BBt)
                H_B = BBt_inv @ H_B_raw

                # 写入梯度 (除以 scale 因为 delta = lora_b(lora_a(x)) * scale)
                if qkv_mod.lora_a[part].weight.grad is not None:
                    qkv_mod.lora_a[part].weight.grad.copy_(H_B * self.scale)
                if qkv_mod.lora_b[part].weight.grad is not None:
                    qkv_mod.lora_b[part].weight.grad.copy_(H_A * self.scale)

        self._saved.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def get_lorapro_model(rank, lora_alpha, dropout_rate, train_dataloader, device, sam_type="sam_base"):
    """LoRA-Pro: LoRA-GA 初始化 + Sylvester 梯度修正（返回模型和 hook 对象）"""
    model = get_loraga_model(rank, lora_alpha, dropout_rate, train_dataloader, device, sam_type)
    pro_hook = LoRAProGradHook(model, rank, lora_alpha)
    return model, pro_hook


def measure_inference_time(model, input_tensor, num_runs=10):
    """
    测量模型的推理时间。

    :param model: 要测试的模型
    :param input_tensor: 输入张量
    :param num_runs: 运行次数
    :return: 平均推理时间
    """
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            # _ = model.vision_encoder(input_tensor)
            _ = model(input_tensor)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def create_model_for_inference(model, lora_weights_path: str, device: str = "cpu") -> SamModel:
    """
    一步到位创建用于推理的、加载了LoRA权重的SAM模型。
    Args:
        lora_weights_path (str): 保存的LoRA权重文件 (.pth) 的路径。
        device (str): 模型加载到的设备 ('cpu' or 'cuda')。
    Returns:
        SamModel: 一个准备好进行推理的完整模型。
    """
    print("--- 开始构建推理模型 ---")
    # custom_model_types = ['loradsc_qv', 'loradsc_qkv', 'loradsc_qk', ]
    # if args.ft_type in custom_model_types:
    # 步骤 2: 加载微调后的LoRA权重
    print(f"步骤 2/2: 从 '{lora_weights_path}' 加载微调后的LoRA权重...")
    try:
        # 加载只包含LoRA参数的state_dict
        lora_state_dict = torch.load(lora_weights_path, map_location=torch.device(device))
        lora_state_dict = {k.replace('_orig_mod.', ''): v for k, v in lora_state_dict.items()}

        # 使用 strict=False 将LoRA权重加载到模型中
        # 这会填充 lora_a_q, lora_b_q 等参数，同时忽略文件中不存在的基础模型参数
        missing_keys, unexpected_keys = model.load_state_dict(lora_state_dict, strict=False)
        
        print("LoRA权重加载成功！")
        if unexpected_keys:
             print(f"  - 警告: 权重文件中有多余的键: {unexpected_keys}")
        # `missing_keys` 会列出所有基础模型的参数，这是正常现象
        # print(f"  - {len(missing_keys)} 个基础模型参数被正确忽略。")

    except FileNotFoundError:
        print(f"  - 错误: 找不到LoRA权重文件: {lora_weights_path}。模型将使用随机初始化的LoRA权重。")
    except Exception as e:
        print(f"  - 错误: 加载LoRA权重时发生错误: {e}")

    # 将模型移动到指定设备并设置为评估模式
    model.to(device)
    model.eval()
    
    print("--- 推理模型构建完成，已设为评估模式 ---")
    return model


if __name__ == "__main__":
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")

    layer_0 = hgsam_model.vision_encoder.layers[0]
    print(layer_0)

    linear = hgsam_model.vision_encoder.layers[0].attn.qkv
    print(linear.in_features)
    print(linear.out_features)
    print('----before---')
    print(hgsam_model.vision_encoder)
    # x = torch.randn(2, 196, 768)
    # output = linear(x)
    # print(output[:, :, 768])
    # print(linear.bias)

    print("---after---")
    # model = get_loradsc_model(rank=16, lora_alpha=16, dropout_rate=0.0, 
    #                           ft_q = True, ft_k = False , ft_v = True, add_dsc_conv=False)
    # model = get_sam_loraDSC_qv_vision_encoder(rank=16, lora_alpha=16, dropout=0.0, add_dsc_conv=True)

    # model = loraConv_attnqkv(lora_rank=16, lora_alpha=16, dropout_rate=0.0, add_std_conv=False)

    model = get_loradsc_gated_model(rank=16, lora_alpha=16, dropout_rate=0.0,
                                       ft_q=True, ft_k=False, ft_v=True, add_dsc_conv=True,
                                       gate_init=1e-3, sam_type="sam_base")

    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            print(name, param.shape)
    

    # print(model.vision_encoder)
    print_trainable_parameters(model)

    img = torch.rand((1, 3, 1024, 1024))
    vs_encoder_output = model.vision_encoder(img)
