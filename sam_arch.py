from transformers import SamModel
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from hf_finetune_engine import print_trainable_parameters

class LoRA_encoder_attn_Conv(nn.Module):
    """
    对sam的encoder的attention部分进行修改, 将qkv的线性层添加LoRA或者LoRAConv
    """
    def __init__(self, qkv_layer, merge, rank=16, lora_alpha=16, dropout=0.05, add_conv = True):
        super(LoRA_encoder_attn_Conv, self).__init__()
        self.in_features = qkv_layer.in_features
        self.out_features = qkv_layer.out_features
        self.merge = merge
        self.rank = rank
        self.dropout_rate = dropout
        self.lora_alpha = lora_alpha
        self.add_conv = add_conv

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
            self.dropout = nn.Dropout(self.dropout_rate)
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

class LoRASam_qvConv(nn.Module):
    def __init__(self, qkv_layer, merge, rank=16, lora_alpha=16, dropout=0.05, add_conv = True):
        super(LoRASam_qvConv, self).__init__()
        self.in_features = qkv_layer.in_features
        self.out_features = qkv_layer.out_features
        self.merge = merge
        self.rank = rank
        self.dropout_rate = dropout
        self.lora_alpha = lora_alpha
        self.add_conv = add_conv

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
            # print(f"qkv shape: {qkv.shape}")
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            # print(f"q shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
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
            # print(f"被调用了: {output.shape}")
            return output
        else:
            return self.dropout(self.linear(x))

class LoRASam_qv_DepWiseConv(nn.Module):
    def __init__(self, qkv_layer, merge, rank=16, lora_alpha=16, dropout=0.05, add_conv = True):
        super(LoRASam_qv_DepWiseConv, self).__init__()
        self.in_features = qkv_layer.in_features
        self.out_features = qkv_layer.out_features
        self.merge = merge
        self.rank = rank
        self.dropout_rate = dropout
        self.lora_alpha = lora_alpha
        self.add_conv = add_conv

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

        if hasattr(self, 'lora_conv_q'):
            nn.init.kaiming_uniform_(self.lora_conv_q.weight, a=math.sqrt(5))
        if hasattr(self, 'lora_conv_v'):
            nn.init.kaiming_uniform_(self.lora_conv_v.weight, a=math.sqrt(5))

    def forward(self, x):
        print(f"x.shape: {x.shape}")
        if self.rank > 0 and self.merge:
            qkv = self.linear(x)
            print(f"qkv shape: {qkv.shape}")
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            print(f"q shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
            if self.add_conv:
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
            gate_weights_q = self.gate_q(gate_input_q)  # [batch, num_experts]
            gate_weights_q = torch.softmax(gate_weights_q, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [batch, num_experts, 1, 1, 1]

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
            gate_weights_v = self.gate_v(gate_input_v)  # [batch, num_experts]
            gate_weights_v = torch.softmax(gate_weights_v, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [batch, num_experts, 1, 1, 1]

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

def get_sam_loraconv_qv_vision_encoder(rank=16, dropout=0.05):
    """
    lora qv 纯 conv
    """
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    for name, param in hgsam_model.named_parameters():
    # 冻结某些层
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)
    for layers in hgsam_model.vision_encoder.layers:
        layers.attn.qkv = LoRASam_qvConv(qkv_layer = layers.attn.qkv, merge = True, rank=rank, lora_alpha=rank, dropout=dropout, add_conv=True)
    return hgsam_model

def get_sam_lora_qv_encoder(rank=16, dropout=0.05):
    """
    自己实现的LoRASam_qvConv 使用add_conv = False变成了纯LoRA qv 
    """
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    for name, param in hgsam_model.named_parameters():
    # 冻结某些层
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)
    for layers in hgsam_model.vision_encoder.layers:
        layers.attn.qkv = LoRASam_qvConv(qkv_layer = layers.attn.qkv, merge = True, rank=rank, lora_alpha=rank, dropout=dropout, add_conv=False)
    return hgsam_model

def get_LoRA_DepWiseConv_Samqv_vision_encoder(rank=16, dropout=0.05, add_conv=True):
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
        layers.attn.qkv = LoRASam_qv_DepWiseConv(qkv_layer = layers.attn.qkv, merge = True, rank=rank, lora_alpha=rank, dropout=dropout, add_conv=add_conv)
    return hgsam_model

def get_LoRA_Moe_DepwiseConv_Samqv_vison_encoder(rank=16, dropout=0.05):
    """
    lora moe机制合并的深度可分离卷积 qv
    """
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    for name, param in hgsam_model.named_parameters():
    # 冻结某些层
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)
    for layers in hgsam_model.vision_encoder.layers:
        layers.attn.qkv = LoRA_Moe_DepwiseConv_Samqv(qkv_layer = layers.attn.qkv, merge = True, rank=rank, lora_alpha=rank, dropout=dropout, num_experts=3, kernel_sizes=[3, 5, 7])
    return hgsam_model

def loraConv_attnqkv(rank=16, dropout=0.05):
    """
    lora qkv conv
    """
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
    for name, param in hgsam_model.named_parameters():
    # 冻结某些层
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
            param.requires_grad_(False)
    for layers in hgsam_model.vision_encoder.layers:
        layers.attn.qkv = LoRA_encoder_attn_Conv(qkv_layer = layers.attn.qkv, merge = True, rank=rank, lora_alpha=rank, dropout=dropout, add_conv=True)
    return hgsam_model


if __name__ == "__main__":
    hgsam_model = SamModel.from_pretrained("./HuggingfaceModel/sam_vit_base/model")
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
    # model = hgsam_model
    # model = get_sam_lora_conv_qkv_vision_encoder()
    model = get_LoRA_DepWiseConv_Samqv_vision_encoder()
    # model = get_LoRA_Moe_DepwiseConv_Samqv_vison_encoder(rank=4)
    # model = loraConv_attnqkv()

    
    print(model.vision_encoder)
    print_trainable_parameters(model)

    img = torch.rand((1, 3, 1024, 1024))
    vs_encoder_output = model.vision_encoder(img)
    # print(vs_encoder_output)

    # x = torch.rand((1, 3, 1024, 1024))

    # output = hgsam_model.vision_encoder(x)
    # print(type(output))

    # # t_layer_0, block = hgsam_model.vision_encoder.layers[0]

    # print(hgsam_model.vision_encoder.layers[0])

    # for name, child in hgsam_model.vision_encoder.layers[0].named_children():
    #     print('------')
    #     print(name)
    #     print('||')
    #     print(child)
    #     print('------')


    # linear = hgsam_model.vision_encoder.layers[0].attn.qkv
    # print(linear.bias)
    # print(linear.weight)
    # print(linear.in_features)
    # print(linear.out_features)
    # y = torch.rand(768)
    # y_out = linear(y)
    # print(y_out)

    for t_layer_i, blk in enumerate(hgsam_model.vision_encoder.layers):
        w_qkv_linear = blk.attn.qkv.in_features
        w_qkv_bias = blk.attn.qkv.bias
        w_proj_linear = blk.attn.proj.weight
        w_proj_bias = blk.attn.proj.bias
        print(w_qkv_linear)
        # print(w_qkv_linear.in_features)
        print(w_qkv_bias.shape)
        print(w_proj_linear.shape)
        print(w_proj_bias.shape)