# @copyright https://github.com/lucidrains/conformer/blob/master/conformer/conformer.py
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()
    
class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)
    
    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads = 8,
            dim_head = 64,
            dropout = 0.,
            max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)
    
    def forward(
            self,
            x,
            context = None, 
            mask = None,
            context_mask = None
        ):
        b, n, device, h, max_pos_emb, has_context = x.shape[0], x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        contxt = default(context, x)

        # 线性变换生成 Q, K, V
        q, k, v = (self.to_q(x), *self.to_kv(contxt).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 计算注意力分数 (Dot-product attention)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # 添加相对位置偏差 (Relative positional bias)
        if not has_context:
            seq = torch.arange(n, device = device)
            dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
            dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
            rel_pos_emb = self.rel_pos_emb(dist)
            pos_attn = einsum('b h i d, i j d -> b h i j', q, rel_pos_emb) * self.scale
            dots = dots + pos_attn

        # 掩码处理 (Masking)
        if exists(mask) or exists(context_mask):
            if not exists(mask):
                mask = torch.ones((b, n), device = device).bool()
            if not exists(context_mask):
                context_mask = mask if not has_context else torch.ones((b, k.shape[-2]), device = device).bool()

            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        # Softmax 与 加权求和
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    """
    Conformer 中的前馈网络模块 (FFN)
    通常使用 Swish 激活函数
    """
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    """
    Conformer 中的卷积模块
    结构: LayerNorm -> 1x1 Conv -> GLU -> Depthwise Conv -> BatchNorm -> Swish -> 1x1 Conv
    """
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim = 1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerBlock(nn.Module):
    """
    单个 Conformer 块
    结构: 0.5 * FFN -> MHSA -> Conv -> 0.5 * FFN -> LayerNorm
    这种“三明治”结构也被称为 Macaron 风格
    """
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.ff1 = Scale(0.5, PreNorm(dim, FeedForward(dim, mult = ff_mult, dropout = ff_dropout)))
        self.attn = PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout))
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = Scale(0.5, PreNorm(dim, FeedForward(dim, mult = ff_mult, dropout = ff_dropout)))
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x

class Conformer(nn.Module):
    """
    Conformer 编码器模型
    由多个 ConformerBlock 堆叠而成
    """
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                conv_dropout = conv_dropout,
                conv_causal = conv_causal
            ))

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x
