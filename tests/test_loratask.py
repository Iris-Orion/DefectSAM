"""
utils/loratask.py 单元测试。

加载真实 SAM-base 模型，测试 FusedQKVSplitLinear 是否正确替换 vision encoder 中的 qkv 层。
不依赖 GPU 或真实数据集。
"""
import copy
import argparse
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.loratask as lt
from transformers import SamConfig, SamModel


# ─── SAM-base 风格 fixture（2层，与真实 qkv 维度一致）────────────────

def _make_sam_config():
    """与 sam-vit-base 架构完全一致，仅将层数缩减为 2 以加速测试。"""
    config = SamConfig()
    config.vision_config.num_hidden_layers = 2
    config.vision_config.num_global_attn_indices = [1]
    return config


@pytest.fixture(scope="module")
def sam_base():
    """SAM-base 随机权重模型，整个测试模块共享一份（不修改）。"""
    return SamModel(_make_sam_config())


@pytest.fixture
def sam_fresh():
    """每个测试独立的 SAM 模型副本，用于需要原地修改的测试。"""
    return SamModel(_make_sam_config())


@pytest.fixture
def sam_qv_split(sam_fresh):
    """已执行 qkv 拆分的 SAM 模型。"""
    lt.prepare_sam_qkv_for_qv_peft(sam_fresh, 'vision_encoder')
    return sam_fresh


# 无法匹配 qkv/q_proj/v_proj 的极简 encoder，仅用于"未找到"错误路径
class _NoLinearEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()


class _ModelNoLinear:
    """只有 vision_encoder 和 mask_decoder 属性，但 vision_encoder 不含目标线性层。"""
    def __init__(self):
        self.vision_encoder = _NoLinearEncoder()
        self.mask_decoder = _NoLinearEncoder()


# ─── A. FusedQKVSplitLinear ───────────────────────────────────────
# 使用真实 SAM-base qkv 尺寸：Linear(768, 2304)

SAM_HIDDEN = 768
SAM_QKV_OUT = 2304  # 768 * 3


def test_fused_qkv_split_linear_output_matches_original():
    """包装后的 forward 输出应与原始 qkv 线性层完全一致。"""
    torch.manual_seed(0)
    qkv = nn.Linear(SAM_HIDDEN, SAM_QKV_OUT)
    wrapper = lt.FusedQKVSplitLinear(qkv)
    x = torch.randn(3, SAM_HIDDEN)
    with torch.no_grad():
        out_ref = qkv(x)
        out_wrapped = wrapper(x)
    assert out_wrapped.shape == (3, SAM_QKV_OUT)
    assert torch.allclose(out_wrapped, out_ref, atol=1e-5), \
        "FusedQKVSplitLinear 的输出与原始层不一致"


def test_fused_qkv_split_linear_no_bias():
    """无 bias 的情况：has_bias=False，输出形状正确。"""
    qkv = nn.Linear(SAM_HIDDEN, SAM_QKV_OUT, bias=False)
    wrapper = lt.FusedQKVSplitLinear(qkv)
    assert not wrapper.has_bias
    x = torch.randn(2, SAM_HIDDEN)
    assert wrapper(x).shape == (2, SAM_QKV_OUT)


def test_fused_qkv_split_linear_type_error():
    """传入非 nn.Linear 应抛出 TypeError。"""
    with pytest.raises(TypeError, match="nn.Linear"):
        lt.FusedQKVSplitLinear(nn.ReLU())


def test_fused_qkv_split_linear_value_error_not_divisible():
    """out_features 不能被 3 整除时应抛出 ValueError。"""
    with pytest.raises(ValueError, match="divisible by 3"):
        lt.FusedQKVSplitLinear(nn.Linear(SAM_HIDDEN, SAM_HIDDEN + 1))


def test_fused_qkv_split_linear_submodule_weights_correct():
    """q/k/v 子层权重应分别等于原始 qkv 权重的对应切片。"""
    torch.manual_seed(1)
    qkv = nn.Linear(SAM_HIDDEN, SAM_QKV_OUT)
    wrapper = lt.FusedQKVSplitLinear(qkv)
    head_dim = SAM_QKV_OUT // 3
    with torch.no_grad():
        assert torch.allclose(wrapper.q_proj.weight, qkv.weight[:head_dim])
        assert torch.allclose(wrapper.k_proj.weight, qkv.weight[head_dim:2 * head_dim])
        assert torch.allclose(wrapper.v_proj.weight, qkv.weight[2 * head_dim:])


# ─── B. filter_target_modules ─────────────────────────────────────

def test_filter_target_modules_finds_qkv_linear(sam_base):
    result = lt.filter_target_modules(sam_base.vision_encoder.named_modules(), ['qkv'])
    assert len(result) > 0
    assert all('qkv' in name for name in result)


def test_filter_target_modules_excludes_non_linear():
    result = lt.filter_target_modules(_NoLinearEncoder().named_modules(), ['act'])
    assert result == []


def test_filter_target_modules_multiple_substrings(sam_qv_split):
    result = lt.filter_target_modules(
        sam_qv_split.vision_encoder.named_modules(), ['q_proj', 'v_proj']
    )
    names = '\n'.join(result)
    assert 'q_proj' in names
    assert 'v_proj' in names
    assert 'k_proj' not in names


def test_filter_target_modules_no_match_returns_empty(sam_base):
    result = lt.filter_target_modules(
        sam_base.vision_encoder.named_modules(), ['nonexistent']
    )
    assert result == []


# ─── C. prepare_sam_qkv_for_qv_peft ──────────────────────────────

def test_prepare_sam_qkv_wraps_layers(sam_fresh):
    """调用后每个 layer.attn.qkv 都应被替换为 FusedQKVSplitLinear。"""
    result = lt.prepare_sam_qkv_for_qv_peft(sam_fresh, 'vision_encoder')
    assert result is sam_fresh
    for layer in sam_fresh.vision_encoder.layers:
        assert isinstance(layer.attn.qkv, lt.FusedQKVSplitLinear)


def test_prepare_sam_qkv_idempotent(sam_fresh):
    """重复调用不应二次包装。"""
    lt.prepare_sam_qkv_for_qv_peft(sam_fresh, 'vision_encoder')
    ids_first = [id(layer.attn.qkv) for layer in sam_fresh.vision_encoder.layers]
    lt.prepare_sam_qkv_for_qv_peft(sam_fresh, 'vision_encoder')
    ids_second = [id(layer.attn.qkv) for layer in sam_fresh.vision_encoder.layers]
    assert ids_first == ids_second


def test_prepare_sam_qkv_wrong_target_raises(sam_fresh):
    """target_part != 'vision_encoder' 应抛出 ValueError。"""
    with pytest.raises(ValueError):
        lt.prepare_sam_qkv_for_qv_peft(sam_fresh, 'mask_decoder')


# ─── D. get_sam_target_modules ────────────────────────────────────

def test_get_sam_target_modules_finds_qkv(sam_base, capsys):
    result = lt.get_sam_target_modules(sam_base, 'vision_encoder')
    assert len(result) > 0
    assert all('qkv' in name for name in result)


def test_get_sam_target_modules_unknown_part_raises(sam_base):
    with pytest.raises(ValueError, match="target_part"):
        lt.get_sam_target_modules(sam_base, 'unknown_part')


def test_get_sam_target_modules_no_match_raises():
    """vision_encoder 中无匹配层时应抛出 ValueError。"""
    with pytest.raises(ValueError, match="未找到"):
        lt.get_sam_target_modules(_ModelNoLinear(), 'vision_encoder')


# ─── E. get_sam_qv_target_modules_for_peft ───────────────────────

def test_get_sam_qv_target_modules_finds_q_and_v(sam_qv_split, capsys):
    result = lt.get_sam_qv_target_modules_for_peft(sam_qv_split, 'vision_encoder')
    names = '\n'.join(result)
    assert 'q_proj' in names
    assert 'v_proj' in names
    assert 'k_proj' not in names


def test_get_sam_qv_target_modules_wrong_target_raises(sam_qv_split):
    with pytest.raises(ValueError):
        lt.get_sam_qv_target_modules_for_peft(sam_qv_split, 'mask_decoder')


def test_get_sam_qv_target_modules_no_qv_raises(sam_base):
    """真实 SAM 在 qkv 拆分前只有 qkv，没有 q_proj/v_proj，应抛出 ValueError。"""
    with pytest.raises(ValueError, match="q_proj/v_proj"):
        lt.get_sam_qv_target_modules_for_peft(sam_base, 'vision_encoder')


# ─── F. create_model_from_type 错误路径 ───────────────────────────

def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        ft_type='lora_attn_qv',
        lora_rank=4, lora_alpha=4, lora_dropout=0.0,
        sam_type='sam_base', num_epochs=1,
        save_custom_lora=False, save_hf_format=False,
        use_loraplus_optim=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_create_model_from_type_unknown_raises():
    args = _make_args(ft_type='totally_unknown_type')
    with pytest.raises(ValueError, match="Unknown model type"):
        lt.create_model_from_type(args)


def test_create_model_from_type_adalora_without_dataloader_raises(monkeypatch):
    monkeypatch.setattr(lt.SamModel, 'from_pretrained', lambda path: object())
    args = _make_args(ft_type='adalora_encoder')
    with pytest.raises(ValueError, match="train_dataloader"):
        lt.create_model_from_type(args, train_dataloader=None)


def test_create_model_from_type_dora_with_save_custom_lora_raises(monkeypatch):
    monkeypatch.setattr(lt.SamModel, 'from_pretrained', lambda path: object())
    args = _make_args(ft_type='dora_qv_encoder', save_custom_lora=True)
    with pytest.raises(ValueError, match="dora_qv_encoder"):
        lt.create_model_from_type(args)


def test_create_model_from_type_lokr_with_save_custom_lora_raises(monkeypatch):
    monkeypatch.setattr(lt.SamModel, 'from_pretrained', lambda path: object())
    args = _make_args(ft_type='lokr_qv_encoder', save_custom_lora=True)
    with pytest.raises(ValueError, match="lokr_qv_encoder"):
        lt.create_model_from_type(args)
