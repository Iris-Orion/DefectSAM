"""
utils/loratask.py 单元测试。

覆盖：FusedQKVSplitLinear（含错误路径）、filter_target_modules、
      prepare_sam_qkv_for_qv_peft（包括幂等性与错误路径）、
      get_sam_target_modules / get_sam_qv_target_modules_for_peft 的错误路径、
      create_model_from_type 的可在 CPU 上触发的错误分支。

不依赖 GPU、SAM 权重或真实数据集。
"""
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


# ─── 辅助：最小假模型 ──────────────────────────────────────────────

class _FakeAttn:
    def __init__(self):
        self.qkv = nn.Linear(8, 12)


class _FakeLayer:
    def __init__(self):
        self.attn = _FakeAttn()


class _FakeVisionEncoder:
    def __init__(self):
        self.layers = [_FakeLayer(), _FakeLayer()]


class _FakeModelForQKVPeft:
    def __init__(self):
        self.vision_encoder = _FakeVisionEncoder()


class _FakeAttentionQKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 12)


class _FakeEncoderWithQKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = _FakeAttentionQKV()
        self.layer1 = _FakeAttentionQKV()


class _FakeAttentionQV(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 4)
        self.k_proj = nn.Linear(8, 4)
        self.v_proj = nn.Linear(8, 4)


class _FakeEncoderWithQV(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = _FakeAttentionQV()


class _FakeEncoderNoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()


class _FakeModelWithVisionEncoder:
    def __init__(self, encoder):
        self.vision_encoder = encoder
        self.mask_decoder = _FakeEncoderWithQKV()


# ─── A. FusedQKVSplitLinear ───────────────────────────────────────

def test_fused_qkv_split_linear_output_matches_original():
    """包装后的 forward 输出应与原始 qkv 线性层完全一致。"""
    torch.manual_seed(0)
    qkv = nn.Linear(8, 12)
    wrapper = lt.FusedQKVSplitLinear(qkv)
    x = torch.randn(3, 8)
    with torch.no_grad():
        out_ref = qkv(x)
        out_wrapped = wrapper(x)
    assert out_wrapped.shape == (3, 12)
    assert torch.allclose(out_wrapped, out_ref, atol=1e-5), \
        "FusedQKVSplitLinear 的输出与原始层不一致"


def test_fused_qkv_split_linear_no_bias():
    """无 bias 的情况：has_bias=False，输出形状正确。"""
    qkv = nn.Linear(8, 12, bias=False)
    wrapper = lt.FusedQKVSplitLinear(qkv)
    assert not wrapper.has_bias
    x = torch.randn(2, 8)
    out = wrapper(x)
    assert out.shape == (2, 12)


def test_fused_qkv_split_linear_type_error():
    """传入非 nn.Linear 应抛出 TypeError。"""
    with pytest.raises(TypeError, match="nn.Linear"):
        lt.FusedQKVSplitLinear(nn.ReLU())


def test_fused_qkv_split_linear_value_error_not_divisible():
    """out_features 不能被 3 整除时应抛出 ValueError。"""
    qkv = nn.Linear(8, 10)  # 10 % 3 != 0
    with pytest.raises(ValueError, match="divisible by 3"):
        lt.FusedQKVSplitLinear(qkv)


def test_fused_qkv_split_linear_submodule_weights_correct():
    """q/k/v 子层权重应分别等于原始 qkv 权重的对应切片。"""
    torch.manual_seed(1)
    qkv = nn.Linear(8, 12)
    wrapper = lt.FusedQKVSplitLinear(qkv)
    with torch.no_grad():
        assert torch.allclose(wrapper.q_proj.weight, qkv.weight[:4])
        assert torch.allclose(wrapper.k_proj.weight, qkv.weight[4:8])
        assert torch.allclose(wrapper.v_proj.weight, qkv.weight[8:])


# ─── B. filter_target_modules ─────────────────────────────────────

def test_filter_target_modules_finds_qkv_linear():
    """模块名包含 'qkv' 且类型为 Linear 时应被选中。"""
    encoder = _FakeEncoderWithQKV()
    result = lt.filter_target_modules(encoder.named_modules(), ['qkv'])
    assert any('qkv' in name for name in result)


def test_filter_target_modules_excludes_non_linear():
    """非 Linear 的模块即使名称匹配也不应被选中。"""
    encoder = _FakeEncoderNoLinear()
    result = lt.filter_target_modules(encoder.named_modules(), ['relu'])
    assert result == []


def test_filter_target_modules_multiple_substrings():
    """多个 target_substrings 时，只要任一匹配就应包含该模块。"""
    encoder = _FakeEncoderWithQV()
    result = lt.filter_target_modules(encoder.named_modules(), ['q_proj', 'v_proj'])
    names = '\n'.join(result)
    assert 'q_proj' in names
    assert 'v_proj' in names
    assert 'k_proj' not in names


def test_filter_target_modules_no_match_returns_empty():
    """没有任何匹配时返回空列表。"""
    encoder = _FakeEncoderWithQKV()
    result = lt.filter_target_modules(encoder.named_modules(), ['nonexistent'])
    assert result == []


# ─── C. prepare_sam_qkv_for_qv_peft ──────────────────────────────

def test_prepare_sam_qkv_wraps_layers():
    """调用后每个 layer.attn.qkv 都应被替换为 FusedQKVSplitLinear。"""
    model = _FakeModelForQKVPeft()
    result = lt.prepare_sam_qkv_for_qv_peft(model, 'vision_encoder')
    assert result is model
    for layer in model.vision_encoder.layers:
        assert isinstance(layer.attn.qkv, lt.FusedQKVSplitLinear)


def test_prepare_sam_qkv_idempotent():
    """重复调用不应二次包装（已是 FusedQKVSplitLinear 则跳过）。"""
    model = _FakeModelForQKVPeft()
    lt.prepare_sam_qkv_for_qv_peft(model, 'vision_encoder')
    ids_after_first = [id(layer.attn.qkv) for layer in model.vision_encoder.layers]
    lt.prepare_sam_qkv_for_qv_peft(model, 'vision_encoder')
    ids_after_second = [id(layer.attn.qkv) for layer in model.vision_encoder.layers]
    assert ids_after_first == ids_after_second


def test_prepare_sam_qkv_wrong_target_raises():
    """target_part != 'vision_encoder' 应抛出 ValueError。"""
    model = _FakeModelForQKVPeft()
    with pytest.raises(ValueError):
        lt.prepare_sam_qkv_for_qv_peft(model, 'mask_decoder')


# ─── D. get_sam_target_modules 错误路径 ───────────────────────────

def test_get_sam_target_modules_finds_qkv(capsys):
    model = _FakeModelWithVisionEncoder(_FakeEncoderWithQKV())
    result = lt.get_sam_target_modules(model, 'vision_encoder')
    assert len(result) > 0
    assert all('qkv' in name for name in result)


def test_get_sam_target_modules_unknown_part_raises():
    model = _FakeModelWithVisionEncoder(_FakeEncoderWithQKV())
    with pytest.raises(ValueError, match="target_part"):
        lt.get_sam_target_modules(model, 'unknown_part')


def test_get_sam_target_modules_no_match_raises():
    """encoder 中没有 q_proj/k_proj/v_proj/qkv 时应抛出 ValueError。"""
    model = _FakeModelWithVisionEncoder(_FakeEncoderNoLinear())
    with pytest.raises(ValueError, match="未找到"):
        lt.get_sam_target_modules(model, 'vision_encoder')


# ─── E. get_sam_qv_target_modules_for_peft 错误路径 ──────────────

def test_get_sam_qv_target_modules_finds_q_and_v(capsys):
    model = _FakeModelWithVisionEncoder(_FakeEncoderWithQV())
    result = lt.get_sam_qv_target_modules_for_peft(model, 'vision_encoder')
    names = '\n'.join(result)
    assert 'q_proj' in names
    assert 'v_proj' in names
    assert 'k_proj' not in names


def test_get_sam_qv_target_modules_wrong_target_raises():
    model = _FakeModelWithVisionEncoder(_FakeEncoderWithQV())
    with pytest.raises(ValueError):
        lt.get_sam_qv_target_modules_for_peft(model, 'mask_decoder')


def test_get_sam_qv_target_modules_no_qv_raises():
    """模型中没有 q_proj/v_proj 时（只有 qkv 整合层）应抛出 ValueError。"""
    model = _FakeModelWithVisionEncoder(_FakeEncoderWithQKV())
    with pytest.raises(ValueError, match="q_proj/v_proj"):
        lt.get_sam_qv_target_modules_for_peft(model, 'vision_encoder')


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
