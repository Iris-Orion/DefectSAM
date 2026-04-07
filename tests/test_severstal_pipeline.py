"""
Severstal 微调 pipeline 单元测试。

只覆盖 CPU 可测的纯函数与轻量逻辑，不依赖 SAM 权重 / GPU / 真实数据集图片。
集成测试用 @pytest.mark.integration 标记，默认跳过；用 `-m integration` 单独运行。

运行方式：
    pytest tests/test_severstal_pipeline.py -v               # 跑全部 unit
    pytest tests/test_severstal_pipeline.py -v -m integration  # 跑集成
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# 让仓库根目录可被 import
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import severstal as sv  # noqa: E402
from utils import helper_function as hf  # noqa: E402
from utils.utils import compute_dice_score, compute_iou_score  # noqa: E402


# ─── A. RLE / mask 编解码 ─────────────────────────────────────

def test_rle2mask_raw_empty():
    """全空缺陷 df → 返回 (256,1600,4) uint8 全零 mask。"""
    df = pd.DataFrame({"ImageId": ["x.jpg"], "ClassId": [0], "EncodedPixels": [None]})
    m = sv._rle2mask_raw("x.jpg", df)
    assert m.shape == (256, 1600, 4)
    assert m.dtype == np.uint8
    assert m.sum() == 0


def test_rle2mask_raw_single_class_pixels():
    """构造一行 RLE，验证对应通道在解码后含有正确数量的前景像素。"""
    df = pd.DataFrame({
        "ImageId": ["a.jpg"],
        "ClassId": [2],  # → channel index 1
        "EncodedPixels": ["1 5 100 3"],
    })
    m = sv._rle2mask_raw("a.jpg", df)
    assert m.shape == (256, 1600, 4)
    # channel 1（class_id=2-1）总共应该有 5+3 = 8 个像素被置 1
    assert int(m[..., 1].sum()) == 8
    # 其他通道全 0
    assert m[..., 0].sum() == 0
    assert m[..., 2].sum() == 0
    assert m[..., 3].sum() == 0


def test_no_inmemory_cache_module_attr():
    """B1 回归保护：模块不应再拥有 _MASK_CACHE 这个无界字典。"""
    assert not hasattr(sv, "_MASK_CACHE"), (
        "_MASK_CACHE 已被移除，禁止再添加进程级内存缓存（会导致 worker OOM）"
    )


def test_rle2mask_disk_cache_hit(tmp_path):
    """放一个 .npz 到 tmp_path 中，rle2mask 应直接读取磁盘而不需要 df 内容。"""
    fake_mask = np.zeros((256, 1600, 4), dtype=np.uint8)
    fake_mask[10, 20, 0] = 1
    np.savez_compressed(tmp_path / "img.npz", mask=fake_mask)

    # 传入空 df：如果走到 _rle2mask_raw 会得到全 0 mask；
    # 走磁盘路径才会得到我们写入的有 1 像素的 mask
    empty_df = pd.DataFrame(columns=["ImageId", "ClassId", "EncodedPixels"])
    out = sv.rle2mask("img.jpg", empty_df, cache_dir=str(tmp_path))
    assert out.shape == (256, 1600, 4)
    assert out[10, 20, 0] == 1
    assert out.sum() == 1


def test_rle2mask_fallback_no_cache(tmp_path):
    """磁盘未命中时 fallback 到 RLE 解码，且不写任何缓存。"""
    df = pd.DataFrame({
        "ImageId": ["b.jpg"],
        "ClassId": [1],
        "EncodedPixels": ["1 4"],
    })
    out = sv.rle2mask("b.jpg", df, cache_dir=str(tmp_path))
    assert out.shape == (256, 1600, 4)
    assert int(out[..., 0].sum()) == 4
    # 确认没有副产物文件被写到 tmp_path
    assert not any(tmp_path.iterdir())


# ─── B. letterbox / bbox ──────────────────────────────────────

def _make_dummy_dataset():
    """构造一个不依赖磁盘的最小 SteelDataset_WithBoxPrompt 占位实例，用于复用其方法。"""
    instance = sv.SteelDataset_WithBoxPrompt.__new__(sv.SteelDataset_WithBoxPrompt)
    instance.is_train = False
    return instance


def test_letterbox_img_np_shape_and_padding():
    ds = _make_dummy_dataset()
    img = np.full((256, 1600, 3), 200, dtype=np.uint8)
    out = ds.letterbox_img_np(img, [1024, 1024])
    assert out.shape == (1024, 1024, 3)
    assert out.dtype == np.uint8
    # padding 区一定包含 (128,128,128) 灰背景
    assert (out == 128).any()
    # 中心区一定保留有原值 200
    assert (out == 200).any()


def test_letterbox_mask_shape_and_padding():
    ds = _make_dummy_dataset()
    mask = np.ones((256, 1600, 4), dtype=np.uint8)
    out = ds.letterbox_mask(mask, (1024, 1024))
    assert out.shape == (1024, 1024, 4)
    # padding 区为 0
    assert (out == 0).any()
    # 缩放后原内容应仍存在
    assert out.sum() > 0


def test_letterbox_mask_values_preserved():
    ds = _make_dummy_dataset()
    mask = np.zeros((256, 1600, 1), dtype=np.uint8)
    mask[100:150, 200:300, 0] = 1
    out = ds.letterbox_mask(mask, (1024, 1024))
    assert out.shape == (1024, 1024, 1)
    assert out.sum() > 0


def test_get_bounding_box_empty():
    """severstal.get_bounding_box: 空 mask 返回 [0,0,0,0]。"""
    mask = torch.zeros((256, 1600), dtype=torch.float32)
    bbox = sv.get_bounding_box(mask, perturb=False)
    assert bbox == [0, 0, 0, 0]


def test_get_bounding_box_no_perturb_matches_rect():
    mask = torch.zeros((256, 1600), dtype=torch.float32)
    mask[50:80, 100:200] = 1
    bbox = sv.get_bounding_box(mask, perturb=False)
    # bbox = [x_min, y_min, x_max, y_max]
    assert bbox[0] == 100
    assert bbox[1] == 50
    assert bbox[2] == 199
    assert bbox[3] == 79


def test_get_bounding_box_perturb_within_image():
    """perturb=True 下 bbox 仍应落在图像范围内。"""
    np.random.seed(0)
    mask = torch.zeros((256, 1600), dtype=torch.float32)
    mask[50:80, 100:200] = 1
    for _ in range(20):
        bbox = sv.get_bounding_box(mask, perturb=True)
        x_min, y_min, x_max, y_max = bbox
        assert 0 <= x_min <= x_max <= 1600
        assert 0 <= y_min <= y_max <= 256


# ─── C. df 处理 ────────────────────────────────────────────────

def _write_minimal_csv(tmp_path):
    """写一个含 60 行（30 张 unique 图）的最小 train.csv，无需依赖磁盘图片。"""
    rows = []
    for i in range(30):
        for c in (1, 2):
            rows.append({
                "ImageId": f"img_{i:03d}.jpg",
                "ClassId": c,
                "EncodedPixels": "1 5",
            })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_traindf_preprocess_ratio_float_safe(tmp_path, monkeypatch, capsys):
    """B3 回归：0.6+0.2+0.2 浮点和不严格等于 1.0，但 abs(...)<1e-6 不应误报。"""
    csv_path = _write_minimal_csv(tmp_path)
    # build_mask_cache 会尝试创建目录并写文件，传入 tmp_path 让其无副作用
    monkeypatch.setattr(sv, "build_mask_cache", lambda *a, **kw: None)
    train_df, val_df, test_df = sv.traindf_preprocess(
        split_seed=42,
        csv_path=str(csv_path),
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        include_no_defect=False,
    )
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_traindf_preprocess_ratio_invalid(tmp_path, monkeypatch):
    csv_path = _write_minimal_csv(tmp_path)
    monkeypatch.setattr(sv, "build_mask_cache", lambda *a, **kw: None)
    with pytest.raises(ValueError):
        sv.traindf_preprocess(
            csv_path=str(csv_path),
            train_ratio=0.5, val_ratio=0.2, test_ratio=0.2,
            include_no_defect=False,
        )


def test_traindf_preprocess_split_disjoint(tmp_path, monkeypatch):
    csv_path = _write_minimal_csv(tmp_path)
    monkeypatch.setattr(sv, "build_mask_cache", lambda *a, **kw: None)
    train_df, val_df, test_df = sv.traindf_preprocess(
        split_seed=42,
        csv_path=str(csv_path),
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        include_no_defect=False,
    )
    train_ids = set(train_df["ImageId"].unique())
    val_ids = set(val_df["ImageId"].unique())
    test_ids = set(test_df["ImageId"].unique())
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids | test_ids) == 30


# ─── D. 指标 ───────────────────────────────────────────────────

def test_compute_dice_perfect():
    pred = torch.full((2, 1, 8, 8), 10.0)        # sigmoid > 0.5 → 全 1
    target = torch.ones((2, 1, 8, 8))
    score = compute_dice_score(pred, target)
    assert abs(score - 1.0) < 1e-3


def test_compute_dice_zero_intersection():
    pred = torch.full((2, 1, 8, 8), -10.0)       # sigmoid < 0.5 → 全 0
    target = torch.ones((2, 1, 8, 8))
    score = compute_dice_score(pred, target)
    assert score < 0.01


def test_compute_iou_perfect():
    pred = torch.full((2, 1, 8, 8), 10.0)
    target = torch.ones((2, 1, 8, 8))
    score = compute_iou_score(pred, target)
    assert abs(score - 1.0) < 1e-3


def test_compute_iou_disjoint():
    pred = torch.full((1, 1, 4, 4), 10.0)
    target = torch.zeros((1, 1, 4, 4))
    # pred 全 1, target 全 0 → intersection=0, union=16, iou≈0
    score = compute_iou_score(pred, target)
    assert score < 0.01


# ─── E. seed / DDP helpers / config ──────────────────────────

def test_set_seed_reproducible():
    hf.set_seed(123, seed_offset=0)
    a = (torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random())
    hf.set_seed(123, seed_offset=0)
    b = (torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random())
    assert a == b


def test_set_seed_offset_diverges():
    hf.set_seed(123, seed_offset=0)
    a = torch.rand(3).tolist()
    hf.set_seed(123, seed_offset=1)
    b = torch.rand(3).tolist()
    assert a != b


def test_setup_ddp_no_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    info = hf.setup_ddp()
    assert info["ddp"] is False
    assert info["master_process"] is True
    assert info["rank"] == 0
    assert info["world_size"] == 1
    assert info["device"] is None


def test_setup_ddp_with_env(monkeypatch):
    """mock RANK/LOCAL_RANK/WORLD_SIZE + init_process_group / set_device 为 no-op。"""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(hf, "init_process_group", lambda *a, **kw: None)
    monkeypatch.setattr(hf, "_configure_nccl_compat_env", lambda: {})
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **kw: None)
    info = hf.setup_ddp()
    assert info["ddp"] is True
    assert info["master_process"] is True
    assert info["rank"] == 0
    assert info["local_rank"] == 0
    assert info["world_size"] == 1
    # device 应该是 cuda:0 类型（即便没有 GPU 也只是构造一个 torch.device 对象）
    assert info["device"].type == "cuda"


def test_get_severstal_ft_args_defaults(monkeypatch):
    from utils.config import get_severstal_ft_args
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = get_severstal_ft_args()
    # finetune defaults
    assert args.batch_size == 2
    assert args.num_epochs == 50
    # 我们新增的开关默认 False（关闭训练阶段 HD95）
    assert args.train_hd95 is False
    # 默认未启用 mini dataset
    assert args.mini_dataset is False


# ─── F. CUDAPrefetcher (CPU) ──────────────────────────────────

def test_cuda_prefetcher_iterates_all():
    """在 CPU 上跑 CUDAPrefetcher，验证迭代次数 = 输入长度。"""
    from utils.finetune_engine import CUDAPrefetcher

    class FakeLoader:
        def __init__(self, items):
            self.items = items
        def __iter__(self):
            return iter(self.items)
        def __len__(self):
            return len(self.items)

    batches = [{"x": torch.tensor([float(i)])} for i in range(5)]
    loader = FakeLoader(batches)
    prefetcher = CUDAPrefetcher(loader, device=torch.device("cpu"))
    seen = list(prefetcher)
    assert len(seen) == 5
    assert [b["x"].item() for b in seen] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_cuda_prefetcher_empty():
    from utils.finetune_engine import CUDAPrefetcher

    class FakeLoader:
        def __iter__(self):
            return iter([])
        def __len__(self):
            return 0

    prefetcher = CUDAPrefetcher(FakeLoader(), device=torch.device("cpu"))
    assert list(prefetcher) == []


# ─── G. 集成测试（默认跳过） ─────────────────────────────────

@pytest.mark.integration
def test_steel_dataset_getitem_smoke():
    """需要真实的 severstal 数据集 + 缓存目录。手动开启时使用。"""
    data_path = REPO_ROOT / "data" / "severstal_steel_defect_detection"
    if not (data_path / "train.csv").exists():
        pytest.skip("severstal 数据集不可用")
    train_df, _, _ = sv.traindf_preprocess(
        csv_path=str(data_path / "train.csv"),
        include_no_defect=False,
    )
    train_transforms, _ = sv.get_severstal_ft_albumentations_transforms()
    ds = sv.SteelDataset_WithBoxPrompt(train_df, data_path=str(data_path),
                                        transforms=train_transforms, is_train=True)
    sample = ds[0]
    assert sample["image"].shape == (3, 1024, 1024)
    assert sample["mask"].shape == (256, 1600)
    assert sample["letterboxed_mask"].shape == (1024, 1024)
    assert sample["bbox"].shape == (4,)
