"""
finetune_engine.run_finetune_engine 回归测试。

只覆盖这次原地重构涉及的控制流和副作用边界，不依赖 GPU / SAM 权重 / 真实数据集。
"""
import sys
from pathlib import Path

import pytest
import torch

# 让仓库根目录可被 import
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
UTILS_ROOT = REPO_ROOT / "utils"
if str(UTILS_ROOT) not in sys.path:
    sys.path.append(str(UTILS_ROOT))

from utils import finetune_engine as fe  # noqa: E402


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.vision_encoder = torch.nn.Identity()

    def forward(self, *args, **kwargs):
        del args, kwargs
        return self.weight

    def to(self, device):
        self._device = device
        return self


class FakeDDP(torch.nn.Module):
    def __init__(self, module, device_ids=None, find_unused_parameters=False):
        super().__init__()
        del device_ids, find_unused_parameters
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def train(self, mode=True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def modules(self):
        return self.module.modules()

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)


class DummyScheduler:
    def step(self):
        return None


class DummyMFUEstimator:
    def __init__(self, sam_type, gpu_type):
        self.sam_type = sam_type
        self.gpu_type = gpu_type

    def summary(self):
        return f"mfu({self.sam_type},{self.gpu_type})"


class DummyMFUTracker:
    def __init__(self, estimator, batch_size, ema_alpha):
        self.estimator = estimator
        self.batch_size = batch_size
        self.ema_alpha = ema_alpha


class FakeGradScaler:
    def __init__(self, device_type, enabled):
        self.device_type = device_type
        self.enabled = enabled


def _base_hyperparameters(**overrides):
    hyperparameters = {
        "seed": 7,
        "save_hf_format": False,
        "save_custom_lora": False,
        "ft_type": "loradsc_qv_residual_gated",
        "lora_rank": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "num_epochs": 1,
        "warmup_ratio": 0.0,
        "patience": 2,
        "min_delta": 0.0,
        "use_loraplus_optim": False,
        "disable_early_stop": True,
        "grad_clip": 1.0,
        "multimask": False,
        "train_hd95": False,
        "use_swanlab": False,
        "swanlab_project": "unit-test",
        "task_name": "engine",
        "no_compile": True,
        "batch_size": 1,
        "sam_type": "sam_base",
    }
    hyperparameters.update(overrides)
    return hyperparameters


def _patch_common_runtime(monkeypatch):
    monkeypatch.setattr(fe, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(fe, "get_lr_scheduler", lambda *args, **kwargs: DummyScheduler())
    monkeypatch.setattr(fe, "SAMMFUEstimator", DummyMFUEstimator)
    monkeypatch.setattr(fe, "MFUTracker", DummyMFUTracker)
    monkeypatch.setattr(fe, "print_trainable_parameters", lambda model: None)
    monkeypatch.setattr(fe, "debug_print_optimizer_param_groups", lambda optimizer: None)
    monkeypatch.setattr(fe.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(fe.torch.amp, "GradScaler", lambda *args, **kwargs: None)


def test_run_finetune_engine_single_gpu_smoke(tmp_path, monkeypatch):
    _patch_common_runtime(monkeypatch)

    swanlab_events = {"init": 0, "logs": [], "finish": 0}
    save_model_calls = []
    saved_log_epochs = []
    eval_results = iter([
        (0.2, 0.8, 0.7, 1.2),  # val
        (0.1, 0.9, 0.8, 1.0),  # final test
    ])

    def fake_swanlab_init(**kwargs):
        del kwargs
        swanlab_events["init"] += 1
        return object()

    def fake_swanlab_log(payload, step=None):
        swanlab_events["logs"].append((step, payload))

    def fake_swanlab_finish():
        swanlab_events["finish"] += 1

    def fake_train_one_epoch(*args, **kwargs):
        global_step = kwargs["global_step"]
        return 0.3, 0.4, 0.5, 6.0, 0.25, global_step + 1

    def fake_evaluate(*args, **kwargs):
        del args, kwargs
        return next(eval_results)

    def fake_save_model(**kwargs):
        save_model_calls.append(kwargs["target_dir"])
        return str(tmp_path / "best_model.pth")

    def fake_save_training_logs(**kwargs):
        saved_log_epochs.append(kwargs["epoch"])
        return {"last_completed_epoch": kwargs["epoch"]}

    monkeypatch.setattr(fe.swanlab, "init", fake_swanlab_init)
    monkeypatch.setattr(fe.swanlab, "log", fake_swanlab_log)
    monkeypatch.setattr(fe.swanlab, "finish", fake_swanlab_finish)
    monkeypatch.setattr(fe, "_train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(fe, "_evaluate", fake_evaluate)
    monkeypatch.setattr(fe, "save_model", fake_save_model)
    monkeypatch.setattr(fe, "save_training_logs", fake_save_training_logs)

    model = DummyModel()
    monkeypatch.setattr(
        fe.torch,
        "load",
        lambda path, map_location=None: {"model_state_dict": model.state_dict()},
    )
    history, returned_model = fe.run_finetune_engine(
        train_dataloader=[object()],
        val_dataloader=[object()],
        test_dataloader=[object()],
        model=model,
        device="cpu",
        hyperparameters=_base_hyperparameters(use_swanlab=True),
        process_batch_fn=lambda *args, **kwargs: None,
        save_dir=str(tmp_path),
    )

    assert returned_model is model
    assert save_model_calls == [str(tmp_path)]
    assert saved_log_epochs == [1, 1]
    assert swanlab_events["init"] == 1
    assert swanlab_events["finish"] == 1
    assert len(swanlab_events["logs"]) == 3
    assert history["best_epoch"] == 1
    assert history["final_test_metrics"]["dice"] == 0.9


def test_run_finetune_engine_num_epochs_zero_logs_epoch_zero(tmp_path, monkeypatch):
    _patch_common_runtime(monkeypatch)

    saved_log_epochs = []

    def fail_train_one_epoch(*args, **kwargs):
        raise AssertionError("_train_one_epoch should not run when num_epochs == 0")

    def fake_evaluate(*args, **kwargs):
        del args, kwargs
        return 0.5, 0.6, 0.7, 8.0

    def fake_save_training_logs(**kwargs):
        saved_log_epochs.append(kwargs["epoch"])
        return {"last_completed_epoch": kwargs["epoch"]}

    monkeypatch.setattr(fe, "_train_one_epoch", fail_train_one_epoch)
    monkeypatch.setattr(fe, "_evaluate", fake_evaluate)
    monkeypatch.setattr(fe, "save_training_logs", fake_save_training_logs)

    history, _ = fe.run_finetune_engine(
        train_dataloader=[object()],
        val_dataloader=[object()],
        test_dataloader=[object()],
        model=DummyModel(),
        device="cpu",
        hyperparameters=_base_hyperparameters(num_epochs=0),
        process_batch_fn=lambda *args, **kwargs: None,
        save_dir=str(tmp_path),
    )

    assert saved_log_epochs == [0]
    assert history["train_loss"] == []
    assert history["final_test_metrics"]["hd95"] == 8.0


def test_run_finetune_engine_hf_load_failure_raises_before_final_eval(tmp_path, monkeypatch):
    _patch_common_runtime(monkeypatch)

    eval_call_count = {"count": 0}

    def fake_train_one_epoch(*args, **kwargs):
        global_step = kwargs["global_step"]
        return 0.3, 0.4, 0.5, 6.0, 0.25, global_step + 1

    def fake_evaluate(*args, **kwargs):
        del args, kwargs
        eval_call_count["count"] += 1
        return 0.2, 0.8, 0.7, 1.2

    class DummyConfig:
        base_model_name_or_path = "fake-base"

    monkeypatch.setattr(fe, "_train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(fe, "_evaluate", fake_evaluate)
    monkeypatch.setattr(fe, "save_model", lambda **kwargs: str(tmp_path / "hf_best"))
    monkeypatch.setattr(fe, "save_training_logs", lambda **kwargs: {"last_completed_epoch": kwargs["epoch"]})
    monkeypatch.setattr(fe.PeftConfig, "from_pretrained", lambda path: DummyConfig())
    monkeypatch.setattr(fe.SamModel, "from_pretrained", lambda path: DummyModel())
    monkeypatch.setattr(fe, "prepare_base_model_for_hf_adapter_loading", lambda base_model, ft_type: base_model)
    monkeypatch.setattr(
        fe.PeftModel,
        "from_pretrained",
        lambda base_model, best_model_path: (_ for _ in ()).throw(ValueError("boom")),
    )

    with pytest.raises(RuntimeError, match="Failed to load Hugging Face PEFT model"):
        fe.run_finetune_engine(
            train_dataloader=[object()],
            val_dataloader=[object()],
            test_dataloader=[object()],
            model=DummyModel(),
            device="cpu",
            hyperparameters=_base_hyperparameters(save_hf_format=True),
            process_batch_fn=lambda *args, **kwargs: None,
            save_dir=str(tmp_path),
        )

    assert eval_call_count["count"] == 1


def test_run_finetune_engine_ddp_non_master_uses_broadcast_best_path(tmp_path, monkeypatch):
    _patch_common_runtime(monkeypatch)

    torch_load_paths = []
    save_model_calls = []
    broadcast_calls = []
    eval_results = iter([
        (0.2, 0.8, 0.7, 1.2),  # val
        (0.1, 0.9, 0.8, 1.0),  # final test
    ])

    def fake_train_one_epoch(*args, **kwargs):
        global_step = kwargs["global_step"]
        return 0.3, 0.4, 0.5, 6.0, 0.25, global_step + 1

    def fake_evaluate(*args, **kwargs):
        del args, kwargs
        return next(eval_results)

    def fake_broadcast_object_list(obj_list, src):
        del src
        broadcast_calls.append(True)
        obj_list[0] = "shared_best.pth"

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(fe, "init_process_group", lambda backend: None)
    monkeypatch.setattr(fe.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(fe, "DDP", FakeDDP)
    monkeypatch.setattr(fe.dist, "is_available", lambda: True)
    monkeypatch.setattr(fe.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(fe.dist, "barrier", lambda: None)
    monkeypatch.setattr(fe.dist, "broadcast_object_list", fake_broadcast_object_list)
    monkeypatch.setattr(fe, "_train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(fe, "_evaluate", fake_evaluate)
    monkeypatch.setattr(fe, "save_model", lambda **kwargs: save_model_calls.append(kwargs))
    monkeypatch.setattr(fe, "save_training_logs", lambda **kwargs: {"last_completed_epoch": kwargs["epoch"]})
    monkeypatch.setattr(
        fe.torch,
        "load",
        lambda path, map_location=None: (torch_load_paths.append((path, map_location)) or {}),
    )

    history, returned_model = fe.run_finetune_engine(
        train_dataloader=[object()],
        val_dataloader=[object()],
        test_dataloader=[object()],
        model=DummyModel(),
        device="cpu",
        hyperparameters=_base_hyperparameters(save_custom_lora=True, use_swanlab=False),
        process_batch_fn=lambda *args, **kwargs: None,
        save_dir=str(tmp_path),
        ddp_info={"world_size": 2},
    )

    assert isinstance(returned_model, FakeDDP)
    assert save_model_calls == []
    assert broadcast_calls == [True]
    assert torch_load_paths == [("shared_best.pth", "cuda:0")]
    assert "final_test_metrics" not in history


def test_run_finetune_engine_amp_startup_enables_gradscaler_when_bf16_unsupported(tmp_path, monkeypatch):
    _patch_common_runtime(monkeypatch)

    scaler_inits = []
    train_seen_scalers = []
    eval_seen_scalers = []

    def fake_grad_scaler(device_type, enabled):
        scaler = FakeGradScaler(device_type=device_type, enabled=enabled)
        scaler_inits.append(scaler)
        return scaler

    def fake_train_one_epoch(*args, **kwargs):
        train_seen_scalers.append(args[6])
        global_step = kwargs["global_step"]
        return 0.3, 0.4, 0.5, 6.0, 0.25, global_step + 1

    def fake_evaluate(*args, **kwargs):
        eval_seen_scalers.append(args[5])
        return 0.2, 0.8, 0.7, 1.2

    monkeypatch.setattr(fe.torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(fe.torch.amp, "GradScaler", fake_grad_scaler)
    monkeypatch.setattr(fe, "_train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(fe, "_evaluate", fake_evaluate)
    monkeypatch.setattr(fe, "save_model", lambda **kwargs: str(tmp_path / "best_model.pth"))
    monkeypatch.setattr(fe, "save_training_logs", lambda **kwargs: {"last_completed_epoch": kwargs["epoch"]})

    model = DummyModel()
    monkeypatch.setattr(
        fe.torch,
        "load",
        lambda path, map_location=None: {"model_state_dict": model.state_dict()},
    )

    history, returned_model = fe.run_finetune_engine(
        train_dataloader=[object()],
        val_dataloader=[object()],
        test_dataloader=[object()],
        model=model,
        device="cpu",
        hyperparameters=_base_hyperparameters(),
        process_batch_fn=lambda *args, **kwargs: None,
        save_dir=str(tmp_path),
    )

    assert returned_model is model
    assert len(scaler_inits) == 1
    assert scaler_inits[0].device_type == "cuda"
    assert scaler_inits[0].enabled is True
    assert train_seen_scalers == [scaler_inits[0]]
    assert eval_seen_scalers == [scaler_inits[0], scaler_inits[0]]
    assert history["final_test_metrics"]["dice"] == 0.8


# ─── H. _select_best_mask ─────────────────────────────────────────

def test_select_best_mask_5d_input_shape_and_selection():
    """5D pred_masks [B,1,3,H,W] + 3D iou_scores [B,1,3] → 正确形状与最高IoU掩膜被选中。"""
    B, H, W = 2, 4, 4
    pred_masks = torch.zeros(B, 1, 3, H, W)
    pred_masks[0, 0, 2] = 5.0   # batch 0: mask index 2 应被选
    pred_masks[1, 0, 1] = 3.0   # batch 1: mask index 1 应被选
    iou_scores = torch.tensor([[[0.1, 0.2, 0.9]], [[0.1, 0.8, 0.3]]])  # [2,1,3]
    result = fe._select_best_mask(pred_masks, iou_scores)
    assert result.shape == (B, 1, H, W)
    assert result[0, 0].mean().item() == pytest.approx(5.0)
    assert result[1, 0].mean().item() == pytest.approx(3.0)


def test_select_best_mask_4d_input_already_squeezed():
    """4D pred_masks [B,3,H,W] + 2D iou_scores [B,3] → 不需要 squeeze 也能正常运行。"""
    pred_masks = torch.zeros(1, 3, 4, 4)
    pred_masks[0, 0] = 7.0   # mask index 0 应被选
    iou_scores = torch.tensor([[0.9, 0.1, 0.2]])   # [1,3]
    result = fe._select_best_mask(pred_masks, iou_scores)
    assert result.shape == (1, 1, 4, 4)
    assert result[0, 0].mean().item() == pytest.approx(7.0)


# ─── I. _sample_points_from_mask ──────────────────────────────────

def test_sample_points_foreground_returns_fg_label():
    """mask 中有前景像素时，sample_from='foreground' 应返回 label=1 的点。"""
    mask = torch.zeros(32, 32)
    mask[10:20, 10:20] = 1.0
    coords, labels = fe._sample_points_from_mask(mask, num_points=1, sample_from='foreground')
    assert coords.shape == (1, 2)
    assert labels.shape == (1,)
    assert labels[0] == 1
    # coords 格式 (x, y)，应在前景区域 [10,20)
    x, y = coords[0]
    assert 10 <= int(x) < 20
    assert 10 <= int(y) < 20


def test_sample_points_tensor_3d_input_handled():
    """3D Tensor 输入应被自动转换并正常采样。"""
    mask = torch.ones(1, 16, 16)  # 全前景
    coords, labels = fe._sample_points_from_mask(mask, num_points=2, sample_from='foreground')
    assert coords.shape == (2, 2)
    assert all(l == 1 for l in labels)


def test_sample_points_empty_fg_falls_back_to_background():
    """全背景 mask 下前景采样 fallback 到背景点（label=0）。"""
    mask = torch.zeros(32, 32)  # 全零，无前景
    coords, labels = fe._sample_points_from_mask(mask, num_points=1, sample_from='foreground')
    assert coords.shape == (1, 2)
    assert labels[0] == 0


def test_sample_points_padding_when_insufficient():
    """请求点数超过可用点数时，不足部分用 [0,0] / label=0 填充。"""
    mask = torch.zeros(4, 4)
    mask[0, 0] = 1.0  # 只有 1 个前景像素
    coords, labels = fe._sample_points_from_mask(mask, num_points=5, sample_from='foreground')
    assert coords.shape == (5, 2)
    assert labels.shape == (5,)


# ─── J. _sample_correction_points ────────────────────────────────

def test_sample_correction_fn_dominant_returns_fg_label():
    """假阴性远多于假阳性时，应采样前景纠正点（label=1）。"""
    import numpy as np
    gt = np.zeros((32, 32), dtype=np.float32)
    gt[5:25, 5:25] = 1.0   # 大前景区域
    pred = np.zeros((32, 32), dtype=np.float32)   # 预测全零 → 大量 FN，无 FP
    coords, labels = fe._sample_correction_points(pred, gt, num_points=1)
    assert labels[0] == 1


def test_sample_correction_fp_dominant_returns_bg_label():
    """假阳性远多于假阴性时，应采样背景纠正点（label=0）。"""
    import numpy as np
    gt = np.zeros((32, 32), dtype=np.float32)
    gt[0, 0] = 1.0   # 极小前景
    pred = torch.full((1, 32, 32), 10.0)   # sigmoid → 全 1，产生大量 FP
    coords, labels = fe._sample_correction_points(pred, gt, num_points=1)
    assert labels[0] == 0


def test_sample_correction_no_errors_pads_with_zeros():
    """预测与 GT 完全一致（无错误区域）时，coords 应以零填充。"""
    import numpy as np
    gt = np.ones((8, 8), dtype=np.float32)
    pred = np.ones((8, 8), dtype=np.float32)   # 完全正确
    coords, labels = fe._sample_correction_points(pred, gt, num_points=2)
    assert coords.shape == (2, 2)
    assert labels.shape == (2,)


# ─── K. severstal_get_offset ──────────────────────────────────────

def test_severstal_get_offset_bounds_and_ordering():
    """severstal_get_offset 返回 4 个整数，且 start < end，均在 pred 256×256 空间内。"""
    result = fe.severstal_get_offset()
    assert len(result) == 4
    crop_y_start, crop_x_start, crop_y_end, crop_x_end = result
    assert 0 <= crop_y_start < crop_y_end <= 256
    assert 0 <= crop_x_start < crop_x_end <= 256


def test_severstal_get_offset_is_deterministic():
    """两次调用结果相同（纯函数）。"""
    assert fe.severstal_get_offset() == fe.severstal_get_offset()
