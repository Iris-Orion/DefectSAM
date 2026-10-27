"""
MFU (Model FLOPs Utilization) estimator for the SAM image encoder (ViT-B/L/H).

Per transformer block, FLOPs = linear layers + attention:
  - linear: QKV + output projection + MLP = (6 + 2 + 16) * d^2 * T = 24 * d^2 * T
  - windowed attention: 2 * T_w * T * d   (T_w = window_size^2)
  - global attention:   2 * T^2 * d
Forward + backward ~= 3 x forward (backward ~= 2 x forward).

Reference: PaLM Appendix B (https://arxiv.org/abs/2204.02311)
"""

import time


_SAM_CONFIGS = {
    'sam_base': dict(embed_dim=768,  depth=12, num_heads=12,
                  window_size=14, global_attn_indices=[2, 5, 8, 11]),
    'sam_large': dict(embed_dim=1024, depth=24, num_heads=16,
                  window_size=14, global_attn_indices=[5, 11, 17, 23]),
    'sam_huge': dict(embed_dim=1280, depth=32, num_heads=16,
                  window_size=14, global_attn_indices=[7, 15, 23, 31]),
}

# Peak BF16 tensor-core throughput, in FLOPS.
_GPU_PEAK_FLOPS = {
    'rtx5090': 838e12,
    'rtx4090': 330e12,
    'a100':    312e12,
    'h100':    989e12,   # SXM
}


class SAMMFUEstimator:
    """Estimate training MFU of the SAM image encoder from measured step time."""

    def __init__(self,
                 sam_type: str = 'sam_base',
                 image_size: int = 1024,
                 patch_size: int = 16,
                 gpu_type: str = 'rtx5090',
                 peak_flops: float = None):
        """
        Args:
            sam_type:   one of the _SAM_CONFIGS keys.
            gpu_type:   key into _GPU_PEAK_FLOPS; unknown types fall back to the RTX 5090 peak.
            peak_flops: peak GPU throughput in FLOPS; overrides gpu_type.
        """
        assert sam_type in _SAM_CONFIGS, \
            f"sam_type 必须是 {list(_SAM_CONFIGS.keys())} 之一"

        cfg = _SAM_CONFIGS[sam_type]
        self.sam_type   = sam_type
        self.d          = cfg['embed_dim']
        self.L          = cfg['depth']
        self.T          = (image_size // patch_size) ** 2   # total patch tokens
        self.T_w        = cfg['window_size'] ** 2            # tokens per attention window
        self.n_global   = len(cfg['global_attn_indices'])

        self.peak_flops = peak_flops if peak_flops is not None \
                          else _GPU_PEAK_FLOPS.get(gpu_type, 838e12)

        self._flops_per_sample = self._compute_fwdbwd_flops()

    def _compute_fwdbwd_flops(self) -> float:
        """Theoretical forward + backward FLOPs for one image."""
        d, L, T, T_w = self.d, self.L, self.T, self.T_w
        n_global   = self.n_global
        n_windowed = L - n_global

        matmul_per_block = 24 * d**2 * T

        attn_windowed = 2 * T_w * T * d
        attn_global   = 2 * T**2  * d

        flops_fwd = (L * matmul_per_block
                     + n_windowed * attn_windowed
                     + n_global   * attn_global)

        return 3.0 * flops_fwd

    def estimate(self, batch_size: int, dt: float) -> float:
        """
        Args:
            batch_size: images processed in the step (micro-batch x grad_accum_steps).
            dt:         wall-clock time of the step, in seconds.

        Returns:
            MFU in [0, 1]; > 1 means the FLOP count is underestimated or the GPU exceeds the tabulated peak.
        """
        flops_achieved = self._flops_per_sample * batch_size / dt
        return flops_achieved / self.peak_flops

    def summary(self) -> str:
        return (f"SAMMFUEstimator | {self.sam_type.upper()} | "
                f"T={self.T} patches | "
                f"FLOPs/sample(fwd+bwd)={self._flops_per_sample/1e12:.2f}T | "
                f"Peak={self.peak_flops/1e12:.0f}TFLOPS")


class MFUTracker:
    """Exponential moving average of MFU over training steps."""

    def __init__(self,
                 estimator: SAMMFUEstimator,
                 batch_size: int,
                 ema_alpha: float = 0.9):
        self.estimator  = estimator
        self.batch_size = batch_size
        self.alpha      = ema_alpha
        self._mfu_ema   = -1.0

    def update(self, dt: float):
        mfu = self.estimator.estimate(self.batch_size, dt)
        if self._mfu_ema < 0:
            self._mfu_ema = mfu
        else:
            self._mfu_ema = self.alpha * self._mfu_ema + (1 - self.alpha) * mfu

    @property
    def mfu(self) -> float:
        """Current EMA MFU, or -1 before the first update()."""
        return self._mfu_ema

    def status(self) -> str:
        if self._mfu_ema < 0:
            return "MFU: N/A"
        return f"MFU: {self._mfu_ema * 100:.2f}%"
