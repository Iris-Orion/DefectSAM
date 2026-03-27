"""
MFU (Model FLOPs Utilization) 估算器，针对 SAM 图像编码器 (ViT-B/L/H)。

理论基础：
  - 每个 Transformer block 的 FLOPs = 线性层 + 注意力
  - 线性层：QKV + 输出投影 + MLP = (6 + 2 + 16) * d² * T = 24 * d² * T
  - 注意力：
      - 窗口注意力：2 * T_w * T * d   (T_w = window_size²)
      - 全局注意力：2 * T² * d
  - 前向 + 反向 ≈ 3 × 前向 (反向是前向的 2 倍)

参考：PaLM Appendix B (https://arxiv.org/abs/2204.02311)
"""

import time


# SAM 各版本 ViT 架构参数
_SAM_CONFIGS = {
    'sam_base': dict(embed_dim=768,  depth=12, num_heads=12,
                  window_size=14, global_attn_indices=[2, 5, 8, 11]),
    'sam_large': dict(embed_dim=1024, depth=24, num_heads=16,
                  window_size=14, global_attn_indices=[5, 11, 17, 23]),
    'sam_huge': dict(embed_dim=1280, depth=32, num_heads=16,
                  window_size=14, global_attn_indices=[7, 15, 23, 31]),
}

# GPU BF16/FP16 Tensor Core 峰值算力 (单位：FLOPS)
_GPU_PEAK_FLOPS = {
    'rtx5090': 838e12,   # RTX 5090 BF16 tensor core ~838 TFLOPS
    'rtx4090': 330e12,   # RTX 4090 BF16
    'a100':    312e12,   # A100 BF16
    'h100':    989e12,   # H100 BF16 SXM
}


class SAMMFUEstimator:
    """
    估算 SAM 图像编码器训练时的 MFU（Model FLOPs Utilization）。

    用法示例：
        estimator = SAMMFUEstimator(sam_type='vit_b', gpu_type='rtx5090')
        mfu = estimator.estimate(batch_size=4, dt=1.2)   # dt 单位秒
        print(f"MFU: {mfu*100:.2f}%")
    """

    def __init__(self,
                 sam_type: str = 'sam_base',
                 image_size: int = 1024,
                 patch_size: int = 16,
                 gpu_type: str = 'rtx5090',
                 peak_flops: float = None):
        """
        Args:
            sam_type:    SAM 版本，'vit_b' | 'vit_l' | 'vit_h'
            image_size:  输入图像边长，默认 1024
            patch_size:  patch 大小，默认 16
            gpu_type:    GPU 型号，用于查表峰值算力；
                         也可直接传 peak_flops 覆盖
            peak_flops:  手动指定 GPU 峰值算力（FLOPS），覆盖 gpu_type
        """
        assert sam_type in _SAM_CONFIGS, \
            f"sam_type 必须是 {list(_SAM_CONFIGS.keys())} 之一"

        cfg = _SAM_CONFIGS[sam_type]
        self.sam_type   = sam_type
        self.d          = cfg['embed_dim']
        self.L          = cfg['depth']
        self.T          = (image_size // patch_size) ** 2   # 总 patch 数
        self.T_w        = cfg['window_size'] ** 2            # 每个窗口的 token 数
        self.n_global   = len(cfg['global_attn_indices'])

        self.peak_flops = peak_flops if peak_flops is not None \
                          else _GPU_PEAK_FLOPS.get(gpu_type, 838e12)

        self._flops_per_sample = self._compute_fwdbwd_flops()

    def _compute_fwdbwd_flops(self) -> float:
        """计算单张图像一次前向+反向的理论 FLOPs。"""
        d, L, T, T_w = self.d, self.L, self.T, self.T_w
        n_global   = self.n_global
        n_windowed = L - n_global

        # 每个 block 的线性层 FLOPs：QKV(6d²T) + 输出投影(2d²T) + MLP(16d²T)
        matmul_per_block = 24 * d**2 * T

        # 注意力 FLOPs
        attn_windowed = 2 * T_w * T * d   # 窗口注意力（大多数层）
        attn_global   = 2 * T**2  * d     # 全局注意力（少数层）

        flops_fwd = (L * matmul_per_block
                     + n_windowed * attn_windowed
                     + n_global   * attn_global)

        # 前向 + 反向 ≈ 3 × 前向
        return 3.0 * flops_fwd

    def estimate(self, batch_size: int, dt: float) -> float:
        """
        根据实际训练耗时估算 MFU。

        Args:
            batch_size: 该 step 处理的图像数（micro-batch × grad_accum_steps）
            dt:         该 step 的实际耗时，单位秒

        Returns:
            MFU（0~1 的浮点数），>1 表示估算 FLOP 数偏低或 GPU 超频
        """
        flops_achieved = self._flops_per_sample * batch_size / dt
        return flops_achieved / self.peak_flops

    def summary(self) -> str:
        """打印估算器基本参数。"""
        return (f"SAMMFUEstimator | {self.sam_type.upper()} | "
                f"T={self.T} patches | "
                f"FLOPs/sample(fwd+bwd)={self._flops_per_sample/1e12:.2f}T | "
                f"Peak={self.peak_flops/1e12:.0f}TFLOPS")


class MFUTracker:
    """
    在训练循环中对 MFU 做指数滑动平均（EMA），
    并在每个 batch 后提供 formatted 输出字符串。

    用法：
        tracker = MFUTracker(estimator, batch_size=4, ema_alpha=0.9)
        t0 = time.time()
        ... # forward + backward
        tracker.update(time.time() - t0)
        print(tracker.status())   # "MFU: 32.14%"
    """

    def __init__(self,
                 estimator: SAMMFUEstimator,
                 batch_size: int,
                 ema_alpha: float = 0.9):
        self.estimator  = estimator
        self.batch_size = batch_size
        self.alpha      = ema_alpha
        self._mfu_ema   = -1.0      # -1 表示尚未有值

    def update(self, dt: float):
        """用本次 step 的耗时更新 EMA。"""
        mfu = self.estimator.estimate(self.batch_size, dt)
        if self._mfu_ema < 0:
            self._mfu_ema = mfu
        else:
            self._mfu_ema = self.alpha * self._mfu_ema + (1 - self.alpha) * mfu

    @property
    def mfu(self) -> float:
        """当前 EMA MFU（-1 表示尚未更新过）。"""
        return self._mfu_ema

    def status(self) -> str:
        if self._mfu_ema < 0:
            return "MFU: N/A"
        return f"MFU: {self._mfu_ema * 100:.2f}%"
