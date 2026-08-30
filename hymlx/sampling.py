"""里程碑 11：flow matching 取樣器。

跟 mdream 那顆 HiDream 的差別要記住：HiDream 是 CONST 排程加 8 倍噪聲縮放，
`denoised = x - v * sigma`；混元這邊是**乾淨的 flow matching**——模型直接輸出速度 v，
Euler 就是 `x += v * (sigma_next - sigma)`，sigma 從 1 走到 0，dt 是負的。

sigma 的時間位移是 SD3 那條 `sigma' = shift * sigma / (1 + (shift - 1) * sigma)`，
`shift = 3.0`。餵給模型的 timestep 是 `sigma * 1000`。

CFG 是標準式：`pred = pred_uncond + scale * (pred_cond - pred_uncond)`。
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import mlx.core as mx
import numpy as np


def sd3_shift(sigmas: np.ndarray, shift: float) -> np.ndarray:
    return (shift * sigmas) / (1.0 + (shift - 1.0) * sigmas)


def sigma_schedule(num_steps: int, shift: float = 3.0,
                   num_train_timesteps: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (sigmas[N+1], timesteps[N])。sigmas 由 1 遞減到 0。"""
    # 全程 float32：官方用 torch.linspace（float32），差一個 dtype 就差 1e-4 個 timestep
    sigmas = np.linspace(np.float32(1.0), np.float32(0.0), num_steps + 1, dtype=np.float32)
    if shift != 1.0:
        sigmas = sd3_shift(sigmas, np.float32(shift)).astype(np.float32)
    return sigmas, (sigmas[:-1] * np.float32(num_train_timesteps)).astype(np.float32)


def cfg(pred_cond: mx.array, pred_uncond: mx.array, scale: float) -> mx.array:
    return pred_uncond + scale * (pred_cond - pred_uncond)


def euler_sample(
    denoiser: Callable[[mx.array, float, int], mx.array],
    latents: mx.array,
    num_steps: int = 50,
    shift: float = 3.0,
    callback: Optional[Callable[[int, mx.array], None]] = None,
) -> mx.array:
    """`denoiser(x, timestep, step_index) -> v`（CFG 已經合併過）。"""
    sigmas, timesteps = sigma_schedule(num_steps, shift)
    x = latents.astype(mx.float32)
    for i in range(num_steps):
        v = denoiser(x, float(timesteps[i]), i).astype(mx.float32)
        dt = float(sigmas[i + 1] - sigmas[i])          # 負的
        x = x + v * dt
        mx.eval(x)
        if callback is not None:
            callback(i, x)
    return x


def init_latents(batch: int, channels: int, height: int, width: int,
                 downsample: int = 16, seed: int = 0) -> mx.array:
    """官方用 torch 的 randn；為了能對答案，種子相同時這裡也走 numpy 的標準常態。"""
    rs = np.random.RandomState(seed)
    return mx.array(rs.randn(batch, channels, height // downsample,
                             width // downsample).astype(np.float32))
