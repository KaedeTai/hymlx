"""里程碑 5：2D 圖像 RoPE 與混合注意力遮罩。

文字 token 走因果注意力，一張圖像內部的 token 互相全連通。圖像 token 的位置
不是一維遞增，而是排在一個二維格點上，格點中心對齊「同樣長度的文字」會落在的
平均位置——這樣圖像佔的位置區間跟 h*w 個文字 token 一樣長，接在後面的文字不必
知道前面是圖還是字。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

# (區段, (高, 寬))；高為 None 代表這段是圖像的控制 token（<boi>、尺寸、比例……），
# 位置照文字排。
ImageInfo = Tuple[slice, Tuple[Optional[int], Optional[int]]]


def rope_positions(seq_len: int, image_infos: Optional[Sequence[ImageInfo]] = None) -> np.ndarray:
    """回傳 (seq_len, 2) 的 (y, x) 整數位置。"""
    ys: List[np.ndarray] = []
    xs: List[np.ndarray] = []
    last = 0
    for sec, (h, w) in image_infos or []:
        L = sec.start
        if last < L:
            ys.append(np.arange(last, L, dtype=np.float32))
            xs.append(np.arange(last, L, dtype=np.float32))
        elif h is None:
            # 交錯資料裡這些 token 的位置跟前面重疊，照文字排就好。
            ys.append(np.arange(sec.start, sec.stop, dtype=np.float32))
            xs.append(np.arange(sec.start, sec.stop, dtype=np.float32))
            continue
        # 讓格點的重心落在 L + (w*h)/2，也就是同長度文字的中點。
        beta_y = L + (w * h - h) / 2
        beta_x = L + (w * h - w) / 2
        gy = np.repeat(beta_y + np.arange(h, dtype=np.float32), w)
        gx = np.tile(beta_x + np.arange(w, dtype=np.float32), h)
        ys.append(gy)
        xs.append(gx)
        last = L + w * h
    ys.append(np.arange(last, seq_len, dtype=np.float32))
    xs.append(np.arange(last, seq_len, dtype=np.float32))
    y = np.concatenate(ys).astype(np.int64)[:seq_len]
    x = np.concatenate(xs).astype(np.int64)[:seq_len]
    return np.stack([y, x], axis=1)


def build_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: Optional[Sequence[ImageInfo]] = None,
    base: float = 10000.0,
    base_rescale_factor: float = 1.0,
    dtype=mx.float32,
) -> Tuple[mx.array, mx.array]:
    """回傳 (cos, sin)，形狀都是 (seq_len, n_elem)。

    頻率分成兩半交錯：偶數對給 y、奇數對給 x，所以每個頻率 y、x 各有一份。
    """
    assert n_elem % 4 == 0, f"n_elem 要能被 4 整除，拿到 {n_elem}"
    if base_rescale_factor != 1.0:
        base = base * base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (base ** (np.arange(0, n_elem, 2, dtype=np.float32) / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)

    pos = rope_positions(seq_len, image_infos).astype(np.float32)  # (S, 2)
    idx = (pos[:, None, :] * theta).reshape(seq_len, n_elem // 2)
    idx = np.concatenate([idx, idx], axis=1)
    return mx.array(np.cos(idx)).astype(dtype), mx.array(np.sin(idx)).astype(dtype)


def build_batch_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: Optional[Sequence[Optional[Sequence[ImageInfo]]]] = None,
    **kw,
) -> Tuple[mx.array, mx.array]:
    infos = list(image_infos) if image_infos is not None else [None]
    pairs = [build_2d_rope(seq_len, n_elem, image_infos=i, **kw) for i in infos]
    return mx.stack([c for c, _ in pairs]), mx.stack([s for _, s in pairs])


def rotate_half(x: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def apply_rope(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, unsqueeze_dim: int = 1):
    """q、k 是 (B, heads, S, D)，cos/sin 是 (B, S, D)。"""
    cos = mx.expand_dims(cos, unsqueeze_dim)
    sin = mx.expand_dims(sin, unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def build_attention_mask(
    seq_len: int,
    full_attn_slices: Sequence[slice] = (),
    batch: int = 1,
) -> mx.array:
    """(B, 1, S, S) 的布林遮罩：下三角，加上每張圖像自己的方塊全開。"""
    m = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    for s in full_attn_slices:
        m[s, s] = True
    return mx.array(np.broadcast_to(m, (batch, 1, seq_len, seq_len)).copy())


def additive_mask(mask: mx.array, dtype=mx.float32) -> mx.array:
    """布林遮罩轉成給 softmax 前相加的 0 / -inf。"""
    neg = mx.array(-np.inf, dtype=dtype)
    return mx.where(mask, mx.array(0.0, dtype=dtype), neg)
