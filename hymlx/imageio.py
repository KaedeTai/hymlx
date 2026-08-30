"""里程碑 8：latent <-> token 的接口。

混元沒有把 VAE latent 直接切 patch 丟進 transformer，而是各接了半個 UNet：
`patch_embed`（UNetDown）把 32 通道的 latent 變成 4096 維的 token，
`final_layer`（UNetUp）再變回去。兩邊的 ResBlock 都吃時間步 embedding，用
adaptive GroupNorm（`h = norm(h) * (1 + scale) + shift`）把 timestep 注入。

patch_size = 1，所以「down」跟「up」其實都不改解析度：token 數 = latent 的 H x W。
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def timestep_embedding(t: mx.array, dim: int = 256, max_period: float = 10000.0) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t.reshape(-1, 1).astype(mx.float32) * freqs.reshape(1, -1)
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_size: int = 256, max_period: float = 10000.0,
                 out_size: int | None = None):
        super().__init__()
        self.freq_size = freq_size
        self.max_period = max_period
        self.fc1 = nn.Linear(freq_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size or hidden_size)

    def __call__(self, t: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(timestep_embedding(t, self.freq_size, self.max_period))))


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, ch, eps=1e-5, pytorch_compatible=True)


class ResBlock(nn.Module):
    """帶 adaptive GroupNorm 的殘差塊。輸入輸出都是 NHWC。"""

    def __init__(self, cin: int, emb_ch: int, cout: int):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.in_norm = _gn(cin)
        self.in_conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb_lin = nn.Linear(emb_ch, 2 * cout)
        self.out_norm = _gn(cout)
        self.out_conv = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = None if cin == cout else nn.Conv2d(cin, cout, 1)

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        h = self.in_conv(nn.silu(self.in_norm(x)))
        e = self.emb_lin(nn.silu(emb))                    # (B, 2*cout)
        scale, shift = mx.split(e, 2, axis=-1)
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]
        h = self.out_norm(h) * (1.0 + scale) + shift
        h = self.out_conv(nn.silu(h))
        return (x if self.skip is None else self.skip(x)) + h


class UNetDown(nn.Module):
    """latent (B, 32, H, W) -> tokens (B, H*W, 4096)。"""

    def __init__(self, cin: int = 32, emb_ch: int = 4096, hidden: int = 1024, cout: int = 4096):
        super().__init__()
        self.conv_in = nn.Conv2d(cin, hidden, 3, padding=1)
        self.block = ResBlock(hidden, emb_ch, cout)

    def __call__(self, x: mx.array, emb: mx.array) -> Tuple[mx.array, int, int]:
        x = x.transpose(0, 2, 3, 1)                       # NCHW -> NHWC
        h = self.block(self.conv_in(x), emb)
        B, th, tw, C = h.shape
        return h.reshape(B, th * tw, C), th, tw


class UNetUp(nn.Module):
    """tokens (B, H*W, 4096) -> latent (B, 32, H, W)。"""

    def __init__(self, cin: int = 4096, emb_ch: int = 4096, hidden: int = 1024, cout: int = 32):
        super().__init__()
        self.block = ResBlock(cin, emb_ch, hidden)
        self.out_norm = _gn(hidden)
        self.out_conv = nn.Conv2d(hidden, cout, 3, padding=1)

    def __call__(self, x: mx.array, emb: mx.array, token_h: int, token_w: int) -> mx.array:
        B, _, C = x.shape
        h = self.block(x.reshape(B, token_h, token_w, C), emb)
        h = self.out_conv(nn.silu(self.out_norm(h)))
        return h.transpose(0, 3, 1, 2)                    # NHWC -> NCHW


# ----------------------------------------------------------------------------
# 權重載入：torch conv 是 (out, in, kH, kW)，MLX 是 (out, kH, kW, in)。
# ----------------------------------------------------------------------------

def _conv(mod, w: Dict[str, np.ndarray], p: str, dtype) -> int:
    mod.weight = mx.array(w[p + ".weight"]).transpose(0, 2, 3, 1).astype(dtype)
    mod.bias = mx.array(w[p + ".bias"]).astype(dtype)
    return 2


def _plain(mod, w: Dict[str, np.ndarray], p: str, dtype) -> int:
    mod.weight = mx.array(w[p + ".weight"]).astype(dtype)
    mod.bias = mx.array(w[p + ".bias"]).astype(dtype)
    return 2


def load_timestep_embedder(m: TimestepEmbedder, w, prefix: str, dtype=mx.float32) -> int:
    return _plain(m.fc1, w, prefix + "mlp.0", dtype) + _plain(m.fc2, w, prefix + "mlp.2", dtype)


def _load_resblock(b: ResBlock, w, p: str, dtype) -> int:
    n = _plain(b.in_norm, w, p + "in_layers.0", dtype)
    n += _conv(b.in_conv, w, p + "in_layers.2", dtype)
    n += _plain(b.emb_lin, w, p + "emb_layers.1", dtype)
    n += _plain(b.out_norm, w, p + "out_layers.0", dtype)
    n += _conv(b.out_conv, w, p + "out_layers.3", dtype)
    if b.skip is not None:
        n += _conv(b.skip, w, p + "skip_connection", dtype)
    return n


def load_unet_down(m: UNetDown, w, prefix: str = "patch_embed.", dtype=mx.float32) -> int:
    return _conv(m.conv_in, w, prefix + "model.0", dtype) + \
        _load_resblock(m.block, w, prefix + "model.1.", dtype)


def load_unet_up(m: UNetUp, w, prefix: str = "final_layer.", dtype=mx.float32) -> int:
    n = _load_resblock(m.block, w, prefix + "model.0.", dtype)
    n += _plain(m.out_norm, w, prefix + "model.1.0", dtype)
    n += _conv(m.out_conv, w, prefix + "model.1.2", dtype)
    return n
