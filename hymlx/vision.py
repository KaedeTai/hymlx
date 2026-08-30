"""里程碑 7：SigLIP2 (so400m/16, NaFlex) 視覺塔 + aligner。

參考圖走的是這條路：影像先被切成 16x16 的 patch 攤平成向量（不是卷積，是一顆
Linear），位置編碼是把 16x16 的表格用**帶抗鋸齒的雙線性插值**縮放到這張圖實際的
patch 格點大小。那個 antialias 是整個移植裡唯一沒有 MLX 對應算子的東西，所以這裡
自己算插值權重矩陣——反正只有 16 -> (h, w)，一次性的小矩陣。

`vision_model.head`（attention pooling）沒有被用到：混元只取 `last_hidden_state`
餵給 aligner，所以這裡不移植那顆頭。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    num_patches: int = 256
    patch_size: int = 16
    layer_norm_eps: float = 1e-6

    @classmethod
    def from_json(cls, cfg: Dict) -> "VisionConfig":
        keys = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in cfg.items() if k in keys})


# ----------------------------------------------------------------------------
# 抗鋸齒雙線性插值：照 aten/src/ATen/native/UpSampleKernel.cpp 的 _compute_weights_aa
# ----------------------------------------------------------------------------

def aa_weights(in_size: int, out_size: int) -> np.ndarray:
    """回傳 (out_size, in_size) 的權重矩陣，每列和為 1。"""
    scale = in_size / out_size
    support = scale if scale >= 1.0 else 1.0
    inv = 1.0 / support
    W = np.zeros((out_size, in_size), dtype=np.float64)
    for i in range(out_size):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        for j in range(xmin, xmax):
            t = abs((j + 0.5 - center) * inv)
            if t < 1.0:
                W[i, j] = 1.0 - t
        s = W[i].sum()
        if s:
            W[i] /= s
    return W.astype(np.float32)


def resize_pos_embed(pos: mx.array, h: int, w: int) -> mx.array:
    """pos 是 (P, P, D)；回傳 (h*w, D)。"""
    P = pos.shape[0]
    wy = mx.array(aa_weights(P, h))
    wx = mx.array(aa_weights(P, w))
    out = mx.einsum("ia,jb,abd->ijd", wy, wx, pos)
    return out.reshape(h * w, -1)


class Embeddings(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = int(cfg.num_patches ** 0.5)
        self.patch_embedding = nn.Linear(cfg.num_channels * cfg.patch_size ** 2, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.num_patches, cfg.hidden_size)

    def __call__(self, pixel_values: mx.array, spatial_shapes) -> mx.array:
        h = self.patch_embedding(pixel_values)
        pos = self.position_embedding.weight.reshape(self.grid, self.grid, -1)
        L = pixel_values.shape[1]
        rows = []
        for (ph, pw) in spatial_shapes:
            r = resize_pos_embed(pos, int(ph), int(pw))
            pad = L - r.shape[0]
            if pad > 0:                       # 官方用第一個 patch 的位置編碼填滿尾巴
                r = mx.concatenate([r, mx.broadcast_to(r[:1], (pad, r.shape[1]))], axis=0)
            rows.append(r[:L])
        return h + mx.stack(rows)


class VitAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.nh = cfg.num_attention_heads
        self.hd = cfg.hidden_size // self.nh
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, S, _ = x.shape
        def split(t):
            return t.reshape(B, S, self.nh, self.hd).transpose(0, 2, 1, 3)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, S, self.nh * self.hd))


class VitLayer(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        e = cfg.layer_norm_eps
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=e)
        self.self_attn = VitAttention(cfg)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=e)
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        # hidden_act 是 gelu_pytorch_tanh，不是精確的 erf 版本
        return x + self.fc2(nn.gelu_approx(self.fc1(self.layer_norm2(x))))


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = Embeddings(cfg)
        self.layers = [VitLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def __call__(self, pixel_values: mx.array, spatial_shapes,
                 attention_mask: Optional[mx.array] = None) -> mx.array:
        mask = None
        if attention_mask is not None:
            # (B, S) 的 1/0 -> (B, 1, 1, S) 的 0 / -inf
            m = attention_mask.astype(mx.bool_)[:, None, None, :]
            mask = mx.where(m, mx.array(0.0, dtype=mx.float32),
                            mx.array(-3.4028235e38, dtype=mx.float32))
        h = self.embeddings(pixel_values, spatial_shapes)
        for lyr in self.layers:
            h = lyr(h, mask)
        return self.post_layernorm(h)


class Aligner(nn.Module):
    """mlp_gelu, depth 2：Linear(1152, 4096) -> GELU(精確版) -> Linear(4096, 4096)"""

    def __init__(self, input_dim: int = 1152, n_embed: int = 4096, depth: int = 2):
        super().__init__()
        self.fcs = [nn.Linear(input_dim, n_embed)] + \
                   [nn.Linear(n_embed, n_embed) for _ in range(depth - 1)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fcs[0](x)
        for fc in self.fcs[1:]:
            x = fc(nn.gelu(x))
        return x


# ----------------------------------------------------------------------------
# 權重載入
# ----------------------------------------------------------------------------

def _lin(mod, w, p, dtype):
    mod.weight = mx.array(w[p + ".weight"]).astype(dtype)
    mod.bias = mx.array(w[p + ".bias"]).astype(dtype)
    return 2


def load_vision(tower: VisionTower, w: Dict[str, np.ndarray],
                prefix: str = "vision_model.", dtype=mx.float32) -> int:
    n = _lin(tower.embeddings.patch_embedding, w, prefix + "embeddings.patch_embedding", dtype)
    tower.embeddings.position_embedding.weight = \
        mx.array(w[prefix + "embeddings.position_embedding.weight"]).astype(dtype)
    n += 1
    for i, lyr in enumerate(tower.layers):
        p = f"{prefix}encoder.layers.{i}."
        n += _lin(lyr.layer_norm1, w, p + "layer_norm1", dtype)
        n += _lin(lyr.layer_norm2, w, p + "layer_norm2", dtype)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            n += _lin(getattr(lyr.self_attn, name), w, p + "self_attn." + name, dtype)
        n += _lin(lyr.fc1, w, p + "mlp.fc1", dtype)
        n += _lin(lyr.fc2, w, p + "mlp.fc2", dtype)
    n += _lin(tower.post_layernorm, w, prefix + "post_layernorm", dtype)
    return n


def load_aligner(a: Aligner, w: Dict[str, np.ndarray],
                 prefix: str = "vision_aligner.layers.", dtype=mx.float32) -> int:
    n = 0
    for i, fc in enumerate(a.fcs):
        n += _lin(fc, w, f"{prefix}{i * 2}", dtype)
    return n
