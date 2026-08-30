"""里程碑 6：混元的 32 層 decoder。

跟一般 LLM 的差別有四個，每一個都值得寫下來，因為每一個都是 port 常見的錯法：

1. **qkv 是打包成一條的**，而且切法不是 [Q|K|V] 三段，是照
   `(n_kv_heads, n_kv_groups + 2, head_dim)` 交錯——每個 KV head 後面跟著它自己
   那組 Q head，再接一個 K 一個 V。
2. **RoPE 在 qk_norm 之前。** 大多數實作反過來。順序不同結果不同。
3. **MoE 路由是 softmax → top-k → 除以選中機率之和**，不是 top-k 之後再 softmax。
4. **共享專家是加上去的**，跟被選中的 8 個專家並聯，不佔 top-k 名額。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .rope import apply_rope


def _per_layer(v, i):
    return v[i] if isinstance(v, (list, tuple)) else v


@dataclass
class TextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_head_dim: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    moe_intermediate_size: Any
    moe_topk: Any
    num_experts: int
    num_shared_expert: Any
    use_mixed_mlp_moe: bool = True
    moe_layer_num_skipped: int = 0
    use_qk_norm: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"

    @classmethod
    def from_json(cls, cfg: Dict[str, Any]) -> "TextConfig":
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in keys})


class Attention(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.groups = self.n_heads // self.n_kv
        self.hd = cfg.attention_head_dim
        self.scale = self.hd ** -0.5
        self.use_qk_norm = cfg.use_qk_norm
        self.qkv_proj = nn.Linear(cfg.hidden_size, (self.n_heads + 2 * self.n_kv) * self.hd,
                                  bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.hd, cfg.hidden_size, bias=cfg.attention_bias)
        if cfg.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.hd, eps=cfg.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.hd, eps=cfg.rms_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        B, S, _ = x.shape
        g, hd = self.groups, self.hd
        qkv = self.qkv_proj(x).reshape(B, S, self.n_kv, g + 2, hd)
        q = qkv[:, :, :, :g, :].reshape(B, S, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = qkv[:, :, :, g:g + 1, :].reshape(B, S, self.n_kv, hd).transpose(0, 2, 1, 3)
        v = qkv[:, :, :, g + 1:, :].reshape(B, S, self.n_kv, hd).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos, sin)
        if self.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, S, self.n_heads * hd)
        return self.o_proj(o)


class MLP(nn.Module):
    """SwiGLU，但 gate 與 up 是同一顆權重的兩半：**前半是 up，後半才是 gate**。"""

    def __init__(self, hidden: int, inter: int, bias: bool = False):
        super().__init__()
        self.gate_and_up_proj = nn.Linear(hidden, 2 * inter, bias=bias)
        self.down_proj = nn.Linear(inter, hidden, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        up, gate = mx.split(self.gate_and_up_proj(x), 2, axis=-1)
        return self.down_proj(up * nn.silu(gate))


class SwitchMLP(nn.Module):
    """64 顆專家疊成一張權重，用 gather_mm 只算被選中的那 8 顆。"""

    def __init__(self, hidden: int, inter: int, n_experts: int):
        super().__init__()
        self.up_w = mx.zeros((n_experts, inter, hidden))
        self.gate_w = mx.zeros((n_experts, inter, hidden))
        self.down_w = mx.zeros((n_experts, hidden, inter))

    def __call__(self, x: mx.array, inds: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))                       # (B, S, 1, 1, D)
        up = mx.gather_mm(x, self.up_w.swapaxes(-1, -2), rhs_indices=inds)
        gate = mx.gather_mm(x, self.gate_w.swapaxes(-1, -2), rhs_indices=inds)
        y = mx.gather_mm(up * nn.silu(gate), self.down_w.swapaxes(-1, -2), rhs_indices=inds)
        return y.squeeze(-2)                                  # (B, S, k, D)


class Gate(nn.Module):
    def __init__(self, hidden: int, n_experts: int):
        super().__init__()
        self.wg = nn.Linear(hidden, n_experts, bias=False)


class MoE(nn.Module):
    def __init__(self, cfg: TextConfig, layer_idx: int):
        super().__init__()
        inter = _per_layer(cfg.moe_intermediate_size, layer_idx)
        self.top_k = _per_layer(cfg.moe_topk, layer_idx)
        self.use_shared = cfg.use_mixed_mlp_moe
        self.gate = Gate(cfg.hidden_size, cfg.num_experts)
        self.experts = SwitchMLP(cfg.hidden_size, inter, cfg.num_experts)
        if self.use_shared:
            n_sh = _per_layer(cfg.num_shared_expert, layer_idx)
            self.shared_mlp = MLP(cfg.hidden_size, inter * n_sh, bias=cfg.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        # 路由永遠在 fp32 算：官方的 wg 就是 fp32，且 softmax 之後還要再除一次。
        logits = self.gate.wg(x.astype(mx.float32))
        g = mx.softmax(logits, axis=-1, precise=True)
        inds = mx.argpartition(-g, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        w = mx.take_along_axis(g, inds, axis=-1)
        w = w / mx.maximum(w.sum(axis=-1, keepdims=True), 1e-8)
        y = self.experts(x, inds)
        y = (y * mx.expand_dims(w.astype(y.dtype), -1)).sum(axis=-2)
        return y + self.shared_mlp(x) if self.use_shared else y


class DecoderLayer(nn.Module):
    def __init__(self, cfg: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(cfg)
        dense = cfg.num_experts <= 1 or layer_idx < cfg.moe_layer_num_skipped
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias) if dense \
            else MoE(cfg, layer_idx)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x, cos, sin, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Decoder(nn.Module):
    """model.wte -> 32 x DecoderLayer -> model.ln_f。lm_head 另外掛。"""

    def __init__(self, cfg: TextConfig, layers: Optional[Sequence[int]] = None):
        super().__init__()
        self.cfg = cfg
        self.layer_ids = list(range(cfg.num_hidden_layers) if layers is None else layers)
        self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg, i) for i in self.layer_ids]
        self.ln_f = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x, cos, sin, mask=None, embed: bool = True):
        h = self.wte(x) if embed else x
        for lyr in self.layers:
            h = lyr(h, cos, sin, mask)
        return self.ln_f(h)


# ----------------------------------------------------------------------------
# 權重載入
# ----------------------------------------------------------------------------

def _lin(mod, w, p, dtype):
    mod.weight = mx.array(w[p + ".weight"]).astype(dtype)
    n = 1
    if p + ".bias" in w:
        mod.bias = mx.array(w[p + ".bias"]).astype(dtype)
        n += 1
    return n


def load_layer(layer: DecoderLayer, w: Dict[str, np.ndarray], prefix: str,
               n_experts: int, dtype=mx.float32) -> int:
    n = 0
    a = layer.self_attn
    n += _lin(a.qkv_proj, w, prefix + "self_attn.qkv_proj", dtype)
    n += _lin(a.o_proj, w, prefix + "self_attn.o_proj", dtype)
    if a.use_qk_norm:
        a.query_layernorm.weight = mx.array(w[prefix + "self_attn.query_layernorm.weight"]).astype(dtype)
        a.key_layernorm.weight = mx.array(w[prefix + "self_attn.key_layernorm.weight"]).astype(dtype)
        n += 2
    layer.input_layernorm.weight = mx.array(w[prefix + "input_layernorm.weight"]).astype(dtype)
    layer.post_attention_layernorm.weight = mx.array(w[prefix + "post_attention_layernorm.weight"]).astype(dtype)
    n += 2

    m = layer.mlp
    if isinstance(m, MoE):
        m.gate.wg.weight = mx.array(w[prefix + "mlp.gate.wg.weight"]).astype(mx.float32)
        n += 1
        if m.use_shared:
            n += _lin(m.shared_mlp.gate_and_up_proj, w, prefix + "mlp.shared_mlp.gate_and_up_proj", dtype)
            n += _lin(m.shared_mlp.down_proj, w, prefix + "mlp.shared_mlp.down_proj", dtype)
        ups, gates, downs = [], [], []
        for e in range(n_experts):
            gu = w[prefix + f"mlp.experts.{e}.gate_and_up_proj.weight"]
            up, gate = np.split(gu, 2, axis=0)
            ups.append(up); gates.append(gate)
            downs.append(w[prefix + f"mlp.experts.{e}.down_proj.weight"])
            n += 2
        m.experts.up_w = mx.array(np.stack(ups)).astype(dtype)
        m.experts.gate_w = mx.array(np.stack(gates)).astype(dtype)
        m.experts.down_w = mx.array(np.stack(downs)).astype(dtype)
    else:
        n += _lin(m.gate_and_up_proj, w, prefix + "mlp.gate_and_up_proj", dtype)
        n += _lin(m.down_proj, w, prefix + "mlp.down_proj", dtype)
    return n


def load_decoder(dec: Decoder, w: Dict[str, np.ndarray], dtype=mx.float32,
                 with_embed: bool = True) -> int:
    n = 0
    if with_embed:
        dec.wte.weight = mx.array(w["model.wte.weight"]).astype(dtype)
        dec.ln_f.weight = mx.array(w["model.ln_f.weight"]).astype(dtype)
        n += 2
    for lyr, i in zip(dec.layers, dec.layer_ids):
        n += load_layer(lyr, w, f"model.layers.{i}.", dec.cfg.num_experts, dtype)
    return n
