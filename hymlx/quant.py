"""里程碑 12：量化。

權重 157 GiB，機器 128 GB——不量化就跑不動，這不是為了快，是為了能跑。
（`FEASIBILITY.md` 裡量過：1024² 一次前向計算 2.8 s、讀權重 0.08 s，是**計算受限**。
量化讓它放得進記憶體，不會讓它變快。）

哪些留原樣，理由跟 mdream 的 `PIXEL_SHIMS` 一樣：**在像素與 token 之間轉換的那幾層
不量化**。它們只佔 0.6% 的參數，但誤差會直接變成畫面上的格子與色偏：

    patch_embed / final_layer / time_embed / time_embed_2 / timestep_emb
    所有 norm、MoE 的路由 gate（官方本來就是 fp32）
    VAE、視覺塔

量的是 attention 的 qkv/o、共享專家、64 顆專家、wte 與 lm_head——也就是那 91.7%。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# 不量化的前綴（比對的是完整權重名）
KEEP_FP = (
    "patch_embed.", "final_layer.", "time_embed.", "time_embed_2.", "timestep_emb.",
    "vae.", "vision_model.", "vision_aligner.",
    "model.ln_f.",
)
# 名字裡出現這些就不量化（layer 內部）
KEEP_FP_SUFFIX = (
    "layernorm.weight", "norm.weight", "mlp.gate.wg.weight",
)


def should_quantize(name: str, min_dim: int = 64) -> bool:
    if any(name.startswith(p) for p in KEEP_FP):
        return False
    if any(name.endswith(s) for s in KEEP_FP_SUFFIX):
        return False
    return name.endswith(".weight")


class QuantLinear(nn.Module):
    """只做推論的量化線性層。權重是 x @ W.T，跟 nn.Linear 一致。"""

    def __init__(self, bits: int = 4, group_size: int = 64, bias: bool = False):
        super().__init__()
        self.bits, self.group_size = bits, group_size
        self.weight = mx.zeros((1, 1), dtype=mx.uint32)
        self.scales = mx.zeros((1, 1))
        self.biases = mx.zeros((1, 1))
        self.bias = None

    @classmethod
    def from_array(cls, w: mx.array, bits: int = 4, group_size: int = 64,
                   bias: Optional[mx.array] = None) -> "QuantLinear":
        q = cls(bits, group_size)
        q.weight, q.scales, q.biases = mx.quantize(w, group_size=group_size, bits=bits)
        q.bias = bias
        return q

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.quantized_matmul(x, self.weight, scales=self.scales, biases=self.biases,
                                transpose=True, group_size=self.group_size, bits=self.bits)
        return y if self.bias is None else y + self.bias


def gather_sort(x: mx.array, indices: mx.array):
    """把 (token, 專家) 配對照專家編號排好。

    不排的話 `gather_qmm` 會為每一個配對獨立做一次矩陣向量乘，每次都要把那顆專家
    21 MB 的權重整包讀一遍——4096 個 token x 8 顆 = 33k 次，700 GB 的讀取量。
    排好之後同一顆專家的 token 連在一起，變成一次真正的 GEMM。
    實測：一步 106 s -> 見 notes/plan.md。
    """
    M = indices.shape[-1]
    flat = indices.flatten()
    order = mx.argsort(flat)
    inv = mx.argsort(order)
    return x.flatten(0, -3)[order // M], flat[order], inv


def scatter_unsort(x: mx.array, inv: mx.array, shape) -> mx.array:
    return mx.unflatten(x[inv], 0, shape)


class QuantSwitchMLP(nn.Module):
    """64 顆專家的量化版。每顆專家自己一組 scale/bias。"""

    def __init__(self, bits: int = 4, group_size: int = 64):
        super().__init__()
        self.bits, self.group_size = bits, group_size
        for n in ("up", "gate", "down"):
            setattr(self, f"{n}_w", mx.zeros((1, 1, 1), dtype=mx.uint32))
            setattr(self, f"{n}_s", mx.zeros((1, 1, 1)))
            setattr(self, f"{n}_b", mx.zeros((1, 1, 1)))

    def _mm(self, x: mx.array, n: str, inds: mx.array, sorted_: bool) -> mx.array:
        return mx.gather_qmm(x, getattr(self, f"{n}_w"), scales=getattr(self, f"{n}_s"),
                             biases=getattr(self, f"{n}_b"), rhs_indices=inds,
                             transpose=True, group_size=self.group_size, bits=self.bits,
                             sorted_indices=sorted_)

    def __call__(self, x: mx.array, inds: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = inds.size >= 64
        idx, inv = inds, None
        if do_sort:
            x, idx, inv = gather_sort(x, inds)
        up = self._mm(x, "up", idx, do_sort)
        gate = self._mm(x, "gate", idx, do_sort)
        y = self._mm(up * nn.silu(gate), "down", idx, do_sort)
        if do_sort:
            y = scatter_unsort(y, inv, inds.shape)
        return y.squeeze(-2)


def quantize_stack(w: mx.array, bits: int, group_size: int) -> Tuple[mx.array, mx.array, mx.array]:
    """對 (E, out, in) 的專家堆做量化；mx.quantize 只吃最後一維分組，直接餵就對。"""
    return mx.quantize(w, group_size=group_size, bits=bits)


def bits_per_param(bits: int, group_size: int) -> float:
    """含 scale 與 bias（各 fp16）的實際位元數。"""
    return bits + 2 * 16 / group_size
