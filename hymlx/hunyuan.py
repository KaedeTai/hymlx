"""里程碑 10：完整前向。

生成模式的前向跟一般 LLM 差在頭尾，中間 32 層是一樣的：

    h = wte(input_ids)
    h[影像位置]   = patch_embed(latent, time_embed(t))     # 連續的 latent 蓋掉離散 token
    h[timestep]  = timestep_emb(t)
    h = 32 層 decoder(2D RoPE, 文字因果 + 影像全連通)
    pred        = final_layer(h[影像位置], time_embed_2(t))

注意 **`ln_f` 在生成模式不會被用到**——它只服務文字 logits。影像預測直接從最後一層
decoder 的輸出接 `final_layer`。這一點看程式碼很容易漏掉，因為 `ln_f` 就掛在旁邊。

權重 157 GiB，放不進 128 GB。所以這裡支援 **streaming**：一次只把一層的權重讀進來，
算完就丟。這樣可以在不量化的情況下跑出精確的全前向，用來當量化版本的對照組。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .imageio import (TimestepEmbedder, UNetDown, UNetUp, load_timestep_embedder,
                      load_unet_down, load_unet_up)
from .model import DecoderLayer, TextConfig, load_layer
from .rope import build_batch_2d_rope, build_attention_mask


class ShardedWeights:
    """照 safetensors 的 index 隨用隨讀。開過的檔案留著，不會重複開。"""

    def __init__(self, snapshot: str, index_path: str | Path):
        from safetensors import safe_open
        self._open = safe_open
        self.snapshot = snapshot
        self.map: Dict[str, str] = json.load(open(index_path))["weight_map"]
        self._handles: Dict[str, Any] = {}

    def _h(self, shard: str):
        if shard not in self._handles:
            # 權重是 bf16，safetensors 的 numpy 後端讀不出來，走 torch 後端再轉 fp32
            self._handles[shard] = self._open(f"{self.snapshot}/{shard}", framework="pt")
        return self._handles[shard]

    def __contains__(self, name: str) -> bool:
        return name in self.map

    def __getitem__(self, name: str) -> np.ndarray:
        return self._h(self.map[name]).get_tensor(name).float().numpy()

    def prefix(self, p: str) -> Dict[str, np.ndarray]:
        return {k: self[k] for k in self.map if k.startswith(p)}


class HunyuanImage3(nn.Module):
    """生成模式的完整模型。`resident_layers=None` 代表全部常駐。"""

    def __init__(self, cfg: Dict[str, Any], weights: ShardedWeights,
                 dtype=mx.float32, resident: bool = False):
        super().__init__()
        self.raw_cfg = cfg
        self.tc = TextConfig.from_json(cfg)
        self.w = weights
        self.dtype = dtype
        self.hidden = cfg["hidden_size"]
        self.latent_channels = cfg["vae"]["latent_channels"]
        self.n_layers = cfg["num_hidden_layers"]
        self.head_dim = cfg["attention_head_dim"]
        self.rope_theta = cfg["rope_theta"]
        self.max_pos = cfg["max_position_embeddings"]

        self.wte = nn.Embedding(cfg["vocab_size"], self.hidden)
        self.wte.weight = mx.array(weights["model.wte.weight"]).astype(dtype)
        self.time_embed = TimestepEmbedder(self.hidden)
        self.time_embed_2 = TimestepEmbedder(self.hidden)
        self.timestep_emb = TimestepEmbedder(self.hidden)
        self.patch_embed = UNetDown(self.latent_channels, self.hidden,
                                    cfg.get("patch_embed_hidden_dim", 1024), self.hidden)
        self.final_layer = UNetUp(self.hidden, self.hidden,
                                  cfg.get("patch_embed_hidden_dim", 1024), self.latent_channels)
        head = {k: weights[k] for k in weights.map
                if k.split(".")[0] in ("time_embed", "time_embed_2", "timestep_emb",
                                       "patch_embed", "final_layer")}
        load_timestep_embedder(self.time_embed, head, "time_embed.", dtype)
        load_timestep_embedder(self.time_embed_2, head, "time_embed_2.", dtype)
        load_timestep_embedder(self.timestep_emb, head, "timestep_emb.", dtype)
        load_unet_down(self.patch_embed, head, "patch_embed.", dtype)
        load_unet_up(self.final_layer, head, "final_layer.", dtype)
        mx.eval(self.parameters())

        self.resident: Optional[List[DecoderLayer]] = None
        if resident:
            self.resident = []
            for i in range(self.n_layers):
                lyr = DecoderLayer(self.tc, i)
                load_layer(lyr, self.w.prefix(f"model.layers.{i}."),
                           f"model.layers.{i}.", self.tc.num_experts, dtype)
                mx.eval(lyr.parameters())
                self.resident.append(lyr)

    # -- 每層權重的來源 ------------------------------------------------------
    def _layer(self, i: int) -> DecoderLayer:
        if self.resident is not None:
            return self.resident[i]
        lyr = DecoderLayer(self.tc, i)
        load_layer(lyr, self.w.prefix(f"model.layers.{i}."),
                   f"model.layers.{i}.", self.tc.num_experts, self.dtype)
        mx.eval(lyr.parameters())
        return lyr

    # -- 前向 ---------------------------------------------------------------
    def embed(self, tokens: mx.array, latents: mx.array, t: mx.array,
              image_mask: mx.array, timestep_index: Optional[mx.array]) -> mx.array:
        """把離散 token embedding 與連續的 latent / timestep 混在同一條序列上。"""
        h = self.wte(tokens)
        B, S, D = h.shape
        img_seq, th, tw = self.patch_embed(latents, self.time_embed(t))
        idx = mx.array(np.where(np.array(image_mask, copy=False))[1].reshape(B, -1))
        h = _scatter_rows(h, idx, img_seq)
        if timestep_index is not None:
            ts = self.timestep_emb(t).reshape(B, -1, D)
            h = _scatter_rows(h, timestep_index, ts)
        return h

    def __call__(self, tokens: mx.array, latents: mx.array, t: mx.array,
                 image_mask: mx.array, timestep_index: Optional[mx.array],
                 cos: mx.array, sin: mx.array, mask: mx.array,
                 token_h: int, token_w: int,
                 progress=None) -> mx.array:
        h = self.embed(tokens, latents, t, image_mask, timestep_index)
        for i in range(self.n_layers):
            lyr = self._layer(i)
            h = lyr(h, cos, sin, mask)
            if self.resident is None:
                mx.eval(h)
                del lyr
            if progress is not None:
                progress(i)
        # 生成模式不過 ln_f
        B, S, D = h.shape
        idx = mx.array(np.where(np.array(image_mask, copy=False))[1].reshape(B, -1))
        img = mx.take_along_axis(h, idx[:, :, None], axis=1)
        return self.final_layer(img, self.time_embed_2(t), token_h, token_w)


def _scatter_rows(h: mx.array, idx: mx.array, src: mx.array) -> mx.array:
    """h[b, idx[b, k], :] = src[b, k, :]。"""
    D = h.shape[-1]
    K = idx.shape[1]
    return mx.put_along_axis(h, mx.broadcast_to(idx[:, :, None], (idx.shape[0], K, D)),
                             src.astype(h.dtype), axis=1)
