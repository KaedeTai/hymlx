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
from .model import DecoderLayer, TextConfig, load_layer, load_layer_quantized
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


class QuantizedHunyuan(nn.Module):
    """從 `tools/convert.py` 產出的目錄載入。整包約 46 GiB，可以整個常駐。"""

    def __init__(self, qdir: str | Path, dtype=mx.bfloat16, layers: Optional[int] = None,
                 requant: Optional[Dict[str, Tuple[int, int]]] = None):
        """`requant` 讓某些權重在**載入時**改成別的位元數，不必另外產生檔案。
        例如 {"": (4, 64)} 全部降成 4-bit；
        {"mlp.experts": (4, 64)} 只把 64 顆專家降成 4-bit，attention 維持原樣。"""
        super().__init__()
        qdir = Path(qdir)
        cfg = json.load(open(qdir / "config.json"))
        qc = json.load(open(qdir / "quant_config.json"))
        self.qdir, self.raw_cfg, self.q = qdir, cfg, (qc["bits"], qc["group_size"])
        self.wdtype = dtype
        self.tc = TextConfig.from_json(cfg)
        self.hidden = cfg["hidden_size"]
        self.latent_channels = cfg["vae"]["latent_channels"]
        self.n_layers = layers or cfg["num_hidden_layers"]
        self.head_dim = cfg["attention_head_dim"]
        self.rope_theta = cfg["rope_theta"]

        head = mx.load(str(qdir / "head.safetensors"))
        # wte 在檔案裡是量化的（省 0.8 GiB 磁碟），但查表用不上量化，
        # 載入時解一次成 bf16 常駐（1.1 GiB）。
        self.wte = mx.dequantize(head["model.wte.weight"], head["model.wte.scales"],
                                 head["model.wte.biases"],
                                 group_size=self.q[1], bits=self.q[0]).astype(dtype)

        self.time_embed = TimestepEmbedder(self.hidden)
        self.time_embed_2 = TimestepEmbedder(self.hidden)
        self.timestep_emb = TimestepEmbedder(self.hidden)
        ph = cfg.get("patch_embed_hidden_dim", 1024)
        self.patch_embed = UNetDown(self.latent_channels, self.hidden, ph, self.hidden)
        self.final_layer = UNetUp(self.hidden, self.hidden, ph, self.latent_channels)
        load_timestep_embedder(self.time_embed, head, "time_embed.", dtype)
        load_timestep_embedder(self.time_embed_2, head, "time_embed_2.", dtype)
        load_timestep_embedder(self.timestep_emb, head, "timestep_emb.", dtype)
        load_unet_down(self.patch_embed, head, "patch_embed.", dtype)
        load_unet_up(self.final_layer, head, "final_layer.", dtype)

        import mlx.nn as _nn
        from .quant import QuantLinear as _QL
        self.ln_f = _nn.RMSNorm(self.hidden, eps=cfg["rms_norm_eps"])
        self.ln_f.weight = head["model.ln_f.weight"].astype(dtype)
        self.lm_head = _QL(*self.q)
        self.lm_head.weight = head["lm_head.weight"]
        self.lm_head.scales = head["lm_head.scales"]
        self.lm_head.biases = head["lm_head.biases"]

        self._vision = self._aligner = self._encoder = None   # 編輯才載
        self.layers = []
        for i in range(self.n_layers):
            # 一律用檔案本身的位元數建構，個別模組的覆寫交給 load_layer_quantized。
            # 之前在這裡先猜一個 base，結果混合精度時 attention 的權重還是 8-bit、
            # bits 卻被設成 4，形狀對不上就爆了。
            lyr = DecoderLayer(self.tc, i, self.q)
            load_layer_quantized(lyr, mx.load(str(qdir / f"layer_{i:02d}.safetensors")),
                                 src=self.q, override=requant)
            mx.eval(lyr.parameters())
            self.layers.append(lyr)
        mx.eval(self.parameters())

    def embed_tokens(self, tokens: mx.array) -> mx.array:
        return self.wte[tokens]

    def logits(self, tokens: mx.array, cos: mx.array, sin: mx.array,
               mask=None, caches=None) -> mx.array:
        """純文字模式：走 ln_f + lm_head。影像模式**不走** ln_f，這是兩條不同的路。

        caches 是 32 個 [k, v]；給了就是增量解碼，只餵新的 token。
        """
        h = self.embed_tokens(tokens)
        for i, lyr in enumerate(self.layers):
            h = lyr(h, cos, sin, mask, None if caches is None else caches[i])
        h = self.ln_f(h[:, -1:, :])
        return self.lm_head(h)[:, 0]

    def new_caches(self):
        return [[] for _ in range(self.n_layers)]

    # ------------------------------------------------------------------
    # 影像生成的 KV cache
    #
    # 取樣的每一步，文字前綴（系統提示詞 + prompt + CoT，兩千多個 token）都是
    # 一模一樣的，重算 50 次是白費。而且一旦把前綴的 k/v 快取起來，**遮罩就整個
    # 不需要了**：影像 token 本來就看得到全部的文字與彼此，而 timestep token
    # 不能看影像——把它單獨先跑一遍就自然滿足了，不必materialise 一張
    # 6278x6278 的遮罩（fp32 315 MB，每層都要讀一次）。
    # 序列尾巴 <eoi> 之後的文字也不用算，final_layer 根本不讀它們。
    # ------------------------------------------------------------------

    # -- 參考圖（編輯用）------------------------------------------------
    # 參考圖的 token 在取樣過程中不變（時間步固定 0，就是「乾淨的圖」），
    # 而且在序列上排在生成影像之前，所以整段可以放進前綴快取：每步成本不變，
    # 只有一次性的前綴變長。

    def _load_cond_encoders(self):
        """視覺塔與 VAE 編碼器只有編輯才用得到，用到才載（約 2.2 GiB）。"""
        if getattr(self, "_vision", None) is not None:
            return
        import json as _json
        from .vae import Encoder, VAEConfig, load_encoder
        from .vision import Aligner, VisionConfig, VisionTower, load_aligner, load_vision
        cfg = self.raw_cfg
        vw = mx.load(str(self.qdir / "vision.safetensors"))
        self._vision = VisionTower(VisionConfig.from_json(cfg["vit"]))
        load_vision(self._vision, vw, dtype=self.wdtype)
        al = cfg["vit_aligner"]
        self._aligner = Aligner(**{k: al[k] for k in ("input_dim", "n_embed", "depth")})
        load_aligner(self._aligner, vw, dtype=self.wdtype)
        vc = cfg["vae"]
        K = ("in_channels", "out_channels", "latent_channels", "block_out_channels",
             "layers_per_block", "ffactor_spatial", "ffactor_temporal",
             "upsample_match_channel", "downsample_match_channel", "scaling_factor")
        self._vae_cfg = VAEConfig(**{k: vc[k] for k in K})
        # VAE 在 1024x1024 x 4 幀 x 128 通道上跑，一個中間張量 fp32 就 2 GiB。
        # 官方自己的 vae_autocast_dtype 是 float16，所以 bf16 不是偷工，是照做。
        self._encoder = Encoder(self._vae_cfg)
        load_encoder(self._encoder, mx.load(str(self.qdir / "vae.safetensors")),
                     dtype=mx.bfloat16)
        mx.eval(self._vision.parameters(), self._aligner.parameters(),
                self._encoder.parameters())

    def encode_cond_image(self, pixels: mx.array) -> mx.array:
        """(B, 3, H, W) 的 [-1,1] 像素 -> (B, 32, h, w) 的 latent。

        官方對靜態圖是沿時間軸 expand 成 4 幀再編碼，取的是後驗的平均
        （logvar 平均 -14.5，標準差約 7e-4，取樣與取平均沒有差別）。
        """
        self._load_cond_encoders()
        # CFG 的兩列共用同一張參考圖，編碼一次就好——編兩次是把 28 GiB 的峰值變兩倍。
        one = pixels[0:1]
        x = mx.repeat(one[:, :, None], self._vae_cfg.ffactor_temporal, axis=2)
        z = self._encoder(x.astype(mx.bfloat16))[:, :self._vae_cfg.latent_channels, 0]
        z = (z * self._vae_cfg.scaling_factor).astype(mx.float32)
        mx.eval(z)
        return mx.broadcast_to(z, (pixels.shape[0],) + z.shape[1:])

    def embed_prefix(self, tokens: mx.array, cond=None) -> mx.array:
        """前綴的 embedding。有參考圖就把 VAE latent 與 ViT 特徵蓋進去。"""
        h = self.embed_tokens(tokens)
        if cond is None:
            return h
        self._load_cond_encoders()
        B, S, D = h.shape
        t0 = mx.zeros((B,))
        # 每一段都當場求值。不這樣做的話 VAE 編碼器、視覺塔、後面 32 層 decoder
        # 會全部堆在同一張懶惰的圖裡，峰值記憶體多出二十幾 GiB——VAE 編碼器光第一層
        # 卷積在 (2, 4, 1024, 1024, 128) 就是 2.1 GB。
        lat = self.encode_cond_image(mx.array(cond.vae_pixels)); mx.eval(lat)
        img, _, _ = self.patch_embed(lat, self.time_embed(t0)); mx.eval(img)
        del lat; mx.clear_cache()
        h = _scatter_rows(h, mx.array(cond.vae_index), img); mx.eval(h)
        del img
        feat = self._aligner(self._vision(mx.array(cond.vit_pixels), cond.vit_shapes,
                                          mx.array(cond.vit_mask)))
        mx.eval(feat)
        h = _scatter_rows(h, mx.array(cond.vit_index), feat); mx.eval(h)
        del feat; mx.clear_cache()
        h = _scatter_rows(h, mx.array(cond.ts_index),
                          self.timestep_emb(t0).reshape(B, -1, D))
        return h

    def prefill_prefix(self, tokens: mx.array, cos: mx.array, sin: mx.array,
                       mask: mx.array, cond=None):
        """把 <timestep> 之前的整段跑一次，留下每一層的 k/v。只做一次。"""
        caches = self.new_caches()
        h = self.embed_prefix(tokens, cond)
        mx.eval(h)
        if cond is not None:
            # 視覺塔與 VAE 編碼器只在這裡用一次。模型本體已經佔 84 GiB，
            # 編輯的序列又比文生圖長一倍，留著這 2.2 GiB 會把機器推進 swap。
            self._vision = self._aligner = self._encoder = None
            mx.clear_cache()
        for i, lyr in enumerate(self.layers):
            h = lyr(h, cos, sin, mask, caches[i])
            mx.eval(h, *caches[i])          # 逐層求值，峰值才壓得下來
        del h
        mx.clear_cache()
        return caches

    def image_step(self, prefix_caches, latents: mx.array, t: mx.array,
                   cos_ts: mx.array, sin_ts: mx.array,
                   cos_img: mx.array, sin_img: mx.array,
                   token_h: int, token_w: int) -> mx.array:
        """一步去噪。prefix_caches 不會被改到，每一步都從它重新長出來。"""
        caches = [list(c) for c in prefix_caches]

        # 1. timestep token 先單獨走完 32 層。它在序列上排在影像之前，看不到影像，
        #    所以先跑它，之後影像那一段就可以完全不用遮罩。
        h = self.timestep_emb(t).reshape(t.shape[0], 1, self.hidden)
        for i, lyr in enumerate(self.layers):
            h = lyr(h, cos_ts, sin_ts, None, caches[i])
            mx.eval(h, *caches[i])

        # 2. 影像 token：對前綴、timestep、彼此全連通，mask=None 就是正確答案
        h, _, _ = self.patch_embed(latents, self.time_embed(t))
        for i, lyr in enumerate(self.layers):
            h = lyr(h, cos_img, sin_img, None, caches[i])
            mx.eval(h, *caches[i])
        return self.final_layer(h, self.time_embed_2(t), token_h, token_w)

    def __call__(self, tokens, latents, t, image_mask, timestep_index,
                 cos, sin, mask, token_h, token_w, progress=None):
        h = self.embed_tokens(tokens)
        B, S, D = h.shape
        img_seq, _, _ = self.patch_embed(latents, self.time_embed(t))
        idx = mx.array(np.where(np.array(image_mask, copy=False))[1].reshape(B, -1))
        h = _scatter_rows(h, idx, img_seq)
        if timestep_index is not None:
            h = _scatter_rows(h, timestep_index, self.timestep_emb(t).reshape(B, -1, D))
        for i, lyr in enumerate(self.layers):
            h = lyr(h, cos, sin, mask)
            if progress is not None:
                progress(i)
        img = mx.take_along_axis(h, idx[:, :, None], axis=1)
        return self.final_layer(img, self.time_embed_2(t), token_h, token_w)
