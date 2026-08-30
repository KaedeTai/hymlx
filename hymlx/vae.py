"""HunyuanImage-3.0 的 3D KL 自編碼器，解碼器部分（latent → 像素）。

移植自官方 `autoencoder_kl_3d.py`。三個要注意的地方：

- **MLX 的 conv 是 channels-last (NDHWC)，torch 是 NCDHW。** 整個解碼器內部一律用
  NDHWC，只在進出口轉置一次，權重載入時也把 (out, in, kD, kH, kW) 轉成
  (out, kD, kH, kW, in)。中間每層都轉會慢而且容易錯。

- **官方的 `Conv3d` 子類只是記憶體優化**：張量大於 2 GiB 時沿時間軸切塊、手動補
  padding 再拼回來，註解自己寫「與 nn.Conv3d 的數值差異在 1e-5 內」。那是等價的
  普通卷積，這裡直接用普通卷積，不複製那套切塊。

- **UpsampleDCAE 是 pixel-shuffle 加一條 repeat_interleave 捷徑**，不是插值。
  conv 出 C×factor 個通道再重排成空間（和時間）解析度，捷徑把輸入通道重複到同樣
  數量後做一樣的重排。兩者相加。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 32
    block_out_channels: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 1024])
    layers_per_block: int = 2
    ffactor_spatial: int = 16
    ffactor_temporal: int = 4
    upsample_match_channel: bool = True
    downsample_match_channel: bool = True
    scaling_factor: float = 0.562679178327931


def swish(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class GroupNorm3d(nn.Module):
    """GroupNorm over (T, H, W) for NDHWC input, 32 groups, eps 1e-6."""

    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.groups, self.eps, self.channels = groups, eps, channels
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        b, t, h, w, c = x.shape
        g = self.groups
        # group over channels; statistics span T, H, W and the channels in a group
        xg = x.reshape(b, t * h * w, g, c // g).astype(mx.float32)
        mean = xg.mean(axis=(1, 3), keepdims=True)
        var = xg.var(axis=(1, 3), keepdims=True)
        xg = (xg - mean) * mx.rsqrt(var + self.eps)
        out = xg.reshape(b, t, h, w, c).astype(x.dtype)
        return out * self.weight + self.bias


class Conv3d(nn.Module):
    """NDHWC 3D convolution with symmetric zero padding."""

    def __init__(self, cin: int, cout: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.weight = mx.zeros((cout, kernel, kernel, kernel, cin))
        self.bias = mx.zeros((cout,))
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        return mx.conv3d(x, self.weight, stride=1, padding=self.padding) + self.bias


class ResnetBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.norm1 = GroupNorm3d(cin)
        self.conv1 = Conv3d(cin, cout, 3, 1)
        self.norm2 = GroupNorm3d(cout)
        self.conv2 = Conv3d(cout, cout, 3, 1)
        if cin != cout:
            self.nin_shortcut = Conv3d(cin, cout, 1, 0)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        if self.cin != self.cout:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Full attention over every (t, h, w) position, one head."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = GroupNorm3d(channels)
        self.q = Conv3d(channels, channels, 1, 0)
        self.k = Conv3d(channels, channels, 1, 0)
        self.v = Conv3d(channels, channels, 1, 0)
        self.proj_out = Conv3d(channels, channels, 1, 0)

    def __call__(self, x: mx.array) -> mx.array:
        b, t, h, w, c = x.shape
        hn = self.norm(x)
        q, k, v = self.q(hn), self.k(hn), self.v(hn)
        shape = (b, 1, t * h * w, c)
        o = mx.fast.scaled_dot_product_attention(
            q.reshape(shape), k.reshape(shape), v.reshape(shape), scale=c ** -0.5)
        return x + self.proj_out(o.reshape(b, t, h, w, c))


class UpsampleDCAE(nn.Module):
    def __init__(self, cin: int, cout: int, temporal: bool):
        super().__init__()
        self.factor = 8 if temporal else 4
        self.r1 = 2 if temporal else 1
        self.conv = Conv3d(cin, cout * self.factor, 3, 1)
        self.repeats = self.factor * cout // cin
        self.cout = cout

    def _shuffle(self, x: mx.array) -> mx.array:
        # torch: "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)"
        # here channels are last, so the same split runs over the last axis.
        b, t, h, w, _ = x.shape
        r1, c = self.r1, self.cout
        x = x.reshape(b, t, h, w, r1, 2, 2, c)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)      # b t r1 h r2 w r3 c
        return x.reshape(b, t * r1, h * 2, w * 2, c)

    def __call__(self, x: mx.array) -> mx.array:
        h = self._shuffle(self.conv(x))
        short = mx.repeat(x, self.repeats, axis=-1)
        return h + self._shuffle(short)


class Decoder(nn.Module):
    """block_out_channels is given encoder-order and reversed here, exactly as
    AutoencoderKLConv3D does — the decoder runs 1024 -> 128, not 128 -> 1024.
    Getting this wrong is caught immediately by a state_dict shape mismatch,
    which is why loading the reference's own weights is the first test."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        import math
        self.cfg = cfg
        bo = list(reversed(cfg.block_out_channels))
        block_in = bo[0]
        self.conv_in = Conv3d(cfg.latent_channels, block_in, 3, 1)
        self.mid_block_1 = ResnetBlock(block_in, block_in)
        self.mid_attn_1 = AttnBlock(block_in)
        self.mid_block_2 = ResnetBlock(block_in, block_in)

        self.levels: List[dict] = []
        for i, ch in enumerate(bo):
            blocks = []
            for _ in range(cfg.layers_per_block + 1):
                blocks.append(ResnetBlock(block_in, ch))
                block_in = ch
            up = None
            spatial = i < math.log2(cfg.ffactor_spatial)
            temporal = i < math.log2(cfg.ffactor_temporal)
            if spatial or temporal:
                cout = bo[i + 1] if cfg.upsample_match_channel else block_in
                up = UpsampleDCAE(block_in, cout, temporal)
                block_in = cout
            self.levels.append({"block": blocks, "upsample": up})
        self.norm_out = GroupNorm3d(block_in)
        self.conv_out = Conv3d(block_in, cfg.out_channels, 3, 1)

    def __call__(self, z: mx.array) -> mx.array:
        """z is (B, C, T, H, W) as the reference gives it; returns the same layout."""
        z = z.transpose(0, 2, 3, 4, 1)                    # -> NDHWC
        repeats = list(reversed(self.cfg.block_out_channels))[0] // self.cfg.latent_channels
        h = self.conv_in(z) + mx.repeat(z, repeats, axis=-1)
        h = self.mid_block_2(self.mid_attn_1(self.mid_block_1(h)))
        for lvl in self.levels:
            for blk in lvl["block"]:
                h = blk(h)
            if lvl["upsample"] is not None:
                h = lvl["upsample"](h)
        h = self.conv_out(swish(self.norm_out(h)))
        return h.transpose(0, 4, 1, 2, 3)                 # -> NCTHW


def load_decoder(dec: Decoder, w: dict, prefix: str = "vae.decoder.", dtype=mx.float32) -> int:
    """torch (out, in, kD, kH, kW) -> MLX (out, kD, kH, kW, in)."""
    n = 0

    def conv(mod, p):
        nonlocal n
        mod.weight = mx.array(w[p + ".weight"]).transpose(0, 2, 3, 4, 1).astype(dtype)
        mod.bias = mx.array(w[p + ".bias"]).astype(dtype)
        n += 2

    def gn(mod, p):
        nonlocal n
        mod.weight = mx.array(w[p + ".weight"]).astype(dtype)
        mod.bias = mx.array(w[p + ".bias"]).astype(dtype)
        n += 2

    def resnet(mod, p):
        gn(mod.norm1, p + ".norm1"); conv(mod.conv1, p + ".conv1")
        gn(mod.norm2, p + ".norm2"); conv(mod.conv2, p + ".conv2")
        if mod.cin != mod.cout:
            conv(mod.nin_shortcut, p + ".nin_shortcut")

    P = prefix
    conv(dec.conv_in, P + "conv_in")
    resnet(dec.mid_block_1, P + "mid.block_1")
    a = dec.mid_attn_1
    gn(a.norm, P + "mid.attn_1.norm")
    for nm, m in (("q", a.q), ("k", a.k), ("v", a.v), ("proj_out", a.proj_out)):
        conv(m, f"{P}mid.attn_1.{nm}")
    resnet(dec.mid_block_2, P + "mid.block_2")
    for i, lvl in enumerate(dec.levels):
        for j, blk in enumerate(lvl["block"]):
            resnet(blk, f"{P}up.{i}.block.{j}")
        if lvl["upsample"] is not None:
            conv(lvl["upsample"].conv, f"{P}up.{i}.upsample.conv")
    gn(dec.norm_out, P + "norm_out")
    conv(dec.conv_out, P + "conv_out")
    return n


class DownsampleDCAE(nn.Module):
    """Pixel-unshuffle plus a mean-pooled shortcut — the mirror of UpsampleDCAE."""

    def __init__(self, cin: int, cout: int, temporal: bool):
        super().__init__()
        self.factor = 8 if temporal else 4
        self.r1 = 2 if temporal else 1
        self.conv = Conv3d(cin, cout // self.factor, 3, 1)
        self.group_size = self.factor * cin // cout
        self.cout = cout

    def _unshuffle(self, x: mx.array) -> mx.array:
        # torch: "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w"
        b, t, h, w, c = x.shape
        r1 = self.r1
        x = x.reshape(b, t // r1, r1, h // 2, 2, w // 2, 2, c)
        x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)          # b f h w r1 r2 r3 c
        return x.reshape(b, t // r1, h // 2, w // 2, r1 * 4 * c)

    def __call__(self, x: mx.array) -> mx.array:
        h = self._unshuffle(self.conv(x))
        short = self._unshuffle(x)
        b, t, hh, w, c = short.shape
        short = short.reshape(b, t, hh, w, c // self.group_size, self.group_size).mean(axis=-1)
        return h + short


class Encoder(nn.Module):
    """Pixels -> latent. Unlike the decoder this takes block_out_channels in the
    given order, and uses `num_res_blocks` per level rather than +1."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        import math
        self.cfg = cfg
        bo = cfg.block_out_channels
        block_in = bo[0]
        self.conv_in = Conv3d(cfg.in_channels, block_in, 3, 1)
        self.levels: List[dict] = []
        for i, ch in enumerate(bo):
            blocks = []
            for _ in range(cfg.layers_per_block):
                blocks.append(ResnetBlock(block_in, ch))
                block_in = ch
            down = None
            spatial = i < math.log2(cfg.ffactor_spatial)
            temporal = spatial and i >= math.log2(cfg.ffactor_spatial // cfg.ffactor_temporal)
            if spatial or temporal:
                cout = bo[i + 1] if cfg.downsample_match_channel else block_in
                down = DownsampleDCAE(block_in, cout, temporal)
                block_in = cout
            self.levels.append({"block": blocks, "downsample": down})
        self.mid_block_1 = ResnetBlock(block_in, block_in)
        self.mid_attn_1 = AttnBlock(block_in)
        self.mid_block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = GroupNorm3d(block_in)
        self.conv_out = Conv3d(block_in, 2 * cfg.latent_channels, 3, 1)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 3, 4, 1)
        h = self.conv_in(x)
        for lvl in self.levels:
            for blk in lvl["block"]:
                h = blk(h)
            if lvl["downsample"] is not None:
                h = lvl["downsample"](h)
        h = self.mid_block_2(self.mid_attn_1(self.mid_block_1(h)))
        # mirror of the decoder's repeat_interleave shortcut: mean-pool groups of
        # channels down to 2 * latent_channels and add it to conv_out.
        gs = self.cfg.block_out_channels[-1] // (2 * self.cfg.latent_channels)
        b, t, hh, ww, c = h.shape
        short = h.reshape(b, t, hh, ww, c // gs, gs).mean(axis=-1)
        h = self.conv_out(swish(self.norm_out(h))) + short
        return h.transpose(0, 4, 1, 2, 3)


def load_encoder(enc: Encoder, w: dict, prefix: str = "vae.encoder.", dtype=mx.float32) -> int:
    n = 0

    def conv(mod, p):
        nonlocal n
        mod.weight = mx.array(w[p + ".weight"]).transpose(0, 2, 3, 4, 1).astype(dtype)
        mod.bias = mx.array(w[p + ".bias"]).astype(dtype); n += 2

    def gn(mod, p):
        nonlocal n
        mod.weight = mx.array(w[p + ".weight"]).astype(dtype)
        mod.bias = mx.array(w[p + ".bias"]).astype(dtype); n += 2

    def resnet(mod, p):
        gn(mod.norm1, p + ".norm1"); conv(mod.conv1, p + ".conv1")
        gn(mod.norm2, p + ".norm2"); conv(mod.conv2, p + ".conv2")
        if mod.cin != mod.cout:
            conv(mod.nin_shortcut, p + ".nin_shortcut")

    P = prefix
    conv(enc.conv_in, P + "conv_in")
    for i, lvl in enumerate(enc.levels):
        for j, blk in enumerate(lvl["block"]):
            resnet(blk, f"{P}down.{i}.block.{j}")
        if lvl["downsample"] is not None:
            conv(lvl["downsample"].conv, f"{P}down.{i}.downsample.conv")
    resnet(enc.mid_block_1, P + "mid.block_1")
    a = enc.mid_attn_1
    gn(a.norm, P + "mid.attn_1.norm")
    for nm, m in (("q", a.q), ("k", a.k), ("v", a.v), ("proj_out", a.proj_out)):
        conv(m, f"{P}mid.attn_1.{nm}")
    resnet(enc.mid_block_2, P + "mid.block_2")
    gn(enc.norm_out, P + "norm_out")
    conv(enc.conv_out, P + "conv_out")
    return n
