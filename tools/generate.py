"""端到端生成：prompt -> 圖。

    python tools/generate.py "a photo of a red fox in snow" -o fox.png

流程刻意沒做 KV cache。官方在第一步之後只重餵影像 token，靠 cache 省下文字前綴；
但這個模型的文字前綴只有 55 個 token，影像有 4096 個，省下的不到 1.5%，
換來的是一整套 cache 的正確性風險。等哪天測出來真的差很多再說。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from hymlx.conditioning import Conditioner
    from hymlx.hunyuan import QuantizedHunyuan
    from hymlx.rope import build_batch_2d_rope, build_attention_mask
    from hymlx.sampling import cfg as apply_cfg, sigma_schedule
    from hymlx.vae import Decoder, VAEConfig, decode_image, load_decoder

    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-o", "--out", default="out.png")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=str(Path.home() / "models/hymlx-4bit"))
    ap.add_argument("--layers", type=int, default=None, help="只跑前 N 層（測速用）")
    ap.add_argument("--system-prompt", default="en_vanilla",
                    help="en_vanilla / en_unified / none")
    ap.add_argument("--cot-file", default=None,
                    help="tools/recaption.py 產生的 <think>…</think><recaption>…</recaption>")
    ap.add_argument("--recaption", default=None,
                    help="官方的圖像階段一定會餵一段 recaption（think/recaption 產生的）。"
                         "沒有它模型是在分布外跑。這裡先讓人手動給。")
    a = ap.parse_args()

    t0 = time.time()
    cond = Conditioner()
    sp = None if a.system_prompt == "none" else cond.system_prompt(a.system_prompt)
    cot = None
    tk = cond._tokenizer
    if a.cot_file:
        cot = [Path(a.cot_file).read_text()]
    elif a.recaption:
        cot = [tk.recaption_token + a.recaption + tk.end_of_recaption_token]
    c = cond.build(a.prompt, image_size=a.size, system_prompt=sp, cot_text=cot)
    print(f"序列 {c.tokens.shape}  影像 token {c.token_h}x{c.token_w}  "
          f"輸出 {c.image_width}x{c.image_height}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    m = QuantizedHunyuan(a.model, layers=a.layers)
    print(f"模型載入 {time.time()-t0:.0f}s")

    D = m.head_dim
    cos_l, sin_l = [], []
    for info in c.rope_image_info:
        cc, ss = build_batch_2d_rope(c.seq_len, D, image_infos=[info], base=m.rope_theta)
        cos_l.append(cc[0]); sin_l.append(ss[0])
    cos, sin = mx.stack(cos_l), mx.stack(sin_l)
    mask = mx.concatenate([build_attention_mask(c.seq_len, sl) for sl in c.full_attn_slices])
    amask = mx.where(mask, mx.array(0.0, mx.float32), mx.array(-3.4028235e38, mx.float32))

    tokens = mx.array(c.tokens.astype(np.int32))
    image_mask = mx.array(c.gen_image_mask)
    ts_index = None if c.gen_timestep_index is None else mx.array(c.gen_timestep_index)

    rs = np.random.RandomState(a.seed)
    x = mx.array(rs.randn(1, m.latent_channels, c.token_h, c.token_w).astype(np.float32))
    sigmas, timesteps = sigma_schedule(a.steps, a.shift)
    B = c.batch                                     # 2：有條件 + 無條件

    t_all = time.time()
    for i in range(a.steps):
        t_step = time.time()
        xin = mx.repeat(x, B, axis=0)
        tt = mx.full((B,), float(timesteps[i]))
        pred = m(tokens, xin, tt, image_mask, ts_index, cos, sin, amask,
                 c.token_h, c.token_w)
        pred = pred.astype(mx.float32)
        v = apply_cfg(pred[0:1], pred[1:2], a.guidance) if B == 2 else pred
        x = x + v * float(sigmas[i + 1] - sigmas[i])
        mx.eval(x)
        print(f"  步 {i+1:>2}/{a.steps}  {time.time()-t_step:.1f}s", flush=True)
    print(f"取樣 {time.time()-t_all:.0f}s（{(time.time()-t_all)/a.steps:.1f}s/步）")

    # VAE 解碼
    vcfg = json.loads((Path(a.model) / "config.json").read_text())["vae"]
    sf = vcfg.get("scaling_factor")
    lat = x / sf if sf else x
    dec = Decoder(VAEConfig(**{k: vcfg[k] for k in (
        "in_channels", "out_channels", "latent_channels", "block_out_channels",
        "layers_per_block", "ffactor_spatial", "ffactor_temporal",
        "upsample_match_channel", "downsample_match_channel", "scaling_factor")}))
    load_decoder(dec, mx.load(str(Path(a.model) / "vae.safetensors")))
    mx.eval(dec.parameters())
    # 一定要走 decode_image：T=1 的 latent 會放大成 4 幀，官方取的是最後一幀
    img = np.array(decode_image(dec, lat), copy=False)[0, :, -1]
    img = np.clip((img + 1) / 2, 0, 1)
    from PIL import Image
    Image.fromarray((img.transpose(1, 2, 0) * 255).round().astype(np.uint8)).save(a.out)
    print(f"寫出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
