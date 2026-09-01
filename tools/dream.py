"""prompt -> CoT -> 圖，一支到底。模型只載一次，可以連跑好幾個題目。

    python tools/dream.py "a hamster ..." "a desk ..." --steps 14
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def generate_cot(m, cond, prompt, sys_prompt, max_new=2048, temperature=0.6,
                 top_p=0.95, top_k=1024, seed=0, image=None, log=print):
    """跑文字階段，讓模型自己寫 <think>…</think><recaption>…</recaption>。

    官方的圖像階段吃的是這段文字，不是使用者的原句。沒有它模型是在分布外跑。

    編輯（給了 `image`）時，官方的 `generate_image` **兩個階段都傳 image**：
    文字階段的序列裡就有那張參考圖（1024² 佔 4096 個 VAE token + 1024 個 ViT
    token），模型是看著圖寫 recaption，寫出來的是「改完之後的圖長什麼樣」，
    不是把使用者那句短指令重複一遍。所以這裡也要照做——序列變長六倍，
    rope 要帶 2D 段落，遮罩要讓參考圖那一塊全連通，embedding 要自己蓋。
    """
    from hymlx.rope import build_batch_2d_rope, build_attention_mask
    tk = cond._tokenizer
    tc = cond.build_text(prompt, system_prompt=sys_prompt, bot_task="think",
                         image=image, with_cond=True)
    toks = list(tc.tokens[0])
    plen = len(toks)
    rng = np.random.RandomState(seed)
    caches = m.new_caches()
    D = m.head_dim
    S = len(toks)
    info = tc.rope_image_info[0] if image is not None else None
    MAXPOS = S + max_new + 8
    # 前綴的位置是整段的前綴（rope_positions 對圖像段之後的文字接著往下排），
    # 所以一次建到 MAXPOS，前綴直接切片就好。
    cos_all, sin_all = build_batch_2d_rope(MAXPOS, D, image_infos=[info], base=m.rope_theta)
    cos, sin = cos_all[:, :S], sin_all[:, :S]
    am = mx.where(build_attention_mask(S, tc.full_attn_slices[0]),
                  mx.array(0.0, mx.float32), mx.array(-3.4028235e38, mx.float32))
    t0 = time.time()
    if image is None:
        lg = m.logits(mx.array(np.array(toks, dtype=np.int32)[None]), cos, sin, am, caches)
    else:
        h = m.embed_prefix(mx.array(tc.tokens.astype(np.int32)), tc.cond)
        mx.eval(h)
        m._vision = m._aligner = m._encoder = None      # 特徵已快取，卸掉編碼器
        mx.clear_cache()
        lg = m.logits(None, cos, sin, am, caches, embeds=h)
        del h
    del am
    mx.clear_cache()

    def sample(logits):
        v = np.array(logits.astype(mx.float32), copy=False)[0] / max(temperature, 1e-6)
        idx = np.argsort(-v)[:top_k]
        pr = np.exp(v[idx] - v[idx].max()); pr /= pr.sum()
        keep = int(np.searchsorted(np.cumsum(pr), top_p)) + 1
        idx, pr = idx[:keep], pr[:keep] / pr[:keep].sum()
        return int(rng.choice(idx, p=pr))

    def step(tid):
        pos = len(toks)
        out = m.logits(mx.array(np.array([[tid]], dtype=np.int32)),
                       cos_all[:, pos:pos + 1], sin_all[:, pos:pos + 1], None, caches)
        toks.append(tid)
        return out

    stage, nxt = "think", sample(lg)
    for _ in range(max_new):
        lg = step(nxt)
        if nxt == tk.end_of_think_token_id and stage == "think":
            lg = step(tk.convert_tokens_to_ids(tk.recaption_token))   # 官方的 stage transition
            stage = "recaption"
        if nxt == tk.end_of_recaption_token_id:
            break
        nxt = sample(lg)
    cot = tk.think_token + tk.decode(toks[plen:])
    n = len(toks) - plen
    # 撞到上限代表 recaption 是被切斷的（可能斷在字中間），影像階段吃到的
    # 就是半句話。官方的 max_new_tokens 是 2048，我一開始寫 1100 就踩到過。
    trunc = "（**撞到上限，被切斷**）" if n >= max_new else ""
    log(f"    CoT {n} token，{time.time()-t0:.0f}s{trunc}")
    del caches
    mx.clear_cache()
    return cot


def render(m, cond, dec, vc, prompt, cot, sys_prompt, size, steps, guidance, shift,
           seed, out_path, cfg_steps=None, image=None, model_dir=None, log=print):
    from hymlx.rope import build_batch_2d_rope, build_attention_mask
    from hymlx.sampling import cfg as apply_cfg, sigma_schedule
    from hymlx.vae import decode_image
    from PIL import Image

    c = cond.build(prompt, image_size=size, system_prompt=sys_prompt,
                   cot_text=[cot] if cot else None, image=image)
    D = m.head_dim
    cos = mx.stack([build_batch_2d_rope(c.seq_len, D, image_infos=[i], base=m.rope_theta)[0][0]
                    for i in c.rope_image_info])
    sin = mx.stack([build_batch_2d_rope(c.seq_len, D, image_infos=[i], base=m.rope_theta)[1][0]
                    for i in c.rope_image_info])
    P = int(c.gen_timestep_index[0][0])
    istart = int(np.where(c.gen_image_mask[0])[0][0])
    iend = int(np.where(c.gen_image_mask[0])[0][-1]) + 1
    assert istart == P + 1
    # 前綴的遮罩：文字因果，但參考圖那一整塊（VAE + ViT）要全連通
    pre_full = [s_ for s_ in c.full_attn_slices[0] if s_.stop <= P]
    pm = mx.broadcast_to(build_attention_mask(P, pre_full)[0], (2, 1, P, P))
    pam = mx.where(pm, mx.array(0.0, mx.float32), mx.array(-3.4028235e38, mx.float32))
    t0 = time.time()
    caches = m.prefill_prefix(mx.array(c.tokens[:, :P].astype(np.int32)),
                              cos[:, :P], sin[:, :P], pam, cond=c.cond)
    del pm, pam; mx.clear_cache()
    # 後段不做 CFG 時，只留有條件那一列的 k/v
    caches1 = [[k[0:1], v[0:1]] for k, v in caches]
    kcfg = steps if cfg_steps is None else min(cfg_steps, steps)
    log(f"    序列 {c.seq_len}，前綴 {P}"
        f"{'（含參考圖 %d 個 token）' % (len(c.cond.vae_index[0]) + len(c.cond.vit_index[0])) if c.cond else ''}"
        f"，預填 {time.time()-t0:.0f}s")
    cos_ts, sin_ts = cos[:, P:P + 1], sin[:, P:P + 1]
    cos_img, sin_img = cos[:, istart:iend], sin[:, istart:iend]

    rs = np.random.RandomState(seed)
    x = mx.array(rs.randn(1, m.latent_channels, c.token_h, c.token_w).astype(np.float32))
    sigmas, timesteps = sigma_schedule(steps, shift)
    t0 = time.time()
    for i in range(steps):
        tt = float(timesteps[i])
        if i < kcfg:
            p = m.image_step(caches, mx.repeat(x, 2, axis=0), mx.full((2,), tt),
                             cos_ts, sin_ts, cos_img, sin_img,
                             c.token_h, c.token_w).astype(mx.float32)
            v = apply_cfg(p[0:1], p[1:2], guidance)
        else:
            v = m.image_step(caches1, x, mx.full((1,), tt),
                             cos_ts[0:1], sin_ts[0:1], cos_img[0:1], sin_img[0:1],
                             c.token_h, c.token_w).astype(mx.float32)
        x = x + v * float(sigmas[i + 1] - sigmas[i])
        mx.eval(x); mx.clear_cache()
    dt = time.time() - t0
    del caches, caches1; mx.clear_cache()
    if dec is None:
        from hymlx.vae import Decoder, load_decoder
        dec = Decoder(vc)
        load_decoder(dec, mx.load(str(model_dir / "vae.safetensors")), dtype=mx.bfloat16)
        mx.eval(dec.parameters())
    img = np.array(decode_image(dec, (x / vc.scaling_factor).astype(mx.bfloat16)
                                ).astype(mx.float32), copy=False)[0, :, -1]
    img = np.clip((img + 1) / 2, 0, 1).transpose(1, 2, 0)
    Image.fromarray((img * 255).round().astype(np.uint8)).save(out_path)
    log(f"    {steps} 步 {dt:.0f}s（{dt/steps:.1f}s/步）-> {Path(out_path).name}")
    return dt


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from hymlx.conditioning import Conditioner
    from hymlx.hunyuan import QuantizedHunyuan
    from hymlx.vae import Decoder, VAEConfig, load_decoder

    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="+")
    ap.add_argument("--model", default=str(Path.home() / "models/hymlx-8bit"))
    ap.add_argument("--outdir", default=str(Path.home() / "models/hymlx-out"))
    ap.add_argument("--names", default=None, help="逗號分隔的輸出檔名（不含副檔名）")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system-prompt", default="en_unified")
    ap.add_argument("--no-cot", action="store_true")
    ap.add_argument("--cot-file", default=None, help="重用先前產生的 CoT，省掉文字階段")
    ap.add_argument("--requant", default=None,
                    help="載入時即時重量化，不必另外產生檔案。"
                         "例如 '4,64' 全部降 4-bit；'mlp.experts=4,64' 只降專家。")
    ap.add_argument("--image", default=None,
                    help="參考圖（編輯）。同一張圖會同時走 VAE 與 ViT 兩條路。")
    ap.add_argument("--cfg-steps", type=int, default=None,
                    help="只有前 N 步做 CFG，後面用有條件那條就好。"
                         "14 步時給 6 大約省 20%%，圖幾乎一樣。")
    a = ap.parse_args()

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    names = a.names.split(",") if a.names else [f"dream_{i}" for i in range(len(a.prompts))]
    t0 = time.time()
    cond = Conditioner()
    rq = None
    if a.requant:
        rq = {}
        for part in a.requant.split(";"):
            key, _, spec = part.rpartition("=")
            b, g = (int(v) for v in spec.split(","))
            rq[key] = (b, g)
    m = QuantizedHunyuan(a.model, requant=rq)
    vcfg = json.load(open(Path(a.model) / "config.json"))["vae"]
    K = ("in_channels", "out_channels", "latent_channels", "block_out_channels",
         "layers_per_block", "ffactor_spatial", "ffactor_temporal",
         "upsample_match_channel", "downsample_match_channel", "scaling_factor")
    vc = VAEConfig(**{k: vcfg[k] for k in K})
    dec = None      # 解碼器最後才用，先不佔記憶體
    sp = None if a.system_prompt == "none" else cond.system_prompt(a.system_prompt,
                                                                   "think_recaption")
    print(f"載入 {time.time()-t0:.0f}s\n")

    for prompt, name in zip(a.prompts, names):
        print(f"[{name}] {prompt}", flush=True)
        t_sub = time.time()
        if a.cot_file:
            cot = Path(a.cot_file).read_text()
        elif a.no_cot:
            cot = None
        else:
            cot = generate_cot(m, cond, prompt, sp, seed=a.seed, image=a.image)
        if cot:
            (outdir / f"{name}_cot.txt").write_text(cot)
        render(m, cond, dec, vc, prompt, cot, sp, a.size, a.steps, a.guidance,
               a.shift, a.seed, outdir / f"{name}.png", cfg_steps=a.cfg_steps,
               image=a.image, model_dir=Path(a.model))
        print(f"    小計 {time.time()-t_sub:.0f}s\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
