"""產生官方要的 CoT：<think>...</think><recaption>...</recaption>。

官方的圖像階段不是直接吃使用者的 prompt，而是先跑一次**文字**生成，讓模型自己寫一段
think + recaption，再把那段文字放進圖像序列裡。我一直手寫 recaption，那是分布外。

順帶一提，這也是整個移植最鋒利的一個測試：模型如果能吐出通順的英文，
attention、RoPE、MoE 路由、lm_head 就都是對的——亂碼騙不了人。

沒有 KV cache，每一個 token 都重算整條序列。文字階段只有幾百個 token，一次前向約 1 秒。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from hymlx.conditioning import Conditioner
    from hymlx.hunyuan import QuantizedHunyuan
    from hymlx.rope import build_batch_2d_rope, build_attention_mask

    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default=str(Path.home() / "models/hymlx-8bit"))
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system-prompt", default="en_unified")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    cond = Conditioner()
    tk = cond._tokenizer
    sp = cond.system_prompt(a.system_prompt, "think_recaption")
    toks, _ = cond.build_text(a.prompt, system_prompt=sp, bot_task="think")
    toks = list(toks[0])
    print(f"提示序列 {len(toks)} token，系統提示詞 {a.system_prompt}")

    m = QuantizedHunyuan(a.model)
    D = m.head_dim
    end_think = tk.end_of_think_token_id
    end_recap = tk.end_of_recaption_token_id
    recap_tok = tk.convert_tokens_to_ids(tk.recaption_token)
    rng = np.random.RandomState(a.seed)

    t0 = time.time()
    stage = "think"
    caches = m.new_caches()
    prompt_len = len(toks)
    # 預填：整條提示序列走一次因果注意力，把每一層的 k/v 存起來
    S = len(toks)
    cos, sin = build_batch_2d_rope(S, D, image_infos=[None], base=m.rope_theta)
    amask = mx.where(build_attention_mask(S), mx.array(0.0, mx.float32),
                     mx.array(-3.4028235e38, mx.float32))
    lg = m.logits(mx.array(np.array(toks, dtype=np.int32)[None]), cos, sin, amask, caches)
    print(f"  預填 {S} token 用了 {time.time()-t0:.0f}s", flush=True)

    # 增量解碼用的 RoPE 表：純文字沒有影像，位置就是 0,1,2,...
    MAXPOS = S + a.max_new + 8
    cos_all, sin_all = build_batch_2d_rope(MAXPOS, D, image_infos=[None], base=m.rope_theta)

    def sample(logits):
        lg = np.array(logits.astype(mx.float32), copy=False)[0] / max(a.temperature, 1e-6)
        idx = np.argsort(-lg)[:a.top_k]
        pr = np.exp(lg[idx] - lg[idx].max()); pr /= pr.sum()
        keep = int(np.searchsorted(np.cumsum(pr), a.top_p)) + 1
        idx, pr = idx[:keep], pr[:keep] / pr[:keep].sum()
        return int(rng.choice(idx, p=pr))

    def step(tid):
        pos = len(toks)
        out = m.logits(mx.array(np.array([[tid]], dtype=np.int32)),
                       cos_all[:, pos:pos + 1], sin_all[:, pos:pos + 1], None, caches)
        toks.append(tid)
        return out

    nxt = sample(lg)
    for n in range(a.max_new):
        lg = step(nxt)
        if nxt == end_think and stage == "think":
            lg = step(recap_tok)          # 官方的 stage transition
            stage = "recaption"
        if nxt == end_recap:
            break
        nxt = sample(lg)
        if n % 40 == 0:
            print(f"  {n:>3} token，{time.time()-t0:.0f}s，"
                  f"…{tk.decode(toks[-24:])[-70:]!r}", flush=True)

    cot = tk.decode(toks[prompt_len:])
    print(f"\n產生 {len(toks)-prompt_len} 個 token，{time.time()-t0:.0f}s\n")
    print("=" * 70)
    print(tk.think_token + cot)
    print("=" * 70)
    if a.out:
        Path(a.out).write_text(tk.think_token + cot)
        print(f"寫出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
