"""把官方的 bf16 權重轉成 MLX 的量化檢查點。

一次處理一層，處理完就丟，所以峰值記憶體大約 10 GiB，不需要 157 GiB。
專家是一顆一顆量化再疊起來的——`mx.quantize` 的分組只沿最後一維，逐顆量化再
stack 跟先 stack 再量化在數值上完全相同，但前者的峰值記憶體小得多。

用法：
    python tools/convert.py --bits 4 --group-size 64 --out ~/models/hymlx-4bit
"""
from __future__ import annotations

import argparse
import glob
import os
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open


def snapshot() -> str:
    return glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                         "models--tencent--HunyuanImage-3.0-Instruct/snapshots/*"))[0]


class Reader:
    def __init__(self, snap: str, index: dict):
        self.snap, self.map, self.h = snap, index["weight_map"], {}

    def get(self, name: str, dtype=mx.bfloat16) -> mx.array:
        sh = self.map[name]
        if sh not in self.h:
            self.h[sh] = safe_open(f"{self.snap}/{sh}", framework="pt")
        t = self.h[sh].get_tensor(name)
        return mx.array(t.float().numpy()).astype(dtype)

    def close(self):
        self.h.clear()


def q(w: mx.array, bits: int, gs: int):
    return mx.quantize(w, group_size=gs, bits=bits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--out", type=str, default=str(Path.home() / "models/hymlx-4bit"))
    ap.add_argument("--layers", type=str, default=None, help="例如 0-3，只轉部分層（除錯用）")
    ap.add_argument("--prune-source", action="store_true",
                    help="轉完一層之後，把已經沒有人要用的原始 shard 刪掉。"
                         "磁碟塞不下 157 GiB 原檔加上輸出時用。任何一刻資料不是在原檔"
                         "就是在新檔，不會兩頭空。")
    a = ap.parse_args()

    study = Path.home() / "repos/hunyuan-study"
    snap = snapshot()
    cfg = json.load(open(study / "config.json"))
    index = json.load(open(study / "model.safetensors.index.json"))
    r = Reader(snap, index)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bits, gs = a.bits, a.group_size
    mx.set_default_device(mx.cpu)

    NE = cfg["num_experts"]
    NL = cfg["num_hidden_layers"]
    layer_ids = range(NL)
    if a.layers:
        lo, hi = (int(x) for x in a.layers.split("-"))
        layer_ids = range(lo, hi + 1)

    total_bytes = 0
    t_all = time.time()

    def free_disk_gib():
        st = os.statvfs(out)
        return st.f_bavail * st.f_frsize / 2 ** 30

    def prune(done_upto: int):
        """刪掉只服務 layer <= done_upto 的 shard。"""
        if not a.prune_source:
            return
        still_needed = set()
        for name, sh in index["weight_map"].items():
            top = name.split(".")[0]
            if top in ("vae", "vision_model", "vision_aligner", "patch_embed", "final_layer",
                       "time_embed", "time_embed_2", "timestep_emb", "lm_head") or name.startswith("model.wte") \
                    or name == "model.ln_f.weight":
                continue
            if name.startswith("model.layers."):
                li = int(name.split(".")[2])
                if li > done_upto:
                    still_needed.add(sh)
        freed = 0
        for sh in sorted(set(index["weight_map"].values())):
            if sh in still_needed:
                continue
            f = Path(snap) / sh
            if f.exists():
                real = f.resolve()
                sz = real.stat().st_size
                real.unlink(missing_ok=True)
                f.unlink(missing_ok=True)
                freed += sz
        if freed:
            print(f"    刪掉已用完的原始 shard {freed / 2**30:.1f} GiB，"
                  f"剩餘磁碟 {free_disk_gib():.0f} GiB", flush=True)

    # --- 頭尾：全部保持 bf16，這幾層是像素與 token 的介面，不能量 ---
    head = {}
    for name in index["weight_map"]:
        top = name.split(".")[0]
        if top in ("patch_embed", "final_layer", "time_embed", "time_embed_2", "timestep_emb"):
            head[name] = r.get(name)
        elif name == "model.ln_f.weight":
            head[name] = r.get(name)
    # wte 與 lm_head 量化（各 545M 參數，bf16 就要 1.1 GiB）
    for name in ("model.wte.weight", "lm_head.weight"):
        w, s, b = q(r.get(name), bits, gs)
        head[name] = w; head[name[:-7] + ".scales"] = s; head[name[:-7] + ".biases"] = b
    mx.eval(list(head.values()))
    mx.save_safetensors(str(out / "head.safetensors"), head)
    total_bytes += (out / "head.safetensors").stat().st_size
    print(f"  head: {len(head)} 個張量 "
          f"{(out / 'head.safetensors').stat().st_size / 2**30:.2f} GiB")
    del head

    # --- VAE 與視覺塔：bf16 原樣搬過來 ---
    for tag, prefixes in (("vae", ("vae.",)), ("vision", ("vision_model.", "vision_aligner."))):
        d = {n: r.get(n) for n in index["weight_map"] if n.startswith(prefixes)}
        mx.eval(list(d.values()))
        mx.save_safetensors(str(out / f"{tag}.safetensors"), d)
        sz = (out / f"{tag}.safetensors").stat().st_size
        total_bytes += sz
        print(f"  {tag}: {len(d)} 個張量 {sz / 2**30:.2f} GiB")
        del d

    # --- 每一層 ---
    for i in layer_ids:
        t0 = time.time()
        p = f"model.layers.{i}."
        d = {}
        for nm in ("self_attn.qkv_proj", "self_attn.o_proj",
                   "mlp.shared_mlp.gate_and_up_proj", "mlp.shared_mlp.down_proj"):
            w, s, b = q(r.get(p + nm + ".weight"), bits, gs)
            d[nm + ".weight"], d[nm + ".scales"], d[nm + ".biases"] = w, s, b
        for nm in ("self_attn.query_layernorm.weight", "self_attn.key_layernorm.weight",
                   "input_layernorm.weight", "post_attention_layernorm.weight"):
            d[nm] = r.get(p + nm)
        d["mlp.gate.wg.weight"] = r.get(p + "mlp.gate.wg.weight", mx.float32)

        stacks = {"up": [[], [], []], "gate": [[], [], []], "down": [[], [], []]}
        for e in range(NE):
            gu = r.get(p + f"mlp.experts.{e}.gate_and_up_proj.weight")
            up, gate = mx.split(gu, 2, axis=0)
            for nm, w in (("up", up), ("gate", gate),
                          ("down", r.get(p + f"mlp.experts.{e}.down_proj.weight"))):
                for j, t in enumerate(q(w, bits, gs)):
                    stacks[nm][j].append(t)
            mx.eval([t[-1] for v in stacks.values() for t in v])
        for nm, (ws, ss, bs) in stacks.items():
            d[f"mlp.experts.{nm}.weight"] = mx.stack(ws)
            d[f"mlp.experts.{nm}.scales"] = mx.stack(ss)
            d[f"mlp.experts.{nm}.biases"] = mx.stack(bs)
        del stacks
        mx.eval(list(d.values()))
        f = out / f"layer_{i:02d}.safetensors"
        mx.save_safetensors(str(f), d)
        sz = f.stat().st_size; total_bytes += sz
        print(f"  layer {i:>2}: {len(d)} 個張量 {sz / 2**30:.2f} GiB  {time.time()-t0:.0f}s"
              f"  磁碟剩 {free_disk_gib():.0f} GiB", flush=True)
        del d
        r.close()
        prune(i)

    meta = dict(bits=bits, group_size=gs, source="tencent/HunyuanImage-3.0-Instruct",
                num_hidden_layers=NL, num_experts=NE,
                keep_bf16=["patch_embed", "final_layer", "time_embed", "time_embed_2",
                           "timestep_emb", "norms", "mlp.gate.wg", "vae", "vision"])
    (out / "quant_config.json").write_text(json.dumps(meta, indent=2))
    shutil.copy(study / "config.json", out / "config.json")
    print(f"\n  共 {total_bytes / 2**30:.1f} GiB，{time.time()-t_all:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
