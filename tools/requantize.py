"""把 8-bit 的檢查點就地降到 6-bit，不必重下原始權重。

代價：多一次量化。實測在真實權重上只比「從 bf16 直接做 6-bit」差 3.2%
（理論值 sqrt(1 + (1/4)^2) = 3.1%，因為 8-bit 的步階是 6-bit 的 1/4），
換掉 25 分鐘的重新下載，划算。

保持 bf16 的那些（norm、mlp.gate.wg、patch_embed、final_layer、timestep、VAE、視覺塔）
原樣搬過去。
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx

CHUNK = 8          # 專家一次處理幾顆，控制峰值記憶體


def requant(w, s, b, src_bits, dst_bits, gs):
    """解量化再量化。專家堆很大（64 x 6144 x 4096），分塊處理。"""
    if w.ndim == 3 and w.shape[0] > CHUNK:
        outs = []
        for i in range(0, w.shape[0], CHUNK):
            d = mx.dequantize(w[i:i + CHUNK], s[i:i + CHUNK], b[i:i + CHUNK],
                              group_size=gs, bits=src_bits).astype(mx.float32)
            outs.append(mx.quantize(d, group_size=gs, bits=dst_bits))
            mx.eval(outs[-1])
            del d
        return tuple(mx.concatenate([o[j] for o in outs], axis=0) for j in range(3))
    d = mx.dequantize(w, s, b, group_size=gs, bits=src_bits).astype(mx.float32)
    return mx.quantize(d, group_size=gs, bits=dst_bits)


def convert_file(src: Path, dst: Path, src_bits: int, dst_bits: int, gs: int):
    d = mx.load(str(src))
    out, n = {}, 0
    names = {k[:-7] for k in d if k.endswith(".scales")}
    for k, v in d.items():
        base = k.rsplit(".", 1)[0]
        if base in names and k.endswith((".weight", ".scales", ".biases")):
            continue
        out[k] = v
    for base in sorted(names):
        w, s, b = requant(d[base + ".weight"], d[base + ".scales"], d[base + ".biases"],
                          src_bits, dst_bits, gs)
        out[base + ".weight"], out[base + ".scales"], out[base + ".biases"] = w, s, b
        n += 1
    mx.eval(list(out.values()))
    mx.save_safetensors(str(dst), out)
    del d, out
    mx.clear_cache()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "models/hymlx-8bit"))
    ap.add_argument("--out", default=str(Path.home() / "models/hymlx-6bit"))
    ap.add_argument("--bits", type=int, default=6)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qc = json.load(open(src / "quant_config.json"))
    sb, gs = qc["bits"], qc["group_size"]
    print(f"{sb}-bit -> {a.bits}-bit（group {gs}），來源 {src}")
    mx.set_default_device(mx.cpu)

    t_all = time.time()
    total = 0
    for f in sorted(src.glob("*.safetensors")):
        if f.name in ("vae.safetensors", "vision.safetensors"):
            shutil.copy(f, out / f.name)     # 本來就是 bf16，原樣搬
            sz = (out / f.name).stat().st_size
            total += sz
            print(f"  {f.name}: 原樣複製 {sz/2**30:.2f} GiB", flush=True)
            continue
        t0 = time.time()
        n = convert_file(f, out / f.name, sb, a.bits, gs)
        sz = (out / f.name).stat().st_size
        total += sz
        print(f"  {f.name}: 重量化 {n} 組 {f.stat().st_size/2**30:.2f} -> "
              f"{sz/2**30:.2f} GiB  {time.time()-t0:.0f}s", flush=True)

    meta = dict(qc); meta["bits"] = a.bits
    meta["derived_from"] = f"{sb}-bit（二次量化，實測比從 bf16 直接做差 3.2%）"
    (out / "quant_config.json").write_text(json.dumps(meta, indent=2))
    shutil.copy(src / "config.json", out / "config.json")
    print(f"\n  共 {total/2**30:.1f} GiB，{time.time()-t_all:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
