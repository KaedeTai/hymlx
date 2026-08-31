"""里程碑 12：量化的代價與速度。

問的問題是「4-bit 比 bf16 差多少」，不是「4-bit 離 fp32 多遠」——因為官方自己就是
用 bf16 跑的，bf16 的誤差是這個模型的自然本底。所以 bf16 那一列就是及格線，
量化的誤差只要落在同一個數量級就算堪用。
"""
from __future__ import annotations
import contextlib, gc, glob, json, sys, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hymlx.conditioning import ensure_official_code, register_official_package, snapshot_dir
register_official_package(ensure_official_code()); STUDY = Path(snapshot_dir())
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()
SNAP = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--tencent--HunyuanImage-3.0-Instruct/snapshots/*"))[0]
CFG = json.load(open(STUDY / "config.json"))
IDX = json.load(open(STUDY / "model.safetensors.index.json"))["weight_map"]
QDIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/hymlx-q-test")


def load_np(prefix):
    want = {k: v for k, v in IDX.items() if k.startswith(prefix)}
    out = {}
    for sh in sorted(set(want.values())):
        with safe_open(f"{SNAP}/{sh}", framework="pt") as f:
            for k, s in want.items():
                if s == sh:
                    out[k] = f.get_tensor(k).float().numpy()
    return out


def rel(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)


def main() -> int:
    import mlx.core as mx
    from hymlx.model import DecoderLayer, TextConfig, load_layer, load_layer_quantized
    from hymlx.quant import bits_per_param
    from hymlx.rope import build_batch_2d_rope, build_attention_mask
    print(f"里程碑 12 — 量化（來源 {QDIR}）\n")

    tc = TextConfig.from_json(CFG)
    S = 256
    info = [(slice(8, 8 + 196), (14, 14))]
    cos, sin = build_batch_2d_rope(S, CFG["attention_head_dim"], image_infos=[info],
                                   base=CFG["rope_theta"])
    mask = build_attention_mask(S, [info[0][0]])
    rs = np.random.RandomState(0)
    x = (rs.randn(1, S, CFG["hidden_size"]) * 0.02).astype(np.float32)

    mx.set_default_device(mx.gpu)
    pref = "model.layers.0."
    wn = load_np(pref)

    ref_layer = DecoderLayer(tc, 0)
    load_layer(ref_layer, wn, pref, CFG["num_experts"], mx.float32)
    mx.eval(ref_layer.parameters())
    r = np.array(ref_layer(mx.array(x), cos, sin, mask), copy=False)
    t0 = time.time()
    for _ in range(3):
        mx.eval(ref_layer(mx.array(x), cos, sin, mask))
    t_fp32 = (time.time() - t0) / 3
    del ref_layer; gc.collect()

    bf = DecoderLayer(tc, 0)
    load_layer(bf, wn, pref, CFG["num_experts"], mx.bfloat16)
    mx.eval(bf.parameters())
    g = np.array(bf(mx.array(x).astype(mx.bfloat16), cos.astype(mx.bfloat16),
                    sin.astype(mx.bfloat16), mask).astype(mx.float32), copy=False)
    e_bf = rel(g, r)
    t0 = time.time()
    for _ in range(3):
        mx.eval(bf(mx.array(x).astype(mx.bfloat16), cos.astype(mx.bfloat16),
                   sin.astype(mx.bfloat16), mask))
    t_bf = (time.time() - t0) / 3
    del bf, wn; gc.collect()
    print(f"  bf16（官方自己跑的精度）  相對誤差 {e_bf:.3e}   {t_bf*1000:.0f} ms/層")

    qcfg = json.load(open(QDIR / "quant_config.json"))
    bits, gs = qcfg["bits"], qcfg["group_size"]
    ql = DecoderLayer(tc, 0, (bits, gs))
    n = load_layer_quantized(ql, mx.load(str(QDIR / "layer_00.safetensors")))
    mx.eval(ql.parameters())
    gq = np.array(ql(mx.array(x), cos, sin, mask), copy=False)
    e_q = rel(gq, r)
    t0 = time.time()
    for _ in range(3):
        mx.eval(ql(mx.array(x), cos, sin, mask))
    t_q = (time.time() - t0) / 3
    sz = (QDIR / "layer_00.safetensors").stat().st_size / 2 ** 30
    print(f"  {bits}-bit / group {gs}       相對誤差 {e_q:.3e}   {t_q*1000:.0f} ms/層  "
          f"({n} 個張量, {sz:.2f} GiB/層, {bits_per_param(bits, gs):.2f} bit/參數)")
    print(f"  fp32 對照                                      {t_fp32*1000:.0f} ms/層\n")

    ratio = e_q / max(e_bf, 1e-12)
    ok = e_q < 20 * e_bf
    print(f"  {'✅' if ok else '❌'} 量化誤差是 bf16 本底的 {ratio:.1f} 倍"
          f"（門檻 20 倍以內）")
    print(f"     32 層估計 {sz * CFG['num_hidden_layers']:.1f} GiB，"
          f"加上頭尾與 VAE 約 {sz * CFG['num_hidden_layers'] + 4.4:.1f} GiB —— 放得進 128 GB")

    print("\n  " + ("PASS — 量化可用" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
