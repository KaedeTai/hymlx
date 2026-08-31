"""里程碑 6：完整 decoder。逐層權重 157 GiB，全部載進來對不了，所以策略是
**單層對到底、跨層抽驗**：layer 0 整層（attention + MoE + 兩個 norm + 殘差）
在 fp32 下對官方模組，另外抽 layer 31 只對 attention，確認逐層索引沒錯位。
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


def load(prefixes):
    want = {k: v for k, v in IDX.items() if any(k.startswith(p) for p in prefixes)}
    out = {}
    for sh in sorted(set(want.values())):
        with safe_open(f"{SNAP}/{sh}", framework="pt") as f:
            for k, s in want.items():
                if s == sh:
                    out[k] = f.get_tensor(k).float()
    return out


def rel(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)


def main() -> int:
    import mlx.core as mx
    from hymlx.model import TextConfig, DecoderLayer, load_layer
    from hymlx.rope import build_batch_2d_rope, build_attention_mask
    from hy.configuration_hunyuan_image_3 import HunyuanImage3Config
    from hy.modeling_hunyuan_image_3 import HunyuanImage3DecoderLayer
    print("里程碑 6 — decoder 層\n")

    tc = TextConfig.from_json(CFG)
    rcfg = HunyuanImage3Config(**CFG); rcfg._attn_implementation = "sdpa"

    # 一段文字 + 一張 6x6 的圖 + 一段文字，剛好把因果與全連通兩種遮罩都走到
    S, D = 64, CFG["attention_head_dim"]
    info = [(slice(8, 8 + 36), (6, 6))]
    cos, sin = build_batch_2d_rope(S, D, image_infos=[info], base=CFG["rope_theta"])
    mask = build_attention_mask(S, [info[0][0]])
    rcos = torch.from_numpy(np.array(cos, copy=False)); rsin = torch.from_numpy(np.array(sin, copy=False))
    rmask = torch.from_numpy(np.array(mask, copy=False))
    rs = np.random.RandomState(0)
    x = (rs.randn(1, S, CFG["hidden_size"]) * 0.02).astype(np.float32)

    ok = True
    for li, full in ((0, True), (31, False)):
        pref = f"model.layers.{li}."
        want = [pref] if full else [pref + "self_attn.", pref + "input_layernorm.",
                                    pref + "post_attention_layernorm."]
        t0 = time.time(); w = load(want)
        gib = sum(v.numel() * 4 for v in w.values()) / 2 ** 30
        print(f"  layer {li}: {len(w)} 個張量 ({gib:.2f} GiB fp32) 載入 {time.time()-t0:.1f}s")

        ref = HunyuanImage3DecoderLayer(rcfg, layer_idx=li).float().eval()
        miss, unexp = ref.load_state_dict({k[len(pref):]: v for k, v in w.items()}, strict=False)
        miss = [m for m in miss if full or m.startswith("mlp.")]
        print(f"    官方模組: missing={len(miss)} unexpected={len(unexp)}")
        with torch.no_grad():
            if full:
                r = ref(torch.from_numpy(x), attention_mask=rmask, custom_pos_emb=(rcos, rsin))[0].numpy()
            else:
                r = ref.self_attn(ref.input_layernorm(torch.from_numpy(x)), attention_mask=rmask,
                                  custom_pos_emb=(rcos, rsin))[0].numpy()
        del ref; gc.collect()

        wn = {k: v.numpy() for k, v in w.items()}
        del w; gc.collect()
        for dev in (mx.cpu, mx.gpu):
            mx.set_default_device(dev)
            lyr = DecoderLayer(tc, li)
            n = load_layer(lyr, wn, pref, CFG["num_experts"]) if full else None
            if not full:
                from hymlx.model import _lin
                a = lyr.self_attn
                _lin(a.qkv_proj, wn, pref + "self_attn.qkv_proj", mx.float32)
                _lin(a.o_proj, wn, pref + "self_attn.o_proj", mx.float32)
                a.query_layernorm.weight = mx.array(wn[pref + "self_attn.query_layernorm.weight"])
                a.key_layernorm.weight = mx.array(wn[pref + "self_attn.key_layernorm.weight"])
                lyr.input_layernorm.weight = mx.array(wn[pref + "input_layernorm.weight"])
                n = 6
            mx.eval(lyr.parameters())
            t0 = time.time()
            if full:
                got = np.array(lyr(mx.array(x), cos, sin, mask), copy=False)
            else:
                got = np.array(lyr.self_attn(lyr.input_layernorm(mx.array(x)), cos, sin, mask), copy=False)
            dt = time.time() - t0
            e = rel(got, r)
            tol = 1e-5 if dev == mx.cpu else 2e-3
            good = e < tol and got.shape == r.shape; ok &= good
            what = "整層" if full else "只有 attention"
            print(f"    {'✅' if good else '❌'} {'CPU' if dev==mx.cpu else 'GPU'} {what:<14} "
                  f"誤差 {e:.3e} (容差 {tol:.0e})  載入 {n} 個張量  {dt:.2f}s")
            del lyr; gc.collect()
        del wn; gc.collect()

    print("\n  " + ("PASS — decoder 層對得上" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
