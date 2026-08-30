"""里程碑 7：SigLIP2 視覺塔 + aligner 對官方實作。"""
from __future__ import annotations
import contextlib, glob, json, sys
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
STUDY = Path.home() / "repos/hunyuan-study"; sys.path.insert(0, str(STUDY))
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
    import torch.nn.functional as F
    from hymlx.vision import (VisionConfig, VisionTower, Aligner, aa_weights,
                              load_vision, load_aligner)
    from hy.siglip2 import Siglip2VisionTransformer, LightProjector
    print("里程碑 7 — SigLIP2 + aligner\n")
    ok = True

    # --- 先單獨驗抗鋸齒插值權重，這是唯一沒有 MLX 對應算子的東西 ---
    worst = 0.0
    for out_hw in [(7, 11), (16, 16), (23, 5), (32, 32), (4, 40)]:
        P = 16
        src = np.random.RandomState(7).randn(1, 3, P, P).astype(np.float32)
        r = F.interpolate(torch.from_numpy(src), size=out_hw, mode="bilinear",
                          align_corners=False, antialias=True).numpy()
        wy, wx = aa_weights(P, out_hw[0]), aa_weights(P, out_hw[1])
        g = np.einsum("ia,jb,ncab->ncij", wy, wx, src)
        worst = max(worst, rel(g, r))
    good = worst < 1e-5; ok &= good
    print(f"  {'✅' if good else '❌'} antialias 雙線性插值      最差誤差 {worst:.3e}（5 種輸出尺寸）")

    w = load(["vision_model.", "vision_aligner."])
    wn = {k: v.numpy() for k, v in w.items()}
    print(f"  {len(w)} 個張量 ({sum(v.numel()*4 for v in w.values())/2**30:.2f} GiB fp32)")

    vcfg = VisionConfig.from_json(CFG["vit"])
    ref = Siglip2VisionTransformer(CFG["vit"]).float().eval()
    miss, unexp = ref.load_state_dict({k[len("vision_model."):]: v for k, v in w.items()
                                       if k.startswith("vision_model.")}, strict=False)
    print(f"  官方視覺塔: missing={len(miss)} unexpected={len(unexp)}")
    rp = LightProjector(CFG["vit_aligner"]).float().eval()
    rp.load_state_dict({k[len("vision_aligner."):]: v for k, v in w.items()
                        if k.startswith("vision_aligner.")})

    # 兩張不同格點的圖，尾巴補到同一個長度，順便測 attention_mask
    rs = np.random.RandomState(11)
    shapes = [(12, 9), (7, 5)]
    L = 128
    px = (rs.randn(2, L, 3 * 16 * 16) * 0.5).astype(np.float32)
    am = np.zeros((2, L), dtype=np.float32)
    for i, (h, ww_) in enumerate(shapes):
        am[i, :h * ww_] = 1.0
    ss = torch.tensor(shapes, dtype=torch.long)
    with torch.no_grad():
        r_last = ref(torch.from_numpy(px), attention_mask=torch.from_numpy(am),
                     spatial_shapes=ss).last_hidden_state
        r_proj = rp(r_last).numpy()
    r_last = r_last.numpy()

    # 參考實作自己的 bf16 損失當包絡
    with torch.no_grad():
        m32 = ref.to("mps")
        o32 = m32(torch.from_numpy(px).to("mps"), attention_mask=torch.from_numpy(am).to("mps"),
                  spatial_shapes=ss.to("mps")).last_hidden_state.float().cpu().numpy()
        ob = ref.bfloat16()(torch.from_numpy(px).to("mps").bfloat16(),
                            attention_mask=torch.from_numpy(am).to("mps").bfloat16(),
                            spatial_shapes=ss.to("mps")).last_hidden_state.float().cpu().numpy()
    ref.float().to("cpu")
    env = rel(ob, o32)
    print(f"  參考實作 bf16 包絡: {env:.3e}\n")

    for dev in (mx.cpu, mx.gpu):
        mx.set_default_device(dev)
        tower = VisionTower(vcfg); n = load_vision(tower, wn); mx.eval(tower.parameters())
        al = Aligner(**{k: CFG["vit_aligner"][k] for k in ("input_dim", "n_embed", "depth")})
        na = load_aligner(al, wn); mx.eval(al.parameters())
        got = tower(mx.array(px), shapes, mx.array(am))
        e1 = rel(got, r_last)
        e2 = rel(al(got), r_proj)
        good = e1 < env and e2 < env; ok &= good
        print(f"  {'✅' if good else '❌'} {'CPU' if dev==mx.cpu else 'GPU'}  視覺塔 {e1:.3e}  "
              f"+aligner {e2:.3e}   ({n}+{na} 個張量)")

    print("\n  " + ("PASS — 視覺塔對得上" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
