"""里程碑 5：2D RoPE 與注意力遮罩，對官方實作。這一段是純索引與三角函數，
沒有 GEMM，所以容差直接用 fp32 的機器精度，不需要 bf16 包絡。"""
from __future__ import annotations
import contextlib, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hymlx.conditioning import ensure_official_code, register_official_package, snapshot_dir
register_official_package(ensure_official_code()); STUDY = Path(snapshot_dir())
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()
CFG = json.load(open(STUDY / "config.json"))


def rel(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)


def main() -> int:
    import mlx.core as mx
    from hymlx import rope as R
    from hy.modeling_hunyuan_image_3 import build_batch_2d_rope as ref_rope, apply_rotary_pos_emb as ref_apply
    print("里程碑 5 — 2D RoPE + 注意力遮罩\n")

    D = CFG["attention_head_dim"]; base = CFG["rope_theta"]
    ok = True
    cases = [
        ("純文字", 128, None),
        ("一張 16x16 圖 @20", 300, [(slice(20, 20 + 256), (16, 16))]),
        ("圖後接重疊控制 token", 400, [(slice(4, 260), (16, 16)), (slice(260, 264), (None, None))]),
        ("兩張圖", 700, [(slice(4, 4 + 256), (16, 16)), (slice(266, 266 + 400), (20, 20))]),
    ]
    for name, S, info in cases:
        rc, rs = ref_rope(S, D, image_infos=[info], base=base)
        gc, gs = R.build_batch_2d_rope(S, D, image_infos=[info], base=base)
        e = max(rel(gc, rc.numpy()), rel(gs, rs.numpy()))
        good = e < 1e-6 and tuple(gc.shape) == tuple(rc.shape); ok &= good
        print(f"  {'✅' if good else '❌'} {name:<22} cos/sin {tuple(gc.shape)}  誤差 {e:.3e}")

    # 位置整數本身要完全一致
    _, _, allpos = ref_rope(300, D, image_infos=[[(slice(20, 20 + 256), (16, 16))]],
                            base=base, return_all_pos=True)
    mine = R.rope_positions(300, [(slice(20, 20 + 256), (16, 16))])
    exact = np.array_equal(mine, allpos[0].squeeze(1).numpy()); ok &= exact
    print(f"  {'✅' if exact else '❌'} 位置索引逐一相同 (y, x)")

    # apply
    rs_ = np.random.RandomState(0)
    B, H, S = 1, 4, 300
    q = rs_.randn(B, H, S, D).astype(np.float32); k = rs_.randn(B, H, S, D).astype(np.float32)
    rc, rsn = ref_rope(S, D, image_infos=[[(slice(20, 20 + 256), (16, 16))]], base=base)
    rq, rk = ref_apply(torch.from_numpy(q), torch.from_numpy(k), rc, rsn)
    gc, gsn = R.build_batch_2d_rope(S, D, image_infos=[[(slice(20, 20 + 256), (16, 16))]], base=base)
    mq, mk = R.apply_rope(mx.array(q), mx.array(k), gc, gsn)
    e = max(rel(mq, rq.numpy()), rel(mk, rk.numpy()))
    good = e < 1e-6; ok &= good
    print(f"  {'✅' if good else '❌'} apply_rope             誤差 {e:.3e}")

    # 遮罩
    S = 300; sl = slice(20, 20 + 256)
    ref_m = torch.ones(S, S, dtype=torch.bool).tril(0).repeat(1, 1, 1)
    ref_m[0, sl, sl] = True
    ref_m = ref_m.unsqueeze(1)
    mine_m = np.array(R.build_attention_mask(S, [sl]), copy=False)
    same = np.array_equal(mine_m, ref_m.numpy()); ok &= same
    print(f"  {'✅' if same else '❌'} 遮罩 {mine_m.shape} 逐一相同  "
          f"(圖像方塊全開 {mine_m[0,0,sl,sl].all()}，圖像看不到後面文字 {not mine_m[0,0,25,290]})")

    print("\n  " + ("PASS — RoPE 與遮罩對得上" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
