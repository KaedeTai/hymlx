"""里程碑 11：flow matching 取樣器對官方 FlowMatchDiscreteScheduler。

這裡的容差是 float32 的機器精度（1e-7），不是 bf16 包絡：整個排程只有 linspace、
一次除法和一次乘法，官方用 `torch.linspace` 我用 `np.linspace`，兩者在最後一位
可能差 1 ulp，這就是全部的誤差來源。
"""
from __future__ import annotations
import contextlib, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
STUDY = Path.home() / "repos/hunyuan-study"; sys.path.insert(0, str(STUDY))
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()


def main() -> int:
    import mlx.core as mx
    from hymlx.sampling import sigma_schedule, cfg, euler_sample
    from hy.hunyuan_image_3_pipeline import FlowMatchDiscreteScheduler, ClassifierFreeGuidance
    print("里程碑 11 — flow matching 取樣器\n")
    ok = True

    for n, sh in ((50, 3.0), (28, 3.0), (20, 1.0)):
        s = FlowMatchDiscreteScheduler(shift=sh, reverse=True, solver="euler")
        s.set_timesteps(n)
        ms, mt = sigma_schedule(n, sh)
        e1 = np.abs(s.sigmas.numpy() - ms).max()
        e2 = np.abs(s.timesteps.numpy() - mt).max() / 1000.0
        good = e1 < 2e-7 and e2 < 2e-7; ok &= good
        print(f"  {'✅' if good else '❌'} {n} 步 shift={sh}  sigma 差 {e1:.2e}  "
              f"timestep 相對差 {e2:.2e}   sigma[0..2]={ms[:3].round(4)}")

    # Euler 的一整條軌跡：拿同一串固定的 v 餵兩邊，走完 50 步比終點
    n = 50
    s = FlowMatchDiscreteScheduler(shift=3.0, reverse=True, solver="euler")
    s.set_timesteps(n)
    rs = np.random.RandomState(2)
    vs = [(rs.randn(1, 32, 8, 8) * 0.3).astype(np.float32) for _ in range(n)]
    x0 = (rs.randn(1, 32, 8, 8)).astype(np.float32)

    xr = torch.from_numpy(x0.copy())
    for i, t in enumerate(s.timesteps):
        xr = s.step(torch.from_numpy(vs[i]), t, xr, return_dict=False)[0]

    xm = euler_sample(lambda x, t, i: mx.array(vs[i]), mx.array(x0), num_steps=n, shift=3.0)
    e = np.abs(np.array(xm, copy=False) - xr.numpy()).max() / max(np.abs(xr.numpy()).max(), 1e-12)
    good = e < 1e-6; ok &= good
    print(f"\n  {'✅' if good else '❌'} 50 步 Euler 軌跡終點  相對誤差 {e:.3e}")

    # CFG
    op = ClassifierFreeGuidance()
    a = (rs.randn(1, 32, 8, 8)).astype(np.float32); b = (rs.randn(1, 32, 8, 8)).astype(np.float32)
    r = op(torch.from_numpy(a), torch.from_numpy(b), 2.5, step=0).numpy()
    g = np.array(cfg(mx.array(a), mx.array(b), 2.5), copy=False)
    good = np.abs(g - r).max() < 1e-6; ok &= good
    print(f"  {'✅' if good else '❌'} CFG（scale 2.5）      最大差 {np.abs(g-r).max():.3e}")

    print("\n  " + ("PASS — 取樣器對得上" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
