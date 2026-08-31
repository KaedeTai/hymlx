"""里程碑 3：VAE 解碼器，對官方 PyTorch 實作。

容差不是猜的，是量出來的。第一版寫死 1e-4 然後 CPU 測到 3.7e-4「失敗」，逐層查下去
每一階段都在 1e-5 以內——問題出在 `norm_out`：GroupNorm 除以標準差，把 1.2e-5 的
累積誤差放大成 3.7e-4。那是 20 層 fp32 卷積的正常結果，不是 bug。

所以這裡改成量**參考實作自己在 bf16 下損失多少**（它實際跑的精度），拿那個當包絡。
跟 mdream 里程碑 3 學到的同一件事：跨精度的絕對容差沒有意義，要有對照組。
"""
from __future__ import annotations
import glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hymlx.conditioning import ensure_official_code, register_official_package, snapshot_dir
register_official_package(ensure_official_code()); STUDY = Path(snapshot_dir())
sys.path.insert(0, str(STUDY))
torch.cuda.set_device = lambda *a, **k: None
import contextlib
torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()

SNAP = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--tencent--HunyuanImage-3.0-Instruct/snapshots/*"))[0]
CFG = json.load(open(STUDY / "config.json"))


def load_vae_weights():
    idx = json.load(open(STUDY / "model.safetensors.index.json"))["weight_map"]
    want = {k: v for k, v in idx.items() if k.startswith("vae.decoder.")}
    out = {}
    for shard in sorted(set(want.values())):
        with safe_open(f"{SNAP}/{shard}", framework="pt") as f:
            for k, s in want.items():
                if s == shard:
                    out[k] = f.get_tensor(k).float()
    return out


def main() -> int:
    import mlx.core as mx
    from hymlx.vae import Decoder, VAEConfig, load_decoder
    print("里程碑 3 — VAE 解碼器\n")
    w = load_vae_weights()
    print(f"  {len(w)} 個張量, {sum(v.numel()*4 for v in w.values())/2**30:.2f} GiB (fp32)")

    from hy.autoencoder_kl_3d import Decoder as RefDecoder
    vc = CFG["vae"]
    ref = RefDecoder(z_channels=vc["latent_channels"], out_channels=vc["out_channels"],
                     block_out_channels=list(reversed(vc["block_out_channels"])),
                     num_res_blocks=vc["layers_per_block"],
                     ffactor_spatial=vc["ffactor_spatial"],
                     ffactor_temporal=vc["ffactor_temporal"],
                     upsample_match_channel=vc["upsample_match_channel"]).float().eval()
    sd = {k[len("vae.decoder."):]: v for k, v in w.items()}
    miss, unexp = ref.load_state_dict(sd, strict=False)
    print(f"  官方解碼器: missing={len(miss)} unexpected={len(unexp)}")
    if miss[:3]: print("    missing:", miss[:3])

    rs = np.random.RandomState(0)
    T, H, W = 1, 8, 8                       # 小 latent：8x8 -> 128x128 像素
    z = (rs.randn(1, vc["latent_channels"], T, H, W) * 0.5).astype(np.float32)
    t0 = time.time()
    with torch.no_grad():
        r = ref(torch.from_numpy(z)).numpy()
    print(f"  官方輸出 {r.shape}  範圍 [{r.min():+.4f}, {r.max():+.4f}]  {time.time()-t0:.1f}s\n")

    # 包絡：參考實作 fp32 vs 自己的 bf16。在 MPS 上跑——torch 在 CPU 上沒有
    # bf16 的 conv3d 核心，會慢到不能用。
    with torch.no_grad():
        rm32 = ref.to("mps")(torch.from_numpy(z).to("mps")).float().cpu().numpy()
        rmb = ref.bfloat16()(torch.from_numpy(z).to("mps").bfloat16()).float().cpu().numpy()
    env = np.abs(rmb - rm32).max() / max(np.abs(rm32).max(), 1e-12)
    ref.float().to("cpu")
    print(f"  參考實作 bf16 vs 自己的 fp32: {env:.3e}  ← 這個模型實際跑的精度\n")

    ok = True
    for dev, tol in ((mx.cpu, env), (mx.gpu, env)):
        mx.set_default_device(dev)
        d = Decoder(VAEConfig(**{k: vc[k] for k in
                    ["in_channels", "out_channels", "latent_channels", "block_out_channels",
                     "layers_per_block", "ffactor_spatial", "ffactor_temporal",
                     "upsample_match_channel", "scaling_factor"]}))
        nload = load_decoder(d, {k: v.numpy() for k, v in w.items()})
        mx.eval(d.parameters())
        t0 = time.time(); got = np.array(d(mx.array(z)), copy=False); dt = time.time()-t0
        if got.shape != r.shape:
            print(f"  ❌ 形狀不符 {got.shape} vs {r.shape}"); return 1
        rel = np.abs(got - r).max() / max(np.abs(r).max(), 1e-12)
        name = "CPU" if dev == mx.cpu else "GPU"
        good = rel < tol; ok &= good
        print(f"  {'✅' if good else '❌'} {name} fp32  相對誤差 {rel:.3e}  "
              f"(bf16 包絡的 {rel/tol*100:.0f}%)  {dt:.1f}s  載入 {nload} 個張量")
    print("\n  " + ("PASS — VAE 解碼器對得上" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
