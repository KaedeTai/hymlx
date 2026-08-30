"""里程碑 4：VAE 編碼器（參考圖用）。容差同樣量參考實作自己的 bf16 包絡。"""
from __future__ import annotations
import glob, json, sys, contextlib
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
STUDY = Path.home() / "repos/hunyuan-study"; sys.path.insert(0, str(STUDY))
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()
SNAP = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--tencent--HunyuanImage-3.0-Instruct/snapshots/*"))[0]
CFG = json.load(open(STUDY / "config.json")); vc = CFG["vae"]

def load(prefix):
    idx = json.load(open(STUDY / "model.safetensors.index.json"))["weight_map"]
    want = {k: v for k, v in idx.items() if k.startswith(prefix)}
    out = {}
    for sh in sorted(set(want.values())):
        with safe_open(f"{SNAP}/{sh}", framework="pt") as f:
            for k, s in want.items():
                if s == sh: out[k] = f.get_tensor(k).float()
    return out

def main() -> int:
    import mlx.core as mx
    from hymlx.vae import Encoder, VAEConfig, load_encoder
    from hy.autoencoder_kl_3d import Encoder as RefEncoder
    print("里程碑 4 — VAE 編碼器\n")
    w = load("vae.encoder.")
    print(f"  {len(w)} 個張量")
    ref = RefEncoder(in_channels=vc["in_channels"], z_channels=vc["latent_channels"],
                     block_out_channels=vc["block_out_channels"],
                     num_res_blocks=vc["layers_per_block"],
                     ffactor_spatial=vc["ffactor_spatial"], ffactor_temporal=vc["ffactor_temporal"],
                     downsample_match_channel=vc["downsample_match_channel"]).float().eval()
    miss, unexp = ref.load_state_dict({k[len("vae.encoder."):]: v for k, v in w.items()}, strict=False)
    print(f"  官方編碼器: missing={len(miss)} unexpected={len(unexp)}")
    if miss[:3]: print("    missing:", miss[:3])

    rs = np.random.RandomState(1)
    x = (rs.rand(1, 3, 4, 128, 128).astype(np.float32) * 2 - 1)   # T=4 -> latent T=1
    with torch.no_grad():
        r = ref(torch.from_numpy(x)).numpy()
        rm32 = ref.to("mps")(torch.from_numpy(x).to("mps")).float().cpu().numpy()
        rmb = ref.bfloat16()(torch.from_numpy(x).to("mps").bfloat16()).float().cpu().numpy()
    ref.float().to("cpu")
    env = np.abs(rmb - rm32).max() / max(np.abs(rm32).max(), 1e-12)
    print(f"  官方輸出 {r.shape}  範圍 [{r.min():+.3f}, {r.max():+.3f}]")
    print(f"  參考實作 bf16 包絡: {env:.3e}\n")

    ok = True
    cfgk = ["in_channels","out_channels","latent_channels","block_out_channels","layers_per_block",
            "ffactor_spatial","ffactor_temporal","upsample_match_channel","downsample_match_channel","scaling_factor"]
    for dev in (mx.cpu, mx.gpu):
        mx.set_default_device(dev)
        e = Encoder(VAEConfig(**{k: vc[k] for k in cfgk}))
        n = load_encoder(e, {k: v.numpy() for k, v in w.items()}); mx.eval(e.parameters())
        got = np.array(e(mx.array(x)), copy=False)
        if got.shape != r.shape:
            print(f"  ❌ 形狀 {got.shape} vs {r.shape}"); return 1
        rel = np.abs(got - r).max() / max(np.abs(r).max(), 1e-12)
        good = rel < env; ok &= good
        print(f"  {'✅' if good else '❌'} {'CPU' if dev==mx.cpu else 'GPU'} fp32  相對誤差 {rel:.3e}  "
              f"(bf16 包絡的 {rel/env*100:.0f}%)  載入 {n} 個張量")
    print("\n  " + ("PASS — VAE 編碼器對得上" if ok else "FAIL"))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
