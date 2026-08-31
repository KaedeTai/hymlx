"""里程碑 8：patch_embed / final_layer / timestep embedder 對官方實作。"""
from __future__ import annotations
import contextlib, glob, json, sys
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


def envelope(mod, args, ref32):
    """量官方實作自己在 bf16（MPS）下的損失，當作容差上限。"""
    m = mod.to("mps")
    a32 = [a.to("mps") if torch.is_tensor(a) else a for a in args]
    ab = [a.to("mps").bfloat16() if torch.is_tensor(a) else a for a in args]
    with torch.no_grad():
        o32 = m(*a32)
        ob = m.bfloat16()(*ab)
    o32 = (o32[0] if isinstance(o32, tuple) else o32).float().cpu().numpy()
    ob = (ob[0] if isinstance(ob, tuple) else ob).float().cpu().numpy()
    mod.float().to("cpu")
    return rel(ob, o32)


def main() -> int:
    import mlx.core as mx
    from hymlx.imageio import (UNetDown, UNetUp, TimestepEmbedder,
                               load_unet_down, load_unet_up, load_timestep_embedder)
    from hy.modeling_hunyuan_image_3 import UNetDown as RefDown, UNetUp as RefUp, \
        TimestepEmbedder as RefT
    print("里程碑 8 — patch_embed / final_layer / timestep\n")

    H = CFG["hidden_size"]; LC = CFG["vae"]["latent_channels"]; PH = 1024
    w = load(["patch_embed.", "final_layer.", "time_embed.", "time_embed_2.", "timestep_emb."])
    wn = {k: v.numpy() for k, v in w.items()}
    print(f"  {len(w)} 個張量")
    rs = np.random.RandomState(3)
    t = np.array([0.37, 0.91], dtype=np.float32)
    lat = (rs.randn(2, LC, 8, 8) * 0.5).astype(np.float32)
    ok = True

    # --- timestep embedder ---
    rt = RefT(hidden_size=H).float().eval()
    rt.load_state_dict({k[len("time_embed."):]: v for k, v in w.items() if k.startswith("time_embed.")})
    with torch.no_grad():
        r_emb = rt(torch.from_numpy(t)).numpy()
    for dev in (mx.cpu, mx.gpu):
        mx.set_default_device(dev)
        m = TimestepEmbedder(H); n = load_timestep_embedder(m, wn, "time_embed."); mx.eval(m.parameters())
        e = rel(m(mx.array(t)), r_emb); tol = 1e-5 if dev == mx.cpu else 1e-3
        good = e < tol; ok &= good
        print(f"  {'✅' if good else '❌'} {'CPU' if dev==mx.cpu else 'GPU'} time_embed      "
              f"誤差 {e:.3e} (容差 {tol:.0e})  {n} 個張量")

    emb = torch.from_numpy(r_emb)

    # --- patch_embed (UNetDown) ---
    rd = RefDown(patch_size=1, in_channels=LC, emb_channels=H, hidden_channels=PH, out_channels=H).float().eval()
    rd.load_state_dict({k[len("patch_embed."):]: v for k, v in w.items() if k.startswith("patch_embed.")})
    with torch.no_grad():
        r_tok, th, tw = rd(torch.from_numpy(lat), emb)
    r_tok = r_tok.numpy()
    env_d = envelope(rd, [torch.from_numpy(lat), emb], r_tok)
    for dev in (mx.cpu, mx.gpu):
        mx.set_default_device(dev)
        m = UNetDown(LC, H, PH, H); n = load_unet_down(m, wn, "patch_embed."); mx.eval(m.parameters())
        got, gh, gw = m(mx.array(lat), mx.array(r_emb))
        e = rel(got, r_tok); good = e < env_d and (gh, gw) == (th, tw); ok &= good
        print(f"  {'✅' if good else '❌'} {'CPU' if dev==mx.cpu else 'GPU'} patch_embed     "
              f"誤差 {e:.3e} (bf16 包絡 {env_d:.3e})  token {gh}x{gw}  {n} 個張量")

    # --- final_layer (UNetUp) ---
    ru = RefUp(patch_size=1, in_channels=H, emb_channels=H, hidden_channels=PH,
               out_channels=LC, out_norm=True).float().eval()
    ru.load_state_dict({k[len("final_layer."):]: v for k, v in w.items() if k.startswith("final_layer.")})
    tok = torch.from_numpy((rs.randn(2, th * tw, H) * 0.5).astype(np.float32))
    with torch.no_grad():
        r_lat = ru(tok, emb, th, tw).numpy()
    env_u = envelope(ru, [tok, emb, th, tw], r_lat)
    for dev in (mx.cpu, mx.gpu):
        mx.set_default_device(dev)
        m = UNetUp(H, H, PH, LC); n = load_unet_up(m, wn, "final_layer."); mx.eval(m.parameters())
        got = np.array(m(mx.array(tok.numpy()), mx.array(r_emb), th, tw), copy=False)
        e = rel(got, r_lat); good = e < env_u and got.shape == r_lat.shape; ok &= good
        print(f"  {'✅' if good else '❌'} {'CPU' if dev==mx.cpu else 'GPU'} final_layer     "
              f"誤差 {e:.3e} (bf16 包絡 {env_u:.3e})  {got.shape}  {n} 個張量")

    print("\n  " + ("PASS — latent <-> token 的接口對得上" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
