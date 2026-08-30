"""VAE 端到端來回：真圖 -> latent -> 真圖。

模組層級的測試（test_vae_decoder / test_vae_encoder）逐張量對得上，卻抓不到這個錯：
**T=1 的 latent 解碼出來是 4 幀，官方 `AutoencoderKLConv3D.decode` 取的是最後一幀**
（`decoded[:, :, -1:]`），那一行在 wrapper 裡，不在 `Decoder` 模組裡。
取第一幀的圖看起來像模型壞掉——16 px 的格子、沒有內容。

教訓：對到模組還不夠，wrapper 裡的那幾行也要對。端到端的來回是最便宜的保險。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SRC = Path.home() / "models/mdream-out/age_comfy_1152.png"
MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "models/hymlx-4bit"))


def main() -> int:
    import mlx.core as mx
    from PIL import Image
    from hymlx.vae import (Decoder, Encoder, VAEConfig, decode_image,
                           load_decoder, load_encoder)
    print("VAE 端到端來回\n")
    if not SRC.exists():
        print(f"  找不到測試圖 {SRC}"); return 1

    vcfg = json.load(open(MODEL / "config.json"))["vae"]
    K = ("in_channels", "out_channels", "latent_channels", "block_out_channels",
         "layers_per_block", "ffactor_spatial", "ffactor_temporal",
         "upsample_match_channel", "downsample_match_channel", "scaling_factor")
    vc = VAEConfig(**{k: vcfg[k] for k in K})
    w = mx.load(str(MODEL / "vae.safetensors"))
    enc = Encoder(vc); load_encoder(enc, w)
    dec = Decoder(vc); load_decoder(dec, w)
    mx.eval(enc.parameters()); mx.eval(dec.parameters())

    im = Image.open(SRC).convert("RGB").resize((1024, 1024), Image.LANCZOS)
    orig = np.asarray(im).astype(np.float32) / 255.0
    a = (orig * 2 - 1).transpose(2, 0, 1)[None, :, None]
    a = np.repeat(a, vcfg["ffactor_temporal"], axis=2)   # 官方對靜態圖沿時間 expand
    lat = enc(mx.array(a))[:, :vcfg["latent_channels"]] * vc.scaling_factor
    la = np.array(lat, copy=False)
    print(f"  latent {la.shape}  std {la.std():.3f}（乘過 scaling_factor {vc.scaling_factor:.4f}）")

    ok = True
    out = decode_image(dec, lat / vc.scaling_factor)
    for tag, frame in (("最後一幀（正確）", -1), ("第一幀（錯的，留著當對照）", 0)):
        raw = np.array(dec(lat / vc.scaling_factor), copy=False)[0, :, frame]
        b = np.clip((raw + 1) / 2, 0, 1).transpose(1, 2, 0)
        psnr = 10 * np.log10(1 / max(((b - orig) ** 2).mean(), 1e-12))
        good = psnr > 18 if frame == -1 else psnr < 18
        if frame == -1:
            ok &= good
        print(f"  {'✅' if good else '❌'} {tag:<26} PSNR {psnr:5.1f} dB  "
              f"平均絕對差 {np.abs(b-orig).mean()*255:5.2f}/255")

    same = np.array_equal(np.array(out, copy=False)[0, :, -1],
                          np.array(dec(lat / vc.scaling_factor), copy=False)[0, :, -1])
    ok &= same
    print(f"  {'✅' if same else '❌'} decode_image 取的就是最後一幀")
    print("\n  " + ("PASS — VAE 來回正確" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
