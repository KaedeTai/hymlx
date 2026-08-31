"""步數掃描：同一顆種子、同一段 CoT，只改步數，找甜蜜點。

模型只載入一次、前綴只預填一次，四個步數共用，省掉重複的固定成本。
最後拼一張對照圖：上排整張、下排臉部 1:1 裁切（皮膚與眼睛最會暴露步數不夠）。
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / "repos/hymlx"))
import numpy as np, mlx.core as mx
from PIL import Image, ImageDraw
from hymlx.conditioning import Conditioner
from hymlx.hunyuan import QuantizedHunyuan
from hymlx.rope import build_batch_2d_rope, build_attention_mask
from hymlx.sampling import cfg as apply_cfg, sigma_schedule
from hymlx.vae import Decoder, VAEConfig, decode_image, load_decoder

QD = Path.home() / "models/hymlx-8bit"
OUT = Path.home() / "models/hymlx-out"
PROMPT = "a portrait photograph of an elderly fisherman, weathered face, natural window light"
STEPS = [14, 18, 22, 28]
SEED, GUID, SHIFT = 7, 2.5, 3.0

cond = Conditioner()
cot = (OUT / "portrait_cot.txt").read_text()
c = cond.build(PROMPT, image_size="1024x1024",
               system_prompt=cond.system_prompt("en_unified"), cot_text=[cot])
m = QuantizedHunyuan(QD)
D = m.head_dim
cos = mx.stack([build_batch_2d_rope(c.seq_len, D, image_infos=[i], base=m.rope_theta)[0][0]
                for i in c.rope_image_info])
sin = mx.stack([build_batch_2d_rope(c.seq_len, D, image_infos=[i], base=m.rope_theta)[1][0]
                for i in c.rope_image_info])
P = int(c.gen_timestep_index[0][0])
istart = int(np.where(c.gen_image_mask[0])[0][0])
iend = int(np.where(c.gen_image_mask[0])[0][-1]) + 1
pmask = mx.broadcast_to(build_attention_mask(P)[0], (2, 1, P, P))
pam = mx.where(pmask, mx.array(0.0, mx.float32), mx.array(-3.4028235e38, mx.float32))
t0 = time.time()
caches = m.prefill_prefix(mx.array(c.tokens[:, :P].astype(np.int32)), cos[:, :P], sin[:, :P], pam)
del pmask, pam; mx.clear_cache()
print(f"序列 {c.seq_len}，前綴 {P}，預填 {time.time()-t0:.0f}s", flush=True)
cos_ts, sin_ts = cos[:, P:P + 1], sin[:, P:P + 1]
cos_img, sin_img = cos[:, istart:iend], sin[:, istart:iend]

vcfg = json.load(open(QD / "config.json"))["vae"]
K = ("in_channels", "out_channels", "latent_channels", "block_out_channels", "layers_per_block",
     "ffactor_spatial", "ffactor_temporal", "upsample_match_channel",
     "downsample_match_channel", "scaling_factor")
vc = VAEConfig(**{k: vcfg[k] for k in K})
dec = Decoder(vc); load_decoder(dec, mx.load(str(QD / "vae.safetensors"))); mx.eval(dec.parameters())

times = {}
for n in STEPS:
    rs = np.random.RandomState(SEED)
    x = mx.array(rs.randn(1, m.latent_channels, c.token_h, c.token_w).astype(np.float32))
    sigmas, timesteps = sigma_schedule(n, SHIFT)
    t0 = time.time()
    for i in range(n):
        p = m.image_step(caches, mx.repeat(x, 2, axis=0), mx.full((2,), float(timesteps[i])),
                         cos_ts, sin_ts, cos_img, sin_img, c.token_h, c.token_w).astype(mx.float32)
        x = x + apply_cfg(p[0:1], p[1:2], GUID) * float(sigmas[i + 1] - sigmas[i])
        mx.eval(x); mx.clear_cache()
    dt = time.time() - t0; times[n] = dt
    img = np.array(decode_image(dec, x / vc.scaling_factor), copy=False)[0, :, -1]
    img = np.clip((img + 1) / 2, 0, 1).transpose(1, 2, 0)
    Image.fromarray((img * 255).round().astype(np.uint8)).save(OUT / f"portrait_{n:02d}.png")
    print(f"  {n:>2} 步  {dt:.0f}s（{dt/n:.1f}s/步）-> portrait_{n:02d}.png", flush=True)

# 對照圖：上排整張縮圖，下排臉部 1:1
CW = 420
cx, cy = 512, 430
box = (cx - CW // 2, cy - CW // 2, cx + CW // 2, cy + CW // 2)
th = 340
sheet = Image.new("RGB", (th * len(STEPS), th + CW + 34), "white")
d = ImageDraw.Draw(sheet)
for j, n in enumerate(STEPS):
    im = Image.open(OUT / f"portrait_{n:02d}.png")
    sheet.paste(im.resize((th, th), Image.LANCZOS), (j * th, 24))
    sheet.paste(im.crop(box).resize((th, th * CW // th), Image.NEAREST) if False else im.crop(box),
                (j * th + (th - CW) // 2 if th > CW else j * th, th + 30))
    d.text((j * th + 8, 6), f"{n} 步  {times[n]:.0f}s", fill="black")
sheet.save(OUT / "portrait_sweep.png")
print("\n寫出 portrait_sweep.png（上排整張、下排臉部 1:1）")
print("步數/秒數:", {k: round(v) for k, v in times.items()})
