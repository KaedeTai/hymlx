from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path.home() / "models/hymlx-out"
STEPS = [14, 18, 22, 28]
SECS = {14: 167, 18: 236, 22: 280, 28: 374}
CX, CY, CW = 520, 380, 380
TH = 380

def load_font(size):
    for p in ("/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

font, small = load_font(25), load_font(19)
ims = {n: Image.open(OUT / f"portrait_{n:02d}.png").convert("RGB") for n in STEPS}
box = (CX - CW // 2, CY - CW // 2, CX + CW // 2, CY + CW // 2)

def detail(im):
    """臉部區域的高頻能量。步數不夠時皮膚會偏蠟，這個值會低。
    跟構圖無關，所以比『跟 28 步的差距』更能講品質。"""
    g = np.asarray(im.crop(box).convert("L")).astype(np.float32)
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return np.abs(lap).mean()

det = {n: detail(ims[n]) for n in STEPS}
ref28 = det[28]

pad, gap = 14, 10
W = len(STEPS) * TH + (len(STEPS) + 1) * gap
H = 44 + TH + gap + CW + 36 + pad
sheet = Image.new("RGB", (W, H), (250, 250, 249))
d = ImageDraw.Draw(sheet)
for j, n in enumerate(STEPS):
    x = gap + j * (TH + gap)
    d.text((x, 10), f"{n} 步 · {SECS[n]}s（{SECS[n]/60:.1f} 分）",
           fill=(20, 20, 20), font=font)
    sheet.paste(ims[n].resize((TH, TH), Image.LANCZOS), (x, 44))
    sheet.paste(ims[n].crop(box), (x, 44 + TH + gap))
d.text((gap, H - 28),
       "上排：整張 1024×1024 縮圖　下排：臉部 380×380 原尺寸裁切（未縮放）　"
       "同一顆種子、同一段 CoT，只有步數不同",
       fill=(90, 90, 90), font=small)
sheet.save(OUT / "portrait_sweep.png")
print("寫出", sheet.size)
for n in STEPS:
    print(f"  {n:>2} 步  {SECS[n]:>3}s  臉部高頻 {det[n]:6.2f}  = 28 步的 {det[n]/ref28*100:5.1f}%")
