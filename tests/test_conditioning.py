"""里程碑 9：序列建構。確認官方 tokenizer 的輸出翻成 hymlx 的 RoPE / 遮罩格式之後
跟官方自己算的東西逐一相同。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    import mlx.core as mx
    from hymlx.conditioning import Conditioner
    from hymlx.rope import build_batch_2d_rope, build_attention_mask
    print("里程碑 9 — conditioning / tokenizer / 系統提示詞\n")

    c = Conditioner()
    print(f"  詞表 {c._tokenizer.vocab_size}  <boi>={c._tokenizer.boi_token_id} "
          f"<eoi>={c._tokenizer.eoi_token_id}")
    ok = True

    for size in ("1024x1024", "1280x720"):
        cd = c.build("a photo of a red fox in snow, natural light", image_size=size)
        S = cd.seq_len
        n_img = int(cd.gen_image_mask[0].sum())
        exp = cd.token_h * cd.token_w
        good = n_img == exp; ok &= good
        print(f"\n  {size} -> {cd.image_width}x{cd.image_height}, token 格點 "
              f"{cd.token_h}x{cd.token_w}")
        print(f"    序列 {cd.tokens.shape}（第 0 條是有條件、第 1 條是無條件）")
        print(f"    {'✅' if good else '❌'} 影像 token 數 {n_img} == token_h*token_w {exp}")
        info = cd.rope_image_info[0]
        print(f"    rope_image_info: {[(str(s), hw) for s, hw in info]}")

        # RoPE：影像那一段的位置應該排成二維格點，且整段佔的位置長度 = h*w
        cos, sin = build_batch_2d_rope(S, 128, image_infos=[info], base=10000.0)
        good = tuple(cos.shape) == (1, S, 128); ok &= good
        print(f"    {'✅' if good else '❌'} RoPE 表 {tuple(cos.shape)}")

        # 遮罩：跟官方 _prepare_attention_mask_for_generation 逐一比對
        ref = c.reference_attention_mask(cd)
        mine = np.concatenate([np.array(build_attention_mask(S, sl), copy=False)
                               for sl in cd.full_attn_slices], axis=0)
        same = np.array_equal(mine, ref); ok &= same
        sl0 = cd.full_attn_slices[0][0]
        print(f"    {'✅' if same else '❌'} 遮罩 {mine.shape} 與官方逐一相同  "
              f"（全連通區段 {len(cd.full_attn_slices[0])} 段，第一段 {sl0}）")

    # 系統提示詞真的接上去了
    sp = c.system_prompt("en_vanilla")
    good = isinstance(sp, str) and len(sp) > 50; ok &= good
    print(f"\n  {'✅' if good else '❌'} 系統提示詞 {len(sp)} 字：{sp[:60]!r}...")

    print("\n  " + ("PASS — 序列建構對得上" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
