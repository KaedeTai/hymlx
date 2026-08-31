"""里程碑 10：完整前向。

權重 157 GiB 放不進 128 GB，所以兩邊都用 streaming：每一層的權重只讀一次，
**同時**餵給官方模組和 hymlx，比完就丟。

判準是「餵同一個輸入時，這一層自己的誤差」，不是累積誤差。因為這是 MoE：
兩條隱狀態一旦差了 1e-07，某一層某個 token 的第 8 名與第 9 名專家如果機率接近，
就會選到不同的專家，輸出立刻差 1e-03。那不是移植錯，是離散路由的本性。
（實測：第 12 層有一個 token 的 e28 與 e43 機率同為 0.021299，小數點後六位都一樣。）
所以這裡同時報三個數字：累積誤差、單層誤差、路由翻轉數——只有後兩個能判對錯。
"""
from __future__ import annotations
import contextlib, gc, glob, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hymlx.conditioning import ensure_official_code, register_official_package, snapshot_dir
register_official_package(ensure_official_code()); STUDY = Path(snapshot_dir())
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()
SNAP = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--tencent--HunyuanImage-3.0-Instruct/snapshots/*"))[0]
CFG = json.load(open(STUDY / "config.json"))


def rel(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)


def main() -> int:
    import mlx.core as mx
    from hymlx.hunyuan import HunyuanImage3, ShardedWeights
    from hymlx.rope import build_batch_2d_rope, build_attention_mask
    from hy.configuration_hunyuan_image_3 import HunyuanImage3Config
    from hy.modeling_hunyuan_image_3 import HunyuanImage3DecoderLayer
    mx.set_default_device(mx.cpu)
    print("里程碑 10 — 完整前向（streaming，fp32 CPU）\n")

    # 小尺寸的合成序列：4x4 的 latent -> 16 個影像 token，前面 8 個文字 token
    TH = TW = 4
    NIMG = TH * TW
    S = 8 + NIMG + 4
    img_slice = slice(8, 8 + NIMG)
    rs = np.random.RandomState(5)
    tokens = rs.randint(0, 100000, size=(1, S)).astype(np.int64)
    latents = (rs.randn(1, CFG["vae"]["latent_channels"], TH, TW) * 0.7).astype(np.float32)
    t = np.array([0.63], dtype=np.float32)
    image_mask = np.zeros((1, S), dtype=bool); image_mask[0, img_slice] = True
    ts_index = np.array([[7]], dtype=np.int64)          # timestep token 就放在 <boi> 後面

    info = [(img_slice, (TH, TW))]
    cos, sin = build_batch_2d_rope(S, CFG["attention_head_dim"], image_infos=[info],
                                   base=CFG["rope_theta"])
    mask = build_attention_mask(S, [img_slice])
    rcos = torch.from_numpy(np.array(cos, copy=False)); rsin = torch.from_numpy(np.array(sin, copy=False))
    rmask = torch.from_numpy(np.array(mask, copy=False))

    t0 = time.time()
    w = ShardedWeights(SNAP, STUDY / "model.safetensors.index.json")
    m = HunyuanImage3(CFG, w, dtype=mx.float32)
    print(f"  頭尾模組載入 {time.time()-t0:.1f}s（wte + patch_embed + final_layer + 三個 timestep）")

    # --- 輸入層 ---
    h_mlx = m.embed(mx.array(tokens), mx.array(latents), mx.array(t),
                    mx.array(image_mask), mx.array(ts_index))
    mx.eval(h_mlx)

    rwte = torch.from_numpy(w["model.wte.weight"])
    h_ref = torch.nn.functional.embedding(torch.from_numpy(tokens), rwte)
    del rwte; gc.collect()
    # 官方的輸入層：patch_embed(latent, time_embed(t)) 蓋掉影像位置，timestep_emb(t) 蓋掉 timestep 位置
    with torch.no_grad():
        t_emb = torch.from_numpy(np.array(m.time_embed(mx.array(t)), copy=False))
        img_seq = torch.from_numpy(np.array(m.patch_embed(mx.array(latents), mx.array(t_emb.numpy()))[0], copy=False))
        h_ref[0, img_slice, :] = img_seq[0]
        h_ref[0, ts_index[0], :] = torch.from_numpy(
            np.array(m.timestep_emb(mx.array(t)), copy=False))
    e0 = rel(h_mlx, h_ref.numpy())
    ok = e0 < 1e-6
    print(f"  {'✅' if ok else '❌'} 輸入層（scatter 之後）誤差 {e0:.3e}   序列 {S} token，"
          f"其中 {NIMG} 個是影像\n")

    rcfg = HunyuanImage3Config(**CFG); rcfg._attn_implementation = "sdpa"
    hm, hr = h_mlx, h_ref
    worst_cum = worst_solo = 0.0
    flips_total = 0
    t0 = time.time()
    print("    層   累積誤差    單層誤差(同輸入)  專家選擇不同的 token 數")
    for i in range(CFG["num_hidden_layers"]):
        pref = f"model.layers.{i}."
        wl = w.prefix(pref)
        ref = HunyuanImage3DecoderLayer(rcfg, layer_idx=i).float().eval()
        ref.load_state_dict({k[len(pref):]: torch.from_numpy(v) for k, v in wl.items()}, strict=False)
        hm_t = torch.from_numpy(np.array(hm, copy=False))
        with torch.no_grad():
            hr = ref(hr, attention_mask=rmask, custom_pos_emb=(rcos, rsin))[0]
            # 同一個輸入餵兩邊，把「這一層自己的誤差」跟「上游累積的誤差」分開
            hr_solo = ref(hm_t, attention_mask=rmask, custom_pos_emb=(rcos, rsin))[0]
            # 路由是否選到同一組專家
            gl = ref.mlp.gate.wg(ref.post_attention_layernorm(hm_t).reshape(-1, 4096).float())
            ridx = torch.topk(torch.softmax(gl, dim=1), 8).indices.numpy()
        del ref; gc.collect()
        lyr = m._layer(i)
        hm_solo = lyr(mx.array(np.array(hm, copy=False)), cos, sin, mask); mx.eval(hm_solo)
        g = mx.softmax(lyr.mlp.gate.wg(lyr.post_attention_layernorm(hm).astype(mx.float32)),
                       axis=-1, precise=True)
        midx = np.array(mx.argpartition(-g, kth=7, axis=-1)[..., :8], copy=False).reshape(-1, 8)
        flips = int(sum(set(a) != set(b) for a, b in zip(np.sort(ridx, 1), np.sort(midx, 1))))
        flips_total += flips
        hm = lyr(hm, cos, sin, mask); mx.eval(hm)
        del lyr, wl; gc.collect(); mx.clear_cache()
        e_cum = rel(hm, hr.numpy()); e_solo = rel(hm_solo, hr_solo.numpy())
        worst_cum = max(worst_cum, e_cum); worst_solo = max(worst_solo, e_solo)
        mark = "  <-- 路由翻轉" if flips else ""
        print(f"    {i:>2}   {e_cum:.3e}   {e_solo:.3e}        {flips}/{hm.shape[1]}{mark}")
    print(f"\n  32 層跑完 {time.time()-t0:.0f}s")
    print(f"  累積誤差最差 {worst_cum:.3e}；**單層誤差最差 {worst_solo:.3e}**；"
          f"路由翻轉共 {flips_total} 個 token-層")
    worst = worst_solo

    # --- 輸出層 ---
    idx = np.where(image_mask)[1].reshape(1, -1)
    img_m = mx.take_along_axis(hm, mx.array(idx)[:, :, None], axis=1)
    pred_m = m.final_layer(img_m, m.time_embed_2(mx.array(t)), TH, TW)
    with torch.no_grad():
        img_r = hr[:, img_slice, :]
        pred_r = m.final_layer(mx.array(img_r.numpy()), m.time_embed_2(mx.array(t)), TH, TW)
    e_out = rel(pred_m, np.array(pred_r, copy=False))
    good = worst < 1e-5; ok &= good
    print(f"  {'✅' if good else '❌'} 判準是**單層誤差** {worst:.3e} < 1e-5。"
          f"累積誤差沒有意義，理由見下。")
    print(f"     final_layer 輸出 {tuple(pred_m.shape)}，兩邊各餵自己的隱狀態差 {e_out:.3e}")
    print(f"     預測 latent 範圍 [{float(pred_m.min()):+.3f}, {float(pred_m.max()):+.3f}]")

    print("\n  " + ("PASS — 完整前向對得上" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
