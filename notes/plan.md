# HunyuanImage-3.0 → MLX：里程碑

規矩沿用 mdream，那套在上一個專案救了五、六次：

- **每一階段都跟官方 PyTorch 實作對數字，再寫下一階段。** 不要整個蓋完再除錯。
- **CPU 用 fp32 容差證明算術**（MLX 的 CPU GEMM 準，邏輯錯誤藏不住）；
  **GPU 另外對 Metal fp32 GEMM 的底線**（這台機器實測約 8e-4）。
- 官方模組在 CPU 上要 stub 掉 `torch.cuda.set_device` 和 `nvtx.range`，兩者不影響數值。

| # | 階段 | 狀態 |
|---|---|---|
| 1 | 權重盤點（157 GiB，91.7% 是專家） | ✅ 完成 |
| 2 | 單層 MoE 對照 | ✅ 完成，CPU 3.622e-07（補上 `norm_topk_prob` 之後） |
| 3 | VAE 解碼器（latent → 像素） | ✅ 完成，CPU 3.692e-04（bf16 包絡的 3%） |
| 4 | VAE 編碼器（參考圖用） | ✅ 完成，CPU 7.018e-06 / GPU 2.406e-05（bf16 包絡 8.29e-3） |
| 5 | 2D 圖像 RoPE + 非因果注意力遮罩 | ✅ 完成，逐一相同 / 5.96e-08 |
| 6 | 完整 32 層 decoder | ✅ 單層對到底（layer 0 整層 CPU 9.7e-07 / GPU 1.0e-03），layer 31 抽驗 attention |
| 7 | SigLIP2 + aligner | ✅ 完成，CPU 2.4e-06（bf16 包絡 2.6e-02） |
| 8 | UNetDown / UNetUp / TimestepEmbedder | ✅ 完成，CPU 4.1e-06（bf16 包絡 4.2e-03） |
| 9 | conditioning、tokenizer、系統提示詞 | ✅ 完成，遮罩與官方逐一相同（刻意沿用官方 tokenizer） |
| 10 | 完整前向 | ✅ streaming 對照 32 層，單層誤差最差 1.035e-06 |
| 11 | 取樣器（flow matching） | ✅ 完成，50 步軌跡 1.7e-07 |
| 12 | 量化 + 實測 | 8-bit 可用；**4-bit 不能用**，理由見下 |

## 踩過的坑

- **解碼器的 `block_out_channels` 是反過來的**：`AutoencoderKLConv3D` 傳給 `Decoder` 的是
  `list(reversed(...))`，所以解碼器走 1024 → 128 而不是 128 → 1024。這個錯誤被
  state_dict 的形狀不符當場抓到——先載參考實作自己的權重是最便宜的第一個測試。
- **容差要量不要猜。** 第一版寫死 1e-4，CPU 測到 3.7e-4「失敗」，逐層查下去每一階段
  都在 1e-5 以內：是 `norm_out` 的 GroupNorm 除以標準差，把累積誤差放大 30 倍。
  改成量參考實作自己在 bf16 下的損失（1.17e-2）當包絡，我的 fp32 移植是它的 3%。
- **編碼器的收尾有一條捷徑**：`conv_out` 之後要加上把 1024 channel 每 16 個
  取平均壓成 64 channel 的 shortcut（正好是解碼器 `repeat_interleave` 的鏡像）。
  漏掉的話逐層對比每一層都是 1e-5，整體卻錯 63%——逐層測試不會抓到只出現在
  `forward` 裡、不屬於任何子模組的運算。要嘛整體測、要嘛照著 `forward` 讀。
- **MoE 的累積誤差不能拿來判對錯。** 32 層 streaming 對照時，累積相對誤差在第 12 層
  從 4.8e-07 一口氣跳到 2.9e-03，看起來像是那一層寫錯了。實際上：餵同一個輸入時
  第 12 層自己的誤差是 4.1e-08，路由完全一致；真正發生的是兩條**已經差了 4.8e-07**
  的隱狀態跨過了 top-8 的邊界——那個 token 的專家 28 與專家 43 機率同為 0.021299，
  小數點後六位都一樣，torch 挑一顆、MLX 挑另一顆。一顆專家換掉、權重 2.1%，
  就是 1e-03 等級的差。**判準要用單層誤差加上路由翻轉數，不是端到端的累積誤差。**
- **torch 在 CPU 上沒有 bf16 的 conv3d 核心**，量包絡要在 MPS 上跑，不然會卡死。
- 官方的 `Conv3d` 子類只是記憶體優化（>2 GiB 時沿時間軸切塊），註解自己寫數值差異
  在 1e-5 內。不要複製那套切塊，用普通卷積就好。

## 生成一直畫不對，查出來的三個原因

按嚴重程度排，第一個是根因：

1. **分詞器被 transformers 5.x 悄悄弄壞成逐字元。**
   `HunyuanImage3TokenizerFast.from_pretrained()` 載進來的 backend 少了 BPE merges
   與 ByteLevel pre-tokenizer，於是 `'a photo of ...'`（43 字元）被編成 **34 個
   單字元 token**（正確是 11 個），中文則整段消失、編出 0 個 token。
   它不報錯，只是讓模型讀到一串字元湯。這害我把「圖畫不對」誤判成量化問題查了好幾輪。
   修法：用 `tokenizers.Tokenizer.from_file()` 自己讀 tokenizer.json，再
   `cls(tokenizer_object=raw, **config)`，並在載入時當場驗一次（見 `conditioning.py`）。

2. **4-bit 在高噪聲端把訊號整個蓋掉。** 同一組輸入比 4-bit 與 bf16 對乾淨 latent 的估計：
   sigma=0.5 相關 **+0.9863**，sigma=1.0 相關 **+0.0259**——完全不相關。
   高噪聲端模型要決定「要畫什麼」，那個訊號小而細，4-bit 的量化雜訊直接蓋掉它，
   軌跡從隨機方向出發。改用 8-bit（8.5 bit/參數，84 GiB）就有照片級的結構。
   **低噪聲端 4-bit 完全夠用**：從 sigma=0.6 的真 latent 走回去，兩張臉結構完整。

3. **VAE 解碼取錯幀。** T=1 的 latent 解出來是 4 幀，官方
   `AutoencoderKLConv3D.decode` 取的是**最後一幀**（`decoded[:, :, -1:]`）。
   那一行在 wrapper 裡不在 `Decoder` 模組裡，所以模組級的逐張量比對抓不到。
   取第一幀的圖看起來像模型壞掉：16 px 的格子、沒有內容。
   `tests/test_vae_roundtrip.py` 現在把它釘住（PSNR 20.1 dB vs 取錯幀的 13.9 dB）。

**方法上的教訓**：模組逐張量對得上不代表整條路是對的。這三個錯全部落在模組之間的
接縫——wrapper 的一行、外部函式庫的行為、精度的選擇。端到端的來回測試（VAE 來回、
讓模型吐一段英文）比再多的模組級比對都便宜也都有效。

## 已知的事實

- MoE 路由：softmax → top-8 → **除以選中機率之和**（`norm_topk_prob=True`）→ 加共享專家。
  `mlx-lm` 的 `hunyuan` 模組少了那個除法，不補的話誤差 40%。
- `gate_and_up_proj` 拆兩半：**前半是 up（線性支路），後半是 gate（過 silu）**。
- `qkv_proj` 依 `(n_kv_heads, n_kv_groups+2, head_dim, -1)` 拆。
- 圖像 token 數 = (H/16) × (W/16)，`patch_size=1`。1024² → 4096 token。
- 速度是**計算受限**不是頻寬受限：1024² 一次前向計算 2.8 s、讀權重 0.08 s。
  量化讓它放得進 128 GB，不會讓它變快。

## 現在的狀態（原始權重已刪之後）

原始 157 GiB 的 bf16 權重已經刪掉，換成 `~/models/hymlx-8bit`（84 GiB）。
因此**需要原始權重當對照組的測試現在跑不了**：
`test_vae_decoder` / `test_vae_encoder` / `test_decoder_layer` / `test_imageio` /
`test_vision` / `test_full_forward` / `test_quant`。它們的結果留在這份文件的表裡。
還能跑的是不需要對照組的四支：`test_rope`、`test_sampler`、`test_conditioning`、
`test_vae_roundtrip`。要重跑前面那七支，得重新下載官方權重（約 25 分鐘）。

實測（M5 Max 128 GB，8-bit，1024x1024，CFG 所以每步兩個 batch）：

| 步數 | 取樣 | 每步 | 總計（含載入 8s、tokenize 4s、前綴預填 10s） |
|---|---|---|---|
| 50（無快取，舊） | 1006 s | 20.1 s | 17.5 分 |
| 28 | 334 s | 11.9 s | 6.0 分 |
| 10 | 112 s | 11.2 s | 2.2 分 |

think + recaption（1100 token，有 KV cache）約 100 s。

**影像路徑的 KV cache**（`prefill_prefix` + `image_step`）是 2.1 倍的來源，做法是：

- 文字前綴（系統提示詞 + prompt + CoT，2178 個 token）每一步都一樣，只跑一次。
- 快取之後**遮罩整個不需要了**。影像 token 本來就看得到全部文字與彼此；
  唯一的例外是 timestep token 不能看影像——把它單獨先跑完 32 層就自然滿足，
  不必 materialise 一張 6278x6278 的遮罩（fp32 315 MB，每層都要讀一次）。
- 序列尾巴 `<eoi>` 之後的 3 個 token 不用算，`final_layer` 不讀它們。

跟舊路徑的相關係數 0.966；差異是已知的 MoE 混沌（bf16 捨入被路由翻轉放大），
不是邏輯錯誤。

**每步的時間會從 9.4 s 漂到 12 s 之後穩住**，`mx.clear_cache()` 沒有影響，
所以是機器在持續負載下降頻，不是記憶體壓力。

剩下的瓶頸是 MoE 本身：4097 個 token x 2 batch x 8 顆專家，約佔 85% 的浮點運算，
實測約 14 TFLOPS。要再快只有降精度（4-bit 快約 1.4 倍，但高噪聲端不能用——
可以考慮前幾步 8-bit、後面 4-bit 的混合排程，代價是兩份權重要同時常駐 130 GiB，
這台機器放不下）。

## 步數要幾步

人像掃描（同一顆種子、同一段 CoT，只改步數，`tools/sweep_steps.py`）：

| 步數 | 取樣 | 總計 | 看起來 |
|---|---|---|---|
| 14 | 167 s | 2.8 分 | **構圖還沒定**，跟 22/28 是不同的取景 |
| 18 | 236 s | 3.9 分 | 構圖定了，紋理略軟 |
| 22 | 280 s | 4.7 分 | **甜蜜點**，1:1 看跟 28 步幾乎分不出來 |
| 28 | 374 s | 6.2 分 | 基準 |

**要注意的是步數改變的是軌跡，不是同一張圖的品質。** 14 步跟 28 步是兩張不同的
照片（構圖不同），不是同一張的糊版本。所以「跟 28 步差多少」那種指標會把
「構圖不同」跟「細節不夠」混在一起，數字沒有意義——我試過用臉部高頻能量當指標，
結果 14 步反而比 28 步高 26%，因為它的臉在畫面裡比較小，同樣的裁切框進更多背景邊緣。
能講的只有兩件事：構圖在 18 步定下來，紋理在 22 步收斂。
