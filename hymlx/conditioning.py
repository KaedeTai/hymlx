"""里程碑 9：conditioning、tokenizer、系統提示詞。

這一層**刻意不移植**。理由寫在這裡免得以後自己忘記為什麼偷懶：

tokenizer 加上 image processor 一共兩千多行純 Python，沒有一個矩陣乘法。它做的事
是把 prompt、系統提示詞、`<boi>`、尺寸 token、比例 token、timestep token、4096 個
影像 token、`<eoi>` 排成一條序列，並回報每一段落在哪個 slice。重寫一份只會多出一
份會跟上游不同步的 bug，而且它的錯不會顯示成數值誤差，會顯示成「圖是壞的」——
那是這個專案最貴的一種錯。所以直接呼叫官方那份，只把輸出翻成 numpy / MLX。

這層真正屬於我們的工作只有兩件：把序列資訊翻成 `hymlx.rope` 吃的格式，以及確認
翻譯結果跟官方自己 `_prepare_attention_mask_for_generation` 產生的遮罩逐一相同。
"""
from __future__ import annotations

import contextlib
import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# 官方那幾支 Python（tokenizer、image processor、序列組裝）我們**不重寫也不夾帶**：
# 兩千多行純邏輯、沒有一個矩陣乘法，重寫只會多一份會跟上游不同步的 bug。
# 這裡在執行時從官方的 HF repo 抓下來，使用者以騰訊的授權取得，跟權重同一條路。
OFFICIAL_MODULES = (
    "__init__.py",
    "configuration_hunyuan_image_3.py",
    "tokenization_hunyuan_image_3.py",
    "image_processor.py",
    "system_prompt.py",
    "siglip2.py",
    "autoencoder_kl_3d.py",
    "hunyuan_image_3_pipeline.py",
    "cache_utils.py",
    "modeling_hunyuan_image_3.py",
    "utils/__init__.py",
    "utils/import_utils.py",
)
REPO_ID = "tencent/HunyuanImage-3.0-Instruct"


def ensure_official_code(repo_id: str = REPO_ID) -> str:
    """把官方的 .py 抓進 HF 快取，回傳那個目錄。"""
    from huggingface_hub import hf_hub_download
    d = None
    for f in OFFICIAL_MODULES:
        d = Path(hf_hub_download(repo_id, f)).parent
        if "/" in f:
            d = d.parent
    return str(d)


def register_official_package(snapshot: str, name: str = "hy") -> None:
    """把快照目錄掛成一個叫 `hy` 的套件，讓官方檔案裡的相對匯入解得開。

    不複製任何檔案——只是給 Python 一個 `__path__`。
    """
    import sys as _sys
    import types as _types
    if name in _sys.modules:
        return
    pkg = _types.ModuleType(name)
    pkg.__path__ = [snapshot]
    _sys.modules[name] = pkg


def snapshot_dir() -> str:
    pat = str(Path.home() / ".cache/huggingface/hub/"
              "models--tencent--HunyuanImage-3.0-Instruct/snapshots/*")
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError("找不到 HunyuanImage-3.0-Instruct 的快照")
    return hits[0]


def _install_cuda_stubs() -> None:
    """官方模組假設跑在 CUDA 上。這兩個都只是 profiling / 裝置管理，不影響數值。"""
    import torch
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()


@dataclass
class CondImages:
    """參考圖：同一張圖同時走 VAE 與 ViT 兩條路（`cond_image_type = "vae_vit"`）。

    VAE 那條給像素級的細節，ViT 那條給語意。兩段在序列上相鄰，注意力上算同一個
    區塊（`cond_token_attn_type = "joint_full"`）。時間步固定 0，代表「這是乾淨的圖」。
    """
    vae_pixels: np.ndarray          # (B, 3, H, W)，值域 [-1, 1]
    vae_index: np.ndarray           # (B, 4096) 序列上的位置
    vit_pixels: np.ndarray          # (B, 1024, 768) 已經 patch 攤平
    vit_index: np.ndarray           # (B, 1024)
    vit_shapes: list                # 每一列的 (h, w) patch 格點
    vit_mask: np.ndarray            # (B, 1024) 哪些 patch 是真的
    ts_index: np.ndarray            # (B, 1) 參考圖的 timestep token 位置


@dataclass
class TextConditioning:
    """文字階段（think / recaption）的序列資訊。

    純文生圖時只有 tokens 有用；**編輯時文字階段的序列本身也含參考圖**，
    所以跟影像階段一樣需要 rope 的 2D 段落、全連通區塊、以及參考圖本身。
    """
    tokens: np.ndarray                      # (B, S) int32
    stop_ids: Any
    rope_image_info: List[Any]
    full_attn_slices: List[List[slice]]
    cond: Optional["CondImages"] = None
    raw: Any = None


@dataclass
class Conditioning:
    """一次生成需要的全部序列資訊。slice 都是對 tokens 的索引。"""
    tokens: np.ndarray                      # (B, S) int32
    gen_image_mask: np.ndarray              # (B, S) bool，哪些位置是被生成的影像 token
    gen_timestep_index: Optional[np.ndarray]
    rope_image_info: List[List[Tuple[slice, Tuple[int, int]]]]
    full_attn_slices: List[List[slice]]
    token_h: int
    token_w: int
    image_height: int
    image_width: int
    cond: Optional["CondImages"] = None     # 參考圖（編輯用）
    raw: Any = None                         # 官方的 TokenizerEncodeOutput，除錯用

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def batch(self) -> int:
        return int(self.tokens.shape[0])


class Conditioner:
    """官方 tokenizer / image processor 的薄殼。

    借用 `HunyuanImage3ForCausalMM` 上三個純邏輯的方法（它們只碰 config、
    tokenizer、image_processor），不需要把 80B 的權重載進來。
    """

    def __init__(self, snapshot: Optional[str] = None):
        _install_cuda_stubs()
        self.snapshot = snapshot or snapshot_dir()
        register_official_package(ensure_official_code())

        from hy.configuration_hunyuan_image_3 import HunyuanImage3Config
        from hy.image_processor import HunyuanImage3ImageProcessor
        from hy.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
        from hy.tokenization_hunyuan_image_3 import HunyuanImage3TokenizerFast
        from transformers import GenerationConfig

        self.raw_config = json.load(open(f"{self.snapshot}/config.json"))
        self.config = HunyuanImage3Config(**self.raw_config)
        self.image_processor = HunyuanImage3ImageProcessor(self.config)
        self._patch_vit_processor()
        self._tokenizer = self._load_tokenizer(HunyuanImage3TokenizerFast)
        self.generation_config = GenerationConfig.from_pretrained(self.snapshot)
        self._cls = HunyuanImage3ForCausalMM
        # 借用純邏輯的方法：它們只碰 config / tokenizer / image_processor，
        # 不需要 80B 的權重。
        for name in ("check_inputs", "prepare_message_list",
                     "_validate_and_batchify_text", "_validate_and_batchify_image"):
            fn = getattr(HunyuanImage3ForCausalMM, name, None)
            if fn is not None and not hasattr(self, name):
                setattr(self, name, fn.__get__(self, Conditioner)
                        if not isinstance(inspect.getattr_static(HunyuanImage3ForCausalMM, name),
                                          staticmethod) else fn)

    def _patch_vit_processor(self):
        """transformers 5.x 的 Siglip2 image processor 預設回傳 list，官方的
        `vit_process_image` 直接對它呼叫 `.squeeze(0)` 就爆了。補上 `return_tensors="pt"`。

        跟分詞器那個是同一類問題：官方程式碼寫在舊版 transformers 上，新版改了回傳
        型別。這種錯至少會當場拋例外，比分詞器那個安靜地壞掉好處理。
        """
        import torch
        ip = self.image_processor

        def vit_process_image(image):
            origin = image.size
            inputs = ip.vit_info.processor(image, return_tensors="pt")
            px = inputs["pixel_values"]
            if not torch.is_tensor(px):
                px = torch.as_tensor(np.asarray(px))
            extra = {}
            for k in set(inputs.keys()) - {"pixel_values"}:
                v = inputs[k]
                if not torch.is_tensor(v):
                    v = torch.as_tensor(np.asarray(v))
                extra[k] = v.squeeze(0)
            return ip.as_image_tensor(px.squeeze(0), image_type=ip.vit_info.image_type,
                                      origin_size=origin, **extra)

        ip.vit_process_image = vit_process_image

    def _load_tokenizer(self, cls):
        """`from_pretrained` 在 transformers 5.x 會把 tokenizer.json 的 BPE merges 與
        ByteLevel pre-tokenizer 丟掉，變成**逐字元**分詞，中文更是整段消失。

        壞掉的樣子不會報錯，只會讓序列變長四倍、語意全毀：
            from_pretrained:  'a photo of ...' -> 34 個 token ['a','p','h','o','t','o',...]
            正確的:            'a photo of ...' -> 11 個 token ['a','Ġphoto','Ġof',...]
        它害我把生成失敗誤判成量化問題查了好幾輪。所以這裡直接用 `tokenizers`
        自己讀檔，再把它塞進包裝類別，並且當場驗一次。
        """
        import json as _json
        from tokenizers import Tokenizer as _RawTok
        raw = _RawTok.from_file(f"{self.snapshot}/tokenizer.json")
        cfg = _json.load(open(f"{self.snapshot}/tokenizer_config.json"))
        for k in ("tokenizer_class", "auto_map", "added_tokens_decoder"):
            cfg.pop(k, None)
        tk = cls(tokenizer_object=raw, **cfg)
        probe = "a photo of a red fox in snow, natural light"
        ids = tk.encode(probe, add_special_tokens=False)
        if len(ids) > 20 or tk.decode(ids) != probe:
            raise RuntimeError(
                f"分詞器壞了：{len(probe)} 個字元編成 {len(ids)} 個 token"
                f"（應該是 11 個）。BPE merges 或 pre-tokenizer 沒載進來。")
        return tk

    # -- 借來的方法 ---------------------------------------------------------
    def _preprocess(self, **kw):
        return self._cls.preprocess_inputs(self, **kw)

    def _rope_info(self, output, sections):
        return self._cls.build_batch_rope_image_info(self, output, sections)

    def _full_attn_slices(self, output, bsz):
        return [self.image_processor.prepare_full_attn_slices(output, i) for i in range(bsz)]

    # -- 對外 ---------------------------------------------------------------
    def system_prompt(self, kind: str = "en_vanilla", bot_task: str = "image") -> str:
        import hy.system_prompt as sp
        return sp.get_system_prompt(kind, bot_task)

    def build(self, prompt: str, image_size: str = "1024x1024",
              system_prompt: Optional[str] = None, cfg: bool = True,
              image=None, **kw) -> Conditioning:
        """文生圖（或帶參考圖的編輯）的序列。cfg=True 會多做一份無條件序列。"""
        out = self._preprocess(
            prompt=prompt, image=image, mode="gen_image",
            system_prompt=system_prompt, image_size=image_size,
            cfg_factor=2 if cfg else 1, bot_task="auto", **kw,
        )
        o, sections = out["output"], out["sections"]
        gi = out["batch_gen_image_info"][0]
        bsz = int(o.tokens.shape[0])
        rope = self._rope_info(o, sections)
        cond = self._cond_images(o, out["batch_cond_images"], bsz) if image is not None else None
        return Conditioning(
            cond=cond,
            tokens=o.tokens.numpy().astype(np.int32),
            gen_image_mask=o.gen_image_mask.numpy().astype(bool),
            gen_timestep_index=(None if o.gen_timestep_scatter_index is None
                                else o.gen_timestep_scatter_index.numpy()),
            rope_image_info=rope,
            full_attn_slices=self._full_attn_slices(o, bsz),
            token_h=gi.token_height, token_w=gi.token_width,
            image_height=gi.image_height, image_width=gi.image_width,
            raw=o,
        )

    def build_text(self, prompt: str, system_prompt=None, bot_task: str = "think",
                   image=None, with_cond: bool = False, **kw):
        """gen_text 模式的序列（產生 think / recaption 用）。

        預設回傳 (tokens, stop_ids)，跟以前一樣。`with_cond=True` 回傳
        `TextConditioning`——編輯時要用，因為官方在文字階段就把參考圖放進序列了
        （`generate_image` 兩個階段都傳 `image=image`），模型是「看著圖」寫
        recaption，不是只看指令文字。
        """
        out = self._preprocess(prompt=prompt, image=image, mode="gen_text",
                               system_prompt=system_prompt,
                               cfg_factor=1, bot_task=bot_task, **kw)
        o = out["output"]
        toks = o.tokens.numpy().astype(np.int32)
        if not with_cond:
            return toks, out["stop_token_id"]
        bsz = int(o.tokens.shape[0])
        return TextConditioning(
            tokens=toks,
            stop_ids=out["stop_token_id"],
            rope_image_info=self._rope_info(o, out["sections"]),
            full_attn_slices=self._full_attn_slices(o, bsz),
            cond=(self._cond_images(o, out["batch_cond_images"], bsz)
                  if image is not None else None),
            raw=o,
        )

    @staticmethod
    def _idx(mask) -> np.ndarray:
        m = np.asarray(mask)
        return np.where(m)[1].reshape(m.shape[0], -1).astype(np.int32)

    def _cond_images(self, o, batch_cond_images, bsz) -> "CondImages":
        import torch
        # cfg 的兩列共用同一張參考圖：無條件那列丟掉文字，但圖要留著
        per_row = [bc[0] for bc in batch_cond_images]
        if len(per_row) < bsz:
            per_row = per_row * bsz
        vae = np.stack([np.asarray(ci.vae_image, dtype=np.float32) for ci in per_row])
        vit = np.stack([np.asarray(ci.vit_image, dtype=np.float32) for ci in per_row])
        shapes, vmask = [], []
        for ci in per_row:
            k = ci.vit_image.vision_encoder_kwargs
            shapes.append(tuple(int(v) for v in k["spatial_shapes"]))
            vmask.append(np.asarray(k["pixel_attention_mask"], dtype=np.float32))
        return CondImages(
            vae_pixels=vae, vae_index=self._idx(o.vae_image_mask),
            vit_pixels=vit, vit_index=self._idx(o.vit_image_mask),
            vit_shapes=shapes, vit_mask=np.stack(vmask),
            ts_index=np.asarray(o.cond_timestep_scatter_index, dtype=np.int32),
        )

    def reference_attention_mask(self, c: Conditioning) -> np.ndarray:
        """官方 `_prepare_attention_mask_for_generation` 的結果，用來對答案。"""
        import torch
        m = torch.ones(c.seq_len, c.seq_len, dtype=torch.bool).tril(0).repeat(c.batch, 1, 1)
        for i, sl in enumerate(c.full_attn_slices):
            for s in sl:
                m[i, s, s] = True
        return m.unsqueeze(1).numpy()
