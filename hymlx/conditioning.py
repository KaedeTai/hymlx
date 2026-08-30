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


def _study_path() -> Path:
    return Path.home() / "repos/hunyuan-study"


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

    def __init__(self, study: Optional[Path] = None, snapshot: Optional[str] = None):
        import sys
        study = Path(study or _study_path())
        if str(study) not in sys.path:
            sys.path.insert(0, str(study))
        _install_cuda_stubs()

        from hy.configuration_hunyuan_image_3 import HunyuanImage3Config
        from hy.image_processor import HunyuanImage3ImageProcessor
        from hy.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
        from hy.tokenization_hunyuan_image_3 import HunyuanImage3TokenizerFast
        from transformers import GenerationConfig

        self.snapshot = snapshot or snapshot_dir()
        self.raw_config = json.load(open(study / "config.json"))
        self.config = HunyuanImage3Config(**self.raw_config)
        self.image_processor = HunyuanImage3ImageProcessor(self.config)
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
        return Conditioning(
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

    def build_text(self, prompt: str, system_prompt=None, bot_task: str = "think", **kw):
        """gen_text 模式的序列（產生 think / recaption 用）。回傳 (tokens, stop_ids)。"""
        out = self._preprocess(prompt=prompt, mode="gen_text", system_prompt=system_prompt,
                               cfg_factor=1, bot_task=bot_task, **kw)
        o = out["output"]
        return o.tokens.numpy().astype(np.int32), out["stop_token_id"]

    def reference_attention_mask(self, c: Conditioning) -> np.ndarray:
        """官方 `_prepare_attention_mask_for_generation` 的結果，用來對答案。"""
        import torch
        m = torch.ones(c.seq_len, c.seq_len, dtype=torch.bool).tril(0).repeat(c.batch, 1, 1)
        for i, sl in enumerate(c.full_attn_slices):
            for s in sl:
                m[i, s, s] = True
        return m.unsqueeze(1).numpy()
