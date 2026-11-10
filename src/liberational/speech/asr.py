"""Speech recognition: Parakeet on MLX (Apple Silicon) or sherpa-onnx, or a whisper.cpp server.

asr = ASR(); text = asr(audio16k)   # float32 mono at 16 kHz
"""

import numpy as np

from liberational import config as cfg
from liberational import model_runtime

# asr.engine: parakeet (default) | parakeet-onnx | whisper. "parakeet" runs on MLX on Apple
# Silicon and falls back to the same model on sherpa-onnx everywhere else.
ASR_ENGINE = cfg.get("asr.engine").lower()
if ASR_ENGINE == "parakeet" and not cfg.has_mlx():
    ASR_ENGINE = "parakeet-onnx"
ASR_MODEL_NAME = {
    "parakeet": "Parakeet TDT 0.6B v3 (MLX)",
    "parakeet-onnx": "Parakeet TDT 0.6B v3 (sherpa-onnx)",
    "whisper": "whisper.cpp medium (whisper service)",
}.get(ASR_ENGINE, ASR_ENGINE)


MLX_CACHE_LIMIT = 1 << 30  # bytes of freed GPU buffers MLX may keep for reuse


class ASR:
    """Multilingual ASR. Default Parakeet TDT 0.6B v3 via parakeet-mlx (in-process, MLX,
    thread-affine: construct and call it on the same thread), or via sherpa-onnx without MLX.
    Parakeet detects the language automatically across its 25 supported European languages.
    asr.engine=whisper uses whisper.cpp medium via the whisper service instead."""

    def __init__(self) -> None:
        self.engine = ASR_ENGINE
        if self.engine == "whisper":
            from liberational.speech.whisper import WhisperASR

            self.w = WhisperASR()
        elif self.engine == "parakeet-onnx":
            from liberational.speech.sherpa import SherpaParakeet

            self.w = SherpaParakeet()
        else:
            import mlx.core as mx
            from parakeet_mlx import from_pretrained
            from parakeet_mlx.audio import get_logmel

            self.mx, self.get_logmel = mx, get_logmel
            # MLX keeps freed GPU buffers for reuse, and every utterance length makes new ones, so with no limit the
            # cache only grows: 17 GB in the listen service after 1.5 h (2026-10-31; 19 GB after 200 test clips).
            # A cap keeps it to MLX_CACHE_LIMIT with no slowdown (133-140 ms a clip, capped or not).
            mx.set_cache_limit(MLX_CACHE_LIMIT)
            self.m = from_pretrained(cfg.hf_repo("asr_parakeet"))
            model_runtime.loaded(
                "models.hf.asr_parakeet", cfg.hf_repo("asr_parakeet"), "speech to text", "Apple GPU", owner=self
            )

    def __call__(self, audio16k: np.ndarray) -> str:
        if self.engine in ("whisper", "parakeet-onnx"):
            return self.w(audio16k)
        audio16k = np.ascontiguousarray(audio16k, dtype=np.float32)
        if audio16k.size < 1600:  # < 0.1 s: nothing to decode
            return ""
        mel = self.get_logmel(self.mx.array(audio16k), self.m.preprocessor_config)
        out = self.m.generate(mel)[0].text.strip()
        return "" if out.lower().strip(".!? ") in {"uh", "um", "hmm", "mm", "mm-hmm", "mhm"} else out
