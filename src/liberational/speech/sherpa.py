"""Portable ASR on sherpa-onnx: the same Parakeet model the Apple Silicon path runs on MLX.

  SherpaParakeet()(audio16k) -> str

The model comes from config/defaults.toml ([models.files.asr_parakeet]); fetch it with `mise run models`.
TTS off Apple Silicon is in speech/kokoro.py.
"""

import os
import pathlib

import numpy as np

from liberational import config as cfg
from liberational import model_runtime

FILLERS = {"uh", "um", "hmm", "mm", "mm-hmm", "mhm"}


def _threads() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def _need(path: pathlib.Path, what: str) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}; run `mise run models`")
    return path


class SherpaParakeet:
    """Parakeet TDT 0.6B v3 (int8) via sherpa-onnx. Thread-safe; CPU by default."""

    name = "Parakeet TDT 0.6B v3 (sherpa-onnx)"

    def __init__(self, model_dir: str | os.PathLike[str] | None = None, provider: str = "cpu") -> None:
        import sherpa_onnx

        d = _need(pathlib.Path(model_dir) if model_dir else cfg.model_path("asr_parakeet"), "Parakeet model")
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=_threads(),
            sample_rate=16000,
            feature_dim=128,  # Matches v3's preprocessor and exported encoder feat_dim metadata.
            decoding_method="greedy_search",
            model_type="nemo_transducer",
            provider=provider,
        )
        device = "CPU" if provider == "cpu" else "GPU (cuda)" if provider == "cuda" else provider
        model_runtime.loaded("models.files.asr_parakeet", d.name, "speech to text", device, owner=self)

    def __call__(self, audio16k: np.ndarray) -> str:
        audio16k = np.ascontiguousarray(audio16k, dtype=np.float32)
        if audio16k.size < 1600:  # < 0.1 s: nothing to decode
            return ""
        s = self.rec.create_stream()
        s.accept_waveform(16000, audio16k)
        self.rec.decode_stream(s)
        out = s.result.text.strip()
        return "" if out.lower().strip(".!? ") in FILLERS else out
