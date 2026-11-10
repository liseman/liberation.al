"""Qwen3-TTS zero-shot voice cloning on an NVIDIA GPU, through faster-qwen3-tts (CUDA graphs, streaming).

The same model the Macs run on MLX (Qwen3-TTS-12Hz-0.6B-Base), for Linux machines with an NVIDIA GPU: install the
voice-clone extra (`mise run setup` does) and the weights from [models.hf.tts_clone_cuda] (`mise run models`).
generate() takes mlx-audio's arguments and yields results shaped like mlx-audio's (.audio, .sample_rate), so callers
treat both engines alike:

    m = qwen3_cuda.load()
    for r in m.generate(text="Hi.", ref_audio="data/voices/owner_ref.wav", ref_text="...", stream=True):
        play(r.audio, r.sample_rate)

Why faster-qwen3-tts and not the Qwen team's qwen-tts package: qwen-tts runs Hugging Face generate() eagerly, about
540 small forward passes for 3 s of speech, and measured a real-time factor of 2.2 (slower than speech) on the Linux
machine with no streaming output. faster-qwen3-tts captures CUDA graphs and streams. Measured on the Linux machine
(warm, about 3.4 s of speech): first audio about 230 ms, real-time factor about 0.55, 2.8 GB of VRAM.
"""

import importlib.util
import os
import pathlib
import subprocess
import threading
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from liberational import config as cfg
from liberational import model_runtime

CODEC_HZ = 12  # codec steps per second of audio (the 12Hz model)
LANGUAGE = "Auto"  # Qwen3-TTS picks the language from the text

_lock = threading.Lock()  # one process-wide model; its CUDA graphs can't be replayed by two threads at once
_model: Qwen3Cuda | None = None


def model_dir() -> pathlib.Path:
    return cfg.ROOT / cfg.load()["models"]["hf"]["tts_clone_cuda"]["path"]


def available() -> tuple[bool, str]:
    """(ok, reason): whether voice cloning can run on CUDA here."""
    if importlib.util.find_spec("faster_qwen3_tts") is None:
        return False, "the voice-clone extra isn't installed (mise run setup)"
    import torch

    if not torch.cuda.is_available():
        return False, "no CUDA GPU"
    if not (model_dir() / "config.json").exists():
        return False, f"the model isn't downloaded to {model_dir()} (mise run models)"
    return True, ""


def pick_device() -> str:
    """tts.clone_device when set, else the GPU with the most free memory (the other models tend to fill the first)."""
    if want := cfg.get("tts.clone_device"):
        return want
    # nvidia-smi rather than torch.cuda.mem_get_info, which would open a CUDA context (~256 MB) on every GPU.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        free = {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines())}
        return f"cuda:{max(free, key=free.__getitem__)}"
    except OSError, ValueError, subprocess.SubprocessError:
        return "cuda:0"


class Qwen3Cuda:
    def __init__(self, device: str) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        self.device = device
        self.m = FasterQwen3TTS.from_pretrained(
            str(model_dir()), device=device, dtype=torch.bfloat16, local_files_only=True
        )
        self.m.warmup()  # capture the CUDA graphs now, not on the first reply (about 2.5 s)
        self._ref_mtimes: dict[str, int] = {}
        self.name = f"Qwen3-TTS clone on CUDA ({device})"
        where = f"GPU ({device})" if device.startswith("cuda") else "CPU" if device == "cpu" else device
        model_runtime.loaded("models.hf.tts_clone_cuda", cfg.hf_repo("tts_clone_cuda"), "voice", where, owner=self)

    def _forget_stale_reference(self, ref_audio: str | os.PathLike[str]) -> None:
        """faster-qwen3-tts caches the encoded reference by path, and re-recording owner_ref.wav keeps its path."""
        try:
            mtime = os.stat(ref_audio).st_mtime_ns
        except OSError:
            return
        key = str(ref_audio)
        if self._ref_mtimes.get(key, mtime) != mtime:
            cache = getattr(self.m, "_voice_prompt_cache", {})
            for k in [k for k in cache if k and str(k[0]) == key]:
                cache.pop(k, None)
        self._ref_mtimes[key] = mtime

    def generate(
        self,
        text: str,
        ref_audio: str | os.PathLike[str],
        ref_text: str,
        stream: bool = True,
        streaming_interval: float = 0.5,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **_: object,
    ) -> Iterator[SimpleNamespace]:
        """Yield SimpleNamespace(audio=float32 array, sample_rate=int): chunks of about streaming_interval seconds
        when stream is true, else one result with the whole utterance."""
        kw: dict[str, float] = {"max_new_tokens": int(max_tokens)} if max_tokens else {}
        if temperature is not None:
            kw["temperature"] = float(temperature)
        chunk = max(1, round(streaming_interval * CODEC_HZ))
        with _lock:
            self._forget_stale_reference(ref_audio)
            parts: list[NDArray[np.float32]] = []
            sr = 24000
            for audio, sr, _timing in self.m.generate_voice_clone_streaming(
                text=text, language=LANGUAGE, ref_audio=str(ref_audio), ref_text=ref_text, chunk_size=chunk, **kw
            ):
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if stream:
                    yield SimpleNamespace(audio=audio, sample_rate=sr)
                else:
                    parts.append(audio)
            if not stream and parts:
                yield SimpleNamespace(audio=np.concatenate(parts), sample_rate=sr)


def load() -> Qwen3Cuda:
    """The process-wide CUDA clone model (loaded on first use, on pick_device())."""
    global _model
    with _lock:
        if _model is None:
            _model = Qwen3Cuda(pick_device())
    return _model
