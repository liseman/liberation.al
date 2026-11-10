"""Render text in the owner's cloned voice (Qwen3-TTS zero-shot, owner.voice_ref + .txt), on MLX or CUDA.

Used by OpenClaw's tts-local-cli provider for phone calls:
  python clone_tts_cli.py "<text>" <output.wav>
Writes a mono 24 kHz WAV. Exits non-zero on failure so the caller sees the error.
"""

import sys

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from liberational import config as cfg
from liberational import model_runtime

REF_WAV = cfg.path("owner.voice_ref")
REF_TXT = REF_WAV.with_suffix(".txt")
MODEL = cfg.hf_repo("tts_clone")
SR = 24000


def resample(a: np.ndarray, sr_in: int, sr_out: int) -> NDArray[np.float32]:
    import math

    from scipy.signal import resample_poly

    g = math.gcd(sr_in, sr_out)
    return resample_poly(a, sr_out // g, sr_in // g).astype(np.float32)


@model_runtime.reporting("clone-tts")
def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: clone_tts_cli.py <text> <output.wav>")
    text, out = sys.argv[1].strip(), sys.argv[2]
    if not text:
        sys.exit("empty text")
    if not cfg.has_mlx():
        from liberational.speech import voice_clone

        if not voice_clone.can_clone():
            sys.exit("voice cloning needs MLX (Apple Silicon) or an NVIDIA GPU with the voice-clone extra")
        voice_clone.load()
        audio, out_sr = voice_clone.say(text, ref=REF_WAV)
        if out_sr != SR:
            audio = resample(audio, out_sr, SR)
        sf.write(out, audio, SR, subtype="PCM_16")
        print(f"wrote {out} ({len(audio) / SR:.2f}s)")
        return
    import mlx.core as mx
    from mlx_audio.tts.utils import load as load_tts

    ref, sr = sf.read(str(REF_WAV), dtype="float32")
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    if sr != SR:
        ref = resample(ref, sr, SR)
    ref_audio = mx.array(ref)
    ref_text = REF_TXT.read_text().strip()

    model = load_tts(MODEL)
    model_runtime.loaded("models.hf.tts_clone", MODEL, "voice", "Apple GPU", owner=model)
    parts: list[NDArray[np.float32]] = []
    out_sr = SR
    for r in model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, stream=False):
        parts.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
        out_sr = r.sample_rate
    if not parts:
        sys.exit("no audio generated")
    audio = np.concatenate(parts)
    if out_sr != SR:
        audio = resample(audio, out_sr, SR)
    sf.write(out, audio, SR, subtype="PCM_16")
    print(f"wrote {out} ({len(audio) / SR:.2f}s)")


if __name__ == "__main__":
    main()
