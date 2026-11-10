#!/usr/bin/env python3
"""Phase 1 harness: real ASR -> LLM -> TTS round trip, no mocks.
Target: <700 ms from end-of-speech to first audio out, short replies. Apple Silicon (MLX) only.

Usage:
  python phase1_harness.py                # synth test utterance via Kokoro, loop it
  python phase1_harness.py --wav foo.wav  # use your own 16 kHz mono recording
  python phase1_harness.py --mic          # record 3 s from default input
"""

import argparse
import json
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import requests

from liberational import config as cfg

if TYPE_CHECKING:
    from liberational.speech.asr import ASR

OLLAMA = cfg.ollama_url()
LLM = cfg.get("models.ollama.llm")
SYSTEM = (
    "You are a voice assistant speaking through bone conduction. "
    "Answer in one short sentence, under 20 words. No markdown."
)


def t() -> float:
    return time.perf_counter()


# ---------- ASR (pack.ASR: Parakeet by default, PACK_ASR=whisper for whisper.cpp) ----------
_asr: ASR | None = None


def asr_load() -> ASR:
    global _asr
    from liberational.speech.asr import ASR

    _asr = ASR()
    return _asr


def asr(audio16k: np.ndarray) -> str:
    assert _asr is not None, "call asr_load() first"
    return _asr(audio16k.astype(np.float32))


# ---------- LLM (Ollama) ----------
def llm_stream(prompt: str) -> Iterator[str]:
    """Yield text chunks. Thinking disabled for latency."""
    r = requests.post(
        f"{OLLAMA}/api/chat",
        stream=True,
        json={
            "model": LLM,
            "stream": True,
            "think": False,
            "keep_alive": -1,
            "options": {"num_predict": 40, "temperature": 0.3},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        },
    )
    for line in r.iter_lines():
        if not line:
            continue
        d = json.loads(line)
        c = d.get("message", {}).get("content", "")
        if c:
            yield c
        if d.get("done"):
            break


# ---------- TTS (Kokoro, MLX on Apple Silicon) ----------
_tts: Any = None  # an mlx_audio Kokoro model once tts_load() runs


def tts_load() -> Any:
    global _tts
    from mlx_audio.tts.utils import load as load_tts

    _tts = load_tts(cfg.hf_repo("tts_kokoro"))
    return _tts


def tts(text: str) -> tuple[np.ndarray, int]:
    """Returns (audio float32, sample_rate). First chunk only matters for latency."""
    out = list(_tts.generate(text=text, voice=cfg.get("tts.voice"), speed=1.1))
    sr = out[0].sample_rate
    audio = np.concatenate([np.array(o.audio) for o in out])
    return audio, sr


# ---------- loop ----------
def run_once(audio16k: np.ndarray, play: bool = False) -> float:
    t0 = t()
    text = asr(audio16k)
    t1 = t()
    # Stream LLM; synthesize at first sentence/clause boundary for low latency
    buf, first_chunk_t, reply = "", None, ""
    audio_parts: list[np.ndarray] = []
    sr = 24000
    for chunk in llm_stream(text):
        reply += chunk
        buf += chunk
        if any(p in buf for p in ".?!,") and len(buf) > 12:
            a, sr = tts(buf.strip())
            audio_parts.append(a)
            buf = ""
            if first_chunk_t is None:
                first_chunk_t = t()
    if buf.strip():
        a, sr = tts(buf.strip())
        audio_parts.append(a)
        if first_chunk_t is None:
            first_chunk_t = t()
    t3 = t()
    first = (first_chunk_t or t3) - t0
    print(
        f"  ASR {1000 * (t1 - t0):.0f} ms | LLM+TTS->first audio {1000 * ((first_chunk_t or t3) - t1):.0f} ms | total-to-first-audio {1000 * first:.0f} ms | full {1000 * (t3 - t0):.0f} ms"
    )
    print(f"  heard: {text!r}\n  reply: {reply.strip()!r}")
    if play and audio_parts:
        import sounddevice as sd

        sd.play(np.concatenate(audio_parts), sr)
        sd.wait()
    return first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("-n", type=int, default=3)
    a = ap.parse_args()

    print("loading models...")
    t0 = t()
    asr_load()
    tts_load()
    requests.post(f"{OLLAMA}/api/generate", json={"model": LLM, "keep_alive": -1})  # warm
    print(f"  loaded in {t() - t0:.1f}s")

    if a.wav:
        import soundfile as sf

        audio, sr = sf.read(a.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        assert sr == 16000, "need 16 kHz"
    elif a.mic:
        import sounddevice as sd

        print("speak now (3 s)...")
        audio = sd.rec(3 * 16000, samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        audio = audio[:, 0]
    else:
        # Self-test: synthesize a question with Kokoro, resample 24k->16k
        q, sr = tts("What time does the sun set tonight?")
        audio = np.interp(np.arange(0, len(q), sr / 16000), np.arange(len(q)), q).astype(np.float32)

    lat = [run_once(audio, play=a.play) for _ in range(a.n)]
    best = min(lat[1:]) if len(lat) > 1 else lat[0]
    print(
        f"\nbest warm time-to-first-audio: {1000 * best:.0f} ms  ->  {'PASS' if best < 0.7 else 'FAIL'} (target <700 ms)"
    )


if __name__ == "__main__":
    main()
