#!/usr/bin/env python3
"""Benchmark voice cloning on CUDA (speech/qwen3_cuda.py) with the owner's reference clip.

  uv run tools/bench_clone_cuda.py                       # 4 runs, prints first audio, total, real-time factor
  uv run tools/bench_clone_cuda.py --out /tmp/clone.wav  # also save the last run

Needs an NVIDIA GPU, the voice-clone extra, and the model (`mise run setup`, `mise run models`). The first run
includes encoding the reference; later runs are what conversations see.
"""

import argparse
import time

import numpy as np
import soundfile as sf

from liberational import config as cfg
from liberational.speech import qwen3_cuda
from liberational.speech.voice_clone import ref_text_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text", default="Sure. The coffee place on the left is open until nine tonight.")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    ok, why = qwen3_cuda.available()
    if not ok:
        raise SystemExit(f"can't clone on CUDA here: {why}")
    ref = cfg.path("owner.voice_ref")
    t = time.perf_counter()
    m = qwen3_cuda.load()
    print(f"{m.name}: loaded and warmed up in {time.perf_counter() - t:.1f} s")
    import torch

    gpu = torch.device(m.device).index or 0
    for run in range(a.runs):
        t, first, parts, sr = time.perf_counter(), None, [], 24000
        for r in m.generate(
            text=a.text, ref_audio=ref, ref_text=ref_text_for(ref), stream=True, streaming_interval=0.33
        ):
            first = first or time.perf_counter() - t
            parts.append(r.audio)
            sr = r.sample_rate
        total, audio = time.perf_counter() - t, np.concatenate(parts)
        assert first is not None  # set by the first chunk, and concatenate() above needs at least one
        secs = len(audio) / sr
        print(
            f"run {run + 1}: first audio {1000 * first:.0f} ms, total {1000 * total:.0f} ms, {secs:.2f} s, rtf {total / secs:.2f}"
        )
    print(f"peak VRAM {torch.cuda.max_memory_allocated(gpu) / 2**30:.2f} GB on {m.device}")
    if a.out:
        sf.write(a.out, audio, sr)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
