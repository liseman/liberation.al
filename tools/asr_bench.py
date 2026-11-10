#!/usr/bin/env python3
"""Benchmark the active ASR engine on the A/B clip set in data/asr_ab/ (your own clips; none ship with the repository).

  uv run tools/asr_bench.py                  # engine from config (asr.engine)
  PACK_ASR=parakeet-onnx uv run tools/asr_bench.py
  uv run tools/asr_bench.py --verbose        # print every transcript

There are no human reference transcripts, so accuracy is reported as word
disagreement: corpus word error rate (total word edits over total reference
words) against another engine's output, the live whisper.cpp transcript and
Parakeet MLX as run on the original Mac mini (results.json). Filler words (uh,
um, mm-hmm) are dropped on both sides first, since the pack's ASR discards them
on purpose. Lower is closer. Latency is per clip, warm.
"""

import argparse
import json
import pathlib
import re
import statistics
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from liberational.config import ROOT
from liberational.speech import asr as pack_asr

DATA = ROOT / "data" / "asr_ab"


FILLERS = {"uh", "um", "hmm", "mm", "mm-hmm", "mhm"}


def words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9' -]+", " ", text.lower()).split() if w.strip("-") and w not in FILLERS]


def edits(hyp: str, ref: str) -> tuple[int, int]:
    """(word edit distance, reference word count) of hyp against ref."""
    h, r = words(hyp), words(ref)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1], len(r)


def load16k(path: pathlib.Path) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a if sr == 16000 else resample_poly(a, 16000, sr).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    meta = json.loads((DATA / "meta.json").read_text())
    base = {e["i"]: e for e in json.loads((DATA / "results.json").read_text())}

    asr = pack_asr.ASR()
    asr(load16k(DATA / meta[0]["wav"]))  # warm up
    ms: list[float] = []
    vs_whisper, vs_mac = [0, 0], [0, 0]
    for e in meta:
        audio = load16k(DATA / e["wav"])
        t = time.perf_counter()
        text = asr(audio)
        ms.append((time.perf_counter() - t) * 1000)
        for acc, ref in ((vs_whisper, e["whisper_live"]), (vs_mac, base.get(e["i"], {}).get("p"))):
            if ref is not None:
                ed, n = edits(text, ref)
                acc[0] += ed
                acc[1] += n
        if a.verbose:
            print(f"[{e['i']:02} {e['dur']:4.1f}s] {ms[-1]:5.0f} ms  {text}")

    print(f"engine: {pack_asr.ASR_MODEL_NAME}  ({len(meta)} clips, {sum(e['dur'] for e in meta):.0f} s of speech)")
    print(
        f"latency ms: median {statistics.median(ms):.0f}, p90 {sorted(ms)[int(len(ms) * 0.9)]:.0f}, max {max(ms):.0f}"
    )
    print(f"disagreement vs whisper.cpp live:   {vs_whisper[0] / max(1, vs_whisper[1]):.1%} corpus WER")
    print(f"disagreement vs Parakeet MLX (Mac): {vs_mac[0] / max(1, vs_mac[1]):.1%} corpus WER")


if __name__ == "__main__":
    main()
