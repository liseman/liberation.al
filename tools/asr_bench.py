#!/usr/bin/env python3
"""Benchmark the active ASR engine: accuracy against reference transcripts, agreement with other engines, latency.

  uv run tools/asr_bench.py                       # the shipped clip set in tools/asr_clips/, engine from asr.engine
  PACK_ASR=parakeet-onnx uv run tools/asr_bench.py
  uv run tools/asr_bench.py --verbose             # print every transcript beside its reference
  uv run tools/asr_bench.py --clips data/my_clips # your own set, in the same layout
  uv run tools/asr_bench.py --save                # also store this engine's transcripts in the manifest
  uv run tools/asr_bench.py synth                 # regenerate the shipped clips with Kokoro (see below)

A clip set is a directory with a clips.json manifest and the audio files it names (any format soundfile reads, any
sample rate; it is resampled to 16 kHz mono):

  {"clips": [{"file": "01.ogg", "text": "reference transcript", "voice": "af_heart",
              "engines": {"<engine name>": "that engine's transcript"}}]}

"text" is the reference: the tool reports corpus word error rate (total word edits over total reference words)
against it. Each entry under "engines" is another engine's transcript of the same clip, recorded with --save on
some earlier run; the tool reports disagreement (the same corpus WER) against each one, so a fresh clone can
compare engines without running them all. Both sides are lowercased, digits are spelled out ("25" scores the same
as "twenty five"), and filler words (uh, um, mm-hmm) are dropped first, since the pack's ASR discards them on
purpose. Lower is better. Latency is per clip, after one warm-up clip.

The shipped set (tools/asr_clips/, about a minute of speech) is synthetic: Kokoro v1.0 stock voices reading
made-up sentences, so it can be published (tools/asr_clips/NOTICE.md has the licenses). Its references were
checked clip by clip against a transcription: Parakeet TDT 0.6B v3 matches every one word for word once numbers
are spelled out, so the speech says what the text says. After a `synth`, check them again the same way. Clean synthetic speech is
easier than a real room, so read its WER as a floor and a regression check, not as field accuracy. `synth` needs
the Kokoro ONNX files (`mise run models`) and espeak-ng; it rewrites the audio from the manifest's text and voice
and keeps the recorded engine transcripts, which then need a --save rerun per engine.
"""

import argparse
import json
import pathlib
import re
import statistics
import time
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

CLIPS = pathlib.Path(__file__).resolve().parent / "asr_clips"
FILLERS = {"uh", "um", "hmm", "mm", "mhm"}  # "mm-hmm" splits into two "mm"s once hyphens become spaces
SYNTH_SPEED = 1.0  # Kokoro's natural pace; the assistant's 1.1 is tuned for replies, not for test speech


def _number(m: re.Match[str]) -> str:
    from num2words import num2words

    return " " + num2words(int(m[0].replace(",", ""))).replace("-", " ").replace(",", "") + " "


def words(text: str) -> list[str]:
    """Lowercase words with fillers dropped and digits spelled out, so "25" and "twenty-five" score the same."""
    t = re.sub(r"\d+(?:,\d{3})*", _number, text.lower()).replace("-", " ")
    return [w for w in re.sub(r"[^a-z0-9' ]+", " ", t).split() if w not in FILLERS]


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


def wer(pairs: list[tuple[str, str]]) -> float:
    """Corpus WER of (hypothesis, reference) pairs: total word edits over total reference words."""
    ed = n = 0
    for hyp, ref in pairs:
        e, k = edits(hyp, ref)
        ed, n = ed + e, n + k
    return ed / max(1, n)


def load16k(path: pathlib.Path) -> NDArray[np.float32]:
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return np.asarray(a if sr == 16000 else resample_poly(a, 16000, sr), dtype=np.float32)


def read_manifest(clips: pathlib.Path) -> dict[str, Any]:
    return json.loads((clips / "clips.json").read_text())


def write_manifest(clips: pathlib.Path, manifest: dict[str, Any]) -> None:
    (clips / "clips.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def synth(clips: pathlib.Path) -> None:
    """Rewrite every clip's audio from its text and Kokoro voice, as 16 kHz mono Ogg Opus."""
    from liberational.speech.kokoro import Kokoro

    manifest = read_manifest(clips)
    voices: dict[str, Kokoro] = {}
    for c in manifest["clips"]:
        tts = voices.get(c["voice"]) or voices.setdefault(c["voice"], Kokoro(voice=c["voice"], speed=SYNTH_SPEED))
        audio, sr = tts.synth(c["text"])
        pad = np.zeros(int(0.3 * sr), np.float32)  # a little silence each side, as a recorder would leave
        audio16 = resample_poly(np.concatenate([pad, audio, pad]), 16000, sr).astype("float32")
        sf.write(clips / c["file"], audio16, 16000, format="OGG", subtype="OPUS")
        print(f"{c['file']}  {len(audio16) / 16000:4.1f} s  {c['voice']}")


def bench(clips: pathlib.Path, verbose: bool, save: bool) -> None:
    from liberational.speech import asr as pack_asr

    manifest = read_manifest(clips)
    entries = manifest["clips"]
    asr = pack_asr.ASR()
    asr(load16k(clips / entries[0]["file"]))  # warm up
    name = pack_asr.ASR_MODEL_NAME
    ms: list[float] = []
    secs = 0.0
    vs_ref: list[tuple[str, str]] = []
    vs_engine: dict[str, list[tuple[str, str]]] = {}
    for c in entries:
        audio = load16k(clips / c["file"])
        secs += len(audio) / 16000
        t = time.perf_counter()
        text = asr(audio)
        ms.append((time.perf_counter() - t) * 1000)
        vs_ref.append((text, c["text"]))
        for other, ref in c.get("engines", {}).items():
            if other != name:
                vs_engine.setdefault(other, []).append((text, ref))
        if save:
            c.setdefault("engines", {})[name] = text
        if verbose:
            e, n = edits(text, c["text"])
            print(f"[{c['file']} {len(audio) / 16000:4.1f}s] {ms[-1]:5.0f} ms  {e}/{n} edits")
            print(f"  ref: {c['text']}\n  hyp: {text}")

    print(f"engine: {name}  ({len(entries)} clips, {secs:.0f} s of audio)")
    p90 = sorted(ms)[int(len(ms) * 0.9)]
    print(f"latency ms: median {statistics.median(ms):.0f}, p90 {p90:.0f}, max {max(ms):.0f}")
    print(f"accuracy vs reference transcripts: {wer(vs_ref):.1%} corpus WER")
    for other, pairs in sorted(vs_engine.items()):
        print(f"disagreement vs {other}: {wer(pairs):.1%} corpus WER ({len(pairs)} clips)")
    if save:
        write_manifest(clips, manifest)
        print(f"saved this engine's transcripts to {clips / 'clips.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("command", nargs="?", choices=["bench", "synth"], default="bench")
    ap.add_argument("--clips", type=pathlib.Path, default=CLIPS, help="clip set directory (default: the shipped set)")
    ap.add_argument("--verbose", action="store_true", help="print every transcript")
    ap.add_argument("--save", action="store_true", help="store this engine's transcripts in clips.json")
    a = ap.parse_args()
    if a.command == "synth":
        synth(a.clips)
    else:
        bench(a.clips, a.verbose, a.save)


if __name__ == "__main__":
    main()
