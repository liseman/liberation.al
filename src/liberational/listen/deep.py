"""The scheduled second pass over stored utterances: Parakeet plus Qwen3-ASR 0.6B.

Both models hear the same padded, level-normalized audio. Qwen runs in this short-lived process through native
Transformers (>=5.13); CUDA, MPS, and CPU are supported, with local weights only. It returns transcription text,
not Whisper probabilities. Empty transcription means no speech; a missing detected language alone does not.
Existing utterance boundaries remain the segment timestamps, so no forced aligner is loaded.

Agreement is accepted directly. Disagreements go to a constrained local chooser; a wide unresolved disagreement
keeps the live line uncertain. This deliberately drops the old Whisper-specific preference on wide disagreements.
The derived revisions table stores all three texts without changing source utterances or owner corrections.
Model/API: https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf (Apache-2.0).
"""

import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import logs, model_runtime, ollama, ollama_health
from liberational.listen.revisions import Revisions, revisions

log = logs.get("deep-transcribe")
SR = 16_000
PAD_S = 0.25
MIN_S = 0.6  # shorter lines aren't worth a second pass
CONTEXT = 2  # lines each side the merge model sees
CLOSE = 0.5  # maximum word error for an uncorroborated near revision
MAX_TOKENS = 512


def words(text: str | None) -> list[str]:
    return re.sub(r"[^\w' ]+", " ", (text or "").casefold().replace("-", " ").replace("_", " ")).split()


def wer(ref: str | None, hyp: str | None) -> float:
    """Word error of `hyp` against `ref` (edits over reference words; 0 when both are empty)."""
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


@dataclass
class Result:
    text: str
    method: str  # kept | parakeet | qwen | agree | merged | uncertain
    uncertain: bool = False


def merge(
    live: str,
    parakeet: str,
    qwen: str | None,
    choose: Callable[[str, str, str], str | None],
    names: Iterable[str] = (),
) -> Result:
    """Combine independent passes, allowing the chooser only words heard or known names."""
    p, w = (parakeet or "").strip(), (qwen or "").strip()
    if not p and not w:
        return Result(live, "kept")
    if not p or not w:
        text, method = (p, "parakeet") if p else (w, "qwen")
        if words(text) == words(live):
            return Result(live, "kept")
        return Result(text, method) if wer(live, text) <= CLOSE else Result(live, "uncertain", uncertain=True)
    if words(p) == words(w):
        return Result(p, "agree")
    allowed = set(words(live)) | set(words(p)) | set(words(w)) | {t for n in names for t in words(n)}
    out = choose(live, p, w)
    if out and words(out) and set(words(out)) <= allowed:
        return Result(out.strip(), "merged")
    return Result(p, "parakeet") if wer(p, w) <= CLOSE else Result(live, "uncertain", uncertain=True)


def ollama_choose(context: Sequence[str] = (), names: Sequence[str] = ()) -> Callable[[str, str, str], str | None]:
    """The merge model: the pack's small local model, at a fixed small context (never its default)."""

    def choose(live: str, p: str, w: str) -> str | None:
        system = (
            "You correct one line of a room-microphone transcript. You get three transcriptions of the same audio "
            "(A, B, C) and the lines spoken around it. Write the words most likely actually said, using ONLY words "
            "that appear in A, B or C, or one of the listed names. Prefer what two of them agree on. No explanation. "
            'Reply with JSON: {"text": "..."}'
        )
        user = (
            (f"Names: {', '.join(names)}\n" if names else "")
            + ("Around it:\n" + "\n".join(f"- {c}" for c in context) + "\n" if context else "")
            + f"A: {live}\nB: {p}\nC: {w}"
        )
        model = cfg.get("models.ollama.llm")
        try:
            reply = ollama.chat(
                model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                num_ctx=4096,
                num_predict=120,
                temperature=0,
                think=False,
                schema="json",
                timeout=60,
            )
            text = json.loads(reply["message"]["content"]).get("text")
        except (OSError, ValueError, KeyError, ollama_health.Unavailable) as e:
            log.warning("merge model: %s", e)
            return None
        return text if isinstance(text, str) else None

    return choose


def level(x: np.ndarray) -> np.ndarray:
    """`x` brought to a speaking level (audio/loudness.py, as Review's clips are): stored room audio can sit near
    -50 dBFS, where both models hear nothing; the live pass applies the mic's gain before ASR."""
    from liberational.audio import loudness

    if not len(x) or not np.count_nonzero(x):
        return x
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    return loudness.normalize(pcm, SR).astype(np.float32) / 32767


class Qwen:
    """One locally installed native Transformers model, owned by the scheduled process until exit."""

    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        except ImportError as ex:
            raise RuntimeError("Qwen3-ASR needs the deep-asr extra; run `mise run setup`") from ex

        torch.set_num_threads(2)
        device = str(cfg.get("asr.deep_device"))
        if not device:
            if torch.cuda.is_available():
                free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
                device = f"cuda:{max(range(len(free)), key=free.__getitem__)}"
            else:
                device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        repo = cfg.hf_repo("asr_deep")
        self.processor: Any = AutoProcessor.from_pretrained(repo, local_files_only=True)
        model: Any = Qwen3ASRForConditionalGeneration.from_pretrained(
            repo, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
        )
        self.model: Any = model.to(device).eval()
        model_runtime.loaded("models.hf.asr_deep", repo, "speech to text", device, owner=self)

    def __call__(self, audio16k: np.ndarray) -> str | None:
        import torch

        x = np.asarray(audio16k, dtype=np.float32)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("Qwen3-ASR requires finite mono audio at 16000 Hz")
        if not x.size or not np.count_nonzero(x):
            return None
        inputs = self.processor.apply_transcription_request(
            audio=x, language=None, prompt=None, processor_kwargs={"sampling_rate": SR, "return_tensors": "pt"}
        ).to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=MAX_TOKENS)
        generated = output[0, inputs["input_ids"].shape[1] :]
        eos = self.model.generation_config.eos_token_id
        stops = eos if isinstance(eos, list) else [eos]
        if len(generated) >= MAX_TOKENS and int(generated[-1]) not in stops:
            raise RuntimeError("Qwen3-ASR reached its token limit; refusing an incomplete transcription")
        parsed = self.processor.decode(generated, return_format="parsed")
        text = str(parsed["transcription"]).strip()
        return text or None


def run(
    lines: Sequence[dict[str, Any]],
    audio: Callable[[dict[str, Any]], np.ndarray | None],
    parakeet: Callable[[np.ndarray], str],
    qwen: Callable[[np.ndarray], str | None],
    store: Revisions,
    *,
    names: Sequence[str] = (),
    choose: Callable[[Sequence[str]], Callable[[str, str, str], str | None]] = lambda ctx: ollama_choose(ctx),
    limit: int | None = None,
) -> dict[str, int]:
    """Second pass over `lines` (one day's utterances, in time order), skipping lines already revised."""
    done, counts = store.done(), {}
    todo = [
        (i, e)
        for i, e in enumerate(lines)
        if e.get("id")
        and str(e["id"]) not in done
        and not e.get("password")
        and e.get("audio")
        and (e.get("ts_end", 0) - e.get("ts_start", 0)) >= MIN_S
    ]
    for i, e in todo[:limit]:
        x = audio(e)
        if x is None or not np.count_nonzero(x):
            continue
        p = parakeet(x)
        # A failed backend is unavailable, not a no-speech result. Leave this line pending for retry.
        w = qwen(x)
        near = [*lines[max(0, i - CONTEXT) : i], *lines[i + 1 : i + 1 + CONTEXT]]
        ctx = [str(c.get("text") or "") for c in near]
        r = merge(str(e.get("text") or ""), p, w, choose(ctx), names)
        store.put(
            {
                "id": str(e["id"]),
                "ts_start": e.get("ts_start"),
                "device": e.get("device"),
                "live": e.get("text"),
                "parakeet": p,
                "qwen": w,
                "text": r.text,
                "method": r.method,
                "uncertain": int(r.uncertain),
                "at": time.time(),
            }
        )
        counts[r.method] = counts.get(r.method, 0) + 1
    return counts


@model_runtime.reporting("deep-transcribe")
def main(argv: Sequence[str] | None = None) -> None:
    """`pack-deep-transcribe`: the second pass over a day's speech (default: yesterday and today until an hour ago)."""
    import argparse
    import datetime as dt

    from liberational import config as cfg
    from liberational import speech_log
    from liberational.people import registry
    from liberational.speech.asr import ASR

    parser = argparse.ArgumentParser(prog="pack-deep-transcribe", description=main.__doc__)
    parser.add_argument("--day", help="YYYY-MM-DD (default: yesterday and today)")
    parser.add_argument("--limit", type=int, default=None, help="at most this many lines (a trial run)")
    args = parser.parse_args(argv)
    state = Path(cfg.path("paths.state"))
    if args.day:
        start = dt.datetime.fromisoformat(args.day).timestamp()
        end = start + 86400
    else:
        today = dt.datetime.combine(dt.date.today(), dt.time()).timestamp()
        start, end = today - 86400, time.time() - 3600
    sl = speech_log.shared()
    lines = sorted(sl.range(start, end), key=lambda e: e.get("ts_start") or 0)

    def audio(e: dict[str, Any]) -> np.ndarray | None:
        a = e["audio"]
        rate, x = sl.audio(a["stream"], a["start_ns"] - int(PAD_S * 1e9), a["end_ns"] + int(PAD_S * 1e9), 0)
        x = np.asarray(x, np.float32)
        if rate != SR and len(x):
            x = np.interp(np.arange(0, len(x), rate / SR), np.arange(len(x)), x).astype(np.float32)
        return level(x)

    names = [p.display_name for p in registry.at(state).people() if p.display_name]
    second = Qwen()
    counts = run(
        lines,
        audio,
        ASR(),
        second,
        revisions(state),
        names=names,
        choose=lambda ctx: ollama_choose(ctx, names),
        limit=args.limit,
    )
    log.info("%d lines in range; revised: %s", len(lines), counts)
