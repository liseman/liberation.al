"""Evaluate finished phone calls without listening: intelligibility, latency, reply quality.

For each call under calls/:
  * ASR re-transcribes bot.wav (what the caller actually heard) and scores it against the text Bob meant to say (word error rate).
  * Response latency comes from the bridge log ("first audio ... after end of caller speech").
  * Flags replies that claim capabilities the bridge does not have, and counts filler words.
Writes calls/<day>/<call>/eval.json and prints a summary.
"""

import json
import re

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from liberational import config as cfg
from liberational.phone import bridge_log

CALLS = cfg.path("paths.calls")
OVERCLAIM = re.compile(
    r"\b(take messages|set reminders?|book (an |a )?appointments?|send (a |an )?(message|text|email)|schedule|order|check my (calendar|email)|look (it )?up|i can (send|book|set|schedule|remind|text|call|order|buy)|i('| a)m able to (send|book|set))",
    re.I,
)
FILLERS = re.compile(r"\b(um+|uh+|er+|ah+)\b", re.I)


def wer(ref: str, hyp: str) -> float:
    r, h = ref.lower().split(), hyp.lower().split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / max(1, len(r))


def latencies(call_sid: str) -> list[float]:
    return [float(m.group(1)) for m in re.finditer(rf"\[{call_sid}\] first audio ([\d.]+)s", bridge_log())]


def main() -> None:
    from liberational.speech.asr import ASR

    asr = ASR()
    out = []
    for d in sorted(CALLS.glob("*/*")):
        tp, bp = d / "transcript.jsonl", d / "bot.wav"
        if not tp.exists() or not bp.exists():
            continue
        turns = [json.loads(l) for l in tp.read_text().splitlines() if l.strip()]
        bot_turns = [t["text"] for t in turns if t["speaker"] == "bot"]
        audio, sr = sf.read(bp, dtype="float32")
        a16 = resample_poly(audio, 16000, sr).astype(np.float32)
        heard = asr(a16)
        ref = " ".join(bot_turns)
        cid = re.sub(r"^[0-9_]+_", "", d.name)
        bot_words = sum(len(t.split()) for t in bot_turns)
        res = {
            "call": d.name,
            "bot_turns": len(bot_turns),
            "bot_words": bot_words,
            "wer_bot_audio_vs_intended": round(wer(ref, heard), 3),
            "asr_of_bot_audio": heard[:300],
            "first_audio_latency_s": latencies(cid),
            "overclaims": [t for t in bot_turns if OVERCLAIM.search(t)],
            "filler_words": sum(len(FILLERS.findall(t)) for t in bot_turns),
            "avg_reply_words": round(bot_words / max(1, len(bot_turns)), 1),
        }
        (d / "eval.json").write_text(json.dumps(res, indent=2))
        out.append(res)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
