"""What two or more named people talked about most in a window: their conversations, each summarised by the local
model, media playback set aside, topics ranked by time spent.

The previous answer counted recurring words and returned "chinese, khan, mongols" for a two-person conversation.
Those were an audiobook about Genghis Khan playing in the room, which speaker identification attributed to them.
Here each conversation (lines no more than GAP_S apart, where every named person speaks at least twice) is
classified as conversation or media and given a short topic; media is reported but not ranked.
"""

import datetime as dt
import json
import math
import pathlib
import re
import time
from typing import Any

from liberational import config as cfg
from liberational import ollama, ollama_health

GAP_S = 180.0
MIN_LINES = 2
MAX_CHARS = 6000  # transcript sent to the model per conversation
PROMPT = (
    "Below is an automatic transcript from a room microphone, with speaker labels that can be wrong. "
    "Decide whether it is mainly people talking to each other (CONVERSATION) or mainly media playing in the room "
    "(MEDIA: a podcast, audiobook, video, TV or narration; long fluent monologues on one subject are a sign). "
    "Then give the main topic in at most 6 words, and one short factual sentence. Reply with only JSON: "
    '{"kind": "CONVERSATION" or "MEDIA", "topic": "...", "summary": "..."}'
)


def _llm(text: str, timeout: float = 60.0, *, background: bool = True) -> dict[str, str] | None:
    model = cfg.get("models.ollama.llm")  # the pack's small local model; fast and always loaded
    try:
        reply = ollama.chat(
            model,
            [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
            num_ctx=4096,
            num_predict=160,
            temperature=0,
            think=False,
            schema="json",
            timeout=timeout,
            background=background,
        )
        out = json.loads(reply["message"]["content"])
        if isinstance(out, dict) and out.get("topic"):
            return {k: str(out.get(k) or "").strip() for k in ("kind", "topic", "summary")}
    except OSError, ValueError, KeyError, ollama_health.Unavailable:
        return None
    return None


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def conversations(lines: list[dict[str, Any]], people: list[str]) -> list[list[dict[str, Any]]]:
    want = {_norm(p) for p in people}
    out, cur = [], []
    for e in sorted(lines, key=lambda x: x.get("ts_start") or 0):
        if not (e.get("text") or "").strip():
            continue
        if cur and e["ts_start"] - cur[-1]["ts_start"] > GAP_S:
            out.append(cur)
            cur = []
        cur.append(e)
    if cur:
        out.append(cur)
    keep = []
    for c in out:
        counts: dict[str, int] = {}
        for e in c:
            counts[_norm(e.get("speaker") or "")] = counts.get(_norm(e.get("speaker") or ""), 0) + 1
        if all(counts.get(p, 0) >= MIN_LINES for p in want):
            keep.append(c)
    return keep


def _render(c: list[dict[str, Any]]) -> str:
    text = "\n".join(f"{e.get('speaker') or 'unknown'}: {e['text'].strip()}" for e in c)
    return text[:MAX_CHARS]


def analyse(
    state: pathlib.Path, people: list[str], start: float, end: float, *, background: bool = True
) -> dict[str, Any]:
    """The conversations `people` had in [start, end), from the speech log, each line named through the event log
    (people/transcript_view.py), since a stored line names a voice, never a person. Foreground history questions
    pass background=False, since their answer may be the active voice turn itself."""
    from liberational import speech_log
    from liberational.people import transcript_view

    lo, hi = max(start, 0.0), end if math.isfinite(end) else time.time() + 1.0
    heard = [e for e in speech_log.shared().range(lo, hi) if lo <= e["ts_start"] < hi] if lo < hi else []
    lines = transcript_view.View(state).lines(heard)
    convs = conversations(lines, people)
    items = []
    for c in convs:
        a, b = c[0]["ts_start"], c[-1].get("ts_end") or c[-1]["ts_start"]
        got = _llm(_render(c), background=background)
        items.append(
            {
                "start": a,
                "end": b,
                "minutes": max(1, round((b - a) / 60)),
                "lines": len(c),
                "kind": (got or {}).get("kind", "UNKNOWN").upper(),
                "topic": (got or {}).get("topic", ""),
                "summary": (got or {}).get("summary", ""),
            }
        )
    return {"people": people, "window": [start, end], "conversations": items}


def _clock(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M")


def answer_text(res: dict[str, Any], label: str = "yesterday") -> str:
    names = " and ".join(p.title() for p in res["people"])
    talk = [c for c in res["conversations"] if c["kind"] != "MEDIA" and c["topic"]]
    media = [c for c in res["conversations"] if c["kind"] == "MEDIA"]
    if not talk:
        base = f"I didn't find a conversation between {names} {label}."
        if media:
            base += (
                " The only stretches with both their voices were media playing: "
                + "; ".join(f"{_clock(c['start'])} to {_clock(c['end'])} ({c['topic']})" for c in media)
                + "."
            )
        return base
    ranked = sorted(talk, key=lambda c: -c["minutes"])
    top = ranked[0]
    text = (
        f"{names} talked most about {top['topic'].rstrip('.')} ({top['minutes']} min, "
        f"{_clock(top['start'])} to {_clock(top['end'])}): {top['summary']}"
    )
    if len(ranked) > 1:
        text += (
            " Other conversations: "
            + "; ".join(f"{c['topic'].rstrip('.')} ({c['minutes']} min, {_clock(c['start'])})" for c in ranked[1:6])
            + "."
        )
    if media:
        text += (
            " Not counted, because it was media playing rather than them talking: "
            + "; ".join(f"{_clock(c['start'])} to {_clock(c['end'])} ({c['topic']})" for c in media)
            + "."
        )
    text += (
        f" From {len(res['conversations'])} stretches where both spoke, each summarised by the local model;"
        " speaker labels in the transcript can be wrong."
    )
    return text
