"""Liberational Pack: the "who is this" whisper.

Follows the ambient transcript (the speech log on Continuum, each line's person per the event log through
people/transcript_view.py). When a known voice other than the owner's speaks and hasn't been announced recently, it
speaks ONE short line in the owner's ear via the clone voice:

    "Alice. Last here Tuesday. She just got health insurance."

Facts come from state/people/<name>.md (people/profiles.py). Unknown voices are not announced: there is nothing to
say about them. No LLM in the loop by default (--llm polishes the line with the local LLM).

Run:  uv run python -m liberational.people.whisper_names            # daemon (service: whisper-names)
      ... --once --dry-run                                # evaluate current state, print, do not speak
      ... --test alice                                    # build + speak the line for a name
State: state/whisper_names.json  {announced: {key: ts}, seen: {key: [ts,...]}}
"""

import argparse
import json
import os
import re
import time
from collections.abc import Callable
from typing import Any

from liberational import config as cfg
from liberational import logs, ollama, speech_log
from liberational.people import profiles, transcript_view

STATE = cfg.path("paths.state")
PEOPLE = STATE / "people"
OUT = STATE / "whisper_names.json"
DEBOUNCE_S = float(os.environ.get("PACK_WHISPER_DEBOUNCE", "1800"))  # one line per person per 30 min


LOG = logs.get("whisper-names")


def log(*a: object, level: int = logs.INFO) -> None:
    LOG.log(level, " ".join(str(x) for x in a))


def load_state() -> dict[str, Any]:
    try:
        return json.loads(OUT.read_text())
    except Exception:
        return {"announced": {}, "seen": {}}


def save_state(s: dict[str, Any]) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(s))
    os.replace(tmp, OUT)


def norm(n: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (n or "").strip().lower()).strip("-")


def is_owner(name: str | None) -> bool:
    """Never announce the owner (matched by name or speaker label); nobody matches when no owner is set."""
    return norm(name) in {norm(cfg.owner_name()), norm(cfg.owner_label())} - {""}


def when(ts: float, now: float | None = None) -> str:
    now = now or time.time()
    d = now - ts
    if d < 3600:
        return f"{int(d // 60)} minutes ago"
    if d < 86400:
        return f"{int(d // 3600)} hours ago"
    if d < 7 * 86400:
        return time.strftime("%A", time.localtime(ts))
    return time.strftime("%B %-d", time.localtime(ts))


def facts_for(name: str, n: int = 2) -> list[str]:
    found = profiles.files_of(PEOPLE, [name])  # "dana cole.md", as profiles writes it, not "dana-cole.md"
    if not found:
        return []
    p = found[0]
    bullets = [l[2:].strip() for l in p.read_text().splitlines() if l.startswith("- ")]
    first = norm(name).split("-")[0]
    # only facts phrased about the person ("Alice ...", "Alice's ..."), under 90 chars: short enough to say, less ASR noise
    bullets = [b for b in bullets if re.match(rf"{re.escape(first)}\b", b, re.I) and len(b) <= 90]
    return bullets[-n:]


def is_auto_label(label: str | None) -> bool:
    return not label or bool(re.fullmatch(r"Speaker \d+", label)) or label.lower() == "unknown"


class TranscriptTail:
    """The utterances committed to the speech log since the last call (from when it was made)."""

    def __init__(self) -> None:
        self.seq = speech_log.shared().mark()[0]

    def events(self, now: float) -> list[dict[str, Any]]:
        lines, self.seq = speech_log.shared().since(self.seq)
        out = []
        fresh = [
            e
            for e in lines
            if now - e.get("ts_end", 0) <= 30  # not stale backlog
            and (e.get("text") or "").strip()
        ]
        for e in transcript_view.view(fresh, STATE) if fresh else ():  # who said it, per the event log
            label = e.get("speaker")
            if is_owner(label):
                continue
            if is_auto_label(label):
                continue  # an unknown voice: nothing to say about it
            out.append({"key": "voice:" + norm(label), "name": label, "ts": e.get("ts_end", now)})
        return out


# ---------------------------------------------------------------- line
def build_line(ev: dict[str, Any], state: dict[str, Any], now: float) -> str:
    seen = state["seen"].get(ev["key"], [])
    first = ev["name"].split()[0]
    parts = [first.capitalize() + "."]
    parts.append(f"Last here {when(seen[-1], now)}." if seen else "First time I've logged them.")
    for f in facts_for(ev["name"], 1):
        f = re.sub(r"^" + re.escape(first) + r"'s\b", "Their", f, flags=re.I)
        f = re.sub(r"^" + re.escape(first) + r"\b", "They", f, flags=re.I)
        f = re.sub(
            r"^They (is|has|was|does|wants|needs|likes|plans|works|lives)\b",
            lambda m: (
                "They " + {"is": "are", "has": "have", "was": "were", "does": "do"}.get(m.group(1), m.group(1)[:-1])
            ),
            f,
        )
        parts.append(f.rstrip(".") + ".")
    return " ".join(parts)


def polish(line: str) -> str:
    """Optional: rephrase into one spoken sentence with the fast local model. Keeps facts; never adds."""
    try:
        reply = ollama.chat(
            cfg.get("models.ollama.llm"),
            [
                {
                    "role": "system",
                    "content": "Rewrite as one natural spoken sentence, max 15 words. Keep every fact, add nothing, no names you were not given.",
                },
                {"role": "user", "content": line},
            ],
            num_ctx=4096,
            num_predict=40,
            temperature=0.2,
            think=False,
            truncate=False,
            timeout=8,
            background=True,
        )
        out = reply["message"]["content"].strip()
        return out if 0 < len(out) <= 160 else line
    except Exception:
        return line


# ---------------------------------------------------------------- main loop
def step(
    tail: TranscriptTail,
    state: dict[str, Any],
    now: float,
    speak: Callable[[str], object] | None,
    use_llm: bool,
    dry: bool,
) -> int:
    events = tail.events(now)
    spoken = 0
    for ev in events:
        last = state["announced"].get(ev["key"], 0)
        if now - last < DEBOUNCE_S:
            state["seen"].setdefault(ev["key"], [])
            continue
        line = build_line(ev, state, now)
        if use_llm:
            line = polish(line)
        log(f"{ev['key']} -> {line!r}")
        state["announced"][ev["key"]] = now
        state["seen"].setdefault(ev["key"], []).append(now)
        state["seen"][ev["key"]] = state["seen"][ev["key"]][-50:]
        if not dry and speak:
            speak(line)
        spoken += 1
        break  # one line per tick; the rest wait for the next pass (and may be debounced by then)
    # record sightings even when debounced, so "last here" stays current
    for ev in events:
        s = state["seen"].setdefault(ev["key"], [])
        if not s or now - s[-1] > 300:
            s.append(now)
    if events or spoken:
        save_state(state)
    return spoken


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--llm", action="store_true", help="polish the line with the local LLM (models.ollama.llm)")
    ap.add_argument("--test", metavar="NAME", help="build and speak the line for NAME and exit")
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    state = load_state()
    now = time.time()
    speak = None
    if not a.dry_run:
        from liberational.speech import voice_clone

        t0 = time.time()
        voice_clone.load()
        voice_clone.say("Ready.", out=None)
        log(f"voice ready in {time.time() - t0:.1f}s")
        speak = lambda text: voice_clone.say(text, play=True)
    if a.test:
        ev = {"key": "voice:" + norm(a.test), "name": a.test, "ts": now}
        line = build_line(ev, state, now)
        line = polish(line) if a.llm else line
        print(line)
        speak and speak(line)
        return
    tail = TranscriptTail()
    log(f"following the speech log; debounce {DEBOUNCE_S:.0f}s")
    while True:
        try:
            step(tail, state, time.time(), speak, a.llm, a.dry_run)
        except Exception as ex:
            log(f"{ex}", level=logs.ERROR)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
