"""Questions about what the cameras saw: "what did cam0 see in the last hour", "what was on camera at 3pm". The
captions come from the vision log (`camera.caption` on Continuum); this reads them, it keeps nothing."""

import argparse
import dataclasses
import datetime as dt
import itertools
import json
import re
import time
from collections.abc import Mapping
from typing import Any

from liberational import vision_log
from liberational.vision import views

RETENTION = 7 * 86400
GAP = 30
WORDS = dict(
    zip(
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"],
        range(13),
    )
)
WORDS.update(a=1, an=1)
NEAR = 300  # a clock-time question ("at 3pm") reads the caption nearest that time, within this many seconds

CLOCK = re.compile(
    r"\b(?:at|around|about)\s+(?:(noon|midday|midnight)|(\d{1,2})(?::(\d\d))?\s*([ap])\.?\s?m\b\.?|(\d{1,2}):(\d\d))",
    re.I,
)
# Any other time word beside a clock time ("two days ago at 3pm", "at 3pm last week"): a clock time means today or
# yesterday only, so the rest would be dropped and the answer would come from the wrong day.
OTHER_TIME = re.compile(
    r"\b(?:last|past|previous|since|ago|earlier|days?|weeks?|months?|years?|tomorrow|tonight|morning|afternoon|"
    r"evening|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\d{4}-\d{2}-\d{2}",
    re.I,
)
# Parts of the day are refused until their hours are defined in one place.
PART_OF_DAY = re.compile(r"\b(?:this|in the)\s+(?:morning|afternoon|evening)\b", re.I)


@dataclasses.dataclass(frozen=True)
class At:
    """A moment asked about by its clock time ("at 3pm", "yesterday at 15:30"), as an epoch time."""

    ts: float


CAMERA_Q = re.compile(r"\b(?:cameras?|cams?\s*(?:[0-9]+|zero|one|two)|what do you see)\b", re.I)


def number(s: str) -> float:
    return WORDS[s] if s in WORDS else float(s)


def at_clock(q: str, now: float) -> At | None:
    """The moment a clock time in the request names: today's when it has passed, else yesterday's (or yesterday's
    when the request says so). None when it names no clock time; ValueError for one that can't be (25:00, or a time
    later today asked about with "today")."""
    m = CLOCK.search(q)
    if not m:
        return None
    if OTHER_TIME.search(q[: m.start()] + " " + q[m.end() :]):
        raise ValueError("Please ask about one time: a clock time today or yesterday, such as yesterday at 3pm.")
    word, h12, m12, ap, h24, m24 = m.groups()
    if word:
        hour, minute = (0, 0) if word.lower() == "midnight" else (12, 0)
    elif h24 is not None:
        hour, minute = int(h24), int(m24)
    else:
        hour, minute = int(h12), int(m12 or 0)
        if not 1 <= hour <= 12:
            raise ValueError("Please say a time like 3pm or 15:00.")
        hour = hour % 12 + (12 if ap.lower() == "p" else 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("Please say a time like 3pm or 15:00.")
    today = dt.datetime.fromtimestamp(now).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if re.search(r"\byesterday\b", q):
        moment = today - dt.timedelta(days=1)
    elif today.timestamp() <= now:
        moment = today
    elif re.search(r"\btoday\b", q):
        raise ValueError("That time hasn't come yet today.")
    else:
        moment = today - dt.timedelta(days=1)  # the last time it was that time
    if now - moment.timestamp() > RETENTION:
        raise ValueError("History requests must be between 1 second and 7 days.")
    return At(moment.timestamp())


def parse(request: str, now: float | None = None) -> tuple[list[str], float | At | None]:
    """The cameras a request names (all of them when it names none) and the time it asks about: a window of the
    last so many seconds ("in the last 2 hours"), a moment by its clock time ("at 3pm"), or None for now. A past
    time it can't read raises ValueError, so it is never answered with what the cameras show now."""
    configured_cameras, remaining = views.configured_camera_mentions(request)
    q = remaining.lower()
    cams = re.findall(r"\bcam(?:era)?\s*(\d+|zero|one|two|three|four|five)\b", q)
    cams = sorted(set(configured_cameras) | {"cam%d" % number(x) for x in cams}) or views.camera_ids()
    m = re.search(r"\b(?:last|past|previous)\s+(?:(\d+(?:\.\d+)?|[a-z]+)\s+)?(seconds?|minutes?|hours?|days?)\b", q)
    seconds = None
    if m:
        try:
            seconds = number(m[1] or "one") * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m[2][0]]
        except KeyError, ValueError:
            raise ValueError("Please specify a numeric duration, such as last 1 hour.")
        if not 0 < seconds <= RETENTION:
            raise ValueError("History requests must be between 1 second and 7 days.")
        return cams, seconds
    moment = at_clock(q, time.time() if now is None else now)
    if moment is not None:
        return cams, moment
    if PART_OF_DAY.search(q):
        raise ValueError("Please say a time like 3pm, or a window such as last 2 hours.")
    if re.search(r"\b(last|past|previous|yesterday|since|ago)\b", q):
        raise ValueError("Please specify a relative window, such as last 1 hour.")
    return cams, None


def query(request: str, log: vision_log.VisionLog | None = None, now: float | None = None) -> dict[str, Any]:
    """The answer to a question about what the cameras saw, from the vision log (`vision_log.at()` by default):
    each camera's captions over the window it asks about (the last ten minutes when it names none), or nearest the
    moment it names."""
    now = time.time() if now is None else now
    log = log or vision_log.at()
    cams, window = parse(request, now)
    if isinstance(window, At):
        return moment_query(request, cams, window, log, now)
    seconds = window
    start = now - (seconds if seconds is not None else 600)
    result: dict[str, Any] = {"request": request, "start": start, "end": now, "window_seconds": seconds, "cameras": []}
    for cam in cams:
        rows = [(r["ts"], r["caption"]) for r in log.captions(start, now + 0.001, [cam]) if r.get("caption")]
        last = rows[-1][0] if rows else _last_seen(log, cam, now)
        gaps = [b[0] - a[0] for a, b in itertools.pairwise(rows)]
        partial = not rows or rows[0][0] > start + GAP or rows[-1][0] < now - GAP or any(g > GAP for g in gaps)
        changes = []
        previous = None
        for ts, cap in rows:
            normalized = re.sub(r"^(same|changed)\b[.:\s-]*", "", cap, flags=re.I).strip()
            # VLM Changed labels are observations, not verified physical changes.
            if re.match(r"^changed\b", cap, re.I) and normalized != previous:
                changes.append({"ts": ts, "caption": cap})
            previous = normalized
        result["cameras"].append(
            dict(
                cam=cam,
                count=len(rows),
                first=rows[0][0] if rows else None,
                last=last,
                stale=last is not None and now - last > GAP,
                partial=partial,
                max_gap_seconds=max(gaps, default=0),
                change_count=len(changes),
                changes=changes[-5:],
                changes_truncated=len(changes) > 5,
                latest=rows[-1][1] if rows else None,
            )
        )
    result["answer"] = render(result)
    return result


def _last_seen(log: vision_log.VisionLog, cam: str, now: float) -> float | None:
    """When `cam` was last captioned before `now`, looked for over the last day."""
    rows = log.captions(now - 86400, now + 0.001, [cam])
    return rows[-1]["ts"] if rows else None


def moment_query(request: str, cams: list[str], at: At, log: vision_log.VisionLog, now: float) -> dict[str, Any]:
    """A clock-time question: each camera's caption nearest that moment, within NEAR seconds. Never the current view:
    with nothing near it, the answer says so."""
    result: dict[str, Any] = {"request": request, "at": at.ts, "window_seconds": None, "cameras": []}
    for cam in cams:
        rows = log.captions(at.ts - NEAR, min(at.ts + NEAR, now) + 0.001, [cam])
        near = min(rows, key=lambda r: abs(r["ts"] - at.ts), default=None)
        result["cameras"].append(
            dict(
                cam=cam,
                ts=near["ts"] if near else None,
                caption=near.get("caption") if near else None,
                since=near.get("since") if near and near.get("unchanged") else None,
            )
        )
    lines = [
        f"{c['cam']} at {stamp(c['ts'])}: "
        + (f"unchanged since {stamp(c['since'])}: " if c.get("since") else "")
        + c["caption"]
        if c["caption"]
        else f"{c['cam']} at {stamp(c['ts'])}: no caption (the vision model was starting up)."
        if c["ts"]
        else f"{c['cam']}: no observations within {NEAR // 60} minutes of {stamp(at.ts)}."
        for c in result["cameras"]
    ]
    result["answer"] = " ".join(lines)
    return result


def stamp(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def render(result: Mapping[str, Any]) -> str:
    lines = []
    for c in result["cameras"]:
        name = c["cam"]
        if not c["count"]:
            lines.append(
                f"{name}: no observations in the requested window."
                + (f" Last observation {stamp(c['last'])}, stale." if c["last"] else " No saved history.")
            )
            continue
        if result["window_seconds"] is None:
            lines.append(
                f"{name}, observed at {stamp(c['last'])}" + (" (stale)" if c["stale"] else "") + ": " + c["latest"]
            )
            continue
        status = (
            "Partial coverage; cannot account for the whole requested window"
            if c["partial"]
            else "Sampled coverage, not continuous video"
        )
        line = f"{name}: {status}. {c['count']} observations from {stamp(c['first'])} to {stamp(c['last'])}."
        if c["stale"]:
            line += " Latest observation is stale."
        if c["changes"]:
            line += (
                " Caption-reported changes (not independently verified)"
                + ("; latest five" if c["changes_truncated"] else "")
                + ": "
                + "; ".join(stamp(e["ts"]) + " " + e["caption"] for e in c["changes"])
            )
        else:
            line += " No captions explicitly reported a change in the available samples; this does not prove nothing changed."
        line += " Latest caption: " + c["latest"]
        lines.append(line)
    return " ".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    try:
        r = query(a.question)
    except ValueError as e:
        p.error(str(e))
    print(json.dumps(r, indent=2) if a.json else r["answer"])


if __name__ == "__main__":
    main()
