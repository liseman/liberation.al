"""How long the room's electric lights were on, measured from the recorded video's colour cast (the cameras' frames on
the broker).

The cameras adjust exposure automatically, so overall brightness hardly changes between night and evening, and a
brightness threshold called the lights "on" all day. The colour cast does change: the workshop's lamps
are warm, so with them on the picture's blue channel drops well below green; at night the picture is neutral (only
device LEDs), and daylight is neutral too but brighter. Checked by eye on cam0 frames at 10:00 and 14:00 (lamps lit,
warm), 12:30 (daylight, neutral and bright) and 22:30 (dark, neutral).

Per camera and day: one frame every STEP_S from the recording, its mean colour, and the blue-minus-green gap. The
day's night baseline is the median gap of its darkest samples; a sample is "lamps on" when its gap is well below
that baseline (more than half the way to the day's warmest tenth). Cameras vote per slot. Ended days are cached.
"""

import datetime as dt
import json
import pathlib
from statistics import median
from typing import Any

import cv2
import numpy as np

from liberational import vision_log

STEP_S = 300
MIN_SPREAD = 8.0  # a day whose warmest samples are within this of the night baseline never had the lamps on
DAYLIGHT_LIFT = 10.0  # neutral samples this much brighter than the night's are daylight, not dark
VERSION = 2


SAMPLE_WITHIN_S = 2.0  # a sample is the camera's frame nearest its time, this close


def _effective_timezone(tz: dt.tzinfo | None) -> dt.tzinfo:
    if tz is not None:
        return tz
    local = dt.datetime.now().astimezone().tzinfo
    return local if local is not None else dt.UTC


def _colour(frame: Any) -> tuple[float, float, float] | None:
    """A frame's mean (red, green, blue), read on a 64 by 36 copy, or None for no frame."""
    if frame is None:
        return None
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    b, g, r = (float(small[:, k].mean()) for k in range(3))
    return r, g, b


def sample(state: pathlib.Path, cam: str, start: float, end: float, step: float = STEP_S) -> list[dict[str, float]]:
    """(ts, mean, gap) every step seconds in [start, end) where the camera has video on the broker (`<cam>.video`)."""
    log = vision_log.at(state)
    rows: list[dict[str, float]] = []
    t = start
    while t < end:
        c = _colour(log.frame(cam, t, SAMPLE_WITHIN_S))
        if c is not None:
            rows.append({"ts": t, "mean": (c[0] + c[1] + c[2]) / 3, "gap": c[2] - c[1]})
        t += step
    return rows


def classify(rows: list[dict[str, float]]) -> list[str]:
    """'on' (lamps), 'daylight' or 'dark' for each sample of one camera over one day."""
    if not rows:
        return []
    by_mean = sorted(rows, key=lambda r: r["mean"])
    dark = by_mean[: max(3, len(rows) // 10)]
    base = median(r["gap"] for r in dark)
    night_mean = median(r["mean"] for r in dark)
    warmest = sorted(r["gap"] for r in rows)[: max(3, len(rows) // 10)]
    spread = base - median(warmest)
    cut = base - max(MIN_SPREAD, spread / 2)
    out = []
    for r in rows:
        if spread >= MIN_SPREAD and r["gap"] <= cut:
            out.append("on")
        elif r["mean"] >= night_mean + DAYLIGHT_LIFT:
            out.append("daylight")
        else:
            out.append("dark")
    return out


def _spans(slots: list[tuple[float, str]], label: str, step: float) -> list[tuple[float, float]]:
    spans: list[list[float]] = []
    for ts, s in slots:
        if s != label:
            continue
        if spans and ts - spans[-1][1] <= step * 0.5:  # the very next slot; any other label between ends a span
            spans[-1][1] = ts + step
        else:
            spans.append([ts, ts + step])
    return [(a, b) for a, b in spans]


def analyse_day(
    state: pathlib.Path,
    day: dt.date,
    cameras: tuple[str, ...] = ("cam0", "cam1", "cam2"),
    until: float | None = None,
    tz: dt.tzinfo | None = None,
) -> dict[str, Any]:
    zone = _effective_timezone(tz)
    start_dt = dt.datetime.combine(day, dt.time(), tzinfo=zone)
    next_day_dt = start_dt + dt.timedelta(days=1)
    start = start_dt.timestamp()
    day_end = next_day_dt.timestamp()
    end = min(day_end, until or day_end)
    ended = end >= day_end and day < dt.datetime.now(zone).date()
    cache = state / "lights" / f"{day.isoformat()}.json"
    if ended and cache.exists():
        try:
            got = json.loads(cache.read_text())
            if got.get("version") == VERSION:
                return got
        except OSError, ValueError:
            pass
    votes: dict[float, list[str]] = {}
    per_cam = {}
    for cam in cameras:
        rows = sample(state, cam, start, end)
        labels = classify(rows)
        per_cam[cam] = {"samples": len(rows), "on": labels.count("on"), "daylight": labels.count("daylight")}
        for r, lab in zip(rows, labels):
            votes.setdefault(r["ts"], []).append(lab)
    slots = []
    for ts in sorted(votes):
        v = votes[ts]
        on = v.count("on")
        slots.append((ts, "on" if on * 2 > len(v) else ("daylight" if v.count("daylight") * 2 > len(v) else "dark")))
    res = {
        "version": VERSION,
        "day": day.isoformat(),
        "step_s": STEP_S,
        "cameras": per_cam,
        "measured_s": len(slots) * STEP_S,
        "window_s": end - start,
        "on": _spans(slots, "on", STEP_S),
        "daylight": _spans(slots, "daylight", STEP_S),
    }
    res["on_s"] = sum(b - a for a, b in res["on"])
    res["daylight_s"] = sum(b - a for a, b in res["daylight"])
    if ended:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(res))
    return res


def _hm(seconds: float) -> str:
    m = round(seconds / 60)
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def _clock(ts: float, tz: dt.tzinfo | None = None) -> str:
    return dt.datetime.fromtimestamp(ts, tz).strftime("%H:%M")


def answer_text(res: dict[str, Any], label: str = "yesterday", tz: dt.tzinfo | None = None) -> str:
    if not res["measured_s"]:
        return f"I can't tell how long the lights were on {label}: there's no recorded video for that time."
    spans = ", ".join(f"{_clock(a, tz)} to {_clock(b, tz)}" for a, b in res["on"]) or "never"
    text = f"The lights were on for about {_hm(res['on_s'])} {label}: {spans}."
    if res["daylight"]:
        text += (
            " The room was lit by daylight "
            + ", ".join(f"{_clock(a, tz)} to {_clock(b, tz)}" for a, b in res["daylight"])
            + "; the lamps may also have been on then, which the cameras can't tell apart from daylight."
        )
    gap = res["window_s"] - res["measured_s"]
    if gap > STEP_S * 2:
        text += f" About {_hm(gap)} had no recording to check."
    text += (
        " Measured from the cameras' colour every 5 minutes: the lamps give the picture a warm cast that"
        " daylight and the dark room don't."
    )
    return text
