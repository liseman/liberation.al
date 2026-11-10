"""Ask's answers from linked devices: "is Alex home", "when did Sam get here", "where did I last have my
AirPods".

A question is answered here only when an owner link names the person or thing it asks about (proximity/devices.py);
otherwise it goes the way it always did. A person's answer joins their devices' sightings with their face visits
(People), and every answer says these are sightings of a device, not proof of a person. Where no sensor was running
it says it can't tell, never that someone left.
"""

import datetime as dt
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from liberational import config, segment_log
from liberational.proximity import devices

HOME = re.compile(r"\b(?:is|am)\s+(?P<who>[\w'-]+)\s+(?:at\s+home|home|here|around|in)\b\s*\??\s*$", re.I)
ARRIVED = re.compile(
    r"\bwhen\s+did\s+(?P<who>[\w'-]+)\s+(?:get\s+(?:here|home|back|in)|arrive|come\s+(?:home|in|back))\b", re.I
)
THING = re.compile(
    r"\bwhere\s+(?:did\s+(?:i|we)\s+(?:last\s+)?(?:have|leave|see|put)|(?:are|is|were|was))\s+"
    r"(?:my|our|the)?\s*(?P<thing>.+?)(?:\s+last)?\s*\??\s*$",
    re.I,
)
SELF = frozenset({"i", "me"})
DEVICE_NOTE = "These are sightings of a linked device, not proof of who carried it."


def _who(name: str) -> str:
    return config.owner_or() if name.casefold() in SELF else name


def asked(question: str, state: Path) -> tuple[str, str] | None:
    """("home" | "arrived" | "thing", the name) when an owner link names who or what the question asks about."""
    found: tuple[str, str, str] | None = None
    if m := HOME.search(question):
        found = ("home", _who(m["who"]), "person")
    elif m := ARRIVED.search(question):
        found = ("arrived", _who(m["who"]), "person")
    elif m := THING.search(question):
        found = ("thing", m["thing"].strip(), "thing")
    if found is None:
        return None
    try:
        links = devices.at(state).links()
    except Exception:
        # Unknown link evidence must not turn a recognized device question into an ungrounded model answer.
        return (found[0], found[1]) if found[0] == "home" else None
    kind, name, target = found
    hit = next((x for x in links if x.target == target and x.name.casefold() == name.casefold()), None)
    return (kind, hit.name) if hit is not None else None


def unlinked_home(question: str, state: Path) -> str | None:
    """The name in "is X home" when no owner link names X, for Ask to answer from camera and voice presence instead
    (memory/history_qa.py); None for any other question, or one a link answers."""
    m = HOME.search(question)
    if m is None or asked(question, state) is not None:
        return None
    return _who(m["who"])


def _clock(ts: float, tz: dt.tzinfo | None, now: float) -> str:
    t = dt.datetime.fromtimestamp(ts, tz)
    today = dt.datetime.fromtimestamp(now, tz).date()
    when = (
        "today"
        if t.date() == today
        else "yesterday"
        if t.date() == today - dt.timedelta(days=1)
        else t.date().isoformat()
    )
    return f"{t:%H:%M} {when}"


def _ago(seconds: float) -> str:
    minutes = int(seconds / 60 + 0.5)
    return "just now" if minutes < 1 else f"{minutes} min ago" if minutes < 120 else f"{int(minutes / 60 + 0.5)} h ago"


def _signal(record: Mapping[str, Any]) -> str:
    value = devices.rssi(record)
    return f", signal {value:g} dBm" if value is not None else ""


def _visits(state: Path, name: str, start: float, end: float) -> list[dict[str, Any]]:
    try:
        rows = segment_log.at(state).range(segment_log.FACE_VISITS, start, end)
    except Exception:
        return []
    return sorted(
        (v for v in rows if str(v.get("name", "")).casefold() == name.casefold()), key=lambda v: v["ts_start"]
    )


def _review(ts: float, stamp: Callable[[float], str]) -> str:
    return "/review#" + urlencode(dict(day=stamp(ts)[:10], t=ts))


def _unavailable(error: Exception) -> dict[str, Any]:
    return dict(
        answer="I can't tell: linked-device evidence is unavailable.",
        evidence=[],
        coverage={"read_error": True},
        warnings=[type(error).__name__],
    )


def answer(
    kind: str,
    name: str,
    query: Mapping[str, Any],
    state: Path,
    stamp: Callable[[float], str],
    tz: dt.tzinfo | None,
    now: float | None = None,
) -> dict[str, Any]:
    """{answer, evidence, coverage, warnings} for a device question."""
    now = time.time() if now is None else now
    gap = devices.gap_s()
    try:
        reg = devices.at(state)
        if kind == "thing":
            return _thing(reg, name, now, tz, stamp)
    except Exception as error:
        return _unavailable(error)
    start = (
        float(query["start"])
        if float(query.get("start") or 0) > 0
        else now - float(config.get("devices.lookback_h")) * 3600
    )
    if kind == "arrived" and float(query.get("start") or 0) <= 0:
        start = dt.datetime.fromtimestamp(now, tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    try:
        seen = reg.target(name, "person", start, now)
    except Exception as error:
        return _unavailable(error)
    visits = _visits(state, name, start, now)
    coverage = {
        "start": start,
        "stays": [{"arrived": s.start, "last_seen": s.end, "places": list(s.places)} for s in seen.stays]
        if seen
        else [],
        "sensing": [{"start": a, "end": b} for a, b in seen.sensing] if seen else [],
        "face_visits": len(visits),
    }
    evidence = [
        dict(
            ts=s.start,
            source="device",
            link=_review(s.start, stamp),
            quote=f"{name}'s device: {s.sightings} sightings by {', '.join(s.places)}",
            method="presence.device_span-style stay of a linked device (#1329), unconfirmed",
            event=s.first_id,
        )
        for s in (seen.stays if seen else ())
    ] + [
        dict(
            ts=float(v["ts_start"]),
            camera=v.get("cam"),
            source="camera",
            quote=f"{name} seen on {v.get('cam')}",
            link="/review#" + urlencode(dict(day=stamp(float(v["ts_start"]))[:10], t=float(v["ts_start"]))),
            method="face visit (People)",
            event=v.get("id"),
        )
        for v in visits[-5:]
    ]
    if seen is None:
        return dict(answer=f"No device is linked to {name}.", evidence=[], coverage=coverage, warnings=[])
    last = seen.last
    face_tail = ""
    if visits:
        face_tail = (
            f" Last seen on camera at {_clock(float(visits[-1].get('ts_end') or visits[-1]['ts_start']), tz, now)}."
        )
    if kind == "home":
        status = devices.home_status(seen, now, gap)
        if status == "recent":
            assert last is not None
            text = (
                f"Probably: {name}'s device was seen {_ago(now - float(last['end']))} by "
                f"{devices.place(last, last['source'])}{_signal(last)}."
            )
        elif status == "unknown":
            when = f" {name}'s device was last seen at {_clock(float(last['end']), tz, now)}." if last else ""
            text = (
                f"I can't tell: recent coverage is incomplete or cannot establish absence for the linked devices.{when}"
            )
        elif last is not None:
            text = (
                f"Probably not: {name}'s device was last seen at {_clock(float(last['end']), tz, now)} by "
                f"{devices.place(last, last['source'])}, and not since, though the sensors were running."
            )
        else:
            text = f"Probably not: {name}'s device hasn't been seen recently, though the sensors were running."
        return dict(answer=text + face_tail + " " + DEVICE_NOTE, evidence=evidence, coverage=coverage, warnings=[])
    # arrived
    if seen.stays:
        stay = seen.stays[-1]
        text = f"{name}'s device arrived at {_clock(stay.start, tz, now)}, heard by {', '.join(stay.places)}"
        text += "." if now - stay.end <= gap else f", and was last seen at {_clock(stay.end, tz, now)}."
        if len(seen.stays) > 1:
            earlier = ", ".join(_clock(s.start, tz, now) for s in seen.stays[:-1])
            text += f" It also arrived earlier at {earlier}."
    elif devices.home_status(seen, now, gap) == "not_seen":
        text = f"{name}'s device wasn't seen in the available network observations."
    else:
        text = "I can't tell when the person arrived: no linked-device sighting establishes an arrival."
    if visits:
        text += f" First seen on camera at {_clock(float(visits[0]['ts_start']), tz, now)}."
    return dict(answer=text + " " + DEVICE_NOTE, evidence=evidence, coverage=coverage, warnings=[])


def _lives(reg: devices.Devices, name: str) -> str:
    """ "Your TV lives in the Living room (Home), as you set it. " when a device linked as the thing called `name` has
    a location, else ""."""
    mine = {(x.kind, x.value) for x in reg.links() if x.target == "thing" and x.name.casefold() == name.casefold()}
    where = next((x for x in reg.locations() if (x.kind, x.value) in mine), None)
    if where is None:
        return ""
    names = {o["id"]: o["name"] for o in devices.location_options()}
    return f"Your {name} lives in {where.words(names)}, as you set it. "


def _thing(
    reg: devices.Devices, name: str, now: float, tz: dt.tzinfo | None, stamp: Callable[[float], str]
) -> dict[str, Any]:
    """The newest sighting of a thing, a day at a time back to devices.thing_lookback_days, after where it lives when
    the owner said so (a device's location)."""
    days = int(config.get("devices.thing_lookback_days"))
    lives = _lives(reg, name)
    end = now
    for _ in range(days):
        start = end - 86400.0
        seen = reg.target(name, "thing", start, end)
        if seen is None:
            return dict(answer=f"Nothing is named {name}.", evidence=[], coverage={}, warnings=[])
        if seen.last is not None:
            last = seen.last
            at = float(last["end"])
            where = devices.place(last, last["source"])
            text = lives + (
                f"Your {name} was last seen at {_clock(at, tz, now)} ({_ago(now - at)}) by {where}{_signal(last)}; "
                "a stronger signal (nearer 0) means closer to that sensor. "
                + DEVICE_NOTE.replace("who carried it", "where it is now")
            )
            evidence = [
                dict(
                    ts=at,
                    source="device",
                    link=_review(at, stamp),
                    quote=f"{name} heard by {where}",
                    method="bluetooth.sighting",
                    event=last["id"],
                )
            ]
            return dict(
                answer=text,
                evidence=evidence,
                coverage={"last_seen": at, "place": where, "rssi": devices.rssi(last)},
                warnings=[],
            )
        end = start
    return dict(
        answer=lives + f"Your {name} hasn't been seen in the last {days} days.",
        evidence=[],
        coverage={"last_seen": None},
        warnings=["Not seen is not proof it is gone: it may be off, out of range, or a sensor may have been down."],
    )
