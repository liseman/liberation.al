"""Activity questions that aren't counts (#1863), from see-actions' model: when someone sat down, what someone was doing.

- "When did I sit down?" (`when`): the times of that kind's `action.span` rows over the window, over the time
  `action.coverage` says was watched. Until the owner's ratings check a kind (memory/events.py's promotion), each time
  is marked unverified.
- "What did Sam do in the kitchen?" (`doing` with a zone): the model's top classes (`action.window`, NTU-120 or Kinetics-600 names)
  over the tracks that visited that zone (`zone.visit`, see-actions' track ids), and, when a person is named, only
  the tracks whose face named them (`face.match`).
- "What was I doing at 3?" (`doing` without a zone): the same over every track, or the named person's.
- Stillness and repeated movement, which the model doesn't detect (`unsupported`), say so plainly.

Model classes are guesses, never confirmed. Time `action.coverage` doesn't cover is unknown, never "nothing".
"""

import collections
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode

from liberational import segment_log
from liberational.memory import event_counts

UNSUPPORTED = event_counts.UNSUPPORTED
DOING = re.compile(
    r"\bwhat\s+(?:was|were|is|am|are|did|has|have)\s+(?P<who>[a-z]+)\s+(?:been\s+)?(?:doing|do|up\s+to)\b", re.I
)
ZONE = re.compile(
    r"\b(?:in|at|by|near)\s+(?:the\s+|my\s+)?(?P<zone>[a-z][a-z ]{1,40}?)(?=\s+(?:today|yesterday|last|this|at|on|between|from|around|before|after)\b|[?.!]|$)",
    re.I,
)
SOMEONE = frozenset({"i", "me", "we", "someone", "anyone", "anybody", "somebody", "they", "he", "she", "you"})
TOP_GUESSES = 3
MAX_EVIDENCE = 10


@dataclass(frozen=True)
class Asked:
    mode: Literal["when", "doing", "unsupported"]
    kind: str | None = None  # a model kind, for `when`
    zone: str | None = None  # a drawn zone's name, for `doing`
    who: str | None = None  # a named person, for `doing`


def _zone(question: str, state: Any) -> str | None:
    names = event_counts.zone_names(state)
    m = ZONE.search(question)
    if m is None:
        return None
    said = m["zone"].strip().lower()
    return next((n for n in names if n.lower() == said or event_counts.same(n, said)), None)


def asked(question: str, state: Any) -> Asked | None:
    """What a non-count activity question asks, or None when it isn't one (counts are event_counts')."""
    from liberational.memory import activity_kinds

    if event_counts.TRANSCRIPT_ACTION.search(question):
        return None
    if event_counts.action_kind(question) is not None:
        return None  # a count: event_counts' while answer_counts is on, else the usual history answer
    # A kind's own name ("when was phone_use detected"), else its plain words.
    kind = next(
        (k for k in sorted(activity_kinds.model_kinds(state)) if "_" in k and re.search(rf"\b{k}\b", question, re.I)),
        None,
    ) or next((k for k, pattern in event_counts._ACTION_WORDS if pattern.search(question)), None)
    if kind is not None and not DOING.search(question):
        # "When did I sit down", "how long was I on my phone", "did anyone eat": when the model saw it.
        return Asked("when", kind=kind)
    if (m := DOING.search(question)) is not None:
        who = m["who"].lower()
        return Asked("doing", zone=_zone(question, state), who=None if who in SOMEONE else m["who"])
    if event_counts.retired_question(question):
        return Asked("unsupported")
    return None


def _watched(state: Any, start: float, end: float, cameras: list[str]) -> float:
    rows = segment_log.at(state).range(segment_log.ACTION_COVERAGE, start, end)
    return sum(
        max(0.0, min(float(r["ts_end"]), end) - max(float(r["ts_start"]), start))
        for r in rows
        if not cameras or r.get("cam") in cameras
    )


def _link(stamp: Callable[[float], str], ts: float, cam: Any) -> str:
    return "/review#" + urlencode(dict(day=stamp(ts)[:10], t=ts, cam=cam or ""))


def _no_watch(what: str) -> dict[str, Any]:
    return dict(
        answer=f"The camera model hasn't watched any of that time, so I can't say {what}.",
        evidence=[],
        coverage={"available": False, "watched_seconds": 0.0},
        warnings=["Nothing covers the window: this is not a report that nothing happened."],
    )


def _by(e: Mapping[str, Any]) -> str:
    """by Dana, for a span whose person's face was recognized; nothing for someone not recognized."""
    return f" by {e['person_name']}" if e.get("person_name") else ""


def when(
    a: Asked, query: Mapping[str, Any], state: Any, stamp: Callable[[float], str], clock: Callable[[float], str]
) -> dict[str, Any]:
    """When `a.kind` happened in the window, from `action.span`, marked unverified until its kind is checked."""
    from liberational.memory import events

    assert a.kind is not None
    start, end, cameras = float(query["start"]), float(query["end"]), list(query.get("cameras") or [])
    what = a.kind.replace("_", " ")
    watched = _watched(state, start, end, cameras)
    if watched <= 0:
        return _no_watch(f"when anyone did {what}")
    promoted, unverified = events.count_evidence(state, start, end)
    checked = event_counts.uncalibrated(state, a.kind) is None
    found = sorted(
        (
            e
            for e in [*promoted, *unverified]
            if e.get("kind") == a.kind and (not cameras or e.get("camera") in cameras)
        ),
        key=lambda e: float(e["start"]),
    )
    mins = round(watched / 60)
    if not found:
        text = f"The camera model saw no {what} in the {mins} min it watched of that time."
        if not checked:
            text += f" Its {what} detections aren't checked yet, so it may have missed some."
    else:
        times = ", ".join(f"{clock(float(e['start']))} on {e.get('camera')}{_by(e)}" for e in found[-MAX_EVIDENCE:])
        text = f"{what.capitalize()}: {times}."
        if not checked:
            text = f"Unverified (the camera model's {what} detections aren't checked yet): " + text
        if watched < (end - start) - 60:
            text += f" It watched {mins} min of that time; the rest is unknown."
    evidence = [
        dict(
            ts=float(e["start"]),
            camera=e.get("camera"),
            source="camera",
            quote=f"{'' if checked else 'Unverified: '}{what} on {e.get('camera')}{_by(e)}",
            link=_link(stamp, float(e["start"]), e.get("camera")),
            method="see-actions model action (action.span)" + ("" if checked else ", not yet checked by ratings"),
            stream="action.span",
            event=e.get("id"),
        )
        for e in found[-MAX_EVIDENCE:]
    ]
    return dict(
        answer=text,
        evidence=evidence,
        coverage={"available": True, "watched_seconds": watched, "checked": checked, "found": len(found)},
        warnings=[] if checked else ["The model's detections of this kind aren't checked yet."],
    )


def _tracks_of(state: Any, a: Asked, start: float, end: float) -> tuple[set[tuple[str, str]] | None, list[str]]:
    """((camera, track id) the zone and person keep to, or None for every track; and why any were left out)."""
    keep: set[tuple[str, str]] | None = None
    notes: list[str] = []
    log = segment_log.at(state)
    if a.zone is not None:
        from liberational.memory import zone_counts

        ids = set(zone_counts._matching(state, a.zone, event_counts.same))
        visits = [v for v in log.range(segment_log.ZONE_VISITS, start, end) if v.get("zone") in ids]
        keep = {(str(v.get("cam")), str(v.get("track_id"))) for v in visits if v.get("subject_kind") == "person"}
    if a.who is not None:
        named = {
            (str(m.get("cam")), str(m.get("track_id")))
            for m in log.range(segment_log.FACE_MATCHES, start, end)
            if isinstance(m.get("name"), str) and event_counts.same(m["name"], a.who)
        }
        keep = named if keep is None else keep & named
        notes.append(f"Only tracks whose face was recognized as {a.who} are counted.")
    return keep, notes


def _guesses(state: Any, a: Asked, start: float, end: float, cameras: list[str]) -> list[tuple[str, int, str]]:
    """The model's most frequent top classes over the window: (class name, windows, a camera), most first."""
    from liberational.see import action_classes, action_clips, movinet_classes

    keep, _ = _tracks_of(state, a, start, end)
    tally: collections.Counter[str] = collections.Counter()
    where: dict[str, str] = {}
    for w in segment_log.at(state).range(segment_log.ACTION_WINDOWS, start, end):
        cam = str(w.get("cam"))
        if (cameras and cam not in cameras) or (keep is not None and (cam, str(w.get("track_id"))) not in keep):
            continue
        classes = w.get("classes") or []
        # MoViNet's windows name Kinetics-600 classes, ST-GCN++'s NTU-120 ones
        names = movinet_classes.CLASSES if w.get("model") == action_clips.MODEL else action_classes.CLASSES
        if not classes or not 0 <= int(classes[0]) < len(names):
            continue
        name = names[int(classes[0])]
        tally[name] += 1
        where.setdefault(name, cam)
    return [(name, n, where[name]) for name, n in tally.most_common(TOP_GUESSES)]


def doing(a: Asked, query: Mapping[str, Any], state: Any, stamp: Callable[[float], str]) -> dict[str, Any]:
    """What someone did (in a zone, when one is named), as the model's top guesses."""
    start, end, cameras = float(query["start"]), float(query["end"]), list(query.get("cameras") or [])
    who = a.who or "anyone"
    place = f" in {a.zone}" if a.zone else ""
    watched = _watched(state, start, end, cameras)
    if watched <= 0:
        return _no_watch(f"what {who} did{place}")
    keep, notes = _tracks_of(state, a, start, end)
    if keep is not None and not keep:
        text = f"I didn't see {who}{place} in the time the camera model watched ({round(watched / 60)} min)."
        return dict(answer=text, evidence=[], coverage={"available": True, "watched_seconds": watched}, warnings=[])
    guesses = _guesses(state, a, start, end, cameras)
    if not guesses:
        text = f"The camera model has no guess at what {who} did{place}: it scored no movement long enough to read."
    else:
        parts = ", ".join(f"{name} ({n} {'look' if n == 1 else 'looks'})" for name, n, _ in guesses)
        text = f"The camera model's best guesses at what {who} did{place}: {parts}. These are model guesses, not confirmed."
    text = " ".join([text, *notes])
    evidence = guess_evidence(guesses, start, stamp)
    return dict(answer=text, evidence=evidence, coverage={"available": True, "watched_seconds": watched}, warnings=[])


def guess_evidence(guesses: list[tuple[str, int, str]], at: float, stamp: Callable[[float], str]) -> list[dict]:
    return [
        dict(
            ts=at,
            camera=cam,
            source="camera",
            quote=f"Camera model's guess (unverified): {name}, top class in {n} of its looks",
            link=_link(stamp, at, cam),
            method="see-actions action model's top class per scored window (action.window), a guess",
            stream="action.window",
        )
        for name, n, cam in guesses
    ]
