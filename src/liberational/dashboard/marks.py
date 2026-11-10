"""Right and wrong marks: POST /measurement/mark saves one measurement label.

From People, a mark says whether the name the pack gave one transcript line, or one recorded face sighting, was right.
It is saved as a line of `identification.jsonl` under `paths.measurement` (the format in docs/MEASUREMENT.md), naming
that one record (`line` or `sighting`), so `tools/measure.py --part identification` scores exactly it. A mark never
changes who a line or sighting is assigned to: that is still the owner's tag ("Actually someone else").

From Review's Ask, a mark (`kind: "answer"`) says whether an answer was right, and is saved as a question in
`retrieval.jsonl` for `tools/measure.py --part retrieval`: a right answer's cited records become the evidence it must
find again, and a wrong one's correction becomes the text it must (or must not) say, or the moment it must cite. A mark
changes no answer.

From Review's timeline, a mark (`kind: "event"`) is one span of one camera where something happened, with how many
times it happened counted by watching. It is saved in `events.jsonl` for `tools/measure.py --part events`,
in the format docs/MEASUREMENT.md gives: the camera, what happened in the owner's words, the span, the count, and
optionally who it involved, measured attributes (a value and a unit), and notes.

Also from Review's timeline, a mark (`kind: "entries"`) is one window of one camera with every entry and exit the owner
saw in it, and the stretches they could not tell. It is saved in `entries.jsonl` for `tools/measure.py --part
entries`, in the format docs/MEASUREMENT.md gives. A window with no events and no unsure stretch needs `none_seen`, so
an empty list always means the owner watched it and nobody came or went. Windows on one camera must not overlap: a
mark that overlaps another window is refused unless it says `replace`, which drops the windows it overlaps.

A second mark on the same record, answer, or span of a camera replaces the first. The labels hold names and what was said, so they stay
in data/measurement/, which git never tracks.
"""

import datetime as dt
import itertools
import json
import math
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import json_state
from liberational.dashboard import routes
from liberational.memory.ask import MAX_QUESTION
from liberational.people import registry

if TYPE_CHECKING:
    from liberational.dashboard.server import H

Reply = tuple[int, dict[str, Any]]

FILE = "identification.jsonl"
RETRIEVAL = "retrieval.jsonl"
EVENTS = "events.jsonl"
ENTRIES = "entries.jsonl"
MAX_ENTRY_EVENTS = 500
MAX_UNSURE = 50
CONDITIONS = ("camera_moving", "low_light", "occlusion", "crowded", "no_events")
FEEDBACK = "answer_feedback.jsonl"  # a plain "yes, that was right" with nothing to check
MAX_COUNT = 10_000
MAX_ATTRIBUTES = 20
MAX_NOTES = 50
MAX_NOTE = 200
SOURCES = ("transcript", "camera", "call", "journal")  # the evidence sources tools/measure.py checks
MAX_CITED = 200  # Ask keeps up to 100 records and its search rounds add up to 60, any of which an answer cites
MAX_SAYS = 200
_lock = threading.Lock()  # one rewrite at a time: the dashboard serves requests on threads


def _name(v: object) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() and len(v.strip()) <= 80 else None


def _number(v: object) -> float | None:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v < 0:
        return None
    return float(v)


def label(body: dict[str, Any]) -> dict[str, Any] | str:
    """The label a mark makes, or why the mark can't be one."""
    shown, right, actual = _name(body.get("shown")), body.get("right"), body.get("actual")
    if shown is None:
        return "shown is the name the pack gave, 1 to 80 characters"
    if not isinstance(right, bool):
        return "right is true (the name was right) or false (it's someone else)"
    if actual is not None and (right or _name(actual) is None):
        return "actual names who it really was, and only on a wrong mark"
    actual = _name(actual)
    if actual is not None and actual.casefold() == shown.casefold():
        return "actual is someone other than the name shown"
    kind = body.get("kind")
    if kind == "line":
        start, end = _number(body.get("ts_start")), _number(body.get("ts_end"))
        if start is None or end is None or end < start:
            return "a line needs its ts_start and ts_end"
        device = _name(body.get("device"))
        # the line's key: two microphones' lines in one millisecond are two marks, not one replacing the other
        key = registry.line_key(start, device)
        out: dict[str, Any] = {"id": f"people-line-{key}", "modality": "voice", "start": start, "end": end}
        out["line"] = start
        if device is not None:
            out["device"] = device
    elif kind == "shot":
        sighting, ts, camera = body.get("sighting"), _number(body.get("ts")), _name(body.get("camera"))
        if not isinstance(sighting, str) or not sighting.strip() or ts is None or not camera:
            return "a shot needs its sighting id, ts, and camera"
        out = {"id": f"people-shot-{sighting}", "modality": "face", "camera": camera, "at": ts, "sighting": sighting}
    else:
        return 'kind is "line", "shot", "answer", "event", or "entries"'
    # Right: the person shown was there. Wrong: whoever it really was, if the owner said; else nobody known, so the
    # name shown counts as wrong.
    out["people"] = [shown] if right else [actual] if actual else []
    out["note"] = f"marked {'right' if right else 'wrong'} in People (shown as {shown})"
    return out


def _says(v: object) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() and len(v.strip()) <= MAX_SAYS else None


def _day(v: object) -> bool:
    try:
        return isinstance(v, str) and dt.date.fromisoformat(v).isoformat() == v
    except ValueError:
        return False


def answer(body: dict[str, Any]) -> dict[str, Any] | str:
    """The retrieval label a mark on an Ask answer makes, or why it can't be one.

    A right mark keeps the records the answer cited (each must be found again within a second, from the same source)
    and, optionally, text a right answer says. A wrong mark keeps what the owner says a right answer says, what it
    must not say, or the moment Review was on that it should cite (within 30 s, the measurement default)."""
    question, day, right = body.get("question"), body.get("day"), body.get("right")
    asked = _number(body.get("asked_at"))
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION:
        return f"question is the question as it was asked, 1 to {MAX_QUESTION} characters"
    if asked is None:
        return "asked_at is when it was asked, in epoch seconds"
    if day is not None and not _day(day):
        return "day is the Review day it was asked on, YYYY-MM-DD"
    if not isinstance(right, bool):
        return "right is true (the answer was right) or false (it was wrong)"
    cited, moment = body.get("cited", []), body.get("moment")
    if not isinstance(cited, list) or len(cited) > MAX_CITED:
        return f"cited is the records the answer cited, at most {MAX_CITED}"
    evidence: list[dict[str, Any]] = []
    bad = f"each cited record has its ts, and a source of {', '.join(SOURCES)}, or none"
    for c in cited:
        if not isinstance(c, dict):
            return bad
        ts, source = _number(c.get("ts")), c.get("source")
        if ts is None or (source is not None and source not in SOURCES):
            return bad
        evidence.append({"at": ts, "within_s": 1, **({"source": source} if source else {})})
    texts = {k: body.get(k) for k in ("says", "not_says")}
    if any(v is not None and _says(v) is None for v in texts.values()):
        return f"says and not_says are text, 1 to {MAX_SAYS} characters"
    says, not_says = _says(texts["says"]), _says(texts["not_says"])
    if says and not_says and not_says.casefold() in says.casefold():
        return "not_says is inside says, so the label could never pass"
    if moment is not None and _number(moment) is None:
        return "moment is the time Review was on, in epoch seconds"
    if right and (not_says or moment is not None):
        return "not_says and moment are for a wrong mark"
    if not right and evidence:
        return "cited is for a right mark: a wrong answer's records aren't what it should cite"
    if moment is not None:
        evidence.append({"at": _number(moment), "within_s": 30})
    if not (says or not_says or evidence):
        return (
            "a right mark needs the records the answer cited, or what a right answer says"
            if right
            else "a wrong mark needs what a right answer says, what it shouldn't say, or the moment it should cite"
        )
    out: dict[str, Any] = {"id": f"ask-{asked:.3f}", "question": question.strip(), "asked_at": asked}
    if day is not None:
        out["day"] = day
    if says:
        out["answer_includes"] = [says]
    if not_says:
        out["answer_excludes"] = [not_says]
    if evidence:
        out["evidence"] = evidence
    out["note"] = f"marked {'right' if right else 'wrong'} in Ask"
    return out


def event(body: dict[str, Any]) -> dict[str, Any] | str:
    """The event label a mark makes (docs/MEASUREMENT.md, events.jsonl), or why it can't be one. Its id names the
    camera and the span, so marking the same span again replaces it."""
    camera, what = _name(body.get("camera")), _name(body.get("what"))
    person = _name(body.get("person"))
    if body.get("person") not in (None, "") and person is None:
        return "person is who it involved, 1 to 80 characters, or left out"
    if camera is None:
        return "camera is the camera the span is on"
    if what is None:
        return "what is what happened, in your own words, 1 to 80 characters"
    start, end = _number(body.get("start")), _number(body.get("end"))
    if start is None or end is None or end <= start:
        return "a span needs its start and a later end"
    count = body.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= MAX_COUNT:
        return f"count is how many times it happened, counted by watching: a whole number from 0 to {MAX_COUNT}"
    out: dict[str, Any] = {
        "id": f"review-{camera}-{start:.3f}-{end:.3f}",
        "camera": camera,
        "what": what,
        "start": start,
        "end": end,
        "count": count,
    }
    if person is not None:
        out["person"] = person
    attributes = body.get("attributes") or {}
    if not isinstance(attributes, dict) or len(attributes) > MAX_ATTRIBUTES:
        return f"attributes are up to {MAX_ATTRIBUTES} measurements, each by its name"
    kept: dict[str, dict[str, Any]] = {}
    for name, a in attributes.items():
        key, value = _name(name), a.get("value") if isinstance(a, dict) else None
        unit = _name(a.get("unit")) if isinstance(a, dict) else None
        if key is None or isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            return f"attribute {name!r} needs a name and a value, a number"
        if unit is None:
            return f"attribute {name!r} needs a unit, like m or s"
        kept[key] = {"value": value, "unit": unit}
    if kept:
        out["attributes"] = kept
    notes = body.get("notes") or []
    if not isinstance(notes, list) or len(notes) > MAX_NOTES:
        return f"notes are up to {MAX_NOTES}, each a note and optionally which time it is about"
    said = []
    for n in notes:
        note = n.get("note") if isinstance(n, dict) else None
        nth = n.get("nth") if isinstance(n, dict) else None
        if not isinstance(note, str) or not note.strip() or len(note.strip()) > MAX_NOTE:
            return f"each note says something, up to {MAX_NOTE} characters"
        if nth is not None and (isinstance(nth, bool) or not isinstance(nth, int) or not 1 <= nth <= count):
            return "a note's nth is which time it is about, from 1 to the count"
        said.append({"note": note.strip()} | ({"nth": nth} if nth is not None else {}))
    if said:
        out["notes"] = said
    return out


def entries(body: dict[str, Any]) -> dict[str, Any] | str:
    """The entry-window label a mark makes (docs/MEASUREMENT.md, entries.jsonl), or why it can't be one. Its id names
    the camera and the window, so labeling the same window again replaces it."""
    camera = _name(body.get("camera"))
    if camera is None:
        return "camera is the camera the window is on"
    start, end = _number(body.get("start")), _number(body.get("end"))
    if start is None or end is None or end <= start:
        return "a window needs its start and a later end"
    raw = body.get("events")
    if not isinstance(raw, list) or len(raw) > MAX_ENTRY_EVENTS:
        return f"events are up to {MAX_ENTRY_EVENTS} entries and exits"
    seen = []
    for e in raw:
        kind, at = (e.get("kind"), _number(e.get("at"))) if isinstance(e, dict) else (None, None)
        if kind not in ("entry", "exit") or at is None:
            return "each event is an entry or an exit, with its time"
        if not start <= at < end:
            return "each entry and exit is inside the window: from its start, and before its end"
        seen.append({"kind": kind, "at": at})
    seen.sort(key=lambda e: (e["at"], e["kind"]))
    raw = body.get("unsure") or []
    if not isinstance(raw, list) or len(raw) > MAX_UNSURE:
        return f"unsure stretches are up to {MAX_UNSURE}"
    unsure = []
    for u in raw:
        a, b = (_number(u.get("start")), _number(u.get("end"))) if isinstance(u, dict) else (None, None)
        reason = u.get("reason") if isinstance(u, dict) else None
        if a is None or b is None or not start <= a < b <= end:
            return "each unsure stretch has a start and a later end, inside the window"
        if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > MAX_NOTE:
            return f"each unsure stretch says why it could not be told, up to {MAX_NOTE} characters"
        unsure.append({"start": a, "end": b, "reason": reason.strip()})
    unsure.sort(key=lambda u: (u["start"], u["end"]))
    if any(u["end"] > v["start"] for u, v in itertools.pairwise(unsure)):
        return "unsure stretches must not overlap"
    for e in seen:
        if any(u["start"] <= e["at"] < u["end"] for u in unsure):
            return "an entry or exit inside an unsure stretch is a guess: leave it out, or shorten the stretch"
    conditions = body.get("conditions") or []
    if not isinstance(conditions, list) or any(c not in CONDITIONS for c in conditions):
        return f"conditions are any of {', '.join(CONDITIONS)}"
    if not seen and not unsure:
        if body.get("none_seen") is not True:
            return "no entries or exits: say none_seen once you have watched the whole window"
        conditions = [*conditions, "no_events"]
    note = body.get("note")
    if note is not None and (not isinstance(note, str) or len(note.strip()) > MAX_NOTE):
        return f"note is up to {MAX_NOTE} characters"
    out: dict[str, Any] = {
        "id": f"review-{camera}-{start:.3f}-{end:.3f}",
        "camera": camera,
        "start": start,
        "end": end,
        "events": seen,
        "unsure": unsure,
        "conditions": sorted(set(conditions)),
    }
    if isinstance(note, str) and note.strip():
        out["note"] = note.strip()
    return out


def _overlaps(new: dict[str, Any]) -> Callable[[dict[str, Any]], bool]:
    """Whether a saved entry window is another window on new's camera that overlaps it."""

    def test(old: dict[str, Any]) -> bool:
        a, b = old.get("start"), old.get("end")
        return (
            old.get("id") != new["id"]
            and old.get("camera") == new["camera"]
            and isinstance(a, int | float)
            and isinstance(b, int | float)
            and a < new["end"]
            and new["start"] < b
        )

    return test


def save(
    new: dict[str, Any],
    file: str = FILE,
    conflict: Callable[[dict[str, Any]], bool] | None = None,
    replace: bool = False,
) -> list[dict[str, Any]]:
    """Write new into the label file, in place of any earlier label with its id. Other lines are kept as they are,
    even ones that don't parse, since the owner may be editing the file by hand. Labels `conflict` matches are
    returned and nothing is written, unless `replace`, which drops them."""
    folder = cfg.path("paths.measurement")
    path = folder / file
    with _lock:
        folder.mkdir(parents=True, exist_ok=True)
        kept, clashes = [], []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    old = json.loads(line)
                    same = old.get("id") == new["id"]
                except ValueError, AttributeError:
                    old, same = None, False
                if isinstance(old, dict) and conflict is not None and conflict(old):
                    clashes.append(old)
                    if replace:
                        continue
                if line.strip() and not same:
                    kept.append(line)
        if clashes and not replace:
            return clashes
        lines = "".join(line + "\n" for line in [*kept, json.dumps(new)])
        json_state.write_text(path, lines, durable=True)  # the owner's hand-made labels
    return clashes


def _bare_yes(body: dict[str, Any]) -> bool:
    """A right mark with nothing to check: no cited records and no text, for a valid question and time."""
    return (
        body.get("right") is True
        and not body.get("cited")
        and not body.get("says")
        and body.get("not_says") is None
        and body.get("moment") is None
        and isinstance(body.get("question"), str)
        and bool(body["question"].strip())
        and _number(body.get("asked_at")) is not None
        and (body.get("day") is None or _day(body.get("day")))
    )


def mark(body: object, control: bool) -> Reply:
    """(code, reply) for POST /measurement/mark."""
    if not control:
        return 403, {"ok": False, "error": "marks can only be made on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    kind = body.get("kind")
    if kind == "answer" and _bare_yes(body):
        # Yes on an answer that cited nothing: a one-click thumbs-up, not a retrieval label (measure.py needs
        # something to check, and nobody wants a form after Yes).
        made = {
            "id": f"ask-{body['asked_at']:.3f}",
            "question": body["question"].strip(),
            "asked_at": body["asked_at"],
            "right": True,
            **({"day": body["day"]} if body.get("day") is not None else {}),
        }
        save(made, FEEDBACK)
        return 200, {"ok": True, "label": made}
    if kind == "entries":
        window = entries(body)
        if isinstance(window, str):
            return 400, {"ok": False, "error": window}
        clashes = save(window, ENTRIES, conflict=_overlaps(window), replace=body.get("replace") is True)
        if clashes and body.get("replace") is not True:
            spans = [{"id": c.get("id"), "start": c.get("start"), "end": c.get("end")} for c in clashes]
            return 409, {"ok": False, "error": "this window overlaps one already labeled", "overlaps": spans}
        return 200, {"ok": True, "label": window, "replaced": len(clashes)}
    made, file = (
        (answer(body), RETRIEVAL)
        if kind == "answer"
        else (event(body), EVENTS)
        if kind == "event"
        else (label(body), FILE)
    )
    if isinstance(made, str):
        return 400, {"ok": False, "error": made}
    save(made, file)
    return 200, {"ok": True, "label": made}


# ---- routes (dashboard/routes.py): what the server answers for this area ----


@routes.post("/measurement/mark")
def _post_measurement_mark(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = mark(body, h._can_teach())
    return h._json(o, code)
