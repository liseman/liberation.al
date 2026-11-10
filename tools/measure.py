#!/usr/bin/env python3
"""Score today's identification and history answers against the hand-made measurement set.

  uv run tools/measure.py                       # default parts, labels from paths.measurement
  uv run tools/measure.py --part retrieval      # one part: identification, retrieval, or events
  uv run tools/measure.py --part identification --sweep  # precision and recall at thresholds from today's up
  uv run tools/measure.py --json                # the full result, for keeping beside a release
  uv run tools/measure.py --dir tests/data/measurement --state-dir /some/state
  uv run tools/measure.py --part retrieval --ask             # score the answers Ask gives, with the local model
  uv run tools/measure.py --part retrieval --rounds 0 --json # the same with the search loop off
  uv run tools/measure.py --part retrieval --vectors off --json  # retrieval with the semantic index off
  uv run tools/measure.py --part retrieval --ask --journal off --json  # Ask with the journal off
  uv run tools/measure.py --compare single.json loop.json    # retrieval in two --json results, side by side
  uv run tools/measure.py --part readings       # how well the model reads questions (calls the local model)

For entry/exit scoring, select a counter and tolerance explicitly. Run as a module
so callbacks in the repository's tools namespace can be imported:
  uv run python -m tools.measure --part entries --entry-counter module:function --entry-tolerance 2
Default runs do not load a counter. A callback controls its own I/O.

To score an event counter against the event labels, select it the same way:
  uv run python -m tools.measure --part events --event-counter some.module:counter --event-margin 30

The label formats, and how to add labels, are in docs/MEASUREMENT.md. This is the measurement set from
docs/DOMAIN-MODEL.md, which steps 3, 7, and 8 need before they change what the user sees.

- Identification compares each labeled clip or frame with the names today's stores gave the same moment: face names
  on see-actions' face matches (`face.match` in the segment store, read through vision/sightings.py) for a camera clip, and resolved speaker labels on transcript
  lines for a voice clip. It reports precision, recall, counts per person, and a confusion list (labeled name -> predicted name).
- Retrieval asks each labeled question through memory/history_qa.collect(), the answering code the dashboard and the
  voice loop use, at the question's own time, and checks the answer text and the evidence it returned. collect()
  has the local model read each question once (memory/question.py), so every run needs it, and each row keeps
  the reading it got under query.reading. Recorded-frame
  analysis is turned off for the run, because that path writes caches and can queue scans; --as-configured keeps it.
  With --ask, each question goes through memory/ask.execute() in local mode instead, so the answer checks read the
  words the user hears, the evidence checks read the evidence Ask answered from (the search rounds' finds included),
  and each question's latency, search rounds, and model are recorded. It also counts the expected evidence that was
  in the answering model's input (evidence_given: the items left after Ask's input budget), which tells a search
  that missed a line from a model that had it and didn't use it. That calls the local Ollama model. --rounds N
  implies --ask and sets recall.max_rounds for the run (0 turns recall.enabled off), for comparing the search loop
  with a single pass. --vectors on or off sets semantic.enabled for the run, through either path, for comparing
  search by words alone with search by words and meaning (the index itself has to be built already: the vector
  worker builds it as entries are written). --journal on or off sets journal.enabled for the run, for comparing
  answers with and without the journal. A question marked "draft" is scored like any other and counted apart, so
  proposed questions can be run before someone confirms them.
- Events (anything that happened on video: someone sat down, opened the fridge, or walked past, or the dog got on
  the couch) are loaded and checked, each with how many times it happened in its span. With an explicit
  --event-counter, each labeled span goes to the counter as a window around it (camera, what happened, and times
  only), and the report gives each span's count error and how well the counter's span lines up with it (overlap, and
  how far its start and end are off), plus the spans it missed and the ones it found where none is labeled.

The built-in scorers read; they do not write to state. While the semantic index is on and built, collect() embeds each
question with the local embedding model (models.ollama.embedding). With --ask, Ask's path (collect, the search rounds, and the answer)
writes nothing either while recorded-frame analysis is off, but its model calls load the model into
Ollama, which keeps it loaded. The numbers describe the labels you have, so a small set gives unstable numbers, and a
score is not a claim about accuracy in general.
"""

import argparse
import collections
import dataclasses
import datetime as dt
import importlib
import itertools
import json
import math
import os
import pathlib
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal, Protocol, TypeGuard, overload
from unittest import mock

from liberational import config as cfg
from liberational.domain import kernel

PARTS = ("identification", "retrieval", "events")
MODALITIES = ("face", "voice")
SOURCES = ("transcript", "camera", "call", "journal")
# Segment streams an expected evidence item can name as its source: Ask's evidence items say which stream
# they came from (`stream`), so a label can expect a walk-by, a zone visit, a model action, or movement.
STREAM_SOURCES = ("walkby.span", "zone.visit", "action.span", "action.window", "movement.track", "movement.coverage")
NOBODY = "nobody"
PLAYBACK_FLAGS = ("playback", "is_playback", "cloned_playback", "synthetic", "is_synthetic")
GENERIC = re.compile(r"(?:speaker\s*\d+|unknown)", re.I)
ASK_SETTINGS = (
    "ask.model",
    "ask.num_ctx",
    "ask.think",
    "recall.enabled",
    "recall.max_rounds",
    "recall.model",
    "journal.enabled",
    "models.ollama.llm",
    "semantic.enabled",
    "models.ollama.embedding",
)  # recorded with every --ask run


class LabelError(ValueError):
    """A label line that does not follow docs/MEASUREMENT.md. The message names the file and line."""


class RunError(ValueError):
    """A run that can't be done as asked: a setting the environment would override, or a result file it can't read."""


# ---- loading ---------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Clip:
    """A labeled clip or frame: who was really there. `people` may be empty (nobody known was there)."""

    id: str
    modality: str
    start: float
    end: float
    people: tuple[str, ...]
    unsure: tuple[str, ...] = ()
    camera: str | None = None
    device: str | None = None
    frame: bool = False
    record: float | None = None  # one voice line only: its ts_start (People's marks)
    sighting: str | None = None  # one face only: its sighting id (its face.match row's id)


@dataclasses.dataclass(frozen=True)
class Expected:
    """One record the answer should cite: near `at`, and containing `includes` when it is set."""

    at: float
    within_s: float
    includes: str | None = None
    source: str | None = None


@dataclasses.dataclass(frozen=True)
class Question:
    id: str
    question: str
    asked_at: float
    day: str | None = None
    answer_includes: tuple[str, ...] = ()
    answer_excludes: tuple[str, ...] = ()
    evidence: tuple[Expected, ...] = ()
    draft: bool = False
    # Each group passes when the answer holds any one of its texts ("three" or "3").
    answer_any: tuple[tuple[str, ...], ...] = ()


@dataclasses.dataclass(frozen=True)
class Event:
    """One span on a camera where something happened, what it was, and how many times, counted by watching. `person`
    is None when no one is named (an event may involve no one)."""

    id: str
    camera: str
    what: str
    start: float
    end: float
    count: int
    person: str | None = None
    attributes: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)  # name -> {"value", "unit"}
    notes: tuple[dict[str, Any], ...] = ()  # {"note", "nth"}; no nth is about the whole span


def instant(value: Any, field: str) -> float:
    """Epoch seconds (as the Review URL's t= shows them), or ISO 8601 with a zone. A time with no zone is refused."""
    if isinstance(value, str):
        try:
            value = dt.datetime.fromisoformat(value)
        except ValueError:
            raise LabelError(f"{field}: not an ISO 8601 time: {value!r}")
    try:
        return kernel.instant(value)
    except (TypeError, ValueError) as e:
        raise LabelError(f"{field}: {e}")


@overload
def _text(d: dict[str, Any], key: str, required: Literal[True] = True) -> str: ...
@overload
def _text(d: dict[str, Any], key: str, required: bool) -> str | None: ...
def _text(d: dict[str, Any], key: str, required: bool = True) -> str | None:
    v = d.get(key)
    if v is None and not required:
        return None
    if not isinstance(v, str) or not v.strip():
        raise LabelError(f"{key} must be non-empty text")
    return v.strip()


def _names(d: dict[str, Any], key: str) -> tuple[str, ...]:
    v = d.get(key, [])
    if not isinstance(v, list) or not all(isinstance(n, str) and n.strip() for n in v):
        raise LabelError(f"{key} must be a list of names")
    return tuple(n.strip() for n in v)


def _texts(d: dict[str, Any], key: str) -> tuple[str, ...]:
    v = d.get(key, [])
    if not isinstance(v, list) or not all(isinstance(n, str) and n.strip() for n in v):
        raise LabelError(f"{key} must be a list of text")
    return tuple(n.strip() for n in v)


def _groups(d: dict[str, Any], key: str) -> tuple[tuple[str, ...], ...]:
    v = d.get(key, [])
    if not isinstance(v, list) or not all(isinstance(g, list) and g for g in v):
        raise LabelError(f"{key} must be a list of lists of text")
    return tuple(_texts({key: g}, key) for g in v)


def _span(d: dict[str, Any], frame_window: float) -> tuple[float, float, bool]:
    """(start, end, frame) from start and end, or from `at` for a single frame (widened by frame_window each way)."""
    if "at" in d:
        if "start" in d or "end" in d:
            raise LabelError("give at, or start and end, not both")
        at = instant(d["at"], "at")
        return max(0.0, at - frame_window), at + frame_window, True
    if "start" not in d or "end" not in d:
        raise LabelError("needs start and end, or at")
    start, end = instant(d["start"], "start"), instant(d["end"], "end")
    if end < start:
        raise LabelError("end is before start")
    return start, end, False


def _clip(d: dict[str, Any], frame_window: float) -> Clip:
    modality = d.get("modality")
    if modality not in MODALITIES:
        raise LabelError(f"modality must be one of {', '.join(MODALITIES)}, not {modality!r}")
    if "people" not in d:
        raise LabelError("people is required (an empty list means nobody known was there)")
    people, unsure = _names(d, "people"), _names(d, "unsure")
    if {p.casefold() for p in people} & {u.casefold() for u in unsure}:
        raise LabelError("a name is in both people and unsure")
    camera = _text(d, "camera", required=modality == "face")
    start, end, frame = _span(d, frame_window)
    return Clip(
        _text(d, "id"),
        modality,
        start,
        end,
        people,
        unsure,
        camera,
        _text(d, "device", False),
        frame,
        *_record(d, modality),
    )


def _record(d: dict[str, Any], modality: str) -> tuple[float | None, str | None]:
    """(line, sighting): the one record a label is about, when it names one: `sighting` (an id) for a face, `line`
    (its ts_start) for a voice. Then only that record is scored, not everyone else in the clip."""
    key, other = ("sighting", "line") if modality == "face" else ("line", "sighting")
    if other in d:
        raise LabelError(f"{other} is for {'voice' if other == 'line' else 'face'} labels")
    if key not in d:
        return None, None
    v = d[key]
    if key == "sighting":
        if not isinstance(v, str) or not v.strip():
            raise LabelError("sighting must be a sighting id, a non-empty string")
        return None, v
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v < 0:
        raise LabelError("line must be a line ts_start, a number at least 0")
    return float(v), None


def _expected(e: Any) -> Expected:
    if not isinstance(e, dict):
        raise LabelError("each evidence item is an object")
    within = e.get("within_s", 30)
    if isinstance(within, bool) or not isinstance(within, int | float) or not math.isfinite(within) or within < 0:
        raise LabelError("within_s must be a number of seconds, at least 0")
    source = e.get("source")
    if source is not None and source not in (*SOURCES, *STREAM_SOURCES):
        raise LabelError(f"evidence source must be one of {', '.join((*SOURCES, *STREAM_SOURCES))}, not {source!r}")
    return Expected(instant(e.get("at"), "evidence at"), float(within), _text(e, "includes", False), source)


def _question(d: dict[str, Any], _frame_window: float) -> Question:
    day = d.get("day")
    if day is not None:
        try:
            dt.date.fromisoformat(day)
        except TypeError, ValueError:
            raise LabelError(f"day must be YYYY-MM-DD, not {day!r}")
    evidence = d.get("evidence", [])
    if not isinstance(evidence, list):
        raise LabelError("evidence must be a list")
    draft = d.get("draft", False)
    if not isinstance(draft, bool):
        raise LabelError("draft must be true or false")
    q = Question(
        _text(d, "id"),
        _text(d, "question"),
        instant(d.get("asked_at"), "asked_at"),
        day,
        _texts(d, "answer_includes"),
        _texts(d, "answer_excludes"),
        tuple(_expected(e) for e in evidence),
        draft,
        _groups(d, "answer_any"),
    )
    if not (q.answer_includes or q.answer_any or q.answer_excludes or q.evidence):
        raise LabelError("nothing to check: give answer_includes, answer_any, answer_excludes, or evidence")
    return q


def _number(v: Any) -> bool:
    return not isinstance(v, bool) and isinstance(v, int | float) and math.isfinite(v)


def _whole(v: Any) -> TypeGuard[int]:
    return not isinstance(v, bool) and isinstance(v, int)


OLD_EVENT_FIELDS = {"activity": "what", "reps": "count"}  # refused, with the new name; a note's rep is now nth
OLD_EVENT_FILES = ("squats.jsonl", "activities.jsonl")  # a run stops while one is there, so no label is skipped


def _event(d: dict[str, Any], _frame_window: float) -> Event:
    for old, new in OLD_EVENT_FIELDS.items():
        if old in d:
            raise LabelError(f"{old} is now {new} (docs/MEASUREMENT.md, Moving old activity labels)")
    count = d.get("count")
    if not _whole(count) or count < 0:
        raise LabelError("count must be a whole number, at least 0")
    attributes = d.get("attributes", {})
    if not isinstance(attributes, dict):
        raise LabelError("attributes must be an object: a name for each measurement")
    for name, a in attributes.items():
        if not name.strip() or not isinstance(a, dict) or not _number(a.get("value")):
            raise LabelError(f"attribute {name!r} needs a value, a number")
        _text(a, "unit")
    notes = d.get("notes", [])
    if not isinstance(notes, list):
        raise LabelError("notes must be a list")
    for n in notes:
        if not isinstance(n, dict):
            raise LabelError("each note is an object with a note and an optional nth")
        if "rep" in n:
            raise LabelError("a note's rep is now nth (docs/MEASUREMENT.md, Moving old activity labels)")
        nth = n.get("nth")
        if nth is not None and not (_whole(nth) and 1 <= nth <= count):
            raise LabelError(f"a note's nth is which time it is about, a whole number from 1 to the count ({count})")
        _text(n, "note")
    start, end, _ = _span(d, 0.0)
    return Event(
        _text(d, "id"),
        _text(d, "camera"),
        _text(d, "what"),
        start,
        end,
        count,
        _text(d, "person") if "person" in d else None,
        {k.strip(): dict(v) for k, v in attributes.items()},
        tuple(dict(n) for n in notes),
    )


@dataclasses.dataclass(frozen=True)
class ReadLabel:
    """A question and the reading a person agreed is right (memory/question.py's Reading), or `same_as`: the id of
    the label it is a reworded copy of, whose reading it shares."""

    id: str
    question: str
    reading: Any  # question.Reading; None for a reworded copy, which takes its original's
    same_as: str | None = None


def _read_label(d: dict[str, Any], _frame_window: float) -> ReadLabel:
    from liberational.memory import question

    same_as = _text(d, "same_as", required=False)
    fields = d.get("reading")
    if (fields is None) == (same_as is None):
        raise LabelError("give reading (the fields that differ from the defaults) or same_as, not both")
    reading = None
    if fields is not None:
        if not isinstance(fields, dict):
            raise LabelError("reading must be an object")
        reading = question.handed(fields)
        choices = dict(
            source=question.SOURCES,
            measure=question.MEASURES,
            movement=question.MOVEMENTS,
            part=question.PARTS,
            span_unit=question.UNITS,
            span_kind=question.SPAN_KINDS,
        )
        if reading is None:
            raise LabelError(f"reading has a field the form doesn't: {sorted(fields)}")
        for name, allowed in choices.items():
            if getattr(reading, name) not in allowed:
                raise LabelError(f"reading.{name} must be one of {list(allowed)}")
    return ReadLabel(_text(d, "id"), _text(d, "question"), reading, same_as)


PARSERS = {"identification": _clip, "retrieval": _question, "events": _event, "readings": _read_label}


def load(directory: str | pathlib.Path, part: str, frame_window: float = 1.0) -> list[Any]:
    """The labels of one part from <directory>/<part>.jsonl, or [] when the file is not there. Raises LabelError."""
    path = pathlib.Path(directory) / f"{part}.jsonl"
    if not path.exists():
        return []
    out: list[Any] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            d = json.loads(line)
            if not isinstance(d, dict):
                raise LabelError("a label is a JSON object")
            label = PARSERS[part](d, frame_window)
            if part == "entries" and isinstance(label, EntryWindow):
                _entry_windows([*out, label])
        except (LabelError, json.JSONDecodeError) as e:
            raise LabelError(f"{path}:{n}: {e}") from None
        if label.id in seen:
            raise LabelError(f"{path}:{n}: id {label.id!r} is used twice")
        seen.add(label.id)
        out.append(label)
    return out


# ---- entry/exit measurements ------------------------------------------------------------------------------------


EntryKind = Literal["entry", "exit"]
ENTRY_KINDS: tuple[EntryKind, ...] = ("entry", "exit")


@dataclasses.dataclass(frozen=True)
class EntryEvent:
    kind: EntryKind
    at: float


@dataclasses.dataclass(frozen=True)
class EntrySpan:
    start: float
    end: float
    reason: str = ""


@dataclasses.dataclass(frozen=True)
class EntryWindow:
    id: str
    camera: str
    start: float
    end: float
    events: tuple[EntryEvent, ...]
    unsure: tuple[EntrySpan, ...] = ()
    conditions: tuple[str, ...] = ()


class EntryCounter(Protocol):
    """A counter receives the observation window only, never labels or uncertainty."""

    def __call__(self, *, camera: str, start: float, end: float, state: pathlib.Path) -> Mapping[str, Any]: ...


def _entry_events(raw: Any, start: float, end: float) -> tuple[EntryEvent, ...]:
    if not isinstance(raw, list):
        raise LabelError("events must be a list")
    out = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"kind", "at"} or item["kind"] not in ENTRY_KINDS:
            raise LabelError("each event needs kind entry or exit and at")
        at = instant(item["at"], "event.at")
        if not start <= at < end:
            raise LabelError("event.at must be inside [start, end)")
        out.append(EntryEvent(item["kind"], at))
    return tuple(out)


def _entry_spans(raw: Any, start: float, end: float) -> tuple[EntrySpan, ...]:
    if not isinstance(raw, list):
        raise LabelError("uncertainty spans must be a list")
    out = []
    for item in raw:
        if (
            not isinstance(item, dict)
            or not {"start", "end"} <= item.keys()
            or item.keys() - {"start", "end", "reason"}
        ):
            raise LabelError("each uncertainty span needs start/end and optional reason")
        a, b = instant(item["start"], "span.start"), instant(item["end"], "span.end")
        reason = item.get("reason", "")
        if not isinstance(reason, str) or not start <= a < b <= end:
            raise LabelError("uncertainty span must have a string reason and lie inside the window")
        out.append(EntrySpan(a, b, reason))
    out.sort(key=lambda span: (span.start, span.end))
    if any(a.end > b.start for a, b in itertools.pairwise(out)):
        raise LabelError("uncertainty spans must not overlap")
    return tuple(out)


def _entries(d: dict[str, Any], _frame_window: float) -> EntryWindow:
    if d.keys() - {"id", "camera", "start", "end", "events", "unsure", "conditions", "note"}:
        raise LabelError("unknown entry-window field")
    start, end = instant(d.get("start"), "start"), instant(d.get("end"), "end")
    if start >= end:
        raise LabelError("entry window needs start < end")
    conditions = d.get("conditions", [])
    if not isinstance(conditions, list) or any(not isinstance(tag, str) or not tag.strip() for tag in conditions):
        raise LabelError("conditions must be a list of nonempty strings")
    if "note" in d and not isinstance(d["note"], str):
        raise LabelError("note must be a string")
    return EntryWindow(
        _text(d, "id"),
        _text(d, "camera"),
        start,
        end,
        _entry_events(d.get("events"), start, end),
        _entry_spans(d.get("unsure", []), start, end),
        tuple(conditions),
    )


def _entry_windows(windows: Sequence[EntryWindow]) -> None:
    seen: set[str] = set()
    last: dict[str, float] = {}
    for window in sorted(windows, key=lambda w: (w.camera, w.start, w.end)):
        if window.id in seen:
            raise LabelError(f"entry window id {window.id!r} is used twice")
        if window.start < last.get(window.camera, -math.inf):
            raise LabelError(f"entry windows overlap for camera {window.camera!r}")
        seen.add(window.id)
        last[window.camera] = window.end


def _entry_tolerance(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise RunError("entry tolerance must be finite and nonnegative")


def match_events(expected: Sequence[float], predicted: Sequence[float], tolerance: float) -> list[tuple[int, int]]:
    """Maximum-cardinality, then minimum-error ordered matching; return original indices.

    Absolute distance on a line admits an optimal noncrossing assignment. Dynamic
    programming avoids greedy nearest-neighbor choices that can lose a match.
    Equal optima prefer matching the current pair, then skipping a prediction.
    """
    _entry_tolerance(tolerance)
    if any(
        isinstance(t, bool) or not isinstance(t, int | float) or not math.isfinite(t) for t in (*expected, *predicted)
    ):
        raise RunError("matching timestamps must be finite numbers")
    left, right = (
        sorted(enumerate(expected), key=lambda x: (x[1], x[0])),
        sorted(enumerate(predicted), key=lambda x: (x[1], x[0])),
    )
    n, m = len(left), len(right)
    scores = [[(0, 0.0) for _ in range(m + 1)] for _ in range(n + 1)]
    moves = [["" for _ in range(m)] for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best, move = scores[i][j + 1], "prediction"
            if scores[i + 1][j] > best:
                best, move = scores[i + 1][j], "label"
            distance = abs(left[i][1] - right[j][1])
            if distance <= tolerance:
                count, error = scores[i + 1][j + 1]
                candidate = (count + 1, error - distance)
                if candidate >= best:
                    best, move = candidate, "match"
            scores[i][j], moves[i][j] = best, move
    pairs = []
    i = j = 0
    while i < n and j < m:
        move = moves[i][j]
        if move == "match":
            pairs.append((left[i][0], right[j][0]))
            i += 1
            j += 1
        elif move == "label":
            i += 1
        else:
            j += 1
    return pairs


def _entry_region(at: float, unsure: Sequence[EntrySpan]) -> int | None:
    for index, span in enumerate(unsure):
        if at < span.start:
            return index
        if at < span.end:
            return None
    return len(unsure)


def _entry_metrics(tp: int, fp: int, fn: int, supported: bool = True) -> dict[str, Any]:
    return dict(
        tp=tp if supported else None,
        fp=fp if supported else None,
        fn=fn if supported else None,
        precision=tp / (tp + fp) if supported and tp + fp else None,
        recall=tp / (tp + fn) if supported and tp + fn else None,
        supported=supported,
    )


def score_entries(
    windows: Sequence[EntryWindow],
    state: str | pathlib.Path,
    counter: EntryCounter,
    tolerance: float,
) -> dict[str, Any]:
    """Score a blinded counter; its unknown coverage never erases labeled misses."""
    _entry_tolerance(tolerance)
    _entry_windows(windows)
    per_window: list[dict[str, Any]] = []
    per_camera: dict[str, Any] = {}
    for window in windows:
        try:
            raw = counter(camera=window.camera, start=window.start, end=window.end, state=pathlib.Path(state))
            if not isinstance(raw, Mapping) or raw.keys() - {"events", "supported_kinds", "unknown", "diagnostics"}:
                raise LabelError("counter result must be a mapping with events, supported_kinds, unknown, diagnostics")
            supported = raw.get("supported_kinds")
            if (
                not isinstance(supported, list)
                or any(kind not in ENTRY_KINDS for kind in supported)
                or len(set(supported)) != len(supported)
            ):
                raise LabelError("supported_kinds must list unique entry/exit kinds")
            predictions = _entry_events(raw.get("events"), window.start, window.end)
            if any(event.kind not in supported for event in predictions):
                raise LabelError("counter returned events of an unsupported kind")
            unknown = _entry_spans(raw.get("unknown", []), window.start, window.end)
            diagnostics = raw.get("diagnostics", {})
            if not isinstance(diagnostics, Mapping):
                raise LabelError("counter diagnostics must be a mapping")
            diagnostics = json.loads(json.dumps(dict(diagnostics), allow_nan=False))
        except Exception as exc:
            raise RunError(f"entry counter failed for {window.id}: {exc}") from exc
        kinds: dict[str, Any] = {}
        excluded_labels = sum(_entry_region(event.at, window.unsure) is None for event in window.events)
        excluded_predictions = sum(_entry_region(event.at, window.unsure) is None for event in predictions)
        for kind in ENTRY_KINDS:
            labels = [
                event.at
                for event in window.events
                if event.kind == kind and _entry_region(event.at, window.unsure) is not None
            ]
            guesses = [
                event.at
                for event in predictions
                if event.kind == kind and _entry_region(event.at, window.unsure) is not None
            ]
            matches: list[dict[str, float]] = []
            if kind in supported:
                for region in range(len(window.unsure) + 1):
                    left = [at for at in labels if _entry_region(at, window.unsure) == region]
                    right = [at for at in guesses if _entry_region(at, window.unsure) == region]
                    matches.extend(
                        dict(expected=left[i], predicted=right[j], delta_s=right[j] - left[i])
                        for i, j in match_events(left, right, tolerance)
                    )
            kinds[kind] = dict(
                **_entry_metrics(
                    len(matches), len(guesses) - len(matches), len(labels) - len(matches), kind in supported
                ),
                matches=matches,
                labeled=len(labels),
                predicted=len(guesses),
            )
        row = dict(
            id=window.id,
            camera=window.camera,
            start=window.start,
            end=window.end,
            conditions=list(window.conditions),
            by_kind=kinds,
            excluded_labels=excluded_labels,
            excluded_predictions=excluded_predictions,
            excluded_seconds=sum(span.end - span.start for span in window.unsure),
            unknown=[dataclasses.asdict(span) for span in unknown],
            unknown_seconds=sum(span.end - span.start for span in unknown),
            diagnostics=diagnostics,
        )
        per_window.append(row)
        camera = per_camera.setdefault(
            window.camera,
            dict(
                windows=0,
                by_kind={},
                excluded_labels=0,
                excluded_predictions=0,
                excluded_seconds=0.0,
                unknown_seconds=0.0,
            ),
        )
        camera["windows"] += 1
        for field in ("excluded_labels", "excluded_predictions", "excluded_seconds", "unknown_seconds"):
            camera[field] += row[field]
        for kind, metrics in kinds.items():
            total = camera["by_kind"].setdefault(
                kind, dict(tp=0, fp=0, fn=0, supported_windows=0, unsupported_windows=0)
            )
            total["supported_windows" if metrics["supported"] else "unsupported_windows"] += 1
            if metrics["supported"]:
                for field in ("tp", "fp", "fn"):
                    total[field] += metrics[field]
    for camera in per_camera.values():
        for total in camera["by_kind"].values():
            total.update(
                precision=total["tp"] / (total["tp"] + total["fp"])
                if not total["unsupported_windows"] and total["tp"] + total["fp"]
                else None,
                recall=total["tp"] / (total["tp"] + total["fn"])
                if not total["unsupported_windows"] and total["tp"] + total["fn"]
                else None,
                supported=not total["unsupported_windows"],
            )
    return dict(windows=len(windows), tolerance_s=tolerance, per_camera=per_camera, per_window=per_window)


def _load_callback(spec: str, what: str) -> Any:
    """Import an explicitly selected local callback. Importing arbitrary code is opt-in."""
    if spec.count(":") != 1:
        raise RunError(f"{what} must be module:function")
    module, name = spec.split(":")
    if not module or not name or not all(part.isidentifier() for part in module.split(".")) or not name.isidentifier():
        raise RunError(f"{what} must be module:function")
    try:
        callback = getattr(importlib.import_module(module), name)
    except Exception as exc:
        raise RunError(f"cannot load {what} {spec}: {exc}") from exc
    if not callable(callback):
        raise RunError(f"{what} {spec} is not callable")
    return callback


def load_entry_counter(spec: str) -> EntryCounter:
    return _load_callback(spec, "entry counter")


PARSERS["entries"] = _entries


# ---- identification ------------------------------------------------------------------------------------------------


Named = tuple[str, float | None]  # a name given, and the similarity it was given at (None: a person chose it)


SIGHTING_SLACK_S = 60.0  # a one-sighting label's id is looked for this far around its clip


def _face_names(state: pathlib.Path, clip: Clip) -> tuple[list[Named], int]:
    """(names on the camera's sightings in the clip, with each one's similarity; faces found in it), read-only. A
    name is read as the person's name now (the registry's `known_as` at the sighting's time), as People and the
    timeline show it, so a label written after a rename measures recognition, not the rename."""
    from liberational.people import transcript_view
    from liberational.vision import sightings

    if clip.sighting is not None:  # one sighting: only it is scored, not the others in the frame
        near = sightings.faces(clip.start - SIGHTING_SLACK_S, clip.end + SIGHTING_SLACK_S, clip.camera, state)
        found = [f for f in near if f["id"] == clip.sighting]
    else:
        found = sightings.faces(clip.start, clip.end + 0.001, clip.camera, state)
    view = transcript_view.View(state)

    def now(name: str, ts: float) -> str:
        person = view.person_at(name, ts)
        return person.display_name if person is not None else name

    return [(now(f["name"], f["ts"]), _sim(f["sim"])) for f in found if f["name"]], len(found)


def _sim(v: object) -> float | None:
    """A stored similarity as a number, or None when there is none (a name a person gave)."""
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) else None


def _voice_lines(lo: float, hi: float) -> Iterator[tuple[float, float, dict[str, Any]]]:
    """The speech log's utterances overlapping [lo, hi], as (start, end, line)."""
    from liberational import speech_log

    for e in speech_log.shared().range(lo, hi + 1e-3):
        start = float(e["ts_start"])
        yield start, float(e.get("ts_end") or start), e


def _voice_names(state: pathlib.Path, clips: Sequence[Clip]) -> dict[str, tuple[list[Named], int]]:
    """{clip id: (resolved speaker names of lines overlapping the clip, with each line's speaker_sim; lines found)},
    read-only."""
    from liberational.memory.history_qa import bot_labels
    from liberational.people import transcript_view
    from liberational.speech.playback_guard import overlaps

    view, bots = transcript_view.View(state), bot_labels()  # names as history answers read them (step 4)
    windows = state / "playback_windows.json"
    voice = [c for c in clips if c.modality == "voice"]
    found: dict[str, tuple[list[Named], int]] = {c.id: ([], 0) for c in voice}
    spans = [(c.start, c.end) for c in voice]
    lines = _voice_lines(min(a for a, _ in spans), max(b for _, b in spans)) if spans else iter(())
    for start, end, e in lines:
        for c in voice:
            if start > c.end or end < c.start or (c.device and e.get("device") != c.device):
                continue
            if c.record is not None and abs(start - c.record) > 1e-3:  # one line: only it is scored
                continue
            names, n = found[c.id]
            # The person's name now, as People shows the line (`named`): a label written after a rename names them so.
            name = view.named([e])[0].get("speaker") or transcript_view.UNKNOWN
            # The pack's own playback is left out as history answers leave it out: still a line, never a name.
            playback = any(e.get(k) for k in PLAYBACK_FLAGS) or overlaps(start, end, windows)
            if not playback and name.casefold() not in bots and not GENERIC.fullmatch(name):
                names.append((name, _sim(e.get("speaker_sim"))))
            found[c.id] = (names, n + 1)
    return found


def _found(clips: Sequence[Clip], state: pathlib.Path) -> dict[str, tuple[list[Named], int]]:
    """{clip id: (names given in it, with their similarities; records found)}, read from state once."""
    voices = _voice_names(state, clips)
    return {c.id: _face_names(state, c) if c.modality == "face" else voices[c.id] for c in clips}


def score_identification(clips: Sequence[Clip], state: str | pathlib.Path) -> dict[str, Any]:
    """Precision, recall, counts per person, and confusions of today's names against the labeled clips.

    Per clip, a predicted name the label lists is a true positive, a predicted name it does not list is a false
    positive (unless it is in `unsure`), and a labeled name nobody predicted is a false negative. A confusion pairs
    each missed name with each extra name in the same clip, or with "nobody" when one side is empty.
    """
    return _tally(clips, _found(clips, pathlib.Path(state)), lambda _modality, _sim: True)


def _tally(
    clips: Sequence[Clip],
    found_in: Mapping[str, tuple[list[Named], int]],
    keep: Callable[[str, float | None], bool],
) -> dict[str, Any]:
    """score_identification's counts, keeping only the names keep(modality, similarity) allows."""
    tp = fp = fn = 0
    per_person: collections.defaultdict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    confusions: collections.Counter[tuple[str, str]] = collections.Counter()
    rows: list[dict[str, Any]] = []
    silent: list[str] = []
    for c in clips:
        given, found = found_in[c.id]
        names = [n for n, sim in given if keep(c.modality, sim)]
        display: dict[str, str] = {}
        for n in [*c.people, *names]:
            display.setdefault(n.casefold(), n)
        labeled = {p.casefold() for p in c.people}
        predicted = {n.casefold() for n in names} - {u.casefold() for u in c.unsure}
        hits, extra, missed = labeled & predicted, predicted - labeled, labeled - predicted
        for k, group in (("tp", hits), ("fp", extra), ("fn", missed)):
            for n in group:
                per_person[display[n]][k] += 1
        tp, fp, fn = tp + len(hits), fp + len(extra), fn + len(missed)
        for m in sorted(missed) or [NOBODY]:
            for x in sorted(extra) or [NOBODY]:
                if (m, x) != (NOBODY, NOBODY):
                    confusions[(display.get(m, m), display.get(x, x))] += 1
        if not found:
            silent.append(c.id)
        rows.append(
            {
                "id": c.id,
                "modality": c.modality,
                "labeled": sorted(display[n] for n in labeled),
                "predicted": sorted(display[n] for n in predicted),
                "missed": sorted(display[n] for n in missed),
                "extra": sorted(display[n] for n in extra),
                "records": found,
            }
        )
    return {
        "clips": len(clips),
        "by_modality": dict(collections.Counter(c.modality for c in clips)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "per_person": dict(sorted(per_person.items(), key=lambda kv: kv[0].casefold())),
        "confusions": [
            {"labeled": m, "predicted": x, "count": n}
            for (m, x), n in sorted(confusions.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "no_records": silent,
        "per_clip": rows,
    }


# Today's thresholds: voice names a line at speaker_sim >= 0.55 (audio/ambient.py THRESHOLD), a face at
# sim > 0.45 (vision/faces.py THRESH, vision/sightings.py FACE_THRESH). Copied here so this reads no model code.
TODAY = {"voice": 0.55, "face": 0.45}
SWEEP_TOP, SWEEP_STEP = 0.90, 0.05


def _named_at(modality: str, sim: float | None, threshold: float) -> bool:
    """Whether a name given at sim would still be given at threshold, compared as the live system compares it. A
    name with no similarity was a person's choice (a hand label): every threshold keeps it."""
    if sim is None:
        return True
    return sim >= threshold if modality == "voice" else sim > threshold


def sweep_identification(
    clips: Sequence[Clip], state: str | pathlib.Path, top: float = SWEEP_TOP, step: float = SWEEP_STEP
) -> dict[str, Any]:
    """Precision and recall per modality at thresholds from today's up to top, from the similarities stored on each
    transcript line and sighting. Read-only, and it changes no default.

    Only today's threshold and stricter ones can be swept: a record today left unnamed keeps no candidate name, so
    what a lower threshold would have named isn't stored."""
    found_in = _found(clips, pathlib.Path(state))
    out: dict[str, Any] = {}
    for modality, today in TODAY.items():
        mine = [c for c in clips if c.modality == modality]
        rows = []
        for i in range(int(round((top - today) / step)) + 1):
            t = round(today + i * step, 4)
            r = _tally(mine, found_in, lambda m, sim, t=t: _named_at(m, sim, t))
            rows.append(
                {
                    "threshold": t,
                    "today": i == 0,
                    "tp": r["tp"],
                    "fp": r["fp"],
                    "fn": r["fn"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                }
            )
        out[modality] = {"clips": len(mine), "today": today, "rows": rows}
    return out


# ---- retrieval -----------------------------------------------------------------------------------------------------


def _number(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _cites(item: dict[str, Any], e: Expected) -> bool:
    """Whether an evidence item is the record e expects: near its time, holding its text, from its source. An item
    with a numeric start and end (a journal entry covers a stretch of time) is near when e.at falls in that span,
    widened by within_s each way; any other item is near when its ts is within within_s of e.at."""
    try:
        if _number(item.get("start")) and _number(item.get("end")):
            near = item["start"] - e.within_s <= e.at <= item["end"] + e.within_s
        else:
            near = abs(float(item.get("ts")) - e.at) <= e.within_s  # ty: ignore[invalid-argument-type] no ts: TypeError, caught below
    except TypeError, ValueError:
        return False
    text = str(item.get("quote") or item.get("text") or "")
    source = item.get("stream") if e.source in STREAM_SOURCES else item.get("source")
    if e.source == "journal" and item.get("kind") == "journal":
        source = "journal"
    return (
        near
        and (e.includes is None or e.includes.casefold() in text.casefold())
        and (e.source is None or source == e.source)
    )


def _given(contexts: Sequence[str]) -> set[str] | None:
    """The ids of the evidence items the answering model received: the "evidence" rows of the first context
    ask.local_answer was given (ask.evidence_context's JSON), or None when no model was asked."""
    if not contexts:
        return None
    try:
        rows = json.loads(contexts[0]).get("evidence") or []
    except ValueError, AttributeError:
        return set()
    return {str(row.get("id")) for row in rows if isinstance(row, dict)}


def _ask(q: Question, state: str | pathlib.Path, row: dict[str, Any]) -> tuple[dict[str, Any], set[str] | None]:
    """ask.execute() for one question in local mode, timed, with its search rounds and model added to row; and the
    ids of the evidence the answering model was given (_given), None when the answer needed no model."""
    from liberational.memory import ask

    contexts: list[str] = []
    answer = ask.local_answer
    trace: dict[str, Any] = {}

    def recorded(question: str, context: str | None, *args: Any, **kwargs: Any) -> tuple[str, str]:
        if context is not None:
            contexts.append(context)
        return answer(question, context, *args, **kwargs, trace=trace)

    began = time.perf_counter()
    try:
        with mock.patch.object(ask, "local_answer", recorded):
            r = ask.execute(q.question, mode="local", state_dir=state, now=q.asked_at, day=q.day)
    finally:
        row["latency_s"] = round(time.perf_counter() - began, 3)
    rounds = r.get("rounds")
    row.update(status=r.get("status"), model=r.get("model"), rounds=len(rounds) if isinstance(rounds, list) else 0)
    # The answer call's own timings and the model's reasoning (ask.think), when a model answered
    row.update({k: trace[k] for k in ("first_token_s", "model_s") if k in trace})
    if trace.get("thinking"):
        row["thinking"] = trace["thinking"]
    if r.get("status") not in ("ok", "partial"):
        # An error answer is still text ("The local model could not complete the answer."), which an exclusion
        # would pass, so a failed answer fails the answer check.
        why = f" ({r['error']})" if r.get("error") else ""
        row["failures"].append(f"execute() gave status {r.get('status')}{why}")
    # Ask's worker is stopped at its deadline, so an answer that took longer never reaches the user.
    if row["latency_s"] > (limit := ask.deadline("local")):
        row["failures"].append(
            f"execute() gave its answer after {row['latency_s']:.0f} s, past Ask's {limit} s deadline"
        )
    return r, _given(contexts)


def score_retrieval(
    questions: Sequence[Question],
    state: str | pathlib.Path,
    as_configured: bool = False,
    use_ask: bool = False,
    rounds: int | None = None,
    vectors: bool | None = None,
    journal: bool | None = None,
) -> dict[str, Any]:
    """Ask each question at its own time, and check its answer and evidence.

    By default each question goes through history_qa.collect(). With use_ask, or with rounds (which sets
    recall.max_rounds for the run), it goes through ask.execute() in local mode instead. vectors True or False sets
    semantic.enabled for the run, and journal True or False sets journal.enabled; the result then says which
    ("vectors", "journal"). Draft questions are scored like the rest and counted apart ("drafts")."""
    from liberational.memory import history_qa

    use_ask = use_ask or rounds is not None
    fn = "execute()" if use_ask else "collect()"
    # Recorded-frame analysis writes a frame cache, so a scoring run turns it off. The environment beats
    # config.override, so the PACK_ variable is set too.
    off = {} if as_configured else {"review": {"recorded_frame_analysis": False}}
    env = {} if as_configured else {"PACK_REVIEW_RECORDED_FRAME_ANALYSIS": "false"}
    if rounds is not None:
        off = cfg.merge(off, {"recall": {"enabled": rounds > 0, "max_rounds": rounds}})
    if vectors is not None:
        off = cfg.merge(off, {"semantic": {"enabled": bool(vectors)}})
    if journal is not None:
        off = cfg.merge(off, {"journal": {"enabled": bool(journal)}})
    rows: list[dict[str, Any]] = []
    with cfg.override(off), mock.patch.dict(os.environ, env):
        if rounds is not None and (cfg.get("recall.max_rounds"), bool(cfg.get("recall.enabled"))) != (
            rounds,
            rounds > 0,
        ):
            # Setting the variables here, as for frames, would hide a choice someone made on purpose, so stop instead.
            raise RunError(
                "PACK_RECALL_MAX_ROUNDS or PACK_RECALL_ENABLED is set in the environment and wins over --rounds; "
                "unset it first"
            )
        if vectors is not None and bool(cfg.get("semantic.enabled")) != bool(vectors):
            raise RunError("PACK_SEMANTIC_ENABLED is set in the environment and wins over --vectors; unset it first")
        if journal is not None and bool(cfg.get("journal.enabled")) != bool(journal):
            raise RunError("PACK_JOURNAL_ENABLED is set in the environment and wins over --journal; unset it first")
        frames = bool(cfg.get("review.recorded_frame_analysis"))
        settings: dict[str, Any] = {k: cfg.get(k, None) for k in ASK_SETTINGS}
        for q in questions:
            row: dict[str, Any] = {"id": q.id, "question": q.question, "failures": []}
            if q.draft:
                row["draft"] = True
            r: dict[str, Any]
            given: set[str] | None = None
            try:
                if use_ask:
                    r, given = _ask(q, state, row)
                else:
                    r = history_qa.collect(q.question, state, now=q.asked_at, day=q.day)
            except Exception as e:  # one broken question must not hide the rest of the report
                r = {"answer": None, "evidence": []}
                row["failures"].append(f"{fn} raised {type(e).__name__}: {e}")
            answer = r.get("answer") or ""
            evidence = r.get("evidence") or []
            row["is_history"] = r.get("is_history", True)
            row["query"] = r.get("query") or {}  # its "reading" is how the model read the question
            if not row["is_history"]:
                row["failures"].append(f"{fn} did not treat this as a history question")
            row["answer"], row["evidence_returned"] = answer, len(evidence)
            row["answer_checked"] = bool(q.answer_includes or q.answer_any or q.answer_excludes)
            if row["answer_checked"] and not answer:
                # Transcript and caption answers are written later from the evidence, so collect() has no text. With
                # no text, an exclusion would pass without checking anything, so a missing answer fails the check.
                row["failures"].append(
                    "collect() gave evidence but no answer text; check evidence instead"
                    if evidence and not use_ask
                    else f"{fn} gave no answer text"
                )
            for text in q.answer_includes:
                if text.casefold() not in answer.casefold():
                    row["failures"].append(f"answer lacks {text!r}")
            for group in q.answer_any:
                if not any(text.casefold() in answer.casefold() for text in group):
                    row["failures"].append(f"answer lacks any of {list(group)!r}")
            for text in q.answer_excludes:
                if text.casefold() in answer.casefold():
                    row["failures"].append(f"answer has {text!r}")
            row["answer_ok"] = not any(f.startswith(("answer", f"{fn} gave")) for f in row["failures"])
            row["evidence_expected"] = len(q.evidence)
            row["evidence_found"] = 0
            # What the answering model read: the evidence items in its input, after the input budget. An answer that
            # needed no model (a recorded answer) was given all of it.
            model_input = [i for i in evidence if given is None or str(i.get("id")) in given]
            if use_ask:
                row["evidence_given"] = 0
                row["model_input"] = None if given is None else len(model_input)
                row["evidence_ids"] = [str(i.get("id")) for i in evidence]
                row["given_ids"] = [str(i.get("id")) for i in model_input]
            for e in q.evidence:
                if use_ask and any(_cites(item, e) for item in model_input):
                    row["evidence_given"] += 1
                if any(_cites(item, e) for item in evidence):
                    row["evidence_found"] += 1
                else:
                    what = f" with {e.includes!r}" if e.includes else ""
                    row["failures"].append(f"no evidence within {e.within_s:g}s of {history_qa.stamp(e.at)}{what}")
            row["passed"] = not row["failures"]
            rows.append(row)
    checked = [r for r in rows if r["answer_checked"]]
    out: dict[str, Any] = {
        "questions": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "answers_checked": len(checked),
        "answers_ok": sum(r["answer_ok"] for r in checked),
        "evidence_expected": sum(r["evidence_expected"] for r in rows),
        "evidence_found": sum(r["evidence_found"] for r in rows),
        "recorded_frame_analysis": frames,
        "timezone": str(history_qa.TZ),
        "per_question": rows,
    }
    if vectors is not None:
        out["vectors"] = bool(vectors)
    if journal is not None:
        out["journal"] = bool(journal)
    if drafts := [r for r in rows if r.get("draft")]:
        out["drafts"] = len(drafts)
        out["drafts_passed"] = sum(r["passed"] for r in drafts)
    if use_ask:  # a collect-only run keeps exactly the fields it always had
        latencies = [r["latency_s"] for r in rows]
        out.update(
            through="ask.execute",
            settings=settings,
            latency_median_s=round(statistics.median(latencies), 3) if latencies else None,
            latency_max_s=max(latencies) if latencies else None,
            rounds=sum(r.get("rounds", 0) for r in rows),
            evidence_given=sum(r["evidence_given"] for r in rows),
        )
    return out


# ---- events ----------------------------------------------------------------------------------------------------


class EventCounter(Protocol):
    """A counter gets a window and what happened, never the person, the labeled span, or its count. It returns
    {"spans": [{"start", "end", "count"}, ...], "diagnostics": {...}}: the stretches where it saw the event, and how
    many times in each."""

    def __call__(
        self, *, camera: str, what: str, start: float, end: float, state: pathlib.Path
    ) -> Mapping[str, Any]: ...


@dataclasses.dataclass(frozen=True)
class CountedSpan:
    start: float
    end: float
    count: int


def load_event_counter(spec: str) -> EventCounter:
    return _load_callback(spec, "event counter")


def _counted_spans(raw: Any, start: float, end: float) -> tuple[CountedSpan, ...]:
    if not isinstance(raw, Mapping) or raw.keys() - {"spans", "diagnostics"}:
        raise LabelError("counter result must be a mapping with spans and optional diagnostics")
    spans = raw.get("spans")
    if not isinstance(spans, list):
        raise LabelError("spans must be a list")
    out = []
    for item in spans:
        if not isinstance(item, Mapping) or set(item) != {"start", "end", "count"}:
            raise LabelError("each span is an object with start, end, and count")
        count = item["count"]
        if not _number(item["start"]) or not _number(item["end"]) or not item["start"] <= item["end"]:
            raise LabelError("a span needs a start and an end, in order")
        if not _whole(count) or count < 0:
            raise LabelError("count must be a whole number, at least 0")
        if item["start"] < start or item["end"] > end:
            raise LabelError("a span is outside the window the counter was given")
        out.append(CountedSpan(float(item["start"]), float(item["end"]), count))
    return tuple(sorted(out, key=lambda c: (c.start, c.end)))


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Seconds two spans share (0 when they only touch or don't meet)."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    """Shared time over the time either covers: 1 for the same span, 0 when they don't overlap."""
    shared = overlap(a0, a1, b0, b1)
    union = max(a1, b1) - min(a0, b0)
    return shared / union if union > 0 else float(a0 == b0 and a1 == b1)


def score_events(
    spans: Sequence[Event],
    state: str | pathlib.Path = ".",
    counter: EventCounter | None = None,
    margin: float = 30.0,
) -> dict[str, Any]:
    """Without a counter, a summary of the labels. With one, each labeled span goes to the counter as a window widened
    by margin seconds each way, and the span it found that overlaps the label most is compared with it: the count error
    (its count minus the labeled count, or minus the labeled count alone when it found nothing there) and how well the
    two line up. A found span that overlaps no labeled span of the same event on that camera is an extra."""
    by_what = collections.Counter(s.what.casefold() for s in spans)
    out: dict[str, Any] = dict(
        spans=len(spans),
        occurrences=sum(s.count for s in spans),
        kinds=dict(sorted(by_what.items())),
        scored=counter is not None,
    )
    if counter is None:
        out["note"] = "no counter given (--part events --event-counter module:function)"
        return out
    if not _number(margin) or margin < 0:
        raise RunError("event margin must be a number of seconds, at least 0")
    per_span: list[dict[str, Any]] = []
    for label in spans:
        start, end = label.start - margin, label.end + margin
        try:
            raw = counter(camera=label.camera, what=label.what, start=start, end=end, state=pathlib.Path(state))
            found = _counted_spans(raw, start, end)
            diagnostics = json.loads(json.dumps(dict(raw.get("diagnostics", {})), allow_nan=False))
        except Exception as exc:
            raise RunError(f"event counter failed for {label.id}: {exc}") from exc
        labeled = [o for o in spans if o.camera == label.camera and o.what.casefold() == label.what.casefold()]
        touching = [c for c in found if overlap(c.start, c.end, label.start, label.end) > 0]
        best = max(touching, key=lambda c: (overlap(c.start, c.end, label.start, label.end), -c.start), default=None)
        extra = [c for c in found if not any(overlap(c.start, c.end, o.start, o.end) > 0 for o in labeled)]
        predicted = best.count if best else 0
        per_span.append(
            dict(
                id=label.id,
                person=label.person,
                camera=label.camera,
                what=label.what,
                count=label.count,
                predicted_count=predicted,
                count_error=predicted - label.count,
                found=best is not None,
                overlap=iou(best.start, best.end, label.start, label.end) if best else 0.0,
                start_offset_s=best.start - label.start if best else None,
                end_offset_s=best.end - label.end if best else None,
                pieces=len(touching),  # more than one: the counter split the span
                extra_spans=len(extra),
                diagnostics=diagnostics,
            )
        )
    errors = [abs(r["count_error"]) for r in per_span]
    matched = [r for r in per_span if r["found"]]
    out.update(
        margin_s=margin,
        found=len(matched),
        missed=len(per_span) - len(matched),
        exact_counts=sum(r["count_error"] == 0 for r in per_span),
        mean_abs_count_error=statistics.fmean(errors) if errors else None,
        mean_overlap=statistics.fmean(r["overlap"] for r in matched) if matched else None,
        per_span=per_span,
    )
    return out


# ---- report --------------------------------------------------------------------------------------------------------


def _ratio(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _settings_line(s: dict[str, Any]) -> str:
    model = s.get("recall.model") or "unset (ask.model, else models.ollama.llm)"
    answer = s.get("ask.model") or "unset (models.ollama.llm)"
    context = s.get("ask.num_ctx") or "unset (the model's usual)"
    loop = f"on, {s.get('recall.max_rounds')} rounds" if s.get("recall.enabled") else "off"
    journal = "on" if s.get("journal.enabled") else "off"
    vectors = f"on, {s.get('models.ollama.embedding')}" if s.get("semantic.enabled") else "off"
    return (
        f"ask.model {answer}, ask.num_ctx {context}, "
        f"search loop {loop}, recall.model {model}, models.ollama.llm {s.get('models.ollama.llm')}, "
        f"journal {journal}, semantic index {vectors}"
    )


def _vectors(r: dict[str, Any]) -> str:
    """The switches a run set (--vectors, --journal), as a clause of the report's heading."""
    out = "" if "vectors" not in r else f", semantic index {'on' if r['vectors'] else 'off'}"
    return out + ("" if "journal" not in r else f", journal {'on' if r['journal'] else 'off'}")


def _sweep_lines(sweep: dict[str, Any]) -> list[str]:
    """A table per modality: precision and recall at each threshold, today's marked."""
    out = ["Identification by threshold (from the stored similarities; * is today's):"]
    for modality, s in sweep.items():
        out.append(f"  {modality}: {s['clips']} labeled {'clip' if s['clips'] == 1 else 'clips'}")
        if not s["clips"]:
            continue
        out.append(f"    {'threshold':>10} {'precision':>10} {'recall':>7} {'right':>6} {'wrong':>6} {'missed':>7}")
        for r in s["rows"]:
            mark = "*" if r["today"] else " "
            out.append(
                f"    {mark + format(r['threshold'], '.2f'):>10} {_ratio(r['precision']):>10} {_ratio(r['recall']):>7}"
                f" {r['tp']:>6} {r['fp']:>6} {r['fn']:>7}"
            )
    out += [
        "  Only today's threshold and stricter ones: a record left unnamed keeps no candidate, so what a lower",
        "  threshold would name isn't stored. A name with no similarity (one someone labeled) counts at every one.",
        "",
    ]
    return out


# The fields of a reading that are scored. `text` (the journal's search words) has more than one right wording, so it
# isn't. A refused time is scored as refused or not, since the phrase it carries is only for the message.
READ_FIELDS = (
    "history",
    "source",
    "measure",
    "movement",
    "direction",
    "intent",
    "now",
    "people",
    "day",
    "part",
    "clock",
    "clock_end",
    "since",
    "span",
    "span_unit",
    "span_kind",
    "other_time",
)


def _read_value(reading: Any, field: str) -> Any:
    value = getattr(reading, field)
    if field == "people":
        return sorted(p.casefold() for p in value)
    if field == "other_time":
        return bool(value)
    return value


def score_readings(labels: Sequence[ReadLabel]) -> dict[str, Any]:
    """Read each labeled question with the model (memory/question.py's read: one call each to recall.model) and
    compare the reading with the label, field by field.

    A reworded copy (`same_as`) is scored against its original's labeled reading, and the pair agrees when the
    model gave both the same reading, right or wrong. A question the model gives no reading for counts every field
    as wrong, and a question labeled as not about history is scored on that field alone. Returns per-field counts, the number read exactly right, the pairs that agreed, how long a call took,
    and one row per question with the fields it got wrong (expected and got)."""
    from liberational.memory import question

    by_id = {label.id: label for label in labels}
    for label in labels:
        if label.same_as is not None and (label.same_as not in by_id or by_id[label.same_as].reading is None):
            raise LabelError(
                f"readings.jsonl: {label.id!r} is same_as {label.same_as!r}, which is not a labeled reading"
            )
    question.forget()  # every question is read by the model in this run, never taken from an earlier one
    got: dict[str, Any] = {}
    fields = {name: {"right": 0, "of": 0} for name in READ_FIELDS}
    rows: list[dict[str, Any]] = []
    seconds: list[float] = []
    for label in labels:
        expected = label.reading if label.reading is not None else by_id[str(label.same_as)].reading
        row: dict[str, Any] = {"id": label.id, "question": label.question, "wrong": {}}
        started = time.monotonic()
        try:
            got[label.id] = question.read(label.question)
        except question.Unreadable as error:
            got[label.id] = None
            row["unreadable"] = str(error)
        seconds.append(time.monotonic() - started)
        # A question that isn't about history is routed elsewhere, so only that one field matters for it.
        for name in READ_FIELDS if expected.history else ("history",):
            want = _read_value(expected, name)
            have = None if got[label.id] is None else _read_value(got[label.id], name)
            fields[name]["of"] += 1
            if got[label.id] is not None and have == want:
                fields[name]["right"] += 1
            else:
                row["wrong"][name] = {"expected": want, "got": have}
        row["exact"] = not row["wrong"]
        if label.same_as is not None:
            row["same_as"] = label.same_as
        rows.append(row)
    pairs = [label for label in labels if label.same_as is not None]
    for label in pairs:
        ours, theirs = got[label.id], got[str(label.same_as)]
        agreed = (
            ours is not None
            and theirs is not None
            and all(_read_value(ours, name) == _read_value(theirs, name) for name in READ_FIELDS)
        )
        next(row for row in rows if row["id"] == label.id)["agrees_with_original"] = agreed
    return {
        "questions": len(labels),
        "exact": sum(row["exact"] for row in rows),
        "unreadable": sum("unreadable" in row for row in rows),
        "fields": fields,
        "pairs": len(pairs),
        "pairs_agreed": sum(bool(row.get("agrees_with_original")) for row in rows),
        "seconds_median": statistics.median(seconds) if seconds else None,
        "seconds_max": max(seconds) if seconds else None,
        "settings": {k: cfg.get(k, None) for k in ("recall.model", "ask.model", "models.ollama.llm")},
        "per_question": rows,
    }


def report(results: dict[str, dict[str, Any]], directory: str | pathlib.Path, state: str | pathlib.Path) -> str:
    out = [f"Measurement set: {directory}", f"State: {state}", ""]
    if (r := results.get("identification")) is not None:
        kinds = ", ".join(f"{n} {m}" for m, n in sorted(r["by_modality"].items()))
        out.append(f"Identification: {r['clips']} labeled clips and frames" + (f" ({kinds})" if kinds else ""))
        if r["clips"]:
            out.append(
                f"  precision {_ratio(r['precision'])} ({r['tp']} of {r['tp'] + r['fp']} names given were right), "
                f"recall {_ratio(r['recall'])} ({r['tp']} of {r['tp'] + r['fn']} labeled people were named)"
            )
            width = max(len(p) for p in r["per_person"]) if r["per_person"] else 0
            for person, n in r["per_person"].items():
                out.append(f"  {person:<{width}}  right {n['tp']}  wrong {n['fp']}  missed {n['fn']}")
            if r["confusions"]:
                out.append("  confusions (labeled -> named):")
                out += [f"    {c['count']}  {c['labeled']} -> {c['predicted']}" for c in r["confusions"]]
            if r["no_records"]:
                out.append(
                    "  no faces or lines at all in: " + ", ".join(r["no_records"]) + " (not processed yet, or missed)"
                )
        out.append("")
    if (sw := results.get("identification_sweep")) is not None:
        out += _sweep_lines(sw)
    if (r := results.get("retrieval")) is not None:
        how = "on" if r["recorded_frame_analysis"] else "off"
        through = "ask.execute in local mode" if r.get("through") else "history_qa.collect"
        out.append(
            f"Retrieval: {r['questions']} questions through {through} "
            f"(time zone {r['timezone']}, recorded frames {how}{_vectors(r)})"
        )
        if r.get("through"):
            out.append("  " + _settings_line(r["settings"]))
        if r["questions"]:
            out.append(
                f"  passed {r['passed']} of {r['questions']}; answers {r['answers_ok']} of {r['answers_checked']}; "
                f"evidence found {r['evidence_found']} of {r['evidence_expected']}"
                + (f", {r['evidence_given']} of them given to the model" if "evidence_given" in r else "")
            )
            if r.get("drafts"):
                out.append(f"  {r['drafts']} are drafts, not yet confirmed; {r['drafts_passed']} of them passed")
            if r.get("through"):
                out.append(
                    f"  latency median {r['latency_median_s']:.1f} s, max {r['latency_max_s']:.1f} s; "
                    f"search rounds {r['rounds']} in all"
                )
            for q in r["per_question"]:
                if not q["passed"]:
                    draft = " (draft)" if q.get("draft") else ""
                    out.append(f"  FAIL {q['id']}{draft}: " + "; ".join(q["failures"]))
        out.append("")
    if (r := results.get("readings")) is not None:
        out.append(f"Readings: {r['questions']} labeled questions, read by the model (memory/question.py)")
        if r["questions"]:
            out.append(f"  read exactly right {r['exact']} of {r['questions']}; no reading for {r['unreadable']}")
            out.append(f"  reworded copies read like their original {r['pairs_agreed']} of {r['pairs']}")
            out.append(f"  one reading: median {r['seconds_median']:.1f} s, max {r['seconds_max']:.1f} s")
            worst = sorted(r["fields"].items(), key=lambda item: item[1]["right"] - item[1]["of"])
            for name, counts in worst:
                if counts["right"] < counts["of"]:
                    out.append(f"  {name}: {counts['right']} of {counts['of']}")
            for q in r["per_question"]:
                if not q["exact"]:
                    wrong = "; ".join(f"{k} {v['expected']!r} read as {v['got']!r}" for k, v in q["wrong"].items())
                    out.append(f"  MISREAD {q['id']}: " + (q.get("unreadable") and "no reading" or wrong))
        out.append("")
    if (r := results.get("entries")) is not None:
        out.append(f"Entries/exits: {r['windows']} labeled camera windows; tolerance {r['tolerance_s']:g} s")
        for camera, values in r["per_camera"].items():
            for kind, counts in values["by_kind"].items():
                out.append(
                    f"  {camera} {kind}: precision {_ratio(counts['precision'])}, recall {_ratio(counts['recall'])}; "
                    f"{counts['unsupported_windows']} unsupported windows"
                )
            out.append(
                f"  {camera}: excluded {values['excluded_seconds']:g} s; counter unknown {values['unknown_seconds']:g} s"
            )
        out.append("")
    if (r := results.get("events")) is not None:
        kinds = ", ".join(f"{n} {what}" for what, n in r["kinds"].items())
        spans = "span" if r["spans"] == 1 else "spans"
        times = "occurrence" if r["occurrences"] == 1 else "occurrences"
        out.append(
            f"Events: {r['spans']} labeled {spans}" + (f" ({kinds})" if kinds else "") + f", {r['occurrences']} {times}"
        )
        if not r["scored"]:
            out.append(f"  not scored: {r['note']}")
        elif r["spans"]:
            out.append(
                f"  found {r['found']} of {r['spans']}; exact counts {r['exact_counts']} of {r['spans']}; "
                f"mean count error {r['mean_abs_count_error']:.1f} times; mean overlap {_ratio(r['mean_overlap'])}"
            )
            for s in r["per_span"]:
                where = (
                    f"overlap {s['overlap']:.2f}, start {s['start_offset_s']:+.1f} s, end {s['end_offset_s']:+.1f} s"
                    if s["found"]
                    else "not found"
                )
                more = (f", in {s['pieces']} pieces" if s["pieces"] > 1 else "") + (
                    f", {s['extra_spans']} extra" if s["extra_spans"] else ""
                )
                out.append(
                    f"  {s['id']}: {s['what']}, counted {s['predicted_count']} of {s['count']} "
                    f"({s['count_error']:+d}); {where}{more}"
                )
        out.append("")
    if any(
        results[p].get("windows")
        or results[p].get("clips")
        or results[p].get("questions")
        or results[p].get("per_span")
        for p in results
    ):
        out.append("These numbers describe these labels only. A small set gives unstable numbers.")
    if not any(
        results[p].get("windows") or results[p].get("clips") or results[p].get("questions") or results[p].get("spans")
        for p in results
    ):
        out.append(f"No labels yet in {directory}. docs/MEASUREMENT.md says how to add them.")
    return "\n".join(out).rstrip() + "\n"


def compare(path_a: str | pathlib.Path, path_b: str | pathlib.Path) -> str:
    """The retrieval parts of two --json results side by side: totals, then the questions whose result changed."""
    runs = []
    for path in (path_a, path_b):
        try:
            r = json.loads(pathlib.Path(path).read_text()).get("retrieval")
        except (OSError, ValueError, AttributeError) as e:
            raise RunError(f"can't read {path}: {e}") from None
        if not isinstance(r, dict) or not isinstance(r.get("per_question"), list):
            raise RunError(f"{path} has no retrieval result (make it with --part retrieval --json)")
        runs.append(r)
    a, b = runs

    def how(r: dict[str, Any]) -> str:
        return _settings_line(r["settings"]) if r.get("through") else "history_qa.collect only" + _vectors(r)

    def latency(r: dict[str, Any]) -> str:
        return "n/a" if r.get("latency_median_s") is None else f"{r['latency_median_s']:.1f} s"

    out = [f"A: {path_a}", f"   {how(a)}", f"B: {path_b}", f"   {how(b)}", ""]
    for label, key, total in (
        ("passed", "passed", "questions"),
        ("answers", "answers_ok", "answers_checked"),
        ("evidence found", "evidence_found", "evidence_expected"),
    ):
        out.append(f"{label:<16}{a[key]} of {a[total]}  ->  {b[key]} of {b[total]}")
    if "evidence_given" in a and "evidence_given" in b:
        given = (
            f"{a['evidence_given']} of {a['evidence_expected']}  ->  {b['evidence_given']} of {b['evidence_expected']}"
        )
        out.append(f"{'given to model':<16}{given}")
    out.append(f"{'latency median':<16}{latency(a)}  ->  {latency(b)}")
    rows_a = {q["id"]: q for q in a["per_question"]}
    rows_b = {q["id"]: q for q in b["per_question"]}
    both = [i for i in rows_a if i in rows_b]
    for label, ids in (
        ("better (failed in A, passed in B)", [i for i in both if rows_b[i]["passed"] > rows_a[i]["passed"]]),
        ("worse (passed in A, failed in B)", [i for i in both if rows_b[i]["passed"] < rows_a[i]["passed"]]),
        ("only in A", [i for i in rows_a if i not in rows_b]),
        ("only in B", [i for i in rows_b if i not in rows_a]),
    ):
        if ids:
            out.append(f"{label}: " + ", ".join(ids))
    return "\n".join(out) + "\n"


def run(
    directory: str | pathlib.Path,
    state: str | pathlib.Path,
    parts: Sequence[str] = PARTS,
    frame_window: float = 1.0,
    as_configured: bool = False,
    use_ask: bool = False,
    rounds: int | None = None,
    vectors: bool | None = None,
    journal: bool | None = None,
    *,
    entry_counter: EntryCounter | None = None,
    entry_tolerance: float | None = None,
    event_counter: EventCounter | None = None,
    event_margin: float = 30.0,
    sweep: bool = False,
) -> dict[str, dict[str, Any]]:
    results = {}
    if "entries" in parts:
        if entry_counter is None or entry_tolerance is None:
            raise RunError("entries requires an explicit counter and tolerance")
        results["entries"] = score_entries(load(directory, "entries"), state, entry_counter, entry_tolerance)
    if "readings" in parts:
        results["readings"] = score_readings(load(directory, "readings"))
    if "identification" in parts:
        clips = load(directory, "identification", frame_window)
        results["identification"] = score_identification(clips, state)
        if sweep:
            results["identification_sweep"] = sweep_identification(clips, state)
    if "retrieval" in parts:
        results["retrieval"] = score_retrieval(
            load(directory, "retrieval"), state, as_configured, use_ask, rounds, vectors, journal
        )
    if "events" in parts:
        for name in OLD_EVENT_FILES:
            if (old := pathlib.Path(directory) / name).exists():
                raise LabelError(
                    f"{old}: {name} is now events.jsonl. Move its lines as docs/MEASUREMENT.md says "
                    "(Moving old activity labels), then remove it"
                )
        results["events"] = score_events(load(directory, "events"), state, event_counter, event_margin)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", help="label directory (default: paths.measurement)")
    ap.add_argument("--state-dir", help="state to score (default: paths.state)")
    ap.add_argument(
        "--part",
        choices=(*PARTS, "entries", "readings"),
        action="append",
        help="score only this part (repeatable); entries and readings run only when named",
    )
    ap.add_argument("--entry-counter", help="explicit module:function entry/exit counter")
    ap.add_argument("--entry-tolerance", type=float, help="explicit matching tolerance in seconds")
    ap.add_argument("--event-counter", help="explicit module:function event counter")
    ap.add_argument(
        "--event-margin", type=float, default=30.0, help="seconds each way around a labeled span the counter sees"
    )
    ap.add_argument("--frame-window", type=float, default=1.0, help="seconds each way around a labeled frame")
    ap.add_argument("--as-configured", action="store_true", help="keep recorded-frame analysis as configured")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--ask", action="store_true", help="score retrieval through ask.execute (calls the local model)")
    ap.add_argument("--rounds", type=int, help="set recall.max_rounds for the run, 0 for no loop (implies --ask)")
    ap.add_argument("--vectors", choices=("on", "off"), help="set semantic.enabled for the run")
    ap.add_argument("--journal", choices=("on", "off"), help="set journal.enabled for the run")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare the retrieval part of two --json results")
    ap.add_argument(
        "--sweep", action="store_true", help="identification: precision and recall at thresholds from today's up (#291)"
    )
    a = ap.parse_args(argv)
    if not (math.isfinite(a.frame_window) and a.frame_window >= 0):
        ap.error("--frame-window must be a number of seconds, at least 0")
    if a.rounds is not None and a.rounds < 0:
        ap.error("--rounds must be a whole number, at least 0")
    if "entries" in (a.part or ()):
        if not a.entry_counter or a.entry_tolerance is None:
            ap.error("--part entries requires --entry-counter and --entry-tolerance")
        if not math.isfinite(a.entry_tolerance) or a.entry_tolerance < 0:
            ap.error("--entry-tolerance must be finite and nonnegative")
    elif a.entry_counter is not None or a.entry_tolerance is not None:
        ap.error("entry options require explicit --part entries")
    if a.sweep and "identification" not in (a.part or PARTS):
        ap.error("--sweep scores identification: use it with --part identification")
    if a.event_counter is not None and "events" not in (a.part or ()):
        ap.error("--event-counter requires explicit --part events")
    if not (math.isfinite(a.event_margin) and a.event_margin >= 0):
        ap.error("--event-margin must be a number of seconds, at least 0")
    if a.compare:
        try:
            print(compare(*a.compare), end="")
        except RunError as e:
            print(f"measure: {e}", file=sys.stderr)
            return 2
        return 0
    directory = pathlib.Path(a.dir) if a.dir else cfg.path("paths.measurement")
    state = pathlib.Path(a.state_dir) if a.state_dir else cfg.path("paths.state")
    try:
        vectors = None if a.vectors is None else a.vectors == "on"
        journal = None if a.journal is None else a.journal == "on"
        results = run(
            directory,
            state.resolve(),
            a.part or PARTS,
            a.frame_window,
            a.as_configured,
            a.ask,
            a.rounds,
            vectors,
            journal,
            entry_counter=load_entry_counter(a.entry_counter) if "entries" in (a.part or ()) else None,
            entry_tolerance=a.entry_tolerance,
            event_counter=load_event_counter(a.event_counter) if a.event_counter else None,
            event_margin=a.event_margin,
            sweep=a.sweep,
        )
    except (LabelError, RunError) as e:
        print(f"measure: {e}", file=sys.stderr)
        return 2
    print(json.dumps(results, indent=2, ensure_ascii=False) if a.json else report(results, directory, state), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
