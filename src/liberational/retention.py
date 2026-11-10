"""Retention past the 30-day window: keep the meaning, drop the measurement.

Each night `pack-video-window` runs this after it ages out the video. It takes every ended local day older than the
window (`video_window.days`) whose day page is written (a `journal.page` at level `day`, a quiet day's marker
included: the page is what the day is remembered by, so a day not yet summed up is never thinned), oldest first, at
most `MAX_DAYS` a run, and for each:

1. **Condenses** the dense coverage streams (`action.coverage`, `walkby.coverage`, `animal.coverage`,
   `zone.coverage`, about one row a second a camera) into contiguous intervals, and the device sightings
   (`network.device_observation`, `bluetooth.sighting`, one row a poll) into visits: rows of one camera with the same
   fields, at most `COVERAGE_GAP_S` (coverage) or `VISIT_GAP_S` (sightings) apart, within one UTC hour, become one
   row from the first's start to the last's end (coverage sums its `frames`); and the positions (`position.sample`,
   about one row a second a tracked person or pet) into paths, one sample per track per `POSITION_EVERY_S` and its
   last. They go to each stream's
   `.condensed` companion (`segment_log.CONDENSED`), which every `segment_log.Log` read merges in.
2. **Drops** the per-frame rows that the kept spans already condense (`pose.track` into `action.span`,
   `walkby.span`, `presence.span`, and `zone.visit`; `face.match` and `face.signal` into `face.visit`), the per-frame
   face embeddings beside them (`face.embedding`; taught face and voice vectors live in the knowledge store and
   `faces/db.npz`, never here), and the condensed streams' originals: one segment-store forget of kind `reset` over
   the day, so what was learned from the rows (the vector index, the journal) stands.
3. **Drops silent audio**: every audio chunk of the day (`*.audio`, about 15 s each) that no speech touches is aged
   out like old video. Speech is an utterance (`speech.utterance`, by its audio span) or a second the voice gate
   heard speech (`speech.activity`), widened by `SPEECH_MARGIN_S` on each side.

Kept: spans and events (`action.span`, `walkby.span`, `zone.visit`, `presence.span`, `face.visit`), captions,
transcripts, voice embeddings, the journal, and taught state. `movement.*` and `action.window` aren't touched yet.

Each finished day is recorded in `state/retention.json` with what went, and logged per stream. A crash part way
through a day leaves it unrecorded, so the next night does it again: a camera whose condensed rows for the day are
already written isn't written twice, and a forget or an age-out of what's gone already does nothing.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liberational import json_state, logs
from liberational import segment_log as S

log = logs.get("retention")

MARKS = "retention.json"
MAX_DAYS = 7  # days one run takes, oldest first: a night's work stays bounded after a long gap
COVERAGE_GAP_S = 2.0  # coverage rows this close are one interval (they come about a second apart)
VISIT_GAP_S = 60.0  # sightings this close are one visit (well under devices.gap_s, which joins stays)
SPEECH_MARGIN_S = 5.0  # audio this close to speech is kept
GATE_S = 1.0  # a speech.activity row says the gate heard speech for about this long (rows come every second)
HOUR_S = 3600.0

POSES = "pose.track"  # see-actions writes it with its own schema; it's dropped by name
DROPPED = (POSES, S.FACE_MATCHES, S.FACE_SIGNALS, S.FACE_EMBEDDINGS)
COVERAGE = (S.ACTION_COVERAGE, S.WALKBY_COVERAGE, S.ANIMAL_COVERAGE, S.ZONE_COVERAGE)
SIGHTINGS = (S.NET_OBSERVATIONS, S.BLE_SIGHTINGS)
POSITIONS = (S.POSITIONS, S.PLACED, S.PLACED_PATHS)
POSITION_EVERY_S = 10.0  # one position per track (or entity) per this long is kept past the window

Record = dict[str, Any]


# Merging: pure, over one camera's records of one day, in time order.


def _coverage_key(r: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (k, json.dumps(v, sort_keys=True))
            for k, v in r.items()
            if k not in ("id", "ts_start", "ts_end", "cam", "frames")
        )
    )


def _net_key(r: Mapping[str, Any]) -> tuple[Any, ...]:
    volatile = ("id", "inventory_id", "start", "end", "ts_start", "source")
    return tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in r.items() if k not in volatile))


def _ble_key(r: Mapping[str, Any]) -> tuple[Any, ...]:
    """What links and readers tell a Bluetooth device and its sensor by; the rest (signal, counters) is per poll."""
    from liberational.proximity import ble

    return (
        r.get("identifier"),
        r.get("identifier_kind"),
        ble.payload_fingerprint(dict(r)),
        r.get("local_name"),
        r.get("sensor_name"),
        r.get("adapter"),
        r.get("source"),
    )


def _hour(t: float) -> int:
    return int(t // HOUR_S)


def merge_coverage(rows: Iterable[Mapping[str, Any]], gap: float = COVERAGE_GAP_S) -> list[Record]:
    """Coverage rows as contiguous intervals: rows with the same fields, at most `gap` apart and starting in one UTC
    hour, become one row from the first's start to the latest end, with their `frames` summed."""
    open_: dict[tuple[Any, ...], Record] = {}
    out: list[Record] = []
    for r in sorted(rows, key=lambda r: (r["ts_start"], str(r.get("id", "")))):
        key = _coverage_key(r)
        cur = open_.get(key)
        if (
            cur is not None
            and float(r["ts_start"]) - float(cur["ts_end"]) <= gap
            and _hour(float(r["ts_start"])) == _hour(float(cur["ts_start"]))
        ):
            cur["ts_end"] = max(float(cur["ts_end"]), float(r["ts_end"]))
            cur["frames"] = int(cur.get("frames", 0)) + int(r.get("frames", 0))
            continue
        cur = dict(r)
        open_[key] = cur
        out.append(cur)
    return out


def merge_sightings(
    rows: Iterable[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], tuple[Any, ...]], gap: float = VISIT_GAP_S
) -> list[Record]:
    """Sightings as visits: one device's rows with the same `key`, at most `gap` apart and starting in one UTC hour,
    become the first row stretched to the latest end."""
    open_: dict[tuple[Any, ...], Record] = {}
    out: list[Record] = []
    for r in sorted(rows, key=lambda r: (float(r["start"]), str(r.get("id", "")))):
        k = key(r)
        cur = open_.get(k)
        if (
            cur is not None
            and float(r["start"]) - float(cur["end"]) <= gap
            and _hour(float(r["start"])) == _hour(float(cur["start"]))
        ):
            cur["end"] = max(float(cur["end"]), float(r["end"]))
            continue
        cur = dict(r)
        open_[k] = cur
        out.append(cur)
    return out


def thin_positions(rows: Iterable[Mapping[str, Any]], every: float = POSITION_EVERY_S) -> list[Record]:
    """Positions thinned to paths (spatial/positions.py's `thin`): of each path's (a placed row's `path`, else its
    track's, or with none its entity's) samples from one source and camera, the first in every `every` seconds and the
    last. A walk's shape and its ends survive; the second-by-second jitter doesn't. A row with neither track nor
    entity is kept as its own path, and so is every zone row: it is a visit's whole span, not a sample of one."""
    from liberational.spatial import positions

    def key(r: Mapping[str, Any]) -> object:
        if r.get("source") == positions.ZONE:
            return f"row:{r.get('id')}"
        line = r.get("path") or r.get("track")
        return positions.path_key(line, r.get("entity"), r.get("source"), r.get("camera"), str(r.get("id")))

    return [dict(r) for r in positions.thin(list(rows), key, lambda r: float(r["ts_start"]), every)]


def merged(stream: str, rows: Sequence[Mapping[str, Any]]) -> list[Record]:
    if stream in POSITIONS:
        return thin_positions(rows)
    if stream == S.NET_OBSERVATIONS:
        return merge_sightings(rows, _net_key)
    if stream == S.BLE_SIGHTINGS:
        return merge_sightings(rows, _ble_key)
    return merge_coverage(rows)


def speech_spans(
    utterances: Iterable[Mapping[str, Any]], activity: Iterable[Mapping[str, Any]], stream: str
) -> list[tuple[float, float]]:
    """Where `stream` (an audio stream) has speech: its utterances' audio spans and the seconds its gate heard speech,
    each widened by SPEECH_MARGIN_S, merged, in order."""
    spans: list[tuple[float, float]] = []
    for u in utterances:
        audio = u.get("audio") or {}
        if audio.get("stream") == stream and audio.get("start_ns") is not None:
            spans.append((int(audio["start_ns"]) / 1e9, int(audio.get("end_ns") or audio["start_ns"]) / 1e9))
        elif not audio and stream.startswith(f"{u.get('device')}."):
            spans.append((float(u["ts_start"]), float(u.get("ts_end", u["ts_start"]))))
    for a in activity:
        if a.get("stream") == stream and a.get("in_speech") is True:
            spans.append((float(a["ts_start"]), float(a["ts_start"]) + GATE_S))
    out: list[tuple[float, float]] = []
    for lo, hi in sorted((lo - SPEECH_MARGIN_S, hi + SPEECH_MARGIN_S) for lo, hi in spans):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def silent(chunks: Iterable[tuple[str, float, float]], speech: Sequence[tuple[float, float]]) -> list[str]:
    """The ids of the chunks (id, start, end) that no speech span touches."""
    return [cid for cid, a, b in chunks if not any(lo < b and hi > a for lo, hi in speech)]


# The days to do, and their record.


@dataclass
class Marks:
    """`state/retention.json`: each finished day with what went."""

    path: Path
    days: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def load(cls, state: Path) -> Marks:
        path = state / MARKS
        try:
            days = json.loads(path.read_text()).get("days", {})
        except FileNotFoundError, ValueError, AttributeError:
            days = {}
        return cls(path, {str(k): dict(v) for k, v in days.items()} if isinstance(days, dict) else {})

    def done(self, day: str, counts: Mapping[str, int]) -> None:
        self.days[day] = dict(counts)
        json_state.write(self.path, {"days": self.days}, indent=1, sort_keys=True)


def first_day(segments: Path) -> str | None:
    """The local day of the oldest hour any segment stream holds, or None for an empty store."""
    oldest: str | None = None
    try:
        streams = [e for e in os.scandir(segments) if e.is_dir() and not e.name.startswith(".")]
    except FileNotFoundError:
        return None
    for stream in streams:
        for cam in os.scandir(stream.path):
            if not cam.is_dir():
                continue
            for f in os.scandir(cam.path):
                hour = f.name[:10]
                if hour.isdigit() and (oldest is None or hour < oldest):
                    oldest = hour
    if oldest is None:
        return None
    at = dt.datetime.strptime(oldest, "%Y%m%d%H").replace(tzinfo=dt.UTC).timestamp()
    return local_day(at)


def local_day(t: float) -> str:
    from liberational.memory.history_qa import local_zone

    return dt.datetime.fromtimestamp(t, local_zone()).date().isoformat()


def due(first: str | None, done: Iterable[str], now: float, days: float, written: Callable[[str], bool]) -> list[str]:
    """The days to do this run, oldest first: from `first` to the last day ended `days` ago, not done yet, whose day
    page is written; at most MAX_DAYS."""
    from liberational.memory_log import day_bounds

    if first is None:
        return []
    finished = set(done)
    out: list[str] = []
    waiting: list[str] = []
    day = dt.date.fromisoformat(first)
    while len(out) < MAX_DAYS:
        name = day.isoformat()
        if day_bounds(name)[1] > now - days * 86400:
            break
        if name not in finished:
            (out if written(name) else waiting).append(name)
        day += dt.timedelta(days=1)
    if waiting:
        log.info(
            "%d days past the window have no day page yet and are kept whole (%s to %s)",
            len(waiting),
            waiting[0],
            waiting[-1],
        )
    return out


def day_written(state: Path) -> Callable[[str], bool]:
    """Whether a day's page is written: a `journal.page` at level `day` for it, a quiet day's marker included."""
    from liberational import memory_log

    pages = memory_log.at(state)

    def written(day: str) -> bool:
        a, _ = memory_log.day_bounds(day)
        found = memory_log.current(pages.range(memory_log.PAGES, a, a + 1.0))
        return any(r.get("level") == "day" and r.get("period") == day for r in found)

    return written


# Doing a day's segments.


class Store:
    """The segment store's side of a day: raw reads (no condensed companion), appends, and forgets."""

    def __init__(self, base: Path) -> None:
        self.base = base

    def cameras(self, stream: str) -> list[str]:
        try:
            return sorted(e.name for e in os.scandir(self.base / stream) if e.is_dir())
        except FileNotFoundError:
            return []

    def read(self, stream: str, start: float, end: float, cam: str) -> list[Record]:
        """`stream`'s own records of `cam` starting in [start, end): each row belongs to the one day it starts in."""
        from continuum_py import read_segments

        got = read_segments(self.base, stream, S.STREAMS[stream].schema, S.ns(start), S.ns(end), [cam])
        return [r for r in S.records(stream, got) if start <= float(r["ts_start"]) < end]

    def append(self, stream: str, cam: str, rows: Sequence[Mapping[str, Any]]) -> None:
        writers = S.Writers(self.base)
        try:
            writers.append(stream, cam, rows)
            writers.sync()
        finally:
            writers.close()

    def forget(self, streams: Sequence[str], start: float, end: float) -> None:
        from continuum_py import forget_segments

        forget_segments(self.base, S.ns(start), S.ns(end), list(streams), None, "reset")

    def size(self, stream: str, start: float, end: float) -> int:
        """Bytes of `stream`'s hour files that start in [start, end), for the log."""
        first = time.strftime("%Y%m%d%H", time.gmtime(start))
        last = time.strftime("%Y%m%d%H", time.gmtime(end - 1e-6))
        total = 0
        for cam in self.cameras(stream):
            for f in os.scandir(self.base / stream / cam):
                if first <= f.name[:10] <= last and f.is_file():
                    total += f.stat().st_size
        return total


def segments_day(store: Store, day: str) -> dict[str, int]:
    """Condenses and drops one day's segment rows; returns rows kept as condensed and bytes dropped, per stream."""
    from liberational.memory_log import day_bounds

    a, b = day_bounds(day)
    counts: dict[str, int] = {}
    for stream in (*COVERAGE, *SIGHTINGS, *POSITIONS):
        for cam in store.cameras(stream):
            rows = store.read(stream, a, b, cam)
            if not rows:
                continue
            if store.read(S.condensed(stream), a, b, cam):
                continue  # condensed on a run that stopped before its forget
            kept = merged(stream, rows)
            store.append(S.condensed(stream), cam, kept)
            counts[f"{stream} rows"] = counts.get(f"{stream} rows", 0) + len(rows)
            counts[f"{stream} kept"] = counts.get(f"{stream} kept", 0) + len(kept)
    gone = (*DROPPED, *COVERAGE, *SIGHTINGS, *POSITIONS)
    for stream in gone:
        size = store.size(stream, a, b)
        if size:
            counts[f"{stream} bytes"] = size
    store.forget(gone, a, b)
    return counts


def report(day: str, counts: Mapping[str, int]) -> None:
    for name in sorted(counts):
        log.info("%s: %s %d", day, name, counts[name])
