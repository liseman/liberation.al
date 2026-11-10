"""Revised transcript lines from the overnight second pass (listen/deep.py), by utterance id:
`<paths.state>/transcript_revisions_qwen.sqlite3`. Kept apart from deep.py, which loads the speech models, so the
transcript view (and everything that imports it) reads revisions without depending on them.
"""

import bisect
import functools
import math
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liberational import knowledge

CORRECTIONS = "transcript.correction"
_NS = 1_000_000_000
# A line nobody in the room said: played back, synthetic, or heard from a TV or radio (`media`). Readers that
# speak for people (the journal, Ask, the timeline) leave out a line with any of these set.
NOT_SAID = ("playback", "is_playback", "cloned_playback", "synthetic", "is_synthetic", "media")
MEDIA_SOURCES = ("owner", "classifier")
NOT_MEDIA = "not-media"  # the owner's Not TV on some lines: their marks taken back, those lines alone, with an Undo  # who marked a line as a TV's: the owner's "This is TV", or the audio tagger


def _standing(records: Sequence[knowledge.Record], action: str) -> list[knowledge.Record]:
    undone = {r.value.get("undo") for r in records if r.value.get("action") == "undo"}
    return [r for r in records if r.value.get("action") == action and r.id not in undone]


@dataclass(frozen=True)
class Corrections:
    """The corrections, read and worked out once per change of them. The audio tagger writes one per TV line,
    and readers on hot paths (filter_lines for every line read, review_revision polled by every Live page) asked for
    all of them each time, each a query and a JSON parse per record ."""

    stored: tuple[knowledge.StoredRecord, ...] = ()
    rejected: tuple[knowledge.Record, ...] = ()
    media: tuple[knowledge.Record, ...] = ()
    hidden_ids: frozenset[str] = frozenset()
    hidden_keys: frozenset[str] = frozenset()
    # Who marked each line that is a TV's now, by id and by key: its standing marks newer than its last standing Not TV.
    media_by_id: Mapping[str, str] = field(default_factory=dict)
    media_by_key: Mapping[str, str] = field(default_factory=dict)
    newest: int = 0
    spans: tuple[tuple[float, float], ...] = ()

    def revision(self, start: float | None = None, end: float | None = None) -> int:
        """Latest recorded rejection or Undo affecting the window; zero before its first correction."""
        if start is None and end is None:
            return self.newest
        lo = 0 if start is None else int(max(0.0, start) * _NS)
        hi = 2**63 - 1 if end is None or not math.isfinite(end) else int(end * _NS)
        starts, ends, positions, longest = self._by_start
        # Only a correction starting within `longest` before lo can reach past it (a journal pass asks this
        # for every entry, and the audio tagger's corrections number in the thousands).
        first, last = bisect.bisect_left(starts, lo - longest), bisect.bisect_left(starts, hi)
        return max((positions[i] for i in range(first, last) if ends[i] > lo), default=0)

    def affected(self, start: float, end: float) -> bool:
        """Whether any correction's span overlaps [start, end) (seconds; a span is at least a microsecond long)."""
        starts, ends, longest = self._spans
        first, last = bisect.bisect_left(starts, start - longest - 1.0), bisect.bisect_left(starts, end)
        return any(ends[i] > start for i in range(first, last))

    @functools.cached_property
    def _by_start(self) -> tuple[list[int], list[int], list[int], int]:
        """The stored corrections by start: their starts, ends (at least a nanosecond past the start), positions,
        and the longest one's length, so a window's look is a bisect and a short scan."""
        rows = sorted((r.record.start_ns, max(r.record.end_ns, r.record.start_ns + 1), r.position) for r in self.stored)
        return (
            [r[0] for r in rows],
            [r[1] for r in rows],
            [r[2] for r in rows],
            max((r[1] - r[0] for r in rows), default=0),
        )

    @functools.cached_property
    def _spans(self) -> tuple[list[float], list[float], float]:
        """`spans` as their starts, their ends (at least a microsecond past the start), and the longest's length."""
        spans = sorted(self.spans)
        ends = [max(b, a + 1e-6) for a, b in spans]
        return [a for a, _ in spans], ends, max((e - a for (a, _), e in zip(spans, ends, strict=True)), default=0.0)


_EMPTY = Corrections()
_indexes: dict[str, tuple[tuple[str, int, int], Corrections]] = {}
_indexes_lock = threading.Lock()


def _build(stored: list[knowledge.StoredRecord]) -> Corrections:
    records = [r.record for r in stored]
    rejected, media = _standing(records, "reject"), _standing(records, "media")
    hidden_ids, hidden_keys = _lines_of(rejected)
    # A line's last standing Not TV takes back the marks made before it, on that line alone; a newer mark stands.
    position = {r.record.id: r.position for r in stored}
    off: dict[str, int] = {}
    for r in _standing(records, NOT_MEDIA):
        for line in r.value["lines"]:
            off[line["line"]] = max(off.get(line["line"], 0), position[r.id])
    # Who marked each line (`media_by`): the owner's "This is TV", or the audio tagger's ear. An owner's mark wins.
    by_id: dict[str, str] = {}
    by_key: dict[str, str] = {}
    for r in sorted(media, key=lambda r: r.value.get("source") == "owner"):
        for line in r.value["lines"]:
            if position[r.id] <= off.get(line["line"], 0):
                continue
            source = str(r.value.get("source") or "owner")
            if line.get("id"):
                by_id[line["id"]] = source
            by_key[line["line"]] = source
    return Corrections(
        stored=tuple(stored),
        rejected=tuple(rejected),
        media=tuple(media),
        hidden_ids=frozenset(hidden_ids),
        hidden_keys=frozenset(hidden_keys),
        media_by_id=by_id,
        media_by_key=by_key,
        newest=max((r.position for r in stored), default=0),
        spans=tuple(sorted({(r.record.start_ns / _NS, r.record.end_ns / _NS) for r in stored})),
    )


def _index(state: Path) -> Corrections:
    """The corrections' index, read again only when they changed: an insert, a removal, or another database file."""
    path = Path(state) / "knowledge.sqlite3"
    if not path.exists():
        return _EMPTY
    store = knowledge.at(path)
    generation = store.generation([CORRECTIONS])
    with _indexes_lock:
        hit = _indexes.get(str(path))
    if hit is not None and hit[0] == generation:
        return hit[1]
    # A write landing between the generation and the read is in the read; the next call sees a newer generation.
    index = _build(store.query(CORRECTIONS))
    with _indexes_lock:
        _indexes[str(path)] = (generation, index)
    return index


def _corrections(state: Path) -> list[knowledge.StoredRecord]:
    """Owner corrections share the existing knowledge store, never the raw speech or overnight ASR tables."""
    return list(_index(state).stored)


def review_revision(state: Path, start: float | None = None, end: float | None = None) -> int:
    """Latest recorded rejection or Undo affecting the window; zero before its first correction."""
    return _index(state).revision(start, end)


def review_snapshot(state: Path) -> Corrections:
    """The corrections in one read, for a reader asking about many windows (its `revision` and `spans`). Each
    `review_revision` call reads the knowledge store's generation again, a transaction apiece."""
    return _index(state)


def reviewed_spans(state: Path) -> list[tuple[float, float]]:
    """Windows with owner review history, so old corrected journal windows can be revisited directly."""
    return list(_index(state).spans)


def _active(state: Path) -> list[knowledge.Record]:
    return list(_index(state).rejected)


def _media(state: Path) -> list[knowledge.Record]:
    """The standing marks of lines as a TV's or radio's."""
    return list(_index(state).media)


def rejected_subjects(state: Path) -> frozenset[str]:
    """Utterance subjects currently rejected by the owner, including batch members."""
    return frozenset(f"utterance:{line['line']}" for r in _active(state) for line in r.value["lines"])


def _is_line(row: Any) -> bool:
    return (
        isinstance(row, dict) and not row.get("assistant") and any(f in row for f in ("text", "speaker", "speaker_id"))
    )


def _marked(row: dict[str, Any], ids: AbstractSet[str], keys: AbstractSet[str]) -> bool:
    from liberational.people import registry

    return row.get("id") in ids or (isinstance(row.get("ts_start"), int | float) and registry.key_of(row) in keys)


def _lines_of(records: Sequence[knowledge.Record]) -> tuple[set[str], set[str]]:
    ids = {line["id"] for r in records for line in r.value["lines"] if line.get("id")}
    return ids, {line["line"] for r in records for line in r.value["lines"]}


def media_subjects(state: Path) -> frozenset[str]:
    """Utterance subjects currently marked as a TV's or radio's."""
    return frozenset(f"utterance:{key}" for key in _index(state).media_by_key)


def filter_lines(state: Path, items: Iterable[Any]) -> list[Any]:
    """Every ordinary reader's lines: rejected speech hidden, and a line marked as a TV's or radio's kept with `media`
    set, and `media_by` saying who marked it ("owner" or "classifier"), so the transcript shows it and readers that speak for people leave it out (NOT_SAID). Raw rows and
    assistant replies are kept as they are."""
    index = _index(state)
    if not index.rejected and not index.media_by_key:
        return list(items)
    hidden_ids, hidden_keys = index.hidden_ids, index.hidden_keys
    by_id, by_key = index.media_by_id, index.media_by_key
    media = bool(by_key)
    out = []
    for row in items:
        if not _is_line(row):
            out.append(row)
        elif _marked(row, hidden_ids, hidden_keys):
            continue
        elif media and (source := _source(row, by_id, by_key)) is not None:
            out.append({**row, "media": True, "media_by": source})
        else:
            out.append(row)
    return out


def _source(row: dict[str, Any], by_id: Mapping[str, str], by_key: Mapping[str, str]) -> str | None:
    from liberational.people import registry

    if (source := by_id.get(str(row.get("id")))) is not None:
        return source
    return by_key.get(registry.key_of(row)) if isinstance(row.get("ts_start"), int | float) else None


_review_lock = threading.RLock()


def current_rejection(state: Path, keys: Sequence[str]) -> str | None:
    """The existing durable token for exactly this active batch, so a request retry is harmless."""
    wanted = set(keys)
    return next((r.id for r in _active(state) if wanted == {line["line"] for line in r.value["lines"]}), None)


def reject_lines(state: Path, entries: Sequence[Mapping[str, Any]]) -> str:
    """Record one atomic owner rejection for the whole visible turn. Returns its durable Undo token."""
    from liberational.people import registry

    if not entries:
        raise ValueError("Choose at least one transcript line.")
    lines = [{"id": e.get("id"), "line": registry.key_of(e)} for e in entries]
    keys = {e["line"] for e in lines}
    with _review_lock:
        active = _active(state)
        for record in active:
            if keys == {line["line"] for line in record.value["lines"]}:
                return record.id
        if keys & {line["line"] for record in active for line in record.value["lines"]}:
            raise ValueError("One of those lines was already rejected. Refresh the transcript.")
        token = str(uuid.uuid7())
        start = min(int(float(e["ts_start"]) * _NS) for e in entries)
        end = max(int(float(e.get("ts_end") or e["ts_start"]) * _NS) for e in entries)
        record = knowledge.Record(
            token,
            CORRECTIONS,
            token,
            start,
            max(start, end),
            time.time_ns(),
            {"action": "reject", "reason": "not_transcribed_correctly", "lines": lines},
            tuple(f"utterance:{key}" for key in keys),
        )
        knowledge.at(Path(state) / "knowledge.sqlite3").put([record])
        return token


def current_media(state: Path, keys: Sequence[str], source: str = "owner") -> str | None:
    """The standing token of `source`'s mark of exactly these lines while they're all still a TV's, so a request retry
    is harmless. Another source's mark is not this one: the owner's Undo never takes back the tagger's mark."""
    wanted = set(keys)
    index = _index(state)
    if not wanted <= index.media_by_key.keys():
        return None
    return next(
        (
            r.id
            for r in index.media
            if str(r.value.get("source") or "owner") == source and wanted == {line["line"] for line in r.value["lines"]}
        ),
        None,
    )


def mark_media(state: Path, entries: Sequence[Mapping[str, Any]], source: str = "owner") -> str:
    """Record one atomic mark of these lines as heard from a TV or radio: kept in the transcript, left out of
    people, voices, the journal, and Ask. Lines already marked are left as they are. Returns its durable Undo token."""
    from liberational.people import registry

    if source not in MEDIA_SOURCES:
        raise ValueError(f"A line is marked by one of {', '.join(MEDIA_SOURCES)}.")
    if not entries:
        raise ValueError("Choose at least one transcript line.")
    with _review_lock:
        keys = [registry.key_of(e) for e in entries]
        if token := current_media(state, keys, source):
            return token
        marked = _index(state).media_by_key
        fresh = [e for e in entries if registry.key_of(e) not in marked]
        if not fresh:
            raise ValueError("Those lines are marked as a TV's already.")
        lines = [{"id": e.get("id"), "line": registry.key_of(e)} for e in fresh]
        token = str(uuid.uuid7())
        start = min(int(float(e["ts_start"]) * _NS) for e in fresh)
        end = max(int(float(e.get("ts_end") or e["ts_start"]) * _NS) for e in fresh)
        record = knowledge.Record(
            token,
            CORRECTIONS,
            token,
            start,
            max(start, end),
            time.time_ns(),
            {"action": "media", "source": source, "lines": lines},
            tuple(f"utterance:{line['line']}" for line in lines),
        )
        knowledge.at(Path(state) / "knowledge.sqlite3").put([record])
        return token


def unmark_media(state: Path, keys: Sequence[str]) -> tuple[str | None, list[str]]:
    """The owner's Not TV: these lines, and no others, are a TV's no more, whoever marked them. One write, so a
    double click or a race with an Undo finds nothing left to unmark instead of failing partway.
    Returns its Undo token and the lines unmarked; (None, []) when none of them is a TV's now."""
    wanted = list(dict.fromkeys(keys))
    with _review_lock:
        index = _index(state)
        lines = [key for key in wanted if key in index.media_by_key]
        if not lines:
            return None, []
        marks = [r for r in index.media if {line["line"] for line in r.value["lines"]} & set(lines)]
        ids = {line["line"]: line.get("id") for r in marks for line in r.value["lines"]}
        token = str(uuid.uuid7())
        record = knowledge.Record(
            token,
            CORRECTIONS,
            token,
            min(r.start_ns for r in marks),
            max(r.end_ns for r in marks),
            time.time_ns(),
            {"action": NOT_MEDIA, "lines": [{"id": ids.get(key), "line": key} for key in lines]},
            tuple(f"utterance:{key}" for key in lines),
        )
        knowledge.at(Path(state) / "knowledge.sqlite3").put([record])
        return token, lines


def undo_correction(state: Path, token: str) -> list[str]:
    """Restore only the rejection, TV mark, or Not TV named by the token; a later independent correction cannot be
    undone by it."""
    with _review_lock:
        all_records = [r.record for r in _corrections(state)]
        rejection = next(
            (r for r in all_records if r.id == token and r.value.get("action") in ("reject", "media", NOT_MEDIA)), None
        )
        if rejection is None:
            raise KeyError("Nothing to undo there.")
        if not any(r.id == token for r in _standing(all_records, str(rejection.value["action"]))):
            raise ValueError("That correction was undone already.")
        record = knowledge.Record(
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"transcript-rejection-undo:{token}")),
            CORRECTIONS,
            token,
            rejection.start_ns,
            rejection.end_ns,
            time.time_ns(),
            {"action": "undo", "undo": token},
            rejection.subjects,
        )
        knowledge.at(Path(state) / "knowledge.sqlite3").put([record])
        return [line["line"] for line in rejection.value["lines"]]


class Revisions:
    """The revised lines, by utterance id: `<paths.state>/transcript_revisions_qwen.sqlite3`. Readers cache the table and
    reread it when the file changes."""

    COLS = ("id", "ts_start", "device", "live", "parakeet", "qwen", "text", "method", "uncertain", "at")

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: tuple[float, dict[str, dict[str, Any]]] | None = None

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.execute(
            "create table if not exists revisions (id text primary key, ts_start real, device text, live text, "
            "parakeet text, qwen text, text text, method text, uncertain integer, at real)"
        )
        return db

    def done(self) -> set[str]:
        if not self.path.exists():
            return set()
        with self._db() as db:
            return {r[0] for r in db.execute("select id from revisions")}

    def put(self, row: dict[str, Any]) -> None:
        with self._db() as db:
            db.execute(
                f"insert or replace into revisions ({', '.join(self.COLS)}) values ({', '.join('?' * len(self.COLS))})",
                [row.get(c) for c in self.COLS],
            )

    def all(self) -> dict[str, dict[str, Any]]:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return {}
        if self._cache is None or self._cache[0] != mtime:
            with self._db() as db:
                rows = db.execute(f"select {', '.join(self.COLS)} from revisions").fetchall()
            self._cache = (mtime, {r[0]: dict(zip(self.COLS, r, strict=True)) for r in rows})
        return self._cache[1]

    def apply(self, e: Any) -> Any:
        """`e` with its revised text, when there is one: the live text kept as `text_live`."""
        if not isinstance(e, dict) or not e.get("id"):
            return e
        r = self.all().get(str(e["id"]))
        if r is None or r["uncertain"] or r["method"] == "kept" or not r["text"]:
            return e
        return {**e, "text": r["text"], "text_live": e.get("text"), "revised": r["method"]}


_shared: dict[str, Revisions] = {}


def revisions(state: Path) -> Revisions:
    key = str(state)
    if key not in _shared:
        _shared[key] = Revisions(Path(state) / "transcript_revisions_qwen.sqlite3")
    return _shared[key]
