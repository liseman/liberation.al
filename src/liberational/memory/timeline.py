"""Timeline and encounters: who was seen or heard, and when (step 7 of docs/DOMAIN-MODEL.md).

An encounter (`domain.memory.Encounter`) is a stretch of time in which one person was seen, heard, or both. It is a
layer 2 projection: built from layer 1 and the identity assertions, never stored, and never entered by hand.

Where encounters come from:

- Seen: face visits in the segment store (face.visit, read through vision/sightings.py): see-actions groups each
  name's matches per camera into visits. A visit carries the name it was matched under, so the name is resolved to a
  person through the registry (`Registry.known_as`: the person called that now, or before a rename or a merge).
- Heard: the speech log's utterances (on Continuum, liberational/speech_log.py) read through `people/transcript_view.View`, which says who said each line by person_of
  (`View.speakers`), so a label given only in the event log counts. Assistant lines, lines marked as playback or heard
  while the pack was speaking (playback_windows.json), and lines that name nobody are left out. Phone calls are not
  presence, so calls/ is not read.
- A person the registry doesn't know, or has forgotten, has no encounters: person_of resolves them to nobody.

Each person's visits and lines are sorted, and joined into one encounter while the gap between them is at most
`timeline.gap_s`. The encounter records whether the person was seen, heard, or both, and keeps the visits and lines
it was built from as evidence. Display names come from the registry, so a rename shows the new name.

Face detections are not in the event log yet, so `TodayStores` is the one adapter that reads them, with the speech
log. When they move, it is the only part to replace.

Cache: the timeline is built on first use and kept until one of the files in SOURCES, the speech log, or the face
visits change, the way dashboard/presence.py caches Review's chips, plus where the event log stands (`Registry.mark`), because a label given
only in the log changes who someone is. The speech log grows with every line heard, so `TodayStores` also keeps
the lines it has read and reads only new ones until the log or the voice list changes, or a purge lands. On a made-up year
(180,000 lines and 5,000 visits) a full read took about 3 s, a read after new lines about 0.3 s, and a cached answer
well under a millisecond. Ask answers each question in a fresh worker process (memory/ask.py), so an Ask question
always pays the full read; the cache serves the dashboard process, the journal's day pages, and repeated reads in one process.

Switch: `timeline.enabled` (on by default). Off, nothing here runs, and "who was here" and summaries are exactly what
they were before step 7. "Last seen" is the old answer only while `detections.last_seen` is off too, or there is no
detections.sqlite3: history_qa's detector path reads live face matches whatever this switch says. Without an event
log there are no person ids, so `load` returns None and callers keep their old path too.
"""

import bisect
import collections
import dataclasses
import json
import math
import pathlib
import threading
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from liberational import config as cfg
from liberational import speech_log
from liberational.domain.kernel import PersonId
from liberational.domain.memory import Encounter, Modality
from liberational.intervals import interval_groups
from liberational.listen.revisions import NOT_SAID
from liberational.people import registry, transcript_view

SOURCES = ("playback_windows.json",)
PLAYBACK = NOT_SAID  # played back, synthetic, or a TV's: nobody here said it
BATCH = 5000  # transcript lines resolved at a time, so reading a year of them never holds it all as dicts


def enabled() -> bool:
    return bool(cfg.get("timeline.enabled"))


@dataclasses.dataclass(frozen=True, slots=True)
class Visit:
    """One face visit on one camera, as vision/sightings.py groups sightings. `name` is as stored on the visit."""

    name: str
    cam: str
    start: float
    end: float
    best_sim: float
    payload_ids: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class Line:
    """One heard transcript line: its time span and text. Who said it is the Observed person it belongs to."""

    start: float
    end: float
    text: str
    payload_ids: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class Observed:
    """One visit or line, and the person it resolves to."""

    person: PersonId
    evidence: Visit | Line

    @property
    def modality(self) -> Modality:
        return Modality.SEEN if isinstance(self.evidence, Visit) else Modality.HEARD


@dataclasses.dataclass(frozen=True, slots=True)
class Entry:
    """One encounter, with the visits and lines it was built from, oldest first. `encounter` is the domain record;
    it is made when asked for, since a year holds tens of thousands and a question reads a few."""

    person: PersonId
    start: float
    end: float
    seen_or_heard: frozenset[Modality]
    visits: tuple[Visit, ...]
    lines: tuple[Line, ...]

    @property
    def encounter(self) -> Encounter:
        return Encounter(self.person, self.start, self.end, self.seen_or_heard)


@dataclasses.dataclass(frozen=True, slots=True)
class Present:
    """One person with at least one encounter overlapping a time range, and those encounters."""

    person: PersonId
    name: str
    seen_or_heard: frozenset[Modality]
    entries: tuple[Entry, ...]


def payload_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Canonical existing payload identities, never display labels or invented projection identities."""
    many = record.get("payload_ids")
    candidates = [record.get("id"), record.get("payload_id")]
    if isinstance(many, list | tuple):
        candidates.extend(many)
    found: list[str] = []
    for value in candidates:
        if not isinstance(value, str):
            continue
        try:
            canonical = str(uuid.UUID(value))
        except ValueError:
            continue
        if value == canonical and value not in found:
            if value == record.get("id"):
                # Raw log rows get this identity from their header. Body fields cannot add fake sources.
                return (value,)
            found.append(value)
    return tuple(found)


def project(observed: Iterable[Observed], gap: float) -> list[Entry]:
    """Encounters from visits and lines: per person, in time order, joined while the gap is at most `gap` seconds.
    Sorted by start."""
    by_person: dict[PersonId, list[Observed]] = collections.defaultdict(list)
    for o in observed:
        by_person[o.person].append(o)
    out: list[Entry] = []
    for person, items in by_person.items():
        items.sort(key=lambda o: (o.evidence.start, o.evidence.end))
        for group in interval_groups(items, gap, start=lambda o: o.evidence.start, end=lambda o: o.evidence.end):
            out.append(_entry(person, group))
    out.sort(key=lambda e: (e.start, e.end, e.person))
    return out


def _entry(person: PersonId, group: list[Observed]) -> Entry:
    return Entry(
        person,
        min(o.evidence.start for o in group),
        max(o.evidence.end for o in group),
        frozenset(o.modality for o in group),
        tuple(o.evidence for o in group if isinstance(o.evidence, Visit)),
        tuple(o.evidence for o in group if isinstance(o.evidence, Line)),
    )


class Timeline:
    """Every encounter, by person and by time. Built by `load`; answers without reading any file."""

    def __init__(self, entries: Iterable[Entry], names: Mapping[PersonId, str]) -> None:
        self.entries = sorted(entries, key=lambda e: (e.start, e.end, e.person))
        self.names = dict(names)
        self._starts = [e.start for e in self.entries]
        self._longest = max((e.end - e.start for e in self.entries), default=0.0)

    def name(self, person: PersonId) -> str:
        return self.names.get(person, person)

    def encounters(self, person: PersonId | None = None, start: float = 0.0, end: float = math.inf) -> list[Entry]:
        """Encounters overlapping [start, end], of one person or everyone, oldest first."""
        # An encounter that overlaps starts no later than `end`, and no earlier than the longest one before `start`.
        lo = bisect.bisect_left(self._starts, start - self._longest)
        hi = bisect.bisect_right(self._starts, end)
        return [e for e in self.entries[lo:hi] if e.end >= start and (person is None or e.person == person)]

    def last(self, person: PersonId, start: float = 0.0, end: float = math.inf) -> Entry | None:
        """The encounter of `person` overlapping [start, end] that ended last, or None."""
        found = self.encounters(person, start, end)
        return max(found, key=lambda e: e.end) if found else None

    def present(self, start: float, end: float) -> list[Present]:
        """Who was present in [start, end], in the order they first appear: everyone with an encounter overlapping
        it, with its encounters. Seen, heard, or both comes from the part inside the range (a visit that overlaps
        it, a line that starts in it), since an encounter can run past either end. Someone with neither there (the
        range falls in a gap of their encounter) is left out."""
        by_person: dict[PersonId, list[Entry]] = {}
        for e in self.encounters(None, start, end):
            by_person.setdefault(e.person, []).append(e)
        out = []
        for pid, entries in by_person.items():
            how = frozenset(m for e in entries for m in within(e, start, end))
            if how:
                out.append(Present(pid, self.name(pid), how, tuple(entries)))
        return out


def within(entry: Entry, start: float, end: float) -> frozenset[Modality]:
    """How a person was met in the part of an encounter inside [start, end]: seen when a visit overlaps it, heard
    when a line starts in it (the lines history answers cite)."""
    seen = any(v.end >= start and v.start <= end for v in entry.visits)
    heard = any(start <= ln.start <= end for ln in entry.lines)
    return frozenset(({Modality.SEEN} if seen else set()) | ({Modality.HEARD} if heard else set()))


class TodayStores:
    """Layer 1 as it's held today: face visits from the vision log, and utterances from the speech log on
    Continuum (liberational/speech_log.py), read through people/transcript_view.View. Face detections are not in the
    event log yet; when they are, an adapter that reads them from the log replaces this one and nothing else changes.

    It keeps the lines it has read, and who said each, between reads. While the event log and the voices and forwards
    in the voice list stay the same, a read takes only the lines committed since the last one (`poll`), so a growing
    speech log costs little. When any of those change, or a purge lands (forgetting a person purges their lines),
    every line is read again."""

    def __init__(self, state: pathlib.Path) -> None:
        self.state = state
        self._seq = 0  # the speech log's commit sequence read up to
        self._purge: str | None = None  # the newest purge seen when the lines were read
        self._pending: list[dict[str, Any]] = []  # lines polled but not resolved yet
        self._key: object = None  # what the people of those lines were resolved against
        self._heard: list[tuple[PersonId, Line]] = []
        self._seen: list[dict[str, Any]] = []  # the face visits read at the last poll

    def poll(self) -> tuple[int, str | None, tuple[object, ...]]:
        """Takes the lines committed since the last poll, starting over after a new purge, and every face visit, and
        returns where the speech log stands (the newest commit sequence and purge) and what the visits are (how many,
        and the newest), for the cache's stamp."""
        from liberational.vision import sightings

        self._seen = sightings.visits_between(0.0, None, self.state)
        newest = max(((v["end"], v["start"], str(v["cam"]), str(v["name"])) for v in self._seen), default=None)
        log = speech_log.shared()
        purge = log.newest_purge(self._purge)
        if purge != self._purge:
            self._seq, self._purge, self._pending, self._heard, self._key = 0, purge, [], [], None
        lines, self._seq = log.since(self._seq)
        self._pending += lines
        return self._seq, self._purge, (len(self._seen), newest)

    def read(self) -> tuple[list[Observed], dict[PersonId, str]] | None:
        """(every visit and line that resolves to a person, the display name of every person), or None without an
        event log."""
        if not registry.log_exists(self.state):
            return None  # nothing labeled since step 3, so no person ids; and a reader never creates the log
        from liberational.listen import revisions

        log = (registry.at(self.state).mark(), revisions.review_revision(self.state))
        view = transcript_view.View(self.state)
        if view.reg is None:
            return None
        lines = self._lines(view, log)
        windows = self._playback()
        if windows is None:
            lines = []
        elif windows:  # heard while the pack was speaking: history_qa leaves these out too
            lo, hi = min(w[0] for w in windows), max(w[1] for w in windows)
            lines = [
                (p, ln)
                for p, ln in lines
                if ln.end < lo or ln.start > hi or not any(ln.start <= b and ln.end >= a for a, b in windows)
            ]
        observed = self._visits(view.reg) + [Observed(p, ln) for p, ln in lines]
        return observed, {p.id: p.display_name for p in view.reg.people()}

    def _visits(self, reg: registry.Registry) -> list[Observed]:
        known: dict[str, PersonId | None] = {}
        out = []
        for v in self._seen:
            name, start, end = v["name"], _instant(v["start"]), _instant(v["end"])
            if not registry.is_name(name) or start is None or end is None or end < start:
                continue
            if name not in known:
                known[name] = reg.known_as(name)
            if (pid := known[name]) is not None:
                out.append(
                    Observed(pid, Visit(name, str(v["cam"]), start, end, float(v["best_sim"] or 0.0), payload_ids(v)))
                )
        return out

    def _lines(self, view: transcript_view.View, log: object) -> list[tuple[PersonId, Line]]:
        """Every heard line that names a person, and who. Only the lines polled since the last read are resolved while
        what they were resolved against is the same: the event log's files as they were before `view` read them, and
        the voices and their names `view` read."""
        key = (log, tuple(view.names.items()))
        if key != self._key:
            self._heard, self._pending = [], speech_log.shared().since(0)[0]
        pending, self._pending = self._pending, []
        for i in range(0, len(pending), BATCH):
            self._heard += self._resolve(view, pending[i : i + BATCH])
        self._key = key
        return self._heard

    @staticmethod
    def _resolve(view: transcript_view.View, lines: list[dict[str, Any]]) -> list[tuple[PersonId, Line]]:
        from liberational.listen import revisions

        heard, spans = [], []
        for e in revisions.filter_lines(view.state, lines):
            if e.get("assistant") or any(e.get(k) for k in PLAYBACK):
                continue
            start, end = _instant(e.get("ts_start")), _instant(e.get("ts_end"))
            if start is None:
                continue
            heard.append(e)
            spans.append((start, end if end is not None and end >= start else start))
        out = []
        for (e, pid), (start, end) in zip(view.speakers(heard), spans):
            if pid is not None:
                out.append((pid, Line(start, end, str(e.get("text", "")), payload_ids(e))))
        return out

    def _playback(self) -> list[tuple[float, float]] | None:
        """When the pack was speaking (speech/playback_guard.py), or None when the file can't be read, which leaves
        every line out, the way playback_guard.overlaps does."""
        try:
            rows = json.loads((self.state / "playback_windows.json").read_text())
            return [(float(r["start"]), float(r["end"])) for r in rows]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, TypeError, KeyError) as e:
            registry.LOG.warning(f"timeline: playback windows unreadable, so no line counts as heard: {e!r}")
            return None


def _instant(t: object) -> float | None:
    """`t` as an instant an Encounter accepts (epoch seconds), else None."""
    if isinstance(t, bool) or not isinstance(t, int | float) or not 0 <= t <= 1e12:
        return None  # not a number, not finite, or milliseconds: the bounds kernel.check_instant uses
    return float(t)


def _stat(path: pathlib.Path) -> tuple[int, int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_ino, st.st_mtime_ns, st.st_size


def _signature(state: pathlib.Path) -> tuple[object, ...]:
    """The files' sizes and times, and where the event log stands (a label given only in the log changes who someone
    is), so the cache refreshes on either."""
    log = registry.at(state).mark() if registry.log_exists(state) else None
    from liberational.listen import revisions

    return (*(_stat(state / n) for n in SOURCES), (log, revisions.review_revision(state)))


_cache: dict[str, tuple[object, Timeline | None]] = {}
_stores: dict[str, TodayStores] = {}
_lock = threading.Lock()


def load(state: str | pathlib.Path | None = None) -> Timeline | None:
    """The timeline for `state` (paths.state by default), rebuilt only when a source changed. None without an event
    log, or when it can't be read (logged), so a caller keeps its old path."""
    state = pathlib.Path(state if state is not None else cfg.path("paths.state"))
    with _lock:  # one build at a time; a second caller waits and takes the first one's result
        if len(_cache) > 8:
            _cache.clear()
            _stores.clear()
        stores = _stores.setdefault(str(state), TodayStores(state))
        built = None
        try:
            stamp = (_signature(state), stores.poll(), cfg.nonnegative_seconds("timeline.gap_s"))
            hit = _cache.get(str(state))
            if hit is not None and hit[0] == stamp:
                return hit[1]
            read = stores.read()
            if read is not None:
                built = Timeline(project(read[0], stamp[2]), read[1])
        except Exception as e:
            registry.LOG.warning(f"timeline: not built, so answers take their old path: {e!r}")
            _stores.pop(str(state), None)  # start over next time
            return None
        # Cached under the files as they were before reading, so a change made meanwhile is read next time.
        _cache[str(state)] = (stamp, built)
        return built


def person(name: str, state: str | pathlib.Path | None = None) -> PersonId | None:
    """The person a name asked about is (the registry's `known_as`), or None. Names reach here only from a question."""
    state = pathlib.Path(state if state is not None else cfg.path("paths.state"))
    if not registry.is_name(name) or not registry.log_exists(state):
        return None
    try:
        return registry.at(state).known_as(name)
    except Exception as e:
        registry.LOG.warning(f"timeline: {name!r} not looked up: {e!r}")
        return None


def encounters(
    who: PersonId | None = None, start: float = 0.0, end: float = math.inf, state: str | pathlib.Path | None = None
) -> list[Entry]:
    """Encounters overlapping [start, end], of one person or everyone, oldest first ([] without a timeline)."""
    tl = load(state)
    return tl.encounters(who, start, end) if tl is not None else []


def present(start: float, end: float, state: str | pathlib.Path | None = None) -> list[Present]:
    """Who was present in [start, end], seen, heard, or both ([] without a timeline)."""
    tl = load(state)
    return tl.present(start, end) if tl is not None else []
