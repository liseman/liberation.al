"""Who and what was around during one day, as time spans, for Review's chips (GET /review/presence?day=).

People come from the face worker's visits (vision/sightings.py, on the vision log) and named transcript lines;
objects from the caption model's grounding stored with each caption record (the Continuum vision log); notable events from the day's journal
page's linked bullets (memory/journal_pages.py), one chip per section, and from what the assistant said out loud. When enabled, anonymous per-camera
presence starts add event chips. Their short playback spans are context only, so total_s stays zero.

A chip is {"id", "kind": "person" | "object" | "event", "label", "spans": [[start, end], ...], "total_s", "count"};
event chips also carry "notes", one per span. Named person chips carry "speech" separately from their combined
"spans", so a camera filter keeps utterances even when they overlap a face visit. A person chip seen on recorded
video also carries "cams": {"cam0":
[[start, end], ...]}, when each camera view saw them (the cameras are views of one rig, not rooms). Faces
nobody has named make one more person chip, "person:unknown" ("Unknown faces"), with the same "cams". The result is
cached until one of the sources changes, and for a day still going, for at most CAPTION_REFRESH_S (new captions).

An ended day's answer is kept until something about that day changes: its segment files (an hour either side,
as stat()s, so a late row or a purge shows), the deletions log, the speech purges, the registry, the transcript
review, and the journal's pages. A line heard today doesn't make yesterday's history be worked out again, and the
movement read that dominates a load (8 to 12 s) is skipped while it holds. Review gets `for_page`'s answer: the chips
without their evidence and source ids, and movement as its counts; the 25 MB of tracks, barriers, and unknown spans
stay with presence()'s other readers (Ask).
"""

import functools
import math
import multiprocessing
import pathlib
import re
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import continuum_ready, logs, speech_log, vision_log
from liberational.dashboard import routes
from liberational.intervals import interval_groups
from liberational.memory import presence_counts
from liberational.people import registry, transcript_view
from liberational.see.faces import VISIT_GAP_S
from liberational.vision import sightings

if TYPE_CHECKING:
    from liberational.dashboard.server import H

OBJECT_PAD = 2.0  # an observation covers about this long
EVENT_PAD = 30.0  # a summary bullet that links a single moment
CAPTION_REFRESH_S = 60  # how long a day still going keeps its cached chips, for the captions since
MIN_GROUNDINGS = 2  # objects seen in fewer separate groundings are left out
KINDS = ("person", "object", "event")
SYNONYMS = {"television": "tv", "tv set": "tv", "couch": "sofa", "cellphone": "phone", "cell phone": "phone"}
NOT_PEOPLE = re.compile(r"(speaker \d+|unknown.*|multiple speakers?)?$", re.I)
# Summary sections that aren't events: names (people already have their own chips), and the sections the summarizer
# appends itself (sampled captions, what the summary covers, and the walk-by counts).
SKIP_SECTIONS = {"people present", "counts", "evidence"}  # a day page's sections code writes, not events
LINK = re.compile(r"\[[^\]]*\]\(#t=([\d.]+)\)")
SPAN_LINK = re.compile(r"\[[^\]]*\]\(#t=([\d.]+)\)\s*-\s*\[[^\]]*\]\(#t=([\d.]+)\)")


def day_bounds(day: str) -> tuple[float, float]:
    """[start, end) of a local calendar day 'YYYY-MM-DD' as epoch seconds (DST days are 23 or 25 hours)."""
    t = time.strptime(day, "%Y-%m-%d")
    start = time.mktime(t)
    nxt = time.mktime((t.tm_year, t.tm_mon, t.tm_mday + 1, 0, 0, 0, 0, 0, -1))
    return start, nxt


def merge(spans: Iterable[tuple[float, float]], gap: float) -> list[list[float]]:
    """Sorted spans with any closer than gap seconds joined."""
    return [
        [group[0][0], max(span[1] for span in group)]
        for group in interval_groups(sorted(spans), gap, start=lambda span: span[0], end=lambda span: span[1])
    ]


# Words that end in "s" without being plurals, so the singular rule leaves them alone ("news" is not "new").
NOT_PLURAL = frozenset({"news", "lens", "canvas", "gas", "series", "species", "atlas", "iris", "chaos", "yes"})


def object_label(raw: object) -> str:
    """One name per kind of thing: lowercase, spaces, singular, and common synonyms folded together."""
    s = re.sub(r"[\s_-]+", " ", str(raw or "")).strip().lower()
    if not s:
        return ""
    words = s.split()
    w = words[-1]
    if len(w) > 2 and w not in NOT_PLURAL and not w.endswith(("ss", "us", "is")):
        if w.endswith("ies"):
            w = w[:-3] + "y"
        elif w.endswith(("ches", "shes", "xes", "sses", "lenses")):
            w = w[:-2]
        elif w.endswith("s"):
            w = w[:-1]
    s = " ".join(words[:-1] + [w])
    return SYNONYMS.get(s, s)


UNKNOWN_FACES = "person:unknown"  # the chip for faces nobody has named


def _chip(
    kind: str,
    label: str,
    spans: Sequence[Sequence[float]],
    notes: list[str] | None = None,
    cams: dict[str, list[list[float]]] | None = None,
    chip_id: str | None = None,
    speech: list[list[float]] | None = None,
) -> dict[str, Any]:
    c = {
        "id": chip_id or f"{kind}:{label.lower()}",
        "kind": kind,
        "label": label,
        "spans": [[round(s, 3), round(e, 3)] for s, e in spans],
        "total_s": round(sum(e - s for s, e in spans), 1),
        "count": len(spans),
    }
    if notes is not None:
        c["notes"] = notes
    if speech is not None:
        c["speech"] = [[round(a, 3), round(b, 3)] for a, b in speech]
    if cams:
        c["cams"] = {cam: [[round(a, 3), round(b, 3)] for a, b in v] for cam, v in sorted(cams.items())}
    return c


def people(state: pathlib.Path, start: float, end: float) -> list[dict[str, Any]]:
    spans, labels = {}, {}
    spoken: dict[str, list[tuple[float, float]]] = {}
    seen: dict[str, dict[str, list[tuple[float, float]]]] = {}  # who each camera view saw, from the visits
    view = transcript_view.View(state)  # who each line and visit is now, through the event log

    def add(name: str | None, s: float, e: float) -> tuple[str, float, float] | None:
        if not name or NOT_PEOPLE.match(name.strip()):
            return
        k = name.strip().lower()
        labels.setdefault(k, name.strip())
        s = max(s, start)
        e = max(min(e, end), s + 1)
        spans.setdefault(k, []).append((s, e))
        return k, s, e

    for v in sightings.visits_between(start, end, state):
        if v["start"] >= end:
            continue
        name, cam, s, e = v["name"], v["cam"], v["start"], v["end"]
        who = view.current(name) if isinstance(name, str) else name  # a rename since applies
        add(who, s, e)
        if who and not NOT_PEOPLE.match(who.strip()):
            a = max(s, start)
            seen.setdefault(who.strip().lower(), {}).setdefault(str(cam), []).append((a, max(min(e, end), a + 1)))
    for e in view.named(speech_log.shared().range(start, end)):  # who said it, by their name now
        if e.get("assistant"):
            continue
        if heard := add(e.get("speaker"), e["ts_start"], e.get("ts_end") or e["ts_start"] + 1):
            name, a, b = heard
            spoken.setdefault(name, []).append((a, b))
    gap = cfg.nonnegative_seconds("dashboard.person_gap_s")
    chips = [
        _chip(
            "person",
            labels[k],
            merge(v, gap),
            cams={cam: merge(w, gap) for cam, w in seen.get(k, {}).items()},
            speech=merge(spoken.get(k, []), gap),
        )
        for k, v in spans.items()
    ]
    unknown = unknown_faces(state, start, end)
    return chips + ([unknown] if unknown else [])


def unknown_faces(
    state: pathlib.Path,
    start: float,
    end: float,
    *,
    unidentified_tracks_only: bool = False,
    observed: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The "Unknown faces" chip: when each camera view saw a face nobody has named, from the face worker's sightings
    (visits only hold named people), grouped per camera with the visit gap. None when there were none. `observed`:
    the window's sightings when the caller has them (sightings.faces otherwise)."""
    if observed is None:
        observed = sightings.faces(start, end, state=state)
    if unidentified_tracks_only:
        # Arrival is a physical-track event, not a second arrival whenever a name flickers off. Keep uncertainty
        # in Review, but exclude anonymous arrivals for the same identified track run. A reset epoch, another
        # camera/track, or a gap beyond the visit continuity limit remains a distinct run.
        tracked: dict[tuple[str, str], list[dict[str, Any]]] = {}
        remaining = []
        for face in observed:
            track = face.get("track_id")
            if track:
                tracked.setdefault((str(face["cam"]), str(track)), []).append(face)
            else:
                remaining.append(face)
        for group in tracked.values():
            for run in interval_groups(
                sorted(group, key=lambda f: float(f["ts"])),
                VISIT_GAP_S,
                start=lambda f: float(f["ts"]),
                end=lambda f: float(f["ts"]),
            ):
                if not any(f.get("name") for f in run):
                    remaining.extend(run)
        observed = remaining
    by_cam: dict[str, list[float]] = {}
    for f in observed:
        if f["name"] is None:
            by_cam.setdefault(str(f["cam"]), []).append(float(f["ts"]))
    if not by_cam:
        return None
    gap = VISIT_GAP_S
    cams = {
        cam: [[g[0], max(g[-1], g[0] + 1)] for g in interval_groups(sorted(ts), gap, start=float, end=float)]
        for cam, ts in by_cam.items()
    }
    spans = merge(((a, b) for v in cams.values() for a, b in v), cfg.nonnegative_seconds("dashboard.person_gap_s"))
    return _chip("person", "Unknown faces", spans, cams=cams, chip_id=UNKNOWN_FACES)


def objects(state: pathlib.Path, start: float, end: float) -> list[dict[str, Any]]:
    seen, groundings = {}, {}
    try:
        rows = [
            (r["ts"], r.get("objects")) for r in vision_log.at(state).captions(start, end) if start <= r["ts"] < end
        ]
    except OSError:  # the broker is unreachable
        rows = []
    for ts, items in rows:
        for o in items if isinstance(items, list) else ():
            label = object_label(o.get("label") if isinstance(o, dict) else o)
            if not label:
                continue
            seen.setdefault(label, []).append((ts, ts + OBJECT_PAD))
            groundings.setdefault(label, set()).add(o.get("ts", ts) if isinstance(o, dict) else ts)
    return [
        _chip("object", label, merge(v, cfg.nonnegative_seconds("dashboard.object_gap_s")))
        for label, v in seen.items()
        if len(groundings[label]) >= MIN_GROUNDINGS
    ]


def events(state: pathlib.Path, day: str, start: float, end: float) -> list[dict[str, Any]]:
    """The Review chips for what happened: one per section of the day's journal page (memory/journal_pages.py, its
    "## section" bullets with their citations' time links), and one for what the assistant said."""
    from liberational import memory_log
    from liberational.memory import journal_pages

    out = []
    page = memory_log.at(state).page(journal_pages.DAY, day)
    md = journal_pages.markdown(page, state) if page else ""
    section, bullets = None, {}
    for ln in md.splitlines():
        if ln.startswith("## "):
            section = ln[3:].strip()
            if section.lower() in SKIP_SECTIONS:
                section = None
            continue
        if not section or not ln.lstrip().startswith("- "):
            continue
        m = SPAN_LINK.search(ln) or LINK.search(ln)
        if not m:
            continue
        a = float(m.group(1))
        b = float(m.group(2)) if m.re is SPAN_LINK else a + EVENT_PAD
        if not start <= a < end:
            continue
        note = LINK.sub(lambda x: "", ln.lstrip()[2:]).strip(" -:").strip()
        note = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s+", " ", note)).replace("**", "")
        bullets.setdefault(section, []).append((a, max(b, a + 1), note[:160]))
    for section, items in bullets.items():
        items.sort()
        label = re.split(r"\s*[&/]\s*", section)[0].strip() or section
        out.append(_chip("event", label, [(a, b) for a, b, _ in items], [n for *_, n in items]))
    spoke = sorted(
        (e["ts_start"], max(e.get("ts_end") or 0, e["ts_start"] + 1), (e.get("text") or "")[:160])
        for e in speech_log.shared().range(start, end, [speech_log.REPLIES])
        if not e.get("status") and e.get("text")
    )
    if spoke:
        out.append(
            _chip("event", f"{cfg.assistant_name()} spoke", [(a, b) for a, b, _ in spoke], [n for *_, n in spoke])
        )
    return out


def _ended_stamp(state: pathlib.Path, day: str, start: float, end: float) -> tuple[Any, ...]:
    """What an ended day's history depends on (see the module's docs): its sources without the speech log's
    position, which every line heard since moves, and its segment files."""
    from liberational import segment_log

    return (
        tuple(s for s in _sources(state, day) if s[0] != "speech"),
        segment_log.at(state).stamp(start, end),
        speech_log.shared().stamp(start, end),
    )


def _sources(state: pathlib.Path, day: str) -> tuple[tuple[str, int | None, int | None], ...]:
    from liberational import memory_log
    from liberational.listen import revisions

    # A new journal page (or a purge of one) moves the memory log's mark on its stream.
    out: list[tuple[str, int | None, int | None]] = [
        ("journal.page", memory_log.at(state).mark((memory_log.PAGES,)), None)
    ]
    # A line heard or said, or a purge, moves the speech log's mark.
    seq, purge = speech_log.shared().mark((speech_log.UTTERANCES, speech_log.REPLIES))
    out.append(("speech", seq, None))
    out.append(("speech-purge", None, None) if purge is None else ("speech-purge:" + purge, None, None))
    # A label given only in the event log changes who someone is: where the registry's streams stand says so.
    mark = registry.at(state).mark() if registry.log_exists(state) else ()
    out.append(("events", max(mark, default=None), None))
    out.append(("transcript-review", revisions.review_revision(state), None))
    return tuple(out)


def sampled_presence(
    state: pathlib.Path, start: float, end: float, context: float, result: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Per-camera start markers; their playback padding is not occupancy duration. result: collect's answer for the
    window, when the caller has it (an ended day's, from its job)."""
    if result is None:
        result = presence_counts.collect(state, start=start, end=end)
    if result.get("error"):
        raise presence_counts.Unavailable("Presence history unavailable", pending=bool(result.get("retryable")))
    chips = []
    for camera, observed in sorted(result["cameras"].items()):
        episodes = [episode for episode in observed["episodes"] if episode["counted"]]
        if not observed["count"] or not episodes:
            continue
        notes = []
        for episode in episodes:
            note = "Sampled presence only; not a physical-entry or unique-person count."
            if observed["unknown_seconds"]:
                note += f" Unknown coverage: {observed['unknown_seconds']:.1f} s."
            if episode["left_censored"]:
                note += " Start boundary uncertain."
            if episode["right_censored"]:
                note += " End boundary uncertain."
            notes.append(note)
        chips.append(
            dict(
                id=f"presence:{camera}",
                camera=camera,
                kind="event",
                label=f"{camera} came into view",
                spans=[[episode["start"], min(episode["start"] + context, end)] for episode in episodes],
                total_s=0,
                count=observed["count"],
                notes=notes,
            )
        )
    return chips


def tracked_movement(result: dict[str, Any], end: float) -> list[dict[str, Any]]:
    """One plain chip per camera for tracked movement: every track's supported intervals joined into the
    stretches someone was in view, with all the tracks' source ids and edge evidence, so a moment still plays.
    Entries and exits get no chip of their own (the camera's "came into view" chip already says that), and nearby
    arrival times get none either. Presentation only: the counts and records behind it are unchanged."""
    if not result.get("available"):
        return []
    chips = []
    for camera, observed in sorted(result.get("cameras", {}).items()):
        if not observed.get("available"):
            continue
        tracks = [track for track in observed.get("tracks", ()) if track["supported_s"] > 0]
        intervals = sorted(
            (float(lo), min(end, float(hi)))
            for track in tracks
            for lo, hi in track["supported_intervals"]
            if min(end, float(hi)) > float(lo)
        )
        if not intervals:
            continue
        stretches = [
            [group[0][0], max(hi for _, hi in group)]
            for group in interval_groups(intervals, 0, start=lambda span: span[0], end=lambda span: span[1])
        ]
        note = (
            "When the camera followed someone moving in the picture. Stretches that overlap are joined, so this "
            "is not a count of different people."
        )
        if observed["partial"]:
            note += " Some of the day wasn't fully covered, so there may be more."
        chips.append(
            dict(
                id=f"movement:visible:{camera}",
                camera=camera,
                kind="event",
                label=f"Someone in view on {camera}",
                spans=stretches,
                total_s=round(sum(hi - lo for lo, hi in stretches), 1),
                count=len(stretches),
                notes=[note],
                source_ids=sorted({source for track in tracks for source in track["source_ids"]}),
                evidence=[*observed.get("entries", ()), *observed.get("exits", ())],
            )
        )
    return chips


_cache: dict[
    tuple[str, str, float, float, bool, float | None, float | None, float | None],
    tuple[object, dict[str, Any]],
] = {}


def presence(day: str | None = None, state: str | pathlib.Path | None = None) -> dict[str, Any]:
    """{"day", "start", "end", "chips": [...]} for one local day, people first, then objects, then events, each kind
    by most time present."""
    state = pathlib.Path(state or cfg.path("paths.state"))
    day = day or time.strftime("%Y-%m-%d")
    start, end = day_bounds(day)
    enabled = presence_counts.enabled()
    settings = presence_counts.settings() if enabled else None
    context = cfg.get("presence.chip_context_s") if enabled else None
    if enabled and (
        isinstance(context, bool) or not isinstance(context, int | float) or not math.isfinite(context) or context <= 0
    ):
        raise ValueError("presence.chip_context_s must be finite and positive")
    context = float(context) if context is not None else None
    key = (
        str(state),
        day,
        cfg.nonnegative_seconds("dashboard.person_gap_s"),
        cfg.nonnegative_seconds("dashboard.object_gap_s"),
        enabled,
        settings["gap"] if settings is not None else None,
        settings["absence"] if settings is not None else None,
        context,
    )
    measured: dict[str, Any] | None = None
    ended = end <= time.time()
    if ended:
        # Nothing about an ended day changed: no reading at all, the movement read included. A stamp that
        # can't be read is no stamp: the day is worked out as before, and its answer isn't kept under it.
        try:
            stamp: Any = ("ended", _ended_stamp(state, day, start, end))
        except Exception:
            stamp = None
        hit = _cache.get(key)
        if stamp is not None and hit and hit[0] == stamp:
            return hit[1]
    try:
        # Refresh the separate movement source even for an ended day's cached episode/chip answer.
        measured = presence_counts.collect(state, start=start, end=end) if enabled else None
        tracked = (measured or {}).get("movement")
        # today's captions and sampled presence move with the logs, which have no file to stat
        if not ended:
            live = int(time.time() // CAPTION_REFRESH_S)
            stamp = (_sources(state, day), live, tracked)
            hit = _cache.get(key)
            if hit and hit[0] == stamp:
                return hit[1]
        chips = people(state, start, end) + objects(state, start, end) + events(state, day, start, end)
        chips.sort(key=lambda c: (KINDS.index(c["kind"]), -c["total_s"], c["label"]))
        if context is not None:
            chips.extend(sampled_presence(state, start, end, context, result=measured))
            chips.extend(tracked_movement(tracked or {}, end))
    except presence_counts.Unavailable as exc:
        return dict(
            day=day, start=start, end=end, chips=[], partial=True, pending=exc.pending, error=type(exc).__name__
        )
    res = {"day": day, "start": start, "end": end, "chips": chips}
    if measured is not None:
        res["movement"] = measured["movement"]
    if len(_cache) > 16:
        _cache.clear()
    if stamp is not None:
        _cache[key] = (stamp, res)
    return res


@dataclass
class _Request:
    future: Future[dict[str, Any]]
    completed_at: float | None = None
    previous: dict[str, Any] | None = None


_requests: dict[tuple[Any, ...], _Request] = {}
_request_lock = threading.Lock()
_executor: Executor | None = None  # made on first use (_pool)


def _pool() -> Executor:
    """One process of its own for day loads (under _request_lock). A load of today is about 10 s of pure Python
    once a minute, and run on a dashboard thread it held the interpreter lock that every request needs, so each one
    waited behind it."""
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    return _executor


def _submit(load: functools.partial[dict[str, Any]]) -> Future[dict[str, Any]]:
    """Submit a day load (under _request_lock). A process that died while idle leaves the pool broken, and its
    next submit raises at once: that pool is dropped and the load goes to a new one."""
    global _executor
    try:
        return _pool().submit(load)
    except BrokenProcessPool as error:
        _LOG.warning(f"Review history process had stopped; starting a new one: {error}")
        _executor = None
        return _pool().submit(load)


_MOVEMENT_COUNTS = (
    "available",
    "entry_count",
    "exit_count",
    "coarrivals",
    "supported_s",
    "unknown_seconds",
    "partial",
    "errors",
)
_NOT_FOR_PAGE = ("source_ids", "evidence")  # Ask's and the summary's, not Review's


def for_page(res: dict[str, Any]) -> dict[str, Any]:
    """The day's answer as Review reads it: the chips without their evidence and source ids, and movement as
    each camera's counts. The tracks, barriers, and unknown spans were 25 MB a day, sent to a page that never reads
    them, and pickled across processes every minute."""
    out = dict(res)
    out["chips"] = [{k: v for k, v in c.items() if k not in _NOT_FOR_PAGE} for c in res.get("chips", [])]
    movement = res.get("movement")
    if isinstance(movement, dict):
        out["movement"] = {
            **{k: v for k, v in movement.items() if k != "cameras"},
            "cameras": {
                cam: {k: v for k, v in counts.items() if k in _MOVEMENT_COUNTS}
                for cam, counts in (movement.get("cameras") or {}).items()
                if isinstance(counts, dict)
            },
        }
    return out


def _load(day: str, state: str) -> dict[str, Any]:
    """One day's answer, for for_request's job; in its own process, the settings as they are now."""
    if multiprocessing.parent_process() is not None:
        cfg.load.cache_clear()  # a setting saved since: the dashboard rereads its pack.toml, and so does this
        # Today is read again every minute: each ended hour's records are kept in this process, and only the hours
        # still going are read from the store.
        from liberational import memtrim, segment_log

        segment_log.keep_hours()
        memtrim.start()  # what a load freed goes back to the system; the kept hours stay
    start, end = day_bounds(day)
    pending = dict(day=day, start=start, end=end, chips=[], partial=True, pending=True)
    try:
        started = time.monotonic()
        result = presence(day, state)
        elapsed = time.monotonic() - started
        if elapsed > 5 and not result.get("error") and not result.get("pending"):
            _LOG.info(f"Review history for {day} loaded in {elapsed:.1f} s")
        return for_page(result)
    except Exception as exc:
        if continuum_ready.transient(exc):
            _LOG.warning("Review history waiting for the broker: %s", type(exc).__name__)
            return pending
        _LOG.exception(f"Review history failed: {type(exc).__name__}: {exc}")
        return dict(day=day, start=start, end=end, chips=[], partial=True, pending=False, error=type(exc).__name__)


def _finished(job: _Request, day: str) -> dict[str, Any]:
    """A finished job's answer. One whose process died or couldn't run it is an error, retried on the next poll in a
    new process."""
    global _executor
    error = job.future.exception()
    if error is None:
        return job.future.result()
    if isinstance(error, BrokenProcessPool):
        _executor = None
    _LOG.warning(f"Review history load stopped: {type(error).__name__}: {error}")
    start, end = day_bounds(day)
    return dict(day=day, start=start, end=end, chips=[], partial=True, pending=False, error=type(error).__name__)


def for_request(
    day: str | None = None, state: str | pathlib.Path | None = None, max_age_s: float = CAPTION_REFRESH_S
) -> dict[str, Any]:
    """Start or poll one background day load without waiting on it. Only successful loads
    get the refresh interval (a load older than `max_age_s` starts the next): startup stays pending and a failed load
    retries on the next poll, one job at a time."""
    state = pathlib.Path(state or cfg.path("paths.state"))
    day = day or time.strftime("%Y-%m-%d")
    start, end = day_bounds(day)
    key = (
        str(state),
        day,
        cfg.get("presence.enabled"),
        cfg.get("presence.gap_s"),
        cfg.get("presence.absence_s"),
        cfg.get("presence.chip_context_s"),
        cfg.get("dashboard.person_gap_s"),
        cfg.get("dashboard.object_gap_s"),
    )
    pending = dict(day=day, start=start, end=end, chips=[], partial=True, pending=True)
    now = time.monotonic()
    with _request_lock:
        job = _requests.get(key)
        previous = None
        if job is not None:
            if not job.future.done():
                return job.previous or pending
            previous = _finished(job, day)
            if (
                not previous.get("error")
                and not previous.get("pending")
                and now - (job.completed_at if job.completed_at is not None else now) < max_age_s
            ):
                return previous
        if len(_requests) >= 16 and key not in _requests:
            finished = next((k for k, value in _requests.items() if value.future.done()), None)
            if finished is None:
                return pending
            del _requests[finished]
        previous = previous if previous and not previous.get("pending") else None
        job = _Request(_submit(functools.partial(_load, day, str(state))), previous=previous)
        job.future.add_done_callback(lambda _: setattr(job, "completed_at", time.monotonic()))
        _requests[key] = job
        return previous or pending


WARM_EVERY_S = 30.0  # how often the dashboard checks today's answer in the background
# How old today's answer may get while nobody asks. An open Review polls every minute (CHIPS_MS), and each poll keeps
# the minute's refresh; a load of today is about 10 s of CPU, so warming every minute regardless was a sixth of a core
# all day. A visit after a quiet spell gets this answer at once and the fresh one on its next poll.
IDLE_WARM_S = 600.0
_LOG = logs.get("dashboard")


def keep_warm(state: str | pathlib.Path | None = None, stop: threading.Event | None = None) -> threading.Thread:
    """Keep today's Review answer at most IDLE_WARM_S old in the background, so a visit finds it ready."""
    halt = stop or threading.Event()

    def run() -> None:
        while not halt.is_set():
            try:
                if presence_counts.enabled():
                    for_request(None, state, max_age_s=IDLE_WARM_S)
            except Exception as ex:  # keep trying: one bad pass mustn't stop the next
                _LOG.warning(f"presence: today's Review answer not brought up to date: {type(ex).__name__}: {ex}")
            halt.wait(WARM_EVERY_S)

    thread = threading.Thread(target=run, name="presence-warm", daemon=True)
    thread.start()
    return thread


# ---- routes (dashboard/routes.py): what the server answers for this area ----


@routes.get("/review/presence")
def _get_review_presence(h: H) -> None:
    day = h._query().get("day") or None
    if day and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        return h._json({"error": "day must be YYYY-MM-DD"}, 400)
    return h._json(for_request(day, h.state))
