"""Live's "What's happening": a plain log of the last few hours, newest at the bottom.

GET /live/happening?since=&limit= gives {"items": [...], "updated": cursor}, items in time order:

    {"id", "ts", "kind": "arrival" | "departure" | "talk", "text", "people": [names], "cam", "camera_name"}

- arrival and departure come from who the camera views saw (the face worker's visits on the vision log, and faces
  nobody has named): "Kara came in" when they show up after LEAVE_S unseen, "Kara left" once LEAVE_S have passed
  since their last sighting (the face worker watches the cameras live, so it has seen on past them). Faces nobody
  has named are "Someone not yet named".
- talk is a finished conversation (lines with gaps up to TALK_GAP_S, then that much quiet), told by the local model
  as one line with names: "Kara asked Luca about the couch" (memory/plain_text.py, named lines). Only known people
  are named; anyone else is "someone". Never first person: the owner is named like anyone. A line not written
  within HOLD_S goes out as "Kara and Luca talked", and the model's line replaces it when it comes.
- ts is when it happened, for showing; rows come late (a visit once it closes, the model), so the cursor is where
  each row was written instead: its position in the dashboard process's journal of rows. updated is the
  newest; a poll with since=updated gets every row written or revised since (a revision keeps its id, so the page
  replaces it), however late its ts. Without since, the log as it is now. With since, limit gives the oldest rows
  written since, and updated is the last of them, so a tab far behind catches up without skipping any.
- cam and camera_name: the camera view an arrival or departure was seen on and its name (vision.camera_names), else
  None.

Rows and the cursor live in the dashboard process (_Rows): a journal with one entry for each row written or changed,
numbered 1, 2, and on, `row` its id on the page, a revision naming the first entry of its row in `revises`. A
restart starts the journal and the cursor over: a tab whose cursor is past the new end gets the rows afresh, and the
model is asked again for the talk lines. The log is built at most once every BUILD_S seconds (every open Live tab
polls it), reads only the transcript's last LOOKBACK_S, and never waits on the model. Only the first build waits;
after that a poll answers from the last build while one background build at a time brings it up to date (a build
takes 1 to 5 s under load).
"""

import math
import pathlib
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from liberational import logs, speech_log
from liberational.dashboard import presence, routes
from liberational.intervals import interval_groups
from liberational.memory import plain_text
from liberational.people import registry, transcript_view
from liberational.vision import sightings, views

if TYPE_CHECKING:
    from liberational.dashboard.server import H

LOOKBACK_S = 6 * 3600
EDGE_S = 3600  # read this much before the window too, so what is cut at its start never reads as new
LEAVE_S = 120.0  # unseen this long on the camera views (scanned) is gone; back after it is a new arrival
TALK_GAP_S = 90.0
HOLD_S = 180.0
BUILD_S = 8.0
MAX_TALK_LINES, MAX_TALK_CHARS = 20, 1500
# Face sightings older than SETTLED_S are read again only every SETTLED_REFRESH_S, or at once when a purge, a hidden
# record, or a name given moves their key: rereading the window's seven hours each build was most of what the
# dashboard decoded at idle, about 190,000 records a minute with Live open.
SETTLED_S = 900.0
SETTLED_REFRESH_S = 300.0
_built: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_building: dict[str, threading.Thread] = {}  # the background build running for a state
_lock = threading.Lock()  # _built, _building, and a build's rows with their records, so a cursor never passes a row
_build_lock = threading.Lock()  # one build at a time: _record keeps the rows written so far
LOG = logs.get("dashboard")


# ---- reading ----


def _seen(
    state: pathlib.Path, start: float, end: float, view: transcript_view.View
) -> dict[str, tuple[str, bool, list]]:
    """{chip id: (who, unknown, [(start, end, cam)])}: each person's sightings on the camera views, by their name now
    (visits), and faces nobody has named."""
    out: dict[str, tuple[str, bool, list]] = {}
    for v in sightings.visits_between(start, end, state):
        if v["start"] >= end:
            continue
        who = view.current(v["name"]).strip() if isinstance(v["name"], str) else ""
        if who and not presence.NOT_PEOPLE.match(who):
            spans = out.setdefault(f"person:{who.lower()}", (who, False, []))[2]
            spans.append((float(v["start"]), float(v["end"]), str(v["cam"])))
    unknown = presence.unknown_faces(
        state, start, end, unidentified_tracks_only=True, observed=_faces(state, start, end)
    )
    if unknown:
        spans = [(float(a), float(b), cam) for cam, v in (unknown.get("cams") or {}).items() for a, b in v]
        out[presence.UNKNOWN_FACES] = ("Someone not yet named", True, spans)
    return out


@dataclass
class _Settled:
    """The face sightings of [start, split), read at `read_at` under `key`."""

    key: tuple[Any, ...]
    start: float
    split: float
    rows: list[dict[str, Any]]
    read_at: float


_settled: dict[str, _Settled] = {}


def _segment_revision(state: pathlib.Path) -> int | None:
    """The segment store's deletion revision (a forget moves it), or None when it can't be read."""
    try:
        from continuum_py import segment_revision

        from liberational import segment_log

        return int(segment_revision(segment_log.root(state)))
    except Exception:
        return None


def _faces(state: pathlib.Path, start: float, end: float) -> list[dict[str, Any]]:
    """The face sightings of [start, end) (sightings.faces): the part older than SETTLED_S kept between builds, the
    rest read each time. The kept part is read again after SETTLED_REFRESH_S, when the window starts before it, and
    when its key moves (a forget in the segment store, or the event log's names); with that key unknown, every build
    reads it all."""
    revision = _segment_revision(state)
    names = registry.at(state).mark() if registry.log_exists(state) else None
    key = (revision, names)
    held = _settled.get(str(state))
    if (
        held is None
        or revision is None
        or held.key != key
        or start < held.start
        or not held.read_at <= end < held.read_at + SETTLED_REFRESH_S
    ):
        split = max(start, end - SETTLED_S)
        held = _Settled(key, start, split, sightings.faces(start, split, state=state), end)
        _settled[str(state)] = held
    kept = [f for f in held.rows if float(f["ts"]) >= start]
    return kept + sightings.faces(held.split, end, state=state)


# ---- building ----
def _comings(state: pathlib.Path, start: float, now: float, view: transcript_view.View) -> list[dict[str, Any]]:
    """Arrivals and departures over [start, now]."""
    named, out = views.names(), []
    for chip, (who, unknown, spans) in _seen(state, start, now, view).items():
        for group in interval_groups(sorted(spans), LEAVE_S, start=lambda r: r[0], end=lambda r: r[1]):
            first, last = group[0], max(group, key=lambda r: r[1])
            base = {"people": [] if unknown else [who]}
            out.append(
                {
                    "id": f"a:{chip}:{first[0]:.0f}",
                    "ts": first[0],
                    "kind": "arrival",
                    "text": f"{who} came in",
                    "cam": first[2],
                    "camera_name": named.get(first[2]),
                    **base,
                }
            )
            if last[1] < now - LEAVE_S:  # the face worker saw on past them, without them
                out.append(
                    {
                        "id": f"d:{chip}:{last[1]:.0f}",
                        "ts": last[1],
                        "kind": "departure",
                        "text": f"{who} left",
                        "cam": last[2],
                        "camera_name": named.get(last[2]),
                        **base,
                    }
                )
    return out


def _fallback(names: list[str]) -> str:
    """The line a conversation gets until (or unless) the model tells it: who talked, without what about."""
    if not names:
        return "Someone talked"
    if len(names) == 1:
        return f"{names[0]} talked"
    if len(names) == 2:
        return f"{names[0]} and {names[1]} talked"
    return f"{', '.join(names[:2])}, and others talked"


def _talks(
    state: pathlib.Path, start: float, now: float, view: transcript_view.View, rewrites: plain_text.Rewrites
) -> list[dict[str, Any]]:
    """The finished conversations over [start, now] as talk rows. Each line is said by its person (View.speakers),
    named only when they are someone known; a conversation whose line isn't written yet goes out after HOLD_S with
    the plain line."""
    raw = [e for e in speech_log.shared().range(start, now + 1.0) if e["ts_start"] >= start]
    read = [(e, pid) for e, pid in view.speakers(raw) if (e.get("text") or "").strip()]
    read.sort(key=lambda r: r[0]["ts_start"])
    groups = interval_groups(
        read, TALK_GAP_S, start=lambda r: r[0]["ts_start"], end=lambda r: r[0].get("ts_end") or r[0]["ts_start"]
    )
    talks = []
    for g in groups:
        end = max(e.get("ts_end") or e["ts_start"] for e, _ in g)
        if end > now - TALK_GAP_S:
            continue  # still going
        who: list[str] = []
        said = []
        for e, pid in g[-MAX_TALK_LINES:]:
            name = (e.get("speaker") or "").strip()
            label = name if pid is not None and name else "someone"
            if label != "someone" and name not in who:
                who.append(name)
            said.append(f"{label}: {e['text'].strip()}")
        talks.append((g[0][0]["ts_start"], end, who, "\n".join(said)[-MAX_TALK_CHARS:]))
    lines_for = rewrites.many(src for *_x, src in talks)
    rows = []
    for s0, end, who, src in talks:
        stage, line = lines_for[src]
        if stage == "pending" and now - end < HOLD_S:
            continue  # give the model a moment before the plain line
        told = stage == "ready" and line
        row = {"id": f"t:{s0:.3f}", "ts": s0, "kind": "talk", "text": line["text"] if told else _fallback(who)}
        row.update(people=who, cam=None, camera_name=None)
        rows.append(row)
    return rows


def _build(state: pathlib.Path, now: float, rewrites: plain_text.Rewrites) -> list[dict[str, Any]]:
    """The log as it is now, oldest first, over the last LOOKBACK_S (read from EDGE_S before that)."""
    start = now - LOOKBACK_S - EDGE_S
    view = transcript_view.View(state)
    rows = _comings(state, start, now, view) + _talks(state, start, now, view, rewrites)
    rows = [r for r in rows if r["ts"] >= now - LOOKBACK_S]
    rows.sort(key=lambda r: (r["ts"], r["id"]))
    return rows


# ---- the rows written so far, and the cursor ----
FIELDS = ("kind", "text", "people", "cam", "camera_name")


class _Rows:
    """The rows of a state written so far, in this process: `kept` is {row id: {"row", "first"}} (`first` the
    position of the row's first entry), and `journal` has one entry for each row written or changed, oldest first,
    each with its `position` (1, 2, and on; `position` is the newest)."""

    def __init__(self) -> None:
        self.kept: dict[str, dict[str, Any]] = {}
        self.journal: list[dict[str, Any]] = []
        self.position = 0

    def add(self, body: dict[str, Any]) -> int:
        """Journal one entry; returns its position."""
        self.position += 1
        self.journal.append({**body, "position": self.position})
        return self.position

    def since(self, position: int) -> list[dict[str, Any]]:
        """The entries written after `position`, oldest first."""
        return [e for e in self.journal if e["position"] > position]

    def trim(self, before: float) -> None:
        """Let go of the rows and entries that happened before `before`."""
        self.journal = [e for e in self.journal if e["ts_start"] >= before]
        for k in [k for k, v in self.kept.items() if v["row"]["ts"] < before]:
            del self.kept[k]


def _row_of(e: dict[str, Any]) -> dict[str, Any]:
    """A journal entry as the page's row."""
    row = {"id": e["row"], "ts": e["ts_start"]}
    row.update({k: e.get(k) for k in FIELDS})
    if e.get("removed"):
        row["removed"] = True
    return row


_rows: dict[str, _Rows] = {}


def _rows_of(state: pathlib.Path) -> _Rows:
    return _rows.setdefault(str(state), _Rows())


def _record(state: pathlib.Path, rows: list[dict[str, Any]], now: float) -> None:
    """Journal each row that is new or changed since it was last written (a revision naming its row's first entry),
    and let go of rows older than the window."""
    held = _rows_of(state)
    for r in rows:
        had = held.kept.get(r["id"])
        if had is not None and had["row"] == r:
            continue
        body = {k: r.get(k) for k in FIELDS}
        body.update(row=r["id"], ts_start=r["ts"], revises=had["first"] if had else None)
        position = held.add(body)
        held.kept[r["id"]] = {"row": r, "first": had["first"] if had else position}
    current = {r["id"] for r in rows}
    for rid, previous in held.kept.items():
        old = previous["row"]
        if (
            rid not in current
            and rid.startswith((f"a:{presence.UNKNOWN_FACES}:", f"d:{presence.UNKNOWN_FACES}:"))
            and not old.get("removed")
            and old["ts"] >= now - LOOKBACK_S
        ):
            # A later named sample can establish that an anonymous episode was the same physical track.
            held.add({"row": rid, "ts_start": old["ts"], "removed": True, "revises": previous["first"]})
            previous["row"] = {**old, "removed": True}
    held.trim(now - LOOKBACK_S - EDGE_S)


class TalkLines(plain_text.Rewrites):
    """plain_text's named-line writer for conversations, its lines kept in this process."""

    def __init__(self, state: pathlib.Path, **kw: Any) -> None:
        super().__init__(
            state,
            lambda t: plain_text._ask(plain_text.NAMED_SYSTEM, f"Conversation:\n{t}"),
            checker=plain_text.check_named,
            **kw,
        )


_talk_lines: dict[str, TalkLines] = {}


def happening(
    state: pathlib.Path,
    since: float | None = None,
    limit: int = 50,
    now: float | None = None,
    rewrites: plain_text.Rewrites | None = None,
) -> dict[str, Any]:
    """GET /live/happening (see the module docstring)."""
    now = time.time() if now is None else now
    key = str(state)
    if rewrites is None:
        rewrites = _talk_lines.setdefault(key, TalkLines(pathlib.Path(state)))
    held = _rows_of(state)
    with _lock:
        hit = _built.get(key)
        if hit is not None and (now - hit[0] >= BUILD_S or now < hit[0]) and key not in _building:
            thread = threading.Thread(target=_refresh, args=(state, now, rewrites), name="happening-build", daemon=True)
            _building[key] = thread
            thread.start()
        # Read with the rows: a build journals its rows and publishes them together under this lock.
        newest = held.position if hit is not None else None
    if hit is None:
        with _build_lock:
            with _lock:
                hit = _built.get(key)  # another first request may have built it while this one waited
                newest = held.position if hit is not None else None
            if hit is None:
                _publish(state, now, _build(state, now, rewrites))
                with _lock:
                    hit = _built[key]
                    newest = held.position
    assert newest is not None
    rows = hit[1]
    limit = max(1, limit)
    if since is None or since > newest:  # past the end: a page from before a restart, sent afresh
        return {"items": rows[-limit:], "updated": newest}
    with _lock:
        entries, newest = held.since(int(since)), held.position
    if not entries:
        return {"items": [], "updated": since}
    taken = entries[:limit]
    updated = taken[-1]["position"] if len(entries) > limit else newest
    latest: dict[str, dict[str, Any]] = {}
    for e in taken:
        latest[e["row"]] = _row_of(e)
    items = sorted(latest.values(), key=lambda r: (r["ts"], r["id"]))
    return {"items": items, "updated": updated}


def _publish(state: pathlib.Path, now: float, rows: list[dict[str, Any]]) -> None:
    """Journal the rows and make them the answer in one step (under _lock, the caller holding _build_lock)."""
    with _lock:
        _record(state, rows, now)
        _built[str(state)] = (now, rows)


def _refresh(state: pathlib.Path, now: float, rewrites: plain_text.Rewrites) -> None:
    """A background build: polls keep the last build until it lands, and keep it if it fails (the next poll past
    BUILD_S tries again)."""
    key = str(state)
    try:
        with _build_lock:
            _publish(state, now, _build(state, now, rewrites))
    except Exception as error:
        LOG.warning("live happening build failed; the last one stands: %r", error)
    finally:
        with _lock:
            _building.pop(key, None)


# ---- routes (dashboard/routes.py): what the server answers for this area ----


@routes.get("/live/happening")
def _get_live_happening(h: H) -> None:
    q = h._query()
    try:
        since = float(q["since"]) if q.get("since") else None
        limit = max(1, min(200, int(q.get("limit") or 50)))
    except ValueError:
        return h._json({"error": "since is a time in seconds, and limit a number"}, 400)
    if since is not None and not math.isfinite(since):
        return h._json({"error": "since is a time in seconds"}, 400)
    try:
        return h._json(happening(h.state, since, limit))
    except Exception as ex:
        LOG.error(f"live happening failed: {ex!r}")
        return h._json({"error": "Couldn't read what's been "}, 500)
