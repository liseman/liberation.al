"""The Journal tab's reads: the journal as a column view, month down to the journal's own events.

Every level is one read-only GET on /journal (a page load of /journal with text/html gets the app instead):

    GET /journal                      the months, newest first: {items}
    GET /journal?month=YYYY-MM-01     that month's weeks (each overlapping it, Monday first) and its page
    GET /journal?week=YYYY-MM-DD      that week's seven days (from its Monday) and its page
    GET /journal?day=YYYY-MM-DD       that day's hour entries and its page
    GET /journal?hour=<epoch>         the event entries of the hour starting then (the journal's next level down)
    GET /journal?entry=<id>&start=<epoch>   one hour's or event's evidence, resolved: {status, sources}

A listing reply is {items, page}. `items` are the next column's rows: {key, label, ...}, months, weeks, and days newest
first, hours and events in time order with their entry id, text, and how many sources they cite. `page` is the
period's journal.page (day, week, or month; memory/journal_pages.py) as Markdown with speaker tokens as current names
and its `#t=` citations as Review links, or null while none is shown, with `why` saying why (`why_no_page`): not
written yet, being written again after a transcript correction, nothing recorded, the model failed, or too old to be
written. Pages and entries are only read here; the journal worker writes them.

Evidence is read for the one entry the tab shows, not for every row, since resolving a camera moment asks the broker
what video it still has. The sources the entry cites (memory/journal.py `evidence_source`) go through
memory/journal_evidence.py's `resolve_sources` in one batch (a broker read per camera, not per source), the same as a
day page's citations (/journal/evidence): the recording
while it's kept, then a promoted clip or still, audio for speech, or "footage expired" / "no footage was kept". This
adds who and when, a Review link, and the audio's URL.
"""

import calendar
import datetime as dt
import math
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from liberational import config as cfg
from liberational import memory_log
from liberational.dashboard import routes
from liberational.memory import history_qa, journal, journal_evidence, journal_pages

if TYPE_CHECKING:
    from liberational.dashboard.server import H

Reply = tuple[int, dict[str, Any]]

# How many months the first column lists, the current one first.
MONTHS_SHOWN = 12
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _local(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, history_qa.TZ)


def _date(value: str) -> dt.date:
    if not _DATE.fullmatch(value):
        raise ValueError(value)
    return dt.date.fromisoformat(value)


# The latest time a query may name: well past any recording, and far inside what datetime can hold.
_LATEST = 4102444800.0  # 2100-01-01


def _epoch(value: str) -> float:
    t = float(value)
    if not math.isfinite(t) or not 0 <= t <= _LATEST:
        raise ValueError(value)
    return t


def event_level() -> int:
    """The level the Events column lists: the journal's level below the hour's (its second finest of four, 3 minutes
    by default), so an hour holds about twenty at most, each with its own evidence. With only two levels, the finest."""
    levels = journal.levels()
    return levels[1] if len(levels) > 2 else levels[0]


def _pages(level: str, start: float, end: float, state: Path) -> dict[str, dict[str, Any]]:
    """The current page at `level` of each period starting in [start, end), by period."""
    mlog = memory_log.at(state)
    newest: dict[str, dict[str, Any]] = {}
    for r in memory_log.reviewed(memory_log.current(mlog.range(memory_log.PAGES, start, end)), state):
        if r.get("level") != level or r.get("status") or not start <= float(r["ts_start"]) < end:
            continue
        if str(r["id"]) > str(newest.get(r["period"], {}).get("id", "")):
            newest[r["period"]] = r
    return newest


def why_no_page(level: str, start: dt.date, state: Path, now: float) -> str:
    """Why the tab shows no page for a period: "rewriting" (a transcript correction hid it, and the journal's next pass
    writes it again), "quiet" (nothing was recorded in it), "failed" (the model failed on it, tried again when its
    inputs change), "pending" (not written yet: it hasn't settled, or the next batch writes it), or "old" (none was
    written, and it is older than the journal looks back, so none will be)."""
    have = journal_pages.stored(memory_log.at(state), level, start)
    if have is not None:
        return {memory_log.QUIET: "quiet", memory_log.FAILED: "failed"}.get(str(have.get("status")), "rewriting")
    if not journal_pages.settled(level, start, now):
        return "pending"
    today = _local(now).date()
    lookback = max(1, int(cfg.get("journal.reconcile_days", 2)))
    looked_at = journal_pages.candidates(level, today, lookback, journal_pages.corrected_days(state))
    return "pending" if start in looked_at else "old"


def _page_reply(level: str, start: dt.date, state: Path, now: float) -> dict[str, Any]:
    """A listing's `page` for the period at `level` starting on `start`, and `why` when there is none to show."""
    view = page_view(memory_log.at(state).page(level, start.isoformat()), state)
    return {"page": view, "why": None if view else why_no_page(level, start, state, now)}


def page_view(record: Mapping[str, Any] | None, state: Path) -> dict[str, Any] | None:
    """A page for the tab: its Markdown with names, and its `#t=` citations opening Review at that moment."""
    if record is None:
        return None
    md = journal_pages.markdown(record, state).replace("](#t=", "](/review#t=")
    return {"level": record.get("level"), "period": record.get("period"), "md": md}


def evidence(entry: Mapping[str, Any], state: Path) -> dict[str, Any]:
    """An entry's evidence as the tab shows it: whether it was verified, and each source resolved to what it can show
    now (journal_evidence.resolve's `stage` and its fields), with who, when, and a Review link."""
    ev = entry.get("evidence") or {}
    day = _local(float(entry["ts_start"])).date().isoformat()
    sources = [s for s in ev.get("sources") or [] if isinstance(s, Mapping)]
    out = []
    for s, resolved in zip(sources, journal_evidence.resolve_sources(state, day, sources), strict=True):
        start = float(s.get("ts_start") or 0.0)
        end = max(start, float(s.get("ts_end") or start))
        item: dict[str, Any] = {
            **resolved,
            "kind": s.get("kind"),
            "stream": s.get("stream"),
            "people": [p.get("name") for p in s.get("people") or [] if p.get("name")],
            "start": start,
            "end": end,
            "review": f"/review#t={start:.1f}",
        }
        if item.get("stage") == "audio":
            item["audio"] = "/audio/range?" + urlencode({"start": f"{start:.2f}", "end": f"{max(end, start + 1):.2f}"})
        out.append(item)
    return {"status": ev.get("status"), "sources": out}


def _entry(r: Mapping[str, Any], state: Path, *, by_start: bool) -> dict[str, Any]:
    """An hour or event entry as a row: keyed by its start (an hour, which the next column asks for) or its id."""
    return {
        "key": str(int(float(r["ts_start"]))) if by_start else str(r["id"]),
        "id": str(r["id"]),
        "label": _local(float(r["ts_start"])).strftime("%H:%M"),
        "start": float(r["ts_start"]),
        "end": float(r["ts_end"]),
        "text": journal.render(str(r.get("text") or ""), state),
        "sources": len((r.get("evidence") or {}).get("sources") or []),
    }


def entry_evidence(entry_id: str, start: float, state: Path) -> Reply:
    """Resolve an entry link through corrections before showing its evidence."""
    rows = memory_log.at(state).range(memory_log.JOURNAL, start, start + 1)
    by_id = {str(r["id"]): r for r in rows}
    corrections = {
        str(r["supersedes"]): str(r["id"]) for r in sorted(rows, key=lambda r: str(r["id"])) if r.get("supersedes")
    }
    visited: set[str] = set()
    while entry_id in corrections and entry_id not in visited:
        visited.add(entry_id)
        entry_id = corrections[entry_id]
    found = by_id.get(entry_id)
    if found is None:
        return 404, {"error": "no such entry"}
    return 200, evidence(found, state)


def months(state: Path, today: dt.date) -> Reply:
    first = today.replace(day=1)
    starts = [first]
    for _ in range(MONTHS_SHOWN - 1):
        starts.append((starts[-1] - dt.timedelta(days=1)).replace(day=1))
    a = journal_pages.bounds(journal_pages.MONTH, starts[-1])[0]
    b = journal_pages.bounds(journal_pages.MONTH, first)[1]
    pages = _pages(journal_pages.MONTH, a, b, state)
    items = [{"key": m.isoformat(), "label": m.strftime("%B %Y"), "page": m.isoformat() in pages} for m in starts]
    return 200, {"items": items, "page": None}


def weeks(month: dt.date, state: Path, now: float) -> Reply:
    today = _local(now).date()
    month = month.replace(day=1)
    end = month + dt.timedelta(days=calendar.monthrange(month.year, month.month)[1])
    monday = journal_pages.period_start(journal_pages.WEEK, month)
    starts = []
    while monday < end and monday <= today:
        starts.append(monday)
        monday += dt.timedelta(days=7)
    if not starts:
        return 200, {"items": [], "page": None}
    a = journal_pages.bounds(journal_pages.WEEK, starts[0])[0]
    b = journal_pages.bounds(journal_pages.WEEK, starts[-1])[1]
    pages = _pages(journal_pages.WEEK, a, b, state)
    items = []
    for w in reversed(starts):
        last = w + dt.timedelta(days=6)
        label = f"{w.strftime('%b')} {w.day} to {'' if last.month == w.month else last.strftime('%b') + ' '}{last.day}"
        items.append({"key": w.isoformat(), "label": label, "page": w.isoformat() in pages})
    return 200, {"items": items, **_page_reply(journal_pages.MONTH, month, state, now)}


def days(week: dt.date, state: Path, now: float) -> Reply:
    today = _local(now).date()
    week = journal_pages.period_start(journal_pages.WEEK, week)
    a, b = journal_pages.bounds(journal_pages.WEEK, week)
    pages = _pages(journal_pages.DAY, a, b, state)
    top = journal.levels()[-1]
    counts: dict[str, int] = {}
    for r in memory_log.at(state).entries(a, b, level=top):
        if a <= float(r["ts_start"]) < b:
            d = _local(float(r["ts_start"])).date().isoformat()
            counts[d] = counts.get(d, 0) + 1
    items = []
    for i in reversed(range(7)):
        d = week + dt.timedelta(days=i)
        if d > today:
            continue
        key = d.isoformat()
        items.append(
            {"key": key, "label": d.strftime("%a %b ") + str(d.day), "page": key in pages, "hours": counts.get(key, 0)}
        )
    return 200, {"items": items, **_page_reply(journal_pages.WEEK, week, state, now)}


def hours(day: dt.date, state: Path, now: float) -> Reply:
    a, b = journal.day_bounds(day)
    top = journal.levels()[-1]
    rows = [r for r in memory_log.at(state).entries(a, b, level=top) if a <= float(r["ts_start"]) < b]
    items = [_entry(r, state, by_start=True) for r in rows]
    return 200, {"items": items, **_page_reply(journal_pages.DAY, day, state, now)}


def events(start: float, state: Path) -> Reply:
    end = start + journal.levels()[-1]
    rows = [r for r in memory_log.at(state).entries(start, end, level=event_level()) if start <= r["ts_start"] < end]
    return 200, {"items": [_entry(r, state, by_start=False) for r in rows], "page": None}


def listing(query: Mapping[str, str], state: Path, now: float | None = None) -> Reply:
    """GET /journal: the column a query asks for (see the module docstring)."""
    now = time.time() if now is None else now
    today = _local(now).date()
    try:
        if query.get("entry"):
            return entry_evidence(query["entry"], _epoch(query.get("start", "")), state)
        if query.get("hour"):
            return events(_epoch(query["hour"]), state)
        if query.get("day"):
            return hours(_date(query["day"]), state, now)
        if query.get("week"):
            return days(_date(query["week"]), state, now)
        if query.get("month"):
            return weeks(_date(query["month"]), state, now)
    except ValueError:
        return 400, {"error": "month, week, and day are YYYY-MM-DD dates; hour and start are epoch times"}
    return months(state, today)


# ---- routes (dashboard/routes.py): what the server answers for this area ----


@routes.get("/journal")
def _get_journal(h: H) -> None:
    code, reply = listing(h._query(), h.state)
    return h._json(reply, code)


@routes.get("/journal/today")
def _get_journal_today(h: H) -> None:
    from liberational.memory import ask, journal_pages

    try:
        day = ask.validate_day(h._query().get("day", ""))
    except ValueError:
        return h._json({"error": "day must be YYYY-MM-DD"}, 400)
    code, reply = journal_pages.today_page(h.state, time.time(), dt.date.fromisoformat(day) if day else None)
    return h._json(reply, code)


@routes.get("/journal/evidence")
def _get_journal_evidence(h: H) -> None:
    from liberational.memory import journal_evidence

    day = h._query().get("day") or ""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        return h._json({"error": "day must be YYYY-MM-DD"}, 400)
    return h._json({"day": day, "sources": journal_evidence.resolve_day(h.state, day)})


@routes.get_re(r"/journal/evidence/(\d{4}-\d{2}-\d{2})/([0-9A-Za-z_.-]{1,96})")
def _get_journal_evidence_x_x(h: H, m: re.Match[str]) -> None:
    from liberational.memory import journal_evidence

    found = journal_evidence.file(h.state, m.group(1), m.group(2))
    if found is None:
        return h._send(404, "text/plain", b"")
    # Byte ranges (206), which Safari and iOS need to play an mp4 at all.
    return h._media(found, "video/mp4" if found.suffix == ".mp4" else "image/jpeg")
