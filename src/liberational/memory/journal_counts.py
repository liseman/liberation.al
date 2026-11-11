"""The Counts tables of the journal's day and week pages (memory/journal_pages.py): numbers from code reading
structured streams, never from the model. The model only decides which counts are worth a
sentence in the prose; it never computes or edits a number here.

What is counted, by default only what the owner has named: the names the owner gave detected
events (`activity.named`, read through memory/event_counts.py) and the zones the owner drew and named
(memory/zone_counts.py). Presence episodes (memory/presence_counts.py) are per camera "someone came into view"
counts that nobody named, so they stay out of the default table.

Each row is one named thing over the period:
- The value is its count from the count modules (event_counts.answer, which hands zones to zone_counts), plus the
  time it took when the spans have length (a zone's visits, an event's spans: their summed time inside the period).
- Time nobody watched is unknown, never zero: a period the detectors didn't cover at all reads "unknown (not
  watched)", one they covered in part says how much ("watched 9 h of 24 h"), and a count the modules won't give
  (unverified candidates, an unchecked model kind, a retired kind) reads "unknown".
- "Usual" is the median and range over recent comparable periods with enough footage: for a day, the DAY_HISTORY
  days before it (memory/typical_counts.py, which leaves out a day watched for less than MIN_COVERED of it); for a
  week, the WEEK_HISTORY weeks before it, under the same rule. Empty when no earlier period had enough footage.
"""

import datetime as dt
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liberational.memory import event_counts, history_qa, journal, typical_counts, zone_counts

DAY_HISTORY = typical_counts.DEFAULT_PERIODS  # days before a day that its "Usual" reads
WEEK_HISTORY = 4  # weeks before a week that its "Usual" reads
WHOLE_SLACK_S = 60.0  # coverage this close to the whole period counts as the whole period


@dataclass(frozen=True)
class CountRow:
    """One row of a Counts table: what was counted, its value for the period (a count or a duration, already
    worded), and the usual range for comparison (empty when there is no history to compare with)."""

    what: str
    value: str
    usual: str = ""


@dataclass(frozen=True)
class _Measure:
    count: int | None  # None: the modules gave no count
    dwell: float  # seconds the counted spans took inside the period
    covered: float | None  # seconds of the period the detectors watched; None when the modules didn't say
    length: float  # seconds in the period


def named_things(state: Path) -> list[tuple[str, bool]]:
    """(name, is a zone) for everything the owner has named, in table order: event names, then zone names, each
    sorted; a zone that shares an event's name is counted once, as the event."""
    events = sorted({n for n in event_counts.known_names(state) if event_counts.usable(n)}, key=str.casefold)
    zones = sorted(
        {z for z in zone_counts.names(state) if not any(event_counts.same(e, z) for e in events)}, key=str.casefold
    )
    return [(n, False) for n in events] + [(z, True) for z in zones]


def _measure(name: str, start: float, end: float, state: Path) -> _Measure:
    query = {"start": start, "end": end, "question": ""}
    reply = event_counts.answer(name, query, state, history_qa.stamp, history_qa.TZ)
    count = reply.get("count")
    coverage = reply.get("coverage") or {}
    # A reply that stopped before reading coverage (a kind it won't count, no detector installed) says nothing about
    # what was watched.
    covered = typical_counts._covered_seconds(coverage, start, end) if "covered" in coverage else None
    return _Measure(None if count is None else int(count), float(reply.get("dwell_s") or 0.0), covered, end - start)


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return f"{round(seconds)} s"
    hours, minutes = divmod(minutes, 60)
    if not hours:
        return f"{minutes} min"
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def _value(m: _Measure, zone: bool) -> str:
    if m.count is None:
        return "unknown (not watched)" if m.covered == 0 else "unknown"
    one, many = ("visit", "visits") if zone else ("time", "times")
    text = f"{m.count} {one if m.count == 1 else many}"
    if m.count and m.dwell >= 60:
        text += f", {_duration(m.dwell)}"
    if m.covered is not None and m.covered < m.length - WHOLE_SLACK_S:
        text += f" (watched {_duration(m.covered)} of {_duration(m.length)})"
    return text


def _usual(values: Sequence[int], unit: str) -> str:
    if not values:
        return ""
    median = statistics.median(values)
    lo, hi = min(values), max(values)
    spread = f"{lo} to {hi}, " if lo != hi else ""
    return f"{median:g} ({spread}{len(values)} {unit if len(values) == 1 else unit + 's'})"


def _day_usual(name: str, date: dt.date, state: Path) -> str:
    start = journal.day_bounds(date - dt.timedelta(days=DAY_HISTORY))[0]
    end = journal.day_bounds(date)[0]
    query = {"start": start, "end": end, "question": ""}  # no question: whole local days, never nights
    reply = typical_counts.answer(
        name,
        query,
        state,
        history_qa.stamp,
        history_qa.TZ,
        now=end,
        count=event_counts.answer,
        scan=event_counts.scan,
    )
    periods: list[dict[str, Any]] = (reply.get("coverage") or {}).get("periods") or []
    return _usual([int(p["count"]) for p in periods], "day")


def _week_bounds(week_start: dt.date) -> tuple[float, float]:
    return journal.day_bounds(week_start)[0], journal.day_bounds(week_start + dt.timedelta(days=7))[0]


def _week_usual(name: str, week_start: dt.date, state: Path) -> str:
    values = []
    for back in range(WEEK_HISTORY, 0, -1):
        start, end = _week_bounds(week_start - dt.timedelta(weeks=back))
        m = _measure(name, start, end, state)
        if m.count is not None and (m.covered or 0.0) >= typical_counts.MIN_COVERED * m.length:
            values.append(m.count)
    return _usual(values, "week")


def day_counts(date: dt.date, *, state: Path | None = None, until: float | None = None) -> list[CountRow]:
    """The counts of a local day, in table order; with `until`, of the day up to then (today so far), so the hours
    still to come aren't read as unwatched."""
    where = history_qa.STATE if state is None else state
    start, end = journal.day_bounds(date)
    end = end if until is None else max(start, min(end, until))
    return [
        CountRow(name, _value(_measure(name, start, end, where), zone), _day_usual(name, date, where))
        for name, zone in named_things(where)
    ]


def week_counts(week_start: dt.date, *, state: Path | None = None) -> list[CountRow]:
    """The counts of the week starting on Monday `week_start`, in table order."""
    where = history_qa.STATE if state is None else state
    start, end = _week_bounds(week_start)
    return [
        CountRow(name, _value(_measure(name, start, end, where), zone), _week_usual(name, week_start, where))
        for name, zone in named_things(where)
    ]


def render_counts_table(rows: list[CountRow], level: str) -> str:
    """A Markdown table of `rows` for a `level` ("day" or "week") page; empty for no rows, so the section is left
    out. A value is a count or a duration as worded; "Usual" is the typical range."""
    if not rows:
        return ""
    period = "Today" if level == "day" else "Week"
    lines = [f"| What | {period} | Usual |", "| --- | ---: | --- |"]
    lines += [f"| {_cell(r.what)} | {_cell(r.value)} | {_cell(r.usual)} |" for r in rows]
    return "\n".join(lines)


def _cell(text: str) -> str:
    """A table cell's text: one line, with no pipe to break the row."""
    return " ".join(str(text).split()).replace("|", "/")
