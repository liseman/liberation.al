"""Reports: daily aggregates over the whole corpus, for the dashboard's Reports tab (dashboard/reports.py).

A report never ships raw rows. Each one turns a local day of records into one small aggregate (`Report.day`), and
turns the stored aggregates of every day into the figures a chart draws (`Report.assemble`). Days are the unit
cached, so a year of data is read once and then only today, and days that changed, are read again:

- **Store.** `state/reports.sqlite3` keeps each report's day aggregates (with the report's version, so a change to
  what a day holds rereads every day), the last purge seen, the Reports tab's chosen set and order (`layout`), and
  System health's samples (dashboard/report_health.py).
- **Settled days.** A day is final once SETTLE_S has passed since it ended (late workers, such as faces on a
  backlog, still add records for a while). Until then it is reread whenever its copy is older than FRESH_S.
- **Purges.** Forgetting a person or deleting a stretch takes records away. Each read checks for purges newer than
  the last one seen: a purge of a time range drops the days it touches, and any other purge (of ids or a person)
  drops every day, so nothing forgotten lives on in a report. A range purge only of streams no report reads
  (`UNREAD_STREAM_PREFIXES`: the broker's worker checkpoints, which it purges every few seconds from 0 ns) is
  skipped, and the rest are clipped to the days reports cover and joined before they become days, so a batch of new
  purges costs at most MAX_DAYS days however wide it is. The list
  is of streams no report reads rather than of those they do: a stream left off it costs a reread, never a report
  that keeps what was forgotten. If the saved purge cursor no longer exists (such as after a broker reset), its
  missing history cannot validate cached days: retire the cursor and invalidate every day once, then read from
  the beginning on the next check. Other broker errors keep the cursor and retry normally.
- **Building.** A request never reads a day itself. It gets every day stored now and the list of days still
  `pending`, and `Builder` reads those in the background, newest first, so the page fills in while it polls.
- **Coverage.** A day records which hours had recording (`Sources.recorded_hours`) or the coverage records its
  counts use. An hour nobody could have been seen or heard in is None in the figures, never a zero, and the charts
  draw it as a gap.

Who's around and Conversations read the timeline (memory/timeline.py: face visits and heard lines, each resolved to
a person through the event log). Day aggregates hold person ids, never names; names are looked up when assembled, so
a rename shows the new name and a forgotten person (whom the timeline no longer resolves) drops out.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import math
import multiprocessing
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from liberational import logs, proc_table
from liberational.memory.conversation_topics import STOP

LOG = logs.get("reports")

MAX_DAYS = 400  # the most days a report reads back from today: a year and some
SETTLE_S = 6 * 3600.0  # a day this long past its end is final
FRESH_S = 300.0  # a day not final yet is reread when its copy is older than this
REPLY_S = 60.0  # two people's lines this close are one exchange between them
TOP_WORDS_DAY = 60  # words kept per day; a word that never makes a day's top is too rare to chart
TOP_WORDS = 12  # words charted over time
TOP_PAIRS = 20
WORD = re.compile(r"[a-z][a-z'-]{2,}")
# Streams no report reads, so a range purge of only these leaves every day as it is: name prefixes and exact names.
UNREAD_STREAM_PREFIXES = ("worker.",)  # the broker's own worker state, such as worker.checkpoint.<hash>.parts


def day_bounds(day: str) -> tuple[float, float]:
    """[start, end) of a local day 'YYYY-MM-DD' in epoch seconds (23 or 25 hours across a clock change)."""
    t = time.strptime(day, "%Y-%m-%d")
    return time.mktime(t), time.mktime((t.tm_year, t.tm_mon, t.tm_mday + 1, 0, 0, 0, 0, 0, -1))


def hour_bounds(day: str) -> list[tuple[float, float]]:
    """The 24 local clock hours of a day as [start, end). On a spring-forward day the skipped hour is empty
    (start == end), and on a fall-back day the repeated hour is folded into its slot."""
    t = time.strptime(day, "%Y-%m-%d")
    starts = [time.mktime((t.tm_year, t.tm_mon, t.tm_mday, h, 0, 0, 0, 0, -1)) for h in range(24)]
    end = day_bounds(day)[1]
    # mktime moves a skipped 02:00 to 03:00, so that slot ends where it starts.
    return [(a, max(a, starts[h + 1] if h < 23 else end)) for h, a in enumerate(starts)]


def day_of(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def days_between(first: str, last: str) -> list[str]:
    """Every day from first to last, inclusive, oldest first."""
    a, b = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    return [(a + dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def merged(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Overlapping or touching spans joined, sorted."""
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def per_hour(spans: Iterable[tuple[float, float]], hours: Sequence[tuple[float, float]]) -> list[float]:
    """Seconds of the joined spans inside each hour."""
    joined = merged(spans)
    return [round(sum(max(0.0, min(b, hb) - max(a, ha)) for a, b in joined), 1) for ha, hb in hours]


def hour_index(ts: float, hours: Sequence[tuple[float, float]]) -> int | None:
    for h, (a, b) in enumerate(hours):
        if a <= ts < b:
            return h
    return None


def terms(text: str, names: frozenset[str]) -> list[str]:
    """The substantive words of a line: lowercase, three letters or more, not a stop word or a person's name."""
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", text.lower())
    return [w for w in WORD.findall(text) if w not in STOP and w not in names and "'" not in w]


# ---- what a report reads ----


@dataclass(frozen=True)
class Seen:
    """One stretch a person was seen (a face visit) or heard (a line), with the line's text when heard."""

    person: str
    start: float
    end: float
    heard: bool
    text: str = ""


class Sources(Protocol):
    """Where reports read from; tests give a fake."""

    def people(self) -> Mapping[str, str] | None:
        """Every person's id and display name now, or None when there is no timeline (no event log)."""
        ...

    def seen(self, start: float, end: float) -> list[Seen]:
        """Every visit and heard line overlapping [start, end) that resolves to a person."""
        ...

    def recorded_hours(self, kind: str, day: str) -> list[bool]:
        """For each of the day's 24 hours, whether anything was recorded: kind "video" (any camera) or "audio"."""
        ...

    def first_day(self) -> str | None:
        """The first local day anything was recorded, or None."""
        ...

    def identity_mark(self) -> str | None:
        """Current identity and activity-evidence revision: labels, merges, owner ratings/names, and promotion
        settings can change a past report even after its source day is settled."""
        ...

    def events(self, start: float, end: float) -> list[dict[str, Any]]:
        """Promoted activity spans overlapping [start, end): {"name", "start"} at least (memory/events.py)."""
        ...

    def activity_coverage(self, start: float, end: float) -> list[tuple[str, float, float]]:
        """(camera, start, end) of each stretch the activity worker went through."""
        ...

    def zone_visits(self, start: float, end: float) -> list[dict[str, Any]]:
        """Zone visits overlapping [start, end): the segment store's zone.visit records (memory/zone_counts.py)."""
        ...

    def zone_coverage(self, start: float, end: float) -> list[dict[str, Any]]:
        """zone.coverage records overlapping [start, end): {"subject_kind", "ts_start", "ts_end"} at least."""
        ...

    def action_spans(self, start: float, end: float) -> list[dict[str, Any]]:
        """Model action candidates, excluding rejected/replaced spans, with an explicit verified flag."""
        ...

    def segment_rows(self, stream: str, start: float, end: float) -> list[dict[str, Any]]:
        """One day of a typed segment stream, retaining its coverage fields."""
        ...


class Report(Protocol):
    name: str
    version: int
    identity: bool  # its days depend on current identities or owner activity evidence/settings

    def day(self, src: Sources, day: str) -> dict[str, Any]: ...

    def assemble(self, days: Mapping[str, Mapping[str, Any]], src: Sources, arg: str | None) -> dict[str, Any]: ...


def _covered(src: Sources, day: str, kinds: Sequence[str]) -> list[bool]:
    hours = [False] * 24
    for kind in kinds:
        hours = [a or b for a, b in zip(hours, src.recorded_hours(kind, day), strict=True)]
    return hours


class WhosAround:
    """Who was home or in view, by day and hour: seconds each person was seen or heard in each hour.

    Day: {"cov": [24 bools], "people": {person id: [24 seconds]}}. An hour with no recording but someone in it (an
    imported clip) counts as covered."""

    name = "whos_around"
    version = 1
    identity = True

    def day(self, src: Sources, day: str) -> dict[str, Any]:
        lo, hi = day_bounds(day)
        hours = hour_bounds(day)
        spans: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
        for s in src.seen(lo, hi):
            spans[s.person].append((max(s.start, lo), min(max(s.end, s.start + 1.0), hi)))
        people = {p: per_hour(v, hours) for p, v in spans.items()}
        cov = _covered(src, day, ("video", "audio"))
        cov = [c or any(v[h] > 0 for v in people.values()) for h, c in enumerate(cov)]
        return {"cov": cov, "people": {p: v for p, v in people.items() if any(v)}}

    def assemble(self, days: Mapping[str, Mapping[str, Any]], src: Sources, arg: str | None) -> dict[str, Any]:
        names = src.people() or {}
        totals: collections.Counter[str] = collections.Counter()
        rows = []
        for day, body in sorted(days.items()):
            hours = hour_bounds(day)
            cells: list[dict[str, Any] | None] = []
            for h in range(24):
                if not body["cov"][h]:
                    cells.append(None)
                    continue
                present = [(p, v[h]) for p, v in body["people"].items() if v[h] > 0 and p in names]
                for p, s in present:
                    totals[p] += s  # everyone, so the picker keeps the whole roster while one person is shown
                here = [(p, s) for p, s in present if arg in (None, p)]
                here.sort(key=lambda x: (-x[1], names[x[0]]))
                cells.append({"t": hours[h][0], "who": [[p, round(s / 60, 1)] for p, s in here]})
            rows.append({"day": day, "hours": cells})
        people = [
            {"id": p, "name": names[p], "hours": round(totals[p] / 3600, 1)}
            for p in sorted(totals, key=lambda p: (-totals[p], names[p]))
        ]
        return {"days": rows, "people": people}


class Conversations:
    """Who talks, when, about what, and with whom, from heard lines.

    Day: {"cov": [24 bools], "lines": {person: [24 line counts]}, "speech_s": {person: seconds}, "first": time of
    the day's first line, "words": {word: count} (the day's top words), "pairs": {"a|b": [exchanges, first time]}}."""

    name = "conversations"
    version = 1
    identity = True

    def day(self, src: Sources, day: str) -> dict[str, Any]:
        lo, hi = day_bounds(day)
        hours = hour_bounds(day)
        names = src.people() or {}
        name_words = frozenset(w for n in names.values() for w in n.lower().split())
        lines = sorted((s for s in src.seen(lo, hi) if s.heard and lo <= s.start < hi), key=lambda s: s.start)
        counts: dict[str, list[int]] = {}
        speech: collections.defaultdict[str, float] = collections.defaultdict(float)
        words: collections.Counter[str] = collections.Counter()
        pairs: dict[str, list[float]] = {}
        for i, s in enumerate(lines):
            h = hour_index(s.start, hours)
            if h is not None:
                counts.setdefault(s.person, [0] * 24)[h] += 1
            speech[s.person] += max(0.0, s.end - s.start)
            words.update(terms(s.text, name_words))
            prev = lines[i - 1] if i else None
            if prev is not None and prev.person != s.person and s.start - prev.end <= REPLY_S:
                key = "|".join(sorted((prev.person, s.person)))
                pair = pairs.setdefault(key, [0, prev.start])
                pair[0] += 1
        return {
            "cov": _covered(src, day, ("audio",)),
            "lines": counts,
            "speech_s": {p: round(v, 1) for p, v in speech.items()},
            "first": lines[0].start if lines else None,
            "words": dict(words.most_common(TOP_WORDS_DAY)),
            "pairs": pairs,
        }

    def assemble(self, days: Mapping[str, Mapping[str, Any]], src: Sources, arg: str | None) -> dict[str, Any]:
        names = src.people() or {}
        known = [d for d in sorted(days)]
        by_hour: dict[str, list[int]] = {}
        latest: dict[tuple[str, int], str] = {}  # (person, hour): the last day they talked in that hour
        speech: collections.defaultdict[str, float] = collections.defaultdict(float)
        daily = []
        words: collections.Counter[str] = collections.Counter()
        weekly: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        week_covered: dict[str, bool] = {}
        week_best: dict[tuple[str, str], tuple[int, str]] = {}  # (week, word): (most uses on one day, that day)
        pairs: dict[str, list[Any]] = {}
        for day in known:
            body = days[day]
            covered = any(body["cov"]) or bool(body["lines"])
            week = monday(day)
            week_covered[week] = week_covered.get(week, False) or covered
            total = 0
            for p, hours in body["lines"].items():
                if p not in names:
                    continue
                row = by_hour.setdefault(p, [0] * 24)
                for h, n in enumerate(hours):
                    row[h] += n
                    total += n
                    if n:
                        latest[(p, h)] = day
                speech[p] += body["speech_s"].get(p, 0.0)
            daily.append({"day": day, "lines": total if covered else None, "t": body["first"]})
            for w, n in body["words"].items():
                words[w] += n
                weekly[week][w] += n
                if n > week_best.get((week, w), (0, ""))[0]:
                    week_best[(week, w)] = (n, day)
            for key, (n, first) in body["pairs"].items():
                a, b = key.split("|")
                if a not in names or b not in names:
                    continue
                pair = pairs.setdefault(key, [a, b, 0, day, first])
                pair[2] += n
                pair[3], pair[4] = day, first  # days go oldest first, so this ends on the latest
        people = sorted(by_hour, key=lambda p: (-sum(by_hour[p]), names[p]))
        weeks = sorted(week_covered)
        kind, _, word = (arg or "").partition(":")
        top = [word] if kind == "word" and word in words else [w for w, _ in words.most_common(TOP_WORDS)]
        return {
            "people": [
                {
                    "id": p,
                    "name": names[p],
                    "lines": sum(by_hour[p]),
                    "speech_min": round(speech[p] / 60, 1),
                    "by_hour": [
                        {"n": n, "day": latest.get((p, h)), "t": _hour_start(latest.get((p, h)), h)}
                        for h, n in enumerate(by_hour[p])
                    ],
                }
                for p in people
            ],
            "daily": daily,
            "words": [
                {
                    "word": w,
                    "total": words[w],
                    "weeks": [
                        {
                            "week": wk,
                            "n": weekly[wk][w] if week_covered[wk] else None,
                            "day": week_best.get((wk, w), (0, None))[1],
                        }
                        for wk in weeks
                    ],
                }
                for w in top
            ],
            "pairs": [
                {"a": a, "b": b, "a_name": names[a], "b_name": names[b], "n": n, "day": day, "t": first}
                for a, b, n, day, first in sorted(pairs.values(), key=lambda x: -x[2])[:TOP_PAIRS]
            ],
        }


def _typical(values: Sequence[int | None]) -> float | None:
    """The median of the days that were covered, or None when none were."""
    known = sorted(v for v in values if v is not None)
    if not known:
        return None
    mid = len(known) // 2
    return float(known[mid]) if len(known) % 2 else (known[mid - 1] + known[mid]) / 2


def _hours_covered(spans: Iterable[tuple[float, float]], hours: Sequence[tuple[float, float]]) -> list[bool]:
    return [s > 0 for s in per_hour(spans, hours)]


class Activity:
    """Events, zone visits, and pet activity over time, each over the coverage records its counts use.

    Day: {"events": {name: [24 counts]}, "event_cov": [24 bools] (the activity worker's coverage), "zones": {zone:
    [24 counts]}, "pets": {label: [24 counts]}, "zone_cov" and "pet_cov": [24 bools] (zone coverage for any subject,
    and for animals), "first": {series key: time of its first moment that day}}. Events count the promoted spans
    (memory/events.py's gate); visits count those of at least zone_counts.MIN_DWELL_S, after joining a visit split by
    its track (zone_counts.merged), as a count answer does."""

    name = "activity"
    version = 1
    identity = True  # owner names/ratings and promotion settings can change settled event counts
    SHOWN = 6  # series charted; the rest are summed into "Other events"

    def day(self, src: Sources, day: str) -> dict[str, Any]:
        from liberational.memory import zone_counts

        lo, hi = day_bounds(day)
        hours = hour_bounds(day)
        first: dict[str, float] = {}

        def count(into: dict[str, list[int]], key: str, series: str, ts: float) -> None:
            h = hour_index(ts, hours)
            if h is None:
                return
            into.setdefault(key, [0] * 24)[h] += 1
            first[series] = min(first.get(series, ts), ts)

        events: dict[str, list[int]] = {}
        for e in src.events(lo, hi):
            name = str(e.get("name") or "something not named yet")
            count(events, name, f"event:{name}", float(e["start"]))
        zones: dict[str, list[int]] = {}
        pets: dict[str, list[int]] = {}
        groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
        for v in src.zone_visits(lo, hi):
            groups[(str(v.get("zone")), str(v.get("label")))].append(v)
        for g in groups.values():
            for v in zone_counts.merged(g):
                ts = float(v["ts_start"])
                if v["dwell_s"] < zone_counts.MIN_DWELL_S or not lo <= ts < hi:
                    continue
                zone = str(v.get("zone_name") or v.get("zone"))
                count(zones, zone, f"zone:{zone}", ts)
                if v.get("subject_kind") == "animal":
                    label = str(v.get("label") or "animal")
                    count(pets, label, f"pet:{label}", ts)
        cov = src.zone_coverage(lo, hi)
        return {
            "events": events,
            "event_cov": _hours_covered(((a, b) for _, a, b in src.activity_coverage(lo, hi)), hours),
            "zones": zones,
            "zone_cov": _hours_covered(((c["ts_start"], c["ts_end"]) for c in cov), hours),
            "pets": pets,
            "pet_cov": _hours_covered(
                ((c["ts_start"], c["ts_end"]) for c in cov if c.get("subject_kind") == "animal"), hours
            ),
            "first": first,
        }

    def _series(
        self, days: Mapping[str, Mapping[str, Any]], group: str, cov: str, prefix: str, only: str | None
    ) -> list[dict[str, Any]]:
        totals: collections.Counter[str] = collections.Counter()
        for body in days.values():
            for key, hours in body[group].items():
                totals[key] += sum(hours)
        keys = [only] if only is not None else [k for k, _ in totals.most_common()]
        shown, rest = keys[: self.SHOWN], keys[self.SHOWN :]
        out = []
        for key in shown + (["Other events"] if rest and group == "events" else []):
            members = rest if key == "Other events" and rest else [key]
            per_day: list[dict[str, Any]] = []
            for day in sorted(days):
                body = days[day]
                covered = any(body[cov]) or any(sum(body[group].get(m, [])) for m in members)
                n = sum(sum(body[group].get(m, [])) for m in members) if covered else None
                times = [body["first"][f"{prefix}:{m}"] for m in members if f"{prefix}:{m}" in body["first"]]
                per_day.append({"day": day, "n": n, "t": min(times) if times and n else None})
            by_hour = [0] * 24
            latest: list[str | None] = [None] * 24
            for day in sorted(days):
                for m in members:
                    for h, n in enumerate(days[day][group].get(m, [])):
                        by_hour[h] += n
                        if n:
                            latest[h] = day
            out.append(
                {
                    "key": f"{prefix}:{key}",
                    "name": key,
                    "total": sum(totals[m] for m in members),
                    "typical": _typical([d["n"] for d in per_day]),
                    "days": per_day,
                    "by_hour": [
                        {"n": n, "day": latest[h], "t": _hour_start(latest[h], h)} for h, n in enumerate(by_hour)
                    ],
                }
            )
        return out

    def assemble(self, days: Mapping[str, Mapping[str, Any]], src: Sources, arg: str | None) -> dict[str, Any]:
        kind, _, value = (arg or "").partition(":")
        focus = value if kind in ("event", "zone", "pet") and value else None
        return {
            "events": self._series(days, "events", "event_cov", "event", focus if kind == "event" else None)
            if focus is None or kind == "event"
            else [],
            "zones": self._series(days, "zones", "zone_cov", "zone", focus if kind == "zone" else None)
            if focus is None or kind == "zone"
            else [],
            "pets": self._series(days, "pets", "pet_cov", "pet", focus if kind == "pet" else None)
            if focus is None or kind == "pet"
            else [],
            "covered_days": {
                "events": sum(1 for b in days.values() if any(b["event_cov"])),
                "zones": sum(1 for b in days.values() if any(b["zone_cov"])),
                "pets": sum(1 for b in days.values() if any(b["pet_cov"])),
            },
        }


def monday(day: str) -> str:
    d = dt.date.fromisoformat(day)
    return (d - dt.timedelta(days=d.weekday())).isoformat()


def _hour_start(day: str | None, h: int) -> float | None:
    return hour_bounds(day)[h][0] if day else None


def _hour_series(days: Mapping[str, Mapping[str, Any]], only: str | None = None) -> list[dict[str, Any]]:
    """Count series with independent per-source coverage, used by walk-bys and device observations."""
    keys = sorted({key for body in days.values() for key in body["counts"]})
    out = []
    for key in keys:
        if only is not None and key != only:
            continue
        rows: list[dict[str, Any]] = []
        for day, body in sorted(days.items()):
            counts = body["counts"].get(key, [0] * 24)
            times = body["times"].get(key, [None] * 24)
            cov = body["cov"].get(key, [False] * 24)
            hours = [{"n": n, "t": times[h]} if cov[h] or n else None for h, n in enumerate(counts)]
            known = [h for h in hours if h is not None]
            first = [h["t"] for h in known if h["t"] is not None]
            rows.append(
                {
                    "day": day,
                    "n": sum(h["n"] for h in known) if known else None,
                    "t": min(first) if first else None,
                    "hours": hours,
                }
            )
        out.append({"key": key, "name": key, "total": sum(r["n"] or 0 for r in rows), "days": rows})
    return sorted(out, key=lambda s: (-s["total"], s["key"]))


class Walkbys:
    """Promoted crossing spans by camera and hour, with that camera's walk-by coverage.

    One span is one crossing observation; this report does not infer unique people or count frames. Each hourly
    bucket links to its first crossing, rather than an invented time inside the hour.
    """

    name = "walkbys"
    version = 1
    identity = True

    def day(self, src: Sources, day: str) -> dict[str, Any]:
        lo, hi = day_bounds(day)
        hours = hour_bounds(day)
        counts: dict[str, list[int]] = {}
        times: dict[str, list[float | None]] = {}
        coverage: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
        for cam, a, b in src.activity_coverage(lo, hi):
            coverage[cam].append((a, b))
        for e in src.events(lo, hi):
            if e.get("kind") != "crossing" or not lo <= float(e["start"]) < hi:
                continue
            cam = str(e.get("camera") or "Unknown camera")
            h = hour_index(float(e["start"]), hours)
            if h is not None:
                counts.setdefault(cam, [0] * 24)[h] += 1
                values = times.setdefault(cam, [None] * 24)
                values[h] = min(values[h] or float(e["start"]), float(e["start"]))
        return {
            "counts": counts,
            "times": times,
            "cov": {cam: _hours_covered(spans, hours) for cam, spans in coverage.items()},
        }

    def assemble(self, days: Mapping[str, Mapping[str, Any]], src: Sources, arg: str | None) -> dict[str, Any]:
        return {
            "cameras": _hour_series(days, arg),
            "covered_days": sum(any(any(c) for c in b["cov"].values()) for b in days.values()),
        }


class PersonActivity:
    """Per-person model span durations, including explicitly unverified candidates, never inferred between spans.

    Spans of the same person and kind are unioned before splitting at hours, so overlapping windows/cameras do not
    multiply duration. Unknown/unmatched people are omitted. Names are resolved at assembly, as in Who's around.
    The producer's coverage names neither the evaluated person nor the kinds it evaluated; no-positive slots stay
    unknown even when another model or camera was running, rather than asserting a person's zero activity.
    """

    name = "person_activity"
    version = 1
    identity = True

    def day(self, src: Sources, day: str) -> dict[str, Any]:
        from liberational import segment_log

        lo, hi = day_bounds(day)
        hours = hour_bounds(day)
        groups: dict[str, dict[str, list[tuple[float, float]]]] = {}
        verified: dict[str, dict[str, bool]] = {}
        for span in src.action_spans(lo, hi):
            person, kind = span.get("person"), span.get("kind")
            if not isinstance(person, str) or not isinstance(kind, str):
                continue
            a, b = max(lo, float(span["start"])), min(hi, float(span["end"]))
            if b <= a:
                continue
            groups.setdefault(person, {}).setdefault(kind, []).append((a, b))
            flags = verified.setdefault(person, {})
            flags[kind] = flags.get(kind, True) and span.get("verified") is True
        people = {}
        for person, kinds in groups.items():
            people[person] = {}
            for kind, spans in kinds.items():
                joined = merged(spans)
                times = [min((max(a, ha) for a, b in joined if b > ha and a < hb), default=None) for ha, hb in hours]
                people[person][kind] = {
                    "seconds": per_hour(joined, hours),
                    "times": times,
                    "verified": verified[person][kind],
                }
        coverage = src.segment_rows(segment_log.ACTION_COVERAGE, lo, hi)
        cov = _hours_covered(((max(lo, c["ts_start"]), min(hi, c["ts_end"])) for c in coverage), hours)
        return {"people": people, "cov": cov}

    def assemble(self, days: Mapping[str, Mapping[str, Any]], src: Sources, arg: str | None) -> dict[str, Any]:
        names = src.people() or {}
        people = []
        for person in sorted({p for body in days.values() for p in body["people"]}):
            if person not in names or arg not in (None, person):
                continue
            series = []
            kinds = sorted({k for body in days.values() for k in body["people"].get(person, {})})
            for kind in kinds:
                rows: list[dict[str, Any]] = []
                verified = True
                for day, body in sorted(days.items()):
                    item = body["people"].get(person, {}).get(kind)
                    seconds = item["seconds"] if item else [0] * 24
                    times = item["times"] if item else [None] * 24
                    if item:
                        verified = verified and item["verified"]
                    hours = [{"minutes": round(s / 60, 2), "t": times[h]} if s else None for h, s in enumerate(seconds)]
                    known = [h for h in hours if h is not None]
                    first = [t for t in times if t is not None]
                    rows.append(
                        {
                            "day": day,
                            "minutes": round(sum(seconds) / 60, 2) if known else None,
                            "t": min(first) if first else None,
                            "hours": hours,
                        }
                    )
                series.append(
                    {
                        "key": kind,
                        "name": kind,
                        "total_min": round(sum(r["minutes"] or 0 for r in rows), 2),
                        "verified": verified,
                        "days": rows,
                    }
                )
            people.append({"id": person, "name": names[person], "series": series})
        return {"people": people, "covered_days": sum(any(b["cov"]) for b in days.values())}


class Devices:
    """Distinct device identifiers observed each hour, separated by radio and network source.

    A network row counts only when it explicitly says connected: cached/unknown and disconnected inventory rows
    are not fresh sightings. Coverage alone never establishes device absence. Totals sum hourly distinct counts,
    so they are device-hours, not unique devices across a day or people present.
    """

    name = "devices"
    version = 1
    identity = False

    def day(self, src: Sources, day: str) -> dict[str, Any]:
        from liberational import segment_log

        lo, hi = day_bounds(day)
        hours = hour_bounds(day)
        counts: dict[str, list[int]] = {}
        times: dict[str, list[float | None]] = {}
        cov: dict[str, list[bool]] = {}
        for key, stream, coverage in (
            ("bluetooth", segment_log.BLE_SIGHTINGS, segment_log.BLE_COVERAGE),
            ("network", segment_log.NET_OBSERVATIONS, segment_log.NET_INVENTORY),
        ):
            seen: list[set[str]] = [set() for _ in range(24)]
            first: list[float | None] = [None] * 24
            for row in src.segment_rows(stream, lo, hi):
                if key == "network" and row.get("connected") is not True:
                    continue
                identifier = row.get("identifier") or row.get("mac")
                if not identifier:
                    continue
                # BLE rows retain the first and last advertisement in a compacted run. Both endpoints are real
                # observations, including when the run crosses midnight; do not fill the time between them.
                for ts in (float(row["start"]), float(row["end"])):
                    h = hour_index(ts, hours)
                    if h is not None:
                        seen[h].add(str(identifier))
                        first[h] = min(first[h] or ts, ts)
            # Reuse the device reader's evidence gate: an inactive BLE sensor, or a partial/cached router poll,
            # cannot turn an hour with no positive sighting into a zero. Current BLE coverage explicitly withholds
            # negatives; its empty hours therefore remain unknown.
            from liberational.proximity.devices import _negative_coverage

            coverage_rows = [
                c for c in src.segment_rows(coverage, lo, hi) if _negative_coverage({**c, "stream": coverage})
            ]
            covered = _hours_covered(
                (
                    (max(lo, float(c["start"])), min(hi, float(c["end"])))
                    for c in coverage_rows
                    if float(c["end"]) > float(c["start"])
                ),
                hours,
            )
            # A zero-duration coverage row is one completed poll, not a fabricated interval around it.
            for c in coverage_rows:
                h = hour_index(float(c["start"]), hours)
                if h is not None:
                    covered[h] = True
            if any(seen) or any(covered):
                counts[key], times[key], cov[key] = [len(s) for s in seen], first, covered
        return {"counts": counts, "times": times, "cov": cov}

    def assemble(self, days: Mapping[str, Mapping[str, Any]], src: Sources, arg: str | None) -> dict[str, Any]:
        sources = _hour_series(days, arg)
        for source in sources:
            source["name"] = {"bluetooth": "Bluetooth", "network": "Network"}[source["key"]]
        return {
            "sources": sources,
            "covered_days": sum(any(any(c) for c in b["cov"].values()) for b in days.values()),
            "caveat": "Distinct identifiers observed each hour, not people or proof of absence. Totals are device-hours; Bluetooth addresses may rotate.",
        }


REPORTS: dict[str, Report] = {
    r.name: r for r in (WhosAround(), Conversations(), Activity(), Walkbys(), PersonActivity(), Devices())
}


# ---- the store ----


class Store:
    """state/reports.sqlite3: day aggregates, the last purge seen, and the Reports tab's layout."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS days (report TEXT, day TEXT, version INTEGER, final INTEGER,"
                " made REAL, body TEXT, PRIMARY KEY (report, day))"
            )
            db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS samples (ts REAL PRIMARY KEY, body TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS recorded (kind TEXT, day TEXT, hours TEXT, PRIMARY KEY (kind, day))")

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def days(self, report: str, version: int) -> dict[str, tuple[bool, float, dict[str, Any]]]:
        """{day: (final, when it was read, aggregate)} of the report's current version."""
        with self._lock, self._db() as db:
            rows = db.execute(
                "SELECT day, final, made, body FROM days WHERE report = ? AND version = ?", (report, version)
            ).fetchall()
        return {day: (bool(final), made, json.loads(body)) for day, final, made, body in rows}

    def put(self, report: str, version: int, day: str, final: bool, made: float, body: Mapping[str, Any]) -> None:
        with self._lock, self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO days VALUES (?, ?, ?, ?, ?, ?)",
                (report, day, version, int(final), made, json.dumps(body, separators=(",", ":"))),
            )

    def drop(self, days: Iterable[str] | None = None) -> None:
        """Forget the given days of every report, or every day, with what was recorded on them."""
        with self._lock, self._db() as db:
            if days is None:
                db.execute("DELETE FROM days")
                db.execute("DELETE FROM recorded")
            else:
                db.executemany("DELETE FROM days WHERE day = ?", [(d,) for d in days])
                db.executemany("DELETE FROM recorded WHERE day = ?", [(d,) for d in days])

    def unsettle(self, reports: Iterable[str]) -> None:
        """Mark every day of these reports for rereading, still shown until it's read again."""
        with self._lock, self._db() as db:
            db.executemany("UPDATE days SET final = 0, made = 0 WHERE report = ?", [(r,) for r in reports])

    def recorded(self, kind: str, day: str) -> list[bool] | None:
        with self._lock, self._db() as db:
            row = db.execute("SELECT hours FROM recorded WHERE kind = ? AND day = ?", (kind, day)).fetchone()
        return json.loads(row[0]) if row else None

    def set_recorded(self, kind: str, day: str, hours: Sequence[bool]) -> None:
        with self._lock, self._db() as db:
            db.execute("INSERT OR REPLACE INTO recorded VALUES (?, ?, ?)", (kind, day, json.dumps(list(hours))))

    def add_sample(self, ts: float, body: Mapping[str, Any]) -> None:
        """One System health sample (dashboard/report_health.py)."""
        with self._lock, self._db() as db:
            db.execute("INSERT OR REPLACE INTO samples VALUES (?, ?)", (ts, json.dumps(body, separators=(",", ":"))))

    def samples(self, since: float) -> list[tuple[float, dict[str, Any]]]:
        with self._lock, self._db() as db:
            rows = db.execute("SELECT ts, body FROM samples WHERE ts >= ? ORDER BY ts", (since,)).fetchall()
        return [(ts, json.loads(body)) for ts, body in rows]

    def drop_samples(self, before: float) -> None:
        with self._lock, self._db() as db:
            db.execute("DELETE FROM samples WHERE ts < ?", (before,))

    def meta(self, key: str) -> Any:
        with self._lock, self._db() as db:
            row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key: str, value: Any) -> None:
        with self._lock, self._db() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))


def read_by_reports(streams: Iterable[str]) -> bool:
    """Whether a range purge of these streams (by name) could take anything a report reads. No streams means all."""
    names = list(streams)
    return not names or any(not n.startswith(UNREAD_STREAM_PREFIXES) for n in names)


def purged_days(purges: Sequence[Mapping[str, Any]], window: tuple[str, str]) -> set[str] | None:
    """The days of window (first and last day, inclusive) that new purges touch, or None when one took ids or a
    person (it could be on any day). A purge's "streams" are the names of the streams it named (live_purges adds
    them); a range purge only of streams no report reads touches nothing. Each range is clipped to the window and
    the ranges are joined before they become days, so this costs at most the window's days."""
    lo_w, hi_w = day_bounds(window[0])[0], day_bounds(window[1])[1]
    spans: list[tuple[float, float]] = []
    for p in purges:
        span = p.get("range")
        if not (isinstance(span, list | tuple) and len(span) == 2 and all(isinstance(x, int) for x in span)):
            return None
        lo, hi = span[0] / 1e9, span[1] / 1e9
        if not (math.isfinite(lo) and math.isfinite(hi) and hi >= lo):
            return None
        names = p.get("streams")
        if isinstance(names, list | tuple) and not read_by_reports(str(n) for n in names):
            continue
        lo, hi = max(lo, lo_w), min(hi, hi_w)
        if hi > lo or (hi == lo and lo_w <= lo < hi_w):
            spans.append((lo, hi))
    out: set[str] = set()
    for lo, hi in merged(spans):
        out.update(days_between(day_of(lo), day_of(max(lo, hi - 1e-3))))
    return out


# ---- reading and building ----


class _MissingPurgeCursor(Exception):
    """The broker no longer has a purge cursor, so cached aggregates cannot be validated against its history."""


class Reports:
    """The Reports tab's reads: each report's figures from the stored days, with the days still to read built in
    the background (one thread, newest day first)."""

    def __init__(
        self,
        store: Store,
        src: Sources,
        *,
        purges: Callable[[str | None], tuple[list[dict[str, Any]], str | None]] | None = None,
        now: Callable[[], float] = time.time,
        reports: Mapping[str, Report] = REPORTS,
        background: bool = True,
    ) -> None:
        self.store, self.src, self.now, self.reports = store, src, now, reports
        self._purges = purges
        self.background = background  # off: requested days only queue, and drain() builds them (tests)
        self._queue: list[tuple[str, str]] = []
        self._wake = threading.Condition()
        self._thread: threading.Thread | None = None
        self.pause_s = 0.05  # between days, so a rebuild of a year leaves the dashboard its share of the machine
        # Bumped by every invalidation. A day read across one is not stored (it may hold what was just taken away or
        # re-resolved); it's queued again instead.
        self._generation = 0
        self._publish = threading.Lock()
        self._read = _SettledHours(self)
        self.state: Path | None = None  # set by at(): where the worker process reads days from (READ_IN_WORKER)

    def days(self) -> list[str]:
        """Every day a report covers: from the first recording (at most MAX_DAYS back) to today."""
        first, today = self.window()
        return days_between(first, today)

    def window(self) -> tuple[str, str]:
        """The first and last day a report covers (see days)."""
        today = day_of(self.now())
        first = self.src.first_day() or today
        earliest = (dt.date.fromisoformat(today) - dt.timedelta(days=MAX_DAYS - 1)).isoformat()
        return (max(first, earliest), today) if first <= today else (today, today)

    def _check_purges(self) -> None:
        if self._purges is None:
            return
        seen = self.store.meta("purge")
        try:
            new, newest = self._purges(seen)
        except _MissingPurgeCursor:
            with self._publish:
                if self.store.meta("purge") != seen:
                    return  # another reader already retired this cursor
                self._generation += 1
                self.store.drop()
                self.store.set_meta("purge", None)
            LOG.warning("saved purge cursor no longer exists: cached reports cleared; next check starts fresh")
            return
        if newest == seen:
            return
        if seen is None and not new:
            self.store.set_meta("purge", newest)
            return
        touched = purged_days(new, self.window())
        if touched != set():
            with self._publish:
                self._generation += 1
                self.store.drop(touched)
            LOG.info(f"purges since the last read: {'every day' if touched is None else f'{len(touched)} days'} reread")
        self.store.set_meta("purge", newest)

    def _check_identity(self) -> None:
        mark = self.src.identity_mark()
        seen = self.store.meta("identity")
        if mark == seen:
            return
        if seen is not None:
            with self._publish:
                self._generation += 1
                self.store.unsettle(r.name for r in self.reports.values() if r.identity)
            LOG.info("identities or activity evidence changed: dependent reports are read again")
        self.store.set_meta("identity", mark)

    def stale(self, day: str, final: bool, made: float) -> bool:
        return not final and self.now() - made > FRESH_S

    def view(self, name: str, arg: str | None = None) -> dict[str, Any]:
        """{"days": how many days the report covers, "pending": days not read yet (newest first), the figures}."""
        report = self.reports[name]
        try:
            self._check_purges()
        except Exception as e:  # a purge check that fails leaves the stored days as they are, until the next read
            LOG.warning(f"purges not checked: {type(e).__name__}: {e}")
        if report.identity:
            try:
                self._check_identity()
            except Exception as e:
                LOG.warning(f"identity/activity evidence changes not checked: {type(e).__name__}: {e}")
        wanted = self.days()
        stored = self.store.days(report.name, report.version)
        pending = [d for d in reversed(wanted) if d not in stored or self.stale(d, *stored[d][:2])]
        if pending:
            self.request(name, pending)
        have = {d: stored[d][2] for d in wanted if d in stored}
        return {
            "report": name,
            "first": wanted[0],
            "last": wanted[-1],
            "pending": [d for d in pending if d not in stored],
            "refreshing": [d for d in pending if d in stored],
            **report.assemble(have, self.src, arg),
        }

    def build(self, name: str, day: str) -> bool:
        """Read one day of one report and store it; False (and queued again) when an invalidation came while it was
        being read."""
        report = self.reports[name]
        final = day_bounds(day)[1] + SETTLE_S < self.now()
        generation = self._generation
        body = self._day(name, day)
        with self._publish:
            if self._generation == generation:
                self.store.put(report.name, report.version, day, final, self.now(), body)
                return True
        self.request(name, [day])
        return False

    def _day(self, name: str, day: str) -> dict[str, Any]:
        """One day of one report, read in the worker process when the dashboard runs one (READ_IN_WORKER)."""
        if not READ_IN_WORKER or self.state is None:
            return self.reports[name].day(self._read, day)
        return _in_worker(self.state, name, day)

    def request(self, name: str, days: Sequence[str]) -> None:
        with self._wake:
            queued = set(self._queue)
            self._queue.extend((name, d) for d in days if (name, d) not in queued)
            self._wake.notify()
            if self.background and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, daemon=True, name="reports-build")
                self._thread.start()

    def _run(self) -> None:
        while True:
            with self._wake:
                if not self._queue:
                    self._wake.wait(timeout=60)
                    if not self._queue:
                        self._thread = None
                        return
                # Newest first across reports: today's page fills in before last year's.
                self._queue.sort(key=lambda job: job[1], reverse=True)
                name, day = self._queue.pop(0)
            try:
                self.build(name, day)
            except Exception as e:
                LOG.warning(f"{name} {day} not read: {type(e).__name__}: {e}")
            time.sleep(self.pause_s)

    def drain(self) -> None:
        """Build everything queued now, on this thread (tests)."""
        while self._queue:
            name, day = self._queue.pop(0)
            self.build(name, day)


class _SettledHours:
    """The Sources a day is read from: the real ones, with each settled day's recorded hours kept in the store, so
    rereading a day (after a labeling, say) doesn't ask the broker about every hour again."""

    def __init__(self, owner: Reports) -> None:
        self.owner = owner

    def people(self) -> Mapping[str, str] | None:
        return self.owner.src.people()

    def seen(self, start: float, end: float) -> list[Seen]:
        return self.owner.src.seen(start, end)

    def recorded_hours(self, kind: str, day: str) -> list[bool]:
        kept = self.owner.store.recorded(kind, day)
        if kept is not None:
            return kept
        hours = self.owner.src.recorded_hours(kind, day)
        if day_bounds(day)[1] + SETTLE_S < self.owner.now():
            self.owner.store.set_recorded(kind, day, hours)
        return hours

    def first_day(self) -> str | None:
        return self.owner.src.first_day()

    def identity_mark(self) -> str | None:
        return self.owner.src.identity_mark()

    def events(self, start: float, end: float) -> list[dict[str, Any]]:
        return self.owner.src.events(start, end)

    def activity_coverage(self, start: float, end: float) -> list[tuple[str, float, float]]:
        return self.owner.src.activity_coverage(start, end)

    def zone_visits(self, start: float, end: float) -> list[dict[str, Any]]:
        return self.owner.src.zone_visits(start, end)

    def zone_coverage(self, start: float, end: float) -> list[dict[str, Any]]:
        return self.owner.src.zone_coverage(start, end)

    def action_spans(self, start: float, end: float) -> list[dict[str, Any]]:
        return self.owner.src.action_spans(start, end)

    def segment_rows(self, stream: str, start: float, end: float) -> list[dict[str, Any]]:
        return self.owner.src.segment_rows(stream, start, end)


# ---- the live sources ----


class LiveSources:
    """Sources over this machine's records: the timeline for people, and the broker for what was recorded."""

    def __init__(self, state: Path) -> None:
        self.state = state
        self._first: tuple[float, str | None] = (0.0, None)
        self._coverage: dict[tuple[float, float], tuple[float, list[tuple[str, float, float]]]] = {}

    def _timeline(self) -> Any:
        from liberational.memory import timeline

        return timeline.load(self.state) if timeline.enabled() else None

    def people(self) -> Mapping[str, str] | None:
        tl = self._timeline()
        return dict(tl.names) if tl is not None else None

    def seen(self, start: float, end: float) -> list[Seen]:
        tl = self._timeline()
        if tl is None:
            return []
        out = []
        for e in tl.encounters(None, start, end):
            out += [Seen(e.person, v.start, v.end, False) for v in e.visits if v.end >= start and v.start < end]
            out += [Seen(e.person, ln.start, ln.end, True, ln.text) for ln in e.lines if start <= ln.start < end]
        return out

    def _streams(self, kind: str) -> list[str]:
        from liberational import speech_log, vision_log

        if kind == "video":
            return [f"{cam}.video" for cam in vision_log.at(self.state).video_cameras()]
        return sorted(speech_log.shared().mic_freshness())

    def recorded_hours(self, kind: str, day: str) -> list[bool]:
        """One one-item query per stream and hour, so a day of frames is never listed."""
        import continuum_py

        from liberational import vision_log
        from liberational.continuum import to_ns

        hours = hour_bounds(day)
        streams = self._streams(kind)

        async def go(client: Any) -> list[bool]:
            out = [False] * 24
            for stream in streams:
                for h, (a, b) in enumerate(hours):
                    if out[h] or b <= a:
                        continue
                    try:
                        found = await client.query(stream, to_ns(a), to_ns(b), all=False, max_items=1)
                    except continuum_py.ContinuumError:
                        break
                    out[h] = bool(found["items"])
            return out

        return vision_log.at(self.state).link.call(go)

    def first_day(self) -> str | None:
        """The first recording, timeline, or report segment day, looked up at most every ten minutes.

        The segment store's UTC hour filenames are its index. Inspect only report inputs, without loading their
        rows, so device-only history is available even when no camera or microphone was ever recorded.
        """
        at, first = self._first
        if time.monotonic() - at < 600 and first is not None:
            return first
        from liberational import segment_log, vision_log

        found = []
        try:
            days = vision_log.at(self.state).video_days()
            found += days[:1]
        except Exception as e:
            LOG.warning(f"recorded days not read: {type(e).__name__}: {e}")
        tl = self._timeline()
        if tl is not None and tl.entries:
            found.append(day_of(tl.entries[0].start))
        for stream in (
            segment_log.WALKBY_SPANS,
            segment_log.WALKBY_COVERAGE,
            segment_log.ACTION_SPANS,
            segment_log.ACTION_COVERAGE,
            segment_log.ZONE_VISITS,
            segment_log.ZONE_COVERAGE,
            segment_log.NET_OBSERVATIONS,
            segment_log.NET_INVENTORY,
            segment_log.BLE_SIGHTINGS,
            segment_log.BLE_COVERAGE,
        ):
            for part in (segment_log.root(self.state) / stream).glob("*/*.arrows"):
                match = re.fullmatch(r"(\d{10})-\d+-s\d+(?:-\d+(?:-v\d+)?)?\.arrows", part.name)
                if match is None or not part.is_file():
                    continue
                try:
                    hour = dt.datetime.strptime(match[1], "%Y%m%d%H").replace(tzinfo=dt.UTC)
                except ValueError:
                    continue  # a file outside the segment store's hour naming contract is not an index entry
                found.append(day_of(hour.timestamp()))
        first = min(found) if found else None
        self._first = (time.monotonic(), first)
        return first

    def identity_mark(self) -> str | None:
        from liberational import config
        from liberational.memory import events
        from liberational.people import registry

        have = registry.log_exists(self.state)
        identities = registry.at(self.state).mark() if have else None
        activity = events._store(self.state).revision((events.FEEDBACK, events.NAMED)) if have else 0
        promotion = tuple(
            config.get(key)
            for key in ("events.answer_counts", "activity.promotion_min_ratings", "activity.promotion_min_precision")
        )
        return str((identities, activity, promotion))

    def events(self, start: float, end: float) -> list[dict[str, Any]]:
        from liberational.memory import events

        return events.events(self.state, start, end)

    def activity_coverage(self, start: float, end: float) -> list[tuple[str, float, float]]:
        """The cameras' activity coverage, kept COVERAGE_KEEP_S: Activity and Walk-bys each read a day's, about 1.4
        million spans and 10 s of Python a day, one right after the other."""
        from liberational.memory import events

        now = time.monotonic()
        kept = self._coverage.get((start, end))
        if kept is not None and now - kept[0] < COVERAGE_KEEP_S:
            return kept[1]
        spans = events.coverage(self.state, start, end)
        self._coverage = {k: v for k, v in self._coverage.items() if now - v[0] < COVERAGE_KEEP_S}
        self._coverage[(start, end)] = (now, spans)
        return spans

    def zone_visits(self, start: float, end: float) -> list[dict[str, Any]]:
        from liberational import segment_log

        return segment_log.at(self.state).range(segment_log.ZONE_VISITS, start, end)

    def zone_coverage(self, start: float, end: float) -> list[dict[str, Any]]:
        import numpy as np

        from liberational import segment_log

        # Each kind of subject's joined stretches, a camera at a time: what the reports read is whether an hour was
        # looked at, for any subject and for animals.
        log = segment_log.at(self.state)
        kinds = {
            k
            for c in log.table(segment_log.ZONE_COVERAGE, start, end, ("subject_kind",)).values()
            for k in c["subject_kind"]
        }
        out = []
        for kind in sorted(kinds):
            of_kind = (
                ("subject_kind",),
                lambda c, kind=kind: np.array([k == kind for k in c["subject_kind"]], dtype=bool),
            )
            for cam, found in log.covered(segment_log.ZONE_COVERAGE, start, end, where=of_kind).items():
                out += [{"cam": cam, "subject_kind": kind, "ts_start": a, "ts_end": b} for a, b in found.spans]
        return sorted(out, key=lambda r: (r["ts_start"], r["cam"], r["subject_kind"]))

    def action_spans(self, start: float, end: float) -> list[dict[str, Any]]:
        from liberational import segment_log
        from liberational.domain.kernel import PersonId
        from liberational.memory import events
        from liberational.people import registry

        promoted, unverified = events.count_evidence(self.state, start, end)
        rows = [
            {**row, "verified": verified}
            for rows, verified in ((promoted, True), (unverified, False))
            for row in rows
            if row.get("stream") == segment_log.ACTION_SPANS
        ]
        if not registry.log_exists(self.state):
            return [{**row, "person": None} for row in rows]
        reg = registry.at(self.state)
        with reg.lock:
            reg.refresh(registry.PEOPLE)
            return [
                {**row, "person": reg.resolve(PersonId(str(row["person"]))) if row.get("person") else None}
                for row in rows
            ]

    def segment_rows(self, stream: str, start: float, end: float) -> list[dict[str, Any]]:
        from liberational import segment_log

        return segment_log.at(self.state).range(stream, start, end)


def live_purges(state: Path) -> Callable[[str | None], tuple[list[dict[str, Any]], str | None]]:
    """A reader of the broker's purges after a purge id: (the new purges, the newest purge id)."""

    names: dict[int, str] = {}  # stream id to name; ids are never reused, so read again only for one not known yet

    def read(after: str | None) -> tuple[list[dict[str, Any]], str | None]:
        """Each purge also gets "streams": the names of the streams it named (purged_days skips the ones no report
        reads). An id the broker doesn't list is kept as its number, and counts as a stream reports read."""
        from continuum_py import BrokerError

        from liberational import vision_log

        async def go(client: Any) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            cursor = after
            while True:
                try:
                    page = await client.list_purges(after=cursor, limit=256)
                except BrokerError as error:
                    if cursor is None or not error.args or error.args[0] != "NotFound":
                        raise
                    raise _MissingPurgeCursor(cursor) from error
                if not page:
                    break
                out += page
                if page[-1]["purge_id"] == cursor:
                    break
                cursor = page[-1]["purge_id"]
            if any(i not in names for p in out for i in p.get("stream_ids") or ()):
                names.update({int(s["stream_id"]): str(s["name"]) for s in await client.list_streams()})
            for p in out:
                p["streams"] = [names.get(i, str(i)) for i in p.get("stream_ids") or ()]
            return out

        found = vision_log.at(state).link.call(go)
        return found, (found[-1]["purge_id"] if found else after)

    return read


_shared: dict[str, Reports] = {}
_shared_lock = threading.Lock()


def at(state: Path) -> Reports:
    """The Reports reader for a state directory, one per process."""
    with _shared_lock:
        key = str(state)
        if key not in _shared:
            _shared[key] = Reports(Store(state / "reports.sqlite3"), LiveSources(state), purges=live_purges(state))
            _shared[key].state = state
        return _shared[key]


# ---- the worker process ----

COVERAGE_KEEP_S = 60.0  # a day's activity coverage is read once for the reports built one after another
DAY_TIMEOUT_S = 900.0  # a worker that takes longer than this over one day is stopped, and the next read starts another
READ_IN_WORKER: bool = False  # the dashboard turns it on: tests and other importers read days in their own process
_executor: ProcessPoolExecutor | None = None
_executor_lock = threading.Lock()


def _read_day(state: str, name: str, day: str) -> dict[str, Any]:
    """One day of one report, in the worker: its own Reports over the same store, which never builds or queues."""
    if multiprocessing.parent_process() is not None:
        from liberational import config as cfg

        cfg.load.cache_clear()  # a setting saved since: the dashboard rereads its pack.toml, and so does this
        from liberational import memtrim

        memtrim.start()  # what a day's read freed goes back to the system
    reader = at(Path(state))
    return reader.reports[name].day(reader._read, day)


def _in_worker(state: Path, name: str, day: str) -> dict[str, Any]:
    """Read one day in one process of its own. A day of all six reports is about 35 s of pure Python, and on
    the dashboard's own thread it held the interpreter lock every request needs; today is read again every FRESH_S
    while Reports is open, and every day after a purge or a naming. A worker that died or ran past DAY_TIMEOUT_S is
    replaced, and this day raises (the builder logs it, and the next view asks for it again)."""
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=proc_table.ignore_interrupt
            )
        executor = _executor
    try:
        return executor.submit(_read_day, str(state), name, day).result(timeout=DAY_TIMEOUT_S)
    except BrokenProcessPool, TimeoutError:
        with _executor_lock:
            if _executor is executor:
                _executor = None
        for process in list((getattr(executor, "_processes", None) or {}).values()):
            process.terminate()  # one still running past the timeout
        executor.shutdown(wait=False, cancel_futures=True)
        raise
