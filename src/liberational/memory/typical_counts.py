"""What's typical for a count, over days or nights (#1318): "how many times do I generally get up to use the bathroom
at night", "how often does the cat usually use the litter box".

history_qa sends a count question here when it names a detected event or a zone (memory/event_counts.py) and asks
what is usual rather than what happened once ("generally", "usually", "on average", "per night"). The window is cut
into periods, a local day or a night (NIGHT_HOURS, 22:00 to 07:00 so an early-morning trip counts), and each period is
counted the way a single count question is (event_counts.answer, which hands zones to zone_counts). The answer is the
median and the range over the periods with enough checked footage (MIN_COVERED of the period), and which days those
were. A period without enough footage is left out and said so, never counted as zero.

With no window in the question (or one shorter than two periods) it reads the last DEFAULT_PERIODS whole periods.
"""

import datetime as dt
import re
import statistics
from collections.abc import Callable, Mapping
from typing import Any

TYPICAL = re.compile(
    r"\b(?:generally|usually|typically|normally|on average|average|in general|on a typical|most (?:days|nights)"
    r"|per (?:day|night)|a (?:day|night)|each (?:day|night)|every (?:day|night))\b",
    re.I,
)
NIGHT = re.compile(r"\b(?:at night|nights?|overnight|nightly)\b", re.I)
NIGHT_HOURS = (22, 31)  # local hours [start, end) of the night that starts on a date: 22:00 to 07:00 the next day
DEFAULT_PERIODS = 14
MIN_COVERED = 0.5  # a period counts when at least this much of it was checked

#: event_counts.answer: (asked, query, state, stamp, tz, person=...) -> {answer, evidence, coverage, warnings, count}.
Count = Callable[..., dict[str, Any]]


def asks(question: str) -> bool:
    """Whether a count question asks what is typical over days or nights."""
    return TYPICAL.search(question) is not None


def periods(start: float, end: float, now: float, tz: dt.tzinfo | None, night: bool) -> list[tuple[float, float]]:
    """The whole days (or nights) in [start, end), or the last DEFAULT_PERIODS before `now` when that holds fewer
    than two. A period still going on at `now` isn't whole, so it isn't one."""
    first, last = NIGHT_HOURS if night else (0, 24)

    def period(day: dt.date) -> tuple[float, float]:
        base = dt.datetime.combine(day, dt.time(), tz)
        return (base + dt.timedelta(hours=first)).timestamp(), (base + dt.timedelta(hours=last)).timestamp()

    def back(n: int) -> list[tuple[float, float]]:
        day = dt.datetime.fromtimestamp(now, tz).date()
        out: list[tuple[float, float]] = []
        while len(out) < n:
            a, b = period(day)
            if b <= now:
                out.append((a, b))
            day -= dt.timedelta(days=1)
        return out[::-1]

    end = min(end, now)
    if start > 0:
        day = dt.datetime.fromtimestamp(start, tz).date() - dt.timedelta(days=1)
        inside = []
        while (p := period(day))[0] < end:
            if p[0] >= start and p[1] <= end:
                inside.append(p)
            day += dt.timedelta(days=1)
        if len(inside) >= 2:
            return inside
    return back(DEFAULT_PERIODS)


def _covered_seconds(coverage: Mapping[str, Any], start: float, end: float) -> float:
    """How much of [start, end) the count's coverage spans hold, overlaps counted once."""
    spans = sorted(
        (max(float(c["start"]), start), min(float(c["end"]), end))
        for c in coverage.get("covered") or []
        if float(c["end"]) > start and float(c["start"]) < end
    )
    total, reach = 0.0, start
    for a, b in spans:
        a = max(a, reach)
        if b > a:
            total += b - a
            reach = b
    return total


def _number(x: float) -> str:
    return f"{x:g}"


def answer(
    asked: str,
    query: Mapping[str, Any],
    state: Any,
    stamp: Callable[[float], str],
    tz: dt.tzinfo | None,
    now: float,
    count: Count,
    person: Any = None,
) -> dict[str, Any]:
    """{answer, evidence, coverage, warnings} for what is typical for `asked` per day or night."""
    question = str(query.get("question") or "")
    night = NIGHT.search(question) is not None
    unit, units = ("night", "nights") if night else ("day", "days")
    found = periods(float(query["start"]), float(query["end"]), now, tz, night)
    counted: list[tuple[float, float, int]] = []
    missing: list[tuple[float, float]] = []
    evidence: list[dict[str, Any]] = []
    for a, b in found:
        r = count(asked, {**query, "start": a, "end": b}, state, stamp, tz, person=person)
        n = r.get("count")
        if n is None or _covered_seconds(r.get("coverage") or {}, a, b) < MIN_COVERED * (b - a):
            missing.append((a, b))
            continue
        counted.append((a, b, int(n)))
        evidence.extend(r.get("evidence") or [])

    def day_of(a: float) -> str:
        return dt.datetime.fromtimestamp(a, tz).date().isoformat()

    coverage = {
        "periods": [{"start": a, "end": b, "count": n} for a, b, n in counted],
        "missing": [{"start": a, "end": b} for a, b in missing],
        "unit": unit,
        "min_covered": MIN_COVERED,
    }
    if not counted:
        return dict(
            answer=f"None of the last {len(found)} {units} had enough checked footage to count {asked}, so I can't say"
            " what's typical.",
            evidence=[],
            coverage=coverage,
            warnings=["No period had enough coverage: this is not a typical count of zero."],
        )
    values = [n for _, _, n in counted]
    median = statistics.median(values)
    span = f"{day_of(counted[0][0])} to {day_of(counted[-1][0])}"
    which = "night starting" if night else "day"
    text = (
        f"{asked}: typically {_number(median)} a {unit} (median), from {min(values)} to {max(values)}, over "
        f"{len(counted)} {units if len(counted) != 1 else unit} with footage ({span}, each {which} counted"
        + (" 22:00 to 07:00)." if night else ").")
    )
    if missing:
        text += (
            f" {len(missing)} {units if len(missing) != 1 else unit} didn't have enough checked footage and aren't"
            " counted."
        )
    per = ", ".join(f"{day_of(a)}: {n}" for a, _, n in counted)
    text += f" By {unit}: {per}."
    return dict(answer=text, evidence=evidence, coverage=coverage, warnings=[])
