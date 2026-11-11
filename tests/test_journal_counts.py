"""The journal's code-built Counts tables: what the owner has named, counted from the count modules over a
day or a week, with unwatched time unknown and never zero, and "Usual" from earlier periods with enough footage.

Zones run through the real zone and event count modules over an in-memory segment store; named events through
event_counts with a stand-in for memory/events.py (its events() and coverage(), as test_event_counts does). All
dates are in October 2026, clear of the 1 November DST change, so every day is 24 hours.
"""

import datetime as dt
import types
from zoneinfo import ZoneInfo

import pytest

from liberational import vision_log
from liberational.memory import event_counts, history_qa, journal_counts, zone_counts
from liberational.see import zones

LA = ZoneInfo("America/Los_Angeles")
DAY = dt.date(2026, 10, 23)  # a Friday
MONDAY = dt.date(2026, 10, 12)
H = 3600.0
BOX = [[0.1, 0.6], [0.3, 0.6], [0.3, 0.9], [0.1, 0.9]]


def midnight(day: dt.date) -> float:
    return dt.datetime.combine(day, dt.time(), LA).timestamp()


class FakeEvents(types.ModuleType):
    """memory/events.py as a stand-in: the spans it found and the recordings it covered, set per test."""

    def __init__(self) -> None:
        super().__init__("liberational.memory.events")
        self.found: list[dict] = []
        self.covered: list[tuple[str, float, float]] = []

    def events(self, state, start, end):
        return [e for e in self.found if e["start"] < end and e["end"] > start]

    def coverage(self, state, start, end):
        return [c for c in self.covered if c[1] < end and c[2] > start]

    def named(self, state):
        return [(e["name"], e.get("kind")) for e in self.found if e.get("name") and e["name"] != e.get("kind")]


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(history_qa, "TZ", LA)
    monkeypatch.setattr(event_counts, "_NAMES", {})
    monkeypatch.setattr(zone_counts, "_NAMES", {})


@pytest.fixture
def events(monkeypatch):
    fake = FakeEvents()
    monkeypatch.setattr(event_counts, "module", lambda: fake)
    return fake


@pytest.fixture
def litter(tmp_path, vision, events):
    """A zone the owner drew and named "Litter box", with helpers to add checked footage and a cat's visits."""
    d = zones.define(tmp_path, "cam0", zones.parse_zones([{"name": "Litter box", "polygon": BOX}]), None, now=0.0)
    zone = d.zones[0].id

    def cover(a: float, b: float) -> None:
        vision.add(
            vision_log.ZONE_COVERAGE,
            ts_start=a,
            ts_end=b,
            cam="cam0",
            subject_kind="animal",
            definition=d.id,
            zones=[zone],
            frames=1000,
        )

    def visit(at: float, seconds: float = 120.0) -> None:
        vision.add(
            vision_log.ZONE_VISITS,
            ts_start=at,
            ts_end=at + seconds,
            cam="cam0",
            zone=zone,
            zone_name="Litter box",
            definition=d.id,
            subject_kind="animal",
            track_id=f"run:cam0:{at}",
            label="cat",
            dwell_s=seconds,
            frames=10,
            entry_seen=True,
            exit_seen=True,
            ended="left",
            candidate=True,
        )

    return tmp_path, cover, visit


def test_a_watched_day_counts_visits_and_their_time_with_the_usual_from_earlier_days(litter):
    state, cover, visit = litter
    today = midnight(DAY)
    cover(today, today + 24 * H)
    for hour in (7, 13, 21):
        visit(today + hour * H)  # three visits of 2 min each
    # Earlier days: three watched whole (2, 0, and 4 visits), one watched only 2 h, which is too little to compare.
    for back, visits in ((1, 2), (2, 0), (3, 4)):
        day = midnight(DAY - dt.timedelta(days=back))
        cover(day, day + 24 * H)
        for k in range(visits):
            visit(day + (8 + k) * H)
    thin = midnight(DAY - dt.timedelta(days=4))
    cover(thin, thin + 2 * H)
    visit(thin + H)
    assert journal_counts.day_counts(DAY, state=state) == [
        journal_counts.CountRow("Litter box", "3 visits, 6 min", "2 (0 to 4, 3 days)")
    ]


def test_a_partly_watched_day_says_how_much_was_watched(litter):
    state, cover, visit = litter
    today = midnight(DAY)
    cover(today + 8 * H, today + 17 * H)
    visit(today + 10 * H, seconds=300)
    (row,) = journal_counts.day_counts(DAY, state=state)
    assert (row.value, row.usual) == ("1 visit, 5 min (watched 9 h of 24 h)", "")


def test_today_so_far_counts_up_to_now_and_reads_the_hours_to_come_as_not_yet_unwatched(litter):
    state, cover, visit = litter
    today = midnight(DAY)
    cover(today, today + 12 * H)  # watched all morning, and it is noon
    visit(today + 10 * H)
    visit(today + 15 * H)  # after now: a row this test plants, which a count up to now leaves out
    (row,) = journal_counts.day_counts(DAY, state=state, until=today + 12 * H)
    assert row.value == "1 visit, 2 min"


def test_a_day_nobody_watched_is_unknown_never_zero(litter):
    state, _, visit = litter
    visit(midnight(DAY) + 10 * H)  # a visit row with no checked footage under it
    (row,) = journal_counts.day_counts(DAY, state=state)
    assert row.value == "unknown (not watched)"


def test_a_watched_day_with_no_visit_is_a_real_zero(litter):
    state, cover, _ = litter
    cover(midnight(DAY), midnight(DAY) + 24 * H)
    (row,) = journal_counts.day_counts(DAY, state=state)
    assert row.value == "0 visits"


def span(start: float, minutes: float, name: str, kind: str = "model_action") -> dict:
    return {"id": f"ev-{start:.0f}", "start": start, "end": start + minutes * 60, "camera": "cam0", "count": None,
            "name": name, "segment": "cam0/x.mp4", "person": None, "kind": kind}  # fmt: skip


def test_only_what_the_owner_named_is_counted_events_first_then_zones(litter, events):
    state, cover, _ = litter
    today = midnight(DAY)
    cover(today, today + 24 * H)
    events.covered = [("cam0", today, today + 24 * H)]
    events.found = [
        span(today + 8 * H, 20, "fed the cat"),
        span(today + 18 * H, 25, "fed the cat"),
        span(today + 12 * H, 1, "model_action"),  # the detector's own kind as its name: nobody named it
        span(today + 9 * H, 0, "Doorbell"),  # a point event: no time to add up
    ]
    rows = journal_counts.day_counts(DAY, state=state)
    assert [(r.what, r.value) for r in rows] == [
        ("Doorbell", "1 time"),
        ("fed the cat", "2 times, 45 min"),
        ("Litter box", "0 visits"),
    ]


def test_a_name_the_count_modules_refuse_is_unknown_not_unwatched(events, tmp_path):
    today = midnight(DAY)
    events.covered = [("cam0", today, today + 24 * H)]
    # Repeated movement is a retired kind the camera model doesn't detect, so event_counts won't count it.
    events.found = [span(today + 9 * H, 0, "repeated_movement")]
    (row,) = journal_counts.day_counts(DAY, state=tmp_path)
    assert (row.what, row.value) == ("repeated_movement", "unknown")


def test_a_week_counts_its_seven_days_with_the_usual_from_the_weeks_before(events, tmp_path):
    week = midnight(MONDAY)
    events.covered = [("cam0", week - 4 * 7 * 24 * H, week + 7 * 24 * H)]  # four weeks back and this one, all watched
    for back, n in ((4, 3), (3, 5), (2, 1), (1, 4)):
        start = midnight(MONDAY - dt.timedelta(weeks=back))
        events.found += [span(start + (24 * k + 9) * H, 0, "fed the cat") for k in range(n)]
    events.found += [span(week + (24 * k + 9) * H, 0, "fed the cat") for k in range(6)]
    events.found.append(span(week + 7 * 24 * H + 9 * H, 0, "fed the cat"))  # next Monday: not this week
    assert journal_counts.week_counts(MONDAY, state=tmp_path) == [
        journal_counts.CountRow("fed the cat", "6 times", "3.5 (1 to 5, 4 weeks)")
    ]


def test_a_week_with_too_little_footage_is_left_out_of_the_usual(events, tmp_path):
    week = midnight(MONDAY)
    one_back = midnight(MONDAY - dt.timedelta(weeks=1))
    events.covered = [("cam0", one_back, one_back + 24 * H), ("cam0", week, week + 7 * 24 * H)]  # 1 of 7 days
    events.found = [span(one_back + 9 * H, 0, "fed the cat"), span(week + 9 * H, 0, "fed the cat")]
    (row,) = journal_counts.week_counts(MONDAY, state=tmp_path)
    assert (row.value, row.usual) == ("1 time", "")


def test_nothing_named_means_no_rows_and_no_table(events, tmp_path):
    assert journal_counts.day_counts(DAY, state=tmp_path) == []
    assert journal_counts.render_counts_table([], "day") == ""


def test_the_table_is_written_as_the_rows_say_with_no_cell_breaking_a_row():
    rows = [
        journal_counts.CountRow("Litter box", "3 visits, 6 min", "2 (0 to 4, 3 days)"),
        journal_counts.CountRow("in | out", "unknown (not watched)"),
    ]
    assert journal_counts.render_counts_table(rows, "week") == (
        "| What | Week | Usual |\n"
        "| --- | ---: | --- |\n"
        "| Litter box | 3 visits, 6 min | 2 (0 to 4, 3 days) |\n"
        "| in / out | unknown (not watched) |  |"
    )
