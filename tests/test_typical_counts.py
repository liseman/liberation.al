"""Typical counts over days or nights for Ask (#1318): "how many times do I generally get up to use the bathroom at
night"."""

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from liberational import vision_log
from liberational.memory import event_counts, typical_counts, zone_counts
from liberational.memory import history_qa as h
from liberational.see import zones

LA = ZoneInfo("America/Los_Angeles")
DAY = dt.datetime(2026, 10, 23, tzinfo=LA).timestamp()  # local midnight
NOW = DAY + 20 * 3600
BOX = [[0.1, 0.6], [0.3, 0.6], [0.3, 0.9], [0.1, 0.9]]
QUESTION = "how many times do I generally get up to use the bathroom at night"
H = 3600.0


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(h, "TZ", LA)
    zone_counts._NAMES.clear()
    event_counts._NAMES.clear()


def night(n: int) -> float:
    """22:00 local on the date n days before DAY."""
    return dt.datetime.fromtimestamp(DAY, LA).replace(hour=22).timestamp() - n * 86400.0


@pytest.fixture
def bathroom(tmp_path, vision):
    d = zones.define(tmp_path, "cam0", zones.parse_zones([{"name": "Bathroom", "polygon": BOX}]), None, now=DAY)
    zone = d.zones[0].id

    def cover(a: float, b: float) -> None:
        vision.add(
            vision_log.ZONE_COVERAGE,
            ts_start=a,
            ts_end=b,
            cam="cam0",
            subject_kind="person",
            definition=d.id,
            zones=[zone],
            frames=1000,
        )

    def visit(at: float) -> None:
        vision.add(
            vision_log.ZONE_VISITS,
            ts_start=at,
            ts_end=at + 120,
            cam="cam0",
            zone=zone,
            zone_name="Bathroom",
            definition=d.id,
            subject_kind="person",
            track_id=f"run:cam0:{at}",
            label="person",
            dwell_s=120.0,
            frames=10,
            entry_seen=True,
            exit_seen=True,
            ended="left",
            candidate=True,
        )

    return tmp_path, cover, visit


def test_a_typical_night_is_the_median_over_the_nights_with_footage(bathroom):
    state, cover, visit = bathroom
    # Three nights checked whole: 2, 0, and 3 trips. One night checked for two hours: not enough to count.
    for n in (1, 2, 3):
        cover(night(n), night(n) + 9 * H)
    cover(night(4), night(4) + 2 * H)
    for at in (night(1) + 2 * H, night(1) + 8 * H):  # 00:00 and 06:00: both inside a 22:00 to 07:00 night
        visit(at)
    for k in range(3):
        visit(night(3) + (1 + 2 * k) * H)
    visit(night(4) + H)  # on the night without enough footage: not counted
    visit(night(2) - 4 * H)  # 18:00, before the night starts: not a night trip
    reply = h.collect(QUESTION, state, NOW)
    assert reply["paths"] == ["events", "typical"]
    text = reply["answer"]
    assert text.startswith("Bathroom: typically 2 a night (median), from 0 to 3, over 3 nights with footage")
    assert "11 nights didn't have enough checked footage and aren't counted." in text
    assert [p["count"] for p in reply["coverage"]["events"]["periods"]] == [3, 0, 2]


def test_no_checked_night_is_unknown_not_a_typical_zero(bathroom):
    state, cover, _ = bathroom
    cover(night(1), night(1) + H)
    reply = h.collect(QUESTION, state, NOW)
    assert "can't say what's typical" in reply["answer"]
    assert reply["coverage"]["events"]["periods"] == []


def test_a_typical_day_without_night_words_counts_whole_local_days(bathroom):
    state, cover, visit = bathroom
    for n in (1, 2):
        day = DAY - n * 86400.0
        cover(day, day + 24 * H)
    visit(DAY - 86400.0 + 12 * H)
    reply = h.collect("how many times do I usually use the bathroom", state, NOW)
    assert reply["answer"].startswith("Bathroom: typically 0.5 a day (median), from 0 to 1, over 2 days with footage")


def test_a_window_in_the_question_sets_the_periods_and_one_shorter_than_two_falls_back():
    week = typical_counts.periods(DAY - 7 * 86400.0, NOW, NOW, LA, night=False)
    assert (len(week), week[0][0], week[-1][1]) == (7, DAY - 7 * 86400.0, DAY), "today isn't whole yet"
    today = typical_counts.periods(DAY, NOW, NOW, LA, night=True)
    assert len(today) == typical_counts.DEFAULT_PERIODS
    assert today[-1] == (night(1), night(1) + 9 * H), "the last whole night ended at 07:00 this morning"


def test_a_question_about_one_window_keeps_its_plain_count(bathroom):
    state, cover, visit = bathroom
    cover(DAY + 6 * H, NOW)
    visit(DAY + 8 * H)
    reply = h.collect("how many times did I use the bathroom today", state, NOW)
    assert reply["paths"] == ["events"]
    assert reply["answer"].startswith("Bathroom: 1 visit")
