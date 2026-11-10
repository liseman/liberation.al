"""Counting zone visits for Ask (#1316): "how many times did the cat use the litter box today"."""

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from liberational import vision_log
from liberational.memory import event_counts, zone_counts
from liberational.memory import history_qa as h
from liberational.see import zones

LA = ZoneInfo("America/Los_Angeles")
DAY = dt.datetime(2026, 10, 23, tzinfo=LA).timestamp()  # local midnight
NOW = DAY + 20 * 3600
BOX = [[0.1, 0.6], [0.3, 0.6], [0.3, 0.9], [0.1, 0.9]]
QUESTION = "how many times did the cat use the litter box today"


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(h, "TZ", LA)
    zone_counts._NAMES.clear()
    event_counts._NAMES.clear()


@pytest.fixture
def litter(tmp_path, vision):
    """cam0 with a Litter box zone, covered from 06:00 to 20:00 for animals and people."""
    d = zones.define(tmp_path, "cam0", zones.parse_zones([{"name": "Litter box", "polygon": BOX}]), None, now=DAY)
    zone = d.zones[0].id
    for kind in ("animal", "person"):
        vision.add(
            vision_log.ZONE_COVERAGE,
            ts_start=DAY + 6 * 3600,
            ts_end=NOW,
            cam="cam0",
            subject_kind=kind,
            definition=d.id,
            zones=[zone],
            frames=1000,
        )
    return tmp_path, vision, zone, d.id


def visit(vision, zone, definition, at, dwell, label="cat", ended="left"):
    """A visit entering at DAY + at seconds, staying `dwell` seconds."""
    return vision.add(
        vision_log.ZONE_VISITS,
        ts_start=DAY + at,
        ts_end=DAY + at + dwell,
        cam="cam0",
        zone=zone,
        zone_name="Litter box",
        definition=definition,
        subject_kind="person" if label == "person" else "animal",
        track_id=f"run:cam0:{at}",
        label=label,
        dwell_s=float(dwell),
        frames=10,
        entry_seen=True,
        exit_seen=ended == "left",
        ended=ended,
        candidate=True,
    )


def ask(state, question=QUESTION, **query):
    q = h.parse(question, NOW)
    q.update(query)
    asked = event_counts.wanted(question, state)
    assert asked is not None
    return event_counts.answer(asked, q, state, h.stamp, LA)


def test_a_count_question_naming_a_zone_is_routed_to_it(litter):
    state, *_ = litter
    assert event_counts.wanted(QUESTION, state) == "Litter box"


def test_the_cat_counts_long_visits_joining_a_split_track_and_skipping_passes_and_people(litter):
    state, vision, zone, definition = litter
    morning = 7 * 3600
    visit(vision, zone, definition, morning, 40)  # 07:00:00 to 07:00:40
    visit(vision, zone, definition, morning + 50, 20)  # 10 s later: the same use, split by its track
    visit(vision, zone, definition, 9 * 3600, 5)  # 09:00, 5 s: walking past
    second = visit(vision, zone, definition, 13 * 3600, 30)  # 13:00
    visit(vision, zone, definition, 14 * 3600, 60, label="person")  # someone cleaning it
    reply = ask(state)
    assert reply["answer"].startswith("Litter box: 2 visits of at least 10 s by cat (07:00, 13:00).")
    assert [e["ts"] for e in reply["evidence"]] == [DAY + morning, second["ts_start"]]
    assert reply["coverage"]["count_available"] is True
    assert "not confirmed" in reply["answer"]


def test_a_cat_question_with_only_people_checked_is_unknown_not_zero(tmp_path, vision):
    d = zones.define(tmp_path, "cam0", zones.parse_zones([{"name": "Litter box", "polygon": BOX}]), None, now=DAY)
    vision.add(
        vision_log.ZONE_COVERAGE,
        ts_start=DAY + 6 * 3600,
        ts_end=NOW,
        cam="cam0",
        subject_kind="person",
        definition=d.id,
        zones=[d.zones[0].id],
        frames=1000,
    )
    assert ask(tmp_path)["coverage"]["count_available"] is False
    reply = ask(tmp_path, "how many times was the litter box used today")
    assert "No animals were looked for in it, so they aren't counted." in reply["answer"]


def test_without_a_species_it_counts_everyone_and_says_what_they_were(litter):
    state, vision, zone, definition = litter
    visit(vision, zone, definition, 7 * 3600, 40)
    visit(vision, zone, definition, 14 * 3600, 60, label="person")
    reply = ask(state, "how many times was the litter box used today")
    assert (
        "2 visits of at least 10 s by anyone or anything (07:00, 14:00): 1 by a cat, 1 by a person." in reply["answer"]
    )


def test_no_coverage_of_the_zone_is_unknown_not_zero(tmp_path, vision):
    zones.define(tmp_path, "cam0", zones.parse_zones([{"name": "Litter box", "polygon": BOX}]), None, now=DAY)
    reply = ask(tmp_path)
    assert reply["coverage"]["count_available"] is False
    assert "can't count" in reply["answer"]
    assert reply["warnings"] == ["Nothing covers the zone in the window: this is not a count of zero."]


def test_partial_coverage_says_how_much_was_checked(litter):
    state, *_ = litter
    reply = ask(state)
    assert "no visit of at least 10 s by cat in the checked footage" in reply["answer"]
    assert "Footage on cam0 was checked for 14 h 0 min of that time." in reply["answer"]


def test_a_short_visit_ended_by_a_gap_is_reported_apart(litter):
    state, vision, zone, definition = litter
    visit(vision, zone, definition, 8 * 3600, 4, ended="censored")
    reply = ask(state)
    assert "1 shorter visit ended in a gap and may have been longer." in reply["answer"]
    assert reply["evidence"] == []


def test_an_event_with_the_zone_s_name_keeps_its_own_answer(litter, monkeypatch):
    state, *_ = litter
    monkeypatch.setattr(event_counts, "known_names", lambda state: ["litter box"])
    assert event_counts._is_zone(state, "Litter box") is False


def test_a_renamed_zone_counts_its_visits_under_the_new_name(litter):
    state, vision, zone, definition = litter
    visit(vision, zone, definition, 7 * 3600, 40)
    current = zones.current(state)["cam0"]
    zones.define(
        state, "cam0", zones.parse_zones([{"id": zone, "name": "Cat box", "polygon": BOX}]), current.id, now=DAY + 1
    )
    zone_counts._NAMES.clear()
    reply = ask(state, "how many times did the cat use the cat box today")
    assert reply["answer"].startswith("Cat box: 1 visit of at least 10 s by cat (07:00).")


def test_ask_answers_a_litter_box_count_from_zone_visits(litter):
    state, vision, zone, definition = litter
    visit(vision, zone, definition, 7 * 3600, 40)
    reply = h.collect(QUESTION, state, NOW)
    assert reply["paths"] == ["events"]
    assert reply["answer"].startswith("Litter box: 1 visit of at least 10 s by cat (07:00).")


@pytest.fixture
def may(monkeypatch):
    from liberational.memory import subject_activity

    monkeypatch.setattr(subject_activity, "_pets", lambda state: [{"id": "p-may", "name": "May", "species": "cat"}])


def test_a_pet_asked_about_by_name_counts_her_species_visits_and_says_it_is_an_assumption(litter, may):
    state, vision, zone, definition = litter
    visit(vision, zone, definition, 8 * 3600, 60)
    visit(vision, zone, definition, 9 * 3600, 30, label="person")
    visit(vision, zone, definition, 10 * 3600, 40, label="dog")
    reply = h.collect("how many times did May use the litter box today", state, NOW)
    assert reply["answer"].startswith(
        "Litter box: 1 visit of at least 10 s by the cat (probably May, the only cat you've named) (08:00)."
    )
    assert "any cat seen is taken to be May" in reply["answer"]
    assert "can't tell people apart" not in reply["answer"]


def test_when_the_cat_last_used_the_litter_box_is_its_newest_long_visit(litter, may):
    state, vision, zone, definition = litter
    visit(vision, zone, definition, 8 * 3600, 60)
    visit(vision, zone, definition, 11 * 3600, 90)
    visit(vision, zone, definition, 12 * 3600, 4)  # passing by
    visit(vision, zone, definition, 13 * 3600, 60, label="person")
    reply = h.collect("when did the cat last use the litter box?", state, NOW)
    assert reply["paths"] == ["animal-last-seen"]
    assert reply["answer"].startswith(
        "The cat (probably May, the only cat you've named) was last in Litter box at 2026-10-23 at 11:01 on cam0, "
        "for 2 min."
    )
    assert "checked 8 h 59 min of footage for animals since then" in reply["answer"]


def test_a_zone_that_isn_t_drawn_is_not_a_last_seen_question(litter, may):
    state, *_ = litter
    from liberational.memory import subject_activity

    assert subject_activity.last_asked("when did the cat last use the scratching post", state) is None


def test_the_last_litter_box_visit_survives_a_rename_of_the_zone(litter, may):
    state, vision, zone, definition = litter
    visit(vision, zone, definition, 8 * 3600, 60)  # recorded while it was "Litter box"
    current = zones.current(state)["cam0"]
    zones.define(
        state, "cam0", zones.parse_zones([{"id": zone, "name": "Cat box", "polygon": BOX}]), current.id, now=DAY + 1
    )
    zone_counts._NAMES.clear()
    reply = h.collect("when did the cat last use the cat box today?", state, NOW)
    assert reply["paths"] == ["animal-last-seen"]
    assert "was last in Cat box at 2026-10-23 at 08:01" in reply["answer"]
