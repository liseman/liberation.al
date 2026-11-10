"""Activity questions that aren't counts: when a model action happened, and what someone did in a zone,
from see-actions' spans and scored windows, honest about unchecked kinds and unwatched time."""

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from liberational import segment_log
from liberational.memory import activity_questions as aq
from liberational.memory import event_counts
from liberational.memory import history_qa as h
from liberational.see import zones

LA = ZoneInfo("America/Los_Angeles")
DAY = dt.datetime(2026, 10, 23, tzinfo=LA).timestamp()  # local midnight
NOW = DAY + 20 * 3600
H = 3600.0
BOX = [[0.1, 0.6], [0.3, 0.6], [0.3, 0.9], [0.1, 0.9]]
WINDOW = {"start": DAY + 9 * H, "end": DAY + 10 * H}
TYPING, PHONE = 29, 28  # NTU-120: typing on a keyboard, playing with phone or tablet


@pytest.fixture(autouse=True)
def local_time(monkeypatch):
    monkeypatch.setattr(h, "TZ", LA)


def clock(ts: float) -> str:
    return event_counts._clock(ts, LA)


def watched(vision, a=DAY + 9 * H, b=DAY + 10 * H, cam="cam0"):
    vision.add(segment_log.ACTION_COVERAGE, ts_start=a, ts_end=b, cam=cam, frames=1000, model="m")


def action(vision, at, kind="sit_down", cam="cam0", person=None, person_name=None):
    return vision.add(segment_log.ACTION_SPANS, ts_start=at, ts_end=at + 2, cam=cam, kind=kind, candidate=True,
                      track_id="7", subject_type="person", camera_moved=False, person=person,
                      person_name=person_name)  # fmt: skip


def window(vision, at, track, top, cam="cam0", model="m"):
    vision.add(segment_log.ACTION_WINDOWS, ts_start=at, ts_end=at + 4, cam=cam, track_id=track,
               classes=[top, 1, 2, 3, 4], scores=[0.6, 0.1, 0.1, 0.1, 0.1], model=model)  # fmt: skip


def test_a_window_the_model_never_watched_is_unknown_not_nothing(tmp_path, vision):
    got = aq.when(aq.Asked("when", kind="sit_down"), WINDOW, tmp_path, h.stamp, clock)
    assert "hasn't watched any of that time" in got["answer"]
    assert got["evidence"] == [] and got["coverage"]["available"] is False


def test_an_unchecked_kind_lists_its_times_as_unverified(tmp_path, vision):
    watched(vision)
    first, second = action(vision, DAY + 9 * H + 600), action(vision, DAY + 9 * H + 2400)
    got = aq.when(aq.Asked("when", kind="sit_down"), WINDOW, tmp_path, h.stamp, clock)
    assert got["answer"].startswith("Unverified (the camera model's sit down detections aren't checked yet)")
    assert f"{clock(first['ts_start'])} on cam0, {clock(second['ts_start'])} on cam0" in got["answer"]
    assert [e["event"] for e in got["evidence"]] == [first["id"], second["id"]]
    assert got["warnings"]


def test_none_seen_by_an_unchecked_kind_says_it_may_have_missed_some(tmp_path, vision):
    watched(vision)
    action(vision, DAY + 9 * H + 600, kind="drink")  # another kind's
    got = aq.when(aq.Asked("when", kind="sit_down"), WINDOW, tmp_path, h.stamp, clock)
    assert got["answer"] == (
        "The camera model saw no sit down in the 60 min it watched of that time. Its sit down detections aren't "
        "checked yet, so it may have missed some."
    )


def kitchen(tmp_path, vision, track="7"):
    defined = zones.define(tmp_path, "cam0", zones.parse_zones([{"name": "Kitchen", "polygon": BOX}]), None, now=DAY)
    vision.add(segment_log.ZONE_VISITS, ts_start=DAY + 9 * H, ts_end=DAY + 9 * H + 900, cam="cam0",
               zone=defined.zones[0].id, zone_name="Kitchen", subject_kind="person", track_id=track, dwell_s=900.0)  # fmt: skip


def test_what_someone_did_in_a_zone_is_the_model_guess_over_its_visitors_tracks(tmp_path, vision):
    watched(vision)
    kitchen(tmp_path, vision)
    for n in range(3):
        window(vision, DAY + 9 * H + 60 * n, "7", TYPING)
    window(vision, DAY + 9 * H + 300, "8", PHONE)  # someone not in the kitchen
    question = "What did someone do in the kitchen today?"
    a = aq.asked(question, tmp_path)
    assert a == aq.Asked("doing", zone="Kitchen")
    got = aq.doing(a, WINDOW, tmp_path, h.stamp)
    assert got["answer"] == (
        "The camera model's best guesses at what anyone did in Kitchen: typing on a keyboard (3 looks). "
        "These are model guesses, not confirmed."
    )


@pytest.mark.parametrize(("name", "seen"), [("Sam", True), ("Dana", False)])
def test_what_a_named_person_did_in_a_zone_keeps_to_their_recognized_tracks(tmp_path, vision, name, seen):
    watched(vision)
    kitchen(tmp_path, vision)
    window(vision, DAY + 9 * H + 60, "7", TYPING)
    vision.add(segment_log.FACE_MATCHES, ts_start=DAY + 9 * H + 30, cam="cam0", track_id="7", name="Sam", sim=0.8)
    got = aq.doing(aq.Asked("doing", zone="Kitchen", who=name), WINDOW, tmp_path, h.stamp)
    if seen:
        assert "typing on a keyboard (1 look)" in got["answer"]
    else:
        assert got["answer"].startswith("I didn't see Dana in Kitchen")


def test_ask_answers_when_a_model_action_happened(tmp_path, vision):
    watched(vision, DAY, NOW)
    action(vision, DAY + 9 * H + 600)
    reply = h.collect("When did I sit down today?", tmp_path, NOW)
    assert reply["paths"] == ["activity"]
    assert reply["answer"].startswith("Unverified")
    assert reply["evidence"][0]["method"].startswith("see-actions model action")
    assert reply["evidence"][0]["stream"] == "action.span"  # tools/measure.py can expect it by stream


def test_what_was_i_doing_is_the_model_guess_over_the_window(tmp_path, vision):
    watched(vision, DAY, NOW)
    window(vision, DAY + 9 * H + 60, "7", TYPING)
    reply = h.collect("What was I doing at 9am today?", tmp_path, NOW)
    assert reply["paths"] == ["activity"]
    assert "typing on a keyboard (1 look)" in reply["answer"] and "not confirmed" in reply["answer"]


def test_a_movinet_window_names_its_kinetics_class(tmp_path, vision):
    watched(vision, DAY, NOW)
    window(vision, DAY + 9 * H + 60, "7", 405, model="movinet-a2-k600")  # Kinetics-600 405: reading book
    reply = h.collect("What was I doing at 9am today?", tmp_path, NOW)
    assert "reading book (1 look)" in reply["answer"]


def test_when_says_who_did_it_where_their_face_was_recognized(tmp_path, vision):
    watched(vision, DAY, NOW)
    action(vision, DAY + 9 * H + 600, person="0190a5b8-0000-7000-8000-00000000000a", person_name="Dana")
    action(vision, DAY + 9 * H + 1200)
    reply = h.collect("When did anyone sit down today?", tmp_path, NOW)
    first, second = reply["answer"].split(": ", 1)[1].split(", ")
    assert first.endswith("on cam0 by Dana")
    assert second.endswith("on cam0.")  # someone not recognized: no name
    assert reply["evidence"][0]["quote"].endswith("by Dana")
