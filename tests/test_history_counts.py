"""How count questions route through history_qa: walk-bys and entries from see-actions' spans and movement only."""

import datetime as dt
from unittest.mock import create_autospec

import pytest

from liberational import config, vision_log
from liberational.memory import crossing_counts, movement_counts, presence_counts, video_evidence
from liberational.memory import history_qa as history

NOW = dt.datetime(2026, 10, 28, 15, tzinfo=history.TZ).timestamp()
NO_FRAMES = {"presence": {"enabled": False}, "review": {"recorded_frame_analysis": False}}


def database(vision, rows, **fields):
    """Caption records from rows of (ts, cam, yaw, caption, observation dict or None)."""
    for ts, cam, yaw, caption, observation in rows:
        record = dict(ts_start=ts, cam=cam, yaw_deg=yaw, caption=caption, **fields)
        if observation is not None:
            record["observation"] = observation
        vision.add(vision_log.CAPTIONS, **record)


def status(name):
    return dict(status=name)


def refuse(*args, **kwargs):
    raise AssertionError("a walk-by count runs no frame analysis")


def test_a_walk_by_question_without_presence_is_answered_from_walk_by_spans_alone(tmp_path, monkeypatch, vision):
    # Captions saying "a person" are no walk-by evidence: with no walk-by spans, nothing watched, so it's unknown.
    database(vision, [(10, "cam0", 0, "no people", None), (50, "cam0", 0, "person by the door", None)])
    monkeypatch.setattr(video_evidence, "collect", refuse)
    with config.override(NO_FRAMES):
        result = history.collect("how many times did someone walk by in last 24 hours?", tmp_path, now=100)
    assert "unknown" in result["answer"]
    assert "person by the door" not in result["answer"]
    assert result["evidence"] == []
    assert set(result["crossings"]) == {"path_b", "answer"}
    assert result["crossings"]["path_b"]["result"]["cameras"] == {}


ENTRY_QUESTIONS = [
    pytest.param("how many times did you see someone walk into frame in last 24 hours?", id="walk-into-frame"),
    pytest.param("how many people came in today?", id="came-in"),
    pytest.param("how often did anyone enter the room yesterday?", id="enter-the-room"),
    pytest.param("how many times did someone walk by in last 24 hours?", id="walk-by"),
    pytest.param("can you tell me how many people came in today?", id="tell-me"),
]


@pytest.mark.parametrize("day", [None, "2026-10-28"], ids=["no-day", "review-day"])
@pytest.mark.parametrize("question", ENTRY_QUESTIONS)
def test_entry_count_questions_take_the_walk_by_path_on_every_route(question, day):
    query = history.parse(question, NOW, selected_day=day)
    assert history.is_history_question(question)
    assert history.is_crossing_question(question, query)


@pytest.mark.parametrize("day", [None, "2026-10-28"], ids=["no-day", "review-day"])
@pytest.mark.parametrize(
    "question",
    [
        pytest.param("how many people did I talk to today?", id="talk-to"),
        pytest.param("how many times did I enter my password today?", id="enter-password"),
        pytest.param("how many of the people who came in today did I talk to?", id="talked-to-arrivals"),
        pytest.param("how many people did I see enter today and end up speaking to?", id="speaking-to-arrivals"),
        pytest.param("how many people who came in today have I spoken with?", id="spoken-with-arrivals"),
    ],
)
def test_count_questions_that_are_not_entries_stay_off_the_walk_by_path(question, day):
    assert not history.is_crossing_question(question, history.parse(question, NOW, selected_day=day))


@pytest.mark.parametrize(
    ("question", "kind"),
    [
        pytest.param("tell me what people were doing today", "video", id="scene"),
        pytest.param("recap what everyone said today", "mixed", id="recap"),
    ],
)
def test_review_day_questions_keep_their_kind(question, kind):
    assert history.parse(question, NOW, selected_day="2026-10-28")["kind"] == kind


def test_a_walk_by_question_reads_walk_by_spans_and_movement_not_sampled_presence(tmp_path, monkeypatch, vision):
    # A walk-by count comes from walk-by spans and movement, never from sampled presence.
    monkeypatch.setattr(
        video_evidence,
        "collect",
        create_autospec(video_evidence.collect, side_effect=AssertionError("no video analysis")),
    )
    monkeypatch.setattr(
        presence_counts,
        "collect",
        create_autospec(presence_counts.collect, side_effect=AssertionError("no sampled presence")),
    )
    with config.override({"presence": {"enabled": True}, "review": {"recorded_frame_analysis": True}}):
        result = history.collect("how many times did someone walk by camera0 in last 24 hours?", tmp_path, now=NOW)
    assert result["paths"] == ["presence"]
    assert "presence" not in result
    assert set(result["coverage"]) >= {"walked_past", "movement"}
    assert "walked by" in result["answer"]


def test_a_walk_by_question_whose_sources_both_fail_says_so_never_zero(tmp_path, monkeypatch):
    def unreadable(*args, **kwargs):
        raise OSError("broker unreachable")

    monkeypatch.setattr(crossing_counts, "from_spans", unreadable)
    monkeypatch.setattr(
        movement_counts, "collect", lambda *a, **k: {"available": False, "errors": ["OSError"], "cameras": {}}
    )
    with config.override({"presence": {"enabled": True, "support_cadence_s": 0}}):
        result = history.collect("how many times did someone come into view today?", tmp_path, now=NOW)
    assert result["coverage"]["read_error"]
    assert result["paths"] == ["presence"]
    assert result["answer"] == "History source could not be read safely."


@pytest.mark.parametrize(
    ("question", "happened"),
    [
        pytest.param("how many times did someone come into view today?", "came into view", id="entries"),
        pytest.param("how many times did someone walk by today?", "walked by", id="walk-bys"),
    ],
)
def test_an_unwatched_count_says_what_it_was_asked_about(tmp_path, question, happened):
    with config.override({"presence": {"enabled": True, "support_cadence_s": 0}}):
        result = history.collect(question, tmp_path, now=NOW)
    assert f"can't say how many times someone {happened}." in result["answer"]
