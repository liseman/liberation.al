"""Review's recorded days (GET /review/days): which days have video, room audio, or transcript lines to show."""

import json
import time
import urllib.request

import pytest

from liberational import speech_log
from liberational.dashboard import context, server, transcript_lines
from speech_fake import fake


def noon(day: str) -> float:
    """Local noon of a day, so the day a line falls on doesn't depend on the test machine's time zone."""
    return time.mktime(time.strptime(f"{day} 12:00:00", "%Y-%m-%d %H:%M:%S"))


@pytest.fixture
def state(tmp_path, vision):
    state = tmp_path / "state"
    vision.video.append(("cam0", noon("2026-10-24"), noon("2026-10-24") + 300))  # a camera's video on the broker
    at = int(noon("2026-10-22") * 1e9)
    lines = [
        # a line whose audio is kept on its microphone's stream: an audio day
        {
            "ts_start": noon("2026-10-22"),
            "text": "Morning.",
            "audio": {"stream": "mic.audio", "start_ns": at, "end_ns": at},
        },
        {"ts_start": noon("2026-10-25"), "text": "Evening."},
    ]
    fake().write(lines)
    return state


def test_days_are_the_ones_with_kept_video_or_lines_and_their_audio(state):
    days = transcript_lines.recorded_days(state)
    assert days == [
        {"day": "2026-10-22", "video": False, "audio": True, "transcript": True},
        {"day": "2026-10-24", "video": True, "audio": False, "transcript": False},
        {"day": "2026-10-25", "video": False, "audio": False, "transcript": True},
    ]


def test_a_fresh_state_has_no_days(tmp_path, vision):
    assert transcript_lines.recorded_days(tmp_path / "state") == []


def test_the_route_lists_the_days(state, serve, monkeypatch):
    monkeypatch.setattr(context, "STATE", state)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    with urllib.request.urlopen(serve(server.H) + "/review/days", timeout=5) as r:
        body = json.loads(r.read())
    assert [d["day"] for d in body["days"]] == ["2026-10-22", "2026-10-24", "2026-10-25"]
    assert body["today"] == time.strftime("%Y-%m-%d")


def test_lines_committed_or_purged_later_move_the_days(state):
    def days():
        return [d["day"] for d in transcript_lines.recorded_days(state)]

    assert days() == ["2026-10-22", "2026-10-24", "2026-10-25"]
    fake().say({"ts_start": noon("2026-10-26"), "text": "Later."})
    assert days()[-1] == "2026-10-26"
    fake().purge(speech_log.UTTERANCES, lambda e: e["text"] == "Evening.")  # a forgotten person's line
    assert "2026-10-25" not in days()


def test_new_lines_are_added_without_reading_every_line_again(state, monkeypatch):
    reads = []
    real = transcript_lines.heard_lines
    monkeypatch.setattr(transcript_lines, "heard_lines", lambda *a: reads.append(a) or real(*a))
    transcript_lines.recorded_days(state)
    fake().say({"ts_start": noon("2026-10-20"), "text": "An old recording, ingested late."})
    days = transcript_lines.recorded_days(state)
    assert days[0] == {"day": "2026-10-20", "video": False, "audio": False, "transcript": True}
    assert len(reads) == 1  # only the first call read them all


def test_a_review_correction_reads_every_line_again(state, monkeypatch):
    from liberational.listen import revisions

    reads = []
    real = transcript_lines.heard_lines
    monkeypatch.setattr(transcript_lines, "heard_lines", lambda *a: reads.append(a) or real(*a))
    transcript_lines.recorded_days(state)
    monkeypatch.setattr(revisions, "review_revision", lambda state, *a: 7)  # a line rejected since
    transcript_lines.recorded_days(state)
    assert len(reads) == 2


def test_a_purge_reads_every_line_again(state, monkeypatch):
    reads = []
    real = transcript_lines.heard_lines
    monkeypatch.setattr(transcript_lines, "heard_lines", lambda *a: reads.append(a) or real(*a))
    transcript_lines.recorded_days(state)
    fake().purge(speech_log.UTTERANCES, lambda e: False)  # a purge, even one that took none of these lines
    transcript_lines.recorded_days(state)
    assert len(reads) == 2


def test_video_days_are_asked_again_only_after_a_while_and_todays_until_it_has_some(state, vision, monkeypatch):
    monkeypatch.setattr(transcript_lines, "REVIEW_TODAY_VIDEO_S", 0.0)
    transcript_lines.recorded_days(state)
    transcript_lines.recorded_days(state)
    assert vision.video_days_asks == 1
    vision.video.append(("cam0", time.time() - 60, time.time()))  # today's first video
    assert transcript_lines.recorded_days(state)[-1] == {
        "day": time.strftime("%Y-%m-%d"),
        "video": True,
        "audio": False,
        "transcript": False,
    }
    assert vision.video_days_asks == 1  # found by asking about today alone
    asks = vision.video_asks
    transcript_lines.recorded_days(state)
    assert vision.video_asks == asks  # today has video now: not asked again
    monkeypatch.setattr(transcript_lines, "REVIEW_VIDEO_DAYS_S", 0.0)
    transcript_lines.recorded_days(state)
    assert vision.video_days_asks == 2


def test_todays_video_is_asked_about_at_most_once_a_while_while_it_has_none(state, vision):
    for _ in range(4):
        transcript_lines.recorded_days(state)  # the first asks every day; the rest only about today, once a while
    assert vision.video_days_asks == 1
    assert vision.video_asks == 1
