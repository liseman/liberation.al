"""Times someone came into view (src/liberational/memory/presence_counts.py): see-zones' presence episodes over
see-actions' coverage, read through the suite's fake segment log."""

import pytest

from liberational import segment_log
from liberational.memory import presence_counts as p

T = 1_800_000_000.0


def watched(vision, start, end, cam="cam0"):
    """see-actions analyzed [start, end) of `cam`, in one-second rows as it writes them."""
    at = start
    while at < end:
        vision.add(segment_log.ACTION_COVERAGE, ts_start=T + at, ts_end=T + min(at + 1, end), cam=cam, frames=30)
        at += 1


def episode(vision, start, end, cam="cam0", left=False, right=False):
    vision.add(
        segment_log.PRESENCE_SPANS,
        ts_start=T + start,
        ts_end=T + end,
        cam=cam,
        frames=30,
        max_people=1,
        tracks=1,
        left_censored=left,
        right_censored=right,
    )


def collect(start, end, cameras=()):
    return p.collect("state", start=T + start, end=T + end, cameras=cameras)


def test_episodes_are_counted_where_they_start_in_the_window(vision):
    watched(vision, 0, 100)
    episode(vision, 5, 20)  # starts before the window: shown, not counted
    episode(vision, 40, 45)
    episode(vision, 70, 90)
    got = collect(10, 100)["cameras"]["cam0"]
    assert got["count"] == 2
    assert [s - T for s in got["starts"]] == [40, 70]
    assert [e["counted"] for e in got["episodes"]] == [False, True, True]
    assert (got["unknown"], got["partial"]) == ([], False)


def test_time_see_actions_did_not_analyze_is_unknown_and_makes_the_answer_partial(vision):
    watched(vision, 0, 30)
    watched(vision, 60, 100)
    got = collect(0, 100)["cameras"]["cam0"]
    assert [(u["start"] - T, u["end"] - T) for u in got["unknown"]] == [(30, 60)]
    assert got["unknown_seconds"] == pytest.approx(30)
    assert (got["count"], got["partial"]) == (0, True)


def test_a_camera_never_analyzed_is_not_listed_and_none_watching_says_so(vision):
    watched(vision, 0, 100, cam="cam1")
    result = collect(0, 100, cameras=("cam0",))
    assert result["cameras"] == {} and result["partial"] is True
    assert p.sampled_answer_text(result).startswith("No camera was watching then")


def test_a_left_censored_start_is_counted_and_marked(vision):
    watched(vision, 0, 100)
    episode(vision, 50, 60, left=True)
    got = collect(0, 100)["cameras"]["cam0"]
    assert (got["count"], got["censored_starts"], got["partial"]) == (1, 1, True)


def test_an_unreadable_store_answers_cant_say_never_nobody(vision, monkeypatch):
    def broken(stream, start, end):
        raise OSError("segments unreadable")

    monkeypatch.setattr(vision, "range", broken)
    result = collect(0, 100)
    assert result["error"] and result["cameras"] == {}
    assert p.sampled_answer_text(result).startswith("I can't read the camera history")


@pytest.mark.parametrize(
    ("episodes", "gap", "expected"),
    [
        pytest.param([(10, 20)], False, "Someone was in view of the camera once.", id="once"),
        pytest.param(
            [(10, 20), (40, 50), (70, 80)], False, "Someone was in view of the camera 3 separate times.", id="three"
        ),
        pytest.param([], False, "Nobody was seen in view of the camera.", id="nobody-while-watched"),
        pytest.param(
            [(10, 20)], True, "Someone was in view of the camera once. It wasn't watching the whole time", id="partial"
        ),
    ],
)
def test_the_answer_says_how_many_times_never_who(vision, episodes, gap, expected):
    watched(vision, 0, 30)
    watched(vision, 30 + (20 if gap else 0), 100)
    for a, b in episodes:
        episode(vision, a, b)
    text = p.sampled_answer_text(collect(0, 100))
    assert text.startswith(expected)
    assert text.endswith("not different people, and it can't say who.")


def test_a_window_nobody_watched_is_never_a_zero(vision):
    watched(vision, 0, 5)  # analyzed before the window
    watched(vision, 200, 300)  # and after it
    text = p.sampled_answer_text(collect(10, 100))
    assert text.startswith("No camera was watching then") and "Nobody was seen" not in text


@pytest.mark.parametrize("bad", [(10.0, 10.0), (20.0, 10.0), (float("nan"), 10.0)])
def test_a_window_must_have_a_positive_length(bad):
    with pytest.raises(ValueError, match="positive duration"):
        p.collect("state", start=bad[0], end=bad[1])
