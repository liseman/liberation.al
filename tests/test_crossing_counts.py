"""Walk-bys over a window from the segment store's walk-by spans, and the chain rules that make them
(memory/crossing_counts.py, Review path B)."""

import math

import pytest

from liberational import config as cfg
from liberational import segment_log
from liberational.memory import crossing_counts as cc
from liberational.memory import walked_past
from liberational.vision import crossings as X

T = 1_800_000_000.0
LINE = {"kind": "vertical", "position": 0.5}


def walk_by(vision, ts, direction="left_to_right", *, cam="cam0", crossings=1, followed=True):
    return vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=ts,
        ts_end=ts,
        cam=cam,
        kind="crossing",
        attributes={"direction": direction, "crossings": crossings, "followed": followed},
    )


def test_walk_bys_are_counted_per_camera_and_direction_with_what_was_analysed(tmp_path, vision):
    # The walk-by worker looked at cam0 for 19 s, then nothing for the rest of the minute; cam1 for all of it.
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=T, ts_end=T + 19, cam="cam0", frames=38)
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=T, ts_end=T + 60, cam="cam1", frames=120)
    walk_by(vision, T + 3)
    walk_by(vision, T + 7, crossings=3, followed=False)
    walk_by(vision, T + 9, "right_to_left")
    walk_by(vision, T + 4, cam="cam1")
    result = cc.from_spans(tmp_path, T, T + 60)
    cam0, cam1 = result["cameras"]["cam0"], result["cameras"]["cam1"]
    assert cam0["crossings"] == {"left_to_right": 2, "right_to_left": 1, "total": 3}
    assert cam1["crossings"]["total"] == 1, "each camera on its own, never summed"
    assert (cam0["scanned_seconds"], cam0["coverage_of_requested"]) == (19.0, round(19 / 60, 6))
    assert cam0["unrecorded_or_unscanned_intervals"] == [[T + 19, T + 60]]
    assert cam1["unrecorded_or_unscanned_intervals"] == [] and not cam1["partial"]
    assert [e["ts"] for e in cam0["examples"]] == [T + 3, T + 7, T + 9]
    assert (cam0["line_crossings"]["total"], cam0["unverified"]) == (5, 1)
    assert result["partial"]
    assert "people walked past about 3 times" in cc.answer_text(result)


def test_a_walk_by_while_the_camera_moved_flags_the_count_as_uncertain(tmp_path, vision):
    still = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=T + 3,
        ts_end=T + 3,
        cam="cam0",
        kind="crossing",
        attributes={"direction": "left_to_right", "crossings": 1, "followed": True},
        camera_moved=False,
    )
    shaken = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=T + 9,
        ts_end=T + 9,
        cam="cam0",
        kind="crossing",
        attributes={"direction": "right_to_left", "crossings": 1, "followed": True},
        camera_moved=True,
        uncertainty={"camera_motion": "unknown"},
    )
    motion = cc.from_spans(tmp_path, T, T + 60)["cameras"]["cam0"]["camera_motion"]
    assert motion == {"uncertain": True, "flagged_files": [shaken["id"]]}
    assert still["id"] not in motion["flagged_files"]


def test_a_still_camera_reports_no_motion_caution(tmp_path, vision):
    vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=T + 3,
        ts_end=T + 3,
        cam="cam0",
        kind="crossing",
        attributes={"direction": "left_to_right", "crossings": 1, "followed": True},
        camera_moved=False,
    )
    motion = cc.from_spans(tmp_path, T, T + 60)["cameras"]["cam0"]["camera_motion"]
    assert motion == {"uncertain": False, "flagged_files": []}


def test_a_window_with_nothing_analysed_is_unknown_not_zero(tmp_path, vision):
    result = cc.from_spans(tmp_path, T, T + 60)
    assert result["cameras"] == {} and result["partial"]
    text = cc.answer_text(result)
    assert "unknown" in text and "0" not in text


# -- what counts as walking past: the chain rules see/walk_bys.py applies as it watches


def seen(points, tid, where, t0, t1, line=LINE):
    """Track `tid`'s foot point every 0.25 s from t0 to t1 at x = where(t), as [t, distance from the line]."""
    k = 0
    while t0 + k / 4 <= t1:
        t = t0 + k / 4
        points.setdefault(tid, []).append([t, X.side_value(line, where(t - t0), 0.8)])
        k += 1


def cross(events, ts, tid, direction="left_to_right"):
    events.append({"id": f"{tid}-{len(events)}", "ts_start": ts, "track_id": tid, "direction": direction})


def glide(a, b, seconds):
    return lambda t: a + (b - a) * min(1.0, t / seconds)


def walked(points, events, line=LINE):
    """(walk-bys, crossings) over the chains."""
    found = cc.chains(events, points)
    return sum(cc.qualifies(c, line) for c in found), sum(len(c["events"]) for c in found)


def test_someone_hovering_at_the_line_never_walked_past():
    points, events = {}, []
    seen(points, "desk", lambda t: 0.5 + 0.08 * (-1) ** int(t), T, T + 60)  # shifting either side of it
    for k in range(10):
        cross(events, T + 2 + 5 * k, "desk", "left_to_right" if k % 2 else "right_to_left")
    assert walked(points, events) == (0, 10)


def test_someone_going_clearly_from_one_side_to_the_other_walked_past_once():
    points, events = {}, []
    seen(points, "a", glide(0.2, 0.8, 4), T + 10, T + 14)
    cross(events, T + 12, "a")
    assert walked(points, events) == (1, 1)


def test_someone_who_stays_in_view_and_goes_back_and_forth_counts_once():
    points, events = {}, []
    seen(points, "a", lambda t: 0.5 + 0.3 * math.sin(t), T, T + 30)
    for k in range(9):
        cross(events, T + 1 + 3.1 * k, "a", "left_to_right" if k % 2 == 0 else "right_to_left")
    assert walked(points, events) == (1, 9)
    (chain,) = cc.chains(events, points)
    walk = cc.walk(chain)
    assert (walk["ts"], walk["direction"], walk["crossings"]) == (T + 1, "left_to_right", 9), "its first crossing"


def test_a_track_that_broke_and_restarted_where_it_ended_is_one_walk_by():
    points, events = {}, []
    seen(points, "a", glide(0.2, 0.49, 2), T + 10, T + 12)
    seen(points, "b", glide(0.51, 0.8, 2), T + 12.5, T + 14.5)
    cross(events, T + 12.6, "b")
    assert walked(points, events) == (1, 1)


def test_a_track_that_starts_long_after_another_ended_is_not_joined_to_it():
    points, events = {}, []
    seen(points, "a", glide(0.2, 0.49, 2), T + 10, T + 12)
    seen(points, "b", glide(0.51, 0.8, 2), T + 20, T + 22)
    cross(events, T + 20.1, "b")
    assert walked(points, events) == (0, 1), "neither half went clearly past the line"


def test_two_people_walking_past_together_are_two():
    points, events = {}, []
    for tid, at in (("a", T + 10), ("b", T + 10.25)):
        seen(points, tid, glide(0.2, 0.8, 4), at, at + 4)
        cross(events, at + 2, tid)
    assert walked(points, events) == (2, 2)


def test_a_line_near_the_edge_asks_only_half_the_room_on_its_short_side():
    near_edge = {"kind": "vertical", "position": 0.9}
    points, events = {}, []
    seen(points, "a", glide(0.5, 0.97, 4), T + 10, T + 14, near_edge)
    cross(events, T + 13, "a")
    assert walked(points, events, near_edge) == (1, 1)


@pytest.mark.parametrize(
    ("directions", "phrases"),
    [
        (["left_to_right", "right_to_left"], ["mixed directions", "left to right", "right to left"]),
        (["top_to_bottom", "bottom_to_top"], ["mixed directions", "top to bottom", "bottom to top"]),
        (["left_to_right"], ["left to right"]),
    ],
)
def test_direction_text_reports_observed_image_directions(directions, phrases):
    result = {"cameras": {"cam0": {"direction_examples": [{"direction": d} for d in directions]}}}
    answer = walked_past.direction_text(result)
    for phrase in phrases:
        assert phrase in answer
    assert "cam0" in answer and "image" in answer
    assert "who" in answer and "destination" in answer


def test_direction_text_keeps_cameras_separate_and_marks_partial_coverage():
    answer = walked_past.direction_text(
        {
            "partial": True,
            "cameras": {
                "cam1": {"direction_examples": [{"direction": "right_to_left"}]},
                "cam0": {"examples": [{"direction": "left_to_right"}], "partial": True},
            },
        }
    )
    assert "On cam0, observed movement was left to right" in answer
    assert "On cam1, observed movement was right to left" in answer
    assert answer.index("cam0") < answer.index("cam1")
    assert "partial" in answer and "other directions" in answer


@pytest.mark.parametrize(
    "result",
    [
        None,
        {"cameras": {}},
        {"cameras": {"cam0": {"crossings": {"left_to_right": 8, "total": 8}, "examples": [{"direction": "unknown"}]}}},
    ],
)
def test_direction_text_does_not_infer_direction_or_absence_from_counts(result):
    answer = walked_past.direction_text(result)
    assert "direction evidence" in answer.lower() and "unavailable" in answer
    assert "left to right" not in answer
    assert "Nobody" not in answer


def test_explicit_empty_direction_examples_do_not_fall_back_to_legacy_examples():
    answer = walked_past.direction_text(
        {
            "cameras": {
                "cam0": {
                    "direction_examples": [],
                    "examples": [{"direction": "left_to_right"}],
                }
            }
        }
    )
    assert "unavailable" in answer and "left to right" not in answer


def test_native_direction_witness_survives_beyond_regular_example_prefix(tmp_path, vision):
    for k in range(cc.MAX_EXAMPLES + 1):
        vision.add(
            segment_log.WALKBY_SPANS,
            ts_start=T + k,
            ts_end=T + k,
            cam="cam0",
            kind="crossing",
            attributes={"direction": "left_to_right", "crossings": 1, "followed": True},
        )
    witness = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=T + 300,
        ts_end=T + 300,
        cam="cam0",
        kind="crossing",
        attributes={"direction": "right_to_left", "crossings": 1, "followed": True},
    )
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=T, ts_end=T + 400, cam="cam0", frames=800)
    result = cc.from_spans(tmp_path, T, T + 400)
    camera = result["cameras"]["cam0"]
    assert camera["crossings"] == {"left_to_right": 201, "right_to_left": 1, "total": 202}
    assert len(camera["examples"]) == 200 and camera["examples_truncated"]
    assert {e["direction"] for e in camera["examples"]} == {"left_to_right"}
    assert [e["direction"] for e in camera["direction_examples"]] == ["left_to_right", "right_to_left"]
    assert camera["direction_examples"][0] == camera["examples"][0]
    assert camera["direction_examples"][1] == {
        "ts": T + 300,
        "direction": "right_to_left",
        "camera": "cam0",
        "link": cc.link(T + 300),
        "payload_id": witness["id"],
    }
    assert "mixed directions" in walked_past.direction_text(result)


def test_direction_examples_keep_each_native_direction_once_and_ignore_unknown(tmp_path, vision):
    records = [
        vision.add(
            segment_log.WALKBY_SPANS,
            ts_start=T + k,
            ts_end=T + k,
            cam="cam0",
            kind="crossing",
            attributes={"direction": direction, "crossings": 1, "followed": True},
        )
        for k, direction in enumerate(
            [
                "unknown",
                "top_to_bottom",
                "bottom_to_top",
                "left_to_right",
                "right_to_left",
                "side_a_to_side_b",
                "side_b_to_side_a",
                "top_to_bottom",
            ]
        )
    ]
    examples = cc.from_spans(tmp_path, T, T + 60)["cameras"]["cam0"]["direction_examples"]
    assert [e["direction"] for e in examples] == [
        "top_to_bottom",
        "bottom_to_top",
        "left_to_right",
        "right_to_left",
        "side_a_to_side_b",
        "side_b_to_side_a",
    ]
    assert [e["payload_id"] for e in examples] == [records[k]["id"] for k in (1, 2, 3, 4, 5, 6)]
    assert [e["ts"] for e in examples] == [T + 1, T + 2, T + 3, T + 4, T + 5, T + 6]


def test_direction_count_answer_keeps_per_direction_passages_and_default_wording():
    crossed = {
        "partial": True,
        "cameras": {
            "cam0": {
                "direction_examples": [{"direction": "left_to_right"}, {"direction": "right_to_left"}],
                "crossings": {"left_to_right": 3, "right_to_left": 1, "total": 4},
            }
        },
    }
    answer = walked_past.direction_text(crossed, include_counts=True)
    assert "left to right (3 observed passages)" in answer
    assert "right to left (1 observed passage)" in answer
    assert "not distinct people" in answer and "partial" in answer
    direction_only = walked_past.direction_text(crossed)
    assert (
        "observed movement was mixed directions: left to right and right to left in the camera image." in direction_only
    )
    assert "observed passages" not in direction_only


@pytest.mark.parametrize("quantity", [None, "3", -1, True])
def test_direction_quantity_unavailable_does_not_borrow_total(quantity):
    answer = walked_past.direction_text(
        {
            "cameras": {
                "cam0": {
                    "direction_examples": [{"direction": "left_to_right"}],
                    "crossings": {"left_to_right": quantity, "total": 9},
                }
            }
        },
        include_counts=True,
    )
    assert "left to right (passage quantity unavailable)" in answer
    assert "9" not in answer


def test_native_segment_direction_witnesses_and_quantities(tmp_path, vision):
    first = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=T + 1,
        cam="cam0",
        kind="crossing",
        attributes={"direction": "side_a_to_side_b", "crossings": 2, "followed": True},
    )
    vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=T + 2,
        cam="cam0",
        kind="crossing",
        attributes={"direction": "side_a_to_side_b", "crossings": 1, "followed": True},
    )
    reverse = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=T + 3,
        cam="cam0",
        kind="crossing",
        attributes={"direction": "side_b_to_side_a", "crossings": 1, "followed": True},
    )
    line = {"kind": "segment", "p1": [0.1, 0.2], "p2": [0.9, 0.8]}
    with cfg.override({"crossings": {"lines": {"cam0": line}}}):
        result = cc.from_spans(tmp_path, T, T + 60)
    cam = result["cameras"]["cam0"]
    assert cam["crossings"] == {"side_a_to_side_b": 2, "side_b_to_side_a": 1, "total": 3}
    assert cam["line_crossings"]["total"] == 4
    assert cam["direction_examples"] == [
        {
            "ts": T + 1,
            "camera": "cam0",
            "direction": "side_a_to_side_b",
            "payload_id": first["id"],
            "link": cc.link(T + 1),
        },
        {
            "ts": T + 3,
            "camera": "cam0",
            "direction": "side_b_to_side_a",
            "payload_id": reverse["id"],
            "link": cc.link(T + 3),
        },
    ]
    direction_only = walked_past.direction_text(result)
    assert "line side A to line side B" in direction_only
    assert "line side B to line side A" in direction_only
    assert "mixed directions" in direction_only
    answer = walked_past.direction_text(result, include_counts=True)
    assert "line side A to line side B (2 observed passages)" in answer
    assert "line side B to line side A (1 observed passage)" in answer
