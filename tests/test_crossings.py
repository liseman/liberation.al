"""Crossing geometry and tracking: line geometry, the tracker, and hysteresis, with no real data."""

import pytest

from liberational import config as cfg
from liberational.vision import crossings as X


def walker(x0, x1, frames=30, y=0.9, h=0.4, w=0.1, fps=5.0, score=0.9):
    """Detections of one person walking horizontally: [(ts, [(x1, y1, x2, y2, score)])]."""
    out = []
    for i in range(frames):
        cx = x0 + (x1 - x0) * i / (frames - 1)
        out.append((i / fps, [(cx - w / 2, y - h, cx + w / 2, y, score)]))
    return out


def run(detections, line=None, **over):
    conf = dict(X.settings(), **over)
    line = line or dict(X.DEFAULT_LINE)
    tracker, events = X.Tracker(max_gap=conf["max_gap_seconds"]), []
    for ts, dets in detections:
        for tr in tracker.update(dets, ts):
            events += X.crossings_of(tr, line, conf["hysteresis"], conf["min_track_frames"], conf["min_track_seconds"])
    for tr in tracker.flush():
        events += X.crossings_of(tr, line, conf["hysteresis"], conf["min_track_frames"], conf["min_track_seconds"])
    return events


def test_default_line_and_directions_are_explicit():
    line, is_default = X.line_for("cam0", dict(X.settings(), lines={}))
    assert is_default and line == dict(kind="vertical", position=0.5)
    assert "default vertical line at 50%" in X.line_description(line, True)
    assert X.direction_names(line) == ("left_to_right", "right_to_left")


def test_configured_line_per_camera_and_bad_config_falls_back():
    conf = dict(
        X.settings(), lines={"cam1": {"kind": "vertical", "position": 0.3}, "cam2": {"kind": "vertical", "position": 4}}
    )
    assert X.line_for("cam1", conf) == (dict(kind="vertical", position=0.3), False)
    assert X.line_for("cam2", conf)[1] is True  # invalid position is ignored, not trusted
    assert X.line_for("cam0", conf)[1] is True
    with pytest.raises(ValueError):
        X.normalize_line(dict(kind="segment", p1=[0.2, 0.2], p2=[0.2, 0.2]))
    seg = X.normalize_line(dict(kind="segment", p1=[0.5, 0.0], p2=[0.5, 1.0]))
    assert X.side_value(seg, 0.7, 0.5) * X.side_value(seg, 0.3, 0.5) < 0


def test_bidirectional_crossings_counted_once_each():
    right = run(walker(0.1, 0.9))
    left = run(walker(0.9, 0.1))
    assert [d for _, d in right] == [0] and [d for _, d in left] == [1]


def test_hysteresis_ignores_jitter_on_the_line():
    cxs = [0.5 + (0.01 if i % 2 else -0.01) for i in range(30)]  # never leaves the hysteresis band
    assert run([(i / 5, [(cx - 0.05, 0.5, cx + 0.05, 0.9, 0.9)]) for i, cx in enumerate(cxs)]) == []
    # a person who lingers just on one side and returns without passing far enough
    out_and_back = walker(0.1, 0.52, frames=15)
    back = [(3 + i / 5, dets) for i, (_, dets) in enumerate(reversed(walker(0.1, 0.52, 15)))]
    assert run(out_and_back + back) == []


def test_short_tracks_are_not_evidence():
    assert run(walker(0.2, 0.8, frames=2)) == []
    assert run(walker(0.1, 0.9, frames=4), min_track_seconds=5.0) == []


def test_gap_resets_track_and_prevents_phantom_crossing():
    first = walker(0.1, 0.3, frames=15)
    second = [(ts + 20.0, d) for ts, d in walker(0.7, 0.9, frames=15)]  # 20 s later, other side
    assert run(first + second) == []  # two separate tracks, neither crossed


def test_two_people_cross_independently():
    a = walker(0.1, 0.9, y=0.9)
    b = walker(0.9, 0.1, y=0.6, h=0.3)
    merged = [(ts, da + db) for (ts, da), (_, db) in zip(a, b)]
    assert sorted(d for _, d in run(merged)) == [0, 1]


def test_low_score_detection_keeps_a_track_alive_through_a_dip():
    frames = walker(0.1, 0.9, frames=30)
    dipped = [
        (ts, [(x1, y1, x2, y2, 0.4 if 10 <= i < 14 else s)]) for i, (ts, [(x1, y1, x2, y2, s)]) in enumerate(frames)
    ]
    assert [d for _, d in run(dipped)] == [0]


@pytest.mark.parametrize("appearances", [None, [[1.0, 0.0]]])
def test_expired_track_cannot_be_resurrected_by_a_nearby_detection(appearances):
    tracker = X.Tracker(max_gap=1.0)
    box = (0.2, 0.3, 0.4, 0.9, 0.9)
    tracker.update([box], 0.0, appearances=appearances)
    ended = tracker.update([box], 1.01, appearances=appearances)
    assert [track.id for track in ended] == [1]
    assert tracker.assigned == {0: 2}


@pytest.mark.parametrize("start", [100.0, 1_800_000_000.0])
def test_track_retains_exact_gap_boundary_despite_timestamp_roundoff(start):
    tracker = X.Tracker(max_gap=0.2)
    box = (0.2, 0.3, 0.4, 0.9, 0.9)
    tracker.update([box], start)
    assert tracker.update([box], start + 0.2) == []
    assert tracker.assigned == {0: 1}
    assert tracker.update([box], start + 0.4) == []
    assert tracker.assigned == {0: 1}
    # Ten microseconds beyond the boundary is real elapsed time, even at a Unix epoch.
    ended = tracker.update([box], start + 0.60001)
    assert [track.id for track in ended] == [1]
    assert tracker.assigned == {0: 2}


def tracked_box(center, score=0.9):
    return (center - 0.08, 0.3, center + 0.08, 0.9, score)


@pytest.mark.parametrize("reverse", [False, True])
def test_appearance_preserves_ids_through_crossing_and_reordered_detections(reverse):
    tracker = X.Tracker()
    snapshots = []
    for step, (a, b) in enumerate(((0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3))):
        order = [1, 0] if reverse and step % 2 else [0, 1]
        boxes = [tracked_box(a), tracked_box(b)]
        features = [[3.0, 0.0], [0.0, 2.0]]
        tracker.update([boxes[i] for i in order], step * 0.1, appearances=[features[i] for i in order])
        snapshots.append({order[i]: tid for i, tid in tracker.assigned.items()})
    assert snapshots == [{0: 1, 1: 2}] * 5


def test_appearance_track_survives_a_short_occlusion_at_its_predicted_position():
    tracker = X.Tracker(max_gap=1.0)
    tracker.update([tracked_box(0.2)], 0.0, appearances=[[1.0, 0.0]])
    tracker.update([tracked_box(0.25)], 0.1, appearances=[[1.0, 0.0]])
    assert tracker.update([], 0.4, appearances=[]) == []
    tracker.update([tracked_box(0.35)], 0.6, appearances=[[1.0, 0.0]])
    assert tracker.assigned == {0: 1}


def test_two_crossing_people_keep_ids_when_one_is_briefly_occluded():
    tracker = X.Tracker(max_gap=1.0)
    tracker.update([tracked_box(0.3), tracked_box(0.7)], 0.0, appearances=[[1.0, 0.0], [0.0, 1.0]])
    tracker.update([tracked_box(0.4), tracked_box(0.6)], 0.1, appearances=[[1.0, 0.0], [0.0, 1.0]])
    tracker.update([tracked_box(0.5), tracked_box(0.5)], 0.2, appearances=[[0.0, 1.0], [1.0, 0.0]])
    assert tracker.assigned == {0: 2, 1: 1}
    tracker.update([tracked_box(0.4)], 0.3, appearances=[[0.0, 1.0]])
    assert tracker.assigned == {0: 2}
    tracker.update([tracked_box(0.3), tracked_box(0.7)], 0.4, appearances=[[0.0, 1.0], [1.0, 0.0]])
    assert tracker.assigned == {0: 2, 1: 1}


def test_low_score_rescue_keeps_appearance_without_starting_a_new_track():
    tracker = X.Tracker()
    tracker.update([tracked_box(0.3)], 0.0, appearances=[[1.0, 0.0]])
    tracker.update([tracked_box(0.31, 0.4), tracked_box(0.8, 0.4)], 0.1, appearances=[[0.8, 0.6], [0.0, 1.0]])
    assert tracker.assigned == {0: 1}
    assert len(tracker.active) == 1
    assert tracker.active[0].appearance == pytest.approx([1.0, 0.0])
    tracker.update([tracked_box(0.32)], 0.2, appearances=[[1.0, 0.0]])
    assert tracker.assigned == {0: 1}


@pytest.mark.parametrize("feature", [None, [], [0.0, 0.0], [float("nan"), 0.0], [float("inf"), 0.0], [[1.0]], ["bad"]])
def test_unusable_appearance_falls_back_to_motion(feature):
    tracker = X.Tracker()
    tracker.update([tracked_box(0.3)], 0.0, appearances=[[1.0, 0.0]])
    tracker.update([tracked_box(0.31)], 0.1, appearances=[feature])
    assert tracker.assigned == {0: 1}
    assert tracker.active[0].appearance == pytest.approx([1.0, 0.0])


def test_incompatible_appearance_dimensions_fall_back_to_motion():
    tracker = X.Tracker()
    tracker.update([tracked_box(0.3)], 0.0, appearances=[[1.0, 0.0]])
    tracker.update([tracked_box(0.31)], 0.1, appearances=[[1.0, 0.0, 0.0]])
    assert tracker.assigned == {0: 1}
    assert tracker.active[0].appearance == pytest.approx([1.0, 0.0])


def test_appearance_cannot_match_a_person_outside_the_motion_gate():
    tracker = X.Tracker()
    tracker.update([tracked_box(0.2)], 0.0, appearances=[[1.0, 0.0]])
    tracker.update([tracked_box(0.8)], 0.1, appearances=[[1.0, 0.0]])
    assert tracker.assigned == {0: 2}


def test_contradictory_appearance_can_remain_unmatched_at_the_same_position():
    tracker = X.Tracker()
    tracker.update([tracked_box(0.3)], 0.0, appearances=[[1.0, 0.0]])
    tracker.update([tracked_box(0.3)], 0.1, appearances=[[0.0, 1.0]])
    assert tracker.assigned == {0: 2}


def test_misaligned_appearance_input_does_not_change_track_state():
    tracker = X.Tracker()
    tracker.update([tracked_box(0.3)], 0.0, appearances=[[1.0, 0.0]])
    with pytest.raises(ValueError, match="align"):
        tracker.update([tracked_box(0.4)], 0.1, appearances=[])
    assert tracker.assigned == {0: 1}
    assert tracker.active[0].last_ts == 0.0


def test_global_assignment_does_not_let_a_greedy_pair_strand_a_second_person():
    tracker = X.Tracker(max_center=0.12)
    tracker.update([tracked_box(0.3), tracked_box(0.45)], 0.0, appearances=[None, None])
    tracker.update([tracked_box(0.36), tracked_box(0.20)], 0.1, appearances=[None, None])
    assert tracker.assigned == {1: 1, 0: 2}


def test_legacy_callers_without_appearance_keep_their_greedy_assignment():
    tracker = X.Tracker(max_center=0.12)
    tracker.update([tracked_box(0.3), tracked_box(0.45)], 0.0)
    tracker.update([tracked_box(0.36), tracked_box(0.20)], 0.1)
    assert tracker.assigned == {0: 1, 1: 3}


def test_histogram_bins_and_bounds():
    bins = X.histogram([0.5, 1.5, 1.6, 9.9, 10.0, -1], 0, 10, bins=10)
    assert [b["count"] for b in bins] == [1, 2, 0, 0, 0, 0, 0, 0, 0, 1]


def test_settings_are_bounded():
    with cfg.override({"crossings": {"hysteresis": 7, "min_track_frames": 0, "max_gap_seconds": 0}}):
        s = X.settings()
    assert (s["hysteresis"], s["min_track_frames"], s["max_gap_seconds"]) == (0.2, 2, 0.2)


def _crossed(tracker, frames):
    """Feed frames to a tracker; the crossings of every track it returns, as (ts, direction)."""
    ended = []
    for ts, dets in frames:
        ended += tracker.update(dets, ts)
    ended += tracker.flush()
    conf = X.settings()
    line = dict(X.DEFAULT_LINE)
    return [
        (round(ts, 2), d)
        for t in ended
        for ts, d in X.crossings_of(t, line, conf["hysteresis"], conf["min_track_frames"], conf["min_track_seconds"])
    ]


def _faint_until_past_the_line():
    """Someone walking left to right who scores under `high` until they're past the midpoint."""
    frames = walker(0.2, 0.8, frames=20)
    return [(ts, [(*d[:4], 0.45 if (d[0] + d[2]) / 2 < 0.55 else 0.8) for d in dets]) for ts, dets in frames]


@pytest.mark.parametrize(("tentative", "expected"), [(False, []), (True, [(2.2, 0)])])
def test_a_person_seen_faintly_until_past_the_line_crosses_once_their_track_is_tentative_first(tentative, expected):
    assert _crossed(X.Tracker(tentative=tentative), _faint_until_past_the_line()) == expected


def test_a_tentative_track_stays_unassigned_until_a_confident_detection_matches_it():
    tracker = X.Tracker(tentative=True)
    tracker.update([(0.2, 0.5, 0.3, 0.9, 0.45)], 0.0)
    assert tracker.assigned == {} and [t.tentative for t in tracker.active] == [True]
    tracker.update([(0.21, 0.5, 0.31, 0.9, 0.45)], 0.2)
    assert tracker.assigned == {}
    tracker.update([(0.22, 0.5, 0.32, 0.9, 0.8)], 0.4)
    assert tracker.assigned == {0: 1} and [t.tentative for t in tracker.active] == [False]
    assert len(tracker.active[0].points) == 3  # its faint frames are its history


def test_a_faint_box_that_never_scores_high_never_becomes_a_track():
    tracker = X.Tracker(tentative=True, tentative_s=3.0)
    prop = (0.8, 0.8, 0.87, 0.95, 0.4)  # a static false box, as on the patio
    ended = [t for k in range(40) for t in tracker.update([prop], k * 0.25)]
    assert ended == [] and tracker.assigned == {}
    assert all(9.75 - t.first_ts <= 3.0 for t in tracker.active), "a tentative track lives at most tentative_s"
    assert tracker.flush() == []
