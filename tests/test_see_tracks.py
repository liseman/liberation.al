"""Source-backed movement fragments, without models, services, or runtime state."""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from liberational.see.tracks import MovementPipeline


def ns(ts):
    """A fixture time's header ns: the fixtures' times are short decimals, some before zero."""
    return round(ts * 1_000_000_000)


def body(ts, tracks=(), *, cam="cam0", **extra):
    return {
        "id": f"b:{cam}:{ts}",
        "ts_start": ts,
        "_start_ns": ns(ts),
        "cam": cam,
        "boxes": [[x - 0.04, 0.3, x + 0.04, 0.8, 0.9] for _, x in tracks],
        "track_ids": [tid for tid, _ in tracks],
        **extra,
    }


def motion(ts, *, shift=0, cam="cam0", epoch=0, reference=None):
    return {
        "id": f"m:{cam}:{ts}",
        "ts_start": ts,
        "_start_ns": ns(ts),
        "cam": cam,
        "motion": {
            "status": "measured",
            "transform": [[1, 0, shift], [0, 1, 0], [0, 0, 1]],
            "epoch": epoch,
            "reference_ts": ts - 0.5 if reference is None else reference,
            "current_ts": ts,
        },
    }


def feed(p, ts, tracks=(), **extra):
    previous = p.snapshot()["motion"]
    reference = previous[-1]["ts"] if previous else ts - 0.5
    p.feed_motion(motion(ts, reference=reference))
    return p.feed_body(body(ts, tracks, **extra))


def track_outputs(rows):
    return [out for out in rows if out.kind == "track"]


def test_observed_empty_frames_support_zero_with_full_body_and_motion_lineage():
    p = MovementPipeline("cam0")
    feed(p, 0)
    feed(p, 0.5)
    out = p.flush()
    coverage = [o for o in out if o.kind == "coverage"]
    assert coverage[0].record["supported"] == [[0, 0.5]]
    assert coverage[0].record["tracking"] == "complete"
    assert set(coverage[0].parents) == {"b:cam0:0", "b:cam0:0.5", "m:cam0:0", "m:cam0:0.5"}


def test_crossing_people_reorder_without_merging_tracks_or_direction():
    p = MovementPipeline("cam0")
    feed(p, 0, [("a", 0.2), ("b", 0.8)])
    feed(p, 0.5, [("b", 0.6), ("a", 0.4)])
    feed(p, 1, [("a", 0.6), ("b", 0.4)])
    out = track_outputs(p.flush())
    assert {o.record["track_id"]: o.record["direction"] for o in out} == {"a": "right", "b": "left"}
    assert all(o.record["supported"] == [[0, 1]] for o in out)
    assert all(o.record["entry"]["status"] == "unknown" and o.record["exit"]["status"] == "unknown" for o in out)
    assert all(set(o.parents) == {f"{kind}:cam0:{ts}" for kind in ("b", "m") for ts in (0, 0.5, 1)} for o in out)


def test_short_occlusion_has_unknown_gap_not_full_span_dwell_or_exit():
    p = MovementPipeline("cam0")
    feed(p, 0, [("a", 0.3)])
    feed(p, 0.5, [("a", 0.35)])
    feed(p, 1)
    feed(p, 1.5, [("a", 0.45)])
    feed(p, 2, [("a", 0.5)])
    out = track_outputs(p.flush())
    assert len(out) == 1
    assert out[0].record["supported"] == [[0, 0.5], [1.5, 2]]
    assert out[0].record["gaps"] == [[0.5, 1.5]]
    assert out[0].record["exit"]["status"] == "unknown"
    assert "b:cam0:1" in out[0].parents


def test_sustained_edge_travel_needs_source_backed_absence_on_both_sides():
    p = MovementPipeline("cam0")
    outputs = []
    for ts, tracks in [
        (0, []),
        (0.5, [("a", 0.05)]),
        (1, [("a", 0.15)]),
        (1.5, [("a", 0.25)]),
        (2, [("a", 0.65)]),
        (2.5, [("a", 0.85)]),
        (3, [("a", 0.95)]),
        (3.5, []),
        (4, []),
        (4.5, []),
    ]:
        outputs.extend(feed(p, ts, tracks))
    out = track_outputs(outputs)
    assert len(out) == 1
    record = out[0].record
    assert record["entry"] == {
        "status": "supported",
        "edge": "left",
        "time": [0, 0.5],
        "reason": "edge_motion_and_evaluated_absence",
    }
    assert record["exit"] == {
        "status": "supported",
        "edge": "right",
        "time": [3, 4.5],
        "reason": "edge_motion_and_evaluated_absence",
    }
    assert record["supported"] == [[0.5, 3]]
    assert record["ts_start"] == 0 and record["ts_end"] == 4.5
    assert record["direction"] == "right"
    assert set(out[0].parents) == {
        f"{kind}:cam0:{ts}" for kind in ("b", "m") for ts in (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)
    }


@pytest.mark.parametrize("bad", ["legacy", "null", "error", "moved", "stale_motion"])
def test_uncertain_frames_never_become_zero_coverage_or_supported_edges(bad):
    p = MovementPipeline("cam0")
    feed(p, 0)
    for ts, x in [(0.5, 0.05), (1, 0.15), (1.5, 0.25)]:
        rec = body(ts, [("a", x)])
        if bad == "legacy":
            rec.pop("track_ids")
        elif bad == "null":
            rec["track_ids"] = [None]
        elif bad == "error":
            rec["error"] = "detector unavailable"
        p.feed_motion(motion(ts, shift=0.2 if bad == "moved" else 0)) if bad != "stale_motion" else None
        p.feed_body(rec)
    out = p.flush()
    coverage = next(o for o in out if o.kind == "coverage")
    assert coverage.record["tracking"] == "partial"
    assert all(o.record["entry"]["status"] == "unknown" for o in track_outputs(out))


def test_restart_snapshot_replays_the_same_outputs_and_exact_dependencies():
    p = MovementPipeline("cam0")
    feed(p, 0, [("a", 0.2)])
    feed(p, 0.5, [("a", 0.3)])
    snapshot = json.loads(json.dumps(p.snapshot(), allow_nan=False))
    dependencies = p.state_parents()
    restored = MovementPipeline("cam0", state=snapshot)
    assert restored.state_parents() == dependencies
    assert set(dependencies) == {"b:cam0:0", "b:cam0:0.5", "m:cam0:0", "m:cam0:0.5"}
    for pipeline in (p, restored):
        feed(pipeline, 1)
        feed(pipeline, 1.5, [("a", 0.5)])
    assert restored.flush() == p.flush()
    with pytest.raises(ValueError, match="incompatible"):
        MovementPipeline("other", state=snapshot)
    with pytest.raises(ValueError, match="incompatible"):
        MovementPipeline("cam0", conf={"sample_gap": 0.2}, state=snapshot)


def test_fragments_bound_points_parents_and_keep_unknown_occlusion_on_rollover():
    p = MovementPipeline("cam0", conf={"max_points": 3, "max_parents": 12})
    out = []
    for ts, tracks in [
        (0, [("a", 0.3)]),
        (0.5, [("a", 0.35)]),
        (1, []),
        (1.5, [("a", 0.45)]),
        (2, [("a", 0.5)]),
        (2.5, [("a", 0.55)]),
    ]:
        out += feed(p, ts, tracks)
        state = p.snapshot()
        assert len(state["tracks"]["a"]["samples"]) <= 3
    out += p.flush()
    tracks = track_outputs(out)
    assert len(tracks) >= 2
    assert all(len(o.record["points"]) <= 3 for o in tracks)
    assert all(len(o.parents) <= 12 for o in out)
    assert all(not (lo < 1.5 and hi > 0.5) for o in tracks for lo, hi in o.record["supported"])
    assert all(o.record["entry"]["status"] == "unknown" for o in tracks)


def test_equal_timestamps_for_two_tracks_and_other_cameras_do_not_drop_people():
    p = MovementPipeline("cam0")
    p.feed_body(body(0, [("foreign", 0.5)], cam="other"))
    p.feed_motion(motion(0, cam="other"))
    feed(p, 0, [("a", 0.3), ("b", 0.7)])
    duplicate = body(0, [("a", 0.3), ("b", 0.7)])
    assert p.feed_body(duplicate) == []
    feed(p, 0.5, [("a", 0.35), ("b", 0.65)])
    assert {o.record["track_id"] for o in track_outputs(p.flush())} == {"a", "b"}


def test_long_input_gap_and_interior_disappearance_are_censored():
    p = MovementPipeline("cam0")
    feed(p, 0, [("a", 0.3)])
    feed(p, 0.5, [("a", 0.4)])
    out = feed(p, 10, [("a", 0.5)]) + p.flush()
    tracks = track_outputs(out)
    assert len(tracks) == 2
    assert all(o.record["entry"]["status"] == o.record["exit"]["status"] == "unknown" for o in tracks)
    assert not any(lo < 10 and hi > 0.5 for o in tracks for lo, hi in o.record["supported"])


def test_camera_motion_epoch_change_and_rotation_censor_direction():
    for changed in ({"epoch": 1}, {"rotate": True}):
        p = MovementPipeline("cam0")
        feed(p, 0, [("a", 0.2)])
        rec = motion(0.5, epoch=changed.get("epoch", 0))
        if changed.get("rotate"):
            rec["motion"]["transform"] = [[0.9, -0.2, 0], [0.2, 0.9, 0], [0, 0, 1]]
        p.feed_motion(rec)
        p.feed_body(body(0.5, [("a", 0.4)]))
        tracks = track_outputs(p.flush())
        assert tracks[0].record["direction"] == "unknown"
        assert tracks[0].record["camera_motion"] != "stable"
        assert {"m:cam0:0", "m:cam0:0.5"} <= set(tracks[0].parents)


def test_timestamp_disorder_closes_old_track_without_asserting_exit():
    p = MovementPipeline("cam0")
    feed(p, 1, [("a", 0.2)])
    feed(p, 1.5, [("a", 0.3)])
    out = feed(p, 0.5, [("a", 0.1)])
    old = track_outputs(out)[0]
    assert old.record["closure"] == "timestamp_order"
    assert old.record["exit"]["status"] == "unknown"
    assert all(o.record["ts_start"] <= o.record["ts_end"] for o in out)


def test_new_track_after_long_camera_gap_does_not_create_an_empty_track_fragment():
    p = MovementPipeline("cam0")
    feed(p, 0)
    assert track_outputs(feed(p, 30, [("a", 0.5)])) == []
    tracks = track_outputs(p.flush())
    assert len(tracks) == 1 and tracks[0].record["points"] == [[30, 0.5, 0.8]]
    assert tracks[0].record["entry"]["status"] == "unknown"


def test_parent_budget_is_respected_through_long_sampled_absence():
    p = MovementPipeline("cam0", conf={"max_parents": 8, "max_gap": 2})
    outputs = []
    for k in range(30):
        outputs += feed(p, k / 10, [("a", 0.3)] if k < 3 else [])
    outputs += p.flush()
    assert all(len(o.parents) <= 8 for o in outputs)
    assert all(o.record["exit"]["status"] == "unknown" for o in track_outputs(outputs))
    assert all(len(t["samples"]) <= 32 for t in p.snapshot()["tracks"].values())


def test_cropped_feet_do_not_override_a_supported_side_edge():
    p = MovementPipeline("cam0")
    feed(p, 0)
    for ts, x in [(0.5, 0.05), (1, 0.15), (1.5, 0.25)]:
        p.feed_motion(motion(ts))
        rec = body(ts, [("a", x)])
        rec["boxes"][0][3] = 1.0
        p.feed_body(rec)
    track = track_outputs(p.flush())[0]
    assert track.record["entry"]["status"] == "supported"
    assert track.record["entry"]["edge"] == "left"


def test_malformed_motion_is_unknown_and_snapshot_remains_json_safe():
    p = MovementPipeline("cam0")
    rec = motion(0)
    rec["motion"] = "unreadable"
    p.feed_motion(rec)
    rec = motion(0.5)
    rec["motion"]["epoch"] = float("nan")
    p.feed_motion(rec)
    p.feed_body(body(0.5, [("a", 0.3)]))
    json.dumps(p.snapshot(), allow_nan=False)
    assert track_outputs(p.flush())[0].record["camera_motion"] == "unknown"


def test_parent_budget_rollover_does_not_erase_a_trailing_occlusion():
    p = MovementPipeline("cam0", conf={"max_parents": 8})
    out = []
    feed(p, -0.5)
    for ts, tracks in [(0, [("a", 0.3)]), (0.5, [("a", 0.35)]), (1, []), (1.5, [("a", 0.45)])]:
        out += feed(p, ts, tracks)
    out += p.flush()
    assert all(not (lo < 1.5 and hi > 0.5) for o in track_outputs(out) for lo, hi in o.record["supported"])


@pytest.mark.parametrize("transform", [[1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1]])
def test_legacy_flat_affine_motion_remains_supported(transform):
    p = MovementPipeline("cam0")
    for ts in (0, 0.5):
        rec = motion(ts)
        rec["motion"]["transform"] = transform
        p.feed_motion(rec)
        p.feed_body(body(ts))
    coverage = next(o for o in p.flush() if o.kind == "coverage")
    assert coverage.record["supported"] == [[0, 0.5]]
    assert coverage.record["camera_motion"] == "stable"


@pytest.mark.parametrize(
    "transform",
    [
        [[1, 0, 0], [0, 1], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0.2, 0, 1]],
        [[1, 0, float("nan")], [0, 1, 0], [0, 0, 1]],
        [1, 0, 0, 0, 1, 0, 0],
    ],
)
def test_malformed_or_projective_motion_is_unknown(transform):
    p = MovementPipeline("cam0")
    for ts in (0, 0.5):
        rec = motion(ts)
        rec["motion"]["transform"] = transform
        p.feed_motion(rec)
        p.feed_body(body(ts))
    coverage = next(o for o in p.flush() if o.kind == "coverage")
    assert coverage.record["camera_motion"] == "unknown"
    assert coverage.record["supported"] == []


@pytest.mark.parametrize("motion_shift", [0.03, 0.5])
def test_incremental_camera_drift_cannot_establish_personal_edge_travel(motion_shift):
    p = MovementPipeline("cam0", conf={"motion_shift": motion_shift})
    feed(p, 0)
    for ts, x, shift in [(0.5, 0.05, 0), (1, 0.07, 0.02), (1.5, 0.09, 0.02)]:
        p.feed_motion(motion(ts, shift=shift))
        p.feed_body(body(ts, [("a", x)]))
    track = track_outputs(p.flush())[0]
    assert track.record["entry"]["status"] == "unknown"
    assert track.record["direction"] != "right"
    assert {"m:cam0:0.5", "m:cam0:1", "m:cam0:1.5"} <= set(track.parents)


def test_motion_between_body_frames_is_not_hidden_by_a_later_inverse_step():
    p = MovementPipeline("cam0")
    feed(p, 0, [("a", 0.3)])
    p.feed_motion(motion(0.25, shift=0.2, reference=0))
    p.feed_motion(motion(0.5, shift=-0.2, reference=0.25))
    p.feed_body(body(0.5, [("a", 0.3)]))
    track = track_outputs(p.flush())[0]
    assert track.record["camera_motion"] == "moved"
    assert {"m:cam0:0.25", "m:cam0:0.5"} <= set(track.parents)


def test_missing_motion_chain_is_unknown_instead_of_extrapolating_latest():
    p = MovementPipeline("cam0", conf={"max_motion": 1})
    feed(p, 0, [("a", 0.3)])
    p.feed_motion(motion(0.25, reference=0))
    p.feed_motion(motion(0.5, reference=0.25))
    p.feed_body(body(0.5, [("a", 0.4)]))
    out = p.flush()
    assert track_outputs(out)[0].record["camera_motion"] == "unknown"
    assert next(o for o in out if o.kind == "coverage").record["tracking"] == "partial"


def test_track_id_split_in_an_occupied_boundary_region_is_not_an_entry():
    p = MovementPipeline("cam0")
    feed(p, 0, [("old", 0.05)])
    for ts, x in [(0.5, 0.05), (1, 0.15), (1.5, 0.25)]:
        feed(p, ts, [("new", x)])
    track = next(o for o in track_outputs(p.flush()) if o.record["track_id"] == "new")
    assert track.record["entry"]["status"] == "unknown"
    assert "b:cam0:0" in track.parents


def test_people_elsewhere_do_not_block_a_clear_edge_entry():
    p = MovementPipeline("cam0")
    feed(p, 0, [("other", 0.8)])
    for ts, x in [(0.5, 0.05), (1, 0.15), (1.5, 0.25)]:
        feed(p, ts, [("new", x), ("other", 0.8)])
    track = next(o for o in track_outputs(p.flush()) if o.record["track_id"] == "new")
    assert track.record["entry"]["status"] == "supported"


def test_successor_track_in_exit_region_blocks_false_exit():
    p = MovementPipeline("cam0")
    out = []
    for ts, tracks in [
        (0, [("old", 0.65)]),
        (0.5, [("old", 0.85)]),
        (1, [("old", 0.95)]),
        (1.5, [("new", 0.95)]),
        (2, [("new", 0.95)]),
        (2.5, [("new", 0.95)]),
    ]:
        out += feed(p, ts, tracks)
    track = next(o for o in track_outputs(out) if o.record["track_id"] == "old")
    assert track.record["exit"]["status"] == "unknown"


def test_out_of_phase_complete_motion_brackets_support_observed_intervals():
    p = MovementPipeline("cam0")
    p.feed_motion(motion(0, reference=-0.1))
    for body_ts, motion_ts, reference, x in [(0.05, 0.1, 0, 0.2), (0.15, 0.2, 0.1, 0.3), (0.25, 0.3, 0.2, 0.4)]:
        p.feed_motion(motion(motion_ts, reference=reference))
        p.feed_body(body(body_ts, [("a", x)]))
    track = track_outputs(p.flush())[0]
    assert track.record["camera_motion"] == "stable"
    assert track.record["direction"] == "right"
    assert track.record["supported"] == [[0.05, 0.25]]
    assert {"m:cam0:0", "m:cam0:0.1", "m:cam0:0.2", "m:cam0:0.3"} <= set(track.parents)


def test_coverage_publication_includes_late_birth_before_complete_window():
    p = MovementPipeline("cam0")
    for half in range(31):
        feed(p, half / 2)
    out = []
    for half in range(31, 42):
        x = min(0.25, 0.05 + (half - 31) * 0.1)
        out = feed(p, half / 2, [("a", x)])
    coverage = next(o for o in out if o.kind == "coverage")
    tracks = track_outputs(out)
    assert coverage.record["ts_end"] == 20
    assert len(tracks) == 1
    assert tracks[0].record["entry"]["status"] == "supported"
    assert out.index(tracks[0]) < out.index(coverage)


def test_eager_publication_preserves_young_entry_qualification():
    p = MovementPipeline("cam0", conf={"max_duration_s": 2})
    for ts in (0, 0.5, 1, 1.5):
        feed(p, ts)
    feed(p, 2, [("a", 0.05)])
    early = feed(p, 2.5, [("a", 0.15)])
    assert track_outputs(early)[0].record["entry"]["status"] == "unknown"
    feed(p, 3, [("a", 0.25)])
    assert track_outputs(p.flush())[0].record["entry"]["status"] == "supported"


def test_capacity_replacement_of_every_track_does_not_evict_uninitialized_track():
    p = MovementPipeline("cam0", conf={"max_tracks": 2})
    feed(p, 0, [("a", 0.2), ("b", 0.4)])
    out = feed(p, 0.5, [("c", 0.6), ("d", 0.8)])
    assert {o.record["track_id"] for o in track_outputs(out)} == {"a", "b"}
    assert set(p.snapshot()["tracks"]) == {"c", "d"}
    assert all(track["samples"] for track in p.snapshot()["tracks"].values())


def test_out_of_order_motion_arrival_cannot_hide_chronological_epoch_change():
    p = MovementPipeline("cam0")
    for ts, epoch, reference in [(1, 1, 0.5), (0.5, 0, 0), (0, 0, -0.5), (1.5, 1, 1)]:
        p.feed_motion(motion(ts, epoch=epoch, reference=reference))
    p.feed_body(body(0))
    for ts, x in [(0.5, 0.05), (1, 0.15), (1.5, 0.25)]:
        p.feed_body(body(ts, [("a", x)]))
    out = p.flush()
    track = track_outputs(out)[0]
    assert track.record["entry"]["status"] == "unknown"
    assert track.record["camera_motion"] == "unknown"
    assert next(o for o in out if o.kind == "coverage").record["tracking"] == "partial"
    assert {"m:cam0:0.5", "m:cam0:1"} <= set(track.parents)


@pytest.mark.parametrize("hz", [10, 15, 30])
def test_edge_window_grows_beyond_sample_floor_at_real_pose_rates(hz):
    p = MovementPipeline("cam0")
    feed(p, -1 / hz)
    out = []
    for step in range(int(hz * 0.4) + 1):
        out.extend(feed(p, step / hz, [("a", 0.05 + 0.15 * step / hz)]))
    out.extend(p.flush())
    assert any(o.record["entry"]["status"] == "supported" for o in track_outputs(out))


def test_slow_boundary_travel_can_qualify_after_more_than_three_samples():
    p = MovementPipeline("cam0")
    feed(p, 0)
    for step in range(1, 7):
        feed(p, step * 0.5, [("a", 0.04 + step * 0.01)])
    track = track_outputs(p.flush())[0]
    assert track.record["entry"]["status"] == "supported"


@pytest.mark.parametrize("hz", [10, 15, 30])
def test_exit_window_grows_backwards_from_boundary_at_real_pose_rates(hz):
    p = MovementPipeline("cam0")
    out = []
    for step in range(int(hz * 0.4) + 1):
        out.extend(feed(p, step / hz, [("a", 0.89 + 0.15 * step / hz)]))
    for step in range(1, 5):
        out.extend(feed(p, 0.4 + step * 0.5))
    assert any(o.record["exit"]["status"] == "supported" for o in track_outputs(out))


def _inside(output):
    """Whether every parent of a track lies in its stream's stretch: the fixtures' ids are "b:cam:ts" and "m:cam:ts"."""
    stretches = {name: (lo, hi) for name, lo, hi in output.spans}
    for parent in output.parents:
        kind, _, ts = parent.split(":")
        lo, hi = stretches["body" if kind == "b" else "motion"]
        if not lo <= ns(float(ts)) <= hi:
            return False
    return True


@settings(max_examples=150, deadline=None)
@given(
    steps=st.lists(
        st.tuples(
            st.sampled_from([0.25, 0.5, 1.0, 2.5]),  # time since the last frame
            st.booleans(),  # whether a motion record comes with the frame
            st.lists(
                st.sampled_from([("a", 0.2), ("a", 0.4), ("b", 0.6), ("b", 0.8)]), max_size=2, unique_by=lambda t: t[0]
            ),
        ),
        min_size=1,
        max_size=24,
    )
)
def test_a_tracks_stretches_hold_every_record_it_was_made_from(steps):
    p = MovementPipeline("cam0", conf={"max_parents": 16, "max_motion": 8})
    out, ts, last_motion = [], 0.0, None
    for gap, with_motion, tracks in steps:
        ts += gap
        if with_motion:
            p.feed_motion(motion(ts, reference=last_motion if last_motion is not None else ts - 0.5))
            last_motion = ts
        out += p.feed_body(body(ts, tracks))
    out += p.flush()
    tracks = track_outputs(out)
    assert all(o.spans is not None and _inside(o) for o in tracks)


def test_a_snapshot_without_header_ns_keeps_per_frame_parents():
    p = MovementPipeline("cam0")
    feed(p, 0, [("a", 0.2)])
    feed(p, 0.5, [("a", 0.3)])
    snapshot = json.loads(json.dumps(p.snapshot(), allow_nan=False))
    for track in snapshot["tracks"].values():
        for evidence in track["evidence"]:
            del evidence["motion_span_ns"]  # as saved before records kept their header ns
            del evidence["ns"]
    for row in snapshot["motion"]:
        del row["ns"], row["lo_ns"]
    restored = MovementPipeline("cam0", state=snapshot)
    (track,) = track_outputs(restored.flush())
    assert track.spans is None
    assert set(track.parents) == {"b:cam0:0", "b:cam0:0.5", "m:cam0:0", "m:cam0:0.5"}
