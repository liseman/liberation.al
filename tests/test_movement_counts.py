"""Movement counts retain source evidence, uncertainty, and camera-local identity."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from liberational import segment_log
from liberational.memory import movement_counts as movement
from liberational.see.tracks import MovementPipeline

TRACKS = "movement.track"
COVERAGE = "movement.coverage"


def cover(vision, start=0.0, end=20.0, *, cam="cam0", **fields):
    return vision.add(
        COVERAGE,
        ts_start=start,
        ts_end=end,
        cam=cam,
        version=1,
        frames=20,
        supported=[[start, end]],
        gaps=[],
        tracking="complete",
        camera_motion="stable",
        finished=True,
        closure="open",
        **fields,
    )


def fragment(vision, track="run:cam0:1", start=2.0, end=8.0, *, cam="cam0", **fields):
    data = dict(
        ts_start=start,
        ts_end=end,
        cam=cam,
        track_id=track,
        version=1,
        points=[[start, 0.0, 0.5], [end, 1.0, 0.5]],
        supported=[[start, end]],
        gaps=[],
        direction="right",
        camera_motion="stable",
        finished=True,
        closure="exit",
    )
    data.update(fields)
    return vision.add(TRACKS, **data)


def edge(at, *, status="supported", side="left", reason=None):
    return dict(status=status, edge=side, time=[at, at], reason=reason)


def test_supported_entry_exit_and_dwell_keep_camera_and_source_ids(tmp_path, vision):
    coverage = cover(vision)
    source = fragment(vision, entry=edge(2.0), exit=edge(8.0, side="right"))
    result = movement.collect(tmp_path, start=0, end=20)
    camera = result["cameras"]["cam0"]
    assert result["available"] and not result["partial"]
    assert (camera["entry_count"], camera["exit_count"]) == (1, 1)
    assert camera["entries"][0]["source_ids"] == [source["id"]]
    assert camera["source_ids"] == [coverage["id"]]
    assert camera["tracks"][0]["supported_s"] == 6.0
    assert camera["tracks"][0]["directions"][0]["direction"] == "right"
    assert "1 supported entries and 1 supported exits on cam0" in movement.answer_text(result)


def test_track_birth_and_unknown_edges_are_not_entries_or_exits(tmp_path, vision):
    cover(vision)
    fragment(
        vision,
        entry=edge(2, status="unknown", reason="interior-birth"),
        exit=edge(8, status="unknown", reason="occlusion"),
    )
    camera = movement.collect(tmp_path, start=0, end=20)["cameras"]["cam0"]
    assert (camera["entry_count"], camera["exit_count"]) == (0, 0)
    assert len(camera["uncertain_events"]) == 2
    assert camera["partial"]


def test_half_open_boundaries_and_straddling_uncertainty(tmp_path, vision):
    cover(vision)
    fragment(vision, track="at-start", start=9, end=11, entry=edge(10))
    fragment(vision, track="at-end", start=19, end=21, entry=edge(20))
    fragment(
        vision,
        track="straddles-start",
        start=9,
        end=11,
        entry={"status": "supported", "edge": "left", "time": [9.9, 10.1]},
    )
    fragment(
        vision,
        track="straddles-end",
        start=19,
        end=21,
        entry={"status": "supported", "edge": "left", "time": [19.9, 20.0]},
    )
    camera = movement.collect(tmp_path, start=10, end=20)["cameras"]["cam0"]
    assert camera["entry_count"] == 1
    assert camera["entries"][0]["track_id"] == "at-start"
    assert {event["track_id"] for event in camera["uncertain_events"]} == {"straddles-start", "straddles-end"}


def test_exact_track_edges_are_deduplicated_without_merging_simultaneous_people(tmp_path, vision):
    cover(vision)
    a = fragment(vision, entry=edge(2))
    duplicate = fragment(vision, entry=edge(2))
    b = fragment(vision, track="run:cam0:2", entry=edge(2))
    camera = movement.collect(tmp_path, start=0, end=20)["cameras"]["cam0"]
    assert camera["entry_count"] == 2
    assert camera["entries"][0]["source_ids"] == sorted([a["id"], duplicate["id"]])
    assert camera["entries"][1]["source_ids"] == [b["id"]]
    assert len(camera["coarrivals"]) == 1


def test_fragment_dwell_is_a_union_clipped_to_the_query_not_a_sum(tmp_path, vision):
    cover(vision)
    fragment(vision, start=2, end=8)
    fragment(vision, start=6, end=12)
    camera = movement.collect(tmp_path, start=5, end=10)["cameras"]["cam0"]
    assert camera["tracks"][0]["supported_intervals"] == [[5.0, 10.0]]
    assert camera["tracks"][0]["supported_s"] == 5.0


def test_occlusion_and_coverage_gaps_do_not_become_dwell_or_supported_edges(tmp_path, vision):
    coverage = cover(vision)
    coverage["gaps"] = [[5, 7]]
    fragment(vision, start=2, end=10, gaps=[[3, 4]], entry=edge(6), exit=edge(10, side="right"))
    camera = movement.collect(tmp_path, start=0, end=20)["cameras"]["cam0"]
    assert camera["tracks"][0]["supported_intervals"] == [[2.0, 3.0], [4.0, 5.0], [7.0, 10.0]]
    assert camera["tracks"][0]["supported_s"] == 5.0
    assert (camera["entry_count"], camera["exit_count"]) == (0, 1)
    assert camera["partial"]


def test_zero_width_barrier_invalidates_crossing_edge_interval(tmp_path, vision):
    coverage = cover(vision)
    coverage["gaps"] = [[5, 5]]
    fragment(vision, entry={"status": "supported", "edge": "left", "time": [4.9, 5.1]})
    camera = movement.collect(tmp_path, start=0, end=20)["cameras"]["cam0"]
    assert camera["entry_count"] == 0
    assert camera["barriers"][0]["start"] == camera["barriers"][0]["end"] == 5
    assert camera["partial"]


@pytest.mark.parametrize("motion", ["moved", "unknown"])
def test_camera_motion_cannot_prove_a_boundary_event(tmp_path, vision, motion):
    cover(vision)
    fragment(vision, entry=edge(2), camera_motion=motion)
    camera = movement.collect(tmp_path, start=0, end=20)["cameras"]["cam0"]
    assert camera["entry_count"] == 0
    assert camera["tracks"][0]["supported_s"] == 0
    assert camera["tracks"][0]["directions"][0]["direction"] == "unknown"
    assert camera["partial"]


@pytest.mark.parametrize("cameras", [(), ("cam0",)])
def test_missing_coverage_is_unavailable_not_zero(tmp_path, vision, cameras):
    fragment(vision, entry=edge(2))
    result = movement.collect(tmp_path, start=0, end=20, cameras=cameras)
    assert not result["available"] and result["partial"]
    assert result["cameras"]["cam0"]["entry_count"] is None
    assert "cannot establish" in movement.answer_text(result)


def test_supported_empty_coverage_can_establish_zero(tmp_path, vision):
    cover(vision)
    camera = movement.collect(tmp_path, start=0, end=20)["cameras"]["cam0"]
    assert camera["available"] and not camera["partial"]
    assert (camera["entry_count"], camera["exit_count"]) == (0, 0)


def test_partial_tracking_keeps_supported_counts_as_lower_bounds(tmp_path, vision):
    coverage = cover(vision)
    coverage["tracking"] = "partial"
    fragment(vision, entry=edge(2))
    result = movement.collect(tmp_path, start=0, end=20)
    camera = result["cameras"]["cam0"]
    assert camera["entry_count"] == 1 and camera["partial"]
    assert camera["supported_intervals"] == []
    assert movement.answer_text(result).startswith("At least 1 supported entries")


def test_coarrival_can_include_a_neighbor_outside_the_count_window(tmp_path, vision):
    cover(vision)
    fragment(vision, track="neighbor", start=8.5, end=9.5, entry=edge(9))
    fragment(vision, track="inside", start=10, end=12, entry=edge(10))
    camera = movement.collect(tmp_path, start=10, end=11, group_window=3)["cameras"]["cam0"]
    assert camera["entry_count"] == 1
    assert camera["coarrivals"][0]["track_ids"] == ["neighbor", "inside"]


def test_coarrival_is_same_camera_and_all_possible_event_times_must_be_close(tmp_path, vision):
    cover(vision)
    cover(vision, cam="cam1")
    fragment(vision, track="wide", start=1, end=8, entry={"status": "supported", "edge": "left", "time": [1, 5]})
    fragment(vision, track="other", entry=edge(4))
    fragment(vision, track="other-camera", cam="cam1", entry=edge(4))
    result = movement.collect(tmp_path, start=0, end=20, group_window=2)
    assert result["cameras"]["cam0"]["coarrivals"] == []
    assert result["cameras"]["cam1"]["coarrivals"] == []


def test_read_error_is_unavailable_and_does_not_fall_back_to_zero(tmp_path, monkeypatch):
    class Offline:
        def table(self, stream, start, end, names):
            raise ConnectionRefusedError("store unreadable")

    monkeypatch.setattr(segment_log, "at", lambda state: Offline())
    result = movement.collect(tmp_path, start=0, end=20)
    assert result["errors"] == ["ConnectionRefusedError"]
    assert not result["available"] and result["partial"]


def test_reads_only_one_expanded_window_per_stream_and_keeps_no_day_cache(tmp_path, monkeypatch):
    class Empty:
        def __init__(self):
            self.calls = []

        def table(self, stream, start, end, names):
            self.calls.append((stream, start, end))
            return {}

        def range(self, stream, start, end):
            self.calls.append((stream, start, end))
            return []

    log = Empty()
    monkeypatch.setattr(segment_log, "at", lambda state: log)
    movement.collect(tmp_path, start=10, end=20, group_window=3)
    movement.collect(tmp_path, start=10, end=20, group_window=3)
    assert log.calls == [(COVERAGE, 7, 23), (TRACKS, 7, 23)] * 2


@pytest.mark.parametrize("group_window", [-1, float("inf"), float("nan"), True])
def test_invalid_group_window_fails_before_reading(tmp_path, group_window):
    with pytest.raises(ValueError):
        movement.collect(tmp_path, start=0, end=20, group_window=group_window)


def test_unknown_durations_are_unioned_without_losing_gap_reasons(tmp_path, vision):
    coverage = cover(vision)
    coverage["gaps"] = [[5, 7]]
    fragment(vision, start=2, end=10, gaps=[[6, 8]])
    camera = movement.collect(tmp_path, start=0, end=20)["cameras"]["cam0"]
    assert camera["unknown_seconds"] == 3.0
    assert [(u["start"], u["end"]) for u in camera["unknown"]] == [(5.0, 8.0)]
    assert camera["unknown"][0]["reasons"] == ["track-occlusion", "tracking-gap", "unsupported-tracking"]


def test_missing_camera_does_not_hide_supported_evidence_on_another_camera(tmp_path, vision):
    cover(vision)
    result = movement.collect(tmp_path, start=0, end=20, cameras=("cam0", "cam1"))
    assert result["available"] and result["partial"]
    assert result["cameras"]["cam0"]["entry_count"] == 0
    assert result["cameras"]["cam1"]["entry_count"] is None


def test_malformed_coverage_is_actionable_and_cannot_establish_zero(tmp_path, vision):
    coverage = cover(vision)
    coverage["supported"] = [[4, 3]]
    result = movement.collect(tmp_path, start=0, end=20)
    assert not result["available"] and result["partial"]
    assert result["errors"] == ["Invalid movement coverage: " + coverage["id"]]
    assert result["cameras"]["cam0"]["entry_count"] is None


def test_real_track_fragments_count_supported_absence_boundaries_without_adding_dwell(tmp_path, vision):
    pipeline = MovementPipeline("cam0")
    outputs = []
    sequence = [
        (0, None),
        (0.5, 0.05),
        (1, 0.15),
        (1.5, 0.25),
        (2, 0.65),
        (2.5, 0.85),
        (3, 0.95),
        (3.5, None),
        (4, None),
        (4.5, None),
        (5, None),
    ]
    for ts, x in sequence:
        pipeline.feed_motion(
            {
                "id": f"motion:{ts}",
                "cam": "cam0",
                "ts_start": ts,
                "motion": {
                    "status": "measured",
                    "transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "epoch": 0,
                    "reference_ts": ts - 0.5,
                    "current_ts": ts,
                },
            }
        )
        outputs.extend(
            pipeline.feed_body(
                {
                    "id": f"body:{ts}",
                    "cam": "cam0",
                    "ts_start": ts,
                    "boxes": [] if x is None else [[x - 0.04, 0.3, x + 0.04, 0.8, 0.9]],
                    "track_ids": [] if x is None else ["run:cam0:1"],
                }
            )
        )
    outputs.extend(pipeline.flush())
    source_track = next(out.record for out in outputs if out.kind == "track")
    assert source_track["entry"]["status"] == source_track["exit"]["status"] == "supported"
    assert source_track["gaps"] == [[0, 0.5], [3, 4.5]]
    for out in outputs:
        vision.add(TRACKS if out.kind == "track" else COVERAGE, **out.record)
    result = movement.collect(tmp_path, start=0, end=5)
    camera = result["cameras"]["cam0"]
    assert (camera["entry_count"], camera["exit_count"]) == (1, 1)
    assert camera["entries"][0]["time"] == [0.0, 0.5]
    assert camera["exits"][0]["time"] == [3.0, 4.5]
    assert camera["tracks"][0]["supported_s"] == 2.5
    assert camera["tracks"][0]["supported_intervals"] == [[0.5, 3.0]]
    assert not camera["partial"]


def test_track_continuity_barrier_still_vetoes_an_otherwise_supported_edge(tmp_path, vision):
    cover(vision)
    fragment(vision, gaps=[[2, 2]], entry=edge(2))
    camera = movement.collect(tmp_path, start=0, end=20)["cameras"]["cam0"]
    assert camera["entry_count"] == 0
    assert camera["partial"]
    assert camera["barriers"][0]["start"] == camera["barriers"][0]["end"] == 2


def test_real_startup_censor_does_not_discard_a_supported_exit(tmp_path, vision):
    pipeline = MovementPipeline("cam0")
    outputs = []
    for ts, x in [
        (0, 0.2),
        (0.5, 0.6),
        (1, 0.85),
        (1.5, 0.95),
        (2, None),
        (2.5, None),
        (3, None),
        (3.5, None),
        (4, None),
    ]:
        pipeline.feed_motion(
            {
                "id": f"motion:{ts}",
                "cam": "cam0",
                "ts_start": ts,
                "motion": {
                    "status": "measured",
                    "transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "epoch": 0,
                    "reference_ts": ts - 0.5,
                    "current_ts": ts,
                },
            }
        )
        outputs.extend(
            pipeline.feed_body(
                {
                    "id": f"body:{ts}",
                    "cam": "cam0",
                    "ts_start": ts,
                    "boxes": [] if x is None else [[x - 0.04, 0.3, x + 0.04, 0.8, 0.9]],
                    "track_ids": [] if x is None else ["run:cam0:1"],
                }
            )
        )
    outputs.extend(pipeline.flush())
    source = next(out.record for out in outputs if out.kind == "track")
    assert source["entry"] == {"status": "unknown", "reason": "startup"}
    assert source["exit"]["status"] == "supported"
    for out in outputs:
        vision.add(TRACKS if out.kind == "track" else COVERAGE, **out.record)
    result = movement.collect(tmp_path, start=0, end=4)
    camera = result["cameras"]["cam0"]
    assert result["errors"] == []
    assert (camera["entry_count"], camera["exit_count"]) == (0, 1)
    assert camera["uncertain_events"][0]["kind"] == "entry"
    assert camera["uncertain_events"][0]["reason"] == "startup"
    assert camera["partial"]


def test_coverage_cannot_support_time_beyond_its_own_record_header(tmp_path, vision):
    source = cover(vision, start=0, end=20)
    source["supported"] = [[0, 40]]
    result = movement.collect(tmp_path, start=21, end=22)
    camera = result["cameras"]["cam0"]
    assert not camera["available"] and camera["entry_count"] is None
    assert result["errors"]


def test_edge_time_outside_track_header_is_unavailable_not_zero(tmp_path, vision):
    cover(vision, start=0, end=40)
    fragment(vision, start=0, end=20, entry=edge(30))
    result = movement.collect(tmp_path, start=21, end=22)
    camera = result["cameras"]["cam0"]
    assert not camera["available"] and camera["entry_count"] is None
    assert result["errors"]


def test_track_dwell_cannot_extend_beyond_its_own_header(tmp_path, vision):
    cover(vision, start=0, end=40)
    fragment(vision, start=0, end=20, supported=[[0, 40]])
    result = movement.collect(tmp_path, start=21, end=22)
    camera = result["cameras"]["cam0"]
    assert camera["tracks"] == []
    assert not camera["available"] and camera["entry_count"] is None


def test_append_between_reads_cannot_mix_old_empty_tracks_with_new_complete_coverage(tmp_path, vision, monkeypatch):
    original = vision.range
    appended = []

    def racing_read(stream, start, end, cameras=None):
        rows = original(stream, start, end, cameras)
        if not appended:
            cover(vision)
            fragment(vision, entry=edge(2))
            appended.append(True)  # ordinary append deliberately leaves invalidation revisions unchanged
        return rows

    monkeypatch.setattr(vision, "range", racing_read)
    first = movement.collect(tmp_path, start=0, end=20)
    assert not first["available"] and first["partial"]
    assert first["cameras"]["cam0"]["entry_count"] is None
    second = movement.collect(tmp_path, start=0, end=20)
    assert second["cameras"]["cam0"]["entry_count"] == 1


@pytest.mark.parametrize(("frames", "minimum_sources"), [(42, 1), (62, 2)], ids=["eager", "eager-and-natural"])
def test_real_late_born_track_is_published_before_coverage_can_establish_zero(
    tmp_path, vision, frames, minimum_sources
):
    pipeline = MovementPipeline("cam0")
    outputs = []
    for index in range(frames):
        ts = index / 2
        x = None if ts <= 15 else {15.5: 0.05, 16.0: 0.15, 16.5: 0.25}.get(ts, 0.4)
        pipeline.feed_motion(
            {
                "id": f"motion:{ts}",
                "cam": "cam0",
                "ts_start": ts,
                "motion": {
                    "status": "measured",
                    "transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "epoch": 0,
                    "reference_ts": ts - 0.5,
                    "current_ts": ts,
                },
            }
        )
        outputs.extend(
            pipeline.feed_body(
                {
                    "id": f"body:{ts}",
                    "cam": "cam0",
                    "ts_start": ts,
                    "boxes": [] if x is None else [[x - 0.04, 0.3, x + 0.04, 0.8, 0.9]],
                    "track_ids": [] if x is None else ["run:cam0:late"],
                }
            )
        )
    # Do not flush at shutdown: the running worker must publish active tracks before its coverage watermark.
    for out in outputs:
        vision.add(TRACKS if out.kind == "track" else COVERAGE, **out.record)
    camera = movement.collect(tmp_path, start=14, end=19)["cameras"]["cam0"]
    assert camera["available"]
    assert camera["entry_count"] == 1
    assert camera["entries"][0]["time"] == [15.0, 15.5]
    assert len(camera["entries"][0]["source_ids"]) >= minimum_sources


_spans = st.lists(
    st.tuples(st.integers(0, 60), st.integers(0, 15)).map(lambda t: (float(t[0]), float(t[0] + t[1]))), max_size=10
)


@given(support=_spans, barriers=_spans, observed=_spans.filter(bool), gaps=_spans, at=_spans.filter(bool))
def test_the_day_index_answers_as_the_whole_day_lists_do(support, barriers, observed, gaps, at):
    """The per-camera index is only a faster way to ask the same questions of a day's spans."""
    supported = movement._subtract(support, barriers)
    index = movement._Index(supported, barriers)
    span = (min(lo for lo, _ in observed + gaps), max(hi for _, hi in observed + gaps))
    expected = movement._subtract(movement._intersect(observed, supported), [*barriers, *gaps])
    assert index.own(observed, gaps, span) == expected
    points = [lo for lo, hi in gaps if lo == hi]
    for edge in at:
        reference = movement._covered(edge, supported, [*barriers, *((p, p) for p in points)])
        assert index.covered(edge, points) == reference


def mixed_day(add):
    """Coverage of every kind `collect` tells apart, and a track across it: mostly plain rows, as a real day is."""
    for at in range(0, 40, 2):
        add(COVERAGE, at, at + 2)
    add(COVERAGE, 40, 44, tracking="partial")
    add(COVERAGE, 44, 48, gaps=[[45, 46]])
    add(COVERAGE, 48, 52, camera_motion="moved")
    add(COVERAGE, 52, 56, frames=0)
    add(COVERAGE, 56, 60, supported=[[59, 57]])  # reversed: an invalid row
    add(COVERAGE, 60, 64, supported=[[60, 61], [62, 64]])
    add(TRACKS, 4, 30, entry=edge(4.0), exit=edge(30.0, side="right"))


def test_the_column_path_for_plain_coverage_answers_as_the_row_checks_do(tmp_path, vision, monkeypatch):
    def add(stream, start, end, **fields):
        if stream == TRACKS:
            return fragment(vision, start=start, end=end, **fields)
        row = dict(
            version=1,
            frames=20,
            supported=[[start, end]],
            gaps=[],
            tracking="complete",
            camera_motion="stable",
            finished=True,
            closure="open",
        )
        return vision.add(COVERAGE, ts_start=start, ts_end=end, cam="cam0", **{**row, **fields})

    mixed_day(add)
    fast = movement.collect(tmp_path, start=0, end=70)
    plain = movement._plain(vision.table(COVERAGE, 0, 70, movement.COVERAGE_COLUMNS)["cam0"])
    assert plain.tolist() == [True] * 20 + [False] * 6
    monkeypatch.setattr(movement, "_plain", lambda columns: np.zeros(len(columns["ts_ns"]), dtype=bool))
    assert movement.collect(tmp_path, start=0, end=70) == fast, "every row through the row checks: the same answer"
    camera = fast["cameras"]["cam0"]
    assert (camera["entry_count"], camera["exit_count"]) == (None, None), "the invalid row leaves no count"
    assert len(fast["errors"]) == 1
    assert len(camera["source_ids"]) == 23, "the steady rows with frames: plain, partial, gapped, and two-span"


def test_plain_coverage_is_told_from_the_stores_own_columns(tmp_path, monkeypatch):
    pytest.importorskip("continuum_py").SegmentWriter  # noqa: B018 (the native store must be built)
    store = tmp_path / "segments"
    writers = segment_log.Writers(store)
    T = 1_800_000_000.0

    def add(stream, start, end, **fields):
        row = dict(
            id=segment_log.new_id(),
            ts_start=T + start,
            ts_end=T + end,
            version=1,
            frames=20,
            gaps=[],
            supported=[[T + start, T + end]],
            tracking="complete",
            camera_motion="stable",
            finished=True,
            closure="open",
        )
        if stream == COVERAGE:
            shifted = {k: [[T + a, T + b] for a, b in v] for k, v in fields.items() if k in ("supported", "gaps")}
            writers.append(COVERAGE, "cam0", [{**row, **fields, **shifted}])

    mixed_day(add)
    writers.close()
    log = segment_log.Log(store)
    monkeypatch.setattr(segment_log, "at", lambda state=None: log)
    columns = log.table(COVERAGE, T, T + 70, movement.COVERAGE_COLUMNS)["cam0"]
    assert movement._plain(columns).tolist() == [True] * 20 + [False] * 6
    fast = movement.collect(tmp_path, start=T, end=T + 70)
    monkeypatch.setattr(movement, "_plain", lambda columns: np.zeros(len(columns["ts_ns"]), dtype=bool))
    assert movement.collect(tmp_path, start=T, end=T + 70) == fast
    # Complete tracking only: the 20 plain rows (40 s), the row around its gap (3 s), and the two-span row (3 s).
    assert fast["cameras"]["cam0"]["supported_s"] == pytest.approx(46)
