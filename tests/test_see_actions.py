"""see-actions' per-frame path: tracked pose rows, coverage runs, and gaps (see/actions.py)."""

import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from liberational import segment_log
from liberational.see import action_animals, actions, stgcn, vision_backend
from liberational.see import hearts as hearts_module
from liberational.see.vision_backend import People

T = 500_000 * 3_600_000_000_000  # on an hour boundary, so a few seconds of rows stay in one segment
STEP = 40_000_000  # 25 fps keeps the flush boundaries on whole frames
# The starting activities' NTU-120 classes the scripted model scores (memory/activity_kinds.py's STARTING)
NTU = {"sit_down": "sitting down", "stand_up": "standing up", "fall": "falling", "drink": "drink water"}


class Rows:
    def __init__(self) -> None:
        self.appends: list[dict[str, Any]] = []

    def append(self, columns: dict[str, Any]) -> None:
        self.appends.append(dict(columns))


def person(x: float, score: float = 0.9, joint: tuple[float, float, float] = (50.0, 25.0, 0.9)) -> People:
    """One person whose box starts at x (normalized), with every joint at `joint` (frame pixels)."""
    box = np.array([[x, 0.2, x + 0.2, 0.9, score]], np.float32)
    return People(box, np.tile(np.array(joint, np.float32), (1, actions.JOINTS, 1)))


def nobody() -> People:
    return People(np.zeros((0, 5), np.float32), np.zeros((0, 0, 3), np.float32))


@dataclass(frozen=True)
class Decoded:
    """A decoded frame, as the shared decode's `DecodedFrame` carries it."""

    id: str
    capture_ns: int
    gap_before: bool = False
    pixels: np.ndarray = field(default_factory=lambda: np.zeros((150, 200), np.uint8))  # NV12 of a 200 x 100 frame
    source_header: dict[str, Any] = field(default_factory=dict)
    color_range: str = "full"
    color_matrix: str = "bt601"


@dataclass
class Script:
    """A backend whose images are the frames themselves, 200 wide and 100 high, with scripted people."""

    frames: list[People]
    uploaded: list[str] = field(default_factory=list)
    cropped: list[tuple[str, tuple[int, int, int, int]]] = field(default_factory=list)
    name: str = "script"

    def upload(self, frame: vision_backend.Frame) -> str:
        self.uploaded.append(frame.id)
        return frame.id

    def captures(self, frame: vision_backend.Frame) -> bool:
        return False

    def size(self, image: str) -> tuple[int, int]:
        return 100, 200

    def people(self, image: str) -> People:
        return self.frames[self.uploaded.index(image)]

    def embed_animals(self, image: str, boxes: np.ndarray) -> np.ndarray:
        raise AssertionError("the pipeline embeds no animals")

    def verify_animals(self, image: str, boxes: np.ndarray, species: list[str]) -> list[bool]:
        raise AssertionError("the pipeline checks no animals")

    def crop(self, image: str, box: tuple[int, int, int, int]) -> np.ndarray:
        self.cropped.append((image, box))
        x1, y1, x2, y2 = box
        return np.zeros((y2 - y1, x2 - x1, 3), np.uint8)

    def score(self, windows: np.ndarray) -> np.ndarray:
        raise AssertionError("nothing scores without windows or spans to write")

    def shift(self, image: str) -> tuple[float, float] | None:
        return (0.0, 0.0)


def frame_id(i: int) -> str:
    return str(uuid.UUID(int=0x0199A000_0000_7000_8000_000000000000 + i))


LONG = 6_000_000_000  # a gap longer than actions.LOST_S


def at(i: int, gaps: frozenset[int], long: bool = True) -> int:
    """Frame i's capture time: 25 fps, with every gap at or before it LONG later when the gaps are long."""
    return T + i * STEP + (LONG * sum(g <= i for g in gaps) if long else 0)


def run(
    frames: list[People],
    gaps: frozenset[int] = frozenset(),
    *,
    poses: Any = None,
    coverage: Any = None,
    long: bool = True,
    model: str = "test-model",
) -> tuple[Any, Any, actions.ActionPipeline]:
    poses, coverage = poses if poses is not None else Rows(), coverage if coverage is not None else Rows()
    pipeline = actions.ActionPipeline(Script(frames), poses, coverage, model)
    for i in range(len(frames)):
        pipeline.feed(Decoded(frame_id(i), at(i, gaps, long), gap_before=i in gaps))
    pipeline.flush()
    return poses, coverage, pipeline


def coverage_rows(coverage: Rows) -> list[tuple[int, int, int]]:
    return [(int(a["ts_ns"][0]), int(a["end_ns"][0]), int(a["frames"][0])) for a in coverage.appends]


def test_a_tracked_person_is_a_pose_row_every_frame_appended_about_once_a_second():
    poses, coverage, _ = run([person(0.1 + i * 0.002) for i in range(45)])
    # A track counts from its fifth frame (MIN_HITS). Frame 25 is 1 s after frame 0, so the first append holds
    # frames 4 to 25 and the closing flush the rest.
    assert [len(a["frame_id"]) for a in poses.appends] == [22, 19]
    assert {int(t) for a in poses.appends for t in a["track_id"]} == {1}
    assert list(poses.appends[1]["frame_id"][:2]) == [frame_id(26), frame_id(27)]
    assert int(poses.appends[1]["ts_ns"][0]) == T + 26 * STEP
    assert coverage_rows(coverage) == [(T, T + 25 * STEP, 26), (T + 25 * STEP, T + 44 * STEP, 19)]


def test_the_pose_and_coverage_rows_record_the_configured_model_name():
    name = vision_backend.model_name("cuda")
    poses, coverage, _ = run([person(0.1) for _ in range(actions.MIN_HITS + 1)], model=name)
    assert {model for append in poses.appends for model in append["model"]} == {name}
    assert {model for append in coverage.appends for model in append["model"]} == {name}


def test_joints_are_stored_normalized_and_scaled_and_an_unplaced_joint_stays_unplaced():
    people = person(0.1)
    people.joints[0, 1] = (-1.0, -1.0, 0.0)
    poses, _, _ = run([people] * actions.MIN_HITS)
    row = poses.appends[0]
    assert row["joints"].dtype == np.int16
    assert list(row["joints"][0, :6]) == [2500, 2500, 9000, -10000, -10000, 0]  # (50, 25) of a 200 x 100 frame
    assert row["box"][0] == pytest.approx([0.1, 0.2, 0.3, 0.9, 0.9])
    assert list(row["model"]) == ["test-model"]


def test_a_gap_ends_the_tracks_and_the_coverage_run_so_neither_bridges_it():
    poses, coverage, _ = run([person(0.1) for _ in range(20)], gaps=frozenset({10}))
    tracks = [int(t) for a in poses.appends for t in a["track_id"]]
    assert tracks == [1] * 6 + [2] * 6  # each counted from its fifth frame: 4 to 9, then 14 to 19
    after = T + 10 * STEP + LONG
    assert coverage_rows(coverage) == [(T, T + 9 * STEP, 10), (after, after, 1), (after, after + 9 * STEP, 9)]


def test_a_short_gap_keeps_the_track_and_only_breaks_the_coverage_run():
    # A frame lapped in the decoder's ring: the tracker bridges it, so the person keeps one track ID.
    poses, coverage, _ = run([person(0.1) for _ in range(20)], gaps=frozenset({10}), long=False)
    assert [int(t) for a in poses.appends for t in a["track_id"]] == [1] * 16
    assert coverage_rows(coverage) == [(T, T + 9 * STEP, 10), (T + 10 * STEP, T + 19 * STEP, 10)]


def test_a_track_seen_for_fewer_than_min_hits_frames_writes_nothing():
    blip = [person(0.6)] * (actions.MIN_HITS - 2) + [nobody()] * 50
    poses, coverage, _ = run(blip)
    assert poses.appends == []
    assert coverage_rows(coverage)[0][2] > 0  # its frames still count as watched


def test_a_weak_detection_starts_no_track_but_the_frames_still_count_as_watched():
    poses, coverage, _ = run([person(0.1, score=0.4) for _ in range(5)] + [nobody()] * 3)
    assert poses.appends == []
    assert coverage_rows(coverage) == [(T, T + 7 * STEP, 8)]


def test_a_detection_without_a_confident_pose_starts_no_track():
    poses, _, _ = run([person(0.1, joint=(50.0, 25.0, 0.3)) for _ in range(10)])
    assert poses.appends == []


def test_a_tracked_person_whose_pose_fades_keeps_their_track():
    sure, faded = person(0.1), person(0.1, joint=(50.0, 25.0, 0.3))
    poses, _, _ = run([sure] * 6 + [faded] * 6)
    rows = [t for a in poses.appends for t in a["track_id"]]
    assert len(rows) == 8  # frames 4 to 11: counted from the fifth hit, through the faded frames
    assert len(set(rows)) == 1


def test_a_track_whose_pose_stays_unsure_for_five_seconds_ends_and_a_sure_pose_starts_another():
    sure, faded = person(0.1), person(0.1, joint=(50.0, 25.0, 0.3))
    poses, _, _ = run([sure] * 6 + [faded] * 130 + [sure] * 10)
    rows = [int(t) for a in poses.appends for t in a["track_id"]]
    # The last sure pose is frame 5; frame 131 is 5.04 s after it, so the track's rows are frames 4 to 130. The
    # faded frames after start nothing, and the sure ones from frame 136 start a track counted from frame 140.
    assert [rows.count(t) for t in sorted(set(rows))] == [127, 6]


def test_a_long_track_keeps_a_thinned_path_for_its_crossing_check():
    _, _, pipeline = run([person(0.1) for _ in range(250)])  # 10 s
    (track,) = pipeline.tracker.active
    # 10 s at 25 fps: one point about every PATH_SPACING_S, plus the latest, not every frame and not only the end.
    assert 10 / actions.PATH_SPACING_S - 5 <= len(track.points) <= 10 / actions.PATH_SPACING_S + 3


def test_rows_round_trip_through_the_segment_store(tmp_path):
    continuum_py = pytest.importorskip("continuum_py")
    if not hasattr(continuum_py, "SegmentWriter"):
        pytest.skip("this continuum_py predates the segment store")
    poses = continuum_py.SegmentWriter(tmp_path, "pose.track", "cam0", actions.POSE_SCHEMA)
    coverage = continuum_py.SegmentWriter(tmp_path, "action.coverage", "cam0", actions.COVERAGE_SCHEMA)
    run([person(0.1 + i * 0.002) for i in range(30)], gaps=frozenset({20}), poses=poses, coverage=coverage)
    got = continuum_py.read_segments(tmp_path, "pose.track", actions.POSE_SCHEMA, T, T + 3_600_000_000_000)
    kept = [*range(4, 20), *range(24, 30)]  # each track counted from its fifth frame
    assert got["ts_ns"].tolist() == [at(i, frozenset({20})) for i in kept]
    assert got["frame_id"] == [frame_id(i) for i in kept]
    assert got["track_id"].tolist() == [1] * 16 + [2] * 6
    assert got["joints"].shape == (22, actions.JOINTS * 3)
    assert got["joints"][0, :3].tolist() == [2500, 2500, 9000]
    spans = continuum_py.read_segments(tmp_path, "action.coverage", actions.COVERAGE_SCHEMA, T, T + 3_600_000_000_000)
    after = T + 20 * STEP + LONG
    assert list(zip(spans["ts_ns"].tolist(), spans["end_ns"].tolist(), spans["frames"].tolist(), strict=True)) == [
        (T, T + 19 * STEP, 20),
        (after, after, 1),  # the gap's frame is more than FLUSH_NS after the last flush, so it flushes at once
        (after, after + 9 * STEP, 9),
    ]


def test_the_pipeline_never_needs_torch():
    # What touches the accelerator stays behind the backend seam, so the rig's Mac backend can replace CUDA.
    blocked = "import sys; sys.modules['torch'] = None; import liberational.see.actions"
    subprocess.run([sys.executable, "-c", blocked], check=True)


def test_a_machine_without_a_supported_accelerator_says_so(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(vision_backend.NoBackend, match="CUDA"):
        vision_backend.load("unused")


class Scored(Script):
    """A backend whose action model says what `kinds_at` gives for each window it scores, by call: each key a starting
    kind (its NTU class) or an NTU class name."""

    def __init__(self, frames: list[People], kinds_at: list[dict[str, float]]) -> None:
        super().__init__(frames)
        self.kinds_at, self.calls = kinds_at, 0

    def score(self, windows: np.ndarray) -> np.ndarray:
        out = np.full((len(windows), len(stgcn.CLASSES)), 0.001, np.float32)
        for row in out:
            for kind, p in self.kinds_at[min(self.calls, len(self.kinds_at) - 1)].items():
                row[stgcn.CLASSES.index(NTU.get(kind, kind))] = p
            self.calls += 1
        return out


def scored(
    n: int,
    kinds_at: list[dict[str, float]],
    gaps: frozenset[int] = frozenset(),
    kinds: dict[str, tuple[int, ...]] | None = None,
) -> tuple[Rows, list[dict[str, Any]], actions.ActionPipeline]:
    windows, spans = Rows(), []
    pipeline = actions.ActionPipeline(
        Scored([person(0.1) for _ in range(n)], kinds_at),
        Rows(),
        Rows(),
        "test-model",
        windows=windows,
        spans=spans.extend,
        action_model="test-actions",
        first_track=1_000_001,
        kinds=None if kinds is None else lambda: kinds,
    )
    for i in range(n):
        pipeline.feed(Decoded(frame_id(i), at(i, gaps), gap_before=i in gaps))
    pipeline.close()
    return windows, spans, pipeline


def test_a_full_window_is_scored_every_half_second_and_a_held_kind_is_one_span():
    windows, spans, _ = scored(125, [{"sit_down": 0.9}])
    # The track counts from frame 4, so frame 103 fills the first 100-frame window; at 25 fps the next score comes
    # 0.52 s later, at frame 116.
    assert [int(e) for a in windows.appends for e in a["end_ns"]] == [T + 103 * STEP, T + 116 * STEP]
    assert [int(s) for a in windows.appends for s in a["ts_ns"]] == [T + 4 * STEP, T + 17 * STEP]
    top = windows.appends[0]
    assert int(top["classes"][0, 0]) == stgcn.CLASSES.index("sitting down")
    assert float(top["scores"][0, 0]) == pytest.approx(0.9)
    assert [int(t) for t in top["track_id"]] == [1_000_001, 1_000_001]
    (span,) = spans
    assert (span["kind"], span["track_id"], span["subject_type"]) == ("sit_down", "1000001", "person")
    assert (span["ts_start"], span["ts_end"]) == pytest.approx(((T + 4 * STEP) / 1e9, (T + 116 * STEP) / 1e9))
    assert span["attributes"]["observations"] == 2
    assert "sitting down" in span["uncertainty"]["interpretation"]


SQUAT, SIT = stgcn.CLASSES.index("squat down"), stgcn.CLASSES.index("sitting down")


@pytest.mark.parametrize(
    ("kinds", "scores", "expected"),
    [
        pytest.param(None, {"sitting down": 0.9}, ["sit_down"], id="starting-list"),
        pytest.param({"squats": (SQUAT, SIT)}, {"squat down": 0.3, "sitting down": 0.3}, ["squats"], id="summed"),
        pytest.param({"squats": (SQUAT,)}, {"sitting down": 0.9}, [], id="sit-down-deleted"),
    ],
)
def test_the_owners_activities_are_scored_from_their_classes(kinds, scores, expected):
    _, spans, _ = scored(125, [scores], kinds=kinds)
    assert [s["kind"] for s in spans] == expected


def test_a_span_says_which_of_its_classes_it_was_read_from():
    _, (span,), _ = scored(125, [{"squat down": 0.6}], kinds={"squats": (SQUAT, SIT)})
    assert "NTU-120 'squat down' or 'sitting down', peak 0.60" in span["uncertainty"]["interpretation"]


def test_a_short_dip_below_the_threshold_keeps_one_span():
    # Windows end at frames 103, 116, 129, and 142: seen, not seen, then seen twice, all within HOLD_NS.
    _, spans, _ = scored(144, [{"stand_up": 0.8}, {"stand_up": 0.2}, {"stand_up": 0.7}, {"stand_up": 0.7}])
    assert [(s["kind"], s["attributes"]["observations"]) for s in spans] == [("stand_up", 3)]


def test_a_kind_unseen_for_longer_than_the_hold_closes_its_span_and_a_return_opens_another():
    # Windows end every 13 frames from frame 103: seen, unseen for eleven (6.2 s), then seen at frame 259.
    _, spans, _ = scored(260, [{"stand_up": 0.8}, *[{"stand_up": 0.2}] * 11, {"stand_up": 0.7}])
    assert [(s["ts_end"], s["attributes"]["observations"]) for s in spans] == [
        (pytest.approx((T + 103 * STEP) / 1e9), 1),
        (pytest.approx((T + 259 * STEP) / 1e9), 1),
    ]


def test_a_person_cut_off_by_the_frames_edge_is_tracked_but_not_scored():
    poses, windows = Rows(), Rows()
    pipeline = actions.ActionPipeline(
        Scored([person(0.0) for _ in range(125)], [{"fall": 0.9}]), poses, Rows(), "m", windows=windows
    )
    for i in range(125):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    pipeline.close()
    assert sum(len(a["ts_ns"]) for a in poses.appends) == 121
    assert windows.appends == []


def sure_joints(n: int) -> np.ndarray:
    """A pose whose first `n` joints are above JOINT_SURE and the rest just under it."""
    joints = np.zeros((actions.JOINTS, 3), np.float32)
    joints[:, 2] = actions.JOINT_SURE
    joints[:n, 2] = 0.9
    return joints


@pytest.mark.parametrize(
    ("sure", "scored"), [pytest.param(6, 1, id="six-joints"), pytest.param(5, 0, id="five-joints")]
)
def test_a_pose_with_few_confident_joints_is_tracked_but_not_scored(sure, scored):
    backend = Scored([], [{"fall": 0.9}])
    scoring = actions.Scoring(backend.score, "m")
    for i in range(stgcn.CLIP):
        scoring.add(1, T + i * STEP, sure_joints(sure), still_box())
    scoring.due(T + stgcn.CLIP * STEP)
    assert backend.calls == scored


def still_box(dy: float = 0.0) -> np.ndarray:
    return np.array([0.4, 0.2 + dy, 0.6, 0.6 + dy, 0.9], np.float32)  # 0.4 high, so STILL_MOVE is 0.02


def test_a_box_that_stays_put_for_two_minutes_stops_being_scored_until_it_moves():
    backend, second = Scored([], [{"fall": 0.1}]), 1_000_000_000
    scoring = actions.Scoring(backend.score, "m")
    joints = sure_joints(17)
    calls = []
    for i in range(140):  # one frame a second; it wobbles by less than STILL_MOVE
        scoring.add(1, T + i * second, joints, still_box(0.015 * (i % 2)))
        scoring.due(T + i * second)
        calls.append(backend.calls)
    # The window is full at second 99 and scored each second after, until the box has been still for 120 s.
    assert calls[119] - calls[98] == 21
    assert calls[139] == calls[119]
    scoring.add(1, T + 140 * second, joints, still_box(0.03))
    scoring.due(T + 140 * second)
    assert backend.calls == calls[139] + 1


def test_kinds_under_the_threshold_make_windows_but_no_spans():
    windows, spans, _ = scored(110, [{"fall": 0.3}])
    assert sum(len(a["ts_ns"]) for a in windows.appends) == 1
    assert spans == []


def test_a_gap_closes_spans_and_the_window_starts_again_from_empty():
    windows, spans, _ = scored(212, [{"drink": 0.9}], gaps=frozenset({105}))
    assert [s["track_id"] for s in spans] == ["1000001", "1000002"]  # the gap ended the first track
    ends = [int(e) for a in windows.appends for e in a["end_ns"]]
    assert ends[0] == T + 103 * STEP
    # The new track counts from frame 109, so its first full window ends at frame 208.
    assert T + 208 * STEP + LONG in ends
    assert all(not (T + 105 * STEP + LONG <= e < T + 208 * STEP + LONG) for e in ends)


class FaceCalls:
    """A face stage that records what the pipeline hands it, cropping one fixed box per track it's given."""

    def __init__(self) -> None:
        self.fed: list[tuple[str, int, list[int], list[tuple[float, float]]]] = []
        self.patches: list[tuple[int, ...]] = []
        self.calls: list[str] = []

    def feed(self, crop, width, height, capture_ns, frame_id, people, frame_seq=0) -> None:
        people = list(people)
        self.fed.append((frame_id, capture_ns, [t for t, _ in people], [tuple(j[0][:2]) for _, j in people]))
        for _ in people:
            self.patches.append(crop((10, 10, 40, 50)).shape)

    def end(self, track_id: int) -> None:
        self.calls.append(f"end {track_id}")

    def gap(self) -> None:
        self.calls.append("gap")

    def flush(self, now_ns: int | None) -> None:
        self.calls.append("flush" if now_ns is not None else "close")


def test_the_face_stage_gets_each_confirmed_track_with_its_pixel_joints_and_crops_from_the_backend():
    faces = FaceCalls()
    # A track counts from its fifth frame (MIN_HITS), so the stage gets frames 4 and 5.
    backend = Script([person(0.1)] * (actions.MIN_HITS + 1) + [nobody()])
    pipeline = actions.ActionPipeline(backend, Rows(), Rows(), "test-model", faces=cast(Any, faces))
    for i in range(actions.MIN_HITS + 2):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    pipeline.close()
    assert [(f, t) for f, _, t, _ in faces.fed] == [(frame_id(4), [1]), (frame_id(5), [1])]
    assert faces.fed[0][3] == [(50.0, 25.0)], "joints in frame pixels, as the crop needs them"
    assert backend.cropped == [(frame_id(4), (10, 10, 40, 50)), (frame_id(5), (10, 10, 40, 50))]
    assert faces.patches == [(40, 30, 3), (40, 30, 3)]
    assert faces.calls == ["gap", "flush", "close"], "closing ends the tracks, writes, then closes open visits"


class Pets(Script):
    """A backend whose image check passes every animal and whose embeddings are all one unit vector."""

    def verify_animals(self, image: str, boxes: np.ndarray, species: list[str]) -> list[bool]:
        return [True] * len(boxes)

    def embed_animals(self, image: str, boxes: np.ndarray) -> np.ndarray:
        return np.tile(np.eye(1, 384, dtype=np.float32), (len(boxes), 1))


def test_an_animal_in_the_detector_pass_is_written_by_the_animal_stage_and_a_stop_ends_its_coverage():
    cat = People(nobody().boxes, nobody().joints, np.array([[0.5, 0.5, 0.8, 0.9, 0.9, 15]], np.float32))
    frames = [cat] * 75  # three seconds: past the stage's confirmation
    backend = Pets(frames)
    written: dict[str, list[dict[str, Any]]] = {}
    stage = action_animals.Animals("cam0", lambda stream, rows: written.setdefault(stream, []).extend(rows), backend)
    pipeline = actions.ActionPipeline(backend, Rows(), Rows(), "test-model", animals=stage)
    for i in range(len(frames)):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    pipeline.close()
    assert {r["species"] for r in written[segment_log.ANIMAL_DETECTIONS]} == {"cat"}
    spans = [(r["ts_start"], r["ts_end"]) for r in written[segment_log.ANIMAL_COVERAGE]]
    assert spans[0][0] == pytest.approx(T / 1e9)
    assert spans[-1][1] <= (T + 74 * STEP) / 1e9, "coverage spans the frames looked at, no further"


class Fisheye(Scored):
    """A fisheye backend: frame joints at (50, 25) in frame pixels, and view joints at (96, 64) in a 192 x 256 view."""

    def __init__(self, frames: list[People], kinds_at: list[dict[str, float]]) -> None:
        super().__init__(frames, kinds_at)
        self.scored: list[np.ndarray] = []

    def people(self, image: str) -> People:
        found = super().people(image)
        view = np.tile(np.array((96.0, 64.0, 0.9), np.float32), (len(found.boxes), actions.JOINTS, 1))
        return People(found.boxes, found.joints, view_joints=view, view_size=(192, 256))

    def score(self, windows: np.ndarray) -> np.ndarray:
        self.scored.append(windows)
        return super().score(windows)


def test_a_fisheye_camera_scores_each_persons_view_joints_not_the_frames():
    backend = Fisheye([person(0.1) for _ in range(110)], [{}])
    pipeline = actions.ActionPipeline(backend, Rows(), Rows(), "m", windows=Rows(), spans=lambda s: None)
    for i in range(110):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    (windows,) = backend.scored
    # (96, 64) of a 192 x 256 view is (0.5, 0.25); the frame joints (50, 25) of 200 x 100 would be (0.25, 0.25).
    assert windows[0, 0, 0].tolist() == pytest.approx([0.5, 0.25, 0.9])


def test_a_closed_span_names_its_tracks_person_or_no_one():
    who = {1_000_001: ("0190a5b8-0000-7000-8000-00000000000a", "Dana")}
    scoring = actions.Scoring(Scored([], [{"sit_down": 0.9}]).score, "m", who=lambda t: who.get(t, (None, None)))
    for track in (1_000_001, 1_000_002):
        for i in range(stgcn.CLIP):
            scoring.add(track, T + i * STEP, sure_joints(17), still_box())
    scoring.due(T + stgcn.CLIP * STEP)
    scoring.gap()
    by_track = {s["track_id"]: (s["person"], s["person_name"]) for s in scoring.take_spans()}
    assert by_track == {"1000001": ("0190a5b8-0000-7000-8000-00000000000a", "Dana"), "1000002": (None, None)}


# ---- people on a screen ----

ON_TV = np.array([[0.0, 0.1, 0.35, 0.95, 0.9, 62]], np.float32)  # a TV whose box holds person(0.1) whole


def shown(people: People, tv: np.ndarray = ON_TV) -> People:
    """The same people, with a TV in the same detector pass."""
    return People(people.boxes, people.joints, people.animals, screens=tv)


class Ended:
    """A movement stage that records the tracks it samples and the ones that end through it."""

    def __init__(self) -> None:
        self.ends: list[int] = []
        self.sampled: list[list[int]] = []

    def due(self, now_ns: int) -> bool:
        return True

    def sample(self, frame_id, capture_ns, people, shift) -> None:
        self.sampled.append([t for t, _ in people])

    def ended(self, track) -> None:
        self.ends.append(track.id)

    def gap(self) -> None:
        pass

    def flush_coverage(self) -> None:
        pass


def test_a_person_a_tv_shows_writes_no_pose_row_and_reaches_neither_faces_nor_movement():
    faces, movement = FaceCalls(), Ended()
    # 40 frames of someone on the TV, then nobody for longer than LOST_S, so the track ends while frames go on.
    gone = int(actions.LOST_S * 1e9 / STEP) + 10
    backend = Script([shown(person(0.1))] * 40 + [nobody()] * gone)
    rows = Rows()
    pipeline = actions.ActionPipeline(
        backend, rows, Rows(), "test-model", faces=cast(Any, faces), movement=cast(Any, movement)
    )
    for i in range(40 + gone):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    counts = pipeline.screens.counts
    pipeline.close()
    assert rows.appends == []
    assert faces.fed == []
    assert all(sampled == [] for sampled in movement.sampled)
    assert movement.ends == [], "its end is no crossing or walk-by"
    assert (counts.detections, counts.masked, counts.born) == (40, 40, 1)


def test_a_track_born_on_a_screen_counts_from_the_frame_it_steps_out_of_it():
    # From x = 0.1, 0.0045 a frame: the box is less than SCREEN_INSIDE inside the padded TV from x > 0.2, frame 23,
    # and it has been out for RELEASE_NS (25 frames) at frame 48.
    frames = [shown(person(0.1 + i * 0.0045)) for i in range(70)]
    poses, _, pipeline = run_with(frames)
    written = [f for a in poses.appends for f in a["frame_id"]]
    assert written[0] == frame_id(48)
    assert {int(t) for a in poses.appends for t in a["track_id"]} == {1}, "the same track, found to be real"
    assert pipeline.screens.counts.released == 1


def test_someone_who_walks_in_and_stands_in_front_of_a_tv_stays_counted():
    # Born at x = 0.5, outside the TV, then still at x = 0.1, inside its box.
    frames = [shown(person(0.5 - i * 0.02)) for i in range(20)] + [shown(person(0.1))] * 25
    poses, _, _ = run_with(frames)
    written = [f for a in poses.appends for f in a["frame_id"]]
    assert written[-1] == frame_id(44)
    assert len(written) == 45 - actions.MIN_HITS + 1


def test_a_screen_zone_keeps_out_who_it_shows_without_a_detected_screen():
    zone = ((0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0))
    poses, _, _ = run_with([person(0.1)] * 30, screen_zones=lambda: [zone])
    assert poses.appends == []


def run_with(frames: list[People], **kw: Any) -> tuple[Rows, Rows, actions.ActionPipeline]:
    poses, coverage = Rows(), Rows()
    pipeline = actions.ActionPipeline(Script(frames), poses, coverage, "test-model", **kw)
    for i in range(len(frames)):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    pipeline.flush()
    return poses, coverage, pipeline


class Playing(Script):
    """A backend whose crops are a new picture every frame: a TV playing."""

    def crop(self, image: str, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        seed = int(uuid.UUID(image)) & 0xFFFF
        return np.random.default_rng(seed).integers(0, 256, (y2 - y1, x2 - x1, 3), dtype=np.uint8)


def _tv_zone() -> tuple[tuple[float, float], ...]:
    """A screen zone drawn over the test's TV."""
    x1, y1, x2, y2 = (float(v) for v in ON_TV[0][:4])
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def test_a_detected_tv_alone_is_never_on(tmp_path, monkeypatch):
    """The detector's TV class also finds monitors: only a drawn screen zone counts as a TV that's on."""
    from liberational.see import screens

    monkeypatch.setattr(screens, "MIN_ON_NS", 0)
    path = tmp_path / "screens_on.json"
    written = screens.ScreensOn(path, clock=lambda: 10**9)
    frames = [shown(nobody())] * 150
    pipeline = actions.ActionPipeline(Playing(frames), Rows(), Rows(), "test-model", camera="cam0", screens_on=written)
    for i in range(len(frames)):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    pipeline.gap()
    assert written.cams == {}


def test_a_playing_tv_is_written_as_on_for_tagging_what_was_heard_then(tmp_path, monkeypatch):
    from liberational.see import screens

    monkeypatch.setattr(screens, "MIN_ON_NS", 0)  # 6 s here, not a show's minutes

    path = tmp_path / "screens_on.json"
    written = screens.ScreensOn(path, clock=lambda: 10**9)
    frames = [shown(nobody())] * 150  # 6 s at 25 fps, a TV in every frame, nobody in the room
    pipeline = actions.ActionPipeline(
        Playing(frames),
        Rows(),
        Rows(),
        "test-model",
        camera="cam0",
        screens_on=written,
        screen_zones=lambda: [_tv_zone()],
    )
    for i in range(len(frames)):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    pipeline.gap()  # ends the span, and writes it
    spans = screens.ScreensOn(path).cams["cam0"]
    assert len(spans) == 1
    start, end = spans[0]
    # On from its third change: samples land on frames, so each is up to a frame (40 ms) late.
    assert start == pytest.approx((T + 3 * screens.SAMPLE_NS) / 1e9, abs=3 * STEP / 1e9)
    assert end == pytest.approx((T + 149 * STEP) / 1e9, abs=screens.SAMPLE_NS / 1e9)


def test_a_still_tv_is_never_on():
    from liberational.see import screens

    written = screens.ScreensOn(Path("/nonexistent/never-written.json"), clock=lambda: 0.0)
    frames = [shown(nobody())] * 150
    pipeline = actions.ActionPipeline(Script(frames), Rows(), Rows(), "test-model", camera="cam0", screens_on=written)
    for i in range(len(frames)):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    pipeline.gap()
    assert written.cams == {}


def test_a_tv_the_detector_misses_for_a_few_frames_releases_nobody():
    # Someone on the TV, and three frames where the detector found no TV: still on it.
    frames = [shown(person(0.1))] * 20 + [person(0.1)] * 3 + [shown(person(0.1))] * 20
    poses, _, pipeline = run_with(frames)
    assert poses.appends == []
    assert pipeline.screens.counts.released == 0


def test_a_released_track_ends_into_movement_with_only_its_path_after_it_left_the_screen():
    paths: list[list[float]] = []

    class Paths(Ended):
        def ended(self, track) -> None:
            paths.append([p[0] for p in track.points])

    movement = Paths()
    gone = int(actions.LOST_S * 1e9 / STEP) + 10
    frames = [shown(person(0.1 + i * 0.0045)) for i in range(70)] + [nobody()] * gone
    pipeline = actions.ActionPipeline(Script(frames), Rows(), Rows(), "test-model", movement=cast(Any, movement))
    for i in range(len(frames)):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    assert len(paths) == 1
    assert min(paths[0]) >= (T + 23 * STEP) / 1e9 - 1e-6, "nothing from before it left the screen"


class HeartsRecorder:
    """A heart-rate hook that keeps what the pipeline asked it to do."""

    def __init__(self) -> None:
        self.fed: list[tuple[int, int]] = []  # (capture_ns, tracks seen)
        self.reads: list[int] = []
        self.ended: list[int] = []
        self.gaps = 0
        self.flushes = 0

    def feed(self, capture_ns: int, _width: int, _height: int, people: Any, _crop: Any) -> None:
        self.fed.append((capture_ns, len(people)))

    def due(self, capture_ns: int) -> None:
        self.reads.append(capture_ns)

    def end(self, track_id: int) -> None:
        self.ended.append(track_id)

    def gap(self) -> None:
        self.gaps += 1

    def flush(self) -> None:
        self.flushes += 1


def test_the_heart_rate_hook_sees_every_frame_of_a_counted_track_and_the_flushes_between():
    hearts = HeartsRecorder()
    frames = [person(0.1)] * 45 + [nobody()] * 20
    pipeline = actions.ActionPipeline(
        Script(frames), Rows(), Rows(), "test-model", hearts=cast(Any, hearts), camera="cam0"
    )
    for i in range(len(frames)):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    pipeline.flush()
    assert len(hearts.fed) == len(hearts.reads)
    # A track counts from its fifth frame, so frames 4 to 44 of the 45 with a person are fed; the frames with nobody
    # in them feed nothing, and a track nobody is in is not ended by a short empty run.
    assert len(hearts.fed) == 41
    assert hearts.fed[0][0] == T + 4 * STEP
    assert {tracks for _ns, tracks in hearts.fed} == {1}
    assert hearts.ended == [1] or hearts.ended == []
    assert hearts.gaps == 0, "a short run of empty frames is not a gap"
    assert hearts.flushes >= 2


class Blind(Script):
    """A backend whose crop fails, as a decoder's bad frame does: the face is there, its pixels can't be read."""

    def crop(self, image: str, box: tuple[int, int, int, int]) -> np.ndarray:
        self.cropped.append((image, box))
        raise RuntimeError("this frame's pixels can't be read")


class Streams:
    """A `segment_log.Writers` that keeps what was appended to each stream."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = {}

    def append(self, stream: str, cam: str, records: list[dict[str, Any]]) -> None:
        self.rows.setdefault(stream, []).extend(records)


def face_person(x: float = 0.1) -> People:
    """One person whose pose places the nose and both eyes, so see/hearts.py has a face to sample."""
    joints = np.zeros((1, actions.JOINTS, 3), np.float32)
    # Six confident joints (MIN_JOINTS starts a track): nose, both eyes, both ears, and a shoulder, in the 200 x 100
    # frame; the pupils 20 px apart, so the face patches are small but real.
    for i, (jx, jy) in enumerate(
        ((100.0, 30.0), (90.0, 25.0), (110.0, 25.0), (80.0, 27.0), (120.0, 27.0), (100.0, 60.0))
    ):
        joints[0, i] = (jx, jy, 0.9)
    return People(np.array([[x, 0.2, x + 0.2, 0.9, 0.9]], np.float32), joints)


class Logs:
    """The module's log, kept: the log itself doesn't propagate to pytest's capture (logs.py turns that off)."""

    def __init__(self) -> None:
        self.errors: list[tuple[tuple, dict]] = []

    def exception(self, *args: Any, **kwargs: Any) -> None:
        self.errors.append((args, kwargs))


def test_a_failing_heart_rate_does_not_take_the_camera_s_vision_down(monkeypatch):
    """A heart rate is an extra: an error in it costs the camera its heart rate, never its people, skeleton, or
    coverage."""
    logs = Logs()
    monkeypatch.setattr(hearts_module, "LOG", logs)
    into = Streams()
    pulses = hearts_module.Hearts("cam0", 30.0, into)
    poses, coverage = Rows(), Rows()
    frames = [face_person(0.1)] * 45
    pipeline = actions.ActionPipeline(Blind(frames), poses, coverage, "test-model", hearts=pulses, camera="cam0")
    for i in range(len(frames)):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    pipeline.flush()
    assert pulses.off == "an error while sampling a face"
    assert len(logs.errors) == 1, "said once, with the traceback"
    assert poses.appends, "the camera's pose rows still come"
    assert coverage_rows(coverage) == [(T, T + 25 * STEP, 26), (T + 25 * STEP, T + 44 * STEP, 19)]
    assert into.rows == {}, "and the heart rate wrote nothing"
