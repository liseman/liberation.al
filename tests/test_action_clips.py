"""MoViNet's clips beside ST-GCN++ in see-actions (see/action_clips.py): crops at 5 fps, scored every 2 s, the
skips, and spans."""

from typing import Any

import numpy as np
import pytest
import torch

from liberational.see import action_clips, actions, gpu_pose
from test_see_actions import Decoded, Rows, Scored, frame_id, person

T = 500_000 * 3_600_000_000_000
FPS30 = 33_333_333
FIFTH = 200_000_000
READ_BOOK = 405  # Kinetics-600


class Scorer:
    """MoViNet's place: each call's clips get `read_at[call]` on 'reading book', and the clips' shapes are kept."""

    def __init__(self, read_at: list[float]) -> None:
        self.read_at, self.shapes = read_at, []

    def __call__(self, clips: np.ndarray) -> np.ndarray:
        out = np.full((len(clips), 600), 0.0001, np.float32)
        out[:, READ_BOOK] = self.read_at[min(len(self.shapes), len(self.read_at) - 1)]
        self.shapes.append(clips.shape)
        return out


def crop(image: Any, box: np.ndarray) -> np.ndarray:
    return np.zeros((2, 2, 3), np.uint8)


BOX = np.array([0.4, 0.2, 0.6, 0.6, 0.9], np.float32)
EDGE_BOX = np.array([0.0, 0.2, 0.2, 0.6, 0.9], np.float32)


def feed(clips: action_clips.Clips, times: list[int], box: np.ndarray = BOX, sure: int = 17) -> None:
    for ts in times:
        clips.add(1, ts, None, box, sure)
        clips.due(ts)


def test_crops_at_five_fps_from_thirty_fill_a_clip_scored_every_two_seconds():
    score = Scorer([0.1])
    clips = action_clips.Clips(crop, score)
    feed(clips, [T + i * FPS30 for i in range(360)])
    # A crop at frame 0, then one due every 200 ms, taken up to 50 ms early: frames 5, 11, 17, and on (6k - 1). The
    # 50th is frame 293, 9.77 s in; the next score is 2 s later, at frame 353, whose clip starts at frame 59.
    windows = clips.take_windows()
    assert windows is not None
    assert [int(e) for e in windows["end_ns"]] == [T + 293 * FPS30, T + 353 * FPS30]
    assert [int(s) for s in windows["ts_ns"]] == [T, T + 59 * FPS30]
    assert windows["model"] == [action_clips.MODEL] * 2
    assert int(windows["classes"][0, 0]) == READ_BOOK
    assert score.shapes == [(1, 50, 2, 2, 3)] * 2


@pytest.mark.parametrize(
    ("read_at", "spans"), [pytest.param([0.6, 0.6], 1, id="two-clips"), pytest.param([0.6, 0.1], 0, id="one-clip")]
)
def test_a_kind_is_a_span_only_when_two_clips_see_it(read_at, spans):
    clips = action_clips.Clips(crop, Scorer(read_at))
    feed(clips, [T + i * FIFTH for i in range(60)])  # scored at crops 49 and 59
    clips.gap()
    closed = clips.take_spans()
    assert [(s["kind"], s["attributes"]["observations"]) for s in closed] == [("read", 2)] * spans
    assert all("'reading book'" in s["uncertainty"]["interpretation"] for s in closed)


def test_clips_score_the_owners_activities_and_no_longer_a_deleted_one():
    # "read" was deleted and "study" added, read from the same Kinetics class
    clips = action_clips.Clips(crop, Scorer([0.6, 0.6]), kinds=lambda: {"study": (READ_BOOK,)})
    feed(clips, [T + i * FIFTH for i in range(60)])
    clips.gap()
    assert [s["kind"] for s in clips.take_spans()] == ["study"]


def test_a_person_still_for_minutes_keeps_being_scored():
    score = Scorer([0.9])
    clips = action_clips.Clips(crop, score)
    feed(clips, [T + i * FIFTH for i in range(1200)])  # 4 minutes in one place
    assert len(score.shapes) == 116  # crops 49, 59, and on to 1199


@pytest.mark.parametrize(("sure", "calls"), [pytest.param(6, 1, id="six-joints"), pytest.param(5, 0, id="five")])
def test_a_clip_whose_pose_shows_too_few_joints_is_not_scored(sure, calls):
    score = Scorer([0.9])
    feed(action_clips.Clips(crop, score), [T + i * FIFTH for i in range(50)], sure=sure)
    assert len(score.shapes) == calls


def test_a_box_at_the_frames_edge_stays_in_the_clip():
    score = Scorer([0.9])
    clips = action_clips.Clips(crop, score)
    feed(clips, [T + i * FIFTH for i in range(49)])
    feed(clips, [T + 49 * FIFTH], box=EDGE_BOX)
    assert len(score.shapes) == 1


def test_a_clip_restarted_by_a_gap_keeps_one_span():
    clips = action_clips.Clips(crop, Scorer([0.6]))
    feed(clips, [T + i * FIFTH for i in range(60)])  # scored at crops 49 and 59 (11.8 s)
    # 1.3 s unseen restarts the clip: it fills again, and is scored, 9.8 s after it comes back (22.9 s in)
    feed(clips, [T + 1_300_000_000 + i * FIFTH for i in range(60, 110)])
    clips.gap()
    assert [(s["kind"], s["attributes"]["observations"]) for s in clips.take_spans()] == [("read", 3)]


def test_a_track_unseen_for_over_a_second_starts_its_clip_over():
    score = Scorer([0.9])
    clips = action_clips.Clips(crop, score)
    feed(clips, [T + i * FIFTH for i in range(40)])
    feed(clips, [T + 1_300_000_000 + i * FIFTH for i in range(40, 89)])
    assert score.shapes == []
    feed(clips, [T + 1_300_000_000 + 89 * FIFTH])
    assert len(score.shapes) == 1


def test_the_pipeline_writes_movinet_windows_and_spans_beside_the_skeleton_models():
    windows, spans = Rows(), []
    score = Scorer([0.9])
    pipeline = actions.ActionPipeline(
        Scored([person(0.1) for _ in range(400)], [{"fall": 0.1}]),
        Rows(),
        Rows(),
        "m",
        windows=windows,
        spans=spans.extend,
        clips=action_clips.Clips(crop, score),
    )
    for i in range(400):  # 16 s at 25 fps
        pipeline.feed(Decoded(frame_id(i), T + i * 40_000_000))
    pipeline.close()
    assert action_clips.MODEL in {m for a in windows.appends for m in a["model"]}
    assert [s["kind"] for s in spans if "Kinetics" in s["uncertainty"]["interpretation"]] == ["read"]


def test_a_crop_is_a_square_around_the_box_black_past_the_edge_and_rgb():
    backend = gpu_pose.TorchBackend.__new__(gpu_pose.TorchBackend)  # crop needs no models
    image = torch.zeros((100, 200, 3), dtype=torch.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 10, 20, 30  # BGR
    inside = backend.clip(image, np.array([0.4, 0.4, 0.6, 0.6], np.float32))
    assert inside.shape == (224, 224, 3)
    assert (inside == np.array([30, 20, 10], np.uint8)).all()
    # A 40 x 40 box at the top left corner: a 50-pixel square from -5, -5, so its first tenth is past the frame.
    corner = backend.clip(image, np.array([0.0, 0.0, 0.2, 0.4], np.float32))
    assert (corner[0, 0] == 0).all() and (corner[-1, -1] == np.array([30, 20, 10], np.uint8)).all()
