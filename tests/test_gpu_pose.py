"""The GPU pose path computes what the CPU path does (vision/crossings.py and vision/pose_model.py)."""

import json
import pathlib
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from liberational import config as cfg
from liberational import model_runtime
from liberational.vision import crossings, pose_model

pytest.importorskip("onnxruntime")

from liberational.see import gpu_pose

MODELS = pathlib.Path(__file__).resolve().parents[1] / "data" / "models"
# Every device this machine's torch has: the CPU always, a CUDA GPU on the Linux machine, MPS on the Mac.
DEVICES = [
    torch.device("cpu"),
    *([torch.device("cuda", torch.cuda.device_count() - 1)] if torch.cuda.is_available() else []),
    *([torch.device("mps")] if torch.backends.mps.is_available() else []),
]


@pytest.fixture(params=DEVICES, ids=str)
def device(request: pytest.FixtureRequest) -> torch.device:
    return request.param


@pytest.fixture
def cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return torch.device("cuda", torch.cuda.device_count() - 1)


def models() -> pathlib.Path:
    """The installed models' folder, skipped when this checkout has none (a fresh worktree has none until
    mise run models)."""
    detector, pose = cfg.model_path("person_detector"), cfg.model_path("pose_live")
    if detector.exists() and pose.exists():
        return detector.parent
    pytest.skip("worktree model weights are missing; run mise run models")


def session_of(role: str, batch: int = 1) -> tuple[pathlib.Path, tuple[int, ...]]:
    """The configured model for `role` ("person_detector" or "pose_live") and the fixed input it declares, with
    `batch` on the first axis. Nothing here names a file or a resolution: a switched detector is a different file at
    a different size, and these tests are about the path, not about which checkpoint is installed today."""
    path = cfg.model_path(role)
    if not path.exists():
        pytest.skip(f"{path.name} is missing; run mise run models")
    height, width = gpu_pose.model_input(path)
    return path, (batch, 3, height, width)


def smooth_frame(h: int = 720, w: int = 1280, seed: int = 0) -> np.ndarray:
    """A camera-like BGR frame: low-frequency color, so interpolation differences stay at rounding size."""
    small = np.random.default_rng(seed).integers(0, 256, (h // 40, w // 40, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def test_nv12_converts_as_opencv_does(device):
    rng = np.random.default_rng(1)
    h, w = 720, 1280
    y = cv2.resize(rng.integers(16, 236, (h // 8, w // 8), dtype=np.uint8), (w, h))
    uv = cv2.resize(rng.integers(16, 241, (h // 16, w // 16, 2), dtype=np.uint8), (w // 2, h // 2))
    nv12 = np.concatenate([y, uv.reshape(h // 2, w)])
    want = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12).astype(int)
    got = gpu_pose.nv12_to_bgr(torch.from_numpy(nv12).to(device), h, w).cpu().numpy().astype(int)
    assert np.abs(got - want).max() <= 2


def nv12_of(bgr: np.ndarray, color_range: str, kr: float, kb: float) -> np.ndarray:
    """Encode BGR (H x W x 3, even sides, flat 2 x 2 blocks) as NV12 with the forward equations of the standard."""
    b, g, r = (bgr[..., i].astype(np.float64) for i in range(3))
    y = kr * r + (1 - kr - kb) * g + kb * b
    cb, cr = (b - y) / (2 * (1 - kb)), (r - y) / (2 * (1 - kr))
    if color_range == "full":
        cb, cr = cb + 128, cr + 128
    else:
        y, cb, cr = 16 + y * 219 / 255, 128 + cb * 224 / 255, 128 + cr * 224 / 255
    h, w = y.shape
    uv = np.stack((cb[::2, ::2], cr[::2, ::2]), 2).reshape(h // 2, w)
    return np.concatenate((y, uv)).round().clip(0, 255).astype(np.uint8)


@pytest.mark.parametrize("color_range", ["full", "limited"])
@pytest.mark.parametrize("color_matrix", ["bt601", "bt709", "bt2020nc"])
def test_nv12_decodes_with_the_signaled_range_and_matrix(color_range, color_matrix, device):
    kr, kb = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722), "bt2020nc": (0.2627, 0.0593)}[color_matrix]
    colors = np.array(
        [[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [40, 90, 200], [128, 128, 128]],
        np.uint8,
    )
    bgr = np.repeat(np.repeat(colors[None], 2, axis=0), 2, axis=1)  # each color a flat 2 x 2 block
    nv12 = nv12_of(bgr, color_range, kr, kb)
    got = gpu_pose.nv12_to_bgr(
        torch.from_numpy(nv12).to(device), 2, 14, color_range=color_range, color_matrix=color_matrix
    )
    assert np.abs(got.cpu().numpy().astype(int) - bgr.astype(int)).max() <= 2


def test_nv12_with_an_unspecified_range_and_matrix_decodes_as_ffmpeg_defaults(device):
    nv12 = nv12_of(np.full((2, 2, 3), [40, 90, 200], np.uint8), "limited", 0.299, 0.114)
    frame = torch.from_numpy(nv12).to(device)
    unspecified = gpu_pose.nv12_to_bgr(frame, 2, 2, color_range="unspecified", color_matrix="unspecified")
    assert torch.equal(unspecified, gpu_pose.nv12_to_bgr(frame, 2, 2, color_range="limited", color_matrix="bt601"))


def test_nv12_with_an_unknown_matrix_is_refused(device):
    with pytest.raises(ValueError, match="color matrix"):
        gpu_pose.nv12_to_bgr(torch.zeros((3, 2), dtype=torch.uint8, device=device), 2, 2, color_matrix="smpte240m")


def test_detector_input_changes_bgr_to_normalized_rgb_without_padding(device):
    frame = torch.tensor([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=torch.uint8, device=device)
    got = gpu_pose.detector_input(frame, 2).cpu().numpy()
    want = np.array(
        [
            [
                [[-2.11790393, -2.11790393], [2.24890830, 2.24890830]],
                [[-2.03571429, 2.42857143], [-2.03571429, 2.42857143]],
                [[2.64000000, -1.80444444], [-1.80444444, 2.64000000]],
            ]
        ],
        np.float32,
    )
    np.testing.assert_allclose(got, want, atol=5e-7)


def test_detector_resize_stretches_to_square_and_preserves_fractional_pixel_values(device):
    # A 3px row resized to 2 samples x=.25 and x=1.75: intensities32 and223.25, not rounded integers.
    frame = torch.tensor([[[0, 0, 0], [128, 128, 128], [255, 255, 255]]], dtype=torch.uint8, device=device)
    got = gpu_pose.detector_input(frame, 2).cpu().numpy()
    levels = np.array([32 / 255, 223.25 / 255], np.float32)
    want = np.array(
        [[(levels - 0.485) / 0.229] * 2, [(levels - 0.456) / 0.224] * 2, [(levels - 0.406) / 0.225] * 2], np.float32
    )[None]
    np.testing.assert_allclose(got, want, atol=5e-7)


def test_detector_input_matches_the_portable_float_resize(device):
    frame = smooth_frame()
    want = crossings.detector_input(frame, 512)
    got = gpu_pose.detector_input(torch.from_numpy(frame).to(device), 512)
    np.testing.assert_allclose(got.cpu().numpy(), want, atol=2e-5)


@pytest.mark.parametrize(
    "box",
    [
        pytest.param([500.0, 100.0, 700.0, 650.0], id="tall"),
        pytest.param([100.0, 300.0, 600.0, 450.0], id="wide"),
        pytest.param([1150.0, -40.0, 1350.0, 400.0], id="past-the-edge"),
    ],
)
def test_pose_crops_match_the_cpu_crop(box, device):
    frame = smooth_frame(seed=2)
    want, want_t = pose_model.crop(frame, box, (192, 256))
    got, t = gpu_pose.crops(torch.from_numpy(frame).to(device), torch.tensor([box], device=device), (192, 256))
    assert t.cpu().numpy()[0] == pytest.approx(np.array(want_t), rel=1e-5)
    levels = np.abs(got.cpu().numpy()[0] - want) * pose_model.STD[:, None, None]
    assert levels.max() <= 3
    assert levels.mean() < 0.5


@pytest.mark.parametrize(
    "neighbors, expected",
    [
        ([], []),
        ([[20, 20, 40, 40]], []),  # self alone cannot change a protected target
        ([[44, 20, 64, 40]], []),  # outside even the padded crop plus interpolation margin
        ([[41, 20, 61, 40]], [0]),  # misses the target box but intersects its padded crop
        ([[18, 45, 22, 49]], [0]),  # the 3:4 aspect extends the crop below a square target
    ],
)
def test_only_targets_with_a_neighbor_in_the_padded_crop_need_masking(neighbors, expected):
    boxes = np.array([[20, 20, 40, 40], [0, 0, 5, 5]], np.float32)
    assert gpu_pose.neighbor_targets(boxes, np.array(neighbors, np.float32).reshape(-1, 4), (24, 32)) == expected


@pytest.mark.parametrize("neighbor_score", [0.8, 0.5])
def test_people_mask_strong_neighbors_across_batches_but_preserve_overlap_and_weak_boxes(
    monkeypatch, device, neighbor_score
):
    detector, pose = pathlib.Path("person.onnx"), pathlib.Path("pose.onnx")
    # Normalized RF-DETR cxcywh, from target/strong-neighbor/weak-neighbor visible boxes.
    xyxy = torch.tensor(
        [
            [20, 20, 40, 40],
            [36, 16, 56, 36],
            [16, 36, 36, 56],
            [1, 58, 4, 61],
            [5, 58, 8, 61],
            [9, 58, 12, 61],
            [13, 58, 16, 61],
            [17, 58, 20, 61],
            [54, 58, 57, 61],
            [58, 58, 61, 61],
        ],
        device=device,
        dtype=torch.float32,
    )
    boxes = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), 1)[None] / 64
    logits = torch.full((1, 10, 91), -30.0, device=device)
    logits[0, :, 1] = torch.logit(
        torch.tensor([0.9, neighbor_score, 0.4, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85], device=device)
    )
    inputs = []

    class Session:
        def __init__(self, path, shape):
            self.path = path
            self.input = torch.zeros(tuple(shape), device=device)
            self.output_names = ["labels", "dets"] if path == detector else ["simcc_x", "simcc_y"]

        def run(self, blob=None):
            if self.path == detector:
                return [logits, boxes]
            inputs.append(self.input.clone())
            batch = self.input.shape[0]
            return [torch.ones((batch, 26, 48), device=device), torch.ones((batch, 26, 64), device=device)]

    monkeypatch.setattr(gpu_pose, "model_input", lambda path: (64, 64) if path == detector else (32, 24))
    monkeypatch.setattr(gpu_pose, "session_for", lambda path, device, shape: Session(path, shape))
    model = gpu_pose.GpuPose(detector, pose, device, score=0.2)
    frame = torch.tensor([10, 20, 30], dtype=torch.uint8, device=device).expand(64, 64, 3)
    found = model.people(frame)
    assert len(found.boxes) == 10
    target = inputs[0][0].cpu().numpy() * pose_model.STD[:, None, None] + pose_model.MEAN[:, None, None]
    # Source (41.46,21.67) is in the neighbor beyond the target's one-pixel interpolation margin.
    np.testing.assert_allclose(target[:, 8, 23], [0, 0, 0], atol=1e-5)
    # Source x=40.42 is outside the target but still protected by the interpolation margin.
    np.testing.assert_allclose(target[:, 8, 22], [30, 20, 10], atol=1e-5)
    # Source (38.33,21.67) overlaps the target; (20.63,45.63) belongs only to the weak neighbor.
    np.testing.assert_allclose(target[:, 8, 20], [30, 20, 10], atol=1e-5)
    np.testing.assert_allclose(target[:, 31, 3], [30, 20, 10], atol=1e-5)


@pytest.mark.parametrize(
    "sparse, internal",
    [
        (1, 0),
        (16, 14),
        (17, 15),
        (18, 16),
        (19, 17),
        (20, 18),
        (21, 19),
        (22, 20),
        (23, 21),
        (24, 22),
        (25, 23),
        (72, 62),  # tv
        (73, 63),  # laptop
    ],
)
def test_rfdetr_sparse_categories_map_to_existing_person_and_animal_ids(sparse, internal, device):
    boxes = torch.tensor([[0.1, 0.9, 0.4, 0.4]], device=device)
    logits = torch.full((1, 91), -30.0, device=device)
    logits[0, sparse] = 0  # independent sigmoid gives .5, not class softmax
    got = gpu_pose.rfdetr_candidates(boxes, logits).cpu().numpy()
    assert got.shape == (91, 6)
    assert got[0] == pytest.approx([0, 0.7, 0.3, 1, 0.5, internal])


def test_rfdetr_ranks_all_query_class_pairs_before_filtering_and_keeps_slot90(device):
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]] * 4, device=device)
    logits = torch.full((4, 91), 2.0, device=device)
    logits[:, 1] = -2  # Person scores lose to360 nonperson pairs, even though each exceeds a low threshold.
    logits[3, 90] = 3  # Slot90 is a valid COCO category, not a removable background column.
    got = gpu_pose.rfdetr_candidates(boxes, logits).cpu().numpy()
    assert got.shape == (300, 6)
    assert got[0] == pytest.approx([0.4, 0.4, 0.6, 0.6, 0.95257413, -1])
    assert not np.any(got[:, 5] == 0)
    assert got[:, 4].min() == pytest.approx(0.88079708)


def test_rfdetr_scene_keeps_overlapping_people_and_animals_without_nms(device):
    boxes = torch.tensor([[0.4, 0.5, 0.6, 0.8], [0.45, 0.5, 0.6, 0.8]], device=device)
    logits = torch.full((2, 91), -30.0, device=device)
    logits[:, 1] = torch.tensor([2.1972246, 1.3862944], device=device)
    logits[:, 17] = torch.tensor([1.0986123, 0.8472979], device=device)
    cand = gpu_pose.rfdetr_candidates(boxes, logits)
    people, animals = gpu_pose.scene(cand.cpu().numpy(), 0.5)
    np.testing.assert_allclose(people, [[0.1, 0.1, 0.7, 0.9, 0.9], [0.15, 0.1, 0.75, 0.9, 0.8]], atol=1e-6)
    np.testing.assert_allclose(animals, [[0.1, 0.1, 0.7, 0.9, 0.75, 15], [0.15, 0.1, 0.75, 0.9, 0.7, 15]], atol=1e-6)


def test_screens_from_the_same_pass_are_kept_apart_from_people_and_animals(device):
    boxes = torch.tensor([[0.3, 0.5, 0.4, 0.6], [0.7, 0.5, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]], device=device)
    logits = torch.full((3, 91), -30.0, device=device)
    logits[0, 72] = 0.0  # a tv, .5
    logits[1, 73] = -1.3862944  # a laptop, .2: under SCREEN_SCORE
    logits[2, 17] = 0.0  # a cat, .5
    cand = gpu_pose.rfdetr_candidates(boxes, logits).cpu().numpy()
    shown = gpu_pose.screens(cand)
    np.testing.assert_allclose(shown, [[0.1, 0.2, 0.5, 0.8, 0.5, 62]], atol=1e-6)
    people, animals = gpu_pose.scene(cand, 0.1)
    assert people.shape == (0, 5)
    assert animals[:, 5].tolist() == [15], "a screen is never an animal"


@pytest.mark.parametrize("tight", [False, True], ids=["square-for-the-check", "tight-for-the-embedder"])
@pytest.mark.parametrize("box", [(0.4, 0.4, 0.5, 0.6, 0.9), (0.0, 0.0, 0.05, 0.05, 0.8), (0.9, 0.7, 1.0, 1.0, 0.7)])
def test_an_animal_crop_is_the_cpu_paths_crop(box, tight, device):
    from liberational.vision import animal_verifier

    frame = smooth_frame()
    (got,) = gpu_pose.animal_crops(torch.from_numpy(frame).to(device), np.array([box], np.float32), tight=tight)
    np.testing.assert_array_equal(got, (animal_verifier.tight if tight else animal_verifier.square)(frame, box))


def test_simcc_decodes_as_the_cpu_decode_does(device):
    rng = np.random.default_rng(6)
    simcc_x = rng.normal(0, 1, (3, 26, 384)).astype(np.float32)
    simcc_y = rng.normal(0, 1, (3, 26, 512)).astype(np.float32)
    simcc_x[1, 4] = -1.0  # a joint with no positive peak
    want_locs, want_conf = pose_model.decode(simcc_x, simcc_y)
    locs, conf = gpu_pose.decode(torch.from_numpy(simcc_x).to(device), torch.from_numpy(simcc_y).to(device))
    np.testing.assert_allclose(locs.cpu().numpy(), want_locs)
    np.testing.assert_allclose(conf.cpu().numpy(), want_conf)
    t = (640.0, 360.0, 0.4)
    np.testing.assert_allclose(
        gpu_pose.to_frame(locs, torch.tensor([t] * 3, device=device), (192, 256)).cpu().numpy(),
        pose_model.to_frame(want_locs, t, (192, 256)),
        rtol=1e-5,
    )


@pytest.mark.parametrize(("role", "batch"), [("person_detector", 1), ("pose_live", gpu_pose.POSE_BATCHES[-1])])
def test_a_graph_replay_runs_each_new_input_as_a_plain_session_does(role, batch, cuda):
    import onnxruntime as ort

    path, shape = session_of(role, batch)
    session = gpu_pose.GraphSession(path, cuda, shape)
    plain = ort.InferenceSession(str(path), providers=[("CUDAExecutionProvider", {"device_id": cuda.index})])
    rng = np.random.default_rng(7)
    for _ in range(2):  # a replay of a stale input would match only the first, or neither
        x = rng.normal(0, 1, shape).astype(np.float32)
        staged = torch.from_numpy(x).to(cuda)
        with torch.cuda.device(cuda):
            torch.cuda._sleep(200_000_000)  # keep torch's stream busy, so the input copy lands late
        got = [o.cpu().numpy() for o in session.run(staged)]
        want = plain.run(None, {plain.get_inputs()[0].name: x})
        for g, w in zip(got, want, strict=True):
            np.testing.assert_allclose(g, np.asarray(w), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(("role", "batch"), [("person_detector", 1), ("pose_live", gpu_pose.POSE_BATCHES[-1])])
def test_on_its_own_stream_a_replay_is_queued_behind_torchs_work_without_a_host_wait(role, batch, cuda):
    import onnxruntime as ort

    path, shape = session_of(role, batch)
    plain = ort.InferenceSession(str(path), providers=[("CUDAExecutionProvider", {"device_id": cuda.index})])
    stream = torch.cuda.Stream(cuda)
    rng = np.random.default_rng(9)
    with torch.cuda.stream(stream):
        session = gpu_pose.GraphSession(path, cuda, shape)
        for _ in range(2):  # a replay of a stale input would match only the first, or neither
            x = rng.normal(0, 1, shape).astype(np.float32)
            torch.cuda._sleep(200_000_000)  # torch's stream stays busy, so the input copy lands late
            outputs = session.run(torch.from_numpy(x).pin_memory().to(cuda, non_blocking=True))
            assert not stream.query()  # run returned while the stream still had the sleep, the copy, and the replay
            got = [o.cpu().numpy() for o in outputs]
            want = plain.run(None, {plain.get_inputs()[0].name: x})
            for g, w in zip(got, want, strict=True):
                np.testing.assert_allclose(g, np.asarray(w), atol=1e-4, rtol=1e-4)


def smooth_nv12(seed: int, h: int = 720, w: int = 1280) -> np.ndarray:
    """A camera-like NV12 frame: smooth luma, then smooth interleaved chroma at half resolution."""
    rng = np.random.default_rng(seed)
    y = cv2.resize(rng.integers(16, 236, (h // 40, w // 40), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    uv = cv2.resize(
        rng.integers(64, 192, (h // 80, w // 40), dtype=np.uint8), (w, h // 2), interpolation=cv2.INTER_CUBIC
    )
    return np.vstack((y, uv))


@pytest.mark.parametrize("flip", [False, True])
def test_a_cameras_frames_replayed_on_its_stream_find_exactly_what_eager_work_finds(flip, cuda):
    detector, pose = session_of("person_detector")[0], session_of("pose_live")[0]
    frames = [
        Decoded(smooth_nv12(seed), {"capture": {"flip": flip}}, color_range="limited", color_matrix="bt709")
        for seed in (1, 2)
    ]
    boxes = np.array([[100, 100, 400, 600], [300, 120, 600, 650], [900, 50, 1200, 700]], np.float32)

    def run(backend: gpu_pose.TorchBackend) -> list[list[np.ndarray]]:
        out = []
        for frame in frames:
            image = backend.upload(frame)
            found = backend.people(image)
            joints = backend.pose.skeletons(image, boxes, neighbors=boxes[:2])
            # The image is good until the next upload, so it is read back now.
            out.append([image.cpu().numpy(), *(o.cpu().numpy() for o in backend.pose.detector.outputs), joints])
            assert found.joints.shape[0] == len(found.boxes)
        return out

    eager = run(gpu_pose.TorchBackend(detector, pose, cuda))  # on the default stream: nothing is replayed
    with torch.cuda.stream(torch.cuda.Stream(cuda)):
        backend = gpu_pose.TorchBackend(detector, pose, cuda)
        assert backend.captures(frames[0])
        replayed = run(backend)
        assert not backend.captures(frames[1])
    for got, want in zip(replayed, eager, strict=True):
        for g, w in zip(got, want, strict=True):
            np.testing.assert_array_equal(g, w)


def test_a_frame_with_people_waits_on_the_host_only_for_its_candidates_and_its_joints(monkeypatch, cuda):
    detector, pose = pathlib.Path("person.onnx"), pathlib.Path("pose.onnx")
    # Ten overlapping people (two pose batches, neighbors masked across them), normalized cxcywh.
    xyxy = torch.tensor([[i * 5, 10, i * 5 + 12, 40] for i in range(10)], dtype=torch.float32, device=cuda)
    boxes = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), 1)[None] / 64
    logits = torch.full((1, 10, 91), -30.0, device=cuda)
    logits[0, :, 1] = 3.0
    simcc = (torch.rand((8, 26, 48), device=cuda), torch.rand((8, 26, 64), device=cuda))

    class Session:
        def __init__(self, path, shape):
            self.path = path
            self.input = torch.zeros(tuple(shape), device=cuda)
            self.output_names = ["labels", "dets"] if path == detector else ["simcc_x", "simcc_y"]

        def run(self, blob=None):
            if self.path == detector:
                return [logits, boxes]
            batch = self.input.shape[0]
            return [simcc[0][:batch], simcc[1][:batch]]

    monkeypatch.setattr(gpu_pose, "model_input", lambda path: (64, 64) if path == detector else (32, 24))
    monkeypatch.setattr(gpu_pose, "session_for", lambda path, device, shape: Session(path, shape))
    waits = []
    real_wait = gpu_pose.wait
    monkeypatch.setattr(gpu_pose, "wait", lambda device: (waits.append(device), real_wait(device)))
    model = gpu_pose.GpuPose(detector, pose, cuda, score=0.2)
    frame = torch.full((64, 64, 3), 100, dtype=torch.uint8, device=cuda)
    model.people(frame)  # the first frame allocates the pinned buffers
    waits.clear()
    torch.cuda.set_sync_debug_mode("error")  # any other wait (a small upload, a list index, .item()) raises
    try:
        found = model.people(frame)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert found.joints.shape == (10, 26, 3)
    assert waits == [cuda, cuda]


@dataclass(frozen=True)
class Decoded:
    pixels: np.ndarray
    source_header: dict[str, Any]
    color_range: str = "full"
    color_matrix: str = "bt601"
    id: str = "frame"
    capture_ns: int = 0
    gap_before: bool = False


@pytest.mark.parametrize(("flip", "corner"), [(False, 10), (True, 160)])
def test_a_decoded_frame_comes_out_upright(flip, corner, device):
    y = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]], np.uint8)
    nv12 = np.concatenate([y, np.full((2, 4), 128, np.uint8)])  # no chroma: gray, so B = G = R = Y at full range
    bgr = gpu_pose.upright_bgr(Decoded(nv12, {"capture": {"flip": flip}}), device)
    assert bgr.shape == (4, 4, 3)
    assert bgr[0, 0].tolist() == [corner] * 3


def test_a_staged_upload_is_exact_and_the_next_frame_never_overwrites_it(cuda):
    staging = gpu_pose.Staging()
    rng = np.random.default_rng(8)
    first = rng.integers(0, 256, (1080, 1280), dtype=np.uint8)
    second = rng.integers(0, 256, (1080, 1280), dtype=np.uint8)
    a = staging.upload(first, cuda)
    b = staging.upload(second, cuda)  # reuses the same pinned buffer, after the first copy is done
    np.testing.assert_array_equal(a.cpu().numpy(), first)
    np.testing.assert_array_equal(b.cpu().numpy(), second)


def test_people_go_down_to_the_low_pass_threshold_but_animals_keep_the_detectors(device):
    boxes = torch.tensor([[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]], device=device)
    logits = torch.full((2, 91), -30.0, device=device)
    logits[0, 1] = -1.3862944  # person .2
    logits[1, 17] = -1.3862944  # cat .2
    cand = gpu_pose.rfdetr_candidates(boxes, logits)
    people, animals = gpu_pose.scene(cand.cpu().numpy(), 0.1)
    assert people.shape == (1, 5)
    assert people[0] == pytest.approx([0.15, 0.15, 0.35, 0.35, 0.2])
    assert animals.shape == (0, 6)


SESSIONS = [("person_detector", 1), ("pose_live", 2)]


@pytest.mark.parametrize(("role", "batch"), SESSIONS)
def test_a_models_session_on_each_device_gives_the_cpu_runs_outputs(role, batch, device):
    import onnxruntime as ort

    if device.type == "cuda":
        pytest.skip("a CUDA graph is checked against a plain CUDA session (the graph replay test)")
    path, shape = session_of(role, batch)
    session = gpu_pose.session_for(path, device, shape)
    plain = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(7).normal(size=shape).astype(np.float32)
    want = plain.run(None, {plain.get_inputs()[0].name: x})
    got = [o.cpu().numpy() for o in session.run(torch.from_numpy(x).to(device))]
    for g, w in zip(got, (np.asarray(w) for w in want), strict=True):
        # Core ML runs in half precision on the Neural Engine or the GPU; the CPU keeps float32.
        scale = 2e-2 if getattr(session, "provider", "") == gpu_pose.COREML else 1e-3
        np.testing.assert_allclose(g, w, atol=scale * float(np.abs(w).max()))


def test_a_model_runs_where_its_device_says(device):
    path, shape = session_of("person_detector")
    session = gpu_pose.session_for(path, device, shape)
    want = {"cpu": "CPUExecutionProvider", "mps": gpu_pose.COREML, "cuda": "CUDAExecutionProvider"}[device.type]
    assert getattr(session, "provider", "CUDAExecutionProvider") == want


def test_without_core_mls_provider_a_mac_session_runs_on_the_cpu_and_says_so(monkeypatch, caplog):
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    path, shape = session_of("person_detector")
    session = gpu_pose.HostSession(path, torch.device("cpu"), shape, coreml=True)
    assert session.provider == "CPUExecutionProvider"
    assert "no Core ML provider" in caplog.text


@pytest.mark.parametrize(
    ("person_score", "expected_people"),
    [
        (0.59, [[10 / 64, 10 / 64, 50 / 64, 50 / 64, 0.9]]),
        (0.60, [[10 / 64, 10 / 64, 50 / 64, 50 / 64, 0.9], [20 / 64, 10 / 64, 60 / 64, 50 / 64, 0.6]]),
    ],
)
def test_named_detector_outputs_use_calibrated_person_floor_and_keep_overlapping_animals(
    monkeypatch, person_score, expected_people
):
    detector, pose = pathlib.Path("person.onnx"), pathlib.Path("pose.onnx")
    boxes = torch.tensor([[[0.46875, 0.46875, 0.625, 0.625], [0.625, 0.46875, 0.625, 0.625]]])
    logits = torch.full((1, 2, 91), -30.0)
    logits[0, :, 1] = torch.logit(torch.tensor([0.9, person_score]))
    logits[0, :, 17] = torch.tensor([2.1972246, 1.3862944])

    class Session:
        def __init__(self, path, shape):
            self.path = path
            self.input = torch.zeros(tuple(shape))
            self.output_names = ["labels", "dets"] if path == detector else ["simcc_x", "simcc_y"]

        def run(self, blob=None):
            if self.path == detector:
                return [logits, boxes]
            batch = self.input.shape[0]
            return [torch.ones((batch, 26, 24)), torch.ones((batch, 26, 32))]

    monkeypatch.setattr(gpu_pose, "model_input", lambda path: (64, 64) if path == detector else (16, 12))
    monkeypatch.setattr(gpu_pose, "session_for", lambda path, device, shape: Session(path, shape))
    found = gpu_pose.GpuPose(detector, pose, torch.device("cpu")).people(torch.zeros((64, 64, 3), dtype=torch.uint8))
    np.testing.assert_allclose(found.boxes, expected_people, atol=1e-6)
    np.testing.assert_allclose(
        found.animals,
        [[10 / 64, 10 / 64, 50 / 64, 50 / 64, 0.9, 15], [20 / 64, 10 / 64, 60 / 64, 50 / 64, 0.8, 15]],
        atol=1e-6,
    )


@pytest.mark.parametrize("box_shape,logit_shape", [((2, 5), (2, 91)), ((2, 4), (2, 80)), ((2, 4), (1, 91))])
def test_detector_rejects_mismatched_query_or_class_shapes(box_shape, logit_shape):
    with pytest.raises(gpu_pose.DetectorUnavailable, match="91 sparse COCO"):
        gpu_pose.rfdetr_candidates(torch.zeros(box_shape), torch.zeros(logit_shape))


@pytest.mark.parametrize(
    ("names", "want"),
    [
        (["dets", "labels"], (0, 1)),  # Roboflow's own exporter
        (["pred_boxes", "pred_logits"], (0, 1)),  # the community conversions of the same checkpoints
        (["pred_logits", "pred_boxes"], (1, 0)),  # order is not assumed
    ],
)
def test_both_ways_an_rf_detr_export_names_its_raw_tensors_are_read(names, want):
    assert gpu_pose.rfdetr_outputs(names) == want


def test_a_detector_that_hands_back_decoded_outputs_is_refused():
    # A model that has already decoded its boxes and scores is a different contract from this path, and running it
    # would silently read one tensor as the other.
    with pytest.raises(gpu_pose.DetectorUnavailable, match="boxes output and a logits output"):
        gpu_pose.rfdetr_outputs(["boxes", "scores"])


def test_a_backend_reports_its_detector_and_pose_on_the_device_onnx_runtime_bound(tmp_path, monkeypatch):
    detector, pose = tmp_path / "detector.onnx", tmp_path / "pose.onnx"
    bound = SimpleNamespace(
        get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        get_provider_options=lambda: {"CUDAExecutionProvider": {"device_id": "1"}, "CPUExecutionProvider": {}},
    )

    class Session:
        def __init__(self, path, shape):
            self.session = bound
            self.input = torch.zeros(tuple(shape))
            self.output_names = ["dets", "labels"] if path == detector else ["simcc_x", "simcc_y"]

    monkeypatch.setattr(gpu_pose, "model_input", lambda path: (64, 64) if path == detector else (32, 24))
    monkeypatch.setattr(gpu_pose, "session_for", lambda path, device, shape: Session(path, shape))
    with cfg.override({"paths": {"state": str(tmp_path)}}), model_runtime.reporting("see-actions"):
        backend = gpu_pose.TorchBackend(detector, pose, torch.device("cpu"))
        (report,) = (tmp_path / "model_loads").glob("*.json")
        models = json.loads(report.read_text())["models"]
    assert backend is not None
    assert {(m["key"], m["name"], m["device"]) for m in models} == {
        ("models.files.person_detector", "detector.onnx", "GPU 1"),
        ("models.files.pose_live", "pose.onnx", "GPU 1"),
    }


class _Counted:
    """A model that counts how often it's built, and whose score can be held mid-call."""

    built = 0

    def __init__(self) -> None:
        type(self).built += 1
        self.inside = threading.Event()
        self.release = threading.Event()
        self.overlaps = 0
        self.active = 0

    def score(self, x: int) -> int:
        self.active += 1
        self.overlaps += self.active > 1
        self.inside.set()
        self.release.wait(5)
        self.active -= 1
        return x * 2


@pytest.fixture
def fresh_shared(monkeypatch):
    monkeypatch.setattr(gpu_pose, "_shared", {})
    _Counted.built = 0


def test_cameras_on_one_device_share_one_copy_of_a_model(fresh_shared):
    cpu = torch.device("cpu")
    first = gpu_pose.shared(cpu, "siglip2", _Counted)
    second = gpu_pose.shared(cpu, "siglip2", _Counted)
    assert first is second
    assert _Counted.built == 0  # nothing loads until it's used
    first._load().release.set()
    assert second.score(3) == 6
    assert _Counted.built == 1
    gpu_pose.shared(cpu, "dinov3", _Counted).loaded()  # another model, or another device, is its own copy
    gpu_pose.shared(torch.device("meta"), "siglip2", _Counted).loaded()
    assert _Counted.built == 3


def test_a_shared_model_takes_one_call_at_a_time(fresh_shared):
    model = gpu_pose.shared(torch.device("cpu"), "movinet", _Counted).loaded()
    inner = model._load()
    results = []
    first = threading.Thread(target=lambda: results.append(model.score(1)))
    first.start()
    assert inner.inside.wait(5)
    inner.inside.clear()
    second = threading.Thread(target=lambda: results.append(model.score(2)))
    second.start()
    assert not inner.inside.wait(0.2)  # the second camera waits for the first call to finish
    inner.release.set()
    first.join(5)
    second.join(5)
    assert sorted(results) == [2, 4]
    assert inner.overlaps == 0
