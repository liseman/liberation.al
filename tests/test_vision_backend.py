"""see-actions' backend choice (src/liberational/see/vision_backend.py): CUDA, then an Apple Silicon Mac, and
the CPU only by name."""

import pathlib
from typing import cast

import numpy as np
import pytest
import torch

from liberational import config as cfg
from liberational.see import gpu_pose, vision_backend

pytest.importorskip("onnxruntime")


class Made:
    def __init__(self, kind: str, *args: object, **kwargs: object) -> None:
        self.kind, self.args, self.kwargs = kind, args, kwargs


@pytest.fixture
def machine(monkeypatch):
    """A machine with what a test says it has, and the backends it'd build recorded instead of built."""
    has = {"cuda": False, "mps": False}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: has["cuda"])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: has["mps"])
    for kind, cls in (("cuda", "CudaBackend"), ("apple", "AppleBackend"), ("cpu", "CpuBackend")):
        monkeypatch.setattr(gpu_pose, cls, lambda *a, kind=kind, **k: Made(kind, *a, **k))
    return has


@pytest.mark.parametrize(
    ("cuda", "mps", "asked", "built"),
    [
        pytest.param(True, False, "auto", "cuda", id="cuda-first"),
        pytest.param(False, True, "auto", "apple", id="a-mac"),
        pytest.param(True, True, "apple", "apple", id="apple-by-name"),
        pytest.param(True, False, "cpu", "cpu", id="cpu-by-name-even-with-a-gpu"),
        pytest.param(False, False, "cpu", "cpu", id="cpu-by-name"),
    ],
)
def test_the_backend_is_the_first_this_machine_has(machine, cuda, mps, asked, built):
    machine.update(cuda=cuda, mps=mps)
    made = vision_backend.load("models", backend=asked)
    assert isinstance(made, Made) and made.kind == built


def test_cuda_goes_on_the_last_device_unless_one_is_named(machine):
    machine["cuda"] = True
    assert cast(Made, vision_backend.load("models")).args[2] == torch.device("cuda", 1)
    assert cast(Made, vision_backend.load("models", 0)).args[2] == torch.device("cuda", 0)


@pytest.mark.parametrize(
    ("cuda", "mps", "asked"),
    [
        pytest.param(False, False, "auto", id="nothing-auto: never the cpu unasked"),
        pytest.param(False, True, "cuda", id="cuda-on-a-mac"),
        pytest.param(True, False, "apple", id="apple-without-mps"),
    ],
)
def test_without_the_backend_asked_for_it_says_so(machine, cuda, mps, asked):
    machine.update(cuda=cuda, mps=mps)
    with pytest.raises(vision_backend.NoBackend, match="--backend cpu"):
        vision_backend.load("models", backend=asked)


def test_an_unknown_backend_is_refused(machine):
    with pytest.raises(ValueError, match="isn't one of"):
        vision_backend.load("models", backend="tpu")


MODELS = pathlib.Path(__file__).resolve().parents[1] / "data" / "models"


def test_the_model_files_are_the_ones_config_names(machine, tmp_path):
    # see-actions must open the files [models.files.*] names: editing pose_live.path as defaults.toml documents
    # used to leave the service opening the old filename and failing at camera load.
    machine["cuda"] = True
    for name in ("detector.onnx", "pose.onnx", "actions.pth", "stills.pth"):
        (tmp_path / name).write_bytes(b"")
    declared = {
        "person_detector": {"path": str(tmp_path / "detector.onnx")},
        "pose_live": {"path": str(tmp_path / "pose.onnx")},
        "action_stgcnpp": {"path": str(tmp_path / "actions.pth")},
        "action_movinet": {"path": str(tmp_path / "stills.pth")},
    }
    with cfg.override({"models": {"files": declared}}):
        made = cast(Made, vision_backend.load(tmp_path))
    assert made.args[:2] == (tmp_path / "detector.onnx", tmp_path / "pose.onnx")
    assert made.kwargs["action_weights"] == tmp_path / "actions.pth"
    assert made.kwargs["clip_weights"] == tmp_path / "stills.pth"
    (tmp_path / "stills.pth").unlink()
    with cfg.override({"models": {"files": declared}}):
        assert cast(Made, vision_backend.load(tmp_path)).kwargs["clip_weights"] is None, (
            "a clip model that isn't downloaded leaves the still kinds to ST-GCN++"
        )


def test_the_model_name_a_row_records_is_the_configured_detector_and_pose():
    declared = {
        "person_detector": {"path": "elsewhere/rfdetr-large.onnx"},
        "pose_live": {"path": "elsewhere/pose2.onnx"},
    }
    with cfg.override({"models": {"files": declared}}):
        assert vision_backend.model_name("cuda") == "rfdetr-large+pose2/cuda"
        assert vision_backend.detector_model() == "rfdetr-large"
    assert vision_backend.detector_model() == cfg.model_path("person_detector").stem


def models() -> pathlib.Path:
    for root in (MODELS,):
        detector = cfg.model_path("person_detector").name
        if all(
            (root / f).exists() for f in (detector, "rtmpose-m_halpe26_256x192.onnx", "stgcnpp_ntu120_xsub_hrnet_j.pth")
        ):
            return root
    pytest.skip("model weights are missing; run mise run models")


class Frame:
    id, capture_ns, gap_before = "f", 0, False
    source_header: dict = {}  # noqa: RUF012
    color_range, color_matrix = "limited", "bt601"
    pixels = np.full((720 * 3 // 2, 1280), 128, np.uint8)


@pytest.mark.parametrize(
    "asked",
    ["cpu", pytest.param("apple", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="a Mac"))],
)
def test_a_backend_runs_a_frame_end_to_end(asked):
    backend = vision_backend.load(models(), backend=asked)
    image = backend.upload(Frame())
    assert backend.size(image) == (720, 1280)
    people = backend.people(image)
    assert people.boxes.shape[1:] == (5,) and people.joints.shape[2:] == (3,)
    assert backend.crop(image, (10, 20, 50, 60)).shape == (40, 40, 3)
    assert backend.shift(image) is None and backend.shift(image) == (0.0, 0.0)
    if (models() / "stgcnpp_ntu120_xsub_hrnet_j.pth").exists():
        assert backend.score(np.zeros((1, 90, 17, 3), np.float32)).shape[0] == 1
    assert set(cast(gpu_pose.TorchBackend, backend).providers()) >= {"detector", "pose1"}
