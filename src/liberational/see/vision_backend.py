"""The seam between see-actions and the hardware it runs on.

see-actions runs on the Linux machine while it's developed on cam0, and on the Mac in the end. So
everything that depends on the accelerator sits behind `Backend`, and only NumPy arrays and plain values cross it:

- `upload` takes a decoded frame (host NV12 with its signaled color range and matrix, and its capture header) and
  returns the backend's own image, upright. Callers only pass that image back; they never look inside it.
- `size` is that image's height and width in pixels.
- `people` finds the image's people and their skeletons, as `People` in NumPy.
- `score` runs the action model on windows of one person's joints (see/stgcn.py), class probabilities out.
- `shift` estimates how far the camera moved since the backend's last `shift` call: x and y as fractions of the
  frame, or None on the first call.
- `providers` says where each model runs (its ONNX Runtime provider), so a model Core ML declined shows.

With a fisheye lens's calibration (see/fisheye.py), a backend rectifies each person into a virtual pinhole
view before pose, and `People.view_joints` holds the joints in that view's coordinates.

Animals ride the same detector pass (see/action_animals.py): `People.animals` holds the frame's COCO animal boxes,
`embed_animals` gives each box's appearance embedding, and `verify_animals` says whether each box's crop shows its
species and not part of a person.

Backends (`load` picks one): `gpu_pose.CudaBackend` (ONNX Runtime CUDA graphs and torch on CUDA, the Linux machine),
`gpu_pose.AppleBackend` (the same ONNX models on ONNX Runtime's Core ML provider, and torch on MPS for the rest:
the Mac), and `gpu_pose.CpuBackend` (everything on the CPU, only when asked for by name).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class Frame(Protocol):
    """The part of the shared full-rate decode's `DecodedFrame` see-actions reads."""

    @property
    def id(self) -> str: ...

    @property
    def capture_ns(self) -> int: ...

    @property
    def gap_before(self) -> bool: ...  # frames are missing before this one

    @property
    def pixels(self) -> np.ndarray: ...  # host NV12, height * 3 / 2 rows of width bytes, in sensor orientation

    @property
    def source_header(self) -> Mapping[str, Any]: ...

    @property
    def color_range(self) -> str: ...

    @property
    def color_matrix(self) -> str: ...


@dataclass(frozen=True)
class People:
    """One frame's people: `boxes` N x 5 (x1, y1, x2, y2 normalized to the frame, score) and `joints` N x K x 3
    (x, y in frame pixels, confidence; a joint the model didn't place is -1, -1, 0), as the CPU path gives them."""

    boxes: NDArray[np.float32]
    joints: NDArray[np.float32]
    # N x 6: x1, y1, x2, y2 normalized to the frame, the score, and the COCO class id (vision/crossings.COCO_ANIMALS)
    animals: NDArray[np.float32] = field(default_factory=lambda: np.zeros((0, 6), np.float32))
    # On a fisheye camera: N x K x 3, each person's joints in their own virtual pinhole view (x, y in the
    # view's pixels, the pose input's size, and confidence; -1, -1, 0 where not placed), as perspective as the action
    # model's training skeletons were. None on a plain camera, whose `joints` already are.
    view_joints: NDArray[np.float32] | None = None
    view_size: tuple[int, int] | None = None  # the views' size in pixels, width by height, with `view_joints`
    # N x 6: the screens seen in the same inference, x1, y1, x2, y2 normalized to the frame, the score, and the
    # COCO class id (vision/crossings.COCO_SCREENS)
    screens: NDArray[np.float32] = field(default_factory=lambda: np.zeros((0, 6), np.float32))


class Backend[Image](Protocol):
    name: str  # what the coverage and pose rows record alongside the model, such as "cuda"

    def upload(self, frame: Frame, /) -> Image: ...  # good until the next upload

    def captures(self, frame: Frame, /) -> bool: ...  # this frame's work must run alone on the GPU (a graph capture)

    def size(self, image: Image, /) -> tuple[int, int]: ...

    def people(self, image: Image, /) -> People: ...

    def embed_animals(self, image: Image, boxes: np.ndarray, /) -> np.ndarray: ...  # N x 384 float32, unit length

    def verify_animals(self, image: Image, boxes: np.ndarray, species: list[str], /) -> list[bool]: ...

    def crop(self, image: Image, box: tuple[int, int, int, int], /) -> np.ndarray: ...  # BGR uint8, on the host

    def score(self, windows: np.ndarray, /) -> np.ndarray: ...

    def shift(self, image: Image, /) -> tuple[float, float] | None: ...


class NoBackend(RuntimeError):
    """This machine has no accelerator see-actions supports yet."""


BACKENDS = ("auto", "cuda", "apple", "cpu")


def load(models: Any, device: int | None = None, backend: str = "auto", lens: Any = None) -> Backend[Any]:
    """The backend for this machine, with the person detector, pose model, and action model from `models` (a
    directory holding the person detector ([models.files.person_detector]), rtmpose-m_halpe26_256x192.onnx, the ST-GCN++ weights, and MoViNet's if downloaded).

    `backend` "auto" picks CUDA when torch sees a CUDA device (`device`, else the last one), then Apple when it sees
    MPS (an Apple Silicon Mac: Core ML for the models, MPS for the rest). "cuda" and "apple" insist on theirs. "cpu"
    runs everything on the CPU and is only ever chosen by name: at 30 fps it would take the machine.

    `lens` is the camera's fisheye calibration (see/fisheye.py), or None for a plain camera: with one, each person is
    resampled into a virtual pinhole view before pose."""
    import pathlib

    import torch

    from liberational.see import gpu_pose

    root = pathlib.Path(models)
    from liberational import config as cfg

    files = (root / cfg.model_path("person_detector").name, root / "rtmpose-m_halpe26_256x192.onnx")
    weights = root / "stgcnpp_ntu120_xsub_hrnet_j.pth"
    # MoViNet for the still kinds (see/action_clips.py), when its weights are downloaded: without them, only ST-GCN++.
    clips = root / "movinet_a2_k600.pth"
    clip_weights = clips if clips.exists() else None
    kind = choose(backend)
    if kind == "cuda":
        index = torch.cuda.device_count() - 1 if device is None else device
        return gpu_pose.CudaBackend(
            *files, torch.device("cuda", index), action_weights=weights, clip_weights=clip_weights, lens=lens
        )
    if kind == "apple":
        return gpu_pose.AppleBackend(*files, action_weights=weights, clip_weights=clip_weights, lens=lens)
    return gpu_pose.CpuBackend(*files, action_weights=weights, clip_weights=clip_weights, lens=lens)


def choose(backend: str = "auto") -> str:
    """Which backend `load` builds for `backend` on this machine: "cuda", "apple", or "cpu" (see `load`). Raises
    NoBackend when the machine has none of what `backend` asks for, a GPU that has gone missing included."""
    import torch

    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r} isn't one of {', '.join(BACKENDS)}")
    if backend in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if backend in ("auto", "apple") and torch.backends.mps.is_available():
        return "apple"
    if backend == "cpu":
        return "cpu"
    wanted = {"auto": "a CUDA GPU or an Apple Silicon Mac", "cuda": "a CUDA GPU", "apple": "an Apple Silicon Mac (MPS)"}
    raise NoBackend(f"see-actions needs {wanted[backend]}; this machine has none (--backend cpu runs on the CPU)")
