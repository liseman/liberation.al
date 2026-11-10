"""Reference RTMPose inference, crop transforms, and SimCC decoding.

see-actions' pose path (see/gpu_pose.py) uses the crop constants and is checked against this reference math.

The model is RTMPose (Apache 2.0, OpenMMLab) exported to ONNX: a top-down model, so each person's box is cropped
(padded 1.25x, to the model's 3:4 aspect), normalized, and decoded from its SimCC outputs (per-axis peaks at
half-pixel resolution). The input size is read from the model: RTMPose-l halpe26 at 288x384 offline, RTMPose-m
at 192x256 live.
"""

import logging
import pathlib
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from liberational import model_runtime
from liberational.vision.onnx_runtime import DetectorUnavailable, preload_gpu_libraries

_LOG = logging.getLogger(__name__)

Size = tuple[int, int]  # a model's input, width by height: (192, 256) or (288, 384) for RTMPose
PADDING = 1.25  # a box is grown this much before cropping, as the model was trained
MEAN = np.array([123.675, 116.28, 103.53], np.float32)  # RGB
STD = np.array([58.395, 57.12, 57.375], np.float32)
SPLIT = 2.0  # SimCC bins per pixel


# ---- the model


def crop_transform(box: Sequence[float], size: Size) -> tuple[float, float, float]:
    """(cx, cy, s) for a box in pixels: the crop is the box grown PADDING times and widened or heightened to the
    input's aspect, centered on the box, scaled by s (input pixels per frame pixel)."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    w, h = (x2 - x1) * PADDING, (y2 - y1) * PADDING
    iw, ih = size
    if w > h * iw / ih:
        h = w * ih / iw
    else:
        w = h * iw / ih
    return (x1 + x2) / 2, (y1 + y2) / 2, iw / max(w, 1e-6)


def crop(
    frame: NDArray[Any], box: Sequence[float], size: Size
) -> tuple[NDArray[np.float32], tuple[float, float, float]]:
    """The model's input for one box (3 x height x width, RGB, normalized) and its crop_transform."""
    import cv2

    iw, ih = size
    cx, cy, s = t = crop_transform(box, size)
    m = np.array([[s, 0, iw / 2 - s * cx], [0, s, ih / 2 - s * cy]], np.float32)
    patch = cv2.warpAffine(frame, m, (iw, ih), flags=cv2.INTER_LINEAR)
    rgb = patch[:, :, ::-1].astype(np.float32)
    return ((rgb - MEAN) / STD).transpose(2, 0, 1), t


def decode(simcc_x: NDArray[Any], simcc_y: NDArray[Any]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """SimCC outputs (N x K x bins) -> (N x K x 2 locations in input pixels, N x K confidences). A joint's confidence
    is the lower of its two axis peaks; a joint with no positive peak is placed at -1, -1 with confidence 0."""
    x = np.argmax(simcc_x, axis=2).astype(np.float32) / SPLIT
    y = np.argmax(simcc_y, axis=2).astype(np.float32) / SPLIT
    conf = np.minimum(np.max(simcc_x, axis=2), np.max(simcc_y, axis=2)).astype(np.float32)
    locs = np.stack([x, y], axis=2)
    locs[conf <= 0] = -1
    return locs, np.maximum(conf, 0)


def to_frame(locs: NDArray[np.float32], t: tuple[float, float, float], size: Size) -> NDArray[np.float32]:
    """Input-pixel locations back to frame pixels (the inverse of crop's affine)."""
    cx, cy, s = t
    out = np.empty_like(locs)
    out[..., 0] = (locs[..., 0] - size[0] / 2) / s + cx
    out[..., 1] = (locs[..., 1] - size[1] / 2) / s + cy
    return out


class Session(Protocol):
    """The part of an onnxruntime.InferenceSession the model uses."""

    def run(self, output_names: Any, input_feed: dict[str, Any], /) -> Sequence[Any]: ...


class PoseModel:
    """RTMPose over ONNX Runtime: estimate(frame, boxes) -> one (K x 3 array of x, y in frame pixels and confidence)
    per box."""

    def __init__(
        self,
        session: Session,
        input_name: str,
        name: str,
        version: str,
        provider: str = "?",
        batched: bool = True,
        size: Size = (192, 256),
        cpu: Callable[[], Session] | None = None,
    ) -> None:
        self.session, self.input, self.name, self.version, self.provider = session, input_name, name, version, provider
        # A CPU session to fall back to when the accelerated one fails at run time. CoreML passed load's probe runs
        # (batches of 1 and 2) but failed on live frames, which crashed see-bodies every ~20 s (the Mac,
        # 2026-11-02: 23k failures).
        self.cpu = cpu
        self.size = size  # the input, width by height, read from the model when it's loaded
        self.batched = batched  # False for an export with a fixed batch of 1: each person is run alone

    def _run(self, blob: NDArray[np.float32]) -> tuple[NDArray[Any], NDArray[Any]]:
        try:
            simcc_x, simcc_y = (np.asarray(o) for o in self.session.run(None, {self.input: blob})[:2])
        except Exception as exc:
            if self.cpu is None:
                raise DetectorUnavailable(f"Pose model failed on {self.provider}: {exc}") from exc
            _LOG.warning("Pose model failed on %s; using the CPU from now on: %s", self.provider, exc)
            self.session, self.provider, self.cpu = self.cpu(), "CPUExecutionProvider", None
            model_runtime.loaded("models.files.pose_live", f"{self.name}.onnx", "body pose", self.provider, owner=self)
            return self._run(blob)
        if simcc_x.ndim != 3 or simcc_x.shape[:2] != simcc_y.shape[:2] or simcc_x.shape[0] != len(blob):
            raise DetectorUnavailable(f"Pose model output has shapes {simcc_x.shape} and {simcc_y.shape}")
        return simcc_x, simcc_y

    def estimate(self, frame: NDArray[Any], boxes: Sequence[Sequence[float]]) -> list[NDArray[np.float32]]:
        if not boxes:
            return []
        crops = [crop(frame, b, self.size) for b in boxes]
        blob = np.stack([c[0] for c in crops]).astype(np.float32)
        if self.batched:
            simcc_x, simcc_y = self._run(blob)
        else:
            parts = [self._run(blob[i : i + 1]) for i in range(len(blob))]
            simcc_x, simcc_y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        locs, conf = decode(simcc_x, simcc_y)
        out = []
        for i, (_, t) in enumerate(crops):
            pts = to_frame(locs[i], t, self.size)
            pts[conf[i] <= 0] = -1
            out.append(np.concatenate([pts, conf[i][:, None]], axis=1))
        return out


def load(path: pathlib.Path, provider: str = "auto") -> PoseModel:
    """The model from its weights, on the GPU when one can run it (checked with a real run), else the CPU."""
    import hashlib

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise DetectorUnavailable("ONNX Runtime is unavailable") from exc
    if not path.exists():
        raise DetectorUnavailable("Pose model weights are missing; run mise run models.")
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    preload_gpu_libraries(ort)
    available = ort.get_available_providers()
    accelerated = [p for p in ("CoreMLExecutionProvider", "CUDAExecutionProvider") if p in available]
    chosen = accelerated if provider == "auto" else [provider] if provider in available else []
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2  # Keep reference CPU inference bounded without a retired runtime setting.
    for providers in ([*chosen, "CPUExecutionProvider"], ["CPUExecutionProvider"]):
        try:
            s = ort.InferenceSession(str(path), options, providers=providers)
            model_input = s.get_inputs()[0]
            name, (ih, iw) = model_input.name, model_input.shape[2:4]
            if not (isinstance(iw, int) and isinstance(ih, int)):
                raise DetectorUnavailable(f"Pose model input has no fixed size: {model_input.shape}")
            s.run(None, {name: np.zeros((1, 3, ih, iw), np.float32)})
            try:  # a frame with two people is one batch, when the export allows it
                s.run(None, {name: np.zeros((2, 3, ih, iw), np.float32)})
                batched = True
            except Exception:
                batched = False
            on_cpu = providers == ["CPUExecutionProvider"] or s.get_providers()[0] == "CPUExecutionProvider"

            def cpu_session() -> Session:
                return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

            model = PoseModel(
                s, name, path.stem, digest, s.get_providers()[0], batched, (iw, ih), None if on_cpu else cpu_session
            )
            model_runtime.loaded("models.files.pose_live", path.name, "body pose", model.provider, owner=model)
            return model
        except Exception as exc:
            if providers == ["CPUExecutionProvider"]:
                raise DetectorUnavailable(f"Pose model cannot run: {exc}") from exc
            _LOG.warning("Pose model acceleration failed; using CPU: %s", exc)
    raise AssertionError("unreachable")
