"""RF-DETR Small people/animals and RTMPose skeletons, keeping frame work on its torch device.

RF-DETR takes a square RGB, ImageNet-normalized image and returns normalized boxes plus COCO logits.
Global query/class top-k decoding replaces YOLOX grids and NMS. Only selected boxes and keypoints return
from the GPU; pose crops, neighbor masking, and SimCC decoding remain on the frame's device.
CUDA uses fixed ONNX Runtime graph bindings. MPS uses Core ML with host I/O; CPU uses ONNX Runtime's
CPU provider. Provider placement is reported rather than inferred from accelerator availability.
"""

import ctypes
import functools
import inspect
import logging
import pathlib
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from liberational import model_runtime
from liberational.see.vision_backend import Frame, People
from liberational.vision import crossings, pose_model
from liberational.vision.onnx_runtime import DetectorUnavailable, preload_gpu_libraries

_LOG = logging.getLogger(__name__)
POSE_BATCHES = (1, 2, 4, 8)  # a pose graph for each; a frame with more people than the largest replays it again
_MEAN = torch.tensor(pose_model.MEAN.tolist())  # RGB
_STD = torch.tensor(pose_model.STD.tolist())
_NORMS: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _norm(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The pose model's mean and std on `device`, uploaded once: a host-to-GPU copy per call waits for the GPU's
    queue to drain, and these two were a fifth of a frame's time."""
    if device not in _NORMS:
        _NORMS[device] = (_MEAN.to(device)[None, :, None, None], _STD.to(device)[None, :, None, None])
    return _NORMS[device]


def block_on_sync(devices: Sequence[int]) -> bool:
    """Make the host sleep, not spin, while it waits on these GPUs: CUDA's default spin-wait took most of a
    30 fps frame loop's CPU (21% of a core against 13% blocking). It must run before torch or ONNX Runtime
    first touches a device, so a service calls it at startup. False when CUDA's runtime library can't be found or
    refuses, which leaves the default: correct, only busier."""
    libs = sorted((pathlib.Path(torch.__file__).parent.parent / "nvidia").glob("*/lib/libcudart.so*"))
    if not libs:
        _LOG.warning("CUDA runtime library not found; GPU waits will spin")
        return False
    cudart = ctypes.CDLL(str(libs[0]))
    schedule_blocking_sync = 4  # cudaDeviceScheduleBlockingSync
    for device in devices:
        if device in _BLOCKING:
            continue  # set once: a second backend on the device finds its context already made
        if cudart.cudaSetDevice(device) != 0 or cudart.cudaSetDeviceFlags(schedule_blocking_sync) != 0:
            _LOG.warning("CUDA refused blocking sync on device %d; GPU waits will spin", device)
            return False
        _BLOCKING.add(device)
    return True


_BLOCKING: set[int] = set()


class GraphSession:
    """One ONNX model on the CUDA provider, replayed as a CUDA graph over fixed input and output buffers.

    The graph replays on the torch stream that was current when the session was made, ordered after the torch work
    queued there before `run` and before the work queued after it, so on that stream `run` neither waits for the
    input nor syncs at its end: the host waits once, when it reads what the frame found. On the legacy
    default stream the replay is not ordered (a replay read a stale input), so a session made there, or run
    from another stream, waits for torch's work first and lets ONNX Runtime sync at the end, as before."""

    def __init__(self, path: pathlib.Path, device: torch.device, input_shape: Sequence[int]) -> None:
        import onnxruntime as ort

        preload_gpu_libraries(ort)
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise DetectorUnavailable("ONNX Runtime has no CUDA provider")
        index = device.index if device.index is not None else torch.cuda.current_device()
        self.device = torch.device("cuda", index)
        stream = torch.cuda.current_stream(self.device)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        cuda = {
            "device_id": str(index),
            "enable_cuda_graph": "1",
            "has_user_compute_stream": "1",
            "user_compute_stream": str(stream.cuda_stream),
        }
        self.session = ort.InferenceSession(str(path), options, providers=[("CUDAExecutionProvider", cuda)])
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise DetectorUnavailable(f"{path.name} did not load on the CUDA provider")
        self.output_names = [output.name for output in self.session.get_outputs()]
        model_input = self.session.get_inputs()[0]
        self.input = torch.zeros(tuple(input_shape), dtype=torch.float32, device=self.device)
        self.outputs: list[torch.Tensor] = []
        self.binding = self.session.io_binding()
        self.binding.bind_input(
            model_input.name, "cuda", index, np.float32, list(self.input.shape), self.input.data_ptr()
        )
        # Output shapes can be symbolic (the batch, the joints), so one CPU run of the fixed input says what they are.
        cpu = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        probe = cpu.run(None, {model_input.name: np.zeros(tuple(input_shape), np.float32)})
        for output, value in zip(self.session.get_outputs(), probe, strict=True):
            shape = list(np.asarray(value).shape)
            buffer = torch.empty(tuple(shape), dtype=torch.float32, device=self.device)
            self.outputs.append(buffer)
            self.binding.bind_output(output.name, "cuda", index, np.float32, shape, buffer.data_ptr())
        self.session.run_with_iobinding(self.binding)  # the first run with the binding captures the graph
        self.stream = stream
        self.queued = ort.RunOptions()  # a replay ONNX Runtime doesn't wait for: the stream orders it
        self.queued.add_run_config_entry("disable_synchronize_execution_providers", "1")

    def ordered(self) -> bool:
        """Whether a replay now is ordered on torch's current stream: the session's own, not the legacy default."""
        return self.stream.cuda_stream != 0 and torch.cuda.current_stream(self.device) == self.stream

    def run(self, x: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Copy `x` into the input buffer (same shape) and replay the model; the output buffers are reused. On the
        session's stream this only queues the replay: read the outputs with torch work on that stream, and the host
        waits for them when it copies them back."""
        if x is not None:
            self.input.copy_(x)
        if self.ordered():
            self.session.run_with_iobinding(self.binding, self.queued)
        else:
            wait(self.device)
            self.session.run_with_iobinding(self.binding)
        return self.outputs


class Replay:
    """Torch work over fixed buffers, captured once as a CUDA graph on the current stream's device and replayed: one
    launch, and one release of the GIL, in place of one per op. A frame's ~70 small ops, each releasing the GIL
    around its launch, left six camera threads handing it back and forth, 15 to 45 sleeps and wakes a frame.

    `work` reads only buffers that outlive the replay and returns its output, which every replay overwrites. It runs
    once eagerly first, so constants it makes on first use exist before the capture. The capture is thread-local,
    so the other cameras' threads go on using the GPU meanwhile."""

    def __init__(self, work: Callable[[], torch.Tensor], device: torch.device) -> None:
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            work()
            self.graph = torch.cuda.CUDAGraph()
            self.graph.capture_begin(capture_error_mode="thread_local")
            try:
                self.output = work()
            finally:
                self.graph.capture_end()
        current.wait_stream(side)

    def replay(self) -> torch.Tensor:
        """Queue the work on the current stream; its output is ready for work queued after it."""
        self.graph.replay()
        return self.output


def wait(device: torch.device) -> None:
    """Block until the work queued on `device` is done: on CUDA through a blocking event, yielding the CPU instead
    of spinning on it; on MPS through its synchronize; on the CPU, torch's work is already done."""
    if device.type == "cuda":
        done = torch.cuda.Event(blocking=True)
        done.record(torch.cuda.current_stream(device))
        done.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


COREML = "CoreMLExecutionProvider"
# Core ML's own program format with fixed shapes (each pose batch size is its own session), on any compute unit
# Core ML picks: the Neural Engine, the GPU, or the CPU.
COREML_OPTIONS = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "1"}


class HostSession:
    """One ONNX model on ONNX Runtime with host input and output, for a backend without CUDA graphs: Core ML's
    provider on a Mac (`coreml`), the CPU otherwise. Its `input` buffer and `outputs` are tensors on `device`, as
    GraphSession's are, so GpuPose drives both the same way. `provider` is what the model's first node landed on:
    Core ML, or the CPU when the provider wasn't available or wouldn't take the model."""

    def __init__(self, path: pathlib.Path, device: torch.device, input_shape: Sequence[int], *, coreml: bool) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session: Any = None
        if coreml and COREML in ort.get_available_providers():
            for provider in ((COREML, COREML_OPTIONS), COREML):  # an older ONNX Runtime refuses the options
                try:
                    self.session = ort.InferenceSession(
                        str(path), options, providers=[provider, "CPUExecutionProvider"]
                    )
                    break
                except Exception as e:
                    _LOG.warning("%s on Core ML (%s): %s", path.name, provider, e)
        elif coreml:
            _LOG.warning("ONNX Runtime has no Core ML provider; %s runs on the CPU", path.name)
        if self.session is None:
            self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.provider = str(self.session.get_providers()[0])
        self.device = device
        self.name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input = torch.zeros(tuple(input_shape), dtype=torch.float32, device=device)
        self.outputs: list[torch.Tensor] = []
        self.run()  # the first run compiles Core ML's program, so a frame never pays for it

    def run(self, x: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Copy `x` into the input buffer (same shape), run the model on the host copy, and put the outputs on the
        device. Reading the input to the host waits for the device work that filled it."""
        if x is not None:
            self.input.copy_(x)
        outs = self.session.run(None, {self.name: self.input.cpu().numpy()})
        self.outputs = [torch.from_numpy(np.asarray(o, np.float32)).to(self.device) for o in outs]
        return self.outputs


def session_for(path: pathlib.Path, device: torch.device, input_shape: Sequence[int]) -> GraphSession | HostSession:
    """The session a model runs in on `device`: a CUDA graph on CUDA, Core ML's provider with host I/O on MPS (a
    Mac), and the CPU provider on the CPU."""
    if device.type == "cuda":
        return GraphSession(path, device, input_shape)
    return HostSession(path, device, input_shape, coreml=device.type == "mps")


# Luma weights (Kr, Kb) per color matrix. An unspecified matrix or range decodes as FFmpeg's scaler does by default:
# BT.601, limited range.
MATRICES = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722), "bt2020nc": (0.2627, 0.0593)}
MATRICES["unspecified"] = MATRICES["bt601"]


def nv12_to_bgr(
    nv12: torch.Tensor, height: int, width: int, *, color_range: str = "limited", color_matrix: str = "bt601"
) -> torch.Tensor:
    """An NV12 frame (height * 3 / 2 rows of `width` bytes: Y, then interleaved U and V at half resolution) as an
    H x W x 3 uint8 BGR frame, decoded with the frame's own signaled range and matrix (`DecodedFrame.color_range`
    and `color_matrix`). JPEG-sourced capture signals full-range BT.601, and network cameras are often BT.709."""
    if color_matrix not in MATRICES:
        raise ValueError(f"unsupported color matrix {color_matrix!r}")
    if color_range not in {"full", "limited", "unspecified"}:
        raise ValueError(f"unsupported color range {color_range!r}")
    kr, kb = MATRICES[color_matrix]
    kg = 1.0 - kr - kb
    y = nv12[:height].float()
    uv = nv12[height : height + height // 2].view(height // 2, width // 2, 2).float()
    uv = uv.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    u, v = uv[..., 0] - 128.0, uv[..., 1] - 128.0
    if color_range == "full":
        luma = y
    else:
        luma = (y - 16.0) * (255.0 / 219.0)
        u, v = u * (255.0 / 224.0), v * (255.0 / 224.0)
    r = luma + 2 * (1 - kr) * v
    b = luma + 2 * (1 - kb) * u
    g = luma - (2 * kb * (1 - kb) / kg) * u - (2 * kr * (1 - kr) / kg) * v
    return torch.stack((b, g, r), dim=2).round_().clamp_(0, 255).to(torch.uint8)


def detector_input(frame: torch.Tensor, size: int) -> torch.Tensor:
    """BGR uint8 pixels to RF-DETR's square RGB float input, with no padding or integer resize rounding."""
    rgb = frame.flip(2).permute(2, 0, 1)[None].float() / 255.0
    resized = F.interpolate(rgb, size=(size, size), mode="bilinear", align_corners=False, antialias=False)
    mean, std = _detector_norm(frame.device)
    return (resized - mean) / std


@functools.lru_cache(maxsize=8)
def _detector_norm(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None],
        torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None],
    )


CANDIDATES = 300  # RF-DETR's global query/class top-k, before selecting person and animal categories
LOW_SCORE = 0.6  # RF-DETR person floor, measured through the six-joint birth gate on the retained stills
ANIMAL_SCORE = crossings.ANIMAL_SCORE
NEIGHBOR_SCORE = 0.5


@functools.lru_cache(maxsize=8)
def _detector_classes(device: torch.device) -> torch.Tensor:
    ids = [-1] * 91
    for internal, sparse in (*crossings.RF_CLASS_IDS.items(), *crossings.RF_SCREEN_IDS.items()):
        ids[sparse] = internal
    return torch.tensor(ids, dtype=torch.float32, device=device)


def rfdetr_candidates(boxes: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Fixed-shape normalized xyxy, sigmoid score, internal class ID; unmapped COCO slots have ID -1.

    Rank all query/class pairs before class filtering, as upstream does. RF-DETR is trained without NMS;
    overlap alone must not remove adjacent people or animals.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4 or logits.ndim != 2 or logits.shape != (len(boxes), 91):
        raise DetectorUnavailable("RF-DETR requires query boxes and 91 sparse COCO logits")
    scores, flat = logits.sigmoid().flatten().topk(min(CANDIDATES, logits.numel()))
    queries = torch.div(flat, 91, rounding_mode="floor")
    categories = flat.remainder(91)
    selected = boxes[queries]
    xyxy = torch.cat((selected[:, :2] - selected[:, 2:] / 2, selected[:, :2] + selected[:, 2:] / 2), 1)
    return torch.cat((xyxy.clamp(0, 1), scores[:, None], _detector_classes(logits.device)[categories, None]), 1)


def scene(
    cand: NDArray[np.float32], score: float, animal_score: float = ANIMAL_SCORE
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Filter globally ranked RF-DETR candidates without applying class-specific NMS."""
    valid = (cand[:, 2] > cand[:, 0]) & (cand[:, 3] > cand[:, 1])
    people = cand[valid & (cand[:, 5] == crossings.COCO_PERSON) & (cand[:, 4] >= score), :5]
    animals = cand[valid & np.isin(cand[:, 5], list(crossings.COCO_ANIMALS)) & (cand[:, 4] >= animal_score)]
    return people, animals


def screens(cand: NDArray[np.float32], score: float = crossings.SCREEN_SCORE) -> NDArray[np.float32]:
    """The screens among the same globally ranked candidates: N x 6, as `scene`'s animals."""
    valid = (cand[:, 2] > cand[:, 0]) & (cand[:, 3] > cand[:, 1])
    return cand[valid & np.isin(cand[:, 5], list(crossings.COCO_SCREENS)) & (cand[:, 4] >= score)]


def neighbor_targets(boxes: np.ndarray, neighbors: np.ndarray, size: pose_model.Size) -> list[int]:
    """Targets whose padded crop intersects another strong box, checked on the host before any GPU mask work.

    A one-pixel conservative crop margin includes interpolation/rounding boundaries. Identical boxes are the
    target itself; its pixels are protected by the mask, so it never requires masking alone."""
    if not len(boxes) or not len(neighbors):
        return []
    iw, ih = size
    center = (boxes[:, :2] + boxes[:, 2:4]) / 2
    sides = (boxes[:, 2:4] - boxes[:, :2]) * pose_model.PADDING
    width = np.maximum(np.maximum(sides[:, 0], sides[:, 1] * iw / ih), 1e-6)
    half = np.column_stack((width / 2, width * ih / iw / 2))
    low, high = center - half - 1, center + half + 1
    intersects = (
        (neighbors[None, :, 0] <= high[:, None, 0])
        & (neighbors[None, :, 2] >= low[:, None, 0])
        & (neighbors[None, :, 1] <= high[:, None, 1])
        & (neighbors[None, :, 3] >= low[:, None, 1])
        & np.any(boxes[:, None, :] != neighbors[None, :, :], axis=2)
    )
    return np.flatnonzero(np.any(intersects, axis=1)).tolist()


def crops(
    frame: torch.Tensor,
    boxes: torch.Tensor,
    size: pose_model.Size,
    *,
    neighbors: torch.Tensor | None = None,
    masked: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose inputs for boxes in frame pixels (N x 4), as `pose_model.crop` cuts them: N x 3 x height x width, RGB,
    normalized, and each crop's (cx, cy, s) as N x 3. With `neighbors` (strong person boxes in frame pixels),
    black out their pixels outside the target box; keep the target and all overlaps, plus one source pixel for
    bilinear boundary samples. The stock padding/aspect stays intact. People inside the target remain inseparable.
    `masked` (N booleans on the frame's device, every box when None) says which crops are masked: a host list would
    be a host-to-GPU copy, and a wait, per index."""
    h, w = frame.shape[:2]
    iw, ih = size
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    bw, bh = (x2 - x1) * pose_model.PADDING, (y2 - y1) * pose_model.PADDING
    wide = bw > bh * iw / ih
    bw = torch.where(wide, bw, bh * iw / ih)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = iw / bw.clamp(min=1e-6)
    # cv2.warpAffine samples frame pixel x = (u - iw / 2) / s + cx for output pixel u (pixel centers on integers).
    # In grid_sample's normalized coordinates (align_corners=False), that's x_n = a * u_n + b.
    theta = torch.zeros((len(boxes), 2, 3), dtype=torch.float32, device=frame.device)
    theta[:, 0, 0] = iw / (w * s)
    theta[:, 0, 2] = (2 * cx - 1 / s + 1) / w - 1
    theta[:, 1, 1] = ih / (h * s)
    theta[:, 1, 2] = (2 * cy - 1 / s + 1) / h - 1
    grid_ = F.affine_grid(theta, [len(boxes), 3, ih, iw], align_corners=False)
    bgr = frame.permute(2, 0, 1)[None].float().expand(len(boxes), -1, -1, -1)
    patch = F.grid_sample(bgr, grid_, mode="bilinear", padding_mode="zeros", align_corners=False)
    if neighbors is not None and len(neighbors) and len(boxes):
        # Every crop's source pixel coordinates, and whether each lies in a neighbor (N x ih x iw), in a few kernels:
        # a Python loop over the neighbors launched a handful per neighbor.
        scale = s[:, None, None]
        x = (torch.arange(iw, device=frame.device)[None, None, :] - iw / 2) / scale + cx[:, None, None]
        y = (torch.arange(ih, device=frame.device)[None, :, None] - ih / 2) / scale + cy[:, None, None]
        inside = (
            (x >= x1[:, None, None] - 1)
            & (x <= x2[:, None, None] + 1)
            & (y >= y1[:, None, None] - 1)
            & (y <= y2[:, None, None] + 1)
        )
        nx = (x[..., None] >= neighbors[:, 0]) & (x[..., None] <= neighbors[:, 2])  # N x 1 x iw x M
        ny = (y[..., None] >= neighbors[:, 1]) & (y[..., None] <= neighbors[:, 3])  # N x ih x 1 x M
        neighbor = (nx & ny).any(dim=3)
        black = neighbor & ~inside
        if masked is not None:
            black &= masked[:, None, None]
        patch = patch.masked_fill(black[:, None], 0)
    rgb = patch.flip(1)
    mean, std = _norm(frame.device)
    return (rgb - mean) / std, torch.stack((cx, cy, s), 1)


def decode(simcc_x: torch.Tensor, simcc_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`pose_model.decode` on the GPU: N x K x 2 locations in input pixels and N x K confidences."""
    vx, ix = simcc_x.max(dim=2)
    vy, iy = simcc_y.max(dim=2)
    locs = torch.stack((ix.float(), iy.float()), 2) / pose_model.SPLIT
    conf = torch.minimum(vx, vy)
    locs[conf <= 0] = -1
    return locs, conf.clamp(min=0)


def to_frame(locs: torch.Tensor, t: torch.Tensor, size: pose_model.Size) -> torch.Tensor:
    cx, cy, s = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    x = (locs[..., 0] - size[0] / 2) / s + cx
    y = (locs[..., 1] - size[1] / 2) / s + cy
    return torch.stack((x, y), 2)


class GpuPose:
    """RF-DETR Small people/animals and RTMPose skeletons for frames already on `device`."""

    def __init__(
        self,
        detector: pathlib.Path,
        pose: pathlib.Path,
        device: torch.device,
        *,
        score: float = LOW_SCORE,
        min_box_height: float = 0.0,
        lens: Any = None,
    ) -> None:
        self.device = device
        self.lens = lens  # a fisheye.Lens: each person is resampled into a virtual pinhole view before pose
        size = model_input(detector)[0]  # the detector's square input, as the model declares it
        self.detector = session_for(detector, device, (1, 3, size, size))
        self.size = int(self.detector.input.shape[2])
        if set(self.detector.output_names) != {"dets", "labels"}:
            raise DetectorUnavailable("RF-DETR requires named dets and labels outputs")
        self._boxes_output = self.detector.output_names.index("dets")
        self._logits_output = self.detector.output_names.index("labels")
        shape = model_input(pose)
        self.pose_size: pose_model.Size = (shape[1], shape[0])
        self.pose = {n: session_for(pose, device, (n, 3, shape[0], shape[1])) for n in POSE_BATCHES}
        self.score, self.min_box_height = score, min_box_height
        self._pinned = device.type == "cuda"
        self._cand = torch.empty((CANDIDATES, 6), dtype=torch.float32, pin_memory=self._pinned)
        self._ranked: Replay | None = None
        self.capture_failed = False  # a capture that failed leaves this backend's torch work eager

    def replays(self) -> bool:
        """Whether this frame's torch work can be captured and replayed (Replay): on CUDA, with the detector's replay
        ordered on the current stream, unless a capture failed."""
        return not self.capture_failed and isinstance(self.detector, GraphSession) and self.detector.ordered()

    def replay(self, work: Callable[[], torch.Tensor]) -> Replay | None:
        """`work` captured as a Replay, or None (and every later frame eager) when the capture fails."""
        try:
            return Replay(work, self.device)
        except RuntimeError:
            _LOG.warning("a CUDA graph capture failed; this camera's frame work runs eagerly", exc_info=True)
            self.capture_failed = True
            return None

    def captures(self) -> bool:
        """Whether the next frame captures a replay (`people`'s candidates), which must run alone on the GPU."""
        return self.replays() and self._ranked is None

    def prepare(self, frame: torch.Tensor) -> None:
        """Fill the detector's input from a BGR frame, for `people(..., prepared=True)`: the upload's replay does it
        with the frame's conversion."""
        self.detector.input.copy_(detector_input(frame, self.size))

    def _candidates(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        return rfdetr_candidates(outputs[self._boxes_output][0], outputs[self._logits_output][0])

    def people(self, frame: torch.Tensor, *, prepared: bool = False) -> People:
        """People and skeletons in one BGR frame (H x W x 3 uint8 on this device), its detector input already
        filled when `prepared` (`prepare`). The host waits twice: once for the globally ranked box candidates and
        once for the joints (only the first when nobody is there). Everything between, model runs included, is
        queued on the frame's stream."""
        h, w = frame.shape[:2]
        if not prepared:
            self.prepare(frame)
        outputs = self.detector.run()  # a graph session's fixed output buffers, which its replay reads
        if self.replays() and self._ranked is None:
            self._ranked = self.replay(lambda: self._candidates(outputs))
        cand = self._ranked.replay() if self._ranked is not None and self.replays() else self._candidates(outputs)
        self._cand[: len(cand)].copy_(cand, non_blocking=True)
        wait(self.device)
        boxes, animals = scene(self._cand[: len(cand)].numpy(), self.score)
        shown = screens(self._cand[: len(cand)].numpy())
        boxes = boxes[(boxes[:, 3] - boxes[:, 1]) >= self.min_box_height]
        if not len(boxes):
            return People(boxes, np.zeros((0, 0, 3), np.float32), animals, screens=shown)
        px = boxes[:, :4] * np.array([w, h, w, h], np.float32)
        if self.lens is None:
            joints = self.skeletons(frame, px, neighbors=px[boxes[:, 4] >= NEIGHBOR_SCORE])
            return People(boxes, joints, animals, screens=shown)
        joints, view_joints = self.skeletons(frame, px, views=True)
        return People(boxes, joints, animals, view_joints, self.pose_size, screens=shown)

    def skeletons(
        self, frame: torch.Tensor, boxes: np.ndarray, *, views: bool = False, neighbors: np.ndarray | None = None
    ) -> Any:
        """The skeleton in each box (N x 4, x1, y1, x2, y2 in frame pixels) of a BGR frame on this device: N x K x 3,
        x and y in frame pixels and the confidence, -1, -1, 0 for a joint not placed. With `views` (a fisheye lens),
        also each skeleton in its virtual view's pixels, as a second array. Plain crops mask `neighbors` outside
        their target; pass every strong box, including boxes in other pose batches. The host waits once."""
        masked = neighbor_targets(boxes, neighbors, self.pose_size) if neighbors is not None else []
        # The boxes, which of them are masked, and the strong neighbors, in one host-to-GPU copy: from pinned memory
        # on CUDA it is queued, not waited for. Rows: N boxes (x1, y1, x2, y2, masked), then M neighbors.
        strong_rows = np.asarray(neighbors, np.float32).reshape(-1, 4) if masked else np.zeros((0, 4), np.float32)
        host = torch.empty((len(boxes) + len(strong_rows), 5), dtype=torch.float32, pin_memory=self._pinned)
        rows = host.numpy()
        rows[: len(boxes), :4] = boxes[:, :4]
        rows[: len(boxes), 4] = 0
        rows[masked, 4] = 1
        rows[len(boxes) :, :4] = strong_rows
        rows[len(boxes) :, 4] = 0
        packed = host.to(self.device, non_blocking=self._pinned)
        px, flags = packed[: len(boxes), :4], packed[: len(boxes), 4] > 0
        strong = packed[len(boxes) :, :4] if masked else None
        largest = POSE_BATCHES[-1]
        joints, in_views = [], []
        for start in range(0, len(px), largest):
            part = px[start : start + largest]
            if self.lens is None:
                part_masked = flags[start : start + largest]
                blob, t = crops(frame, part, self.pose_size, neighbors=strong, masked=part_masked)
            else:
                from liberational.see import fisheye

                blob, view = fisheye.crops(frame, part, self.pose_size, self.lens)
            session = self.pose[next(n for n in POSE_BATCHES if n >= len(part))]
            session.input[len(part) :].zero_()
            session.input[: len(part)].copy_(blob)
            simcc_x, simcc_y = session.run()
            locs, conf = decode(simcc_x[: len(part)], simcc_y[: len(part)])
            if self.lens is None:
                pts = to_frame(locs, t, self.pose_size)
            else:
                pts = fisheye.to_frame(locs, view, self.pose_size, self.lens, frame.shape[:2])
                seen = locs.clone()
                seen[conf <= 0] = -1
                in_views.append(torch.cat((seen, conf[..., None]), 2))
            pts[conf <= 0] = -1
            joints.append(torch.cat((pts, conf[..., None]), 2))
        found = torch.cat(joints) if not (views and in_views) else torch.cat((torch.cat(joints), torch.cat(in_views)))
        host = torch.empty(found.shape, dtype=found.dtype, pin_memory=self._pinned)
        host.copy_(found, non_blocking=True)
        wait(self.device)
        out = host.numpy()
        if views and in_views:
            return out[: len(px)], out[len(px) :]
        return out


def model_input(path: pathlib.Path) -> tuple[int, int]:
    """A model's input height and width, read from the model."""
    import onnxruntime as ort

    shape: Any = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_inputs()[0].shape
    return int(shape[2]), int(shape[3])


class Staging:
    """A pinned host buffer per frame shape, so a frame's upload doesn't wait for the GPU's queue to drain (a
    pageable copy does, about 1.9 ms a frame on cam0). An event marks each copy, and the buffer isn't written again
    until that copy is done."""

    def __init__(self) -> None:
        self._buffers: dict[tuple[int, ...], tuple[torch.Tensor, torch.cuda.Event]] = {}

    def upload(self, pixels: np.ndarray, device: torch.device, *, into: torch.Tensor | None = None) -> torch.Tensor:
        """The pixels on `device`: a new tensor, or copied `into` a device tensor of their shape."""
        if device.type != "cuda":
            out = torch.from_numpy(np.ascontiguousarray(pixels)).to(device)
            return out if into is None else into.copy_(out)
        shape = tuple(pixels.shape)
        if shape not in self._buffers:
            self._buffers[shape] = (torch.empty(shape, dtype=torch.uint8, pin_memory=True), torch.cuda.Event())
        host, copied = self._buffers[shape]
        copied.synchronize()
        host.numpy()[...] = pixels
        out = host.to(device, non_blocking=True) if into is None else into.copy_(host, non_blocking=True)
        copied.record(torch.cuda.current_stream(device))
        return out


def upright_bgr(frame: Frame, device: torch.device, staging: Staging | None = None) -> torch.Tensor:
    """A decoded frame as H x W x 3 uint8 BGR on `device`, rotated 180 degrees when its capture header says the
    camera is mounted flipped, as the CPU readers do (see/frames.py)."""
    if staging is not None:
        nv12 = staging.upload(frame.pixels, device)
    else:
        nv12 = torch.from_numpy(np.ascontiguousarray(frame.pixels)).to(device)
    return upright(nv12, frame.color_range, frame.color_matrix, flipped(frame))


def flipped(frame: Frame) -> bool:
    """Whether the frame's capture header says the camera is mounted upside down."""
    return bool((frame.source_header.get("capture") or {}).get("flip"))


def upright(nv12: torch.Tensor, color_range: str, color_matrix: str, flip: bool) -> torch.Tensor:
    """An NV12 frame on its device as upright H x W x 3 uint8 BGR (`upright_bgr` after the upload)."""
    height, width = nv12.shape[0] * 2 // 3, nv12.shape[1]
    bgr = nv12_to_bgr(nv12, height, width, color_range=color_range, color_matrix=color_matrix)
    return bgr.flip(0, 1) if flip else bgr


def animal_crops(image: torch.Tensor, boxes: np.ndarray, *, tight: bool = False) -> list[np.ndarray]:
    """Each normalized box's crop of a BGR frame on the GPU, copied to the host: square with context
    (`animal_verifier.square`), as the animal check reads it, or with `tight` the box itself, as the embedder does."""
    from liberational.vision.animal_verifier import square_bounds, tight_bounds

    h, w = int(image.shape[0]), int(image.shape[1])
    bounds = tight_bounds if tight else square_bounds
    out = []
    for box in np.asarray(boxes, np.float64):
        a, b, c, d = bounds(h, w, box)
        out.append(image[b:d, a:c].cpu().numpy())
    return out


SHIFT_SIZE = (90, 160)  # the gray thumbnail the camera's shift is measured on, height by width


def area_resize(x: torch.Tensor, size: Any) -> torch.Tensor:
    """F.interpolate(mode="area") of an NCHW tensor, kept on its device. MPS implements area only when the input
    divides evenly by the output, and a person's crop rarely does (see-actions' camera threads stopped on it on
    the Mac), so there it runs on the CPU and comes back."""
    out = (size, size) if isinstance(size, int) else tuple(size)
    if x.device.type == "mps" and (x.shape[-2] % out[0] or x.shape[-1] % out[1]):
        return F.interpolate(x.cpu(), size=out, mode="area").to(x.device)
    return F.interpolate(x, size=out, mode="area")


def phase_shift(frame: torch.Tensor, previous: torch.Tensor | None) -> tuple[tuple[float, float] | None, torch.Tensor]:
    """How far a BGR frame (H x W x 3 uint8) moved from the one whose spectrum is `previous`, by phase correlation
    on a gray thumbnail: (x, y) as fractions of the frame (whole thumbnail pixels), and this frame's spectrum for
    the next call. None the first time. The scene's moving people are a small part of it, so the peak is the
    camera's own shift."""
    gray = frame.float().mean(dim=2)[None, None]
    small = area_resize(gray, SHIFT_SIZE)[0, 0]
    small = small - small.mean()
    window = torch.outer(
        torch.hann_window(SHIFT_SIZE[0], device=frame.device), torch.hann_window(SHIFT_SIZE[1], device=frame.device)
    )
    spectrum = torch.fft.fft2(small * window)
    if previous is None:
        return None, spectrum
    cross = spectrum * previous.conj()
    surface = torch.fft.ifft2(cross / (cross.abs() + 1e-9)).real
    peak = int(torch.argmax(surface))
    dy, dx = divmod(peak, SHIFT_SIZE[1])
    dy = dy - SHIFT_SIZE[0] if dy > SHIFT_SIZE[0] // 2 else dy
    dx = dx - SHIFT_SIZE[1] if dx > SHIFT_SIZE[1] // 2 else dx
    return (dx / SHIFT_SIZE[1], dy / SHIFT_SIZE[0]), spectrum


class Shared:
    """One copy of a model on a device for every camera on it, one call at a time. Each camera's backend loaded
    its own SigLIP 2, DINOv3, ST-GCN++, and MoViNet, about 4 GB on GPU 0 for six cameras, though they run rarely.

    A call runs on the calling camera's own CUDA stream (each camera thread sets one), and every method these models
    have returns host data, which waits for that stream: so when the lock passes to the next camera nothing of the last
    call is still running. The weights are copied to the device on the first caller's stream, so the device is
    synchronized once after loading, before any other camera uses them. The model loads on first use, or at once with
    `loaded()`."""

    def __init__(self, device: torch.device, build: Callable[[], Any]) -> None:
        self.device = device
        self._build = build
        self._model: Any = None
        self._lock = threading.Lock()

    def loaded(self) -> Shared:
        with self._lock:
            self._load()
        return self

    def _load(self) -> Any:
        if self._model is None:
            self._model = self._build()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        return self._model

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return getattr(self._load(), name)(*args, **kwargs)

        return call


_shared: dict[tuple[str, str], Shared] = {}
_shared_lock = threading.Lock()


def shared(device: torch.device, key: str, build: Callable[[], Any]) -> Shared:
    """The one `Shared` copy of the model `key` (its name and weights) on `device`, made by `build` when first used."""
    with _shared_lock:
        return _shared.setdefault((str(device), key), Shared(device, build))


class TorchBackend:
    """see-actions' `vision_backend.Backend` on a torch device: its images are torch tensors on `device`. The models
    run as `session_for` places them (CUDA graphs on CUDA, Core ML on a Mac's MPS, the CPU on the CPU), and the
    frame's conversion, letterbox, crops, shift, and the action model run as torch ops on the device."""

    name = "torch"
    _prepared: torch.Tensor | None = None  # the last upload's image, when its replay filled the detector's input

    def __init__(
        self,
        detector: pathlib.Path,
        pose: pathlib.Path,
        device: torch.device,
        *,
        action_weights: pathlib.Path | None = None,
        clip_weights: pathlib.Path | None = None,
        lens: Any = None,
    ) -> None:
        self.device = device
        self.clip_weights = clip_weights
        self.pose = GpuPose(detector, pose, self.device, lens=lens)
        # The dashboard's model status: each model's provider and device as ONNX Runtime bound them. The pose model
        # has one session per batch size, all placed alike, so the first stands for them.
        first_pose = next(iter(self.pose.pose.values()))
        for key, path, purpose, session in (
            ("models.files.person_detector", detector, "people detection", self.pose.detector),
            ("models.files.pose_live", pose, "body poses", first_pose),
        ):
            model_runtime.loaded(key, path.name, purpose, model_runtime.ort_device(session.session), owner=self)
        self._previous: torch.Tensor | None = None
        self._staging = Staging()
        # On CUDA, each frame shape's conversion and detector input as one replay (Replay), reading a fixed NV12
        # buffer and writing a fixed BGR one: an image is good until the next upload.
        self._uploads: dict[tuple[Any, ...], tuple[torch.Tensor, Replay]] = {}
        self.actions: Shared | None = None
        if action_weights is not None:
            from liberational.see import stgcn

            # Loaded now, as before (see-actions loads each camera's models in turn, before any captures graphs).
            weights = action_weights
            self.actions = shared(
                self.device, f"stgcn:{weights}", lambda: stgcn.ActionModel(weights, self.device)
            ).loaded()

    def upload(self, frame: Frame) -> torch.Tensor:
        """The frame as an upright BGR image on the device, good until the next upload. On CUDA the conversion and
        the detector's input are one replay over fixed buffers, captured on the first frame of each shape."""
        self._prepared = None
        key = self._key(frame)
        if self.pose.replays() and key not in self._uploads:
            nv12 = torch.zeros(frame.pixels.shape, dtype=torch.uint8, device=self.device)

            def convert() -> torch.Tensor:
                bgr = upright(nv12, *key[1:])
                self.pose.prepare(bgr)
                return bgr

            replay = self.pose.replay(convert)
            if replay is not None:
                self._uploads[key] = (nv12, replay)
        if not self.pose.replays() or key not in self._uploads:
            return upright_bgr(frame, self.device, self._staging)
        nv12, replay = self._uploads[key]
        self._staging.upload(frame.pixels, self.device, into=nv12)
        self._prepared = replay.replay()
        return self._prepared

    @staticmethod
    def _key(frame: Frame) -> tuple[Any, ...]:
        return (tuple(frame.pixels.shape), frame.color_range, frame.color_matrix, flipped(frame))

    def captures(self, frame: Frame) -> bool:
        """Whether this frame's work captures a CUDA graph (its shape's first frame): torch refuses a replay while
        any thread captures, and a capture fails when another thread's GPU work cuts in, so the caller runs such
        a frame alone on the GPU, as a model load does."""
        return self.pose.replays() and (self._key(frame) not in self._uploads or self.pose.captures())

    def size(self, image: torch.Tensor) -> tuple[int, int]:
        return int(image.shape[0]), int(image.shape[1])

    def people(self, image: torch.Tensor) -> People:
        if image is self._prepared:  # the upload's replay filled the detector's input already
            return self.pose.people(image, prepared=True)
        return self.pose.people(image)

    # The animal models load on first use, on this backend's device, so see-actions without animals never pays; one
    # copy serves every camera on the device (Shared).
    @functools.cached_property
    def _embedder(self) -> Any:
        from liberational.vision.animal_embedder import DinoEmbedder

        return shared(self.device, "dinov3", lambda: DinoEmbedder(device=str(self.device)))

    @functools.cached_property
    def _verifier(self) -> Any:
        from liberational.vision.animal_verifier import SiglipVerifier

        return shared(self.device, "siglip2", lambda: SiglipVerifier(device=str(self.device)))

    def embed_animals(self, image: torch.Tensor, boxes: np.ndarray) -> np.ndarray:
        return self._embedder.embed_crops(animal_crops(image, boxes, tight=True))

    def verify_animals(self, image: torch.Tensor, boxes: np.ndarray, species: list[str]) -> list[bool]:
        crops_ = animal_crops(image, boxes)
        return [bool(self._verifier.verify_crop(crop, s)) for crop, s in zip(crops_, species, strict=True)]

    def crop(self, image: torch.Tensor, box: tuple[int, int, int, int]) -> np.ndarray:
        """A patch of the upright frame (x1, y1, x2, y2 in its pixels) as host BGR: only the patch leaves the GPU."""
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2].contiguous().cpu().numpy()

    def score(self, windows: np.ndarray) -> np.ndarray:
        if self.actions is None:
            raise RuntimeError("this backend was loaded without the action model")
        return self.actions.score(windows)

    @functools.cached_property
    def _clip_model(self) -> Any:
        from liberational.see import movinet

        weights = self.clip_weights
        if weights is None:
            raise RuntimeError("this backend was loaded without the clip model")
        return shared(self.device, f"movinet:{weights}", lambda: movinet.ClipModel(weights, self.device))

    def clip(self, image: torch.Tensor, box: np.ndarray) -> np.ndarray:
        """A person's crop for the clip model: a square 1.25 times the box's longer side, centred on it, black
        past the frame's edge, at movinet.SIZE as host RGB uint8."""
        from liberational.see import movinet

        height, width = int(image.shape[0]), int(image.shape[1])
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        x1, x2, y1, y2 = x1 * width, x2 * width, y1 * height, y2 * height
        side = max(round(max(x2 - x1, y2 - y1) * 1.25), 2)
        left, top = round((x1 + x2) / 2 - side / 2), round((y1 + y2) / 2 - side / 2)
        square = torch.zeros((side, side, 3), dtype=image.dtype, device=image.device)
        sx, sy, ex, ey = max(left, 0), max(top, 0), min(left + side, width), min(top + side, height)
        if ex > sx and ey > sy:
            square[sy - top : ey - top, sx - left : ex - left] = image[sy:ey, sx:ex]
        small = area_resize(square.permute(2, 0, 1)[None].float(), (movinet.SIZE, movinet.SIZE))
        return small[0].flip(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

    def score_clips(self, clips: np.ndarray) -> np.ndarray:
        return self._clip_model.score(clips)

    def shift(self, image: torch.Tensor) -> tuple[float, float] | None:
        x, self._previous = phase_shift(image, self._previous)
        return x

    def providers(self) -> dict[str, str]:
        """Where each model runs: the detector's and the pose sessions' ONNX Runtime provider (what a bench reports,
        so a model that fell back to the CPU shows)."""
        sessions = {"detector": self.pose.detector, **{f"pose{n}": p for n, p in self.pose.pose.items()}}
        return {k: getattr(v, "provider", "CUDAExecutionProvider") for k, v in sessions.items()}


class CudaBackend(TorchBackend):
    """On a CUDA GPU: the models as CUDA graphs, and the host sleeping, not spinning, on the GPU's waits."""

    name = "cuda"

    def __init__(
        self,
        detector: pathlib.Path,
        pose: pathlib.Path,
        device: torch.device,
        *,
        action_weights: pathlib.Path | None = None,
        clip_weights: pathlib.Path | None = None,
        lens: Any = None,
    ) -> None:
        index = device.index if device.index is not None else 0
        block_on_sync([index])  # before anything touches the device
        super().__init__(
            detector,
            pose,
            torch.device("cuda", index),
            action_weights=action_weights,
            clip_weights=clip_weights,
            lens=lens,
        )


# torch has one MPS stream per process and it isn't thread-safe: see-actions runs one thread (and one backend) per
# camera, and two of them encoding at once aborted the process in Metal ("commit command buffer with uncommitted
# encoder", on the Mac). Every Apple backend call holds this lock; the GPU is shared anyway.
MPS_LOCK = threading.RLock()


def _mps_serialized(method: Any) -> Any:
    @functools.wraps(method)
    def locked(self: Any, *args: Any, **kwargs: Any) -> Any:
        with MPS_LOCK:
            return method(self, *args, **kwargs)

    return locked


class AppleBackend(TorchBackend):
    """On an Apple Silicon Mac (the rig's Mac mini): the models on ONNX Runtime's Core ML provider, and the rest in
    torch on MPS."""

    name = "apple"

    def __init__(
        self,
        detector: pathlib.Path,
        pose: pathlib.Path,
        *,
        action_weights: pathlib.Path | None = None,
        clip_weights: pathlib.Path | None = None,
        lens: Any = None,
    ) -> None:
        with MPS_LOCK:
            super().__init__(
                detector, pose, torch.device("mps"), action_weights=action_weights, clip_weights=clip_weights, lens=lens
            )


for _name, _method in list(vars(TorchBackend).items()):
    if not _name.startswith("_") and inspect.isfunction(_method):
        setattr(AppleBackend, _name, _mps_serialized(_method))


class CpuBackend(TorchBackend):
    """On the CPU alone, only when asked for: the same numbers as the others, far too slow for 30 fps on more than a
    little video. For checking the other backends against, and a machine with nothing else."""

    name = "cpu"

    def __init__(
        self,
        detector: pathlib.Path,
        pose: pathlib.Path,
        *,
        action_weights: pathlib.Path | None = None,
        clip_weights: pathlib.Path | None = None,
        lens: Any = None,
    ) -> None:
        super().__init__(
            detector, pose, torch.device("cpu"), action_weights=action_weights, clip_weights=clip_weights, lens=lens
        )
