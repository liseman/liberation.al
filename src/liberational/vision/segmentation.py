"""On-demand object masks for the Objects page.

A mask is asked for two ways. A box or point the owner chose prompts an image model once, and that is always the
point-prompt model. "Find objects" (`find_all`) asks for objects across the region the owner is looking at, and there
`find_backend` picks one of two methods at run time from the weights on disk (`objects.mask_backend` pins a choice):

- SAM 3, when its weights are installed and a GPU has room for them: one concept prompt, the name the owner typed,
  which finds far more of what is asked for than any grid of points (the measurement pass measured it).
- the point grid this has always run: one image embedding and 16x16 points, the masks kept by predicted IoU and
  stability, tiny, whole-view, and duplicate ones dropped, at most `FIND_MAX`. It runs on SAM 2 when its weights are
  installed and on EdgeTAM otherwise, and it is the only option for a run with no name to look for.

Nothing runs without the owner asking. Inference runs in one child process, with a deadline, so a model fault cannot
take the dashboard down. The child stays warm between requests with the model loaded (a cold start is mostly importing
torch and transformers, many seconds on a busy machine) and is stopped after `IDLE_S` without one, so an idle dashboard
holds no GPU memory for it (a warm child can hold both the concept model and the point-prompt one, if a run of each
came within that window). Only one request runs at once. Weights must already be installed by `mise run models`; a
request never downloads weights. A concept prompt with no GPU that has room for it falls back to the point grid for
that request, and the answer says so. The optional `segmentation` extra declares PyTorch and Transformers with the timm
vision backbone. CUDA, Apple Silicon MPS, and CPU use the same model interface; the concept model needs an NVIDIA GPU,
so an Apple Silicon or CPU install keeps the point grid.

API reference: https://huggingface.co/docs/transformers/model_doc/edgetam and /sam3
Model and access terms: https://huggingface.co/yonigozlan/EdgeTAM-hf and https://huggingface.co/facebook/sam3
"""

import functools
import io
import math
import multiprocessing
import os
import pathlib
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

import numpy as np
from PIL import Image

from liberational import config as cfg
from liberational import model_runtime


class Unavailable(RuntimeError):
    """No object model can provide a mask right now; callers must not silently use a rectangular crop instead."""


FIND_GRID = 16  # Find objects' point prompts: this many a side over the image (256 points)
FIND_BATCH = 8  # Keep full-resolution mask decoder batches within the shared GPU budget.
FIND_MIN_IOU = 0.7  # a mask's predicted IoU must reach this
FIND_MIN_STABILITY = 0.85  # and its stability: the share of its pixels that stay in when the cutoff moves by 1 logit
FIND_MIN_AREA = 0.001  # masks smaller than this share of the image are noise
FIND_MAX_AREA = 0.9  # and larger than this the whole view, not an object
FIND_OVERLAP = 0.7  # a mask this much like (IoU) a better one is the same object
FIND_MAX = 30  # at most this many masks a run

# The models Find objects can use, by their [models.hf] key. EdgeTAM is installed by every install; the other two are
# optional downloads, and SAM 3's is gated.
CONCEPT = "object_segmentation_concept"  # SAM 3: one concept prompt, the name of the thing
GRID = "object_segmentation_points"  # SAM 2: the same grid on a larger backbone
TODAY = "object_segmentation"  # EdgeTAM: the grid as it has always run
KINDS = {CONCEPT: "sam3", GRID: "sam2", TODAY: "edgetam"}
NAMES = {CONCEPT: "SAM 3", GRID: "SAM 2", TODAY: "EdgeTAM"}  # each model's name in the owner's words
CONCEPT_MIN_SCORE = 0.5  # a concept mask's own score must reach this (the measurement's operating point)
CONCEPT_BYTES = 5 * 2**30  # free GPU memory the concept model needs: 4.6 GB measured, plus the decoder's room
# The reason a concept run fell back, said with the name of the model that ran instead (`find_all` joins the two).
NO_ROOM = "SAM 3 had no room on the GPU (it needs about 5 GB free)"
BACKENDS = ("auto", "sam3", "sam2", "edgetam")


@dataclass(frozen=True)
class Result:
    mask: np.ndarray
    score: float
    model: str


@dataclass(frozen=True)
class Backend:
    """One model, which it is, and where it loads from."""

    kind: str  # "sam3", "sam2", or "edgetam"
    key: str  # its [models.hf] key
    repo: str  # what a Result names as the model that made its mask
    model_ref: str  # the directory to load from


@dataclass(frozen=True)
class Found:
    """A Find objects run: its masks best first, the model that found them, and why it was not the first choice."""

    objects: list[Result]
    backend: Backend
    note: str = ""


class Cancelled(Unavailable):
    """The caller replaced this specific segmentation request."""


class _NoRoom(Unavailable):
    """The concept model did not fit on a GPU: this request uses the point grid instead."""


@dataclass
class _Job:
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    process: BaseProcess | None = None


_lock = threading.Lock()
_jobs_lock = threading.Lock()
_jobs: dict[str, _Job] = {}
_cancelled: dict[str, float] = {}
_current = threading.local()
_active: _Job | None = None


def _job_key(key: str) -> str:
    if not isinstance(key, str) or not key or len(key) > 128:
        raise ValueError("An object mask job must be a nonempty identifier of at most 128 characters.")
    return key


def cancel(key: str) -> bool:
    """Stop only the named request, including one whose HTTP request has not started yet.

    Tombstones are bounded and expire after twenty minutes. Wait until the owner has reaped its child and
    released the busy slot before permitting the UI to submit its replacement. No service or unrelated job is killed.
    """
    key = _job_key(key)
    with _jobs_lock:
        now = time.monotonic()
        expired = [name for name, at in _cancelled.items() if now - at > 1200]
        for name in expired:
            del _cancelled[name]
        if len(_cancelled) >= 1024:
            del _cancelled[next(iter(_cancelled))]
        _cancelled[key] = now
        job = _jobs.get(key)
        if job is None:
            return False
    return _stop(job)


def _stop(job: _Job) -> bool:
    with _jobs_lock:
        job.cancelled.set()
        if job.process is not None and job.process.is_alive():
            job.process.terminate()
    if not job.done.wait(4):
        raise Unavailable("The object model is still stopping the previous selection. Try Freeze View again shortly.")
    return True


def cancel_current() -> bool:
    """Freeze replaces this dashboard's active segmentation job, even after a page reload lost its client ID.

    Only the process registered by this module is signalled. No other model or service is affected, and the
    busy slot must be released before the new snapshot can be returned.
    """
    with _jobs_lock:
        job = _active
    return _stop(job) if job is not None else False


@contextmanager
def _request(key: str | None) -> Generator[None]:
    global _active
    job = _Job()
    with _jobs_lock:
        if key is not None:
            _job_key(key)
            if key in _cancelled and time.monotonic() - _cancelled[key] <= 1200:
                raise Cancelled("The object selection was cancelled.")
            _cancelled.pop(key, None)
        if not _lock.acquire(blocking=False):
            raise Unavailable("An object is already being selected. Try again shortly.")
        if key is not None:
            _jobs[key] = job
        _current.job = job
        _active = job
    try:
        yield
        if job.cancelled.is_set():
            raise Cancelled("The object selection was cancelled.")
    finally:
        with _jobs_lock:
            if key is not None and _jobs.get(key) is job:
                del _jobs[key]
            _current.job = None
            if _active is job:
                _active = None
            _lock.release()
            job.done.set()


def _device(device: str, need: int = 0) -> str:
    """The device to load on: "auto" takes the GPU with the most free memory, else Apple Silicon's, else the CPU.
    With `need` bytes, anything but an NVIDIA GPU with that much free memory is a clean refusal instead."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
            device = f"cuda:{max(range(len(free)), key=free.__getitem__)}"
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
    if need:
        if not device.startswith("cuda"):
            raise _NoRoom(NO_ROOM)
        index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
        if torch.cuda.mem_get_info(index)[0] < need:
            raise _NoRoom(NO_ROOM)
    return device


@functools.cache
def _load(key: str, model_ref: str, device: str) -> tuple[Any, Any, str]:
    """A point-prompt model's model and processor on `device` ("auto": the GPU with the most free memory), in the
    child process, once per child. SAM 2 and EdgeTAM share the interface, so only the class differs."""
    import torch
    from transformers import EdgeTamModel, Sam2Model, Sam2Processor

    torch.set_num_threads(2)
    device = _device(device)
    family = Sam2Model if key == GRID else EdgeTamModel
    model: Any = family.from_pretrained(model_ref, local_files_only=True)
    model = model.to(device).eval()
    processor = Sam2Processor.from_pretrained(model_ref, local_files_only=True)
    model_runtime.loaded(f"models.hf.{key}", model_ref, "object outlines", device, owner=model)
    return model, processor, device


DETECTOR = "detector_model."  # every tensor of the image model inside SAM 3's published checkpoint, by that prefix


@functools.cache
def _load_concept(model_ref: str, device: str) -> tuple[Any, Any, str]:
    """SAM 3's image model and processor, in the child process, once per child. The published checkpoint is the video
    layout: its `detector_model.*` tensors are the image model, and they load into `Sam3Model` by name (1468 of them,
    checked against the checkpoint, with none missing and none unexpected). A device without room is `_NoRoom`, and
    nothing is loaded."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, Sam3Model, Sam3Processor

    torch.set_num_threads(2)
    device = _device(device, CONCEPT_BYTES)
    config = AutoConfig.from_pretrained(model_ref, local_files_only=True)
    model: Any = Sam3Model(config.detector_config)
    weights = {
        name[len(DETECTOR) :]: value
        for name, value in load_file(pathlib.Path(model_ref) / "model.safetensors").items()
        if name.startswith(DETECTOR)
    }
    report = model.load_state_dict(weights, strict=False)
    del weights
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            f"SAM 3's weights do not fit its image model: {len(report.missing_keys)} missing, "
            f"{len(report.unexpected_keys)} unexpected"
        )
    model = model.to(device).eval()
    processor = Sam3Processor.from_pretrained(model_ref, local_files_only=True)
    model_runtime.loaded(f"models.hf.{CONCEPT}", model_ref, "object outlines", device, owner=model)
    return model, processor, device


def _infer(
    image: bytes, prompt: dict[str, list[float]], key: str, model_ref: str, device: str
) -> tuple[np.ndarray, float]:
    """The third-party inference boundary, only called in the child process."""
    import torch

    model, processor, device = _load(key, model_ref, device)
    raw = Image.open(io.BytesIO(image)).convert("RGB")
    width, height = raw.size
    kwargs: dict[str, Any] = {}
    if "bbox" in prompt:
        x, y, w, h = prompt["bbox"]
        kwargs["input_boxes"] = [[[x * width, y * height, (x + w) * width, (y + h) * height]]]
    else:
        x, y = prompt["point"]
        kwargs["input_points"] = [[[[x * width, y * height]]]]
        kwargs["input_labels"] = [[[1]]]
    inputs = processor(images=raw, return_tensors="pt", **kwargs).to(device)
    with torch.inference_mode():
        outputs = model(
            pixel_values=inputs["pixel_values"],
            **{key: inputs[key] for key in ("input_points", "input_labels", "input_boxes") if key in inputs},
            multimask_output=False,
        )
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    mask = masks[0, 0].numpy().astype(bool)
    score = float(outputs.iou_scores[0, 0, 0].item())
    return mask, score


def _infer_all(image: bytes, key: str, model_ref: str, device: str) -> list[tuple[np.ndarray, float]]:
    """Every object the point grid finds in the image (see the module's docs), best first, as (mask, score): the score
    is the predicted IoU times the stability. Only called in the child process."""
    import torch

    model, processor, device = _load(key, model_ref, device)
    raw = Image.open(io.BytesIO(image)).convert("RGB")
    width, height = raw.size
    inputs = processor(images=raw, return_tensors="pt").to(device)
    grid = (np.arange(FIND_GRID) + 0.5) / FIND_GRID
    points = [[float(x * width), float(y * height)] for y in grid for x in grid]
    logits, ious = [], []
    with torch.inference_mode():
        embeddings = model.get_image_embeddings(inputs["pixel_values"])
        for start in range(0, len(points), FIND_BATCH):
            batch = points[start : start + FIND_BATCH]
            prompts = processor(
                images=raw,
                input_points=[[[point] for point in batch]],
                input_labels=[[[1] for _ in batch]],
                return_tensors="pt",
            ).to(device)
            out = model(
                input_points=prompts["input_points"],
                input_labels=prompts["input_labels"],
                image_embeddings=embeddings,
                multimask_output=True,
            )
            logits.append(out.pred_masks[0].float())
            ious.append(out.iou_scores[0].float())
        low, iou = torch.cat(logits), torch.cat(ious)  # points x 3 candidate masks: keep each point's best
        best = iou.argmax(1)
        rows = torch.arange(len(low), device=low.device)
        low, iou = low[rows, best], iou[rows, best]
        stability = (low > 1).flatten(1).sum(1) / (low > -1).flatten(1).sum(1).clamp(min=1)
        keep = (iou >= FIND_MIN_IOU) & (stability >= FIND_MIN_STABILITY)
        low, score = low[keep], (iou * stability)[keep]
        if not len(low):
            return []
        masks = processor.post_process_masks(low[None, :, None].cpu(), inputs["original_sizes"].cpu())[0][:, 0]
    return select(masks.numpy().astype(bool), score.cpu().numpy())


def select(masks: np.ndarray, scores: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """The masks to show, best score first: none smaller than FIND_MIN_AREA or larger than FIND_MAX_AREA of the
    image, none more than FIND_OVERLAP like a better one, at most FIND_MAX."""
    kept: list[tuple[np.ndarray, float]] = []
    areas = masks.reshape(len(masks), -1).mean(1) if len(masks) else np.zeros(0)
    for k in np.argsort(-scores, kind="stable"):
        mask = masks[k]
        if not FIND_MIN_AREA <= areas[k] <= FIND_MAX_AREA:
            continue
        if any((mask & other).sum() > FIND_OVERLAP * (mask | other).sum() for other, _ in kept):
            continue
        kept.append((mask, float(scores[k])))
        if len(kept) == FIND_MAX:
            break
    return kept


def _infer_concept(image: bytes, concept: str, model_ref: str, device: str) -> list[tuple[np.ndarray, float]]:
    """The objects SAM 3 finds for one concept (the name the owner typed), best first, as (mask, score). Only called in
    the child process."""
    import torch

    model, processor, device = _load_concept(model_ref, device)
    raw = Image.open(io.BytesIO(image)).convert("RGB")
    width, height = raw.size
    inputs = processor(images=raw, text=concept, return_tensors="pt").to(device)
    dtype = next(model.parameters()).dtype
    inputs = {name: (value.to(dtype) if value.is_floating_point() else value) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    found = processor.post_process_instance_segmentation(
        outputs, threshold=CONCEPT_MIN_SCORE, mask_threshold=0.5, target_sizes=[(height, width)]
    )[0]
    return select(found["masks"].cpu().numpy().astype(bool), found["scores"].cpu().numpy())


def _find(
    image: bytes, key: str, model_ref: str, device: str
) -> tuple[list[tuple[np.ndarray, tuple[int, ...], float]]]:
    """`_infer_all` with its masks packed: 30 full masks would be tens of MB through the pipe."""
    return ([(np.packbits(m), m.shape, s) for m, s in _infer_all(image, key, model_ref, device)],)


def _find_concept(
    image: bytes, concept: str, model_ref: str, device: str
) -> tuple[list[tuple[np.ndarray, tuple[int, ...], float]]]:
    """`_infer_concept` with its masks packed, as `_find` does."""
    return ([(np.packbits(m), m.shape, s) for m, s in _infer_concept(image, concept, model_ref, device)],)


MISSING = "Object mask dependencies are missing. Install them with uv sync --inexact --extra segmentation, then restart the dashboard."
# Library errors may contain credentials or local paths: a stable, actionable message instead.
FAILED = "The object model could not load or run. Check model access and installed weights with mise run models; no selection was saved."


@model_runtime.reporting("objects")
def _serve(conn: Connection) -> None:
    """The warm child: runs each `(target, args)` it is sent and answers `(True, *result)` or `(False, why)`, until
    its pipe closes. `_load` is cached here, so the model loads once per child."""
    if hasattr(os, "nice"):
        os.nice(19)
    while True:
        try:
            message = conn.recv()
        except EOFError, OSError:
            return
        if message is None:
            return
        target, args = message
        try:
            conn.send((True, *target(*args)))
        except ImportError:
            conn.send((False, MISSING))
        except _NoRoom:
            conn.send((False, NO_ROOM))
        except Exception:
            conn.send((False, FAILED))


IDLE_S = 120.0  # the warm child is stopped after this long without a request, so an idle dashboard holds no GPU memory
REAP_EVERY_S = 5.0  # how often the idle child is looked at


@dataclass
class _Server:
    process: BaseProcess
    conn: Connection
    used: float


_server: _Server | None = None  # the warm child, when one is running; only touched holding _lock
_reaper: threading.Thread | None = None


def _shutdown(server: _Server) -> None:
    """Ends a warm child: its exit frees everything it held on the GPU."""
    try:
        server.conn.close()
    except OSError:
        pass
    process = server.process
    if process.is_alive():
        process.terminate()
    if process.pid is not None:
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()


def reap_idle(now: float | None = None) -> bool:
    """Stops the warm child when it has been idle IDLE_S, unless a request holds it. Returns whether it stopped one."""
    global _server
    if not _lock.acquire(blocking=False):
        return False
    try:
        server = _server
        if server is None or (time.monotonic() if now is None else now) - server.used < IDLE_S:
            return False
        _server = None
        _shutdown(server)
        return True
    finally:
        _lock.release()


def _reap_forever() -> None:
    while True:
        time.sleep(REAP_EVERY_S)
        reap_idle()


def _start() -> _Server:
    """A new warm child (and the reaper thread, once)."""
    global _reaper
    context = multiprocessing.get_context("spawn")
    conn, theirs = context.Pipe(duplex=True)
    process = context.Process(target=_serve, args=(theirs,), daemon=True)
    process.start()
    theirs.close()
    if _reaper is None or not _reaper.is_alive():
        _reaper = threading.Thread(target=_reap_forever, name="object-masks-idle", daemon=True)
        _reaper.start()
    return _Server(process, conn, time.monotonic())


def _run(target: Any, args: tuple[Any, ...], timeout_s: float) -> tuple[Any, ...]:
    """Runs `target(*args)` in the warm child, starting one if none is running, and returns what it gave. Call holding
    _lock (inside `_request`). A cancel, a timeout, or a child that died ends the child; the next request starts
    another."""
    global _server
    job: _Job | None = getattr(_current, "job", None)
    with _jobs_lock:
        if job is not None and job.cancelled.is_set():
            raise Cancelled("The object selection was cancelled.")
        if _server is None or not _server.process.is_alive():
            _server = _start()
        server = _server
        if job is not None:
            job.process = server.process
    ok = False
    try:
        try:
            server.conn.send((target, args))
        except (OSError, ValueError) as exc:
            raise Unavailable("The object model stopped before returning a mask. No selection was saved.") from exc
        deadline = time.monotonic() + timeout_s
        while True:
            if job is not None and job.cancelled.is_set():
                raise Cancelled("The object selection was cancelled.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Unavailable(
                    "The object model took too long. Try again when the machine is less busy; no selection was saved."
                )
            if server.conn.poll(min(remaining, 0.05)):
                break
        try:
            result = server.conn.recv()
        except (EOFError, OSError) as exc:
            if job is not None and job.cancelled.is_set():
                raise Cancelled("The object selection was cancelled.") from exc
            raise Unavailable("The object model stopped before returning a mask. No selection was saved.") from exc
        ok = True
        if not result[0]:
            raise _NoRoom(result[1]) if result[1] == NO_ROOM else Unavailable(result[1])
        return tuple(result[1:])
    finally:
        server.used = time.monotonic()
        if not ok:
            with _jobs_lock:
                if _server is server:
                    _server = None
            _shutdown(server)


def segment(image: bytes, *, bbox: object = None, point: object = None, job: str | None = None) -> Result:
    """A mask for exactly one normalized box [x,y,w,h] or point [x,y]: a prompt, not a concept, so the point-prompt
    model answers (SAM 2 when its weights are installed, else EdgeTAM)."""
    from liberational.memory.objects import _box

    if (bbox is None) == (point is None):
        raise ValueError("Choose one box or point.")
    if bbox is not None:
        prompt = {"bbox": list(_box(bbox))}
    else:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError("point must be normalized [x,y]")
        xy = [float(v) for v in point]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in xy):
            raise ValueError("point outside image")
        prompt = {"point": xy}
    chosen = points_backend(forced())
    with _request(job):
        mask, score = _run(
            _infer,
            (image, prompt, chosen.key, chosen.model_ref, str(cfg.get("objects.segmentation_device"))),
            _timeout(),
        )
    with Image.open(io.BytesIO(image)) as raw:
        shape = (raw.height, raw.width)
    if not isinstance(mask, np.ndarray) or mask.shape != shape or not mask.any() or not math.isfinite(score):
        raise Unavailable("The object model did not find a usable mask there. Draw a clearer box and try again.")
    return Result(mask.astype(bool), score, chosen.repo)


def _timeout() -> float:
    """One request's deadline, including the model's load."""
    timeout_s = float(cfg.get("objects.segmentation_timeout_s"))
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("objects.segmentation_timeout_s must be positive")
    return timeout_s


def forced() -> str:
    """`objects.mask_backend`: "auto" (the default), or the one backend the owner pinned."""
    want = str(cfg.get("objects.mask_backend", "auto")).strip().lower()
    if want not in BACKENDS:
        raise Unavailable(f'objects.mask_backend is "{want}"; it must be one of {", ".join(BACKENDS)}.')
    return want


def installed(key: str) -> bool:
    """Whether a declared model's weights are installed. Only a local copy counts: a request never downloads weights,
    and Hugging Face's own cache is filled by `mise run models`, which puts every one of these under data/models."""
    local = str(cfg.get(f"models.hf.{key}.path", ""))
    return bool(local) and (cfg.path(f"models.hf.{key}.path") / "config.json").is_file()


def _model(key: str) -> Backend:
    """One declared model: its repo, and the installed directory it loads from, or its repo when it isn't installed."""
    repo = str(cfg.get(f"models.hf.{key}.repo", key))
    return Backend(KINDS[key], key, repo, str(cfg.path(f"models.hf.{key}.path")) if installed(key) else repo)


def points_backend(want: str = "auto") -> Backend:
    """The model that takes point and box prompts: SAM 2 when its weights are installed, else EdgeTAM, which every
    install has. "edgetam" keeps today's model, and "sam2" without its weights is an error rather than a silent swap."""
    if want != "edgetam" and installed(GRID):
        return _model(GRID)
    if want == "sam2":
        raise Unavailable(
            f"Find objects is set to SAM 2 (objects.mask_backend), and {_model(GRID).repo} is not installed: "
            "run `mise run models`, then try again."
        )
    return _model(TODAY)


def find_backend(concept: str = "") -> tuple[Backend, str]:
    """The backend a Find objects run uses, and a plain-words note when it is not the first choice: SAM 3's concept
    prompt for the name given, else the point grid (see the module's docs)."""
    want = forced()
    if want in ("auto", "sam3"):
        if installed(CONCEPT):
            if concept:
                return _model(CONCEPT), ""
            return points_backend(want), (
                "SAM 3 looks for the name you give it, so name the object first; this run found everything in view "
                "instead."
            )
        if want == "sam3":
            raise Unavailable(
                f"Find objects is set to SAM 3 (objects.mask_backend), and {_model(CONCEPT).repo} is not installed: "
                "run `mise run models`, and give this machine's Hugging Face login access to it first."
            )
    return points_backend(want), ""


def describe() -> dict[str, str]:
    """Which model Find objects uses and why, in plain words: the models status and the Objects page show it. The
    words say what each model does for the owner, never our internals or a command (the docs carry those)."""
    try:
        want = forced()
        if installed(CONCEPT) and want in ("auto", "sam3"):
            return {
                "backend": KINDS[CONCEPT],
                "model": _model(CONCEPT).repo,
                "why": "SAM 3 is installed, so Find objects looks for the name you type, and finds far more of what "
                "you ask for. It loads when you ask, and stops when it has been idle.",
            }
        if want == "sam3":
            return {
                "backend": KINDS[CONCEPT],
                "model": _model(CONCEPT).repo,
                "why": "SAM 3 is chosen in Settings and isn't installed, so Find objects can't run until it is.",
            }
        if want == "sam2" or installed(GRID):
            return {
                "backend": KINDS[GRID],
                "model": _model(GRID).repo,
                "why": "SAM 3, which finds objects by name, isn't installed. SAM 2 finds everything in view instead.",
            }
        return {
            "backend": KINDS[TODAY],
            "model": _model(TODAY).repo,
            "why": "SAM 3, which finds objects by name, isn't installed. EdgeTAM finds everything in view instead.",
        }
    except Unavailable as exc:  # a bad setting must not take the Objects page or the machine's status down
        return {"backend": "unavailable", "model": "", "why": str(exc)}


def mask_backend_row() -> dict[str, str]:
    """The mask backend as one row of the models status (`model_runtime.configured`): its purpose in the list's two
    or three words, its model, and, in plain words, why that one is in use."""
    seen = describe()
    return {
        "key": "objects.mask_backend",
        "name": seen["model"] or "no object model",
        "purpose": "find objects",
        "why": seen["why"],
        "device": "unknown",
    }


def find_all(image: bytes, *, concept: str = "", job: str | None = None) -> Found:
    """The objects found in the image, best first (see the module's docs); none is an answer, not an error. A concept
    whose model no GPU has room for falls back to the point grid for this run only, and says why."""
    chosen, note = find_backend(concept)
    device = str(cfg.get("objects.segmentation_device"))
    with _request(job):
        if chosen.kind == KINDS[CONCEPT]:
            try:
                (found,) = _run(_find_concept, (image, concept, chosen.model_ref, device), _timeout())
            except _NoRoom as exc:
                chosen = points_backend(forced())
                note = f"{exc}, so {NAMES[chosen.key]} looked instead."
                (found,) = _run(_find, (image, chosen.key, chosen.model_ref, device), _timeout())
        else:
            (found,) = _run(_find, (image, chosen.key, chosen.model_ref, device), _timeout())
    with Image.open(io.BytesIO(image)) as raw:
        shape = (raw.height, raw.width)
    out = []
    for packed, mask_shape, score in found:
        if tuple(mask_shape) != shape or not math.isfinite(score):
            raise Unavailable("The object model returned masks that don't fit this view. Try again.")
        mask = np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)
        out.append(Result(mask, float(score), chosen.repo))
    return Found(out, chosen, note)
