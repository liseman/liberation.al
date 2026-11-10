"""On-demand EdgeTAM object masks for the Objects page.

Each mask is asked for: a box or point the owner chose prompts the EdgeTAM image model once (#1481), or the owner's
"Find objects" asks for every object in the region they're looking at (`find_all`, #2020): one image embedding and a
grid of point prompts, the masks kept by predicted IoU and stability, tiny, whole-view, and duplicate ones dropped,
at most `FIND_MAX`. Nothing runs without the owner asking. Inference runs in one child process, with a
deadline, so a model fault cannot take the dashboard down. The child stays warm between requests with the model loaded
(a cold start is mostly importing torch and transformers, many seconds on a busy machine) and is stopped after
`IDLE_S` without one, so an idle dashboard holds no GPU memory for it.
Only one request runs at once. Weights must already be installed by `mise run models`; a request never downloads
weights or substitutes a different segmentation method. The optional `segmentation` extra declares PyTorch and
Transformers with the timm vision backbone. CUDA, Apple Silicon MPS, and CPU use the same model interface.

API reference: https://huggingface.co/docs/transformers/model_doc/edgetam
Model and access terms: https://huggingface.co/yonigozlan/EdgeTAM-hf
"""

import functools
import io
import math
import multiprocessing
import os
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
    """EdgeTAM cannot currently provide a mask; callers must not silently use a rectangular crop instead."""


FIND_GRID = 16  # Find objects' point prompts: this many a side over the image (256 points)
FIND_BATCH = 8  # Keep full-resolution mask decoder batches within the shared GPU budget.
FIND_MIN_IOU = 0.7  # a mask's predicted IoU must reach this
FIND_MIN_STABILITY = 0.85  # and its stability: the share of its pixels that stay in when the cutoff moves by 1 logit
FIND_MIN_AREA = 0.001  # masks smaller than this share of the image are noise
FIND_MAX_AREA = 0.9  # and larger than this the whole view, not an object
FIND_OVERLAP = 0.7  # a mask this much like (IoU) a better one is the same object
FIND_MAX = 30  # at most this many masks a run


@dataclass(frozen=True)
class Result:
    mask: np.ndarray
    score: float
    model: str


class Cancelled(Unavailable):
    """The caller replaced this specific segmentation request."""


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
        raise ValueError("EdgeTAM job must be a nonempty identifier of at most 128 characters.")
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
        raise Unavailable("EdgeTAM is still stopping the previous selection. Try Freeze View again shortly.")
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
                raise Cancelled("EdgeTAM selection was cancelled.")
            _cancelled.pop(key, None)
        if not _lock.acquire(blocking=False):
            raise Unavailable("EdgeTAM is already selecting another object. Try again shortly.")
        if key is not None:
            _jobs[key] = job
        _current.job = job
        _active = job
    try:
        yield
        if job.cancelled.is_set():
            raise Cancelled("EdgeTAM selection was cancelled.")
    finally:
        with _jobs_lock:
            if key is not None and _jobs.get(key) is job:
                del _jobs[key]
            _current.job = None
            if _active is job:
                _active = None
            _lock.release()
            job.done.set()


@functools.cache
def _load(model_ref: str, device: str) -> tuple[Any, Any, str]:
    """EdgeTAM's model and processor on `device` ("auto": the GPU with the most free memory), in the child process, once
    per child."""
    import torch
    from transformers import EdgeTamModel, Sam2Processor

    torch.set_num_threads(2)
    if device == "auto":
        if torch.cuda.is_available():
            free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
            device = f"cuda:{max(range(len(free)), key=free.__getitem__)}"
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
    model: Any = EdgeTamModel.from_pretrained(model_ref, local_files_only=True)
    model = model.to(device).eval()
    processor = Sam2Processor.from_pretrained(model_ref, local_files_only=True)
    model_runtime.loaded("models.hf.object_segmentation", model_ref, "object outlines", device, owner=model)
    return model, processor, device


def _infer(image: bytes, prompt: dict[str, list[float]], model_ref: str, device: str) -> tuple[np.ndarray, float]:
    """The third-party inference boundary, only called in the child process."""
    import torch

    model, processor, device = _load(model_ref, device)
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


def _infer_all(image: bytes, model_ref: str, device: str) -> list[tuple[np.ndarray, float]]:
    """Every object EdgeTAM finds in the image (see the module's docs), best first, as (mask, score): the score is the
    predicted IoU times the stability. Only called in the child process."""
    import torch

    model, processor, device = _load(model_ref, device)
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


def _find(image: bytes, model_ref: str, device: str) -> tuple[list[tuple[np.ndarray, tuple[int, ...], float]]]:
    """`_infer_all` with its masks packed: 30 full masks would be tens of MB through the pipe."""
    return ([(np.packbits(m), m.shape, s) for m, s in _infer_all(image, model_ref, device)],)


MISSING = "EdgeTAM dependencies are missing. Install it with uv sync --inexact --extra segmentation, then restart the dashboard."
# Library errors may contain credentials or local paths: a stable, actionable message instead.
FAILED = "EdgeTAM could not load or run. Check model access and installed weights with mise run models; no selection was saved."


@model_runtime.reporting("objects")
def _serve(conn: Connection) -> None:
    """The warm child: runs each `(target, args)` it is sent and answers `(True, *result)` or `(False, why)`, until
    its pipe closes. `_load` is cached here, so the model loads once per child (#2033)."""
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
        _reaper = threading.Thread(target=_reap_forever, name="edgetam-idle", daemon=True)
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
            raise Cancelled("EdgeTAM selection was cancelled.")
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
            raise Unavailable("EdgeTAM stopped before returning a mask. No selection was saved.") from exc
        deadline = time.monotonic() + timeout_s
        while True:
            if job is not None and job.cancelled.is_set():
                raise Cancelled("EdgeTAM selection was cancelled.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Unavailable(
                    "EdgeTAM took too long. Try again when the machine is less busy; no selection was saved."
                )
            if server.conn.poll(min(remaining, 0.05)):
                break
        try:
            result = server.conn.recv()
        except (EOFError, OSError) as exc:
            if job is not None and job.cancelled.is_set():
                raise Cancelled("EdgeTAM selection was cancelled.") from exc
            raise Unavailable("EdgeTAM stopped before returning a mask. No selection was saved.") from exc
        ok = True
        if not result[0]:
            raise Unavailable(result[1])
        return tuple(result[1:])
    finally:
        server.used = time.monotonic()
        if not ok:
            with _jobs_lock:
                if _server is server:
                    _server = None
            _shutdown(server)


def segment(image: bytes, *, bbox: object = None, point: object = None, job: str | None = None) -> Result:
    """An EdgeTAM mask from exactly one normalized box [x,y,w,h] or point [x,y]."""
    from liberational.memory.objects import _box

    if (bbox is None) == (point is None):
        raise ValueError("Choose one box or point for EdgeTAM.")
    if bbox is not None:
        prompt = {"bbox": list(_box(bbox))}
    else:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError("point must be normalized [x,y]")
        xy = [float(v) for v in point]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in xy):
            raise ValueError("point outside image")
        prompt = {"point": xy}
    repo, model_ref, timeout = _settings()
    with _request(job):
        mask, score = _run(_infer, (image, prompt, model_ref, str(cfg.get("objects.segmentation_device"))), timeout)
    with Image.open(io.BytesIO(image)) as raw:
        shape = (raw.height, raw.width)
    if not isinstance(mask, np.ndarray) or mask.shape != shape or not mask.any() or not math.isfinite(score):
        raise Unavailable("EdgeTAM did not find a usable object mask. Draw a clearer box and try again.")
    return Result(mask.astype(bool), score, repo)


def _settings() -> tuple[str, str, float]:
    """The model's repo, where to load it from, and the request's deadline."""
    repo = cfg.hf_repo("object_segmentation")
    local = cfg.get("models.hf.object_segmentation.path", "")
    model_ref = str(cfg.path("models.hf.object_segmentation.path")) if local else repo
    timeout = float(cfg.get("objects.segmentation_timeout_s"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("objects.segmentation_timeout_s must be positive")
    return repo, model_ref, timeout


def find_all(image: bytes, *, job: str | None = None) -> list[Result]:
    """Every object EdgeTAM finds in the image, best first (see the module's docs); none is an answer, not an error."""
    repo, model_ref, timeout = _settings()
    with _request(job):
        (found,) = _run(_find, (image, model_ref, str(cfg.get("objects.segmentation_device"))), timeout)
    with Image.open(io.BytesIO(image)) as raw:
        shape = (raw.height, raw.width)
    out = []
    for packed, mask_shape, score in found:
        if tuple(mask_shape) != shape or not math.isfinite(score):
            raise Unavailable("EdgeTAM returned masks that don't fit this view. Try again.")
        mask = np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)
        out.append(Result(mask, float(score), repo))
    return out
