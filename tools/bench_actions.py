"""Bench see-actions' per-frame work on this machine's vision backend at a paced frame rate.

    .venv/bin/python tools/bench_actions.py --streams 1 --seconds 30
    .venv/bin/python tools/bench_actions.py --backend apple --streams 3 --video clip.mp4

Each stream runs on its own thread with its own backend, as pack-see-actions runs each camera: on CUDA the thread
makes its own CUDA stream before loading, so its ONNX Runtime sessions queue on it without host waits and its frame
work replays as CUDA graphs; on the default stream both fall back to the eager, synchronized path, which is
not what runs live. Backends load and capture their graphs one at a time (`actions_worker.GpuGate.loading`), and
each frame runs inside `GpuGate.running`, as live. Each stream is paced at `--fps` (30 by default): a frame slot that
passes while an earlier frame runs is dropped, not caught up, so the rate it reaches is what the machine sustains.
Per frame it does what see-actions does on the accelerator:

- `upload`: the NV12 frame to the backend's upright image;
- `people`: detection and the pose batch;
- `crop`: a head crop of each person, once a second per person (the head-face path);
- `score`: the action model on one window per person, every half second.

`--video` loops a recording (any file OpenCV reads, people in it make the pose batches real); without it, a
synthetic frame has nobody in it and `--people` stands in for the score and crop counts. It prints, per stage, the
median and 95th percentile time in ms, then the frame rate each stream reached, how many frame slots it dropped, the
process's CPU (percent of one core), and where each model ran (a model on CPUExecutionProvider on a Mac means
Core ML declined it). Nothing is written anywhere.
"""

import argparse
import contextlib
import math
import pathlib
import resource
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

STAGES = ("upload", "people", "crop", "score")
CLIP = 90  # frames in an action window: 3 s at 30 fps
SCORE_EVERY_S = 0.5
CROP_EVERY_S = 1.0


def bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    """A BGR frame (even height and width) as NV12: the Y plane, then U and V interleaved at half resolution."""
    import cv2

    h, w = bgr.shape[:2]
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)  # Y, then all of U, then all of V
    y, u, v = i420[:h], i420[h : h + h // 4].reshape(-1), i420[h + h // 4 :].reshape(-1)
    uv = np.empty(u.size * 2, np.uint8)
    uv[0::2], uv[1::2] = u, v
    return np.vstack((y, uv.reshape(h // 2, w)))


def synthetic(h: int = 1080, w: int = 1920, seed: int = 0) -> np.ndarray:
    """A camera-like BGR frame with nobody in it."""
    import cv2

    small = np.random.default_rng(seed).integers(0, 256, (h // 40, w // 40, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def video_frames(path: pathlib.Path, limit: int = 300) -> list[np.ndarray]:
    """Up to `limit` frames of a recording as NV12, decoded up front so decoding isn't in the measurement."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    out = []
    while len(out) < limit:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        out.append(bgr_to_nv12(bgr[: h - h % 2, : w - w % 2]))
    cap.release()
    if not out:
        raise SystemExit(f"no frames in {path}")
    return out


@dataclass
class Frame:
    """What a backend's `upload` reads of a decoded frame."""

    pixels: np.ndarray
    id: str = "bench"
    capture_ns: int = 0
    gap_before: bool = False
    source_header: dict[str, Any] = field(default_factory=dict)
    color_range: str = "limited"
    color_matrix: str = "bt601"


@dataclass
class Stream:
    """One stream's measurements: seconds per stage call, frames done, and frame slots dropped (passed while an
    earlier frame was still running)."""

    times: dict[str, list[float]] = field(default_factory=lambda: {s: [] for s in STAGES})
    frames: int = 0
    dropped: int = 0
    seconds: float = 0.0
    providers: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def timed(times: list[float], call: Callable[..., Any], *args: Any) -> Any:
    """`call(*args)`, its time added to `times`."""
    t = time.perf_counter()
    out = call(*args)
    times.append(time.perf_counter() - t)
    return out


def run_stream(
    backend: Any,
    frames: Iterator[np.ndarray],
    stats: Stream,
    *,
    fps: float,
    seconds: float,
    people: int,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    gate: Callable[[], contextlib.AbstractContextManager[Any]] = contextlib.nullcontext,
    alone: Callable[[], contextlib.AbstractContextManager[Any]] = contextlib.nullcontext,
) -> None:
    """Paces `backend` at `fps` for `seconds`: each frame's stages, a head crop per person once a second, and one
    action window per person every half second. Slots that pass while a frame runs are dropped and counted, so a
    slow machine isn't asked to catch up. Each frame's work runs inside `gate()`, or `alone()` for a frame whose work
    captures a CUDA graph (the backend's `captures`), as see-actions' `follow` runs it."""
    period, start, total = 1.0 / fps, clock(), round(fps * seconds)
    last_crop, last_score, k = -1e9, -1e9, 0
    window = np.zeros((1, CLIP, 26, 3), np.float32)
    while k < total:
        now = clock()
        due = math.floor((now - start) / period + 1e-9)
        if due > k:  # slots that passed while the last frame ran: dropped, not caught up
            stats.dropped += min(due, total) - k
            k = due
            if k >= total:
                break
        elif now < start + k * period:
            sleep(start + k * period - now)
            now = start + k * period
        k += 1
        frame = Frame(next(frames))
        with alone() if backend.captures(frame) else gate():
            image = timed(stats.times["upload"], backend.upload, frame)
            found = timed(stats.times["people"], backend.people, image)
            n = max(len(found.boxes), people)
            h, w = backend.size(image)
            if now - last_crop >= CROP_EVERY_S:
                last_crop = now
                for i in range(n):
                    box = found.boxes[i] if i < len(found.boxes) else np.array([0.4, 0.2, 0.6, 0.8, 1.0])
                    x1, y1, x2 = int(box[0] * w), int(box[1] * h), int(box[2] * w)
                    side = max(8, (x2 - x1) // 2)
                    timed(stats.times["crop"], backend.crop, image, (x1, y1, x1 + side, y1 + side))
            if n and now - last_score >= SCORE_EVERY_S:
                last_score = now
                timed(stats.times["score"], backend.score, np.repeat(window, n, axis=0))
        stats.frames += 1
    stats.seconds = clock() - start


def report(streams: list[Stream], cpu_seconds: float, wall: float) -> list[str]:
    """The printed report: per stage over every stream, then each stream's rate, then the CPU."""
    lines = [f"{'stage':<8} {'calls':>7} {'p50 ms':>8} {'p95 ms':>8}"]
    for stage in STAGES:
        ts = [t for s in streams for t in s.times[stage]]
        if ts:
            p50, p95 = np.percentile(np.asarray(ts) * 1e3, [50, 95])
            lines.append(f"{stage:<8} {len(ts):>7} {p50:>8.2f} {p95:>8.2f}")
    for i, s in enumerate(streams):
        if s.error:
            lines.append(f"stream {i}: failed: {s.error}")
            continue
        rate = s.frames / s.seconds if s.seconds else 0.0
        lines.append(f"stream {i}: {rate:.1f} fps, {s.frames} frames, {s.dropped} dropped")
    lines.append(f"cpu: {100 * cpu_seconds / wall:.0f}% of one core over {wall:.1f} s")
    if streams and streams[0].providers:
        lines.append("models: " + ", ".join(f"{k} on {v}" for k, v in streams[0].providers.items()))
    return lines


def load_stream(models: pathlib.Path, backend: str, gpu: int | None) -> Any:
    """A stream's backend, loaded as see-actions loads a camera's: on CUDA, first a CUDA stream of this thread's own
    (on `gpu`, else the last GPU), which its sessions and graphs bind to."""
    from liberational.see import vision_backend

    if vision_backend.choose(backend) == "cuda":
        import torch

        gpu = torch.cuda.device_count() - 1 if gpu is None else gpu
        torch.cuda.set_device(gpu)
        torch.cuda.set_stream(torch.cuda.Stream(torch.device("cuda", gpu)))
    return vision_backend.load(models, gpu, backend)


def main(argv: list[str] | None = None) -> int:
    from liberational import config as cfg
    from liberational.see import vision_backend
    from liberational.see.actions_worker import GpuGate

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=vision_backend.BACKENDS, default="auto")
    parser.add_argument("--streams", type=int, default=1, help="streams at once, each its own backend (1 to 3)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--video", type=pathlib.Path, default=None, help="a recording to loop (default: synthetic)")
    parser.add_argument("--people", type=int, default=2, help="people to score and crop when none are found")
    parser.add_argument("--models", type=pathlib.Path, default=None, help="default: the person detector's folder")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device for every stream (default: the last)")
    args = parser.parse_args(argv)

    models = args.models or cfg.model_path("person_detector").parent
    source = video_frames(args.video) if args.video else [bgr_to_nv12(synthetic())]
    streams = [Stream() for _ in range(args.streams)]
    names: list[str] = []
    gate = GpuGate()
    measured: dict[str, float] = {}

    def started() -> None:  # every stream loaded and warm: the measurement starts
        usage = resource.getrusage(resource.RUSAGE_SELF)
        measured["cpu"], measured["t"] = usage.ru_utime + usage.ru_stime, time.monotonic()

    ready = threading.Barrier(args.streams, action=started)

    def go(i: int) -> None:
        backend = None
        try:
            # Loaded and warmed alone, as see-actions loads a camera: Core ML compiles, MPS and CUDA settle, and the
            # first upload captures this thread's CUDA graphs, all outside the measurement.
            with gate.loading():
                backend = load_stream(models, args.backend, args.gpu)
                streams[i].providers = getattr(backend, "providers", dict)()
                names.append(backend.name)
                image = backend.upload(Frame(source[0]))
                backend.people(image)
                backend.crop(image, (0, 0, 64, 64))
                backend.score(np.zeros((max(1, args.people), CLIP, 26, 3), np.float32))
        except Exception as e:
            streams[i].error = f"{type(e).__name__}: {e}"
        ready.wait()
        if backend is None:
            return

        def loop() -> Iterator[np.ndarray]:
            while True:
                yield from source

        try:
            run_stream(
                backend,
                loop(),
                streams[i],
                fps=args.fps,
                seconds=args.seconds,
                people=args.people,
                gate=gate.running,
                alone=gate.loading,
            )
        except Exception as e:  # one stream failing still reports the others
            streams[i].error = f"{type(e).__name__}: {e}"

    threads = [threading.Thread(target=go, args=(i,)) for i in range(args.streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    wall = time.monotonic() - measured["t"]
    print(
        f"backend {names[0] if names else args.backend}, {args.streams} stream(s) at {args.fps:g} fps for {args.seconds:g} s"
    )
    print("\n".join(report(streams, usage.ru_utime + usage.ru_stime - measured["cpu"], wall)))
    return 1 if any(s.error for s in streams) else 0


if __name__ == "__main__":
    raise SystemExit(main())
