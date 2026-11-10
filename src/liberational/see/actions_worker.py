"""`pack-see-actions`: every camera's frames, at its analysis rate (`vision.actions_fps`, and
`vision.actions_camera_fps` by name: cam0 at its full 30), through people, skeletons, tracks, and the action
model, as `pose.track`, `action.window`, `action.span`, and `action.coverage` in the segment store. With MoViNet's
weights downloaded, each person's crops are scored too, for reading, watching TV, a phone, and sleep.

It follows the shared decode (`shared_frames.decoded_frames`, NV12 on the host) at the camera's rate, which the
decoder applies before it downloads a picture from the GPU (`shared_frames.Schedule`), and hands each frame to
`actions.ActionPipeline` on this machine's vision backend (see/vision_backend.py: CUDA on the Linux machine, Core ML and
MPS on the Mac, `--backend` to choose), whose tracked heads also give the faces (`see.head_faces`: `face.match`,
`face.visit`, and `face.embedding`). The rows go to one `continuum_py.SegmentWriter` per stream under
`paths.state/segments`, synced every SYNC_S and on stop. Once a minute it logs each camera's frames, frame rate, time
per frame, and how far behind live it is.

Each camera runs in its own thread, with its own backend (its own ONNX Runtime sessions and CUDA graphs) and its
own CUDA stream: in one loop, every camera waited on every other's GPU work, and six cameras got 14 fps on cam0 and
6 on the rest. By default the main camera (the highest rate) is on the last GPU and the rest on the first;
`vision.actions_camera_gpus` overrides individual cameras. CUDA graph capture is global, so the backends load one at
a time, and while one loads, the running cameras' frames wait (`GpuGate`). The face models are shared, behind a lock.

A camera that fails to load, or fails later, is logged once and tried again with backoff while the others run
on; the process exits 1, for the service manager to start it again, only once no camera is running.

A restart starts new tracks from the live frames, as a gap would; there is no checkpoint to resume. Track IDs
start from the run's start time in milliseconds, times 1000, so they stay unique across restarts.
"""

import argparse
import asyncio
import contextlib
import signal
import threading
import time
from collections.abc import AsyncIterator, Callable, Generator, Sequence
from pathlib import Path
from typing import Any, cast

from liberational import config as cfg
from liberational import continuum_link, logs, model_runtime, segment_log
from liberational.memory import activity_kinds
from liberational.people import face_shots, pet_shots
from liberational.see import action_clips, actions, pose_boxes, screens, vision_backend
from liberational.see.action_animals import Animals
from liberational.see.action_movement import Movement
from liberational.see.shared_frames import Schedule

log = logs.get("see-actions")
NO_BACKEND_WAIT_S = 300  # how long a process with no backend to run on rests before it exits to look again


def _rest(stop: threading.Event, seconds: float) -> None:
    """Wait `seconds`, or until a stop signal."""
    stop.wait(seconds)


POSES = "pose.track"
WINDOWS = "action.window"
COVERAGE = "action.coverage"
SYNC_S = 5.0
REPORT_S = 60.0
RETRY_S = 5.0  # a failed camera's first wait before it is tried again, doubled each failure in a row
RETRY_MAX_S = 300.0  # the longest wait between a failed camera's tries
HEALTHY_S = 60.0  # a camera that ran this long before failing starts its backoff over, and its failure is logged again


class Counted:
    """A sink that counts the rows it passes on."""

    def __init__(self, sink: actions.Sink) -> None:
        self.sink, self.rows = sink, 0

    def append(self, columns: dict[str, Any]) -> None:
        self.rows += len(next(iter(columns.values())))
        self.sink.append(columns)


async def follow(
    frames: AsyncIterator[vision_backend.Frame],
    pipeline: actions.ActionPipeline,
    stop: asyncio.Event,
    *,
    sync: Callable[[], None] = lambda: None,
    clock: Callable[[], float] = time.monotonic,
    every_ns: int = 0,
    camera: str = "",
    gate: Callable[[], contextlib.AbstractContextManager[Any]] = contextlib.nullcontext,
    alone: Callable[[], contextlib.AbstractContextManager[Any]] = contextlib.nullcontext,
) -> int:
    """Feed the pipeline a frame at most every `every_ns` of capture time until `frames` ends or `stop` is set; the
    frames analyzed. Frames skipped for the rate are not gaps: coverage runs on through them. The pipeline's work on
    each frame, and its close, are inside `gate()`, and waiting for the next frame is outside it. A frame whose work
    captures a CUDA graph (the backend's `captures`, a camera's first frame) runs inside `alone()` instead."""
    seen = 0
    synced = reported = clock()
    rate = logs.Periodic()  # the rate line each minute: INFO every 10 minutes, else DEBUG (#59)
    window = {"frames": 0, "busy": 0.0}
    schedule = Schedule(every_ns)
    try:
        async for frame in frames:
            if not schedule.take(frame.capture_ns, frame.gap_before):
                continue
            with alone() if pipeline.backend.captures(frame) else gate():
                began = time.perf_counter()
                pipeline.feed(frame)
                window["busy"] += time.perf_counter() - began
                window["frames"] += 1
                seen += 1
                now = clock()
                if now - synced >= SYNC_S:
                    pipeline.flush()
                    sync()
                    synced = now
            if now - reported >= REPORT_S:
                lag = time.time() - frame.capture_ns / 1e9
                log.log(
                    rate.level(),
                    "%s: %d frames in %.0f s (%.1f a second), %.1f ms a frame, %.1f s behind live",
                    camera or "camera",
                    window["frames"],
                    now - reported,
                    window["frames"] / (now - reported),
                    1000 * window["busy"] / max(1, window["frames"]),
                    lag,
                )
                window = {"frames": 0, "busy": 0.0}
                reported = now
            if stop.is_set():
                break
    finally:
        with gate():
            pipeline.close()
            sync()
    return seen


def fps_of(camera: str) -> float:
    """A camera's analysis rate: `vision.actions_camera_fps` by name, else `vision.actions_fps`."""
    per = cfg.get("vision.actions_camera_fps") or {}
    return float(per.get(camera, cfg.get("vision.actions_fps")))


_screens_on: screens.ScreensOn | None = None
_screens_on_lock = threading.Lock()


def screens_on() -> screens.ScreensOn:
    """The process's one writer of <state>/screens_on.json, shared by its cameras. Locked: cameras
    start in threads of their own, and two that both found none would each make one, and each write only its own
    cameras' spans over the other's."""
    global _screens_on
    with _screens_on_lock:
        if _screens_on is None:
            _screens_on = screens.ScreensOn(cfg.path("paths.state") / "screens_on.json")
        return _screens_on


SCREEN_ZONES_S = 60.0  # how often a camera's screen zones are read again: a newly drawn one counts within this


class ScreenZones:
    """A camera's zones of kind "screen" (see/zones.py), read again at most every SCREEN_ZONES_S. A read that fails
    keeps the last ones."""

    def __init__(self, camera: str, state: Path, clock: Callable[[], float] = time.monotonic) -> None:
        self.camera, self.state, self.clock = camera, state, clock
        self.polygons: list[tuple[tuple[float, float], ...]] = []
        self.read_at = -SCREEN_ZONES_S

    def __call__(self) -> list[tuple[tuple[float, float], ...]]:
        now = self.clock()
        if now - self.read_at >= SCREEN_ZONES_S:
            self.read_at = now
            from liberational.see import zones

            try:
                self.polygons = zones.screen_polygons(zones.current(self.state).get(self.camera))
            except Exception as ex:
                log.warning(f"screen zones of {self.camera} not read: {type(ex).__name__}: {ex}")
        return self.polygons


class Camera:
    """One camera's pipeline and its writers."""

    def __init__(
        self,
        camera: str,
        backend: Any,
        spans: segment_log.Writers,
        faces: Callable[[str], Any],
        live_poses: Callable[[dict[str, Any]], None] | None = None,
        pet_photos: pet_shots.Writer | None = None,
    ) -> None:
        from continuum_py import SegmentWriter

        root = segment_log.root()
        self.camera, self.fps = camera, fps_of(camera)
        self.poses = SegmentWriter(root, POSES, camera, actions.POSE_SCHEMA)
        self.windows = SegmentWriter(root, WINDOWS, camera, actions.WINDOW_SCHEMA)
        self.coverage = SegmentWriter(root, COVERAGE, camera, actions.COVERAGE_SCHEMA)
        self.spans = spans
        self.counted = Counted(self.poses)
        heads = faces(camera)
        who = heads.who if heads is not None else None
        clip_weights = getattr(backend, "clip_weights", None)
        live = activity_kinds.Live(cfg.path("paths.state"))  # the owner's activities, reread each minute
        clips = (
            action_clips.Clips(backend.clip, backend.score_clips, who=who, kinds=live.kinetics)
            if clip_weights is not None
            else None
        )
        self.pipeline = actions.ActionPipeline(
            backend,
            self.counted,
            self.coverage,
            f"yolox-nano+rtmpose-m-halpe26/{backend.name}",
            windows=self.windows,
            spans=lambda closed: spans.append(segment_log.ACTION_SPANS, camera, closed),
            first_track=int(time.time() * 1000) * 1000 + 1,
            movement=Movement(camera, lambda stream, records: spans.append(stream, camera, records)),
            animals=Animals(
                camera, lambda stream, records: spans.append(stream, camera, records), backend, shots=pet_photos
            ),
            faces=heads,
            clips=clips,
            live=(lambda ns, people: live_poses(pose_boxes.record(camera, ns, people))) if live_poses else None,
            camera=camera,
            screen_zones=ScreenZones(camera, cfg.path("paths.state")),
            screens_on=screens_on(),
            kinds=live.ntu,
        )

    def sync(self) -> None:
        self.poses.sync()
        self.windows.sync()
        self.coverage.sync()
        self.spans.sync()

    def close(self) -> None:
        self.poses.close()
        self.windows.close()
        self.coverage.close()


class Locked:
    """An object whose methods run one call at a time: the face models are shared by the camera threads."""

    def __init__(self, inner: Any) -> None:
        self._inner, self._lock = inner, threading.Lock()

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._inner, name)
        if not callable(value):
            return value

        def locked(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return value(*args, **kwargs)

        return locked


class LiveBoxes:
    """The `face.boxes` and `pose.boxes` rings' one writer each, for the Live tab's face boxes and skeletons, shared by
    the camera threads. A ring takes a single writer whose claim lasts its client's session, so one broker client
    holds both, on a small event loop of its own; `push` and `push_poses` are synchronous and go one at a time. Without
    a broker, or a ring, they push nothing, after one warning."""

    def __init__(self) -> None:
        from liberational.see import face_boxes, pose_boxes

        self._faces, self._poses = face_boxes.encode, pose_boxes.encode
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="see-actions-live-boxes", daemon=True)
        self._thread.start()

        async def open_() -> tuple[Any, Any, Any]:
            from continuum_py import Client

            try:
                client = await Client.connect(str(continuum_link.socket_path()), client_name="see-actions")
            except Exception as error:
                log.warning("no live face boxes or skeletons: the broker is unavailable (%s)", error)
                return None, None, None
            writers = []
            for module in (face_boxes, pose_boxes):
                try:
                    writers.append(await module.writer(client))
                except Exception as error:
                    log.warning("no live %s: the broker's ring is unavailable (%s)", module.STREAM, error)
                    writers.append(None)
            return client, writers[0], writers[1]

        self._client, self._writer, self._pose_writer = asyncio.run_coroutine_threadsafe(open_(), self._loop).result(30)

    def push(self, record: dict[str, Any]) -> None:
        if self._writer is not None:
            with self._lock:
                self._writer.push(self._faces(record))

    def push_poses(self, record: dict[str, Any]) -> None:
        if self._pose_writer is not None:
            data = self._poses(record)
            with self._lock:
                self._pose_writer.push(data)

    def close(self) -> None:
        client = self._client
        if client is not None:

            async def close_() -> None:  # the client's own methods need its loop running when they're called
                await continuum_link.close_quietly(client)

            asyncio.run_coroutine_threadsafe(close_(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)


def gpu_for(camera_fps: float, main_fps: float, devices: int, *, camera: str = "") -> int:
    """The main camera (the highest rate) on the last GPU, the others on the first, so the main camera's frames never
    queue behind theirs. One GPU takes them all. A configured camera overrides that default; invalid indexes fail
    before camera threads start, rather than stranding other threads at the model-loading barrier."""
    per = cfg.get("vision.actions_camera_gpus") or {}
    if camera in per:
        device = per[camera]
        if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device < devices:
            raise ValueError(f"vision.actions_camera_gpus.{camera} must be a CUDA device index in [0, {devices})")
        return device
    return devices - 1 if camera_fps >= main_fps or devices < 2 else 0


async def watch(
    camera: Camera,
    decoded: Callable[..., AsyncIterator[Any]],
    stop: threading.Event,
    gate: Callable[[], contextlib.AbstractContextManager[Any]] = contextlib.nullcontext,
    alone: Callable[[], contextlib.AbstractContextManager[Any]] = contextlib.nullcontext,
) -> None:
    """Follow one camera until `stop` is set, in its own thread's event loop, its GPU work inside `gate()` (or
    `alone()` for a frame that captures a CUDA graph)."""
    done = asyncio.Event()

    async def stopping() -> None:
        while not stop.is_set():
            await asyncio.sleep(0.2)
        done.set()

    # No on_break: the next frame's gap_before says how long the break was, and only a long one ends tracks. The
    # camera's rate is applied by the decoder, before a picture it skips is downloaded or copied, so follow
    # takes every frame it is given.
    frames = decoded(camera.camera, format="nv12", every_ns=int(1e9 / camera.fps))
    task = asyncio.create_task(
        follow(
            frames,
            camera.pipeline,
            done,
            sync=camera.sync,
            camera=camera.camera,
            gate=gate,
            alone=alone,
        )
    )
    waiter = asyncio.create_task(stopping())
    await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    if not task.done():
        task.cancel()
    try:
        seen = await task
        log.info("%s stopped after %d frames, %d pose rows", camera.camera, seen, camera.counted.rows)
    except asyncio.CancelledError:
        log.info("%s stopped while waiting for a frame; %d pose rows", camera.camera, camera.counted.rows)
    finally:
        waiter.cancel()
        camera.close()


class GpuGate:
    """Loads one at a time, alone: CUDA graph capture is global, so while a camera's backend loads, no other camera
    may touch the GPU. A camera's first frame captures its frame work's graphs too (gpu_pose.Replay), so it holds
    `loading()` as a load does. Running cameras hold `running()` for each frame's work, any number at once;
    `loading()` waits for those in hand to finish and holds new ones back until the load is done. A waiting load goes
    before new frames, so the cameras' steady stream of frames can't keep it waiting."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._running = 0
        self._loading = False
        self._waiting = 0  # loads waiting for the frames in hand to finish

    @contextlib.contextmanager
    def loading(self) -> Generator[None]:
        with self._changed:
            self._waiting += 1
            try:
                self._changed.wait_for(lambda: not self._loading and not self._running)
            finally:
                self._waiting -= 1
            self._loading = True
        try:
            yield
        finally:
            with self._changed:
                self._loading = False
                self._changed.notify_all()

    @contextlib.contextmanager
    def running(self) -> Generator[None]:
        with self._changed:
            self._changed.wait_for(lambda: not self._loading and not self._waiting)
            self._running += 1
        try:
            yield
        finally:
            with self._changed:
                self._running -= 1
                if not self._running:
                    self._changed.notify_all()


def run_cameras(
    names: Sequence[str],
    load: Callable[[str], Any],
    follow: Callable[[str, Any, GpuGate], None],
    stop: threading.Event,
    *,
    retry_s: float = RETRY_S,
    retry_max_s: float = RETRY_MAX_S,
    healthy_s: float = HEALTHY_S,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Run each camera in a thread of its own until `stop`: `load(name)` its backend, alone on the GPU (`GpuGate`),
    then `follow(name, loaded, gate)` its frames, holding `gate.running()` for its GPU work. False if it stopped
    because no camera could run, else True.

    A camera whose load or follow fails, or whose frames end, is logged once and tried again after `retry_s`, doubled
    each failure in a row up to `retry_max_s`, while the others run on. A failed follow keeps its loaded
    backend; only a failed load loads again. One that followed for `healthy_s` before failing starts over, as a new
    failure. Once every camera is down at the same time, `stop` is set and the process should exit, for the service
    manager to start it again (a GPU gone bad, say, comes back only in a new process). No camera waits on another to
    start, so one that never loads can't hold the rest."""
    gate = GpuGate()
    down: set[str] = set()
    tally = threading.Lock()
    gave_up = threading.Event()

    def failed(name: str, stage: str, streak: int, error: BaseException | None) -> None:
        if streak == 1:
            what = f"{name} failed while {stage}" if error is not None else f"{name}'s frames ended"
            log.error("%s; trying it again in the background", what, exc_info=error)
        else:
            log.debug("%s failed again while %s (%d in a row): %s", name, stage, streak, error)
        with tally:
            down.add(name)
            if down >= set(names) and not stop.is_set():
                log.error("no camera can run; stopping so the service starts again")
                gave_up.set()
                stop.set()

    def run(name: str) -> None:
        loaded: Any = None
        streak = 0
        while not stop.is_set():
            stage, began = "loading", clock()
            error: BaseException | None = None
            try:
                if loaded is None:
                    with gate.loading():
                        loaded = load(name)
                    if streak:
                        log.warning("%s loaded after %d failed tries", name, streak)
                if stop.is_set():
                    return
                stage, began = "following", clock()
                with tally:
                    down.discard(name)
                follow(name, loaded, gate)
                if stop.is_set():
                    return
            except Exception as e:
                error = e
            if stage == "following" and clock() - began >= healthy_s:
                streak = 0
            streak += 1
            failed(name, stage, streak, error)
            # The exponent capped: 2 ** 1024 doesn't convert to a float, and the OverflowError would end
            # this thread, so a camera that failed that many times in a row was never tried again.
            stop.wait(min(retry_max_s, retry_s * 2 ** min(streak - 1, 64)))

    threads = [threading.Thread(target=run, args=(name,), name=f"see-actions-{name}") for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        while thread.is_alive():
            thread.join(timeout=0.5)
    return not gave_up.is_set()


@model_runtime.reporting("see-actions")
def main(argv: Sequence[str] | None = None) -> None:
    """`pack-see-actions`: people, skeletons, tracks, actions, movement, walk-bys, faces, and animals from every
    camera, one thread per camera."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--camera", action="append", default=None, help="a camera to follow (repeatable; default: every camera)"
    )
    parser.add_argument("--manifest", type=Path, default=Path("config/devices.toml"))
    parser.add_argument(
        "--backend",
        choices=vision_backend.BACKENDS,
        default="auto",
        help="where the models run: auto (CUDA, else an Apple Silicon Mac), cuda, apple, or cpu (only by name)",
    )
    args = parser.parse_args(argv)

    try:
        vision_backend.choose(args.backend)
    except vision_backend.NoBackend as e:
        # A GPU gone missing (on the Linux machine, a boot without a working driver) is fixed outside this process,
        # and torch won't see it come back in this one: say so once, rest, then exit so the service manager starts a
        # fresh process that looks again, rather than restarting every few seconds.
        log.error("%s; looking again in %d min", e, NO_BACKEND_WAIT_S // 60)
        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        _rest(stop, NO_BACKEND_WAIT_S)
        raise SystemExit(1) from None

    import torch

    from liberational.people import registry
    from liberational.see import faces as face_models
    from liberational.see import fisheye, gpu_pose, subjects
    from liberational.see import frames as camera_manifest
    from liberational.see.head_faces import HeadFaces
    from liberational.see.shared_frames import decoded_frames
    from liberational.vision import faces as insightface

    names = args.camera or [c.name for c in camera_manifest.cameras_from_manifest(args.manifest)]
    rates = {name: fps_of(name) for name in names}
    devices = torch.cuda.device_count() if args.backend in ("auto", "cuda") else 0  # no CUDA setup otherwise
    try:
        gpus = {
            name: gpu_for(rates[name], max(rates.values()), devices, camera=name) if devices else None for name in names
        }
    except ValueError as e:
        parser.error(str(e))
    if devices:
        gpu_pose.block_on_sync(list(range(devices)))  # before any thread makes a CUDA context
    insightface.app()  # load the face models now, not on the first look, which would stall the frames
    detector, matcher = Locked(face_models.InsightFaceDetector()), Locked(face_models.GalleryMatcher())
    people = subjects.RegistryPeople(registry.at(cfg.path("paths.state")))
    models = cfg.model_path("person_detector").parent
    stop = threading.Event()
    live = LiveBoxes()
    shots = face_shots.Writer(cfg.path("paths.state"))  # People's face thumbnails, one a sighting
    pet_photos = pet_shots.Writer(cfg.path("paths.state"))  # Pets' sighting photos, one a track
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    def load(name: str) -> tuple[Any, Any]:
        gpu = gpus[name]
        if gpu is not None:
            # A stream of its own: torch's current stream is per thread, and on the device's shared default stream
            # one camera's waits would wait for every other camera's work.
            torch.cuda.set_device(gpu)
            torch.cuda.set_stream(torch.cuda.Stream(torch.device("cuda", gpu)))
        # Alone on the GPU (run_cameras' GpuGate): CUDA graph capture is global, so while one thread captures, no
        # other may touch the GPU, loading or running.
        lens = fisheye.lens_for(args.manifest, name)  # a calibrated fisheye: each person is rectified
        return vision_backend.load(models, gpu, args.backend, lens), lens

    def follow(name: str, loaded: tuple[Any, Any], gate: GpuGate) -> None:
        backend, lens = loaded
        gpu = gpus[name]
        writers = segment_log.Writers(segment_log.root())
        try:
            with gate.running():
                camera = Camera(
                    name,
                    backend,
                    writers,
                    lambda cam: HeadFaces(
                        cam,
                        cast(Any, detector),
                        cast(Any, matcher),
                        writers,
                        people=people,
                        live=live.push,
                        shots=shots,
                    ),
                    live_poses=live.push_poses,
                    pet_photos=pet_photos,
                )
            kind = "a fisheye, rectified per person" if lens is not None else "plain"
            log.info("following %s at %g fps on %s %s (%s)", name, camera.fps, backend.name, gpu, kind)
            asyncio.run(watch(camera, decoded_frames, stop, gate.running, gate.loading))
        finally:
            writers.close()

    ok = run_cameras(names, load, follow, stop)
    live.close()
    if not ok:
        raise SystemExit(1)  # no camera could run: the service manager starts it again
