"""`pack-surroundings`: collects pictures of each place from its cameras while "learn surroundings" is on (#256).
docs/SPATIAL.md has the design; spatial/surroundings.py is the switch, and spatial/surroundings_frames.py the gate and
the store. The same pictures give each camera its own point cloud (spatial/surroundings_clouds.py decides when and
from which picture; spatial/camera_clouds.py makes and keeps it).

The service follows the switch live (`surroundings.read`, every `SWITCH_EVERY_S`). While it's off, or no camera is
chosen, it holds no connection to the shared decoder and no model: it costs one small file's `stat` a second. While
it's on it follows each chosen camera's BGR pictures from see-decode (`shared_frames.decoded_frames`, the pictures
the other workers share, never a decode of its own), one every `SAMPLE_EVERY_S`, and looks at each (`Sampler.look`),
cheapest check first:

1. **dark**: not lit colour (`cameras.daylight`: a dark frame, or a camera's night mode);
2. **same**: its 32 x 18 thumbnail is at least `SAME_THUMB` alike to the camera's last look the gate judged (5
   below), and the kept frame that stood for that view is still kept; so a still camera stops here, for a few
   milliseconds a sample. The comparison is to that look, not to the sample before, so a scene that drifts slowly
   (the daylight) still gets past once it has drifted far enough. A look skipped for blur or for someone in view
   judged nothing and is never compared with: on the Linux machine two cameras whose first looks had a person in
   them read "same" from then on and kept nothing. And when the kept frame is dropped, the view is open again;
3. **blurred**: under `BLUR` of the camera's median sharpness over its last `SHARP_WINDOW` looked-at frames (cameras
   differ ten times over in what their sharp frame scores, so there's no one bar);
4. **person**: see-actions' detector, on the CPU, finds people or animals (`Living`) whose boxes, each grown by
   `PAD`, cover more than `MAX_COVERED` of the frame. With less covered the frame goes on to the gate with those
   pixels marked (`cover`), kept beside it so whatever reads the picture can leave them out.
   Skipping every frame with someone in it was the first rule, and it starved the cameras that matter: on the Linux
   machine two cameras of a room someone sat in had a person in every one of 32 looks. Without the detector's model
   nothing is marked, and the log says so once;
5. the gate (`surroundings_frames.Store.offer`): **copy**, or kept as a new **view**, a **changed** one, or a
   **sharper** copy.

The looks run on one worker thread, so the models never take more than their own threads of CPU, and a camera whose
look is still running when its next picture comes passes that picture by. What it's doing is in
`state/surroundings/status.json` (`STATUS`), rewritten every `STATUS_EVERY_S` while it changes: whether learning is
on, and per camera the counts of each outcome since the service started, when a frame was last kept, and the
outcomes of its last `RECENT` looks (`recent`), which is how a page says why a camera's frames aren't being kept now.

A camera the decoder refuses or stops serving is logged and tried again after `RETRY_S`; the other cameras carry on.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import math
import signal
import statistics
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import json_state, logs
from liberational.spatial import surroundings
from liberational.spatial import surroundings_clouds as clouds
from liberational.spatial import surroundings_frames as frames

LOG = logs.get("surroundings")

SAMPLE_EVERY_S = 5.0
SWITCH_EVERY_S = 1.0
STATUS_EVERY_S = 5.0
RETRY_S = 30.0
SAME_THUMB = 0.995
THUMB = (32, 18)
BLUR = 0.5
SHARP_WINDOW = 20
MIN_SHARP_HISTORY = 5  # looked-at frames a camera needs before one can be called blurred
STATUS = "status.json"  # under the store's folder
RECENT = 12  # looks a camera's "recent" outcomes cover: a minute of samples
OUTCOMES = ("dark", "same", "blurred", "person", "copy", "view", "changed", "sharper")

#: The people and animals in a BGR frame, as boxes (`Living`).
Detector = Callable[[np.ndarray], Sequence[tuple[float, float, float, float]]]
#: A camera's pictures, one every so often (`pictures`).
Pictures = Callable[[str], AsyncIterator[np.ndarray]]


PERSON_SCORE = 0.5  # RF-DETR's score for a person: under see-actions' tracking floor, as a missed person costs more
THREADS = 2  # the person detector's CPU threads
PAD = 0.15  # of a box's own size, added on every side of it before it's covered
MAX_COVERED = 0.25  # of the frame: with more of it than this covered by people, the frame is skipped


Box = tuple[float, float, float, float]  # x1, y1, x2, y2, fractions of the frame


def living(dets: np.ndarray, logits: np.ndarray) -> list[Box]:
    """The people and animals among RF-DETR's queries: `dets` (N x 4: centre x, centre y, width, height, fractions
    of the frame) whose `logits` (N x classes, sparse COCO) make a person at PERSON_SCORE or an animal at the
    detector's own animal floor, as clipped boxes. One body can be two queries; their union is what's covered."""
    from liberational.vision import crossings

    scores = 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64)))
    person = scores[:, crossings.RF_CLASS_IDS[crossings.COCO_PERSON]] >= PERSON_SCORE
    animals = [slot for coco, slot in crossings.RF_CLASS_IDS.items() if coco != crossings.COCO_PERSON]
    animal = (scores[:, animals] >= crossings.ANIMAL_SCORE).any(axis=1)
    out: list[Box] = []
    for cx, cy, w, h in np.asarray(dets, dtype=np.float64)[person | animal]:
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def cover(boxes: Sequence[Box], width: int, height: int, pad: float = PAD) -> np.ndarray:
    """The pixels of a `width` x `height` frame that `boxes` cover (H x W bool), each box grown by `pad` of its own
    size on every side: a box is drawn tight, and a hand or a shadow reaches past it."""
    out = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        dx, dy = pad * (x2 - x1), pad * (y2 - y1)
        out[
            max(0, math.floor((y1 - dy) * height)) : math.ceil((y2 + dy) * height),
            max(0, math.floor((x1 - dx) * width)) : math.ceil((x2 + dx) * width),
        ] = True
    return out


class Living:
    """The people and animals in a BGR frame, as boxes: see-actions' detector (RF-DETR,
    [models.files.person_detector]) on the CPU, about 0.14 s a frame. Loaded on the first frame that gets this far.
    Without the model, or with one that isn't RF-DETR's, every frame reads as empty, and the log says so once."""

    def __init__(self) -> None:
        self._run: Callable[[np.ndarray], list[Box]] | None = None
        self._missing = False

    def _load(self) -> Callable[[np.ndarray], list[Box]] | None:
        import onnxruntime as ort

        from liberational.vision import crossings

        path = cfg.model_path("person_detector")
        if not path.exists():
            LOG.warning("no person detector at %s: frames with people in them aren't skipped", path)
            return None
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        entry = session.get_inputs()[0]
        names = [o.name for o in session.get_outputs()]
        if set(names) != {"dets", "labels"}:
            LOG.warning("the person detector at %s isn't RF-DETR's: frames with people in them aren't skipped", path)
            return None
        size = int(entry.shape[2])

        def run(pixels: np.ndarray) -> list[Box]:
            got = dict(zip(names, session.run(None, {entry.name: crossings.detector_input(pixels, size)}), strict=True))
            return living(np.asarray(got["dets"])[0], np.asarray(got["labels"])[0])

        return run

    def __call__(self, pixels: np.ndarray) -> list[Box]:
        if self._run is None and not self._missing:
            self._run = self._load()
            self._missing = self._run is None
        return [] if self._run is None else self._run(np.asarray(pixels))

    def unload(self) -> None:
        """Free the model while nothing is being sampled."""
        self._run = None


def thumbnail(pixels: np.ndarray) -> np.ndarray:
    """A frame's 32 x 18 grey thumbnail, zero mean and unit length, so two frames' likeness is a dot product that an
    exposure step doesn't move."""
    import cv2

    grey = cv2.cvtColor(np.asarray(pixels, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    small -= small.mean()
    return small / max(float(np.linalg.norm(small)), 1e-6)


class Sampler:
    """Looks at sampled frames and keeps the ones the gate takes. One per service; `look` is called from one thread."""

    def __init__(
        self,
        store: frames.Store,
        describe: Callable[[np.ndarray], np.ndarray],
        detect: Detector | None,
        lit: Callable[[np.ndarray], bool],
        sharpness: Callable[[np.ndarray], float],
    ) -> None:
        self.store = store
        self.describe = describe
        self.detect = detect
        self.lit = lit
        self.sharpness = sharpness
        self.counts: dict[str, collections.Counter[str]] = {}
        self.recent: dict[str, collections.deque[str]] = {}  # each camera's last RECENT outcomes
        self.last_kept: dict[str, float] = {}
        #: Each camera's last look the gate judged: its thumbnail, and the kept frame that stands for that view.
        self._judged: dict[str, tuple[np.ndarray, str]] = {}
        self._sharp: dict[str, collections.deque[float]] = {}

    def look(self, camera: str, pixels: np.ndarray, now: float) -> str:
        """Look at one sampled frame of `camera`; the outcome, one of OUTCOMES."""
        outcome = self._look(camera, pixels, now)
        self.counts.setdefault(camera, collections.Counter())[outcome] += 1
        self.recent.setdefault(camera, collections.deque(maxlen=RECENT)).append(outcome)
        return outcome

    def _look(self, camera: str, pixels: np.ndarray, now: float) -> str:
        if not self.lit(pixels):
            return "dark"
        thumb = thumbnail(pixels)
        judged = self._judged.get(camera)
        if judged is not None and float(judged[0] @ thumb) >= SAME_THUMB:
            if any(f.id == judged[1] for f in self.store.kept):
                return "same"
            del self._judged[camera]  # the kept frame that stood for this view is gone: the view is open again
        sharp = self.sharpness(pixels)
        history = self._sharp.setdefault(camera, collections.deque(maxlen=SHARP_WINDOW))
        usual = statistics.median(history) if len(history) >= MIN_SHARP_HISTORY else None
        history.append(sharp)
        if usual is not None and sharp < BLUR * usual:
            return "blurred"
        covered = None
        boxes = self.detect(pixels) if self.detect is not None else ()
        if boxes:
            covered = cover(boxes, pixels.shape[1], pixels.shape[0])
            if float(covered.mean()) > MAX_COVERED:
                return "person"
        decision, kept = self.store.offer(camera, pixels, self.describe(pixels), sharp, now=now, covered=covered)
        # Only now is this view judged: it's kept, or a kept frame already shows it. A look skipped for blur or for
        # someone in view judged nothing, so the next look is taken on its own.
        standing = kept.id if kept is not None else decision.nearest
        if standing is not None:
            self._judged[camera] = (thumb, standing)
        if kept is not None:
            self.last_kept[camera] = now
            LOG.info(
                "%s: kept a frame (%s, %.2f alike to the nearest kept); %d kept in all",
                camera,
                decision.reason,
                decision.alike,
                len(self.store.kept),
            )
        return decision.reason

    def forget(self, camera: str) -> None:
        """Start a camera's "same" and "blurred" checks afresh (it's being followed again after a break)."""
        self._judged.pop(camera, None)
        self._sharp.pop(camera, None)

    def status(self, switch: surroundings.Switch, now: float) -> dict[str, Any]:
        return {
            "t": now,
            "on": switch.on,
            "sampling": list(switch.cameras) if switch.on else [],
            "kept": len(self.store.kept),
            "cameras": {
                camera: {
                    **{o: counts.get(o, 0) for o in OUTCOMES},
                    "last_kept": self.last_kept.get(camera),
                    # what the last looks came to, so a page can say why a camera's frames aren't being kept now
                    "recent": dict(collections.Counter(self.recent.get(camera, ()))),
                }
                for camera, counts in sorted(self.counts.items())
            },
        }


async def pictures(camera: str) -> AsyncIterator[np.ndarray]:
    """`camera`'s BGR pictures from the shared decoder, one every SAMPLE_EVERY_S."""
    from liberational.see import shared_frames

    async for frame in shared_frames.decoded_frames(camera, format="bgr", every_ns=int(SAMPLE_EVERY_S * 1e9)):
        yield frame.pixels


class Service:
    """Follows the switch, and the chosen cameras while it's on."""

    def __init__(
        self,
        sampler: Sampler,
        read: Callable[[], surroundings.Switch] = surroundings.read,
        source: Pictures = pictures,
        clock: Callable[[], float] = time.time,
        on_idle: Callable[[], None] = lambda: None,
        monotonic: Callable[[], float] = time.monotonic,
        clouds: clouds.Clouds | None = None,
    ) -> None:
        self.monotonic = monotonic
        self.sampler = sampler
        self.clouds = clouds
        self.read = read
        self.source = source
        self.clock = clock
        self.on_idle = on_idle
        self.following: dict[str, asyncio.Task[None]] = {}
        self._retry_at: dict[str, float] = {}
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="surroundings")
        self._busy: set[str] = set()
        self._status: dict[str, Any] | None = None
        self._status_at = 0.0
        self._report = logs.Periodic()

    async def _follow(self, camera: str) -> None:
        loop = asyncio.get_running_loop()
        self.sampler.forget(camera)
        if self.clouds is not None:
            self.clouds.forget(camera)
        async for pixels in self.source(camera):
            if camera in self._busy:
                continue  # its last look is still running: pass this picture by
            self._busy.add(camera)
            try:
                await loop.run_in_executor(self._pool, self._look, camera, pixels, self.clock())
            finally:
                self._busy.discard(camera)

    def _look(self, camera: str, pixels: np.ndarray, now: float) -> None:
        """One sampled picture: the sampler's look, then the camera's own cloud's (on the same thread, since they
        share the detector)."""
        self.sampler.look(camera, pixels, now)
        if self.clouds is not None:
            self.clouds.look(camera, pixels, now)

    def step(self) -> surroundings.Switch:
        """Bring what's followed in line with the switch as it reads now; the switch."""
        switch = self.read()
        want = set(switch.cameras) if switch.on else set()
        now = self.monotonic()
        for camera, task in list(self.following.items()):
            if task.done():
                del self.following[camera]
                error = None if task.cancelled() else task.exception()
                if error is not None:
                    LOG.warning(
                        "%s: no pictures from the shared decoder (%s); trying again in %.0f s", camera, error, RETRY_S
                    )
                    self._retry_at[camera] = now + RETRY_S
            elif camera not in want:
                task.cancel()
                del self.following[camera]
                LOG.info("%s: stopped sampling", camera)
        for camera in sorted(want - set(self.following)):
            if now >= self._retry_at.get(camera, 0.0):
                self.following[camera] = asyncio.get_running_loop().create_task(self._follow(camera))
                LOG.info("%s: sampling a frame every %.0f s", camera, SAMPLE_EVERY_S)
        if not self.following:
            self.on_idle()
        return switch

    def write_status(self, switch: surroundings.Switch, force: bool = False) -> None:
        now = self.clock()
        status = self.sampler.status(switch, now)
        same = self._status is not None and {**status, "t": 0} == {**self._status, "t": 0}
        if same or (not force and now - self._status_at < STATUS_EVERY_S):
            return
        self._status, self._status_at = status, now
        try:
            json_state.write(self.sampler.store.folder / STATUS, status)
        except OSError as error:
            LOG.warning("couldn't write the status file: %s", error)
        if status["cameras"]:
            LOG.log(
                self._report.level(),
                "kept %d; %s",
                status["kept"],
                "; ".join(
                    f"{camera}: " + ", ".join(f"{o} {n}" for o in OUTCOMES if (n := row[o]))
                    for camera, row in status["cameras"].items()
                ),
            )

    async def run(self, stop: asyncio.Event) -> None:
        was_on: bool | None = None
        try:
            while not stop.is_set():
                switch = self.step()
                if switch.on != was_on:
                    LOG.info("learning surroundings is %s", "on" if switch.on else "off")
                self.write_status(switch, force=switch.on != was_on)
                was_on = switch.on
                try:
                    await asyncio.wait_for(stop.wait(), SWITCH_EVERY_S)
                except TimeoutError:
                    pass
        finally:
            tasks = list(self.following.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._pool.shutdown(wait=True)
            if self.clouds is not None:
                self.clouds.close()


def main(argv: Sequence[str] | None = None) -> None:
    """`pack-surroundings`: collects the chosen cameras' pictures while "learn surroundings" is on, keeps the ones
    that show something new, and keeps each camera's own point cloud up to date. It needs see-decode for pictures;
    while the switch is off it needs nothing."""
    import argparse

    from liberational.spatial import cameras

    parser = argparse.ArgumentParser(prog="pack-surroundings", description=main.__doc__)
    parser.parse_args(argv)
    logs.setup()
    describer, detect = frames.Describer(), Living()
    sampler = Sampler(
        frames.Store(frames.root(cfg.path("paths.state"))), describer, detect, cameras.daylight, cameras.sharpness
    )

    def idle() -> None:
        describer.unload()
        detect.unload()

    async def run() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        own = clouds.for_service(cfg.path("paths.state"), detect, cameras.daylight, thumbnail)
        await Service(sampler, on_idle=idle, clouds=own).run(stop)

    asyncio.run(run())


if __name__ == "__main__":
    main()
