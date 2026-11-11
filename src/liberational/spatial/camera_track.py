"""Tracking a camera while it moves: its pose over time on the plan, held to the plan's walls (#516).

spatial/cloud_scene.py places a camera when it gets a cloud, which takes a view that has held still. Between two
such views a moving camera has no pose. Here it gets one, a few times a second, from its live frames.

**The tracker** is MASt3R-SLAM's own frame tracker (spatial/pointmap_runner.py, in its runtime, as a process that
stays up for the length of a track): each frame is matched to the current keyframe and its pose solved against that
keyframe's points, and a frame that shares too little with the keyframe becomes the next one. It says where each
frame's camera is in the track's own frame (the first keyframe's camera), as a similarity. Nothing else of
MASt3R-SLAM runs: no loop closing, no relocalising.

**On the plan** (`Track`). A track starts where the camera stood (`cloud_scene.standing`). Each keyframe gives a
cloud, made as any camera's is (spatial/camera_clouds.py `make`: cleaned, its floor found, levelled). The first
keyframe's cloud is placed by its walls near where the camera stood, facing any way (spatial/cloud_align.py
`locate`: by the time a move is noticed the camera has turned). That fixes how the track's frame lies on the plan.
Every later keyframe is first put where the track says it is, then held to the plan's walls from there
(`cloud_align.hold`: squared and fitted within the bounds any cloud is). When the walls take it, the track's lie on
the plan is set again from that keyframe: what the track had drifted is gone, the camera's height and tilt come from
the keyframe's own floor, and its scale from the plan's walls. When they don't (no wall in view, or nothing that
fits), the keyframe stays where the track put it and the track says it is running unheld since the last one that
was. Frames between keyframes are placed through the same lie.

**Lost.** A frame the tracker can't match is skipped. After `LOST_FRAMES` in a row the track is begun again from
the next frame, where the camera was last placed, and held to the walls as a first keyframe is; after `RESTARTS` of
those the track ends.

**In the scene.** The last `KEEP` keyframes' clouds (thinned to `THIN` points) and the camera's path are kept
beside its cloud (`track.json`, `track/<n>.ply`); spatial/cloud_scene.py draws them with the camera's own points and
shows the camera where the track has it. Nothing here writes a spot or the plan.

**When it starts and stops** (`Tracker`). The surroundings service samples each chosen camera every few seconds.
A camera starts being tracked when two samples in a row are less than `MOVING` alike with nobody in view (someone
walking past a still camera changes its picture too, so a start that sees no motion ends within `REST_S`), it has
a place in the scene, the runtime is set up, and a GPU has the memory. One camera is tracked at a time. It ends when
the camera has rested: its pose within `REST` of where it was `REST_S` ago. It also ends after `MAX_S`, and is then
left alone for `COOLDOWN_S`. When the camera's view has held still long enough for a cloud, that cloud is placed
from where the track left the camera (`cloud_scene.align`), the track's files go, and it is an ordinary camera
again.

**What it costs**, measured on the Linux machine beside the other services: about 10 s to load the model at the
start of a track, then 0.2 s of one GPU a frame at `TRACK_EVERY_S` (about two fifths of that GPU's time while a
camera moves) and 2.9 GB of its memory, all given back when the track ends. A made-up pan (one still picture turned
through 30 degrees and back) was tracked to within 0.3 degrees, the camera wandering up to 7 cm.
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from liberational import json_state, logs
from liberational.memory import places
from liberational.spatial import camera_clouds, cloud_align, cloud_scene, ply

LOG = logs.get("camera-track")

TRACK_EVERY_S = 0.5  # a frame this often while a camera moves
MOVING = 0.9  # two samples in a row under this alike: the view is moving (a still camera reads 0.99 and up)
REST = (0.05, 2.0)  # metres and degrees: a camera within this of where it was `REST_S` ago has come to rest
REST_S = 10.0
MAX_S = 600.0  # a track is ended after this long, moving or not
COOLDOWN_S = 60.0  # and a camera whose track ended is not tracked again for this long
LOST_FRAMES = 6  # frames in a row the tracker couldn't match before the track is begun again
RESTARTS = 5  # times a track is begun again before it ends
KEEP = 6  # keyframes whose clouds are kept for the scene
THIN = 30000  # points of a keyframe's cloud that are kept
PATH = 240  # poses of the path that are kept: two minutes at `TRACK_EVERY_S`
WRITE_EVERY_S = 1.0  # how often the track is written while only its pose changes
ANSWER_S = 30.0  # how long the runner may take over one frame
READY_S = 120.0  # and over loading the model


@dataclass(frozen=True)
class Step:
    """What the tracker said of one frame (spatial/pointmap_runner.py's answer)."""

    ok: bool
    pose: np.ndarray | None = None  # the frame's camera in the track's frame, 4 x 4, a similarity
    started: bool = False  # this frame began a track: what came before is in another frame
    points: camera_clouds.Pointmap | None = None  # a keyframe's points, in its own camera frame


class Session(Protocol):
    """A tracker that takes a camera's frames in order."""

    def step(self, picture: np.ndarray, overlay: np.ndarray | None, reset: bool) -> Step: ...

    def close(self) -> None: ...


class Runner:
    """The tracker in the configured runtime: one process, up from `open` to `close`."""

    def __init__(self, found: camera_clouds.Runtime, gpu: int) -> None:
        self._folder = Path(tempfile.mkdtemp(prefix="camera-track-"))
        job = self._folder / "job.json"
        job.write_text(json.dumps({"repo": str(found.repo), "weights": str(found.weights), "track": True}))
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu)}
        runner = str(Path(__file__).with_name("pointmap_runner.py"))
        self._process = subprocess.Popen(
            ["nice", "-n", str(camera_clouds.NICE), str(found.python), runner, str(job)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self._answers: queue.Queue[dict[str, Any] | None] = queue.Queue()
        threading.Thread(target=self._read, name="camera-track-answers", daemon=True).start()
        self._answer(READY_S)

    def _read(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            try:
                self._answers.put(json.loads(line))
            except ValueError:
                continue  # not an answer: something the runtime printed
        self._answers.put(None)

    def _answer(self, within: float) -> dict[str, Any]:
        try:
            got = self._answers.get(timeout=within)
        except queue.Empty:
            raise RuntimeError(f"the tracker took over {within:.0f} s") from None
        if got is None:
            raise RuntimeError("the tracker stopped")
        return got

    def step(self, picture: np.ndarray, overlay: np.ndarray | None, reset: bool) -> Step:
        import cv2

        cv2.imwrite(str(self._folder / "frame.png"), picture)
        mask = None
        if overlay is not None and overlay.any():
            mask = str(self._folder / "mask.png")
            cv2.imwrite(mask, overlay.astype(np.uint8) * 255)
        out = self._folder / "points.npz"
        asked = {"image": str(self._folder / "frame.png"), "mask": mask, "out": str(out), "reset": reset}
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(json.dumps(asked) + "\n")
            self._process.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"the tracker stopped: {e}") from e
        got = self._answer(ANSWER_S)
        if not got.get("ok"):
            return Step(False)
        points = None
        if got.get("keyframe"):
            with np.load(out) as kept:
                points = camera_clouds.Pointmap(kept["points"], kept["confidence"], kept["colour"], kept["covered"])
        pose = np.asarray(got["pose"], dtype=np.float64).reshape(4, 4)
        return Step(True, pose, bool(got.get("started")), points)

    def close(self) -> None:
        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
            self._process.wait(timeout=10)
        except OSError, subprocess.TimeoutExpired:
            self._process.kill()
        shutil.rmtree(self._folder, ignore_errors=True)


def open_runner() -> Session | None:
    """A tracker in the configured runtime, or None while there is none or no GPU has the memory for it."""
    found = camera_clouds.runtime()
    gpu = camera_clouds.free_gpu()
    if found is None or gpu is None or gpu[1] < camera_clouds.NEED_GB:
        return None
    return Runner(found, gpu[0])


def _floor_pose(matrix: np.ndarray) -> tuple[cloud_scene.At, float]:
    """Where a camera stands on the floor of the scene and how high, from its pose there (4 x 4, a similarity)."""
    ahead = matrix[:3, 2]
    return cloud_scene.At(float(matrix[0, 3]), float(matrix[1, 3]), math.atan2(ahead[0], ahead[1])), float(matrix[2, 3])


def _lie(at: cloud_scene.At, scale: float) -> np.ndarray:
    """A levelled cloud's frame on the plan, for a camera standing at `at`, resized by `scale`."""
    return cloud_scene._transform(at) @ np.diag([scale, scale, scale, 1.0])


@dataclass
class Track:
    """One camera's track on the plan (the module docstring). `take` is given every frame's `Step`, in order."""

    store: camera_clouds.Store
    camera: str
    basis: str  # the owner's spot it is beside (`cloud_scene.SpotAt.basis`)
    rooms: Sequence[cloud_scene.Ring]  # the plan's rooms
    at: cloud_scene.At  # where the camera is now: to begin with, where it stood
    since: float
    height: float = 0.0
    seen: float = 0.0  # when it was last placed
    held: float | None = None  # when a keyframe was last held to the plan's walls
    missed: int = 0  # frames in a row the tracker couldn't match
    restarts: int = 0
    state: str = "tracking"
    pose: np.ndarray | None = None  # the camera in the scene now, 4 x 4 (rigid)
    path: list[tuple[float, float, float, float]] = field(default_factory=list)  # when, x, y, bearing
    keyframes: list[dict[str, Any]] = field(default_factory=list)
    _onto: np.ndarray | None = None  # the track's own frame on the plan, 4 x 4 (a similarity)
    _count: int = 0
    _written: float = 0.0

    def __post_init__(self) -> None:
        self.start = self.at
        self.seen = self.since
        shutil.rmtree(self.folder, ignore_errors=True)
        self.write(self.since, force=True)

    @property
    def folder(self) -> Path:
        return self.store.folder / camera_clouds.slug(self.camera) / cloud_scene.TRACK_FOLDER

    @property
    def begin(self) -> bool:
        """Whether the next frame must begin the track (again): nothing lies on the plan yet."""
        return self._onto is None

    def take(self, step: Step, now: float) -> None:
        if not step.ok or step.pose is None:
            self.missed += 1
            if self.missed >= LOST_FRAMES:
                self._onto, self.missed = None, 0  # lost: begin again from the next frame, where it last was
                self.restarts += 1
                LOG.info("%s: the track was lost; beginning again where it was last placed", self.camera)
            return
        self.missed = 0
        if step.started:
            self._onto = None
        if step.points is not None:
            self._keyframe(step, now)
        if self._onto is None:
            return  # no keyframe lies on the plan yet (its picture gave no cloud): the next frame begins again
        placed = self._onto @ step.pose
        scale = float(np.cbrt(abs(np.linalg.det(placed[:3, :3])))) or 1.0
        placed[:3, :3] /= scale
        self.at, self.height = _floor_pose(placed)
        self.pose, self.seen = placed, now
        self.path = [*self.path, (now, self.at.x, self.at.y, self.at.bearing)][-PATH:]
        self.write(now, force=step.points is not None)

    def _keyframe(self, step: Step, now: float) -> None:
        """Make a keyframe's cloud, put it on the plan, and hold it to the plan's walls."""
        assert step.pose is not None and step.points is not None
        try:
            made = camera_clouds.make(self.camera, step.points, None, now)
        except camera_clouds.Rejected as e:
            LOG.debug("%s: a keyframe gave no cloud (%s)", self.camera, e.reason)
            return
        level = np.asarray(made.cloud.pose, dtype=np.float64).reshape(4, 4)  # the keyframe's camera in its cloud
        rows = ply.vertices(made.ply)
        assert rows is not None
        points = np.c_[rows["x"], rows["y"], rows["z"]].astype(np.float64)
        walls = cloud_align.lines(cloud_align.planes(points))
        edges = cloud_align.edges_of(self.rooms)
        if self._onto is None:  # the first of a track: near where the camera stood, facing any way
            lie = _lie(self.at, 1.0)
            found = cloud_align.locate(walls, edges, self.rooms, points, (self.at.x, self.at.y), anywhere=False)
        else:
            lie = self._onto @ step.pose @ np.linalg.inv(level)
            where, _ = _floor_pose(lie @ level)
            found = cloud_align.hold(walls, where.x, where.y, where.bearing, edges, self.rooms, points)
        if found.state == "placed":
            lie = _lie(cloud_scene.At(found.x, found.y, found.bearing), found.scale)
            self.held = now
        # The track's frame lies on the plan so that this keyframe's camera is where its cloud was put.
        self._onto = lie @ level @ np.linalg.inv(step.pose)
        self._count += 1
        self.folder.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self._count)
        pick = rng.choice(len(points), min(len(points), THIN), replace=False)
        colours = np.c_[rows["red"], rows["green"], rows["blue"]].astype(np.uint8)
        places.write_whole(self.folder / f"{self._count}.ply", ply.dumps(points[pick], colours[pick]))
        row = {"n": self._count, "at": now, "transform": [float(v) for v in lie.ravel()], "points": len(pick),
               "held": found.state == "placed"}  # fmt: skip
        self.keyframes = [*self.keyframes, row][-KEEP:]
        for old in self.folder.glob("*.ply"):
            if old.stem not in {str(k["n"]) for k in self.keyframes}:
                old.unlink(missing_ok=True)
        LOG.debug("%s: keyframe %d %s", self.camera, self._count, "held to the walls" if row["held"] else "unheld")

    def rested(self, now: float) -> bool:
        """Whether the camera has come to rest: placed `REST_S` ago or more, and still within `REST` of it."""
        before = [row for row in self.path if row[0] <= now - REST_S]
        if not before or now - self.since < REST_S:
            return now - self.since >= 2 * REST_S and not self.path  # never placed at all: nothing to follow
        was = cloud_scene.At(*before[-1][1:])
        shift, turn = cloud_scene.moved_by(self.at, was)
        return shift <= REST[0] and turn <= REST[1]

    def write(self, now: float, force: bool = False) -> None:
        if not force and now - self._written < WRITE_EVERY_S:
            return
        self._written = now
        record = {
            "camera": self.camera,
            "basis": self.basis,
            "state": self.state,
            "since": self.since,
            "at": self.seen,
            "xyb": [self.at.x, self.at.y, self.at.bearing],
            "start": [self.start.x, self.start.y, self.start.bearing],
            "height": self.height,
            "pose": None if self.pose is None else [float(v) for v in self.pose.ravel()],
            "held": self.held,
            "keyframes": self.keyframes,
            "path": [[round(t, 2), round(x, 3), round(y, 3), round(b, 4)] for t, x, y, b in self.path],
            "restarts": self.restarts,
        }
        json_state.write(self.folder.parent / cloud_scene.TRACK, record)

    def end(self, now: float) -> None:
        self.state = "ended"
        self.write(now, force=True)


class Follows(Protocol):
    """What the surroundings service asks of a tracker (`Tracker`)."""

    def look(self, camera: str, pixels: np.ndarray, now: float) -> bool: ...

    def step(self, camera: str, pixels: np.ndarray, now: float) -> bool: ...

    def end(self, camera: str, now: float) -> None: ...


#: A camera's frame made ready for the model: flattened, with its overlays marked (spatial/lens.py `prepare`).
Prepare = Callable[[str, np.ndarray], tuple[np.ndarray, np.ndarray | None]]


class Tracker:
    """Decides when a camera is tracked, and runs its track (the module docstring). `look` takes the service's
    samples and `step` the frames of the camera being tracked; both may be called from threads, one call at a
    time each."""

    def __init__(
        self,
        state: Path,
        store: camera_clouds.Store,
        thumbnail: Callable[[np.ndarray], np.ndarray],
        lit: Callable[[np.ndarray], bool],
        detect: Callable[[np.ndarray], Sequence[Any]] | None,
        prepare: Prepare,
        open_session: Callable[[], Session | None] = open_runner,
    ) -> None:
        self.state, self.store = state, store
        self.thumbnail, self.lit, self.detect, self.prepare = thumbnail, lit, detect, prepare
        self.open_session = open_session
        self.active: str | None = None
        self._track: Track | None = None
        self._session: Session | None = None
        self._last: dict[str, np.ndarray] = {}
        self._not_before: dict[str, float] = {}
        self._lock = threading.Lock()
        for camera in store.cameras():  # a track left open by a service that stopped is over
            was = cloud_scene.tracked(store, camera)
            if was is not None and was.get("state") == "tracking":
                json_state.write(
                    store.folder / camera_clouds.slug(camera) / cloud_scene.TRACK, {**was, "state": "ended"}
                )

    def look(self, camera: str, pixels: np.ndarray, now: float) -> bool:
        """Take one of the service's samples of `camera`. Whether its tracking began with it."""
        thumb = self.thumbnail(pixels)
        before, self._last[camera] = self._last.get(camera), thumb
        if before is None or len(before) != len(thumb) or float(before @ thumb) >= MOVING:
            return False
        with self._lock:
            if self.active is not None or now < self._not_before.get(camera, 0.0) or not self.lit(pixels):
                return False
            if self.detect is not None and self.detect(pixels):
                return False  # someone in view changes a still camera's picture too
            spots = cloud_scene.spots_of(self.state)
            spot = spots.get(camera)
            if spot is None or spot.at is None:
                return False
            rooms = cloud_scene.plan_rooms(self.state, spot.place, spot.mirrored)
            stood = cloud_scene.standing(self.store, camera, spots, rooms)
            if stood is None:
                return False
            try:
                session = self.open_session()
            except (OSError, RuntimeError) as e:
                LOG.warning("%s: the tracker didn't start: %s", camera, e)
                session = None
            if session is None:
                self._not_before[camera] = now + COOLDOWN_S
                return False
            self._session = session
            self._track = Track(self.store, camera, spot.basis, rooms, stood, now)
            self.active = camera
            LOG.info("%s: its view is moving; tracking it", camera)
            return True

    def step(self, camera: str, pixels: np.ndarray, now: float) -> bool:
        """Take one frame of the camera being tracked. Whether its track is over."""
        with self._lock:
            track, session = self._track, self._session
            if camera != self.active or track is None or session is None:
                return True
            try:
                picture, overlay = self.prepare(camera, pixels)
                track.take(session.step(picture, overlay, track.begin), now)
            except (OSError, RuntimeError, ValueError) as e:
                LOG.warning("%s: tracking stopped: %s", camera, e)
                self._end(now)
                return True
            if track.rested(now) or now - track.since >= MAX_S or track.restarts > RESTARTS:
                why = "came to rest" if track.rested(now) else "was lost" if track.restarts > RESTARTS else "ran long"
                shift, turn = cloud_scene.moved_by(track.at, track.start)
                LOG.info("%s: the track ended (it %s): %.2f m and %.0f degrees from where it stood, %d keyframes",
                         camera, why, shift, turn, track._count)  # fmt: skip
                self._end(now)
                return True
            return False

    def _end(self, now: float) -> None:
        if self._session is not None:
            self._session.close()
        if self._track is not None:
            self._track.end(now)
            self._not_before[self._track.camera] = now + COOLDOWN_S
        self.active, self._track, self._session = None, None, None

    def end(self, camera: str, now: float) -> None:
        """Stop tracking `camera` now (its frames stopped coming, or the service is stopping)."""
        with self._lock:
            if camera == self.active:
                self._end(now)


def for_service(
    state: Path,
    detect: Callable[[np.ndarray], Sequence[Any]] | None,
    lit: Callable[[np.ndarray], bool],
    thumbnail: Callable[[np.ndarray], np.ndarray],
) -> Tracker:
    """The service's `Tracker`: the pack's store, the configured runtime, and the cameras' lenses."""
    from liberational.spatial import lens as lenses

    def prepare(camera: str, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        found, masks = lenses.for_camera(lenses.manifest(), camera)
        return lenses.prepare(pixels, found, masks)

    store = camera_clouds.Store(camera_clouds.root(state))
    return Tracker(state, store, thumbnail, lit, detect, prepare)
