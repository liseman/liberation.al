"""Tracking a camera while it moves: its pose over time on the plan, held to the plan's walls (#516).

spatial/cloud_scene.py places a camera when it gets a cloud, which takes a view that has held still. Between two
such views a moving camera has no pose. Here it gets one, a few times a second, from its live frames.

**The tracker** is MASt3R-SLAM's own frame tracker (spatial/pointmap_runner.py, in its runtime, as a process that
stays up for the length of a track): each frame is matched to the current keyframe and its pose solved against that
keyframe's points, and a frame that shares too little with the keyframe becomes the next one. It says where each
frame's camera is in the track's own frame (the first keyframe's camera), as a similarity. Nothing else of
MASt3R-SLAM runs: no loop closing, and no relocalising but the look through the track's last keyframes.

**On the plan** (`Track`). A track begins on the picture the camera's resting cloud was made from: that picture
is sent to the tracker as its first keyframe, and its camera's place in the scene is known (`cloud_scene.resting`),
so the track stands on the plan from its first live frame, whichever way the camera has turned by the time the
move is noticed and whether or not a wall is in view. (The first live test showed why: begun on a live frame and
placed "where the camera stood", a track whose first keyframe no wall held was wrong from the start, none of its
26 keyframes after that could be held either, and nothing reached the map.) Each keyframe gives a cloud, made as
any camera's is (spatial/camera_clouds.py `make`: cleaned, its floor found, levelled). It is put where the track
says it is, then held to the plan's walls from there (`cloud_align.hold`: squared and fitted within the bounds any
cloud is). When the walls take it, the track's lie on the plan is set again from that keyframe: what the track had
drifted is gone, the camera's height and tilt come from the keyframe's own floor, and its scale from the plan's
walls. When they don't (no wall in view, or nothing that fits), the keyframe stays where the track put it and the
track says it is running unheld since the last one that was. Frames between keyframes are placed through the same
lie.

**Into the map, as it goes.** Every keyframe is offered to the place's map (spatial/cloud_map.py) as it is made.
One the walls hold is offered as a camera's own fitted cloud is. One they don't is offered where the track puts it
(`TRACKED`): the map adds what it saw in space the map has nothing in, leaves out its points that better-placed
shots looked straight through, moves nothing of its own for it, and refuses it when as a whole it stands somewhere
else. So the map fills in while the camera moves, and the track's record says how many points this move has added
and when the last went in.

**Lost.** A frame that doesn't match the current keyframe is looked for by the tracker among the track's last
keyframes (a camera that swings back, or comes out of a blurred stretch, is found again in the same frame, and the
track goes on). One that matches none is skipped. After `LOST_S` of those the track is begun again from the next
frame: where that is, its walls on the plan's say (near where the camera was last placed, facing any way); failing
that, its surfaces on those the map holds there (spatial/cloud_merge.py `agree`); failing both it stands where the
camera was last placed and its keyframes are not offered until one is held. After `RESTARTS` of those the track
ends.

**When it starts and stops** (`Tracker`). The surroundings service samples each chosen camera every few seconds.
A camera starts being tracked when the edges of two samples in a row are less than `MOVED` alike (`structure`) and
nobody is in either: a lamp, the daylight, or the exposure changes a picture without moving its edges, and someone
coming into or leaving the picture moves them without the camera moving. (Three of the first four live tracks
went nowhere: the camera faced a desk, and each began when the person at it left or came back.) It must also have
a place in the scene, the runtime set up, and a GPU with the memory. One camera is tracked at a time. It ends when
the camera has rested: its pose within `REST` of where it was `REST_S` ago. It also ends after `MAX_S`, and is then
left alone for `COOLDOWN_S`. A track that ends having gone nowhere is forgotten there and then. Otherwise the
camera is asked for a cloud of where it now rests (`rested`), whether or not its view looks changed; when its view
has held still long enough for one, that cloud is placed from where the track left the camera
(`cloud_scene.align`), the track goes, and it is an ordinary camera again.

**What it costs**, measured on the Linux machine beside the other services: about 12 s to load the model at the
start of a track, then 0.16 s of one GPU for a frame (the median; a keyframe takes up to 0.9 s), four frames a
second: about two thirds of that GPU's time while the camera moves. 2.9 GB of GPU memory, given back when the
track ends; nothing is held while no camera moves.
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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from liberational import json_state, logs
from liberational.spatial import camera_clouds, cloud_align, cloud_map, cloud_merge, cloud_scene, ply

LOG = logs.get("camera-track")

TRACK_EVERY_S = 0.25  # a frame this often while a camera moves
MOVED = 0.5  # two samples in a row whose edges (`structure`) are under this alike: the camera moved
TRACKED = 0.3  # how a keyframe the track places counts against one the plan's walls hold (`cloud_map.Offer.score`)
REST = (0.05, 2.0)  # metres and degrees: a camera within this of where it was `REST_S` ago has come to rest
REST_S = 10.0
MAX_S = 600.0  # a track is ended after this long, moving or not
COOLDOWN_S = 60.0  # and a camera whose track ended is not tracked again for this long
LOST_S = 2.0  # how long the tracker may match no keyframe before the track is begun again
RESTARTS = 5  # times a track is begun again before it ends
KEEP = 12  # keyframes the track's record tells of
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
    anchor: bool = False  # the frame was the camera's resting picture, sent to begin the track on its cloud's place
    recovered: bool = False  # lock was lost and found again among the track's earlier keyframes


class Session(Protocol):
    """A tracker that takes a camera's frames in order."""

    def step(self, picture: np.ndarray, overlay: np.ndarray | None, reset: bool, anchor: bool = False) -> Step: ...

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
            stderr=open(self._folder / "errors.log", "w"),  # noqa: SIM115 (the process's, for as long as it runs)
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

    def _said(self) -> str:
        """The end of what the runner wrote to its standard error: why it stopped, when it did."""
        try:
            lines = (self._folder / "errors.log").read_text(errors="replace").strip().splitlines()
        except OSError:
            return ""
        return " | ".join(line.strip() for line in lines[-3:])[-300:]

    def _answer(self, within: float) -> dict[str, Any]:
        try:
            got = self._answers.get(timeout=within)
        except queue.Empty:
            raise RuntimeError(f"the tracker took over {within:.0f} s") from None
        if got is None:
            raise RuntimeError(f"the tracker stopped: {self._said()}")
        return got

    def step(self, picture: np.ndarray, overlay: np.ndarray | None, reset: bool, anchor: bool = False) -> Step:
        import cv2

        cv2.imwrite(str(self._folder / "frame.png"), picture)
        mask = None
        if overlay is not None and overlay.any():
            mask = str(self._folder / "mask.png")
            cv2.imwrite(mask, overlay.astype(np.uint8) * 255)
        out = self._folder / "points.npz"
        asked = {"image": str(self._folder / "frame.png"), "mask": mask, "out": str(out), "reset": reset,
                 "anchor": anchor}  # fmt: skip
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(json.dumps(asked) + "\n")
            self._process.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"the tracker stopped: {e}: {self._said()}") from e
        got = self._answer(ANSWER_S)
        if not got.get("ok"):
            return Step(False)
        points = None
        if got.get("keyframe"):
            with np.load(out) as kept:
                points = camera_clouds.Pointmap(kept["points"], kept["confidence"], kept["colour"], kept["covered"])
        pose = np.asarray(got["pose"], dtype=np.float64).reshape(4, 4)
        return Step(True, pose, bool(got.get("started")), points, anchor, bool(got.get("recovered")))

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


def _keyframe_cloud(cloud: camera_clouds.Cloud, n: int, since: float) -> camera_clouds.Cloud:
    """A keyframe's cloud under a name of its own: its track's start and its number, so the map knows the shot."""
    return replace(cloud, id=f"track{since:.0f}.{n}")


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
    place: str
    basis: str  # the owner's spot it is beside (`cloud_scene.SpotAt.basis`)
    rooms: Sequence[cloud_scene.Ring]  # the plan's rooms
    at: cloud_scene.At  # where the camera is now: to begin with, where it stood
    since: float
    #: The camera in the scene as its resting cloud has it (4 x 4, a similarity carrying the cloud's size), when the
    #: track begins on the picture that cloud was made from; None when it begins on a live frame.
    rest: np.ndarray | None = None
    height: float = 0.0
    seen: float = 0.0  # when it was last placed
    held: float | None = None  # when a keyframe was last held to the plan's walls
    lost: float | None = None  # since when the tracker hasn't matched a frame
    restarts: int = 0
    recovered: int = 0  # times lock was lost and found again among the track's own keyframes
    added: int = 0  # points this move has put into the place's map
    added_at: float | None = None  # and when the last went in
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
        self.write(self.since, force=True)

    @property
    def begin(self) -> bool:
        """Whether the next frame must begin the track (again): nothing lies on the plan yet."""
        return self._onto is None

    @property
    def anchored(self) -> bool:
        """Whether the track is to begin on the camera's resting picture, and hasn't yet."""
        return self.rest is not None and self._onto is None and self._count == 0 and self.restarts == 0

    def take(self, step: Step, now: float) -> None:
        if step.anchor:
            if step.ok and self.rest is not None:
                self._onto = self.rest  # the resting picture is the track's first keyframe: its camera is known
            else:
                self.restarts += 1  # it gave nothing to stand on: the track begins on a live frame instead
            return
        if not step.ok or step.pose is None:
            self.lost = now if self.lost is None else self.lost
            if now - self.lost >= LOST_S:
                # Found among none of the track's keyframes for this long: begin again from the next frame. Where
                # that is, its walls, the map, and failing both where the camera was last placed will say.
                self._onto, self.lost = None, None
                self.restarts += 1
                LOG.info(
                    "%s: the track was lost for %.0f s; beginning again from what it sees now", self.camera, LOST_S
                )
            return
        self.lost = None
        self.recovered += step.recovered
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
        """Make a keyframe's cloud, put it on the plan, and offer it to the place's map (the module docstring's On
        the plan and In the scene)."""
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
        colours = np.c_[rows["red"], rows["green"], rows["blue"]].astype(np.uint8)
        planes = cloud_align.planes(points)
        walls = cloud_align.lines(planes)
        edges = cloud_align.edges_of(self.rooms)
        charted = cloud_map.folder(self.store.folder, self.place)
        tracked = self._onto is not None
        if self._onto is None:
            # A new start, with nothing to say where this is but where the camera was last placed: its walls on
            # the plan's, facing any way; failing that, its surfaces on those the map holds there.
            lie = _lie(self.at, 1.0)
            found = cloud_align.locate(walls, edges, self.rooms, points, (self.at.x, self.at.y), anywhere=False)
            if found.state != "placed":
                here = cloud_merge.Stand(self.at.x, self.at.y, self.at.bearing)
                agreed = cloud_merge.agree(cloud_merge.surfaces(planes), here, cloud_map.surfaces(charted))
                if agreed is not None and agreed.at is not None:
                    lie = _lie(cloud_scene.At(agreed.at.x, agreed.at.y, agreed.at.bearing), agreed.at.scale)
                    tracked = True  # placed by the map: offered where that put it
        else:
            lie = self._onto @ step.pose @ np.linalg.inv(level)
            where, _ = _floor_pose(lie @ level)
            found = cloud_align.hold(walls, where.x, where.y, where.bearing, edges, self.rooms, points)
        held = found.state == "placed"
        if held:
            lie = _lie(cloud_scene.At(found.x, found.y, found.bearing), found.scale)
            self.held = now
        # The track's frame lies on the plan so that this keyframe's camera is where its cloud was put.
        self._onto = lie @ level @ np.linalg.inv(step.pose)
        self._count += 1
        cloud = _keyframe_cloud(made.cloud, self._count, self.since)
        name = f"{self.camera}@track{self.since:.0f}.{self._count}"
        shot = None
        if held:  # the plan's walls hold it: offered as a camera's own fitted cloud is
            stand = cloud_merge.Stand(found.x, found.y, found.bearing, found.scale)
            flat = cloud_align.fit(walls, found.x, found.y, found.bearing, edges)  # to lay its walls on the plan's
            shot = cloud_scene.shot_of(
                name, cloud, stand, found.scale, True, found.score, points, colours, planes, flat
            )
        elif tracked:  # where the track puts it: added where it agrees with what the map holds there
            where, _ = _floor_pose(lie @ level)
            size = float(np.linalg.norm(lie[:3, 0]))
            stand = cloud_merge.Stand(where.x, where.y, where.bearing, size)
            shot = cloud_scene.shot_of(name, cloud, stand, size, False, TRACKED, points, colours, planes, None, lie)
        added = 0
        if shot is not None:
            try:
                added = cloud_map.offer(charted, self.rooms, shot).points
            except (OSError, ValueError) as e:
                LOG.warning("%s: a keyframe wasn't offered to the map: %s", self.camera, e)
        if added:
            self.added, self.added_at = self.added + added, now
        row = {"n": self._count, "at": now, "held": held, "added": added}
        self.keyframes = [*self.keyframes, row][-KEEP:]

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
            "added": self.added,
            "added_at": self.added_at,
            "keyframes": self.keyframes,
            "path": [[round(t, 2), round(x, 3), round(y, 3), round(b, 4)] for t, x, y, b in self.path],
            "restarts": self.restarts,
            "recovered": self.recovered,
        }
        json_state.write(self.store.folder / camera_clouds.slug(self.camera) / cloud_scene.TRACK, record)

    def end(self, now: float) -> None:
        self.state = "ended"
        self.write(now, force=True)


def structure(pixels: np.ndarray) -> np.ndarray:
    """Where a picture's edges are, as a small map (64 x 36, zero mean, unit length), so two pictures' likeness is
    a dot product that says whether the camera moved. Each edge is taken against the edges around it, so a lamp
    coming on, the room dimming, or the exposure stepping leaves it as it was (0.96 to 1.00 alike on a live picture
    relit three ways), someone leaving a third of the picture leaves it 0.80 alike, and a camera turned 2 degrees
    makes it another map (0.10). The thumbnail the sampler compares (spatial/surroundings_worker.py) can't tell
    those apart: a lamp's pool of light read 0.83 and a 2 degree turn 0.87."""
    import cv2

    grey = cv2.cvtColor(np.asarray(pixels, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    grey = cv2.GaussianBlur(grey, (0, 0), 2 * grey.shape[1] / 960)
    strength = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
    around = cv2.GaussianBlur(strength, (0, 0), 25 * grey.shape[1] / 960)
    small = cv2.resize(strength / (around + 1.0), (64, 36), interpolation=cv2.INTER_AREA).ravel()
    small -= small.mean()
    return small / max(float(np.linalg.norm(small)), 1e-6)


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
        lit: Callable[[np.ndarray], bool],
        detect: Callable[[np.ndarray], Sequence[Any]] | None,
        prepare: Prepare,
        open_session: Callable[[], Session | None] = open_runner,
        rested: Callable[[str], None] | None = None,
        structure: Callable[[np.ndarray], np.ndarray] = structure,
    ) -> None:
        self.structure = structure
        self._rest: Callable[[], np.ndarray | None] | None = None  # reads the tracked camera's resting picture
        self.rested = rested  # told a camera whose track ended somewhere else: it wants a cloud of where it rests
        self.state, self.store = state, store
        self.lit, self.detect, self.prepare = lit, detect, prepare
        self.open_session = open_session
        self.active: str | None = None
        self._track: Track | None = None
        self._session: Session | None = None
        self._last: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # each camera's last sample: its edges, and it
        self._not_before: dict[str, float] = {}
        self._lock = threading.Lock()
        for camera in store.cameras():
            was, cloud = cloud_scene.tracked(store, camera), store.current(camera)
            if was is None:
                continue
            if was.get("state") == "tracking":  # a track left open by a service that stopped is over
                was = {**was, "state": "ended"}
                json_state.write(store.folder / camera_clouds.slug(camera) / cloud_scene.TRACK, was)
            # One that ended after the camera's cloud was made left it somewhere its cloud doesn't show.
            if rested is not None and cloud is not None and float(was.get("at") or 0.0) > cloud.at:
                rested(camera)

    def look(self, camera: str, pixels: np.ndarray, now: float) -> bool:
        """Take one of the service's samples of `camera`. Whether its tracking began with it."""
        edges = self.structure(pixels)
        before, self._last[camera] = self._last.get(camera), (edges, pixels)
        if before is None or float(before[0] @ edges) >= MOVED:
            return False
        with self._lock:
            if self.active is not None or now < self._not_before.get(camera, 0.0) or not self.lit(pixels):
                return False
            # Someone in the picture, now or in the sample before, moves its edges too.
            if self.detect is not None and (self.detect(pixels) or self.detect(before[1])):
                return False
            spots = cloud_scene.spots_of(self.state)
            spot = spots.get(camera)
            if spot is None or spot.at is None:
                return False
            rooms = cloud_scene.plan_rooms(self.state, spot.place, spot.mirrored)
            rest = cloud_scene.resting(self.store, camera, spots, rooms)
            if rest is None:
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
            self._track = Track(self.store, camera, spot.place, spot.basis, rooms, rest[0], now, rest[1])
            self._rest = rest[2]
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
                if track.anchored:  # first, the picture its resting cloud was made from: the track stands on it
                    picture = self._rest() if self._rest is not None else None
                    track.take(
                        Step(False, anchor=True) if picture is None else session.step(picture, None, True, True), now
                    )
                picture, overlay = self.prepare(camera, pixels)
                track.take(session.step(picture, overlay, track.begin), now)
            except (OSError, RuntimeError, ValueError) as e:
                LOG.warning("%s: tracking stopped: %s", camera, e)
                self._end(now)
                return True
            if track.rested(now) or now - track.since >= MAX_S or track.restarts > RESTARTS:
                why = "came to rest" if track.rested(now) else "was lost" if track.restarts > RESTARTS else "ran long"
                shift, turn = cloud_scene.moved_by(track.at, track.start)
                LOG.info("%s: the track ended (it %s): %.2f m and %.0f degrees from where it stood, %d keyframes, "
                         "%d points added to the map, lock found again %d times, begun again %d", camera, why, shift,
                         turn, track._count, track.added, track.recovered, track.restarts)  # fmt: skip
                self._end(now)
                return True
            return False

    def _end(self, now: float) -> None:
        if self._session is not None:
            self._session.close()
        if self._track is not None:
            track = self._track
            track.end(now)
            self._not_before[track.camera] = now + COOLDOWN_S
            shift, turn = cloud_scene.moved_by(track.at, track.start)
            if shift <= REST[0] and turn <= REST[1]:
                cloud_scene.drop_track(self.store, track.camera)  # it went nowhere: its cloud still shows its view
            elif self.rested is not None:
                self.rested(track.camera)
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
    rested: Callable[[str], None] | None = None,
) -> Tracker:
    """The service's `Tracker`: the pack's store, the configured runtime, and the cameras' lenses. `rested` is
    told a camera that came to rest somewhere else (spatial/surroundings_clouds.py `Clouds.remake`)."""
    from liberational.spatial import lens as lenses

    def prepare(camera: str, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        found, masks = lenses.for_camera(lenses.manifest(), camera)
        return lenses.prepare(pixels, found, masks)

    store = camera_clouds.Store(camera_clouds.root(state))
    return Tracker(state, store, lit, detect, prepare, rested=rested)
