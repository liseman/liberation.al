"""Each camera's own point cloud: what one camera sees, as coloured points in 3D (#417).

A camera that stays put can't measure a room: nothing it sees shifts against anything else. What it can give is its
own view with depth, and that is what this is: one camera, one cloud, made from one clean picture of that camera and
shown by itself. Clouds of different cameras are never joined here; only a camera that moves can relate two views.

**Making one** (`make`). A picture goes through MASt3R-SLAM's single-frame step in its own runtime
(spatial/pointmap_runner.py, as a subprocess: `Runtime`, `pointmap`), which gives every pixel a point in the
camera's own frame and a confidence. Then, in the pack:

1. *Which pixels count.* A point is kept when the model is sure of it (`CONF`, MASt3R's own bar) and its pixel isn't
   one the camera burns its clock or name into (the device manifest's `masks`, spatial/lens.py). Bright windows,
   glass, and sky usually fall out here.
2. *Accept or reject.* A picture the model is unsure about over most of it gives no cloud (`MIN_SHARE` of its pixels
   must count: the six cameras on the Linux machine kept 0.60 to 0.88), nor does one with no depth to speak of
   (`MIN_DEPTH_SPREAD`). The reason is kept (`Store.note`) and the camera's earlier cloud stays.
3. *The floor* (`floor`). The plane with the most points on it among those that face roughly up in the picture, pass
   under the camera, and have next to nothing under them: the floor is the lowest big surface, which rules out a
   table top. Found on five of those six cameras; the sixth sees almost no floor.
4. *Size* (`Cloud.size`). MASt3R's metres are approximate: on three cameras with a measured mounting height they
   read 17 to 25 percent off, either way. A camera with a height in the device manifest (`mount.height_m`) and a
   floor in view is sized by it: the cloud is scaled so the camera stands that high above its floor ("height").
   Any other cloud stays as the model gave it ("approximate"), and the page says which.
5. *Its frame* (`level`). The cloud is turned level: z is up, the floor is z = 0, the camera stands on the z axis
   looking along +y. With no floor found the camera is taken to be upright and the lowest points stand in for the
   floor, so the view still has a ground to stand on.

**Keeping one** (`Store`): `state/camera_clouds/<camera>/` holds the camera's one cloud, `<id>.ply` (binary, x y z
and red green blue) and `cloud.json` (`Cloud`: when, how many points, how it was sized, the camera's pose and focal
in the cloud's frame, the floor, and the thumbnail of the picture it was made from, which is how a changed view is
noticed). A new cloud replaces the old one: nothing is carried forward. `notes.json` beside the folders says, per
camera, why its last try made no cloud. None of this is a place's revision: a revision's cameras are registrations
the position placer reads, and a camera's own cloud registers nothing.

Cost, measured beside the live services: 7.5 s to load the model, 0.2 s and 2.9 GB of GPU memory a picture, about
0.5 s of CPU for the floor. A pass is skipped, with the reason, when no GPU has `NEED_GB` free.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import json_state, logs
from liberational.memory import places
from liberational.spatial import ply

LOG = logs.get("camera-clouds")

DIR = "camera_clouds"  # under paths.state
RUNTIME = "spatial.pointmap_runtime"
WEIGHTS = "spatial.pointmap_weights"
NEED_GB = 4.0  # free GPU memory a pass needs: 2.9 GB measured, with room for the driver's own
NICE = 10
TIMEOUT_S = 180.0
CONF = 1.5  # MASt3R's confidence bar for a usable point
MIN_SHARE = 0.3  # of a picture's pixels that must give a point; live pictures gave 0.60 to 0.88
MIN_DEPTH_SPREAD = 1.15  # the far tenth of the points must be this many times as far as the near tenth
FLOOR_TOL_M = 0.025
FLOOR_TRIES = 1500
FLOOR_SAMPLE = 40000
FLOOR_UP = 0.5  # cosine: a floor's normal is within 60 degrees of the camera's own up
FLOOR_UNDER = (0.06, 0.02)  # at most this share of points may lie more than this far (m) under the floor
FLOOR_MIN = 0.04  # of the sampled points must be on it
FLOOR_BELOW_M = 0.2  # the floor passes at least this far under the camera
#: Why a camera has no cloud, or no newer one (`Store.note`).
REASONS = ("no_runtime", "dark", "people", "lens", "gpu", "failed", "unsure", "flat")


@dataclass(frozen=True)
class Runtime:
    """Where MASt3R-SLAM is set up: its environment's python, its checkout, and MASt3R's weights."""

    python: Path
    repo: Path
    weights: Path


def runtime() -> Runtime | None:
    """The runtime `spatial.pointmap_runtime` names (a folder with `venv/` and `src/`), or None while the setting is
    empty or what it names isn't there: then no cloud is made, and the page says so."""
    folder = str(cfg.get(RUNTIME, "") or "").strip()
    if not folder:
        return None
    base = cfg.owner_file(folder)
    found = Runtime(base / "venv" / "bin" / "python", base / "src", cfg.owner_file(str(cfg.get(WEIGHTS))))
    if not found.python.is_file() or not found.repo.is_dir() or not found.weights.is_dir():
        return None
    return found


def free_gpu() -> tuple[int, float] | None:
    """The GPU with the most free memory, as (its index in bus order, GB free), or None without `nvidia-smi`."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
            env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        ).stdout
    except OSError, subprocess.SubprocessError:
        return None
    best: tuple[int, float] | None = None
    for line in out.splitlines():
        try:
            index, free = (part.strip() for part in line.split(","))
            row = (int(index), float(free) / 1024)
        except ValueError:
            continue
        if best is None or row[1] > best[1]:
            best = row
    return best


@dataclass(frozen=True)
class Pointmap:
    """What the runner gives for one picture, in the camera's own frame (x right, y down, z forward)."""

    points: np.ndarray  # H x W x 3
    confidence: np.ndarray  # H x W
    colour: np.ndarray  # H x W x 3 uint8
    covered: np.ndarray  # H x W bool: overlay pixels
    seconds: float = 0.0
    peak_gpu_gb: float = 0.0


class NoGpu(RuntimeError):
    """No GPU has the memory a pass needs right now."""


def pointmap(found: Runtime, picture: np.ndarray, overlay: np.ndarray | None) -> Pointmap:
    """`picture` (BGR) through the runner in `found`, niced, on the GPU with the most free memory. Raises NoGpu when
    none has NEED_GB free, and RuntimeError when the runner fails."""
    import cv2

    gpu = free_gpu()
    if gpu is None or gpu[1] < NEED_GB:
        raise NoGpu("no GPU found" if gpu is None else f"{gpu[1]:.1f} GB free, {NEED_GB:.0f} needed")
    with tempfile.TemporaryDirectory(prefix="pointmap-") as tmp:
        folder = Path(tmp)
        cv2.imwrite(str(folder / "picture.png"), picture)
        mask = None
        if overlay is not None and overlay.any():
            mask = str(folder / "mask.png")
            cv2.imwrite(mask, overlay.astype(np.uint8) * 255)
        job = {
            "repo": str(found.repo),
            "weights": str(found.weights),
            "out": str(folder / "report.json"),
            "views": [{"image": str(folder / "picture.png"), "mask": mask, "out": str(folder / "points.npz")}],
        }
        (folder / "job.json").write_text(json.dumps(job))
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu[0])}
        try:
            subprocess.run(
                [str(found.python), str(Path(__file__).with_name("pointmap_runner.py")), str(folder / "job.json")],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                check=True,
                env=env,
                preexec_fn=lambda: os.nice(NICE),
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"the point map runner failed: {(e.stderr or '').strip()[-400:]}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"the point map runner took over {TIMEOUT_S:.0f} s") from e
        report = json.loads((folder / "report.json").read_text())["views"][0]
        with np.load(folder / "points.npz") as got:
            return Pointmap(
                got["points"],
                got["confidence"],
                got["colour"],
                got["covered"],
                report["seconds"],
                report["peak_gpu_gb"],
            )


def floor(points: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float] | None:
    """The floor in a camera-frame cloud (N x 3; y is down): its unit normal, pointing up into the room, and how far
    the camera (the origin) stands above it, in the cloud's own metres. None when no plane qualifies (the module
    docstring says what does). The same cloud gives the same floor: the search is seeded."""
    rng = np.random.default_rng(seed)
    if len(points) < 500:
        return None
    sample = points[rng.choice(len(points), min(len(points), FLOOR_SAMPLE), replace=False)].astype(np.float64)
    up = np.array([0.0, -1.0, 0.0])
    best: tuple[np.ndarray, int] | None = None
    for _ in range(FLOOR_TRIES):
        p = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            continue
        n = n / norm
        if n @ up < 0:
            n = -n
        if n @ up < FLOOR_UP:
            continue
        above = sample @ n - p[0] @ n  # each point's height above the plane
        if -(p[0] @ n) < FLOOR_BELOW_M or float((above < -FLOOR_UNDER[0]).mean()) > FLOOR_UNDER[1]:
            continue
        count = int((np.abs(above) < FLOOR_TOL_M).sum())
        if best is None or count > best[1]:
            best = (np.append(n, -(p[0] @ n)), count)
    if best is None or best[1] < FLOOR_MIN * len(sample):
        return None
    n, height = best[0][:3], float(best[0][3])
    on = sample[np.abs(sample @ n + height) < FLOOR_TOL_M]
    # Refit to the points on it: three points' plane is only as good as those three.
    centre = on.mean(axis=0)
    _, _, vt = np.linalg.svd(on - centre, full_matrices=False)
    fitted = vt[2] if vt[2] @ up > 0 else -vt[2]
    return fitted, float(-(centre @ fitted))


def level(points: np.ndarray, found: tuple[np.ndarray, float] | None) -> tuple[np.ndarray, np.ndarray, bool]:
    """A camera-frame cloud turned level (z up, the floor at z = 0, the camera on the z axis looking along +y), the
    camera's pose in that frame (camera to cloud, 4 x 4), and whether the floor was found. Without one the camera
    is taken to be upright, and the lowest points stand in for the floor."""
    if found is not None:
        up, height = found
    else:
        up = np.array([0.0, -1.0, 0.0])
        height = float(-np.percentile(points @ up, 2)) if len(points) else 0.0
    ahead = np.array([0.0, 0.0, 1.0]) - up[2] * up  # the camera's forward, laid on the floor
    if np.linalg.norm(ahead) < 1e-6:  # looking straight down or up: any direction across the floor will do
        ahead = np.array([1.0, 0.0, 0.0]) - up[0] * up
    y = ahead / np.linalg.norm(ahead)
    x = np.cross(y, up)
    turn = np.stack([x, y, up])
    pose = np.eye(4)
    pose[:3, :3] = turn
    pose[2, 3] = height
    return points @ turn.T + np.array([0.0, 0.0, height]), pose, found is not None


@dataclass(frozen=True)
class Cloud:
    """One camera's cloud as `cloud.json` keeps it. The frame is `level`'s: metres, z up, the floor at z = 0."""

    id: str
    camera: str
    at: float  # when it was made (UTC epoch seconds)
    points: int
    share: float  # of the picture's pixels that gave a point
    size: str  # "height": scaled by the camera's measured mounting height; "approximate": the model's own metres
    factor: float  # what the model's metres were multiplied by (1 when approximate)
    floor_found: bool
    pose: tuple[float, ...]  # camera to cloud, 4 x 4 row by row
    focal: float  # of the picture the model saw, in its pixels
    width: int
    height: int
    seconds: float = 0.0
    peak_gpu_gb: float = 0.0
    frames: int = 1  # the frames whose median the picture was
    thumb: tuple[float, ...] = field(default=(), repr=False)  # the picture's thumbnail, to notice a changed view

    @staticmethod
    def from_dict(value: dict[str, Any]) -> Cloud:
        known = {k: value[k] for k in Cloud.__dataclass_fields__ if k in value}
        known["pose"] = tuple(float(v) for v in known["pose"])
        known["thumb"] = tuple(float(v) for v in known.get("thumb", ()))
        return Cloud(**known)


@dataclass(frozen=True)
class Made:
    """A cloud ready to keep: its record and its PLY."""

    cloud: Cloud
    ply: bytes


class Rejected(ValueError):
    """A picture that gives no cloud; `reason` is one of REASONS."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


def _focal(points: np.ndarray, keep: np.ndarray) -> float:
    """The focal (pixels) a point map implies, from where its points fall across the picture."""
    h, w = keep.shape
    xx = np.arange(w)[None, :] - w / 2
    wide = keep & (np.abs(xx) > w / 8) & (np.abs(points[..., 0]) > 1e-3)
    if not wide.any():
        return float(w)
    found = float(np.median((np.broadcast_to(xx, (h, w)) * points[..., 2] / np.where(wide, points[..., 0], 1.0))[wide]))
    return found if np.isfinite(found) and found > 0 else float(w)


def make(
    camera: str, got: Pointmap, height_m: float | None, now: float, frames: int = 1, thumb: Sequence[float] = ()
) -> Made:
    """`camera`'s cloud from its picture's point map (the module docstring's steps 1 to 5). Raises Rejected."""
    points = np.asarray(got.points, dtype=np.float64)
    keep = (got.confidence > CONF) & ~got.covered & np.isfinite(points).all(axis=2) & (points[..., 2] > 0)
    share = float(keep.mean())
    if share < MIN_SHARE:
        raise Rejected("unsure", f"only {share:.0%} of the picture gave a point; {MIN_SHARE:.0%} needed")
    kept = points[keep]
    near, far = np.percentile(kept[:, 2], [10, 90])
    if far < MIN_DEPTH_SPREAD * near:
        raise Rejected("flat", f"no depth to speak of: {near:.2f} to {far:.2f} m")
    found = floor(kept)
    factor = 1.0
    if found is not None and height_m is not None and height_m > 0:
        factor = height_m / found[1]
    sized = "height" if factor != 1.0 else "approximate"
    scaled = kept * factor
    levelled, pose, floor_found = level(scaled, None if found is None else (found[0], found[1] * factor))
    h, w = keep.shape
    cloud = Cloud(
        id=str(uuid.uuid7()),
        camera=camera,
        at=now,
        points=len(levelled),
        share=round(share, 3),
        size=sized,
        factor=round(factor, 4),
        floor_found=floor_found,
        pose=tuple(float(v) for v in pose.ravel()),
        focal=_focal(points, keep),
        width=w,
        height=h,
        seconds=got.seconds,
        peak_gpu_gb=got.peak_gpu_gb,
        frames=frames,
        thumb=tuple(float(v) for v in thumb),
    )
    return Made(cloud, ply.dumps(levelled, got.colour[keep]))


def root(state: Path | str) -> Path:
    return Path(state) / DIR


def _slug(camera: str) -> str:
    """A camera's folder name: its own name where that is a plain file name, else that with the rest replaced."""
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", camera).strip(".")
    if not slug:
        raise ValueError(f"{camera!r} can't name a camera's folder")
    return slug


class Store:
    """The cameras' clouds under `folder` (`root`): one a camera, replaced whole."""

    NOTES = "notes.json"

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def current(self, camera: str) -> Cloud | None:
        """`camera`'s cloud, or None: it has none, or its record doesn't read."""
        try:
            value = json.loads((self.folder / _slug(camera) / "cloud.json").read_text())
            cloud = Cloud.from_dict(value)
        except OSError, ValueError, KeyError, TypeError:
            return None
        return cloud if cloud.camera == camera and self.file(cloud).is_file() else None

    def cameras(self) -> list[str]:
        """The cameras that have a cloud, in name order."""
        found = []
        for record in sorted(self.folder.glob("*/cloud.json")):
            name = json_state.read(record, {})
            name = name.get("camera") if isinstance(name, dict) else None
            if isinstance(name, str) and self.current(name) is not None:
                found.append(name)
        return sorted(found)

    def file(self, cloud: Cloud) -> Path:
        if not re.fullmatch(r"[0-9a-f-]{36}", cloud.id):
            raise ValueError("a cloud's id is a uuid")
        return self.folder / _slug(cloud.camera) / f"{cloud.id}.ply"

    def replace(self, made: Made) -> None:
        """Keep `made` as its camera's cloud. Its file is written first and its record after, so a reader never finds
        a record whose points aren't there; the cloud before it goes last."""
        folder = self.folder / _slug(made.cloud.camera)
        folder.mkdir(parents=True, exist_ok=True)
        places.write_whole(self.file(made.cloud), made.ply)
        json_state.write(folder / "cloud.json", asdict(made.cloud))
        for old in folder.glob("*.ply"):
            if old.name != f"{made.cloud.id}.ply":
                old.unlink(missing_ok=True)
        self.note(made.cloud.camera, None, made.cloud.at)

    def note(self, camera: str, reason: str | None, at: float, detail: str = "") -> None:
        """Say why `camera`'s last try made no cloud (one of REASONS), or with None that it made one."""
        if reason is not None and reason not in REASONS:
            raise ValueError(f"a note's reason is one of {', '.join(REASONS)}")
        notes = self.notes()
        was = notes.get(camera)
        if reason is None:
            if was is None:
                return
            del notes[camera]
        else:
            if was is not None and was.get("reason") == reason:
                return  # the same reason again: when it was first so stays
            notes[camera] = {"reason": reason, "at": at, "detail": detail}
        self.folder.mkdir(parents=True, exist_ok=True)
        json_state.write(self.folder / self.NOTES, notes)

    def notes(self) -> dict[str, dict[str, Any]]:
        got = json_state.read(self.folder / self.NOTES, {})
        return got if isinstance(got, dict) else {}

    def forget(self, camera: str) -> None:
        """Drop `camera`'s cloud and note."""
        shutil.rmtree(self.folder / _slug(camera), ignore_errors=True)
        self.note(camera, None, time.time())


#: A picture's point map: `pointmap` with the runtime bound, or a test's stand-in.
Runner = Callable[[np.ndarray, np.ndarray | None], Pointmap]


def refresh(
    store: Store,
    camera: str,
    picture: np.ndarray,
    now: float,
    *,
    run: Runner | None,
    frames: int = 1,
    thumb: Sequence[float] = (),
    manifest: Path | None = None,
) -> Cloud | None:
    """Remake `camera`'s cloud from `picture` (BGR, with nobody in it): flatten a wide lens and mark the overlays
    (spatial/lens.py), run the model, and keep the cloud if it's accepted. The cloud, or None with the reason noted
    (`Store.note`) and the earlier cloud left as it was."""
    from liberational.spatial import lens as lenses
    from liberational.spatial import rig

    if run is None:
        store.note(camera, "no_runtime", now)
        return None
    try:
        found, masks = lenses.for_camera(manifest or lenses.manifest(), camera)
        prepared, overlay = lenses.prepare(picture, found, masks)
    except ValueError as e:
        store.note(camera, "lens", now, str(e))
        return None
    mount = rig.load(manifest).get(camera)
    try:
        made = make(camera, run(prepared, overlay), mount.height_m if mount.fixed else None, now, frames, thumb)
    except NoGpu as e:
        store.note(camera, "gpu", now, str(e))
        return None
    except Rejected as e:
        store.note(camera, e.reason, now, str(e))
        return None
    except (RuntimeError, OSError, ValueError, KeyError) as e:
        LOG.warning("%s: no cloud: %s", camera, e)
        store.note(camera, "failed", now, str(e)[-300:])
        return None
    store.replace(made)
    LOG.info(
        "%s: a cloud of %d points (%s size, %.2f s, %.1f GB)",
        camera,
        made.cloud.points,
        made.cloud.size,
        made.cloud.seconds,
        made.cloud.peak_gpu_gb,
    )
    return made.cloud


def runner() -> Runner | None:
    """`pointmap` in the configured runtime, or None while none is configured."""
    found = runtime()
    return None if found is None else (lambda picture, overlay: pointmap(found, picture, overlay))
