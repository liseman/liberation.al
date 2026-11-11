"""Each camera's own point cloud: what one camera sees, as coloured points in 3D (#417).

A camera that stays put can't measure a room: nothing it sees shifts against anything else. What it can give is its
own view with depth, and that is what this is: one camera, one cloud, made from one clean picture of that camera and
shown by itself. Clouds of different cameras are never joined here; only a camera that moves can relate two views.

**Making one** (`make`). A picture goes through MASt3R-SLAM's single-frame step in its own runtime
(spatial/pointmap_runner.py, as a subprocess: `Runtime`, `pointmap`), which gives every pixel a point in the
camera's own frame and a confidence. Then, in the pack:

1. *Which pixels count.* A point is kept when the model is sure of it (`CONF`, MASt3R's own bar) and its pixel isn't
   one the camera burns its clock or name into (the device manifest's `masks`, spatial/lens.py). Bright windows,
   glass, and sky usually fall out here. Then the points that aren't on a surface the camera really saw go
   (`clean`): a pixel at a jump in depth, where the model strings points between a near thing and the far one
   behind it (`JUMP`); a pixel of a surface seen so nearly edge-on that its depth is a guess (`GRAZING`); and a
   speck, a patch of fewer than `SPECK` pixels left standing alone. These are the long sheets and streaks a cloud
   shows from anywhere but its own camera.
2. *Accept or reject.* A picture the model is unsure about over most of it gives no cloud (`MIN_SHARE` of its pixels
   must count: the six cameras on the Linux machine kept 0.60 to 0.88), nor does one with no depth to speak of
   (`MIN_DEPTH_SPREAD`). The reason is kept (`Store.note`) and the camera's earlier cloud stays.
3. *The floor* (`floor`). The plane with the most points on it among those that face roughly up in the picture, pass
   under the camera, and have next to nothing under them: the floor is the lowest big surface, which rules out a
   table top. Found on five of those six cameras; the sixth sees almost no floor. A camera's measured mounting
   height (the device manifest's `mount.height_m`), when it has one, checks the find: a plane that puts the camera
   outside `HEIGHT_FITS` of that height isn't its floor (a sofa seat, with the floor out of sight, read 0.32 m
   under a camera hung at 1.47 m), and is passed over.
4. *Size* (`Cloud.size`). The cloud is left in MASt3R's own metres, and says so: "approximate". Sizing it by the
   mounting height was tried and made it worse. The height reads the floor a metre or two from the camera, the walls
   are six or seven away, and the model's near and far depth don't share one scale: on two cameras the walls landed
   1.11 and 1.24 times too far on the owner's plan when sized by height, against 0.96 and 1.00 in the model's own
   metres (and 1.06 on a third).
5. *Its frame* (`level`, `Cloud.level`). The cloud is turned level: z is up, the floor is z = 0, the camera stands
   on the z axis looking along +y. Its tilt and roll are its floor's ("floor"). With no floor found, its walls say
   which way is up ("walls": spatial/cloud_align.py `stand`; a wall is vertical), and the lowest points stand in
   for the floor. With neither, the camera is taken to be upright ("none"), which a tilted camera isn't: two such
   clouds on the Linux machine stood 11.7 and 12.5 degrees off level, and came upright by their walls. A cloud made
   before walls were used is stood up by them once, where it is kept (`relevel`).

**Which cloud is kept** (`refresh`, `better`). A camera's view looks different by night, so a cloud is made then
too, and a dark picture often shows no floor. While the camera hasn't moved (its picture registers with its anchor,
spatial/cloud_scene.py), a new cloud that is worse than the one it has doesn't replace it: one with no floor against
one with a floor, or one with under `FEWER` of its points. The look is remembered (`Cloud.also`), so the same view
isn't tried again. A camera with no spot has no anchor to say it stayed put, and its newest cloud is kept.

**Keeping one** (`Store`): `state/camera_clouds/<camera>/` holds the camera's one cloud, `<id>.ply` (binary, x y z
and red green blue) and `cloud.json` (`Cloud`: when, how many points, the camera's pose and focal
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import json_state, logs
from liberational.memory import places
from liberational.spatial import cloud_align, ply

LOG = logs.get("camera-clouds")

DIR = "camera_clouds"  # under paths.state
RUNTIME = "spatial.pointmap_runtime"
WEIGHTS = "spatial.pointmap_weights"
NEED_GB = 4.0  # free GPU memory a pass needs: 2.9 GB measured, with room for the driver's own
NICE = 10
TIMEOUT_S = 180.0
CONF = 1.5  # MASt3R's confidence bar for a usable point
MIN_SHARE = 0.25  # of a picture's pixels that must give a point; live pictures gave 0.50 to 0.80 after `clean`
JUMP = 0.04  # a neighbour this much nearer or farther (of the nearer depth) is another surface
GRAZING = 0.17  # the cosine between a surface's normal and the ray to it: under it, the surface is within 10 degrees of edge-on
SPECK = 40  # pixels: a smaller patch standing alone is dropped
MIN_DEPTH_SPREAD = 1.15  # the far tenth of the points must be this many times as far as the near tenth
FLOOR_TOL_M = 0.025
FLOOR_TRIES = 1500
FLOOR_SAMPLE = 40000
FLOOR_UP = 0.5  # cosine: a floor's normal is within 60 degrees of the camera's own up
FLOOR_UNDER = (0.06, 0.02)  # at most this share of points may lie more than this far (m) under the floor
FLOOR_MIN = 0.04  # of the sampled points must be on it
FLOOR_BELOW_M = 0.5  # the floor passes at least this far under the camera: nearer is a table top or a shelf
FEWER = 0.8  # a new cloud with under this share of the kept one's points is the worse of the two
HEIGHT_FITS = (
    0.6,
    1.7,
)  # a floor is the camera's when its measured height over the height found is in this; live: 0.85 to 1.25
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
    #: What the picture matched of each picture it was compared with, in the order asked (spatial/cloud_scene.py).
    matches: tuple[dict[str, Any] | None, ...] = ()


class NoGpu(RuntimeError):
    """No GPU has the memory a pass needs right now."""


#: A picture to compare the new one with, and whether that picture is the one seen from (the reference).
Other = tuple[Path, bool]


def pointmap(found: Runtime, picture: np.ndarray, overlay: np.ndarray | None, others: Sequence[Other] = ()) -> Pointmap:
    """`picture` (BGR) through the runner in `found`, niced, on the GPU with the most free memory, and compared with
    each of `others` in the same run (one load of the model). Raises NoGpu when none has NEED_GB free, and
    RuntimeError when the runner fails."""
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
            "pairs": [
                {
                    "a": str(path if first else folder / "picture.png"),
                    "b": str(folder / "picture.png" if first else path),
                    "out": str(folder / f"pair{i}.json"),
                }
                for i, (path, first) in enumerate(others)
            ],
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
        matches = tuple(json_state.read(folder / f"pair{i}.json") for i in range(len(others)))
        with np.load(folder / "points.npz") as got:
            return Pointmap(
                got["points"],
                got["confidence"],
                got["colour"],
                got["covered"],
                report["seconds"],
                report["peak_gpu_gb"],
                matches,
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
    size: str  # "approximate": the model's own metres
    floor_found: bool
    pose: tuple[float, ...]  # camera to cloud, 4 x 4 row by row
    focal: float  # of the picture the model saw, in its pixels
    width: int
    height: int
    seconds: float = 0.0
    peak_gpu_gb: float = 0.0
    frames: int = 1  # the frames whose median the picture was
    cleaned: float = 0.0  # of the points the model was sure of, the share `clean` dropped
    thumb: tuple[float, ...] = field(default=(), repr=False)  # the picture's thumbnail, to notice a changed view
    level: str = ""  # what it was levelled by: "floor", "walls", or "none"; "" for one made before walls were used
    #: The thumbnail of a later picture of the same view that gave no better cloud (`refresh`): that view is known too.
    also: tuple[float, ...] = field(default=(), repr=False)

    @property
    def levelled(self) -> str:
        """`level`, for an older cloud too."""
        return self.level or ("floor" if self.floor_found else "none")

    @staticmethod
    def from_dict(value: dict[str, Any]) -> Cloud:
        known = {k: value[k] for k in Cloud.__dataclass_fields__ if k in value}
        known["pose"] = tuple(float(v) for v in known["pose"])
        known["thumb"] = tuple(float(v) for v in known.get("thumb", ()))
        known["also"] = tuple(float(v) for v in known.get("also", ()))
        return Cloud(**known)


@dataclass(frozen=True)
class Made:
    """A cloud ready to keep: its record and its PLY."""

    cloud: Cloud
    ply: bytes
    picture: np.ndarray | None = None  # the picture it was made from (BGR), kept for comparing with later ones


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


def clean(points: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """`keep` (H x W bool) without the pixels of `points` (H x W x 3, the camera's frame) that aren't on a surface
    the camera saw: those at a jump in depth, those of a surface seen nearly edge-on, and specks (the module
    docstring's step 1)."""
    import cv2

    depth = points[..., 2]
    jump = np.zeros(keep.shape, dtype=bool)
    for axis in (0, 1):
        a, b = np.moveaxis(depth, axis, 0)[:-1], np.moveaxis(depth, axis, 0)[1:]
        apart = np.abs(a - b) > JUMP * np.minimum(np.abs(a), np.abs(b))
        both = np.moveaxis(jump, axis, 0)
        both[:-1] |= apart
        both[1:] |= apart
    across = np.zeros_like(points)
    down = np.zeros_like(points)
    across[:, 1:-1] = points[:, 2:] - points[:, :-2]
    down[1:-1] = points[2:] - points[:-2]
    normal = np.cross(across, down)
    length = np.linalg.norm(normal, axis=2)
    ray = points / np.maximum(np.linalg.norm(points, axis=2, keepdims=True), 1e-9)
    facing = np.abs((normal * ray).sum(axis=2)) / np.maximum(length, 1e-12)
    grazing = (facing < GRAZING) & (length > 0)
    out = keep & ~jump & ~grazing
    count, labels, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < SPECK)
    return out & ~np.isin(labels, small[small > 0]) if count > 1 else out


def make(
    camera: str, got: Pointmap, height_m: float | None, now: float, frames: int = 1, thumb: Sequence[float] = ()
) -> Made:
    """`camera`'s cloud from its picture's point map (the module docstring's steps 1 to 5). Raises Rejected."""
    points = np.asarray(got.points, dtype=np.float64)
    sure = (got.confidence > CONF) & ~got.covered & np.isfinite(points).all(axis=2) & (points[..., 2] > 0)
    keep = clean(np.nan_to_num(points), sure)
    share = float(keep.mean())
    if share < MIN_SHARE:
        raise Rejected("unsure", f"only {share:.0%} of the picture gave a point; {MIN_SHARE:.0%} needed")
    kept = points[keep]
    near, far = np.percentile(kept[:, 2], [10, 90])
    if far < MIN_DEPTH_SPREAD * near:
        raise Rejected("flat", f"no depth to speak of: {near:.2f} to {far:.2f} m")
    found = floor(kept)
    fits = found is None or not height_m or HEIGHT_FITS[0] <= height_m / found[1] <= HEIGHT_FITS[1]
    if not fits:
        found = None  # not this camera's floor: it would stand far from its measured height above it
    levelled, pose, floor_found = level(kept, found)
    by = "floor" if floor_found else "none"
    if not floor_found:
        upright = cloud_align.stand(levelled, pose)
        if upright is not None:
            levelled, pose, by = cloud_align.apply(upright, levelled), upright @ pose, "walls"
    h, w = keep.shape
    cloud = Cloud(
        id=str(uuid.uuid7()),
        camera=camera,
        at=now,
        points=len(levelled),
        share=round(share, 3),
        size="approximate",
        floor_found=floor_found,
        pose=tuple(float(v) for v in pose.ravel()),
        focal=_focal(points, keep),
        width=w,
        height=h,
        seconds=got.seconds,
        peak_gpu_gb=got.peak_gpu_gb,
        frames=frames,
        thumb=tuple(float(v) for v in thumb),
        cleaned=round(1.0 - float(keep.sum()) / max(int(sure.sum()), 1), 3),
        level=by,
    )
    return Made(cloud, ply.dumps(levelled, got.colour[keep]))


RANK = {"floor": 2, "walls": 1, "none": 0}


def better(new: Cloud, old: Cloud) -> bool:
    """Whether `new` is at least as good a cloud of the same view as `old` (the module docstring)."""
    if RANK[new.levelled] != RANK[old.levelled]:
        return RANK[new.levelled] > RANK[old.levelled]
    return new.points >= FEWER * old.points


def root(state: Path | str) -> Path:
    return Path(state) / DIR


def slug(camera: str) -> str:
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
            value = json.loads((self.folder / slug(camera) / "cloud.json").read_text())
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
        return self.folder / slug(cloud.camera) / f"{cloud.id}.ply"

    def picture(self, cloud: Cloud) -> Path:
        """Where the picture `cloud` was made from is kept (whether or not it is there)."""
        return self.file(cloud).with_suffix(".png")

    def replace(self, made: Made) -> None:
        """Keep `made` as its camera's cloud. Its file is written first and its record after, so a reader never finds
        a record whose points aren't there; the cloud before it goes last."""
        import cv2

        folder = self.folder / slug(made.cloud.camera)
        folder.mkdir(parents=True, exist_ok=True)
        places.write_whole(self.file(made.cloud), made.ply)
        if made.picture is not None:
            ok, png = cv2.imencode(".png", made.picture)
            if ok:
                places.write_whole(self.picture(made.cloud), png.tobytes())
        json_state.write(folder / "cloud.json", asdict(made.cloud))
        for old in [*folder.glob("*.ply"), *folder.glob("*.png")]:
            if old.stem not in (made.cloud.id, "anchor", "before.anchor"):  # spatial/cloud_scene.py's to keep
                old.unlink(missing_ok=True)
        self.note(made.cloud.camera, None, made.cloud.at)

    def seen(self, cloud: Cloud, thumb: Sequence[float]) -> Cloud:
        """Remember a later picture of `cloud`'s view that gave no better cloud (`Cloud.also`)."""
        known = replace(cloud, also=tuple(float(v) for v in thumb))
        json_state.write(self.folder / slug(cloud.camera) / "cloud.json", asdict(known))
        return known

    def relevel(self, camera: str) -> Cloud | None:
        """Stand up, by its walls, a cloud made before walls were used that found no floor (the module docstring's
        step 5). The cloud is kept as a new one (its points turned, nothing else changed), which is returned; None
        when there was nothing to do. A cloud is looked at once: one with no wall is marked "none"."""
        cloud = self.current(camera)
        if cloud is None or cloud.level:
            return None
        if cloud.floor_found:
            json_state.write(self.folder / slug(camera) / "cloud.json", asdict(replace(cloud, level="floor")))
            return None
        points, colours, _ = ply.read(self.file(cloud))
        pose = np.asarray(cloud.pose, dtype=np.float64).reshape(4, 4)
        upright = cloud_align.stand(points, pose)
        if upright is None:
            json_state.write(self.folder / slug(camera) / "cloud.json", asdict(replace(cloud, level="none")))
            return None
        stood = replace(
            cloud, id=str(uuid.uuid7()), level="walls", pose=tuple(float(v) for v in (upright @ pose).ravel())
        )
        picture = self.picture(cloud)
        if picture.is_file():
            shutil.copyfile(picture, self.picture(stood))
        fill = np.full((len(points), 3), 160, dtype=np.uint8) if colours is None else colours
        self.replace(Made(stood, ply.dumps(cloud_align.apply(upright, points), fill)))
        return stood

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
        shutil.rmtree(self.folder / slug(camera), ignore_errors=True)
        self.note(camera, None, time.time())


#: A picture's point map, and what it matched of the pictures it's compared with: `pointmap` with the runtime
#: bound, or a test's stand-in.
Runner = Callable[[np.ndarray, np.ndarray | None, Sequence[Other]], Pointmap]


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
    spots: Callable[[], Mapping[str, Any]] | None = None,
) -> Cloud | None:
    """Remake `camera`'s cloud from `picture` (BGR, with nobody in it): flatten a wide lens and mark the overlays
    (spatial/lens.py), run the model, and keep the cloud if it's accepted. The cloud, or None with the reason noted
    (`Store.note`) and the earlier cloud left as it was.

    With `spots` (the cameras' spots on their plans, spatial/cloud_scene.py `spots_of`), the same run compares the
    picture with the camera's anchor and its neighbours', and what they matched is recorded for the place's scene:
    whether the camera moved, and how each pair of cameras sits."""
    from liberational.spatial import cloud_scene, rig
    from liberational.spatial import lens as lenses

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
        placed = spots() if spots is not None else {}
        compared = cloud_scene.compare(store, camera, placed)
        got = run(prepared, overlay, [(path, label == "anchor") for label, path in compared])
        made = make(camera, got, mount.height_m if mount.fixed else None, now, frames, thumb)
        made = replace(made, picture=prepared)
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
    have = store.current(camera)
    still = bool(compared) and compared[0][0] == "anchor" and cloud_scene.stayed(store, camera, got.matches[:1])
    if have is not None and still and not better(made.cloud, have):
        LOG.info("%s: a new cloud (%s, %d points) is no better than the one it has; kept", camera, made.cloud.level,
                 made.cloud.points)  # fmt: skip
        return store.seen(have, thumb)
    store.replace(made)
    if compared:
        try:
            matches = dict(zip((label for label, _ in compared), got.matches, strict=False))
            cloud_scene.record(store, made.cloud, matches, placed)
        except (OSError, ValueError, KeyError, TypeError) as e:  # the cloud is kept; only its place in the scene isn't
            LOG.warning("%s: its place in the scene wasn't worked out: %s", camera, e)
    elif placed:
        cloud_scene.record(store, made.cloud, {}, placed)
    LOG.info(
        "%s: a cloud of %d points (%.2f s, %.1f GB)",
        camera,
        made.cloud.points,
        made.cloud.seconds,
        made.cloud.peak_gpu_gb,
    )
    return made.cloud


def runner() -> Runner | None:
    """`pointmap` in the configured runtime, or None while none is configured."""
    found = runtime()
    return None if found is None else (lambda picture, overlay, others: pointmap(found, picture, overlay, others))
