"""What every job that works from the cameras' recorded frames shares (part of spatial memory): which cameras are
fixed, reading and scoring their frames, finding the people in a frame, running MASt3R, and a cloud's floor and plan.

- **Cameras.** `fixed_cameras` reads the capture manifest (`config/devices.toml`); `indoor_cameras` leaves out the
  ones mounted looking outdoors (spatial/rig.py).
- **Frames.** `recorded` reads a camera's recorded frame nearest a time from the broker, and `sample_times` gives the
  times a day's midday is tried at. `daylight` says whether a frame is lit colour (not dark, not a camera's night
  mode), and `sharpness` scores its focus and motion blur.
- **People.** `detector` is see-actions' person detector (RF-DETR, on the CPU); `covered_share` says how much of a
  frame its boxes cover and `person_mask` gives their pixels, grown a little, so people can be skipped or masked.
- **MASt3R.** `run_mast3r` runs spatial/mast3r_runner.py in MASt3R's own runtime (`mise run mast3r`, under
  `spatial.runtime`) on the GPU `pick_gpu` chooses. MASt3R (naver/mast3r, code and weights CC BY-NC-SA 4.0, the
  weights also under the non-commercial terms of the datasets they were trained on) is accepted for now;
  docs/SPATIAL.md says so where it lists the model.
- **Floor and plan.** `floor_along` and `fit_floor` find a cloud's floor plane, and `layout` derives its floorplan
  (spatial/floorplan.py, through spatial/tiling.py).
- **Held intrinsics.** `with_held` gives a camera's pose the focal and lens held for it (spatial/lens.py).

Nothing here runs on a schedule or writes a place: the jobs that build maps call these.

`MIN_MATCHES` of 1,000 comes from the first measurements: two views of one room matched 3,528 times, a camera with a
weak tie to them about 600, and views of different spaces at most 187.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import subprocess
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import tzinfo
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import logs
from liberational.memory import places
from liberational.spatial import floorplan, lens, rig, tiling

LOG = logs.get("spatial")


MIN_MATCHES = 1000


MAST3R_SIZE = 512  # the long side MASt3R sees a frame at


#: A frame darker than this mean (0 to 255) is left out: MASt3R's matches, and the 1,000 bar, came from lit frames.
MIN_BRIGHTNESS = 60.0


#: A frame whose channels differ by less than this on average is a night-mode (infrared) grey frame, left out too.
MIN_COLOUR = 4.0


RUNNER = Path(__file__).with_name("mast3r_runner.py")


RUN_TIMEOUT_S = 1800.0


def fixed_cameras(manifest: Path) -> list[str]:
    """The enabled fixed cameras in the capture manifest, in manifest order: not on a rig (`mount.rig`, spatial/rig.py,
    #5), whose poses change as it moves, and with no `fisheye` calibration, which MASt3R's pinhole model can't take."""
    try:
        parsed = tomllib.loads(manifest.read_text())
    except FileNotFoundError:
        return []
    mounts = rig.parse(parsed)
    return [
        str(d["name"])
        for d in parsed.get("device", [])
        if isinstance(d, dict)
        and d.get("kind") in {"camera", "rtsp"}
        and d.get("enabled", True)
        and "fisheye" not in d
        and d.get("name")
        and mounts.get(str(d["name"])).fixed
    ]


def indoor_cameras(cams: Sequence[str], mounts: rig.Mounts) -> list[str]:
    """`cams` without the ones mounted looking outdoors (`mount.outdoor`), in order."""
    out = mounts.outdoor()
    return [c for c in cams if c not in out]


def light(pixels: np.ndarray) -> tuple[float, float]:
    """A BGR frame's mean brightness (0 to 255) and colour (the mean difference between its channels: near 0 for a
    grey, night-mode frame); (0, 0) for a frame that isn't three-channel."""
    img = np.asarray(pixels, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        return (0.0, 0.0)
    spread = np.abs(img[..., 0] - img[..., 1]) + np.abs(img[..., 1] - img[..., 2])
    return (float(img.mean()), float(spread.mean()))


def daylight(pixels: np.ndarray) -> bool:
    """Whether a BGR frame is lit colour, the kind MASt3R matches well, rather than a dark frame or a
    camera's night mode (infrared: grey, its three channels all but equal)."""
    brightness, colour = light(pixels)
    return brightness >= MIN_BRIGHTNESS and colour >= MIN_COLOUR


#: The last day's midday, when a fixed camera's splat training frames are taken (spatial/splats.py): tried every
#: SAMPLE_EVERY_S over MIDDAY (local minutes past midnight, [from, to)) of the last LOOKBACK_S. A splat is one fixed
#: appearance, and the day's light changes it: on Place 1, 10:30 to 14:30 against 09:00 to 16:00 put held-out midday
#: frames 2.0 to 2.5 dB closer (cam0 28.4 to 30.9, living-room 25.7 to 27.7) for 0.5 dB less on a 09:40 frame.
#: Every 20 minutes keeps up to 12 tries a camera, well above the splat's 3-frame minimum.
LOOKBACK_S = 86400.0


SAMPLE_EVERY_S = 1200.0


MIDDAY = (10 * 60 + 30, 14 * 60 + 30)


#: A try takes the recorded frame nearest its time, this close.
RECORDED_WITHIN_S = 2.0


#: One camera's recorded frame nearest a time (BGR), or None.
Reader = Callable[[str, float], Any]


#: The people in a BGR frame: their boxes, (x1, y1, x2, y2) as fractions of its width and height (`detector`).
Detector = Callable[[np.ndarray], list[tuple[float, float, float, float]]]


#: RF-DETR's person score a box needs to count as someone in a frame a map is built from. Lower than
#: see-actions' 0.6 tracking floor: a missed person ends up in the cloud as furniture, an
#: extra box only leaves a patch of them unknown.
PERSON_SCORE = 0.5


#: Each person's box is widened by this share of its size each way before it's left out (a hand, a foot on a stool).
PERSON_GROW = 0.1


def sample_times(
    now: float,
    zone: tzinfo,
    lookback: float = LOOKBACK_S,
    every: float = SAMPLE_EVERY_S,
    window: tuple[int, int] = MIDDAY,
) -> list[float]:
    """The times frames are tried at: every `every` seconds over the last `lookback`, oldest first, the newest `every`
    before now, those whose local time in `zone` is in `window` (minutes past midnight, [from, to))."""
    count = int(lookback // every)
    times = [now - every * (count - i) for i in range(count)]
    return [t for t in times if window[0] <= _minute(dt.datetime.fromtimestamp(t, zone)) < window[1]]


def _minute(when: dt.datetime) -> int:
    return when.hour * 60 + when.minute


def recorded(cam: str, t: float) -> Any:
    """`cam`'s recorded frame nearest `t` from the pack's broker, or None."""
    from liberational import vision_log

    return vision_log.at().frame(cam, t, RECORDED_WITHIN_S)


SHARP_MIN_GRAD = 16.0  # a 3x3 Sobel gradient under this is flat surface or sensor noise, not an edge
SHARP_MIN_EDGES = 0.002  # with fewer pixels than this on an edge, a frame is blank or fogged
SHARP_EDGE_PCT = 95.0  # score the strongest twentieth of the frame's gradients, or all real edges when sparser


def sharpness(pixels: np.ndarray) -> float:
    """How sharp a BGR frame is: its Laplacian energy over Sobel gradient on its strongest edges (higher is sharper;
    motion blur, a smudged or fogged lens, and a refocusing camera all lower it, while a crisp doorway or plain wall
    after a cluttered room keeps its edges' sharpness)."""
    import cv2

    grey = cv2.cvtColor(np.asarray(pixels, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    grad = np.hypot(cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3))
    if float((grad >= SHARP_MIN_GRAD).mean()) < SHARP_MIN_EDGES:
        return 0.0
    bar = max(SHARP_MIN_GRAD, float(np.percentile(grad, SHARP_EDGE_PCT)))
    lap = cv2.Laplacian(grey, cv2.CV_64F)
    edge = grad >= bar
    return float((lap[edge] ** 2).mean() / grad[edge].mean())


def person_boxes(
    dets: np.ndarray, logits: np.ndarray, score: float = PERSON_SCORE
) -> list[tuple[float, float, float, float]]:
    """The people among RF-DETR's queries: `dets` (N x 4, centre x, centre y, width, height, fractions of the frame)
    whose person logit (sparse COCO, see vision/crossings.py) reaches `score` after a sigmoid, as clipped
    (x1, y1, x2, y2) fractions. RF-DETR has no NMS, so one person can be two overlapping boxes; their union is what's
    left out, so that doesn't matter here."""
    from liberational.vision import crossings

    slot = crossings.RF_CLASS_IDS[crossings.COCO_PERSON]
    keep = 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64)[:, slot])) >= score
    out = []
    for cx, cy, w, h in np.asarray(dets, dtype=np.float64)[keep]:
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def detector() -> Detector | None:
    """See-actions' person detector (RF-DETR, [models.files.person_detector]) on the CPU, about 0.14 s a frame, or
    None when the model or onnxruntime isn't installed."""
    path = cfg.model_path("person_detector")
    if not path.exists():
        LOG.info("no person detector at %s", path)
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        LOG.info("no onnxruntime, so no person detector")
        return None
    from liberational.vision import crossings

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    entry = session.get_inputs()[0]
    size = int(entry.shape[2])
    names = [o.name for o in session.get_outputs()]
    if set(names) != {"dets", "labels"}:
        LOG.warning("the person detector at %s isn't RF-DETR's (dets, labels)", path)
        return None

    def people(frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        got = dict(zip(names, session.run(None, {entry.name: crossings.detector_input(np.asarray(frame), size)})))
        return person_boxes(np.asarray(got["dets"])[0], np.asarray(got["labels"])[0])

    return people


def covered_share(boxes: Sequence[Sequence[float]], cells: int = 64) -> float:
    """How much of a frame the boxes (fractions) cover, their union measured on a `cells` square grid."""
    grid = np.zeros((cells, cells), bool)
    for x1, y1, x2, y2 in boxes:
        grid[int(y1 * cells) : math.ceil(y2 * cells), int(x1 * cells) : math.ceil(x2 * cells)] = True
    return float(grid.mean())


def person_mask(boxes: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    """The pixels of a `width` x `height` frame the people's boxes cover, each grown by PERSON_GROW (H x W bool)."""
    out = np.zeros((height, width), bool)
    for x1, y1, x2, y2 in boxes:
        gx, gy = PERSON_GROW * (x2 - x1), PERSON_GROW * (y2 - y1)
        c1, c2 = max(0, math.floor((x1 - gx) * width)), min(width, math.ceil((x2 + gx) * width))
        r1, r2 = max(0, math.floor((y1 - gy) * height)), min(height, math.ceil((y2 + gy) * height))
        out[r1:r2, c1:c2] = True
    return out


def pick_gpu(setting: int, found: Sequence[Mapping[str, Any]] | None) -> str | None:
    """The CUDA_VISIBLE_DEVICES value for the run: GPU `setting` when it's 0 or more, else the one with the most free
    memory, by UUID (NVML and CUDA can number GPUs differently); None with no GPU known."""
    if not found:
        return None
    if setting >= 0:
        match = [g for g in found if int(g["index"]) == setting]
        return str(match[0]["uuid"]) if match else None
    best = max(found, key=lambda g: int(g["total"]) - int(g["used"]))
    return str(best["uuid"])


def checkpoint() -> Path:
    return cfg.owner_file(str(cfg.load()["models"]["hf"]["mast3r"]["path"]))


def run_mast3r(
    runtime: Path,
    images: Mapping[str, Path],
    out: Path,
    gpu: str | None,
    *,
    min_matches: int = MIN_MATCHES,
    window: int | None = None,
    offsets: Sequence[int] | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    timeout_s: float = RUN_TIMEOUT_S,
    nice: int = 0,
    lone_matches: int | None = None,
    save_pairs: bool = False,
) -> dict[str, Any]:
    """Run spatial/mast3r_runner.py in MASt3R's runtime on `images` and return its answer (see that module). `window`
    limits the pairs to images at most that many apart in name order; `pairs` (image names) runs the runner's pairs
    mode on just those, `save_pairs` saving their inferences too, for registration by
    reprojection; `nice` lowers the run's CPU priority; `lone_matches` makes a camera with no pair that strong a place
    of its own."""
    python = runtime / ".venv" / "bin" / "python"
    repo = runtime / "src"
    if not python.exists() or not (repo / "mast3r").is_dir():
        raise RuntimeError(f"MASt3R's runtime isn't installed in {runtime}; run `mise run mast3r`")
    if not (checkpoint() / "model.safetensors").exists():
        raise RuntimeError(f"MASt3R's checkpoint isn't in {checkpoint()}; run `mise run models`")
    out.mkdir(parents=True, exist_ok=True)
    job = out / "job.json"
    job.write_text(
        json.dumps(
            dict(
                repo=str(repo),
                checkpoint=str(checkpoint()),
                images={c: str(p) for c, p in images.items()},
                min_matches=min_matches,
                size=MAST3R_SIZE,
                out=str(out),
                **({"window": window} if window is not None else {}),
                **({"offsets": list(offsets)} if offsets is not None else {}),
                **({"pairs": [list(p) for p in pairs]} if pairs is not None else {}),
                **({"lone_matches": lone_matches} if lone_matches is not None else {}),
                **({"save_pairs": True} if save_pairs else {}),
            )
        )
    )
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    command = [str(python), str(RUNNER), str(job)]
    if nice:
        command = ["nice", "-n", str(nice), *command]
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout_s)
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1:] or ["(no output)"]
        raise RuntimeError(f"MASt3R failed ({done.returncode}): {tail[0][:300]}")
    return json.loads((out / "result.json").read_text())


#: As the runner's floor_from_normals: horizontal surfaces are points whose normal is this close to up (degrees), their
#: heights binned this fine (metres); the floor is the lowest bin holding this share of them, fitted within the band.
FLAT_DEG, FLOOR_BIN, FLOOR_SHARE, FLOOR_BAND = 15.0, 0.04, 0.05, 0.06


FLOOR_POINTS = 200_000  # at most this many of the cloud's points are read for its surfaces


def floor_along(points: np.ndarray, up: Sequence[float] | np.ndarray) -> tuple[float, float, float, float] | None:
    """The cloud's floor (nx, ny, nz, d) as the lowest layer of its horizontal surfaces along `up` that holds
    FLOOR_SHARE of them, with a plane fitted to that layer; None when too few surfaces are horizontal. Surface normals
    come from each point's 16 nearest neighbours."""
    from scipy.spatial import KDTree

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) > FLOOR_POINTS:
        pts = pts[np.random.default_rng(0).choice(len(pts), FLOOR_POINTS, replace=False)]
    if len(pts) < 100:
        return None
    upv = np.asarray(up, dtype=np.float64)
    upv = upv / np.linalg.norm(upv)
    _, idx = KDTree(pts).query(pts, k=16)
    near = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", near, near))
    flat = pts[
        (np.abs(vectors[:, :, 0] @ upv) > math.cos(math.radians(FLAT_DEG))) & (values[:, 0] < 0.05 * values[:, 1])
    ]
    if len(flat) < 50:
        return None
    h = flat @ upv
    counts, edges = np.histogram(h, np.arange(h.min(), h.max() + 2 * FLOOR_BIN, FLOOR_BIN))
    enough = np.nonzero(counts >= max(50, FLOOR_SHARE * len(flat)))[0]
    if not len(enough):
        return None
    level = (edges[enough[0]] + edges[enough[0] + 1]) / 2
    layer = flat[np.abs(h - level) < FLOOR_BAND]
    centre = layer.mean(axis=0)
    normal = np.linalg.svd(layer - centre, full_matrices=False)[2][-1]
    normal = normal if normal @ upv > 0 else -normal
    x, y, z = (float(v) for v in normal)
    return (x, y, z, -float(normal @ centre))


def up_direction(
    poses: Mapping[str, Sequence[float]], sizes: Mapping[str, Sequence[int]] | None = None
) -> tuple[float, float, float]:
    """The cloud's rough up: the mean of the cameras' up axes (each camera's -y, as cameras hang roughly level). A
    camera whose frames are portrait is on its side, so its -y says nothing about up: only landscape cameras count
    (by `sizes`, width and height), unless there are none."""
    level = [c for c in poses if sizes is None or int(sizes[c][0]) >= int(sizes[c][1])] or list(poses)
    ups = [-np.asarray(poses[c], dtype=float).reshape(4, 4)[:3, 1] for c in level]
    up = np.mean(ups, axis=0)
    norm = float(np.linalg.norm(up))
    if norm < 1e-6:
        return (0.0, -1.0, 0.0)
    x, y, z = (up / norm).tolist()
    return (x, y, z)


@dataclass(frozen=True)
class Layout:
    """What the cloud says about its place's floor: the plan (a scene's text) and its map-to-plan transform (12 numbers), when
    the cloud has the walls for a plan, and the floor plane (nx, ny, nz, d), when it has a floor."""

    plan: str | None
    transform: tuple[float, ...] | None
    floor: tuple[float, ...] | None


#: The walked floor on a found floor plane (nx, ny, nz, d): the footprints projected onto it, (N, 3) points in the map
#: frame, and how many footprints each point stands for (spatial/footprints.py reads them as cells of the frame).
Walked = Callable[[tuple[float, ...]], tuple[np.ndarray, np.ndarray]]


def fit_floor(points: np.ndarray, up: tuple[float, float, float]) -> tuple[float, float, float, float] | None:
    """The cloud's floor plane (nx, ny, nz, d: a point's height above it is n . p + d), or None without one."""
    settings = floorplan.Settings(up=up)
    try:
        normal, offset = floorplan.floor_plane(
            np.asarray(points, dtype=np.float64), up, settings.floor_tolerance, np.random.default_rng(settings.seed)
        )
    except ValueError:
        return None
    x, y, z = (float(v) for v in normal)
    return (x, y, z, -float(offset))


def layout(
    points: np.ndarray,
    up: tuple[float, float, float],
    title: str,
    walked: Walked | None = None,
    floor: Sequence[float] | None = None,
) -> Layout:
    """The cloud's floor and, when it has one, its floorplan. A few high cameras' views have the floor and furniture
    but few walls, so where people walked on that floor (`walked`, spatial/footprints.py) closes the rooms
    the walls alone don't. A `floor` given (nx, ny, nz, d: a lone camera's, which the runner finds from its
    surfaces) is taken as it is; otherwise one is fitted to the cloud."""
    settings = floorplan.Settings(up=up)
    floor = floor if floor is not None else fit_floor(points, up)
    if floor is None:
        return Layout(None, None, None)
    floor = tuple(float(x) for x in floor)
    feet, weights = walked(floor) if walked is not None else (None, None)
    try:
        plan = floorplan.floorplan(points, settings, floor=floor, walkable=feet, walk_weights=weights)
    except ValueError as e:
        LOG.info("no floorplan for %s (%d footprints): %s", title, 0 if weights is None else int(weights.sum()), e)
        return Layout(None, None, floor)
    # Through the tiling converter, so the library takes it as it is: rooms on shared centrelines, openings
    # clamped to their walls. Its squaring turn (none for the baseline's axis-aligned rooms) goes into the transform.
    rooms, openings = tiling.from_plan(plan)
    t = tiling.tile(rooms, openings, tiling.Settings(faces=False))
    for note in t.notes:
        LOG.info("%s: %s", title, note)
    c, s = math.cos(t.angle), math.sin(t.angle)
    turned = plan.transform.copy()
    turned[:2] = np.array([[c, -s], [s, c]]) @ plan.transform[:2]
    tr = tuple(float(x) for x in turned.ravel())
    return Layout(tiling.to_scene(t, title, tiling.Settings(faces=False)), tr, floor)


def camera_pose(pose: np.ndarray, focal: float, division: float, size: Sequence[int], at: float) -> places.CameraPose:
    """A registered camera as a place revision keeps it: a pinhole with no lens, else its lens as spatial/lens.py's
    tuple, centred, and its focal the undistorted frame's (`lens.out_scale` of the fitted one), as floor_point casts
    rays through it."""
    w, h = int(size[0]), int(size[1])
    flat = tuple(float(x) for x in np.asarray(pose).ravel())
    if abs(division) < 1e-6:
        return places.CameraPose(flat, float(focal), w, h, at)
    found = lens.Lens(w, h, (focal, focal, (w - 1) / 2, (h - 1) / 2), division)
    return places.CameraPose(flat, float(focal * lens.out_scale(found)), w, h, at, found.as_tuple())


def with_held(camera: str, pose: places.CameraPose, held: lens.Held | None) -> places.CameraPose:
    """`camera`'s `pose` with its `held` intrinsics in place of its own: the focal, and the lens (its configured one
    when `lens.as_registered` has one, since `held.focal` is the undone frame's focal, else the held one). Intrinsics
    measured through another lens than the configured one (`lens.through`) aren't this frame's: `pose` stays as it is."""
    configured = lens.as_registered(camera)
    held = lens.through(held, configured)
    if held is None:
        return pose
    if configured is not None:
        return replace(pose, focal=held.focal * pose.width, lens=configured)
    m = np.asarray(pose.pose, dtype=np.float64).reshape(4, 4)
    found = camera_pose(m, held.focal * pose.width, held.division, (pose.width, pose.height), pose.at)
    return replace(found, background=pose.background)
