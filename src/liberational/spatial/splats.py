"""`pack-spatial-splat`: a 3D Gaussian splat of each fixed-camera place (part of spatial memory),
for the Places 3D view to draw beside, or instead of, the point cloud.

Run by `pack-spatial-cameras` after its daily pass (spatial/cameras.py), or by hand (`pack-spatial-splat [--place
ID]`). For each place, the revision the 3D view shows (the newest whose map is still kept) gets a splat, once, when
it registers fixed cameras:

1. **Frames.** Each registered camera's recorded frames over the last day's daytime, at the times
   spatial/cameras.py tries for its own frame (`cameras.sample_times`: `MIDDAY`, local time, every
   `SAMPLE_EVERY_S`), from the broker (`cameras.recorded`); dark and night-mode ones are left out
   (`cameras.daylight`), and at most `FRAMES_PER_CAMERA`, spread over the day, are kept. A camera with fewer than
   `MIN_FRAMES_PER_CAMERA` takes no part; with none left, the place waits for the next run. Many times of day average
   out the light and the people who pass. Evenings are left out even when the lamps make them lit colour: on Place 1
   a lamp-lit night next to the daytime frames cut the held-out score from 25.8 to 18.7 dB. Each frame is scaled
   and centre-cropped to the size the camera was registered at (`fit`), as MASt3R saw it.
2. **Start.** The revision's map points that some camera sees (`in_view`), at most `MAX_START_POINTS`, are the
   Gaussians' starting centres and colours; the revision's poses and focals are the cameras, taken as they are.
3. **Depth.** Each camera gets one depth image to train against (`depth_priors`, #20): Depth Anything 3's metric
   model on each of its frames (spatial/depth_runner.py, in gsplat's runtime), their per-pixel median, scaled to the
   map by the map points the camera sees (`map_scale`). Without the model's code or weights the splat trains on its
   frames alone, and the log says so.
4. **Train.** spatial/splat_runner.py in gsplat's own runtime (`mise run gsplat`, under `spatial.splat_runtime`),
   niced, on the GPU with the most free memory (`cameras.pick_gpu`): `STEPS` steps, at most `MAX_GAUSSIANS`.
5. **Keep.** The trained 3DGS PLY and its SOG copy (spatial/sog.py) are written beside the revision (memory/places.py
   `write_splat`), and served by the dashboard's `GET /places/splat`. Both are in the map's frame, as is: the 3D view
   (PlayCanvas, #76) draws the SOG without turning it. A splat kept from before the SOG copy (a PLY with an SPZ
   beside it) gets its SOG from its PLY on the next run (`recode`).

Measured with these settings (Place 1: two fixed cameras, 10 daytime frames each, 21k start points, one RTX
3090 shared with the live services): about 2 to 2.5 minutes and 0.2 GiB of GPU memory a place, 244k Gaussians, a 24 MB
PLY, and 27.1 dB PSNR on a frame per camera held out from the same day. Two cameras with little overlap give no new views: the splat looks right from where the cameras
stand, not from between them (6.6 dB from one camera's view trained on the other's). Nothing fills what no camera
sees.

**What the depth prior does** (#20; one place, four fixed cameras with a lens model each, 12 midday frames a camera,
a LiDAR-walked map). Trained on its frames alone, a splat explains each camera's picture with haze: seen from its
own camera its surfaces sit a median 0.36 m off the depth prior, and from a camera left out of training 1.6 to 1.9 m
off. With the prior they sit under 1 cm off from its own cameras and 0.5 to 1.1 m off from a left-out one (what
that camera sees and the others don't has no depth to go by). The picture from the cameras' own views keeps its
score, 29.4 dB held out without and 29.1 dB with. The picture from a left-out camera does not score better (7.6 to
11.7 dB without, 6.8 to 8.4 dB with, and 1 dB apart between two runs of the same settings): at that level PSNR counts
how much of the unseen is filled, and haze fills more of it than surfaces with holes between them do. So the prior
buys geometry, which is what the 3D view moves through, not a novel view's picture. The model's metres needed a
scale of 0.95 to 1.06 to match the map, and the map is no ground truth for it: it has holes, and a room changes
after a walk. The depth model adds about 6 s and 2 GiB of GPU memory to a place's splat.

**A Stray Scanner walkthrough** (`splat_walk`) gets its splat when it's imported: `pack-walkthrough` calls it once the
walk's revision is written and fused (spatial/walkthrough.py `run_stray`). Every saved frame trains, at its ARKit pose
and the phone's own intrinsics, starting from the revision's map points: `WALK_STEPS` steps, at most `WALK_GAUSSIANS`,
and no opacity or scale regularizers. In their place (#43, spatial/splat_runner.py has the measurements) the Gaussians
are held inside the map's box and under a size, each frame's LiDAR depth (`stray.depth_map`) is a loss beside its
image, and each frame's pose is refined as the splat trains. The walk registers `LOOK_FROM` of its frames as cameras
(`look_from`, named `walk-...`), for the 3D view to look from; tier 1 moves them with the walk's cloud into the fused
map's frame, and the move they show (`carried`) brings every frame's pose along. The daily fixed-camera splat leaves
those cameras out. A splat is kept with what it was trained from ("walk" or "cameras", `places.write_splat`), so the
3D view opens it from the right kind of view. A plain video's walk (CUT3R's poses, corrected by the pose graph) is
splatted the same way (#21), without the depth: the pose refinement takes up some of what its poses get wrong. It's
a first look, not a match for a LiDAR walk's: the same 120 frames of one walk through a home scored 15.2 dB held out
as a plain video (14.0 without refinement, which moved each frame a median 6 cm and 2.7 degrees) against 21.6 dB
from the LiDAR export, in the same 11 minutes and 1.3 GiB of GPU memory (`tools/splat_walk_check.py` measures it).

gsplat (nerfstudio-project/gsplat) is Apache-2.0; the trainer adds no model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import tzinfo
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import logs, nvml
from liberational.memory import places
from liberational.spatial import cameras, lens, ply, sog

LOG = logs.get("spatial-splat")

RUNNER = Path(__file__).with_name("splat_runner.py")
FRAMES_PER_CAMERA = 12
#: A camera with fewer lit frames than this takes no part, and a place with no camera left gets no splat this run (the
#: next run will have more). Three is enough: with one frame held out of 12 over the midday window, a splat from 3
#: frames spread over it scored what one from 6 or 11 did (29.4, 29.3, and 29.4 dB), while 1 or 2 frames from its
#: ends scored 22.8 dB: what counts is a frame lit like the one looked at, not how many there are. The bar was 6,
#: which a place missed for a day after every reset or camera rename; at 3 it gets its splat within the first hour
#: of the first midday recorded.
MIN_FRAMES_PER_CAMERA = 3
MAX_START_POINTS = 100_000
STEPS = 7000
MAX_GAUSSIANS = 300_000
#: View-dependent colour to SH band 1 only: a few fixed cameras can't pin down bands 2 and 3. On Place 1 band 1 scored
#: as well as band 3 on held-out frames (27.1 dB both) with about half the files and GPU memory.
SH_DEGREE = 1
RUN_TIMEOUT_S = 1800.0
WALK_TIMEOUT_S = 7200.0  # a walk's splat (`splat_walk`): hundreds of frames, run once per walk
NICE = 10
MIN_DEPTH = 0.05  # metres in front of a camera a start point must be

Reader = cameras.Reader


def fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """`frame` scaled to cover `width` x `height` and centre-cropped to it: how MASt3R's loader made the frame a
    camera was registered from (the long side to 512, the short side cropped to a multiple of 16), so the
    registration's focal and centred principal point hold."""
    import cv2

    h, w = frame.shape[:2]
    scale = max(width / w, height / h)
    sw, sh = max(width, round(w * scale)), max(height, round(h * scale))
    resized = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    x, y = (sw - width) // 2, (sh - height) // 2
    return resized[y : y + height, x : x + width]


def in_view(points: np.ndarray, cams: Mapping[str, places.CameraPose]) -> np.ndarray:
    """Which of `points` (N x 3, the map's frame) at least one camera sees: at least MIN_DEPTH in front of it and
    inside its frame (the camera looks along its +z, OpenCV's convention, principal point at the centre)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    seen = np.zeros(len(pts), dtype=bool)
    for cam in cams.values():
        c2w = np.asarray(cam.pose, dtype=np.float64).reshape(4, 4)
        local = (pts - c2w[:3, 3]) @ c2w[:3, :3]  # world to camera: R^T (p - t)
        z = local[:, 2]
        ahead = z > MIN_DEPTH
        u = np.full(len(pts), -1.0)
        v = np.full(len(pts), -1.0)
        u[ahead] = cam.focal * local[ahead, 0] / z[ahead] + cam.width / 2
        v[ahead] = cam.focal * local[ahead, 1] / z[ahead] + cam.height / 2
        seen |= ahead & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return seen


def spread(items: Sequence[Any], most: int) -> list[Any]:
    """At most `most` of `items`, evenly spaced, the first and last among them when there's room for two."""
    if len(items) <= most:
        return list(items)
    if most == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (most - 1)
    return [items[round(i * step)] for i in range(most)]


def day_frames(cams: Sequence[str], now: float, read: Reader, zone: tzinfo) -> dict[str, list[np.ndarray]]:
    """Each camera's lit recorded frames (`cameras.daylight`) over the last day's daytime in `zone`
    (`sample_times`), at most FRAMES_PER_CAMERA spread over it; a camera with none is left out."""
    got: dict[str, list[np.ndarray]] = {}
    times = cameras.sample_times(now, zone)
    for cam in cams:
        lit = [f for t in times if (f := read(cam, t)) is not None and cameras.daylight(f)]
        if lit:
            got[cam] = spread(lit, FRAMES_PER_CAMERA)
    return got


def target(state: Path, place: str) -> places.Revision | None:
    """The revision a splat of `place` is for: the one the 3D view shows (the newest whose map file is still kept),
    when it registers fixed cameras; else None."""
    folder = places.root(state) / place
    kept = [r for r in places.revisions(state, place) if r.map and (folder / r.map).is_file()]
    return kept[-1] if kept and kept[-1].cameras else None


def start_points(
    points: np.ndarray, colors: np.ndarray | None, cams: Mapping[str, places.CameraPose], most: int = MAX_START_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """The Gaussians' starting centres (float32) and colours (uint8, grey for a map without colours): the map points
    a camera sees, at most `most` of them, picked at random with a fixed seed."""
    mask = in_view(points, cams)
    pts = np.asarray(points, dtype=np.float32)[mask]
    rgb = np.full((len(pts), 3), 128, np.uint8) if colors is None else np.asarray(colors, np.uint8)[mask]
    if len(pts) > most:
        pick = np.sort(np.random.default_rng(0).choice(len(pts), most, replace=False))
        pts, rgb = pts[pick], rgb[pick]
    return pts, rgb


def write_job(
    folder: Path,
    rev: places.Revision,
    frames: Mapping[str, Sequence[np.ndarray]],
    start: tuple[np.ndarray, np.ndarray],
    holdout: Sequence[int] = (),
) -> Path:
    """The runner's job for `rev` in `folder` (spatial/splat_runner.py says what it holds): each frame fitted to its
    camera's registered size and saved as PNG (a wide lens undone and its overlays painted over first, with their mask
    beside it, as spatial/lens.py does for MASt3R), and the start points as an npz. Returns the job file."""
    import cv2

    views = []
    for cam, shots in frames.items():
        pose = rev.cameras[cam]
        # The lens the registration undid, and the camera's overlays, as MASt3R's frame had them.
        undo = lens.Lens.from_tuple(pose.lens) if pose.lens is not None else None
        _, masks = lens.for_camera(lens.manifest(), cam)
        for i, frame in enumerate(shots):
            path = folder / f"{cam}-{i:02d}.png"
            view: dict[str, Any] = {}
            if undo is not None or masks:
                frame, covered = lens.prepare(frame, undo, masks)
                view["mask"] = str(folder / f"{cam}-{i:02d}.mask.png")
                cv2.imwrite(view["mask"], fit(covered.astype(np.uint8) * 255, pose.width, pose.height))
            cv2.imwrite(str(path), fit(frame, pose.width, pose.height))
            views.append(
                {
                    "camera": cam,
                    "image": str(path),
                    "pose": list(pose.pose),
                    "focal": pose.focal,
                    "width": pose.width,
                    "height": pose.height,
                    **view,
                }
            )
    np.savez(folder / "start.npz", points=start[0], colors=start[1])
    job = folder / "job.json"
    job.write_text(
        json.dumps(
            {
                "views": views,
                "init": str(folder / "start.npz"),
                "steps": STEPS,
                "cap": MAX_GAUSSIANS,
                "sh_degree": SH_DEGREE,
                "out": str(folder),
                "holdout": list(holdout),
            }
        )
    )
    return job


#: The depth prior (`depth_priors`): Depth Anything 3's metric model, run by spatial/depth_runner.py in gsplat's
#: runtime from the pinned code `mise run gsplat` puts under it.
DEPTH_RUNNER = Path(__file__).with_name("depth_runner.py")
DEPTH_SOURCE = Path("depth-anything-3") / "src"
DEPTH_TIMEOUT_S = 600.0
#: A camera's depth is scaled to the map by the map points it sees, the nearest in each DEPTH_CELL-pixel cell (a
#: coarse z-buffer, so a thin cloud isn't seen through); with fewer cells than this it keeps the model's own metres.
DEPTH_CELL = 4
MIN_DEPTH_CELLS = 200
DEPTH_UNIT = 0.001  # a depth image's count, in the map's units


def depth_checkpoint() -> Path:
    """Where `mise run models` puts Depth Anything 3's metric weights ([models.hf.depth_anything])."""
    return cfg.owner_file(str(cfg.load()["models"]["hf"]["depth_anything"]["path"])) / "model.safetensors"


def map_scale(depth: np.ndarray, points: np.ndarray, cam: places.CameraPose) -> float | None:
    """What a camera's model depth (`depth`, H x W along the view axis, the camera's registered size) is multiplied by
    to agree with the map: the median, over the cells that hold a map point `cam` sees, of the nearest such point's
    depth over the model's there. None with fewer than MIN_DEPTH_CELLS cells. The median shrugs off what the map
    lacks or the room has since gained (a box put down in front of the camera) and the holes a z-buffer of points
    sees through."""
    c2w = np.asarray(cam.pose, dtype=np.float64).reshape(4, 4)
    local = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - c2w[:3, 3]) @ c2w[:3, :3]
    local = local[local[:, 2] > MIN_DEPTH]
    u = np.floor(cam.focal * local[:, 0] / local[:, 2] + cam.width / 2).astype(np.int64)
    v = np.floor(cam.focal * local[:, 1] / local[:, 2] + cam.height / 2).astype(np.int64)
    inside = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    rows, cols = -(-cam.height // DEPTH_CELL), -(-cam.width // DEPTH_CELL)
    nearest = np.full((rows, cols), np.inf)
    np.minimum.at(nearest, (v[inside] // DEPTH_CELL, u[inside] // DEPTH_CELL), local[inside, 2])
    model = np.asarray(depth, dtype=np.float64)[DEPTH_CELL // 2 :: DEPTH_CELL, DEPTH_CELL // 2 :: DEPTH_CELL]
    seen = np.isfinite(nearest[: model.shape[0], : model.shape[1]]) & (model > 0)
    if int(seen.sum()) < MIN_DEPTH_CELLS:
        return None
    return float(np.median(nearest[: model.shape[0], : model.shape[1]][seen] / model[seen]))


def run_depth(runtime: Path, job: Path, gpu: str | None) -> None:
    """spatial/depth_runner.py on `job` in gsplat's runtime, niced, on `gpu`."""
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    command = ["nice", "-n", str(NICE), str(runtime / ".venv" / "bin" / "python"), str(DEPTH_RUNNER), str(job)]
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=DEPTH_TIMEOUT_S)
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1:] or ["(no output)"]
        raise RuntimeError(f"the depth model failed ({done.returncode}): {tail[0][:300]}")


def depth_priors(
    job: Path, rev: places.Revision, points: np.ndarray, runtime: Path, gpu: str | None
) -> dict[str, float | None]:
    """Give each view of `job` (`write_job`'s) a depth prior (#20) and return each camera's `map_scale` (None where
    the model's own metres were kept). A fixed camera sees one scene, so it gets one depth image: the model's depth
    of each of its frames (spatial/depth_runner.py), their per-pixel median (people passing and the light's changes
    drop out), scaled to the map, and zeroed where the camera's frame is covered (its overlays, and the border a
    lens's undoing leaves), as a 16-bit PNG in DEPTH_UNIT of the map's units. Raises RuntimeError when the model's
    code or weights aren't there, or it fails; the job is then left as it was."""
    import cv2

    source, checkpoint = runtime / DEPTH_SOURCE, depth_checkpoint()
    if not source.is_dir():
        raise RuntimeError(f"Depth Anything 3's code isn't in {source.parent}; run `mise run gsplat`")
    if not checkpoint.is_file():
        raise RuntimeError(f"Depth Anything 3's weights aren't at {checkpoint}; run `mise run models`")
    spec = json.loads(job.read_text())
    views = spec["views"]
    asked = [
        {"image": v["image"], "focal": v["focal"], "out": str(Path(v["image"]).with_suffix(".depth.npy"))}
        for v in views
    ]
    request = job.with_name("depth.json")
    request.write_text(json.dumps({"source": str(source), "checkpoint": str(checkpoint), "views": asked}))
    run_depth(runtime, request, gpu)
    scales: dict[str, float | None] = {}
    for cam in dict.fromkeys(v["camera"] for v in views):
        mine = [i for i, v in enumerate(views) if v["camera"] == cam]
        depth = np.median(np.stack([np.load(asked[i]["out"]) for i in mine]), axis=0)
        scales[cam] = map_scale(depth, points, rev.cameras[cam])
        if scales[cam] is None:
            LOG.info("%s: too few map points in view to scale its depth by; kept the model's metres", cam)
        counts = np.clip(np.round(depth * (scales[cam] or 1.0) / DEPTH_UNIT), 0, 65535).astype(np.uint16)
        mask = views[mine[0]].get("mask")
        covered = cv2.imread(mask, cv2.IMREAD_GRAYSCALE) if mask else None
        if covered is not None:
            counts[covered > 127] = 0
        path = job.with_name(f"{cam}.depth.png")
        cv2.imwrite(str(path), counts)
        for i in mine:
            views[i]["depth"] = str(path)
    spec["depth_unit"] = DEPTH_UNIT
    job.write_text(json.dumps(spec))
    return scales


def run_runner(runtime: Path, job: Path, gpu: str | None, timeout_s: float = RUN_TIMEOUT_S) -> dict[str, Any]:
    """spatial/splat_runner.py on `job` in gsplat's runtime, niced, on `gpu` (a CUDA_VISIBLE_DEVICES value); its
    result.json."""
    python = runtime / ".venv" / "bin" / "python"
    if not python.exists():
        raise RuntimeError(f"gsplat's runtime isn't installed in {runtime}; run `mise run gsplat`")
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    command = ["nice", "-n", str(NICE), str(python), str(RUNNER), str(job)]
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout_s)
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1:] or ["(no output)"]
        raise RuntimeError(f"the splat trainer failed ({done.returncode}): {tail[0][:300]}")
    return json.loads((job.parent / "result.json").read_text())


#: A walkthrough's splat (`splat_walk`): more views than a few fixed cameras, so more steps, a larger cap, and no
#: opacity or scale regularizers (on a walk they let the Gaussians fade until the splat goes transparent).
#: 15,000 steps is what walks were measured with (about 6 minutes for 125 frames at 288 x 512); a phone's frames are
#: saved larger (1024 on the long side), so a real walk takes several times that, well inside WALK_TIMEOUT_S.
WALK_STEPS = 15_000
WALK_GAUSSIANS = 1_000_000
WALK_START_POINTS = 500_000
#: A walk registers this many of its frames, evenly spread, as cameras to look from in the 3D view, named with this
#: prefix; they also carry the walk's frame through tier 1's fusing, so its views can be moved into the fused map's.
LOOK_FROM = 12
LOOK_PREFIX = "walk-"


def look_from(
    poses: np.ndarray,
    frames: Sequence[tuple[np.ndarray, tuple[int, int]]],
    times: Sequence[float],
    most: int = LOOK_FROM,
) -> dict[str, places.CameraPose]:
    """Up to `most` of a walk's frames (`poses` camera to world, `frames` each one's intrinsics and saved size, `times`
    in seconds), evenly spread, as cameras to look from: each named by its time into the walk ("walk-1m05s", a name
    Continuum's per-camera reads take: letters, digits, and hyphens), at its pose and its saved frame's size, with the
    mean of its fx and fy as the focal."""
    n = len(poses)
    count = min(most, n)
    picks = sorted({round(k * (n - 1) / max(1, count - 1)) for k in range(count)})
    out: dict[str, places.CameraPose] = {}
    for i in picks:
        k, (w, h) = frames[i]
        secs = int(times[i] - times[0])
        name = f"{LOOK_PREFIX}{secs // 60}m{secs % 60:02d}s"
        while name in out:
            name += "b"
        out[name] = places.CameraPose(
            tuple(float(x) for x in np.asarray(poses[i]).ravel()), float((k[0, 0] + k[1, 1]) / 2), int(w), int(h)
        )
    return out


def carried(
    before: Mapping[str, places.CameraPose], after: Mapping[str, places.CameraPose]
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """The similarity (rotation, shift, scale) that moved the cameras `before` to where `after` has them, over the
    names both have: the rotation from their axes (tier 1 moves a camera rigidly, so each says it exactly, and a
    straight walk still fixes the turn about its own line), the scale and shift from their centres. None when no name
    is in both."""
    names = [n for n in before if n in after]
    if not names:
        return None
    b = np.array([np.asarray(before[n].pose, dtype=np.float64).reshape(4, 4) for n in names])
    a = np.array([np.asarray(after[n].pose, dtype=np.float64).reshape(4, 4) for n in names])
    u, _, vt = np.linalg.svd(
        np.einsum("nij,nkj->ik", a[:, :3, :3], b[:, :3, :3])
    )  # the mean of a_R b_R^T, made a rotation
    rot = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)) or 1.0)]) @ vt
    cb, ca = b[:, :3, 3], a[:, :3, 3]
    db, da = cb - cb.mean(0), ca - ca.mean(0)
    spread = float((db**2).sum())
    scale = float(np.einsum("ni,ni->", da, db @ rot.T) / spread) if spread > 1e-12 else 1.0
    return rot, ca.mean(0) - scale * rot @ cb.mean(0), scale


def walk_views(
    images: Sequence[Path],
    poses: np.ndarray,
    frames: Sequence[tuple[np.ndarray, tuple[int, int]]],
    moved: tuple[np.ndarray, np.ndarray, float] | None = None,
    depths: Sequence[Path | None] = (),
) -> list[dict[str, Any]]:
    """The runner's views for a walk's saved frames (`images`) at their poses (`poses`, camera to world) and
    intrinsics and sizes (`frames`), moved by the similarity `moved` (rotation, shift, scale) when given, each with
    its depth map when `depths` has one (16-bit millimetres, `stray.depth_map`)."""
    rot, shift, scale = moved if moved is not None else (np.eye(3), np.zeros(3), 1.0)
    views = []
    measured = list(depths) or [None] * len(images)
    for image, pose, (k, (w, h)), depth in zip(images, poses, frames, measured, strict=True):
        c2w = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        m = np.eye(4)
        m[:3, :3] = rot @ c2w[:3, :3]
        m[:3, 3] = scale * rot @ c2w[:3, 3] + shift
        views.append(
            {
                "image": str(image),
                "pose": m.ravel().tolist(),
                "focal": float((k[0, 0] + k[1, 1]) / 2),
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
                "width": int(w),
                "height": int(h),
                **({} if depth is None else {"depth": str(depth)}),
            }
        )
    return views


def train_walk(
    views: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    scale: float = 1.0,
    refine: bool = True,
    holdout: Sequence[int] = (),
    steps: int = WALK_STEPS,
) -> tuple[dict[str, Any], bytes]:
    """Train a walk's splat from `views` (`walk_views`), starting from `points` and `colors` (at most
    WALK_START_POINTS of them, picked with a fixed seed), the Gaussians held inside the points' box, with each
    training view's pose refined while it trains unless `refine` is off (spatial/splat_runner.py's `pose_opt`; off
    is for measuring what it gains) and the views at `holdout` scored, not trained on. `scale` is the map's units to
    a metre (`carried`'s), for the views' depth maps. Returns the runner's result and the trained PLY. Raises
    RuntimeError when training fails."""
    runtime = cfg.path("spatial.splat_runtime")
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.full((len(pts), 3), 128, np.uint8) if colors is None else np.asarray(colors, np.uint8)
    if len(pts) > WALK_START_POINTS:
        pick = np.sort(np.random.default_rng(0).choice(len(pts), WALK_START_POINTS, replace=False))
        pts, rgb = pts[pick], rgb[pick]
    gpu = cameras.pick_gpu(int(cfg.get("spatial.gpu")), nvml.gpus())
    with tempfile.TemporaryDirectory(prefix="spatial-splat-walk-") as tmp:
        folder = Path(tmp)
        np.savez(folder / "start.npz", points=pts, colors=rgb)
        job = folder / "job.json"
        job.write_text(
            json.dumps(
                {
                    "views": list(views),
                    "init": str(folder / "start.npz"),
                    "steps": steps,
                    "cap": WALK_GAUSSIANS,
                    "sh_degree": SH_DEGREE,
                    "out": str(folder),
                    "reg": 0.0,
                    "bound": True,
                    "pose_opt": refine,
                    "depth_unit": 0.001 * scale,  # a depth map's millimetres in the map's units
                    "holdout": list(holdout),
                }
            )
        )
        result = run_runner(runtime, job, gpu, timeout_s=WALK_TIMEOUT_S)
        return {**result, "start_points": len(pts)}, (folder / "splat.ply").read_bytes()


def splat_walk(
    state: Path,
    place: str,
    revision: str,
    images: Sequence[Path],
    poses: np.ndarray,
    frames: Sequence[tuple[np.ndarray, tuple[int, int]]],
    looks: Mapping[str, places.CameraPose],
    depths: Sequence[Path | None] = (),
) -> dict[str, Any] | None:
    """Train and keep a splat for a walk's revision (`place`'s `revision`, what tier 1 made of it) from every saved
    frame (`images`) at its pose (`poses`, the walk's frame) and intrinsics and size (`frames`), moved into the
    revision's map frame by the similarity tier 1 moved the walk's `looks` by (`carried`), starting from the
    revision's map points (at most WALK_START_POINTS); what it did. Every view trains, none held out, each against
    its LiDAR depth map when `depths` has one (a Stray walk's; a plain video has none), with its pose refined as it
    trains: that takes up the last few centimetres of a Stray walk's ARKit poses and the larger errors of a plain
    video's CUT3R and pose-graph poses (#21). None (the log says why) when splats are off, gsplat's runtime isn't
    installed, the frame can't be carried, or training fails."""
    runtime = cfg.path("spatial.splat_runtime")
    if not cfg.get("spatial.splats_enabled") or not (runtime / ".venv" / "bin" / "python").exists():
        LOG.info("no splat of the walk: splats are off or gsplat's runtime isn't installed (`mise run gsplat`)")
        return None
    rev = next((r for r in places.revisions(state, place) if r.id == revision), None)
    moved = carried(looks, rev.cameras) if rev is not None and rev.map is not None else None
    if rev is None or moved is None:
        LOG.warning("no splat of the walk: revision %s of %s has no map or none of the walk's cameras", revision, place)
        return None
    views = walk_views(images, poses, frames, moved, depths)
    points, colors, _ = ply.read(places.root(state) / rev.place / str(rev.map))
    try:
        result, master = train_walk(views, points, colors, scale=moved[2])
        rows = ply.vertices(master)
        if rows is None:
            raise RuntimeError("the splat trainer wrote a PLY this reader can't take")
        web = sog.encode(rows)
        places.write_splat(state, rev, master, web, source="walk")
    except (RuntimeError, subprocess.TimeoutExpired, OSError, ValueError) as e:
        LOG.error("no splat of the walk: %s", e)
        return None
    did = {
        "revision": rev.id,
        "views": len(views),
        "start_points": result.get("start_points"),
        "gaussians": result.get("gaussians"),
        "dropped": result.get("dropped"),
        "scale_p99": result.get("scale_p99"),
        "train_psnr": result.get("train_psnr"),
        "pose_shift": result.get("pose_shift"),
        "pose_turn_deg": result.get("pose_turn_deg"),
        "seconds": result.get("seconds"),
        "peak_vram_gib": result.get("peak_vram_gib"),
        "sog_mb": round(len(web) / 2**20, 1),
    }
    LOG.info("splat of the walk: %s", json.dumps(did))
    return did


def splat_place(
    state: Path, place: str, now: float, read: Reader, runtime: Path, gpu: str | None
) -> dict[str, Any] | None:
    """Train and keep a splat for `place`'s `target` revision; what it did, or None (said why in the log) when
    there's nothing to do: no such revision, it has a splat already, or no camera has a lit frame."""
    rev = target(state, place)
    if rev is None:
        LOG.info("%s: no map with registered cameras to splat", place)
        return None
    if places.has_splat(state, rev):
        LOG.info("%s: revision %s has its splat", place, rev.id)
        return None
    from liberational.memory.history_qa import local_zone  # the journal's clock

    fixed = sorted(c for c in rev.cameras if not c.startswith(LOOK_PREFIX))  # a walk's views have no recordings
    if not fixed:
        LOG.info("%s: revision %s has no fixed cameras to splat", place, rev.id)
        return None
    found = day_frames(fixed, now, read, local_zone())
    frames = {c: f for c, f in found.items() if len(f) >= MIN_FRAMES_PER_CAMERA}
    if not frames:
        have = ", ".join(f"{c} {len(found.get(c, []))}" for c in fixed)
        LOG.info(
            "%s: too few lit daytime frames in the last day for a splat (%s; needs %d)",
            place,
            have,
            MIN_FRAMES_PER_CAMERA,
        )
        return None
    if len(frames) < len(found):
        LOG.info(
            "%s: %s left out of the splat: under %d lit frames",
            place,
            ", ".join(sorted(set(found) - set(frames))),
            MIN_FRAMES_PER_CAMERA,
        )
    points, colors, _ = ply.read(places.root(state) / place / str(rev.map))
    start = start_points(points, colors, {c: rev.cameras[c] for c in frames})
    if not len(start[0]):
        LOG.warning("%s: no map point is in any camera's view; is the map in the cameras' frame?", place)
        return None
    with tempfile.TemporaryDirectory(prefix="spatial-splat-") as tmp:
        job = write_job(Path(tmp), rev, frames, start)
        scales: dict[str, float | None] | None = None
        try:
            scales = depth_priors(job, rev, points, runtime, gpu)
        except (RuntimeError, subprocess.TimeoutExpired, OSError, ValueError, KeyError) as e:
            LOG.warning("%s: training without the depth prior: %s", place, e)
        result = run_runner(runtime, job, gpu)
        master = (Path(tmp) / "splat.ply").read_bytes()
    rows = ply.vertices(master)
    if rows is None:
        raise RuntimeError("the splat trainer wrote a PLY this reader can't take")
    web = sog.encode(rows)
    places.write_splat(state, rev, master, web, source="cameras")
    return {
        "place": place,
        "revision": rev.id,
        "frames": {c: len(f) for c, f in frames.items()},
        "start_points": len(start[0]),
        "depth_scales": None if scales is None else {c: s and round(s, 3) for c, s in scales.items()},
        "gaussians": result.get("gaussians"),
        "train_psnr": result.get("train_psnr"),
        "seconds": result.get("seconds"),
        "peak_vram_gib": result.get("peak_vram_gib"),
        "ply_mb": round(len(master) / 2**20, 1),
        "sog_mb": round(len(web) / 2**20, 1),
    }


def recode(state: Path, place: str) -> int:
    """Give `place`'s kept splats that have a PLY but no SOG their SOG, from the PLY, and remove the SPZ copy they
    had before the 3D view moved to SOG (#76); how many it did. A PLY that doesn't read is logged and left."""
    done = 0
    for rev in places.revisions(state, place):
        master = places.splat_path(state, rev, "gs.ply")
        if places.has_splat(state, rev) or not master.is_file():
            continue
        try:
            rows = ply.vertices(master.read_bytes())
        except ply.PlyError:
            rows = None
        if rows is None:
            LOG.warning("%s: revision %s's splat PLY doesn't read; left without a SOG", place, rev.id)
            continue
        places.write_splat_copy(state, rev, sog.encode(rows))
        master.with_name(f"{rev.id}.spz").unlink(missing_ok=True)
        LOG.info("%s: revision %s's splat has its SOG copy", place, rev.id)
        done += 1
    return done


def splat_all(
    state: Path, now: float | None = None, only: str | None = None, read: Reader = cameras.recorded
) -> list[dict[str, Any]]:
    """`splat_place` for every place (or `only` that one); what each did. A place whose training fails is logged and
    skipped. Nothing runs without gsplat's runtime: that's said once, not per place."""
    for place in [only] if only else places.place_ids(state):
        recode(state, place)  # needs no trainer
    runtime = cfg.path("spatial.splat_runtime")
    if not (runtime / ".venv" / "bin" / "python").exists():
        LOG.info("no splats: gsplat's runtime isn't installed in %s (`mise run gsplat`)", runtime)
        return []
    now = time.time() if now is None else now
    gpu = cameras.pick_gpu(int(cfg.get("spatial.gpu")), nvml.gpus())
    done = []
    for place in [only] if only else places.place_ids(state):
        try:
            did = splat_place(state, place, now, read, runtime, gpu)
        except (RuntimeError, subprocess.TimeoutExpired, OSError, ValueError) as e:
            LOG.error("%s: no splat: %s", place, e)
            continue
        if did is not None:
            LOG.info("splat: %s", json.dumps(did))
            done.append(did)
    return done


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pack-spatial-splat", description=__doc__.split("\n\n")[0])
    parser.add_argument("--place", help="only this place's id (default: every place)")
    args = parser.parse_args(argv)
    logs.setup()
    if sys.platform == "darwin":
        LOG.info("splats need CUDA; skipping on macOS")
        return
    began = time.monotonic()
    did = splat_all(cfg.path("paths.state"), only=args.place)
    LOG.info("splats: %d made in %.0f s", len(did), time.monotonic() - began)


if __name__ == "__main__":
    main()
