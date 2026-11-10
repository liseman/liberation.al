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
3. **Train.** spatial/splat_runner.py in gsplat's own runtime (`mise run gsplat`, under `spatial.splat_runtime`),
   niced, on the GPU with the most free memory (`cameras.pick_gpu`): `STEPS` steps, at most `MAX_GAUSSIANS`.
4. **Keep.** The trained 3DGS PLY and its SPZ copy (`spz`) are written beside the revision (memory/places.py
   `write_splat`), and served by the dashboard's `GET /places/splat`. The PLY is in the map's frame, as is; the
   SPZ is a standard one, in SPZ's right-up-back frame (y and z negated, as Niantic's converter makes it from a PLY).
   Spark (the 3D view's renderer) reads an SPZ as it is, so the view turns the splat 180 degrees about x, which
   undoes that negation exactly (positions, rotations, and SH), to put it back on the map.

Measured with these settings (Place 1: two fixed cameras, 10 daytime frames each, 21k start points, one RTX
3090 shared with the live services): about 2 to 2.5 minutes and 0.2 GiB of GPU memory a place, 244k Gaussians, a 24 MB
PLY and a 4.4 MB SPZ, and 27.1 dB PSNR on a frame per camera held out from the same day. Two cameras with little overlap give no new views: the splat looks right from where the cameras
stand, not from between them (6.6 dB from one camera's view trained on the other's). Nothing fills what no camera
sees.

**A Stray Scanner walkthrough** (`splat_walk`) gets its splat when it's imported: `pack-walkthrough` calls it once the
walk's revision is written and fused (spatial/walkthrough.py `run_stray`). Every saved frame trains, at its ARKit pose
and the phone's own intrinsics, starting from the revision's map points: `WALK_STEPS` steps, at most `WALK_GAUSSIANS`,
and no opacity or scale regularizers, so the Gaussians are held inside the map's box and under a size instead
(spatial/splat_runner.py `bound`, #43: without it one walk's splat streaked out to 1.9 km). The walk registers `LOOK_FROM` of its frames as cameras (`look_from`, named
`walk-...`), for the 3D view to look from; tier 1 moves them with the walk's cloud into the fused map's frame, and the
move they show (`carried`) brings every frame's pose along. The daily fixed-camera splat leaves those cameras out. A
walk filmed with CUT3R's poses gets no splat: its poses didn't agree across the walk well enough for one.

gsplat (nerfstudio-project/gsplat) is Apache-2.0; the trainer adds no model weights.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
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
from liberational.spatial import cameras, lens, ply

LOG = logs.get("spatial-splat")

RUNNER = Path(__file__).with_name("splat_runner.py")
FRAMES_PER_CAMERA = 12
#: A camera with fewer lit frames than this takes no part, and a place with no camera left gets no splat this run (the
#: next run's day will have more): after a reset at midday, Place 1's splat trained on 4 frames a camera and showed
#: in place of a 27 dB one from 10.
MIN_FRAMES_PER_CAMERA = 6
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


# SPZ (nianticlabs/spz, MIT): version 3, the gzip layout every SPZ reader takes (version 4 moved to zstd streams).
SPZ_MAGIC = 0x5053474E  # "NGSP"
SPZ_VERSION = 3
SPZ_FRACTIONAL_BITS = 12  # positions to 1/4096 m
SPZ_COLOR_SCALE = 0.15
SPZ_SH1_BUCKET = 8  # SH band 1 kept to 5 bits, the rest to 4: the library's defaults
SPZ_SH_REST_BUCKET = 16
#: SPZ stores right-up-back (three.js's frame); a PLY is right-down-front. Niantic's converter negates y and z, so
#: the rotation's x, y, z by (y*z, x*z, x*y), and each SH coefficient by the sign these give (its `flipSh`, with
#: x = 1, y = z = -1), bands 1 to 3 in order.
SPZ_FLIP_P = np.array([1.0, -1.0, -1.0])
SPZ_FLIP_Q = np.array([1.0, -1.0, -1.0])
SPZ_FLIP_SH = np.array([-1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1], dtype=np.float64)


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _smallest_three(q: np.ndarray) -> np.ndarray:
    """Unit quaternions (N x 4, x y z w) as SPZ 3's 4 bytes each: the largest component's index in the top 2 bits,
    then each other component, in order, as a sign bit and 9 bits of its magnitude over 1/sqrt(2), the largest made
    positive (q and -q are the same rotation)."""
    largest = np.argmax(np.abs(q), axis=1)
    negate = q[np.arange(len(q)), largest] < 0
    comp = largest.astype(np.uint32)
    for i in range(4):
        keep = largest != i
        neg = ((q[:, i] < 0) ^ negate).astype(np.uint32)
        mag = np.clip(np.floor(511.0 * np.abs(q[:, i]) / math.sqrt(0.5) + 0.5), 0, 511).astype(np.uint32)
        comp = np.where(keep, (comp << 10) | (neg << 9) | mag, comp)
    return comp.astype("<u4").view(np.uint8).reshape(-1, 4)


def spz(rows: np.ndarray) -> bytes:
    """A 3DGS PLY's vertices (`ply.vertices`: x, y, z, f_dc_*, f_rest_* channel by channel, opacity as a logit, log
    scale_*, and rot_0..3 as w, x, y, z) as an SPZ file, converted from the PLY's frame to SPZ's (`SPZ_FLIP_*`)."""
    names = rows.dtype.names or ()
    n = len(rows)
    rest = sorted((k for k in names if k.startswith("f_rest_")), key=lambda k: int(k.rsplit("_", 1)[1]))
    coeffs = len(rest) // 3  # SH coefficients past band 0, per channel
    degree = round(math.sqrt(coeffs + 1)) - 1
    if (degree + 1) ** 2 - 1 != coeffs or degree > 3:
        raise ValueError(f"{len(rest)} f_rest properties isn't SH degree 0 to 3")

    def col(*keys: str) -> np.ndarray:
        return np.stack([np.asarray(rows[k], np.float64) for k in keys], axis=1)

    fixed = np.round(col("x", "y", "z") * SPZ_FLIP_P * (1 << SPZ_FRACTIONAL_BITS))
    fixed = np.clip(fixed, -(1 << 23), (1 << 23) - 1).astype("<i4")
    positions = fixed.view(np.uint8).reshape(n, 3, 4)[:, :, :3]
    alphas = _u8(255.0 / (1.0 + np.exp(-np.asarray(rows["opacity"], np.float64))))
    colors = _u8(col("f_dc_0", "f_dc_1", "f_dc_2") * (SPZ_COLOR_SCALE * 255.0) + 127.5)
    scales = _u8((col("scale_0", "scale_1", "scale_2") + 10.0) * 16.0)
    wxyz = col("rot_0", "rot_1", "rot_2", "rot_3")
    xyzw = np.concatenate([wxyz[:, 1:] * SPZ_FLIP_Q, wxyz[:, :1]], axis=1)
    xyzw /= np.maximum(np.linalg.norm(xyzw, axis=1, keepdims=True), 1e-12)
    rotations = _smallest_three(xyzw)
    # The PLY has each channel's coefficients together; SPZ has each coefficient's channels together.
    flat = col(*rest) if rest else np.zeros((n, 0))
    sh = flat.reshape(n, 3, coeffs).transpose(0, 2, 1) * SPZ_FLIP_SH[:coeffs, None]
    bucket = np.where(np.arange(coeffs) < 3, SPZ_SH1_BUCKET, SPZ_SH_REST_BUCKET)[None, :, None]
    q = np.clip(np.round(sh * 128.0) + 128.0, 0, 255).astype(np.int32)
    sh_bytes = np.clip((q + bucket // 2) // bucket * bucket, 0, 255).astype(np.uint8)
    header = np.array([SPZ_MAGIC, SPZ_VERSION, n], "<u4").tobytes()
    header += bytes([degree, SPZ_FRACTIONAL_BITS, 0, 0])
    body = b"".join(a.tobytes() for a in (positions, alphas, colors, scales, rotations, sh_bytes))
    return gzip.compress(header + body, compresslevel=9, mtime=0)


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


def splat_walk(
    state: Path,
    place: str,
    revision: str,
    images: Sequence[Path],
    poses: np.ndarray,
    frames: Sequence[tuple[np.ndarray, tuple[int, int]]],
    looks: Mapping[str, places.CameraPose],
) -> dict[str, Any] | None:
    """Train and keep a splat for a Stray walk's revision (`place`'s `revision`, what tier 1 made of it) from every
    saved frame (`images`) at its pose (`poses`, the walk's frame) and intrinsics and size (`frames`), moved into the
    revision's map frame by the similarity tier 1 moved the walk's `looks` by (`carried`), starting from the
    revision's map points (at most WALK_START_POINTS); what it did. Every view trains, none held out. None (the log
    says why) when splats are off, gsplat's runtime isn't installed, the frame can't be carried, or training fails.
    ARKit's poses are metric and tracked, so unlike a CUT3R walk's they agree well enough across the walk."""
    runtime = cfg.path("spatial.splat_runtime")
    if not cfg.get("spatial.splats_enabled") or not (runtime / ".venv" / "bin" / "python").exists():
        LOG.info("no splat of the walk: splats are off or gsplat's runtime isn't installed (`mise run gsplat`)")
        return None
    rev = next((r for r in places.revisions(state, place) if r.id == revision), None)
    moved = carried(looks, rev.cameras) if rev is not None and rev.map is not None else None
    if rev is None or moved is None:
        LOG.warning("no splat of the walk: revision %s of %s has no map or none of the walk's cameras", revision, place)
        return None
    rot, shift, scale = moved
    views = []
    for image, pose, (k, (w, h)) in zip(images, poses, frames, strict=True):
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
            }
        )
    points, colors, _ = ply.read(places.root(state) / rev.place / str(rev.map))
    pts = np.asarray(points, dtype=np.float32)
    rgb = np.full((len(pts), 3), 128, np.uint8) if colors is None else np.asarray(colors, np.uint8)
    if len(pts) > WALK_START_POINTS:
        pick = np.sort(np.random.default_rng(0).choice(len(pts), WALK_START_POINTS, replace=False))
        pts, rgb = pts[pick], rgb[pick]
    gpu = cameras.pick_gpu(int(cfg.get("spatial.gpu")), nvml.gpus())
    try:
        with tempfile.TemporaryDirectory(prefix="spatial-splat-walk-") as tmp:
            folder = Path(tmp)
            np.savez(folder / "start.npz", points=pts, colors=rgb)
            job = folder / "job.json"
            job.write_text(
                json.dumps(
                    {
                        "views": views,
                        "init": str(folder / "start.npz"),
                        "steps": WALK_STEPS,
                        "cap": WALK_GAUSSIANS,
                        "sh_degree": SH_DEGREE,
                        "out": str(folder),
                        "reg": 0.0,
                        "bound": True,
                    }
                )
            )
            result = run_runner(runtime, job, gpu, timeout_s=WALK_TIMEOUT_S)
            master = (folder / "splat.ply").read_bytes()
        rows = ply.vertices(master)
        if rows is None:
            raise RuntimeError("the splat trainer wrote a PLY this reader can't take")
        web = spz(rows)
        places.write_splat(state, rev, master, web)
    except (RuntimeError, subprocess.TimeoutExpired, OSError, ValueError) as e:
        LOG.error("no splat of the walk: %s", e)
        return None
    did = {
        "revision": rev.id,
        "views": len(views),
        "start_points": len(pts),
        "gaussians": result.get("gaussians"),
        "dropped": result.get("dropped"),
        "scale_p99": result.get("scale_p99"),
        "train_psnr": result.get("train_psnr"),
        "seconds": result.get("seconds"),
        "peak_vram_gib": result.get("peak_vram_gib"),
        "spz_mb": round(len(web) / 2**20, 1),
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
        result = run_runner(runtime, job, gpu)
        master = (Path(tmp) / "splat.ply").read_bytes()
    rows = ply.vertices(master)
    if rows is None:
        raise RuntimeError("the splat trainer wrote a PLY this reader can't take")
    web = spz(rows)
    places.write_splat(state, rev, master, web)
    return {
        "place": place,
        "revision": rev.id,
        "frames": {c: len(f) for c, f in frames.items()},
        "start_points": len(start[0]),
        "gaussians": result.get("gaussians"),
        "train_psnr": result.get("train_psnr"),
        "seconds": result.get("seconds"),
        "peak_vram_gib": result.get("peak_vram_gib"),
        "ply_mb": round(len(master) / 2**20, 1),
        "spz_mb": round(len(web) / 2**20, 1),
    }


def splat_all(
    state: Path, now: float | None = None, only: str | None = None, read: Reader = cameras.recorded
) -> list[dict[str, Any]]:
    """`splat_place` for every place (or `only` that one); what each did. A place whose training fails is logged and
    skipped. Nothing runs without gsplat's runtime: that's said once, not per place."""
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
