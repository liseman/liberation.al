#!/usr/bin/env python3
"""Fit each fixed camera's own intrinsics (focal and lens) once, to hold them (#3289; spatial/lens.py `Held`).

A camera's focal and lens are its own, true wherever it stands, so every registration after this keeps them and fits
only the camera's pose. They're fitted here against a Stray Scanner (LiDAR) walk: each camera's frame is paired by
MASt3R with the walk's frames, and spatial/registration.py's reprojection fit puts the walk frame's lifted points on
the camera's matched pixels with the phone's frame as the reference (a sharp pinhole whose focal ARKit gives), so the
camera's focal and division lens come out of many pairs, of which the median is held. A camera whose configured lens
(`devices.toml`) undoes its frames first holds only its focal. A camera no walk frame pairs with takes the median of
its own nightly fits over the place's revisions. Each pair also gives the camera's tilt in the map (the walk frames'
poses are known there), which is printed beside the nightly registration's, as a check.

Dry run (the default): prints each camera's fit, then tonight's camera pass seated into a throwaway copy of the state
twice (spatial/dryrun.py), as the nightly run would with today's intrinsics and with the held ones, with today's
position rows that get no floor point and how far apart the two living-room cameras put one person at the same
moment, each way. It saves the fits to `--save`; `--load <file> --write` holds exactly those.

    .venv/bin/python tools/fit_intrinsics.py --place <place id or name> --walk /path/to/stray/export \\
        [--walk-revision <id>] [--pair <camera>:<camera>] [--save <file.json>]
    .venv/bin/python tools/fit_intrinsics.py --place <place> --load <file.json> --write

Run it from the live checkout; it runs MASt3R on the GPU (a few minutes for a two-minute walk and four cameras).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import cameras, dryrun, footprints, lens, registration, splats, stray
from liberational.spatial import walkthrough as W

LOOK_PREFIX = splats.LOOK_PREFIX
MIN_PAIRS = 3  # pairs a camera's walk fit needs to be held; fewer, and its nightly fits' median is held
HISTORY = 30  # the place's newest revisions whose fits of a camera make its nightly median


@dataclasses.dataclass(frozen=True)
class Tilt:
    pitch: float  # degrees below level its view points
    roll: float  # degrees its frame's right side is turned down


def tilt(rotation: np.ndarray) -> Tilt:
    """A camera's tilt from its rotation (camera to map, columns right, down, forward; the map's z up)."""
    forward, right = rotation[:, 2], rotation[:, 0]
    return Tilt(math.degrees(math.asin(-float(forward[2]))), math.degrees(math.asin(-float(right[2]))))


@dataclasses.dataclass(frozen=True)
class WalkFrames:
    images: dict[str, Path]  # name to saved frame
    poses: dict[str, np.ndarray]  # name to its pose in the place's map
    focals: dict[str, float]  # name to its focal in pixels at MASt3R's size


def walk_frames(folder: Path, out: Path, looks: Mapping[str, places.CameraPose]) -> WalkFrames:
    """The walk's frames as pack-walkthrough picks them, saved under `out`, each posed in the place's map: its ARKit
    pose carried by the nearest look the place's map registers (`looks`, the walk's look-from views as the map has
    them, `splats.look_from`), so the walk's drift between looks doesn't reach it."""
    import cv2

    video = folder / "rgb.mp4"
    rows = stray.read_odometry(folder)
    scored = [s for s in W.score(video) if s.index in rows]
    picked = sorted(W.pick(scored))
    turn, code = stray.upright(np.stack([rows[i].c2w for i in picked]))
    c2w = [stray.turned(rows[i].c2w, turn) for i in picked]
    cap = cv2.VideoCapture(str(video))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    frames = [stray.saved_intrinsics(rows[i].k, size, W.FRAME_SIDE, code, turn) for i in picked]
    times = [rows[i].t for i in picked]
    named = splats.look_from(np.stack(c2w), frames, times)
    n = len(picked)
    count = min(splats.LOOK_FROM, n)
    at = sorted({round(k * (n - 1) / max(1, count - 1)) for k in range(count)})
    carry = {
        k: np.asarray(looks[name].pose, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(c2w[k])
        for k, name in zip(at, named, strict=True)
        if name in looks
    }
    if not carry:
        sys.exit("none of the walk's looks is in the place's map: is it the walk the map was built from?")
    out.mkdir(parents=True, exist_ok=True)
    images = W.save_frames(video, picked, out, turn=code)
    poses, focals = {}, {}
    for j in range(n):
        nearest = min(carry, key=lambda k: abs(k - j))
        name = f"f{j:03d}"
        poses[name] = carry[nearest] @ c2w[j]
        k_saved, (w, h) = frames[j]
        focals[name] = float(k_saved[0, 0]) * cameras.MAST3R_SIZE / max(w, h)
    return WalkFrames(images, poses, focals)


@dataclasses.dataclass(frozen=True)
class PairFit:
    walk: str
    focal: float  # share of the camera's frame width
    division: float
    residual: float
    tilt: Tilt


def pair_fits(
    walk: WalkFrames, shots: Mapping[str, Path], current: Mapping[str, places.CameraPose], out: Path
) -> dict[str, list[PairFit]]:
    """Each camera's frame paired with the walk's frames by MASt3R (every JOIN_STRIDE-th, then the ones next to a
    sure pair, as `walkthrough.seat_places` pairs them), each sure pair fitted (`registration.fit`) with the walk frame
    as the reference."""
    images = {**walk.images, **shots}
    names = sorted(walk.images)
    gpu = W.gpu_with_room()

    def run(wanted: Sequence[tuple[str, str]], where: str) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
        if not wanted:
            return [], {}
        answer = cameras.run_mast3r(
            cfg.path("spatial.runtime"),
            images,
            out / where,
            gpu,
            pairs=wanted,
            nice=10,
            timeout_s=max(cameras.RUN_TIMEOUT_S, W.PAIR_S * len(wanted)),
            save_pairs=True,
        )
        with np.load(out / where / answer["pair_file"]) as saved:
            return list(answer["pairs"]), {k: saved[k] for k in saved.files}

    first = [(c, names[k]) for c in sorted(shots) for k in range(0, len(names), W.JOIN_STRIDE)]
    measured, arrays = run(first, "pairs")
    index = {n: k for k, n in enumerate(names)}
    near = {
        (m["a"], names[k])
        for m in measured
        if int(m["matches"]) >= W.JOIN_MATCHES
        for d in range(-W.JOIN_NEAR, W.JOIN_NEAR + 1)
        if 0 <= (k := index[m["b"]] + d) < len(names)
    }
    more, more_arrays = run(sorted(near - set(first)), "pairs-near")
    measured += more
    arrays.update(more_arrays)
    out_fits: dict[str, list[PairFit]] = {c: [] for c in shots}
    for m in measured:
        cam, frame = str(m["a"]), str(m["b"])
        if m.get("forward") is None or int(m["matches"]) < registration.MIN_PAIR_MATCHES:
            continue
        key = f"{frame}__{cam}__"
        if key + "own" not in arrays:
            continue
        pair = registration.Pair(*(arrays[key + f] for f in registration.PAIR_FIELDS))
        ref_img, img = arrays[f"{frame}__image"], arrays[f"{cam}__image"]
        ref_size, size = (ref_img.shape[1], ref_img.shape[0]), (img.shape[1], img.shape[0])
        forward = np.asarray(m["forward"], dtype=np.float64).reshape(4, 4)  # the walk frame in the camera's
        now = current[cam]
        guess = (now.lens[0] if now.lens is not None else now.focal / now.width) * size[0]
        ref_f = walk.focals[frame]
        configured = lens.as_registered(cam) is not None
        got = registration.fit(pair, ref_size, size, forward[:3, :3], (ref_f, guess), not configured, ref_held=ref_f)
        if got is None:
            continue
        rotation = walk.poses[frame][:3, :3] @ got.rotation.T
        out_fits[cam].append(PairFit(frame, got.focal / size[0], got.division, got.residual, tilt(rotation)))
    return out_fits


def history(chain: Sequence[places.Revision], cam: str) -> list[tuple[float, float]]:
    """(focal as a share of the width, division) of `cam` in each of the newest HISTORY revisions that register it,
    as each night's fit left it: a fitted lens's focal is its `lens` tuple's (the stored focal is the undone frame's);
    a camera whose configured lens undoes its frames has division 0."""
    out = []
    for rev in [r for r in chain if cam in r.cameras][-HISTORY:]:
        pose = rev.cameras[cam]
        if pose.lens is None or lens.as_registered(cam) is not None:
            out.append((pose.focal / pose.width, 0.0))
        else:
            out.append((float(pose.lens[0]), float(pose.lens[4])))
    return out


def choose(cam: str, fits: Sequence[PairFit], past: Sequence[tuple[float, float]]) -> lens.Held | None:
    if len(fits) >= MIN_PAIRS:
        return lens.Held(
            float(np.median([f.focal for f in fits])),
            float(np.median([f.division for f in fits])),
            f"walk fit, {len(fits)} pairs",
            float(np.median([f.residual for f in fits])),
        )
    if past:
        return lens.Held(
            float(np.median([p[0] for p in past])),
            float(np.median([p[1] for p in past])),
            f"median of {len(past)} nightly fits",
        )
    print(f"{cam}: no walk pair and no nightly fit; nothing to hold")
    return None


def report(cam: str, fits: Sequence[PairFit], past: Sequence[tuple[float, float]], now: places.CameraPose) -> None:
    if fits:
        f = np.array([p.focal for p in fits])
        d = np.array([p.division for p in fits])
        pitch = np.array([p.tilt.pitch for p in fits])
        roll = np.array([p.tilt.roll for p in fits])
        print(
            f"  {cam}: {len(fits)} walk pairs: focal {np.median(f):.3f} (IQR {np.subtract(*np.percentile(f, [75, 25])):.3f}), "
            f"lens {np.median(d):+.3f} (IQR {np.subtract(*np.percentile(d, [75, 25])):.3f}), "
            f"{np.median([p.residual for p in fits]):.1f} px; tilt by the walk: pitch {np.median(pitch):.1f} "
            f"(IQR {np.subtract(*np.percentile(pitch, [75, 25])):.1f}), roll {np.median(roll):.1f}"
        )
    else:
        print(f"  {cam}: no walk pair held")
    if past:
        f, d = np.array([p[0] for p in past]), np.array([p[1] for p in past])
        print(
            f"    nightly fits ({len(past)}): focal {np.median(f):.3f} ({f.min():.3f} to {f.max():.3f}), lens "
            f"{np.median(d):+.3f} ({d.min():+.3f} to {d.max():+.3f})"
        )
    t = tilt(np.asarray(now.pose, dtype=np.float64).reshape(4, 4)[:3, :3])
    print(f"    registered now: pitch {t.pitch:.1f}, roll {t.roll:.1f}")


def measure(
    state: Path, place: str, pair: tuple[str, str], capture: cameras.Capture, out: Path, held: Mapping[str, lens.Held]
) -> None:
    """Tonight's camera pass seated into a throwaway state, with today's intrinsics and with `held`, and today's rows."""
    day = dt.date.today()
    start = dt.datetime.combine(day, dt.time()).astimezone().timestamp()
    end = time.time()
    now_rows = dryrun.rows(state, place, start, end)
    print(f"{day} rows now (on the plan that holds): {_line(now_rows, pair)}")
    for label, holding in (("tonight's own intrinsics", {}), ("held intrinsics", held)):
        with tempfile.TemporaryDirectory(prefix="fit-intrinsics-") as tmp:
            shadow = dryrun.overlay(state, Path(tmp), place)
            for cam, h in holding.items():
                lens.hold(shadow, cam, h)
            cameras.write_places(
                shadow,
                capture.result,
                out,
                end,
                capture.absent,
                lambda group, s=shadow: footprints.walked(s, group, end),
                capture.gravity,
            )
            newest = places.latest(shadow, place)
            found = dryrun.rows(shadow, place, start, end, newest.id if newest else None)
            print(f"  seated with {label}: {_line(found, pair)}")


def _line(found: Sequence[Any], pair: tuple[str, str]) -> str:
    counts = dryrun.rooms(found)
    agree = dryrun.agreement(found, *pair)
    together = (
        f"{pair[0]} and {pair[1]} put one person {agree.median:.2f} m apart (p90 {agree.p90:.2f}) over {agree.pairs} moments"
        if agree.pairs
        else f"no moment both {pair[0]} and {pair[1]} placed one person"
    )
    return f"{len(found)} rows, {counts[dryrun.NO_POINT]} with no floor point; {together}"


def _place(state: Path, wanted: str) -> places.Place:
    for p in places.places(state):
        if wanted in (p.id, p.name):
            return p
    sys.exit(f"no place {wanted!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", required=True, help="place id or name")
    ap.add_argument("--walk", type=Path, help="the Stray Scanner export the place's map was built from")
    ap.add_argument("--pair", default="camera-b:camera-a", help="two cameras to compare")
    ap.add_argument("--save", type=Path, help="where the dry run saves the fits (.json)")
    ap.add_argument("--fits", type=Path, help="a saved dry run's fits: measure them again without fitting")
    ap.add_argument("--load", type=Path, help="the dry run's saved fits, for --write")
    ap.add_argument("--write", action="store_true", help="hold --load's fits in the live state")
    a = ap.parse_args()
    state = cfg.path("paths.state")
    held_place = _place(state, a.place)
    if a.write:
        if a.load is None:
            sys.exit("--write takes the dry run's --load <file.json>, so what's held is what was checked")
        for cam, v in json.loads(a.load.read_text()).items():
            lens.hold(state, cam, lens.Held(**v))
            print(f"held {cam}: focal {v['focal']:.4f} of its width, lens {v['division']:+.4f} ({v['source']})")
        return
    if a.walk is None and a.fits is None:
        sys.exit("--walk (or --fits) is needed for a dry run")
    rev = places.latest(state, held_place.id)
    if rev is None:
        sys.exit(f"{held_place.name} has no revision")
    fixed = {c: p for c, p in rev.cameras.items() if not c.startswith(LOOK_PREFIX)}
    looks = {c: p for c, p in rev.cameras.items() if c.startswith(LOOK_PREFIX)}
    chain = places.revisions(state, held_place.id)
    pair = tuple(a.pair.split(":", 1))
    if len(pair) != 2:
        sys.exit("--pair is <camera>:<camera>")
    with tempfile.TemporaryDirectory(prefix="fit-intrinsics-") as tmp:
        out = Path(tmp)
        chosen = {c: lens.Held(**v) for c, v in json.loads(a.fits.read_text()).items()} if a.fits else {}
        if a.walk is not None and a.fits is None:
            walk = walk_frames(a.walk, out / "walk", looks)
            print(f"walk: {len(walk.images)} frames posed in the place's map")
            (out / "fixed").mkdir(parents=True)
            shots = cameras.frames(state, sorted(fixed), out / "fixed", time.time(), detect=cameras.detector())
            fits = pair_fits(walk, shots, fixed, out)
        else:
            fits = {}
        print("fits:")
        for cam in sorted(fixed) if a.fits is None else []:
            past = history(chain, cam)
            report(cam, fits.get(cam, []), past, fixed[cam])
            got = choose(cam, fits.get(cam, []), past)
            if got is not None:
                chosen[cam] = got
                print(f"    to hold: focal {got.focal:.4f}, lens {got.division:+.4f} ({got.source})")
        indoor = cameras.indoor_cameras(sorted(fixed), cfg.get("spatial.outdoor_cameras") or [])
        capture = cameras.capture(state, indoor, out / "capture", time.time())
        if capture is None:
            print("no camera had a frame for tonight's pass; nothing measured")
        else:
            measure(state, held_place.id, (pair[0], pair[1]), capture, out / "capture", chosen)
    target = a.save or Path(tempfile.gettempdir()) / f"intrinsics-{held_place.id}-{int(time.time())}.json"
    target.write_text(json.dumps({c: dataclasses.asdict(h) for c, h in chosen.items()}, indent=1))
    print(f"saved the fits to {target}; hold them with --load {target} --write")


if __name__ == "__main__":
    main()
