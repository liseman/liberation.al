#!/usr/bin/env python3
"""A place's fixed cameras' pose series (#3289; spatial/poses.py): backfill them from the place's revisions, and
measure what reading positions through them changes.

Dry run (the default): the `seat` observations the place's tier 1 revisions in its current map frame already hold
(`poses.backfill`), less any already stored, recorded into a throwaway copy of the state (spatial/dryrun.py, whose
copy of the `camera.pose` stream is its own). It prints each camera's observations (time, pitch, roll, position, the
owner's spot), its still stretches and smoothed pose against the raw spread, and today's position rows on the plan
that holds read without and with the series: the rows with no floor point and how far apart two cameras put one
person at the same moment. `--write` records the same observations in the live state.

    .venv/bin/python tools/camera_poses.py --place <place id or name> [--pair <camera>:<camera>] [--day YYYY-MM-DD]
    .venv/bin/python tools/camera_poses.py --place <place> --write

Run it from the live checkout. A new `camera.pose` stream is read by `positions` only after its services restart.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import dryrun, fuse, poses, positions, splats


def tilt(rotation: np.ndarray) -> tuple[float, float]:
    """(pitch, roll) in degrees of a camera rotation (camera to map; columns right, down, forward; map z up)."""
    return math.degrees(math.asin(-float(rotation[2, 2]))), math.degrees(math.asin(-float(rotation[2, 0])))


def new_observations(state: Path, place: str) -> list[poses.Observation]:
    """`poses.backfill`'s observations less those already stored (same camera, source, and time)."""
    found = poses.backfill(state, place)
    have = {
        (o.camera, o.source, round(o.t, 3))
        for o in poses.read(state, 0.0, time.time() + 1.0, sorted({o.camera for o in found}))
    }
    return [o for o in found if (o.camera, o.source, round(o.t, 3)) not in have]


def describe(state: Path, place: str, frame: str | None, cams: Sequence[str]) -> None:
    found = poses.series(state, place, frame, cams)
    for cam in cams:
        s = found.get(cam)
        if s is None:
            print(f"  {cam}: no observations in this frame")
            continue
        located = [o for o in s.observations if o.source != "owner"]
        print(
            f"  {cam}: {len(located)} located observations, owner spots {sum(o.source == 'owner' for o in s.observations)}"
        )
        for o in s.observations:
            when = dt.datetime.fromtimestamp(o.t).astimezone().strftime("%m-%d %H:%M")
            x, y, z = o.pose[:3, 3]
            turn = "" if o.source == "owner" else "pitch {:.1f} roll {:.1f}, ".format(*tilt(o.pose[:3, :3]))
            print(f"    {when} {o.source}: {turn}at {x:.2f},{y:.2f},{z:.2f} (quality {o.quality:.2f})")
        stretches = poses.stretches(list(s.observations))
        print(f"    {len(stretches)} still stretch(es)")
        if located:
            pitches = [tilt(o.pose[:3, :3])[0] for o in located]
            now = s.at(time.time())
            if now is not None:
                p, r = tilt(now[:3, :3])
                print(
                    f"    raw pitch {min(pitches):.1f} to {max(pitches):.1f}; smoothed now pitch {p:.1f} roll {r:.1f}, "
                    f"at {now[0, 3]:.2f},{now[1, 3]:.2f},{now[2, 3]:.2f}"
                )


def line(found: Sequence[positions.Position], pair: tuple[str, str]) -> str:
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
    ap.add_argument("--pair", default="camera-b:camera-a", help="two cameras to compare")
    ap.add_argument("--day", help="count position rows on this local day (default: today, up to now)")
    ap.add_argument("--write", action="store_true", help="record the backfilled observations in the live state")
    a = ap.parse_args()
    state = cfg.path("paths.state")
    held = _place(state, a.place)
    fused = fuse.fused_revision(state, held.id)
    if fused is None:
        sys.exit(f"{held.name} has no fused map")
    cams = sorted(c for c in fused.cameras if not c.startswith(splats.LOOK_PREFIX))
    pair = tuple(a.pair.split(":", 1))
    if len(pair) != 2:
        sys.exit("--pair is <camera>:<camera>")
    fresh = new_observations(state, held.id)
    print(f"{len(fresh)} observations to record from {held.name}'s revisions in frame {fused.frame}")
    if a.write:
        poses.record(state, fresh)
        print("recorded")
        describe(state, held.id, fused.frame, cams)
        return
    day = dt.date.fromisoformat(a.day) if a.day else dt.date.today()
    start = dt.datetime.combine(day, dt.time()).astimezone().timestamp()
    end = min(start + 86400, time.time())
    with tempfile.TemporaryDirectory(prefix="camera-poses-") as tmp:
        shadow = dryrun.overlay(state, Path(tmp), held.id)
        before = dryrun.rows(shadow, held.id, start, end)
        poses.record(shadow, fresh)
        print("each camera's series in the throwaway state:")
        describe(shadow, held.id, fused.frame, cams)
        after = dryrun.rows(shadow, held.id, start, end)
        print(f"{day} rows on the plan that holds, without the series: {line(before, (pair[0], pair[1]))}")
        print(f"{day} rows on the plan that holds, with the series:    {line(after, (pair[0], pair[1]))}")
    print("dry run: nothing written (--write records the observations)")


if __name__ == "__main__":
    main()
