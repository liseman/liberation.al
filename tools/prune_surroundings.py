"""List the learned map's kept views with whether each registers from its camera's spot, and drop those that don't.

A fixed camera that was moved (picked up, carried, put down somewhere new) keeps taking views from where it is now,
and before the map step checks each view against the plan (#410) those views go into the map at the camera's old
spot. This finds them by their registration, not by a time range, so it can be run again after more views arrive:

    .venv/bin/python tools/prune_surroundings.py                     # every kept view and its registration
    .venv/bin/python tools/prune_surroundings.py --prune             # what pruning would change, changing nothing
    .venv/bin/python tools/prune_surroundings.py --prune --yes       # do it
    .venv/bin/python tools/prune_surroundings.py --prune --camera C  # also drop every kept view of camera C
    .venv/bin/python tools/prune_surroundings.py --prune --remove-stray  # also remove a place learned by pairing alone

A view of a camera with a usable spot registers when its floor comes out at the plan's floor within
`surroundings_map.SPOT_SCALE` and its depth, seen from the spot (or its kept refinement), lines up with the plan's
walls and floor (`plan_depth.fit` gives no reason against it). A view placed by pairing into a place in its plan's
frame is checked the same way at the pose it was placed at, except that one seeing too little of the plan (`few`)
is kept. Views of a camera whose spot lacks something can't be checked and are left as they are.

Pruning, under the map step's lock:

- each view that doesn't register (and, with `--camera`, every kept view of that camera) is dropped: it is taken
  out of the map step's state and listed for the sampler to delete from the kept store;
- each place in its plan's frame that held a dropped view is started again: the map step founds it fresh from the
  views kept, as it does when a map moves into its plan's frame;
- with `--remove-stray`, a place the map step learned by pairing alone, with no plan, name, or anything else of the
  owner's, is removed: its state, and its revisions under `state/places/<place>/`, which nothing else refers to.
  (Off by default: on an install with no plan, every map is learned that way.)

Running it again finds nothing more to do until new views arrive. Restart the dashboard afterwards if a place was
removed (it keeps the places it read). The map step must not be stopped: it is what founds the places again.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from liberational import config as cfg
from liberational import locking
from liberational.memory import places
from liberational.spatial import plan_depth, plan_lineup, spot_check, spots
from liberational.spatial import surroundings_frames as frames
from liberational.spatial import surroundings_map as sm


@dataclass(frozen=True)
class View:
    """A kept view and its registration: whether it registers (None when it can't be checked), why not, and the
    number that says so."""

    id: str
    camera: str
    t: float
    place: str | None
    by: str  # "spot", "pairing", or "waiting" (not in a map)
    registers: bool | None
    why: str | None
    detail: str


def check(state_dir: Path) -> tuple[list[View], sm.State]:
    """Every kept view with its registration, and the map step's state they were read against."""
    store = frames.Store(frames.root(state_dir))
    state = sm.State.read(store.folder)
    spotted = spots.read(state_dir)
    geoms: dict[str, spot_check.PlanGeometry | None] = {}

    def geom(place: str) -> spot_check.PlanGeometry | None:
        if place not in geoms:
            held = places.get(state_dir, place)
            parsed = plan_lineup.read_plan(held.plan) if held and held.plan_source == "owner" and held.plan else None
            geoms[place] = None if parsed is None else spot_check.prepare_plan(parsed)
        return geoms[place]

    out = []
    for f in sorted(store.kept, key=lambda k: (k.camera, k.t)):
        entry = state.frames.get(f.id) or {}
        place = entry.get("place")
        by = "waiting" if entry.get("pose") is None else ("spot" if entry.get("spot") else "pairing")
        path = store.folder / f"{f.id}.depth.png"
        spot = spotted.get(f.camera)
        if f.id in state.drop or not path.is_file() or by == "waiting":
            out.append(View(f.id, f.camera, f.t, place, by, None, None, "not in a map"))
            continue
        depth = sm._read_depth(path)
        h, w = depth.shape
        share = entry.get("focal")
        if by == "spot":
            if spot is None or spot.pose is None or not sm.usable(spot):
                lacks = "no spot" if spot is None else ",".join(spot.missing or ("frame",))
                out.append(View(f.id, f.camera, f.t, place, by, None, None, f"its spot lacks {lacks}: not checked"))
                continue
            share = share or spot.pose.focal / spot.pose.width
            pose = np.asarray((spot.refined or spot.pose).pose, dtype=np.float64).reshape(4, 4)
            scale = sm.spot_fit(depth, share * w, pose, None).scale
            if not sm.SPOT_SCALE[0] <= scale <= sm.SPOT_SCALE[1]:
                out.append(View(f.id, f.camera, f.t, place, by, False, "floor", f"floor scale {scale:.3f}"))
                continue
        else:
            pose = np.asarray(entry["pose"], dtype=np.float64).reshape(4, 4)
        g = geom(place) if place and (state.places.get(place) or {}).get("plan_frame") else None
        if g is None or not share:
            ok = by == "pairing"
            out.append(View(f.id, f.camera, f.t, place, by, ok if ok else None, None, "no plan to check against"))
            continue
        fit = plan_depth.fit(depth, plan_depth.expected(g, pose, share * w, h, w))
        numbers = f"share {fit.share:.3f}, past the plan {fit.beyond:.3f}, slope {fit.a:.3f}"
        if fit.why is None or (by == "pairing" and fit.why == "few"):
            out.append(View(f.id, f.camera, f.t, place, by, True, None, numbers))
        else:
            out.append(View(f.id, f.camera, f.t, place, by, False, fit.why, numbers))
    return out, state


@dataclass(frozen=True)
class Pruning:
    drop: tuple[View, ...]
    refound: tuple[str, ...]  # plan-frame places started again
    remove: tuple[str, ...]  # places learned by pairing alone, with nothing of the owner's


def plan(
    state_dir: Path, views: Sequence[View], state: sm.State, cameras: Sequence[str] = (), stray: bool = False
) -> Pruning:
    """What pruning changes."""
    drop = tuple(v for v in views if v.registers is False or (v.camera in cameras and v.id not in state.drop))
    refound = tuple(sorted({v.place for v in drop if v.place and (state.places.get(v.place) or {}).get("plan_frame")}))
    remove = []
    for place, entry in state.places.items():
        if not stray or entry.get("plan_frame"):
            continue
        held = places.get(state_dir, place)
        if held is None or (held.plan_source != "owner" and not held.named):
            remove.append(place)
    return Pruning(drop, refound, tuple(sorted(remove)))


def apply(state_dir: Path, pruning: Pruning) -> None:
    """Carry out `pruning`, under the map step's lock."""
    store_folder = frames.root(state_dir)
    with locking.locked(store_folder / "map"):
        state = sm.State.read(store_folder)
        for v in pruning.drop:
            state.frames.pop(v.id, None)
            if v.id not in state.drop:
                state.drop.append(v.id)
        for place in (*pruning.refound, *pruning.remove):
            state.places.pop(place, None)
            for k in [k for k, entry in state.frames.items() if entry.get("place") == place]:
                del state.frames[k]
        state.write(store_folder)
        for place in pruning.remove:
            folder = places.root(state_dir) / place
            if folder.is_dir():
                shutil.rmtree(folder)


def _when(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--state", type=Path, default=None, help="the state folder (default: paths.state)")
    parser.add_argument("--prune", action="store_true", help="show (or, with --yes, do) what pruning changes")
    parser.add_argument("--camera", action="append", default=[], help="also drop every kept view of this camera")
    parser.add_argument("--remove-stray", action="store_true", help="also remove places learned by pairing alone")
    parser.add_argument("--yes", action="store_true", help="with --prune: do it")
    parser.add_argument("--json", action="store_true", help="print the views as JSON lines")
    args = parser.parse_args(argv)
    state_dir = args.state or cfg.path("paths.state")
    views, state = check(state_dir)
    for v in views:
        if args.json:
            print(json.dumps({**v.__dict__, "kept": _when(v.t)}))
            continue
        verdict = {True: "registers", False: f"DOES NOT REGISTER ({v.why})", None: "not checked"}[v.registers]
        print(
            f"{v.camera:24} {v.id[:8]} kept {_when(v.t)}  {v.by:8} place {(v.place or '-')[-6:]:6}  {verdict}: {v.detail}"
        )
    if not args.prune:
        return
    pruning = plan(state_dir, views, state, args.camera, args.remove_stray)
    print()
    print(f"drop {len(pruning.drop)} view(s): " + ", ".join(f"{v.camera} {v.id[:8]}" for v in pruning.drop))
    print(f"found again from the views kept: {len(pruning.refound)} place(s) " + " ".join(pruning.refound))
    print(f"remove {len(pruning.remove)} place(s) learned by pairing alone: " + " ".join(pruning.remove))
    if not (pruning.drop or pruning.refound or pruning.remove):
        print("nothing to do")
        return
    if not args.yes:
        print("dry run: nothing changed (add --yes to do it)")
        return
    apply(state_dir, pruning)
    print("done")


if __name__ == "__main__":
    main()
