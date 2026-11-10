#!/usr/bin/env python3
"""Rebuild a place's map from its LiDAR walks and its fixed cameras' clouds (#3286; spatial/fuse.py `rebuild` says
how the pieces go together).

The base is each Stray Scanner walk's own map (a revision of the place by `walkthrough.PRODUCER_STRAY` whose map file
is still there), placed in the fused map's frame by the looks it shares with it and fitted to the owner's plan there
(spatial/plan_import.py `register`, seeded where the looks put it). Each fixed camera then stands where the owner
measured it (the plan's `x-camera` spot and height, its facing when given, else its registered one), and tonight's
snapshot of each camera group (spatial/cameras.py `capture` and `snapshot`: frames and MASt3R, read only) is seated
there and polished against the walks. Nothing else of the old map is kept: older machine-made material (soft
MASt3R-only clouds, a phone walk's doubled walls) is expendable. Points outside the owner plan by more than OUTSIDE_M
are left out.

Dry run (the default): the rebuild is written into a throwaway copy of the state (symlinks to the live files, a
backup copy of the knowledge store), and the tool prints the wall error against the owner plan (median and 90th
percentile distance from a wall's face, and the share of wall band points within 10 cm of one) for the current map
and the rebuild, each camera's seat (ICP fitness, and how far from its measured spot it ends), the position rows by
room on the owner plan before and after, which rooms a walk covers, and that the owner plan, its view, its camera
spots, the zones, and the device locations still resolve. It saves the rebuild to `--save` (default under the
system temp dir). `--write --load <saved.npz>` writes exactly that rebuild to the live state, refusing if the place's
fused map moved on since.

    .venv/bin/python tools/rebuild_place.py --place <place id or name> [--walk-revision <id> ...] [--no-cameras] \\
        [--day YYYY-MM-DD] [--save <file.npz>]
    .venv/bin/python tools/rebuild_place.py --place <place> --load <file.npz> --write

Run it from the live checkout, never while `pack-spatial-cameras` runs (the write takes the places lock).
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import math
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import dryrun, floorplan, fuse, ply
from liberational.spatial import plan_import as pi

LOOK_PREFIX = "walk-"  # a walk's look-from views (spatial/splats.py LOOK_PREFIX), not fixed cameras
OUTSIDE_M = 0.3  # metres outside every room of the owner plan (past a wall's centreline) a point may lie and be kept
NEAR_WALL = 0.1  # metres: the report's "within 10 cm of a wall's face"
IDENT = pi.Move(0.0, (0.0, 0.0), mirror=False)  # the owner plan is stored in the plan frame: its page is the frame


@dataclasses.dataclass(frozen=True)
class Walls:
    median: float
    p90: float
    near: float  # share of the band points over the plan within NEAR_WALL of a wall's face
    points: int

    def line(self, label: str) -> str:
        return (
            f"{label}: walls median {self.median * 100:.1f} cm, p90 {self.p90 * 100:.1f} cm from a wall's face; "
            f"{self.near:.0%} of {self.points} wall band points over the plan within {NEAR_WALL * 100:.0f} cm of one"
        )


def wall_stats(plan: pi.Parsed, band: np.ndarray) -> Walls:
    """A map's wall band (plan frame) against the plan's walls, as plan_import's report measures it."""
    w = pi.walls(plan)
    face, _, centre = pi.faces(band, w)
    wall = centre < pi.REPORT_NEAR
    over = np.zeros(len(band), dtype=bool)
    for room in plan.rooms:
        lo = np.min(room.polygon, axis=0) - pi.REPORT_NEAR
        hi = np.max(room.polygon, axis=0) + pi.REPORT_NEAR
        over |= np.all((band >= lo) & (band <= hi), axis=1)
    if not wall.any():
        return Walls(math.nan, math.nan, 0.0, int(over.sum()))
    near = float((face[over] <= NEAR_WALL).mean()) if over.any() else 0.0
    return Walls(float(np.median(face[wall])), float(np.percentile(face[wall], 90)), near, int(over.sum()))


def to_map(transform: Sequence[float], plan_point: np.ndarray) -> np.ndarray:
    """A plan-frame (x, y, height) point back in the map (the 12-number map-to-plan transform is rigid)."""
    t = np.asarray(transform, dtype=np.float64).reshape(3, 4)
    return t[:, :3].T @ (np.asarray(plan_point, dtype=np.float64) - t[:, 3])


def yaw_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The rigid 4x4 turning only about z that best takes the (N, 3) points `src` onto `dst` (both on a floor at z =
    0), with no shift along z."""
    a, b = src[:, :2] - src[:, :2].mean(0), dst[:, :2] - dst[:, :2].mean(0)
    turn = math.atan2(float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()), float((a * b).sum()))
    c, s = math.cos(turn), math.sin(turn)
    out = np.eye(4)
    out[:2, :2] = [[c, -s], [s, c]]
    out[:2, 3] = dst[:, :2].mean(0) - out[:2, :2] @ src[:, :2].mean(0)
    return out


@dataclasses.dataclass(frozen=True)
class WalkPlaced:
    revision: places.Revision
    piece: fuse.Piece
    transform: np.ndarray  # its map into the fused map's frame
    looks: int  # looks it shares with the fused map
    looks_off: float  # metres: the farthest a shared look lands from the fused map's, by the looks' fit
    before: Walls  # as the looks place it
    after: Walls  # fitted to the plan
    moved: tuple[float, float]  # metres and degrees the plan fit moved it


def place_walk(
    state: Path, walk: places.Revision, fused_rev: places.Revision, plan: pi.Parsed, frame: Sequence[float]
) -> WalkPlaced:
    """A walk revision's map in the fused map's frame: levelled by its floor, turned and shifted so its looks land on
    the fused map's own (rigid: the walk is metric LiDAR), then fitted to the owner plan from there."""
    if walk.map is None or walk.floor is None:
        sys.exit(f"walk revision {walk.id} has no map or floor")
    points, colors, _ = ply.read(places.root(state) / walk.place / walk.map)
    level = fuse.leveller(walk.floor)
    # The walk's stored floor is its pose graph's; the floor plane of its own cloud is what tier 1 levels the fused map
    # by (`fuse.relevel`), so the rebuild is levelled by that too, or the next fusion would turn it into a new frame
    # and the owner plan drawn in this one would no longer hold for it
    try:
        normal, d = floorplan.floor_plane(fuse.apply(level, points), (0.0, 0.0, 1.0), 0.03, np.random.default_rng(0))
        level = fuse.leveller((*(float(x) for x in normal), -float(d))) @ level
    except ValueError:
        pass
    shared = sorted(c for c in walk.cameras if c in fused_rev.cameras)
    if len(shared) < 3:
        sys.exit(f"walk revision {walk.id} shares {len(shared)} looks with the fused map; it needs 3")
    src = np.array([fuse.apply(level, np.asarray(walk.cameras[c].pose).reshape(4, 4)[:3, 3][None])[0] for c in shared])
    dst = np.array([np.asarray(fuse_pose(fused_rev, c))[:3, 3] for c in shared])
    turn = yaw_fit(src, dst)
    off = float(np.linalg.norm(fuse.apply(turn, src)[:, :2] - dst[:, :2], axis=1).max())
    start = turn @ level
    placed = fuse.apply(start, points)
    before = wall_stats(plan, pi.band(placed, frame))
    fit = pi.register(plan, pi.band(placed, frame), {}, mirror=False, start=IDENT)
    correction = pi.undo(frame, fit.move)
    transform = correction @ start
    after = wall_stats(plan, pi.band(fuse.apply(transform, points), frame))
    moved = (math.hypot(*fit.move.shift), abs(math.degrees(fit.move.turn)))
    return WalkPlaced(
        walk, fuse.Piece(points, colors, walk.floor, walk.cameras), transform, len(shared), off, before, after, moved
    )


def fuse_pose(rev: places.Revision, cam: str) -> np.ndarray:
    return np.asarray(rev.cameras[cam].pose, dtype=np.float64).reshape(4, 4)


def spots(rev: places.Revision, plan: pi.Parsed, frame: Sequence[float]) -> dict[str, places.CameraPose]:
    """Each fixed camera's pose at the owner's measured spot: its spot and height, turned to its facing when the plan
    gives one (else as registered); a camera with no spot as registered."""
    marks = {c.id: c for c in plan.cameras}
    a = np.asarray(frame, dtype=np.float64).reshape(3, 4)[:, :3]
    out = {}
    for cam, pose in rev.cameras.items():
        if cam.startswith(LOOK_PREFIX):
            continue
        mark = marks.get(cam)
        if mark is None:
            out[cam] = pose
            continue
        m = fuse_pose(rev, cam)
        x, y, h = pi.in_frame(m[:3, 3], frame)[0]
        turn = 0.0
        if mark.yaw is not None:
            look = a @ m[:3, 2]
            turn = pi.turn_to(IDENT, (float(look[0]), float(look[1])), float(mark.yaw))
        m = pi.turn_about(frame, (float(x), float(y)), mark.at, turn) @ m
        height = float(mark.height) if mark.height is not None else float(h)
        m[:3, 3] = to_map(frame, np.array([mark.at[0], mark.at[1], height]))
        out[cam] = dataclasses.replace(pose, pose=tuple(float(v) for v in m.ravel()))
    return out


def inside_plan(plan: pi.Parsed, frame: Sequence[float]) -> Callable[[np.ndarray], np.ndarray]:
    w = pi.walls(plan)

    def inside(points: np.ndarray) -> np.ndarray:
        q = pi.in_frame(points, frame)[:, :2]
        out = pi.inside(plan, q)
        rest = ~out
        if rest.any():
            out[rest] = pi.nearest(q[rest], w)[0] <= OUTSIDE_M
        return out

    return inside


def snapshots(state: Path, cams: Sequence[str], now: float) -> list[fuse.Piece]:
    """Tonight's snapshot of each group of `cams` (frames and MASt3R, read only, in a temporary folder)."""
    from liberational.spatial import cameras

    indoor = cameras.indoor_cameras(cams, cfg.get("spatial.outdoor_cameras") or [])
    out_pieces = []
    with tempfile.TemporaryDirectory(prefix="rebuild-place-") as tmp:
        out = Path(tmp)
        got = cameras.capture(state, indoor, out, now)
        if got is None:
            print("no camera had a frame: the rebuild has the walks only")
            return []
        if got.absent:
            print(f"no usable frame for {', '.join(got.absent)}: they stand at their spots with no cloud")
        for group in got.result.get("groups", []):
            names = [str(c) for c in group["cameras"]]
            if not set(names) <= set(cams):
                print(f"group {', '.join(names)} has a camera of another place; left out")
                continue
            snap = cameras.snapshot(group, out, now, [], got.gravity, seated=True)
            if snap is None or snap.floor is None:
                print(f"group {', '.join(names)} has no floor; left out")
                continue
            out_pieces.append(fuse.Piece(snap.points, snap.colors, snap.floor, snap.poses))
    return out_pieces


def report_cameras(
    title: str,
    cams: Mapping[str, places.CameraPose],
    plan: pi.Parsed,
    frame: Sequence[float],
) -> None:
    print(title)
    a = np.asarray(frame, dtype=np.float64).reshape(3, 4)[:, :3]
    for mark in plan.cameras:
        pose = cams.get(mark.id)
        if pose is None:
            print(f"  {mark.id}: not registered")
            continue
        m = np.asarray(pose.pose, dtype=np.float64).reshape(4, 4)
        x, y, h = pi.in_frame(m[:3, 3], frame)[0]
        look = a @ m[:3, 2]
        bearing = pi.look_bearing(IDENT, (float(look[0]), float(look[1])))
        line = f"  {mark.id}: {math.dist((x, y), mark.at) * 100:.0f} cm from its spot"
        if mark.height is not None:
            line += f", height {h:.2f} m (measured {mark.height:.2f})"
        line += f", faces {bearing:.0f} deg"
        if mark.yaw is not None:
            line += f" (given {mark.yaw:.0f}, {abs((bearing - mark.yaw + 180) % 360 - 180):.0f} off)"
        print(line)


def report_seats(rebuilt: fuse.Rebuilt) -> None:
    for s in rebuilt.seats:
        names = ", ".join(s.cameras)
        if s.why:
            print(f"  group {names}: left out ({s.why})")
            continue
        if s.fitness is None:
            how = "the walks have nothing around it: placed by its cameras' spots"
        elif s.polished:
            how = (
                f"fits the walks at fitness {s.fitness:.2f} where ICP puts it, {s.shift:.2f} m and {s.turn:.1f} deg "
                f"off its spots' placement (scale {s.scale:.3f}): placed there"
            )
        else:
            how = (
                f"ICP against the walks didn't hold (fitness {s.fitness:.2f}, {s.shift:.2f} m and {s.turn:.1f} deg "
                "off): placed by its cameras' spots"
            )
        print(f"  group {names}: {how}; {s.kept} points in, {s.gated} left out by the gates")


def tonight(groups: Sequence[fuse.Piece], rebuilt: fuse.Rebuilt) -> None:
    """Each group's snapshot seated in the rebuilt map as the next `pack-spatial-cameras` run seats it (`fuse.seat`):
    started from the rebuilt cameras' poses, polished by ICP, refused past SEAT_SHIFT_M or SEAT_TURN_DEG."""
    for group in groups:
        names = ", ".join(sorted(group.cameras))
        got = fuse.levelled(group, f"tonight's snapshot of {names}")
        fit = None
        if got is not None and set(group.cameras) <= set(rebuilt.cameras):
            fit = fuse.seat_fit(got[0], got[1], {c: rebuilt.cameras[c] for c in group.cameras}, rebuilt.fused.points)
        if fit is None or fit.reg is None:
            print(f"  {names}: can't be seated")
            continue
        verdict = "seats" if fit.held else "REFUSED"
        print(
            f"  {names}: {verdict}, fitness {fit.reg.fitness:.2f} (needs {fuse.MIN_FITNESS}); its cloud lands "
            f"{fit.shift:.2f} m and {fit.turn:.1f} deg from where its cameras' spots put it, and the cameras stay on "
            "their spots"
        )


def coverage(fused: fuse.Fused, plan: pi.Parsed, frame: Sequence[float]) -> dict[str, list[str]]:
    """The plan's rooms by what covers them: a walk, the cameras only, or nothing (`plan_import.room_coverage`)."""
    out: dict[str, list[str]] = {"walked": [], "cameras only": [], "nothing": []}
    for c in pi.room_coverage(fused.points, fused.walks, plan, frame):
        out[c.kind].append(f"{c.room.name} ({c.walked:.0%} walked, {c.mapped:.0%} mapped)")
    return out


def counts(state: Path, place: str, start: float, end: float) -> collections.Counter[str]:
    """Position rows of `place` in [start, end) by room on the plan that holds (its frame revision)."""
    return dryrun.rooms(dryrun.rows(state, place, start, end))


def print_counts(title: str, before: collections.Counter[str], after: collections.Counter[str]) -> None:
    print(f"{title}: {sum(before.values())} rows before, {sum(after.values())} after")
    for room in sorted(set(before) | set(after), key=lambda r: -(before[r] + after[r])):
        print(f"  {room}: {before[room]} -> {after[room]}")


def resolves(state: Path, place: str) -> list[str]:
    """What the owner taught about `place` that must still resolve: problems found, [] when all does."""
    from liberational.proximity import devices
    from liberational.see import zones

    held = places.get(state, place)
    if held is None:
        return [f"no place {place}"]
    problems = []
    if held.plan_source != "owner" or not held.plan:
        problems.append("the owner plan no longer holds")
    else:
        plan = pi.parse(held.plan)
        rev = next((r for r in places.revisions(state, place) if r.id == held.plan_revision), None)
        newest = places.latest(state, place)
        if rev is None or newest is None or not places.same_plan_frame(rev, newest):
            problems.append("the owner plan isn't drawn in the newest revision's frame")
        if held.view is None:
            problems.append("the owner plan lost its page view")
        regs = places.camera_places(state)
        for mark in plan.cameras:
            if regs.get(mark.id) != place:
                problems.append(f"camera {mark.id} (measured spot) isn't registered to the place")
        rooms = {r.name for r in plan.rooms}
        for loc in devices.at(state).locations():
            if loc.place == place and loc.room is not None and loc.room not in rooms:
                problems.append(f"device {loc.kind} {loc.value} is located in a room the plan doesn't have")
    regs = places.camera_places(state)
    registered = (places.latest(state, place) or held.revision).cameras
    for cam in zones.current(state):
        if cam in registered and regs.get(cam) != place:
            problems.append(f"zones of {cam} no longer file under the place")
    return problems


def save(path: Path, rebuilt: fuse.Rebuilt, sources: Sequence[str], fused_rev: str) -> None:
    f = rebuilt.fused
    meta = {
        "sources": list(sources),
        "fused_revision": fused_rev,
        "cameras": {c: p.as_dict() for c, p in rebuilt.cameras.items()},
    }
    np.savez_compressed(
        path,
        points=f.points,
        colors=f.colors if f.colors is not None else np.zeros((0, 3), np.uint8),
        seen=f.seen,
        missed=f.missed,
        walked=f.walks,
        meta=np.array(json.dumps(meta)),
    )


def load(path: Path) -> tuple[fuse.Rebuilt, list[str], str]:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        colors = z["colors"] if len(z["colors"]) else None
        fused = fuse.Fused(z["points"], colors, z["seen"], z["missed"], walked=z["walked"])
    cams = {c: places.CameraPose.from_dict(p) for c, p in meta["cameras"].items()}
    return fuse.Rebuilt(fused, cams, (), 0), list(meta["sources"]), str(meta["fused_revision"])


def _place(state: Path, wanted: str) -> places.Place:
    for p in places.places(state):
        if wanted in (p.id, p.name):
            return p
    sys.exit(f"no place {wanted!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", required=True, help="place id or name")
    ap.add_argument("--walk-revision", action="append", default=[], help="a Stray walk's revision (default: all)")
    ap.add_argument("--no-cameras", action="store_true", help="leave the fixed cameras' snapshots out")
    ap.add_argument("--day", help="count position rows on this local day (default: today, up to now)")
    ap.add_argument("--save", type=Path, help="where the dry run saves the rebuild (.npz)")
    ap.add_argument("--load", type=Path, help="the dry run's saved rebuild, for --write")
    ap.add_argument("--write", action="store_true", help="write --load's rebuild to the live state")
    a = ap.parse_args()
    state = cfg.path("paths.state")
    held = _place(state, a.place)
    fused_rev = fuse.fused_revision(state, held.id)
    if fused_rev is None:
        sys.exit(f"{held.name} has no fused map")

    if a.write:
        if a.load is None:
            sys.exit("--write takes the dry run's --load <file.npz>, so what's written is what was checked")
        rebuilt, sources, made_on = load(a.load)
        with places.writing(state):
            now_fused = fuse.fused_revision(state, held.id)
            if now_fused is None or now_fused.id != made_on:
                sys.exit(f"{held.name}'s fused map moved on since the dry run ({made_on}); run the dry run again")
            rev = fuse.write_rebuild(state, held.id, rebuilt, sources)
        after = places.get(state, held.id)
        print(f"wrote {rev.id}; the owner plan is drawn on {after.plan_revision if after else None}")
        problems = resolves(state, held.id)
        print("owner inputs resolve" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        return

    if held.plan_source != "owner" or not held.plan:
        sys.exit(f"{held.name} has no owner plan to rebuild against")
    drawn = next((r for r in places.revisions(state, held.id) if r.id == held.plan_revision), None)
    if drawn is None or drawn.frame != fused_rev.frame or drawn.plan is not None:
        sys.exit("the owner plan isn't drawn on the fused map's frame; the rebuild keeps that frame")
    frame = places.plan_frame(drawn)
    assert frame is not None
    plan = pi.parse(held.plan)
    newest = places.latest(state, held.id) or fused_rev
    chain = places.revisions(state, held.id)
    from liberational.spatial import walkthrough

    wanted = a.walk_revision or [
        r.id for r in chain if r.producer == walkthrough.PRODUCER_STRAY and places.read_map(state, r) is not None
    ]
    if not wanted:
        sys.exit("no Stray walk revision with its map; give --walk-revision")
    walks = []
    for wid in wanted:
        rev = next((r for r in chain if r.id == wid), None)
        if rev is None:
            sys.exit(f"{held.name} has no revision {wid}")
        w = place_walk(state, rev, newest, plan, frame)
        print(
            f"walk {rev.id}: {w.looks} looks shared, the farthest {w.looks_off * 100:.0f} cm off; fitted to the plan "
            f"it moved {w.moved[0] * 100:.0f} cm and {w.moved[1]:.1f} deg"
        )
        print("  " + w.before.line("as its looks place it"))
        print("  " + w.after.line("fitted to the plan"))
        walks.append(w)

    assert fused_rev.map is not None  # fused_revision finds only revisions whose map is there
    current = ply.read(places.root(state) / held.id / fused_rev.map)
    before = wall_stats(plan, pi.band(current[0], frame))
    print(before.line(f"current map {fused_rev.id}"))
    report_cameras("cameras now:", {c: p for c, p in newest.cameras.items()}, plan, frame)

    fixed = sorted(c for c in newest.cameras if not c.startswith(LOOK_PREFIX))
    now = time.time()
    groups = [] if a.no_cameras else snapshots(state, fixed, now)
    at_spots = spots(newest, plan, frame)
    rebuilt = fuse.rebuild(
        [(w.piece, w.transform) for w in walks], groups, at_spots, inside_plan(plan, frame), newest.cameras
    )
    after = wall_stats(plan, pi.band(rebuilt.fused.points, frame))
    print(before.line("before"))
    print(after.line("rebuilt"))
    walked_only = rebuilt.fused.points[rebuilt.fused.walks > 0]
    print(wall_stats(plan, pi.band(walked_only, frame)).line("  of which the walks' voxels"))
    print(
        f"rebuilt map: {len(rebuilt.fused.points)} voxels ({len(walked_only)} walked), was {len(current[0])}; "
        f"{rebuilt.outside} walk points left out outside the plan"
    )
    print("groups:")
    report_seats(rebuilt)
    print("tonight's seat of each group in the rebuild (as pack-spatial-cameras will seat it):")
    tonight(groups, rebuilt)
    report_cameras("cameras rebuilt:", rebuilt.cameras, plan, frame)
    for kind, rooms in coverage(rebuilt.fused, plan, frame).items():
        print(f"rooms {kind}: {', '.join(rooms) or 'none'}")
    tilt = _tilt(rebuilt.fused.points)
    print(f"the rebuilt map's own floor is {tilt:.2f} deg off level (tier 1 relevels past {fuse.RELEVEL_DEG})")

    sources = [w.revision.id for w in walks] + [fused_rev.id]
    target = a.save or Path(tempfile.gettempdir()) / f"rebuild-{held.id}-{int(now)}.npz"
    save(target, rebuilt, sources, fused_rev.id)
    day = dt.date.fromisoformat(a.day) if a.day else dt.date.today()
    start = dt.datetime.combine(day, dt.time()).astimezone().timestamp()
    end = min(start + 86400, now)
    with tempfile.TemporaryDirectory(prefix="rebuild-overlay-") as tmp:
        shadow = dryrun.overlay(state, Path(tmp), held.id)
        was = counts(shadow, held.id, start, end)
        rev = fuse.write_rebuild(shadow, held.id, rebuilt, sources)
        print(f"dry run: wrote {rev.id} into a throwaway copy of the state")
        print_counts(f"{day} position rows by room on the owner plan", was, counts(shadow, held.id, start, end))
        problems = resolves(shadow, held.id)
        print("owner inputs resolve after it" if not problems else "PROBLEMS after it:\n  " + "\n  ".join(problems))
    print(f"saved the rebuild to {target}; write it with --load {target} --write")


def _tilt(points: np.ndarray) -> float:
    try:
        normal, _ = floorplan.floor_plane(points, (0.0, 0.0, 1.0), 0.03, np.random.default_rng(0))
    except ValueError:
        return math.nan
    return math.degrees(math.acos(min(1.0, abs(float(normal[2])))))


if __name__ == "__main__":
    main()
