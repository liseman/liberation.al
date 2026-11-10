#!/usr/bin/env python3
"""Import an owner-supplied floorplan as a place's owner plan (#3276; spatial/plan_import.py says how it registers).

Dry run (the default) prints the fit, the cameras, and the day's position rows by room before and after; nothing is
written. `--write` stores the registered plan as the place's owner plan (memory/places.py `correct_plan`, source
owner, drawn in the frame of the revision it was registered to), over whatever held before, which stays in the
store's history.

    .venv/bin/python tools/import_owner_plan.py --place <place id or name> --plan <file.floorplan> \\
        [--extra <more.floorplan>] [--seen <camera>=<room id> ...] [--title "<title>"] [--revision <id>] \\
        [--walk-revision <id>] [--day YYYY-MM-DD] [--anchor-share 0.5] [--write]
    .venv/bin/python tools/import_owner_plan.py --place <place> --history
    .venv/bin/python tools/import_owner_plan.py --place <place> --restore <record id> [--write]

`--extra` adds statements from another file, such as the owner's measured camera spots (`x-camera <camera> at x,y
height <m> yaw <deg>`, in the plan's own coordinates), which hold the fit (`--anchor-share` of its weight). A spot's
`yaw`, when given, is the camera's facing: `--move-camera` then keeps it and fits only the spot near the mark. `--seen`
names the plan room a camera sees, for a camera with no measured spot. `--walk-revision` also measures that revision's
own map (a phone walk's) against the plan, registered alone. The plan file is read, never copied: its comments are
dropped and its title replaced (by `--title`, else the place's name).
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import floorplan, ply, rig
from liberational.spatial import plan_import as pi

LOOK_PREFIX = "walk-"  # a walk's look-from views (spatial/splats.py LOOK_PREFIX), not fixed cameras

Rooms = list[tuple[str, str, tuple[tuple[float, float], ...]]]  # floorplan.rooms_from_dsl's
COMPASS = ("up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left")  # on the plan's page
# A camera is moved to its measured spot only when its view fits the plan clearly better there (`View.score`): by
# this much more, and this many times as much
MOVE_GAIN = 0.15
MOVE_RATIO = 1.5
MIRROR_GAIN = 0.1  # a camera's view must fit this much better mirrored for its video to read as mirrored
# metres: a camera with a given facing goes where its view fits best only within this of its measured spot (tape and
# the owner's marks on the plan agreed to about 8 cm on Place 1); farther, it goes on the spot itself
MARK_TRUST = 0.15
FACING_AGREE = 20.0  # degrees: a facing is stored when the view and the registration agree this well


def _place(state: Path, wanted: str) -> places.Place:
    for p in places.places(state):
        if wanted in (p.id, p.name):
            return p
    sys.exit(f"no place {wanted!r}")


def _revision(state: Path, place: str, wanted: str | None) -> places.Revision:
    chain = places.revisions(state, place)
    if wanted:
        found = next((r for r in chain if r.id == wanted), None)
        if found is None:
            sys.exit(f"{place} has no revision {wanted}")
        return found
    with_map = [r for r in chain if r.map is not None and (places.root(state) / place / r.map).exists()]
    if not with_map:
        sys.exit(f"{place} has no revision with a map")
    return with_map[-1]


def _map_band(state: Path, rev: places.Revision) -> tuple[np.ndarray, Sequence[float]]:
    transform = places.plan_frame(rev)
    if transform is None or rev.map is None:
        sys.exit(f"revision {rev.id} has no floor or plan frame")
    points = ply.read(places.root(state) / rev.place / rev.map)[0]
    return pi.band(points, transform), transform


def _fixed(rev: places.Revision) -> dict[str, tuple[float, ...]]:
    return {c: p.pose for c, p in rev.cameras.items() if not c.startswith(LOOK_PREFIX)}


def _seen(pairs: Sequence[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        cam, sep, room = pair.partition("=")
        if not sep:
            sys.exit(f"--seen takes <camera>=<room id>, not {pair!r}")
        out[cam] = room
    return out


def _report(fit: pi.Fit, plan: pi.Parsed, spots: Mapping[str, tuple[float, ...]]) -> None:
    m = fit.move
    print(
        f"fit ({fit.seeded_by}): turn {math.degrees(m.turn):.2f} deg, shift {m.shift[0]:.3f},{m.shift[1]:.3f} m, "
        f"page mirrored {m.mirror}"
    )
    print(
        f"  walls: median {fit.median * 100:.1f} cm, p90 {fit.p90 * 100:.1f} cm from a wall face (band points within "
        f"{pi.REPORT_NEAR} m of a centreline); {fit.near_face} of {fit.band_points} band points within "
        f"{pi.NEAR_FACE * 100:.0f} cm of a wall face ({fit.near_share:.0%} of those over the plan)"
    )
    print(f"  scale a free fit would take: {fit.scale:.4f} (1 is metric agreement; not applied)")
    names = {r.id: r.name for r in plan.rooms}
    measured = {c.id: c for c in plan.cameras}
    configured = rig.load().heights()
    for c in fit.cameras:
        room = names.get(c.room or "", "outside every room")
        line = f"  camera {c.camera}: at {c.at[0]:.2f},{c.at[1]:.2f} on the plan, in {room}"
        if c.expected is not None:
            line += " (as said)" if c.room == c.expected else f" (said {names.get(c.expected, c.expected)})"
        if c.anchor_error is not None:
            line += f", {c.anchor_error * 100:.0f} cm from its measured spot"
        h = spots[c.camera][2]
        mark = measured.get(c.camera)
        line += f"; height in map {h:.2f} m"
        if mark is not None and mark.height is not None:
            line += f", measured {mark.height:.2f} m"
        if c.camera in configured:
            line += f", configured {float(configured[c.camera]):.2f} m"
        print(line)
    if fit.outside:
        print(f"  outside the room said: {', '.join(fit.outside)}")
    if fit.dropped:
        print(
            f"  left out of the fit, over {pi.ANCHOR_OUTLIER:.0f} m from their measured spots: {', '.join(fit.dropped)}"
        )


def _counts(
    state: Path, place: places.Place, start: float, end: float, frame: str | None, rooms: Rooms | None
) -> collections.Counter[str]:
    from liberational.spatial import positions

    rows = positions.positions(state, start, end, place=place.id, frame_revision=frame)
    out: collections.Counter[str] = collections.Counter()
    for r in rows:
        if r.x is None or r.y is None:
            out["(room only: a zone or no point)"] += 1
        elif not r.in_frame:
            out["(not in the plan's frame)"] += 1
        elif rooms is None:
            out[r.room or "(outside every room)"] += 1
        else:
            found = floorplan.room_at(rooms, r.x, r.y)
            out[found[1] if found else "(outside every room)"] += 1
    return out


def _print_counts(title: str, counts: collections.Counter[str]) -> None:
    print(f"{title}: {sum(counts.values())} rows")
    for room, n in counts.most_common():
        print(f"  {room}: {n}")


@dataclasses.dataclass(frozen=True)
class Registered:
    rev: places.Revision
    transform: Sequence[float]
    spots: dict[str, tuple[float, ...]]  # fixed cameras' (plan x, plan y, height) in the revision's frame
    fit: pi.Fit


def _register(
    state: Path, place: places.Place, revision: str | None, plan: pi.Parsed, seen: Mapping[str, str], share: float
) -> Registered:
    rev = _revision(state, place.id, revision)
    band, transform = _map_band(state, rev)
    poses = _fixed(rev)
    spots = pi.camera_spots(poses, transform)
    anchors = {c.id: c.at for c in plan.cameras if c.id in spots}
    print(f"registering to {place.name} revision {rev.id} ({rev.producer}): {len(band)} wall band points")
    fit = pi.register(
        plan,
        band,
        {c: (s[0], s[1]) for c, s in spots.items()},
        seen,
        anchors,
        anchor_share=share,
        mirror=pi.page_is_mirrored(poses, transform),
    )
    _report(fit, plan, spots)
    return Registered(rev, transform, spots, fit)


def _facing(view: tuple[pi.View, float, float] | None, given: float | None) -> float | None:
    """The yaw a measured spot is stored with: its own when the owner gave one, else its view's when the view is sure
    and its registration agrees (within FACING_AGREE degrees both), else none."""
    if given is not None or view is None:
        return given
    fit, registered, viewed = view
    apart = abs((viewed - registered + 180.0) % 360.0 - 180.0)
    return round(viewed) % 360 if apart <= FACING_AGREE and fit.spread <= FACING_AGREE else None


def _compass(bearing: float) -> str:
    return COMPASS[round(bearing / 45.0) % 8]


def _own_points(state: Path, reg: Registered, camera: str) -> np.ndarray:
    """A fixed camera's own points in the revision's map (its room, `fuse.room_of`), in the plan frame."""
    from liberational.spatial import fuse  # loads the fixed-camera pipeline; only this needs it

    rev = reg.rev
    if rev.map is None:
        return np.zeros((0, 3))
    points = ply.read(places.root(state) / rev.place / rev.map)[0]
    fixed = {c: p for c, p in rev.cameras.items() if not c.startswith(LOOK_PREFIX)}
    return pi.in_frame(points[fuse.room_of(points, camera, fixed)], reg.transform)


def _views(state: Path, reg: Registered, plan: pi.Parsed) -> dict[str, tuple[pi.View, float, float]]:
    """Each measured camera's view fit at its spot (`plan_import.view_fit`), with the bearing on the plan its
    registration says it looks and the one its view says, printed."""
    from liberational.spatial import fuse  # loads the fixed-camera pipeline; only this needs it

    rev = reg.rev
    if rev.map is None:
        return {}
    points = ply.read(places.root(state) / rev.place / rev.map)[0]
    a = np.asarray(reg.transform, dtype=np.float64).reshape(3, 4)[:, :3]
    names = {r.id: r.name for r in plan.rooms}
    # a camera's room is the part of the map it sees nearer to it than to any other fixed camera; a walk's look-from
    # views would crowd it out
    fixed = {c: p for c, p in rev.cameras.items() if not c.startswith(LOOK_PREFIX)}
    out = {}
    for mark in plan.cameras:
        if mark.id not in reg.spots:
            continue
        own = pi.in_frame(points[fuse.room_of(points, mark.id, fixed)], reg.transform)
        x, y, _ = reg.spots[mark.id]
        mark_at = reg.fit.move.point(*mark.at)
        view = pi.view_fit(plan, reg.fit.move, own, (x, y), mark_at)
        look = a @ np.asarray(rev.cameras[mark.id].pose, dtype=np.float64).reshape(4, 4)[:3, 2]
        c, s = math.cos(view.turn), math.sin(view.turn)
        turned = (c * look[0] - s * look[1], s * look[0] + c * look[1])
        registered = pi.look_bearing(reg.fit.move, (float(look[0]), float(look[1])))
        viewed = pi.look_bearing(reg.fit.move, turned)
        apart = abs((viewed - registered + 180.0) % 360.0 - 180.0)
        flipped = pi.view_fit(
            plan, reg.fit.move, pi.mirrored(own, (x, y), (float(look[0]), float(look[1]))), (x, y), mark_at
        )
        down = float((a @ np.asarray(rev.cameras[mark.id].pose, dtype=np.float64).reshape(4, 4)[:3, 1])[2])
        if flipped.score > view.score + MIRROR_GAIN:
            video = "looks MIRRORED"
        elif view.score >= flipped.score:
            video = "not mirrored"
        else:
            video = "mirroring unclear"
        rooms = ", ".join(f"{names.get(r, r)} {share:.0%}" for r, share in view.rooms[:3])
        print(
            f"  {mark.id} faces {_compass(viewed)} ({viewed:.0f} deg) by its view, {_compass(registered)} "
            f"({registered:.0f} deg) by its registration, {apart:.0f} deg apart (its view's best turns span "
            f"{view.spread:.0f} deg either side); sees {rooms or 'no room'}; its view in sight of it on the plan "
            f"{view.inside_now:.0%} and on the walls {view.wall_now:.0%} where registered, {view.inside:.0%} and "
            f"{view.wall:.0%} at its measured spot"
        )
        print(
            f"    video {video}: its view fits {view.score:.2f} as it is, {flipped.score:.2f} mirrored (in sight plus {pi.WALL_WEIGHT} x on the walls); its frame's down points "
            f"{'down, so it is not upside down' if down < 0 else 'UP: the video is upside down'} ({down:+.2f})"
        )
        out[mark.id] = (view, registered, viewed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", required=True, help="place id or name")
    ap.add_argument("--plan", type=Path, help="the owner's plan, floorplan DSL")
    ap.add_argument("--extra", type=Path, action="append", default=[], help="more statements (x-camera spots)")
    ap.add_argument("--seen", action="append", default=[], help="<camera>=<room id> the camera sees")
    ap.add_argument("--title", help="the stored plan's title (default: the place's name)")
    ap.add_argument("--revision", help="register to this revision (default: the newest with a map)")
    ap.add_argument("--walk-revision", help="also measure this revision's own map against the plan")
    ap.add_argument("--day", help="count position rows on this local day (default: today, up to now)")
    ap.add_argument("--anchor-share", type=float, default=0.5, help="weight measured camera spots carry, 0 to 1")
    ap.add_argument("--history", action="store_true", help="list the place's owner plans and stop")
    ap.add_argument("--restore", help="make this earlier owner plan record hold again")
    ap.add_argument(
        "--move-camera",
        action="append",
        default=[],
        help="move this camera and its room to its measured spot when its view fits clearly better there",
    )
    ap.add_argument("--write", action="store_true", help="store it (default: dry run)")
    a = ap.parse_args()
    state = Path(cfg.get("paths.state"))
    place = _place(state, a.place)

    if a.history:
        for r in places.plan_history(state, place.id):
            when = dt.datetime.fromtimestamp(r.start_ns / 1e9).astimezone().isoformat(timespec="seconds")
            rooms = len(floorplan.rooms_from_dsl(str(r.value.get("plan") or "")))
            print(f"{r.id}  {when}  source {r.value.get('source')}  base {r.value.get('base')}  {rooms} rooms")
        return
    if a.restore:
        if a.write:
            places.restore_plan(state, place.id, a.restore)
            print(f"restored {a.restore} as {place.name}'s owner plan")
        else:
            print(f"dry run: would restore {a.restore} as {place.name}'s owner plan (--write to do it)")
        return
    if a.plan is None:
        sys.exit("--plan is needed")

    plan = pi.parse("\n".join(p.read_text() for p in [a.plan, *a.extra]))
    print(
        f"plan: {len(plan.rooms)} rooms, {len(plan.openings)} openings, {len(plan.fixtures)} fixtures, "
        f"{len(plan.cameras)} measured camera spots, {plan.skipped} statements left out"
    )
    seen = _seen(a.seen)
    reg = _register(state, place, a.revision, plan, seen, a.anchor_share)
    views = _views(state, reg, plan)

    moves = {}
    yaws = {c.id: c.yaw for c in plan.cameras if c.yaw is not None}  # facings the owner gave (page bearings)
    for cam in a.move_camera:
        found = views.get(cam)
        if found is None:
            sys.exit(f"--move-camera {cam}: it has no measured spot or isn't registered in {reg.rev.id}")
        view = found[0]
        centre = reg.spots[cam][:2]
        mark = next(c for c in plan.cameras if c.id == cam)
        at = reg.fit.move.point(*mark.at)
        turn, score, now = view.turn, view.score, view.score_now
        if cam in yaws:  # the owner's facing holds; only the spot near the mark is fitted
            pose = np.asarray(reg.rev.cameras[cam].pose, dtype=np.float64).reshape(4, 4)
            look = np.asarray(reg.transform, dtype=np.float64).reshape(3, 4)[:, :3] @ pose[:3, 2]
            turn = pi.turn_to(reg.fit.move, (float(look[0]), float(look[1])), float(yaws[cam]))
            own = _own_points(state, reg, cam)
            mark_at = at
            at, score, now = pi.place_view(plan, reg.fit.move, own, (centre[0], centre[1]), at, turn)
            off = math.dist(reg.fit.move.inverse(np.array([at]))[0], mark.at)
            print(
                f"{cam}: at its given facing {yaws[cam]:.1f} deg, its view fits best {off * 100:.0f} cm from its "
                f"measured spot: score {score:.2f} there, {now:.2f} where registered"
            )
            if off > MARK_TRUST:  # the owner's tape beats a view that fits best that far off: put it on the mark
                at = mark_at
                score = pi.view_score(plan, reg.fit.move, own, (centre[0], centre[1]))(turn, at)
                print(f"{cam}: over {MARK_TRUST * 100:.0f} cm off, so on its measured spot instead: score {score:.2f}")
        if score < now + MOVE_GAIN or score < MOVE_RATIO * now:
            print(
                f"{cam}: its view fits no clearly better at its measured spot ({score:.2f} vs {now:.2f}); not moving it"
            )
            continue
        moves[cam] = pi.turn_about(reg.transform, (centre[0], centre[1]), at, turn)
        print(
            f"{cam}: {'moving' if a.write else 'would move'} it and its room to its measured spot, turned "
            f"{math.degrees(turn):.0f} deg (fuse.move_room): view score {score:.2f} there, {now:.2f} where registered"
        )

    if a.walk_revision:
        walk = _revision(state, place.id, a.walk_revision)
        walk_band, walk_transform = _map_band(state, walk)
        every = {c: p.pose for c, p in walk.cameras.items()}
        walk_fit = pi.register(plan, walk_band, {}, mirror=pi.page_is_mirrored(every, walk_transform))
        print(
            f"walk revision {walk.id} ({walk.producer}) registered alone: median {walk_fit.median * 100:.1f} cm, "
            f"p90 {walk_fit.p90 * 100:.1f} cm from a wall face, {walk_fit.near_share:.0%} of its band within "
            f"{pi.NEAR_FACE * 100:.0f} cm of one, scale a free fit would take {walk_fit.scale:.4f}"
        )

    if a.write and moves:
        from liberational.spatial import fuse

        for cam, transform in moves.items():
            done = fuse.move_room(state, place.id, cam, transform)
            if done is None:
                sys.exit(f"{cam}: move_room found no fused map registering it; nothing more written")
            print(f"{cam}: moved with {done[1]} voxels of its room, revision {done[0].id}")
        print("registering again on the moved map")
        reg = _register(state, place, None, plan, seen, a.anchor_share)
        views = _views(state, reg, plan)

    # each measured spot with the way its view says it looks, as it goes in the plan
    spots = [pi.Camera(c.id, c.at, c.height, _facing(views.get(c.id), c.yaw), c.fov) for c in plan.cameras]
    title = a.title or place.name
    text = pi.to_dsl(plan, reg.fit.move, title, spots)
    moved_rooms = floorplan.rooms_from_dsl(text)
    day = dt.date.fromisoformat(a.day) if a.day else dt.date.today()
    start = dt.datetime.combine(day, dt.time()).astimezone().timestamp()
    end = min(start + 86400, dt.datetime.now().timestamp())
    before = _counts(state, place, start, end, place.plan_revision, None)
    _print_counts(f"{day} before (plan {place.plan_source}, frame {place.plan_revision})", before)
    after = _counts(state, place, start, end, reg.rev.id, moved_rooms)
    _print_counts(f"{day} after (this plan, frame {reg.rev.id})", after)

    if not a.write:
        print("dry run: nothing written (--write to store it)")
        return
    move = reg.fit.move
    # the plan as drawn on its page too, for the Places tab to show upright, with positions turned onto it
    page = pi.to_dsl(plan, pi.Move(0.0, (0.0, 0.0), mirror=False), title, spots)
    view = {"plan": page, "turn": move.turn, "shift": list(move.shift), "mirror": move.mirror}
    places.correct_plan(state, place.id, text, base=reg.rev.id, source="owner", view=view)
    print(f"stored as {place.name}'s owner plan, drawn in revision {reg.rev.id}'s frame; earlier ones: --history")


if __name__ == "__main__":
    main()
