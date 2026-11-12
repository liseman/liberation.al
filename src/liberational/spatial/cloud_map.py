"""A place's map: one point cloud that builds up as its cameras look around (#546).

A camera's own cloud is one shot: what it saw from where it stood. Drawn side by side, shots of the same wall are
several walls. A place's map is the one cloud they add up to, in the plan's frame: every placed shot is offered to
it (a camera's cloud when it rests, and the keyframes of a camera tracked as it moves, spatial/camera_track.py),
so the map fills in as cameras move through the place, and what two shots both saw is there once.

**Adding a shot** (`offer`). In this order:

1. *A shot the plan's walls don't hold is first brought into agreement with the map.* It stands at its camera's
   spot, at the model's own size, and can be a metre out. Its walls and raised flat surfaces are compared with
   those of the held shots already in the map over the space they share, and it is turned, slid, and resized onto
   them within bounds (spatial/cloud_merge.py `agree`, plane to plane). If that can't be done, or it shares
   nothing with them, it goes on as it stands, and says how far out it is. A shot the plan's walls hold is where
   it is: the plan is the reference, not the map.
2. *What was looked through gives way.* A shot saw as far as its points, and no farther (`cloud_merge.View`). A
   point of the map that the new shot looked straight through wasn't there when the shot was taken: a chair that
   was moved, a door that was shut, or an earlier shot's guess at a depth. Those points go. The same is asked the
   other way round, of the new shot's points that an earlier shot looked through.
   When either is more than `SMALL` of what the two have in common, the shots are in conflict, and the question is
   which to believe. The newer one, unless the plan explains it clearly worse than the shots it conflicts with
   (`Offer.score`, by `BETTER`; a shot the plan's walls don't hold scores nothing): then the map stays as it was,
   and the new shot is added without its points that the better shots looked through. And when the conflict is more
   than `WHOLE` of what they have in common, it is not something in the room that moved: the shot as a whole
   stands somewhere else than the map's shots do (its wall a metre inside the room). Such a shot is refused unless
   the plan explains it clearly better, in which case it is the map that was wrong. So the map follows a world
   that changes, a bad shot is not the map for ever, and one bad shot can't erase a good map.
3. *Up to a density.* The map is cut into cells of `CELL_M`, and a cell holds at most `CAP` points. A cell the map
   has nothing in takes the shot's points there. A cell it already has points in keeps them, unless the shot has
   more there: then the shot sees that surface better (nearer, or more squarely), and its points take their
   place. So a camera that stays put adds nothing after its first shot, a camera that moves adds what is new and
   sharpens what it gets closer to, and no cell grows past the cap. The cells are counted where a point is drawn
   flat (a fitted cloud's wall points lie on the plan's wall, its floor points on the floor,
   spatial/cloud_align.py `flatten`), so a wall is one surface however many shots saw it. Each point keeps both
   places: where its shot had it, and where it is drawn flat.

A camera's cloud that is placed again (its working pose changed) takes the place of its earlier self.

**Size.** `CELL_M` is 6 cm and `CAP` 8: about 2,200 points to a square metre of surface, a little under what a
camera's own cloud has of a wall three metres away, so the map looks like the clouds it is made of from where they
were taken. Four live cameras' clouds (420,000 points) made a map of about a third of that. A house of 150 square
metres of floor has some 500 of surfaces: 1.1 million points, about 35 MB on disk. `MOST` (2 million points) is the
bound whatever the house.

**Kept** under the clouds' folder, `maps/<place>/`: `points.npz` (the points, their colours, which shot each came
from, and what each shot saw) and `map.json` (the shots added, with where their surfaces stand, the last `REFUSED`
refused, and the plan the map is of). It is rewritten whole when a shot changes it, under a lock, since the
surroundings service and the dashboard can both offer a shot. It survives restarts. It is machine-made, not taught:
it is built again from the cameras' clouds, `pack-reset-places` removes it with them, and a plan whose walls changed
drops it (the frame it was fitted to is gone).

**How much of the place it covers** (`_floor`) is kept with it: the share of the plan's floor, in cells of
`FLOOR_CELL_M`, that holds a point of the map lying on the floor. It is the floor that was seen, not the walls or
what stands in the rooms, and a cell counts whole, so it reads a little generous far from a camera.

**What it costs**, measured on the Linux machine beside the other services: offering a shot of 100,000 to 120,000
points to a map of 150,000 took 0.2 to 0.3 s of one CPU core and no GPU; a shot the map has already answered,
about 10 ms. Building the map from four cameras' clouds took 1.6 s with their squaring and fit.
"""

from __future__ import annotations

import contextlib
import fcntl
import io
import math
import shutil
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from liberational import json_state, logs
from liberational.memory import places
from liberational.spatial import cloud_align, cloud_merge

LOG = logs.get("camera-clouds")

FOLDER = "maps"  # under the clouds' folder
CELL_M = 0.06  # the cells a map is cut into
CAP = 8  # points a cell takes
MOST = 2_000_000  # points a map holds, whatever the place
SMALL = 0.05  # of what two shots have in common: with less than this looked through, they don't conflict
WHOLE = 0.3  # and with more than this, it is the shot as a whole that is out, not something in the room that moved
BETTER = 0.1  # a newer shot gives way to the ones it conflicts with only when the plan explains it this much worse
MIN_SEEN = 50  # points of an earlier shot a new one must look at before the two are compared
FLOOR_CELL_M = 0.25  # the cells the plan's floor is counted in, for how much of it the map covers
FLOOR_BAND_M = 0.05  # a point this near z = 0 where it is drawn flat lies on the floor
REFUSED = 50  # refused shots that are remembered, so each is tried once and its camera can say why


@dataclass(frozen=True)
class Offer:
    """A shot offered to a place's map: a levelled cloud, and where it stands."""

    id: str  # of this shot at this pose: offered again, the answer is the one given before
    cloud: str  # of the picture: the same picture placed again replaces its earlier self
    camera: str
    at: float  # when the picture was taken
    held: bool  # the plan's walls hold it where it stands
    score: float  # how much of it the plan explains from there (spatial/cloud_align.py `Fit.share`); 0 if unheld
    own: np.ndarray  # N x 3, in the cloud's own frame
    rgb: np.ndarray  # N x 3 uint8
    surfaces: cloud_merge.Surfaces
    stand: cloud_merge.Stand
    up: float  # its size upwards (`Fit.up`)
    camera_pose: np.ndarray  # the camera in the cloud's own frame, 4 x 4
    focal: float
    width: int
    height: int
    fit: cloud_align.Fit | None = None  # for drawing its matched walls and floor flat


@dataclass(frozen=True)
class Outcome:
    """What came of offering a shot."""

    #: "added"; "refused" (`why`: "apart": it couldn't be brought into agreement with the map, or as a whole it
    #: stands somewhere else than the map's shots; "full": the map holds `MOST`)
    state: str
    why: str = ""
    points: int = 0  # of the shot that went into the map
    off_m: float = 0.0  # for "apart": how far its shared surfaces are from the map's
    conflict: float = 0.0  # of what it has in common with the map, the share one looked through the other's
    moved_m: float = 0.0  # how a loose shot was moved to agree: its camera's slide, turn, and resize
    turned_deg: float = 0.0
    resized: float = 1.0
    stand: tuple[float, float, float, float] | None = None  # where it stands in the map: x, y, bearing, scale
    replaced: int = 0  # points of the map it looked through, which gave way
    left_out: int = 0  # points of its own that better-placed shots looked through, which were not added


def folder(clouds: Path, place: str) -> Path:
    """Where `place`'s map is kept, under the clouds' folder."""
    return clouds / FOLDER / place


@contextlib.contextmanager
def _locked(where: Path) -> Generator[None]:
    where.mkdir(parents=True, exist_ok=True)
    with open(where / "lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)


@dataclass
class Held:
    """A map as it is read: its points, and the record of its shots."""

    xyz: np.ndarray  # N x 3 float32: each point where its shot had it
    flat: np.ndarray  # N x 3 float32: and where it is drawn flat
    rgb: np.ndarray  # N x 3 uint8
    shot: np.ndarray  # N uint16: its shot's place in `meta["shots"]`
    meta: dict[str, Any]
    eyes: np.ndarray  # S x 20 float64: for each shot, its `cloud_merge.View`'s eye (16), focal, width, height, spare
    fars: np.ndarray  # S x GRID*GRID float32: and how far it saw


def _empty(plan: str) -> Held:
    none = np.zeros((0, 3), dtype=np.float32)
    return Held(none, none.copy(), np.zeros((0, 3), dtype=np.uint8), np.zeros(0, dtype=np.uint16),
                {"plan": plan, "rev": 0, "shots": [], "refused": []}, np.zeros((0, 20)),
                np.zeros((0, cloud_merge.GRID**2), dtype=np.float32))  # fmt: skip


def record(where: Path) -> dict[str, Any] | None:
    """A map's record (`map.json`) alone, or None when there is no map: cheap, for a scene that only says what a
    camera has in it."""
    got = json_state.read(where / "map.json")
    return got if isinstance(got, dict) and isinstance(got.get("shots"), list) else None


def read(where: Path) -> Held | None:
    """The map kept in `where`, or None when there is none or it doesn't read."""
    meta = record(where)
    if meta is None:
        return None
    try:
        with np.load(where / "points.npz") as kept:
            return Held(kept["xyz"], kept["flat"], kept["rgb"], kept["shot"], meta, kept["eyes"], kept["fars"])
    except OSError, ValueError, KeyError:
        return None


def _floor(flat: np.ndarray, rooms: Sequence[Sequence[tuple[float, float]]]) -> tuple[float, float]:
    """How much of the plan's floor a map covers, and how much floor the plan has, in square metres: the cells of
    `FLOOR_CELL_M` inside the plan's rooms that hold a point of the map lying on the floor, against the rooms' own
    areas. A cell counts whole, so a floor seen sparsely far off counts as seen."""
    area = 0.0
    for ring in rooms:
        corners = np.asarray(ring, dtype=np.float64)
        x, y = corners[:, 0], corners[:, 1]
        area += abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1))) / 2
    on_floor = flat[np.abs(flat[:, 2]) <= FLOOR_BAND_M][:, :2] if len(flat) else np.zeros((0, 2))
    cells = np.unique(np.floor(on_floor.astype(np.float64) / FLOOR_CELL_M).astype(np.int64), axis=0)
    centres = (cells + 0.5) * FLOOR_CELL_M
    inside = int((~cloud_align.outside(centres, rooms, 0.0)).sum()) if len(cells) and len(rooms) else 0
    return min(inside * FLOOR_CELL_M**2, area), area


def _write(where: Path, held: Held, rooms: Sequence[Sequence[tuple[float, float]]]) -> None:
    # Shots with no point left go, and the points are told their shots' new places.
    left = np.bincount(held.shot, minlength=len(held.meta["shots"])) if len(held.shot) else np.zeros(0, dtype=int)
    keep = [i for i in range(len(held.meta["shots"])) if i < len(left) and left[i] > 0]
    if len(keep) != len(held.meta["shots"]):
        to = np.zeros(max(len(held.meta["shots"]), 1), dtype=np.uint16)
        to[keep] = np.arange(len(keep), dtype=np.uint16)
        held.shot = to[held.shot]
        held.meta["shots"] = [held.meta["shots"][i] for i in keep]
        held.eyes, held.fars = held.eyes[keep], held.fars[keep]
    for i, shot in enumerate(held.meta["shots"]):
        shot["points"] = int((held.shot == i).sum())
    held.meta["refused"] = held.meta["refused"][-REFUSED:]
    held.meta["rev"] = int(held.meta.get("rev", 0)) + 1
    covered, floor = _floor(held.flat, rooms)
    held.meta["floor_m2"], held.meta["plan_m2"] = round(covered, 2), round(floor, 2)
    out = io.BytesIO()
    np.savez(out, xyz=held.xyz, flat=held.flat, rgb=held.rgb, shot=held.shot, eyes=held.eyes, fars=held.fars)
    places.write_whole(where / "points.npz", out.getvalue())
    json_state.write(where / "map.json", held.meta)


def _cells(xyz: np.ndarray) -> np.ndarray:
    cell = np.floor(np.asarray(xyz, dtype=np.float64) / CELL_M).astype(np.int64) + 2**20
    return cell[:, 0] << 42 | cell[:, 1] << 21 | cell[:, 2]


def _room(have: np.ndarray, new: np.ndarray, seed: int, sharpen: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """What a shot whose points are in the cells `new` brings to a map whose points are in the cells `have`: which
    of the shot's points go in, and which of the map's give way to them. A cell the map has nothing in takes the
    shot's points there, up to `CAP`. A cell it has points in takes the shot's instead only when the shot has more
    there (it sees that surface better: nearer, or more squarely), and otherwise nothing: the map's stay. Without
    `sharpen` it only fills empty cells. Which of a crowded cell's points go in is drawn by lot, the same lot for
    the same shot."""
    order = np.random.default_rng(seed).permutation(len(new))
    by_cell = order[np.argsort(new[order], kind="stable")]
    cells = new[by_cell]
    first = np.ones(len(cells), dtype=bool)
    first[1:] = cells[1:] != cells[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(len(cells)), 0))
    rank = np.arange(len(cells)) - start
    brings = np.minimum(np.bincount(np.cumsum(first) - 1)[np.cumsum(first) - 1], CAP) if len(cells) else rank
    taken = np.zeros(len(cells), dtype=np.int64)
    if len(have):
        known, count = np.unique(have, return_counts=True)
        at = np.minimum(np.searchsorted(known, cells), len(known) - 1)
        taken = np.where(known[at] == cells, count[at], 0)
    wins = (rank < CAP) & (brings > taken) & (sharpen | (taken == 0))
    gives_way = np.isin(have, cells[wins & (taken > 0)]) if len(have) else np.zeros(0, dtype=bool)
    return by_cell[wins], gives_way


def _lie(stand: cloud_merge.Stand, up: float) -> np.ndarray:
    """A levelled cloud's frame in the scene, for a camera standing at `stand`."""
    s, c = math.sin(stand.bearing), math.cos(stand.bearing)
    out = np.eye(4)
    out[:3, :3] = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([stand.scale, stand.scale, up])
    out[:2, 3] = (stand.x, stand.y)
    return out


def _surfaces_of(shot: Mapping[str, Any]) -> cloud_merge.Placed:
    raw = shot.get("surfaces") or {}
    walls = tuple(cloud_merge.Wall((w[0], w[1]), (w[2], w[3]), w[4], w[5]) for w in raw.get("walls") or ())
    return cloud_merge.Placed(walls, tuple((f[0], f[1], f[2], f[3]) for f in raw.get("flats") or ()))


def _kept_surfaces(found: cloud_merge.Placed) -> dict[str, Any]:
    return {
        "walls": [[*w.centre, *w.normal, w.half, w.weight] for w in found.walls],
        "flats": [list(f) for f in found.flats],
    }


def _outcome(row: Mapping[str, Any]) -> Outcome:
    """A shot's record as the answer it was given."""
    stand = row.get("stand")
    return Outcome(
        str(row.get("state", "")),
        str(row.get("why", "")),
        int(row.get("points", 0)),
        float(row.get("off_m", 0.0)),
        float(row.get("conflict", 0.0)),
        float(row.get("moved_m", 0.0)),
        float(row.get("turned_deg", 0.0)),
        float(row.get("resized", 1.0)),
        (float(stand[0]), float(stand[1]), float(stand[2]), float(stand[3])) if stand else None,
        int(row.get("replaced", 0)),
        int(row.get("left_out", 0)),
    )


def answered(meta: Mapping[str, Any] | None, shot: str) -> dict[str, Any] | None:
    """What a map said of the shot `shot` when it was offered, or None if it never was."""
    for row in [*(meta or {}).get("shots", ()), *(meta or {}).get("refused", ())]:
        if row.get("id") == shot:
            return row
    return None


def _view(held: Held, i: int) -> cloud_merge.View:
    row = held.eyes[i]
    return cloud_merge.View(row[:16].reshape(4, 4), float(row[16]), int(row[17]), int(row[18]), held.fars[i])


def _without(held: Held, gone: np.ndarray) -> None:
    held.xyz, held.flat, held.rgb, held.shot = (a[~gone] for a in (held.xyz, held.flat, held.rgb, held.shot))


def offer(where: Path, rooms: Sequence[Sequence[tuple[float, float]]], shot: Offer) -> Outcome:
    """Offer `shot` to the map kept in `where`, which is of the plan whose rooms are `rooms` (in the scene's frame):
    the module docstring's Adding a shot. A shot offered before gets the answer it got then. A map of a plan with
    other walls (spatial/cloud_align.py `plan_key`) is dropped first."""
    plan = cloud_align.plan_key(rooms)
    with _locked(where):
        kept = read(where)
        held = _empty(plan) if kept is None or kept.meta.get("plan") != plan else kept
        known = answered(held.meta, shot.id)
        if known is not None:
            return _outcome(known)
        about: dict[str, Any] = {"id": shot.id, "cloud": shot.cloud, "camera": shot.camera, "at": shot.at,
                                 "score": shot.score, "held": shot.held}  # fmt: skip

        def refuse(why: str, conflict: float = 0.0) -> Outcome:
            row = {**about, "conflict": conflict, "state": "refused", "why": why, "points": 0}
            held.meta["refused"] = [*held.meta["refused"], row]
            _write(where, held, rooms)
            return _outcome(row)

        # The same picture, placed again: its earlier self goes.
        earlier = [i for i, row in enumerate(held.meta["shots"]) if row.get("cloud") == shot.cloud]
        if earlier:
            _without(held, np.isin(held.shot, earlier))
        stand = shot.stand
        if not shot.held:  # the plan's walls don't hold it: the map's surfaces may
            theirs = cloud_merge.Placed()
            for i, row in enumerate(held.meta["shots"]):
                if i not in earlier and row.get("held"):
                    theirs = theirs + _surfaces_of(row)
            told = cloud_merge.agree(shot.surfaces, stand, theirs)
            if told is not None and told.at is None:  # it stays where it stands, and says how far out that is
                about["off_m"] = told.off_m
            if told is not None and told.at is not None:
                stand = told.at
                about |= {"off_m": told.off_m, "moved_m": told.moved_m, "turned_deg": told.turned_deg,
                          "resized": told.resized}  # fmt: skip
        lie = _lie(stand, shot.up if shot.held else stand.scale)
        xyz = (shot.own @ lie[:3, :3].T + lie[:3, 3]).astype(np.float32)
        view = cloud_merge.view_of(xyz, np.linalg.inv(lie @ shot.camera_pose), shot.focal, shot.width, shot.height)
        # What the new shot looked through of the map, and what earlier shots looked through of it.
        # A point the other also has a point beside (in its cell) is seen by both, not looked through: a floor
        # seen at a low angle runs through many depths in one direction.
        here, there = _cells(xyz), _cells(held.xyz)
        mine, seen = cloud_merge.through(view, held.xyz)
        mine &= ~np.isin(held.shot, earlier) & ~np.isin(there, here)
        shared = np.isin(here, there)
        worse = False
        theirs_through = np.zeros(len(xyz), dtype=bool)
        theirs_seen = np.zeros(len(xyz), dtype=bool)
        rivals = {int(i) for i, n in zip(*np.unique(held.shot[mine], return_counts=True), strict=True) if n >= MIN_SEEN}
        for i in range(len(held.meta["shots"])):
            if i in earlier:
                continue
            looked, saw = cloud_merge.through(_view(held, i), xyz)
            theirs_seen |= saw
            looked &= ~shared
            if looked.sum() >= MIN_SEEN:
                theirs_through |= looked
                rivals.add(i)
        conflict = max(
            float(mine.sum()) / max(int(seen.sum()), 1), float(theirs_through.sum()) / max(int(theirs_seen.sum()), 1)
        )
        about["conflict"] = round(conflict, 3)
        if rivals and conflict > SMALL:
            best = max(float(held.meta["shots"][i].get("score", 0.0)) for i in rivals)
            if conflict > WHOLE and shot.score < best + BETTER:
                LOG.info("%s: its shot is not added to the map: %.0f%% of it stands where the map's shots saw "
                         "nothing, or looks through what they saw", shot.camera, 100 * conflict)  # fmt: skip
                return refuse("apart", conflict=round(conflict, 3))
            worse = shot.score + BETTER < best
        # A shot the plan explains clearly worse than those it conflicts with moves nothing of the map, and its own
        # points that they looked through are left out. Otherwise it is believed: what it looked through goes.
        kept = ~theirs_through if worse else np.ones(len(xyz), dtype=bool)
        left_out = int((~kept).sum())
        replaced = 0 if worse else int(mine.sum())
        if not worse:
            _without(held, mine)
        if len(held.xyz) >= MOST:
            return refuse("full")
        fitted = shot.fit is not None and shot.fit.state == "fitted" and shot.held
        flat = cloud_align.flatten(shot.own, lie, shot.fit)[0].astype(np.float32) if fitted and shot.fit else xyz
        room, gives_way = _room(_cells(held.flat), _cells(flat[kept]), len(held.meta["shots"]), sharpen=not worse)
        room = np.flatnonzero(kept)[room][: max(MOST - len(held.xyz), 0)]
        index = len(held.meta["shots"])
        added = {**about, "state": "added", "why": "", "points": len(room), "replaced": replaced, "left_out": left_out,
                 "stand": [stand.x, stand.y, stand.bearing, stand.scale],
                 "surfaces": _kept_surfaces(cloud_merge.placed(shot.surfaces, stand))}  # fmt: skip
        if not len(room):  # it saw nothing the map lacks: remembered, so it isn't offered again, and holds no point
            held.meta["refused"] = [*held.meta["refused"], {**added, "surfaces": {}}]
        else:
            _without(held, gives_way)
            held.meta["shots"] = [*held.meta["shots"], added]
            held.xyz = np.concatenate([held.xyz, xyz[room]])
            held.flat = np.concatenate([held.flat, flat[room]])
            held.rgb = np.concatenate([held.rgb, np.asarray(shot.rgb, dtype=np.uint8)[room]])
            held.shot = np.concatenate([held.shot, np.full(len(room), index, dtype=np.uint16)])
            eye = np.r_[view.eye.ravel(), view.focal, view.width, view.height, 0.0]
            held.eyes = np.concatenate([held.eyes, eye[None]])
            held.fars = np.concatenate([held.fars, view.far.astype(np.float32)[None]])
        _write(where, held, rooms)
        LOG.info("%s: %d of its shot's %d points added to the map (%d in all)%s", shot.camera, len(room),
                 len(xyz), len(held.xyz), f"; {replaced} points it looked through gave way" if replaced else "")  # fmt: skip
        return _outcome(added)


def count_floor(where: Path, rooms: Sequence[Sequence[tuple[float, float]]]) -> bool:
    """Count the floor a map covers (`_floor`) when the map kept in `where` is of the plan with these rooms and
    hasn't said yet: one kept before this was counted. Whether it was counted now."""
    meta = record(where)
    if meta is None or "plan_m2" in meta or meta.get("plan") != cloud_align.plan_key(rooms):
        return False
    with _locked(where):
        held = read(where)
        if held is None or "plan_m2" in held.meta:
            return False
        held.meta["rev"] = int(held.meta.get("rev", 0)) - 1  # nothing drawn changes
        _write(where, held, rooms)
        return True


def drop(where: Path) -> None:
    """Forget the map kept in `where`."""
    shutil.rmtree(where, ignore_errors=True)


def by_camera(held: Held) -> dict[str, int]:
    """How many of a map's points came from each camera."""
    counts = np.bincount(held.shot, minlength=len(held.meta["shots"])) if len(held.shot) else []
    out: dict[str, int] = {}
    for row, n in zip(held.meta["shots"], counts, strict=False):
        out[row["camera"]] = out.get(row["camera"], 0) + int(n)
    return out


def cameras_of(held: Held, order: Sequence[str]) -> np.ndarray:
    """For each of a map's points, its camera's place in `order` (65535 for a camera that isn't in it)."""
    place = {camera: i for i, camera in enumerate(order)}
    of_shot = np.array([place.get(row["camera"], 65535) for row in held.meta["shots"]] or [65535], dtype=np.uint16)
    return of_shot[held.shot] if len(held.shot) else np.zeros(0, dtype=np.uint16)
