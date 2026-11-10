"""Fixed-camera footprints (part of spatial memory): where people and animals stand on a place's
floor, from see-actions' person and animal tracks on the cameras registered to the place (memory/places.py).

A footprint is one tracked person's foot point in one camera's frame at one moment: the middle of their heels when
pose placed them (halpe26 joints 24 and 25), else of their ankles (15 and 16); a person whose feet pose didn't
place has no footprint. Their box's bottom isn't their feet when furniture hides their legs: on Place 1, 73 to 79% of
such points fell off the map's floor, against 3 to 25% of placed feet's, and they were half of all footprints.
An animal's is the bottom centre of its box (`animal.detection`, see/action_animals.py):
a pet has no pose. Each track is sampled once per SAMPLE_S, so a person standing still for an hour is 3,600 points,
not 30 a second. A footprint is kept as the camera observation it is (camera, u and v as fractions of the
frame, time, track, confidence), not as a position on the floor: each pack-spatial-cameras run registers the cameras
again in a new map frame, so a footprint is projected through the place's current camera pose when it is read
(`project`, places.floor_point), and lands on the plan once the place has one (places.camera_to_plan).

Recording: once a day, from `pack-spatial-cameras` (spatial/cameras.py), each fixed camera's footprints since the
last run go into the position layer (spatial/positions.py) as `fixed-camera` rows, the observation as it is;
`positions.positions` places them on the plan when read. `places/footprints.json` keeps how far each camera is
recorded.

Projected into the map frame, a place's people's footprints are where people walk: its walkable floor, which
spatial/floorplan.py takes beside the place's point cloud (`walkable=`) to close rooms whose walls the fixed cameras
don't see (they hang high and look down, so a camera cloud holds the floor and furniture, and walls only low down).
Animals' footprints are left out of it: a cat on the sofa or a counter isn't standing on the floor. So is a foot on an
occluding edge (`edges`, as positions judges one): pose guesses feet behind a couch's back on its top, and cast
to the floor they land metres past the person, so the couch top and the floor beyond it were learned as walked. Over
14 days on Place 1, that was 84% of cam0's walked footprints
(5,939 of 7,097) and 5% of living-room's; leaving them out took the walked cells (10 cm, two feet or more) from 747 to
390, among them the long streaks along cam0's rays. Those feet that another camera settled had been a median 1.46 m
from where that camera saw the person.
"""

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from liberational import segment_log
from liberational.memory import places

NS = 1_000_000_000
DAY_S = 86400.0
WALK_DAYS = 14  # days of footprints a place's walkable floor reads
FIRST_S = DAY_S  # how far back a camera's first recording reaches
MAX_BACKLOG_S = 7 * DAY_S  # a camera not recorded for longer (a long outage) picks up from this far back
MARKS = "footprints.json"  # under the places folder: each camera's recorded-up-to time
FLUSH_S = 300.0  # pose rows this recent may not be on disk yet: a run records up to this long ago
EDGE_Y = 0.98  # an animal's box reaching this far down the frame is cut off by its edge: its bottom isn't the feet
SAMPLE_S = 1.0  # one footprint per track per this many seconds
JOINT_SURE = 0.3  # a joint placed at least this sure counts as a foot
HEELS, ANKLES = (24, 25), (15, 16)  # halpe26
SCALE = 10000  # pose.track joints are x, y, and confidence times this (see/actions.py)
HOUR_S = 3600.0
#: Metres above the floor a person's foot point is: the heels (halpe26 24 and 25) and ankles (15 and 16) pose places
#: are a few centimetres up, not on the floor. Cast to the floor itself, a foot point lands too far by this height over
#: its ray's slope, more from a low camera looking across a room than from a high one looking down: on Place 1, cam0
#: (1.52 m up) put people about 0.3 m farther than living-room (2.29 m) at 4 to 6 m, and casting both to this height
#: closed most of it. An animal's footprint (its box's bottom) is on the floor.
FOOT_M = 0.07
#: How much floor, in metres along the ray, the surface a foot point's camera sees there with nobody in the way
#: (`places.CameraPose.background`) must hide for the point to be at an occluding edge (`at_edge`): a foot guessed
#: behind that surface can be off by up to that much. Its height alone says less: Place 1's couch back (0.81 to 0.91 m
#: up) hides 1.6 to 2.4 m of floor from cam0, low and looking across the room, while a chair's back hides under 0.8 m
#: from living-room, high and looking down. Flagged by height (0.4 m and up), living-room was at an edge in 89% of the
#: seconds both cameras placed one person, and still 0.24 m from cam0 where cam0 wasn't; by this, 0.4%, and cam0 in
#: 79%, 0.89 m from living-room there. Of 48 feet pose guessed behind the couch it flags 90%, of living-room's 42
#: hand-checked feet on the floor 2%.
EDGE_M = 1.0
#: How high a surface people stand or sit behind may be (a couch's back, a counter, a table), in metres: a foot point
#: whose background cell no registration saw (`filled` found it nowhere) is judged as if this stood there, so it's on
#: an occluding edge when such a surface would hide EDGE_M of floor along its ray, and clear only when even that
#: couldn't. On Place 1, 0.8, 1.0, and 1.2 m all made the same 4 of cam0's and 253 of
#: living-room's 22,454 person rows doubtful.
FURNITURE_M = 1.0
#: Two backgrounds of one camera are one view (`same_view`) when this share of the cells both know agree to SAME_CELL
#: of the camera's height, over at least SAME_MIN_CELLS. Place 1's two cameras' backgrounds on successive nights agree
#: in 92 to 100% of cells.
SAME_VIEW = 0.8
SAME_CELL = 0.1
SAME_MIN_CELLS = 100
CELL = 1 / 500  # the walkable read's quantum of a frame (u and v): about 4 px across a 1920 px frame


@dataclass(frozen=True)
class Footprint:
    """One person's foot point in one camera's frame at one moment."""

    camera: str
    t: float  # epoch seconds
    u: float  # 0 to 1 across the frame, from the left
    v: float  # 0 to 1 down the frame, from the top
    # see-actions' track id: never an identity, but unique per camera across restarts, so "<camera>:<track>" stays one
    # path. A person's is a number (actions_worker starts each run's ids from its start time, in milliseconds times
    # 1000); an animal's is its "<run>:<species>:<n>" (people/pets.py's TRACK).
    track: int | str
    confidence: float  # the detector's score, times the foot joints' confidence when pose placed them


def foot_point(box: Sequence[float], joints: Sequence[int] | None) -> tuple[float, float, float] | None:
    """(u, v, confidence) of a person's feet from a pose.track row's box (x1, y1, x2, y2, score) and joints: their
    heels, else their ankles; None when pose placed neither."""
    if len(box) < 4:
        return None
    score = float(box[4]) if len(box) > 4 else 1.0
    if joints is not None and len(joints) >= 26 * 3:
        for pair in (HEELS, ANKLES):
            placed = [
                (joints[3 * j] / SCALE, joints[3 * j + 1] / SCALE, joints[3 * j + 2] / SCALE)
                for j in pair
                if joints[3 * j + 2] / SCALE >= JOINT_SURE and joints[3 * j] >= 0
            ]
            if placed:
                u = sum(p[0] for p in placed) / len(placed)
                v = sum(p[1] for p in placed) / len(placed)
                return u, v, score * min(p[2] for p in placed)
    return None


def sample(camera: str, rows: Mapping[str, Any]) -> list[Footprint]:
    """One footprint per track per SAMPLE_S from one camera's pose.track columns (`ts_ns`, `track_id`, `box`,
    `joints`, as read_segments gives them), in time order."""
    order = sorted(range(len(rows["ts_ns"])), key=lambda i: int(rows["ts_ns"][i]))
    last: dict[int, float] = {}
    out = []
    for i in order:
        t, track = int(rows["ts_ns"][i]) / NS, int(rows["track_id"][i])
        if t - last.get(track, -math.inf) < SAMPLE_S:
            continue
        found = foot_point([float(b) for b in rows["box"][i]], [int(j) for j in rows["joints"][i]])
        if found is None:
            continue
        last[track] = t
        out.append(Footprint(camera, t, found[0], found[1], track, round(found[2], 3)))
    return out


def sample_animals(camera: str, rows: Mapping[str, Any]) -> list[Footprint]:
    """One footprint per animal track per SAMPLE_S from one camera's animal.detection columns (`ts_ns`, `track_id`,
    `box`): the bottom centre of its box, at half the detector's score (a guess at the feet, as a person's box
    bottom is), in time order. A box cut off by the frame's bottom edge has no feet in view."""
    order = sorted(range(len(rows["ts_ns"])), key=lambda i: int(rows["ts_ns"][i]))
    last: dict[str, float] = {}
    out = []
    for i in order:
        t, track = int(rows["ts_ns"][i]) / NS, str(rows["track_id"][i])
        box = [float(b) for b in rows["box"][i]]
        if t - last.get(track, -math.inf) < SAMPLE_S or len(box) < 5 or box[3] >= EDGE_Y:
            continue
        last[track] = t
        out.append(Footprint(camera, t, (box[0] + box[2]) / 2, box[3], track, round(box[4] * 0.5, 3)))
    return out


def collect(state: Path, start: float, end: float, cameras: Sequence[str]) -> list[Footprint]:
    """The footprints of `cameras` over [start, end), people's and animals', read an hour at a time (pose.track is a
    row per person per frame)."""
    from continuum_py import read_segments

    from liberational.see.actions import POSE_SCHEMA
    from liberational.see.actions_worker import POSES

    root = segment_log.root(state)
    out: list[Footprint] = []
    for camera in cameras:
        at = start
        while at < end:
            upto = min(end, at + HOUR_S)
            lo, hi = round(at * NS), round(upto * NS)
            out += sample(camera, read_segments(root, POSES, POSE_SCHEMA, lo, hi, [camera]))
            animals = read_segments(
                root, segment_log.ANIMAL_DETECTIONS, segment_log.ANIMAL_DETECTION_SCHEMA, lo, hi, [camera]
            )
            out += sample_animals(camera, animals)
            at = upto
    return sorted(out, key=lambda f: f.t)


HEAD = 17  # halpe26's head (the top of it)
HIPS = (11, 12)
#: A sighting is upright when its legs, hips to feet, are at least this share of its head-to-feet height in the frame:
#: someone sitting, bending, or behind furniture has shorter legs there. On Place 1's cam0, people counted as walking
#: measured 1.2 m tall with their hips at 0.3 m; the upright ones 1.8 to 1.9 m with their hips at 0.9 m.
UPRIGHT_LEGS = 0.42


@dataclass(frozen=True)
class Sighting:
    """A person's feet, and the top of their head when pose placed it, in one camera's frame at one moment: what
    tier 1 calibrates the fixed cameras by (spatial/fuse.py: people's height fixes a map's metric scale, and one
    person seen by two cameras at once aligns them). Fractions of the frame, as Footprint's."""

    camera: str
    t: float
    track: int
    u: float
    v: float
    head: tuple[float, float] | None
    upright: bool = False  # pose placed the feet, hips, and head, and the legs are UPRIGHT_LEGS of the body or more


def _placed(joints: Sequence[int], k: int) -> tuple[float, float] | None:
    """Joint `k`'s (x, y) as fractions of the frame, when pose placed it at least JOINT_SURE."""
    if len(joints) < 3 * k + 3 or joints[3 * k] < 0 or joints[3 * k + 2] / SCALE < JOINT_SURE:
        return None
    return joints[3 * k] / SCALE, joints[3 * k + 1] / SCALE


def sightings_of(camera: str, rows: Mapping[str, Any]) -> list[Sighting]:
    """One Sighting per track per SAMPLE_S from one camera's pose.track columns, as `sample` takes them."""
    order = sorted(range(len(rows["ts_ns"])), key=lambda i: int(rows["ts_ns"][i]))
    last: dict[int, float] = {}
    out = []
    for i in order:
        t, track = int(rows["ts_ns"][i]) / NS, int(rows["track_id"][i])
        if t - last.get(track, -math.inf) < SAMPLE_S:
            continue
        joints = [int(j) for j in rows["joints"][i]]
        found = foot_point([float(b) for b in rows["box"][i]], joints)
        if found is None:
            continue
        last[track] = t
        head = _placed(joints, HEAD)
        hips = [p for k in HIPS if (p := _placed(joints, k)) is not None]
        hip_v = sum(p[1] for p in hips) / len(hips) if hips else None
        upright = (
            head is not None
            and hip_v is not None
            and head[1] < hip_v < found[1]
            and (found[1] - hip_v) >= UPRIGHT_LEGS * (found[1] - head[1])
        )
        out.append(Sighting(camera, t, track, found[0], found[1], head, upright))
    return out


def sightings(state: Path, start: float, end: float, cameras: Sequence[str]) -> list[Sighting]:
    """The people `cameras` saw over [start, end) as Sightings, in time order, read an hour at a time; none when the
    state has no segment store."""
    from continuum_py import read_segments

    from liberational.see.actions import POSE_SCHEMA
    from liberational.see.actions_worker import POSES

    root = segment_log.root(state)
    if not root.is_dir():
        return []
    out: list[Sighting] = []
    for camera in cameras:
        at = start
        while at < end:
            upto = min(end, at + HOUR_S)
            out += sightings_of(
                camera, read_segments(root, POSES, POSE_SCHEMA, round(at * NS), round(upto * NS), [camera])
            )
            at = upto
    return sorted(out, key=lambda s: s.t)


@dataclass(frozen=True)
class Cells:
    """One camera's footprints over a stretch as cells of its frame (CELL across): each cell's centre and how many
    footprints fell in it. People walk the same paths, so a fortnight of them is far fewer cells than footprints."""

    camera: str
    uv: np.ndarray  # (K, 2): u and v, fractions of the frame
    counts: np.ndarray  # (K,): footprints in each cell


def _cells(camera: str, keys: Sequence[np.ndarray], counts: Sequence[np.ndarray]) -> Cells | None:
    """Quantized (u, v) keys with counts, from any number of reads, as one camera's Cells; None for none."""
    if not keys:
        return None
    merged, inverse = np.unique(np.concatenate(keys), axis=0, return_inverse=True)
    total = np.bincount(inverse.ravel(), weights=np.concatenate(counts)).astype(np.int64)
    return Cells(camera, (merged + 0.5) * CELL, total)


def _quantize(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys, counts = np.unique(np.floor(uv / CELL).astype(np.int64), axis=0, return_counts=True)
    return keys, counts


def cells(feet: Iterable[Footprint]) -> list[Cells]:
    """Footprints in memory as each camera's Cells, quantized as `walked` reads them."""
    by_camera: dict[str, list[tuple[float, float]]] = {}
    for f in feet:
        by_camera.setdefault(f.camera, []).append((f.u, f.v))
    out = []
    for camera, uv in by_camera.items():
        keys, counts = _quantize(np.asarray(uv, dtype=np.float64).reshape(-1, 2))
        found = _cells(camera, [keys], [counts])
        if found is not None:
            out.append(found)
    return out


def project(
    walked: Iterable[Cells], cameras: Mapping[str, places.CameraPose], floor: Sequence[float] | None
) -> tuple[np.ndarray, np.ndarray]:
    """The cells of `cameras` whose centres reach the floor within places.MAX_FLOOR_M of their camera and aren't on an
    occluding edge (`edges`): (points, an (N, 3) array in the map's frame; weights, the footprints each stands for).
    places.floor_point, a camera at a time over arrays, for people's foot points FOOT_M up. A foot on an edge is
    hidden behind a couch's back or a counter and guessed by pose, or just beyond it, so cast to the floor it lands
    metres past the person, on the furniture's far side: it isn't where anyone walked."""
    empty = np.zeros((0, 3)), np.zeros(0)
    if floor is None:
        return empty
    normal, off = np.asarray(floor[:3], dtype=np.float64), float(floor[3])
    points, weights = [], []
    for c in walked:
        cam = cameras.get(c.camera)
        if cam is None or not len(c.uv):
            continue
        m = np.asarray(cam.pose, dtype=np.float64).reshape(4, 4)
        a = np.asarray(c.uv, dtype=np.float64)
        if cam.lens is not None:  # a wide lens undone first, as floor_point does
            from liberational.spatial import lens

            a = np.column_stack(lens.undistort_points(lens.Lens.from_tuple(cam.lens), a[:, 0], a[:, 1]))
        d = np.column_stack(
            [(a[:, 0] - 0.5) * cam.width / cam.focal, (a[:, 1] - 0.5) * cam.height / cam.focal, np.ones(len(a))]
        )
        rays = d @ m[:3, :3].T
        origin = m[:3, 3]
        height = float(normal @ origin) + off
        rate = rays @ normal
        unit = float(np.linalg.norm(normal))
        above = FOOT_M * unit  # as floor_point: the foot point's plane, in the floor's own units
        if height <= above:
            continue  # a camera at or under the feet
        down = rate < -1e-9  # rays level with or rising off the floor never meet it
        s = np.where(down, (above - height) / np.where(down, rate, -1.0), np.inf)
        length = np.linalg.norm(rays, axis=1)
        # As floor_point: a ray meeting the floor too shallowly lands metres off for a degree of floor tilt.
        steep = -rate / (length * unit) >= math.sin(math.radians(places.MIN_RAY_DEG))
        near = down & steep & (s * length <= places.MAX_FLOOR_M)
        on = origin + s[near, None] * rays[near] - (above / unit**2) * normal
        clear = ~edges(cam, np.asarray(c.uv, dtype=np.float64)[near], on, height / unit, origin, normal / unit)
        points.append(on[clear])
        weights.append(np.asarray(c.counts, dtype=np.float64)[near][clear])
    return (np.concatenate(points), np.concatenate(weights)) if points else empty


def at_edge(
    cam: places.CameraPose, floor: Sequence[float] | None, u: float, v: float, surface: float | None = None
) -> bool | None:
    """Whether a foot point (u, v, fractions of the camera's frame) sits on an occluding edge: what the camera
    sees at that pixel with nobody in the way (a couch's back, a table, a bed) hides at least EDGE_M of floor along the
    ray, so the feet are either hidden behind it and guessed by pose, or just beyond it; from this camera alone the two
    look the same, and the guess can be off by up to the floor hidden. Another camera that sees that floor can tell.
    The floor hidden is the surface's height over the foot's (FOOT_M) times the ray's run over its drop. `surface`, a
    height above the floor, stands in for the background's there (a cell no registration saw, `FURNITURE_M`). None
    when unknown: no background (a camera registered alone) and no `surface`, no floor, nothing sure at that pixel, or
    a ray that doesn't reach the floor. Map units, as `floor_point`: metres once tier 1 has fused the place."""
    if (cam.background is None and surface is None) or floor is None:
        return None
    seen = cam.background.at(u, v) if surface is None and cam.background is not None else None
    feet = places.floor_point(cam, floor, u, v, FOOT_M)
    if (surface is None and seen is None) or feet is None:
        return None
    n = np.asarray(floor[:3], dtype=np.float64)
    unit = float(np.linalg.norm(n))
    n = n / unit
    origin = np.asarray(cam.pose, dtype=np.float64).reshape(4, 4)[:3, 3]
    up = float(n @ origin) + float(floor[3]) / unit
    if up <= FOOT_M:
        return None
    height = surface if surface is not None else float(seen or 0.0) * up
    offset = np.asarray(feet) - origin
    run = float(np.linalg.norm(offset - (offset @ n) * n))  # along the floor, from under the camera to the feet
    return max(height - FOOT_M, 0.0) * run / (up - FOOT_M) >= EDGE_M


def edges(
    cam: places.CameraPose, uv: np.ndarray, feet: np.ndarray, up: float, origin: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """`at_edge` over arrays, for `project`: whether each foot point (uv, (K, 2) fractions of the camera's frame, raw
    as `Background.at` takes them) whose point on the floor is `feet` ((K, 3)) is on an occluding edge, for a camera
    `up` above the floor (unit normal `n`) at `origin`. A cell of the background no registration saw is judged as if
    FURNITURE_M stood there, as positions does; a camera with no background has no edges."""
    out = np.zeros(len(uv), dtype=bool)
    bg = cam.background
    if bg is None or not len(uv) or up <= FOOT_M:
        return out
    heights = np.asarray([np.nan if h is None else h for h in bg.heights], dtype=np.float64)
    inside = (uv[:, 0] >= 0.0) & (uv[:, 0] < 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] < 1.0)
    cell = np.floor(uv[:, 1] * bg.rows).astype(np.int64) * bg.cols + np.floor(uv[:, 0] * bg.cols).astype(np.int64)
    seen = np.where(inside, heights[np.where(inside, cell, 0)], 0.0)
    surface = np.where(np.isnan(seen), FURNITURE_M, seen * up)
    offset = feet - origin
    run = np.linalg.norm(offset - (offset @ n)[:, None] * n, axis=1)  # along the floor, from under the camera
    return inside & (np.maximum(surface - FOOT_M, 0.0) * run / (up - FOOT_M) >= EDGE_M)


def same_view(a: places.Background, b: places.Background) -> bool:
    """Whether two backgrounds of one camera are of one view (the camera wasn't moved or re-aimed between them): of
    the cells both know, at least SAME_VIEW agree to SAME_CELL of the camera's height, over at least SAME_MIN_CELLS."""
    if (a.cols, a.rows) != (b.cols, b.rows):
        return False
    both = [(x, y) for x, y in zip(a.heights, b.heights) if x is not None and y is not None]
    return len(both) >= SAME_MIN_CELLS and sum(abs(x - y) <= SAME_CELL for x, y in both) >= SAME_VIEW * len(both)


def filled(background: places.Background, earlier: Iterable[places.Background]) -> places.Background:
    """`background` with each cell it doesn't know taken from the first of `earlier` (newest first: the camera's
    backgrounds from earlier registrations) of the same view (`same_view`) that knows it. A registration
    leaves out the people in its frame, so a camera's background is unknown where someone sat then, which is
    where people sit and furniture hides their feet; people move and the couch doesn't, so another night's frame
    usually saw it."""
    heights = list(background.heights)
    for other in earlier:
        if None not in heights:
            break
        if not same_view(background, other):
            continue
        for i, h in enumerate(heights):
            if h is None and other.heights[i] is not None:
                heights[i] = other.heights[i]
    return places.Background(background.cols, background.rows, tuple(heights))


def known(background: places.Background) -> int:
    """How many of a background's cells it knows."""
    return sum(h is not None for h in background.heights)


def kept(new: places.Background | None, old: places.Background | None) -> places.Background | None:
    """The background a camera keeps when it is registered again where it was (`fuse.seat`): `new` with its
    unknown cells taken from `old` when the two are one view (`filled`), `new` alone when they are two views (the camera
    was re-aimed), and whichever knows more cells when they share too few to tell. So a sparse snapshot never replaces
    a fuller background of the same view, and a camera with none tonight keeps the one it had."""
    if new is None or old is None:
        return new if old is None else old
    if (new.cols, new.rows) == (old.cols, old.rows):
        shared = sum(a is not None and b is not None for a, b in zip(new.heights, old.heights))
        if shared < SAME_MIN_CELLS:
            return new if known(new) >= known(old) else old
    return filled(new, [old])


def _rows(feet: Iterable[Footprint]) -> list[dict[str, Any]]:
    from liberational.spatial import positions

    return [
        dict(
            t=f.t,
            source=positions.FIXED_CAMERA,
            confidence=f.confidence,
            entity=None,
            track=f"{f.camera}:{f.track}",
            camera=f.camera,
            u=f.u,
            v=f.v,
        )
        for f in feet
    ]


def record(state: Path, cameras: Sequence[str], now: float) -> int:
    """Write each camera's footprints since it was last recorded (FIRST_S back the first time, at most MAX_BACKLOG_S)
    up to FLUSH_S ago, into the position layer; how many. The mark stops at FLUSH_S ago too, so rows still being
    flushed are read by the next run rather than skipped."""
    from liberational.spatial import positions

    path = places.root(state) / MARKS
    try:
        marks = {str(k): float(v) for k, v in json.loads(path.read_text()).items()}
    except FileNotFoundError, ValueError, AttributeError:
        marks = {}
    total = 0
    upto = now - FLUSH_S
    for camera in cameras:
        start = max(marks.get(camera, upto - FIRST_S), upto - MAX_BACKLOG_S)
        if start >= upto:
            continue
        feet = collect(state, start, upto, [camera])
        positions.write_positions(state, _rows(feet))
        total += len(feet)
        marks[camera] = upto
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(marks, indent=1, sort_keys=True))
        tmp.replace(path)  # after each camera, so a failed one doesn't lose the others' marks
    return total


def walked(state: Path, cameras: Sequence[str], now: float, days: float = WALK_DAYS) -> list[Cells]:
    """The `cameras`' recorded people's footprints over the last `days` as Cells (the walkable floor projects them;
    animals' are left out): read as columns a day at a time and quantized as they're read, never a record per
    footprint. A fortnight of two busy cameras is about 740,000 footprints, which made the nightly run build that
    many objects. The store applies forgets as it reads, as `segment_log.Log` does. It reads the raw stream only, not retention's
    `.condensed` companion, which holds rows past the 30-day window: WALK_DAYS must stay under that window."""
    from continuum_py import read_segments

    from liberational.spatial import positions

    root = segment_log.root(state)
    out = []
    for camera in cameras:
        keys: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        at, end = now - days * DAY_S, now
        while at < end:
            upto = min(end, at + DAY_S)
            got = read_segments(
                root,
                segment_log.POSITIONS,
                segment_log.POSITION_SCHEMA,
                round(at * NS),
                round(upto * NS),
                [segment_log.slot(camera)],
            )
            u = np.asarray(got.get("u", []), dtype=np.float64)
            v = np.asarray(got.get("v", []), dtype=np.float64)
            source = np.asarray(got.get("source", []), dtype=object)
            person = np.asarray([positions.animal_track(t) is None for t in got.get("track", [])], dtype=bool)
            ok = np.isfinite(u) & np.isfinite(v) & (source == positions.FIXED_CAMERA) & person
            if ok.any():
                k, c = _quantize(np.column_stack([u[ok], v[ok]]))
                keys.append(k)
                counts.append(c)
            at = upto
        found = _cells(camera, keys, counts)
        if found is not None:
            out.append(found)
    return out
