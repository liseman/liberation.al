"""A place's plan as a Pascal Editor scene: the JSON the dashboard's plan editor loads and saves (part of spatial memory).

A scene is `{"nodes": {<id>: <node>}, "rootNodeIds": [<site id>]}`. Every node has `id`, `type`, `parentId`, and a
free `metadata` record, and its parent lists it in `children`. A plan is one site, one building, and one level, and
under the level:

- `wall`: `start` and `end` (its centreline), `thickness`, and `height`, with its `door` and `window` nodes as
  children. An opening's `position[0]` is how far its middle is along the wall from `start`, and `width` how wide. A
  door whose `openingKind` is `opening` is a doorway with no door in it.
- `zone`: a room, as a `name` and a `polygon`. The editor finds rooms from the walls and keeps these in step.
- nodes the editor makes for itself (a floor `slab`, a `ceiling` per room), which this module carries and moves but
  doesn't read.

Coordinates are the plan frame's: metres, x east and y south on the page, which the editor draws as its x and z. So a
plan's (x, y) is a scene's `[x, z]` unchanged.

What Pascal has no node for rides in `metadata`, which the editor keeps through a save: under the level's
`metadata["liberational"]`, the camera spots (`cameras`: `{<camera>: {"at": [x, y], "height": m, "yaw": degrees
clockwise from up the page, "fov": degrees}}`), `objects`, `unexplored` areas, and `title`; and on a zone, its `kind`.
The editor drops any key added at the top of the document, so nothing of ours goes there.

This module reads and writes that with the standard library only, so `memory/places.py` can use it. The readers that
want arrays, and the converter from the old DSL, are in spatial/plan_import.py, which builds its own shapes from
these.
"""

import json
import math
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

Point = tuple[float, float]
Node = dict[str, Any]
Scene = dict[str, Any]

SIDE = "liberational"  # the key of our records in the level's metadata
VERSION = 1
# What a wall, a door, and a window get when a writer doesn't say (metres).
HEIGHT = 2.4
DOOR_HEIGHT = 2.1
WINDOW_HEIGHT = 1.2
WINDOW_SILL = 0.9
THICKNESS = (0.2, 0.1)  # exterior and partition, when a scene has no wall of the kind to measure
_ON_WALL = 0.02  # how far (metres) a point may be from a wall's centreline and still be on it
_SITE_MARGIN = 2.0  # how far the site reaches past the walls on every side
# The fields that hold points, by node type: a list of [x, z] pairs, or a list of such lists.
_POLYGONS = {"zone": ("polygon",), "slab": ("polygon",), "ceiling": ("polygon",)}
_HOLES = {"zone": ("holes",), "slab": ("holes",), "ceiling": ("holes",)}


class SceneError(ValueError):
    """A plan that isn't a scene this module can use, with why."""


def is_scene(text: str) -> bool:
    """Whether a stored plan is a scene (JSON) and not the old DSL: a DSL plan never starts with a brace."""
    return text.lstrip().startswith("{")


def load(text: str) -> Scene:
    """The scene in `text`. Raises SceneError when it isn't JSON, or has no nodes or no level."""
    try:
        scene = json.loads(text)
    except json.JSONDecodeError as e:
        raise SceneError(f"the plan isn't a scene: {e.msg} at line {e.lineno}") from e
    nodes = scene.get("nodes") if isinstance(scene, dict) else None
    if not isinstance(nodes, dict) or not isinstance(scene.get("rootNodeIds"), list):
        raise SceneError("the plan isn't a scene: it has no nodes")
    if not all(isinstance(n, dict) and isinstance(n.get("type"), str) for n in nodes.values()):
        raise SceneError("the plan isn't a scene: a node has no type")
    level(scene)
    return scene


def dump(scene: Scene) -> str:
    """A scene as the text that is stored: compact, since an editor's save holds every default of every node."""
    return json.dumps(scene, separators=(",", ":"), ensure_ascii=False)


def level(scene: Scene) -> Node:
    """The storey a plan is drawn on: the scene's lowest level (a home's plan has one)."""
    levels = [n for n in scene["nodes"].values() if n["type"] == "level"]
    if not levels:
        raise SceneError("the plan has no level to draw on")
    return min(levels, key=lambda n: _number(n.get("level"), 0.0))


def _number(value: object, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _point(value: object) -> Point | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y)):
            return float(x), float(y)
    return None


def _points(value: object) -> tuple[Point, ...]:
    found = [_point(v) for v in value] if isinstance(value, list) else []
    return tuple(p for p in found if p is not None)


def _under(scene: Scene, parent: Node, kind: str) -> list[Node]:
    """`parent`'s children of one type, in its own order, then any that name it as parent but aren't listed."""
    nodes = scene["nodes"]
    listed = [nodes[i] for i in parent.get("children") or [] if isinstance(i, str) and i in nodes]
    seen = {n["id"] for n in listed if "id" in n}
    rest = [n for n in nodes.values() if n.get("parentId") == parent.get("id") and n.get("id") not in seen]
    return [n for n in (*listed, *rest) if n["type"] == kind]


def records(scene: Scene) -> dict[str, Any]:
    """Our own records in a scene (cameras, objects, unexplored areas, the title): the level's
    `metadata["liberational"]`, or an empty record."""
    side = (level(scene).get("metadata") or {}).get(SIDE)
    return side if isinstance(side, dict) else {}


def _set_records(scene: Scene, side: Mapping[str, Any]) -> None:
    storey = level(scene)
    storey["metadata"] = {**(storey.get("metadata") or {}), SIDE: dict(side)}


def title(scene: Scene) -> str:
    name = records(scene).get("title")
    return name if isinstance(name, str) else ""


def rooms(scene: Scene) -> list[tuple[str, str, str, tuple[Point, ...]]]:
    """Each room as (id, name, kind, polygon in plan metres), in the level's order. A room's id is its zone's id
    without the `zone_` the editor prefixes; its kind is the zone's `metadata["kind"]`, or "other"."""
    out = []
    for zone in _under(scene, level(scene), "zone"):
        polygon = _points(zone.get("polygon"))
        if len(polygon) < 3:
            continue
        kind = (zone.get("metadata") or {}).get("kind")
        name = zone.get("name")
        out.append(
            (
                str(zone.get("id", "")).removeprefix("zone_"),
                name if isinstance(name, str) else "",
                kind if isinstance(kind, str) and kind else "other",
                polygon,
            )
        )
    return out


def walls(scene: Scene) -> list[tuple[str, Point, Point, float]]:
    """Each wall as (id, start, end, thickness), in the level's order. A wall with no thickness of its own has the
    editor's default, 0.1 m."""
    out = []
    for wall in _under(scene, level(scene), "wall"):
        a, b = _point(wall.get("start")), _point(wall.get("end"))
        if a is not None and b is not None and a != b:
            out.append((str(wall.get("id", "")), a, b, _number(wall.get("thickness"), 0.1)))
    return out


def openings(scene: Scene) -> list[tuple[str, str, Point, float]]:
    """Each door, window, and doorway as (kind, wall id, the point its middle is at, width): `door`, `window`, or
    `cased` for a doorway with no door in it."""
    out = []
    for wall in _under(scene, level(scene), "wall"):
        a, b = _point(wall.get("start")), _point(wall.get("end"))
        if a is None or b is None or a == b:
            continue
        length = math.dist(a, b)
        for kind in ("door", "window"):
            for node in _under(scene, wall, kind):
                position = node.get("position")
                d = _number(position[0] if isinstance(position, list) and position else None, length / 2)
                at = (a[0] + (b[0] - a[0]) * d / length, a[1] + (b[1] - a[1]) * d / length)
                named = "cased" if kind == "door" and node.get("openingKind") == "opening" else kind
                out.append((named, str(wall.get("id", "")), at, _number(node.get("width"), 0.9)))
    return out


def room_at(found: Sequence[tuple[str, str, str, Sequence[Point]]], x: float, y: float) -> str | None:
    """The id of the first room whose polygon holds (x, y), or None."""
    for room_id, _, _, poly in found:
        inside = False
        for (ax, ay), (bx, by) in zip(poly, [*poly[1:], poly[0]], strict=True):
            if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                inside = not inside
        if inside:
            return room_id
    return None


def sides(
    found: Sequence[tuple[str, str, str, Sequence[Point]]], a: Point, b: Point, at: Point, reach: float
) -> tuple[str | None, str | None]:
    """The rooms on the left and the right of the wall from `a` to `b`, looked for `reach` metres off it at `at`."""
    length = math.dist(a, b)
    nx, ny = (b[1] - a[1]) / length * reach, -(b[0] - a[0]) / length * reach
    return room_at(found, at[0] + nx, at[1] + ny), room_at(found, at[0] - nx, at[1] - ny)


def thicknesses(scene: Scene) -> tuple[float, float]:
    """The plan's exterior and partition wall thickness, as one number each (the readers of a plan's walls take them
    so): the median of the walls with a room on one side, and of those with a room on both. A kind the scene has no
    wall of gets its usual value."""
    found = rooms(scene)
    outer: list[float] = []
    inner: list[float] = []
    for _, a, b, thickness in walls(scene):
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        left, right = sides(found, a, b, mid, thickness / 2 + 0.05)
        if left is not None and right is not None:
            inner.append(thickness)
        elif left is not None or right is not None:
            outer.append(thickness)
    return (
        statistics.median(outer) if outer else THICKNESS[0],
        statistics.median(inner) if inner else THICKNESS[1],
    )


def spots(scene: Scene) -> list[tuple[str, Point, float | None, float | None, float | None]]:
    """Each camera spot as (camera, at, height, yaw, field of view). A spot with no readable `at` is left out; a
    number that isn't one reads as not given."""
    cameras = records(scene).get("cameras")
    out = []
    for name, spot in (cameras if isinstance(cameras, dict) else {}).items():
        at = _point(spot.get("at")) if isinstance(spot, dict) else None
        if at is None:
            continue
        height, yaw, fov = (
            float(v) if isinstance(v := spot.get(key), (int, float)) and not isinstance(v, bool) else None
            for key in ("height", "yaw", "fov")
        )
        out.append((str(name), at, height, yaw, fov))
    return out


def spot_names(text: str) -> list[str]:
    """The cameras with a spot on a stored scene, or none when the text isn't a scene that reads."""
    try:
        return [name for name, *_ in spots(load(text))]
    except SceneError:
        return []


def rename_cameras(text: str, names: Mapping[str, str]) -> str:
    """A stored scene with each camera spot's camera renamed (`names`: old to new). A scene none of them is on comes
    back as it was, byte for byte."""
    scene = load(text)
    side = records(scene)
    cameras = side.get("cameras")
    if not isinstance(cameras, dict) or not any(name in names for name in cameras):
        return text
    _set_records(scene, {**side, "cameras": {names.get(name, name): spot for name, spot in cameras.items()}})
    return dump(scene)


def moved(text: str, point: Callable[[float, float], Point], yaw: Callable[[float], float]) -> str:
    """A stored scene moved rigidly into another frame: every wall, room, floor, and ceiling outline, the site, and
    our own records, each point through `point` and each camera's facing through `yaw`. Openings ride on their walls,
    so they move with them."""
    scene = load(text)

    def ring(value: object) -> list[list[float]]:
        return [list(point(*p)) for p in _points(value)]

    for node in scene["nodes"].values():
        kind = node["type"]
        if kind == "wall":
            for key in ("start", "end"):
                if (p := _point(node.get(key))) is not None:
                    node[key] = list(point(*p))
        for key in _POLYGONS.get(kind, ()):
            if isinstance(node.get(key), list):
                node[key] = ring(node[key])
        for key in _HOLES.get(kind, ()):
            if isinstance(node.get(key), list):
                node[key] = [ring(hole) for hole in node[key]]
        if kind == "zone" and (p := _point(node.get("seed"))) is not None:
            node["seed"] = list(point(*p))
        if kind == "site" and isinstance(node.get("polygon"), dict):
            node["polygon"] = {**node["polygon"], "points": ring(node["polygon"].get("points"))}
    side = dict(records(scene))
    if isinstance(side.get("cameras"), dict):
        side["cameras"] = {name: _moved_spot(spot, point, yaw) for name, spot in side["cameras"].items()}
    for key in ("objects", "unexplored"):
        if isinstance(side.get(key), dict):
            side[key] = {name: _moved_area(area, point, yaw) for name, area in side[key].items()}
    if side:
        _set_records(scene, side)
    return dump(scene)


def _moved_spot(spot: object, point: Callable[[float, float], Point], yaw: Callable[[float], float]) -> object:
    if not isinstance(spot, dict):
        return spot
    out = dict(spot)
    if (at := _point(spot.get("at"))) is not None:
        out["at"] = list(point(*at))
    if isinstance(spot.get("yaw"), (int, float)):
        out["yaw"] = round(yaw(float(spot["yaw"])), 1) % 360
    return out


def _moved_area(area: object, point: Callable[[float, float], Point], yaw: Callable[[float], float]) -> object:
    """An object or an unexplored area moved: its `at` and its `poly`, and a `rect` as the `poly` it becomes (a turn
    that isn't a quarter makes a rectangle a polygon)."""
    if not isinstance(area, dict):
        return area
    out = dict(area)
    if (at := _point(area.get("at"))) is not None:
        out["at"] = list(point(*at))
    if isinstance(area.get("poly"), list):
        out["poly"] = [list(point(*p)) for p in _points(area["poly"])]
    rect = area.get("rect")
    if isinstance(rect, list) and len(rect) == 4 and all(isinstance(v, (int, float)) for v in rect):
        x, y, w, h = (float(v) for v in rect)
        del out["rect"]
        out["poly"] = [list(point(*p)) for p in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
    if isinstance(area.get("yaw"), (int, float)):
        out["yaw"] = round(yaw(float(area["yaw"])), 1) % 360
    return out


def _along(a: Point, b: Point, p: Point) -> float | None:
    """How far along the wall from `a` to `b` the point `p` is (metres from `a`), or None when it isn't on it."""
    length = math.dist(a, b)
    t = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / length
    off = abs((p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])) / length
    return t if off <= _ON_WALL and -_ON_WALL <= t <= length + _ON_WALL else None


def build(
    *,
    walls: Sequence[tuple[Point, Point, float]],
    openings: Iterable[tuple[str, Point, float]] = (),
    rooms: Iterable[tuple[str, str, str, Sequence[Point]]] = (),
    records: Mapping[str, Any] | None = None,
    height: float = HEIGHT,
) -> Scene:
    """A scene from a plan's parts: `walls` as (start, end, thickness), `openings` as (kind, the point its middle is
    at, width) with kind `door`, `window`, or `cased`, `rooms` as (id, name, kind, polygon), and our own `records`.
    Only the fields this module knows are written; the editor fills in the rest when it loads the scene. Raises
    SceneError for an opening that is on no wall."""
    nodes: dict[str, Node] = {}

    def add(kind: str, node_id: str, parent: str | None, **fields: object) -> Node:
        node: Node = {"object": "node", "id": node_id, "type": kind, "parentId": parent, "visible": True}
        node.update(metadata={}, **fields)
        nodes[node_id] = node
        if parent is not None:
            nodes[parent]["children"].append(node_id)
        return node

    site: dict[str, object] = {}
    if walls:
        xs = [p[0] for a, b, _ in walls for p in (a, b)]
        ys = [p[1] for a, b, _ in walls for p in (a, b)]
        left, right = min(xs) - _SITE_MARGIN, max(xs) + _SITE_MARGIN
        top, bottom = min(ys) - _SITE_MARGIN, max(ys) + _SITE_MARGIN
        corners = [[left, top], [right, top], [right, bottom], [left, bottom]]
        site["polygon"] = {"type": "polygon", "points": corners}
    add("site", "site_main", None, children=[], **site)
    add("building", "building_main", "site_main", children=[])
    storey = add("level", "level_ground", "building_main", children=[], level=0, height=height)
    storey["metadata"] = {SIDE: {"version": VERSION, **(records or {})}}
    for i, (a, b, thickness) in enumerate(walls):
        add(
            "wall",
            f"wall_{i}",
            "level_ground",
            children=[],
            start=list(a),
            end=list(b),
            thickness=thickness,
            height=height,
        )
    for i, (kind, at, width) in enumerate(openings):
        on = next(((j, d) for j, (a, b, _) in enumerate(walls) if (d := _along(a, b, at)) is not None), None)
        if on is None:
            raise SceneError(f"the {kind} at {at[0]:g},{at[1]:g} is on no wall")
        wall, distance = f"wall_{on[0]}", round(on[1], 6)
        if kind == "window":
            position = [distance, WINDOW_SILL + WINDOW_HEIGHT / 2, 0]
            add("window", f"window_{i}", wall, wallId=wall, position=position, width=width, height=WINDOW_HEIGHT)
        else:
            add(
                "door",
                f"door_{i}",
                wall,
                wallId=wall,
                position=[distance, DOOR_HEIGHT / 2, 0],
                width=width,
                height=DOOR_HEIGHT,
                openingKind="opening" if kind == "cased" else "door",
            )
    for room_id, name, kind, polygon in rooms:
        zone = add("zone", f"zone_{room_id}", "level_ground", name=name, polygon=[list(p) for p in polygon])
        zone["metadata"] = {"kind": kind}
    return {"nodes": nodes, "rootNodeIds": ["site_main"]}
