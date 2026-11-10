"""Made-up plans for tests, written in a few lines and stored the way a place's plan is: as a Pascal scene.

A test wants a small home it can read at a glance and change in one word. `scene` takes that as a short notation and
returns the scene's text (spatial/plan_scene.py), with a wall for each stretch of room edge:

    plan "<title>" walls <exterior>/<partition>
    room <id> "<name>" <kind> rect x,y WxH
    room <id> "<name>" <kind> poly x,y x,y x,y ...
    door|window|cased <a>><b> at x,y w<width>
    x-camera <camera> at x,y [height <m>] [yaw <deg>] [fov <deg>]

Coordinates are plan metres, x east and y south. The rooms either side of an opening are for the reader: the opening
goes on the wall its point is on. A line of any other kind is a mistake in the test and raises.

What comes back is a `Plan`: the scene's text, a plain `str` everywhere it is stored or compared, that remembers the
notation it came from. `.replace(old, new)` and `+ "more lines"` change the notation and give the scene of the
result, so a test can still say `PLAN.replace('"Den"', '"Study"')` or `PLAN + "x-camera porch at 1,1\\n"`.
"""

from __future__ import annotations

import re
from typing import SupportsIndex

from liberational.spatial import plan_scene

_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\S+')
_OPENINGS = ("door", "window", "cased")


class Plan(str):
    """A scene's text that remembers the notation it was written in."""

    notation: str

    def replace(self, old: str, new: str, /, count: SupportsIndex = -1) -> Plan:
        return scene(self.notation.replace(old, new, count))

    def __add__(self, more: str) -> Plan:
        return scene(self.notation + more)


def _pair(word: str) -> tuple[float, float]:
    x, y = word.split(",")
    return float(x), float(y)


def scene(notation: str) -> Plan:
    """The plan a notation describes, as a scene's text."""
    title, thickness = "", (0.2, 0.1)
    rooms: list[tuple[str, str, str, tuple[tuple[float, float], ...]]] = []
    openings: list[tuple[str, tuple[float, float], float]] = []
    cameras: dict[str, dict[str, object]] = {}
    for line in notation.splitlines():
        words = _TOKEN.findall(line)
        if not words:
            continue
        head = words[0]
        if head == "plan":
            title = words[1].strip('"') if len(words) > 1 and words[1] != "walls" else ""
            if "walls" in words:
                ext, _, part = words[words.index("walls") + 1].partition("/")
                thickness = (float(ext), float(part or ext))
        elif head == "room":
            room_id, name, kind, shape = words[1], words[2][1:-1].replace('\\"', '"'), words[3], words[4]
            if shape == "rect":
                (x, y), (w, h) = _pair(words[5]), (float(v) for v in words[6].split("x"))
                outline: tuple[tuple[float, float], ...] = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
            else:
                outline = tuple(_pair(w) for w in words[5:])
            rooms.append((room_id, name, kind, outline))
        elif head in _OPENINGS:
            openings.append((head, _pair(words[words.index("at") + 1]), float(words[words.index("at") + 2][1:])))
        elif head == "x-camera":
            spot: dict[str, object] = {"at": list(_pair(words[words.index("at") + 1]))}
            for key in ("height", "yaw", "fov"):
                if key in words:
                    spot[key] = float(words[words.index(key) + 1])
            cameras[words[1]] = spot
        else:
            raise ValueError(f"not a line of a test plan: {line!r}")
    records: dict[str, object] = {"title": title} if title else {}
    if cameras:
        records["cameras"] = cameras
    built = plan_scene.build(
        walls=plan_scene.cut_walls([outline for *_, outline in rooms], *thickness),
        openings=openings,
        rooms=rooms,
        records=records,
    )
    made = Plan(plan_scene.dump(built))
    made.notation = notation
    return made
