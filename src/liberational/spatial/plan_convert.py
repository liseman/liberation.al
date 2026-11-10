"""`pack-plan-convert`: store each place's plan as a Pascal scene, the format the plan editor works on.

  pack-plan-convert                 # print what would change and write a page to look over; change nothing
  pack-plan-convert --yes           # convert every place it can
  pack-plan-convert --place <id>    # one place (with or without --yes)
  pack-plan-convert --check <file>  # where the page goes (default: plan-convert.html under paths.state)

A place's owner plan was text in the floorplan DSL until the plan editor moved to Pascal scenes
(spatial/plan_scene.py). This converts the owner plan that holds for each place (spatial/plan_import.py `to_scene`):
rooms to named zones, each stretch of room edge to a wall, doors, doorways, and windows onto their walls, and the
camera spots, objects, unexplored areas, and fixtures into the scene's own records. A plan drawn on its own page
has that page converted too.

Nothing is rewritten. The scene is saved as one more version of the owner's plan (memory/places.py `correct_plan`,
source "convert"), in the same frame, so the DSL version stays in the plan's history and can be restored from the
Places page. Running it again changes nothing: a plan that is already a scene is left alone.

Before it writes, each conversion is checked by reading the scene back: the same rooms with the same corners, the
same number of doors, doorways, and windows, and every camera spot where it was with the same height, facing, and
field of view. A place whose conversion doesn't check out is reported and not written. What the scene has no place
for (provenance marks, the anchor, revision history) is listed per place.

The page (`--check`) draws each place's plan before and after, side by side, with those counts, to look over before
running with `--yes`. It holds a real home's plan: it stays on this machine.

A place whose only plan is the machine's is reported and left as it is: the machine's plans are not the owner's to
keep, and saving one from the plan editor makes it the owner's, as a scene.

Restart nothing afterwards: the readers take either format, and the Places page reads the plan each time it loads.
"""

from __future__ import annotations

import argparse
import html
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import plan_import, plan_scene

SOURCE = "convert"  # who wrote the converted version, in the plan's history
CHECK_FILE = "plan-convert.html"
SAME_M = 0.001  # a corner or a spot this close to where it was is where it was
_PX_PER_M = 40.0
_MARGIN_M = 1.2  # room for an opening's mark and a camera's name past the walls


@dataclass(frozen=True)
class Counts:
    rooms: int
    walls: int
    doors: int
    doorways: int
    windows: int
    cameras: int

    def words(self) -> str:
        parts = (
            (self.rooms, "room"),
            (self.walls, "wall"),
            (self.doors, "door"),
            (self.doorways, "doorway"),
            (self.windows, "window"),
            (self.cameras, "camera spot"),
        )
        return ", ".join(f"{n} {noun}{'' if n == 1 else 's'}" for n, noun in parts)


@dataclass(frozen=True)
class Outcome:
    """What converting one place's plan comes to."""

    place: places.Place
    state: str  # "convert", "scene" (already one), "machine" (no owner plan), "none" (no plan), or "refused"
    why: str = ""  # for "refused": what stopped it
    before: Counts | None = None
    after: Counts | None = None
    scene: str | None = None  # the plan as a scene, in the plan frame
    page: str | None = None  # its page as a scene, for a plan drawn on its own page
    left: tuple[str, ...] = ()  # the DSL's statements the scene doesn't carry


def _counts(plan: plan_import.Parsed) -> Counts:
    kinds = [o.kind for o in plan.openings]
    return Counts(
        len(plan.rooms),
        len(plan_import.walls(plan).a),
        kinds.count("door"),
        kinds.count("cased"),
        kinds.count("window"),
        len(plan.cameras),
    )


def differs(was: plan_import.Parsed, now: plan_import.Parsed) -> str | None:
    """Why a scene read back isn't the plan it was made from, or None when it is: the rooms and their corners, the
    openings by kind and where they are, and the camera spots."""
    if [(r.id, r.kind, len(r.polygon)) for r in was.rooms] != [(r.id, r.kind, len(r.polygon)) for r in now.rooms]:
        return "its rooms differ"
    for a, b in zip(was.rooms, now.rooms, strict=True):
        if any(math.dist(p, q) > SAME_M for p, q in zip(a.polygon, b.polygon, strict=True)):
            return f"room {a.id} has a corner that moved"
    if len(was.openings) != len(now.openings):
        return f"it has {len(now.openings)} openings where the plan has {len(was.openings)}"

    def in_order(openings: Sequence[plan_import.Opening]) -> list[plan_import.Opening]:
        # a scene lists its openings wall by wall, a DSL plan as written: the same ones, in another order
        return sorted(openings, key=lambda o: (o.kind, round(o.at[0], 3), round(o.at[1], 3)))

    for o, n in zip(in_order(was.openings), in_order(now.openings), strict=True):
        if o.kind != n.kind or math.dist(o.at, n.at) > SAME_M or abs(o.width - n.width) > SAME_M:
            return f"the {o.kind} at {o.at[0]:g},{o.at[1]:g} differs"
    if {c.id: c for c in was.cameras} != {c.id: c for c in now.cameras}:
        return "its camera spots differ"
    return None


def convert(held: places.Place) -> Outcome:
    """What converting one place's plan comes to, writing nothing."""
    if not held.plan:
        return Outcome(held, "none")
    if held.plan_source != "owner":
        return Outcome(held, "scene" if plan_scene.is_scene(held.plan) else "machine")
    if plan_scene.is_scene(held.plan) and (held.view is None or plan_scene.is_scene(held.view.plan)):
        return Outcome(held, "scene")
    try:
        was = plan_import.parse(held.plan)
        made = held.plan if plan_scene.is_scene(held.plan) else None
        left: tuple[str, ...] = ()
        if made is None:
            converted = plan_import.to_scene(held.plan)
            made, left = converted.scene, converted.left
        now = plan_import.parse(made)
        page = None
        if held.view is not None:
            page = held.view.plan if plan_scene.is_scene(held.view.plan) else plan_import.to_scene(held.view.plan).scene
    except ValueError as e:  # PlanError is one
        return Outcome(held, "refused", str(e))
    if (why := differs(was, now)) is not None:
        return Outcome(held, "refused", f"the scene doesn't read back as the plan: {why}", _counts(was), _counts(now))
    return Outcome(held, "convert", "", _counts(was), _counts(now), made, page, left)


def plan(state: Path, only: str | None = None) -> list[Outcome]:
    """Every listed place's outcome (or `only`'s), writing nothing."""
    return [convert(held) for held in places.places(state) if only is None or held.id == only]


def describe(outcomes: Sequence[Outcome]) -> list[str]:
    lines: list[str] = []
    for o in outcomes:
        head = f"{o.place.name} ({o.place.id})"
        if o.state == "convert" and o.before is not None and o.after is not None:
            lines += [f"{head}: to convert", f"  before: {o.before.words()}", f"  after:  {o.after.words()}"]
            lines += [f"  not carried: {line}" for line in o.left]
            if o.page is not None:
                lines.append("  drawn on its own page: the page is converted too")
        elif o.state == "refused":
            lines.append(f"{head}: NOT converted: {o.why}")
        elif o.state == "scene":
            lines.append(f"{head}: already a scene")
        elif o.state == "machine":
            lines.append(f"{head}: only a machine plan, left as it is (save it from the plan editor to keep it)")
        else:
            lines.append(f"{head}: no plan")
    return lines or ["No place is listed."]


def run(state: Path, outcomes: Sequence[Outcome], now: float | None = None) -> list[str]:
    """Save each converted plan as a new version of its place's owner plan. Returns what was done."""
    done = []
    for o in outcomes:
        if o.state != "convert" or o.scene is None:
            continue
        view = None
        if o.place.view is not None and o.page is not None:
            view = {**o.place.view.as_dict(), "plan": o.page}
        places.correct_plan(state, o.place.id, o.scene, now=now, base=o.place.plan_revision, source=SOURCE, view=view)
        done.append(f"{o.place.name}: saved as a scene (the earlier version stays in its history)")
    return done or ["Nothing to convert."]


# The page to look over: each plan drawn before and after.


def _drawing(
    rooms: Sequence[tuple[str, Sequence[plan_import.Point]]],
    walls: Sequence[tuple[plan_import.Point, plan_import.Point, float]],
    openings: Sequence[tuple[str, plan_import.Point, float]],
    spots: Sequence[plan_import.Camera],
) -> str:
    """One plan as an SVG: room outlines with their names, walls at their thickness, a mark at each opening (a
    square for a door or a doorway, a circle for a window), and each camera spot with its facing."""
    points = [p for _, poly in rooms for p in poly] + [p for a, b, _ in walls for p in (a, b)] + [c.at for c in spots]
    if not points:
        return "<p>Nothing to draw.</p>"
    left, top = min(p[0] for p in points) - _MARGIN_M, min(p[1] for p in points) - _MARGIN_M
    width = (max(p[0] for p in points) + _MARGIN_M - left) * _PX_PER_M
    height = (max(p[1] for p in points) + _MARGIN_M - top) * _PX_PER_M

    def xy(p: plan_import.Point) -> tuple[float, float]:
        return (p[0] - left) * _PX_PER_M, (p[1] - top) * _PX_PER_M

    parts = []
    for name, poly in rooms:
        ring = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, poly))
        cx, cy = xy((sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)))
        parts.append(f'<polygon points="{ring}" fill="#eef1f6" stroke="#9aa3b2" stroke-width="1"/>')
        parts.append(f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" font-size="12">{html.escape(name)}</text>')
    for a, b, thickness in walls:
        (x1, y1), (x2, y2) = xy(a), xy(b)
        stroke = max(1.5, thickness * _PX_PER_M)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#1d2433" stroke-width="{stroke:.1f}"/>'
        )
    for kind, at, width_m in openings:
        x, y = xy(at)
        r = max(3.0, width_m * _PX_PER_M / 2)
        if kind == "window":
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#fff" stroke="#2f6fed" stroke-width="2"/>'
            )
        else:
            dash = ' stroke-dasharray="4 3"' if kind == "cased" else ""
            parts.append(
                f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" fill="#fff" '
                f'stroke="#c2410c" stroke-width="2"{dash}/>'
            )
    for c in spots:
        x, y = xy(c.at)
        if c.yaw is not None:
            dx, dy = math.sin(math.radians(c.yaw)) * 30, -math.cos(math.radians(c.yaw)) * 30
            parts.append(
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + dx:.1f}" y2="{y + dy:.1f}" stroke="#7c3aed" stroke-width="2"/>'
            )
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#7c3aed"/>')
        parts.append(f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="11" fill="#7c3aed">{html.escape(c.id)}</text>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.1f} {height:.1f}">{"".join(parts)}</svg>'
    )


def _before(text: str) -> str:
    """A plan drawn the way its DSL reads: rooms, the walls cut from their edges, and the openings' points."""
    parsed = plan_import.parse(text)
    cut = plan_import.walls(parsed)
    walls = [
        ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])), 2 * float(half))
        for a, b, half in zip(cut.a, cut.b, cut.half, strict=True)
    ]
    openings = [(o.kind, o.at, o.width) for o in parsed.openings]
    return _drawing([(r.name or r.id, r.polygon) for r in parsed.rooms], walls, openings, parsed.cameras)


def _after(text: str) -> str:
    """A scene drawn from its own nodes: its zones, its walls with their own thickness, and its openings where
    they hang on their walls."""
    scene = plan_scene.load(text)
    rooms = [(name or room_id, poly) for room_id, name, _, poly in plan_scene.rooms(scene)]
    walls = [(a, b, thickness) for _, a, b, thickness in plan_scene.walls(scene)]
    openings = [(kind, at, width) for kind, _, at, width in plan_scene.openings(scene)]
    spots = [plan_import.Camera(*spot) for spot in plan_scene.spots(scene)]
    return _drawing(rooms, walls, openings, spots)


def check_page(outcomes: Sequence[Outcome]) -> str:
    """The page to look over: each place to convert, its plan before and after side by side with what each holds."""
    sections = []
    for o in outcomes:
        name = html.escape(o.place.name)
        if o.state == "refused":
            sections.append(f"<section><h2>{name}</h2><p class=bad>Not converted: {html.escape(o.why)}</p></section>")
        if o.state != "convert" or o.scene is None or o.before is None or o.after is None:
            continue
        shown_before = o.place.view.plan if o.place.view is not None else (o.place.plan or "")
        shown_after = o.page if o.page is not None else o.scene
        left = "".join(f"<li><code>{html.escape(line)}</code></li>" for line in o.left)
        sections.append(
            f"<section><h2>{name}</h2><div class=pair>"
            f"<figure><figcaption>Before (DSL): {o.before.words()}</figcaption>{_before(shown_before)}</figure>"
            f"<figure><figcaption>After (scene): {o.after.words()}</figcaption>{_after(shown_after)}</figure>"
            f"</div>{f'<p>Not carried into the scene:</p><ul>{left}</ul>' if left else ''}</section>"
        )
    body = "".join(sections) or "<p>No plan to convert.</p>"
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Plan conversion check</title>'
        "<style>body{font:15px system-ui,sans-serif;margin:2em;color:#1d2433}"
        ".pair{display:flex;gap:2em;flex-wrap:wrap;align-items:flex-start}figure{margin:0}"
        "figcaption{margin-bottom:.5em;font-weight:600}svg{border:1px solid #d5d9e2;max-width:100%;height:auto}"
        ".bad{color:#b42318}code{font-size:13px}</style>"
        "<h1>Plan conversion check</h1><p>Each plan as it is now and as the scene it becomes: rooms, walls, a square "
        "at each door (dashed for a doorway with no door), a circle at each window, and each camera spot with the "
        "way it faces. The two drawings of a place should match.</p>" + body + "</html>"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pack-plan-convert", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yes", action="store_true", help="convert; without this, only print what would happen")
    parser.add_argument("--place", help="one place's id; every place without it")
    parser.add_argument(
        "--check", type=Path, help=f"where the page to look over goes (default: {CHECK_FILE} in paths.state)"
    )
    args = parser.parse_args(argv)
    state = cfg.path("paths.state")
    outcomes = plan(state, args.place)
    print("\n".join(describe(outcomes)))
    page = args.check or state / CHECK_FILE
    page.write_text(check_page(outcomes))
    print(f"\nLook the plans over, before and after: {page}")
    if not args.yes:
        print("Nothing was changed. Add --yes to convert.")
        return
    print("\n".join(run(state, outcomes)))


if __name__ == "__main__":
    main()
