"""`pack-reset-places`: start spatial memory again, keeping what the owner taught.

  pack-reset-places          # print what would go and what would stay, change nothing
  pack-reset-places --yes    # do it

Everything the machine made about places goes: each place's revisions, maps, plans, and splats (`state/places/`),
with the cameras' registrations they hold and the tilts fitted for cameras with a spot on a plan (`aims.json`), the files of walks imported from the Places page (`state/walk_imports/`),
and the position layer's stored records (the `position.sample`, `position.placed`, `position.path`, and `camera.pose`
streams of the segment store, with their condensed history). The job that builds maps from what the cameras see
writes all of it again.

What the owner taught stays, since none of it is in those files: place names and plan corrections, with their
history, and each camera's held focal and lens are in the knowledge store, which this never opens for writing. A
place whose owner plan holds is kept on the Places page: it gets one fresh revision with no map, no plan file, and no
camera, under the same id, so the owner's plan is still that place's plan and can still be edited. It has no 3D view
and nobody is placed on it until a map is made for it. A place with only a name, or with nothing of the owner's, is
no longer listed; its name stays in the store with the signals that identify the place.

Nothing writes these files on a schedule, so no service is stopped. Restart the dashboard and the journal afterwards:
they keep what they read.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from liberational import config as cfg
from liberational import json_state
from liberational import segment_log as S
from liberational.memory import places
from liberational.spatial import floorplan, positions

PRODUCER = "places reset v1"
WALK_IMPORTS = "walk_imports"  # under paths.state: what the Places page's walk import kept
#: The position layer's streams: raw footprints, the placer's records and paths, and the cameras' pose series.
STREAMS = (S.POSITIONS, S.PLACED, S.PLACED_PATHS, S.CAMERA_POSES)
#: Signals a kept place carries over: the ones that say where it is. What named its old maps' sources doesn't.
DROPPED_SIGNALS = ("walkthrough", "camera")


@dataclass(frozen=True)
class Kept:
    """A place that stays listed, with the owner's plan that keeps it."""

    place: places.Place
    rooms: int


@dataclass(frozen=True)
class Plan:
    state: Path
    found: tuple[places.Place, ...]  # every place listed now
    kept: tuple[Kept, ...]  # those whose owner plan holds
    targets: tuple[Path, ...]  # what is deleted


def stream_dirs(state: Path) -> list[Path]:
    """The segment store's folders for the position layer's streams and their condensed history, where they exist."""
    root = S.root(state)
    names = [*STREAMS, *(S.condensed(s) for s in STREAMS if s in S.CONDENSED)]
    return [root / n for n in names if (root / n).exists()]


def plan(state: Path) -> Plan:
    """What a reset of `state` would do."""
    found = tuple(places.places(state))
    kept = tuple(
        Kept(p, len(floorplan.rooms_from_dsl(p.plan or ""))) for p in found if p.plan_source == "owner" and p.plan
    )
    base = places.root(state)
    inside = sorted(base.iterdir()) if base.is_dir() else []
    walks = state / WALK_IMPORTS
    targets = [*inside, *([walks] if walks.exists() else []), *stream_dirs(state)]
    return Plan(state, found, kept, tuple(targets))


def size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    return sum(p.lstat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def human(n: int) -> str:
    step = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if step < 1024 or unit == "GiB":
            return f"{step:.0f} {unit}" if unit == "B" else f"{step:.1f} {unit}"
        step /= 1024
    return f"{n} B"


def describe(found: Plan) -> list[str]:
    """The plan, as the dry run prints it."""
    state = found.state
    lines = [f"State: {state}", "Delete:"]
    keeping = {k.place.id for k in found.kept}
    for p in found.found:
        chain = places.revisions(state, p.id)
        folder = places.root(state) / p.id
        maps, splats = len(list(folder.glob("*.map"))), len(list(folder.glob("*.sog")))
        lines.append(
            f"  {p.name} ({p.id}): {len(chain)} revisions, {maps} maps, {splats} splats, "
            f"{len(p.revision.cameras)} camera registrations, {human(size(folder))}"
        )
    listed = {places.root(state) / p.id for p in found.found}
    for path in found.targets:
        if path in listed:
            continue  # a place's folder, named above
        lines.append(f"  {path} ({human(size(path))})")
    lines.append(f"  total {human(sum(size(p) for p in found.targets))}")
    lines.append("Keep:")
    for k in found.kept:
        lines.append(f"  {k.place.name} ({k.place.id}): your corrected plan, {k.rooms} rooms; listed with no map")
    gone = [p for p in found.found if p.id not in keeping]
    if gone:
        names = ", ".join(p.name for p in gone)
        lines.append(f"  not listed until mapped again (no corrected plan of yours): {names}")
    lines.append("  the knowledge store: place names, every plan correction and its history, held camera intrinsics")
    return lines


def _generations(state: Path) -> dict[str, int]:
    """Each place's placer slot generation (spatial/positions.py `place_again`), from the placer's marks."""
    try:
        marks = json.loads((places.root(state) / positions.LAYER_MARKS).read_text())
    except FileNotFoundError, ValueError:
        return {}
    held = marks.get("generations") if isinstance(marks, dict) else None
    return {str(k): int(v) for k, v in held.items()} if isinstance(held, dict) else {}


def run(found: Plan) -> list[str]:
    """Do the reset under the places lock; what it did, as lines."""
    state = found.state
    lines = []
    with places.writing(state):
        generations = {k.place.id: g for k in found.kept if (g := _generations(state).get(k.place.id)) is not None}
        signals = {
            k.place.id: {s: v for s, v in k.place.revision.signals.items() if s not in DROPPED_SIGNALS}
            for k in found.kept
        }
        for path in found.targets:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        lines.append(f"deleted {len(found.targets)} files and folders")
        for k in found.kept:
            rev = places.revise(state, k.place.id, producer=PRODUCER, signals=signals[k.place.id])
            lines.append(f"kept {k.place.name} ({k.place.id}) with your plan, as revision {rev.id}")
        if generations:
            # A kept place's placer slot stays the generation it was: rows forgotten from earlier ones stay hidden.
            json_state.write(
                places.root(state) / positions.LAYER_MARKS, {"generations": generations}, indent=1, sort_keys=True
            )
    lines.append("Restart the dashboard and the journal so they read the places again.")
    return lines


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pack-reset-places", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yes", action="store_true", help="do it; without this, only print what would happen")
    args = parser.parse_args(argv)
    found = plan(cfg.path("paths.state"))
    print("\n".join(describe(found)))
    if not args.yes:
        print("\nNothing was changed. Add --yes to do it.")
        return
    print("\n".join(run(found)))


if __name__ == "__main__":
    main()
