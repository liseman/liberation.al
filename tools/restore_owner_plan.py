#!/usr/bin/env python3
"""Put a place's newest map back on the plan its owner drew (spatial/fuse.py `onto_owner_plan`, #215).

For a place where a fusion turned the map level or rescaled it while the owner's plan was drawn on it: the new
revision stood in a map frame of its own, got a machine plan of numbered rooms, and the owner's plan (still stored,
every room, name, door, and wall) was left drawn in the frame before. Fusions no longer do that; this repairs a place
it already happened to.

It writes the newest fused map again as a new revision in the frame the owner's plan is drawn in, with its cameras
and signals as they are and no machine plan. Nothing the map gained since (a walk's points) is lost, the owner's plan
isn't rewritten, and no earlier revision is touched. It refuses when the place has no owner plan drawn on a map
frame, when the plan is already on the newest map, or when the fixed cameras stand more than `fuse.BACK_M` from
where they stood on the owner's frame (the map moved by more than a levelling turn or a rescale).

A splat made for a revision in the frame left behind isn't shown on the restored one (the 3D view names it as made
for another frame); `pack-spatial-splat --place <id>` makes the restored revision's.

Dry run (the default): says what it found and what it would write.

    .venv/bin/python tools/restore_owner_plan.py --place <place id or name>
    .venv/bin/python tools/restore_owner_plan.py --place <place id or name> --write

Run it from the live checkout, never while `pack-spatial-cameras` or `pack-walkthrough` runs (the write takes the
places lock).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from liberational import config as cfg
from liberational import logs
from liberational.memory import places
from liberational.spatial import fuse


def find(state: Path, wanted: str) -> places.Place | None:
    """The place with this id, an id starting with it, or this name; None when none or several match."""
    known = places.places(state)
    found = [p for p in known if p.id == wanted or p.name == wanted] or [p for p in known if p.id.startswith(wanted)]
    return found[0] if len(found) == 1 else None


def run(state: Path, wanted: str, write: bool, now: float | None = None) -> tuple[bool, str]:
    """Restore the owner's plan onto `wanted`'s newest map, or (without `write`) say what would be done. Returns
    whether a revision was written, and what to tell the person."""
    place = find(state, wanted)
    if place is None:
        return False, f"no one place matches {wanted!r}"
    if not write:
        return False, f"{place.name} ({place.id}): {fuse.onto_owner_plan(state, place.id, now=now, write=False)[1]}"
    with places.writing(state):
        rev, found = fuse.onto_owner_plan(state, place.id, now=now)
    return rev is not None, f"{place.name} ({place.id}): {found}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--place", required=True, help="the place's id (or the start of it), or its name")
    parser.add_argument("--write", action="store_true", help="write the revision (default: a dry run)")
    args = parser.parse_args(argv)
    logs.setup()
    wrote, said = run(cfg.path("paths.state"), args.place, args.write)
    print(said)
    if args.write and not wrote:
        sys.exit(1)


if __name__ == "__main__":
    main()
