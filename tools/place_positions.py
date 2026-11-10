"""Record the fixed cameras' footprints and bring the position layer up to date by hand, or have it placed again.

Nothing does this on a schedule at present. A run works out the tilt of each camera that has a spot on its place's
plan and no tilt yet (spatial/spots.py's `fit`), records each fixed camera's raw footprints since it was last
recorded (spatial/footprints.py's `record`), and places them (spatial/positions.py's `advance`). A footprint is
placed through its camera's spot on the owner's plan, else through its pose in a place's map; a camera with neither
places nobody, and the run says what each camera lacks. `--reset` forgets the placer's records of every place and
places the last 30 days of raw footprints again (after a plan's rooms or a name changed what stored records should
say); the rig's own records are kept. A spot or a registration that moves where a camera puts people needs no
`--reset`: the placer notices and places that place again by itself.

    .venv/bin/python tools/place_positions.py --dry-run   # what a run would place per camera; changes nothing
    .venv/bin/python tools/place_positions.py             # record and place what was seen since the last run
    .venv/bin/python tools/place_positions.py --days 7    # the first recording of a camera reaches 7 days back
    .venv/bin/python tools/place_positions.py --reset     # forget the placed records and place everything again

Run it from the live checkout, at a quiet hour for `--reset`: a busy place is about 20 s of placing for each day.
"""

import argparse
import time
from collections.abc import Sequence

from liberational import config as cfg
from liberational import logs
from liberational.memory import places
from liberational.spatial import cameras, footprints, positions, spots

#: What a camera with a spot lacks, in words (spatial/spots.py's `Spot.missing`).
LACKS = {
    "height": "its spot has no height",
    "facing": "its spot has no facing (yaw)",
    "lens": "it has no held lens and its spot has no fov",
    "tilt": "too few people seen walking to work out its tilt",
    "check": "people come out the wrong height from its spot: its height, lens, or tilt is off",
}


def dry_run(hours: float) -> None:
    state = cfg.path("paths.state")
    fixed = cameras.fixed_cameras(cfg.devices_file())
    print(f"dry run over the last {hours:g} h: nothing is recorded, placed, or kept")
    for row in positions.dry_run(state, fixed, time.time(), hours):
        aim = row["aim"]
        tilt = (
            ""
            if aim is None
            else f"; tilted {aim.pitch:.1f} deg down, rolled {aim.roll:.1f}, from {aim.people} people walking"
            + ("" if aim.tall is None else f" who come out {aim.tall:.2f} m tall")
        )
        by = {"spot": "its spot on the plan", "map": "its pose in the map", None: "nothing: no spot and no pose"}
        lacks = "".join(f"; {LACKS[m]}" for m in row["missing"])
        print(
            f"  {row['camera']}: {row['footprints']:,} footprints, {row['placed']:,} placed on the floor, "
            f"{row['in_a_room']:,} in a room of the plan; read through {by[row['by']]}{tilt}{lacks}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--reset", action="store_true", help="forget the placed records and place everything again")
    parser.add_argument(
        "--dry-run", action="store_true", help="print what a run would place per camera; change nothing"
    )
    parser.add_argument("--hours", type=float, default=24.0, help="how far back a dry run reads (default 24)")
    parser.add_argument(
        "--days", type=float, default=footprints.FIRST_S / 86400, help="how far back a camera's first recording reaches"
    )
    args = parser.parse_args(argv)
    logs.setup()
    if args.dry_run:
        return dry_run(args.hours)
    state = cfg.path("paths.state")
    if args.reset:
        print(f"placed records of {positions.reset(state)} places forgotten; placing them again")
    began = time.monotonic()
    now = time.time()
    with places.writing(state):
        spots.fit(state, now)
    for camera, spot in sorted(spots.read(state).items()):
        if spot.missing:
            print(f"{camera} places nobody yet: " + "; ".join(LACKS[m] for m in spot.missing))
    recorded = footprints.record(state, cameras.fixed_cameras(cfg.devices_file()), now, args.days * 86400)
    print(f"recorded {recorded:,} raw footprints")
    if not places.camera_places(state):
        print("no camera has a spot on a plan or a pose in a map yet, so no footprint can be placed; they wait, raw")
    did = positions.advance(state, now - footprints.FLUSH_S)
    again = f", {did['replaced']} places placed again after a camera moved" if did["replaced"] else ""
    print(f"placed {did['records']:,} records in {did['pieces']} pieces{again}, {time.monotonic() - began:.0f} s")


if __name__ == "__main__":
    main()
