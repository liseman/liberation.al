"""Record the fixed cameras' footprints and bring the position layer up to date by hand, or have it placed again.

Nothing does this on a schedule at present. This records each fixed camera's raw footprints since it was last
recorded (spatial/footprints.py's `record`) and places them (spatial/positions.py's `advance`). A footprint is placed
through its camera's pose in a place's map, so while no place registers a camera it says so and places none: the raw
footprints wait for a map. `--reset` forgets the placer's records of every place and places the last 30 days of raw
footprints again (after a plan's rooms or a name changed what stored records should say); the rig's own records are
kept. A registration that moves where a camera puts people needs no `--reset`: the placer notices and places that
place again by itself.

    .venv/bin/python tools/place_positions.py            # record and place what was seen since the last run
    .venv/bin/python tools/place_positions.py --reset    # forget the placed records and place everything again

Run it from the live checkout, at a quiet hour for `--reset`: a busy place is about 20 s of placing for each day.
"""

import argparse
import time
from collections.abc import Sequence

from liberational import config as cfg
from liberational import logs
from liberational.memory import places
from liberational.spatial import cameras, footprints, positions


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reset", action="store_true", help="forget the placed records and place everything again")
    args = parser.parse_args(argv)
    logs.setup()
    state = cfg.path("paths.state")
    if args.reset:
        print(f"placed records of {positions.reset(state)} places forgotten; placing them again")
    began = time.monotonic()
    now = time.time()
    recorded = footprints.record(state, cameras.fixed_cameras(cfg.devices_file()), now)
    print(f"recorded {recorded:,} raw footprints")
    if not places.camera_places(state):
        print("no camera has a pose in any place's map yet, so no footprint can be placed; they wait, raw, for a map")
    did = positions.advance(state, now - footprints.FLUSH_S)
    again = f", {did['replaced']} places placed again after a registration" if did["replaced"] else ""
    print(f"placed {did['records']:,} records in {did['pieces']} pieces{again}, {time.monotonic() - began:.0f} s")


if __name__ == "__main__":
    main()
