"""Bring the position layer up to date by hand, or have it placed again from scratch.

The nightly `pack-spatial-cameras` run places each day's footprints (spatial/positions.py's `advance`). This does
the same now: after an upgrade, so the Places page has paths before the next night, or with `--reset` after
something changed what stored records should say (a pose series, a plan's rooms, or a name given to a track nobody
could name when it was placed). A registration that moves where a camera puts people needs no `--reset`: the placer
notices and places that place again by itself. `--reset` forgets the placer's records of every place and places the
last 30 days of raw footprints again; the rig's own records are kept.

    .venv/bin/python tools/place_positions.py            # place what was recorded since the last run
    .venv/bin/python tools/place_positions.py --reset    # forget the placed records and place everything again

Run it from the live checkout, at a quiet hour for `--reset`: a busy place is about 20 s of placing for each day.
"""

import argparse
import time
from collections.abc import Sequence

from liberational import config as cfg
from liberational import logs
from liberational.spatial import footprints, positions


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reset", action="store_true", help="forget the placed records and place everything again")
    args = parser.parse_args(argv)
    logs.setup()
    state = cfg.path("paths.state")
    if args.reset:
        print(f"placed records of {positions.reset(state)} places forgotten; placing them again")
    began = time.monotonic()
    did = positions.advance(state, time.time() - footprints.FLUSH_S)
    again = f", {did['replaced']} places placed again after a registration" if did["replaced"] else ""
    print(f"placed {did['records']:,} records in {did['pieces']} pieces{again}, {time.monotonic() - began:.0f} s")


if __name__ == "__main__":
    main()
