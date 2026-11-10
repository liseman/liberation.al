"""What a Places read of a busy place costs: placing the raw footprints on every read (`positions.derive`, how the
page read before the position layer stored placed records) against reading the layer (`positions.positions`).

It makes up a place in a temporary state folder (two cameras over one room, a few people walking at random under each
all day, a new track every ten minutes), records a week of raw footprints, runs the placer over them, and times the
page's read of a day and of the whole week both ways, with the page's own thinning. Nothing of a real install is read
or written. The times are for comparing with each other on one machine, loaded or not; the counts don't depend on it.

    tools/test-slot.sh .venv/bin/python tools/positions_week_read.py            # a week, 2 people a camera, 8 h a day
    tools/test-slot.sh .venv/bin/python tools/positions_week_read.py --days 7 --people 3 --hours 12 --derive-week

`--derive-week` also times the old read over the whole week (minutes, and a few GB, at the default size).
"""

import argparse
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from liberational.memory import places
from liberational.spatial import positions

DAY_S = 86400.0
T0 = 1_800_000_000.0 - 1_800_000_000.0 % DAY_S  # a midnight, UTC
TRACK_S = 600  # a made-up track's length
MAX_POINTS, FINE_SHARE = 2000, 20  # the page's thinning (dashboard/places.py)
FLOOR = (0.0, -1.0, 0.0, 0.0)
TO_PLAN = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0)
PLAN = 'plan "Home" walls 0.2/0.1\nroom r1 "Living" other rect 0,0 8x8\n'


def camera(x: float) -> places.CameraPose:
    """A level camera 2 m up at plan x, looking along +y."""
    return places.CameraPose((1.0, 0, 0, x, 0, 1.0, 0, -2.0, 0, 0, 1.0, 0.0, 0, 0, 0, 1.0), 144.0, 512, 288, T0)


def record_week(state: Path, days: int, people: int, hours: float, rng: np.random.Generator) -> int:
    """A week of raw footprints: under each camera, `people` at a time for `hours` of each day, one row a second
    each, wandering the lower half of the frame. Returns how many rows."""
    total = 0
    for day in range(days):
        for cam in ("cam0", "cam1"):
            rows = []
            for person in range(people):
                seconds = int(hours * 3600)
                u = np.clip(0.5 + np.cumsum(rng.normal(0, 0.004, seconds)), 0.05, 0.95)
                v = np.clip(0.8 + np.cumsum(rng.normal(0, 0.002, seconds)), 0.6, 1.0)
                start = T0 + day * DAY_S + 8 * 3600
                for i in range(seconds):
                    track = f"{cam}:{day * 100000 + person * 1000 + i // TRACK_S}"
                    rows.append(
                        dict(
                            t=start + i,
                            source=positions.FIXED_CAMERA,
                            confidence=0.9,
                            track=track,
                            camera=cam,
                            u=float(u[i]),
                            v=float(v[i]),
                        )
                    )
            positions.write_samples(state, rows)
            total += len(rows)
    return total


def timed[T](what: Callable[[], T]) -> tuple[T, float]:
    began = time.perf_counter()
    found = what()
    return found, time.perf_counter() - began


def every(span: float) -> float | None:
    """The thinning the page asks of the reader for a window `span` long."""
    fine = span / MAX_POINTS / FINE_SHARE
    return fine if fine >= 1 else None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--people", type=int, default=2, help="people at a time under each camera")
    parser.add_argument("--hours", type=float, default=8.0, help="hours of each day they are there")
    parser.add_argument("--derive-week", action="store_true", help="time the old read over the whole week too")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="positions-week-") as tmp:
        state = Path(tmp)
        rev = places.revise(
            state,
            None,
            producer="made-up",
            cameras={"cam0": camera(2.0), "cam1": camera(6.0)},
            floor=FLOOR,
            map_bytes=b"x",
            map_format="ply",
            plan=PLAN,
            plan_transform=TO_PLAN,
            now=T0 - 100,
        )
        raw = record_week(state, args.days, args.people, args.hours, np.random.default_rng(7))
        end = T0 + args.days * DAY_S
        print(f"made up: {raw:,} raw footprints over {args.days} days, 2 cameras, {args.people} people a camera")
        did, took = timed(lambda: positions.advance(state, end))
        print(f"placer (once, as footprints are recorded): {did['records']:,} records written in {took:.1f} s")

        def page(read: Callable[..., list[positions.Position]], span: float) -> tuple[list[positions.Position], float]:
            return timed(
                lambda: read(state, end - span, end, place=rev.place, every=every(span), frame_revision=rev.id)
            )

        def scanned(span: float) -> int:
            """How many stored records the layer's read of `span` decodes (the same scan again, outside the timing)."""
            thinned = (every(span) or 0.0) >= positions.COARSE_S
            scans = positions._scan(state, end - span, end, positions._place_slots(state, rev.place), paths=thinned)
            return sum(len(got["ts_ns"]) for got in scans)

        for name, span in (("day", DAY_S), ("week", args.days * DAY_S)):
            share = span / (args.days * DAY_S)
            if name == "day" or args.derive_week:
                rows, took = page(positions.derive, span)
                print(
                    f"{name}, placed on read (before): {took:7.2f} s, {round(raw * share):,} raw footprints placed,"
                    f" {len(rows):,} rows returned"
                )
            rows, took = page(positions.positions, span)
            print(
                f"{name}, read from the layer (after): {took:7.2f} s, 0 raw footprints placed,"
                f" {scanned(span):,} stored records scanned, {len(rows):,} rows returned"
            )


if __name__ == "__main__":
    main()
