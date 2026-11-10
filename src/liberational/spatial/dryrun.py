"""Dry runs of spatial writes against live state: a throwaway copy of the state to write into, and the
measures the tools print before and after.

`overlay` makes the copy: each entry of the live state linked in, except what a write would change. The place's
folder is a real folder of links (a revision written there stays there, and pruning one unlinks only the link), the
places root's own small state files (seats.json, footprints.json) are copies (a nightly seat's refusal count is
written there), the segment streams a dry run writes (OWN_STREAMS: the cameras' pose observations, spatial/poses.py)
are copies while every other stream is linked, and the knowledge store is a backup copy (a plan or a held lens saved there stays there). Lock files
aren't linked, so nothing in a dry run takes a live lock. A run reads the recordings and the position samples through
the links, as any reader does.

The measures: today's position rows by room on the plan that holds (`rooms`), and how far apart one person lands
when two cameras see them at once (`agreement`): with no hand-checked foot points, two registered cameras' views of
the same moment are the check that needs no labels.
"""

from __future__ import annotations

import collections
import math
import shutil
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from liberational import segment_log as S
from liberational.memory import places
from liberational.spatial import positions

NO_POINT = "(room only: a zone or no point)"
OUT_OF_FRAME = "(not in the plan's frame)"
OUTSIDE = "(outside every room)"
OWN_STREAMS = (S.CAMERA_POSES,)  # segment streams a dry run writes: copied into the overlay, never linked
TOGETHER_S = 0.5  # seconds: two cameras' rows of one entity this close in time are the same moment


def overlay(state: Path, tmp: Path, place: str) -> Path:
    """A throwaway state under `tmp` for a dry run of writes to `place` (see the module docstring)."""
    out = tmp / "state"
    out.mkdir(parents=True)
    for entry in state.iterdir():
        if (
            entry.name in ("places", "segments")
            or entry.name.startswith("knowledge.sqlite3")
            or entry.suffix == ".lock"
        ):
            continue
        (out / entry.name).symlink_to(entry)
    # The segment store by stream: each linked, but the streams a dry run writes (OWN_STREAMS) copied
    (out / "segments").mkdir()
    for entry in (state / "segments").iterdir() if (state / "segments").is_dir() else ():
        if entry.name in OWN_STREAMS:
            shutil.copytree(entry, out / "segments" / entry.name, symlinks=False)
        else:
            (out / "segments" / entry.name).symlink_to(entry)
    (out / "places").mkdir()
    for entry in (state / "places").iterdir():
        if entry.name == place:
            (out / "places" / place).mkdir()
            for f in entry.iterdir():
                (out / "places" / place / f.name).symlink_to(f)
        elif entry.is_file() and entry.suffix == ".json":
            shutil.copy2(entry, out / "places" / entry.name)
        elif entry.suffix != ".lock":
            (out / "places" / entry.name).symlink_to(entry)
    src = sqlite3.connect(f"file:{state / 'knowledge.sqlite3'}?mode=ro", uri=True)
    dst = sqlite3.connect(out / "knowledge.sqlite3")
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    return out


def rows(state: Path, place: str, start: float, end: float, frame: str | None = None) -> list[positions.Position]:
    """`place`'s position rows in [start, end) on the plan of revision `frame` (default: the plan that holds)."""
    if frame is None:
        held = places.get(state, place)
        frame = held.plan_revision if held is not None else None
    return positions.positions(state, start, end, place=place, frame_revision=frame)


def rooms(found: Sequence[positions.Position]) -> collections.Counter[str]:
    """Rows by room, with rows that have no point, aren't in the frame, or are outside every room counted apart."""
    out: collections.Counter[str] = collections.Counter()
    for r in found:
        if r.x is None or r.y is None:
            out[NO_POINT] += 1
        elif not r.in_frame:
            out[OUT_OF_FRAME] += 1
        else:
            out[r.room or OUTSIDE] += 1
    return out


@dataclass(frozen=True)
class Agreement:
    pairs: int  # moments both cameras placed one entity
    median: float  # metres apart
    p90: float


def agreement(found: Sequence[positions.Position], a: str, b: str, within: float = TOGETHER_S) -> Agreement:
    """How far apart cameras `a` and `b` put one entity at the same moment (rows within `within` seconds), over the
    entities either camera named."""
    placed = [r for r in found if r.entity and r.x is not None and r.y is not None and r.in_frame]
    by: dict[tuple[str, str], list[positions.Position]] = collections.defaultdict(list)
    for r in placed:
        if r.camera in (a, b) and r.entity is not None:
            by[(r.camera, r.entity)].append(r)
    gaps = []
    for (cam, entity), mine in by.items():
        if cam != a:
            continue
        theirs = sorted(by.get((b, entity), []), key=lambda r: r.t)
        if not theirs:
            continue
        times = np.array([r.t for r in theirs])
        for r in mine:
            k = int(np.searchsorted(times, r.t))
            near = [j for j in (k - 1, k) if 0 <= j < len(theirs) and abs(theirs[j].t - r.t) <= within]
            if near:
                o = min((theirs[j] for j in near), key=lambda q: abs(q.t - r.t))
                assert r.x is not None and r.y is not None and o.x is not None and o.y is not None
                gaps.append(math.hypot(r.x - o.x, r.y - o.y))
    if not gaps:
        return Agreement(0, math.nan, math.nan)
    return Agreement(len(gaps), float(np.median(gaps)), float(np.percentile(gaps, 90)))
