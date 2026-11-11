"""Sizing a camera's point cloud from a measurement the owner made (#445).

A camera's cloud is in the model's own metres, which are approximate, and not by one factor: measured against an
owner's plan, the model's near and far depth don't share a scale (sizing by the near floor put far walls up to a
quarter too far). So the size comes from the owner: something whose real length they know, pointed at in the
camera's picture.

**A measurement** (`Measure`) is one of two things. *Between two points*: two places in the picture and the real
distance between them (the long edge of a sheet of paper taped to the wall, a door's width). *From the camera*:
one place in the picture and its real distance from the lens (a tape measure to a spot on the wall). The places are
kept as fractions of the picture, so they mean the same thing in the camera's next picture, as long as the camera
hasn't moved.

**It is the owner's.** Measurements live in the knowledge store (`camera.measure`, newest record per camera, each
record the camera's whole list, so a removal is a new record and the old stays in its history). Nothing here writes
one but `add` and `remove`, which the dashboard calls on the owner's action. A camera that has moved (its picture no
longer matches the one it was placed with, spatial/cloud_scene.py) has its measurements set aside, not deleted:
the places no longer point at the same things. They apply again if the camera is found back where it was, and the
owner can remove them.

**Where a place in the picture is in the cloud** (`snap`). The cloud has a point for most pixels, not all: holes
where the model was unsure, and edges that were cleaned away. A place that falls on one is moved to the nearest
pixel that stands on a solid patch (`SOLID` of its neighbours have a point), within `SNAP_PX`, and says how far it
was moved. With nothing solid that near, the measurement is refused.

**The fit** (`fit`, `Fit`). One measurement sets one number: every distance from the camera is multiplied by
`scale`, so the measured thing comes out right, and things much nearer or farther may still be off. Two measurements
at different distances from the camera (`SPREAD` apart) also fit how depth stretches: a distance r becomes
`scale * r ** power`, the simplest curve through both, with `power` held to `POWER`. More than two are fitted
together, least squares on the logarithm. `resize` applies a fit to a cloud, about its camera.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

COLLECTION = "camera.measure"  # in the knowledge store, keyed by camera
KINDS = ("between", "camera")
SNAP_PX = 16  # pixels (of the picture as the model saw it) a place may be moved to reach a solid point
SOLID = 0.6  # of a pixel's 5 x 5 neighbours must have a point for it to count as solid
SPREAD = 1.3  # two measurements this many times apart in distance from the camera tell a stretch from a scale
POWER = (0.6, 1.4)  # how far a stretch may bend depth
LIMITS_M = (0.02, 60.0)  # a real length outside this isn't a measurement of a room


@dataclass(frozen=True)
class Measure:
    """One thing the owner measured in a camera's picture."""

    kind: str  # "between": two places and the distance between them; "camera": one place and its distance from the lens
    points: tuple[tuple[float, float], ...]  # the places, as fractions of the picture: (across, down)
    metres: float
    at: float  # when it was given (UTC epoch seconds)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "points": [list(p) for p in self.points], "metres": self.metres, "at": self.at}

    @staticmethod
    def from_dict(value: object) -> Measure | None:
        if not isinstance(value, dict) or value.get("kind") not in KINDS:
            return None
        try:
            points = tuple((float(u), float(v)) for u, v in value["points"])
            metres, at = float(value["metres"]), float(value.get("at", 0.0))
        except KeyError, TypeError, ValueError:
            return None
        if len(points) != (2 if value["kind"] == "between" else 1) or not math.isfinite(metres) or metres <= 0:
            return None
        return Measure(value["kind"], points, metres, at)


def _store(state: Path) -> Any:
    from liberational import knowledge
    from liberational.people import registry

    return knowledge.at(registry.knowledge_path(state))


def read(state: Path, camera: str) -> list[Measure]:
    """`camera`'s measurements as they hold now (its newest record's); none before any. Reading doesn't create a
    knowledge store."""
    from liberational.people import registry

    if not registry.knowledge_path(state).exists():
        return []
    rows = [row.record for row in _store(state).query(COLLECTION, key=camera)]
    if not rows:
        return []
    newest = max(rows, key=lambda r: r.start_ns)
    listed = newest.value.get("measures")
    return [m for m in (Measure.from_dict(v) for v in (listed if isinstance(listed, list) else [])) if m is not None]


def _put(state: Path, camera: str, measures: Sequence[Measure], now: float | None) -> None:
    from liberational import knowledge

    store = _store(state)
    at_ns = round((time.time() if now is None else now) * 1e9)
    was = [row.record.start_ns for row in store.query(COLLECTION, key=camera)]
    if was and at_ns <= max(was):
        at_ns = max(was) + 1  # the newest record stays the one written last
    value = {"camera": camera, "measures": [m.as_dict() for m in measures]}
    store.put([knowledge.Record(str(uuid.uuid7()), COLLECTION, camera, at_ns, at_ns, time.time_ns(), value)])


def add(state: Path, camera: str, measure: Measure, now: float | None = None) -> list[Measure]:
    """Keep `measure` with `camera`'s others, as the owner's; the list as it now holds."""
    if measure.kind not in KINDS or not LIMITS_M[0] <= measure.metres <= LIMITS_M[1]:
        raise ValueError(f"a measurement is a length between {LIMITS_M[0]} and {LIMITS_M[1]:.0f} m")
    held = [*read(state, camera), measure]
    _put(state, camera, held, now)
    return held


def remove(state: Path, camera: str, index: int | None = None, now: float | None = None) -> list[Measure]:
    """Take `camera`'s measurement at `index` away (all of them with None); the list as it now holds. The record
    before stays in the store's history."""
    held = read(state, camera)
    if index is None:
        held = []
    elif 0 <= index < len(held):
        held = [m for i, m in enumerate(held) if i != index]
    else:
        raise IndexError(index)
    _put(state, camera, held, now)
    return held


class Unsolid(ValueError):
    """A place in the picture with no solid point of the cloud near it."""


def snap(pixels: np.ndarray, u: float, v: float) -> tuple[int, int, float]:
    """The pixel of `pixels` (H x W x 3, NaN where there is no point) nearest the place (`u` across, `v` down, as
    fractions of the picture) that stands on a solid patch, as (row, column, pixels moved). Raises Unsolid when none
    is within SNAP_PX."""
    from scipy.ndimage import uniform_filter

    h, w = pixels.shape[:2]
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        raise Unsolid("that place is outside the picture")
    row, col = min(h - 1, int(v * h)), min(w - 1, int(u * w))
    have = np.isfinite(pixels).all(axis=2)
    solid = have & (uniform_filter(have.astype(np.float32), size=5, mode="constant") >= SOLID)
    r0, r1 = max(0, row - SNAP_PX), min(h, row + SNAP_PX + 1)
    c0, c1 = max(0, col - SNAP_PX), min(w, col + SNAP_PX + 1)
    rows, cols = np.nonzero(solid[r0:r1, c0:c1])
    if not len(rows):
        raise Unsolid("the cloud has no solid point near that place (a hole, glare, or an edge): pick a place on a plain surface")
    apart = np.hypot(rows + r0 - row, cols + c0 - col)
    nearest = int(np.argmin(apart))
    if apart[nearest] > SNAP_PX:
        raise Unsolid("the cloud has no solid point near that place (a hole, glare, or an edge): pick a place on a plain surface")
    return int(rows[nearest] + r0), int(cols[nearest] + c0), float(apart[nearest])


@dataclass(frozen=True)
class Found:
    """A measurement found in a cloud: its ends as rays from the camera (unit direction, and distance in the model's
    metres), where each place was moved to, and how far."""

    measure: Measure
    rays: tuple[tuple[np.ndarray, float], ...]
    points: tuple[tuple[float, float], ...]  # the places used, as fractions of the picture
    moved_px: tuple[float, ...]

    @property
    def range(self) -> float:
        """How far from the camera the measured thing is, in the model's metres."""
        return float(np.mean([r for _, r in self.rays]))

    def length(self, scale: float = 1.0, power: float = 1.0) -> float:
        """The measured length as the cloud has it under a fit."""
        ends = [d * (scale * r**power) for d, r in self.rays]
        return float(np.linalg.norm(ends[0] - ends[1])) if len(ends) == 2 else float(np.linalg.norm(ends[0]))


def find(pixels: np.ndarray, centre: Sequence[float], measure: Measure) -> Found:
    """`measure` in a cloud whose points by pixel are `pixels` and whose camera stands at `centre`. Raises Unsolid."""
    eye = np.asarray(centre, dtype=np.float64)
    h, w = pixels.shape[:2]
    rays, used, moved = [], [], []
    for u, v in measure.points:
        row, col, far = snap(pixels, u, v)
        ray = pixels[row, col].astype(np.float64) - eye
        reach = float(np.linalg.norm(ray))
        if reach < 1e-6:
            raise Unsolid("that place is at the camera itself")
        rays.append((ray / reach, reach))
        used.append(((col + 0.5) / w, (row + 0.5) / h))
        moved.append(far)
    found = Found(measure, tuple(rays), tuple(used), tuple(moved))
    if found.length() < 1e-4:
        raise Unsolid("the two places are the same point of the cloud: pick them farther apart")
    return found


@dataclass(frozen=True)
class Fit:
    """How a cloud's distances from its camera are resized: r becomes scale * r ** power."""

    scale: float = 1.0
    power: float = 1.0

    def at(self, reach: float) -> float:
        """What a distance of `reach` from the camera is multiplied by."""
        return self.scale * max(reach, 1e-6) ** (self.power - 1.0)

    @property
    def stretched(self) -> bool:
        return abs(self.power - 1.0) > 1e-9


def fit(found: Sequence[Found]) -> Fit:
    """The fit that makes `found`'s measurements come out at their real lengths (the module docstring): a scale
    alone from one, or from several the same distance from the camera; a scale and a stretch from measurements at
    different distances."""
    if not found:
        return Fit()
    ratios = [f.measure.metres / f.length() for f in found]
    alone = Fit(float(np.exp(np.mean(np.log(ratios)))))
    reach = [f.range for f in found]
    if len(found) < 2 or max(reach) < SPREAD * min(reach):
        return alone
    from scipy.optimize import least_squares

    def off(params: np.ndarray) -> np.ndarray:
        scale, power = math.exp(params[0]), params[1]
        return np.array([math.log(f.length(scale, power) / f.measure.metres) for f in found])

    got = least_squares(off, [math.log(alone.scale), 1.0], bounds=([-10.0, POWER[0]], [10.0, POWER[1]]))
    return Fit(float(math.exp(got.x[0])), float(got.x[1]))


def resize(points: np.ndarray, centre: Sequence[float], how: Fit) -> tuple[np.ndarray, np.ndarray]:
    """`points` (N x 3, a cloud's level frame, its camera at `centre` over the floor's origin) resized by `how` about
    the camera, and where the camera then stands: as high as the fit makes its height."""
    eye = np.asarray(centre, dtype=np.float64)
    rays = np.asarray(points, dtype=np.float64) - eye
    reach = np.linalg.norm(rays, axis=1, keepdims=True)
    grown = rays * (how.scale * np.maximum(reach, 1e-9) ** (how.power - 1.0))
    stands = np.array([eye[0], eye[1], eye[2] * how.at(float(eye[2]))])
    return grown + stands, stands


@dataclass(frozen=True)
class Sized:
    """A camera's cloud as its measurements size it."""

    #: "measured": sized by the camera's own measurements; "set_aside": it has measurements, but has moved since;
    #: "approximate": it has none (or none that can be found in its cloud)
    state: str
    fit: Fit
    found: tuple[Found, ...] = ()
    unfound: tuple[tuple[Measure, str], ...] = ()  # the measurements that couldn't be found in this cloud, and why


def size(pixels: np.ndarray | None, centre: Sequence[float], measures: Sequence[Measure], moved: bool) -> Sized:
    """How `measures` size a cloud (its points by pixel `pixels`, its camera at `centre`); `moved` says the camera has
    moved since it was placed, which sets them aside."""
    if not measures or pixels is None:
        return Sized("approximate", Fit())
    if moved:
        return Sized("set_aside", Fit())
    found, unfound = [], []
    for measure in measures:
        try:
            found.append(find(pixels, centre, measure))
        except Unsolid as e:
            unfound.append((measure, str(e)))
    if not found:
        return Sized("approximate", Fit(), (), tuple(unfound))
    return Sized("measured", fit(found), tuple(found), tuple(unfound))
