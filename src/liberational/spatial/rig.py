"""How each camera is mounted (#5): a camera is a view, not a room.

A camera is an angle on something that holds it. Some cameras sit on a **rig** that moves (a backpack, a cart): a
rig carries its cameras together, each camera's pose is fixed relative to the rig, and the rig's own pose in a place
changes as it moves. Other cameras are **fixed**: mounted on a wall or a shelf, each registered in its place on its
own (spatial/cameras.py). Neither kind is a room: a camera's name says which camera it is, never where it is, and
where a camera's pixel lands comes from its pose, not from its name.

Both are described on the camera's entry in capture's device manifest (`config/devices.toml`), as `mount`, beside its
lens calibration, and rigs are declared as `[[rig]]` tables:

    [[rig]]
    name = "pack"

    [[device]]
    name = "front"
    kind = "camera"
    match = { usb = "1234:5678" }
    mount = { rig = "pack", at = [0.0, -0.1, 0.05], turn = [0.0, -10.0, 0.0], hfov_deg = 110.0 }

    [[device]]
    name = "hall"
    kind = "rtsp"
    mount = { height_m = 2.1, outdoor = false }

- **On a rig** (`rig` names a `[[rig]]`): `at` is where the camera's lens sits in the rig's frame, in metres, and
  `turn` is which way it looks, in degrees: yaw to the right, pitch up, and roll clockwise as seen from behind the
  camera, applied in that order (yaw, then pitch about the turned right axis, then roll about the view axis). Both
  default to zero. The rig's frame has the same axes as a camera's (x right, y down, z forward, the convention of
  `memory.places.CameraPose`), so a camera at the rig's origin with no turn has the rig's own pose. `hfov_deg` is the
  camera's horizontal field of view (its pinhole focal is worked out from it and the frame's width) and is required:
  a rig camera has no fixed registration to fit its focal from. A rig camera has no `height_m` or `outdoor`: how
  high it is and whether it's indoors depend on where the rig is, which its pose track says.
- **Fixed** (no `rig`): `height_m` is the camera's measured height from the floor to its lens, which puts its place's
  map at metric scale (spatial/fuse.py, `metric`) and checks a lone camera's floor (spatial/cameras.py,
  `implausible`); `outdoor = true` keeps a camera that looks outdoors out of the fixed-camera places, where alone it
  would only become a place of its own. A fixed camera with no `mount` has an unknown height and looks indoors. Its
  pose and focal come from its registration (`memory.places.Revision.cameras`), so `at`, `turn`, and `hfov_deg`
  aren't allowed on it.

A rig's pose in a place is a 4x4 rig-to-map matrix in that place's map frame (16 numbers, row by row, as
`CameraPose.pose` is). Whatever tracks the rig (a walkthrough's poses, the position layer's pose track) gives it;
`camera_pose` turns it and a rig camera's mount into that camera's `CameraPose` at that moment, and `to_plan` puts one
of the camera's pixels on the place's plan through the same floor and plan geometry as a fixed camera
(`memory.places.project_pose`). That is all a position writer needs from this module: the camera's name, the rig's
pose, the frame size, and the pixel.

The manifest is read on each call: it's a small file, and a mount edited by hand takes effect on a reader's next pass.
A mount that doesn't make sense raises `ValueError` naming the camera, rather than being half used.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liberational.memory import places

_RIG_KEYS = {"rig", "at", "turn", "hfov_deg"}
_FIXED_KEYS = {"height_m", "outdoor"}


@dataclass(frozen=True)
class Mount:
    """How one camera is mounted: on a rig (`rig` set) or fixed (`rig` None). See the module docstring."""

    camera: str
    rig: str | None = None
    at: tuple[float, float, float] = (0.0, 0.0, 0.0)  # metres in the rig's frame (x right, y down, z forward)
    turn: tuple[float, float, float] = (0.0, 0.0, 0.0)  # degrees: yaw right, pitch up, roll clockwise
    hfov_deg: float | None = None  # a rig camera's horizontal field of view
    height_m: float | None = None  # a fixed camera's measured floor-to-lens height; None when not measured
    outdoor: bool = False  # a fixed camera that looks outdoors

    @property
    def fixed(self) -> bool:
        """Whether the camera is fixed in its place rather than carried on a rig."""
        return self.rig is None

    def to_rig(self) -> tuple[float, ...]:
        """The camera-to-rig 4x4 matrix, row by row (16 numbers): `turn`'s rotation and `at`'s offset."""
        r = turn_matrix(*self.turn)
        return (*r[0], self.at[0], *r[1], self.at[1], *r[2], self.at[2], 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Mounts:
    """Every camera's mount in a manifest, and the rigs it declares."""

    rigs: tuple[str, ...] = ()
    cameras: dict[str, Mount] = field(default_factory=dict)  # every camera and network camera, by name

    def get(self, camera: str) -> Mount:
        """`camera`'s mount; a camera the manifest doesn't list is fixed, indoors, and unmeasured."""
        return self.cameras.get(camera) or Mount(camera)

    def heights(self) -> dict[str, float]:
        """The fixed cameras' measured heights (metres, floor to lens), by camera."""
        return {c: m.height_m for c, m in self.cameras.items() if m.fixed and m.height_m is not None}

    def outdoor(self) -> set[str]:
        """The fixed cameras that look outdoors."""
        return {c for c, m in self.cameras.items() if m.fixed and m.outdoor}

    def on_rig(self, rig: str | None = None) -> list[str]:
        """The cameras on `rig` (on any rig when None), in manifest order."""
        return [c for c, m in self.cameras.items() if m.rig is not None and (rig is None or m.rig == rig)]


def manifest() -> Path:
    """Capture's device manifest, where the rigs and each camera's mount are."""
    from liberational import config as cfg

    return cfg.ROOT / "config" / "devices.toml"


def load(path: Path | None = None) -> Mounts:
    """The mounts in the manifest at `path` (capture's own when None); none for a manifest that isn't there."""
    try:
        text = (path or manifest()).read_text()
    except FileNotFoundError:
        return Mounts()
    return parse(tomllib.loads(text))


def _triple(value: object, what: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(x, int | float) and not isinstance(x, bool) for x in value)
    ):
        raise ValueError(f"{what} is three numbers")
    a, b, c = (float(x) for x in value)
    if not all(math.isfinite(x) for x in (a, b, c)):
        raise ValueError(f"{what} is three finite numbers")
    return a, b, c


def _number(value: object, what: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{what} is a number")
    return float(value)


def _mount(camera: str, raw: Mapping[str, Any], rigs: Sequence[str]) -> Mount:
    unknown = set(raw) - _RIG_KEYS - _FIXED_KEYS
    if unknown:
        raise ValueError(f"camera {camera!r}: mount has no {', '.join(sorted(unknown))}")
    rig = raw.get("rig")
    if rig is None:
        stray = set(raw) & (_RIG_KEYS - {"rig"})
        if stray:
            raise ValueError(f"camera {camera!r}: {', '.join(sorted(stray))} is for a camera on a rig (mount.rig)")
        height = None if raw.get("height_m") is None else _number(raw["height_m"], f"camera {camera!r}: height_m")
        if height is not None and height <= 0:
            raise ValueError(f"camera {camera!r}: height_m is above the floor (more than 0)")
        outdoor = raw.get("outdoor", False)
        if not isinstance(outdoor, bool):
            raise ValueError(f"camera {camera!r}: outdoor is true or false")
        return Mount(camera, height_m=height, outdoor=outdoor)
    if not isinstance(rig, str) or rig not in rigs:
        raise ValueError(f"camera {camera!r}: mount.rig {rig!r} is no [[rig]] in the manifest")
    stray = set(raw) & _FIXED_KEYS
    if stray:
        raise ValueError(
            f"camera {camera!r}: {', '.join(sorted(stray))} is for a fixed camera; a rig camera's height and "
            "whether it's indoors come from where the rig is"
        )
    if "hfov_deg" not in raw:
        raise ValueError(f"camera {camera!r}: a camera on a rig needs hfov_deg, its horizontal field of view")
    hfov = _number(raw["hfov_deg"], f"camera {camera!r}: hfov_deg")
    if not 0 < hfov < 180:
        raise ValueError(f"camera {camera!r}: hfov_deg is between 0 and 180 (a pinhole's)")
    at = _triple(raw.get("at", [0.0, 0.0, 0.0]), f"camera {camera!r}: at")
    turn = _triple(raw.get("turn", [0.0, 0.0, 0.0]), f"camera {camera!r}: turn")
    return Mount(camera, rig=rig, at=at, turn=turn, hfov_deg=hfov)


def parse(manifest: Mapping[str, Any]) -> Mounts:
    """The mounts of a parsed manifest (its TOML as a mapping). Raises ValueError for a rig or mount that doesn't
    make sense."""
    rigs: list[str] = []
    for entry in manifest.get("rig", []) or []:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError("a [[rig]] needs a name")
        if name in rigs:
            raise ValueError(f"rig {name!r} is named twice")
        rigs.append(name)
    cameras: dict[str, Mount] = {}
    for device in manifest.get("device", []) or []:
        if not isinstance(device, dict) or device.get("kind") not in {"camera", "rtsp"}:
            continue
        name = device.get("name")
        if not isinstance(name, str) or not name:
            continue
        raw = device.get("mount") or {}
        if not isinstance(raw, dict):
            raise ValueError(f"camera {name!r}: mount is a table")
        cameras[name] = _mount(name, raw, rigs)
    return Mounts(tuple(rigs), cameras)


def turn_matrix(yaw: float, pitch: float, roll: float) -> tuple[tuple[float, float, float], ...]:
    """The 3x3 rotation of a camera turned on its rig (degrees: yaw right, pitch up, roll clockwise), in the rig's
    camera-style frame (x right, y down, z forward): yaw about the rig's y, then pitch about the turned x, then roll
    about the turned z. Its columns are the camera's right, down, and forward in the rig's frame."""
    a, b, c = (math.radians(v) for v in (yaw, pitch, roll))
    ry = ((math.cos(a), 0.0, math.sin(a)), (0.0, 1.0, 0.0), (-math.sin(a), 0.0, math.cos(a)))
    rx = ((1.0, 0.0, 0.0), (0.0, math.cos(b), -math.sin(b)), (0.0, math.sin(b), math.cos(b)))
    rz = ((math.cos(c), -math.sin(c), 0.0), (math.sin(c), math.cos(c), 0.0), (0.0, 0.0, 1.0))
    return _mul3(_mul3(ry, rx), rz)


def _mul3(
    p: Sequence[Sequence[float]], q: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    rows = [tuple(sum(p[i][k] * q[k][j] for k in range(3)) for j in range(3)) for i in range(3)]
    return (
        (rows[0][0], rows[0][1], rows[0][2]),
        (rows[1][0], rows[1][1], rows[1][2]),
        (rows[2][0], rows[2][1], rows[2][2]),
    )


def _mul4(p: Sequence[float], q: Sequence[float]) -> tuple[float, ...]:
    return tuple(sum(p[4 * i + k] * q[4 * k + j] for k in range(4)) for i in range(4) for j in range(4))


def camera_pose(
    mount: Mount, rig_in_place: Sequence[float], width: int, height: int, at: float = 0.0
) -> places.CameraPose:
    """A rig camera's pose in a place while the rig stands at `rig_in_place` (its 4x4 rig-to-map matrix, 16 numbers
    row by row), for frames `width` x `height` pixels: a pinhole whose focal is the mount's `hfov_deg` across the
    width. `at` is the moment (UTC epoch seconds). Raises ValueError for a fixed camera, whose pose is its
    registration's."""
    if mount.rig is None or mount.hfov_deg is None:
        raise ValueError(f"camera {mount.camera!r} is fixed: its pose is its registration in its place")
    if len(rig_in_place) != 16:
        raise ValueError("a rig's pose is 16 numbers, a 4x4 matrix row by row")
    focal = width / 2 / math.tan(math.radians(mount.hfov_deg) / 2)
    return places.CameraPose(_mul4([float(x) for x in rig_in_place], mount.to_rig()), focal, width, height, at)


def to_plan(
    rev: places.Revision,
    mount: Mount,
    rig_in_place: Sequence[float],
    width: int,
    height: int,
    u: float,
    v: float,
    above_floor: float = 0.0,
) -> tuple[float, float] | None:
    """Where a rig camera's pixel lands on `rev`'s plan while the rig stands at `rig_in_place` in that revision's map
    frame: plan (x, y) in metres, or None, as `memory.places.project` gives a fixed camera's. `u` and `v` are fractions
    of the frame, and `above_floor` the height of what the pixel shows (a foot point's), as `floor_point` takes them."""
    return places.project_pose(rev, camera_pose(mount, rig_in_place, width, height), u, v, above_floor)
