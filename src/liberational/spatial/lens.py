"""Wide-lens fixed cameras (part of spatial memory): a camera's radial distortion and the overlays
it burns into its frames, so MASt3R, the footprints, and the splats see it as the pinhole they assume.

MASt3R, `places.CameraPose`, `places.floor_point`, and the splat trainer all take a camera as a pinhole with its
principal point at the frame's centre. A wide lens (the Reolink cameras: bedroom and kitchen on the Linux machine) bends
straight lines towards the frame's edges, which bends its share of a place's cloud and misplaces foot points near the
edges. So a camera with a `lens` calibration in its `config/devices.toml` entry has its frames undistorted before MASt3R
and the splats see them, and its foot points undistorted before they're projected onto the floor. A camera registered
with others (spatial/registration.py) can also carry a lens the registration fitted, kept the same way on its pose: its
frames go to MASt3R as they are, and only its rays are undone.

**The model** is the one-parameter division model (Fitzgibbon, 2001): a point seen at normalized position (x, y)
(pixels from the principal point over the focal length) is really at (x, y) / (1 + lambda r^2), with r^2 = x^2 + y^2.
It's stored the way `tools/calibrate_lens.py` writes it, in the pixels of the frame it was calibrated on:

    lens = { width = 896, height = 512, k = [448.0, 448.0, 447.5, 255.5], division = -0.15 }

`k` is fx, fy, cx, cy. The calibration is a plumb-line fit (straight edges in the room must come out straight), which
fixes the distortion but not the focal length, so `k`'s focal is only the scale `division` is in: MASt3R finds the
undistorted frame's own focal. Why not OpenCV's standard (Brown-Conrady) k1, k2: on the Reolinks the barrel needs a k1
of about -0.15 at this scale, where the polynomial stops growing before the frame's edges (it folds), so the edges and
corners would have no undistorted point; the division model never folds and undoes in closed form both ways (chosen
after comparing the fits).

**The undistorted frame** is the same size as the frame, its principal point where the calibration's is, and its scale
`out_scale` times the calibration's, chosen so the whole frame stays in it, every edge and corner on both axes:
undoing a barrel's squeeze pushes the frame's edges outwards, its corners furthest, and a frame cropped to its edges'
middles lost its corners and, on the kitchen's camera, the floor strip along its bottom (a single camera's
floor is all a place has). Where the frame doesn't reach (the black around the barrel's undone outline) is in
`prepare`'s mask with the overlays, so nothing uses it. `undistort_points` maps a pixel of the frame (as fractions of
its width and height, so any resolution of the stream works) to where it is in the undistorted frame, and
`undistort_maps` makes the undistorted frame, so the two always agree.

A calibration is only good where the model is: the division model's undistortion runs to infinity where
1 + lambda r^2 reaches 0, and past it gives nothing. A barrel fitted on edges near the middle of a frame can put that
pole inside it (living-room's portrait frame on the Linux machine: its top and bottom were past it), so `lens_of`
refuses a calibration whose frame reaches past MIN_DENOMINATOR of it.

**Overlays** (a timestamp, the camera's name, a logo) are rectangles of the frame, `masks = [[x1, y1, x2, y2], ...]`
in fractions of its width and height, in the same manifest entry. They are painted over from their surroundings
(`cv2.inpaint`) before the frame is undistorted, so MASt3R finds no matches in them (two cameras burning the same clock
in the same place would otherwise match each other there), and `mask` gives the undistorted frame's pixels they cover,
which MASt3R's cloud and the splat trainer's loss leave out.
"""

from __future__ import annotations

import functools
import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: How close to the division model's pole (1 + lambda r^2 = 0) a frame's corner may be: at 0.35 a corner is undone to
#: about three times its distance from the centre. The Reolinks' corners are at 0.59.
MIN_DENOMINATOR = 0.35


@dataclass(frozen=True)
class Lens:
    """A camera's radial distortion, in the pixels of the frame it was calibrated on."""

    width: int
    height: int
    k: tuple[float, float, float, float]  # fx, fy, cx, cy
    division: float  # lambda: negative for a barrel

    def fractions(self) -> tuple[float, float, float, float]:
        """fx, fy, cx, cy as fractions of the frame's width and height, so any resolution of the stream works."""
        fx, fy, cx, cy = self.k
        return fx / self.width, fy / self.height, (cx + 0.5) / self.width, (cy + 0.5) / self.height

    def as_tuple(self) -> tuple[float, ...]:
        """The calibration as `CameraPose.lens` keeps it: fx, fy, cx, cy in fractions of the frame, then lambda."""
        return (*self.fractions(), self.division)

    @staticmethod
    def from_tuple(value: Sequence[float]) -> Lens:
        """The inverse of `as_tuple`, on a nominal 1 x 1 frame (only fractions matter to the mappings)."""
        fx, fy, cx, cy, division = (float(x) for x in value)
        return Lens(1, 1, (fx, fy, cx - 0.5, cy - 0.5), division)


def lens_of(entry: Mapping[str, Any]) -> Lens | None:
    """A manifest device entry's lens, or None without a `lens` calibration. A malformed one is refused."""
    raw = entry.get("lens")
    if raw is None:
        return None
    k, division = [float(x) for x in raw["k"]], float(raw["division"])
    if len(k) != 4 or k[0] <= 0 or k[1] <= 0 or not all(map(math.isfinite, [*k, division])):
        raise ValueError(f"{entry.get('name')}: lens k needs fx, fy, cx, cy and a finite division")
    width, height = int(raw["width"]), int(raw["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"{entry.get('name')}: lens needs the width and height it was calibrated at")
    found = Lens(width, height, (k[0], k[1], k[2], k[3]), division)
    if denominator(found) < MIN_DENOMINATOR:
        raise ValueError(
            f"{entry.get('name')}: lens division {division} undoes the frame's corners to infinity or past it "
            f"(1 + lambda r^2 is {denominator(found):.2f} there, under {MIN_DENOMINATOR}): fit it again with edges "
            "nearer the frame's corners"
        )
    return found


def _boundary(lens: Lens) -> tuple[np.ndarray, np.ndarray]:
    """The frame's four corners and four edge middles, normalized (pixels from the principal point over the focal)."""
    fx, fy, cx, cy = lens.fractions()
    u = np.array([0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5])
    return (u - cx) / fx, (v - cy) / fy


def denominator(lens: Lens) -> float:
    """The least 1 + lambda r^2 over the frame's corners and edges: under 0, part of the frame has no undistorted
    point; near 0, it's undone to far away."""
    x, y = _boundary(lens)
    return float((1 + lens.division * (x * x + y * y)).min())


def masks_of(entry: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    """A manifest device entry's overlay rectangles (fractions of the frame), none without `masks`. A malformed one is
    refused."""
    out = []
    for m in entry.get("masks") or []:
        x1, y1, x2, y2 = (float(v) for v in m)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(
                f"{entry.get('name')}: a mask is [x1, y1, x2, y2], fractions of the frame, x1 < x2, y1 < y2"
            )
        out.append((x1, y1, x2, y2))
    return out


def for_camera(manifest: Path, camera: str) -> tuple[Lens | None, list[tuple[float, float, float, float]]]:
    """`camera`'s lens and overlay masks in capture's device manifest: (None, []) for a pinhole camera with none, or
    one not in it."""
    try:
        devices = tomllib.loads(manifest.read_text()).get("device", [])
    except FileNotFoundError:
        return None, []
    entry = next((d for d in devices if isinstance(d, dict) and d.get("name") == camera), None)
    return (None, []) if entry is None else (lens_of(entry), masks_of(entry))


def manifest() -> Path:
    """Capture's device manifest, where each camera's `lens` and `masks` are."""
    from liberational import config as cfg

    return cfg.devices_file()


def as_registered(camera: str) -> tuple[float, ...] | None:
    """`camera`'s lens as a registration keeps it (`CameraPose.lens`), or None for a pinhole camera."""
    found, _ = for_camera(manifest(), camera)
    return None if found is None else found.as_tuple()


def _undistort(lens: Lens, xd: np.ndarray, yd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized seen points to where they really are."""
    s = 1 / (1 + lens.division * (xd * xd + yd * yd))
    return xd * s, yd * s


def _distort(lens: Lens, xu: np.ndarray, yu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized real points to where they're seen: r_d solves lambda r_u r_d^2 - r_d + r_u = 0, the root that is r_u
    when lambda is 0."""
    ru = np.hypot(xu, yu)
    q = lens.division * ru
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = np.where(np.abs(q) > 1e-12, (1 - np.sqrt(np.maximum(1 - 4 * q * ru, 0))) / (2 * q), ru)
        s = np.where(ru > 1e-12, rd / ru, 1.0)
    return xu * s, yu * s


@functools.cache
def out_scale(lens: Lens) -> float:
    """The undistorted frame's scale over the calibration's: the largest that keeps the whole frame in it, each of its
    corners and edge middles (undone) inside on both axes; at most 1, so a pincushion's frame isn't enlarged. Worked out
    once per lens: a positions read undistorts tens of thousands of foot points through a handful of lenses."""
    fx, fy, cx, cy = lens.fractions()
    xd, yd = _boundary(lens)
    x, y = _undistort(lens, xd, yd)
    with np.errstate(divide="ignore"):
        # How far each point may sit from the principal point on each axis, in the undistorted frame's fractions, over
        # how far it does at the calibration's scale: the scale that puts it on that edge.
        room_x = np.where(x < 0, cx, 1 - cx) / np.abs(fx * x)
        room_y = np.where(y < 0, cy, 1 - cy) / np.abs(fy * y)
    return float(min(1.0, float(room_x.min()), float(room_y.min())))


def undistort_points(lens: Lens, u: np.ndarray | float, v: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Pixels of the frame, as fractions of its width and height, to where they are in the undistorted frame (fractions
    too; a point pushed off it comes back outside 0 to 1)."""
    fx, fy, cx, cy = lens.fractions()
    x, y = _undistort(lens, (np.asarray(u, float) - cx) / fx, (np.asarray(v, float) - cy) / fy)
    s = out_scale(lens)
    return cx + s * fx * x, cy + s * fy * y


def undistort_maps(lens: Lens, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """cv2.remap's maps for a `width` x `height` frame: for each pixel of the undistorted frame, where it is in the
    frame."""
    fx, fy, cx, cy = lens.fractions()
    s = out_scale(lens)
    uu, vv = np.meshgrid((np.arange(width) + 0.5) / width, (np.arange(height) + 0.5) / height)
    x, y = _distort(lens, (uu - cx) / (s * fx), (vv - cy) / (s * fy))
    return ((cx + fx * x) * width - 0.5).astype(np.float32), ((cy + fy * y) * height - 0.5).astype(np.float32)


def mask(masks: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    """The frame's overlay pixels (H x W bool), from rectangles in fractions of it, each rounded outwards."""
    out = np.zeros((height, width), bool)
    for x1, y1, x2, y2 in masks:
        out[math.floor(y1 * height) : math.ceil(y2 * height), math.floor(x1 * width) : math.ceil(x2 * width)] = True
    return out


def prepare(image: np.ndarray, lens: Lens | None, masks: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """A BGR frame as MASt3R and the splats should see it: overlays painted over from their surroundings, then
    undistorted when there's a `lens`; and the undistorted frame's overlay pixels (H x W bool, all False with no
    masks). The frame keeps its size."""
    import cv2

    height, width = image.shape[:2]
    covered = mask(masks, width, height)
    out = cv2.inpaint(image, covered.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA) if covered.any() else image
    if lens is not None:
        mx, my = undistort_maps(lens, width, height)
        out = cv2.remap(out, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        outside = (mx < 0) | (my < 0) | (mx > width - 1) | (my > height - 1)  # the black the corners left
        moved = cv2.remap(covered.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        covered = moved.astype(bool) | outside
    return out, covered


# A camera's own intrinsics, held. A lens and a focal are properties of the camera, true wherever it stands,
# so they're fitted once (tools/fit_intrinsics.py: against a LiDAR walk's frames, else the median of the camera's
# nightly fits) and every registration after keeps them, fitting only the camera's pose. Fitted afresh each night
# they swung: Place 1's camera-b came out at division -0.18, -0.30, and -0.26 on three runs in one day, and
# its far floor points moved by metres with them. They're owner-backed state, kept by the knowledge store (exported
# with the rest) under `camera.intrinsics`, newest per camera; an earlier fit stays in its history.

INTRINSICS = "camera.intrinsics"


@dataclass(frozen=True)
class Held:
    """A camera's held intrinsics: its focal as a share of its frame's width, and its lens's division (0 for a
    pinhole, or for a camera whose frames its `devices.toml` lens undoes first, whose undone frame is a pinhole)."""

    focal: float
    division: float
    source: str  # how it was fitted, for the record
    residual: float | None = None  # the fit's median reprojection error, pixels at MASt3R's size, when it had one


def held(state: Path) -> dict[str, Held]:
    """Every camera's held intrinsics, newest per camera; {} before any. Reading doesn't create a knowledge store."""
    from liberational import knowledge
    from liberational.people import registry

    if not registry.knowledge_path(state).exists():
        return {}
    newest: dict[str, knowledge.Record] = {}
    for row in knowledge.at(registry.knowledge_path(state)).query(INTRINSICS):
        r = row.record
        if r.key is not None and (r.key not in newest or r.start_ns >= newest[r.key].start_ns):
            newest[r.key] = r
    out = {}
    for cam, r in newest.items():
        v = r.value
        focal, division = v.get("focal"), v.get("division")
        if isinstance(focal, int | float) and isinstance(division, int | float) and focal > 0:
            residual = v.get("residual")
            out[cam] = Held(
                float(focal),
                float(division),
                str(v.get("source") or ""),
                float(residual) if isinstance(residual, int | float) else None,
            )
    return out


def hold(state: Path, camera: str, intrinsics: Held, now: float | None = None) -> None:
    """Store `camera`'s intrinsics as held, over any held before (which stays in the store's history)."""
    import time
    import uuid

    from liberational import knowledge
    from liberational.people import registry

    if intrinsics.focal <= 0:
        raise ValueError("a focal is a positive share of the frame's width")
    store = knowledge.at(registry.knowledge_path(state))
    at_ns = round((time.time() if now is None else now) * 1e9)
    was = [row.record.start_ns for row in store.query(INTRINSICS, key=camera)]
    if was and at_ns <= max(was):
        at_ns = max(was) + 1  # the newest record stays the one written last
    value = {
        "camera": camera,
        "focal": intrinsics.focal,
        "division": intrinsics.division,
        "source": intrinsics.source,
        "residual": intrinsics.residual,
    }
    store.put([knowledge.Record(str(uuid.uuid7()), INTRINSICS, camera, at_ns, at_ns, time.time_ns(), value)])
