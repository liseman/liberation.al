"""What the spatial jobs that work from the cameras' frames share: which cameras are fixed, reading and scoring a
camera's frames, and a camera's held intrinsics.

- **Cameras.** `fixed_cameras` reads the capture manifest (`config/devices.toml`): the cameras on no rig
  (spatial/rig.py), which the position layer places people from.
- **Frames.** `recorded` reads a camera's recorded frame nearest a time from the broker. `daylight` says whether a
  frame is lit colour (not dark, not a camera's night mode), and `sharpness` scores its focus and motion blur.
- **Held intrinsics.** `with_held` gives a camera's pose the focal and lens held for it (spatial/lens.py).

Nothing here runs on a schedule or writes a place.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from liberational.memory import places
from liberational.spatial import lens, rig

#: A frame darker than this mean (0 to 255) is dark.
MIN_BRIGHTNESS = 60.0
#: A frame whose channels differ by less than this on average is a night-mode (infrared) grey frame.
MIN_COLOUR = 4.0
#: `recorded` takes the recorded frame nearest its time, this close.
RECORDED_WITHIN_S = 2.0


def fixed_cameras(manifest: Path) -> list[str]:
    """The enabled fixed cameras in the capture manifest, in manifest order: not on a rig (`mount.rig`, spatial/rig.py,
    #5), whose poses change as it moves, and with no `fisheye` calibration, which a pinhole pose can't take."""
    try:
        parsed = tomllib.loads(manifest.read_text())
    except FileNotFoundError:
        return []
    mounts = rig.parse(parsed)
    return [
        str(d["name"])
        for d in parsed.get("device", [])
        if isinstance(d, dict)
        and d.get("kind") in {"camera", "rtsp"}
        and d.get("enabled", True)
        and "fisheye" not in d
        and d.get("name")
        and mounts.get(str(d["name"])).fixed
    ]


def light(pixels: np.ndarray) -> tuple[float, float]:
    """A BGR frame's mean brightness (0 to 255) and colour (the mean difference between its channels: near 0 for a
    grey, night-mode frame); (0, 0) for a frame that isn't three-channel."""
    img = np.asarray(pixels, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        return (0.0, 0.0)
    spread = np.abs(img[..., 0] - img[..., 1]) + np.abs(img[..., 1] - img[..., 2])
    return (float(img.mean()), float(spread.mean()))


def daylight(pixels: np.ndarray) -> bool:
    """Whether a BGR frame is lit colour, rather than a dark frame or a camera's night mode (infrared: grey, its
    three channels all but equal)."""
    brightness, colour = light(pixels)
    return brightness >= MIN_BRIGHTNESS and colour >= MIN_COLOUR


def recorded(cam: str, t: float) -> Any:
    """`cam`'s recorded frame nearest `t` from the pack's broker, or None."""
    from liberational import vision_log

    return vision_log.at().frame(cam, t, RECORDED_WITHIN_S)


def sharpness(pixels: np.ndarray) -> float:
    """How sharp a BGR frame is: the variance of its grey image's Laplacian (higher is sharper; motion blur, a
    smudged or fogged lens, and a refocusing camera all lower it)."""
    import cv2

    grey = cv2.cvtColor(np.asarray(pixels, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def camera_pose(pose: np.ndarray, focal: float, division: float, size: Sequence[int], at: float) -> places.CameraPose:
    """A camera's pose as a place keeps it: a pinhole with no lens, else its lens as spatial/lens.py's tuple, centred,
    and its focal the undistorted frame's (`lens.out_scale` of the fitted one), as floor_point casts rays through
    it."""
    w, h = int(size[0]), int(size[1])
    flat = tuple(float(x) for x in np.asarray(pose).ravel())
    if abs(division) < 1e-6:
        return places.CameraPose(flat, float(focal), w, h, at)
    found = lens.Lens(w, h, (focal, focal, (w - 1) / 2, (h - 1) / 2), division)
    return places.CameraPose(flat, float(focal * lens.out_scale(found)), w, h, at, found.as_tuple())


def with_held(camera: str, pose: places.CameraPose, held: lens.Held | None) -> places.CameraPose:
    """`camera`'s `pose` with its `held` intrinsics in place of its own: the focal, and the lens (its configured one
    when `lens.as_registered` has one, since `held.focal` is the undone frame's focal, else the held one). Intrinsics
    measured through another lens than the configured one (`lens.through`) aren't this frame's: `pose` stays as it is."""
    configured = lens.as_registered(camera)
    held = lens.through(held, configured)
    if held is None:
        return pose
    if configured is not None:
        return replace(pose, focal=held.focal * pose.width, lens=configured)
    m = np.asarray(pose.pose, dtype=np.float64).reshape(4, 4)
    found = camera_pose(m, held.focal * pose.width, held.division, (pose.width, pose.height), pose.at)
    return replace(found, background=pose.background)
