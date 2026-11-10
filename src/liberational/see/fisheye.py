"""Fisheye lenses for see-actions (#1745, #1891): each lens's calibration, and each person resampled into a virtual
pinhole view before pose.

The rig's IMX291 lenses see 160 degrees, so a person near the rim is bent and squeezed, and RTMPose was trained on
ordinary perspective photos. So, for a camera whose manifest entry has a `fisheye` calibration, each detected box is
re-viewed through a virtual pinhole camera aimed at the ray through the box's center, with a field of view that fits
the box grown as the pose crop grows it (`pose_model.PADDING`, to the model's 3:4 aspect). The view is resampled
from the frame on the device with one `grid_sample` per batch, and pose runs on it. Its joints map back through the
virtual camera and the lens to fisheye pixels, so the tracker, the head crops, and the rows downstream see the same
frame coordinates they always do. A camera without a calibration (cam0) is unchanged.

The lens model is OpenCV's fisheye (equidistant with four distortion terms, `cv2.fisheye`): a ray at angle theta
from the optical axis lands at radius theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
from the principal point, scaled by fx and fy. `tools/calibrate_fisheye.py` finds k = (fx, fy, cx, cy) and
d = (k1, k2, k3, k4) from checkerboard views and writes them under the camera in `config/devices.toml`:

    [[device]]
    name = "fish0"
    fisheye = { width = 1920, height = 1080, k = [fx, fy, cx, cy], d = [k1, k2, k3, k4], rms = 0.4 }

Pixel coordinates follow the pose crop's convention (`gpu_pose.crops`): a crop pixel u at its center integer, and
the view's normalized coordinate (u - width / 2) / f.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from liberational.vision import pose_model


@dataclass(frozen=True)
class Lens:
    """A fisheye lens's calibration in the pixels of the frames see-actions gets (upright)."""

    width: int
    height: int
    k: tuple[float, float, float, float]  # fx, fy, cx, cy
    d: tuple[float, float, float, float]  # k1..k4

    def upright(self, flip: bool) -> Lens:
        """The lens of a camera mounted upside down, whose frames are turned 180 degrees: the principal point turns
        with them (the distortion is radial, so it doesn't change)."""
        if not flip:
            return self
        fx, fy, cx, cy = self.k
        return Lens(self.width, self.height, (fx, fy, self.width - 1 - cx, self.height - 1 - cy), self.d)

    def scaled(self, width: int, height: int) -> Lens:
        """The same lens for frames of another size (a calibration at 1920 x 1080, frames at 1280 x 720)."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        fx, fy, cx, cy = self.k
        return Lens(width, height, (fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5), self.d)


def lens_of(entry: Mapping[str, Any]) -> Lens | None:
    """A manifest device entry's lens, or None without a `fisheye` calibration. A malformed one is refused."""
    raw = entry.get("fisheye")
    if raw is None:
        return None
    k, d = [float(x) for x in raw["k"]], [float(x) for x in raw["d"]]
    if len(k) != 4 or len(d) != 4 or k[0] <= 0 or k[1] <= 0:
        raise ValueError(f"{entry.get('name')}: fisheye k needs fx, fy, cx, cy and d four terms")
    lens = Lens(int(raw["width"]), int(raw["height"]), (k[0], k[1], k[2], k[3]), (d[0], d[1], d[2], d[3]))
    flip = bool((entry.get("video") or {}).get("flip"))
    return lens.upright(flip)


def lens_for(manifest: Path, camera: str) -> Lens | None:
    """The calibrated lens of `camera` in capture's device manifest, or None (a plain camera, or no calibration)."""
    try:
        devices = tomllib.loads(manifest.read_text()).get("device", [])
    except OSError:
        return None
    entry = next((d for d in devices if d.get("name") == camera), None)
    return None if entry is None else lens_of(entry)


def project(lens: Lens, rays: torch.Tensor) -> torch.Tensor:
    """Rays (... x 3, any length, camera coordinates: z out of the lens) to fisheye pixels (... x 2)."""
    fx, fy, cx, cy = lens.k
    k1, k2, k3, k4 = lens.d
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    r = torch.sqrt(x * x + y * y)
    theta = torch.atan2(r, z)
    t2 = theta * theta
    theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    scale = torch.where(r > 1e-12, theta_d / r.clamp(min=1e-12), torch.ones_like(r))
    return torch.stack((fx * x * scale + cx, fy * y * scale + cy), -1)


def unproject(lens: Lens, pixels: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Fisheye pixels (... x 2) to unit rays (... x 3), inverting the distortion by Newton's method."""
    fx, fy, cx, cy = lens.k
    k1, k2, k3, k4 = lens.d
    xd, yd = (pixels[..., 0] - cx) / fx, (pixels[..., 1] - cy) / fy
    theta_d = torch.sqrt(xd * xd + yd * yd)
    theta = theta_d.clone()
    for _ in range(iterations):
        t2 = theta * theta
        f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        theta = theta - f / df
    s = torch.where(theta_d > 1e-12, torch.sin(theta) / theta_d.clamp(min=1e-12), torch.ones_like(theta_d))
    return torch.stack((xd * s, yd * s, torch.cos(theta)), -1)


def _basis(center: torch.Tensor) -> torch.Tensor:
    """N x 3 x 3 rotations whose rows are each view's x, y, and z axes in lens coordinates: z along `center` (N x 3
    unit rays), y as close to the lens's own down as z allows, x completing a right-handed frame."""
    z = F.normalize(center, dim=-1)
    down = torch.tensor([0.0, 1.0, 0.0], device=center.device).expand_as(z)
    y = F.normalize(down - (down * z).sum(-1, keepdim=True) * z, dim=-1)
    x = torch.cross(y, z, dim=-1)
    return torch.stack((x, y, z), 1)


def views(lens: Lens, boxes: torch.Tensor, size: pose_model.Size) -> tuple[torch.Tensor, torch.Tensor]:
    """Each box's virtual pinhole view (boxes N x 4 in fisheye pixels; size the pose input, width by height): its
    rotation (N x 3 x 3, rows its axes) and focal length in view pixels (N). The view looks along the ray through the
    box's center and is just wide enough for the box's edges grown PADDING times, to the input's aspect."""
    iw, ih = size
    x1, y1, x2, y2 = boxes.unbind(-1)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    rot = _basis(unproject(lens, torch.stack((mx, my), -1)))
    # The box's corners and edge midpoints, in each view: their tangents bound the angles the view must cover.
    edge = torch.stack(
        [torch.stack(p, -1) for p in ((x1, y1), (x2, y1), (x1, y2), (x2, y2), (mx, y1), (mx, y2), (x1, my), (x2, my))],
        1,
    )
    inview = torch.einsum("nij,nkj->nki", rot, unproject(lens, edge))
    z = inview[..., 2].clamp(min=1e-6)
    hx = (inview[..., 0] / z).abs().amax(1) * pose_model.PADDING
    hy = (inview[..., 1] / z).abs().amax(1) * pose_model.PADDING
    hx = torch.maximum(hx, hy * iw / ih)  # widen or heighten to the input's aspect, as the plain crop does
    return rot, (iw / 2) / hx.clamp(min=1e-6)


def crops(
    frame: torch.Tensor, boxes: torch.Tensor, size: pose_model.Size, lens: Lens
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Pose inputs for boxes in fisheye pixels (N x 4) of a BGR frame (H x W x 3 uint8), each resampled into its
    virtual view: N x 3 x height x width, RGB, normalized as `gpu_pose.crops` does, and each view (rotations, focal
    lengths) for `to_frame`."""
    from liberational.see import gpu_pose

    h, w = frame.shape[:2]
    lens = lens.scaled(w, h)
    iw, ih = size
    rot, focal = views(lens, boxes, size)
    u = torch.arange(iw, device=frame.device, dtype=torch.float32) - iw / 2
    v = torch.arange(ih, device=frame.device, dtype=torch.float32) - ih / 2
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    view_rays = torch.stack((uu, vv, torch.ones_like(uu)), -1)  # ih x iw x 3, before the focal length
    rays = view_rays[None] / torch.stack((focal, focal, torch.ones_like(focal)), -1)[:, None, None, :]
    lens_rays = torch.einsum("nji,nhwj->nhwi", rot, rays)  # the view's axes back to the lens's
    px = project(lens, lens_rays)
    # grid_sample's normalized coordinates (align_corners=False): -1 and 1 are the frame's outer pixel edges.
    grid_ = torch.stack(((2 * px[..., 0] + 1) / w - 1, (2 * px[..., 1] + 1) / h - 1), -1)
    bgr = frame.permute(2, 0, 1)[None].float().expand(len(boxes), -1, -1, -1)
    patch = F.grid_sample(bgr, grid_, mode="bilinear", padding_mode="zeros", align_corners=False)
    rgb = patch.flip(1)
    mean, std = gpu_pose._norm(frame.device)
    return (rgb - mean) / std, (rot, focal)


def to_frame(
    locs: torch.Tensor, view: tuple[torch.Tensor, torch.Tensor], size: pose_model.Size, lens: Lens, shape: Sequence[int]
) -> torch.Tensor:
    """Joints in view pixels (N x K x 2) back to fisheye frame pixels, through each view and the lens. `shape` is the
    frame's height and width."""
    rot, focal = view
    iw, ih = size
    lens = lens.scaled(int(shape[1]), int(shape[0]))
    xn = (locs[..., 0] - iw / 2) / focal[:, None]
    yn = (locs[..., 1] - ih / 2) / focal[:, None]
    rays = torch.stack((xn, yn, torch.ones_like(xn)), -1)
    return project(lens, torch.einsum("nji,nkj->nki", rot, rays))
