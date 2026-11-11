"""Correcting a fixed camera's monocular depth against the owner's plan (#394).

A camera with a spot on the plan has a known pose, and the plan says where its walls and floor are, so each pixel
that sees a wall or the floor has a depth the plan expects (`expected`). The depth model's output is locally right
(each wall is flat to a centimetre or so) but its scale drifts with range: it reads near floor about 11 percent too
far against far walls. Scaling a whole view by its floor alone (`surroundings_map.spot_fit`, the only correction
before this) therefore pulls far walls 0.4 to 0.6 m short of the plan, and the splat trained on those views inherits
it.

**The model.** On seven spot views of two cameras on the Linux machine, fitted against their plan, the error is a
scale and a shift in inverse depth, `1 / d' = a / d + b`. In five of them the best single scale still left pixels 3.5 to
5 m off reading 6 to 13 percent short: the error depends on range, so no scale removes it. A scale and shift in depth and one in inverse
depth both fit, but the inverse-depth one kept the most pixels on its line in five of the seven views and the
smallest trend left at range. It is also the form this kind of depth model is usually corrected with. Fitted per
view, it moved the walls from 0.37 to 0.58 m short of the plan to within 3.5 cm, with the floor unchanged.

**The fit** (`fit`) is robust, since furniture and people stand in front of walls and on the floor. Those pixels read
nearer than the plan, never farther (rays through doors, archways, and windows are left out, `expected`), and they
lie at scattered depths, so the largest set of pixels one line fits is the room's own surfaces: a RANSAC over pixel
pairs in inverse depth, then least squares on its inliers. The line must hold some of the walls and some of the
floor, or the floor (most of a view's known pixels) would pick a line the walls disagree with. A view where too few
pixels see the plan's walls or floor (`few`), too few agree with any one line (`disagrees`), or the line would bend
the depth too far (`bends`) is not corrected (`Fit.why`): the caller keeps the floor scale it had and reports why.
Nothing is seen through a wall, so a candidate line loses a pixel for each one it would put past the plan's surfaces,
and a fit that puts more than BEYOND of them there is refused (`disagrees`): with a cabinet face across most of a
lone wall, the line through the floor and the face holds more pixels than the true one, and the strip of real wall
around the cabinet then reads past the plan.

**Only over the fitted range** (`apply`). The line is applied between the nearest and farthest inliers' depths, and
outside them the ratio at that end holds: a doorway, a window, or a hallway reads past the far wall, and the line
extended there (with its usual negative `b`) would stretch it without bound.

**Cost.** One fit a kept view: a ray cast of a strided pixel grid against the plan's wall segments, and a few hundred
RANSAC samples. No per-frame or per-training-step work.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from liberational.spatial import plan_import, spot_check

STRIDE = 4  # pixels between the samples the fit reads
WALL_TOP_M = 2.0  # a wall hit above this may be ceiling, which the plan doesn't hold
OPENING_M = 0.05  # a ray whose wall hit moves this far once openings are cut passes through one
MAX_DEPTH_M = 12.0
INLIER = 0.04  # relative depth error a pixel may have and still be the plan's surface
SAMPLES = 400  # RANSAC pixel pairs
MIN_PIXELS = 400  # samples that must see a plan wall or floor
MIN_EACH = 80  # inliers on the walls and on the floor each: both are needed to tell a shift from a scale
MIN_SHARE = 0.12  # share of the samples that see the plan which must agree with the line
MAX_CHANGE = 0.35  # how far the correction may change any inlier's depth, relative; past it the plan or pose is off
BEYOND = 0.10  # share of the samples that see the plan which may read farther than it once corrected
SLOPE = (0.5, 2.0)  # the fitted `a`: the model's depth has the room's shape, so a line far from 1 fits something else
SEED = 394


@dataclass(frozen=True)
class Expected:
    """The depth (metres along the view axis) the plan expects at each sampled pixel of a view, and what it hits:
    `wall` True for a wall, False for the floor. `known` is False where the plan can't say (a ray through an opening,
    above WALL_TOP_M on a wall, or out of every room)."""

    v: np.ndarray  # (N,) sample rows
    u: np.ndarray  # (N,) sample columns
    depth: np.ndarray  # (N,)
    wall: np.ndarray  # (N,) bool
    known: np.ndarray  # (N,) bool


def _cast(origin: np.ndarray, dirs: np.ndarray, w: plan_import.Walls) -> np.ndarray:
    """For (N, 2) horizontal ray directions from `origin` (2,), the ray parameter of the nearest wall face (the
    centreline moved half a wall toward the camera, as `spot_check.raycast_2d` does), inf for none."""
    out = np.full(len(dirs), np.inf)
    if not len(w.a):
        return out
    ab = w.b - w.a
    length = np.hypot(ab[:, 0], ab[:, 1])
    keep = length > 1e-6
    a, ab, length, half = w.a[keep], ab[keep], length[keep], w.half[keep]
    normal = np.stack([-ab[:, 1], ab[:, 0]], axis=1) / length[:, None]
    signed = np.sum((origin[None, :] - a) * normal, axis=1)
    inward = np.where(signed[:, None] >= 0, normal, -normal)
    face = a + inward * np.minimum(half, np.maximum(np.abs(signed) - 0.01, 0.0))[:, None]
    rel = face - origin[None, :]
    for i in range(len(face)):
        denom = dirs[:, 0] * ab[i, 1] - dirs[:, 1] * ab[i, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (rel[i, 0] * ab[i, 1] - rel[i, 1] * ab[i, 0]) / denom
            s = (rel[i, 0] * dirs[:, 1] - rel[i, 1] * dirs[:, 0]) / denom
        hit = (np.abs(denom) > 1e-12) & (t > 1e-3) & (s >= -1e-4) & (s <= 1.0001)
        out = np.where(hit & (t < out), t, out)
    return out


def expected(
    geom: spot_check.PlanGeometry, pose: np.ndarray, focal: float, height: int, width: int, stride: int = STRIDE
) -> Expected:
    """The plan's depth (`geom`, from `spot_check.prepare_plan`) at every `stride`-th pixel of a `width` x `height` pinhole view with `focal` (pixels) at
    `pose` (camera to plan: x, y, and height above the floor; the camera looks along its +z with its principal point
    at the centre, as `surroundings_map.lift` takes it)."""
    c2w = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    v, u = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    v, u = v.ravel(), u.ravel()
    local = np.stack([(u + 0.5 - width / 2) / focal, (v + 0.5 - height / 2) / focal, np.ones(len(u))], axis=1)
    dirs = local @ c2w[:3, :3].T  # a step of 1 along each is a step of 1 m in depth
    centre = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        floor_t = np.where(dirs[:, 2] < -1e-9, -centre[2] / dirs[:, 2], np.inf)
    full = _cast(centre[:2], dirs[:, :2], geom.full)
    cut = _cast(centre[:2], dirs[:, :2], geom.opaque)  # doors, archways, and windows cut
    wall = full < floor_t
    depth = np.minimum(full, floor_t)
    through = wall & ~(np.abs(cut - full) < OPENING_M)
    top = centre[2] + full * dirs[:, 2]
    known = np.isfinite(depth) & (depth <= MAX_DEPTH_M) & ~through & ~(wall & (top > WALL_TOP_M))
    return Expected(v, u, depth, wall, known)


@dataclass(frozen=True)
class Fit:
    """A view's correction, `1 / d' = a / d + b` (`apply`), with how it was found: the samples that saw the plan
    (`seen`), the share of them on the fitted line (`share`), the inliers on walls and on the floor, and the median
    signed relative error of the wall and floor inliers before and after (positive reads farther than the plan).
    `why` says why it isn't to be used (None when it is): `few` (too little of the plan in sight), `disagrees` (too
    few pixels agree with any one line), or `bends` (the line would change the depth too much)."""

    a: float
    b: float
    seen: int
    share: float
    walls: int
    floor: int
    wall_before: float | None
    wall_after: float | None
    floor_before: float | None
    floor_after: float | None
    why: str | None
    near: float = 0.0  # the model's depth at the nearest and farthest inliers: the range the line was fitted on
    far: float = MAX_DEPTH_M
    beyond: float = 0.0  # the share of the samples that see the plan which read farther than it once corrected

    def as_dict(self) -> dict[str, object]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 4)

        return {
            "a": round(self.a, 5),
            "b": round(self.b, 5),
            "seen": self.seen,
            "share": round(self.share, 3),
            "walls": self.walls,
            "floor": self.floor,
            "wall_before": r(self.wall_before),
            "wall_after": r(self.wall_after),
            "floor_before": r(self.floor_before),
            "floor_after": r(self.floor_after),
            "why": self.why,
            "near": round(self.near, 3),
            "far": round(self.far, 3),
            "beyond": round(self.beyond, 3),
        }


def apply(depth: np.ndarray, a: float, b: float, near: float = 0.0, far: float = MAX_DEPTH_M) -> np.ndarray:
    """`depth` (metres, 0 where unknown) corrected by `1 / d' = a / d + b` between `near` and `far` (the depths the
    line was fitted on, `Fit.near` and `Fit.far`), and by the ratio the line gives at the nearer or farther end
    outside them. The line is not extrapolated: with a negative `b` it would stretch a doorway or a window read past
    the far wall without bound, and send it to infinity past `a / -b`. 0 stays 0."""
    d = np.asarray(depth, dtype=np.float64)
    out = np.zeros_like(d)
    seen = d > 0
    held = np.clip(d[seen], max(near, 1e-3), max(far, near, 1e-3))
    out[seen] = d[seen] / (held * (a / held + b))  # d times the ratio d' / d at the held depth
    return out


def _median(x: np.ndarray) -> float | None:
    return float(np.median(x)) if len(x) else None


def fit(depth: np.ndarray, exp: Expected) -> Fit:
    """Fit `1 / d' = a / d + b` between a view's `depth` (H x W metres, 0 where unknown) and the plan's `exp`."""
    d = np.asarray(depth, dtype=np.float64)[exp.v, exp.u]
    use = exp.known & (d > 0) & (d <= MAX_DEPTH_M)
    x, y, wall = 1.0 / d[use], 1.0 / exp.depth[use], exp.wall[use]
    none = Fit(1.0, 0.0, int(use.sum()), 0.0, 0, 0, None, None, None, None, "few")
    if len(x) < MIN_PIXELS or wall.sum() < MIN_EACH or (~wall).sum() < MIN_EACH:
        return none
    rng = np.random.default_rng(SEED)
    i, j = rng.integers(0, len(x), SAMPLES), rng.integers(0, len(x), SAMPLES)
    dx = x[i] - x[j]
    ok = np.abs(dx) > 0.02 * np.maximum(x[i], x[j])
    a_all = np.where(ok, (y[i] - y[j]) / np.where(ok, dx, 1.0), 0.0)
    b_all = y[i] - a_all * x[i]
    ok &= a_all > 0
    # The line with the most pixels on it, less the pixels it would put past the plan's surfaces (nothing is seen
    # through a wall, so those say the line is wrong), among the lines that hold some of the walls and some of the
    # floor: the floor is most of a view's known pixels, and on its own it would pick a line the walls disagree with.
    # Without the penalty, a cabinet face in front of most of a lone wall, with the floor, outvotes the strip of real
    # wall around it.
    best, best_n = None, -1
    for a_c, b_c in zip(a_all[ok], b_all[ok], strict=True):
        err = (a_c * x + b_c) / y - 1.0  # relative depth error, to first order: negative reads farther
        inl = np.abs(err) < INLIER
        n = int(inl.sum()) - int((err < -INLIER).sum())
        if n > best_n and int((inl & wall).sum()) >= MIN_EACH and int((inl & ~wall).sum()) >= MIN_EACH:
            best, best_n = inl, n
    if best is None or not best.any():
        return Fit(1.0, 0.0, len(x), 0.0, 0, 0, None, None, None, None, "disagrees")
    inl = best
    a_ls, b_ls = 1.0, 0.0
    for _ in range(3):  # least squares on the inliers (relative error), then again
        wts = 1.0 / y[inl]
        a_ls, b_ls = np.linalg.lstsq(np.stack([x[inl] * wts, wts], axis=1), y[inl] * wts, rcond=None)[0]
        inl = np.abs((a_ls * x + b_ls) / y - 1.0) < INLIER
        if not inl.any():
            return Fit(1.0, 0.0, len(x), 0.0, 0, 0, None, None, None, None, "disagrees")
    a, b = float(a_ls), float(b_ls)
    near, far = 1.0 / float(x[inl].max()), 1.0 / float(x[inl].min())
    before = y / x - 1.0  # (d - D) / D: positive reads farther than the plan
    after = apply(1.0 / x, a, b, near, far) * y - 1.0
    n_wall, n_floor = int((inl & wall).sum()), int((inl & ~wall).sum())
    out = Fit(
        a,
        b,
        len(x),
        float(inl.mean()),
        n_wall,
        n_floor,
        _median(before[inl & wall]),
        _median(after[inl & wall]),
        _median(before[inl & ~wall]),
        _median(after[inl & ~wall]),
        None,
        near,
        far,
        float((after > INLIER).mean()),
    )
    if n_wall < MIN_EACH or n_floor < MIN_EACH or out.share < MIN_SHARE or out.beyond > BEYOND:
        return _why(out, "disagrees")
    change = np.abs(x[inl] / (a * x[inl] + b) - 1.0)
    if not SLOPE[0] <= a <= SLOPE[1] or float(change.max()) > MAX_CHANGE:
        return _why(out, "bends")
    return out


def _why(f: Fit, why: str) -> Fit:
    return replace(f, why=why)
