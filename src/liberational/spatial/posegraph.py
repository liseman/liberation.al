"""The walkthrough's pose graph (part of spatial memory): a walk's camera poses, corrected so
the places it passes again line up.

A streaming model (CUT3R) gives every frame a camera-to-world pose that holds locally but drifts in heading over a
long walk, so a wall the walk passes twice shows up twice. `solve` keeps the consecutive frames' relative poses
(odometry, tight), adds the revisits MASt3R verified as loop edges (looser), and pulls every camera's up axis toward
the walk's up (gravity) and its height toward the median (height), then finds the poses that fit all of it best.

Pure functions, no I/O and no model, so this runs (and is tested) on any machine:

- `solve`: poses and edges in, corrected poses out. Right-perturbation Gauss-Newton on SE(3) (each pose moves as
  R exp(w), t + R v), with a Levenberg damping that only grows when a step doesn't lower the cost, and the sparse
  normal equations through scipy. The first pose is fixed (the gauge); there's no scale in the state, so a loop
  can't shrink the walk. Every residual is Huber-weighted per block in whitened units (`HUBER`), so a bad loop
  pulls with a bounded force; but a bounded force still bends a long chain (a loop 90 degrees off turned a 64-pose
  rectangle by 16 degrees, 1.3 m at a corner), so a loop that still disagrees by more than `REJECT` after the solve
  is dropped and the walk solved again without it, worst first. The acceptance checks are the first line, Huber
  the second, and this the third.
- `odometry`: the consecutive edges of a pose sequence.
- `loop_scale`: the one number that puts MASt3R's pair translations into CUT3R's scale, or a refusal when the
  short-baseline pairs disagree on it.
- `loop_refusal`: the acceptance checks a MASt3R pair has to pass to become a loop edge (None when it passes).
- `up_axis_error`: whether a frame's camera -y really is the image's up in the world, the check that has to pass
  before the gravity prior may pull (a wrong prior would bend a walk that's already level).

Cameras are OpenCV's (x right, y down, z forward), as DUSt3R's family uses: an upright frame's up is its -y.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

CAMERA_UP = (0.0, -1.0, 0.0)  # an upright camera's up, in its own frame

# Starting noise of each residual (untuned: the first real run is what moves them).
ODOMETRY_DEG, ODOMETRY_M = 2.0, 0.05  # CUT3R between consecutive frames, the part that already holds
LOOP_DEG, LOOP_M = 5.0, 0.20  # a verified MASt3R pair, scale-aligned
GRAVITY_DEG = 5.0  # a phone held upright
HEIGHT_M = 0.3  # a camera's height along up, about the walk's median
HEIGHT_PRIOR = True  # one storey (IMG_7676); a place with stairs turns this off
HUBER = 1.0  # whitened units: a residual block past this pulls with a constant force
REJECT = 4.0  # whitened: a loop still this far off after the solve (about 20 degrees or 0.8 m) is dropped
ITERATIONS = 30
STEP_TOL = 1e-9  # the solve stops once no pose moves more than this (radians or metres)

# A MASt3R pair becomes a loop edge only when it passes every one of these (`loop_refusal`).
MIN_MATCHES = 150  # reciprocal matches: the walkthrough's chaining floor, which the known IMG_7676 loops clear
AGREE_DEG = 10.0  # the two directions' rotations agree this well
AGREE_DIR_DEG = 20.0  # and their translation directions this well
LOOP_UP_DEG = 20.0  # the relative pose keeps the two cameras' up axes this close
LOOP_GATE_M, LOOP_GATE_DEG = 4.0, 45.0  # how far a loop may disagree with the drifted odometry (loose on purpose)
MAX_BASELINE_M = 25.0  # a scale-aligned loop farther apart than a house isn't one
SCALE_SPREAD = 2.0  # the short-baseline scale ratios (10th to 90th percentile) may spread this much


@dataclass(frozen=True)
class Edge:
    """A measured relative pose: `measured` is pose i's inverse times pose j (j in i's frame), with its noise. A
    loop edge may be dropped by `solve` when the rest of the graph disagrees with it; an odometry edge never is."""

    i: int
    j: int
    measured: np.ndarray  # 4x4
    sigma_deg: float
    sigma_m: float
    loop: bool = False


@dataclass(frozen=True)
class Solution:
    """The corrected camera-to-world poses (n x 4 x 4), and how each edge fits them: its rotation error (degrees)
    and translation error (metres), in the order the edges were given (a dropped loop's too). `dropped` are the
    indices of the loop edges `solve` left out."""

    poses: np.ndarray
    iterations: int
    cost: float
    edge_deg: np.ndarray
    edge_m: np.ndarray
    dropped: tuple[int, ...] = ()


def relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose `b` in `a`'s frame (a^-1 b), for 4x4 rigid poses."""
    out = np.eye(4)
    out[:3, :3] = a[:3, :3].T @ b[:3, :3]
    out[:3, 3] = a[:3, :3].T @ (b[:3, 3] - a[:3, 3])
    return out


def odometry(poses: np.ndarray, sigma_deg: float = ODOMETRY_DEG, sigma_m: float = ODOMETRY_M) -> list[Edge]:
    """An edge between each pair of consecutive poses, measuring what the poses already say."""
    return [Edge(k, k + 1, relative(poses[k], poses[k + 1]), sigma_deg, sigma_m) for k in range(len(poses) - 1)]


def world_up(poses: np.ndarray) -> np.ndarray:
    """The walk's up: the mean of the cameras' up axes (as `cameras.up_direction` does with poses and no sizes)."""
    up = np.einsum("nij,j->ni", poses[:, :3, :3], CAMERA_UP).mean(axis=0)
    norm = float(np.linalg.norm(up))
    return up / norm if norm > 1e-9 else np.array(CAMERA_UP)


def solve(
    poses: np.ndarray,
    edges: Sequence[Edge],
    up: np.ndarray | Sequence[float] | None = None,
    gravity: bool = True,
    height: bool = HEIGHT_PRIOR,
    iterations: int = ITERATIONS,
) -> Solution:
    """`poses` (n x 4 x 4 camera-to-world) moved to fit `edges`, and when asked the gravity and height priors about
    `up` (by default `world_up(poses)`). The first pose doesn't move. Loop edges the solve still can't fit within
    `REJECT` are dropped, worst first, each followed by a fresh solve from `poses`."""
    poses = np.array(poses, dtype=np.float64)
    g = world_up(poses) if up is None else np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    level = float(np.median(poses[:, :3, 3] @ g))
    every = _Graph(edges, g, level, gravity, height)
    kept = list(range(len(edges)))
    dropped: list[int] = []
    while True:
        graph = _Graph([edges[k] for k in kept], g, level, gravity, height)
        out, done, cost = _minimize(graph, poses, iterations)
        white = every.whitened(out)
        far = [k for k in kept if edges[k].loop and white[k] > REJECT]
        if not far:
            break
        worst = max(far, key=lambda k: white[k])
        kept.remove(worst)
        dropped.append(worst)
    deg, m = every.errors(out)
    return Solution(out, done, cost, deg, m, tuple(sorted(dropped)))


def _minimize(graph: _Graph, poses: np.ndarray, iterations: int) -> tuple[np.ndarray, int, float]:
    """Levenberg-damped Gauss-Newton from `poses`: the poses it ends at, the iterations it took, and the cost."""
    n = len(poses)
    if n < 2:
        return poses, 0, graph.cost(poses)
    damping = 1e-6
    cost = graph.cost(poses)
    done = 0
    for done in range(1, iterations + 1):
        hessian, gradient = graph.normal(poses, n)
        scale = hessian.diagonal().mean() or 1.0
        while True:
            lhs = hessian + sparse.identity(hessian.shape[0], format="csc") * (damping * scale)
            step = -np.asarray(spsolve(lhs.tocsc(), gradient)).reshape(n - 1, 6)
            trial = _retract(poses, step)
            new = graph.cost(trial)
            if new <= cost or damping > 1e6:
                break
            damping *= 10.0
        if new > cost:
            break
        poses, cost = trial, new
        damping = max(damping / 10.0, 1e-9)
        if float(np.abs(step).max()) < STEP_TOL:
            break
    return poses, done, cost


def loop_scale(odometry_m: Sequence[float], pair_m: Sequence[float]) -> float:
    """The factor that puts MASt3R's lengths into the odometry's scale: the median ratio of lengths the odometry
    gives to the lengths MASt3R gives for the same things (the walkthrough uses each frame's median depth; a first
    try with short-baseline translations spread 8.7x on IMG_7676). Raises ValueError when those ratios spread
    more than `SCALE_SPREAD` (10th to 90th percentile), since loops closed in a scale nobody believes would bend the
    walk."""
    odo, pair = np.asarray(odometry_m, dtype=np.float64), np.asarray(pair_m, dtype=np.float64)
    good = (odo > 1e-6) & (pair > 1e-6)
    if not good.any():
        raise ValueError("no lengths to compare, so there's no scale between the two models")
    ratios = odo[good] / pair[good]
    lo, hi = np.percentile(ratios, [10, 90])
    if hi > SCALE_SPREAD * lo:
        raise ValueError(f"the scales spread {hi / lo:.1f}x (from {lo:.3g} to {hi:.3g})")
    return float(np.median(ratios))


def loop_refusal(
    forward: np.ndarray,
    backward: np.ndarray,
    matches: int,
    guess: np.ndarray,
    scale: float,
    gate_m: float = LOOP_GATE_M,
    gate_deg: float = LOOP_GATE_DEG,
) -> str | None:
    """Why a MASt3R pair isn't a loop edge, or None when it is one. `forward` is camera j in camera i's frame (from
    the pair i, j), `backward` camera i in camera j's frame (from the pair j, i), both in MASt3R's scale; `guess` is
    j in i's frame by the drifted odometry, and `scale` is `loop_scale`'s factor. `gate_m` and `gate_deg` are how far
    the pair may disagree with the guess (drift grows with the walk between the two, so a caller may widen them)."""
    if matches < MIN_MATCHES:
        return f"{matches} matches (needs {MIN_MATCHES})"
    back = np.linalg.inv(backward)
    rot = angle_deg(forward[:3, :3].T @ back[:3, :3])
    if rot > AGREE_DEG:
        return f"the two directions' rotations disagree by {rot:.0f} degrees"
    direction = _between_deg(forward[:3, 3], back[:3, 3])
    if direction > AGREE_DIR_DEG:
        return f"the two directions' translations point {direction:.0f} degrees apart"
    tilt = _between_deg(forward[:3, :3] @ CAMERA_UP, CAMERA_UP)
    if tilt > LOOP_UP_DEG:
        return f"the pair tilts the cameras' up axes {tilt:.0f} degrees apart"
    shift = scale * forward[:3, 3]
    if float(np.linalg.norm(shift)) > MAX_BASELINE_M:
        return f"a {np.linalg.norm(shift):.0f} m baseline"
    off_m = float(np.linalg.norm(shift - guess[:3, 3]))
    off_deg = angle_deg(guess[:3, :3].T @ forward[:3, :3])
    if off_m > gate_m or off_deg > gate_deg:
        return f"{off_m:.1f} m and {off_deg:.0f} degrees off the odometry"
    return None


def loop_edge(forward: np.ndarray, i: int, j: int, scale: float) -> Edge:
    """A verified pair as a loop edge, its translation in the odometry's scale."""
    measured = np.array(forward, dtype=np.float64)
    measured[:3, 3] *= scale
    return Edge(i, j, measured, LOOP_DEG, LOOP_M, loop=True)


def up_axis_error(c2w: np.ndarray, world_points: np.ndarray) -> float:
    """Degrees between the camera's -y as `c2w` puts it in the world, and the image's up in the world as the frame's
    own world pointmap (rows x columns x 3, row 0 at the top) shows it: from its bottom rows' centroid to its top
    rows', less the part along the camera's view. Near 0 when the pose is camera-to-world and -y is up; near 180
    when y is up; anything else when the pose is the other way round."""
    pts = np.asarray(world_points, dtype=np.float64)
    band = max(1, len(pts) // 4)
    image_up = pts[:band].reshape(-1, 3).mean(axis=0) - pts[-band:].reshape(-1, 3).mean(axis=0)
    forward = c2w[:3, 2] / np.linalg.norm(c2w[:3, 2])
    image_up = image_up - (image_up @ forward) * forward
    return _between_deg(c2w[:3, :3] @ CAMERA_UP, image_up)


class _Graph:
    """The residuals, their Jacobians, and the robust cost, over the edges and the priors."""

    def __init__(self, edges: Sequence[Edge], up: np.ndarray, level: float, gravity: bool, height: bool) -> None:
        self.i = np.array([e.i for e in edges], dtype=np.int64)
        self.j = np.array([e.j for e in edges], dtype=np.int64)
        self.z = np.array([e.measured for e in edges], dtype=np.float64).reshape(-1, 4, 4)
        self.sigma = np.array(
            [[math.radians(e.sigma_deg)] * 3 + [e.sigma_m] * 3 for e in edges], dtype=np.float64
        ).reshape(-1, 6)
        self.up = up
        self.level = level
        self.gravity = gravity
        self.height = height
        helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        a = np.cross(up, helper)
        a /= np.linalg.norm(a)
        self.across = np.stack([a, np.cross(up, a)], axis=1)  # 3 x 2: the plane level to up

    def _edges(self, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Each edge's residual (rotation, translation: k x 6), and the pieces its Jacobians are built of."""
        ri, rj = poses[self.i, :3, :3], poses[self.j, :3, :3]
        rz = self.z[:, :3, :3]
        d = np.einsum("kji,kj->ki", ri, poses[self.j, :3, 3] - poses[self.i, :3, 3])  # Ri^T (tj - ti)
        err_r = np.einsum("kji,kjl->kil", rz, np.einsum("kji,kjl->kil", ri, rj))  # Rz^T Ri^T Rj
        rot = Rotation.from_matrix(err_r).as_rotvec() if len(self.i) else np.zeros((0, 3))
        trans = np.einsum("kji,kj->ki", rz, d - self.z[:, :3, 3])
        return np.c_[rot, trans], err_r, d, rz

    def _priors(self, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per pose: the gravity residual (n x 2, the camera up's tilt off up, in radians for small tilts) and the
        height residual (n, metres)."""
        ups = np.einsum("nij,j->ni", poses[:, :3, :3], CAMERA_UP)
        return ups @ self.across, poses[:, :3, 3] @ self.up - self.level

    def _blocks(self, poses: np.ndarray) -> list[np.ndarray]:
        """Every residual block whitened, rows x block size: the edges, then gravity, then height."""
        res = self._edges(poses)[0] / self.sigma
        tilt, rise = self._priors(poses)
        out = [res]
        if self.gravity:
            out.append(tilt / math.sin(math.radians(GRAVITY_DEG)))
        if self.height:
            out.append((rise / HEIGHT_M)[:, None])
        return out

    def cost(self, poses: np.ndarray) -> float:
        return float(sum(_huber(np.linalg.norm(b, axis=1)).sum() for b in self._blocks(poses)))

    def whitened(self, poses: np.ndarray) -> np.ndarray:
        """Each edge's whitened residual norm."""
        return np.linalg.norm(self._edges(poses)[0] / self.sigma, axis=1)

    def errors(self, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        res = self._edges(poses)[0]
        return np.degrees(np.linalg.norm(res[:, :3], axis=1)), np.linalg.norm(res[:, 3:], axis=1)

    def normal(self, poses: np.ndarray, n: int) -> tuple[sparse.csc_matrix, np.ndarray]:
        """The Huber-weighted normal equations (J^T W J, J^T W r) over the free poses (all but the first)."""
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        resid: list[np.ndarray] = []
        start = 0

        def add(jac: np.ndarray, r: np.ndarray, node: np.ndarray, offset: int, w: np.ndarray) -> None:
            """Blocks of rows, one per residual block: `jac` (k x m x 6) on pose `node`'s 6 columns."""
            k, m, _ = jac.shape
            free = node > 0
            if not free.any():
                return
            sw = np.sqrt(w)[free]
            jw = jac[free] * sw[:, None, None]
            row = offset + np.arange(k)[free][:, None, None] * m + np.arange(m)[None, :, None]
            col = (node[free] - 1)[:, None, None] * 6 + np.arange(6)[None, None, :]
            rows.append(np.broadcast_to(row, jw.shape).ravel())
            cols.append(np.broadcast_to(col, jw.shape).ravel())
            vals.append(jw.ravel())

        # Edges: right perturbation of both ends.
        res, err_r, d, rz = self._edges(poses)
        k = len(res)
        if k:
            white = res / self.sigma
            w = _weights(np.linalg.norm(white, axis=1))
            jr_inv = _jr_inv(res[:, :3])
            rj = poses[self.j, :3, :3]
            ri = poses[self.i, :3, :3]
            jj = np.zeros((k, 6, 6))
            jj[:, :3, :3] = jr_inv
            jj[:, 3:, 3:] = np.einsum("kji,kjl->kil", rz, np.einsum("kji,kjl->kil", ri, rj))
            ji = np.zeros((k, 6, 6))
            ji[:, :3, :3] = -np.einsum("kab,kcb,kdc->kad", jr_inv, err_r, rz)  # -Jr^-1 E^T Rz^T
            ji[:, 3:, :3] = np.einsum("kji,kjl->kil", rz, _hat(d))
            ji[:, 3:, 3:] = -np.transpose(rz, (0, 2, 1))
            ji /= self.sigma[:, :, None]
            jj /= self.sigma[:, :, None]
            add(ji, white, self.i, start, w)
            add(jj, white, self.j, start, w)
            resid.append((white * np.sqrt(w)[:, None]).ravel())
            start += k * 6
        tilt, rise = self._priors(poses)
        nodes = np.arange(n)
        rot = poses[:, :3, :3]
        if self.gravity:
            s = math.sin(math.radians(GRAVITY_DEG))
            white = tilt / s
            w = _weights(np.linalg.norm(white, axis=1))
            # u = R c; R exp(w) c ~ u - R [c]x w
            jac = np.zeros((n, 2, 6))
            jac[:, :, :3] = -np.einsum("ia,nib,bc->nac", self.across, rot, _hat(np.array(CAMERA_UP)[None])[0]) / s
            add(jac, white, nodes, start, w)
            resid.append((white * np.sqrt(w)[:, None]).ravel())
            start += n * 2
        if self.height:
            white = rise / HEIGHT_M
            w = _weights(np.abs(white))
            jac = np.zeros((n, 1, 6))
            jac[:, 0, 3:] = np.einsum("i,nij->nj", self.up, rot) / HEIGHT_M  # t + R v
            add(jac, white[:, None], nodes, start, w)
            resid.append(white * np.sqrt(w))
            start += n
        if not vals:  # nothing touches a free pose: no step
            return sparse.csc_matrix((6 * (n - 1), 6 * (n - 1))), np.zeros(6 * (n - 1))
        r = np.concatenate(resid)
        jac_all = sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(start, 6 * (n - 1))
        ).tocsc()
        # A free pose no residual touches would leave the system singular; the damping covers it.
        return (jac_all.T @ jac_all).tocsc(), jac_all.T @ r


def _retract(poses: np.ndarray, step: np.ndarray) -> np.ndarray:
    out = poses.copy()
    rot = poses[1:, :3, :3]
    out[1:, :3, :3] = rot @ Rotation.from_rotvec(step[:, :3]).as_matrix()
    out[1:, :3, 3] = poses[1:, :3, 3] + np.einsum("nij,nj->ni", rot, step[:, 3:])
    return out


def _hat(v: np.ndarray) -> np.ndarray:
    """The cross-product matrix of each row of `v` (k x 3 x 3)."""
    out = np.zeros((len(v), 3, 3))
    out[:, 0, 1], out[:, 0, 2] = -v[:, 2], v[:, 1]
    out[:, 1, 0], out[:, 1, 2] = v[:, 2], -v[:, 0]
    out[:, 2, 0], out[:, 2, 1] = -v[:, 1], v[:, 0]
    return out


def _jr_inv(phi: np.ndarray) -> np.ndarray:
    """The inverse right Jacobian of SO(3) at each rotation vector in `phi` (k x 3 x 3)."""
    theta = np.linalg.norm(phi, axis=1)
    hat = _hat(phi)
    small = theta < 1e-6
    t = np.where(small, 1.0, theta)
    coef = np.where(small, 1.0 / 12.0, 1.0 / t**2 - (1.0 + np.cos(t)) / (2.0 * t * np.sin(np.where(small, 1.0, t))))
    return np.eye(3)[None] + 0.5 * hat + coef[:, None, None] * (hat @ hat)


def _weights(norm: np.ndarray) -> np.ndarray:
    """Huber's weight for a whitened block of this norm (1 inside `HUBER`, falling off as 1/norm past it)."""
    return np.where(norm <= HUBER, 1.0, HUBER / np.maximum(norm, 1e-12))


def _huber(norm: np.ndarray) -> np.ndarray:
    return np.where(norm <= HUBER, 0.5 * norm**2, HUBER * (norm - 0.5 * HUBER))


def angle_deg(rot: np.ndarray) -> float:
    """How far a rotation matrix turns, in degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(rot)) - 1.0) / 2.0))))


def _between_deg(a: np.ndarray | Sequence[float], b: np.ndarray | Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 180.0
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b)))
