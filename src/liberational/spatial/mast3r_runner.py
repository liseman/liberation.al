"""MASt3R over the fixed cameras' frames, run in MASt3R's own runtime. spatial/cameras.py runs it; it is not
imported by anything that runs in the pack's environment, apart from the pure `groups` below.

    <runtime>/.venv/bin/python mast3r_runner.py job.json

MASt3R (naver/mast3r, pinned by `mise run mast3r`) lives in its own venv because it needs an older Python and torch
than the pack's, and its upstream code is kept out of the pack's processes. So this file imports nothing from the pack
and only the standard library at module level; numpy, torch, and MASt3R are imported inside `main`.

The job (JSON): `repo` (the pinned MASt3R checkout, with its dust3r submodule), `checkpoint` (the directory holding
the `model.safetensors` and `config.json` it ships, loaded with safetensors, never a pickle), `images` ({camera: image
path}), `min_matches`, `size` (the long side MASt3R sees, 512), `out` (a directory for the answer and a cache), and
optionally `window` or `offsets`: match only images at most `window` apart in name order, or exactly one of
`offsets` apart (a video's keyframes, named in time order: `pair_edges`), else every pair.

What it does:
1. **Overlap.** Every pair of cameras through MASt3R, both ways, counting its reciprocal matches. The match count is
   the overlap test: in the initial spike, two views of one room matched 3,528 times, a weak tie about 600, and views
   of different spaces at most 187.
2. **Groups.** Cameras joined by pairs with at least `min_matches` matches form a group (`groups`); each group of
   two or more is one place. With `lone_matches` in the job, a camera that sees no other camera's space (no pair
   reaches that bar, `lone`) is a group of its own; one with a weaker tie than `min_matches` but at least
   `lone_matches` (a camera that shares only a sliver of view) is in no group.
3. **Alignment.** Each group through MASt3R's sparse global alignment over its overlapping pairs only (the pair
   results are cached from step 1), for a joint, roughly metric point cloud and each camera's pose. A lone camera's
   group is its frame paired with itself (DUSt3R's monocular mode): its point map in its own frame, its focal
   estimated from it, and the identity pose. In the prototype its depth was within 11.3% (median) of Place 1's
   fused map, against 10.0% for a two-view pair. A lone camera can be pitched far down, so its group also carries
   `up` (`up_from_normals`): the cloud's up, from its surfaces, in the camera's frame, and its `floor`
   (`floor_from_normals`: the lowest layer of horizontal surfaces). A lone camera whose view shows no floor (a
   blocked lens: the kitchen camera's on the Linux machine) is no place; it's listed in the answer's `floorless`.

The answer is `out/result.json`: {"matches": [[a, b, n], ...], "groups": [{"cameras": [...], "cloud": "<npz>",
"poses": {camera: 16 numbers, camera to world, row by row}, "focals": {camera: px}, "sizes": {camera: [w, h]}, and
for a lone camera "up": [x, y, z] and "floor": [nx, ny, nz, d]}], "floorless": [camera, ...]} (the last with
`lone_matches` only),
and each group's cloud an npz of `points` (N x 3 float32, metres) and `colors` (N x 3 uint8).

**Pairs mode**: with `pairs` ([[a, b], ...], image names) in the job, only those
pairs go through MASt3R, nothing is grouped or aligned, and the answer is {"pairs": [{"a", "b", "matches",
"depths", "forward", "backward", "scales", "inliers"}, ...]}. `depths` are a's and b's median depths over their
confident pixels, both in the scale `forward` is in (b's own pointmap comes from the other pass, so its depth is
carried over by the forward similarity's scale), which is how a caller relates the pair to its own scale. MASt3R gives each pair four pointmaps: each view in its own
frame, and each view in the other's frame. `forward` is camera b's pose in camera a's frame (16 numbers, row by row),
the rigid part of the similarity that carries b's own pointmap onto b's pointmap in a's frame (`pair_pose`), and
`backward` is a in b's frame the same way; with `save_pairs` in the job, each pair's inferences also go to
`pairs.npz` as a group's do (`_pair_arrays`), named in the answer's `pair_file`; `scales` are those similarities' scales (near 1 when the two heads agree).
A pair MASt3R is unsure of nearly everywhere has no poses (None) and an `error`. `inliers` count the
reciprocal matches PnP with RANSAC keeps for each direction (`pnp_inliers`: a's points at its
matched pixels against b's matched pixels, with b's focal from its own pointmap), a check that the matches and the
geometry agree. The translations are in MASt3R's scale; a caller puts them in its own.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: Confidence a point needs to be kept in a group's cloud (MASt3R's confidences start at 1).
MIN_CONF = 1.5
#: A pixel takes part in a pair's pose (`pair_pose`) when both its pointmaps are at least this confident.
POSE_CONF = 1.5
#: PnP's RANSAC: iterations, and how far (pixels at MASt3R's input size) a match may land from its pixel to count.
PNP_ITERS, PNP_PX = 200, 5.0
#: Sparse global alignment's two stages, as MASt3R's demo runs them.
LR1, NITER1, LR2, NITER2 = 0.07, 300, 0.014, 300


def groups(cameras: Sequence[str], matches: Iterable[tuple[str, str, int]], min_matches: int) -> list[list[str]]:
    """The cameras joined by pairs with at least `min_matches` matches, as connected groups of two or more, each
    sorted, largest first (then by name)."""
    parent = {c: c for c in cameras}

    def find(c: str) -> str:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for a, b, n in matches:
        if n >= min_matches and a in parent and b in parent and a != b:
            parent[find(a)] = find(b)
    found: dict[str, list[str]] = {}
    for c in cameras:
        found.setdefault(find(c), []).append(c)
    out = [sorted(g) for g in found.values() if len(g) >= 2]
    return sorted(out, key=lambda g: (-len(g), g))


def lone(cameras: Sequence[str], matches: Iterable[tuple[str, str, int]], bar: int) -> list[str]:
    """The cameras whose every pair has fewer than `bar` matches: those that see no other camera's space, in the
    order given. A camera in no pair at all (the only one) is lone too."""
    best = dict.fromkeys(cameras, 0)
    for a, b, n in matches:
        for c in (a, b):
            if c in best:
                best[c] = max(best[c], int(n))
    return [c for c in cameras if best[c] < bar]


#: The surface normals near the camera's up (its -y) that up is taken from, narrowing in steps (degrees).
UP_CONES = (45.0, 30.0, 15.0, 10.0)


#: A lone camera's floor (`floor_from_normals`): horizontal surfaces are pixels whose normal is this close to up
#: (degrees); their heights are binned this fine (metres), the floor is the lowest bin holding this share of them, and
#: the plane is fitted to the pixels this close to it.
FLAT_DEG, FLOOR_BIN, FLOOR_SHARE, FLOOR_BAND = 15.0, 0.04, 0.05, 0.06


def _normals(points: Any, conf: Any, min_conf: float) -> tuple[Any, Any]:
    """Each confident inner pixel's 3D point and unit surface normal (from its neighbours across and down), the
    normal turned to the camera's -y side: (points N x 3, normals N x 3)."""
    import numpy as np

    p = np.asarray(points, dtype=np.float64)
    dx = p[1:-1, 2:] - p[1:-1, :-2]
    dy = p[2:, 1:-1] - p[:-2, 1:-1]
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=-1)
    ok = (np.asarray(conf)[1:-1, 1:-1] > min_conf) & (length > 1e-12)
    n = n[ok] / length[ok][:, None]
    n *= np.where(n[:, 1:2] > 0, -1.0, 1.0)
    return p[1:-1, 1:-1][ok], n


def floor_from_normals(
    points: Any, conf: Any, up: Sequence[float], min_conf: float = MIN_CONF
) -> tuple[float, float, float, float] | None:
    """A single camera's floor (nx, ny, nz, d: a point's height above it is n . p + d), from its point map's
    horizontal surfaces: the lowest layer of them along `up` that holds FLOOR_SHARE of them, a plane fitted to it.
    Pitched down at a room, one camera can see more table, dresser, and bed top than floor (a bedroom camera's
    floor strip was 1.65 m under it, its furniture tops 1.13 and 0.75 m, and a RANSAC floor fit took a top), while
    the floor is still the lowest flat layer. None when too few horizontal surfaces are seen."""
    import numpy as np

    pts, n = _normals(points, conf, min_conf)
    upv = np.asarray(up, dtype=np.float64)
    upv = upv / np.linalg.norm(upv)
    flat = pts[n @ upv > np.cos(np.radians(FLAT_DEG))]
    if len(flat) < 50:
        return None
    h = flat @ upv
    edges = np.arange(h.min(), h.max() + FLOOR_BIN, FLOOR_BIN)
    if len(edges) < 2:
        edges = np.array([h.min(), h.min() + FLOOR_BIN])
    counts, edges = np.histogram(h, edges)
    enough = np.nonzero(counts >= max(50, FLOOR_SHARE * len(flat)))[0]
    if not len(enough):
        return None
    level = (edges[enough[0]] + edges[enough[0] + 1]) / 2
    layer = flat[np.abs(h - level) < FLOOR_BAND]
    centre = layer.mean(axis=0)
    normal = np.linalg.svd(layer - centre, full_matrices=False)[2][-1]
    normal = normal if normal @ upv > 0 else -normal
    x, y, z = normal.tolist()
    return (x, y, z, -float(normal @ centre))


def up_from_normals(points: Any, conf: Any, min_conf: float = MIN_CONF) -> tuple[float, float, float]:
    """A single camera's up in its own frame (OpenCV: y down), from its point map's surfaces (`points`, H x W x 3,
    and `conf`, H x W): each confident pixel's normal from its neighbours, then the mean of the normals near the
    camera's -y, in narrowing cones, so the floor and table tops decide it and the walls drop out. A camera's own -y
    is up only when it hangs level, and one camera alone may be pitched far down. Its -y when nothing is near it."""
    import numpy as np

    _, n = _normals(points, conf, min_conf)
    up = np.array([0.0, -1.0, 0.0])
    for cone in UP_CONES:
        near = n[n @ up > np.cos(np.radians(cone))]
        if len(near) == 0:
            break
        mean = near.mean(axis=0)
        up = mean / np.linalg.norm(mean)
    x, y, z = up.tolist()
    return (x, y, z)


def pair_edges(n: int, window: int | None = None, offsets: Iterable[int] | None = None) -> list[tuple[int, int]]:
    """The index pairs (i, j), i < j, of `n` images in order to match: those `offsets` apart when given, else those
    at most `window` apart, else every pair. A video's offsets are its neighbours and then doubling steps (1, 2,
    3, 4, 8, 16, ...), the log-window scene graph DUSt3R and MASt3R use for video: the neighbours hold the chain
    together, and the long steps tie distant parts of it to each other (where they overlap enough to match), so
    alignment errors don't pile up along it."""
    if offsets is not None:
        steps = sorted({int(k) for k in offsets if int(k) > 0})
        return [(i, i + k) for i in range(n) for k in steps if i + k < n]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if window is None or j - i <= int(window)]


def similarity(src: Any, dst: Any) -> tuple[Any, Any, float]:
    """The rotation R (3 x 3), translation t (3), and scale s that carry the points `src` onto `dst` (both N x 3,
    row for row) best in least squares, dst ~ s R src + t (Umeyama), with R a proper rotation."""
    import numpy as np

    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    flip = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)) or 1.0)])
    rot = u @ flip @ vt
    var = float((a**2).sum() / len(src))
    scale = float(np.trace(np.diag(sig) @ flip) / var) if var > 0 else 1.0
    return rot, mu_d - scale * rot @ mu_s, scale


def pair_pose(own: Any, other: Any, conf_own: Any, conf_other: Any, min_conf: float = POSE_CONF) -> tuple[Any, float]:
    """A camera's pose (4 x 4, camera to the other camera's frame) from its pointmap in its own frame (`own`, H x W x
    3) and in the other camera's frame (`other`), over the pixels both are confident in, and the scale between them.
    The camera's centre is its own frame's origin, so the similarity's translation is where it sits in the other's."""
    import numpy as np

    keep = (np.asarray(conf_own) > min_conf) & (np.asarray(conf_other) > min_conf)
    if keep.sum() < 3:
        raise ValueError("too few confident pixels for a pose")
    rot, shift, scale = similarity(np.asarray(own)[keep], np.asarray(other)[keep])
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rot, shift
    return pose, scale


def pnp_inliers(points: Any, pixels: Any, focal: float, centre: tuple[float, float]) -> int:
    """How many of the matched 3D `points` (N x 3, in one camera's frame) PnP with RANSAC finds consistent with
    their `pixels` (N x 2, x and y) in the other camera (pinhole, `focal` and principal point `centre`); 0 when it
    fails."""
    import cv2
    import numpy as np

    pts, px = np.asarray(points, dtype=np.float64), np.asarray(pixels, dtype=np.float64)
    if len(pts) < 6:
        return 0
    k = np.array([[focal, 0.0, centre[0]], [0.0, focal, centre[1]], [0.0, 0.0, 1.0]])
    ok, _, _, inliers = cv2.solvePnPRansac(
        pts, px, k, None, iterationsCount=PNP_ITERS, reprojectionError=PNP_PX, flags=cv2.SOLVEPNP_SQPNP
    )
    return int(len(inliers)) if ok and inliers is not None else 0


def _pairs(
    images: list[dict[str, Any]], edges: Iterable[tuple[int, int]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """dust3r image pairs over `edges` (index pairs into `images`), both ways round."""
    return [p for i, j in edges for p in ((images[i], images[j]), (images[j], images[i]))]


def main(argv: Sequence[str]) -> int:
    job: Mapping[str, Any] = json.loads(Path(argv[0]).read_text())
    repo = Path(job["repo"])
    sys.path[:0] = [str(repo), str(repo / "dust3r")]
    import numpy as np
    import torch
    from dust3r.utils.image import load_images  # ty: ignore[import-not-found] (MASt3R's runtime only)
    from mast3r.cloud_opt import sparse_ga  # ty: ignore[import-not-found] (MASt3R's runtime only)
    from mast3r.model import AsymmetricMASt3R  # ty: ignore[import-not-found] (MASt3R's runtime only)

    out = Path(job["out"])
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cams = sorted(job["images"])
    paths = [str(job["images"][c]) for c in cams]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    began = time.monotonic()
    # from_pretrained on a local directory reads its model.safetensors: no pickle is loaded.
    model = AsymmetricMASt3R.from_pretrained(str(job["checkpoint"])).to(device).eval()
    size = int(job.get("size", 512))
    imgs = load_images(paths, size=size, verbose=False)
    if job.get("pairs") is not None:
        result = _pair_poses(job, model, imgs, cams, paths, cache, device)
        result["seconds"] = round(time.monotonic() - began, 1)
        (out / "result.json").write_text(json.dumps(result))
        return 0
    every = pair_edges(len(cams), job.get("window"), job.get("offsets"))
    found: dict[Any, Any] = {}
    if every:  # a single camera has no pairs
        pairs = sparse_ga.convert_dust3r_pairs_naming(paths, _pairs(imgs, every))
        found, _ = sparse_ga.forward_mast3r(pairs, model, str(cache), device=device, subsample=8)
    matches = []
    for i, j in every:
        _, corres = found[paths[i], paths[j]]
        _, (_, _, conf) = torch.load(corres, weights_only=True)  # the cache this run just wrote: tensors and floats
        matches.append((cams[i], cams[j], len(conf)))
    result: dict[str, Any] = {"matches": [list(m) for m in matches], "groups": []}
    for k, group in enumerate(groups(cams, matches, int(job["min_matches"]))):
        sub = [paths[cams.index(c)] for c in group]
        sub_imgs = load_images(sub, size=size, verbose=False)
        overlapping = {(a, b) for a, b, n in matches if n >= int(job["min_matches"])}
        edges = [
            (i, j)
            for i in range(len(group))
            for j in range(i + 1, len(group))
            if (group[i], group[j]) in overlapping or (group[j], group[i]) in overlapping
        ]
        scene = sparse_ga.sparse_global_alignment(
            sub,
            _pairs(sub_imgs, edges),
            str(cache),
            model,
            lr1=LR1,
            niter1=NITER1,
            lr2=LR2,
            niter2=NITER2,
            device=device,
            opt_depth=True,
            shared_intrinsics=False,
            matching_conf_thr=0.0,
        )
        poses = scene.get_im_poses().detach().cpu().numpy()
        focals = scene.get_focals().detach().cpu().numpy().ravel()
        pts3d, _, confs = scene.get_dense_pts3d(clean_depth=True)
        points, colors = [], []
        for i in range(len(group)):
            keep = confs[i].detach().cpu().numpy().ravel() > MIN_CONF
            covered = overlay(str(job["images"][group[i]]), *(int(x) for x in sub_imgs[i]["true_shape"][0]), size)
            if covered is not None:
                keep &= ~covered.ravel()
            points.append(pts3d[i].detach().cpu().numpy().reshape(-1, 3)[keep])
            colors.append((np.asarray(scene.imgs[i]).reshape(-1, 3)[keep] * 255).astype(np.uint8))
        cloud = out / f"group{k}.npz"
        np.savez(cloud, points=np.concatenate(points).astype(np.float32), colors=np.concatenate(colors))
        pair_file = out / f"group{k}-pairs.npz"
        np.savez(pair_file, **_pair_arrays(group, [paths[cams.index(c)] for c in group], found, scene.imgs, torch))
        result["groups"].append(
            {
                "cameras": group,
                "cloud": cloud.name,
                "pairs": pair_file.name,
                "poses": {c: [float(x) for x in poses[i].ravel()] for i, c in enumerate(group)},
                "focals": {c: float(focals[i]) for i, c in enumerate(group)},
                "sizes": {c: [int(s) for s in sub_imgs[i]["true_shape"][0][::-1]] for i, c in enumerate(group)},
            }
        )
    if job.get("lone_matches") is not None:
        result["floorless"] = []
        for cam in lone(cams, matches, int(job["lone_matches"])):
            alone = _alone(cam, str(job["images"][cam]), model, size, out, device)
            if alone["floor"] is None:  # a view with no floor in it (blocked, or all wall and ceiling) makes no place
                result["floorless"].append(cam)
            else:
                result["groups"].append(alone)
    result["seconds"] = round(time.monotonic() - began, 1)
    (out / "result.json").write_text(json.dumps(result))
    return 0


def _pair_arrays(
    group: Sequence[str], paths: Sequence[str], found: Mapping[Any, Any], imgs: Any, torch: Any
) -> dict[str, Any]:
    """A group's pair inferences for registration by reprojection, as plain arrays:
    for each ordered pair a, b of its cameras, `a__b__own` (a's point map in its own frame) and `__own_conf`,
    `__other` (b's point map in a's frame) and `__other_conf`, `__xy`, `__xy_other` (their matched pixels) and
    `__match_conf`; and each camera's image at MASt3R's size, `a__image`."""
    import numpy as np

    out: dict[str, Any] = {}
    for i, a in enumerate(group):
        out[f"{a}__image"] = (np.asarray(imgs[i]) * 255).astype(np.uint8)
        for j, b in enumerate(group):
            if i >= j:
                continue
            flip = (paths[i], paths[j]) not in found
            got = found.get((paths[j], paths[i]) if flip else (paths[i], paths[j]))
            if got is None:
                continue
            (first, second), corres = got
            _, (xy1, xy2, conf) = torch.load(corres, weights_only=True)
            x11, c11, x21, c21 = torch.load(first, weights_only=True)
            x22, c22, x12, c12 = torch.load(second, weights_only=True)
            # `first` is from the key's first image's side
            sides = ((a, b, x11, c11, x21, c21, xy1, xy2), (b, a, x22, c22, x12, c12, xy2, xy1))
            if flip:
                sides = ((a, b, x22, c22, x12, c12, xy2, xy1), (b, a, x11, c11, x21, c21, xy1, xy2))
            for ref, cam, own, own_conf, other, other_conf, xy, xy_other in sides:
                key = f"{ref}__{cam}__"
                out[key + "own"] = own.cpu().numpy().astype(np.float32)
                out[key + "own_conf"] = own_conf.cpu().numpy().astype(np.float32)
                out[key + "other"] = other.cpu().numpy().astype(np.float32)
                out[key + "other_conf"] = other_conf.cpu().numpy().astype(np.float32)
                out[key + "xy"] = xy.cpu().numpy().astype(np.int32)
                out[key + "xy_other"] = xy_other.cpu().numpy().astype(np.int32)
                out[key + "match_conf"] = conf.cpu().numpy().astype(np.float32)
    return out


def overlay(image: str, height: int, width: int, size: int) -> Any:
    """The pixels of `image` its camera burns overlays into, at MASt3R's input (`height` x `width`, bool), from
    the `<name>.mask.png` beside it that spatial/cameras.py writes; None without one. The mask is resized and
    centre-cropped as dust3r's `load_images` does the image: its long side to `size`, then a crop to whole patches."""
    import numpy as np
    from PIL import Image

    path = Path(image).with_suffix(".mask.png")
    if not path.exists():
        return None
    m = Image.open(path).convert("L")
    s = max(m.size)
    m = m.resize((round(m.size[0] * size / s), round(m.size[1] * size / s)), Image.Resampling.NEAREST)
    cx, cy = m.size[0] // 2, m.size[1] // 2
    m = m.crop((cx - width // 2, cy - height // 2, cx + width - width // 2, cy + height - height // 2))
    return np.asarray(m) > 127


def _alone(cam: str, path: str, model: Any, size: int, out: Path, device: str) -> dict[str, Any]:
    """A lone camera's group: its frame paired with itself (see the module docstring)."""
    import numpy as np
    import torch
    from dust3r.inference import inference  # ty: ignore[import-not-found] (MASt3R's runtime only)
    from dust3r.post_process import estimate_focal_knowing_depth  # ty: ignore[import-not-found] (MASt3R's runtime)
    from dust3r.utils.image import load_images  # ty: ignore[import-not-found] (MASt3R's runtime only)

    imgs = load_images([path, path], size=size, verbose=False)
    imgs[1]["idx"], imgs[1]["instance"] = 1, "1"
    pred = inference([(imgs[0], imgs[1])], model, device, batch_size=1, verbose=False)["pred1"]
    pts, conf = pred["pts3d"][0], pred["conf"][0]
    h, w = pts.shape[:2]
    centre = torch.tensor([[w / 2, h / 2]], device=pts.device)
    focal = float(estimate_focal_knowing_depth(pts[None], centre, focal_mode="weiszfeld")[0])
    points, confs = pts.detach().cpu().numpy(), conf.detach().cpu().numpy()
    covered = overlay(path, h, w, size)
    if covered is not None:
        confs = np.where(covered, 0.0, confs)  # an overlay is neither cloud nor floor
    keep = confs > MIN_CONF
    rgb = ((imgs[0]["img"][0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    up = up_from_normals(points, confs)
    floor = floor_from_normals(points, confs, up)
    cloud = out / f"alone-{cam}.npz"
    np.savez(cloud, points=points[keep].astype(np.float32), colors=rgb[keep])
    return {
        "cameras": [cam],
        "cloud": cloud.name,
        "poses": {cam: [float(i == j) for i in range(4) for j in range(4)]},
        "focals": {cam: focal},
        "sizes": {cam: [int(w), int(h)]},
        "up": list(up),
        "floor": None if floor is None else list(floor),
    }


def _pair_poses(
    job: Mapping[str, Any],
    model: Any,
    imgs: list[dict[str, Any]],
    cams: list[str],
    paths: list[str],
    cache: Path,
    device: str,
) -> dict[str, Any]:
    """Pairs mode (see the module docstring)."""
    import numpy as np
    import torch
    from dust3r.post_process import estimate_focal_knowing_depth  # ty: ignore[import-not-found] (MASt3R's runtime)
    from mast3r.cloud_opt import sparse_ga  # ty: ignore[import-not-found] (MASt3R's runtime only)

    def depth(pts: Any, conf: Any) -> float:
        """A view's median depth over its confident pixels, in the pair's scale."""
        keep = conf > POSE_CONF
        return float(np.median(pts[..., 2][keep])) if keep.any() else 0.0

    index = {c: k for k, c in enumerate(cams)}
    wanted = [(index[a], index[b]) for a, b in job["pairs"]]
    pairs = sparse_ga.convert_dust3r_pairs_naming(paths, [(imgs[i], imgs[j]) for i, j in wanted])
    found, _ = sparse_ga.forward_mast3r(pairs, model, str(cache), device=device, subsample=8)
    out = []
    for i, j in wanted:
        (path1, path2), corres = found[paths[i], paths[j]]
        # The cache this run just wrote: tensors and floats.
        x11, c11, x21, c21 = (t.numpy() for t in torch.load(path1, map_location="cpu", weights_only=True))
        x22, c22, x12, c12 = (t.numpy() for t in torch.load(path2, map_location="cpu", weights_only=True))
        _, (xy1, xy2, conf) = torch.load(corres, map_location="cpu", weights_only=True)
        xy1, xy2 = np.asarray(xy1), np.asarray(xy2)
        try:
            forward, s_fwd = pair_pose(x22, x21, c22, c21)
            backward, s_back = pair_pose(x11, x12, c11, c12)
        except ValueError as e:  # a pair MASt3R is unsure of everywhere (a blank wall, a blur): no pose, no edge
            out.append({"a": cams[i], "b": cams[j], "matches": len(conf), "forward": None, "backward": None})
            out[-1]["error"] = str(e)
            continue

        def focal(pts: Any) -> float:
            h, w = pts.shape[:2]
            pp = torch.tensor([[w / 2, h / 2]], dtype=torch.float32)
            return float(estimate_focal_knowing_depth(torch.from_numpy(pts)[None], pp, focal_mode="weiszfeld"))

        h2, w2 = x22.shape[:2]
        h1, w1 = x11.shape[:2]
        n_fwd = pnp_inliers(x11[xy1[:, 1], xy1[:, 0]], xy2, focal(x22), (w2 / 2, h2 / 2))
        n_back = pnp_inliers(x22[xy2[:, 1], xy2[:, 0]], xy1, focal(x11), (w1 / 2, h1 / 2))
        out.append(
            {
                "a": cams[i],
                "b": cams[j],
                "matches": len(conf),
                "depths": [depth(x11, c11), depth(x22, c22) * s_fwd],  # both in the first pass's scale, as forward
                "forward": [float(x) for x in forward.ravel()],
                "backward": [float(x) for x in backward.ravel()],
                "scales": [s_fwd, s_back],
                "inliers": [n_fwd, n_back],
            }
        )
    answer: dict[str, Any] = {"pairs": out}
    if job.get("save_pairs"):
        # Each pair's inference as registration by reprojection reads it (`_pair_arrays`)
        arrays: dict[str, Any] = {}
        for i, j in wanted:
            pair_imgs = [((im["img"][0].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1) for im in (imgs[i], imgs[j])]
            arrays.update(_pair_arrays([cams[i], cams[j]], [paths[i], paths[j]], found, pair_imgs, torch))
        pair_file = Path(job["out"]) / "pairs.npz"
        np.savez(pair_file, **arrays)
        answer["pair_file"] = pair_file.name
    return answer


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
