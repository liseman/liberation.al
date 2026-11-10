"""Calibrate a fisheye camera's lens from checkerboard views, for see-actions' per-person rectification.

You need a printed checkerboard on something flat and stiff (a board taped to a clipboard or foam board). The
default is 9 x 6 inner corners, the corners where four squares meet, so a board of 10 x 7 squares. Squares of 25 to
40 mm are fine.

1. Capture about 20 views from the camera, headless. Run it on the machine the camera is plugged into, with the
   camera's service stopped so the device is free:

       .venv/bin/python tools/calibrate_fisheye.py capture --device /dev/video0 --out calib/fish0

   Hold the board in front of the lens and move it slowly: near and far, tilted left, right, up, and down, and
   especially out toward the edges and corners of the image, where the lens bends most. A view is saved only when
   the whole board is found and it has moved since the last saved one; it stops after `--views` (20). Keep the
   whole board in the picture each time, and hold it still for a moment.

2. Solve, and write the result on the camera's entry in `config/devices.toml` (its name there, `--camera`):

       .venv/bin/python tools/calibrate_fisheye.py solve calib/fish0 --camera fish0 --write

   It prints the fit's RMS reprojection error in pixels: under 1 is good. Over 1, capture again with more views
   near the rim. Without `--write` it only prints. With it, it sets on that camera's entry:

       fisheye = { width = 1920, height = 1080, k = [fx, fy, cx, cy], d = [k1, k2, k3, k4], rms = 0.4 }

Calibrate at the resolution the camera streams at (1920 x 1080 for the IMX291s). A calibration at another size is
scaled to the frames see-actions gets. Do one per lens: two copies of the same lens model still differ. Restart
see-actions afterwards; its log says each camera is "a fisheye, rectified per person".
"""

import argparse
import glob
import pathlib
import time
from collections.abc import Sequence

import cv2
import numpy as np

from liberational.see.fisheye import Lens

MIN_VIEWS = 8  # fewer than this and cv2.fisheye.calibrate is poorly conditioned
SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


def board_points(board: tuple[int, int]) -> np.ndarray:
    """The board's inner corners on its own plane, in squares (1 x N x 3; the square size doesn't change intrinsics)."""
    cols, rows = board
    grid = np.zeros((1, cols * rows, 3), np.float64)
    grid[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return grid


def find_corners(image: np.ndarray, board: tuple[int, int], *, fast: bool = False) -> np.ndarray | None:
    """The board's inner corners in a BGR or gray image (1 x N x 2, subpixel), or None when the whole board isn't
    found. `fast` only checks (for capture's live preview)."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + (cv2.CALIB_CB_FAST_CHECK if fast else 0)
    found, corners = cv2.findChessboardCorners(gray, board, flags=flags)
    if not found:
        return None
    if not fast:
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX)
    return corners.reshape(1, -1, 2).astype(np.float64)


def _flag(name: str) -> int:
    """A fisheye calibration flag: under `cv2.fisheye` before OpenCV 5, at the top of `cv2` from 5 on."""
    return int(getattr(cv2.fisheye, name, None) or getattr(cv2, name))


def solve(views: Sequence[np.ndarray], board: tuple[int, int], size: tuple[int, int]) -> tuple[Lens, float]:
    """The lens (`size` is width by height) and the fit's RMS reprojection error in pixels, from each view's corners."""
    if len(views) < MIN_VIEWS:
        raise SystemExit(
            f"{len(views)} views with the whole board found; at least {MIN_VIEWS} are needed (20 is better)"
        )
    objects = [board_points(board)] * len(views)
    K, D = np.zeros((3, 3)), np.zeros((4, 1))
    flags = _flag("CALIB_RECOMPUTE_EXTRINSIC") | _flag("CALIB_CHECK_COND") | _flag("CALIB_FIX_SKEW")
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7)
    rms, K, D, _, _ = cv2.fisheye.calibrate(objects, list(views), size, K, D, flags=flags, criteria=criteria)
    k = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    d = tuple(float(x) for x in np.asarray(D).reshape(4))
    return Lens(size[0], size[1], k, (d[0], d[1], d[2], d[3])), float(rms)


def write_calibration(manifest: pathlib.Path, camera: str, lens: Lens, rms: float) -> None:
    """Sets `fisheye = {...}` on `camera`'s entry in the manifest, keeping the rest of the file as it is."""
    import tomlkit

    doc = tomlkit.parse(manifest.read_text())
    entry = next((d for d in doc.get("device", []) if d.get("name") == camera), None)
    if entry is None:
        raise SystemExit(f"no device named {camera!r} in {manifest}")
    table = tomlkit.inline_table()
    table.update(
        {
            "width": lens.width,
            "height": lens.height,
            "k": [round(x, 3) for x in lens.k],
            "d": [round(x, 6) for x in lens.d],
            "rms": round(rms, 3),
        }
    )
    entry["fisheye"] = table
    manifest.write_text(tomlkit.dumps(doc))


def capture(device: str, out: pathlib.Path, board: tuple[int, int], views: int, width: int, height: int) -> None:
    """Saves views of the board from a camera until there are `views`: one only when the whole board is found and
    its corners have moved a tenth of the image since the last saved view, at most one a second."""
    cap = cv2.VideoCapture(int(device) if device.isdigit() else device)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"couldn't open {device} (is the camera's service still using it?)")
    out.mkdir(parents=True, exist_ok=True)
    saved = len(list(out.glob("view-*.jpg")))
    last: np.ndarray | None = None
    last_at = 0.0
    print(f"capturing {views} views from {device} into {out}: move the board slowly, all over the image")
    while saved < views:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"{device} stopped giving frames")
        corners = find_corners(frame, board, fast=True)
        now = time.monotonic()
        if corners is None or now - last_at < 1.0:
            continue
        moved = last is None or float(np.abs(corners - last).mean()) > 0.1 * min(frame.shape[:2])
        if not moved:
            continue
        cv2.imwrite(str(out / f"view-{saved:02d}.jpg"), frame)
        saved, last, last_at = saved + 1, corners, now
        print(f"  view {saved} of {views} ({frame.shape[1]} x {frame.shape[0]})")
    cap.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", default="9x6", help="inner corners, columns x rows (default 9x6)")
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture", help="save checkerboard views from a camera")
    cap.add_argument("--device", required=True, help="a V4L2 device (/dev/video0) or an OpenCV camera index")
    cap.add_argument("--out", type=pathlib.Path, required=True)
    cap.add_argument("--views", type=int, default=20)
    cap.add_argument("--width", type=int, default=1920)
    cap.add_argument("--height", type=int, default=1080)
    solve_ = sub.add_parser("solve", help="calibrate from saved views and print (or --write) the lens")
    solve_.add_argument("images", help="a folder of views, or a glob of images")
    solve_.add_argument("--camera", help="the camera's name in the manifest (needed with --write)")
    solve_.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("config/devices.toml"))
    solve_.add_argument("--write", action="store_true", help="set the camera's fisheye calibration in the manifest")
    args = parser.parse_args(argv)
    cols, rows = (int(x) for x in args.board.lower().split("x"))
    board = (cols, rows)

    if args.command == "capture":
        capture(args.device, args.out, board, args.views, args.width, args.height)
        return 0
    pattern = str(pathlib.Path(args.images) / "*.jpg") if pathlib.Path(args.images).is_dir() else args.images
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no images match {pattern}")
    views, size = [], None
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"  {path}: not an image, skipped")
            continue
        here = (image.shape[1], image.shape[0])
        if size is not None and here != size:
            raise SystemExit(
                f"{path} is {here[0]} x {here[1]}, the others {size[0]} x {size[1]}: one size per calibration"
            )
        size = here
        corners = find_corners(image, board)
        print(f"  {path}: {'board found' if corners is not None else 'no whole board, skipped'}")
        if corners is not None:
            views.append(corners)
    assert size is not None
    lens, rms = solve(views, board, size)
    print(
        f"{len(views)} views at {size[0]} x {size[1]}: RMS {rms:.3f} px ({'good' if rms < 1 else 'capture more views near the rim'})"
    )
    print(f"k = {list(lens.k)}\nd = {list(lens.d)}")
    if args.write:
        if not args.camera:
            raise SystemExit("--write needs --camera, the camera's name in the manifest")
        write_calibration(args.manifest, args.camera, lens, rms)
        print(f"wrote {args.camera}'s fisheye calibration to {args.manifest}; restart see-actions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
