"""A 3D point for every pixel of one picture, from MASt3R-SLAM's single-frame step, run in its own runtime.
spatial/camera_clouds.py runs it; nothing that runs in the pack's environment imports it.

    <runtime>/venv/bin/python pointmap_runner.py job.json

MASt3R-SLAM (rmurai0610/MASt3R-SLAM, CC BY-NC-SA 4.0, on MASt3R) lives in a folder of its own because it needs its
own torch, a compiled solver, and upstream code the pack's processes stay clear of. The pack downloads none of it:
`spatial.pointmap_runtime` names a folder someone set up, with the checkout in `src/` and its environment in `venv/`.
So this file imports nothing from the pack and only the standard library at module level; numpy, torch, and
MASt3R-SLAM are imported inside `main`.

The job (JSON): `repo` (the MASt3R-SLAM checkout), `weights` (a directory holding MASt3R's `model.safetensors` and
`config.json`, loaded with safetensors, never a pickle), `out` (a file for the report), and `views`
([{"image": path, "mask": path or null, "out": path}]). Each view is one picture, alone: this is what MASt3R-SLAM
does with the first frame of a stream (the frame paired with itself), which gives every pixel a point in the
camera's own frame (x right, y down, z forward, in MASt3R's metres) and a confidence.

The job may also name `pairs` ([{"a": path, "b": path, "out": path}]): two pictures compared, for where the
second camera stands as seen from the first (spatial/cloud_scene.py). MASt3R runs both ways round, each picture
as the reference in turn, which gives every pixel of the other picture a point in the reference's frame. A
similarity fitted between a picture's own points and the same pixels' points in the other's frame is that camera's
position, direction, and scale there. Each pair's `out` is a JSON object (`pair`'s docstring); the pictures may
differ in size and shape.

Each view's `out` is an .npz the size the model saw the picture at (its long side 512, cropped to a multiple of 16):
`points` (H x W x 3 float32), `confidence` (H x W float32), `colour` (H x W x 3 uint8, the picture as the model saw
it), and `covered` (H x W bool: the view's `mask`, white where a camera burns in its clock or name, resized the
same way; all False without one). The report is {"load_seconds", "views": [{"seconds", "peak_gpu_gb"}],
"pairs": [{"seconds", "peak_gpu_gb"}]}.

Measured on six cameras' pictures beside the live services: 7.5 s to load, then 0.2 s and 2.9 GB of GPU memory a
picture, and 0.35 s and 2.9 GB a pair.

**Tracking a camera that moves** (a job with `"track": true`; spatial/camera_track.py runs it). The process stays up
and keeps the model loaded: it prints one JSON line when it is ready ({"ready": true, "load_seconds"}), then reads
one JSON line a frame on its standard input ({"image": path, "mask": path or null, "out": path, "reset": bool,
"anchor": bool}) and answers each with one line. The frames go through MASt3R-SLAM's own frame tracker (`FrameTracker`): each is matched
to the current keyframe and its pose solved against that keyframe's points, and a frame that shares too little with
the keyframe becomes the next one. Nothing else of MASt3R-SLAM runs: no second process, no loop closing, no
relocalising (the pack anchors the keyframes to the plan's walls instead, and starts the track again when it is
lost). A frame sent with `anchor` begins a track as its first keyframe without its points being written: the
picture a camera's own cloud was made from, so the track stands on that cloud's place from its first frame. When a
frame doesn't match the current keyframe it is looked for among the last `KEYFRAMES` keyframes of the track, the
newest first, and tracked from the one it matches (`recovered`): a camera that swings back, or comes out of a
blurred stretch, is found again in the same frame. The answer: `ok` (false when the frame matched none of them),
`pose`
(16 numbers, row by row: the frame's camera in the track's frame, which is the first keyframe's camera frame, as a
similarity: its rotation block carries the frame's scale), `started` (this frame began a track: the first, or one
sent with `reset`), `keyframe` (this frame is a keyframe, and its points were written to `out` as a view's are),
and `seconds`. Only the last `KEYFRAMES` keyframes are held, so memory doesn't grow with the length of the track.
"""

from __future__ import annotations

import json
import os
import sys
import time

SIZE = 512  # the long side MASt3R takes
CONF = 1.5  # MASt3R's confidence bar for a usable point
MIN_SURE = 500  # pixels sure in both frames that a similarity is fitted from
KEYFRAMES = 8  # keyframes of a track that are kept, to find lost lock among


def similarity(src, dst):  # noqa: ANN001, ANN201 (numpy arrays, imported in `main`)
    """The scale, rotation, and shift carrying the points `src` onto `dst` (both N x 3), least squares (Umeyama),
    fitted three times, each on the points within three median errors of the fit before."""
    import numpy as np

    keep = np.ones(len(src), dtype=bool)
    scale, turn, shift = 1.0, np.eye(3), np.zeros(3)
    for _ in range(3):
        a, b = src[keep], dst[keep]
        mean_a, mean_b = a.mean(0), b.mean(0)
        u, d, vt = np.linalg.svd((b - mean_b).T @ (a - mean_a) / len(a))
        sign = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            sign[2, 2] = -1
        turn = u @ sign @ vt
        scale = float(np.trace(np.diag(d) @ sign) / (a - mean_a).var(0).sum())
        shift = mean_b - scale * turn @ mean_a
        error = np.linalg.norm((scale * (turn @ src.T)).T + shift - dst, axis=1)
        keep = error < 3 * max(float(np.median(error[keep])), 1e-6)
    return scale, turn, shift


def pair(model, frame_a, frame_b):  # noqa: ANN001, ANN201 (the runtime's own types)
    """Where camera B stands as seen from camera A, from their two pictures, as a JSON-ready dict:

    - `match`: [the share of B's pixels MASt3R-SLAM's matcher finds in A, and of A's in B]: its tracker's own
      measure of how much two pictures share;
    - `sure`: [the share of B's pixels MASt3R is sure of both in B's own frame and in A's, and of A's]. Two
      pictures of different space have none;
    - `rotation` (9 numbers, row by row) and `shift` (3): B's camera in A's camera frame (x right, y down, z
      forward), the shift in the units of A's own single-picture point map; `scale`: what a length in B's
      single-picture units is in A's. They are the mean of two measurements, one with each picture as the reference;
    - `disagree`: how far those two measurements are apart: [degrees, the same units as `shift`, log scale].

    Without `MIN_SURE` sure pixels each way there is no fit: only `match` and `sure`."""
    import numpy as np
    import torch
    from mast3r_slam import matching  # ty: ignore[import-not-found]
    from mast3r_slam.mast3r_utils import decoder, mast3r_inference_mono  # ty: ignore[import-not-found]

    def flat(x):  # noqa: ANN001, ANN202
        return x.reshape(-1, x.shape[-1]).float().cpu().numpy()

    def one_way(ref, other):  # noqa: ANN001, ANN202
        # MASt3R-SLAM's own asymmetric step stacks the two heads' outputs, which pictures of two shapes can't be.
        with torch.inference_mode():
            for frame in (ref, other):
                if frame.feat is None:
                    frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)
            first, second = decoder(
                model, ref.feat, other.feat, ref.pos, other.pos, ref.img_true_shape, other.img_true_shape
            )
            _, valid = matching.match(
                first["pts3d"].contiguous(), second["pts3d"].contiguous(),
                first["desc"].contiguous(), second["desc"].contiguous(),
            )  # fmt: skip
        return (
            flat(first["pts3d"]), first["conf"].reshape(-1).float().cpu().numpy(),
            flat(second["pts3d"]), second["conf"].reshape(-1).float().cpu().numpy(), float(valid.float().mean()),
        )  # fmt: skip

    alone_a = flat(mast3r_inference_mono(model, frame_a)[0])
    alone_b = flat(mast3r_inference_mono(model, frame_b)[0])
    a_in_a, sure_aa, b_in_a, sure_ba, match_b = one_way(frame_a, frame_b)
    b_in_b, sure_bb, a_in_b, sure_ab, match_a = one_way(frame_b, frame_a)
    both_b = (sure_ba > CONF) & (sure_bb > CONF)
    both_a = (sure_ab > CONF) & (sure_aa > CONF)
    out: dict = {"match": [match_b, match_a], "sure": [float(both_b.mean()), float(both_a.mean())]}
    if both_b.sum() < MIN_SURE or both_a.sum() < MIN_SURE:
        return out
    # One pass's units to that reference's own single-picture units, by the same pixels' depths.
    unit_a = float(np.median(alone_a[sure_aa > CONF, 2] / a_in_a[sure_aa > CONF, 2]))
    unit_b = float(np.median(alone_b[sure_bb > CONF, 2] / b_in_b[sure_bb > CONF, 2]))
    s1, r1, t1 = similarity(b_in_b[both_b], b_in_a[both_b])  # B's frame to A's, with A as the reference
    s2, r2, t2 = similarity(a_in_a[both_a], a_in_b[both_a])  # A's frame to B's, with B as the reference
    # The second, turned round, is B's frame to A's again: p_a = (r2' / s2) (p_b - t2).
    r_back, t_back, s_back = r2.T, -(r2.T @ t2) / s2, 1.0 / s2
    shift_1, shift_2 = t1 * unit_a, t_back * unit_a
    scale_1, scale_2 = s1 * unit_a / unit_b, s_back * unit_a / unit_b
    u, _, vt = np.linalg.svd(r1 + r_back)
    mean = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
    apart = float(np.degrees(np.arccos(np.clip((np.trace(r1 @ r_back.T) - 1) / 2, -1.0, 1.0))))
    out |= {
        "rotation": [float(v) for v in mean.ravel()],
        "shift": [float(v) for v in (shift_1 + shift_2) / 2],
        "scale": float(np.sqrt(scale_1 * scale_2)),
        "disagree": [apart, float(np.linalg.norm(shift_1 - shift_2)), float(abs(np.log(scale_1 / scale_2)))],
    }
    return out


class _Keyframes(list):
    """What `FrameTracker` needs of MASt3R-SLAM's keyframe store, in this process: the last keyframe."""

    def last_keyframe(self):  # noqa: ANN202 (the runtime's own type)
        return self[-1] if self else None


def track(model, device: str, answers) -> None:  # noqa: ANN001 (the runtime's own type, and a text stream)
    """Follow frames named on standard input until it closes, answering on `answers` (the module docstring's
    tracking)."""
    import lietorch  # ty: ignore[import-not-found]
    import numpy as np
    import torch
    from mast3r_slam.frame import create_frame  # ty: ignore[import-not-found]
    from mast3r_slam.mast3r_utils import mast3r_inference_mono, resize_img  # ty: ignore[import-not-found]
    from mast3r_slam.tracker import FrameTracker  # ty: ignore[import-not-found]
    from PIL import Image

    keyframes = _Keyframes()  # the keyframe frames are tracked against: one
    kept: list = []  # and the last `KEYFRAMES` of this track, to find lost lock among
    tracker = FrameTracker(model, keyframes, device)
    last = None
    count = 0

    def say(answer: dict) -> None:
        answers.write(json.dumps(answer) + "\n")
        answers.flush()

    def keep(frame, asked: dict) -> None:  # noqa: ANN001
        h, w = (int(v) for v in frame.img_true_shape[0])
        covered = np.zeros((h, w), dtype=bool)
        if asked.get("mask"):
            mask = np.asarray(Image.open(asked["mask"]).convert("RGB"), dtype=np.float32) / 255.0
            covered = np.asarray(resize_img(mask, SIZE)["unnormalized_img"])[:h, :w, 0] > 127
        np.savez(
            asked["out"],
            points=frame.X_canon.reshape(h, w, 3).float().cpu().numpy(),
            confidence=frame.get_average_conf().reshape(h, w).float().cpu().numpy(),
            colour=(frame.uimg.cpu().numpy()[:h, :w, :3] * 255).astype(np.uint8),
            covered=covered,
        )

    say({"ready": True})
    for line in sys.stdin:
        if not line.strip():
            continue
        asked = json.loads(line)
        started = time.perf_counter()
        rgb = np.asarray(Image.open(asked["image"]).convert("RGB"), dtype=np.float32) / 255.0
        begin = asked.get("reset") or asked.get("anchor") or last is None
        pose = lietorch.Sim3.Identity(1, device=device) if begin else last.T_WC
        frame = create_frame(count, rgb, pose, img_size=SIZE, device=device)
        count += 1
        answer: dict = {"ok": True, "started": bool(begin), "keyframe": bool(begin), "recovered": False}
        if begin:
            points, confidence = mast3r_inference_mono(model, frame)
            frame.update_pointmap(points, confidence)
            keyframes.clear()
            keyframes.append(frame)
            kept.clear()
            kept.append(frame)
            tracker.reset_idx_f2k()
            if asked.get("anchor"):
                answer["keyframe"] = False  # the pack has this picture's cloud already
            else:
                keep(frame, asked)
            last = frame
        else:
            with torch.inference_mode():
                # A frame of another shape than the keyframe (the camera's stream changed size) can't be matched
                # to it: MASt3R-SLAM's matcher stacks the two.
                same = frame.img.shape == keyframes[0].img.shape
                new_keyframe, _, lost = tracker.track(frame) if same else (False, None, True)
                if lost:
                    # Lock on the current keyframe is lost: look for the frame among the keyframes before it, the
                    # newest first (the camera swung back, or a blurred stretch is over).
                    current = keyframes[0]
                    for earlier in reversed(kept):
                        if earlier is current or earlier.img.shape != frame.img.shape:
                            continue
                        keyframes[0] = earlier
                        tracker.reset_idx_f2k()
                        frame.T_WC = earlier.T_WC
                        new_keyframe, _, lost = tracker.track(frame)
                        if not lost:
                            answer["recovered"] = True
                            break
                    if lost:
                        keyframes[0] = current
                        tracker.reset_idx_f2k()
            if lost:
                answer = {"ok": False, "started": False, "keyframe": False, "recovered": False}
            else:
                last = frame
                if new_keyframe:
                    keyframes[0] = frame
                    kept.append(frame)
                    del kept[:-KEYFRAMES]
                    keep(frame, asked)
                    answer["keyframe"] = True
        if answer["ok"]:
            answer["pose"] = [float(v) for v in last.T_WC.matrix()[0].reshape(-1).cpu().numpy()]
        torch.cuda.synchronize()
        answer["seconds"] = round(time.perf_counter() - started, 3)
        say(answer)


def main() -> None:
    with open(sys.argv[1]) as f:
        job = json.load(f)
    repo = os.path.abspath(job["repo"])
    weights = os.path.abspath(job["weights"])
    views = [
        {k: (os.path.abspath(v) if isinstance(v, str) else v) for k, v in view.items()}
        for view in job.get("views") or []
    ]
    report_path = os.path.abspath(job["out"]) if job.get("out") else ""
    os.chdir(repo)  # MASt3R-SLAM reads its config relative to its checkout
    sys.path.insert(0, repo)

    import lietorch  # ty: ignore[import-not-found] (the runtime's own)
    import numpy as np
    import torch
    from mast3r_slam.config import load_config  # ty: ignore[import-not-found]
    from mast3r_slam.frame import create_frame  # ty: ignore[import-not-found]
    from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono, resize_img  # ty: ignore[import-not-found]
    from PIL import Image

    answers = sys.stdout
    if job.get("track"):
        sys.stdout = sys.stderr  # MASt3R and MASt3R-SLAM print as they go: the answers keep a stream of their own
    load_config("config/base.yaml")
    device = "cuda:0"
    started = time.perf_counter()
    model = load_mast3r(path=weights, device=device)
    if job.get("track"):
        track(model, device, answers)
        return
    report = {"load_seconds": round(time.perf_counter() - started, 2), "views": [], "pairs": []}

    def frame_of(path):  # noqa: ANN001, ANN202
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return create_frame(0, rgb, lietorch.Sim3.Identity(1, device=device), img_size=SIZE, device=device)

    for view in views:
        rgb = np.asarray(Image.open(view["image"]).convert("RGB"), dtype=np.float32) / 255.0
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        frame = create_frame(0, rgb, lietorch.Sim3.Identity(1, device=device), img_size=SIZE, device=device)
        points, confidence = mast3r_inference_mono(model, frame)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        h, w = (int(v) for v in frame.img_true_shape[0])
        covered = np.zeros((h, w), dtype=bool)
        if view.get("mask"):
            mask = np.asarray(Image.open(view["mask"]).convert("RGB"), dtype=np.float32) / 255.0
            covered = np.asarray(resize_img(mask, SIZE)["unnormalized_img"])[:h, :w, 0] > 127
        np.savez(
            view["out"],
            points=points.reshape(h, w, 3).float().cpu().numpy(),
            confidence=confidence.reshape(h, w).float().cpu().numpy(),
            colour=(frame.uimg.cpu().numpy()[:h, :w, :3] * 255).astype(np.uint8),
            covered=covered,
        )
        report["views"].append(
            {"seconds": round(seconds, 3), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        )
    for two in job.get("pairs") or []:
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        found = pair(model, frame_of(os.path.abspath(two["a"])), frame_of(os.path.abspath(two["b"])))
        torch.cuda.synchronize()
        with open(os.path.abspath(two["out"]), "w") as f:
            json.dump(found, f)
        report["pairs"].append(
            {
                "seconds": round(time.perf_counter() - t, 3),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
        )
    with open(report_path, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
