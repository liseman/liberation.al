"""`pack-spatial-cameras`: the fixed cameras that see the same space become a place, with a map, a floorplan, and each
camera's pose in it (#2825, part of the spatial memory goal #2794, after the #2786 spike).

Nightly at 01:00 (services/services.toml, niced, off-hours on the GPU) or by hand, never continuously. Fixed cameras
don't move, so a run needs no current frame: each camera's frame is its best from the last day's recordings, and a
run works at any hour (#3008).
0. **Footprints** (#2800). Every fixed camera's tracked people's foot points since the last run go into the position
   layer (spatial/footprints.py, spatial/positions.py), whether or not the camera is registered yet.
1. **Frames.** One frame of each indoor fixed camera (a manifest camera, `config/devices.toml`, not on a rig and with
   no `fisheye` calibration, `fixed_cameras`): its best recorded frame of the last day's daytime
   (`best_frame`), or its best since its view changed when its later lit frames show another view (it was moved, or
   its lens uncovered, `latest_view`). A lone camera MASt3R finds no floor for is tried again in the same run with a
   later frame showing its view now (`retry_floorless`). Frames are tried every `SAMPLE_EVERY_S` over `MIDDAY` (local time) of the last `LOOKBACK_S`
   (`sample_times`), read from the broker (`recorded`); dark and night-mode (infrared grey) ones are passed over, since
   the match counts below came from lit frames. Of the rest, see-actions' person detector (RF-DETR, on the CPU,
   `detector`) picks one with nobody in it, else the one whose people cover least of it, and the sharpest wins; the
   people it found in the chosen frame go beside it in its mask (`<cam>.mask.png`, with the overlays), so they stay
   out of the cloud and the camera's background (#3085: a man lying on Place 1's couch had become 0.77 m of furniture
   behind living-room's feet). The position layer's tracked people (`PERSON_NEAR_S`) stand in when the detector isn't
   installed; they missed him, his feet unplaced. A camera with no lit frame in the window is left out of this run.
2. **MASt3R.** spatial/mast3r_runner.py, in MASt3R's own runtime (`mise run mast3r`, under `spatial.runtime`), on the
   GPU with the most free memory (`spatial.gpu` picks one). It matches every pair, groups the cameras joined by pairs
   with at least `MIN_MATCHES` matches, and aligns each group into one point cloud with every camera's pose. A camera
   in no group (no pair reaching `LONE_MATCHES`) is a group of its own (#3000): its frame
   paired with itself, with its up and floor found from its surfaces, since one camera can be pitched far down and
   see more furniture top than floor; one whose view shows no floor (a blocked lens) is left out. Outdoor cameras
   (`mount.outdoor` in the manifest, spatial/rig.py) take no part.
2a. **Registration.** A group of two or more cameras is then registered again by reprojection (spatial/registration.py,
   #3069): from MASt3R's matches and depths, each camera's pose, focal, and lens are fitted to one reference camera,
   since MASt3R's alignment takes every camera as a pinhole and a wide one bends the whole group (Place 1: the same
   person 0.7 to 1.2 m apart across the room). The group's cloud is re-cast along the fitted rays, and its floor is the
   reference camera's lowest flat layer, tilted to keep the measured camera heights' ratios. A group that can't be
   fitted keeps MASt3R's alignment.
2b. **Gravity.** GeoCalib (spatial/geocalib_runner.py, in its own runtime, `mise run geocalib`, on the CPU) reads
   each frame's gravity and focal. For a camera registered alone, where GeoCalib's pitch is within MAX_PITCH_UNC_DEG,
   it replaces MASt3R's pitch, roll, and focal, keeping MASt3R's yaw and position, and the camera then slides along
   the floor so its walked footprints land where they did (`correct`, #2960); its floor is the lowest flat layer of
   its cloud along GeoCalib's up, when its pitch is within LONE_UP_UNC_DEG (`lone_floor`). Cameras MASt3R registered
   together keep its poses: their registration is what puts one floor point of the room in the same place through
   each of them, and turning one camera about its own centre breaks that. On Place 1 (#2960), giving cam0 GeoCalib's
   pitch moved its share of the floor 0.4 to 0.5 m against living-room's at the floor landmarks both see (a lamp's
   base and a cat scratcher, 0.15 and 0.20 m apart as MASt3R registered them), and sliding it on its footprints
   another 0.5 m. A group seated into a place's map (step 3) keeps MASt3R's poses too: the seat places its cloud by
   them, and a pose turned and slid off its own cloud starts the seat that far off and is written that far off. On
   the joined Place 1 (#3228), the one seat where GeoCalib was applied to the bedroom wrote it 0.45 m from the map's
   bedroom. GeoCalib's numbers are logged beside MASt3R's for every camera either way.
3. **Places.** Each group is written as a revision of a place (memory/places.py): the cloud as a `.map` (binary PLY),
   a floorplan from spatial/floorplan.py when the cloud has the floor and walls for one (the last 14 days of the
   group's footprints, projected onto the floor through the new poses, close rooms whose walls the cameras don't
   see: they hang high and look down), the group's cameras as its
   `camera` signal, and each camera's pose (`Revision.cameras`), which files a fixed camera's rooms under its place.
   Groups go to the places that already have most of their cameras, one group per place per run (a place that split
   keeps its larger part), else to a new place. A place whose group lost cameras only because they had no usable
   frame keeps its last map. A camera that left every group stays registered where it was last seen.
   **A group of some of a place's cameras** (#3115), all registered in that one place, the place with a fused map and
   other cameras besides, goes to it whatever other group does: a place joined from rooms seen apart (#2997, the
   bedroom with Place 1) comes back as one group a room each night. Each is fused straight into the place's map,
   started from where its cameras are registered in it (`fuse.seat`) rather than the wall search, which can't find
   one room in a home; the place's other cameras keep their poses. A group that doesn't seat (it moved, or tonight's
   snapshot is wrong) leaves the place's map as it was and makes no new place, until it has been refused
   `MOVED_NIGHTS` runs in a row (counted per camera in `state/places/seats.json`): then it has moved to a room no
   other camera sees, so it becomes a new place of its own from that night's snapshot, and the old place is rewritten
   without it (`fuse.release`), so its new room's footprints aren't cast into the old one through its old pose. A
   group that was once one MASt3R group with the place's other cameras (`matched_before`) and doesn't seat has
   moved out of their room, and splits off that night (#3146).
4. **Fusion and splats.** Each place's newest revision is fused into its map (spatial/fuse.py, #2801), then the
   revision the 3D view shows gets a 3D Gaussian splat from a day of its cameras' recorded frames
   (spatial/splats.py, #2877) when `spatial.splats_enabled` is on.

MASt3R (naver/mast3r, code and weights CC BY-NC-SA 4.0, the weights also under the non-commercial terms of the
datasets they were trained on, see its CHECKPOINTS_NOTICE) is accepted by Joe for now (2026-11-07); docs/ARCHITECTURE.md
says so where it lists the model.

Its MIN_MATCHES of 1,000 comes from the spike: two views of one room matched 3,528 times, the kitchen camera's weak tie
to them about 600, and views of different spaces at most 187. A wrong camera in a group would bend the place's map,
while a missed one only stays unregistered, so the bar sits well above the weak tie.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import tzinfo
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import logs, nvml
from liberational.memory import places
from liberational.spatial import floorplan, footprints, lens, ply, registration, rig, tiling

LOG = logs.get("spatial-cameras")

PRODUCER = "mast3r f5209afc fixed-cameras v1"
MIN_MATCHES = 1000
MAST3R_SIZE = 512  # the long side MASt3R sees a frame at
#: A camera none of whose pairs reaches this many matches joins no group, and becomes a place of its own (#3000) when
#: its view shows a floor. The same bar as a group's: a camera that shares only a sliver of another's view (the kitchen
#: camera's 350 to 675 matches with Place 1's, against 3,600 within it) is its own room, not hung on that sliver,
#: where it could bend the other place's map. Rooms seen apart are tied later by a walkthrough (#2997).
LONE_MATCHES = MIN_MATCHES
#: A frame darker than this mean (0 to 255) is left out: MASt3R's matches, and the 1,000 bar, came from lit frames.
MIN_BRIGHTNESS = 60.0
#: A frame whose channels differ by less than this on average is a night-mode (infrared) grey frame, left out too.
MIN_COLOUR = 4.0
#: A group of a place's cameras whose seat in its map (`fuse.seat`) is refused this many runs in a row has moved: to a
#: room no other camera sees, as one tonight's run can't tell from a bad snapshot. It is then registered on its own, a
#: new place, and taken out of the old one (`fuse.release`). A run in which it seats starts the count again.
MOVED_NIGHTS = 3
SEATS_FILE = "seats.json"  # under the places root: each camera's refused seats in a row (`refused_seats`)
RUNNER = Path(__file__).with_name("mast3r_runner.py")
RUN_TIMEOUT_S = 1800.0


def fixed_cameras(manifest: Path) -> list[str]:
    """The enabled fixed cameras in the capture manifest, in manifest order: not on a rig (`mount.rig`, spatial/rig.py,
    #5), whose poses change as it moves, and with no `fisheye` calibration, which MASt3R's pinhole model can't take."""
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


def indoor_cameras(cams: Sequence[str], mounts: rig.Mounts) -> list[str]:
    """`cams` without the ones mounted looking outdoors (`mount.outdoor`), in order."""
    out = mounts.outdoor()
    return [c for c in cams if c not in out]


def light(pixels: np.ndarray) -> tuple[float, float]:
    """A BGR frame's mean brightness (0 to 255) and colour (the mean difference between its channels: near 0 for a
    grey, night-mode frame); (0, 0) for a frame that isn't three-channel."""
    img = np.asarray(pixels, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        return (0.0, 0.0)
    spread = np.abs(img[..., 0] - img[..., 1]) + np.abs(img[..., 1] - img[..., 2])
    return (float(img.mean()), float(spread.mean()))


def daylight(pixels: np.ndarray) -> bool:
    """Whether a BGR frame is lit colour, the kind MASt3R matched well in the spike, rather than a dark frame or a
    camera's night mode (infrared: grey, its three channels all but equal)."""
    brightness, colour = light(pixels)
    return brightness >= MIN_BRIGHTNESS and colour >= MIN_COLOUR


#: The last day's midday, when each camera's frame is chosen (and spatial/splats.py's training frames): tried every
#: SAMPLE_EVERY_S over MIDDAY (local minutes past midnight, [from, to)) of the last LOOKBACK_S. A splat is one fixed
#: appearance, and the day's light changes it: on Place 1, 10:30 to 14:30 against 09:00 to 16:00 put held-out midday
#: frames 2.0 to 2.5 dB closer (cam0 28.4 to 30.9, living-room 25.7 to 27.7) for 0.5 dB less on a 09:40 frame
#: (#2877). Every 20 minutes keeps up to 12 tries a camera, well above the splat's 6-frame minimum.
LOOKBACK_S = 86400.0
SAMPLE_EVERY_S = 1200.0
MIDDAY = (10 * 60 + 30, 14 * 60 + 30)
#: A try takes the recorded frame nearest its time, this close.
RECORDED_WITHIN_S = 2.0
#: A frame with someone tracked on its camera within this many seconds is a second choice: people move between
#: frames, and MASt3R matches the room, not them.
PERSON_NEAR_S = 10.0

#: One camera's recorded frame nearest a time (BGR), or None.
Reader = Callable[[str, float], Any]
#: The people in a BGR frame: their boxes, (x1, y1, x2, y2) as fractions of its width and height (`detector`).
Detector = Callable[[np.ndarray], list[tuple[float, float, float, float]]]
#: RF-DETR's person score a box needs to count as someone in a frame chosen for registration (#3085). Lower than
#: see-actions' 0.6 tracking floor: a missed person ends up in the camera's cloud and background as furniture, an
#: extra box only leaves a patch of them unknown.
PERSON_SCORE = 0.5
#: Each person's box is widened by this share of its size each way before it's left out (a hand, a foot on a stool).
PERSON_GROW = 0.1


ALL_DAY = (0, 24 * 60)
#: A camera's view has changed (it was moved, or something stood in front of its lens) when its frame's small grey
#: picture correlates with its last lit frame's less than this (`view_match`: 1 for the same view at any brightness,
#: about 0 for unrelated ones). On 2026-11-08 a bottle stood in front of the kitchen lens over midday, and the camera
#: was moved and cleared at 16:05: every midday frame showed no floor, while a 17:00 one gave a good one.
VIEW_SAME = 0.5
VIEW_SIZE = (32, 18)  # the small grey picture views are compared at, width by height


def sample_times(
    now: float,
    zone: tzinfo,
    lookback: float = LOOKBACK_S,
    every: float = SAMPLE_EVERY_S,
    window: tuple[int, int] = MIDDAY,
) -> list[float]:
    """The times frames are tried at: every `every` seconds over the last `lookback`, oldest first, the newest `every`
    before now, those whose local time in `zone` is in `window` (minutes past midnight, [from, to))."""
    count = int(lookback // every)
    times = [now - every * (count - i) for i in range(count)]
    return [t for t in times if window[0] <= _minute(dt.datetime.fromtimestamp(t, zone)) < window[1]]


def view_match(a: np.ndarray, b: np.ndarray) -> float:
    """How alike two BGR frames' views are: the correlation of their small grey pictures, each less its mean (1 for
    the same view however lit, about 0 for unrelated views)."""
    import cv2

    def small(f: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(np.asarray(f, np.uint8), cv2.COLOR_BGR2GRAY)
        v = cv2.resize(grey, VIEW_SIZE, interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
        v -= v.mean()
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    return float(small(a) @ small(b))


def latest_view(cam: str, times: Sequence[float], read: Reader) -> list[float]:
    """Of `times` (oldest first), those whose frame is lit and shows the view `cam`'s last lit frame of them shows
    (VIEW_SAME): the camera as it is now, after any move."""
    lit = [(t, f) for t in times if (f := read(cam, t)) is not None and daylight(f)]
    if not lit:
        return []
    last = lit[-1][1]
    return [t for t, f in lit if view_match(f, last) >= VIEW_SAME]


def _minute(when: dt.datetime) -> int:
    return when.hour * 60 + when.minute


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


def person_boxes(
    dets: np.ndarray, logits: np.ndarray, score: float = PERSON_SCORE
) -> list[tuple[float, float, float, float]]:
    """The people among RF-DETR's queries: `dets` (N x 4, centre x, centre y, width, height, fractions of the frame)
    whose person logit (sparse COCO, see vision/crossings.py) reaches `score` after a sigmoid, as clipped
    (x1, y1, x2, y2) fractions. RF-DETR has no NMS, so one person can be two overlapping boxes; their union is what's
    left out, so that doesn't matter here."""
    from liberational.vision import crossings

    slot = crossings.RF_CLASS_IDS[crossings.COCO_PERSON]
    keep = 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64)[:, slot])) >= score
    out = []
    for cx, cy, w, h in np.asarray(dets, dtype=np.float64)[keep]:
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def detector() -> Detector | None:
    """See-actions' person detector (RF-DETR, [models.files.person_detector]) on the CPU, about 0.14 s a frame, or
    None when the model or onnxruntime isn't installed (frames are then chosen by the position layer's people)."""
    path = cfg.model_path("person_detector")
    if not path.exists():
        LOG.info("no person detector at %s: frames are chosen by tracked people only", path)
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        LOG.info("no onnxruntime: frames are chosen by tracked people only")
        return None
    from liberational.vision import crossings

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    entry = session.get_inputs()[0]
    size = int(entry.shape[2])
    names = [o.name for o in session.get_outputs()]
    if set(names) != {"dets", "labels"}:
        LOG.warning("the person detector at %s isn't RF-DETR's (dets, labels): frames by tracked people only", path)
        return None

    def people(frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        got = dict(zip(names, session.run(None, {entry.name: crossings.detector_input(np.asarray(frame), size)})))
        return person_boxes(np.asarray(got["dets"])[0], np.asarray(got["labels"])[0])

    return people


def covered_share(boxes: Sequence[Sequence[float]], cells: int = 64) -> float:
    """How much of a frame the boxes (fractions) cover, their union measured on a `cells` square grid."""
    grid = np.zeros((cells, cells), bool)
    for x1, y1, x2, y2 in boxes:
        grid[int(y1 * cells) : math.ceil(y2 * cells), int(x1 * cells) : math.ceil(x2 * cells)] = True
    return float(grid.mean())


def best_frame(
    cam: str,
    times: Sequence[float],
    read: Reader,
    people: Sequence[tuple[float, float]] = (),
    detect: Detector | None = None,
) -> tuple[float, np.ndarray, list[tuple[float, float, float, float]]] | None:
    """`cam`'s best frame of `times`, with its time and the people `detect` found in it: lit (`daylight`); with a
    detector, one with nobody in it if any, else the one whose people cover the least of it (#3085: on Place 1 at
    midday someone was in every living-room frame, and one lying on the couch became 0.77 m of furniture in its
    background); without one, at a time in none of the `people` spans (when someone was tracked on it) if any is; and
    of those the sharpest. None when no frame is lit."""
    lit = [(t, f) for t in times if (f := read(cam, t)) is not None and daylight(f)]
    if not lit:
        return None
    if detect is not None:
        found = [(t, f, detect(f)) for t, f in lit]
        return min(found, key=lambda x: (covered_share(x[2]), -sharpness(x[1])))
    clear = [(t, f) for t, f in lit if not any(a <= t <= b for a, b in people)]
    t, f = max(clear or lit, key=lambda tf: sharpness(tf[1]))
    return t, f, []


def person_mask(boxes: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    """The pixels of a `width` x `height` frame the people's boxes cover, each grown by PERSON_GROW (H x W bool)."""
    out = np.zeros((height, width), bool)
    for x1, y1, x2, y2 in boxes:
        gx, gy = PERSON_GROW * (x2 - x1), PERSON_GROW * (y2 - y1)
        c1, c2 = max(0, math.floor((x1 - gx) * width)), min(width, math.ceil((x2 + gx) * width))
        r1, r2 = max(0, math.floor((y1 - gy) * height)), min(height, math.ceil((y2 + gy) * height))
        out[r1:r2, c1:c2] = True
    return out


def people_spans(state: Path, cam: str, start: float, end: float) -> list[tuple[float, float]]:
    """When someone was tracked on `cam` over [start, end), from the position layer, each span widened by
    PERSON_NEAR_S."""
    from liberational.spatial import positions

    return [(s.t - PERSON_NEAR_S, s.end + PERSON_NEAR_S) for s in positions.samples(state, start, end, cameras=[cam])]


def frames(
    state: Path,
    cams: Sequence[str],
    folder: Path,
    now: float,
    read: Reader = recorded,
    zone: tzinfo | None = None,
    detect: Detector | None = None,
    picked: dict[str, float] | None = None,
    after: Mapping[str, float] | None = None,
) -> dict[str, Path]:
    """Each camera's best recorded frame of the last day's daytime (`best_frame`, people found by `detect`), saved as
    PNG under `folder`, with a wide lens undone and its overlays painted over (spatial/lens.py); a camera with no lit
    frame then is left out. The overlays and the people in the frame go beside it as `<cam>.mask.png`, which
    mast3r_runner.py leaves out of the cloud and spatial/registration.py out of the cloud and the background.

    A camera whose view changed after its midday frame (moved, or its lens uncovered: its later lit frames, every
    SAMPLE_EVERY_S, show another view than that frame, `latest_view`) gets its best frame since the change instead, as
    does one with no lit frame at midday. With `after` (camera to a time), each of those cameras gets its best frame
    later than that time that shows its view now (`retry_floorless`). Each chosen frame's time goes into `picked`."""
    import cv2

    if zone is None:
        from liberational.memory.history_qa import local_zone  # the journal's clock

        zone = local_zone()
    times = sample_times(now, zone)
    day = sample_times(now, zone, window=ALL_DAY)
    got: dict[str, Path] = {}
    for cam in cams:
        cache: dict[float, Any] = {}

        def seen(c: str, t: float, cache: dict[float, Any] = cache) -> Any:
            if t not in cache:
                cache[t] = read(c, t)
            return cache[t]

        spans = people_spans(state, cam, day[0] - PERSON_NEAR_S, day[-1] + PERSON_NEAR_S) if day else []
        found = best_frame(cam, times, seen, spans, detect) if after is None else None
        since = (after or {}).get(cam, found[0] if found is not None else -math.inf)
        recent = latest_view(cam, [t for t in day if t > since], seen)
        why = ""
        if after is not None:
            why = "its view now, after the frame it was tried with"
        elif recent and found is None:
            why = "no lit frame at midday"
        elif recent and found is not None and view_match(found[1], seen(cam, recent[-1])) < VIEW_SAME:
            why = "its view changed after midday"
        if why:
            found = best_frame(cam, recent, seen, spans, detect) if recent else None
        if found is None:
            LOG.info("%s: no lit recorded frame %s; left out of this run", cam, why or "in the last day's daytime")
            continue
        t, pixels, people = found
        if picked is not None:
            picked[cam] = t
        if detect is not None:
            note = f"; {len(people)} people in it, {covered_share(people):.0%} of it left out" if people else ""
        else:
            note = "; someone was in view at every lit time" if any(a <= t <= b for a, b in spans) else ""
        LOG.info(
            "%s: the frame from %s (sharpness %.0f%s)%s",
            cam,
            dt.datetime.fromtimestamp(t, zone).strftime("%Y-%m-%d %H:%M"),
            sharpness(pixels),
            note,
            f", {why}" if why else "",
        )
        # A wide lens undone and its overlays painted over (#2998); the overlays' and people's pixels go beside the
        # frame as `<cam>.mask.png`
        height, width = pixels.shape[:2]
        here = person_mask(people, width, height)
        found_lens, masks = lens.for_camera(lens.manifest(), cam)
        covered = np.zeros((height, width), bool)
        if found_lens is not None or masks:
            pixels, covered = lens.prepare(pixels, found_lens, masks)
            if found_lens is not None and here.any():  # the people's pixels where the undistorted frame has them
                mx, my = lens.undistort_maps(found_lens, width, height)
                here = cv2.remap(here.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        covered |= here
        if covered.any():
            cv2.imwrite(str(folder / f"{cam}.mask.png"), covered.astype(np.uint8) * 255)
        path = folder / f"{cam}.png"
        cv2.imwrite(str(path), pixels)
        got[cam] = path
    return got


def pick_gpu(setting: int, found: Sequence[Mapping[str, Any]] | None) -> str | None:
    """The CUDA_VISIBLE_DEVICES value for the run: GPU `setting` when it's 0 or more, else the one with the most free
    memory, by UUID (NVML and CUDA can number GPUs differently); None with no GPU known."""
    if not found:
        return None
    if setting >= 0:
        match = [g for g in found if int(g["index"]) == setting]
        return str(match[0]["uuid"]) if match else None
    best = max(found, key=lambda g: int(g["total"]) - int(g["used"]))
    return str(best["uuid"])


def checkpoint() -> Path:
    return cfg.ROOT / str(cfg.load()["models"]["hf"]["mast3r"]["path"])


def run_mast3r(
    runtime: Path,
    images: Mapping[str, Path],
    out: Path,
    gpu: str | None,
    *,
    min_matches: int = MIN_MATCHES,
    window: int | None = None,
    offsets: Sequence[int] | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    timeout_s: float = RUN_TIMEOUT_S,
    nice: int = 0,
    lone_matches: int | None = None,
    save_pairs: bool = False,
) -> dict[str, Any]:
    """Run spatial/mast3r_runner.py in MASt3R's runtime on `images` and return its answer (see that module). `window`
    limits the pairs to images at most that many apart in name order; `pairs` (image names) runs the runner's pairs
    mode on just those (a walkthrough's loop edges), `save_pairs` saving their inferences too, for registration by
    reprojection; `nice` lowers the run's CPU priority; `lone_matches` makes a camera with no pair that strong a place
    of its own."""
    python = runtime / ".venv" / "bin" / "python"
    repo = runtime / "src"
    if not python.exists() or not (repo / "mast3r").is_dir():
        raise RuntimeError(f"MASt3R's runtime isn't installed in {runtime}; run `mise run mast3r`")
    if not (checkpoint() / "model.safetensors").exists():
        raise RuntimeError(f"MASt3R's checkpoint isn't in {checkpoint()}; run `mise run models`")
    out.mkdir(parents=True, exist_ok=True)
    job = out / "job.json"
    job.write_text(
        json.dumps(
            dict(
                repo=str(repo),
                checkpoint=str(checkpoint()),
                images={c: str(p) for c, p in images.items()},
                min_matches=min_matches,
                size=MAST3R_SIZE,
                out=str(out),
                **({"window": window} if window is not None else {}),
                **({"offsets": list(offsets)} if offsets is not None else {}),
                **({"pairs": [list(p) for p in pairs]} if pairs is not None else {}),
                **({"lone_matches": lone_matches} if lone_matches is not None else {}),
                **({"save_pairs": True} if save_pairs else {}),
            )
        )
    )
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    command = [str(python), str(RUNNER), str(job)]
    if nice:
        command = ["nice", "-n", str(nice), *command]
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout_s)
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1:] or ["(no output)"]
        raise RuntimeError(f"MASt3R failed ({done.returncode}): {tail[0][:300]}")
    return json.loads((out / "result.json").read_text())


GEOCALIB_RUNNER = Path(__file__).with_name("geocalib_runner.py")
GEOCALIB_TIMEOUT_S = 600.0
#: GeoCalib's gravity and focal replace MASt3R's for a camera registered alone when its pitch is this sure (degrees,
#: GeoCalib's own uncertainty): on Place 1 MASt3R's pitch was 4 to 14 degrees steeper than GeoCalib's on three of four
#: cameras (#2960).
MAX_PITCH_UNC_DEG = 3.0
#: A single camera's floor is found along GeoCalib's up when its pitch is this sure (degrees). The up only picks which
#: flat layer is the floor, and the plane is fitted to that layer, so it can be looser than MAX_PITCH_UNC_DEG. On Place
#: 1's bedroom the runner's floor, from the cloud's own surfaces, cut through the room: 39% of the cloud lay under it
#: and the camera stood 1.10 m above it against 1.52 measured. The lowest layer under GeoCalib's up put the camera
#: 1.51 above it, with 0.3% of the cloud under it (#2960).
LONE_UP_UNC_DEG = 5.0
#: A lone camera found closer than this above its floor (in its cloud's units, about metres) has the wrong floor: on
#: 2026-11-08 an undistorted kitchen frame gave a cloud whose floors put the camera 0.11 above one and 0.08 under the
#: other, against 0.91 m measured, and the place was scaled by 8.3 (#2960).
MIN_LONE_HEIGHT = 0.3
#: Measured over found camera height outside this range means the floor or the cloud is wrong, not MASt3R's scale,
#: which has been within 0.7 to 1.4 of metric (fuse.py refuses such a scale too).
SCALE_BOUNDS = (0.5, 2.0)
#: As the runner's floor_from_normals: horizontal surfaces are points whose normal is this close to up (degrees), their
#: heights binned this fine (metres); the floor is the lowest bin holding this share of them, fitted within the band.
FLAT_DEG, FLOOR_BIN, FLOOR_SHARE, FLOOR_BAND = 15.0, 0.04, 0.05, 0.06
FLOOR_POINTS = 200_000  # at most this many of the cloud's points are read for its surfaces


@dataclass(frozen=True)
class Gravity:
    """GeoCalib's answer for one camera's frame: `up` in the camera's frame (x right, y down, z forward), its focal
    as a fraction of the frame's width, and its uncertainties in degrees."""

    up: tuple[float, float, float]
    focal: float
    pitch_unc: float
    roll_unc: float


def geocalib_checkpoint() -> Path:
    return cfg.path("models.files.geocalib.path")


def run_geocalib(runtime: Path, images: Mapping[str, Path], out: Path, nice: int = 0) -> dict[str, Gravity]:
    """GeoCalib on each camera's frame, in its own runtime (spatial/geocalib_runner.py); {} when the runtime or the
    checkpoint is missing, or the run fails (the cameras keep MASt3R's gravity, and the log says why)."""
    python = runtime / ".venv" / "bin" / "python"
    if not python.exists():
        LOG.warning(
            "GeoCalib's runtime isn't installed in %s (`mise run geocalib`): cameras keep MASt3R's gravity", runtime
        )
        return {}
    if not geocalib_checkpoint().exists():
        LOG.warning(
            "GeoCalib's checkpoint isn't in %s (`mise run models`): cameras keep MASt3R's gravity",
            geocalib_checkpoint(),
        )
        return {}
    out.mkdir(parents=True, exist_ok=True)
    job = out / "geocalib-job.json"
    job.write_text(
        json.dumps(
            {
                "checkpoint": str(geocalib_checkpoint()),
                "images": {c: str(p) for c, p in images.items()},
                "out": str(out),
            }
        )
    )
    command = [str(python), str(GEOCALIB_RUNNER), str(job)]
    if nice:
        command = ["nice", "-n", str(nice), *command]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=GEOCALIB_TIMEOUT_S, cwd=out)
    except subprocess.TimeoutExpired:
        LOG.warning("GeoCalib took over %.0f s: cameras keep MASt3R's gravity", GEOCALIB_TIMEOUT_S)
        return {}
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1:] or ["(no output)"]
        LOG.warning("GeoCalib failed (%s): cameras keep MASt3R's gravity", tail[0])
        return {}
    answer = json.loads((out / "result.json").read_text())
    for cam, error in answer.get("errors", {}).items():
        LOG.warning("GeoCalib failed on %s's frame (%s): it keeps MASt3R's gravity", cam, error)
    return {
        cam: Gravity(
            (float(g["up"][0]), float(g["up"][1]), float(g["up"][2])),
            float(g["focal"]),
            float(g["pitch_unc"]),
            float(g["roll_unc"]),
        )
        for cam, g in answer.get("cameras", {}).items()
    }


def floor_along(points: np.ndarray, up: Sequence[float] | np.ndarray) -> tuple[float, float, float, float] | None:
    """The cloud's floor (nx, ny, nz, d) as the lowest layer of its horizontal surfaces along `up` that holds
    FLOOR_SHARE of them, with a plane fitted to that layer; None when too few surfaces are horizontal. Surface normals
    come from each point's 16 nearest neighbours."""
    from scipy.spatial import KDTree

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) > FLOOR_POINTS:
        pts = pts[np.random.default_rng(0).choice(len(pts), FLOOR_POINTS, replace=False)]
    if len(pts) < 100:
        return None
    upv = np.asarray(up, dtype=np.float64)
    upv = upv / np.linalg.norm(upv)
    _, idx = KDTree(pts).query(pts, k=16)
    near = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", near, near))
    flat = pts[
        (np.abs(vectors[:, :, 0] @ upv) > math.cos(math.radians(FLAT_DEG))) & (values[:, 0] < 0.05 * values[:, 1])
    ]
    if len(flat) < 50:
        return None
    h = flat @ upv
    counts, edges = np.histogram(h, np.arange(h.min(), h.max() + 2 * FLOOR_BIN, FLOOR_BIN))
    enough = np.nonzero(counts >= max(50, FLOOR_SHARE * len(flat)))[0]
    if not len(enough):
        return None
    level = (edges[enough[0]] + edges[enough[0] + 1]) / 2
    layer = flat[np.abs(h - level) < FLOOR_BAND]
    centre = layer.mean(axis=0)
    normal = np.linalg.svd(layer - centre, full_matrices=False)[2][-1]
    normal = normal if normal @ upv > 0 else -normal
    x, y, z = (float(v) for v in normal)
    return (x, y, z, -float(normal @ centre))


def lone_floor(
    camera: str, pose: places.CameraPose, points: np.ndarray, floor: Sequence[float] | None, gravity: Gravity | None
) -> Sequence[float] | None:
    """A single camera's floor: the lowest flat layer along GeoCalib's up (turned into the cloud's frame by MASt3R's
    camera rotation) when GeoCalib is sure enough (LONE_UP_UNC_DEG), else the runner's `floor`. Logged either way."""
    if gravity is None:
        return floor
    if gravity.pitch_unc > LONE_UP_UNC_DEG:
        LOG.info("%s: floor kept from its surfaces (GeoCalib's pitch +-%.1f degrees)", camera, gravity.pitch_unc)
        return floor
    m = np.asarray(pose.pose, dtype=np.float64).reshape(4, 4)
    found = floor_along(points, m[:3, :3] @ np.asarray(gravity.up, dtype=np.float64))
    if found is None:
        LOG.info("%s: no flat layer along GeoCalib's up; floor kept from its surfaces", camera)
        return floor

    def stands(fl: Sequence[float]) -> tuple[float, float]:
        n = np.asarray(fl[:3], dtype=np.float64)
        heights = np.asarray(points, dtype=np.float64).reshape(-1, 3) @ n + float(fl[3])
        return float(m[:3, 3] @ n + float(fl[3])), float((heights < -FLOOR_BAND).mean())

    now, under = stands(found)
    was = f"{stands(floor)[0]:.2f} with {stands(floor)[1]:.0%} under" if floor is not None else "none"
    LOG.info(
        "%s: floor along GeoCalib's up (+-%.1f degrees): the camera %.2f above it, %.1f%% of the cloud under it (was %s)",
        camera,
        gravity.pitch_unc,
        now,
        100 * under,
        was,
    )
    return found


def implausible(camera: str, pose: places.CameraPose, floor: Sequence[float]) -> str | None:
    """Why a lone camera's floor can't be right (too close under the camera, or a scale against the measured height
    outside SCALE_BOUNDS), or None."""
    n = np.asarray(floor[:3], dtype=np.float64)
    height = float(np.asarray(pose.pose, dtype=np.float64).reshape(4, 4)[:3, 3] @ n + float(floor[3])) / float(
        np.linalg.norm(n)
    )
    if height < MIN_LONE_HEIGHT:
        return f"the camera is {height:.2f} above its floor (needs {MIN_LONE_HEIGHT})"
    measured = rig.load().get(camera).height_m
    if measured and not SCALE_BOUNDS[0] <= float(measured) / height <= SCALE_BOUNDS[1]:
        return f"the camera is {height:.2f} above its floor against {float(measured):.2f} m measured"
    return None


def tilt(up: Sequence[float]) -> tuple[float, float]:
    """(pitch down, roll) in degrees of a camera whose frame (x right, y down, z forward) has `up` as up."""
    x, y, z = (float(v) for v in up)
    return math.degrees(math.asin(max(-1.0, min(1.0, -z)))), math.degrees(math.atan2(x, -y))


def regravitate(pose: places.CameraPose, floor: Sequence[float], gravity: Gravity) -> places.CameraPose:
    """`pose` turned about its own centre to GeoCalib's pitch and roll against the floor, facing the same way across
    the floor as MASt3R had it (its yaw), with GeoCalib's focal; its position stays MASt3R's."""
    m = np.asarray(pose.pose, dtype=np.float64).reshape(4, 4)
    n = np.asarray(floor[:3], dtype=np.float64)
    n = n / np.linalg.norm(n)
    ahead = m[:3, 2] - (m[:3, 2] @ n) * n  # MASt3R's view direction across the floor
    if np.linalg.norm(ahead) < 1e-9:
        ahead = m[:3, 1] - (m[:3, 1] @ n) * n  # a camera looking straight down: its frame's down is "ahead"
    ahead = ahead / np.linalg.norm(ahead)
    pitch, roll = (math.radians(a) for a in tilt(gravity.up))
    forward = math.cos(pitch) * ahead - math.sin(pitch) * n
    right = np.cross(forward, n)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    # Roll turns the frame about its forward axis: positive roll tips the camera's x toward up, as `tilt` reads it.
    right, down = math.cos(roll) * right - math.sin(roll) * down, math.cos(roll) * down + math.sin(roll) * right
    m[:3, :3] = np.c_[right, down, forward]
    return replace(pose, pose=tuple(float(x) for x in m.ravel()), focal=gravity.focal * pose.width)


def reanchor(
    old: places.CameraPose, new: places.CameraPose, cells: footprints.Cells | None, floor: Sequence[float]
) -> places.CameraPose:
    """`new` slid along the floor so the camera's walked footprints land, at the median, where `old` put them: turning
    a camera about its centre swings its far floor points metres out, while MASt3R placed it by what it sees. As it
    is without footprints."""
    if cells is None:
        return new
    pairs = [
        (a, b, int(k))
        for (u, v), k in zip(cells.uv.tolist(), cells.counts.tolist(), strict=True)
        if (a := places.floor_point(old, floor, u, v, footprints.FOOT_M)) is not None
        and (b := places.floor_point(new, floor, u, v, footprints.FOOT_M))
    ]
    if not pairs:
        return new
    shift = np.array(
        [np.median(np.repeat(np.array([a[i] - b[i] for a, b, _ in pairs]), [k for *_, k in pairs])) for i in range(3)]
    )
    n = np.asarray(floor[:3], dtype=np.float64) / np.linalg.norm(floor[:3])
    shift -= (shift @ n) * n  # along the floor only: the camera's height stays
    m = np.asarray(new.pose, dtype=np.float64).reshape(4, 4)
    m[:3, 3] += shift
    return replace(new, pose=tuple(float(x) for x in m.ravel()))


def correct(
    poses: Mapping[str, places.CameraPose],
    floor: Sequence[float] | None,
    gravity: Mapping[str, Gravity],
    seen: Sequence[footprints.Cells],
    kept: str | None = None,
) -> dict[str, places.CameraPose]:
    """Each camera with a sure enough GeoCalib answer (pitch uncertainty at most MAX_PITCH_UNC_DEG) given its gravity
    and focal (`regravitate`), then re-anchored on its footprints (`reanchor`); every camera GeoCalib answered for is
    logged, applied or skipped. With a reason MASt3R's poses are `kept` (cameras registered together, or a group
    seated by its cloud: see the module's step 2b) nothing is changed and the comparison is only logged."""
    out = dict(poses)
    if floor is None:
        return out
    n = np.asarray(floor[:3], dtype=np.float64) / np.linalg.norm(floor[:3])
    cells = {c.camera: c for c in seen}
    for cam, pose in sorted(poses.items()):
        g = gravity.get(cam)
        if g is None:
            continue
        mast3r = tilt(tuple(np.asarray(pose.pose, dtype=np.float64).reshape(4, 4)[:3, :3].T @ n))
        geo = tilt(g.up)
        sure = g.pitch_unc <= MAX_PITCH_UNC_DEG
        if kept is not None:
            why = f"kept MASt3R's: {kept}"
        elif sure:
            why = "applied"
        else:
            why = f"skipped (pitch uncertainty over {MAX_PITCH_UNC_DEG:.0f} degrees)"
        LOG.info(
            "%s: MASt3R pitch %.1f roll %.1f focal %.3f, GeoCalib pitch %.1f (+-%.1f) roll %.1f (+-%.1f) focal %.3f: %s",
            cam,
            *mast3r,
            pose.focal / pose.width,
            geo[0],
            g.pitch_unc,
            geo[1],
            g.roll_unc,
            g.focal,
            why,
        )
        if sure and kept is None:
            out[cam] = reanchor(pose, regravitate(pose, floor, g), cells.get(cam), floor)
    return out


def up_direction(
    poses: Mapping[str, Sequence[float]], sizes: Mapping[str, Sequence[int]] | None = None
) -> tuple[float, float, float]:
    """The cloud's rough up: the mean of the cameras' up axes (each camera's -y, as cameras hang roughly level). A
    camera whose frames are portrait is on its side, so its -y says nothing about up: only landscape cameras count
    (by `sizes`, width and height), unless there are none."""
    level = [c for c in poses if sizes is None or int(sizes[c][0]) >= int(sizes[c][1])] or list(poses)
    ups = [-np.asarray(poses[c], dtype=float).reshape(4, 4)[:3, 1] for c in level]
    up = np.mean(ups, axis=0)
    norm = float(np.linalg.norm(up))
    if norm < 1e-6:
        return (0.0, -1.0, 0.0)
    x, y, z = (up / norm).tolist()
    return (x, y, z)


@dataclass(frozen=True)
class Layout:
    """What the cloud says about its place's floor: the plan's DSL and its map-to-plan transform (12 numbers), when
    the cloud has the walls for a plan, and the floor plane (nx, ny, nz, d), when it has a floor."""

    plan: str | None
    transform: tuple[float, ...] | None
    floor: tuple[float, ...] | None


#: The walked floor on a found floor plane (nx, ny, nz, d): the footprints projected onto it, (N, 3) points in the map
#: frame, and how many footprints each point stands for (spatial/footprints.py reads them as cells of the frame).
Walked = Callable[[tuple[float, ...]], tuple[np.ndarray, np.ndarray]]


def fit_floor(points: np.ndarray, up: tuple[float, float, float]) -> tuple[float, float, float, float] | None:
    """The cloud's floor plane (nx, ny, nz, d: a point's height above it is n . p + d), or None without one."""
    settings = floorplan.Settings(up=up)
    try:
        normal, offset = floorplan.floor_plane(
            np.asarray(points, dtype=np.float64), up, settings.floor_tolerance, np.random.default_rng(settings.seed)
        )
    except ValueError:
        return None
    x, y, z = (float(v) for v in normal)
    return (x, y, z, -float(offset))


def layout(
    points: np.ndarray,
    up: tuple[float, float, float],
    title: str,
    walked: Walked | None = None,
    floor: Sequence[float] | None = None,
) -> Layout:
    """The cloud's floor and, when it has one, its floorplan. A few high cameras' views have the floor and furniture
    but few walls, so where people walked on that floor (`walked`, spatial/footprints.py, #2800) closes the rooms
    the walls alone don't. A `floor` given (nx, ny, nz, d: a lone camera's, which the runner finds from its
    surfaces, #3000) is taken as it is; otherwise one is fitted to the cloud."""
    settings = floorplan.Settings(up=up)
    floor = floor if floor is not None else fit_floor(points, up)
    if floor is None:
        return Layout(None, None, None)
    floor = tuple(float(x) for x in floor)
    feet, weights = walked(floor) if walked is not None else (None, None)
    try:
        plan = floorplan.floorplan(points, settings, floor=floor, walkable=feet, walk_weights=weights)
    except ValueError as e:
        LOG.info("no floorplan for %s (%d footprints): %s", title, 0 if weights is None else int(weights.sum()), e)
        return Layout(None, None, floor)
    # Through the tiling converter (#2803), so the library takes it as it is: rooms on shared centrelines, openings
    # clamped to their walls. Its squaring turn (none for the baseline's axis-aligned rooms) goes into the transform.
    rooms, openings = tiling.from_plan(plan)
    t = tiling.tile(rooms, openings, tiling.Settings(faces=False))
    for note in t.notes:
        LOG.info("%s: %s", title, note)
    c, s = math.cos(t.angle), math.sin(t.angle)
    turned = plan.transform.copy()
    turned[:2] = np.array([[c, -s], [s, c]]) @ plan.transform[:2]
    tr = tuple(float(x) for x in turned.ravel())
    return Layout(tiling.to_dsl(t, title, tiling.Settings(faces=False)), tr, floor)


def owners(state: Path, groups: Sequence[Sequence[str]]) -> list[str | None]:
    """The place each group goes to, None for a group that gets a new place. A group whose cameras are all registered
    in one place that has a fused map goes to that place, however many other groups do (#3115): a joined place
    (#2997) holds rooms whose cameras never match each other, so it comes back as one group a room every night, and
    each is seated into its map by its own cameras (`fuse.seat`). Every other group goes to the place sharing most of
    its cameras (by camera signal), best overlap first (then the more recently revised place), each of those places to
    at most one group: two groups that both came from one place (it split) don't both write into it in one run."""
    from liberational.spatial import fuse  # fuse uses this module's layout

    registered = places.camera_places(state)
    out: list[str | None] = [None] * len(groups)
    for i, g in enumerate(groups):
        homes = {registered.get(c) for c in g}
        home = homes.pop() if len(homes) == 1 else None
        if home is not None and fuse.fused_revision(state, home) is not None:
            out[i] = home
    taken = {p for p in out if p is not None}
    revs = [rev for p in places.place_ids(state) if (rev := places.latest(state, p)) is not None]
    pairs = [
        (len(set(rev.signals.get("camera", [])) & set(g)), rev.at, i, rev.place)
        for i, g in enumerate(groups)
        for rev in revs
    ]
    for shared, _, i, place in sorted(pairs, key=lambda x: (-x[0], -x[1], x[2])):
        if shared and out[i] is None and place not in taken:
            out[i] = place
            taken.add(place)
    return out


def _background(reg: registration.Registered, cam: str, floor: Sequence[float]) -> places.Background | None:
    """A registered camera's background, cell by cell of its own frames: a camera whose frames were undistorted
    before MASt3R (a manifest lens) has its cells looked up through that lens."""
    manifest = lens.as_registered(cam)
    undo = None
    if manifest is not None:
        found = lens.Lens.from_tuple(manifest)

        def undo(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return lens.undistort_points(found, u, v)

    return registration.background(reg.maps[cam], reg.poses[cam], floor, undo)


def reproject(
    out: Path, group: Mapping[str, Any], held: Mapping[str, lens.Held] | None = None
) -> registration.Registered | None:
    """A group of cameras registered by reprojection from the runner's saved pair inferences
    (spatial/registration.py), each camera's `held` intrinsics kept, or None when that fails and MASt3R's alignment
    stands."""
    cams = [str(c) for c in group["cameras"]]
    with np.load(out / group["pairs"]) as saved:
        arrays = {k: saved[k] for k in saved.files}
    pairs = {
        (a, b): registration.Pair(*(arrays[f"{a}__{b}__{f}"] for f in registration.PAIR_FIELDS))
        for a in cams
        for b in cams
        if a != b and f"{a}__{b}__own" in arrays
    }
    images = {c: arrays[f"{c}__image"] for c in cams}
    sizes = {c: [int(s) for s in group["sizes"][c]] for c in cams}
    mast3r = {c: np.asarray(group["poses"][c], dtype=np.float64).reshape(4, 4) for c in cams}
    pinhole = [c for c in cams if lens.as_registered(c) is not None]
    focals = {c: float(group["focals"][c]) for c in cams}
    from liberational.spatial import mast3r_runner

    # The overlays and people frames() left out (#3085), as the runner leaves them out of its cloud
    folder = out / str(group.get("folder", ""))  # a retried camera's frames are in their own folder
    covered = {c: mast3r_runner.overlay(str(folder / f"{c}.png"), sizes[c][1], sizes[c][0], MAST3R_SIZE) for c in cams}
    kept = {c: (h.focal, h.division) for c, h in (held or {}).items() if c in cams}
    found = registration.register(cams, pairs, sizes, images, mast3r, focals, pinhole, covered, kept)
    if found is None:
        LOG.info("%s: no reprojection fit; MASt3R's alignment stands", ", ".join(cams))
    return found


def refused_seats(state: Path) -> dict[str, dict[str, Any]]:
    """Each camera whose seat was refused on the last runs: {"place": the place it was refused by, "nights": how
    many runs in a row, "at": the last run's time}. Machine state; {} when there's none or it can't be read, and an
    entry not of that shape is left out, so every count starts again (a camera then needs MOVED_NIGHTS more)."""
    try:
        found = json.loads((places.root(state) / SEATS_FILE).read_text())
    except OSError, ValueError:
        return {}
    if not isinstance(found, dict):
        return {}
    return {
        str(k): v
        for k, v in found.items()
        if isinstance(v, dict)
        and isinstance(v.get("place"), str)
        and isinstance(v.get("nights"), int)
        and isinstance(v.get("at"), int | float)
    }


def _save_seats(state: Path, seats: Mapping[str, Mapping[str, Any]]) -> None:
    path = places.root(state) / SEATS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    places.write_whole(path, json.dumps(seats, sort_keys=True).encode())


def refused(seats: dict[str, dict[str, Any]], cams: Sequence[str], place: str, now: float) -> int:
    """Count one more refused seat for each of `cams` in `place` (in `seats`, in place), once per run (`now`), and
    return how many runs in a row the group has been refused: the fewest of its cameras'."""
    for c in cams:
        was = seats.get(c)
        nights = int(was.get("nights", 0)) if was and was.get("place") == place else 0
        if was is None or was.get("at") != now or was.get("place") != place:
            nights += 1
        seats[c] = {"place": place, "nights": nights, "at": now}
    return min(int(seats[c]["nights"]) for c in cams)


def matched_before(state: Path, place: str, cams: Sequence[str], present: Iterable[str]) -> bool:
    """Whether `cams` were once one MASt3R group with other cameras of `place` that had a frame this run (`present`):
    a fixed-camera snapshot of the place holds some of them beside such a camera. A room of a joined place (#2997)
    never was, since its cameras never matched the other rooms'; a camera of one room that stopped matching the rest
    was (#3146). A partner with no frame tonight couldn't have been matched, so it says nothing (#3173)."""
    mine, here = set(cams), set(present)
    return any(
        r.producer == PRODUCER and mine & set(r.cameras) and (set(r.cameras) - mine) & here
        for r in places.revisions(state, place)
    )


@dataclass(frozen=True)
class Snapshot:
    """One group's cloud from a run, as `write_places` writes or seats it: the cloud in the run's frame, the group's
    cameras posed in it, its floor and up, and the group's walked footprints (`seen`)."""

    points: np.ndarray
    colors: np.ndarray
    poses: dict[str, places.CameraPose]
    floor: tuple[float, ...] | None
    up: tuple[float, float, float]
    seen: list[footprints.Cells]


def snapshot(
    group: Mapping[str, Any],
    out: Path,
    now: float,
    seen: list[footprints.Cells],
    gravity: Mapping[str, Gravity] | None = None,
    seated: bool = False,
    place: str | None = None,
    held: Mapping[str, lens.Held] | None = None,
) -> Snapshot | None:
    """A group of the runner's answer as a snapshot (steps 2a and 2b): its cloud and poses registered again by
    reprojection when it has two or more cameras, its floor, and a lone camera corrected by GeoCalib's `gravity`
    unless it is `seated` into its place's map. A camera's `held` intrinsics (spatial/lens.py, #3289) are kept: the
    reprojection fits only poses, and a lone camera takes its held focal and lens. None when a lone camera's floor is
    implausible (it keeps its last revision; `place` names it in the log)."""
    cams = [str(c) for c in group["cameras"]]
    with np.load(out / group["cloud"]) as cloud:
        points, colors = cloud["points"], cloud["colors"]
    poses = {
        c: places.CameraPose(
            tuple(float(x) for x in group["poses"][c]),
            float(group["focals"][c]),
            int(group["sizes"][c][0]),
            int(group["sizes"][c][1]),
            now,
            lens.as_registered(c),  # the lens frames() undid before MASt3R saw the frame (#2998)
        )
        for c in cams
    }
    # Cameras that see each other's space: registered by reprojection, each with its own lens (#3069), else as
    # MASt3R's alignment left them
    reg = reproject(out, group, held) if len(cams) > 1 and group.get("pairs") else None
    if reg is not None:
        points, colors = reg.points, reg.colors
        poses = {
            c: registration.camera_pose(reg.poses[c], reg.focals[c], reg.divisions[c], group["sizes"][c], now)
            if poses[c].lens is None
            else replace(poses[c], pose=tuple(float(x) for x in reg.poses[c].ravel()), focal=reg.focals[c])
            for c in cams
        }
    # A lone camera's up comes from its surfaces (the runner's `up`): one camera may be pitched far down.
    listed = {c: list(p.pose) for c, p in poses.items()}
    up = tuple(float(x) for x in group["up"]) if "up" in group else up_direction(listed, group["sizes"])
    up3 = (up[0], up[1], up[2])
    floor: Sequence[float] | None
    if reg is not None:
        measured = rig.load().heights()
        heights = {c: measured[c] for c in cams if c in measured}
        floor = registration.floor(points[: reg.own], up3, reg.poses, heights)
        if floor is not None:  # what each camera sees behind people, for footprints.at_edge (#3070)
            poses = {c: replace(p, background=_background(reg, c, floor)) for c, p in poses.items()}
    else:
        floor = group.get("floor") or fit_floor(points, up3)
    if "floor" in group and len(cams) == 1:  # a lone camera: its floor from GeoCalib's up (#2960)
        floor = lone_floor(cams[0], poses[cams[0]], points, floor, (gravity or {}).get(cams[0]))
        if floor is not None:
            up3 = (float(floor[0]), float(floor[1]), float(floor[2]))
            why = implausible(cams[0], poses[cams[0]], floor)
            if why is not None:
                LOG.warning("%s: %s; %s keeps its last revision", cams[0], why, place or "its new place")
                return None
    # GeoCalib corrects a lone camera written as its own place only: cameras registered together, and a group
    # seated by its cloud, keep MASt3R's poses (step 2b, #2960, #3228)
    held_by = "registered with other cameras" if len(cams) > 1 else "seated by its cloud" if seated else None
    poses = correct(poses, floor, gravity or {}, seen, held_by)
    if reg is None:  # a lone camera, or a group MASt3R's alignment left: its held intrinsics replace tonight's
        poses = {c: with_held(c, p, (held or {}).get(c)) for c, p in poses.items()}
    kept = None if floor is None else tuple(float(x) for x in floor)
    return Snapshot(points, colors, poses, kept, up3, seen)


def with_held(camera: str, pose: places.CameraPose, held: lens.Held | None) -> places.CameraPose:
    """`camera`'s `pose` with its `held` intrinsics in place of its own (#3289): the focal, and the lens unless its
    frames are undone by its configured lens first (`lens.as_registered`; the undone frame is a pinhole)."""
    if held is None:
        return pose
    if lens.as_registered(camera) is not None:
        return replace(pose, focal=held.focal * pose.width)
    m = np.asarray(pose.pose, dtype=np.float64).reshape(4, 4)
    found = registration.camera_pose(m, held.focal * pose.width, held.division, (pose.width, pose.height), pose.at)
    return replace(found, background=pose.background)


def write_places(
    state: Path,
    result: Mapping[str, Any],
    out: Path,
    now: float,
    absent: Sequence[str] = (),
    feet: Callable[[Sequence[str]], Sequence[footprints.Cells]] | None = None,
    gravity: Mapping[str, Gravity] | None = None,
) -> list[places.Revision]:
    """Each group in the runner's answer as a revision of its place. A place keeps its last map when its group lost
    cameras only because they had no usable frame this run (`absent`): a poorer snapshot isn't a reason to replace a
    better one. A camera that had a frame and no longer matches has moved, so the place is rewritten without it.
    GeoCalib's `gravity` corrects a lone camera's pitch, roll, and focal (`correct`, #2960)."""
    from liberational.spatial import fuse  # fuse uses this module's layout

    written = []
    groups = [[str(c) for c in g["cameras"]] for g in result.get("groups", [])]
    registered = places.camera_places(state)
    holding = lens.held(state)  # each camera's own intrinsics, kept by every registration (#3289)
    seats = refused_seats(state)
    counted = dict(seats)
    for group, cams, place in zip(result.get("groups", []), groups, owners(state, groups), strict=True):
        was = places.latest(state, place) if place else None
        # Some of a place's own cameras, the rest registered there too: seated into its map by their poses (#3115)
        home = {c for c, p in registered.items() if p == place}
        seated = place is not None and set(cams) < home and fuse.fused_revision(state, place) is not None
        missing = set(was.signals.get("camera", [])) - set(cams) if was else set()
        if missing & set(absent) and not seated:
            LOG.info("kept %s's map: %s had no usable frame", place, ", ".join(sorted(missing & set(absent))))
            continue
        signals = {**(was.signals if was else {}), "camera": cams}
        known = places.get(state, place) if place else None
        title = (known.name if known else "") or "Fixed cameras"
        seen_feet = list(feet(cams)) if feet is not None else []
        snap = snapshot(group, out, now, seen_feet, gravity, seated, place, holding)
        if snap is None:
            continue
        points, colors, poses, floor, up3, seen = snap.points, snap.colors, snap.poses, snap.floor, snap.up, snap.seen
        together = len(cams) > 1
        if seated and place is not None:
            rev = fuse.seat(state, place, points, colors, floor, poses, now) if floor is not None else None
            if rev is not None:
                written.append(rev)
                for c in cams:
                    seats.pop(c, None)
                continue
            # Cameras that matched the place's others before and don't tonight have moved out of their room (#3146);
            # a room of a join that doesn't seat may be a bad snapshot, so it has MOVED_NIGHTS runs to seat again.
            split = floor is not None and matched_before(state, place, cams, (c for g in groups for c in g))
            nights = refused(seats, cams, place, now) if floor is not None and not split else 0
            if not split and nights < MOVED_NIGHTS:
                LOG.info("%s keeps its map: %s didn't seat in it (%d runs in a row)", place, ", ".join(cams), nights)
                continue
            # The cameras moved, to a room no other camera sees. They're registered on their own (a new place, from
            # tonight's snapshot), and the old place is rewritten without them.
            LOG.warning(
                "%s %s: moved; registered on their own as a new place, and %s rewritten without them",
                ", ".join(cams),
                "no longer match the rest and didn't seat"
                if split
                else f"didn't seat in {place} for {nights} runs in a row",
                place,
            )
            dropped = fuse.release(state, place, cams, now)
            if dropped is not None:
                written.append(dropped)
            for c in cams:
                seats.pop(c, None)
            place, signals, title = None, {"camera": cams}, "Fixed cameras"
            if not together:  # a place of its own now: GeoCalib corrects it as any lone camera's
                poses = correct(poses, floor, gravity or {}, seen)
        found = layout(
            points,
            up3,
            title,
            (lambda floor, seen=seen, poses=poses: footprints.project(seen, poses, floor)) if seen else None,
            floor=floor,
        )
        written.append(
            places.revise(
                state,
                place,
                producer=PRODUCER,
                signals=signals,
                map_bytes=ply.dumps(points, colors),
                map_format="ply",
                plan=found.plan,
                plan_transform=found.transform,
                floor=found.floor,
                cameras=poses,
                now=now,
            )
        )
    # A run in which a camera was tried and not refused (its group seated another way, or had no floor) breaks its
    # count: the refusals have to be in a row (#3173). A camera with no frame tonight keeps it.
    for c in {c for g in groups for c in g}:
        if c in seats and seats[c].get("at") != now:
            seats.pop(c)
    if seats != counted:
        _save_seats(state, seats)
    return written


def retry_floorless(
    state: Path,
    result: Mapping[str, Any],
    out: Path,
    now: float,
    picked: Mapping[str, float],
    gpu: str | None,
    detect: Detector | None = None,
    read: Reader = recorded,
    zone: tzinfo | None = None,
) -> tuple[dict[str, Any], dict[str, Gravity]]:
    """The runner's answer with each lone camera it found no floor for (`floorless`: its lens blocked, say) tried
    again in the same run, with its best frame later than the one it was tried with that shows its view now
    (`frames` with `after`), in its own folder `late/` under `out`; and GeoCalib's gravity for those frames. A camera
    uncovered or moved after midday registers that night, not the next. The answer as it was, and no gravity, when
    no camera is floorless or none has a later lit frame."""
    floorless = [str(c) for c in result.get("floorless", []) if c in picked]
    if not floorless:
        return dict(result), {}
    late = out / "late"
    late.mkdir(parents=True, exist_ok=True)
    again = frames(state, floorless, late, now, read, zone, detect=detect, after=picked)
    if not again:
        LOG.info("%s: no floor at midday and no later lit frame to try", ", ".join(floorless))
        return dict(result), {}
    LOG.info("%s: no floor on the frame tried; trying again with a later frame", ", ".join(sorted(again)))
    second = run_mast3r(cfg.path("spatial.runtime"), again, late, gpu, lone_matches=LONE_MATCHES)
    groups = [
        {
            **g,
            "folder": "late",
            "cloud": f"late/{g['cloud']}",
            **({"pairs": f"late/{g['pairs']}"} if g.get("pairs") else {}),
        }
        for g in second.get("groups", [])
    ]
    still = [c for c in result.get("floorless", []) if c not in again] + list(second.get("floorless", []))
    merged = {**result, "groups": [*result.get("groups", []), *groups], "floorless": still}
    return merged, run_geocalib(cfg.path("spatial.geocalib_runtime"), again, late / "geocalib")


@dataclass(frozen=True)
class Capture:
    """A run's steps 1 and 2 (`capture`): the runner's answer, GeoCalib's gravity for each frame, the indoor cameras
    with no usable frame, and how many had one."""

    result: dict[str, Any]
    gravity: dict[str, Gravity]
    absent: list[str]
    cameras: int


def capture(state: Path, indoor: Sequence[str], out: Path, now: float) -> Capture | None:
    """Frames and MASt3R (steps 1 and 2) for the `indoor` cameras, in the folder `out`: each camera's best frame, the
    runner's groups, GeoCalib's gravity, and a lone camera with no floor tried again with a later frame. Reads the
    recordings and writes only under `out`, so a dry run can use it (tools/rebuild_place.py). None when no camera had
    a frame."""
    out.mkdir(parents=True, exist_ok=True)
    detect = detector()
    picked: dict[str, float] = {}
    images = frames(state, indoor, out, now, detect=detect, picked=picked)
    if not images:
        return None
    gpu = pick_gpu(int(cfg.get("spatial.gpu")), nvml.gpus())
    result = run_mast3r(cfg.path("spatial.runtime"), images, out, gpu, lone_matches=LONE_MATCHES)
    gravity = run_geocalib(cfg.path("spatial.geocalib_runtime"), images, out / "geocalib")
    # A lone camera with no floor on its frame (a blocked lens) is tried again with a later one
    result, later = retry_floorless(state, result, out, now, picked, gpu, detect)
    return Capture(result, {**gravity, **later}, [c for c in indoor if c not in images], len(images))


def run(state: Path, now: float | None = None) -> dict[str, Any]:
    """One pass: footprints, frames, MASt3R, places, then the position layer. Returns what it did, for the log."""
    now = time.time() if now is None else now
    try:
        did = _places_pass(state, now)
    finally:
        # Last, so the day's footprints are placed in the frame of the plan this run leaves the place drawing; and
        # whatever became of the places, so a failed registration doesn't hold the day's paths back.
        placed = _place_positions(state, now)
    return {**did, "positions": placed}


def _place_positions(state: Path, now: float) -> dict[str, int] | None:
    """The position layer brought up to where the footprints are recorded (`positions.advance`); None when it
    failed, which the next run makes up for from its mark."""
    from liberational.spatial import positions

    try:
        return positions.advance(state, now - footprints.FLUSH_S)
    except Exception:  # the layer is behind until the next run; the places this run made are still good
        LOG.exception("placing positions failed")
        return None


def _places_pass(state: Path, now: float) -> dict[str, Any]:
    """Footprints, frames, MASt3R, and places."""
    cams = fixed_cameras(cfg.ROOT / "config" / "devices.toml")
    # Footprints first (#2800): every fixed camera's, registered or not, so a place has them by the time it has poses.
    # The log's footprints_recorded is what this run added since each camera's mark (a day's backfill the first time,
    # then only the new rows), not what a plan reads: the walkable floor reads the last 14 days back on every run.
    recorded = footprints.record(state, cams, now)
    # Outdoor cameras (mount.outdoor) see no indoor place: they'd only become places of their own.
    indoor = indoor_cameras(cams, rig.load())
    if not indoor:
        return {"cameras": 0, "places": 0, "why": "no indoor fixed camera", "footprints_recorded": recorded}
    with tempfile.TemporaryDirectory(prefix="spatial-cameras-") as tmp:
        out = Path(tmp)
        got = capture(state, indoor, out, now)
        if got is None:
            why = "no camera had a frame"
            return {"cameras": 0, "places": 0, "why": why, "footprints_recorded": recorded}
        result, gravity, absent = got.result, got.gravity, got.absent
        # Places are written and fused under the lock a walkthrough's join holds too (#3173)
        with places.writing(state):
            revs = write_places(
                state, result, out, now, absent, lambda group: footprints.walked(state, group, now), gravity
            )
            # Tier 1 (#2801): each place's newest revision fused into its map, this run's and any missed before.
            from liberational.spatial import fuse  # fuse uses this module's layout

            fused = fuse.refine_all(state)
    # A splat of each place's newest map (#2877), from a day of its cameras' recorded frames.
    from liberational.spatial import splats

    splatted = splats.splat_all(state, now) if cfg.get("spatial.splats_enabled") else []
    return {
        "fused": sorted(r.place for r in fused),
        "splats": [s["place"] for s in splatted],
        "cameras": got.cameras,
        "matches": result.get("matches", []),
        "floorless": result.get("floorless", []),
        "places": [{"place": r.place, "cameras": sorted(r.cameras), "plan": r.plan is not None} for r in revs],
        "footprints_recorded": recorded,
        "mast3r_s": result.get("seconds"),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pack-spatial-cameras", description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)
    logs.setup()
    if not cfg.get("spatial.cameras_enabled"):
        LOG.info("fixed-camera places are off ([spatial] cameras_enabled)")
        return
    if sys.platform == "darwin":
        LOG.info("fixed-camera places need CUDA; skipping on macOS")
        return
    began = time.monotonic()
    try:
        did = run(cfg.path("paths.state"))
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        LOG.error("fixed-camera places failed: %s", e)
        sys.exit(1)
    LOG.info("fixed-camera places: %s in %.0f s", json.dumps(did), time.monotonic() - began)


if __name__ == "__main__":
    main()
