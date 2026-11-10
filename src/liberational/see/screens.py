"""People on a screen: a face or body a TV or laptop shows is not someone in the room.

The person detector scores every COCO class in the same inference that finds people, so each frame carries the
screens it saw (`People.screens`: tv and laptop boxes). A camera's zones can also mark a screen by hand (a zone of
kind "screen", for one the detector misses, such as a projector's wall). A detection whose box lies mostly inside a
screen (SCREEN_INSIDE of its area), and is small beside a detected screen (at most SCREEN_SHARE_MAX of its area), is
on it.

Tracks, not detections, are kept out. A track born on a screen is a screen track: none of its frames reach faces,
shots, face groups, actions, the Live overlay, pose rows, movement, or crossings. It becomes a real track for good
the first time its box lies mostly outside every screen, since what a screen shows never leaves it, while someone
sitting in front of the TV (whose head and shoulders can sit inside its box from a camera behind the couch) stands,
leans, or walks out of it sooner or later. A real track that later sits in front of a screen stays real. Recordings
are untouched, so the people a screen shows are still in the video, only not counted as anyone.
"""

import json
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from liberational import json_state
from liberational.see.zones import contains

Point = tuple[float, float]
Box = Sequence[float] | NDArray[np.floating]  # x1, y1, x2, y2 (and maybe more), normalized to the frame

SCREEN_INSIDE = 0.8  # a box at least this much inside a screen is on it
# ... and at most this big beside the screen's own box. Someone a screen shows is small on it: a TV evening on cam0
# had its people at 0.26 to 0.40 of the TV's box, while the back of a real head in front of a laptop, inside its box
# from the camera behind, was 0.62 to 0.77 of it. A zone drawn as a screen has no such limit.
SCREEN_SHARE_MAX = 0.5
SCREEN_PAD = 0.01  # a screen's box is padded by this share of the frame on each side: the detector's edges wobble
GRID = 5  # a box is sampled at GRID x GRID points to measure how much of it a zone's polygon covers


def in_box(box: Box, screen: Box) -> float:
    """How much of `box` (x1, y1, x2, y2, normalized) lies inside `screen` (padded by SCREEN_PAD), 0 to 1."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    sx1, sy1, sx2, sy2 = (float(v) for v in screen[:4])
    sx1, sy1, sx2, sy2 = sx1 - SCREEN_PAD, sy1 - SCREEN_PAD, sx2 + SCREEN_PAD, sy2 + SCREEN_PAD
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    w = max(0.0, min(x2, sx2) - max(x1, sx1))
    h = max(0.0, min(y2, sy2) - max(y1, sy1))
    return w * h / area


def share(box: Box, screen: Box) -> float:
    """The box's area as a share of the screen's (unpadded)."""
    area = (float(screen[2]) - float(screen[0])) * (float(screen[3]) - float(screen[1]))
    return (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])) / area if area > 0 else math.inf


def in_polygon(box: Box, polygon: Sequence[Point]) -> float:
    """How much of `box` lies inside the polygon, by a GRID x GRID sample of points across it, 0 to 1."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    steps = [(i + 0.5) / GRID for i in range(GRID)]
    hits = sum(contains(polygon, x1 + (x2 - x1) * u, y1 + (y2 - y1) * v) for u in steps for v in steps)
    return hits / (GRID * GRID)


def on_screen(
    boxes: NDArray[np.float32], screens: NDArray[np.float32], polygons: Sequence[Sequence[Point]] = ()
) -> NDArray[np.bool_]:
    """For each person box (N x >= 4, normalized), whether it lies mostly inside a screen box it is small beside, or a
    screen zone."""
    out = np.zeros(len(boxes), bool)
    shown: list[list[float]] = screens.tolist()
    for i, box in enumerate(boxes.tolist()):
        out[i] = any(in_box(box, s) >= SCREEN_INSIDE and share(box, s) <= SCREEN_SHARE_MAX for s in shown) or any(
            in_polygon(box, p) >= SCREEN_INSIDE for p in polygons
        )
    return out


@dataclass
class Counts:
    """What the mask did since the last report, for the log."""

    detections: int = 0  # person detections assigned to a track
    inside: int = 0  # of them, inside a screen
    masked: int = 0  # track-frames kept out, as a screen track's
    born: int = 0  # screen tracks born
    released: int = 0  # screen tracks found to be real


RELEASE_NS = 10**9  # a screen track is real once its box has been outside every known screen this long
SCREEN_HOLD_NS = 2 * 10**9  # a detected screen is still there this long after the detector last found it


@dataclass
class ScreenTracks:
    """Which of a camera's tracks are on a screen (see the module docstring). A track is released only on evidence
    of leaving: its box outside every known screen for RELEASE_NS, not one frame where the detector missed the
    screen (`known` False, no screen held or drawn, says nothing either way)."""

    screen: dict[int, bool] = field(default_factory=dict)  # track id: still a screen track
    outside_ns: dict[int, int] = field(default_factory=dict)  # a screen track's first frame outside, since
    released_ns: dict[int, int] = field(default_factory=dict)  # when a screen track was found to be real
    counts: Counts = field(default_factory=Counts)

    def update(
        self, assigned: dict[int, int], inside: NDArray[np.bool_], capture_ns: int, known: bool = True
    ) -> set[int]:
        """This frame's tracks (detection: track id), which detections are inside a screen, and whether any screen
        is known: the screen tracks among them now."""
        out = set()
        for det, track_id in assigned.items():
            on = bool(inside[det])
            self.counts.detections += 1
            self.counts.inside += on
            was = self.screen.get(track_id)
            if was is None:
                self.screen[track_id] = on
                self.counts.born += on
            elif was and on:
                self.outside_ns.pop(track_id, None)
            elif was and known:
                since = self.outside_ns.setdefault(track_id, capture_ns)
                if capture_ns - since >= RELEASE_NS:
                    self.screen[track_id] = False
                    self.released_ns[track_id] = since
                    self.outside_ns.pop(track_id, None)
                    self.counts.released += 1
            if self.screen[track_id]:
                out.add(track_id)
                self.counts.masked += 1
        return out

    def is_screen(self, track_id: int) -> bool:
        return self.screen.get(track_id, False)

    def released_at(self, track_id: int) -> int | None:
        """When a track born on a screen was found to be real (its first frame outside), or None."""
        return self.released_ns.get(track_id)

    def end(self, track_id: int) -> None:
        self.screen.pop(track_id, None)
        self.outside_ns.pop(track_id, None)
        self.released_ns.pop(track_id, None)

    def clear(self) -> None:
        self.screen.clear()
        self.outside_ns.clear()
        self.released_ns.clear()

    def take(self) -> Counts:
        """The counts since the last take, and start new ones."""
        counts, self.counts = self.counts, Counts()
        return counts


@dataclass
class HeldScreens:
    """The screens a camera saw lately: one the detector misses for a frame or two is still there, for
    SCREEN_HOLD_NS after it last found it."""

    held: list[tuple[NDArray[np.float32], int]] = field(default_factory=list)  # (screen row, last found)

    def update(self, found: NDArray[np.float32], capture_ns: int) -> NDArray[np.float32]:
        """This frame's detected screens (N x 6): those, and the ones found within SCREEN_HOLD_NS before."""
        kept = [(row, ns) for row, ns in self.held if capture_ns - ns < SCREEN_HOLD_NS]
        for row in found:
            i = next((i for i, (old, _) in enumerate(kept) if iou(old, row) >= SAME_SCREEN), None)
            if i is None:
                kept.append((row.copy(), capture_ns))
            else:
                kept[i] = (row.copy(), capture_ns)
        self.held = kept
        return np.stack([row for row, _ in kept]) if kept else np.zeros((0, 6), np.float32)

    def clear(self) -> None:
        self.held.clear()


# ---- whether a screen is on: a screen box whose pixels keep changing ----

SAMPLE_NS = 500_000_000  # each screen's pixels are compared this often
THUMB = (24, 16)  # a screen is compared as a grey thumbnail this size (width, height)
CHANGE = 0.02  # a mean change of at least this share of full scale between samples is a moving picture
WINDOW = 6  # samples looked back over (3 s)
MOVING = 3  # of which at least this many moved: the screen is on
OFF_NS = 10 * 10**9  # off for this long, the span ends: a still scene or a pause between shows
SAME_SCREEN = 0.5  # a box overlapping a remembered screen this much (IoU) is that screen
MIN_ON_NS = 30 * 10**9  # a span is written once it has lasted this long: a show runs minutes, while people passing an
# off TV, or its reflections, gave cam0 bursts of 1 to 25 s
PLAYING = 62  # the class a screen zone's box is watched as: a TV


def playing(zones: Sequence[Sequence[Point]] = ()) -> NDArray[np.float32]:
    """The screens to watch for a moving picture: each screen zone the owner drew, by its bounding box. A detected TV
    isn't one: the detector's TV class also finds computer monitors, and a monitor in use changes as a show does
    (cam0's desk monitor and a patio reflection made 39 of one day's 41 screen-on minutes)."""
    rows = []
    for z in zones:
        xs, ys = [p[0] for p in z], [p[1] for p in z]
        rows.append(np.array([min(xs), min(ys), max(xs), max(ys), 1.0, PLAYING], np.float32))
    return np.stack(rows) if rows else np.zeros((0, 6), np.float32)


def thumbnail(patch: NDArray[np.uint8]) -> NDArray[np.float32] | None:
    """A screen's patch (BGR, or grey) as a small grey picture, 0 to 1, or None for an empty patch."""
    import cv2

    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return None
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return cv2.resize(grey, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


VISIBLE = 1 / 3  # a screen sample counts only when at least this much of it is clear of people in the room


def covered(screen: Box, real: Sequence[Box]) -> NDArray[np.bool_]:
    """Which cells of a screen's THUMB grid someone in the room is in front of: a person passing a dark TV, or typing
    at a laptop, changes its pixels as a picture does (on cam0, a laptop with someone at it changed in 92% of
    samples, the TV playing in 73%, the TV off in 18%). `real` is the person boxes not on a screen: those a TV shows
    don't block it."""
    w, h = THUMB
    x1, y1, x2, y2 = (float(v) for v in screen[:4])
    xs = x1 + (np.arange(w) + 0.5) / w * (x2 - x1)
    ys = y1 + (np.arange(h) + 0.5) / h * (y2 - y1)
    out = np.zeros((h, w), bool)
    for b in real:
        out |= (ys[:, None] >= b[1]) & (ys[:, None] <= b[3]) & (xs[None, :] >= b[0]) & (xs[None, :] <= b[2])
    return out


def iou(a: Box, b: Box) -> float:
    w = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    h = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = w * h
    union = (
        (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
        + (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
        - inter
    )
    return inter / union if union > 0 else 0.0


@dataclass
class _Screen:
    box: tuple[float, float, float, float]
    last: NDArray[np.float32] | None = None
    moved: list[bool] = field(default_factory=list)
    seen_ns: int = 0


@dataclass
class ScreenActivity:
    """One camera's screens over time: `sample` each one's patch about every SAMPLE_NS, and the spans when any was
    on (its picture moving in MOVING of the last WINDOW samples) come out of `spans`. A span stays open across
    OFF_NS of stillness or a missed screen, and ends after that."""

    screens: list[_Screen] = field(default_factory=list)
    open_ns: tuple[int, int] | None = None  # the open span: (start, last on)
    closed: list[tuple[int, int]] = field(default_factory=list)
    sampled_ns: int | None = None

    def due(self, capture_ns: int) -> bool:
        return self.sampled_ns is None or capture_ns - self.sampled_ns >= SAMPLE_NS

    def sample(
        self, capture_ns: int, patches: Sequence[tuple[Box, NDArray[np.uint8]]], real: Sequence[Box] = ()
    ) -> bool:
        """This frame's screens (normalized box, pixel patch), and the people in the room (`covered`): whether any
        screen is on now. Only the cells clear of people are compared; a screen mostly hidden is skipped."""
        self.sampled_ns = capture_ns
        on = False
        for box, patch in patches:
            thumb = thumbnail(patch)
            if thumb is None:
                continue
            hidden = covered(box, real)
            if 1 - hidden.mean() < VISIBLE:
                continue
            screen = next((s for s in self.screens if iou(s.box, box) >= SAME_SCREEN), None)
            if screen is None:
                screen = _Screen((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                self.screens.append(screen)
            if screen.last is not None and screen.last.shape == thumb.shape:
                change = float(np.abs(thumb - screen.last)[~hidden].mean())
                screen.moved = [*screen.moved[-(WINDOW - 1) :], change >= CHANGE]
            screen.last, screen.seen_ns = thumb, capture_ns
            on = on or sum(screen.moved) >= MOVING
        # A screen unseen for OFF_NS is forgotten (the camera moved, or it was a mistaken box).
        self.screens = [s for s in self.screens if capture_ns - s.seen_ns < OFF_NS]
        if on:
            self.open_ns = (self.open_ns[0] if self.open_ns else capture_ns, capture_ns)
        elif self.open_ns is not None and capture_ns - self.open_ns[1] >= OFF_NS:
            self.closed.append(self.open_ns)
            self.open_ns = None
        return on

    def spans(self) -> list[tuple[int, int]]:
        """Every span closed since the last call, and the open one so far, that lasted at least MIN_ON_NS."""
        out, self.closed = self.closed, []
        found = [*out, self.open_ns] if self.open_ns is not None else out
        return [s for s in found if s[1] - s[0] >= MIN_ON_NS]

    def gap(self) -> None:
        """Frames are missing: the open span ends at the last frame it was on."""
        if self.open_ns is not None:
            self.closed.append(self.open_ns)
            self.open_ns = None
        for s in self.screens:
            s.last, s.moved = None, []


KEEP_S = 24 * 3600.0  # the screen spans file keeps this long a history: tagging reads the last minutes
WRITE_S = 5.0  # it is written at most this often


class ScreensOn:
    """`<state>/screens_on.json`: when each camera's screen was on, the last KEEP_S, as {"cams": {cam: [[start, end],
    ...] (seconds)}, "at": when written}. Shared by every camera of the process; written atomically."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path, self.clock = path, clock
        self.lock = threading.Lock()
        self.cams: dict[str, list[list[float]]] = _read(path)
        self.written = -math.inf

    def add(self, cam: str, spans: Sequence[tuple[int, int]], flush: bool = False) -> None:
        """A camera's spans (ns) closed or still open: merged into its history, and written when due, or at once with
        `flush` (a span closed)."""
        if not spans:
            return
        now = self.clock()
        with self.lock:
            self.cams = {c: [s for s in spans if s[1] >= now - KEEP_S] for c, spans in self.cams.items()}
            self.cams = {c: spans for c, spans in self.cams.items() if spans}
            kept = self.cams.get(cam, [])
            for start_ns, end_ns in spans:
                start, end = start_ns / 1e9, end_ns / 1e9
                if kept and start <= kept[-1][1] + OFF_NS / 1e9:
                    kept[-1][1] = max(kept[-1][1], end)  # the same span, grown (or one that came back soon)
                else:
                    kept.append([start, end])
            self.cams[cam] = kept
            if now - self.written < WRITE_S and not flush:
                return
            self.written = now
            body = json.dumps({"cams": self.cams, "at": now}, separators=(",", ":"))
            # Written under the lock: two cameras writing at once would race on the file, and the slower
            # one's older map would replace the newer.
            try:
                json_state.write_text(self.path, body)
            except OSError:
                pass  # the next span writes it again


def _read(path: Path) -> dict[str, list[list[float]]]:
    try:
        cams = json.loads(path.read_text()).get("cams") or {}
    except OSError, ValueError, AttributeError:
        return {}
    return {
        str(c): [[float(a), float(b)] for a, b in spans]
        for c, spans in cams.items()
        if isinstance(spans, list) and all(isinstance(s, list) and len(s) == 2 for s in spans)
    }


def on_during(path: Path, start: float, end: float, pad_s: float = 5.0) -> bool:
    """Whether any camera's screen was on during [start, end] (seconds), give or take pad_s."""
    return any(a - pad_s <= end and b + pad_s >= start for spans in _read(path).values() for a, b in spans)
