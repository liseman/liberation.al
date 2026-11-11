"""see-actions: people, their skeletons, and their tracks at full frame rate on the main camera, for the
action model.

This module is the per-frame path, apart from where frames come from, what hardware runs the models, and where rows
go, so it tests without a camera, an accelerator, or a store. It never touches the accelerator itself: frames and
people go through a `vision_backend.Backend`, and only NumPy crosses that seam.

- `ActionPipeline.feed` takes a decoded frame (the shared full-rate decode), has the backend upload it and
  find its people and skeletons, follows them with the ByteTrack-style tracker the body worker uses, and buffers
  one pose row per tracked person. About once a second it appends the rows to the pose sink, and a coverage row
  (the span of frames it analyzed) to the coverage sink.
- `Scoring` keeps each track's last `CLIP` frames of joints and has the backend's action model score them
  every SCORE_EVERY_NS. Each score is an `action.window` row (its top classes), and an owner's activity
  (memory/activity_kinds.py) whose NTU classes add up to THRESHOLD or more over consecutive windows becomes
  one `action.span`.
- `action_clips.Clips`, when given, does the same with each track's person crops and MoViNet, for the still kinds
  a skeleton can't show (reading, watching TV, a phone, sleep).
- A gap in the frames (`gap_before`) ends every track, its windows and spans, and the coverage run, so nothing
  bridges frames that were never seen.

The sinks take columns (NumPy arrays, or lists for ids and strings) as the segment store's `SegmentWriter.append`
does, with the schemas below. Frame ids are the shared decode's UUIDs.
"""

import copy
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from liberational import logs
from liberational.memory import activity_kinds
from liberational.see import screens
from liberational.see.action_animals import Animals
from liberational.see.action_classes import CLASSES, CLIP
from liberational.see.action_movement import Movement
from liberational.see.vision_backend import Backend, Frame
from liberational.vision import crossings

LOG = logs.get("see-actions")
SCREEN_LOG_NS = 600 * 10**9  # how often each camera logs what its screen mask did

if TYPE_CHECKING:
    from liberational.see.action_clips import Clips
    from liberational.see.head_faces import HeadFaces

JOINTS = 26  # halpe26
SCALE = 10000  # joints are stored as int16: x and y normalized to the frame, and the confidence, times this
FLUSH_NS = 1_000_000_000  # rows go to the sinks about this often
PATH_SPACING_S = 0.2  # a track's path keeps a point about this often, for the crossing check when it ends
MAX_PATH = 3000  # and at most this many: 10 min of path
# A track counts once it has been matched in this many frames (SORT's min_hits): on cam0, 3 of 10 tracks in 30 s
# were blips of 2 to 4 frames, false boxes at the detector's threshold. 5 frames is 0.17 s at 30 fps.
MIN_HITS = 5
# ByteTrack's thresholds: the detector keeps boxes down to DETECT_SCORE so the tracker's low-score pass can
# hold a dim person's track (a seated, back-turned person scores 0.1 to 0.5), only a box at the tracker's `high`
# (0.5) starts one, and a track survives LOST_S unseen.
LOST_S = 5.0
SCORE_EVERY_NS = 500_000_000  # a track's latest window is scored this often
THRESHOLD = 0.5  # a kind's probability that counts as seen in a window
HOLD_NS = 5_000_000_000  # a span stays open this long after its last window that saw the kind, so a score
# flickering around THRESHOLD stays one span: closing on the first miss gave one patio track 105 spans an hour
EDGE = 0.005  # a box this close to the frame's edge is cut off by it: its pose is a part of a body, and isn't scored
STILL_NS = 120_000_000_000  # a box that stays within STILL_MOVE of where it was for this long isn't scored: an
STILL_MOVE = 0.05  # object the detector takes for a person, or someone asleep (the move is a fraction of box height)
MIN_JOINTS = 6  # a detection whose pose shows fewer of COCO's 17 joints than this above JOINT_SURE starts no track,
# and a window whose median frame does isn't scored: objects taken for people show none, people 13 to 15
JOINT_SURE = 0.5
UNSURE_NS = 5_000_000_000  # a track whose detections have shown no confident pose for this long ends: an object at
# the spot a person left kept their track for 20 minutes, while someone turned away is back within seconds
TOP = 5  # classes kept per window

POSE_SCHEMA = [
    ("ts_ns", "i64"),  # the frame's capture time; the segment store's time key
    ("frame_id", "id"),
    ("track_id", "i64"),
    ("box", "f32[5]"),  # x1, y1, x2, y2 normalized to the frame, and the detector's score
    ("joints", f"i16[{JOINTS * 3}]"),  # x, y, confidence per joint, times SCALE; a joint not placed is -1, -1, 0
    ("model", "str"),
]
WINDOW_SCHEMA = [
    ("ts_ns", "i64"),  # the window's first frame
    ("end_ns", "i64"),  # and its last
    ("track_id", "i64"),
    ("classes", f"i16[{TOP}]"),  # the model's class indices, most likely first: NTU-120 (CLASSES) for ST-GCN++,
    # Kinetics-600 (movinet_classes.CLASSES) for MODEL in see/action_clips.py
    ("scores", f"f32[{TOP}]"),
    ("model", "str"),
]
COVERAGE_SCHEMA = [
    ("ts_ns", "i64"),  # where the analyzed span starts
    ("end_ns", "i64"),  # the last analyzed frame's capture time; the next row of the same run starts here
    ("frames", "i64"),
    ("model", "str"),
]


class Sink(Protocol):
    """Where rows go: `continuum_py.SegmentWriter` in the service, a list in tests."""

    def append(self, columns: dict[str, Any], /) -> Any: ...


def quantize(joints: np.ndarray, width: int, height: int) -> np.ndarray:
    """N x K x 3 joints in frame pixels (as a backend's `People` holds them) as N x (K * 3) int16 of normalized x and y and
    the confidence, times SCALE. A joint the model didn't place stays -1, -1, 0."""
    out = np.empty(joints.shape, np.float64)
    out[..., 0] = joints[..., 0] / width
    out[..., 1] = joints[..., 1] / height
    out[..., 2] = joints[..., 2]
    missing = joints[..., 2] <= 0
    out[missing] = (-1.0, -1.0, 0.0)
    out[..., :2] = out[..., :2].clip(-1.0, 3.0)  # a joint can sit a little outside the frame; int16 holds to 3.27
    return np.rint(out * SCALE).astype(np.int16).reshape(len(joints), -1)


@dataclass
class _Run:
    """The coverage run being built: analyzed frames with no gap between them."""

    start_ns: int  # where the next coverage row starts: the run's first frame, then each row's end
    end_ns: int
    frames: int = 0


def normalized(joints: np.ndarray, width: int, height: int) -> np.ndarray:
    """N x K x 3 joints in frame pixels as x and y normalized to the frame, and the confidence."""
    out = joints.astype(np.float32).copy()
    out[..., 0] /= width
    out[..., 1] /= height
    out[joints[..., 2] <= 0] = 0
    return out


@dataclass
class _Span:
    track_id: int
    kind: str
    start_ns: int
    end_ns: int
    peak: float
    supported: list[tuple[int, int]] = field(default_factory=list)


class Spans:
    """Each track's open spans, one per kind: a window that sees the kind extends its span, and a span no window has
    seen for `hold_ns` closes, as a `segment_log` activity record. One seen by fewer than `min_seen` windows is
    dropped."""

    def __init__(
        self,
        describe: Callable[[_Span], str],
        who: Callable[[int], tuple[str | None, str | None]] | None = None,
        *,
        hold_ns: int = HOLD_NS,
        min_seen: int = 1,
    ) -> None:
        self.describe, self.who, self.hold_ns, self.min_seen = describe, who, hold_ns, min_seen
        self._open: dict[tuple[int, str], _Span] = {}
        self.closed: list[dict[str, Any]] = []  # until taken

    def seen(self, track_id: int, kind: str, start: int, end: int, p: float) -> None:
        span = self._open.get((track_id, kind))
        if span is None:
            self._open[(track_id, kind)] = _Span(track_id, kind, start, end, p, [(start, end)])
        else:
            span.end_ns, span.peak = end, max(span.peak, p)
            span.supported.append((start, end))

    def due(self, capture_ns: int) -> None:
        for key in [k for k, span in self._open.items() if capture_ns - span.end_ns > self.hold_ns]:
            self._close(key)

    def end(self, track_id: int) -> None:
        for key in [k for k in self._open if k[0] == track_id]:
            self._close(key)

    def tracks(self) -> set[int]:
        return {k[0] for k in self._open}

    def take(self) -> list[dict[str, Any]]:
        closed, self.closed = self.closed, []
        return closed

    def _close(self, key: tuple[int, str]) -> None:
        span = self._open.pop(key)
        if len(span.supported) < self.min_seen:
            return
        person, person_name = self.who(span.track_id) if self.who is not None else (None, None)
        self.closed.append(
            {
                "id": str(uuid.uuid7()),
                "ts_start": span.start_ns / 1e9,
                "ts_end": span.end_ns / 1e9,
                "track_id": str(span.track_id),
                "subject_type": "person",
                "species": None,
                "person": person,
                "person_name": person_name,
                "kind": span.kind,
                "attributes": {
                    "supported": [[a / 1e9, b / 1e9] for a, b in span.supported],
                    "observations": len(span.supported),
                },
                "uncertainty": {"interpretation": self.describe(span)},
            }
        )


class Windows:
    """Scored windows as `action.window` columns, until taken."""

    def __init__(self) -> None:
        self.rows: dict[str, list[Any]] = {name: [] for name, _ in WINDOW_SCHEMA}

    def add(self, track_id: int, start: int, end: int, p: np.ndarray, model: str) -> None:
        top = np.argsort(-p)[:TOP]
        self.rows["ts_ns"].append(start)
        self.rows["end_ns"].append(end)
        self.rows["track_id"].append(track_id)
        self.rows["classes"].append(top.astype(np.int16))
        self.rows["scores"].append(p[top].astype(np.float32))
        self.rows["model"].append(model)

    def take(self) -> dict[str, Any] | None:
        if not self.rows["ts_ns"]:
            return None
        columns = {
            "ts_ns": np.array(self.rows["ts_ns"], np.int64),
            "end_ns": np.array(self.rows["end_ns"], np.int64),
            "track_id": np.array(self.rows["track_id"], np.int64),
            "classes": np.stack(self.rows["classes"]),
            "scores": np.stack(self.rows["scores"]),
            "model": list(self.rows["model"]),
        }
        self.rows = {name: [] for name, _ in WINDOW_SCHEMA}
        return columns


def edge_clipped(box: np.ndarray) -> bool:
    """Whether a box (x1, y1, x2, y2 normalized to the frame) is cut off by the frame's edge."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return min(x1, y1) < EDGE or max(x2, y2) > 1 - EDGE


def confident(frames: deque[tuple[int, np.ndarray]]) -> float:
    """The median count, over a window's frames, of COCO joints (halpe26's first 17) above JOINT_SURE."""
    return float(np.median([(joints[:17, 2] > JOINT_SURE).sum() for _, joints in frames]))


@dataclass
class _Still:
    """Where a track's box was when it last moved by more than STILL_MOVE, and when."""

    x: float
    y: float
    since_ns: int


class Scoring:
    """Per-track windows of joints, scored by the action model, as window rows and spans."""

    def __init__(
        self,
        score: Callable[[np.ndarray], np.ndarray],
        model: str,
        who: Callable[[int], tuple[str | None, str | None]] | None = None,
        kinds: Callable[[], Mapping[str, tuple[int, ...]]] | None = None,
    ) -> None:
        """`who(track)` is the track's person at the time a span closes (the face stage's `HeadFaces.who`): their
        PersonId and the name as matched, or (None, None) for someone not recognized. `kinds()` is each activity's
        NTU class indices now (`activity_kinds.Live.ntu`); the starting list's without one."""
        self.score, self.model = score, model
        starting = activity_kinds.starting("ntu")
        self.kinds = kinds if kinds is not None else lambda: starting
        self._seen: dict[tuple[int, str], tuple[int, ...]] = {}  # the classes each open span was read from
        self._frames: dict[int, deque[tuple[int, np.ndarray]]] = {}
        self._scored: dict[int, int] = {}
        self._still: dict[int, _Still] = {}
        self._skip: set[int] = set()  # tracks not scored now: still, or cut off by the frame's edge
        self.windows = Windows()
        self.spans = Spans(
            lambda span: (
                f"{model}: NTU-120 "
                f"{' or '.join(repr(CLASSES[c]) for c in self._seen.pop((span.track_id, span.kind), ()))}, "
                f"peak {span.peak:.2f}"
            ),
            who,
        )

    def add(self, track_id: int, capture_ns: int, joints: np.ndarray, box: np.ndarray) -> None:
        """`box` is x1, y1, x2, y2 normalized to the frame (and the score, unused)."""
        frames = self._frames.setdefault(track_id, deque(maxlen=CLIP))
        frames.append((capture_ns, joints))
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        x, y, reach = (x1 + x2) / 2, (y1 + y2) / 2, STILL_MOVE * (y2 - y1)
        still = self._still.get(track_id)
        if still is None or abs(x - still.x) > reach or abs(y - still.y) > reach:
            still = self._still[track_id] = _Still(x, y, capture_ns)
        if edge_clipped(box) or capture_ns - still.since_ns >= STILL_NS:
            self._skip.add(track_id)
        else:
            self._skip.discard(track_id)

    def due(self, capture_ns: int) -> None:
        """Score every track with a full window that hasn't been scored for SCORE_EVERY_NS, and isn't skipped, then
        close the spans no window has seen for HOLD_NS."""
        ready = [
            t
            for t, frames in self._frames.items()
            if len(frames) == CLIP
            and t not in self._skip
            and confident(frames) >= MIN_JOINTS
            and capture_ns - self._scored.get(t, -(1 << 62)) >= SCORE_EVERY_NS
        ]
        if ready:
            self._score(ready, capture_ns)
        self.spans.due(capture_ns)

    def _score(self, ready: list[int], capture_ns: int) -> None:
        probs = self.score(np.stack([np.stack([j for _, j in self._frames[t]]) for t in ready]))
        for track_id, p in zip(ready, probs, strict=True):
            self._scored[track_id] = capture_ns
            start, end = self._frames[track_id][0][0], self._frames[track_id][-1][0]
            self.windows.add(track_id, start, end, p, self.model)
            for kind, classes in self.kinds().items():
                seen = float(p[list(classes)].sum())
                if seen >= THRESHOLD:
                    self._seen[(track_id, kind)] = classes
                    self.spans.seen(track_id, kind, start, end, seen)

    def end(self, track_id: int) -> None:
        """A track ended: close its spans and forget its window."""
        self.spans.end(track_id)
        for key in [k for k in self._seen if k[0] == track_id]:
            del self._seen[key]
        self._frames.pop(track_id, None)
        self._scored.pop(track_id, None)
        self._still.pop(track_id, None)
        self._skip.discard(track_id)

    def gap(self) -> None:
        for track_id in {*self._frames, *self.spans.tracks()}:
            self.end(track_id)

    def take_windows(self) -> dict[str, Any] | None:
        return self.windows.take()

    def take_spans(self) -> list[dict[str, Any]]:
        return self.spans.take()


def since(track: crossings.Track, ts: float) -> crossings.Track | None:
    """A copy of the track with only its path from `ts` on, or None when fewer than two of its points are."""
    points = [p for p in track.points if p[0] >= ts]
    if len(points) < 2:
        return None
    after = copy.copy(track)
    after.points = points
    after.first_ts = max(track.first_ts, points[0][0])
    return after


class ActionPipeline:
    """One camera's frames in, pose and coverage rows out."""

    def __init__(
        self,
        backend: Backend[Any],
        poses: Sink,
        coverage: Sink,
        model: str,
        *,
        windows: Sink | None = None,
        spans: Callable[[list[dict[str, Any]]], None] | None = None,
        action_model: str = "stgcn++-ntu120-j",
        first_track: int = 1,
        movement: Movement | None = None,
        animals: Animals | None = None,
        flush_ns: int = FLUSH_NS,
        faces: HeadFaces | None = None,
        clips: Clips | None = None,
        live: Callable[[int, list[tuple[int, np.ndarray, np.ndarray]]], None] | None = None,
        camera: str = "",
        screen_zones: Callable[[], Sequence[Sequence[screens.Point]]] | None = None,
        screens_on: screens.ScreensOn | None = None,
        kinds: Callable[[], Mapping[str, tuple[int, ...]]] | None = None,
    ) -> None:
        """`first_track` is this run's first track ID: the service starts it from its start time, so IDs stay
        unique across restarts. Without `windows` and `spans`, nothing is scored. `clips` scores person crops with
        MoViNet (see/action_clips.py) beside the skeleton model, into the same window and span sinks. `live(capture_ns,
        people)` gets each frame's tracked people as they are found, (track, box, joints normalized to the frame) each,
        for the Live overlay's `pose.boxes` ring (see/pose_boxes.py). `screen_zones()` gives the camera's zones of
        kind "screen": with the screens the detector saw, they keep the people a TV shows out of everything
        (see/screens.py). `screens_on` gets when this camera's screens were on (their pictures moving), for
        tagging what was heard then as a TV's. `kinds()` is the owner's activities as NTU class indices
        (`activity_kinds.Live.ntu`); without it, the starting list."""
        self.backend, self.poses, self.coverage, self.model = backend, poses, coverage, model
        self.windows, self.spans, self.movement, self.animals = windows, spans, movement, animals
        self.scoring = (
            Scoring(backend.score, action_model, who=faces.who if faces is not None else None, kinds=kinds)
            if windows is not None or spans is not None
            else None
        )
        self.flush_ns, self.faces, self.clips, self.live = flush_ns, faces, clips, live
        self._last_ns: int | None = None
        self.tracker = crossings.Tracker(max_gap=LOST_S)
        self.tracker.next_id = first_track
        self._sure_ns: dict[int, int] = {}  # each track's last frame whose detection had a confident pose
        self._rows: dict[str, list[Any]] = {name: [] for name, _ in POSE_SCHEMA}
        self._run: _Run | None = None
        self._flushed_ns: int | None = None
        self.camera, self.screen_zones = camera, screen_zones
        self.screens = screens.ScreenTracks()
        self.held = screens.HeldScreens()
        self.screens_on, self.activity = screens_on, screens.ScreenActivity()
        self._screen_logged_ns: int | None = None

    def feed(self, frame: Frame) -> None:
        """Analyze one frame. Frames arrive in capture order; `gap_before` says frames are missing before it.

        A gap longer than LOST_S ends everything open. A shorter one, such as a frame lapped in the decoder's ring when
        a frame took long, only breaks the coverage run: the tracker already bridges a few missing frames. Ending
        every track on each lapped frame gave a person sitting still 400 track IDs a minute on cam0."""
        if frame.gap_before:
            if self._last_ns is None or frame.capture_ns - self._last_ns > LOST_S * 1e9:
                self.gap()
            else:
                self.flush()
                self._run = None
        frame_id, capture_ns = frame.id, frame.capture_ns
        image = self.backend.upload(frame)
        height, width = self.backend.size(image)
        people = self.backend.people(image)
        # People a screen shows, by the screens found now and in the last moments, and the camera's zones.
        zones = self.screen_zones() if self.screen_zones is not None else ()
        held = self.held.update(people.screens, capture_ns)
        inside = screens.on_screen(people.boxes, held, zones)
        if self.screens_on is not None and self.activity.due(capture_ns):
            real = people.boxes[~inside] if len(people.boxes) else people.boxes
            self._sample_screens(image, width, height, capture_ns, screens.playing(zones), real)
        ts = capture_ns / 1e9
        # Only a detection whose pose shows a person starts a track: objects taken for people show no confident
        # joints. One that doesn't still carries an existing track, as a low-score detection, so a person turned away
        # or behind something keeps theirs.
        sure_joints = (people.joints[:, :17, 2] > JOINT_SURE).sum(axis=1)
        sure = sure_joints >= MIN_JOINTS
        starts_below = self.tracker.high - 1e-3
        dets = [
            (x1, y1, x2, y2, score if sure[i] else min(score, starts_below))
            for i, (x1, y1, x2, y2, score) in enumerate(people.boxes.tolist())
        ]
        ended = self.tracker.update(dets, ts)
        for det, track_id in self.tracker.assigned.items():
            if sure[det] or track_id not in self._sure_ns:  # a track is born only on a confident pose
                self._sure_ns[track_id] = capture_ns
        unsure = [t for t in self.tracker.active if capture_ns - self._sure_ns.get(t.id, capture_ns) > UNSURE_NS]
        if unsure:
            gone = {t.id for t in unsure}
            self.tracker.active = [t for t in self.tracker.active if t.id not in gone]
            self.tracker.assigned = {d: t for d, t in self.tracker.assigned.items() if t not in gone}
            ended = [*ended, *unsure]
        # People a screen shows: their tracks run, so one found to be real is kept from then on, but none of
        # their frames go past here.
        on_screen = self.screens.update(self.tracker.assigned, inside, capture_ns, known=bool(len(held)) or bool(zones))
        for track in ended:
            self._sure_ns.pop(track.id, None)
            if self.scoring is not None:
                self.scoring.end(track.id)
            if self.clips is not None:
                self.clips.end(track.id)
            if self.movement is not None and not self.screens.is_screen(track.id):
                released = self.screens.released_at(track.id)
                if released is None:
                    self.movement.ended(track)
                elif (after := since(track, released / 1e9)) is not None:
                    self.movement.ended(after)  # only its path since it left the screen
            if self.faces is not None:
                self.faces.end(track.id)
            self.screens.end(track.id)
        hits = {track.id: track.hits for track in self.tracker.active}
        counted = {
            det: tid
            for det, tid in self.tracker.assigned.items()
            if hits.get(tid, 0) >= MIN_HITS and tid not in on_screen
        }
        if counted:
            joints = quantize(people.joints, width, height)
            if self.scoring is not None:
                # On a fisheye camera, each person's joints in their own perspective view, normalized per
                # person and frame by the view's size: the action model was trained on perspective skeletons, and
                # near the rim a fisheye bends them. Elsewhere, the frame's joints, normalized by the frame.
                if people.view_joints is not None and people.view_size is not None:
                    unit = normalized(people.view_joints, *people.view_size)
                else:
                    unit = normalized(people.joints, width, height)
                for det, track_id in counted.items():
                    self.scoring.add(track_id, capture_ns, unit[det], people.boxes[det])
                self.scoring.due(capture_ns)
            if self.clips is not None:
                for det, track_id in counted.items():
                    self.clips.add(track_id, capture_ns, image, people.boxes[det], int(sure_joints[det]))
                self.clips.due(capture_ns)
            if self.faces is not None:
                self.faces.feed(
                    lambda box: self.backend.crop(image, box),
                    width,
                    height,
                    capture_ns,
                    frame_id,
                    [(track_id, people.joints[det]) for det, track_id in sorted(counted.items())],
                    int(frame.source_header.get("seq") or 0),
                )
            # The Live overlay gets the people whose pose is confident on this frame, as a track starts on: a track
            # carried by an unsure detection (up to UNSURE_NS) draws nothing. The rows' own numbers, so the
            # Live overlay and recorded video's draw the same skeleton.
            shown = [(det, track_id) for det, track_id in sorted(counted.items()) if sure[det]]
            if self.live is not None and shown:
                self.live(
                    capture_ns,
                    [(track_id, people.boxes[det], joints[det].reshape(-1, 3) / SCALE) for det, track_id in shown],
                )
            for det, track_id in sorted(counted.items()):
                self._rows["frame_id"].append(frame_id)
                self._rows["ts_ns"].append(capture_ns)
                self._rows["track_id"].append(track_id)
                self._rows["box"].append(people.boxes[det])
                self._rows["joints"].append(joints[det])
                self._rows["model"].append(self.model)
        for track in self.tracker.active:
            # The tracker appends every frame's anchor; keep one about every PATH_SPACING_S (and the latest), so a
            # person sitting all evening doesn't pile up hundreds of thousands, while the crossing check still has
            # the path.
            points = track.points
            if len(points) >= 3 and points[-2][0] - points[-3][0] < PATH_SPACING_S:
                del points[-2]
            if len(points) > MAX_PATH:
                del points[0]
        self._last_ns = capture_ns
        self._log_screens(capture_ns)
        if self.movement is not None and self.movement.due(capture_ns):
            people_now = [(track_id, people.boxes[det]) for det, track_id in sorted(counted.items())]
            self.movement.sample(frame_id, capture_ns, people_now, self.backend.shift(image))
        if self.animals is not None:
            self.animals.sample(frame_id, capture_ns, image, people)
        if self._run is None:
            self._run = _Run(capture_ns, capture_ns)
        self._run.end_ns = capture_ns
        self._run.frames += 1
        if self._flushed_ns is None:
            self._flushed_ns = capture_ns
        if capture_ns - self._flushed_ns >= self.flush_ns:
            self.flush()
            self._flushed_ns = capture_ns

    def _sample_screens(
        self, image: Any, width: int, height: int, capture_ns: int, shown: np.ndarray, real: np.ndarray
    ) -> None:
        """Each screen's pixels, to tell whether it's on (screens.ScreenActivity), every SAMPLE_NS."""
        patches = []
        for s in shown:
            x1, x2 = (max(0, min(width, round(float(v) * width))) for v in (s[0], s[2]))
            y1, y2 = (max(0, min(height, round(float(v) * height))) for v in (s[1], s[3]))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                patches.append((s[:4], self.backend.crop(image, (x1, y1, x2, y2))))
        self.activity.sample(capture_ns, patches, real.tolist())
        if self.screens_on is not None:
            closed = bool(self.activity.closed)
            self.screens_on.add(self.camera, self.activity.spans(), flush=closed)

    def _log_screens(self, capture_ns: int) -> None:
        """Every SCREEN_LOG_NS, what the screen mask did on this camera, when it did anything."""
        if self._screen_logged_ns is None:
            self._screen_logged_ns = capture_ns
        if capture_ns - self._screen_logged_ns < SCREEN_LOG_NS:
            return
        self._screen_logged_ns = capture_ns
        c = self.screens.take()
        if c.inside or c.born or c.masked:
            LOG.info(
                f"screens: {self.camera or 'camera'} kept {c.masked} of {c.detections} person detections out as on a"
                f" screen ({c.inside} inside one; {c.born} screen tracks, {c.released} found real)"
            )

    def gap(self) -> None:
        """Frames are missing: end every track, its windows and spans, and the coverage run, so none of them
        bridges the gap."""
        if self.scoring is not None:
            self.scoring.gap()
        if self.clips is not None:
            self.clips.gap()
        if self.movement is not None:
            self.movement.gap()
        if self.animals is not None:
            self.animals.gap()
        if self.faces is not None:
            self.faces.gap()
        self.flush()
        self.tracker.flush()
        self._sure_ns.clear()
        self.screens.clear()
        self.held.clear()
        self.activity.gap()
        if self.screens_on is not None:
            self.screens_on.add(self.camera, self.activity.spans(), flush=True)
        self._run = None

    def close(self) -> None:
        """Stopping: close what's open, as a gap does, and append the rest."""
        self.gap()
        if self.faces is not None:
            self.faces.flush(None)

    def flush(self) -> None:
        """Append the buffered pose rows, the windows scored and spans closed, and the coverage since the last row,
        to the sinks."""
        if self.movement is not None:
            self.movement.flush_coverage()
        if self.animals is not None:
            self.animals.flush_coverage()
        for scorer in (self.scoring, self.clips):
            if scorer is None:
                continue
            windows = scorer.take_windows()
            if windows is not None and self.windows is not None:
                self.windows.append(windows)
            closed = scorer.take_spans()
            if closed and self.spans is not None:
                self.spans(closed)
        if self.faces is not None and self._last_ns is not None:
            self.faces.flush(self._last_ns)
        if self._rows["frame_id"]:
            columns = {
                "frame_id": list(self._rows["frame_id"]),
                "ts_ns": np.array(self._rows["ts_ns"], np.int64),
                "track_id": np.array(self._rows["track_id"], np.int64),
                "box": np.array(self._rows["box"], np.float32).reshape(-1, 5),
                "joints": np.array(self._rows["joints"], np.int16).reshape(-1, JOINTS * 3),
                "model": list(self._rows["model"]),
            }
            self.poses.append(columns)
            self._rows = {name: [] for name, _ in POSE_SCHEMA}
        run = self._run
        if run is not None and run.frames:
            self.coverage.append(
                {
                    "ts_ns": np.array([run.start_ns], np.int64),
                    "end_ns": np.array([run.end_ns], np.int64),
                    "frames": np.array([run.frames], np.int64),
                    "model": [self.model],
                }
            )
            run.start_ns, run.frames = run.end_ns, 0
