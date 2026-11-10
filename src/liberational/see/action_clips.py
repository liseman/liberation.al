"""MoViNet beside ST-GCN++ in see-actions (#1977): what a person's skeleton can't show, read from the pixels.

ST-GCN++ reads movement (sitting down, standing up, a fall, picking up). Reading, watching TV, using a phone, and
sleeping hardly move a joint, so `Clips` keeps each track's last FRAMES person crops, one every STEP_NS, and scores
them with MoViNet-A2 (Kinetics-600, see/movinet.py) every EVERY_NS. Each score is an `action.window` row (its top
Kinetics classes, model MODEL), and an activity whose Kinetics classes add up to THRESHOLD or more becomes an
`action.span`, as ST-GCN++'s do. The activities and their classes are the owner's list (memory/activity_kinds.py,
#3244), read through `kinds` each time a clip is scored.

What it skips, from the offline replay of 2026-11-03 18:05 to 19:05 (38 spans at 0.5, ST-GCN++ 0):
- A clip whose median frame has fewer than `actions.MIN_JOINTS` confident joints: an object taken for a person.
- Not a box that stays still for minutes (ST-GCN++'s STILL_NS): reading, watching TV, and sleeping are still. The
  objects that rule was for show no confident joints, so the joints rule already drops them.
- Not a box cut off by the frame's edge, which ST-GCN++ skips because a part of a skeleton misleads it. A crop is
  pixels, black past the edge. On cam0 a person's box touched the edge in a quarter to over half of a long track's
  frames, and each one restarted the clip, so it was rarely scored (#1991).

A span closes HOLD_NS after its last clip that saw the kind and is kept only when MIN_CLIPS clips saw it: in the
replay, most of the 28 read spans were one clip at the edge of a long one. The hold outlasts a clip's refill (10 s
after a gap restarts it), so one restart doesn't split a span in two (#1991).
"""

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from liberational.memory import activity_kinds
from liberational.see import actions
from liberational.see.movinet_classes import CLASSES

MODEL = "movinet-a2-k600"
FRAMES = 50  # a clip: 10 s at 5 fps, as MoViNet-A2 was measured
STEP_NS = 200_000_000  # a crop about this often
EVERY_NS = 2_000_000_000  # a track's latest clip is scored this often
GAP_NS = 1_000_000_000  # a track unseen this long starts its clip over
HOLD_NS = 15_000_000_000  # longer than a clip takes to fill again
MIN_CLIPS = 2
THRESHOLD = 0.5
BATCH = 2  # clips per model call: one takes about 30 ms on a 3090, and 8 at once peak at 2.6 GB


@dataclass
class _Clip:
    crops: deque[tuple[int, np.ndarray, int]] = field(default_factory=lambda: deque(maxlen=FRAMES))
    due_ns: int = 0  # the next crop is taken at or after this (less a quarter STEP_NS of jitter)
    last_ns: int = 0
    scored_ns: int = -(1 << 62)


STARTING = activity_kinds.starting("kinetics")  # the classes before the owner changed the list


class Clips:
    """Per-track clips of person crops, scored by MoViNet, as window rows and spans."""

    def __init__(
        self,
        crop: Callable[[Any, np.ndarray], np.ndarray],
        score: Callable[[np.ndarray], np.ndarray],
        who: Callable[[int], tuple[str | None, str | None]] | None = None,
        kinds: Callable[[], Mapping[str, tuple[int, ...]]] = lambda: STARTING,
    ) -> None:
        """`crop(image, box)` is a person's square crop, SIZE x SIZE x 3 RGB uint8, around `box` (x1, y1, x2, y2
        normalized to the frame); `score(clips)` is MoViNet's class probabilities, N x 600, for N clips of them;
        `kinds()` is each activity's Kinetics class indices now (`activity_kinds.Live.kinetics`)."""
        self.crop, self.score, self.kinds = crop, score, kinds
        self._clips: dict[int, _Clip] = {}
        self._seen: dict[tuple[int, str], tuple[int, ...]] = {}  # the classes each open span was read from
        self.windows = actions.Windows()
        self.spans = actions.Spans(
            lambda span: (
                f"{MODEL}: Kinetics-600 "
                f"{' or '.join(repr(CLASSES[c]) for c in self._seen.pop((span.track_id, span.kind), ()))}, "
                f"peak {span.peak:.2f}"
            ),
            who,
            hold_ns=HOLD_NS,
            min_seen=MIN_CLIPS,
        )

    def add(self, track_id: int, capture_ns: int, image: Any, box: np.ndarray, sure: int) -> None:
        """One frame of a track: `sure` is how many of its COCO joints are above `actions.JOINT_SURE`."""
        clip = self._clips.setdefault(track_id, _Clip())
        if clip.crops and capture_ns - clip.last_ns > GAP_NS:
            clip.crops.clear()
        if clip.crops and capture_ns < clip.due_ns - STEP_NS // 4:
            return
        # A schedule, as see-actions' frame rate is: the next crop is due a step after this one's due time, unless
        # this one came more than a step late.
        behind = not clip.crops or capture_ns - clip.due_ns > STEP_NS
        clip.due_ns = (capture_ns if behind else clip.due_ns) + STEP_NS
        clip.crops.append((capture_ns, self.crop(image, box), sure))
        clip.last_ns = capture_ns

    def due(self, capture_ns: int) -> None:
        """Score every full clip not scored for EVERY_NS whose pose is a person's, then close stale spans."""
        ready = []
        for track_id, clip in self._clips.items():
            if len(clip.crops) < FRAMES or capture_ns - clip.scored_ns < EVERY_NS:
                continue
            clip.scored_ns = capture_ns
            if np.median([sure for *_, sure in clip.crops]) >= actions.MIN_JOINTS:
                ready.append(track_id)
        for i in range(0, len(ready), BATCH):
            self._score(ready[i : i + BATCH])
        self.spans.due(capture_ns)

    def _score(self, ready: list[int]) -> None:
        probs = self.score(np.stack([np.stack([c for _, c, _ in self._clips[t].crops]) for t in ready]))
        for track_id, p in zip(ready, probs, strict=True):
            crops = self._clips[track_id].crops
            start, end = crops[0][0], crops[-1][0]
            self.windows.add(track_id, start, end, p, MODEL)
            for kind, classes in self.kinds().items():
                seen = float(p[list(classes)].sum())
                if seen >= THRESHOLD:
                    self._seen[(track_id, kind)] = classes
                    self.spans.seen(track_id, kind, start, end, seen)

    def end(self, track_id: int) -> None:
        self.spans.end(track_id)
        for key in [k for k in self._seen if k[0] == track_id]:
            del self._seen[key]
        self._clips.pop(track_id, None)

    def gap(self) -> None:
        for track_id in {*self._clips, *self.spans.tracks()}:
            self.end(track_id)

    def take_windows(self) -> dict[str, Any] | None:
        return self.windows.take()

    def take_spans(self) -> list[dict[str, Any]]:
        return self.spans.take()
