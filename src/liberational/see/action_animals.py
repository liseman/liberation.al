"""Animals from see-actions' detector pass: `animal.detection`, `animal.embedding`, and `animal.coverage`, in
the segment store.

The person detector's grid also scores the COCO animal classes (`vision_backend.People.animals`). About every
SAMPLE_NS, `Animals.sample` runs the old body worker's animal rules on them (carried over from see/bodies.py):

- An animal box sitting in the head of a person box is that person's head and is dropped.
- Each species has its own tracker (`vision/crossings.Tracker`), so a cat and a dog never share a track.
- A track is published only once it is confirmed: seen for `confirm_s`, with a box at `peak_score` or more, and
  passed by the backend's image check (`verify_animals`), which looks at an unconfirmed track about every
  CHECK_EVERY_S, up to MAX_CHECKS times. A confirmed track stays confirmed while it lasts.
- A confirmed track's box is embedded (`embed_animals`) when its shorter side is at least MIN_EMBED_PX: first after
  EMBED_EVERY_S, then with the gap doubling up to EMBED_MAX_S.

Each confirmed box is an `animal.detection` row and each embedding an `animal.embedding` row. `animal.coverage` rows
are the spans of samples looked at, with how many, so a span with no detections means no animal was seen, not that
nobody looked. A gap or a stop ends every track and the coverage run, so nothing bridges frames that were never seen.
Animal movement tracks were removed with see-tracks and aren't made here.

Rows are buffered and written by `flush_coverage`, which see-actions calls about once a second, and by `gap`. A
failing image check turns animals off for this stage (an unchecked box is most often a head); a failing embedder
stops embeddings only.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from liberational import logs, segment_log
from liberational.see.vision_backend import Backend, People
from liberational.vision import animal_embedder, crossings

LOG = logs.get("see-actions")
SAMPLE_NS = 200_000_000  # animals are sampled about this often: plenty for a pet, and few backend calls
CHECK_EVERY_S = 1.0  # an unconfirmed animal track's crop is checked at most this often
MAX_CHECKS = 10  # and at most this many times; a track still unconfirmed after them stays withheld
CONFIRM_S = 1.0  # an animal track must be seen this long
PEAK_SCORE = 0.6  # with at least one box this confident (a track starts at 0.5, crossings.Tracker)
EMBED_EVERY_S = 2.0  # a confirmed track's crop is embedded at first this often, the gap doubling each time
EMBED_MAX_S = 300.0  # up to this, so a cat asleep in view all afternoon adds a few dozen, not thousands
HEAD_SHARE = 0.35  # the top of a person box their head is in
HEAD_INSIDE = 0.5  # an animal box this much inside a person's head region is that head
MIN_EMBED_PX = 48  # and only when its box's shorter side has this many pixels
DETECTOR = "rfdetr_small"  # [models.files.person_detector]

Write = Callable[[str, list[dict[str, Any]]], None]


def _covered(inner: Any, outer: Any) -> float:
    """The share of `inner`'s area inside `outer`."""
    w = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    h = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return w * h / max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))


def _on_a_person(box: Any, people: Any) -> bool:
    """Whether an animal box sits in the head of a person box: that person's head. A box beside someone, or low on
    them (a cat on a lap), is not."""
    return any(_covered(box, (p[0], p[1], p[2], p[1] + HEAD_SHARE * (p[3] - p[1]))) >= HEAD_INSIDE for p in people)


@dataclass
class _Candidate:
    """An animal track's way to confirmation, and its embedding cadence."""

    first: float
    peak: float = 0.0
    passed: int = 0
    failed: int = 0
    checked: float = -float("inf")
    confirmed: bool = False
    embedded: float = -float("inf")
    embeds: int = 0


class Animals:
    """One camera's animals. `write(stream, records)` gets their records (into `segment_log.Writers`). `verify`
    False confirms tracks without the image check (for a backend that has none), and their rows say so."""

    def __init__(
        self,
        cam: str,
        write: Write,
        backend: Backend[Any],
        *,
        verify: bool = True,
        confirm_s: float = CONFIRM_S,
        peak_score: float = PEAK_SCORE,
        shots: Any = None,
    ) -> None:
        """`shots` (a `pet_shots.Writer`) is offered each embedded box, for Pets' photo of its track."""
        self.cam, self.write, self.backend, self.shots = cam, write, backend, shots
        self.verify, self.confirm_s, self.peak_score = verify, confirm_s, peak_score
        self.on, self.embedding = True, True
        self.max_gap = float(crossings.settings()["max_gap_seconds"])
        self.embedding_model = animal_embedder.space()  # the model and its crop
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._cover: list[int] | None = None  # the coverage being built: [start ns, end ns, samples]
        self._start()

    def _start(self) -> None:
        """A fresh run: new trackers, and track ids no earlier run used."""
        # The UUID's random tail: a UUIDv7's head is its time, and track numbers restart at 1 with each run.
        self.run = uuid.uuid7().hex[:20]  # starts with the run's time: its photos' hour (face_shots.hour)
        self.trackers = {
            species: crossings.Tracker(max_gap=self.max_gap) for species in crossings.COCO_ANIMALS.values()
        }
        self.candidates: dict[tuple[str, int], _Candidate] = {}
        self._last_ns: int | None = None

    def sample(self, frame_id: str, capture_ns: int, image: Any, people: People) -> None:
        """One analyzed frame: the backend's image and what it found there. Frames closer than SAMPLE_NS to the last
        sample are skipped."""
        if not self.on or (self._last_ns is not None and capture_ns - self._last_ns < SAMPLE_NS):
            return
        ts = capture_ns / 1e9
        persons = [tuple(float(v) for v in box[:5]) for box in people.boxes]
        seen: list[tuple[str, int, tuple[float, ...]]] = []  # (species, track number, box) of each tracked box
        for class_id, species in crossings.COCO_ANIMALS.items():
            detected = [
                tuple(float(v) for v in row[:5])
                for row in people.animals
                if int(row[5]) == class_id and not _on_a_person(row[:4], persons)
            ]
            tracker = self.trackers[species]
            tracker.update(detected, ts)
            for track in tracker.active:  # association needs only the latest point, never a whole day's path
                del track.points[:-2]
            seen += [(species, tid, detected[i]) for i, tid in sorted(tracker.assigned.items())]
        active = {(species, t.id) for species, tracker in self.trackers.items() for t in tracker.active}
        self.candidates = {key: c for key, c in self.candidates.items() if key in active}
        for species, tid, _ in seen:
            self.candidates.setdefault((species, tid), _Candidate(first=ts))
        try:
            self._check(image, ts, seen)
        except Exception as error:
            LOG.warning("%s: the animal check failed, so animals are no longer evaluated: %s", self.cam, error)
            self.gap()
            self.on = False
            return
        confirmed = [(species, tid, box) for species, tid, box in seen if self._confirmed(species, tid, box, ts)]
        ids = [f"{self.run}:{species}:{tid}" for species, tid, _ in confirmed]
        self._add(
            segment_log.ANIMAL_DETECTIONS,
            [
                {
                    "ts_ns": capture_ns,
                    "frame_id": frame_id,
                    "track_id": track_id,
                    "species": species,
                    "box": list(box),
                    "verified": self.verify,
                    "model": DETECTOR,
                }
                for track_id, (species, _, box) in zip(ids, confirmed, strict=True)
            ],
        )
        self._embed(image, capture_ns, confirmed, ids)
        if self._cover is None:
            self._cover = [capture_ns, capture_ns, 0]
        self._cover[1], self._cover[2] = capture_ns, self._cover[2] + 1
        self._last_ns = capture_ns

    def _check(self, image: Any, ts: float, seen: list[tuple[str, int, tuple[float, ...]]]) -> None:
        """One image check, in one backend call, of each unconfirmed track that is due one."""
        if not self.verify:
            return
        due = []
        for species, tid, box in seen:
            c = self.candidates[(species, tid)]
            if not c.confirmed and c.passed + c.failed < MAX_CHECKS and ts - c.checked >= CHECK_EVERY_S:
                due.append((c, species, box))
        if not due:
            return
        boxes = np.array([box for _, _, box in due], np.float32).reshape(-1, 5)
        results = self.backend.verify_animals(image, boxes, [species for _, species, _ in due])
        for (c, _, _), ok in zip(due, results, strict=True):
            c.checked = ts
            if ok:
                c.passed += 1
            else:
                c.failed += 1

    def _confirmed(self, species: str, tid: int, box: tuple[float, ...], ts: float) -> bool:
        c = self.candidates[(species, tid)]
        if not c.confirmed:
            c.peak = max(c.peak, float(box[4]))
            looks = not self.verify or c.passed > c.failed
            c.confirmed = looks and ts - c.first >= self.confirm_s and c.peak >= self.peak_score
        return c.confirmed

    def _embed(
        self, image: Any, capture_ns: int, confirmed: list[tuple[str, int, tuple[float, ...]]], ids: list[str]
    ) -> None:
        """Embeddings, in one backend call, of the confirmed boxes whose tracks are due one."""
        if not self.embedding or not confirmed:
            return
        ts = capture_ns / 1e9
        h, w = self.backend.size(image)
        due = []
        for (species, tid, box), track_id in zip(confirmed, ids, strict=True):
            c = self.candidates[(species, tid)]
            if ts - c.embedded < min(EMBED_MAX_S, EMBED_EVERY_S * 2 ** max(0, c.embeds - 1)):
                continue
            if min((box[2] - box[0]) * w, (box[3] - box[1]) * h) < MIN_EMBED_PX:
                continue
            due.append((c, species, box, track_id))
        if not due:
            return
        try:
            vectors = self.backend.embed_animals(image, np.array([box for _, _, box, _ in due], np.float32))
        except Exception as error:  # optional: a failing embedder stops embeddings, never animals
            LOG.warning("%s: pet embeddings failed, so pets are no longer told apart: %s", self.cam, error)
            self.embedding = False
            return
        rows = []
        for (c, species, box, track_id), vector in zip(due, np.asarray(vectors, np.float32), strict=True):
            c.embedded, c.embeds = ts, c.embeds + 1
            rows.append(
                {
                    "ts_ns": capture_ns,
                    "track_id": track_id,
                    "species": species,
                    "box": list(box),
                    "vector": vector.tolist(),
                    "model": self.embedding_model,
                }
            )
        self._add(segment_log.ANIMAL_EMBEDDINGS, rows)
        if self.shots is not None:
            for _, _, box, track_id in due:
                self.shots.offer(track_id, box, w, h, lambda bounds: self.backend.crop(image, bounds))

    def gap(self) -> None:
        """Frames are missing, or it's stopping: end every track and the coverage run, and write what's buffered."""
        self._start()
        self.flush_coverage(end=True)

    def flush_coverage(self, *, end: bool = False) -> None:
        """Write the buffered detections and embeddings, and the coverage since the last row; `end` breaks the
        coverage run, as a gap does."""
        cover = self._cover
        if cover is not None and cover[2] and cover[1] > cover[0]:
            row = {"id": str(uuid.uuid7()), "ts_start": cover[0] / 1e9, "ts_end": cover[1] / 1e9, "frames": cover[2]}
            self._add(segment_log.ANIMAL_COVERAGE, [row])
            self._cover = [cover[1], cover[1], 0]
        if end:
            self._cover = None
        rows, self._rows = self._rows, {}
        for stream, records in rows.items():
            if records:
                self.write(stream, records)

    def _add(self, stream: str, records: list[dict[str, Any]]) -> None:
        if records:
            self._rows.setdefault(stream, []).extend(records)
