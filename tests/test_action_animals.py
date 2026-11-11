"""see-actions' animals (see/action_animals.py): the old body worker's animal rules on the GPU detector's
boxes, written to the segment store."""

import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from liberational import segment_log
from liberational.see import action_animals
from liberational.see.vision_backend import People, detector_model

T = 500_000 * 3_600_000_000_000  # on an hour boundary
STEP = 250_000_000  # a sample every 0.25 s: past SAMPLE_NS, and exact in float seconds
CAT, DOG = 15, 16  # COCO class ids
PERSON = (0.4, 0.1, 0.6, 0.9, 0.9)
ON_THE_HEAD = (0.45, 0.12, 0.55, 0.25, 0.9)  # inside the person box's top 35%
ON_THE_FLOOR = (0.05, 0.5, 0.25, 0.9, 0.9)  # 128 x 192 px of a 640 x 480 frame


def scene(*animals: tuple[tuple[float, ...], int], people: tuple[tuple[float, ...], ...] = ()) -> People:
    boxes = np.array(people, np.float32).reshape(-1, 5)
    found = np.array([[*box, class_id] for box, class_id in animals], np.float32).reshape(-1, 6)
    return People(boxes, np.zeros((len(boxes), 26, 3), np.float32), found)


@dataclass
class Fake:
    """A backend whose images are sample numbers, 640 x 480, with a scripted image check."""

    looks: bool = True
    checks: list[tuple[int, list[str]]] = field(default_factory=list)
    embedded: list[int] = field(default_factory=list)
    name: str = "fake"

    def upload(self, frame: Any) -> int:
        raise AssertionError("the stage never uploads")

    def captures(self, frame: Any) -> bool:
        return False

    def size(self, image: int) -> tuple[int, int]:
        return 480, 640

    def people(self, image: int) -> People:
        raise AssertionError("the stage is handed its people")

    def shift(self, image: int) -> tuple[float, float] | None:
        raise AssertionError("the animal stage measures no camera shift")

    def crop(self, image: int, box: tuple[int, int, int, int]) -> np.ndarray:
        raise AssertionError("the animal stage crops nothing itself")

    def score(self, windows: np.ndarray) -> np.ndarray:
        raise AssertionError("the stage scores no actions")

    def embed_animals(self, image: int, boxes: np.ndarray) -> np.ndarray:
        self.embedded += [image] * len(boxes)
        vectors = np.zeros((len(boxes), segment_log.EMBEDDING_DIM), np.float32)
        vectors[:, image % segment_log.EMBEDDING_DIM] = 1.0
        return vectors

    def verify_animals(self, image: int, boxes: np.ndarray, species: list[str]) -> list[bool]:
        assert boxes.shape == (len(species), 5)
        self.checks.append((image, list(species)))
        return [self.looks] * len(species)


def frame_id(i: int) -> str:
    return str(uuid.UUID(int=0x0199A000_0000_7000_8000_000000000000 + i))


class Rows(dict[str, list[dict[str, Any]]]):
    def __call__(self, stream: str, records: list[dict[str, Any]]) -> None:
        self.setdefault(stream, []).extend(records)


def run(
    frames: list[People], backend: Fake | None = None, *, gaps: frozenset[int] = frozenset(), write: Any = None
) -> tuple[Rows, Fake, action_animals.Animals]:
    rows, backend = Rows(), backend or Fake()
    animals = action_animals.Animals("cam0", write if write is not None else rows, backend)
    for i, people in enumerate(frames):
        if i in gaps:
            animals.gap()
        animals.sample(frame_id(i), T + i * STEP, i, people)
    animals.flush_coverage()
    return rows, backend, animals


def seconds(ts_ns: int) -> float:
    return (ts_ns - T) / 1e9


def test_an_animal_box_on_a_persons_head_is_dropped_and_one_beside_them_is_kept():
    rows, backend, _ = run([scene((ON_THE_HEAD, CAT), (ON_THE_FLOOR, CAT), people=(PERSON,))] * 8)
    boxes = {tuple(round(v, 4) for v in r["box"]) for r in rows[segment_log.ANIMAL_DETECTIONS]}
    assert boxes == {ON_THE_FLOOR}
    assert all(species == ["cat"] for _, species in backend.checks), "the head is never even checked"


def test_without_a_person_the_same_box_is_an_animal():
    rows, _, _ = run([scene((ON_THE_HEAD, CAT))] * 8)
    assert {r["species"] for r in rows[segment_log.ANIMAL_DETECTIONS]} == {"cat"}


def test_a_box_is_published_from_when_its_track_has_been_seen_long_enough_and_checked():
    rows, backend, _ = run([scene((ON_THE_FLOOR, CAT))] * 8)
    detections = rows[segment_log.ANIMAL_DETECTIONS]
    assert [seconds(r["ts_ns"]) for r in detections] == [1.0, 1.25, 1.5, 1.75]
    assert {r["frame_id"] for r in detections} == {frame_id(i) for i in range(4, 8)}
    assert all(r["verified"] for r in detections)
    assert backend.checks == [(0, ["cat"]), (4, ["cat"])], "about once a second until confirmed, then never"


@pytest.mark.parametrize(
    ("looks", "score"),
    [
        pytest.param(False, 0.9, id="the-image-check-says-no"),
        pytest.param(True, 0.55, id="never-a-confident-box"),
    ],
)
def test_an_unconfirmed_track_publishes_nothing_and_its_checks_run_out(looks, score):
    frames = [scene(((*ON_THE_FLOOR[:4], score), CAT))] * 60  # 15 s
    rows, backend, _ = run(frames, Fake(looks=looks))
    assert segment_log.ANIMAL_DETECTIONS not in rows
    assert segment_log.ANIMAL_EMBEDDINGS not in rows
    assert len(backend.checks) == action_animals.MAX_CHECKS


def test_each_species_is_its_own_track():
    rows, _, _ = run([scene((ON_THE_FLOOR, CAT), (ON_THE_FLOOR, DOG))] * 8)
    by_species = {r["species"]: r["track_id"] for r in rows[segment_log.ANIMAL_DETECTIONS]}
    assert set(by_species) == {"cat", "dog"}
    assert by_species["cat"] != by_species["dog"]


def test_frames_closer_than_the_sample_interval_are_skipped():
    rows = Rows()
    animals = action_animals.Animals("cam0", rows, Fake())
    for i in range(40):  # 25 fps for 1.6 s: every fifth frame is 0.2 s on
        animals.sample(frame_id(i), T + i * 40_000_000, i, scene((ON_THE_FLOOR, CAT)))
    animals.flush_coverage()
    assert [seconds(r["ts_ns"]) for r in rows[segment_log.ANIMAL_DETECTIONS]] == [1.0, 1.2, 1.4]


def test_embeddings_follow_the_doubling_cadence_and_skip_small_boxes():
    small = (0.6, 0.6, 0.65, 0.65, 0.9)  # 32 x 24 px: too small to tell pets apart
    rows, _, _ = run([scene((ON_THE_FLOOR, CAT), (small, DOG))] * 80)  # 20 s
    embeddings = rows[segment_log.ANIMAL_EMBEDDINGS]
    assert [seconds(r["ts_ns"]) for r in embeddings] == [1.0, 3.0, 7.0, 15.0]
    assert {r["species"] for r in embeddings} == {"cat"}
    assert {r["species"] for r in rows[segment_log.ANIMAL_DETECTIONS]} == {"cat", "dog"}
    assert len(embeddings[0]["vector"]) == segment_log.EMBEDDING_DIM


def test_a_failing_embedder_stops_embeddings_but_not_detections():
    class Broken(Fake):
        def embed_animals(self, image: int, boxes: np.ndarray) -> np.ndarray:
            raise RuntimeError("out of memory")

    rows, _, _ = run([scene((ON_THE_FLOOR, CAT))] * 8, Broken())
    assert segment_log.ANIMAL_EMBEDDINGS not in rows
    assert len(rows[segment_log.ANIMAL_DETECTIONS]) == 4


def test_a_failing_image_check_turns_animals_off_and_ends_what_was_open():
    class Broken(Fake):
        def verify_animals(self, image: int, boxes: np.ndarray, species: list[str]) -> list[bool]:
            if image >= 8:
                raise RuntimeError("out of memory")
            return super().verify_animals(image, boxes, species)

    second = (0.7, 0.5, 0.9, 0.9, 0.9)
    frames = [scene((ON_THE_FLOOR, CAT))] * 8 + [scene((ON_THE_FLOOR, CAT), (second, DOG))] * 8
    rows, _, animals = run(frames, Broken())
    assert max(seconds(r["ts_ns"]) for r in rows[segment_log.ANIMAL_DETECTIONS]) == 1.75
    assert animals.on is False
    assert coverage(rows) == [(0.0, 1.75, 8)], "nothing after the failure counts as looked at"


def coverage(rows: Rows) -> list[tuple[float, float, int]]:
    t0 = T / 1e9
    return [
        (round(r["ts_start"] - t0, 3), round(r["ts_end"] - t0, 3), r["frames"])
        for r in rows[segment_log.ANIMAL_COVERAGE]
    ]


def test_frames_with_no_animals_still_count_as_looked_at():
    rows, _, _ = run([scene(people=(PERSON,))] * 8)
    assert segment_log.ANIMAL_DETECTIONS not in rows
    assert coverage(rows) == [(0.0, 1.75, 8)]


def test_a_gap_ends_the_tracks_and_the_coverage_run_and_later_tracks_are_new():
    rows, _, _ = run([scene((ON_THE_FLOOR, CAT))] * 24, gaps=frozenset({12}))
    assert coverage(rows) == [(0.0, 2.75, 12), (3.0, 5.75, 12)], "no coverage bridges the gap"
    before = {r["track_id"] for r in rows[segment_log.ANIMAL_DETECTIONS] if r["ts_ns"] < T + 12 * STEP}
    after = {r["track_id"] for r in rows[segment_log.ANIMAL_DETECTIONS] if r["ts_ns"] >= T + 12 * STEP}
    assert len(before) == len(after) == 1 and before != after
    assert min(seconds(r["ts_ns"]) for r in rows[segment_log.ANIMAL_DETECTIONS] if r["track_id"] in after) == 4.0, (
        "a track after the gap is confirmed again from its own start"
    )


def test_rows_round_trip_through_the_segment_store(tmp_path):
    continuum_py = pytest.importorskip("continuum_py")
    if not hasattr(continuum_py, "SegmentWriter"):
        pytest.skip("this continuum_py predates the segment store")
    writers = segment_log.Writers(tmp_path)
    run([scene((ON_THE_FLOOR, CAT))] * 40, write=lambda stream, records: writers.append(stream, "cam0", records))
    writers.close()
    log = segment_log.Log(tmp_path)
    start, end = T / 1e9, T / 1e9 + 3600
    detections = log.range(segment_log.ANIMAL_DETECTIONS, start, end)
    assert len(detections) == 36
    first = detections[0]
    assert (first["ts_start"], first["cam"], first["frame_id"], first["species"]) == (
        (T + 4 * STEP) / 1e9,
        "cam0",
        frame_id(4),
        "cat",
    )
    assert first["box"] == pytest.approx(list(ON_THE_FLOOR))
    assert (first["verified"], first["model"]) == (True, detector_model())
    embeddings = log.range(segment_log.ANIMAL_EMBEDDINGS, start, end)
    assert [e["ts_start"] - start for e in embeddings] == [1.0, 3.0, 7.0]
    assert embeddings[0]["track_id"] == first["track_id"]
    assert isinstance(embeddings[0]["vector"], list) and len(embeddings[0]["vector"]) == segment_log.EMBEDDING_DIM
    assert embeddings[0]["vector"][4] == 1.0
    (covered,) = log.range(segment_log.ANIMAL_COVERAGE, start, end)
    assert (covered["ts_start"], covered["ts_end"], covered["frames"]) == (start, (T + 39 * STEP) / 1e9, 40)


def test_the_stage_never_needs_torch():
    # What touches the accelerator stays behind the backend seam, so the rig's Mac backend can replace CUDA.
    blocked = "import sys; sys.modules['torch'] = None; import liberational.see.action_animals"
    subprocess.run([sys.executable, "-c", blocked], check=True)
