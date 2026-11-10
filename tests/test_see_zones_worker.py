"""The zone worker on a real segment store: see-actions' rows in, zone visits and coverage out."""

import asyncio
import itertools
import time
import uuid
from pathlib import Path

import pytest

from liberational import segment_log
from liberational.see import zones
from liberational.see import zones_worker as Z
from test_tracked_frames import ActionsRows, S, T

POLYGON = ((0.1, 0.6), (0.3, 0.6), (0.3, 0.9), (0.1, 0.9))
IN, OUT = 0.2, 0.6  # a box centred here has its foot anchor (x, 0.8) inside the polygon at 0.2, outside at 0.6
STEP = S // 2


def box(x: float) -> list[float]:
    return [x - 0.04, 0.5, x + 0.04, 0.8]


def definition(zone_id: str = "zone-litter", name: str = "Litter box", at: float = 0.0) -> zones.Definition:
    return zones.Definition(f"def-{zone_id}-{at}", "cam0", (zones.Zone(zone_id, name, POLYGON),), at)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def state(tmp_path: Path) -> Path:
    pytest.importorskip("continuum_py").SegmentFollowers  # noqa: B018 (the native store must be built)
    return tmp_path


def worker(state: Path, clock: Clock, sets: dict[str, zones.Definition]) -> Z.ZoneWorker:
    return Z.ZoneWorker(state, current=lambda _state: dict(sets), clock=clock)


class AnimalRows:
    """Writes `animal.detection` and `animal.coverage` the way see-actions does, one camera at a time."""

    def __init__(self, root: Path, cam: str = "cam0") -> None:
        self.cam = cam
        self.writers = segment_log.Writers(root)

    def animals(self, ts_ns: int, tracks: list[tuple[str, str, list[float]]]) -> None:
        """One sample's confirmed animals: (track id, species, [x1, y1, x2, y2]) each."""
        if not tracks:
            return
        frame = str(uuid.uuid7())
        self.writers.append(
            segment_log.ANIMAL_DETECTIONS,
            self.cam,
            [
                {
                    "ts_ns": ts_ns,
                    "frame_id": frame,
                    "track_id": track,
                    "species": species,
                    "box": [*box, 0.9],
                    "verified": True,
                    "model": "test",
                }
                for track, species, box in tracks
            ],
        )

    def covered(self, start_ns: int, end_ns: int, frames: int = 5) -> None:
        self.writers.append(
            segment_log.ANIMAL_COVERAGE,
            self.cam,
            [
                {
                    "id": str(uuid.uuid7()),
                    "ts_start": start_ns / 1e9,
                    "ts_end": end_ns / 1e9,
                    "frames": frames,
                }
            ],
        )

    def sync(self) -> None:
        self.writers.sync()

    def close(self) -> None:
        self.writers.close()


def walk(rows: ActionsRows, start_ns: int, xs: list[float | None], track: int = 7) -> int:
    """One frame every half second from `start_ns`, track `track` at each x (None: nobody), covered a second at a
    time. Returns the time after the last frame."""
    at = start_ns
    for n, x in enumerate(xs):
        rows.people(at, [] if x is None else [(track, box(x))])
        if n % 2 == 1:
            rows.covered(at - STEP, at)
        at += STEP
    rows.sync()
    return at


def run(w: Z.ZoneWorker, clock: Clock, rounds: int = 4) -> None:
    for _ in range(rounds):
        w.step(timeout_s=0.3)
        clock.now += 1.0


def written(state: Path, stream: str) -> list[dict]:
    return segment_log.Log(segment_log.root(state)).range(stream, T / 1e9 - 10, T / 1e9 + 1000)


def walk_animals(rows: AnimalRows, start_ns: int, xs: list[float | None], track: str, species: str) -> int:
    """Like `walk`, for one animal track. Returns the time after the last sample."""
    at = start_ns
    for n, x in enumerate(xs):
        rows.animals(at, [] if x is None else [(track, species, box(x))])
        if n % 2 == 1:
            rows.covered(at - STEP, at)
        at += STEP
    rows.sync()
    return at


def test_a_person_walking_through_a_zone_is_one_visit_with_its_dwell(state):
    clock, sets = Clock(), {"cam0": definition()}
    w = worker(state, clock, sets)
    rows = ActionsRows(segment_log.root(state))
    walk(rows, T, [OUT, OUT, *[IN] * 10, *[OUT] * 6])
    run(w, clock)
    w.checkpoint()
    (visit,) = written(state, segment_log.ZONE_VISITS)
    assert (visit["zone_name"], visit["subject_kind"], visit["track_id"], visit["ended"]) == (
        "Litter box",
        "person",
        "7",
        "left",
    )
    assert visit["dwell_s"] == pytest.approx(4.5)  # inside from 1 s to 5.5 s
    assert visit["entry_seen"] is True
    rows.close()


def test_an_animal_in_a_zone_is_a_visit_in_its_species_under_its_track(state):
    clock, sets = Clock(), {"cam0": definition()}
    w = worker(state, clock, sets)
    rows = AnimalRows(segment_log.root(state))
    walk_animals(rows, T, [OUT, OUT, *[IN] * 10, *[OUT] * 6], "run1:cat:3", "cat")
    run(w, clock)
    w.checkpoint()
    (visit,) = written(state, segment_log.ZONE_VISITS)
    assert (visit["zone_name"], visit["subject_kind"], visit["label"], visit["track_id"], visit["ended"]) == (
        "Litter box",
        "animal",
        "cat",
        "run1:cat:3",
        "left",
    )
    assert visit["dwell_s"] == pytest.approx(4.5)  # inside from 1 s to 5.5 s, same span a person gets
    assert visit["entry_seen"] is True
    assert written(state, segment_log.PRESENCE_SPANS) == []  # an animal in view is not someone present
    rows.close()


def test_a_covered_stretch_with_no_animal_is_animal_coverage_and_no_visit(state):
    clock, sets = Clock(), {"cam0": definition()}
    w = worker(state, clock, sets)
    rows = AnimalRows(segment_log.root(state))
    walk_animals(rows, T, [None] * 140, "run1:cat:3", "cat")
    run(w, clock)
    w.checkpoint()
    assert written(state, segment_log.ZONE_VISITS) == []
    (coverage,) = written(state, segment_log.ZONE_COVERAGE)
    assert (coverage["subject_kind"], coverage["zones"]) == ("animal", ["zone-litter"])
    assert coverage["ts_end"] - coverage["ts_start"] >= 60
    rows.close()


def test_a_covered_stretch_with_nobody_in_view_is_coverage_not_unknown(state):
    clock, sets = Clock(), {"cam0": definition()}
    w = worker(state, clock, sets)
    rows = ActionsRows(segment_log.root(state))
    walk(rows, T, [None] * 140)  # 70 s analyzed, nobody tracked
    run(w, clock)
    w.checkpoint()
    assert written(state, segment_log.ZONE_VISITS) == []
    (coverage,) = written(state, segment_log.ZONE_COVERAGE)
    assert (coverage["subject_kind"], coverage["zones"]) == ("person", ["zone-litter"])
    assert coverage["ts_end"] - coverage["ts_start"] >= 60
    rows.close()


def test_a_visit_open_at_a_restart_continues_from_the_checkpoint_and_is_written_once(state):
    clock, sets = Clock(), {"cam0": definition()}
    rows = ActionsRows(segment_log.root(state))
    first = worker(state, clock, sets)
    after = walk(rows, T, [OUT, OUT, IN, IN, IN, IN])
    run(first, clock)
    first.checkpoint()
    first.writers.close()  # the service stops
    again = worker(state, clock, sets)
    walk(rows, after, [IN, IN, *[OUT] * 6])
    run(again, clock)
    again.checkpoint()
    visits = written(state, segment_log.ZONE_VISITS)
    assert [(v["entry_seen"], v["dwell_s"]) for v in visits] == [(True, pytest.approx(2.5))]  # 1 s to 3.5 s, across it
    rows.close()


def test_saving_new_zones_ends_the_open_visits_under_the_old_ones(state):
    clock, sets = Clock(), {"cam0": definition()}
    w = worker(state, clock, sets)
    rows = ActionsRows(segment_log.root(state))
    after = walk(rows, T, [OUT, OUT, IN, IN, IN, IN])
    run(w, clock)
    sets["cam0"] = definition("zone-door", "Door", at=1.0)
    clock.now += Z.ZONES_S
    walk(rows, after, [IN, IN])
    run(w, clock)
    w.checkpoint()
    assert [v["ended"] for v in written(state, segment_log.ZONE_VISITS)] == ["zones_changed"]
    rows.close()


def test_a_camera_without_zones_writes_nothing(state):
    clock = Clock()
    w = worker(state, clock, {})
    rows = ActionsRows(segment_log.root(state))
    walk(rows, T, [OUT, IN, IN, IN, *[OUT] * 6])
    run(w, clock)
    w.checkpoint()
    assert written(state, segment_log.ZONE_VISITS) == written(state, segment_log.ZONE_COVERAGE) == []
    rows.close()


def test_visits_written_after_the_last_checkpoint_are_written_once_after_a_crash(state):
    clock, sets = Clock(), {"cam0": definition()}
    rows = ActionsRows(segment_log.root(state))
    first = worker(state, clock, sets)
    after = walk(rows, T, [OUT, OUT, IN, IN, IN, IN])
    run(first, clock, rounds=2)
    first.checkpoint()
    walk(rows, after, [*[OUT] * 6])  # the visit ends and is written...
    run(first, clock, rounds=3)  # ...within the checkpoint interval, so no checkpoint covers it
    assert len(written(state, segment_log.ZONE_VISITS)) == 1
    first.writers.close()  # then the process dies
    again = worker(state, clock, sets)  # reopens at the checkpoint and reads those frames again
    run(again, clock)
    again.checkpoint()
    assert len(written(state, segment_log.ZONE_VISITS)) == 1
    rows.close()


def test_someone_in_view_then_gone_is_one_presence_episode_with_or_without_zones(state):
    clock = Clock()
    w = worker(state, clock, {})  # presence needs no zones
    rows = ActionsRows(segment_log.root(state))
    walk(rows, T, [None, None, *[OUT] * 6, *[None] * 30])  # in view 1 s to 3.5 s, then nobody for 15 s
    run(w, clock)
    w.checkpoint()
    (ep,) = written(state, segment_log.PRESENCE_SPANS)
    assert (ep["ts_start"] - T / 1e9, ep["ts_end"] - T / 1e9) == (pytest.approx(1.0), pytest.approx(3.5))
    assert (ep["tracks"], ep["left_censored"], ep["right_censored"]) == (1, False, False)
    rows.close()


def test_run_starts_a_step_at_most_once_a_batch_and_a_stop_ends_the_pause_at_once(state, monkeypatch):
    w = worker(state, Clock(), {})  # the fake clock stands still: every step looks instant, so each pause is whole
    began: list[float] = []

    async def go() -> float:
        loop, stop = asyncio.get_running_loop(), asyncio.Event()

        def step() -> None:
            began.append(time.monotonic())
            if len(began) == 3:
                loop.call_soon_threadsafe(stop.set)

        monkeypatch.setattr(w, "step", step)
        start = time.monotonic()
        await w.run(stop, batch_s=0.5)
        return time.monotonic() - start

    took = asyncio.run(go())
    assert len(began) == 3
    assert min(b - a for a, b in itertools.pairwise(began)) >= 0.45, "a step only once a batch"
    assert took < 1.3, "the stop after the third step didn't wait out a third pause (1.5 s)"
