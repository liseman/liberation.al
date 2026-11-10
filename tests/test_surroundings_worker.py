"""Learning surroundings (#256): the sampler's checks on each frame, and the service that follows the switch."""

import asyncio
import itertools
import json
import math

import numpy as np
import pytest

from liberational.spatial import surroundings
from liberational.spatial import surroundings_frames as frames
from liberational.spatial import surroundings_worker as worker
from liberational.vision import crossings


def picture(seed: int, gain: float = 1.0) -> np.ndarray:
    """A made-up 64 x 36 frame; a different seed is a different picture, and `gain` brightens the same one."""
    noise = np.random.default_rng(seed).integers(40, 160, (36, 64, 3)).astype(np.float64)
    return np.clip(noise * gain, 0, 255).astype(np.uint8)


def direction(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)], dtype=np.float32)


class Scene:
    """What the sampler's models would say of each frame, set by the test: keyed by the frame's first pixel."""

    def __init__(self):
        self.angle: dict[int, float] = {}
        self.sharp: dict[int, float] = {}
        self.dark: set[int] = set()
        self.living: set[int] = set()
        self.described = 0

    def key(self, pixels: np.ndarray) -> int:
        return int(pixels[0, 0, 0]) * 65536 + int(pixels[0, 0, 1]) * 256 + int(pixels[0, 0, 2])

    def add(self, seed: int, angle: float = 0.0, sharp: float = 100.0, dark: bool = False, living: bool = False):
        frame = picture(seed)
        k = self.key(frame)
        self.angle[k], self.sharp[k] = angle, sharp
        if dark:
            self.dark.add(k)
        if living:
            self.living.add(k)
        return frame

    def describe(self, pixels):
        self.described += 1
        return direction(self.angle[self.key(pixels)])


@pytest.fixture
def scene():
    return Scene()


@pytest.fixture
def sampler(tmp_path, scene):
    return worker.Sampler(
        frames.Store(tmp_path / "surroundings"),
        scene.describe,
        lambda p: scene.key(p) in scene.living,
        lambda p: scene.key(p) not in scene.dark,
        lambda p: scene.sharp[scene.key(p)],
    )


def test_a_still_camera_keeps_its_first_frame_and_stops_at_the_thumbnail_after(sampler, scene):
    frame = scene.add(1)
    assert [sampler.look("front", frame, float(t)) for t in range(4)] == ["view", "same", "same", "same"]
    assert scene.described == 1  # the models ran once, not once a sample
    assert len(sampler.store.kept) == 1


def test_each_check_skips_its_frame_before_the_gate_sees_it(sampler, scene):
    sampler.look("front", scene.add(1), 0.0)
    assert sampler.look("front", scene.add(2, dark=True), 5.0) == "dark"
    assert sampler.look("front", scene.add(3, living=True), 10.0) == "person"
    assert sampler.look("front", scene.add(4, angle=5), 15.0) == "copy"
    assert sampler.look("front", scene.add(5, angle=90), 20.0) == "view"
    assert scene.described == 3  # the kept two and the copy: never the dark frame or the one with someone in it
    assert [f.reason for f in sampler.store.kept] == ["view", "view"]


def test_a_frame_much_less_sharp_than_the_cameras_usual_is_skipped_once_there_is_a_usual(sampler, scene):
    # The first five looked-at frames set what's usual (100); none of them can be called blurred, however soft.
    outcomes = [sampler.look("front", scene.add(10 + i, angle=72 * i, sharp=100.0), float(i)) for i in range(5)]
    assert "blurred" not in outcomes
    assert sampler.look("front", scene.add(20, angle=36, sharp=49.0), 30.0) == "blurred"
    assert sampler.look("front", scene.add(21, angle=36, sharp=51.0), 35.0) == "changed"  # 0.81 alike to two kept views


def test_a_scene_that_drifts_slowly_gets_past_the_thumbnail_once_it_has_drifted_far_enough(sampler):
    base = picture(1).astype(np.float64)
    other = picture(2).astype(np.float64)
    steps = [np.clip(base + (other - base) * k / 40, 0, 255).astype(np.uint8) for k in range(12)]
    thumbs = [worker.thumbnail(s) for s in steps]
    assert all(float(a @ b) >= worker.SAME_THUMB for a, b in itertools.pairwise(thumbs))  # each step is small
    assert float(thumbs[0] @ thumbs[-1]) < worker.SAME_THUMB  # the whole drift is not
    sampler.describe = lambda p: direction(0)
    sampler.sharpness = lambda p: 100.0
    sampler.lit = lambda p: True
    sampler.detect = None
    outcomes = [sampler.look("front", s, float(i)) for i, s in enumerate(steps)]
    assert outcomes[0] == "view"
    assert "copy" in outcomes[1:]  # a later frame went through to the gate, which found it a copy
    assert outcomes.count("same") >= 5


def test_an_exposure_step_alone_is_still_the_same_thumbnail():
    assert float(worker.thumbnail(picture(1)) @ worker.thumbnail(picture(1, gain=1.3))) >= worker.SAME_THUMB
    assert float(worker.thumbnail(picture(1)) @ worker.thumbnail(picture(2))) < 0.5


def test_each_camera_has_its_own_checks_and_counts(sampler, scene):
    frame = scene.add(1)
    assert sampler.look("front", frame, 0.0) == "view"
    assert sampler.look("hall", frame, 1.0) == "copy"  # hall's first look: not "same", and the store has it already
    status = sampler.status(surroundings.Switch(True, ("front", "hall"), (), "front"), now=9.0)
    assert status["kept"] == 1
    assert status["sampling"] == ["front", "hall"]
    assert (status["cameras"]["front"]["view"], status["cameras"]["front"]["last_kept"]) == (1, 0.0)
    assert (status["cameras"]["hall"]["copy"], status["cameras"]["hall"]["last_kept"]) == (1, None)


def logits(**scores: float) -> np.ndarray:
    """RF-DETR logits for one query: the named classes at the given scores, every other class near zero."""
    row = np.full(91, -10.0)
    slots = {"person": crossings.RF_CLASS_IDS[crossings.COCO_PERSON], "cat": crossings.RF_CLASS_IDS[15]}
    for name, score in scores.items():
        row[slots[name]] = math.log(score / (1 - score))
    return row[None]


@pytest.mark.parametrize(
    ("scores", "found"),
    [({}, 0), ({"person": 0.49}, 0), ({"person": 0.51}, 1), ({"cat": 0.34}, 0), ({"cat": 0.36}, 1)],
    ids=["nothing", "an unsure person", "a person", "an unsure cat", "a cat"],
)
def test_a_person_or_an_animal_sure_enough_is_someone_in_the_frame(scores, found):
    assert worker.living(logits(**scores)) == found


class Cameras:
    """A decoder for the service: each camera gives its queued pictures, then waits; `broken` cameras raise."""

    def __init__(self):
        self.asked: list[str] = []
        self.queues: dict[str, asyncio.Queue] = {}
        self.broken: set[str] = set()

    async def pictures(self, camera):
        self.asked.append(camera)
        if camera in self.broken:
            raise RuntimeError("the shared camera decoder refused it")
        queue = self.queues.setdefault(camera, asyncio.Queue())
        while True:
            yield await queue.get()


async def settle(check, tries: int = 200):
    for _ in range(tries):
        if check():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("never settled")


def test_the_service_follows_the_switch_and_holds_no_camera_while_off(sampler, scene, tmp_path):
    async def run():
        cams = Cameras()
        state = {"switch": surroundings.Switch(False, ("front",), (), "front")}
        idle: list[bool] = []
        service = worker.Service(
            sampler, read=lambda: state["switch"], source=cams.pictures, on_idle=lambda: idle.append(True)
        )
        service.step()
        await asyncio.sleep(0)
        assert (cams.asked, list(service.following), idle) == ([], [], [True])  # off: nothing asked of the decoder

        state["switch"] = surroundings.Switch(True, ("front",), (), "front")
        service.step()
        await settle(lambda: cams.asked == ["front"])
        cams.queues["front"].put_nowait(scene.add(1))
        await settle(lambda: len(sampler.store.kept) == 1)

        state["switch"] = surroundings.Switch(True, ("hall",), ("hall",), "front")  # the picker moved to another camera
        service.step()
        await settle(lambda: cams.asked == ["front", "hall"])
        assert list(service.following) == ["hall"]

        state["switch"] = surroundings.Switch(False, ("hall",), ("hall",), "front")
        service.step()
        assert list(service.following) == []
        service.write_status(state["switch"], force=True)
        status = json.loads((sampler.store.folder / worker.STATUS).read_text())
        assert (status["on"], status["sampling"], status["kept"], status["cameras"]["front"]["view"]) == (
            False,
            [],
            1,
            1,
        )
        stop = asyncio.Event()
        stop.set()
        await service.run(stop)

    asyncio.run(run())


def test_a_camera_the_decoder_refuses_is_left_for_a_while_and_the_others_carry_on(sampler, scene):
    async def run():
        cams = Cameras()
        cams.broken.add("hall")
        switch = surroundings.Switch(True, ("front", "hall"), ("front", "hall"), "front")
        clock = {"now": 100.0}
        service = worker.Service(sampler, read=lambda: switch, source=cams.pictures, monotonic=lambda: clock["now"])
        service.step()
        await settle(lambda: sorted(cams.asked) == ["front", "hall"])
        await settle(lambda: service.following["hall"].done())
        service.step()
        assert list(service.following) == ["front"]
        clock["now"] += worker.RETRY_S - 1
        service.step()
        await asyncio.sleep(0.01)
        assert cams.asked.count("hall") == 1  # not asked again yet
        clock["now"] += 2
        service.step()
        await settle(lambda: cams.asked.count("hall") == 2)
        cams.queues["front"].put_nowait(scene.add(1))
        await settle(lambda: len(sampler.store.kept) == 1)  # the working camera was sampled throughout
        stop = asyncio.Event()
        stop.set()
        await service.run(stop)

    asyncio.run(run())


def test_a_map_step_starts_while_learning_is_on_and_frames_wait_and_what_it_drops_is_dropped(sampler, scene):
    from liberational.spatial import surroundings_map

    async def run():
        steps: list[int] = []
        clock = {"now": 1000.0}

        async def step() -> int:
            # What a real step would leave: the one waiting frame, found to add nothing, is to be dropped.
            steps.append(len(sampler.store.kept))
            state = surroundings_map.State.read(sampler.store.folder)
            state.drop = [f.id for f in sampler.store.kept]
            state.write(sampler.store.folder)
            return 0

        off = surroundings.Switch(False, ("front",), (), "front")
        on = surroundings.Switch(True, ("front",), (), "front")
        service = worker.Service(
            sampler, read=lambda: on, source=Cameras().pictures, map_step=step, monotonic=lambda: clock["now"]
        )
        sampler.look("front", scene.add(1), 0.0)

        service.map_due(off)
        await asyncio.sleep(0.01)
        assert steps == []  # never while learning is off

        service.map_due(on)
        await settle(lambda: steps == [1])
        await settle(lambda: sampler.store.kept == [])  # the step's drop was carried out
        service.map_due(on)  # collects the finished look
        clock["now"] += worker.MAP_GAP_S - 1
        service.map_due(on)
        await asyncio.sleep(0.01)
        assert steps == [1]  # not again before the gap has passed

        clock["now"] += 2
        service.map_due(on)
        await asyncio.sleep(0.05)
        assert steps == [1]  # the gap has passed, but nothing is waiting: no step
        stop = asyncio.Event()
        stop.set()
        await service.run(stop)

    asyncio.run(run())
