"""Learning surroundings (#256): the sampler's checks on each frame, and the service that follows the switch."""

import asyncio
import itertools
import json
import math

import cv2
import numpy as np
import pytest

from liberational.spatial import surroundings
from liberational.spatial import surroundings_frames as frames
from liberational.spatial import surroundings_worker as worker
from liberational.vision import crossings

PERSON = (0.4, 0.5, 0.6, 1.0)  # someone standing in the lower middle: a tenth of the frame


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
        self.living: dict[int, list[tuple[float, float, float, float]]] = {}  # the boxes the detector finds
        self.described = 0

    def key(self, pixels: np.ndarray) -> int:
        return int(pixels[0, 0, 0]) * 65536 + int(pixels[0, 0, 1]) * 256 + int(pixels[0, 0, 2])

    def add(self, seed: int, angle: float = 0.0, sharp: float = 100.0, dark: bool = False, living: str = ""):
        frame = picture(seed)
        k = self.key(frame)
        self.angle[k], self.sharp[k] = angle, sharp
        if dark:
            self.dark.add(k)
        if living:  # "one": someone in a tenth of the frame; "crowd": people over most of it
            self.living[k] = [PERSON] if living == "one" else [(0.0, 0.0, 0.9, 1.0)]
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
        lambda p: scene.living.get(scene.key(p), []),
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
    assert sampler.look("front", scene.add(3, living="crowd"), 10.0) == "person"
    assert sampler.look("front", scene.add(4, angle=5), 15.0) == "copy"
    assert sampler.look("front", scene.add(5, angle=90), 20.0) == "view"
    assert scene.described == 3  # the kept two and the copy: never the dark frame or the one with someone in it
    assert [f.reason for f in sampler.store.kept] == ["view", "view"]


def test_a_look_skipped_for_the_people_in_it_is_not_what_later_looks_are_compared_with(sampler, scene):
    # People fill the frame, then leave without the thumbnail moving much: three looks of one picture, of which
    # the model says the first two are mostly people.
    frame = scene.add(1, living="crowd")
    assert [sampler.look("front", frame, float(t)) for t in range(2)] == ["person", "person"]
    scene.living.clear()
    assert sampler.look("front", frame, 10.0) == "view"
    assert sampler.look("front", frame, 15.0) == "same"
    assert [f.camera for f in sampler.store.kept] == ["front"]


def test_a_frame_with_someone_in_a_small_part_of_it_is_kept_with_them_marked(sampler, scene):
    frame = scene.add(1, living="one")
    assert sampler.look("front", frame, 0.0) == "view"
    [kept] = sampler.store.kept
    read = cv2.imread(str(sampler.store.covered_path(kept)), cv2.IMREAD_GRAYSCALE)
    assert read is not None
    marked = read > 127
    assert marked.shape == frame.shape[:2]
    # The box spans x 0.40 to 0.60 and y 0.50 to 1.00 of a 64 x 36 frame; grown by 0.15 of its size each way that is
    # x 0.37 to 0.63 (columns 23 to 40) and y 0.425 down (rows 15 on).
    assert marked[20, 30] and marked[35, 24] and marked[16, 40]
    assert not marked[20, 22] and not marked[20, 41] and not marked[14, 30]
    sampler.store.remove(kept.id)
    assert not sampler.store.covered_path(kept).exists()


def test_a_frame_with_nobody_in_it_has_nothing_marked(sampler, scene):
    sampler.look("front", scene.add(1), 0.0)
    assert not sampler.store.covered_path(sampler.store.kept[0]).exists()


def test_a_blurred_look_is_not_what_later_looks_are_compared_with(sampler, scene):
    for i in range(5):
        sampler.look("front", scene.add(10 + i, angle=72 * i, sharp=100.0), float(i))
    soft = scene.add(20, angle=36, sharp=10.0)
    assert sampler.look("front", soft, 30.0) == "blurred"
    scene.sharp[scene.key(soft)] = 100.0  # the camera refocused on the same picture
    assert sampler.look("front", soft, 35.0) == "changed"


def test_when_the_kept_frame_of_a_view_is_dropped_the_view_is_kept_again(sampler, scene):
    frame = scene.add(1)
    assert [sampler.look("front", frame, float(t)) for t in range(2)] == ["view", "same"]
    sampler.store.remove(sampler.store.kept[0].id)
    assert sampler.look("front", frame, 10.0) == "view"
    assert len(sampler.store.kept) == 1


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
    box = np.array([[0.5, 0.75, 0.2, 0.5]])  # centre and size, as the detector gives it
    got = worker.living(box, logits(**scores))
    assert len(got) == found
    assert got == [] or got[0] == pytest.approx(PERSON)


def test_a_box_past_the_frames_edge_is_clipped_to_it():
    assert worker.living(np.array([[0.05, 0.5, 0.3, 0.2]]), logits(person=0.9)) == [pytest.approx((0.0, 0.4, 0.2, 0.6))]


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


def test_the_status_says_what_a_cameras_last_looks_came_to_and_forgets_older_ones(sampler, scene):
    frame = scene.add(1)
    assert sampler.look("front", frame, 0.0) == "view"
    for t in range(1, worker.RECENT):
        sampler.look("front", frame, float(t))
    switch = surroundings.Switch(True, ("front",), (), "front")
    assert sampler.status(switch, now=20.0)["cameras"]["front"]["recent"] == {"view": 1, "same": worker.RECENT - 1}
    # one more look pushes the kept one out of the window: the camera has kept nothing lately
    sampler.look("front", scene.add(2, living=True), 30.0)
    front = sampler.status(switch, now=31.0)["cameras"]["front"]
    assert front["recent"] == {"same": worker.RECENT - 1, "person": 1}
    assert (front["view"], front["same"]) == (1, worker.RECENT - 1)  # the counts since the start keep it
