"""pack-see-actions' loop (see/actions_worker.py): every frame fed in order, rows flushed and synced, a clean stop."""

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator, Callable, Generator
from typing import Any

import numpy as np
import pytest

from liberational import config
from liberational.see import actions, actions_worker
from liberational.see import hearts as hearts_module
from liberational.see.vision_backend import People
from test_see_actions import STEP, Decoded, Rows, T, frame_id, person


class Still:
    """A backend with one person standing still in every frame."""

    name = "still"

    def __init__(self, capturing: frozenset[str] = frozenset()) -> None:
        self.capturing = capturing  # the frames whose work captures a CUDA graph

    def upload(self, frame: Any) -> str:
        return frame.id

    def captures(self, frame: Any) -> bool:
        return frame.id in self.capturing

    def size(self, image: str) -> tuple[int, int]:
        return 100, 200

    def people(self, image: str) -> People:
        return person(0.1)

    def embed_animals(self, image: str, boxes: np.ndarray) -> np.ndarray:
        return np.zeros((len(boxes), 384), np.float32)

    def verify_animals(self, image: str, boxes: np.ndarray, species: list[str]) -> list[bool]:
        return [False] * len(boxes)

    def crop(self, image: str, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.zeros((y2 - y1, x2 - x1, 3), np.uint8)

    def score(self, windows: np.ndarray) -> np.ndarray:
        return np.zeros((len(windows), 120), np.float32)

    def shift(self, image: str) -> tuple[float, float] | None:
        return None


async def frames(n: int, stop: asyncio.Event | None = None, stop_after: int = -1) -> AsyncIterator[Decoded]:
    for i in range(n):
        if i == stop_after and stop is not None:
            stop.set()
        yield Decoded(frame_id(i), T + i * STEP)


def pipeline() -> tuple[actions.ActionPipeline, Rows, Rows]:
    poses, coverage = Rows(), Rows()
    return actions.ActionPipeline(Still(), poses, coverage, "m"), poses, coverage


def test_every_frame_is_analyzed_and_the_last_rows_are_flushed_and_synced():
    p, poses, coverage = pipeline()
    synced: list[int] = []
    seen = asyncio.run(actions_worker.follow(frames(10), p, asyncio.Event(), sync=lambda: synced.append(1)))
    assert seen == 10
    assert [f for a in poses.appends for f in a["frame_id"]] == [frame_id(i) for i in range(actions.MIN_HITS - 1, 10)]
    assert [int(a["frames"][0]) for a in coverage.appends] == [10]
    assert synced  # the closing sync, after the closing flush


def test_a_stop_ends_the_loop_after_the_frame_in_hand():
    p, poses, _ = pipeline()
    stop = asyncio.Event()
    seen = asyncio.run(actions_worker.follow(frames(10, stop, stop_after=4), p, stop))
    assert seen == 5
    assert sum(len(a["frame_id"]) for a in poses.appends) == 5 - (actions.MIN_HITS - 1)


def test_rows_are_synced_every_few_seconds_while_it_runs():
    p, _, _ = pipeline()
    synced: list[int] = []
    ticks = iter(np.arange(0.0, 100.0, 1.0))  # one second a frame on the loop's clock
    asyncio.run(
        actions_worker.follow(
            frames(12), p, asyncio.Event(), sync=lambda: synced.append(1), clock=lambda: float(next(ticks))
        )
    )
    assert len(synced) == 12 // int(actions_worker.SYNC_S) + 1  # every SYNC_S, then the closing sync


def test_a_camera_is_analyzed_at_its_rate_and_a_gap_always_gets_through():
    p, _, coverage = pipeline()
    gap_at = 7

    async def frames_with_gap() -> AsyncIterator[Decoded]:
        for i in range(12):
            yield Decoded(frame_id(i), T + i * STEP, gap_before=i == gap_at)

    every = 100_000_000  # 10 fps against 25 fps frames, 40 ms apart
    seen = asyncio.run(actions_worker.follow(frames_with_gap(), p, asyncio.Event(), every_ns=every))
    # Due at 0, 100, and 200 ms (each taken from 25 ms early): frames 0 (0 ms), 2 (80), and 5 (200). The gap's frame
    # 7 (280) always goes and restarts the schedule: 9 (360) is due at 380, and 11 (440) is early for 480.
    assert seen == 5
    assert sum(int(a["frames"][0]) for a in coverage.appends) == 5


def test_a_jittery_source_at_the_target_rate_keeps_every_frame():
    p, _, _ = pipeline()
    jitter = [0, 9, -12, 14, -8, 11, -14, 6, -9, 13]  # ms around a 10 fps source's 100 ms steps

    async def jittery() -> AsyncIterator[Decoded]:
        for i, ms in enumerate(jitter):
            yield Decoded(frame_id(i), T + i * 100_000_000 + ms * 1_000_000)

    assert asyncio.run(actions_worker.follow(jittery(), p, asyncio.Event(), every_ns=100_000_000)) == 10


def test_a_cameras_rate_is_its_own_setting_or_the_default(monkeypatch):
    settings = {"vision.actions_fps": 10.0, "vision.actions_camera_fps": {"cam0": 30.0}}
    monkeypatch.setattr(actions_worker.cfg, "get", lambda key, default=None: settings.get(key, default))
    assert actions_worker.fps_of("cam0") == 30.0
    assert actions_worker.fps_of("patio") == 10.0


@pytest.mark.parametrize("devices,rate,expected", [(1, 30, 0), (2, 30, 1), (3, 30, 2), (2, 10, 0)])
def test_empty_gpu_overrides_keep_rate_based_routing(devices, rate, expected):
    with config.override({"vision": {"actions_camera_gpus": {}}}):
        assert actions_worker.gpu_for(rate, 30, devices, camera="front") == expected


def test_camera_gpu_override_moves_only_the_named_camera():
    with config.override({"vision": {"actions_camera_gpus": {"front": 0, "side": 1}}}):
        assert actions_worker.gpu_for(30, 30, 2, camera="front") == 0
        assert actions_worker.gpu_for(10, 30, 2, camera="side") == 1
        assert actions_worker.gpu_for(30, 30, 2, camera="rear") == 1
        assert actions_worker.gpu_for(10, 30, 2, camera="porch") == 0


@pytest.mark.parametrize("device", [-1, 2, 0.5, "0", True])
def test_invalid_camera_gpu_indexes_fail_before_model_loading(device):
    with (
        config.override({"vision": {"actions_camera_gpus": {"front": device}}}),
        pytest.raises(ValueError, match=r"vision.actions_camera_gpus.front must be a CUDA device index"),
    ):
        actions_worker.gpu_for(30, 30, 2, camera="front")


def test_invalid_gpu_override_exits_before_camera_threads_or_models(monkeypatch, capsys):
    import torch

    from liberational.vision import faces

    # A machine with two GPUs, whether or not this one has any: `main` asks `choose()` for a CUDA backend first, so
    # without this the test measures the no-GPU path (which rests NO_BACKEND_WAIT_S and exits 1) on a machine without
    # CUDA, such as the hosted runner.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(faces, "app", lambda: pytest.fail("loaded models before validating GPU placement"))
    with config.override({"vision": {"actions_camera_gpus": {"front": 2}}}), pytest.raises(SystemExit) as error:
        actions_worker.main(["--camera", "front", "--backend", "cuda"])
    assert error.value.code == 2
    assert "vision.actions_camera_gpus.front must be a CUDA device index" in capsys.readouterr().err


def test_without_a_gpu_it_says_so_once_rests_and_exits_before_loading_models(monkeypatch, caplog):
    import torch

    from liberational.vision import faces

    rested = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(faces, "app", lambda: pytest.fail("loaded models with no backend to run them"))
    monkeypatch.setattr(actions_worker, "_rest", lambda stop, seconds: rested.append(seconds))
    with pytest.raises(SystemExit) as error:
        actions_worker.main(["--camera", "front"])
    assert error.value.code == 1  # a failure, so the service manager starts a fresh process that looks again
    assert rested == [actions_worker.NO_BACKEND_WAIT_S] and actions_worker.NO_BACKEND_WAIT_S >= 60
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "needs a CUDA GPU or an Apple Silicon Mac" in errors[0]


class Source:
    """A camera's frames for a fake follow: each one counted, until the source goes away (it raises) or `stop`."""

    def __init__(self, up: bool = True) -> None:
        self.up, self.frames, self.started = threading.Event(), 0, 0
        if up:
            self.up.set()

    def follow(self, stop: threading.Event) -> None:
        self.started += 1
        while not stop.is_set():
            if not self.up.is_set():
                raise ConnectionError("the decoder refused the camera")
            self.frames += 1
            stop.wait(0.001)


def _until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        threading.Event().wait(0.005)
    return condition()


def _start(names: list[str], load, follow, stop: threading.Event) -> tuple[threading.Thread, list[bool]]:
    result: list[bool] = []
    runner = threading.Thread(
        target=lambda: result.append(
            actions_worker.run_cameras(names, load, follow, stop, retry_s=0.01, retry_max_s=0.02)
        ),
        daemon=True,
    )
    runner.start()
    return runner, result


def _finish(runner: threading.Thread, result: list[bool]) -> bool:
    runner.join(5)
    assert not runner.is_alive(), "a camera thread is still running"
    return result[0]


def test_a_camera_that_fails_to_load_leaves_the_others_running_and_comes_back_when_it_loads(caplog):
    # Back's fisheye entry is broken until it's fixed; front runs on meanwhile, and back is tried again.
    sources = {"front": Source(), "back": Source()}
    fixed, tries = threading.Event(), {"front": 0, "back": 0}

    def load(name: str) -> str:
        tries[name] += 1
        if name == "back" and not fixed.is_set():
            raise ValueError("a malformed fisheye entry")
        return f"{name}-backend"

    stop = threading.Event()
    runner, result = _start(["front", "back"], load, lambda name, backend, gate: sources[name].follow(stop), stop)
    assert _until(lambda: sources["front"].frames > 20 and tries["back"] >= 3)
    assert not stop.is_set() and sources["back"].started == 0
    fixed.set()
    assert _until(lambda: sources["back"].frames > 0)
    stop.set()
    assert _finish(runner, result)
    assert tries["front"] == 1
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "back" in errors[0].getMessage()  # once, not a line every try


def test_a_camera_whose_source_goes_away_is_followed_again_on_the_same_backend_when_it_returns():
    sources = {"front": Source(), "back": Source(up=False)}
    loads: list[str] = []
    stop = threading.Event()

    def load(name: str) -> str:
        loads.append(name)
        return name

    runner, result = _start(["front", "back"], load, lambda name, backend, gate: sources[name].follow(stop), stop)
    assert _until(lambda: sources["back"].started >= 3 and sources["front"].frames > 20)
    sources["back"].up.set()
    assert _until(lambda: sources["back"].frames > 0)
    stop.set()
    assert _finish(runner, result)
    assert sorted(loads) == ["back", "front"]  # its backend kept, not loaded again for each try


def test_it_stops_and_says_so_once_no_camera_can_run():
    # Every camera down at once: the process exits, for the service manager to start a fresh one.
    sources = {"front": Source(), "back": Source(up=False)}
    stop = threading.Event()
    runner, result = _start(
        ["front", "back"], lambda name: name, lambda name, b, gate: sources[name].follow(stop), stop
    )
    assert _until(lambda: sources["front"].frames > 0)
    assert not stop.is_set()
    sources["front"].up.clear()
    assert not _finish(runner, result)
    assert stop.is_set()


def test_a_camera_that_fails_while_another_waits_to_load_is_not_the_last_one_down():
    # The cameras not yet tried aren't down: front failing first must not stop back, still loading.
    loading, release = threading.Event(), threading.Event()
    stop = threading.Event()
    back = Source()

    def load(name: str) -> str:
        if name == "front":
            raise ValueError("front's model is missing")
        loading.set()
        release.wait(5)
        return name

    runner, result = _start(["front", "back"], load, lambda name, b, gate: back.follow(stop), stop)
    assert loading.wait(5)
    release.set()
    assert _until(lambda: back.frames > 0)
    assert not stop.is_set()
    stop.set()
    assert _finish(runner, result)


def test_a_camera_that_keeps_failing_is_still_tried_after_a_thousand_failures():
    # The backoff's 2 ** streak stopped converting to a float at 1025 failures in a row, and the
    # OverflowError ended the camera's supervisor thread, so it was never tried again.
    tries = {"front": 0, "back": 0}
    front = Source()
    stop = threading.Event()

    def load(name: str) -> str:
        tries[name] += 1
        if name == "back":
            raise ValueError("back's camera is unplugged")
        return name

    runner = threading.Thread(
        target=actions_worker.run_cameras,
        args=(["front", "back"], load, lambda name, b, gate: front.follow(stop), stop),
        kwargs={"retry_s": 1e-9, "retry_max_s": 1e-9},
        daemon=True,
    )
    runner.start()
    assert _until(lambda: tries["back"] > 1100, timeout=20), f"back was tried {tries['back']} times, then no more"
    stop.set()
    runner.join(5)
    assert not runner.is_alive()


def test_a_load_waits_for_the_frames_in_hand_and_holds_new_ones_back():
    # CUDA graph capture is global: no camera's GPU work may run while another's backend loads.
    gate = actions_worker.GpuGate()
    order: list[str] = []
    in_frame, release = threading.Event(), threading.Event()

    def frame_in_hand() -> None:
        with gate.running():
            in_frame.set()
            release.wait(5)
            order.append("frame done")

    def a_load() -> None:
        with gate.loading():
            order.append("load")

    def a_new_frame() -> None:
        with gate.running():
            order.append("new frame")

    threads = [threading.Thread(target=frame_in_hand), threading.Thread(target=a_load)]
    threads[0].start()
    assert in_frame.wait(5)
    threads[1].start()
    assert _until(lambda: gate._waiting == 1)  # the load is waiting for the frame in hand
    threads.append(threading.Thread(target=a_new_frame))
    threads[2].start()
    threading.Event().wait(0.05)  # time for a new frame that isn't held back to run first
    release.set()
    for thread in threads:
        thread.join(5)
    assert order == ["frame done", "load", "new frame"]


def test_each_frames_work_is_inside_the_gate_and_the_wait_for_the_next_is_not():
    held: list[bool] = []
    inside = threading.Event()

    @contextlib.contextmanager
    def gate() -> Generator[None]:
        inside.set()
        try:
            yield
        finally:
            inside.clear()

    async def watched() -> AsyncIterator[Decoded]:
        async for frame in frames(3):
            held.append(inside.is_set())  # while the next frame is awaited
            yield frame

    p, _, _ = pipeline()
    real_feed = p.feed

    def feed(frame: Any) -> None:
        held.append(inside.is_set())
        real_feed(frame)

    p.feed = feed
    asyncio.run(actions_worker.follow(watched(), p, asyncio.Event(), gate=gate))
    assert held == [False, True, False, True, False, True]


def test_a_frame_whose_work_captures_a_graph_runs_alone_and_the_rest_inside_the_shared_gate():
    held: list[str] = []
    where: list[str] = []

    def gate_of(name: str) -> Callable[[], contextlib.AbstractContextManager[None]]:
        @contextlib.contextmanager
        def gate() -> Generator[None]:
            where.append(name)
            try:
                yield
            finally:
                where.pop()

        return gate

    poses, coverage = Rows(), Rows()
    p = actions.ActionPipeline(Still(frozenset({frame_id(0)})), poses, coverage, "m")
    real_feed = p.feed

    def feed(frame: Any) -> None:
        held.append(where[-1])
        real_feed(frame)

    p.feed = feed
    asyncio.run(actions_worker.follow(frames(3), p, asyncio.Event(), gate=gate_of("shared"), alone=gate_of("alone")))
    assert held == ["alone", "shared", "shared"]


class Appended:
    """A `segment_log.Writers` that keeps what was appended."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = {}

    def append(self, stream: str, cam: str, records: list[dict[str, Any]]) -> None:
        self.rows.setdefault(stream, []).extend(records)


def test_the_heart_rate_switch_builds_one_or_none():
    with config.override({"vision": {"heart_rate": False}}):
        assert actions_worker.hearts_for("cam0", 30.0, Appended(), None) is None
    with config.override({"vision": {"heart_rate": True}}):
        built = actions_worker.hearts_for("cam0", 30.0, Appended(), None)
        assert isinstance(built, hearts_module.Hearts)
        assert built.measuring, "a 30 fps camera is measured"


def test_a_camera_at_ten_frames_a_second_gets_a_heart_rate_that_measures_nothing():
    with config.override({"vision": {"heart_rate": True}}):
        built = actions_worker.hearts_for("patio", 10.0, Appended(), None)
        assert built is not None and not built.measuring
