"""The see-actions bench (tools/bench_actions.py): its frames are real NV12, its pacing counts what a slow
machine misses, and its report says so."""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest
import torch

PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "bench_actions.py"
spec = importlib.util.spec_from_file_location("bench_actions", PATH)
assert spec and spec.loader
bench = importlib.util.module_from_spec(spec)
sys.modules["bench_actions"] = bench
spec.loader.exec_module(bench)


def test_a_frame_becomes_the_nv12_the_backends_decode():
    from liberational.see import gpu_pose

    bgr = bench.synthetic(64, 96)
    nv12 = bench.bgr_to_nv12(bgr)
    assert nv12.shape == (96, 96)
    back = gpu_pose.nv12_to_bgr(torch.from_numpy(nv12), 64, 96, color_range="limited", color_matrix="bt601")
    # YUV 4:2:0 rounds and halves the chroma: a smooth frame comes back within a few levels, colors in place.
    assert np.abs(back.numpy().astype(int) - bgr.astype(int)).mean() < 3


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


class Backend:
    """Takes `cost` seconds of the clock per people call, and finds `found` people."""

    def __init__(self, clock: Clock, cost: float, found: int = 0) -> None:
        self.clock, self.cost, self.found, self.scored = clock, cost, found, []
        self.uploads = 0

    def captures(self, frame):
        return self.uploads == 0  # the first frame captures the stream's graphs

    def upload(self, frame):
        self.uploads += 1
        return frame.pixels

    def size(self, image):
        return 1080, 1920

    def people(self, image):
        from liberational.see.vision_backend import People

        self.clock.now += self.cost
        boxes = np.tile(np.array([[0.4, 0.2, 0.6, 0.8, 0.9]], np.float32), (self.found, 1))
        return People(boxes, np.zeros((self.found, 26, 3), np.float32))

    def crop(self, image, box):
        return np.zeros((box[3] - box[1], box[2] - box[0], 3), np.uint8)

    def score(self, windows):
        self.scored.append(len(windows))
        return np.zeros((len(windows), 120), np.float32)


def frames():
    while True:
        yield np.zeros((6, 4), np.uint8)


@pytest.mark.parametrize(
    ("cost", "frames_done", "dropped"),
    [
        pytest.param(0.001, 30, 0, id="keeps-up"),
        pytest.param(0.1, 10, 20, id="too-slow-drops-and-counts"),  # 0.1 s a frame: 1 of every 3 slots
    ],
)
def test_a_stream_is_paced_and_a_slow_one_counts_its_dropped_slots(cost, frames_done, dropped):
    clock = Clock()
    backend = Backend(clock, cost, found=2)
    stats = bench.Stream()
    bench.run_stream(backend, frames(), stats, fps=30, seconds=1.0, people=0, clock=clock, sleep=clock.sleep)
    assert (stats.frames, stats.dropped) == (frames_done, dropped)
    assert len(stats.times["people"]) == frames_done
    assert backend.scored and set(backend.scored) == {2}, "one window per person found"
    assert 1 <= len(stats.times["crop"]) // 2 <= 2, "a head crop per person about once a second"


def test_the_report_gives_each_stage_and_each_streams_rate():
    stats = bench.Stream(frames=60, dropped=3, seconds=2.0, providers={"detector": "CoreMLExecutionProvider"})
    stats.times["people"] = [0.004, 0.006]
    lines = bench.report([stats, bench.Stream(error="ValueError: boom")], cpu_seconds=1.0, wall=2.0)
    assert any(line.split()[:2] == ["people", "2"] for line in lines)
    assert "stream 0: 30.0 fps, 60 frames, 3 dropped" in lines
    assert "stream 1: failed: ValueError: boom" in lines
    assert "cpu: 50% of one core over 2.0 s" in lines
    assert "models: detector on CoreMLExecutionProvider" in lines


def test_a_frame_that_captures_a_graph_runs_alone_and_the_rest_inside_the_shared_gate():
    import contextlib

    held = []

    def holding(name):
        @contextlib.contextmanager
        def hold():
            held.append(name)
            yield

        return hold

    clock = Clock()
    bench.run_stream(
        Backend(clock, 0.001),
        frames(),
        bench.Stream(),
        fps=30,
        seconds=0.1,
        people=0,
        clock=clock,
        sleep=clock.sleep,
        gate=holding("running"),
        alone=holding("alone"),
    )
    assert held == ["alone", "running", "running"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_streams_backend_loads_on_a_cuda_stream_of_its_own_thread(monkeypatch):
    # Sessions queue on, and their graphs replay on, the loading thread's own stream; on the default stream
    # they take the eager, synchronized path, which is not what runs live.
    import threading

    from liberational.see import vision_backend

    seen = {}

    def load(models, device, backend):
        seen["device"] = device
        seen["stream"] = torch.cuda.current_stream(device).cuda_stream
        return "backend"

    monkeypatch.setattr(vision_backend, "choose", lambda backend: "cuda")
    monkeypatch.setattr(vision_backend, "load", load)
    thread = threading.Thread(target=lambda: seen.setdefault("loaded", bench.load_stream(PATH, "cuda", 0)))
    thread.start()
    thread.join()
    assert seen["loaded"] == "backend"
    assert seen["device"] == 0
    assert seen["stream"] != 0, "not the legacy default stream"
    assert torch.cuda.current_stream(0).cuda_stream == 0, "the stream is the loading thread's, not this one's"
