import base64
import io
import multiprocessing
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from liberational.vision import segmentation

BOX = [0.1, 0.1, 0.5, 0.5]
MASK = np.zeros((60, 80), dtype=bool)
MASK[10:30, 20:40] = True


def image_bytes():
    out = io.BytesIO()
    Image.new("RGB", (80, 60), "blue").save(out, format="JPEG")
    return out.getvalue()


@pytest.mark.parametrize(
    "prompt",
    [{}, {"bbox": [0, 0, 1, 1], "point": [0.5, 0.5]}, {"bbox": [0.9, 0, 0.5, 1]}, {"point": [float("nan"), 0.5]}],
)
def test_invalid_prompts_are_refused_before_model_work(prompt, monkeypatch):
    called = []
    monkeypatch.setattr(segmentation, "_run", lambda *args: called.append(args))
    with pytest.raises(ValueError):
        segmentation.segment(image_bytes(), **prompt)
    assert called == []


def test_segment_returns_the_actual_model_mask_and_score(monkeypatch):
    mask = np.zeros((60, 80), dtype=bool)
    mask[10:30, 20:40] = True
    calls = []

    def infer(target, args, timeout):
        calls.append(args[1])
        return mask, 0.9

    monkeypatch.setattr(segmentation, "_run", infer)
    monkeypatch.setattr(segmentation.cfg, "hf_repo", lambda name: "fixture/edgetam")
    result = segmentation.segment(image_bytes(), bbox=[0.1, 0.1, 0.5, 0.5])
    assert np.array_equal(result.mask, mask)
    assert result.score == 0.9
    assert result.model == "fixture/edgetam"
    assert calls == [{"bbox": [0.1, 0.1, 0.5, 0.5]}]


@pytest.mark.parametrize("mask", [np.zeros((60, 80), dtype=bool), np.ones((3, 4), dtype=bool)])
def test_empty_or_wrong_sized_model_masks_are_not_saved(mask, monkeypatch):
    monkeypatch.setattr(segmentation, "_run", lambda *args: (mask, 0.8))
    with pytest.raises(segmentation.Unavailable, match="usable object mask"):
        segmentation.segment(image_bytes(), point=[0.5, 0.5])


def test_another_active_segmentation_is_refused_instead_of_queued():
    assert segmentation._lock.acquire(blocking=False)
    try:
        with pytest.raises(segmentation.Unavailable, match="already selecting"):
            segmentation.segment(image_bytes(), point=[0.5, 0.5])
    finally:
        segmentation._lock.release()


class FrozenClock:
    """The segmentation module's clock, held still, so the idle boundary is exactly IDLE_S: on a live clock
    `(used + IDLE_S) - used` can land a hair under IDLE_S for the uptime a runner happens to have, and the reaper
    thread's own look at the child (real clock) could be the one that stops it."""

    now = 1000.0  # 1000.0 + IDLE_S is exact in binary floating point, so the boundary comparison is exact

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)  # the reaper keeps its cadence; the frozen clock is what keeps it from reaping


class FakeConn:
    """The dashboard's end of the warm child's pipe: answers each request with `answers` in turn, or never."""

    def __init__(self, answers=()):
        self.answers = list(answers)
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def poll(self, timeout):
        return bool(self.answers)

    def recv(self):
        return self.answers.pop(0)

    def close(self):
        self.closed = True


class FakeProcess:
    pid = 1

    def __init__(self, **kwargs):
        self.alive = False
        self.terminated = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.alive = False

    def join(self, timeout=None):
        pass


def fake_children(monkeypatch, *answers):
    """Warm children that answer `answers` in turn; returns the processes started."""
    started = []

    class Context:
        def Pipe(self, duplex=False):
            assert duplex, "the warm child is asked and answers over one pipe"
            return FakeConn(answers), FakeConn()

        def Process(self, **kwargs):
            started.append(FakeProcess(**kwargs))
            return started[-1]

    monkeypatch.setattr(segmentation.multiprocessing, "get_context", lambda name: Context())
    return started


@pytest.fixture(autouse=True)
def no_warm_child():
    """Each test starts with no warm child and leaves none running."""
    segmentation._server = None
    yield
    if segmentation._server is not None:
        segmentation._shutdown(segmentation._server)
        segmentation._server = None


def test_a_timeout_ends_the_warm_child_and_the_next_request_starts_another(monkeypatch):
    started = fake_children(monkeypatch)
    with pytest.raises(segmentation.Unavailable, match="too long"):
        segmentation._run(segmentation._infer, (b"image", {"point": [0.5, 0.5]}, "fixture/edgetam", "cpu"), 0.01)
    assert started[0].terminated and segmentation._server is None


def test_the_warm_child_serves_the_next_request_and_stops_only_after_idle_s(monkeypatch):
    started = fake_children(monkeypatch, (True, MASK, 0.9), (True, MASK, 0.8))
    clock = FrozenClock()
    monkeypatch.setattr(segmentation, "time", clock)
    monkeypatch.setattr(segmentation, "REAP_EVERY_S", 3600.0)  # the idle reaper's own tick never reaps in this test
    args = (b"image", {"point": [0.5, 0.5]}, "fixture/edgetam", "cpu")
    assert segmentation._run(segmentation._infer, args, 1)[1] == 0.9
    assert segmentation._run(segmentation._infer, args, 1)[1] == 0.8
    assert len(started) == 1, "one child for both"
    server = segmentation._server
    assert server is not None
    used = server.used
    assert used == clock.now, "the child was used until the second request finished"
    assert not segmentation.reap_idle(used + segmentation.IDLE_S - 1)
    assert segmentation._lock.acquire(blocking=False)
    try:
        assert not segmentation.reap_idle(used + segmentation.IDLE_S), "never while a request holds it"
    finally:
        segmentation._lock.release()
    assert segmentation.reap_idle(used + segmentation.IDLE_S)
    assert started[0].terminated and segmentation._server is None, "its exit frees its GPU memory"


def test_a_failure_the_child_reports_keeps_it_warm(monkeypatch):
    started = fake_children(monkeypatch, (False, segmentation.FAILED))
    with pytest.raises(segmentation.Unavailable, match="could not load or run"):
        segmentation._run(segmentation._infer, (b"image", {"point": [0.5, 0.5]}, "fixture/edgetam", "cpu"), 1)
    assert not started[0].terminated and segmentation._server is not None


def _raise(error):
    raise error


def test_the_child_answers_each_request_and_says_why_one_failed_without_its_details(monkeypatch):
    class Inbox(FakeConn):
        def recv(self):
            return self.answers.pop(0) if self.answers else _raise(EOFError())

    conn = Inbox(
        [
            (lambda a, b: (a + b,), (1, 2)),
            (_raise, (ImportError("no module named torch"),)),
            (_raise, (RuntimeError("/home/secret/token in a path"),)),
        ]
    )
    monkeypatch.setattr(segmentation.os, "nice", lambda n: 0)
    segmentation._serve(cast(Any, conn))
    assert conn.sent == [(True, 3), (False, segmentation.MISSING), (False, segmentation.FAILED)]


def _waiting_infer(image, prompt, model, device, *, marker, ignore_term):
    """A request that never finishes: it says it started (a file: an Event can't go through the pipe) and waits."""
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    open(marker, "w").close()
    threading.Event().wait(30)
    return MASK, 0.9


@pytest.mark.parametrize("ignore_term", [False, True])
@pytest.mark.parametrize("lost_identifier", [False, True])
def test_cancel_ends_only_its_child_and_releases_slot_for_replacement(
    monkeypatch, tmp_path, ignore_term, lost_identifier
):
    context = multiprocessing.get_context("spawn")
    processes = []

    class Context:
        def Pipe(self, duplex=False):
            return context.Pipe(duplex=duplex)

        def Process(self, **kwargs):
            process = context.Process(**kwargs)
            processes.append(process)
            return process

    marker = tmp_path / "started"
    monkeypatch.setattr(segmentation.multiprocessing, "get_context", lambda name: Context())
    monkeypatch.setattr(segmentation, "_infer", partial(_waiting_infer, marker=str(marker), ignore_term=ignore_term))
    job = None if lost_identifier else f"replace-this-view-{ignore_term}"
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(segmentation.segment, image_bytes(), bbox=BOX, job=job)
        try:
            deadline = time.monotonic() + 30
            while not marker.exists():
                assert time.monotonic() < deadline, "synthetic segmentation child did not start"
                time.sleep(0.05)
            assert not segmentation.cancel("another-view")
            assert processes[0].is_alive()
            assert segmentation.cancel_current() if job is None else segmentation.cancel(job)
            with pytest.raises(segmentation.Cancelled, match="cancelled"):
                pending.result(timeout=5)
            assert not processes[0].is_alive()
            assert processes[0].exitcode is not None
            assert segmentation._server is None
            monkeypatch.setattr(segmentation, "_run", lambda *args: (MASK, 0.9))
            assert segmentation.segment(image_bytes(), bbox=BOX, job="replacement-view").score == 0.9
            assert not segmentation.cancel_current() if job is None else not segmentation.cancel(job)
        finally:
            processes[0].terminate()
            processes[0].join(timeout=2)


def test_cancel_before_start_prevents_late_request_without_blocking_other_jobs(monkeypatch):
    calls = []
    monkeypatch.setattr(segmentation, "_run", lambda *args: calls.append(args) or (MASK, 0.9))
    assert not segmentation.cancel("late-view")
    assert not segmentation.cancel("late-view")
    with pytest.raises(segmentation.Cancelled, match="cancelled"):
        segmentation.segment(image_bytes(), bbox=BOX, job="late-view")
    assert calls == []
    assert segmentation.segment(image_bytes(), bbox=BOX, job="independent-view").score == 0.9
    assert len(calls) == 1


@pytest.mark.parametrize("old_key", [None, "previous-page-job"])
@pytest.mark.parametrize("source", ["capture", "import_view"])
def test_new_view_replaces_a_job_even_when_the_browser_lost_its_identifier(monkeypatch, tmp_path, old_key, source):
    from types import SimpleNamespace

    from liberational.dashboard import objects as page

    started = threading.Event()
    cleanup = threading.Event()
    calls = []

    def run(*args):
        calls.append(args)
        if len(calls) == 1:
            current = segmentation._current.job
            started.set()
            while not current.cancelled.wait(0.01) and not cleanup.is_set():
                pass
        return MASK, 0.9

    monkeypatch.setattr(segmentation, "_run", run)
    monkeypatch.setattr(page.objects, "camera_names", lambda: ["fixture-camera"])
    monkeypatch.setattr(page, "store", lambda: SimpleNamespace(state=tmp_path))
    camera = SimpleNamespace(newest=lambda *args, **kwargs: (np.zeros((60, 80, 3), dtype=np.uint8), 1.0, "frame"))
    monkeypatch.setattr(page.vision_log, "at", lambda state: camera)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(segmentation.segment, image_bytes(), bbox=BOX, job=old_key)
        try:
            assert started.wait(2)
            assert not segmentation.cancel("new-page-has-no-old-id")
            frozen = getattr(page, source)(
                {
                    "camera": "fixture-camera",
                    "image": "data:image/jpeg;base64," + base64.b64encode(image_bytes()).decode(),
                    "captured_at": time.time(),
                }
            )
            assert frozen["camera"] == "fixture-camera"
            assert segmentation.segment(image_bytes(), bbox=BOX, job="fresh-view").score == 0.9
            with pytest.raises(segmentation.Cancelled):
                pending.result(timeout=2)
        finally:
            cleanup.set()


def square(top: int, left: int, side: int, size: int = 100) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[top : top + side, left : left + side] = True
    return mask


def test_found_masks_are_best_first_without_tiny_whole_view_or_duplicate_ones():
    masks = np.stack(
        [
            square(10, 10, 20),  # 4% of the view
            square(10, 10, 19),  # 0.9 like the first: the same object
            square(60, 60, 30),  # another object
            square(0, 0, 3),  # 0.09%: noise
            square(0, 0, 100),  # the whole view
        ]
    )
    kept = segmentation.select(masks, np.array([0.8, 0.95, 0.9, 0.99, 0.99]))
    assert [score for _, score in kept] == [0.95, 0.9]
    assert np.array_equal(kept[0][0], masks[1]) and np.array_equal(kept[1][0], masks[2])


def test_at_most_find_max_masks_are_kept(monkeypatch):
    monkeypatch.setattr(segmentation, "FIND_MAX", 2)
    masks = np.stack([square(0, 10 * k, 5) for k in range(5)])
    assert len(segmentation.select(masks, np.linspace(0.9, 0.5, 5))) == 2


def test_find_all_unpacks_each_childs_mask_and_refuses_ones_of_another_size(monkeypatch):
    image = image_bytes()  # 80 x 60
    mask = np.zeros((60, 80), dtype=bool)
    mask[5:20, 10:30] = True
    monkeypatch.setattr(segmentation.cfg, "hf_repo", lambda name: "fixture/edgetam")
    monkeypatch.setattr(segmentation, "_run", lambda *args: ([(np.packbits(mask), mask.shape, 0.7)],))
    [found] = segmentation.find_all(image)
    assert np.array_equal(found.mask, mask) and (found.score, found.model) == (0.7, "fixture/edgetam")
    monkeypatch.setattr(segmentation, "_run", lambda *args: ([(np.packbits(mask), (3, 4), 0.7)],))
    with pytest.raises(segmentation.Unavailable, match="don't fit"):
        segmentation.find_all(image)
    monkeypatch.setattr(segmentation, "_run", lambda *args: ([],))
    assert segmentation.find_all(image) == [], "nothing found is an answer"


@pytest.fixture
def edge_boundary(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = SimpleNamespace(loads=[], prompts=[], forwards=[], embeddings=0, reports=[])

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Model:
        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def get_image_embeddings(self, pixels):
            calls.embeddings += 1
            assert tuple(pixels.shape) == (1, 3, 16, 16)
            return [
                torch.zeros((1, 32, 256, 256)),
                torch.zeros((1, 64, 128, 128)),
                torch.zeros((1, 256, 64, 64)),
            ]

        def __call__(self, **kwargs):
            assert set(kwargs) <= {
                "pixel_values",
                "input_points",
                "input_labels",
                "input_boxes",
                "image_embeddings",
                "multimask_output",
            }, "Processor metadata must not reach the RepViT/timm backbone"
            calls.forwards.append(kwargs)
            if "image_embeddings" not in kwargs:
                return SimpleNamespace(
                    pred_masks=torch.tensor(MASK[None, None, None]),
                    iou_scores=torch.tensor([[[0.9]]]),
                )
            count = kwargs["input_points"].shape[1]
            masks = torch.full((1, count, 3, 8, 8), -2.0)
            masks[:, :, 1, 2:5, 2:5] = 2.0
            return SimpleNamespace(pred_masks=masks, iou_scores=torch.tensor([[[0.1, 0.95, 0.2]]]).repeat(1, count, 1))

    model = Model()

    class Factory:
        @classmethod
        def from_pretrained(cls, model_ref, *, local_files_only):
            calls.loads.append((model_ref, local_files_only))
            return model

    class Processor:
        @classmethod
        def from_pretrained(cls, model_ref, *, local_files_only):
            assert (model_ref, local_files_only) == ("fixture/edgetam", True)
            return cls()

        def __call__(self, *, images, return_tensors, **kwargs):
            assert images.size == (80, 60) and return_tensors == "pt"
            calls.prompts.append(kwargs)
            inputs = Inputs(pixel_values=torch.zeros((1, 3, 16, 16)), original_sizes=torch.tensor([[60, 80]]))
            inputs.update({key: torch.tensor(value) for key, value in kwargs.items()})
            return inputs

        def post_process_masks(self, masks, original_sizes):
            assert original_sizes.tolist() == [[60, 80]]
            return [torch.nn.functional.interpolate(masks[0].float(), size=(60, 80), mode="nearest") > 0]

    def reported(*args, **kwargs):
        calls.reports.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(EdgeTamModel=Factory, Sam2Processor=Processor))
    monkeypatch.setattr(segmentation.model_runtime, "loaded", reported)
    segmentation._load.cache_clear()
    yield calls
    segmentation._load.cache_clear()


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ({"bbox": [0.1, 0.1, 0.5, 0.5]}, {"input_boxes": [[[8.0, 6.0, 48.0, 36.0]]]}),
        ({"point": [0.25, 0.5]}, {"input_points": [[[[20.0, 30.0]]]], "input_labels": [[[1]]]}),
    ],
)
def test_edgetam_prompts_keep_pixel_coordinates_and_return_native_size_masks(edge_boundary, prompt, expected):
    mask, score = segmentation._infer(image_bytes(), prompt, "fixture/edgetam", "cpu")
    assert edge_boundary.loads == [("fixture/edgetam", True)]
    assert edge_boundary.prompts == [expected]
    assert edge_boundary.forwards[0]["multimask_output"] is False
    assert np.array_equal(mask, MASK) and score == pytest.approx(0.9)
    assert edge_boundary.reports[0][0] == ("models.hf.object_segmentation", "fixture/edgetam", "object outlines", "cpu")


def test_edgetam_find_objects_reuses_one_embedding_and_batches_independent_point_prompts(edge_boundary):
    found = segmentation._infer_all(image_bytes(), "fixture/edgetam", "cpu")
    assert edge_boundary.loads == [("fixture/edgetam", True)]
    assert edge_boundary.embeddings == 1
    assert len(edge_boundary.forwards) == 32
    first = edge_boundary.forwards[0]
    last = edge_boundary.forwards[-1]
    assert all(tuple(call["input_points"].shape) == (1, 8, 1, 2) for call in edge_boundary.forwards)
    assert first["input_points"][0, 0, 0].tolist() == [2.5, 1.875]
    assert last["input_points"][0, -1, 0].tolist() == [77.5, 58.125]
    assert all(call["input_labels"].tolist() == [[[1]] * 8] for call in edge_boundary.forwards)
    points = [point for call in edge_boundary.forwards for point in call["input_points"][0, :, 0].tolist()]
    assert points == [[(x + 0.5) * 5, (y + 0.5) * 3.75] for y in range(16) for x in range(16)]
    assert first["image_embeddings"] is last["image_embeddings"]
    assert "pixel_values" not in first and first["multimask_output"] is True
    assert len(found) == 1  # Duplicate grid candidates reduce to one stable, correctly sized object mask.
    mask, score = found[0]
    assert mask.shape == (60, 80) and mask.any() and not mask.all()
    assert score == pytest.approx(0.95)
