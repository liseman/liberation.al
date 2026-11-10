"""ingest/analyze.py: ingested media through the live workers' analysis."""

import asyncio
import io
import json
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import continuum_py
import numpy as np
import pytest

from liberational import continuum_link, segment_log
from liberational.ingest import analyze
from liberational.ingest.store import Store
from liberational.listen.voices import Match, VoiceMatcher, Voiceprints
from liberational.see.caption_worker import CaptionWorker


@pytest.fixture(autouse=True)
def _no_live_voice_gate(monkeypatch):
    """wait_quiet would open the machine's broker socket. These tests never do."""
    monkeypatch.setattr(analyze.voice_turn, "wait_quiet", lambda max_wait_s=None: None)


@dataclass
class FakeItem:
    id: int
    kind: str
    outputs: list[dict[str, Any]]
    stream: str = "ingest-upload"
    name: str = "x"
    status: str = "done"
    original_ts: float | None = 1000.0
    added_ts: float = 2000.0


class FakeSegments:
    """The caption worker's segment writers, as ingest uses them: rows written, a position, sync, and rewind."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, int]] = []
        self.synced = 0

    def position(self, stream: str, cam: str) -> int:
        return len(self.rows)

    def sync(self) -> dict[str, dict[str, int]]:
        self.synced += 1
        return {}

    def rewind(self, to: dict[str, dict[str, int]]) -> None:
        del self.rows[min(at for cams in to.values() for at in cams.values()) :]


@dataclass
class FakeCaptionWorker:
    committed: int = 0
    handled: list[tuple[str, int]] = field(default_factory=list)
    fail_at: int | None = None  # the caption that raises, as a model or write failure would

    def __post_init__(self) -> None:
        import concurrent.futures

        self._model = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.segments = FakeSegments()

    def _handle(self, parent: str, ns: int, out: Any, last: str | None, **_extra: Any) -> str:
        if self.fail_at is not None and len(self.handled) == self.fail_at:
            raise OSError("simulated caption failure")
        self.handled.append((parent, ns))
        self.segments.rows.append((parent, ns))
        self.committed += 1
        return "c"


def no_caption(cam: str) -> SimpleNamespace:
    return SimpleNamespace(feed=lambda *x: None)


def frames(n: int):
    def gen(path: Path, every: float):
        for i in range(n):
            yield np.zeros((4, 4, 3), np.uint8), float(i)

    return gen


def test_video_frames_go_through_captions_timed_at_the_original_and_parented_on_the_chunk():
    cw = FakeCaptionWorker()
    a = analyze.Analyzer(captions=(cw, no_caption), frames=frames(5))
    item = FakeItem(1, "video", [{"kind": "video", "path": "v.mp4", "chunk_id": "chunk-1", "stream": "s.video_v1"}])
    done = asyncio.run(a.item(item))
    assert (done.frames, done.captions) == (5, 3)
    assert cw.handled == [("chunk-1", ns * analyze.NS) for ns in (1000, 1002, 1004)]  # captions every 2 s
    assert len(cw.segments.rows) == 3 and cw.segments.synced == 1


def test_a_video_pass_that_fails_part_way_drops_its_captions_so_the_retry_writes_them_once():
    cw = FakeCaptionWorker(fail_at=2)
    a = analyze.Analyzer(captions=(cw, no_caption), frames=frames(5))
    item = FakeItem(1, "video", [{"kind": "video", "path": "v.mp4", "chunk_id": "chunk-1", "stream": "s.video_v1"}])
    assert asyncio.run(a.item(item)).errors  # the pass failed at its third caption
    assert cw.segments.rows == []  # the two it wrote are rewound
    cw.fail_at = None
    asyncio.run(a.item(item))
    assert [ns // analyze.NS for _, ns in cw.segments.rows] == [1000, 1002, 1004]


def test_a_photo_is_one_frame():
    cw = FakeCaptionWorker()
    a = analyze.Analyzer(captions=(cw, no_caption), frames=frames(30))
    done = asyncio.run(
        a.item(FakeItem(2, "photo", [{"kind": "video", "path": "p.mp4", "chunk_id": "c", "stream": "s"}]))
    )
    assert done.frames == 1 and len(cw.handled) == 1


def test_one_output_failing_keeps_the_others_and_records_why():
    def boom(path: Path) -> Iterator[np.ndarray]:
        raise OSError("no ffmpeg")

    class NoSpeech(analyze.Speech):
        def __init__(self) -> None:  # never reached: audio decoding fails first
            pass

    a = analyze.Analyzer(captions=(FakeCaptionWorker(), no_caption), speech=NoSpeech(), frames=frames(2), audio=boom)
    item = FakeItem(
        3,
        "video",
        [
            {"kind": "audio", "path": "a.flac", "chunk_id": "a", "stream": "s.audio_v1"},
            {"kind": "video", "path": "v.mp4", "chunk_id": "v", "stream": "s.video_v1"},
        ],
    )
    done = asyncio.run(a.item(item))
    assert (done.frames, done.captions) == (2, 1)  # the second frame is 1 s on, inside the 2 s caption gap
    assert done.errors and "no ffmpeg" in done.errors[0]


def test_sweep_analyzes_each_done_item_once(tmp_path):
    items = [
        FakeItem(1, "photo", [{"kind": "video", "path": "p", "chunk_id": "c", "stream": "s"}]),
        FakeItem(2, "photo", [], status="failed"),
    ]
    store = SimpleNamespace(finished=lambda after_id=0, limit=500: [i for i in items if i.id > after_id][:limit])
    ledger = analyze.Ledger(tmp_path / "a.json")
    a = analyze.Analyzer(captions=(FakeCaptionWorker(), no_caption), frames=frames(1))
    assert asyncio.run(analyze.sweep(a, store, ledger)) == 1
    assert asyncio.run(analyze.sweep(a, store, analyze.Ledger(tmp_path / "a.json"))) == 0


def test_sweep_reaches_a_done_item_older_than_the_newest_page(tmp_path):
    filler = FakeItem(1, "photo", [], status="done")
    waiting = FakeItem(2, "photo", [{"kind": "video", "path": "p", "chunk_id": "c", "stream": "s"}])
    newest = FakeItem(3, "photo", [], status="done")

    class Page:
        def recent(self, limit: int) -> list[FakeItem]:
            return [newest][:limit]

        def finished(self, after_id: int = 0, limit: int = 500) -> list[FakeItem]:
            return [i for i in (filler, waiting, newest) if i.id > after_id][:limit]

    ledger = analyze.Ledger(tmp_path / "a.json")
    a = analyze.Analyzer(captions=(FakeCaptionWorker(), no_caption), frames=frames(1))
    assert asyncio.run(analyze.sweep(a, Page(), ledger, limit=1)) == 1


def test_finished_pages_done_items_and_skips_failed(tmp_path):
    store = Store(tmp_path / "ingest.sqlite3")
    first, _new = store.add(sha256="h0", source="upload", name="0", original="o", stream="s")
    store.update(first.id, status="done", outputs=[{"kind": "video"}])
    _skipped, _new = store.add(sha256="h1", source="upload", name="1", original="o", stream="s")
    store.update(_skipped.id, status="failed", outputs=[])
    third, _new = store.add(sha256="h2", source="upload", name="2", original="o", stream="s")
    store.update(third.id, status="done", outputs=[{"kind": "video"}])
    assert [i.id for i in store.finished(limit=1)] == [first.id]
    assert [i.id for i in store.finished(after_id=first.id, limit=10)] == [third.id]


class _Proc:
    def __init__(self, payload: bytes, code: int = 0, err: bytes = b"") -> None:
        self.stdout = io.BytesIO(payload)
        self.stderr = io.BytesIO(err)
        self.returncode: int | None = None
        self.args = ["ffmpeg"]
        self._code = code

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self._code
        return self._code

    def kill(self) -> None:
        self.returncode = -9


def test_audio_blocks_yields_each_tenth_without_joining_the_file(monkeypatch):
    pcm = np.arange(analyze.SR, dtype=np.float32).tobytes()
    monkeypatch.setattr(analyze.subprocess, "Popen", lambda *a, **k: _Proc(pcm))
    got = list(analyze.audio_blocks(Path("clip.mp4")))
    block = int(analyze.SR * analyze.AUDIO_BLOCK_S)
    assert [len(x) for x in got] == [block] * (analyze.SR // block)
    assert np.concatenate(got).tobytes() == pcm


def test_audio_blocks_raises_when_ffmpeg_fails(monkeypatch):
    monkeypatch.setattr(analyze.subprocess, "Popen", lambda *a, **k: _Proc(b"", code=1, err=b"no such file"))
    with pytest.raises(subprocess.CalledProcessError):
        list(analyze.audio_blocks(Path("missing.mp4")))


def test_each_speech_span_waits_for_a_quiet_voice_turn_and_video_frames_do_not(monkeypatch):
    waits: list[int] = []
    monkeypatch.setattr(analyze.voice_turn, "wait_quiet", lambda max_wait_s=None: waits.append(1))

    class Vad:
        def feed(self, audio: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
            if len(audio) >= analyze.SR:
                return []
            return [(0, len(audio), np.asarray(audio, np.float32))]

    class Transcriber:
        async def __call__(self, audio: np.ndarray, device: str) -> None:
            return None

    def blocks(path: Path):
        yield np.zeros(int(analyze.SR * analyze.AUDIO_BLOCK_S), np.float32)
        yield np.zeros(int(analyze.SR * analyze.AUDIO_BLOCK_S), np.float32)

    speech = analyze.Speech(None, Transcriber(), None, lambda: Vad())
    worker = analyze.Analyzer(
        captions=(FakeCaptionWorker(), no_caption),
        speech=speech,
        frames=frames(2),
        audio=blocks,
    )
    item = FakeItem(
        1,
        "video",
        [
            {"kind": "video", "path": "v.mp4", "chunk_id": "v", "stream": "s.video_v1"},
            {"kind": "audio", "path": "a.flac", "chunk_id": "a", "stream": "s.audio_v1"},
        ],
    )
    done = asyncio.run(worker.item(item))
    assert done.frames == 2 and len(waits) == 2  # one wait for each of the two spans the VAD found


def _patch_main(monkeypatch, tmp_path, items: list[FakeItem]) -> None:
    monkeypatch.setattr("liberational.config.path", lambda key: tmp_path)
    monkeypatch.setattr(
        "liberational.ingest.pipeline.default",
        lambda registrar=None: SimpleNamespace(
            store=SimpleNamespace(finished=lambda after_id=0, limit=500: [i for i in items if i.id > after_id][:limit])
        ),
    )


def test_an_idle_run_does_not_connect_or_load_models(monkeypatch, tmp_path):
    _patch_main(monkeypatch, tmp_path, [])
    connected: list[int] = []
    loaded: list[int] = []

    class Client:
        @staticmethod
        async def connect(*args: object, **kwargs: object) -> None:
            connected.append(1)
            raise AssertionError("connected with nothing waiting")

    monkeypatch.setattr("continuum_py.Client", Client)
    monkeypatch.setattr("liberational.see.vlm.make_vlm", lambda *a, **k: loaded.append(1))
    analyze.main(["--once"])
    assert connected == [] and loaded == []


def test_a_waiting_item_connects_before_the_models_load(monkeypatch, tmp_path):
    waiting = FakeItem(1, "photo", [{"kind": "video", "path": "p", "chunk_id": "c", "stream": "s"}])
    _patch_main(monkeypatch, tmp_path, [waiting])
    loaded: list[int] = []

    class Client:
        @staticmethod
        async def connect(*args: object, **kwargs: object) -> None:
            raise RuntimeError("stop before models")

    monkeypatch.setattr("liberational.see.vlm.make_vlm", lambda *a, **k: loaded.append(1))
    monkeypatch.setattr("continuum_py.Client", Client)
    monkeypatch.setattr(continuum_link, "Client", Client)
    with pytest.raises(RuntimeError, match="stop before models"):
        analyze.main(["--once"])
    assert loaded == []


def test_a_broker_gone_when_an_item_waits_is_an_exit_for_restart_not_a_traceback(monkeypatch, tmp_path):
    # Live readers exit EXIT_OUTAGE with one WARN when the broker goes; ingest-analyze raised a traceback instead.
    waiting = FakeItem(1, "photo", [{"kind": "video", "path": "p", "chunk_id": "c", "stream": "s"}])
    _patch_main(monkeypatch, tmp_path, [waiting])

    class Client:
        @staticmethod
        async def connect(*args: object, **kwargs: object) -> None:
            raise continuum_py.ConnectionClosedError("the connection to the broker is closed")

    monkeypatch.setattr("continuum_py.Client", Client)
    monkeypatch.setattr(continuum_link, "Client", Client)
    with pytest.raises(SystemExit) as exited:
        analyze.main(["--once"])
    assert exited.value.code == continuum_link.EXIT_OUTAGE


@pytest.mark.parametrize(
    ("times", "expected"),
    [
        ([0.0, 0.3, 0.6, 0.9, 1.2], [0, 3]),
        ([0.0, 5.0, 10.0], [0, 1, 2]),
        ([1.5, 1.6], [1]),
        ([], []),
    ],
)
def test_choose_keyframes_keeps_the_nearest_one_to_each_second(times: list[float], expected: list[int]) -> None:
    assert analyze.choose_keyframes(times, 1.0) == expected


def test_video_frames_decodes_keyframes_and_skips_the_frames_between_them(monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(list(cmd))
        if "packet=pts_time,flags" in cmd:
            text = "0.000000,K__\n0.033000,___\n1.000000,K__\n"
        else:
            text = "4,4\n"
        return SimpleNamespace(returncode=0, stdout=text, stderr="")

    monkeypatch.setattr(analyze.subprocess, "run", run)
    monkeypatch.setattr(
        analyze.subprocess, "Popen", lambda cmd, **_k: calls.append(list(cmd)) or _Proc(b"\x10" * (4 * 4 * 3 * 2))
    )
    got = list(analyze.video_frames(Path("v.mp4"), 1.0))
    assert [t for _img, t in got] == [0.0, 1.0]
    decode = next(cmd for cmd in calls if cmd[0] == "ffmpeg")
    assert decode[decode.index("-skip_frame") + 1] == "nokey"
    assert decode.index("-skip_frame") < decode.index("-i")


def _turned_clip(directory: Path, degrees: int) -> Path:
    """16x8 clip, red on the left and blue on the right, with a display matrix and no re-encode."""
    coded = directory / "coded.mp4"
    turned = directory / "turned.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=8x8:d=0.1,format=yuv420p",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=8x8:d=0.1,format=yuv420p",
            "-filter_complex",
            "hstack",
            "-c:v",
            "libx264",
            "-g",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(coded),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-display_rotation",
            str(degrees),
            "-i",
            str(coded),
            "-c",
            "copy",
            str(turned),
        ],
        check=True,
    )
    return turned


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
@pytest.mark.parametrize(
    ("degrees", "shape", "top", "bottom"),
    [
        (90, (16, 8, 3), [255, 0, 0], [0, 0, 255]),
        (270, (16, 8, 3), [0, 0, 255], [255, 0, 0]),
        (180, (8, 16, 3), [255, 0, 0], [255, 0, 0]),
    ],
)
def test_video_frames_keeps_a_display_matrix_upright(
    tmp_path: Path, degrees: int, shape: tuple[int, int, int], top: list[int], bottom: list[int]
) -> None:
    frames = list(analyze.video_frames(_turned_clip(tmp_path, degrees), 1.0))
    assert len(frames) == 1
    img, _offset = frames[0]
    assert img.shape == shape
    assert img[0, 0].tolist() == top
    assert img[-1, 0].tolist() == bottom


def test_a_failing_item_retries_on_a_delay_then_stays_failed(tmp_path: Path) -> None:
    store = Store(tmp_path / "ingest.sqlite3")
    item, _new = store.add(sha256="h", source="upload", name="clip", original="o", stream="s")
    store.update(
        item.id,
        status="done",
        original_ts=1_000.0,
        outputs=[{"kind": "video", "path": "v", "chunk_id": "c", "stream": "s.video_v1"}],
    )
    calls = {"n": 0}
    clock = {"t": 1_000.0}

    class Boom:
        async def item(self, item: Any, *, removed: Any = None) -> analyze.Done:
            calls["n"] += 1
            return analyze.Done(errors=["video: RuntimeError: decode"])

    ledger = analyze.Ledger(tmp_path / "a.json")

    def run() -> int:
        return asyncio.run(analyze.sweep(Boom(), store, ledger, now=lambda: clock["t"]))

    assert run() == 1
    row = ledger.row(item.id)
    assert row is not None
    assert row["state"] == "retry"
    assert row["attempts"] == 1
    assert row["next_at"] == 1_030.0
    clock["t"] = 1_020.0
    assert run() == 0 and calls["n"] == 1
    clock["t"] = 1_030.0
    assert run() == 1
    row = ledger.row(item.id)
    assert row is not None
    assert row["attempts"] == 2
    assert row["next_at"] == 1_150.0
    clock["t"] = 1_150.0
    assert run() == 1 and calls["n"] == 3
    failed = store.get(item.id)
    assert failed is not None and failed.status == "failed" and "decode" in (failed.detail or "")
    row = ledger.row(item.id)
    assert row is not None
    assert row["state"] == "failed"
    clock["t"] = 9_000.0
    assert run() == 0 and calls["n"] == 3
    store.update(item.id, status="done", detail="")
    assert run() == 1 and calls["n"] == 4
    row = ledger.row(item.id)
    assert row is not None
    assert row["state"] == "retry" and row["attempts"] == 1


def test_a_slow_failure_waits_from_when_the_pass_returns(tmp_path: Path) -> None:
    store = Store(tmp_path / "ingest.sqlite3")
    item, _new = store.add(sha256="h", source="upload", name="clip", original="o", stream="s")
    store.update(
        item.id,
        status="done",
        original_ts=1_000.0,
        outputs=[{"kind": "video", "path": "v", "chunk_id": "c", "stream": "s.video_v1"}],
    )
    clock = {"t": 1_000.0}

    class Boom:
        async def item(self, item: Any, *, removed: Any = None) -> analyze.Done:
            clock["t"] += 45.0
            return analyze.Done(errors=["video: RuntimeError: decode"])

    ledger = analyze.Ledger(tmp_path / "a.json")

    def run() -> int:
        return asyncio.run(analyze.sweep(Boom(), store, ledger, now=lambda: clock["t"]))

    assert run() == 1
    row = ledger.row(item.id)
    assert row is not None
    assert row["next_at"] == 1_075.0
    clock["t"] = 1_075.0
    assert run() == 1
    row = ledger.row(item.id)
    assert row is not None
    assert row["attempts"] == 2
    assert row["next_at"] == 1_240.0


def test_a_ledger_row_from_before_retries_is_not_analyzed_again(tmp_path: Path) -> None:
    item = FakeItem(1, "photo", [{"kind": "video", "path": "p", "chunk_id": "c", "stream": "s"}])
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"1": {"frames": 1, "errors": ["video: OSError: x"]}}))
    calls: list[int] = []

    class Boom:
        async def item(self, item: Any, *, removed: Any = None) -> analyze.Done:
            calls.append(item.id)
            return analyze.Done()

    store = SimpleNamespace(finished=lambda after_id=0, limit=500: [item] if item.id > after_id else [])
    assert asyncio.run(analyze.sweep(Boom(), store, analyze.Ledger(path))) == 0
    assert calls == []


def test_record_does_not_restore_a_row_another_process_removed(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    ledger = analyze.Ledger(path)
    ledger.record(1, {"frames": 1}, state="done", attempts=1)
    ledger.record(2, {"frames": 1}, state="done", attempts=1)
    analyze.Ledger(path).remove(1)
    ledger.record(3, {"frames": 4}, state="done", attempts=1)
    assert ledger.row(1) is None
    assert ledger.row(2) is not None
    third = ledger.row(3)
    assert third is not None
    assert third["frames"] == 4
    ledger.remove(3)
    ledger.remove(3)
    assert ledger.row(3) is None


def test_a_removed_item_is_not_recorded_even_if_the_pass_finished(tmp_path: Path) -> None:
    item = FakeItem(7, "photo", [{"kind": "video", "path": "p", "chunk_id": "c", "stream": "s"}])

    class Store:
        def __init__(self) -> None:
            self.gone = False

        def finished(self, after_id: int = 0, limit: int = 500) -> list[FakeItem]:
            return [item] if item.id > after_id else []

        def removed(self, item_id: int) -> bool:
            return self.gone

        def item_lock(self, item_id: int, *, blocking: bool = True) -> nullcontext[None]:
            return nullcontext()

    store = Store()

    class Pass:
        async def item(self, item: Any, *, removed: Any = None) -> analyze.Done:
            store.gone = True
            assert callable(removed) and removed() is True
            return analyze.Done(frames=3)

    ledger = analyze.Ledger(tmp_path / "a.json")
    assert asyncio.run(analyze.sweep(Pass(), store, ledger)) == 0
    assert ledger.row(item.id) is None


def test_the_item_lock_is_held_across_the_pass_and_a_busy_lock_is_polled(tmp_path: Path, monkeypatch) -> None:
    item = FakeItem(4, "photo", [{"kind": "video", "path": "p", "chunk_id": "c", "stream": "s"}])
    slept: list[float] = []

    async def no_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(analyze.asyncio, "sleep", no_sleep)

    class Gate:
        def __init__(self) -> None:
            self.tries = 0
            self.held = False

        def __enter__(self) -> None:
            self.tries += 1
            if self.tries < 3:
                raise BlockingIOError
            self.held = True

        def __exit__(self, *_exc: object) -> bool:
            self.held = False
            return False

    gate = Gate()
    seen: list[bool] = []

    class Store:
        def finished(self, after_id: int = 0, limit: int = 500) -> list[FakeItem]:
            return [item] if item.id > after_id else []

        def item_lock(self, item_id: int, *, blocking: bool = True) -> Gate:
            assert blocking is False
            return gate

    class Pass:
        async def item(self, item: Any, *, removed: Any = None) -> analyze.Done:
            seen.append(gate.held)
            return analyze.Done(frames=1)

    class Watching(analyze.Ledger):
        def record(self, *args: Any, **kwargs: Any) -> None:
            seen.append(gate.held)
            super().record(*args, **kwargs)

    ledger = Watching(tmp_path / "a.json")
    assert asyncio.run(analyze.sweep(Pass(), Store(), ledger)) == 1
    assert seen == [True, True]
    assert gate.tries == 3 and len(slept) == 2 and gate.held is False
    recorded = ledger.row(item.id)
    assert recorded is not None
    assert recorded["state"] == "done"


@pytest.mark.parametrize("initial_mark", [None, "p1"])
def test_a_speech_purge_clears_recent_embeddings_and_a_repeat_of_the_same_mark_does_not(
    initial_mark: str | None,
) -> None:
    class Matcher:
        def __init__(self) -> None:
            self.recent = ["stale"]
            self.clears = 0

        async def match(self, *_args: object, **_kwargs: object) -> None:
            return None

        def learn(self, *_args: object, **_kwargs: object) -> None:
            return None

        def forget(self) -> None:
            self.clears += 1
            self.recent.clear()

    class Vad:
        def feed(self, audio: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
            if len(audio) >= analyze.SR:
                return []
            return [(0, 160, np.zeros(160, np.float32))]

    class Heard:
        text = "hello"
        lat_ms = {"asr": 1}  # noqa: RUF012
        embedding = None

    class Transcriber:
        async def __call__(self, _audio: np.ndarray, _device: str) -> Heard:
            return Heard()

    class Client:
        def ingest(self, _header: bytes, inline: bytes = b"") -> None:
            return None

    marks = {"id": initial_mark}
    matcher = Matcher()
    speech = analyze.Speech(Client(), Transcriber(), matcher, lambda: Vad(), newest_purge=lambda _after: marks["id"])

    def run() -> None:
        asyncio.run(
            speech.file(
                iter([np.zeros(160, np.float32)]),
                "mic",
                "s.audio_v1",
                "019a0000-0000-7000-8000-000000000001",
                1_000.0,
                slot="ingest-1",
            )
        )

    run()
    assert matcher.clears == 0 and matcher.recent == ["stale"]
    run()
    assert matcher.clears == 0
    marks["id"] = "p2"
    run()
    assert matcher.clears == 1 and matcher.recent == []


def test_a_second_pass_files_the_same_voice_embedding_ids_and_writes_nothing_to_the_broker(
    monkeypatch, tmp_path
) -> None:
    def boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("uuid7")

    monkeypatch.setattr(uuid, "uuid7", boom)

    class Client:
        def __init__(self) -> None:
            self.payloads: list[tuple[bytes, bytes]] = []

        def ingest(self, header: bytes, inline: bytes = b"") -> None:
            self.payloads.append((bytes(header), bytes(inline)))

    class Cap:
        def feed(self, _img: np.ndarray, ts: float, seq: int) -> Any:
            return SimpleNamespace(
                caption={
                    "ts_start": ts,
                    "cam": "ingest-upload",
                    "frame_seq": seq,
                    "caption": "a room",
                    "observation": {"status": "fresh"},
                },
                objects={
                    "ts_start": ts,
                    "cam": "ingest-upload",
                    "frame_seq": seq,
                    "objects": [],
                    "source": "test",
                },
            )

    class Vad:
        def feed(self, audio: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
            if len(audio) >= analyze.SR:
                return []
            clip = np.zeros(analyze.SR, np.float32)
            return [(0, analyze.SR, clip)]

    class Heard:
        text = "hello there"
        lat_ms = {"asr": 4}  # noqa: RUF012
        embedding = np.zeros(segment_log.VOICE_DIM, np.float32)

    class Transcriber:
        async def __call__(self, _audio: np.ndarray, _device: str) -> Heard:
            return Heard()

    class Matcher(VoiceMatcher):
        async def match(
            self, embedding: np.ndarray, start_ns: int, end_ns: int, *, words: int | None = None
        ) -> Match | None:
            return Match("Speaker 1", 1, 0.8)

    store = tmp_path / "segments"
    people = SimpleNamespace(person_of=lambda _n: None, voice_of=lambda _p: None)
    prints = Voiceprints(tmp_path / "voiceprints.json", lambda: 0)
    matcher = Matcher(prints, people, SimpleNamespace(), segments=store, slot="ingest")  # ty: ignore[invalid-argument-type]

    client = Client()
    cw = CaptionWorker(client, [], lambda _cam: None, model="fake", segments=segment_log.Writers(tmp_path))  # ty: ignore[invalid-argument-type]  # stand-in: only _handle runs
    speech = analyze.Speech(client, Transcriber(), matcher, lambda: Vad(), segments=store)
    worker = analyze.Analyzer(
        captions=(cw, lambda _cam: Cap()),
        speech=speech,
        frames=frames(1),
        audio=lambda _path: iter([np.zeros(160, np.float32)]),
    )
    item = FakeItem(
        1,
        "video",
        [
            {
                "kind": "video",
                "path": "v.mp4",
                "chunk_id": "019a0000-0000-7000-8000-000000000001",
                "stream": "s.video_v1",
            },
            {
                "kind": "audio",
                "path": "a.flac",
                "chunk_id": "019a0000-0000-7000-8000-000000000001",
                "stream": "s.audio_v1",
            },
        ],
    )
    asyncio.run(worker.item(item))
    assert client.payloads == [], "captions and voice embeddings go to the segment store, nothing to the broker"
    filed = segment_log.Log(store).range(segment_log.VOICE_EMBEDDINGS, 0.0, 2.0**32)
    asyncio.run(worker.item(item))
    again = segment_log.Log(store).range(segment_log.VOICE_EMBEDDINGS, 0.0, 2.0**32)
    assert len(filed) == 1 and {r["id"] for r in again} == {filed[0]["id"]}, "a second pass files the same id"
    cw._model.shutdown(wait=False, cancel_futures=True)


def test_real_store_delete_fences_an_active_model_result_and_removes_ledger_without_resurrection(
    tmp_path: Path,
) -> None:
    import threading

    from liberational.ingest.pipeline import Pipeline
    from test_ingest import DeletePurger, FakeRegistrar

    root = tmp_path / "ingest"
    root.mkdir()
    original, converted = root / "original.mov", root / "converted.mp4"
    original.write_bytes(b"original")
    converted.write_bytes(b"converted")
    store = Store(tmp_path / "state" / "ingest.sqlite3")
    item, _ = store.add(
        sha256="imported", source="upload", name="original.mov", original=str(original), stream="ingest-upload"
    )
    item = store.update(
        item.id,
        status="done",
        kind="video",
        original_ts=1000.0,
        outputs=[
            {
                "kind": "video",
                "path": str(converted),
                "chunk_id": "chunk-1",
                "stream": "ingest-upload.video_v1",
            }
        ],
    )
    ledger = analyze.Ledger(tmp_path / "state" / "ingest-analyzed.json")
    ledger.record(item.id, {"frames": 1}, state="retry", next_at=0)
    entered, release = threading.Event(), threading.Event()
    worker = FakeCaptionWorker()

    class InFlight:
        def feed(self, img: np.ndarray, ts: float, seq: int) -> None:
            entered.set()
            assert release.wait(5)

    analyzer = analyze.Analyzer(captions=(worker, lambda _cam: InFlight()), frames=frames(1))
    pipe, purger = Pipeline(store, root, FakeRegistrar()), DeletePurger()

    async def run() -> None:
        task = asyncio.create_task(analyze.sweep(analyzer, store, ledger))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert pipe.delete(item.id, ledger, purger, wait_s=0)["pending"] is True
            assert store.removed(item.id)
            assert purger.items == []  # cancellation must finish before catalog or storage cleanup
            assert ledger.row(item.id) is not None
        finally:
            release.set()
        assert await asyncio.wait_for(task, 5) == 0
        assert worker.handled == []  # the model result returned after the deletion fence, so it cannot commit
        assert pipe.delete(item.id, ledger, purger, wait_s=0)["pending"] is False
        assert ledger.row(item.id) is None
        assert store.get(item.id) is None
        assert not original.exists() and not converted.exists()
        assert pipe.delete(item.id, ledger, purger, wait_s=0)["pending"] is False
        assert purger.items == [item.id]

    try:
        asyncio.run(run())
    finally:
        release.set()
        worker._model.shutdown(wait=True, cancel_futures=True)


def test_purge_during_a_long_file_stops_using_deleted_voice_before_the_next_match() -> None:
    marks = {"id": "before"}
    recent = ["deleted voice"]
    matches = []

    class Matcher:
        async def match(self, *_args: Any, **_kwargs: Any) -> None:
            matches.append(list(recent))

        def learn(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def forget(self) -> None:
            recent.clear()

    class Vad:
        def feed(self, audio: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
            return [] if len(audio) >= analyze.SR else [(0, 160, audio)]

    calls = 0

    async def transcribe(_audio: np.ndarray, _device: str) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            marks["id"] = "after"
        return SimpleNamespace(text="hello", lat_ms={"asr": 1}, embedding=np.zeros(segment_log.VOICE_DIM, np.float32))

    speech = analyze.Speech(
        SimpleNamespace(ingest=lambda _header, **_kwargs: None),
        transcribe,
        Matcher(),
        Vad,
        newest_purge=lambda _after: marks["id"],
    )
    assert (
        asyncio.run(
            speech.file(
                iter([np.zeros(160, np.float32), np.zeros(160, np.float32)]),
                "mic",
                "s.audio_v1",
                "019a0000-0000-7000-8000-000000000001",
                1000.0,
                slot="ingest-1",
            )
        )
        == 2
    )
    assert matches == [["deleted voice"], []]


@pytest.fixture
def purge_preflight(tmp_path):
    """Real Analyzer/Speech, with only the broker, decoder, VAD, matcher and item-lock boundaries replaced."""

    def make(pages=0, *, fails=False, remove_during_scan=False, new_purge_at=None, fail_held_from=None):
        state = SimpleNamespace(
            held=False,
            gone=False,
            lock_entries=0,
            decoded=[],
            matches=[],
            learned=[],
            purge_calls=[],
            clears=0,
            fails=fails,
            held_calls=0,
        )
        purges = [f"purge-{i}" for i in range(pages)]
        item = FakeItem(
            21,
            "audio",
            [
                {
                    "kind": "audio",
                    "path": "clip.flac",
                    "chunk_id": "019a0000-0000-7000-8000-000000000001",
                    "stream": "s.audio_v1",
                }
            ],
        )

        class Gate:
            def __enter__(self):
                state.held = True
                state.lock_entries += 1
                if new_purge_at == "file":
                    purges.append("arrived-after-preflight")

            def __exit__(self, *_exc):
                state.held = False

        class ItemStore:
            def finished(self, after_id=0, limit=500):
                return [item] if after_id < item.id else []

            def removed(self, item_id):
                return state.gone

            def item_lock(self, item_id, *, blocking=True):
                return Gate()

        class Client:
            async def list_purges(self, *, after=None, limit=256):
                state.purge_calls.append((state.held, after))
                if state.held:
                    state.held_calls += 1
                if state.fails or (state.held and fail_held_from is not None and state.held_calls >= fail_held_from):
                    raise OSError("purge metadata unavailable")
                if remove_during_scan:
                    state.gone = True
                index = purges.index(after) + 1 if after is not None else 0
                return [{"purge_id": value} for value in purges[index : index + 1]]

            def ingest(self, header, inline=b""):
                pass

        class Matcher:
            def __init__(self):
                self.recent = ["deleted voice"]

            async def match(self, *_args, **_kwargs):
                state.matches.append(list(self.recent))

            def learn(self, *_args, **_kwargs):
                state.learned.append(list(self.recent))

            def forget(self):
                state.clears += 1
                self.recent.clear()

        class Vad:
            def feed(self, audio):
                return [] if len(audio) >= analyze.SR else [(0, len(audio), audio)]

        async def transcribe(audio, device):
            if new_purge_at == "embedding":
                purges.append("arrived-before-embedding")
            return SimpleNamespace(
                text="hello", lat_ms={"asr": 1}, embedding=np.zeros(segment_log.VOICE_DIM, np.float32)
            )

        def audio(path):
            state.decoded.append(state.held)
            yield np.zeros(160, np.float32)

        speech = analyze.Speech(Client(), transcribe, Matcher(), Vad)
        worker = analyze.Analyzer(speech=speech, audio=audio)
        ledger = analyze.Ledger(tmp_path / "analysis.json")
        return SimpleNamespace(worker=worker, store=ItemStore(), ledger=ledger, state=state, item=item)

    return make


@pytest.mark.parametrize("history_pages", [0, 8])
def test_speech_purge_history_is_preflighted_outside_the_item_lock(purge_preflight, history_pages):
    run = purge_preflight(history_pages)
    assert asyncio.run(analyze.sweep(run.worker, run.store, run.ledger)) == 1
    outside = [after for held, after in run.state.purge_calls if not held]
    inside = [after for held, after in run.state.purge_calls if held]
    assert len(outside) == history_pages + 1
    assert len(inside) == 2  # File start and its one embedding each ask only after the preflight cursor.
    assert run.state.decoded == [True]  # Actual analysis remains fenced through its ledger write.
    row = run.ledger.row(run.item.id)
    assert row is not None
    assert row["state"] == "done"


def test_failed_speech_purge_preflight_never_takes_lock_decodes_or_records(purge_preflight):
    run = purge_preflight(8, fails=True)
    assert asyncio.run(analyze.sweep(run.worker, run.store, run.ledger)) == 0
    assert run.state.purge_calls == [(False, None)]
    assert run.state.lock_entries == 0
    assert run.state.decoded == []
    assert run.ledger.row(run.item.id) is None


def test_item_removed_during_speech_preflight_is_not_decoded_or_recorded(purge_preflight):
    run = purge_preflight(8, remove_during_scan=True)
    assert asyncio.run(analyze.sweep(run.worker, run.store, run.ledger)) == 0
    assert run.state.decoded == []
    assert run.state.matches == []
    assert run.ledger.row(run.item.id) is None


@pytest.mark.parametrize("arrival", ["file", "embedding"])
def test_speech_purge_arriving_after_preflight_clears_recent_before_matching(purge_preflight, arrival):
    run = purge_preflight(8, new_purge_at=arrival)
    assert asyncio.run(analyze.sweep(run.worker, run.store, run.ledger)) == 1
    assert run.state.matches == [[]]
    assert run.state.clears == 1


@pytest.mark.parametrize("point,held_call", [("file", 1), ("embedding", 2)])
def test_failed_incremental_speech_purge_check_cannot_match_or_learn_stale_voice(purge_preflight, point, held_call):
    run = purge_preflight(8, new_purge_at=point, fail_held_from=held_call)
    assert asyncio.run(analyze.sweep(run.worker, run.store, run.ledger)) == 1
    assert run.state.matches == []
    assert run.state.learned == []
    row = run.ledger.row(run.item.id)
    assert row is not None
    assert row["state"] == "retry"
    assert any("speech purge mark unavailable" in error for error in row["errors"])


def test_failed_speech_preflight_leaves_item_pending_until_metadata_recovers(purge_preflight):
    run = purge_preflight(8, fails=True)
    assert asyncio.run(analyze.sweep(run.worker, run.store, run.ledger)) == 0
    assert run.ledger.row(run.item.id) is None
    assert run.state.lock_entries == 0
    run.state.fails = False
    assert asyncio.run(analyze.sweep(run.worker, run.store, run.ledger)) == 1
    row = run.ledger.row(run.item.id)
    assert row is not None and row["state"] == "done"
    assert run.state.decoded == [True]


def _race_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: analyze.Done) -> tuple[int, dict[str, Any]]:
    """Two sweeps that both see one due item. The second blocks on the item lock until the first records."""
    item = FakeItem(4, "photo", [{"kind": "video", "path": "p", "chunk_id": "c", "stream": "s"}])
    ready, release = asyncio.Event(), asyncio.Event()
    calls: list[int] = []
    held = False
    blocked = 0

    class Gate:
        def __enter__(self) -> None:
            nonlocal held, blocked
            if held:
                blocked += 1
                raise BlockingIOError
            held = True

        def __exit__(self, *_exc: object) -> bool:
            nonlocal held
            held = False
            return False

    class Store:
        def finished(self, after_id: int = 0, limit: int = 500) -> list[FakeItem]:
            return [item] if item.id > after_id else []

        def removed(self, item_id: int) -> bool:
            return False

        def item_lock(self, item_id: int, *, blocking: bool = True) -> Gate:
            assert blocking is False
            return Gate()

        def update(self, item_id: int, **_fields: Any) -> None:
            return None

    class Pass:
        async def item(self, item: Any, *, removed: Any = None) -> analyze.Done:
            calls.append(item.id)
            if len(calls) == 1:
                ready.set()
                await release.wait()
            return first

    real_sleep = asyncio.sleep

    async def fast_sleep(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(analyze.asyncio, "sleep", fast_sleep)

    async def race() -> tuple[int, dict[str, Any]]:
        nonlocal blocked
        ledger = analyze.Ledger(tmp_path / "a.json")
        holder = asyncio.create_task(analyze.sweep(Pass(), Store(), ledger, now=lambda: 1_000.0))
        await ready.wait()
        waiter = asyncio.create_task(analyze.sweep(Pass(), Store(), ledger, now=lambda: 1_000.0))
        for _ in range(100):
            if blocked:
                break
            await real_sleep(0)
        assert blocked >= 1
        release.set()
        await holder
        await waiter
        row = ledger.row(item.id)
        assert row is not None
        return len(calls), row

    return asyncio.run(race())


def test_a_waiter_keeps_the_retry_delay_of_the_sweep_that_holds_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, row = _race_sweep(tmp_path, monkeypatch, analyze.Done(errors=["video: RuntimeError: decode"]))
    assert calls == 1
    assert row["state"] == "retry"
    assert row["attempts"] == 1
    assert row["next_at"] == 1_030.0


def test_a_waiter_does_not_analyze_an_item_the_holder_just_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, row = _race_sweep(tmp_path, monkeypatch, analyze.Done(frames=1))
    assert calls == 1
    assert row["state"] == "done"
    assert row["attempts"] == 1


def test_a_waiter_does_not_reopen_an_item_the_holder_just_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale page snapshot still says done after the holder records the third failure.

    The waiter's own snapshot must not count as a user Retry. Reusing one FakeItem would hide
    this, because the holder's update would change the status the waiter already holds.
    """
    output = {"kind": "video", "path": "p", "chunk_id": "c", "stream": "s"}
    ready, release = asyncio.Event(), asyncio.Event()
    calls: list[int] = []
    held = False
    blocked = 0

    class Gate:
        def __enter__(self) -> None:
            nonlocal held, blocked
            if held:
                blocked += 1
                raise BlockingIOError
            held = True

        def __exit__(self, *_exc: object) -> bool:
            nonlocal held
            held = False
            return False

    class Store:
        def __init__(self) -> None:
            self.status = "done"

        def finished(self, after_id: int = 0, limit: int = 500) -> list[FakeItem]:
            if self.status != "done" or after_id >= 8:
                return []
            return [FakeItem(8, "photo", [dict(output)], status="done")]

        def get(self, item_id: int) -> FakeItem:
            return FakeItem(8, "photo", [dict(output)], status=self.status)

        def removed(self, item_id: int) -> bool:
            return False

        def item_lock(self, item_id: int, *, blocking: bool = True) -> Gate:
            assert blocking is False
            return Gate()

        def update(self, item_id: int, **fields: Any) -> None:
            if "status" in fields:
                self.status = str(fields["status"])

    class Pass:
        async def item(self, item: Any, *, removed: Any = None) -> analyze.Done:
            calls.append(item.id)
            if len(calls) == 1:
                ready.set()
                await release.wait()
            return analyze.Done(errors=["video: RuntimeError: decode"])

    real_sleep = asyncio.sleep

    async def fast_sleep(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(analyze.asyncio, "sleep", fast_sleep)
    store = Store()

    async def race() -> None:
        nonlocal blocked
        ledger = analyze.Ledger(tmp_path / "a.json")
        ledger.record(8, {"errors": ["video: earlier"]}, state="retry", attempts=2, next_at=0.0)
        holder = asyncio.create_task(analyze.sweep(Pass(), store, ledger, now=lambda: 1_000.0))
        await ready.wait()
        waiter = asyncio.create_task(analyze.sweep(Pass(), store, ledger, now=lambda: 1_000.0))
        for _ in range(100):
            if blocked:
                break
            await real_sleep(0)
        assert blocked >= 1
        release.set()
        await holder
        await waiter
        row = ledger.row(8)
        assert row is not None
        assert calls == [8]
        assert row["state"] == "failed"
        assert row["attempts"] == 3
        assert store.status == "failed"

    asyncio.run(race())


def test_a_user_retry_still_runs_when_the_ledger_row_is_failed(tmp_path: Path) -> None:
    output = {"kind": "video", "path": "p", "chunk_id": "c", "stream": "s"}
    item = FakeItem(9, "photo", [output])
    calls: list[int] = []

    class Store:
        def finished(self, after_id: int = 0, limit: int = 500) -> list[FakeItem]:
            return [item] if item.id > after_id else []

        def get(self, item_id: int) -> FakeItem:
            return FakeItem(item.id, item.kind, list(item.outputs), status=item.status)

        def removed(self, item_id: int) -> bool:
            return False

    class Pass:
        async def item(self, item: Any, *, removed: Any = None) -> analyze.Done:
            calls.append(item.id)
            return analyze.Done(frames=1)

    ledger = analyze.Ledger(tmp_path / "b.json")
    ledger.record(item.id, {"errors": ["video: earlier"]}, state="failed", attempts=3)
    n = asyncio.run(analyze.sweep(Pass(), Store(), ledger, now=lambda: 1_000.0))
    row = ledger.row(item.id)
    assert n == 1
    assert calls == [item.id]
    assert row is not None
    assert row["state"] == "done"
    assert row["attempts"] == 1


@pytest.fixture
def analyzed_imports(tmp_path):
    from typing import cast

    from continuum_py import Client

    from liberational import config
    from liberational.ingest.pipeline import Pipeline
    from liberational.see.caption_animals import Found
    from liberational.see.captions import CaptionFrame, CaptionPipeline
    from test_ingest import DeletePurger, FakeRegistrar

    streams = (
        segment_log.CAPTIONS,
        segment_log.CAMERA_OBJECTS,
        segment_log.ANIMAL_GROUNDINGS,
        segment_log.ANIMAL_GROUNDING_EMBEDDINGS,
    )
    with config.override({"paths": {"state": str(tmp_path / "state")}}):
        base = segment_log.root()
        root = tmp_path / "ingest"
        root.mkdir()
        store = Store(tmp_path / "state" / "ingest.sqlite3")
        ledger = analyze.Ledger(tmp_path / "state" / "ingest-analyzed.json")
        pipe = Pipeline(store, root, FakeRegistrar())
        state = SimpleNamespace(fail=False, produced=[])
        saved: dict[int, list[tuple[float, CaptionFrame]]] = {}

        class CaptionModel:
            def __init__(self, camera: str) -> None:
                self.camera = camera
                self.track = str(uuid.uuid4())

            def feed(self, _img: np.ndarray, ts: float, seq: int) -> CaptionFrame:
                if state.fail and seq == 2:
                    raise OSError("synthetic model failure after the first frame")
                result = CaptionFrame(
                    caption={
                        "ts_start": ts,
                        "cam": self.camera,
                        "frame_seq": seq,
                        "caption": "A cat beside a box",
                        "observation": {"status": "fresh"},
                    },
                    objects={
                        "ts_start": ts,
                        "cam": self.camera,
                        "frame_seq": seq,
                        "objects": [],
                        "source": "test",
                    },
                    animals=[
                        Found("cat", [0.1, 0.1, 0.4, 0.4], self.track, np.zeros(segment_log.EMBEDDING_DIM, np.float32))
                    ],
                )
                state.produced.append((ts, result))
                return result

        worker = CaptionWorker(
            cast(Client, SimpleNamespace()),
            [],
            lambda _cam: cast(CaptionPipeline, None),
            model="test",
            segments=segment_log.Writers(base),
        )
        analyzer = analyze.Analyzer(captions=(worker, CaptionModel), frames=frames(3))

        def add(label):
            original, converted = root / f"{label}.mov", root / f"{label}.mp4"
            original.write_bytes(label.encode())
            converted.write_bytes(b"converted")
            item, _ = store.add(
                sha256=label, source="upload", name=original.name, original=str(original), stream="ingest-upload"
            )
            return store.update(
                item.id,
                status="done",
                kind="video",
                original_ts=1000.0,
                outputs=[
                    {
                        "kind": "video",
                        "path": str(converted),
                        "chunk_id": str(uuid.uuid4()),
                        "stream": "ingest-upload.video_v1",
                    }
                ],
            )

        def run(item):
            state.produced = []
            result = asyncio.run(analyzer.item(item))
            saved[item.id] = state.produced
            if not result.errors:
                ledger.record(item.id, result.public())
            return result

        def replay(item):
            for ts, output in saved[item.id]:
                worker._handle(item.outputs[0]["chunk_id"], int(ts * analyze.NS), output, None)
            worker.segments.sync()

        def snapshot():
            reader = segment_log.Log(base)
            return {
                stream: sorted(
                    (r["cam"], str(r.get("frame_id", "")), str(r.get("track_id", "")), str(r.get("id", "")))
                    for r in reader.range(stream, 999.0, 1010.0)
                )
                for stream in streams
            }

        try:
            yield SimpleNamespace(
                add=add,
                run=run,
                replay=replay,
                snapshot=snapshot,
                pipe=pipe,
                store=store,
                ledger=ledger,
                purger=DeletePurger(),
                state=state,
                streams=streams,
            )
        finally:
            worker.segments.close()
            worker._model.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("pending", [False, True], ids=["complete", "resume-pending"])
def test_completed_import_delete_removes_all_vision_outputs_without_erasing_overlapping_imports(
    analyzed_imports, pending
):
    from liberational.ingest.pipeline import Pipeline

    env = analyzed_imports
    other, deleted = env.add("other"), env.add("deleted")
    assert not env.run(other).errors
    keep = env.snapshot()
    assert {stream: len(rows) for stream, rows in keep.items()} == dict.fromkeys(env.streams, 2)
    assert not env.run(deleted).errors
    assert {stream: len(rows) for stream, rows in env.snapshot().items()} == dict.fromkeys(env.streams, 4)

    env.purger.finished = not pending
    assert env.pipe.delete(deleted.id, env.ledger, env.purger, wait_s=0)["pending"] is pending
    env.purger.finished = True
    restarted = Pipeline(Store(env.store.path), env.pipe.root, env.pipe.registrar)
    assert restarted.delete(deleted.id, env.ledger, env.purger, wait_s=0)["pending"] is False
    assert env.snapshot() == keep
    assert env.store.get(deleted.id) is None
    assert env.ledger.row(deleted.id) is None
    assert not Path(deleted.original).exists()
    assert not Path(deleted.outputs[0]["path"]).exists()
    assert env.store.get(other.id) is not None
    assert env.ledger.row(other.id) is not None
    assert env.purger.items == [deleted.id]

    later = env.add("later")
    assert not env.run(later).errors
    after_later = env.snapshot()
    assert {stream: len(rows) for stream, rows in after_later.items()} == dict.fromkeys(env.streams, 4)
    assert restarted.delete(deleted.id, env.ledger, env.purger, wait_s=0)["pending"] is False
    assert env.snapshot() == after_later


def test_failed_import_analysis_rewinds_every_vision_stream_before_retry(analyzed_imports):
    env = analyzed_imports
    other, retry = env.add("other"), env.add("retry")
    assert not env.run(other).errors
    keep = env.snapshot()
    env.state.fail = True
    failed = env.run(retry)
    assert len(failed.errors) == 1 and "synthetic model failure" in failed.errors[0]
    env.state.fail = False
    assert not env.run(retry).errors
    assert {stream: len(rows) for stream, rows in env.snapshot().items()} == dict.fromkeys(env.streams, 4)
    assert env.pipe.delete(retry.id, env.ledger, env.purger, wait_s=0)["pending"] is False
    assert env.snapshot() == keep


def test_interrupted_import_vision_cleanup_keeps_ownership_for_deletion_retry(analyzed_imports, monkeypatch):
    import continuum_py

    env = analyzed_imports
    other, deleted = env.add("other"), env.add("deleted")
    assert not env.run(other).errors
    keep = env.snapshot()
    assert not env.run(deleted).errors
    forget = continuum_py.forget_segments_where
    interrupted = False

    def fail_before_groundings(base, column, value, **kwargs):
        nonlocal interrupted
        if column == "frame_id" and not interrupted:
            interrupted = True
            raise OSError("synthetic interruption before grounding deletion")
        return forget(base, column, value, **kwargs)

    monkeypatch.setattr(continuum_py, "forget_segments_where", fail_before_groundings)
    with pytest.raises(OSError, match="synthetic interruption before grounding deletion"):
        env.pipe.delete(deleted.id, env.ledger, env.purger, wait_s=0)
    partial = env.snapshot()
    assert partial[segment_log.ANIMAL_GROUNDING_EMBEDDINGS] == keep[segment_log.ANIMAL_GROUNDING_EMBEDDINGS]
    assert len(partial[segment_log.ANIMAL_GROUNDINGS]) == 4
    assert env.store.get(deleted.id) is not None
    assert env.store.removed(deleted.id)
    assert env.pipe.delete(deleted.id, env.ledger, env.purger, wait_s=0)["pending"] is False
    assert env.snapshot() == keep


def test_deleted_import_vision_tombstones_reject_late_writer_output(analyzed_imports):
    env = analyzed_imports
    other, deleted = env.add("other"), env.add("deleted")
    assert not env.run(other).errors
    keep = env.snapshot()
    assert not env.run(deleted).errors
    assert env.pipe.delete(deleted.id, env.ledger, env.purger, wait_s=0)["pending"] is False
    assert env.snapshot() == keep
    env.replay(deleted)
    assert env.snapshot() == keep
