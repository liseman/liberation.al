import asyncio
import json
import os
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path

import continuum_py
import numpy as np
import pytest

from camera_fake import camera, register_camera
from liberational import segment_log
from liberational.see.caption_worker import CaptionWorker
from liberational.see.captions import CaptionPipeline
from liberational.see.frames import Camera

SPAN = (1_800_000_000.0, 1_800_000_030.0)


class VLM:
    name = "fake"

    def caption(self, img: np.ndarray, memory) -> str:
        return "a grey wall"


def run_worker[T](
    broker: Path, make, frames: int, step_s: float, done: Callable[[continuum_py.Client], Awaitable[T]]
) -> tuple[list[str], T]:
    async def go() -> tuple[list[str], T]:
        async with (
            await continuum_py.Client.connect(str(broker), client_name="worker") as worker_client,
            await continuum_py.Client.connect(str(broker), client_name="camera") as cam_client,
        ):
            stream_id = await register_camera(cam_client)
            worker = make(worker_client)
            stop = asyncio.Event()
            running = asyncio.create_task(worker.run(stop))
            await asyncio.sleep(0.3)
            ids = await camera(cam_client, stream_id, frames, step_s=step_s)
            for _ in range(100):
                if worker.frames >= frames:
                    break
                await asyncio.sleep(0.05)
            stop.set()
            await running
            await asyncio.sleep(0.2)
            return ids, await done(cam_client)

    return asyncio.run(go())


def test_a_steady_view_is_captioned_once_then_repeated_as_unchanged_since_then(broker: Path, tmp_path: Path) -> None:
    def make(client: continuum_py.Client) -> CaptionWorker:
        return CaptionWorker(
            client,
            [Camera("cam0")],
            lambda cam: CaptionPipeline(cam.name, VLM()),
            model="fake",
            min_interval_s=0.0,
            segments=segment_log.Writers(tmp_path),
        )

    async def captions(_: continuum_py.Client) -> list[dict]:
        return segment_log.Log(tmp_path).range(segment_log.CAPTIONS, *SPAN)  # the worker synced as it stopped

    # Nine identical frames 1.5 s apart: captioned at 0 s, repeats at 1.5 s and then 10 s on (12 s).
    frames, (first, second, third) = run_worker(broker, make, 9, 1.5, captions)
    assert first["observation"]["status"] == "fresh" and first["frame_id"] == frames[0]
    assert (second["unchanged"], second["caption"], second["since"]) == (True, "a grey wall", first["ts_start"])
    assert (second["frame_id"], third["frame_id"]) == (frames[1], frames[8])
    assert "unchanged" not in first


def test_a_camera_whose_model_wont_load_stops_the_worker_with_its_error(broker: Path) -> None:
    def broken(cam: Camera) -> CaptionPipeline:
        raise ConnectionError("ollama isn't answering")

    async def go() -> None:
        async with await continuum_py.Client.connect(str(broker), client_name="worker") as client:
            worker = CaptionWorker(client, [Camera("cam0")], broken, model="fake")
            await asyncio.wait_for(worker.run(asyncio.Event()), timeout=5.0)

    with pytest.raises(ConnectionError, match="isn't answering"):
        asyncio.run(go())


@pytest.fixture
def caption_entrypoint(broker, tmp_path, monkeypatch):
    from unittest.mock import create_autospec

    from liberational import continuum_link, voice_turn
    from liberational.see import caption_worker

    manifest = tmp_path / "devices.toml"
    manifest.write_text('[[device]]\nname = "cam0"\nkind = "camera"\n')

    async def activate_voice():
        async with await continuum_py.Client.connect(str(broker), client_name="test-voice") as client:
            client.set_system_state("Thinking")
            await asyncio.wait_for(client.next_event(), timeout=2)

    asyncio.run(activate_voice())
    worker = create_autospec(CaptionWorker, instance=True)
    observed = []

    async def run(stop):
        observed.append((continuum_link.socket_path(), voice_turn.busy()))

    worker.run.side_effect = run
    monkeypatch.setattr(caption_worker, "CaptionWorker", create_autospec(CaptionWorker, return_value=worker))

    def start(socket_args, environment):
        monkeypatch.setenv("CONTINUUM_SOCKET", environment)
        caption_worker.main(["--manifest", str(manifest), "--backend", "ollama", *socket_args])
        return observed, continuum_link.socket_path()

    return start


def test_caption_socket_option_overrides_existing_environment_for_voice_gate(caption_entrypoint, broker, tmp_path):
    observed, selected = caption_entrypoint(["--socket", str(broker)], str(tmp_path / "other.sock"))
    assert observed == [(broker, True)]
    assert selected == broker


def test_caption_socket_option_overrides_default_for_voice_gate(caption_entrypoint, broker):
    observed, selected = caption_entrypoint(["--socket", str(broker)], "")
    assert observed == [(broker, True)]
    assert selected == broker


def test_caption_without_socket_option_preserves_environment_for_voice_gate(caption_entrypoint, broker):
    observed, selected = caption_entrypoint([], str(broker))
    assert observed == [(broker, True)]
    assert selected == broker


def test_caption_entrypoint_reports_in_process_mlx_fallback_from_model_thread(broker, tmp_path, monkeypatch):
    from unittest.mock import create_autospec

    from liberational import config as cfg
    from liberational import model_runtime
    from liberational.see import caption_worker, vlm

    manifest = tmp_path / "devices.toml"
    manifest.write_text('[[device]]\nname = "cam0"\nkind = "camera"\n')
    seen = []

    def unavailable():
        raise ConnectionError("VLM server unavailable")

    class LocalVLM:
        def __init__(self, repo, px):
            self.token = model_runtime.loaded("models.hf.vlm", repo, "camera captions", "metal", owner=self)
            assert self.token is not None, "caption entrypoint must activate reporting on its model thread"

        def warm(self):
            (report,) = (tmp_path / "model_loads").glob("*.json")
            seen.append(json.loads(report.read_text()))

    worker = create_autospec(CaptionWorker, instance=True)
    factory = create_autospec(CaptionWorker, return_value=worker)

    async def run(stop):
        pipeline = factory.call_args.args[2]
        await asyncio.to_thread(pipeline, Camera("cam0"))

    worker.run.side_effect = run
    monkeypatch.setattr(caption_worker, "CaptionWorker", factory)
    monkeypatch.setattr(vlm, "MlxServerVLM", unavailable)
    monkeypatch.setattr(vlm, "MlxVLM", LocalVLM)
    monkeypatch.setattr(vlm.OllamaVLM, "__init__", lambda *args: pytest.fail("unexpected Ollama fallback"))
    monkeypatch.setenv("CONTINUUM_SOCKET", str(broker))
    with cfg.override({"paths": {"state": str(tmp_path)}, "vision": {"objects": False}}):
        caption_worker.main(["--manifest", str(manifest), "--socket", str(broker), "--backend", "mlx-server"])
    assert [(row["pid"], row["service"], row["models"][0]["device"]) for row in seen] == [
        (os.getpid(), "see-captions", "Apple GPU")
    ]
    assert list((tmp_path / "model_loads").glob("*.json")) == []


def test_stopping_during_grounding_keeps_completed_original_outputs(broker: Path, tmp_path: Path) -> None:

    async def run():
        entered, release = threading.Event(), threading.Event()

        def ground(_):
            entered.set()
            assert release.wait(5), "test must release the model thread"
            return "[]"

        async with (
            await continuum_py.Client.connect(str(broker), client_name="object-worker") as client,
            await continuum_py.Client.connect(str(broker), client_name="object-camera") as capture_client,
        ):
            stream = await register_camera(capture_client)
            worker = CaptionWorker(
                client,
                [Camera("cam0")],
                lambda cam: CaptionPipeline(cam.name, VLM(), ground=ground),
                model="fake",
                min_interval_s=0.0,
                segments=segment_log.Writers(tmp_path),
            )
            stop = asyncio.Event()
            task = asyncio.create_task(worker.run(stop))
            try:
                await asyncio.sleep(0.3)
                ids = await camera(capture_client, stream, 1)
                assert await asyncio.to_thread(entered.wait, 5)
                stop.set()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                release.set()
                await asyncio.wait_for(task, 5)
                log = segment_log.Log(tmp_path)  # the worker synced as it stopped
                rows = log.range(segment_log.CAMERA_OBJECTS, *SPAN)
                captions = log.range(segment_log.CAPTIONS, *SPAN)
                assert len(rows) == len(captions) == 1
                assert rows[0]["observation"]["status"] == "fresh"
                assert [rows[0]["frame_id"]] == [captions[0]["frame_id"]] == ids
            finally:
                release.set()
                stop.set()
                await task

    asyncio.run(run())


@pytest.mark.parametrize(
    "kind,every_ns,interval",
    [
        pytest.param("rtsp", 1_000_000_000, 0.0, id="network-in-slots"),
        pytest.param("camera", 0, 1.0, id="usb-as-before"),
    ],
)
def test_a_network_camera_is_read_in_capture_slots_of_the_look_interval(monkeypatch, kind, every_ns, interval):
    from typing import Any, cast

    from liberational.see import caption_worker as CW

    asked = []

    async def newest(client, cam, min_interval_s, **kwargs):
        asked.append((cam.every_ns, min_interval_s))
        return
        yield

    monkeypatch.setattr(CW.frames, "newest", newest)
    worker = CaptionWorker(cast(Any, object()), [], lambda cam: cast(Any, object()), model="fake")
    try:
        asyncio.run(worker._camera(Camera("patio", kind)))
    finally:
        worker._model.shutdown()
    assert asked == [(every_ns, interval)]
