"""The caption worker on Continuum: each camera's frames followed live from the broker (`see.frames`), run
through the caption pipeline (`see.captions`), and written as `camera.caption` and `camera.objects` rows in the
segment store (segment_log.py), with `animal.grounding` and `animal.grounding_embedding` rows for the animals the
captions name (see/caption_animals.py).

Each camera is looked at most every `min_interval_s` (1 s), newest frame first; the change gate
decides whether the VLM sees it, at most once per camera's caption interval (`caption_every_s`, 60 s for network
cameras by default). Every row names its frame (`frame_id`), and a caption repeated for an unchanged view says
since when (`since`). Segment rows are readable as soon as
they're appended and synced at most SYNC_S apart, so a crash loses at most that much. The pipelines are made and run
on one model thread shared by every camera: the in-process MLX backend answers only on the thread that loaded it.
"""

import asyncio
import concurrent.futures
import dataclasses
import functools
import os
import signal
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from continuum_py import Client

from liberational import logs, model_runtime, segment_log
from liberational.people import pet_shots
from liberational.see import caption_animals, frames
from liberational.see.captions import CaptionFrame, CaptionPipeline
from liberational.see.frames import NS, Camera, cameras_from_manifest, check_h264_decoder

log = logs.get("see-captions")

SYNC_S = 5.0  # segment rows are made durable at most this far apart


class CaptionWorker:
    """The caption worker: `run` follows every camera in `cameras` until `stop` is set. `pipeline(camera)` makes each
    camera's pipeline, on the model thread."""

    def __init__(
        self,
        client: Client,
        cameras: Sequence[Camera],
        pipeline: Callable[[Camera], CaptionPipeline],
        *,
        model: str,
        min_interval_s: float = 1.0,
        segments: segment_log.Writers | None = None,
        clock: Callable[[], float] = time.monotonic,
        embedding_model: str = "",
    ) -> None:
        self.client, self.cameras, self.new_pipeline = client, list(cameras), pipeline
        # Captions and object tracks, one writer per (stream, camera); written only from the event loop.
        self.segments = segments if segments is not None else segment_log.Writers(segment_log.root())
        self.clock = clock
        self._synced = clock()
        self.min_interval_s = min_interval_s
        self.model = model
        self.embedding_model = embedding_model  # what made the animal embeddings, for their rows
        self.frames = 0
        self.refusals = frames.Refusals()  # what the broker refused, logged as it comes
        self.committed = 0
        self._model = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="see-captions")

    def _slotted(self, cam: Camera) -> Camera:
        """A network camera read in capture-time slots of `min_interval_s` (frames.newest's `every_ns`), so the
        pictures it won't look at skip color conversion: converting every one of about 30 a second per
        stream cost about a core. The monotonic interval is then the slots' job."""
        if cam.kind != "rtsp" or cam.every_ns or self.min_interval_s <= 0:
            return cam
        return dataclasses.replace(cam, every_ns=round(self.min_interval_s * NS))

    async def run(self, stop: asyncio.Event) -> None:
        tasks = [asyncio.create_task(self._camera(cam)) for cam in self.cameras]
        tasks.append(asyncio.create_task(frames.watch_refusals(self.client, self.refusals)))
        try:
            await frames.until_stopped(stop, tasks)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._model.shutdown(wait=False, cancel_futures=True)
            self.segments.sync()
            self.segments.close()

    async def _camera(self, cam: Camera) -> None:
        loop = asyncio.get_running_loop()
        pipeline = await loop.run_in_executor(self._model, self.new_pipeline, cam)
        last_caption: str | None = None  # the payload id of the camera's last real caption
        slotted = self._slotted(cam)
        interval = 0.0 if slotted.every_ns else self.min_interval_s
        async for img, header in frames.newest(self.client, slotted, interval):
            ts = header["start_ns"] / NS
            seq = (header.get("capture") or {}).get("device_seq", 0)
            running = loop.run_in_executor(self._model, pipeline.feed, img, ts, seq)
            cancelled = False
            try:
                out = await asyncio.shield(running)
            except asyncio.CancelledError:
                # Stopping: the model thread can't be interrupted, so let the frame in it finish and keep what it gave.
                out = await running
                cancelled = True
            last_caption = self._handle(header["id"], header["start_ns"], out, last_caption)
            if cancelled:
                raise asyncio.CancelledError

    def _handle(
        self,
        frame_id: str,
        frame_ns: int,
        out: CaptionFrame,
        last_caption: str | None,
        *,
        payload_for: Callable[[str], str] | None = None,
    ) -> str | None:
        """Writes what a frame gave; returns the id of the camera's last real caption.

        `payload_for(stream)` fixes the ids (ingest retries). The live path leaves it empty and mints UUIDv7s.
        """
        self.frames += 1
        cam = str((out.caption or out.objects or {}).get("cam", ""))
        if out.animals:
            found = caption_animals.rows(
                out.animals, frame_id, frame_ns, f"grounding:{self.model}", self.embedding_model
            )
            for stream, records in found.items():
                if records:
                    self.segments.append(stream, cam, records)
        if out.objects is not None:
            grounding_id = str(uuid.uuid7()) if payload_for is None else payload_for(segment_log.CAMERA_OBJECTS)
            row = {**out.objects, "id": grounding_id, "frame_id": frame_id}
            self.segments.append(segment_log.CAMERA_OBJECTS, cam, [row])
        if out.caption is None:
            self._sync_due()
            return last_caption
        pid = str(uuid.uuid7()) if payload_for is None else payload_for(segment_log.CAPTIONS)
        self.segments.append(segment_log.CAPTIONS, cam, [{**out.caption, "id": pid, "frame_id": frame_id}])
        self._sync_due()
        self.committed += 1
        return pid if out.caption["observation"]["status"] == "fresh" else last_caption

    def _sync_due(self) -> None:
        if self.clock() - self._synced >= SYNC_S:
            self.segments.sync()
            self._synced = self.clock()


def caption_every_s(cam: Camera, network_s: float, own: Mapping[str, Any]) -> float:
    """How long a camera waits between VLM captions: its own `[vision.caption_every_s]` entry, else
    `network_s` for a network (`rtsp`) camera, else 0 (no wait). Raises ValueError for a negative or non-number one."""
    value = own.get(cam.name, network_s if cam.kind == "rtsp" else 0.0)
    if isinstance(value, bool) or not isinstance(value, int | float) or not value >= 0:
        raise ValueError(f"{cam.name}: a caption interval must be 0 or more seconds, not {value!r}")
    return float(value)


@model_runtime.reporting("see-captions")
def main(argv: Sequence[str] | None = None) -> None:
    """`pack-see-captions`: the caption worker as a service. It reads the cameras in capture's manifest from the broker
    (run continuum-broker and continuum-capture with it), with the VLM backend (pack-vlm on Apple Silicon, else
    Ollama) and, when [vision] objects is on and the backend is Ollama, object grounding."""
    import argparse

    from liberational import config as cfg
    from liberational import continuum_link
    from liberational.people import registry
    from liberational.people.pets import PetRegistry
    from liberational.see import vlm as v1
    from liberational.vision import animal_embedder

    parser = argparse.ArgumentParser(prog="pack-see-captions", description=main.__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("config/devices.toml"), help="capture's device manifest")
    parser.add_argument("--socket", type=Path, default=None, help="the broker's socket (default: under paths.state)")
    parser.add_argument(
        "--backend",
        choices=["auto", "mlx-server", "mlx", "ollama"],
        default=cfg.get("vision.backend"),
        help="the VLM backend (default: [vision] backend)",
    )
    args = parser.parse_args(argv)
    if args.socket is not None:
        # Every shared reader and the model thread's voice gate must follow the broker selected by this process.
        os.environ["CONTINUUM_SOCKET"] = str(args.socket)

    cameras = cameras_from_manifest(args.manifest)
    frames.one_opencv_thread()
    if not cameras:
        raise SystemExit(f"no cameras in {args.manifest}")
    check_h264_decoder(cameras)
    backend = args.backend if args.backend != "auto" else "mlx-server" if cfg.has_mlx() else "ollama"
    vlm: list[v1.Backend] = []  # made once, on the model thread
    embed = caption_animals.LazyEmbedder()  # shared by every camera, loaded on the first animal box
    pet_photos = pet_shots.Writer(cfg.path("paths.state"))  # Pets' sighting photos, one a track
    pets = PetRegistry(registry.knowledge_path(None))

    names: list[Any] = [-float("inf"), []]  # when the taught pets' names were read, and them

    def pet_names() -> list[str]:
        """The taught pets' names, read at most once a minute (only fresh captions ask)."""
        if time.monotonic() - names[0] >= 60:
            names[:] = [time.monotonic(), [str(p["name"]) for p in pets.listing()]]
        return names[1]

    def pipeline(cam: Camera) -> CaptionPipeline:
        if not vlm:
            vlm.append(v1.make_vlm(backend))
            vlm[0].warm()
        ollama = vlm[0] if isinstance(vlm[0], v1.OllamaVLM) else None
        ground = ollama.ground if ollama is not None and cfg.get("vision.objects") else None
        every = caption_every_s(cam, cfg.get("vision.network_caption_every_s"), cfg.get("vision.caption_every_s") or {})
        find = None if ollama is None else functools.partial(ollama.ground, prompt=caption_animals.PROMPT)
        animals = caption_animals.CaptionAnimals(cam.name, find, embed, pet_names, shots=pet_photos)
        return CaptionPipeline(cam.name, vlm[0], ground=ground, caption_every_s=every, animals=animals)

    async def run() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        async with continuum_link.connected(client_name="see-captions") as client:
            worker = CaptionWorker(client, cameras, pipeline, model=backend, embedding_model=animal_embedder.space())
            log.info("watching %s (vlm %s)", ", ".join(c.stream for c in cameras), backend)
            await worker.run(stop)

    continuum_link.run_service(run())
