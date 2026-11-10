"""A synthetic cam0 on a real broker, for the vision workers' tests: capture info, the stream, and JPEG frames."""

import asyncio
import uuid

import continuum_py
import cv2
import numpy as np


def capture(w: int, h: int, flip: bool = False, seq: int = 0) -> continuum_py.CaptureInfo:
    return {
        "capture_mono_ns": 0,
        "device_ts": 0,
        "arrival_ns": 0,
        "stride": 0,
        "width": w,
        "height": h,
        "flip": flip,
        "device_seq": seq,
    }


async def register_camera(client: continuum_py.Client) -> int:
    """Registers cam0.video, as capture does before any worker reads it."""
    registered = await client.register_stream(
        "cam0.video", kind="sensor", format="EncodedVideo", shape_name="MJPEG", slab_class=0, nominal_rate_hz=30.0
    )
    return registered["stream_id"]


async def camera(
    client: continuum_py.Client, stream_id: int, frames: int, step_s: float = 1.5, odd_ns: int = 0
) -> list[str]:
    """Publishes `frames` JPEG frames on cam0.video, `step_s` apart in their stamps (plus `odd_ns`, for a time float
    seconds can't hold exactly); returns their ids."""
    arena = client.arena(client.arena_names[0])
    ok, jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), 90, np.uint8))
    assert ok
    ids = []
    for n in range(frames):
        index, generation, offset = arena.write(0, jpeg.tobytes())
        pid = str(uuid.uuid7())
        start = 1_800_000_000 * 1_000_000_000 + round(n * step_s * 1e9) + odd_ns
        header = continuum_py.encode_header(
            {
                "id": pid,
                "stream_id": stream_id,
                "start_ns": start,
                "end_ns": start + 33_000_000,
                "layer": "Layer1Raw",
                "format": "EncodedVideo",
                "shape_name": "MJPEG",
                "parents": [],
                "subjects": [],
                "capture": capture(64, 48, seq=n),
            }
        )
        shm: continuum_py.ShmLocation = {
            "arena": arena.name,
            "arena_generation": arena.generation,
            "slab_class": 0,
            "slot": index,
            "slot_generation": generation,
            "offset": offset,
            "length": len(jpeg.tobytes()),
        }
        client.ingest(header, shm=shm)
        ids.append(pid)
        await asyncio.sleep(0.2)  # each frame is the newest while the worker reads it
    return ids
