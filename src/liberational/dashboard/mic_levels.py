"""Live microphone levels for the dashboard's meters (GET /audio/levels), read from the broker's PCM rings (#548).

v1's microphone service served these from its own port; on Continuum the dashboard reads each `<mic>.audio` ring
itself, under a read-only lease, from where the ring is when the page asks: every WINDOW_S it takes the records that
landed since and sends their RMS and peak in dBFS. The listen worker's `speech.activity` rows (in the segment store)
add a `speech` boolean while its energy gate state is fresh. This is a sound candidate, not Silero-verified speech.
Missing or stale activity omits the flag, so the page shows unavailable rather than quiet. A microphone with nothing
new in a window is left out of that update. No model runs in the dashboard.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import speech_log
from liberational.listen.pcm import STAMP, Format

WINDOW_S = 0.04  # about 25 updates a second, as v1's meters had
ACTIVITY_POLL_S = 0.25
ACTIVITY_FRESH_S = 3.0  # listen publishes transitions and a heartbeat every second
_BYTES = {"S16LE": 2, "S24LE3": 3, "S32LE": 4, "F32LE": 4}


def layout(shape: str, stride: int) -> Format | None:
    """A ring's record layout from its shape name and stride (the rate doesn't matter for a level), or None."""
    kind, _, ch = shape.removeprefix("PCM_").partition("x")
    if kind not in _BYTES or not ch.isdigit() or int(ch) < 1:
        return None
    frame = int(ch) * _BYTES[kind]
    if stride <= STAMP.size or (stride - STAMP.size) % frame:
        return None
    return Format(kind, int(ch), 0, (stride - STAMP.size) // frame)


def db(x: float) -> float:
    return round(20 * float(np.log10(x + 1e-12)), 1)


async def activity(now: float, base: Path | None = None) -> dict[str, bool]:
    """The newest fresh energy-gate state per microphone, from the segment store (`base`, by default under
    paths.state); an absent state is unknown, never quiet."""
    return await asyncio.to_thread(functools.partial(speech_log.activity, now, ACTIVITY_FRESH_S, base=base))


async def _discard(sub: AsyncIterator[object]) -> None:
    """Reads a subscription's events and drops them: the meters read the ring itself, not the batch notices."""
    async for _ in sub:
        pass


async def stream(socket: Path, send: Callable[[dict[str, Any]], None], stop: Callable[[], bool]) -> None:
    """Sends each window's levels through `send` until `stop()` says the page left (or `send` raises)."""
    import continuum_py

    channels = cfg.get("audio.mic_channel") or {}
    async with contextlib.AsyncExitStack() as stack:
        client = await continuum_py.Client.connect(str(socket), client_name="levels")
        stack.push_async_callback(client.close)
        # The meters read the rings directly, and a full subscription queue blocks its connection's socket reader,
        # so the subscriptions and their rings get a connection of their own and every request stays on `client`
        # (#1267). That connection makes no requests, so it uses backpressure rather than the default policy (which
        # closes a connection whose subscription queue overflows, ending its ring leases), and each subscription's
        # events are drained and dropped so the broker never sees the connection stop reading.
        rings = await continuum_py.Client.connect(
            str(socket), client_name="levels-rings", subscription_overflow="backpressure"
        )
        stack.push_async_callback(rings.close)
        readers: dict[str, tuple[Any, Format, int]] = {}
        for s in await client.list_streams():
            name = s["name"]
            if not name.endswith(".audio") or not str(s["shape_name"]).startswith("PCM_"):
                continue
            sub = await rings.subscribe([name])
            stack.push_async_callback(sub.unsubscribe)
            stack.callback(asyncio.create_task(_discard(sub)).cancel)
            if not sub.rings:
                continue
            lease = sub.rings[0]["lease"]
            fmt = layout(str(s["shape_name"]), lease["ring_stride"])
            if fmt is None:
                continue
            mic = name.removesuffix(".audio")
            readers[mic] = (rings.ring_reader(lease), fmt, min(int(channels.get(mic, 0)), fmt.channels - 1))
        if not readers:
            send({"error": "No microphone is recording, so there are no live levels."})
            return
        states: dict[str, bool] = {}
        next_activity = 0.0
        while not stop():
            if time.monotonic() >= next_activity:
                states = await activity(time.time())
                next_activity = time.monotonic() + ACTIVITY_POLL_S
            levels = {}
            for mic, (reader, fmt, ch) in readers.items():
                xs = []
                while (got := reader.read_record()) is not None:
                    xs.append(fmt.decode(got[1], ch)[2])
                if xs:
                    x = np.concatenate(xs)
                    levels[mic] = {"rms": db(float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))),
                                   "peak": db(float(np.max(np.abs(x))))}  # fmt: skip
                    if mic in states:
                        levels[mic]["speech"] = states[mic]
            send({"ts": time.time(), "levels": levels})
            await asyncio.sleep(WINDOW_S)
