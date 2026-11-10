"""The room's audio on Continuum: a span of one microphone's PCM, read back from the broker.

Capture writes each microphone into a ring on `<device>.audio` (stamped records, `listen/pcm.py`), and the tier worker
writes the ring's batches into a chunk file that it commits every 15 seconds. A query of the stream over a span gives those
chunks; `continuum_py.read.records` gives each chunk's batches, a batch being the ring's records exactly as captured,
with the batch's UTC span. So a span of audio is the batches it overlaps, decoded and laid at their UTC times, with
silence where nothing was kept.

Audio in a chunk not committed yet (up to the last 15 seconds) isn't read here: it would need a live ring lease. A
clip asked for that soon has no samples yet, and says so.

Audio registered in place (`<stream>.audio_v1`) is whole FLAC files: each comes back
whole as one record, decoded with `decode_flac` and laid at its start like a batch.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from continuum_py import decode_flac, decode_header
from continuum_py.read import PcmFormat, records
from continuum_py.schema import Item, Locator

from liberational import config as cfg
from liberational.listen.pcm import STAMP, Format

if TYPE_CHECKING:
    from continuum_py import Client

NS = 1_000_000_000
_SHAPE = re.compile(r"PCM_(S16LE|S24LE3|S32LE|F32LE)x(\d+)")
_BYTES = {"S16LE": 2, "S24LE3": 3, "S32LE": 4, "F32LE": 4}
RATES = (8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000)
MAX_SPAN_NS = 330 * NS  # the longest span read at once (a video segment is five minutes)


def split_batch(
    shape: str, batch: bytes, span_ns: int, pcm: PcmFormat | None = None
) -> tuple[Format, list[bytes]] | None:
    """A batch's ring records and their layout, or None for bytes that aren't whole records of `shape`. The stride
    and rate use explicit archive metadata when present. Legacy batches infer stride from sample indexes and rate
    from the batch's span."""
    m = _SHAPE.fullmatch(shape)
    if m is None or len(batch) < STAMP.size:
        return None
    kind, channels = m.group(1), int(m.group(2))
    if channels <= 0:
        return None
    frame = channels * _BYTES[kind]
    if pcm is not None:
        stride = pcm.stride
        block = stride - STAMP.size
        if (
            kind != "F32LE"
            or pcm.channels != channels
            or pcm.rate_hz <= 0
            or block <= 0
            or block % frame
            or len(batch) % stride
        ):
            return None
        return Format(kind, channels, pcm.rate_hz, block // frame), [
            batch[i : i + stride] for i in range(0, len(batch), stride)
        ]
    first = STAMP.unpack_from(batch)[1]
    stride = len(batch)
    if len(batch) >= 2 * (STAMP.size + frame):
        # Where would the second record's stamp be? Try each stride that divides the batch, smallest first, and take
        # the one whose second stamp follows on from the first.
        for frames in range(1, (len(batch) - STAMP.size) // frame + 1):
            s = STAMP.size + frames * frame
            if len(batch) % s == 0 and len(batch) > s and STAMP.unpack_from(batch, s)[1] == first + frames:
                stride = s
                break
    frames = (stride - STAMP.size) // frame
    if (stride - STAMP.size) % frame or len(batch) % stride:
        return None
    n = len(batch) // stride
    rate = min(RATES, key=lambda r: abs(r - n * frames * NS / max(span_ns, 1)))
    return Format(kind, channels, rate, frames), [batch[i : i + stride] for i in range(0, len(batch), stride)]


async def read(client: Client, stream: str, start_ns: int, end_ns: int, channel: int = 0) -> tuple[int, np.ndarray]:
    """`stream`'s audio over [start_ns, end_ns), one channel as float32 at its own rate, with silence where nothing
    was kept: (rate, samples). The rate is 16 kHz when nothing was found at all. Raises ValueError for an empty or
    too long span."""
    if not 0 < end_ns - start_ns <= MAX_SPAN_NS:
        raise ValueError("a span of audio is 1 ns to 330 s")
    result = await client.query(stream, start_ns, end_ns, all=True)
    pieces: list[tuple[int, int, np.ndarray]] = []  # (UTC ns of the first sample, rate, samples)
    for item in result["items"]:
        locator = item["locator"]
        if locator is None or locator["kind"] != "Chunk":
            continue  # still in the ring: not flushed yet
        header = decode_header(item["header"])
        shape = header["shape_name"]
        uri = str(locator.get("uri") or "")
        if uri and not uri.startswith("/") and "://" not in uri:
            # A chunk the tier worker recorded before it wrote absolute paths: relative to the install's root, where
            # the services run.
            item = Item(header=item["header"], locator=Locator(**{**locator, "uri": str(cfg.ROOT / uri)}))
        try:
            found = await records(client, stream, item, start_ns=start_ns, end_ns=end_ns) or []
        except ValueError, OSError:
            continue  # a chunk this machine can't read (not a local file, gone, corrupt): silence there
        for batch in found:
            if bytes(batch.data[:4]) == b"fLaC":  # a file registered in place, whole
                try:
                    rate_hz, channels, pcm = decode_flac(batch.data)
                except ValueError:
                    continue
                interleaved = np.frombuffer(pcm, "<f4").reshape(-1, channels)
                pieces.append((batch.start_ns, rate_hz, interleaved[:, min(channel, channels - 1)].copy()))
                continue
            got = split_batch(shape, batch.data, batch.end_ns - batch.start_ns, batch.pcm)
            if got is None:
                continue
            fmt, recs = got
            ch = min(channel, fmt.channels - 1)
            samples = np.concatenate([fmt.decode(r, ch)[2] for r in recs])
            pieces.append((batch.start_ns, fmt.rate_hz, samples))
    rate = pieces[0][1] if pieces else 16_000
    out = np.zeros(int((end_ns - start_ns) * rate // NS), np.float32)
    for t0, r, samples in pieces:
        if r != rate:
            continue  # a stream doesn't change rate; a batch that says otherwise is skipped, not resampled
        at = (t0 - start_ns) * rate // NS
        lo, hi = max(0, at), min(len(out), at + len(samples))
        if hi > lo:
            out[lo:hi] = samples[lo - at : hi - at]
    return rate, out


async def spans(client: Client, stream: str, start_ns: int, end_ns: int) -> list[tuple[int, int]]:
    """The kept stretches of `stream` over [start_ns, end_ns): each flushed chunk's span, merged where they touch."""
    result = await client.query(stream, start_ns, end_ns, all=True)
    out: list[tuple[int, int]] = []
    for item in result["items"]:
        if item["locator"] is None or item["locator"]["kind"] != "Chunk":
            continue
        h = decode_header(item["header"])
        if out and h["start_ns"] <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], h["end_ns"]))
        else:
            out.append((h["start_ns"], h["end_ns"]))
    return out
