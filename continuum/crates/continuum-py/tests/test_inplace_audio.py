"""FLAC files registered in place through continuum_py, against a real broker process, and the purge and
FLAC helpers the speech importer uses.

Run with `mise run continuum-test`, which builds the broker first. `fixtures/tone.flac` is half a second of a
440 Hz tone, 16 bit mono at 48 kHz (made with ffmpeg's sine source).
"""

import asyncio
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from continuum_py import Client, decode_flac

from conftest import spawn_broker, stop_broker

TONE = Path(__file__).parent / "fixtures" / "tone.flac"
START = 1_793_300_000_250_000_000


@pytest.fixture
def socket(tmp_path: Path) -> Iterator[str]:
    path = tmp_path / "b.sock"
    broker = spawn_broker(path, "--heartbeat-timeout-s", "30")
    try:
        yield str(path)
    finally:
        stop_broker(broker, path)


def device_dir(tmp_path: Path) -> Path:
    """A directory to register: the tone, and an index line for a file that's gone."""
    device = tmp_path / "alsa-input-wave-3"
    (device / "2026-10-29").mkdir(parents=True)
    shutil.copy(TONE, device / "2026-10-29" / "140000.flac")
    (device / "index.jsonl").write_text(
        '{"file": "2026-10-29/140000.flac", "start": 1793300000.25, "sr": 48000}\n'
        '{"file": "2026-10-29/140009.flac", "start": 1793300009.25, "sr": 48000}\n'
    )
    return device


def test_the_archive_registers_once_and_a_second_run_is_a_repeat(socket: str, tmp_path: Path) -> None:
    device = device_dir(tmp_path)

    async def run() -> tuple[list[dict], list[dict]]:
        async with await Client.connect(socket, client_name="importer") as client:
            await client.register_worker("audio-import", [])
            first = await client.register_audio_index("wave3.audio_v1", str(device))
            second = await client.register_audio_index("wave3.audio_v1", str(device))
            return first, second

    first, second = asyncio.run(run())
    assert [e["registered"] for e in first] == [True, False]
    assert first[0]["start_ns"] == START
    assert first[0]["end_ns"] == START + 500_000_000
    assert first[0]["duplicate"] is False
    assert "not on disk" in first[1]["skipped"]
    assert second[0]["duplicate"] is True
    assert (second[0]["payload_id"], second[0]["chunk_id"]) == (first[0]["payload_id"], first[0]["chunk_id"])


def test_a_flac_file_decodes_to_capture_scaled_f32() -> None:
    rate, channels, samples = decode_flac(TONE.read_bytes())
    pcm = np.frombuffer(samples, "<f4")
    assert (rate, channels, len(pcm)) == (48_000, 1, 24_000)
    # 16 bit integers over 2^15: every sample is an exact multiple of 1/32768, inside [-1, 1).
    assert np.all(pcm * 32_768 == np.round(pcm * 32_768))
    assert pcm.min() >= -1.0 and pcm.max() < 1.0
    with pytest.raises(ValueError):
        decode_flac(b"not flac")


def test_a_range_purge_reports_its_id(socket: str) -> None:
    async def run() -> dict:
        async with await Client.connect(socket) as client:
            await client.register_stream("mic.audio", kind="sensor", format="RawBytes", shape_name="PCM", slab_class=0)
            return await client.purge_range(START, START + 1_000_000_000, streams=["mic.audio"], reason="test")

    done = asyncio.run(run())
    assert len(done["purge_id"]) == 36
    assert isinstance(done["removed"], list)
