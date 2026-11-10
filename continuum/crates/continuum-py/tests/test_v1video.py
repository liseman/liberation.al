"""v1's recordings registered in place through continuum_py, against a real broker process, and read back by GOP
for the vision port (#549).

Run with `mise run continuum-test`, which builds the broker first. `fixtures/v1_bframes.mp4` is 2 seconds at 10 fps,
64x48, H.264 with B-frames and a keyframe every 10 frames (made with ffmpeg's testsrc and libx264 -bf 2 -g 10).
"""

import asyncio
import itertools
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from continuum_py import Client, ContinuumError, v1_gop_units

from conftest import spawn_broker, stop_broker

RECORDING = Path(__file__).parent / "fixtures" / "v1_bframes.mp4"
START = 1_793_300_000_250_000_000
MS = 1_000_000


@pytest.fixture
def socket(tmp_path: Path) -> Iterator[str]:
    path = tmp_path / "b.sock"
    broker = spawn_broker(path, "--heartbeat-timeout-s", "30")
    try:
        yield str(path)
    finally:
        stop_broker(broker, path)


def video_dir(tmp_path: Path) -> Path:
    video = tmp_path / "video"
    (video / "cam0" / "2026-10-29").mkdir(parents=True)
    shutil.copy(RECORDING, video / "cam0" / "2026-10-29" / "140000.mp4")
    (video / "index.jsonl").write_text(
        '{"cam": "cam0", "file": "cam0/2026-10-29/140000.mp4", "start": 1793300000.25, "fps": 10.0, "w": 64, "h": 48}\n'
    )
    return video


def test_recordings_register_once_and_a_second_run_is_a_repeat(socket: str, tmp_path: Path) -> None:
    video = video_dir(tmp_path)

    async def run() -> tuple[list[dict], list[dict]]:
        async with await Client.connect(socket, client_name="v1 video importer") as client:
            await client.register_worker("v1-video-import", [])
            first = await client.register_v1_video_index("cam0", str(video))
            second = await client.register_v1_video_index("cam0", str(video))
            return first, second

    first, second = asyncio.run(run())
    assert [e["registered"] for e in first] == [True]
    assert (first[0]["start_ns"], first[0]["end_ns"]) == (START, START + 2_000 * MS)
    assert second[0]["duplicate"] is True
    assert second[0]["chunk_id"] == first[0]["chunk_id"]


def test_a_gop_reads_back_in_decode_order_with_its_times() -> None:
    units = v1_gop_units(RECORDING, START, 1)
    assert len(units) == 10
    assert [u["key"] for u in units] == [True] + [False] * 9
    assert units[0]["bytes"][:5] == b"\x00\x00\x00\x01\x67"
    assert sorted(u["pts_ns"] for u in units) == [START + (10 + k) * 100 * MS for k in range(10)]
    assert all(a["dts_ns"] < b["dts_ns"] for a, b in itertools.pairwise(units))
    with pytest.raises(ContinuumError):
        v1_gop_units(RECORDING, START, 2)
