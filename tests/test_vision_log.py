import asyncio
import collections
from pathlib import Path
from typing import Any, cast

import continuum_py
import pytest
from continuum_py import read

from liberational import segment_log, vision_log

PID = "0190c8e0-0000-7000-8000-000000000001"


def stored_header(stream: str, start_ns: int, end_ns: int) -> bytes:
    """A payload header as the broker stores one, for the `continuum_py.read` tests below."""
    header: continuum_py.PayloadHeader = {
        "id": PID,
        "stream_id": 1,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "layer": "Layer1Raw",
        "format": "Text",
        "shape_name": stream,
        "parents": [],
        "subjects": [],
    }
    return continuum_py.encode_header(header)


def caption(ts: float, cam: str, text: str, pid: int) -> dict:
    return {"ts_start": ts, "ts_end": ts, "cam": cam, "caption": text, "id": f"0190c8e0-0000-7000-8000-{pid:012d}"}


def test_scene_entries_are_the_newest_oldest_first_with_v1s_ts() -> None:
    records = [caption(30.0, "cam1", "c", 3), caption(10.0, "cam0", "a", 1), caption(20.0, "cam0", "b", 2)]
    scene = vision_log.scene_entries(records, limit=2)
    assert [(e["caption"], e["ts"]) for e in scene] == [("b", 20.0), ("c", 30.0)]
    assert vision_log.scene_entries(records, limit=0) == []


def test_the_latest_caption_of_each_camera() -> None:
    records = [caption(30.0, "cam1", "c", 3), caption(10.0, "cam0", "a", 1), caption(20.0, "cam0", "b", 2)]
    latest = vision_log.latest_by_camera(records)
    assert {cam: r["caption"] for cam, r in latest.items()} == {"cam0": "b", "cam1": "c"}


# ---- Against a real broker (the `broker` fixture is in conftest.py)


def test_the_log_reads_the_caption_store_in_time_order_and_the_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(segment_log, "at", lambda state=None: segment_log.Log(segment_log.root(state)))
    # Captions are the caption worker's segment rows: the scene reads them from the log's state.
    writers = segment_log.Writers(segment_log.root(tmp_path))
    for ts, cam, text in [(100.0, "cam0", "empty room"), (101.0, "cam1", "a cat"), (99.0, "cam0", "Sam leaves")]:
        writers.append(segment_log.CAPTIONS, cam, [{"id": segment_log.new_id(), "ts_start": ts, "caption": text}])
    writers.append(segment_log.CAPTIONS, "cam0", [{"id": segment_log.new_id(), "ts_start": 900.0, "caption": "later"}])
    writers.close()
    log = vision_log.VisionLog(state=tmp_path)
    try:
        captions = log.captions(98.0, 150.0)
        scene = log.scene(now=150.0, limit=2)
        latest = log.latest(now=150.0)
    finally:
        log.close()
    assert [r["caption"] for r in captions] == ["Sam leaves", "empty room", "a cat"]
    assert [e["caption"] for e in scene] == ["empty room", "a cat"]
    assert {cam: r["caption"] for cam, r in latest.items()} == {"cam0": "empty room", "cam1": "a cat"}


def test_a_relative_chunk_locator_is_read_under_the_data_root_and_an_absolute_one_as_it_is(
    monkeypatch, tmp_path
) -> None:
    # Locators are stored relative to the data root (#62), so a moved install reads its chunks from where they are now.
    header = stored_header(vision_log.OBJECTS, 5, 5)
    opened = []

    def read_chunk_records(uri, start_ns, end_ns):
        opened.append(uri)
        return [(PID, 5, 5, b"{}")]

    monkeypatch.setattr(read, "read_chunk_records", read_chunk_records)
    relative = {"header": header, "locator": {"kind": "Chunk", "uri": "continuum/tier/7/h/7-1-1.cnk"}}
    got = asyncio.run(read.records(None, vision_log.OBJECTS, relative, root=tmp_path / "moved"))  # ty: ignore[invalid-argument-type] - a chunk read needs no client
    assert got == [read.Record(PID, 5, 5, b"{}")]
    absolute = {"header": header, "locator": {"kind": "Chunk", "uri": "file:///elsewhere/7-1-1.cnk"}}
    asyncio.run(read.records(None, vision_log.OBJECTS, absolute, root=tmp_path))  # ty: ignore[invalid-argument-type] - a chunk read needs no client
    assert opened == [str(tmp_path / "moved/continuum/tier/7/h/7-1-1.cnk"), "/elsewhere/7-1-1.cnk"]


class _SameItem:
    """A client whose query names `item` again: the catalog still has the chunk when it's asked a second time."""

    def __init__(self, item: dict) -> None:
        self.item = item

    async def query(self, stream, start_ns, end_ns, *, all=False):
        return {"items": [self.item]}


def _chunk_item(uri: str) -> dict:
    return {"header": stored_header(vision_log.OBJECTS, 5, 5), "locator": {"kind": "Chunk", "uri": uri}}


def test_a_chunk_file_the_catalog_still_names_but_that_isnt_there_is_missing_not_stale(
    monkeypatch, tmp_path, caplog
) -> None:
    monkeypatch.setattr(read, "read_chunk_records", lambda uri, start_ns, end_ns: None)
    monkeypatch.setattr(read, "_REPORTED", set())
    item = _chunk_item("continuum/tier/7/h/gone.cnk")
    for _ in range(2):
        with pytest.raises(read.ChunkMissing) as missing:
            asyncio.run(read.records(_SameItem(item), vision_log.OBJECTS, item, root=tmp_path))  # ty: ignore[invalid-argument-type] - a fake client with query only
        assert missing.value.filename == str(tmp_path / "continuum/tier/7/h/gone.cnk")
    assert [r.levelname for r in caplog.records if r.name == "continuum_py.read"] == ["ERROR"], (
        "logged once per file, not once per request"
    )


def test_a_chunk_whose_file_is_there_but_moved_again_during_the_retry_is_stale(monkeypatch, tmp_path) -> None:
    (tmp_path / "here.cnk").write_bytes(b"")
    monkeypatch.setattr(read, "read_chunk_records", lambda uri, start_ns, end_ns: None)
    item = _chunk_item("here.cnk")
    with pytest.raises(read.StaleLocator):
        asyncio.run(read.records(_SameItem(item), vision_log.OBJECTS, item, root=tmp_path))  # ty: ignore[invalid-argument-type] - a fake client with query only


def test_a_chunk_still_being_written_is_read_only_as_far_as_its_item_reaches(monkeypatch) -> None:
    # A camera's chunk is committed in parts while its file is open, so the file can hold records the item doesn't
    # name yet (they are still Tier 0 items of their own): a chunk is read over its item's span, never past it.
    header = stored_header(vision_log.OBJECTS, 100, 200)
    asked = []

    def read_chunk_records(uri, start_ns, end_ns):
        asked.append((start_ns, end_ns))
        return [(PID, 150, 150, b"{}")]

    monkeypatch.setattr(read, "read_chunk_records", read_chunk_records)
    item = {"header": header, "locator": {"kind": "Chunk", "uri": "/tier/7-1-1.cnk"}}
    for window in ({}, {"start_ns": 120, "end_ns": 160}, {"start_ns": 50, "end_ns": 10_000}):
        got = asyncio.run(read.records(None, vision_log.OBJECTS, item, root="/state", **window))  # ty: ignore[invalid-argument-type] - a chunk read needs no client
        assert got == [read.Record(PID, 150, 150, b"{}")]
    assert asked == [(100, 200), (120, 160), (100, 200)]
    # A window the item doesn't reach holds none of its records, and the file isn't opened.
    assert asyncio.run(read.records(None, vision_log.OBJECTS, item, root="/state", start_ns=200, end_ns=300)) == []  # ty: ignore[invalid-argument-type] - a chunk read needs no client
    assert len(asked) == 3


def test_video_days_asks_the_broker_for_one_frame_at_a_time() -> None:
    import datetime as dt

    today = dt.date.today()
    noon = [dt.datetime.combine(today - dt.timedelta(days=d), dt.time(12)).timestamp() for d in (2, 0)]
    frames = [vision_log.to_ns(noon[0] + k) for k in range(500)] + [vision_log.to_ns(noon[1])]
    handed: list[int] = []

    class Broker:
        """The query semantics of continuum_py: `all` (the default) pages through every item, ignoring max_items."""

        async def list_streams(self) -> list[dict]:
            return [{"name": "cam0.video"}, {"name": "cam0.caption"}]

        async def query(self, stream: str, lo: int, hi: int, *, all: bool = True, max_items: int = 0) -> dict:
            found = [f for f in frames if lo <= f < hi]
            found = found if all else found[: max_items or 1024]
            handed.append(len(found))
            header = lambda ns: continuum_py.encode_header(
                {"id": PID, "stream_id": 1, "start_ns": ns, "end_ns": ns, "layer": "Layer1Raw", "format": "EncodedVideo",
                 "shape_name": "camera.frame", "parents": [], "subjects": []}
            )  # fmt: skip
            return {"items": [{"header": header(f)} for f in found]}

    days = asyncio.run(vision_log.video_days(Broker()))  # ty: ignore[invalid-argument-type] - a stand-in for the client
    assert days == [(today - dt.timedelta(days=2)).isoformat(), today.isoformat()]
    assert max(handed) == 1  # never the whole day's frames for a yes or no


class _ThreadLink:
    """A broker link's shape: `call` runs on this link's one thread, as each real link runs on its own loop."""

    def __init__(self, socket: object = None, *, client_name: str, timeout: float = 10.0) -> None:
        import concurrent.futures

        self.name, self.timeout = client_name, timeout
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=client_name)
        self.closed = False

    def call(self, fn, timeout: float | None = None):
        def run():
            got = fn(None)
            return (
                asyncio.run(got) if asyncio.iscoroutine(got) else got
            )  # awaited on this link's thread, as a loop does

        return self.pool.submit(run).result(timeout=timeout or 5)

    def close(self) -> None:
        self.closed = True
        self.pool.shutdown(wait=False)


def test_a_picture_decodes_apart_from_every_other_read(monkeypatch: pytest.MonkeyPatch) -> None:
    # After a restart, face shots decoded H.264 on the shared link's thread and plain reads timed out behind
    # them. A picture read must answer while that link is busy, and a read of the main link while a picture decodes.
    import threading

    monkeypatch.setattr(vision_log, "BrokerLink", _ThreadLink)
    release = threading.Event()
    log = vision_log.VisionLog(client_name="vision-log")

    async def stuck(_client):
        return release.wait(5)

    busy = threading.Thread(target=lambda: log.link.call(stuck))
    busy.start()  # the main link is stuck on a long read

    async def evidence(_client, cam, ts):
        return (cam, ts)

    async def frame(_client, cam, ts, _within):
        return (cam, ts)

    monkeypatch.setattr(vision_log, "frame_evidence_at", evidence)
    monkeypatch.setattr(vision_log, "frame_at", frame)
    try:
        assert log.frame_evidence("cam0", 12.5) == ("cam0", 12.5)
        assert log.frame("cam0", 13.0) == ("cam0", 13.0)
    finally:
        release.set()
        busy.join()
    frames = log.frames_link()
    assert frames is log.frames_link(), "one picture link, made once"
    assert cast(_ThreadLink, frames).name == "vision-log-frames"
    log.close()
    assert cast(_ThreadLink, frames).closed and cast(_ThreadLink, log.link).closed


def test_one_scene_read_serves_every_caller_for_a_few_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each /scene poll read the whole hour of captions (0.4 to 1.5 s holding the GIL); several pages polling
    # stalled every other request. One read now serves all callers for SCENE_TTL_S.
    clock = [1000.0]
    monkeypatch.setattr(vision_log.time, "monotonic", lambda: clock[0])
    reads: list[float] = []

    def stored(start: float, end: float) -> list[dict[str, Any]]:
        reads.append(start)
        return [{"id": f"c{len(reads)}", "cam": "cam0", "ts_start": end - 2, "caption": "x"}]

    log = vision_log.VisionLog(client_name="scene-test")
    monkeypatch.setattr(log, "_captions", stored)
    first = log.scene(5000.0)
    clock[0] += vision_log.VisionLog.SCENE_TTL_S - 0.1
    assert log.scene(5004.0) == first and len(reads) == 1, "within a few seconds: the same read"
    clock[0] += 0.2
    assert log.scene(5005.0)[0]["id"] == "c2" and len(reads) == 2, "then a fresh one"
    log.close()


def test_callers_during_a_scene_read_wait_for_it_rather_than_read_again(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    started, release = threading.Event(), threading.Event()
    reads: list[float] = []

    def slow(start: float, end: float) -> list[dict[str, Any]]:
        reads.append(start)
        started.set()
        release.wait(5)
        return [{"id": "c1", "cam": "cam0", "ts_start": end - 2, "caption": "x"}]

    log = vision_log.VisionLog(client_name="scene-test")
    monkeypatch.setattr(log, "_captions", slow)
    results: list[list[dict[str, Any]]] = []
    callers = [threading.Thread(target=lambda: results.append(log.scene(5000.0))) for _ in range(8)]
    for caller in callers:
        caller.start()
    started.wait(5)
    release.set()
    for caller in callers:
        caller.join(5)
    assert len(reads) == 1 and len(results) == 8 and all(r[0]["id"] == "c1" for r in results)
    log.close()


def test_captions_for_some_cameras_read_only_those_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[Any] = []

    def stored(start: float, end: float, cams: list[str] | None = None) -> list[dict[str, Any]]:
        asked.append(cams)
        return [
            {"id": "c1", "cam": "cam0", "ts_start": start + 1, "caption": "x"},
            {"id": "c2", "cam": "cam1", "ts_start": start + 2, "caption": "y"},
        ]

    log = vision_log.VisionLog(client_name="captions-test")
    monkeypatch.setattr(log, "_captions", stored)
    assert [r["id"] for r in log.captions(100.0, 200.0, ("cam0",))] == ["c1"]
    assert [r["id"] for r in log.captions(100.0, 200.0)] == ["c1", "c2"]
    assert asked == [["cam0"], None], "one camera's read, then every camera's"
    log.close()


FRAME_NS = 1_800_000_000_000_000_000
CHUNK_ID = "0190c8e0-0000-7000-8000-0000000000c1"


def chunk_at(length: int, frames: list[tuple[int, bool]], uri: str = "streams/7/2027010100/7-1-0.cnk") -> dict:
    """A `get_chunk` snapshot of one chunk file, `length` bytes long so far."""
    place = {"kind": "Chunk", "tier": "Tier1Warm", "uri": uri, "offset": 0, "length": length}
    return {"chunk_id": CHUNK_ID, "seq": len(frames), "frames": frames, "locations": [place]}


class _GrowingChunk:
    """A broker whose one MJPEG chunk answers `get_chunk` with each of `snapshots` in turn, as a chunk does while
    the tier worker adds a part to it."""

    def __init__(self, queried: dict, snapshots: list[dict]) -> None:
        capture = {
            "capture_mono_ns": 1,
            "device_ts": 1,
            "arrival_ns": 1,
            "stride": 0,
            "width": 8,
            "height": 6,
            "flip": False,
            "device_seq": 1,
        }
        header = continuum_py.encode_header(
            {
                "id": CHUNK_ID,
                "stream_id": 7,
                "start_ns": FRAME_NS,
                "end_ns": FRAME_NS + 1,
                "layer": "Layer1Raw",
                "format": "EncodedVideo",
                "shape_name": "MJPEG",
                "capture": cast(Any, capture),
            }
        )
        self.item = {"header": header, "locator": queried["locations"][0]}
        self.snapshots = collections.deque(snapshots)

    async def query(self, stream: str, start_ns: int, end_ns: int, **_: Any) -> dict:
        return {"items": [self.item], "truncated": False}

    async def get_chunk(self, chunk_id: str) -> dict:
        return self.snapshots.popleft()


def frame_evidence(monkeypatch: pytest.MonkeyPatch, queried: dict, snapshots: list[dict]) -> Any:
    """What `frame_evidence_at` finds for the frame at FRAME_NS, read from a chunk the query saw as `queried` and
    the two `get_chunk` calls see as `snapshots`."""
    import cv2
    import numpy as np

    ok, jpeg = cv2.imencode(".jpg", np.full((6, 8, 3), 200, np.uint8))
    assert ok

    async def records(
        client: Any, stream: str, item: Any, *, root: Any, start_ns: int, end_ns: int
    ) -> list[read.Record]:
        return [read.Record(PID, FRAME_NS, FRAME_NS + 1, jpeg.tobytes())]

    monkeypatch.setattr(read, "records", records)
    client = cast(continuum_py.Client, _GrowingChunk(queried, snapshots))
    return asyncio.run(vision_log.frame_evidence_at(client, "cam0", FRAME_NS / 1e9))


ONE_PART = chunk_at(4_000, [(FRAME_NS, False)])
TWO_PARTS = chunk_at(9_000, [(FRAME_NS, False), (FRAME_NS + 100_000_000, False)])


@pytest.mark.parametrize(
    "queried, snapshots",
    [
        pytest.param(ONE_PART, [ONE_PART, ONE_PART], id="unchanged"),
        pytest.param(ONE_PART, [TWO_PARTS, TWO_PARTS], id="a-part-lands-after-the-query"),
        pytest.param(ONE_PART, [ONE_PART, TWO_PARTS], id="a-part-lands-during-the-read"),
    ],
)
def test_frame_evidence_reads_a_chunk_that_only_grew(
    monkeypatch: pytest.MonkeyPatch, queried: dict, snapshots: list[dict]
) -> None:
    found = frame_evidence(monkeypatch, queried, snapshots)
    assert found is not None
    assert (found.captured_at_ns, found.image.shape, found.metadata["capturedAtUs"]) == (
        FRAME_NS,
        (6, 8, 3),
        FRAME_NS // 1000,
    )


@pytest.mark.parametrize(
    "queried, snapshots",
    [
        pytest.param(ONE_PART, [ONE_PART, chunk_at(4_000, [(FRAME_NS, True)])], id="the-frame-is-cut-during-the-read"),
        pytest.param(
            TWO_PARTS,
            [TWO_PARTS, chunk_at(9_000, [(FRAME_NS, False), (FRAME_NS + 100_000_000, True)])],
            id="another-frame-is-cut-during-the-read",
        ),
        pytest.param(
            ONE_PART,
            [ONE_PART, chunk_at(4_000, [(FRAME_NS, False)], uri="streams/7/2027010100/7-2-0.cnk")],
            id="rewritten-to-another-file-during-the-read",
        ),
        pytest.param(
            chunk_at(4_000, [(FRAME_NS, False)], uri="streams/7/2027010100/7-0-0.cnk"),
            [ONE_PART, ONE_PART],
            id="the-query-named-a-file-the-chunk-has-left",
        ),
        pytest.param(TWO_PARTS, [TWO_PARTS, ONE_PART], id="frames-are-gone-after-the-read"),
    ],
)
def test_frame_evidence_refuses_a_chunk_that_changed_under_the_read(
    monkeypatch: pytest.MonkeyPatch, queried: dict, snapshots: list[dict]
) -> None:
    assert frame_evidence(monkeypatch, queried, snapshots) is None
