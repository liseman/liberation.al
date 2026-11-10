from liberational.see import face_boxes


def face(x: float, size: float) -> dict:
    return {"bbox": [x, 0.0, x + size, size], "name": None, "sim": None, "cluster": "Face 1", "track_id": "t"}


def test_a_record_reads_back_as_written() -> None:
    record = {"cam": "cam0", "frame_seq": 3, "frame_id": "f", "start_ns": 7, "faces": [face(0.1, 0.2)]}
    data = face_boxes.encode(record).ljust(face_boxes.STRIDE, b"\0")
    assert face_boxes.decode(data) == record


def test_a_frame_whose_faces_dont_fit_keeps_its_largest_boxes() -> None:
    faces = [face(i / 100, 0.001 * (i + 1)) for i in range(40)]  # each a bit larger than the last
    record = {"cam": "cam0", "frame_seq": 1, "frame_id": "f", "start_ns": 1, "faces": faces}
    data = face_boxes.encode(record)
    assert len(data) <= face_boxes.STRIDE
    kept = face_boxes.decode(data)
    assert kept is not None
    assert 0 < len(kept["faces"]) < 40
    assert kept["faces"][0] == faces[-1], "the largest box first"
    assert faces[0] not in kept["faces"], "the smallest box went"


def test_an_empty_torn_or_foreign_slot_is_none() -> None:
    assert face_boxes.decode(b"\0" * face_boxes.STRIDE) is None, "never written"
    assert face_boxes.decode(b"\x10\0\0\0{not json at al") is None
    assert face_boxes.decode(b"\xff\xff\0\0{}") is None, "a length past the record"
    assert face_boxes.decode(b"\x02\0\0\0[]") is None, "not an object"


def test_the_ring_reaches_its_live_follower_and_a_lookup_by_time_without_a_catalog_row(broker) -> None:
    import asyncio

    import continuum_py

    record = {"cam": "cam0", "frame_seq": 1, "frame_id": "f1", "start_ns": 5_000, "faces": [face(0.1, 0.2)]}

    async def run() -> tuple[dict, list[dict], list[dict], int]:
        async with await continuum_py.Client.connect(str(broker), client_name="face-boxes") as client:
            writer = await face_boxes.writer(client)
            follower = face_boxes.follow(client)
            first = asyncio.ensure_future(anext(follower))
            await asyncio.sleep(0.2)  # the follower has subscribed and opened its reader
            writer.push(face_boxes.encode(record))
            got = await asyncio.wait_for(first, timeout=5)
            await follower.aclose()
            found = await face_boxes.between(client, 4_000, 6_000)
            outside = await face_boxes.between(client, 6_001, 9_000)
            rows = len((await client.query(face_boxes.STREAM, 0, 10**12))["items"])
            return got, found, outside, rows

    got, found, outside, rows = asyncio.run(run())
    assert got == record
    assert found == [record]
    assert outside == []
    assert rows == 0, "the ring sends no ingest: nothing reaches the catalog"
