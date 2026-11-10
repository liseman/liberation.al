"""`pose.boxes`, see-actions' live people for the Live overlay (see/pose_boxes.py), and the pipeline that feeds it."""

import asyncio

import numpy as np

from liberational.see import actions, pose_boxes
from test_see_actions import Decoded, Rows, Script, frame_id, person

T = 500_000 * 3_600_000_000_000


def joints(x: float) -> np.ndarray:
    out = np.full((actions.JOINTS, 3), (-1.0, -1.0, 0.0))
    out[0] = (x, 0.25, 0.9)
    return out


def test_a_record_holds_each_persons_track_box_and_joints_and_reads_back():
    rec = pose_boxes.record("cam0", 7, [(12, np.array([0.1, 0.2, 0.3, 0.9, 0.85], np.float32), joints(0.123456))])
    person_ = rec["people"][0]
    assert (rec["cam"], rec["start_ns"], person_["track_id"]) == ("cam0", 7, "12")
    assert person_["box"] == [0.1, 0.2, 0.3, 0.9, 0.85]
    assert person_["keypoints"][0] == [0.1235, 0.25, 0.9] and person_["keypoints"][1] == [-1.0, -1.0, 0.0]
    data = pose_boxes.encode(rec).ljust(pose_boxes.STRIDE, b"\0")
    assert pose_boxes.decode(data) == rec


def test_a_frame_whose_people_dont_fit_keeps_its_largest():
    people = [(i, np.array([0.0, 0.0, 0.01 * (i + 1), 0.5, 0.9]), joints(0.5)) for i in range(30)]
    data = pose_boxes.encode(pose_boxes.record("cam0", 1, people))
    assert len(data) <= pose_boxes.STRIDE
    kept = pose_boxes.decode(data)
    assert kept is not None
    tracks = [p["track_id"] for p in kept["people"]]
    assert 0 < len(tracks) < 30 and tracks[0] == "29" and "0" not in tracks


def test_the_pipeline_sends_each_tracked_frame_live_as_its_rows_say():
    sent: list[tuple[int, list]] = []
    poses = Rows()
    pipeline = actions.ActionPipeline(
        Script([person(0.1) for _ in range(10)]), poses, Rows(), "m", live=lambda ns, p: sent.append((ns, p))
    )
    for i in range(10):
        pipeline.feed(Decoded(frame_id(i), T + i * 40_000_000))
    pipeline.close()
    # The track counts from its 5th frame, as the pose rows do; the joints are the rows' own, normalized.
    rows = [int(t) for a in poses.appends for t in a["ts_ns"]]
    assert [ns for ns, _ in sent] == rows == [T + i * 40_000_000 for i in range(4, 10)]
    track, box, live_joints = sent[0][1][0]
    assert track == 1
    assert box.tolist() == poses.appends[0]["box"][0].tolist()
    np.testing.assert_allclose(live_joints[0], (0.25, 0.25, 0.9))  # joint (50, 25) in a 200 x 100 frame


def test_the_ring_reaches_a_reader_without_a_catalog_row(broker):
    import continuum_py

    rec = pose_boxes.record("cam0", 5_000, [(1, np.array([0.1, 0.2, 0.3, 0.9, 0.8]), joints(0.2))])

    async def run() -> tuple[dict | None, int]:
        async with await continuum_py.Client.connect(str(broker), client_name="pose-boxes") as client:
            writer = await pose_boxes.writer(client)
            writer.push(pose_boxes.encode(rec))
            sub = await client.subscribe([pose_boxes.STREAM], start="Live")
            try:
                reader = client.ring_reader(sub.rings[0]["lease"], from_oldest=True)
                got = reader.read_record()
            finally:
                await sub.unsubscribe()
            rows = len((await client.query(pose_boxes.STREAM, 0, 10**12))["items"])
            return (pose_boxes.decode(got[1]) if got else None), rows

    got, rows = asyncio.run(run())
    assert got == rec
    assert rows == 0, "the ring sends no ingest: nothing reaches the catalog"


def test_a_frame_whose_pose_isnt_confident_sends_nothing_live_but_is_still_a_row():
    sent: list[int] = []
    poses = Rows()
    faded = person(0.1, joint=(50.0, 25.0, 0.3))
    pipeline = actions.ActionPipeline(
        Script([person(0.1)] * 6 + [faded] * 4), poses, Rows(), "m", live=lambda ns, p: sent.append(ns)
    )
    for i in range(10):
        pipeline.feed(Decoded(frame_id(i), T + i * 40_000_000))
    pipeline.close()
    assert sent == [T + 4 * 40_000_000, T + 5 * 40_000_000]  # counted from frame 4; frames 6 to 9 are unsure
    assert sum(len(a["ts_ns"]) for a in poses.appends) == 6
