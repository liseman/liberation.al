"""Face thumbnails on disk: written under the deletion revision, pruned by age and size."""

import os
import threading
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from liberational.people import face_shots

NOW = 1_800_000_000.0


def crop_at(state, sid, age_days, size=1000):
    p = face_shots.write(state, sid, "0", b"x" * size)
    os.utime(p, (NOW - age_days * 86400, NOW - age_days * 86400))
    return p


def test_prune_drops_crops_past_the_video_window_and_keeps_the_rest(tmp_path):
    old = crop_at(tmp_path, "old", face_shots.KEEP_DAYS + 1)
    kept = crop_at(tmp_path, "kept", face_shots.KEEP_DAYS - 1)
    assert face_shots.prune(tmp_path, NOW) == 1
    assert (old.exists(), kept.exists()) == (False, True)


def test_prune_drops_the_oldest_until_under_the_cap(tmp_path):
    crops = [crop_at(tmp_path, f"s{k}", age_days=5 - k) for k in range(5)]  # 5000 bytes, oldest first
    assert face_shots.prune(tmp_path, NOW, cap_bytes=3000) == 3  # under 80% of 3000 leaves two
    assert [p.exists() for p in crops] == [False, False, False, True, True]


def test_the_writer_saves_a_square_jpeg_under_the_current_revision(tmp_path):
    face_shots.Writer(tmp_path)("abc", np.full((90, 90, 3), 200, np.uint8))
    (saved,) = face_shots.folder(tmp_path).glob("*.jpg")
    assert saved == face_shots.path(tmp_path, "abc", "0")
    img = cv2.imread(str(saved))
    assert img is not None
    assert img.shape == (face_shots.PX, face_shots.PX, 3)


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        pytest.param((40, 40, 60, 60), (35, 35, 30), id="middle"),
        pytest.param((0, 0, 20, 20), (0, 0, 30), id="kept-inside-the-corner"),
        pytest.param((0, 0, 4, 4), None, id="too-small"),
    ],
)
def test_square_is_half_again_the_face_and_stays_in_the_frame(box, expected):
    assert face_shots.square(100, 80, box) == expected


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        pytest.param([0.9, 0.3, 1.03, 0.6], [0.9, 0.3, 1.0, 0.6], id="past-the-right-edge"),
        pytest.param([-0.1, 0.2, 0.3, 0.5], [0.0, 0.2, 0.3, 0.5], id="past-the-left-edge"),
        pytest.param([1.1, 0.2, 1.3, 0.5], None, id="all-outside"),
    ],
)
def test_in_frame_clips_a_face_box_to_the_frame(box, expected):
    assert face_shots.in_frame(box) == (pytest.approx(expected) if expected else None)


def test_background_jobs_deduplicate_until_a_saved_crop_is_visible(tmp_path, monkeypatch):
    calls = []
    clock = [10.0]
    monkeypatch.setattr(face_shots.time, "monotonic", lambda: clock[0])

    def process(key):
        calls.append(key)
        state, kind, sid, revision = key
        face_shots.write(state, sid, revision, b"jpeg", kind)
        return face_shots.CropResult(True)

    jobs = face_shots.Pending(process)
    assert face_shots.saved(tmp_path) == set()
    assert jobs.offer(tmp_path, "shots", "face-a", "0")
    assert jobs.offer(tmp_path, "shots", "face-a", "0")
    assert calls == []  # listing/admission does no decode
    assert jobs.one()
    assert not jobs.one()
    assert calls == [(tmp_path, "shots", "face-a", "0")]
    assert face_shots.saved(tmp_path) == {"face-a"}  # directory-change invalidation sees background writes
    face_shots.path(tmp_path, "face-a", "0").unlink()
    clock[0] = 11.0  # external changes are indexed within one second even if directory timestamps alias
    assert face_shots.saved(tmp_path) == set()


def test_saved_reads_each_kind_and_revision_from_one_directory_listing(tmp_path):
    shots = face_shots.folder(tmp_path)
    shots.mkdir()
    v, grid = face_shots.VERSION, f"{face_shots.VERSION}-grid{face_shots.GRID_PX}"
    for name in [
        f"a-{v}-r0.jpg",
        f"b-{v}-r1.jpg",
        f"a-{grid}-r0.jpg",
        f"c-{grid}-r1.jpg",
        f".d-{v}-r0.jpg.123.tmp",  # a write in progress
        "e-old-geometry-r0.jpg",
        f"f-{v}-r0.png",
    ]:
        (shots / name).write_bytes(b"x")
    assert face_shots.saved(tmp_path) == {"a", "b"}
    assert face_shots.saved(tmp_path, revision="0") == {"a"}
    assert face_shots.saved(tmp_path, "shots_grid") == {"a", "c"}
    assert face_shots.saved(tmp_path, "shots_grid", "1") == {"c"}
    assert face_shots.saved(tmp_path, "pet_shots") == set()


def test_saved_sees_an_external_write_to_a_settled_directory(tmp_path, monkeypatch):
    shots = face_shots.folder(tmp_path)
    shots.mkdir()
    (shots / f"a-{face_shots.VERSION}-r0.jpg").write_bytes(b"x")
    later = face_shots.time.time_ns() + 60 * 10**9
    monkeypatch.setattr(face_shots.time, "time_ns", lambda: later)  # every change is a minute old: no per-second reads
    assert face_shots.saved(tmp_path, revision="0") == {"a"}
    (shots / f"b-{face_shots.VERSION}-r0.jpg").write_bytes(b"x")  # another process: no cache clear here
    assert face_shots.saved(tmp_path, revision="0") == {"a", "b"}


def test_unavailable_crops_leave_processing_until_new_revision(tmp_path):
    jobs = face_shots.Pending(lambda key: face_shots.CropResult(False))
    assert jobs.offer(tmp_path, "shots", "gone", "0")
    assert jobs.one()
    assert not jobs.offer(tmp_path, "shots", "gone", "0")
    assert jobs.offer(tmp_path, "shots", "gone", "1")


def test_decode_exceptions_are_unavailable_and_never_spin_in_the_queue(tmp_path):
    def broken(key):
        raise OSError("recording cannot be decoded")

    jobs = face_shots.Pending(broken)
    jobs.offer(tmp_path, "shots", "bad", "0")
    assert jobs.one()
    assert not jobs.one()
    assert not jobs.offer(tmp_path, "shots", "bad", "0")


def test_frame_rpc_timeout_stays_pending_then_retries_without_an_error_trace(tmp_path, monkeypatch, vision, caplog):
    from liberational.dashboard import server

    now = [10.0]
    monkeypatch.setattr(face_shots.time, "monotonic", lambda: now[0])
    row = vision.face(NOW, "cam0", "Alice", 0.8)
    vision.frames[("cam0", NOW)] = np.full((120, 160, 3), 200, np.uint8)
    evidence = vision.frame_evidence("cam0", NOW)
    jobs = face_shots.Pending(server._crop_process)
    with patch.object(vision, "frame_evidence", autospec=True, side_effect=[TimeoutError(), evidence]) as read:
        assert jobs.offer(tmp_path, "shots", row["id"], "0")
        assert jobs.one()
        assert not jobs.failed
        assert jobs.offer(tmp_path, "shots", row["id"], "0")
        assert not jobs.one()  # a failed RPC must not run again until its backoff expires
        assert read.call_count == 1
        now[0] += 60
        assert jobs.one()
    assert face_shots.path(tmp_path, row["id"], "0").is_file()
    assert not jobs.pending
    assert not [record for record in caplog.records if record.levelno >= 40]


def test_backoff_does_not_block_other_thumbnail_jobs_and_grows_after_another_timeout(tmp_path, monkeypatch):
    now, calls = [10.0], []
    monkeypatch.setattr(face_shots.time, "monotonic", lambda: now[0])

    def process(key):
        calls.append(key[2])
        return face_shots.CropRetry() if key[2] == "slow" else face_shots.CropResult(True)

    jobs = face_shots.Pending(process)
    jobs.offer(tmp_path, "shots", "slow", "0")
    jobs.offer(tmp_path, "shots", "ready", "0")
    assert jobs.one()
    assert jobs.one()  # the slow job's backoff does not starve the ready one
    assert calls == ["slow", "ready"]
    assert not jobs.one()
    now[0] = 11
    assert jobs.one()
    now[0] = 12
    assert not jobs.one()  # second timeout waits two seconds, not one
    now[0] = 13
    assert jobs.one()
    assert calls == ["slow", "ready", "slow", "slow"]


def test_queue_overflow_and_contention_remain_pending_for_later_admission(tmp_path):
    jobs = face_shots.Pending(lambda key: None)
    jobs.LIMIT = 1
    assert jobs.offer(tmp_path, "shots", "a", "0")
    assert jobs.offer(tmp_path, "shots", "b", "0")
    assert len(jobs.pending) == 1
    assert jobs.one()
    assert len(jobs.pending) == 1
    assert not jobs.failed  # contention must not mark a crop gone
    jobs.process = lambda key: face_shots.CropResult(True)
    assert jobs.one()
    assert jobs.offer(tmp_path, "shots", "b", "0")
    assert jobs.one()
    assert not jobs.pending


def test_a_flood_of_face_crops_neither_keeps_out_nor_delays_a_pet_photo(tmp_path):
    # After every face crop was cleared, thousands of face jobs filled the one queue, so a pet's grid photo was
    # reported pending without being admitted and stayed "processing" for hours.
    done = []
    jobs = face_shots.Pending(lambda key: done.append((key[1], key[2])) or face_shots.CropResult(True))
    jobs.LIMIT = 3
    for sid in "abcd":
        assert jobs.offer(tmp_path, "shots", sid, "0")
    assert jobs.offer(tmp_path, "pet_shots_grid", "cat:3", "0")
    assert jobs.one()
    assert jobs.one()
    assert done == [("shots", "a"), ("pet_shots_grid", "cat:3")]
    assert jobs.offer(tmp_path, "shots", "d", "0")  # the face that overflowed gets in once there's room
    while jobs.one():
        pass
    assert done[2:] == [("shots", "b"), ("shots", "c"), ("shots", "d")]
    assert not jobs.pending


def test_background_worker_shares_the_existing_decode_cap(tmp_path, monkeypatch):
    from liberational.dashboard import server

    monkeypatch.setattr(server, "SHOT_WAIT_S", 0)
    assert server._shot_decode.acquire(blocking=False)
    assert server._shot_decode.acquire(blocking=False)
    try:
        monkeypatch.setattr(server.vision_log, "at", lambda state: None)
        from liberational.vision import sightings

        monkeypatch.setattr(sightings, "shot", lambda sid, state: {"cam": "cam0", "ts": 1, "bbox_norm": [0, 0, 1, 1]})
        assert server._crop_process((tmp_path, "shots", "face-a", "0")) is None
    finally:
        server._shot_decode.release()
        server._shot_decode.release()


def test_revision_revalidation_needs_no_decode_slot_and_marks_the_crop_ready(tmp_path, monkeypatch, vision):
    from liberational.dashboard import server

    row = vision.face(1_800_000_000, "cam0", "Alice", 0.8)
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "_shot_revision", lambda state=None: "1")
    monkeypatch.setattr(server, "SHOT_WAIT_S", 0)
    monkeypatch.setattr(server, "_crop_jobs", face_shots.Pending(server._crop_process))
    face_shots.write(tmp_path, row["id"], "0", b"jpeg")
    assert face_shots.saved(tmp_path, revision="1") == set()
    assert server._ready_faces([row]) == ([], 1, 0)
    assert server._shot_decode.acquire(blocking=False)
    assert server._shot_decode.acquire(blocking=False)
    try:
        result = server._crop_process((tmp_path, "shots", row["id"], "1"))
    finally:
        server._shot_decode.release()
        server._shot_decode.release()
    assert result == face_shots.CropResult(True, False)
    assert server._ready_faces([row]) == ([row], 0, 0)


def test_background_worker_starts_once_at_low_priority_and_stops(tmp_path):
    priority, done = threading.Event(), threading.Event()

    def process(key):
        assert priority.is_set()
        done.set()
        return face_shots.CropResult(True)

    jobs = face_shots.Pending(process)
    jobs.offer(tmp_path, "shots", "a", "0")
    jobs.start(priority.set)
    first = jobs.thread
    try:
        jobs.start(priority.set)
        assert jobs.thread is first
        assert done.wait(5)
    finally:
        jobs.stop()
        assert first is not None
        first.join(5)
    assert not first.is_alive()
    done.clear()
    jobs.offer(tmp_path, "shots", "b", "0")
    jobs.start(priority.set)
    try:
        assert done.wait(5)
    finally:
        jobs.stop()
        assert jobs.thread is not None
        jobs.thread.join(5)
    assert not jobs.thread.is_alive()


def test_group_ready_counts_refresh_without_rebuilding_the_identity_snapshot(tmp_path, monkeypatch):
    from liberational.dashboard import server

    monkeypatch.setattr(server, "STATE", tmp_path)
    jobs = face_shots.Pending(lambda key: face_shots.CropResult(True))
    monkeypatch.setattr(server, "_crop_jobs", jobs)
    shots = [{"id": sid, "ts": t, "url": f"/faces/shot/{sid}.jpg"} for t, sid in enumerate("abcdefgh")]
    face_shots.write(tmp_path, "h", "0", b"jpeg")
    face_shots.write(tmp_path, "h", "0", b"jpeg", "shots_grid")  # outside the old six-shot preview, but ready
    body = {"groups": [{"id": 1, "shots": shots[:6], "_all_shots": shots}], "count": 1}
    code, view = server._ready_face_groups(200, body)
    assert code == 200
    assert [r["id"] for r in view["groups"][0]["shots"]] == ["h"]
    assert view["groups"][0]["shots"][0]["url"] == "/faces/shot/h.jpg?size=80"
    assert view["groups"][0]["shot_count"] == 1
    assert view["groups"][0]["shots_pending"] == view["shots_pending"] == 7
    assert "_all_shots" not in view["groups"][0]
    face_shots.write(tmp_path, "a", "0", b"jpeg")
    face_shots.write(tmp_path, "a", "0", b"jpeg", "shots_grid")
    after = server._ready_face_groups(200, body)[1]
    assert [r["id"] for r in after["groups"][0]["shots"]] == ["a", "h"]
    assert after["shots_pending"] == 6
    jobs.process = lambda key: face_shots.CropResult(False)
    assert jobs.one()
    failed = server._ready_face_groups(200, body)[1]
    assert failed["shots_unavailable"] == 0  # this job's file appeared externally before the queued attempt
    assert jobs.one()
    failed = server._ready_face_groups(200, body)[1]
    assert failed["shots_unavailable"] == failed["groups"][0]["shots_unavailable"] == 1
    assert len(body["groups"][0]["shots"]) == 6  # cached input remains unchanged


def test_pet_photo_readiness_preserves_video_metadata_and_counts_each_track_once(tmp_path, monkeypatch):
    from liberational.dashboard import server
    from liberational.people import pet_shots

    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "_crop_jobs", face_shots.Pending(lambda key: face_shots.CropResult(True)))
    face_shots.write(tmp_path, "a", "0", b"jpeg", pet_shots.KIND)
    face_shots.write(tmp_path, "a", "0", b"jpeg", pet_shots.KIND + "_grid")
    a = {"track_id": "a", "cam": "cam0", "start": 10, "end": 12}
    b = {"track_id": "b", "cam": "cam0", "start": 20, "end": 22}
    body = {"groups": [{"id": "cats", "sightings": [a, b]}], "seen": {"pet-a": [b]}, "names": {}}
    code, view = server._ready_pet_view(200, body)
    assert code == 200
    assert view["groups"][0]["sightings"] == [
        {**a, "photo_ready": True, "thumb_ready": True},
        {**b, "photo_ready": False, "thumb_ready": False},
    ]
    assert view["seen"]["pet-a"] == [
        {**b, "photo_ready": False, "thumb_ready": False, "video_status": "pending", "video_thumb": None}
    ]
    assert view["photos_pending"] == view["groups"][0]["photos_pending"] == 1
    assert view["videos_pending"] == 1
    assert "video_status" not in b  # the cached metadata snapshot remains unchanged


def test_pet_video_preparation_is_independent_of_photo_availability(tmp_path, monkeypatch, vision):
    from liberational.dashboard import server
    from liberational.people import pet_shots

    monkeypatch.setattr(server, "STATE", tmp_path)
    jobs = face_shots.Pending(server._crop_process)
    monkeypatch.setattr(server, "_crop_jobs", jobs)
    face_shots.write(tmp_path, "pet-look", "0", b"pet-photo", pet_shots.KIND)
    face_shots.write(tmp_path, "pet-look", "0", b"pet-photo", pet_shots.KIND + "_grid")
    vision.frames[("cam0", 20)] = np.full((120, 160, 3), 150, np.uint8)
    row = {"track_id": "pet-look", "cam": "cam0", "start": 20, "end": 22, "how": "labeled"}
    body = {"seen": {"pet-a": [row]}, "groups": []}
    first = server._ready_pet_view(200, body)[1]
    assert first["seen"]["pet-a"][0] == {
        **row,
        "photo_ready": True,
        "thumb_ready": True,
        "video_status": "pending",
        "video_thumb": None,
    }
    assert first["photos_pending"] == 0 and first["videos_pending"] == 1
    assert jobs.one()
    after = server._ready_pet_view(200, body)[1]
    assert after["videos_pending"] == after["videos_unavailable"] == 0
    assert after["seen"]["pet-a"][0]["video_status"] == "ready"
    assert after["seen"]["pet-a"][0]["video_thumb"] == "/people/video-thumb.jpg?cam=cam0&start=20.000&end=25.000"


@pytest.mark.parametrize("kind,pixels", [("shots", 160), ("pet_shots", 192)])
def test_grid_variant_uses_saved_jpeg_and_keeps_detail_bytes(tmp_path, monkeypatch, kind, pixels):
    from liberational.dashboard import server

    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "_shot_revision", lambda state=None: "0")
    source = np.random.default_rng(1).integers(0, 256, (pixels, pixels, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", source)
    assert ok
    original = face_shots.write(tmp_path, "saved", "0", encoded.tobytes(), kind)
    result = server._crop_process((tmp_path, kind + "_grid", "saved", "0"))
    grid = face_shots.path(tmp_path, "saved", "0", kind + "_grid")
    assert result == face_shots.CropResult(True, False)
    image = cv2.imread(str(grid))
    assert image is not None and image.shape == (80, 80, 3)
    assert grid.stat().st_size < original.stat().st_size
    assert original.read_bytes() == encoded.tobytes()
    assert face_shots.others(tmp_path, "saved", "1", kind) == [original]
    assert face_shots.saved(tmp_path, kind, "0") == {"saved"}
    assert face_shots.saved(tmp_path, kind + "_grid", "0") == {"saved"}
    face_shots.write(tmp_path, "saved", "0", encoded.tobytes(), kind)
    assert not grid.exists()  # a newer pet/face crop invalidates its small derivative


@pytest.mark.parametrize(
    "kind,url", [("shots", "/faces/shot/saved.jpg?size=80"), ("pet_shots", "/pets/sightings/photo?track=saved&size=80")]
)
def test_grid_http_waits_for_background_variant_and_never_promotes_old_revision(
    tmp_path, monkeypatch, serve, kind, url
):
    import http.client

    from liberational.dashboard import server

    revision = ["0"]
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "_shot_revision", lambda state=None: revision[0])
    jobs = face_shots.Pending(server._crop_process)
    monkeypatch.setattr(server, "_crop_jobs", jobs)
    jpg = face_shots.encode(np.full((160, 160, 3), 123, np.uint8))
    assert jpg is not None
    face_shots.write(tmp_path, "saved", "0", jpg, kind)
    connection = http.client.HTTPConnection(serve(server.H).removeprefix("http://"))
    connection.request("GET", url)
    response = connection.getresponse()
    assert response.status == 202
    assert response.getheader("Cache-Control") == "no-store"
    assert response.read() == b""
    assert jobs.one()
    connection.request("GET", url)
    response = connection.getresponse()
    assert response.status == 200
    image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape == (80, 80, 3)
    etag = response.getheader("ETag")
    assert etag is not None
    assert response.getheader("Cache-Control") == "private, max-age=86400"
    connection.request("GET", url, headers={"If-None-Match": f'"other", W/{etag}'})
    response = connection.getresponse()
    assert response.status == 304
    assert response.read() == b""
    revision[0] = "1"
    connection.request("GET", url)
    response = connection.getresponse()
    assert response.status == 202  # old files cannot be served before source revalidation
    response.read()
    connection.close()


def test_original_detail_counts_and_pet_main_photo_stay_ready_while_grid_variant_is_pending(tmp_path, monkeypatch):
    from liberational.dashboard import server

    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "_crop_jobs", face_shots.Pending(server._crop_process))
    face_shots.write(tmp_path, "face-a", "0", b"jpeg")
    face_shots.write(tmp_path, "face-b", "0", b"jpeg")
    face_shots.write(tmp_path, "face-a", "0", b"jpeg", "shots_grid")
    shots = [{"id": sid, "url": f"/faces/shot/{sid}.jpg", "ts": 1} for sid in ["face-a", "face-b"]]
    body = {"groups": [{"id": 1, "shots": shots}], "count": 1}
    result = server._ready_face_groups(200, body)[1]
    assert len(result["groups"][0]["shots"]) == 1
    assert result["groups"][0]["shot_count"] == 2
    assert result["shots_pending"] == 1
    face_shots.write(tmp_path, "pet-a", "0", b"jpeg", "pet_shots")
    result = server._ready_pet_view(
        200, {"seen": {"pet": [{"track_id": "pet-a", "cam": "cam0", "start": 1, "end": 2}]}}
    )[1]
    assert result["seen"]["pet"][0]["photo_ready"] is True
    assert result["seen"]["pet"][0]["thumb_ready"] is False
    assert result["seen_pending"] == {"pet": 0}
    assert result["photos_pending"] == 1
