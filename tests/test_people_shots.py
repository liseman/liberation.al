"""People's shot strip: every shot the face worker found of a person (sightings.shots) and the face crops cut from
the frames on the vision log (GET /people/<name>/shots, GET /faces/shot/<id>.jpg). Made-up names and synthetic
frames."""

import http.client
import time
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from liberational.dashboard import context, people_reads, photo_files, server
from liberational.people import face_shots, video_thumbs
from liberational.vision import sightings
from liberational.vision import sightings as st

EMPTY = {"total": 0, "shots": [], "next": None}
T = float(int(time.time()) - 3600)  # an hour ago, on a whole second: within the SHOT_DAYS the readers look back
BOX = [40 / 160, 30 / 120, 80 / 160, 70 / 120]  # the white square in the dashboard fixture's frame


@pytest.fixture
def state(tmp_path, vision):
    """Alice twice a second on cam0 for ten seconds, plus one sighting each that the strip treats specially. Each
    face is its own visit here, so each is a photo (a visit keeps its few best)."""
    for sec in range(10):  # cam0: two samples a second; the second of each pair matches better
        for half, sim in ((0.0, 0.6), (0.5, 0.7)):
            vision.face(T + sec + half, "cam0", "Alice", sim)
            vision.visit("Alice", T + sec + half, T + sec + half, "cam0")
    vision.face(T + 5.2, "cam1", "alice", 0.5)  # another camera, same second: kept, and names match any case
    vision.visit("alice", T + 5.2, T + 5.2, "cam1")
    vision.face(T + 6.0, "cam0", "Bob", 0.9)
    vision.visit("Bob", T + 6.0, T + 6.0, "cam0")
    return tmp_path


def test_one_shot_per_camera_per_second_newest_first_paged(state):
    a = st.shots("ALICE", limit=6, state=state)
    assert a["total"] == 11
    assert [round(s["ts"] - T, 1) for s in a["shots"]] == [9.5, 8.5, 7.5, 6.5, 5.5, 5.2]
    assert {s["sim"] for s in a["shots"][:5]} == {0.7}  # the better of each pair
    assert a["next"] == T + 5
    b = st.shots("alice", before=a["next"], limit=6, state=state)
    assert [round(s["ts"] - T, 1) for s in b["shots"]] == [4.5, 3.5, 2.5, 1.5, 0.5]
    assert b["next"] is None  # a short page is the last
    assert not {s["id"] for s in a["shots"]} & {s["id"] for s in b["shots"]}


def test_nobody_has_no_shots(state):
    assert st.shots("Carol", state=state) == EMPTY
    assert st.shot("no-such-id", state=state) is None


@pytest.fixture
def dashboard(tmp_path, monkeypatch, serve, vision):
    """The dashboard over a state dir with one 160 by 120 frame whose only "face" is a white square, a sighting of
    Alice in it, and one whose frame is gone."""
    monkeypatch.setattr(context, "STATE", tmp_path)
    monkeypatch.setattr(photo_files, "_crop_jobs", face_shots.Pending(photo_files._crop_process))
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[30:70, 40:80] = 255
    vision.frames[("cam0", T + 0.5)] = frame
    ids = SimpleNamespace(
        kept=vision.face(T + 0.5, "cam0", "Alice", 0.8, bbox=BOX)["id"],
        gone=vision.face(T + 1.5, "cam0", "Alice", 0.8, bbox=BOX)["id"],
    )
    vision.visit("Alice", T + 0.5, T + 1.5, "cam0")
    return serve(server.H), ids


def get(base, path, headers=None):
    """(status, headers, body) of GET path, sent as is (no client-side normalizing)."""
    c = http.client.HTTPConnection("127.0.0.1", urllib.parse.urlsplit(base).port, timeout=10)
    try:
        c.request("GET", path, headers=headers or {})
        r = c.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        c.close()


def test_a_shot_is_a_cached_crop_of_its_frame(dashboard, tmp_path):
    base, ids = dashboard
    code, headers, body = get(base, f"/faces/shot/{ids.kept}.jpg")
    assert code == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert "max-age=86400" in headers["Cache-Control"]
    assert "immutable" not in headers["Cache-Control"]
    assert headers["ETag"]
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    assert img.shape == (server.SHOT_PX, server.SHOT_PX)
    assert img[70:90, 70:90].mean() > 200  # the face sits in the middle
    assert img[:10, :10].mean() < 60  # with some padding around it
    assert (server._shot_path(ids.kept)).exists()
    assert get(base, f"/faces/shot/{ids.kept}.jpg")[2] == body


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("/faces/shot/99.jpg", id="no-such-sighting"),
        pytest.param("gone", id="frame-gone"),
        pytest.param("/faces/shot/..%2F..%2Fpack.toml", id="encoded-traversal"),
        pytest.param("/faces/shot/x.jpg/../../../pack.toml", id="path-traversal"),
        pytest.param("/faces/shot/a_b.jpg", id="not-an-id"),
    ],
)
def test_a_missing_or_unsafe_shot_is_404(dashboard, path):
    base, ids = dashboard
    assert get(base, f"/faces/shot/{ids.gone}.jpg" if path == "gone" else path)[0] == 404


@pytest.fixture
def unmappable(vision, monkeypatch):
    """Frames decoded at a size their native geometry doesn't match, with no profile saying how: a face's box can't
    be placed on them."""
    plain = vision.frame_evidence

    def evidence(cam, ts):
        found = plain(cam, ts)
        if found is not None:
            found.metadata["nativeGeometry"] = {"width": 1920, "height": 1080}
        return found

    monkeypatch.setattr(vision, "frame_evidence", evidence)


def test_a_shot_whose_frame_has_no_geometry_is_404_and_not_cached(dashboard, unmappable):
    base, ids = dashboard
    assert get(base, f"/faces/shot/{ids.kept}.jpg")[0] == 404
    assert not server._shot_path(ids.kept).exists()


def test_a_scene_photo_whose_frame_has_no_geometry_is_skipped(dashboard, unmappable, tmp_path):
    dest = tmp_path / "photo.jpg"
    assert people_reads._crop("cam0", T + 0.5, BOX, dest) is False
    assert not dest.exists()


def looked_up(*args, **kwargs):
    raise AssertionError("looked up")


def test_a_crop_written_under_the_current_revision_is_served_without_a_lookup(tmp_path, monkeypatch):
    # What see-actions leaves with each sighting: served as is, with no lookup and no decode.
    monkeypatch.setattr(context, "STATE", tmp_path)
    sid = "01a18c4d-4f51-740b-9b2e-d84e49271b7d"
    face_shots.write(tmp_path, sid, photo_files._shot_revision(), b"cut")
    monkeypatch.setattr(sightings, "shot", looked_up)
    path, _etag = photo_files._shot_file(sid)
    assert path is not None and path.read_bytes() == b"cut"


def test_a_crop_from_before_a_forget_is_checked_once_and_kept(dashboard, tmp_path, monkeypatch):
    base, ids = dashboard
    url = f"/faces/shot/{ids.kept}.jpg"
    _, _, pixels = get(base, url)
    monkeypatch.setattr(photo_files, "_shot_revision", lambda state=None: "9")  # a forget elsewhere moved the revision
    monkeypatch.setattr(photo_files.vision_log, "at", lambda state: looked_up())  # kept, not cut again
    assert get(base, url)[2] == pixels
    assert [p.name.rsplit("-r", 1)[1] for p in (tmp_path / "shots").rglob("*.jpg")] == ["9.jpg"]
    monkeypatch.setattr(sightings, "shot", looked_up)  # checked under this revision: no lookup after
    assert get(base, url)[2] == pixels


def test_purged_crop_returns_404_and_removes_cached_file_even_with_old_etag(dashboard, tmp_path, vision, monkeypatch):
    base, ids = dashboard
    path = server._shot_path(ids.kept)
    code, headers, _ = get(base, f"/faces/shot/{ids.kept}.jpg")
    assert code == 200
    assert path.exists()
    vision.records[photo_files.segment_log.FACE_MATCHES] = [
        row for row in vision.records[photo_files.segment_log.FACE_MATCHES] if row["id"] != ids.kept
    ]
    monkeypatch.setattr(
        photo_files, "_shot_revision", lambda state=None: "1"
    )  # the forget that removed it moved the revision
    assert get(base, f"/faces/shot/{ids.kept}.jpg", {"If-None-Match": headers["ETag"]})[0] == 404
    assert not list((tmp_path / "shots").rglob("*.jpg"))


def test_live_cached_crop_revalidates_with_revision_etag_without_reading_jpeg(dashboard, tmp_path, monkeypatch):
    base, ids = dashboard
    url = f"/faces/shot/{ids.kept}.jpg"
    code, headers, body = get(base, url)
    assert code == 200 and body
    cached = server._shot_path(ids.kept)
    original = Path.read_bytes

    def read(path):
        assert path != cached, "304 must not read the cached JPEG"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    connection = http.client.HTTPConnection(urllib.parse.urlsplit(base).netloc, timeout=5)
    try:
        connection.request("GET", url, headers={"If-None-Match": headers["ETag"]})
        response = connection.getresponse()
        assert response.status == 304
        assert response.getheader("ETag") == headers["ETag"]
        assert response.read() == b""
    finally:
        connection.close()


def test_a_changed_face_revision_changes_etag_even_when_crop_pixels_are_the_same(dashboard, monkeypatch):
    base, ids = dashboard
    url = f"/faces/shot/{ids.kept}.jpg"
    assert get(base, url)[0] == 200
    revision = ["0"]
    monkeypatch.setattr(photo_files, "_shot_revision", lambda state=None: revision[0])
    _, before, pixels = get(base, url)
    revision[0] = "1"
    code, after, body = get(base, url, {"If-None-Match": before["ETag"]})
    assert code == 200 and body == pixels
    assert after["ETag"] != before["ETag"]
    assert get(base, url, {"If-None-Match": '"old", W/' + after["ETag"]})[0] == 304


def test_video_thumbnail_recovers_a_real_frame_after_the_visit_start(tmp_path, monkeypatch):
    calls = []
    frame = np.full((90, 160, 3), 170, dtype=np.uint8)

    class Recording:
        def frame(self, cam, ts, within_s=0.001):
            calls.append((cam, ts, within_s))
            return frame if ts == 101.0 else None

        def video_spans(self, cam, start, end):
            assert (cam, start) == ("cam0", 100.0)
            assert end <= 105.0
            return [[101.0, 104.0]]

    monkeypatch.setattr(photo_files.vision_log, "at", lambda state: Recording())
    data = photo_files.video_thumb_jpeg("cam0", 100.0, 110.0)
    assert data is not None
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape[:2] == (180, 320)
    assert image.mean() > 160
    assert calls == [("cam0", 100.0, 0.001), ("cam0", 101.0, 0.05)]


@pytest.mark.parametrize("cam,start,end", [("../x", 100, 101), ("cam0", float("nan"), 101), ("cam0", 101, 100)])
def test_video_thumbnail_refuses_invalid_bounds(cam, start, end):
    with pytest.raises(ValueError):
        photo_files.video_thumb_jpeg(cam, start, end)


def test_video_scene_thumbnail_is_served_for_a_recorded_frame(dashboard, tmp_path):
    base, ids = dashboard
    shot = sightings.shot(ids.kept, tmp_path)
    assert shot is not None
    query = urllib.parse.urlencode({"cam": shot["cam"], "start": shot["ts"], "end": shot["ts"] + 2})
    assert get(base, "/people/video-thumb.jpg?" + query)[0] == 202
    assert photo_files._crop_jobs.one()
    code, headers, body = get(base, "/people/video-thumb.jpg?" + query)
    assert code == 200
    assert headers["Content-Type"] == "image/jpeg"
    image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert max(image.shape[:2]) <= 320
    assert min(image.shape[:2]) <= 180
    assert get(base, "/people/video-thumb.jpg?cam=cam0&start=nope&end=10")[0] == 400


def test_old_geometry_cache_is_not_reused_after_pixel_mapping_changes(dashboard, tmp_path):
    base, ids = dashboard
    folder = face_shots.directory(context.STATE, ids.kept)
    folder.mkdir(parents=True, exist_ok=True)
    old = folder / f"{ids.kept}.jpg"
    old.write_bytes(b"old incorrect geometry")
    code, _, body = get(base, f"/faces/shot/{ids.kept}.jpg")
    assert code == 200 and body != old.read_bytes()
    assert server._shot_path(ids.kept).exists()


def test_a_shot_that_would_wait_too_long_for_a_cut_is_told_to_come_back(dashboard, monkeypatch):
    # Every turn to cut is taken (a grid of old sightings): the request gives its connection back at once.
    base, ids = dashboard
    monkeypatch.setattr(photo_files, "SHOT_WAIT_S", 0.05)
    held = [photo_files._shot_decode.acquire(timeout=1) for _ in range(2)]
    try:
        code, headers, _ = get(base, f"/faces/shot/{ids.kept}.jpg")
    finally:
        for _ in held:
            photo_files._shot_decode.release()
    assert (code, headers["Retry-After"]) == (503, "2")
    assert get(base, f"/faces/shot/{ids.kept}.jpg")[0] == 200


def test_backfill_cuts_the_crops_sightings_lack_and_skips_the_rest(dashboard, tmp_path):
    _, ids = dashboard
    revision = photo_files._shot_revision()
    face_shots.write(tmp_path, "had", revision, b"x")  # not a sighting: left alone
    done = face_shots.backfill(tmp_path, per_second=1000, now=T + 60)
    assert done == {"rows": 2, "had": 0, "cut": 1, "gone": 1}  # the kept sighting cut, the other's frame gone
    path, _ = photo_files._shot_file(ids.kept)
    assert path is not None and path == face_shots.path(tmp_path, ids.kept, revision)
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    assert img[70:90, 70:90].mean() > 200  # the same crop the dashboard cuts: the face in the middle
    assert face_shots.backfill(tmp_path, per_second=1000, now=T + 60)["had"] == 1


def test_a_face_at_the_frames_edge_is_cut_from_the_part_inside(dashboard, vision):
    # The detector's box around a half-seen head runs past the frame: its shot was refused as out of range (a 404).
    base, _ = dashboard
    edge = vision.face(T + 0.5, "cam0", "Alice", 0.8, bbox=[BOX[0], BOX[1], 1.034, BOX[3]])["id"]
    code, _, body = get(base, f"/faces/shot/{edge}.jpg")
    assert code == 200
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (server.SHOT_PX, server.SHOT_PX)


def video_thumb_url(cam="cam0", start=T + 0.5):
    return "/people/video-thumb.jpg?" + urllib.parse.urlencode({"cam": cam, "start": start, "end": start + 2})


def test_card_admits_video_jobs_but_lists_only_ready_thumbnails(dashboard, tmp_path, monkeypatch):
    base, ids = dashboard
    revision = photo_files._shot_revision()
    face_shots.write(tmp_path, ids.kept, revision, b"face")
    face_shots.write(tmp_path, ids.gone, revision, b"face")
    jobs = face_shots.Pending(photo_files._crop_process)
    monkeypatch.setattr(photo_files, "_crop_jobs", jobs)
    card = people_reads.person_card("Alice")
    assert card is not None
    assert card["videos"] == []
    assert card["videos_pending"] == 1
    assert not video_thumbs.folder(tmp_path).exists()  # listing never cuts recorded video
    assert jobs.one()
    after = people_reads.person_card("Alice")
    assert after is not None and len(after["videos"]) == 1
    assert after.get("videos_pending", 0) == 0
    held = [photo_files._shot_decode.acquire(), photo_files._shot_decode.acquire()]
    try:
        assert get(base, after["videos"][0]["thumb"])[0] == 200  # a ready tile needs no decode turn
    finally:
        for _ in held:
            photo_files._shot_decode.release()


def test_a_video_thumbnail_is_decoded_once_then_read_from_its_file(dashboard, tmp_path, monkeypatch):
    base, _ = dashboard
    assert get(base, video_thumb_url())[0] == 202
    assert not video_thumbs.folder(tmp_path).exists()
    assert photo_files._crop_jobs.one()
    code, headers, first = get(base, video_thumb_url())
    assert code == 200 and headers["Cache-Control"] == "private, max-age=86400"
    (saved,) = video_thumbs.folder(tmp_path).glob("*.jpg")
    assert saved.read_bytes() == first

    def no_decode(*args, **kwargs):
        raise AssertionError("a saved thumbnail is read, not decoded again")

    monkeypatch.setattr(photo_files, "video_thumb_jpeg", no_decode)
    assert get(base, video_thumb_url())[:3:2] == (200, first)


def test_a_forget_cuts_a_video_thumbnail_again_from_what_the_video_still_has(dashboard, tmp_path, monkeypatch):
    base, _ = dashboard
    assert get(base, video_thumb_url())[0] == 202
    assert photo_files._crop_jobs.one()
    assert get(base, video_thumb_url())[0] == 200
    monkeypatch.setattr(photo_files, "_shot_revision", lambda state=None: "9")  # a forget elsewhere moved the revision
    monkeypatch.setattr(photo_files, "video_thumb_jpeg", lambda *a, **k: None)  # and took that moment's video
    assert get(base, video_thumb_url())[0] == 202
    assert photo_files._crop_jobs.one()
    assert get(base, video_thumb_url())[0] == 404
    assert list(video_thumbs.folder(tmp_path).glob("*.jpg")) == []  # the old one isn't served, or kept


def test_a_video_thumbnail_admits_background_work_while_all_decode_turns_are_taken(dashboard, monkeypatch):
    base, _ = dashboard
    monkeypatch.setattr(photo_files, "SHOT_WAIT_S", 0.05)
    held = [photo_files._shot_decode.acquire(), photo_files._shot_decode.acquire()]  # both turns busy
    try:
        code, headers, _ = get(base, video_thumb_url())
        assert (code, headers["Retry-After"]) == (202, "2")
        assert headers["Cache-Control"] == "no-store"
        assert photo_files._crop_jobs.one()  # contention leaves the admitted job pending, never a broken tile
        assert photo_files._crop_jobs.pending
    finally:
        for _ in held:
            photo_files._shot_decode.release()
    assert photo_files._crop_jobs.one()
    assert get(base, video_thumb_url())[0] == 200


def test_video_thumbnails_follow_the_video_window_and_the_byte_cap(tmp_path):
    now = 1_800_000_000.0
    old = video_thumbs.write(tmp_path, "cam0", now - 31 * 86400, "1", b"x" * 10)  # its frame is past the window
    keep = [video_thumbs.write(tmp_path, "cam0", now - k * 3600, "1", b"x" * 10) for k in (3, 2, 1)]
    stray = video_thumbs.folder(tmp_path) / "cam0-123-scene-160x90-r1.jpg"  # another size: never served
    stray.write_bytes(b"x")
    assert video_thumbs.prune(tmp_path, now=now, cap_bytes=10**6) == 2
    assert not old.exists() and not stray.exists() and all(p.exists() for p in keep)
    assert video_thumbs.prune(tmp_path, now=now, cap_bytes=30) == 1  # down to 80% (24 of 30 bytes): the oldest goes
    assert [p.exists() for p in keep] == [False, True, True]
