"""Pet thumbnails (#2038): each animal track's photo saved as its largest box comes, served with no lookup, and cut
from recorded video only for a track without one. Synthetic frames and made-up tracks."""

import http.client
import urllib.parse

import cv2
import numpy as np
import pytest

from liberational.dashboard import server
from liberational.people import pet_recognition, pet_shots
from liberational.see.caption_animals import CaptionAnimals

T = 1_800_000_000.0
TRACK = "run1:dog:1"
BOX = [0.25, 0.25, 0.5, 0.5]  # the white square in the frame below


def frame():
    f = np.zeros((120, 160, 3), np.uint8)
    f[30:60, 40:80] = 255
    return f


def sighting(track=TRACK, box=BOX, cam="cam0"):
    return pet_recognition.Sighting(track, "dog", cam, T, T, shot={"ts": T, "box": box, "area": 1.0})


def no_lookup(*args, **kwargs):
    raise AssertionError("looked up")


def test_a_track_is_rewritten_only_when_its_box_is_the_largest_yet(tmp_path):
    writer, cut = pet_shots.Writer(tmp_path), []

    def crop(bounds):
        cut.append(bounds)
        a, b, c, d = bounds
        return frame()[b:d, a:c]

    writer.offer(TRACK, [0.1, 0.1, 0.3, 0.3], 160, 120, crop)
    writer.offer(TRACK, [0.1, 0.1, 0.2, 0.2], 160, 120, crop)  # smaller: kept the first
    writer.offer(TRACK, BOX, 160, 120, crop)  # larger: rewritten
    assert len(cut) == 2
    (saved,) = (tmp_path / pet_shots.KIND).glob("*.jpg")
    img = cv2.imread(str(saved))
    assert img is not None and img.shape == (pet_shots.PX, pet_shots.PX, 3)
    assert img[80:110, 80:110].mean() > 200  # the box in the middle


def test_a_captions_embedded_box_is_offered_for_its_tracks_photo():
    offered = []

    class Shots:
        def offer(self, track, box, width, height, crop):
            offered.append((track.split(":", 1)[1], box, width, height, crop((40, 30, 80, 60)).shape))

    animals = CaptionAnimals("cam0", None, lambda f, b: np.ones(4), shots=Shots())
    animals.look(np.zeros((480, 640, 3), np.uint8), T, "", [{"label": "dog", "box": [100, 100, 600, 600]}])
    assert offered == [("dog:1", [0.1, 0.1, 0.6, 0.6], 640, 480, (30, 40, 3))]


@pytest.fixture
def route(serve, tmp_path, monkeypatch, vision):
    monkeypatch.setattr(server, "STATE", tmp_path)
    vision.frames[("cam0", T)] = frame()
    monkeypatch.setattr(pet_recognition, "recorded", lambda state, now=None: [sighting()])
    host = serve(server.H).removeprefix("http://")

    def get(track):
        c = http.client.HTTPConnection(host, timeout=10)
        try:
            c.request("GET", "/pets/sightings/photo?track=" + urllib.parse.quote(track))
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            c.close()

    return get


def test_a_saved_photo_is_served_without_a_lookup(route, tmp_path, monkeypatch):
    pet_shots.Writer(tmp_path).offer(TRACK, BOX, 160, 120, lambda b: frame()[b[1] : b[3], b[0] : b[2]])
    monkeypatch.setattr(pet_recognition, "recorded", no_lookup)
    code, headers, body = route(TRACK)
    assert (code, headers["Content-Type"]) == (200, "image/jpeg")
    assert "max-age=86400" in headers["Cache-Control"]
    assert body[:2] == b"\xff\xd8"


def test_a_track_without_a_photo_is_cut_once_and_then_served_from_its_file(route, monkeypatch):
    code, _, body = route(TRACK)
    assert code == 200
    monkeypatch.setattr(pet_recognition, "recorded", no_lookup)
    assert route(TRACK) == (200, route(TRACK)[1], body)


@pytest.mark.parametrize("track", [pytest.param("run1:cat:9", id="not-listed"), pytest.param("../x", id="unsafe")])
def test_an_unknown_or_unsafe_track_is_404(route, track):
    assert route(track)[0] == 404


def test_a_photo_that_would_wait_too_long_for_a_cut_is_told_to_come_back(route, monkeypatch):
    monkeypatch.setattr(server, "SHOT_WAIT_S", 0.05)
    held = [server._shot_decode.acquire(timeout=1) for _ in range(2)]
    try:
        code, headers, _ = route(TRACK)
    finally:
        for _ in held:
            server._shot_decode.release()
    assert (code, headers["Retry-After"]) == (503, "2")


def test_backfill_cuts_the_photos_tracks_lack(tmp_path, monkeypatch, vision):
    vision.frames[("cam0", T)] = frame()
    tracks = [sighting(), sighting("run1:dog:2"), sighting("run1:dog:3", cam="cam1")]  # cam1's frame is gone
    monkeypatch.setattr(pet_recognition, "recorded", lambda state, now=None: tracks)
    pet_shots.Writer(tmp_path).offer("run1:dog:2", BOX, 160, 120, lambda b: frame()[b[1] : b[3], b[0] : b[2]])
    done = pet_shots.backfill(tmp_path, per_second=1000, now=T + 60)
    assert done == {"tracks": 3, "had": 1, "cut": 1, "gone": 1}
    assert pet_shots.backfill(tmp_path, per_second=1000, now=T + 60)["had"] == 2


def test_background_pet_jobs_share_one_lookup_before_acquiring_decode_turns(tmp_path, monkeypatch):
    from liberational.dashboard import pets

    calls = []
    tracks = [sighting(), sighting("run1:dog:2")]

    def recorded(state, now=None):
        calls.append(state)
        assert server._shot_decode.acquire(blocking=False)
        assert server._shot_decode.acquire(blocking=False)
        server._shot_decode.release()
        server._shot_decode.release()
        return tracks

    monkeypatch.setattr(pet_recognition, "recorded", recorded)
    monkeypatch.setattr(pet_shots, "cut", lambda state, track: b"jpeg")
    pets._photo_tracks.cache_clear()
    assert server._pet_file(TRACK, tmp_path) == b"jpeg"
    assert server._pet_file("run1:dog:2", tmp_path) == b"jpeg"
    assert calls == [tmp_path]
