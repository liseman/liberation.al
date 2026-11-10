"""Review's server side: what each camera saw at a moment (GET /history/at) and where each video segment ends."""

import http.client
import urllib.parse

import pytest

from liberational.dashboard import server

NOW = 1_800_000_000.0


@pytest.fixture
def notes(vision):
    faces = [{"name": "Alice", "sim": 0.8, "bearing_deg": 10, "bbox": [1, 2, 3, 4]}]
    vision.caption(NOW - 50, "cam0", "An empty desk.", faces=[])
    vision.caption(NOW - 4, "cam0", "Alice at the desk.", faces=faces)
    vision.caption(NOW + 6, "cam1", "A doorway.")  # no faces checked
    return vision


def test_nearest_note_per_camera_within_the_window(notes):
    cams = server.history_at(NOW, 20)["cams"]
    assert set(cams) == {"cam0", "cam1"}
    assert (cams["cam0"]["caption"], cams["cam0"]["dt"]) == ("Alice at the desk.", -4)
    assert (cams["cam0"]["unchanged"], cams["cam0"]["since"]) == (False, None)
    assert cams["cam0"]["faces"] == [{"name": "Alice", "sim": 0.8, "bearing_deg": 10}]  # no boxes
    assert cams["cam1"]["faces"] is None  # no faces checked


def test_window_is_bounded_and_nothing_in_it_is_empty(notes):
    assert server.history_at(NOW, 10_000)["window"] == 600
    assert server.history_at(NOW - 200, 20)["cams"] == {}


def test_history_is_for_this_machine_only(serve, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    base = serve(server.H)

    def status(path, **headers):
        conn = http.client.HTTPConnection(urllib.parse.urlsplit(base).netloc, timeout=5)
        conn.request("GET", path, headers=headers)
        code = conn.getresponse().status
        conn.close()
        return code

    assert status("/history/at?ts=1&window=5") == 200
    assert status("/history/at?ts=1", **{"X-Forwarded-For": "100.64.0.9"}) == 403  # via Tailscale
    assert status("/history/at?ts=x") == 400


def test_remote_history_needs_the_setting_and_a_password(serve, monkeypatch):
    import base64

    base = serve(server.H)
    auth = {"Authorization": "Basic " + base64.b64encode(b"luca:pw").decode(), "X-Forwarded-For": "100.64.0.9"}

    def status(**headers):
        conn = http.client.HTTPConnection(urllib.parse.urlsplit(base).netloc, timeout=5)
        conn.request("GET", "/history/at?ts=1&window=5", headers=headers)
        code = conn.getresponse().status
        conn.close()
        return code

    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_camera_history", lambda: False)
    assert status(**auth) == 403  # off: this machine only
    monkeypatch.setattr(server, "remote_camera_history", lambda: True)
    assert status(**auth) == 200  # on, and the password checked
    assert status(**{"X-Forwarded-For": "100.64.0.9"}) == 401  # on, but no password: the gate before it refuses
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    assert status(**{"X-Forwarded-For": "100.64.0.9"}) == 403  # on, but no dashboard password at all: never


def test_a_note_stored_while_the_view_was_unchanged_shows_its_last_real_caption(vision):
    vision.caption(NOW - 300, "cam0", "Two people on the couch.")
    vision.caption(NOW - 2, "cam0", "Two people on the couch.", unchanged=True, since=NOW - 300, gated=True)
    cam = server.history_at(NOW, 20)["cams"]["cam0"]
    assert (cam["caption"], cam["unchanged"], cam["since"]) == ("Two people on the couch.", True, NOW - 300)
