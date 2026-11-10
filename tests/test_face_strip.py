"""A face's filmstrip on People (issue #60): people/face_photos.sightings_of and its dashboard route, and the photo a
face group gets, cut from a camera frame."""

import json
import time
import urllib.request

import cv2
import numpy as np
import pytest

from liberational import config as cfg
from liberational import segment_log
from liberational.dashboard import server
from liberational.people import face_photos, face_shots
from liberational.vision import faces as fid

T = round(time.time() - 3600.0, 3)  # an hour ago: the strip looks back 30 days


def unit(*xs):
    v = np.zeros(512, np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


@pytest.fixture
def state(tmp_path, monkeypatch, vision):
    """Face groups 5 and 6, and their sightings: {key: sighting id}."""
    state, faces_dir = tmp_path / "state", tmp_path / "faces"
    monkeypatch.setattr(fid, "UNK_DB", faces_dir / "unknown_clusters.npz")
    fid._unk_save([5, 6], np.stack([unit(1), unit(0, 1)]), [4, 2], 7)
    ids = {
        "a": vision.face(T + 100, cluster="Face 5")["id"],
        "b": vision.face(T + 200, cluster="Face 5", quality=0.8)["id"],  # newer, and the best shot
        "c": vision.face(T + 300, cluster="Face 6")["id"],
        "d": vision.face(T + 400, name="Bob", sim=0.6, cluster="Face 5")["id"],  # named: not in the strip
        "e": vision.face(T + 500, cluster="Face 5")["id"],
    }
    vision.records[segment_log.FACE_MATCHES][-1]["det"] = 0.3  # the detector wasn't sure of "e": left out
    monkeypatch.setattr(face_photos, "_failed", {}, raising=False)
    with cfg.override({"paths": {"state": str(state), "faces": str(faces_dir)}}):
        yield ids


@pytest.mark.parametrize(
    ("gid", "kwargs", "expected"),
    [
        pytest.param(5, {}, [("b", 200.0), ("a", 100.0)], id="unnamed-newest-first"),
        pytest.param(6, {}, [("c", 300.0)], id="another-group"),
        pytest.param(5, {"limit": 1}, [("b", 200.0)], id="limit"),
        pytest.param(99, {}, [], id="no-such-group"),
    ],
)
def test_a_groups_strip_is_its_unnamed_sightings_newest_first(state, gid, kwargs, expected):
    want = [{"id": state[k], "ts": T + dt} for k, dt in expected]
    assert face_photos.sightings_of(gid, **kwargs) == want


def test_the_strip_reaches_past_newer_sightings_of_other_faces(state, vision):
    # Codex on PR #71: a face seen before a busy stretch of other faces must still get its strip.
    for k in range(50):
        vision.face(T + 1000 + k, cluster="Face 6")
    assert [s["id"] for s in face_photos.sightings_of(5)] == [state["b"], state["a"]]
    assert len(face_photos.sightings_of(6, limit=3)) == 3


def test_recent_sightings_go_to_their_groups(state):
    got = face_photos.recent_by_group(per=1)
    assert {g: [s["id"] for s in v] for g, v in got.items()} == {5: [state["b"]], 6: [state["c"]]}


def test_a_groups_photo_is_cut_from_its_best_shots_frame(state, vision):
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[20:40, 80:100] = 255  # the default box, [0.4, 0.2, 0.5, 0.4] of 200x100, is white
    vision.frames[("cam0", T + 200)] = frame
    assert face_photos.refresh() == 1  # Face 6 has no frame, so only Face 5 gets one
    photo = cv2.imread(str(face_photos._dest(5)))
    assert photo is not None and photo.shape == (160, 160, 3)
    assert photo[80, 80].min() > 200  # the face's middle is the white box
    assert not face_photos._dest(6).exists()


def test_a_group_whose_photo_couldnt_be_cut_waits_for_new_sightings(state, vision, monkeypatch):
    monkeypatch.setattr(face_photos, "_failed", {}, raising=False)
    reads = []
    real = face_photos._unnamed
    monkeypatch.setattr(face_photos, "_unnamed", lambda *a, **k: reads.append(1) or real(*a, **k))
    assert face_photos.refresh() == 0  # neither group has a frame
    assert face_photos.refresh() == 0
    assert len(reads) == 1  # the second pass had nothing new to try, so it read nothing
    fid._unk_save([5, 6], np.stack([unit(1), unit(0, 1)]), [5, 2], 7)  # Face 5 seen again
    frame = np.zeros((100, 200, 3), np.uint8)
    vision.frames[("cam0", T + 200)] = frame
    assert face_photos.refresh() == 1
    assert len(reads) == 2 and face_photos._dest(5).exists() and not face_photos._dest(6).exists()


@pytest.fixture
def strip(state, serve, monkeypatch, tmp_path):
    """GET a group's strip from a served dashboard: (its reply, the crop jobs it queued). No crop is cut here."""
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    monkeypatch.setattr(server, "STATE", tmp_path / "state")
    jobs = face_shots.Pending(lambda key: None)
    monkeypatch.setattr(server, "_crop_jobs", jobs)
    base = serve(server.H)

    def get(gid):
        with urllib.request.urlopen(f"{base}/faces/unknown/{gid}/sightings", timeout=5) as r:
            assert r.headers["Content-Type"] == "application/json"
            return json.loads(r.read()), jobs

    return get


def test_the_dashboard_serves_the_strip_as_saved_shot_urls(state, strip, tmp_path):
    for key in ("a", "b"):
        face_shots.write(tmp_path / "state", state[key], "0", b"jpeg")
    body, _ = strip(5)
    # Each picture is the sighting's saved shot, newest first (Codex on PR #71).
    assert [x["url"] for x in body["sightings"]] == [f"/faces/shot/{state['b']}.jpg", f"/faces/shot/{state['a']}.jpg"]
    assert (body["total"], body["shots_pending"]) == (2, 0)


def test_a_strip_lists_only_saved_shots_and_queues_the_rest(state, strip, tmp_path):
    face_shots.write(tmp_path / "state", state["b"], "0", b"jpeg")
    body, jobs = strip(5)
    assert [x["id"] for x in body["sightings"]] == [state["b"]]
    assert (body["total"], body["shots_pending"], body["shots_unavailable"]) == (1, 1, 0)
    assert len(jobs.pending) == 1  # "a" waits for the crop worker; the request itself cuts nothing


def test_recent_groups_reuse_complete_rows_without_reading_the_broker(state, vision):
    from liberational.vision import sightings

    rows = sightings.faces(T, T + 1000)
    vision.reads = 0
    assert face_photos.recent_by_group(per=1, rows=rows) == {
        5: [{"id": state["b"], "ts": T + 200, "cam": "cam0"}],
        6: [{"id": state["c"], "ts": T + 300, "cam": "cam0"}],
    }
    assert face_photos.recent_by_group(rows=[]) == {}
    assert vision.reads == 0


@pytest.fixture
def shots(state, vision, monkeypatch, tmp_path):
    """Group 5's two shots with embeddings ("a" looks along x, "b" along y), and the inbox skips in tmp_path."""
    for key, v in (("a", unit(1)), ("b", unit(0, 1))):
        match = next(r for r in vision.records[segment_log.FACE_MATCHES] if r["id"] == state[key])
        vision.add(segment_log.FACE_EMBEDDINGS, ts_start=match["ts_start"], cam="cam0", match_id=state[key], vector=v)
    monkeypatch.setattr(server, "INBOX_SKIPS", tmp_path / "inbox_skips.json")
    return state


def group(gid):
    ids, vecs, counts, _ = fid._unk_load()
    i = [int(x) for x in ids].index(gid)
    return vecs[i], counts[i]


def test_a_rejected_shot_leaves_its_group_and_the_group_keeps_only_the_rest(shots):
    code, reply = server.reject_shots(5, [shots["b"]], True)
    assert (code, reply) == (200, {"ok": True, "removed": 1, "left": 1, "ignored": False})
    assert [s["id"] for s in face_photos.sightings_of(5)] == [shots["a"]]
    vec, count = group(5)
    assert vec == pytest.approx(unit(1), abs=1e-6)  # rebuilt from "a" alone: naming the group teaches only it
    assert count == 3
    # every reader that joins sightings to groups sees it in none, so naming Face 5 won't name it either
    from liberational.vision import sightings

    rejected = next(f for f in sightings.faces(T, T + 1000) if f["id"] == shots["b"])
    assert rejected["cluster"] is None
    assert face_photos.page_of(5, 0, 10) == ([{"id": shots["a"], "ts": T + 100}], 1)


def test_rejecting_a_groups_last_shots_puts_it_aside(shots):
    code, reply = server.reject_shots(5, [shots["a"], shots["b"]], True)
    assert (code, reply["left"], reply["ignored"]) == (200, 0, True)
    assert 5 in server.inbox_skips()["face"]


@pytest.mark.parametrize(
    ("gid", "ids", "control", "code"),
    [
        pytest.param(5, ["b"], False, 403, id="not-this-machine"),
        pytest.param(5, ["c"], True, 400, id="a-shot-of-another-group"),
        pytest.param(5, [], True, 400, id="no-shots"),
        pytest.param(99, ["a"], True, 404, id="no-such-group"),
    ],
)
def test_a_reject_that_cant_apply_changes_nothing(shots, gid, ids, control, code):
    before = group(5)
    got, reply = server.reject_shots(gid, [shots[k] for k in ids], control)
    assert (got, reply["ok"]) == (code, False)
    assert [s["id"] for s in face_photos.sightings_of(5)] == [shots["b"], shots["a"]]
    assert group(5)[1] == before[1]


@pytest.mark.parametrize(
    ("offset", "limit", "expected"),
    [
        pytest.param(0, 1, ["b"], id="first-page"),
        pytest.param(1, 5, ["a"], id="next-page"),
        pytest.param(5, 5, [], id="past-the-end"),
    ],
)
def test_a_groups_shots_come_a_page_at_a_time_with_the_total(state, offset, limit, expected):
    page, total = face_photos.page_of(5, offset, limit)
    assert ([s["id"] for s in page], total) == ([state[k] for k in expected], 2)
