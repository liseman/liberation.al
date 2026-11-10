"""Live's "Who's this?" (vision/face_guess.py and GET /faces/unknown/groups): each unnamed face group's pictures, a
best guess with a score and a band, and the reason line."""

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from liberational import config as cfg
from liberational import segment_log
from liberational.dashboard import server
from liberational.people import registry
from liberational.see import subjects
from liberational.vision import face_guess as G
from liberational.vision import faces as fid
from speech_fake import fake


def at(sim, axis=1):
    """A unit embedding whose similarity with unit(1) (the first axis) is sim, leaning along another axis."""
    v = np.zeros(512, np.float32)
    v[0], v[axis] = sim, math.sqrt(1 - sim * sim)
    return v


E1 = at(1.0)


@pytest.mark.parametrize(
    ("sim", "expected"),
    [(0.25, 0), (0.45, 50), (0.55, 75), (0.65, 100), (0.1, 0), (0.9, 100)],
)
def test_a_similarity_becomes_a_likeness_with_the_live_threshold_at_50(sim, expected):
    assert G.likeness(sim) == expected


@pytest.mark.parametrize(("gap", "expected"), [(0.0, 50), (0.04, 75), (0.08, 100), (0.2, 100)])
def test_the_lead_over_the_runner_up_is_50_for_a_tie_and_75_at_close(gap, expected):
    assert G.lead(gap) == expected


@pytest.mark.parametrize(
    ("photos", "guess"),
    [
        pytest.param(
            [("Kara", 0.61), ("Kara", 0.3), ("Luca", 0.41)],
            {"name": "Kara", "score": 90, "band": "probable", "close": None},
            id="probable-by-the-closest-photo",
        ),
        pytest.param(
            [("Kara", 0.61), ("Luca", 0.60)],
            {"name": "Kara", "score": 56, "band": "maybe", "close": "Luca"},
            id="a-close-runner-up-holds-the-score",
        ),
        pytest.param(
            [("Kara", 0.80), ("Luca", 0.79)],
            {"name": "Kara", "score": 56, "band": "maybe", "close": "Luca"},
            id="never-a-maybe-at-100",
        ),
        pytest.param(
            [("Kara", 0.808), ("Luca", 0.689)],
            {"name": "Kara", "score": 100, "band": "probable", "close": None},
            id="far-ahead-of-a-runner-up-who-is-also-alike",
        ),
        pytest.param([("Luca", 0.49)], {"name": "Luca", "score": 60, "band": "maybe", "close": None}, id="maybe"),
        pytest.param(
            [("Luca", 0.35)], {"name": "Luca", "score": 25, "band": "none", "close": None}, id="no-good-guess"
        ),
    ],
)
def test_the_best_guess_is_the_closest_named_photo(photos, guess):
    names = [n for n, _ in photos]
    embs = np.stack([at(s, axis=k + 1) for k, (_, s) in enumerate(photos)])
    assert G.best_guess(E1, names, embs) == guess


def test_there_is_no_guess_with_nobody_named():
    assert G.best_guess(E1, [], np.zeros((0, 512), np.float32)) is None


PROBABLE = {"name": "Kara", "score": 90, "band": "probable", "close": None}


@pytest.mark.parametrize(
    ("guess", "count", "det", "height", "light", "expected"),
    [
        pytest.param(PROBABLE, 40, 0.9, 0.2, 150.0, "A clear face, close to Kara's photos", id="clear"),
        pytest.param({**PROBABLE, "band": "maybe", "score": 60}, 40, 0.9, 0.2, 150.0,
                     "A clear face, but only partly like Kara", id="clear-maybe"),
        pytest.param({**PROBABLE, "band": "none", "score": 20}, 40, 0.9, 0.2, 150.0,
                     "A clear face, unlike anyone named", id="clear-none"),
        pytest.param(PROBABLE, 40, 0.6, 0.2, 40.0, "Side-on or blurred and dim", id="two-issues"),
        pytest.param(PROBABLE, 3, 0.6, 0.05, 40.0, "Only seen 3 times and side-on or blurred", id="first-two-only"),
        pytest.param(PROBABLE, 40, None, None, None, "A clear face, close to Kara's photos", id="nothing-measured"),
        pytest.param({**PROBABLE, "band": "maybe", "score": 56, "close": "Luca"}, 40, 0.9, 0.06, 150.0,
                     "Looks like Luca too and far from the camera", id="held-by-the-runner-up"),
        pytest.param({**PROBABLE, "score": 80, "close": "Luca"}, 40, 0.9, 0.2, 150.0,
                     "A clear face, close to Kara's photos", id="a-runner-up-that-leaves-it-probable"),
        pytest.param(None, 40, 0.9, 0.2, 150.0, "Nobody is named yet, so there's no one to compare with", id="nobody"),
    ],
)  # fmt: skip
def test_the_reason_names_what_holds_the_guess_back(guess, count, det, height, light, expected):
    assert G.reason(guess, count, det, height, light) == expected


B = float(round(time.time()) - 2 * 86400)  # two days ago: the groups' pictures come from the last 30 days


def add_sightings(vision, ids, rows):
    """rows: (key, seconds after B, its face group or None, name or None, det, face height as a share of the frame).
    Each sighting's id goes in ids under its key."""
    for key, t, group, name, det, h in rows:
        cluster = f"Face {group}" if group is not None else None
        d = vision.face(B + t, "cam0", name, 0.6 if name else None, bbox=[0.1, 0.1, 0.3, 0.1 + h], cluster=cluster)
        d["det"] = det
        ids[key] = d["id"]


@pytest.fixture
def ids():
    """Each sighting's id, by the key a test gave it."""
    return {}


@pytest.fixture
def state(tmp_path, monkeypatch, vision, ids):
    """Face 5 (like Kara, seen 12 times), Face 6 (like nobody, seen 3 times), and Face 7 (put aside); Kara named."""
    state, faces_dir = tmp_path / "state", tmp_path / "faces"
    faces_dir.mkdir()
    monkeypatch.setattr(fid, "UNK_DB", faces_dir / "unknown_clusters.npz")
    monkeypatch.setattr(fid, "DB", faces_dir / "db.npz")
    monkeypatch.setattr(fid, "CHECK", faces_dir / "check.json")
    monkeypatch.setattr(server, "INBOX_SKIPS", tmp_path / "inbox_skips.json")
    vecs = np.stack([E1, at(0.0, 3), at(0.0, 4), at(0.95, 6)])
    fid._unk_save([5, 6, 7, 8], vecs, [12, 3, 9, 4], 9, last_seen=dict.fromkeys([5, 6, 7, 8], time.time()))
    np.savez(faces_dir / "db.npz", names=np.array(["Kara"], dtype=object), embs=np.stack([at(0.61, 2)]))
    rows = [(k, 100.0 + k, 5, None, 0.9, 0.2) for k in range(1, 9)]  # Face 5: 8 sightings
    rows += [(20, 50.0, 6, None, 0.6, 0.05), (21, 60.0, 6, "Kara", 0.9, 0.2)]  # Face 6; one named since
    add_sightings(vision, ids, rows)
    with cfg.override({"paths": {"state": str(state), "faces": str(faces_dir)}}):
        yield state


def test_rows_come_best_guess_first_with_their_newest_shots(state, ids):
    rows = G.groups([{"id": 6, "count": 3}, {"id": 5, "count": 12}], state)
    assert [r["label"] for r in rows] == ["Face 5", "Face 6"]
    five, six = rows
    assert [s["id"] for s in five["shots"]] == [ids[k] for k in (8, 7, 6, 5, 4, 3)]  # the six newest, newest first
    assert five["shots"][0] == {"id": ids[8], "ts": B + 108.0, "url": f"/faces/shot/{ids[8]}.jpg"}
    assert (five["count"], five["last_seen"]) == (12, B + 108.0)
    assert (five["guess"]["name"], five["band"], five["score"]) == ("Kara", "probable", 90)
    assert five["reason"] == "A clear face, close to Kara's photos"
    assert (six["guess"], six["band"], six["last_seen"]) == (None, "none", B + 50.0)  # the named one isn't Face 6's
    assert six["reason"] == "Only seen 3 times and side-on or blurred"


def test_a_dim_photo_shows_in_the_reason_and_last_seen_is_the_newest_recording(state):
    photo = cfg.path("paths.faces") / "_unknown" / "5.jpg"
    photo.parent.mkdir()
    Image.new("RGB", (40, 40), (20, 20, 20)).save(photo)
    os.utime(photo, (B + 5000.0, B + 5000.0))  # the photo can be cut later than any sighting: it says nothing of when
    (five,) = G.groups([{"id": 5, "count": 12}], state)
    assert five["last_seen"] == B + 108.0
    assert five["reason"] == "Dim"


def test_a_group_has_the_sightings_see_actions_matched_to_it(state, ids):
    # Face 8 is like Face 5's person, but see-actions matched none of the sightings to it.
    five, eight = sorted(G.groups([{"id": 5, "count": 12}, {"id": 8, "count": 4}], state), key=lambda r: r["id"])
    assert [s["id"] for s in five["shots"]] == [ids[k] for k in (8, 7, 6, 5, 4, 3)]
    assert (eight["shots"], eight["last_seen"]) == ([], None)


def test_the_route_lists_the_groups_waiting_for_a_name(state, serve, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    server.INBOX_SKIPS.write_text(json.dumps({"face": [7]}))
    groups = server._Fresh("face route", server._unknown_groups, server._faces_owner_key, lambda: 0)
    monkeypatch.setattr(server, "_FACE_GROUPS", groups)
    # The first read creates the empty registry; startup retries that owner-key change before ready.
    assert groups.get()[0] == 200
    assert groups.get()[0] == 200
    assert groups.ready()
    with urllib.request.urlopen(serve(server.H) + "/faces/unknown/groups", timeout=5) as r:
        assert r.status == 200
        body = json.loads(r.read())
    assert (body["count"], body["skipped"]) == (3, [7])
    assert [g["id"] for g in body["groups"]] == [5, 8, 6]


@pytest.mark.parametrize(
    ("days", "listed"),
    [pytest.param(7, [5, 6], id="default"), pytest.param(10, [5, 6, 8], id="longer"), pytest.param(1, [5], id="a-day")],
)
def test_to_name_lists_and_counts_only_groups_seen_in_the_last_unnamed_days(state, monkeypatch, days, listed):
    # Face 5 was seen now, Face 6 two days ago (its sightings and the store agree), and Face 8, which has no
    # recorded sighting left, eight days ago by the store
    ids, vecs, counts, nxt, sums = fid._unk_load_sums()
    fid._unk_save(ids, vecs, counts, nxt, sums, last_seen={6: B, 8: time.time() - 8 * 86400})
    server.INBOX_SKIPS.write_text(json.dumps({"face": [7]}))
    monkeypatch.setattr(server, "STATE", state)
    with cfg.override({"faces": {"unnamed_days": days}}):
        code, body = server._unknown_groups()
    assert code == 200
    assert (sorted(g["id"] for g in body["groups"]), body["count"]) == (listed, len(listed))


def test_group_sighting_pages_count_and_page_only_saved_crops(state, ids, serve, monkeypatch):
    from liberational.people import face_shots

    monkeypatch.setattr(server, "STATE", state)
    monkeypatch.setattr(server, "_crop_jobs", face_shots.Pending(server._crop_process))
    face_shots.write(state, ids[2], "0", b"jpeg")
    face_shots.write(state, ids[4], "0", b"jpeg")
    base = serve(server.H)
    with urllib.request.urlopen(base + "/faces/unknown/5/sightings?offset=0&limit=1", timeout=5) as response:
        before = json.loads(response.read())
    assert before == {
        "sightings": [{"id": ids[4], "ts": B + 104, "url": f"/faces/shot/{ids[4]}.jpg"}],
        "total": 2,
        "shots_pending": 6,
        "shots_unavailable": 0,
    }
    face_shots.write(state, ids[8], "0", b"jpeg")
    with urllib.request.urlopen(base + "/faces/unknown/5/sightings?offset=1&limit=2", timeout=5) as response:
        after = json.loads(response.read())
    assert [row["id"] for row in after["sightings"]] == [ids[4], ids[2]]
    assert (after["total"], after["shots_pending"]) == (3, 5)


def test_the_route_says_when_the_groups_cannot_be_read(state, serve, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")

    def broken():
        raise ValueError("a half-written file")

    monkeypatch.setattr(fid, "_unk_load", broken)
    groups = server._Fresh("broken face route", server._unknown_groups, server._faces_owner_key, lambda: 0)
    monkeypatch.setattr(server, "_FACE_GROUPS", groups)
    url = serve(server.H) + "/faces/unknown/groups"
    with urllib.request.urlopen(url, timeout=5) as r:
        assert r.status == 202
    work = server._FACE_GROUPS.cold
    assert work is not None
    assert work.result(timeout=5)[0] == 500
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url, timeout=5)
    assert e.value.code == 500


def test_a_face_heard_with_a_named_voice_suggests_that_name_but_never_the_owners(state, monkeypatch, vision, ids):
    """Face 6 is seen three times while Dana speaks; the owner, Joe, talks near Face 5 all the while."""
    from voices import write_voices

    add_sightings(vision, ids, [(30, 400.0, 6, None, 0.9, 0.2), (31, 900.0, 6, None, 0.9, 0.2)])
    write_voices(state, {1: "Dana", 2: "Joe"})
    lines = [(49.0, 52.0, 1), (399.0, 402.0, 1), (899.0, 903.0, 1), (104.0, 106.0, 2), (106.5, 108.0, 2)]
    fake().write(
        [
            {"ts_start": B + a, "ts_end": B + b, "speaker_id": sid, "speaker": "", "speaker_sim": 0.8}
            for a, b, sid in lines
        ]
    )
    monkeypatch.setattr(server, "STATE", state)
    with cfg.override({"owner": {"name": "Joe"}}):
        code, body = server.unknown_groups_view()
    assert code == 200
    by = {g["id"]: g for g in body["groups"]}
    six = by[6]["together"]
    assert (six["name"], six["moments"], six["times"]) == ("Dana", 3, [B + 900.0, B + 400.0, B + 50.0])
    assert by[5]["together"] is None  # only the owner was heard near Face 5


def test_a_row_with_a_suggestion_sorts_with_the_maybes(monkeypatch):
    monkeypatch.setattr(server, "_named_sightings", lambda windows, people: [])
    rows: list[dict[str, Any]] = [
        {"id": 1, "band": "none", "score": 0, "last_seen": 300.0},
        {"id": 2, "band": "none", "score": 0, "last_seen": 100.0},
        {"id": 3, "band": "maybe", "score": 60, "last_seen": 50.0},
    ]
    pairs = {2: [(100.0, "Dana"), (500.0, "Dana")]}
    server._with_together(rows, lambda r, seen: pairs.get(r["id"], []), [(100.0, 500.0)], server._People())
    assert [r["id"] for r in rows] == [3, 2, 1]
    assert rows[1]["together"]["name"] == "Dana" and rows[0]["together"] is None


@pytest.fixture
def heard(state, monkeypatch, vision, ids):
    """Face 6 seen at 50, 400, and 900 (plus Face 5's eight at 101 to 108), all after B, lines written by the test
    (their times after B too), Joe the owner."""
    from voices import write_voices

    add_sightings(vision, ids, [(30, 400.0, 6, None, 0.9, 0.2), (31, 900.0, 6, None, 0.9, 0.2)])
    monkeypatch.setattr(server, "STATE", state)

    def lines(voices, spoken, heard_as=""):
        write_voices(state, voices)
        fake().write(
            [
                {"ts_start": B + a, "ts_end": B + b, "speaker_id": sid, "speaker": heard_as, "speaker_sim": 0.8}
                for a, b, sid in spoken
            ]
        )

    with cfg.override({"owner": {"name": "Joe"}}):
        yield lines


def six(code_body):
    code, body = code_body
    assert code == 200
    return next(g for g in body["groups"] if g["id"] == 6)["together"]


def test_a_suggestion_names_the_person_as_they_are_called_now(heard, state):
    heard({1: "Dana"}, [(49.0, 52.0, 1), (399.0, 402.0, 1), (899.0, 903.0, 1)], heard_as="Dana")
    registry.at(state).rename("Dana", "Dee", "test")  # the lines keep the name they were heard as
    assert six(server.unknown_groups_view())["name"] == "Dee"  # "Yes, Dee", never a new person called Dana


def test_the_owner_is_left_out_by_who_they_are_not_how_a_name_is_spelled(heard, state):
    heard({2: "Joe"}, [(49.0, 52.0, 2), (399.0, 402.0, 2), (899.0, 903.0, 2)])
    registry.at(state).rename("Joe", "Jordan", "test")  # the owner's name in the config is still Joe
    assert six(server.unknown_groups_view()) is None


def test_a_voice_whose_own_face_was_seen_a_moment_later_is_explained(heard, state, vision):
    """Dana's face is in view 2.5 s after Face 6 is, and Face 6 is 2.5 s after Dana's line ends: farther than the
    window from the line, so the read must reach twice the window."""
    heard({1: "Dana"}, [(45.0, 47.5, 1), (395.0, 397.5, 1), (895.0, 897.5, 1)])
    for t in (52.5, 402.5, 902.5):
        vision.face(B + t, "cam0", "Dana", 0.7)
    assert six(server.unknown_groups_view()) is None


def test_named_windows_batch_only_their_overlaps_and_match_the_unbatched_read(tmp_path, vision, monkeypatch):
    from liberational.vision import sightings

    monkeypatch.setattr(server, "STATE", tmp_path)
    vision.face(B + 1, name="Dana")
    vision.face(B + 6, name="Alice")  # the gap is not co-occurrence evidence
    vision.face(B + 11, name="Alice")

    class People:
        def name(self, written):
            return written

    windows = [(B, B + 2), (B + 10, B + 12)]
    fast = server._named_sightings(
        windows, cast(server._People, People()), face_batch=sightings.FaceBatch(B, B + 12.001, tmp_path)
    )
    assert fast == [(B + 1, "Dana"), (B + 11, "Alice")]
    assert server._named_sightings(windows, cast(server._People, People())) == fast


def test_exact_nanosecond_index_is_sorted_even_when_float_and_id_order_disagree():
    from liberational.vision import sightings

    epoch = 1_800_000_000.0
    rounded = math.nextafter(epoch, math.inf)
    late = {
        "id": "a",
        "ts_start": rounded,
        "ts_end": rounded,
        "_start_ns": 1800000000000000300,
        "_end_ns": 1800000000000000300,
    }
    early = {
        "id": "z",
        "ts_start": rounded,
        "ts_end": rounded,
        "_start_ns": 1800000000000000130,
        "_end_ns": 1800000000000000130,
    }
    index = sightings._WindowIndex([late, early])
    assert list(index.windows([(epoch, rounded)])) == [[early]]
    assert list(index.windows([(epoch, rounded + 1)])) == [[late, early]]  # legacy float/id return order


def test_face_snapshot_is_one_read_reused_until_the_next_request(tmp_path, vision, monkeypatch):
    from liberational.vision import sightings

    # Only this thread's reads count: the fake answers any thread, and a warm-up thread an earlier test's server
    # started (test_dashboard_warm) can read through it while this runs.
    here = threading.get_ident()
    mine = [0]
    read = vision.range

    def counted(*args, **kwargs):
        if threading.get_ident() == here:
            mine[0] += 1
        return read(*args, **kwargs)

    monkeypatch.setattr(vision, "range", counted)
    first = vision.face(B + 1, name="Dana")
    reads = mine[0]
    batch = sightings.FaceBatch(B, B + 10, tmp_path)
    initial = batch.faces([(B, B + 5)])
    assert initial is not None
    assert [f["id"] for f in initial] == [first["id"]]
    second = vision.face(B + 2, name="Alice")
    reused = batch.faces([(B, B + 10)])
    assert reused is not None
    assert [f["id"] for f in reused] == [first["id"]]
    assert mine[0] - reads == 1
    fresh = sightings.FaceBatch(B, B + 10, tmp_path)
    updated = fresh.faces([(B, B + 10)])
    assert updated is not None
    assert [f["id"] for f in updated] == [first["id"], second["id"]]


def test_an_unavailable_face_read_fails_instead_of_returning_an_empty_list(state, vision, monkeypatch):
    import continuum_py

    def unavailable(*args, **kwargs):
        raise continuum_py.ConnectionClosedError("broker unavailable")

    monkeypatch.setattr(vision, "range", unavailable)
    with pytest.raises(continuum_py.ConnectionClosedError, match="broker unavailable"):
        G._scan([5], state)


def test_a_recent_legacy_group_is_hidden_after_its_imported_sightings_are_deleted(state, vision, ids, monkeypatch):
    from liberational import segment_log

    monkeypatch.setattr(server, "STATE", state)
    monkeypatch.setattr(server, "_FACE_GROUP_LAST_SEEN", {})
    before = server._unknown_groups()
    assert before[0] == 200
    assert 6 in [g["id"] for g in before[1]["groups"]]
    # Delete the imported sightings. Its old aggregate has no update provenance and stays on disk.
    vision.records[segment_log.FACE_MATCHES] = [
        r for r in vision.records[segment_log.FACE_MATCHES] if r["id"] not in (ids[20], ids[21])
    ]
    after = server._unknown_groups()
    assert after[0] == 200
    assert [g["id"] for g in after[1]["groups"]] == [5, 8, 7]
    assert 6 in [g["id"] for g in fid.unknown_groups()]  # no guessed mutation of the shared aggregate
    assert 6 not in [g["id"] for g in server._unknown_groups()[1]["groups"]]  # no resurrection next refresh


def test_empty_group_filter_keeps_unknown_older_and_incomplete_history(state, vision, monkeypatch):
    from liberational.vision import sightings

    monkeypatch.setattr(server, "STATE", state)
    monkeypatch.setattr(server, "_FACE_GROUP_LAST_SEEN", {(str(state), 6): B + 50})
    rows = [{"id": 6, "last_seen": None}, {"id": 8, "last_seen": None}]
    incomplete = sightings.FaceBatch(0, (sightings.SHOT_DAYS + 2) * 86400, state)
    assert server._hide_emptied_face_groups(rows, incomplete) == rows
    old = sightings.FaceBatch(B + 1000, B + 2000, state)
    reads = vision.reads
    assert server._hide_emptied_face_groups(rows, old) == rows
    assert vision.reads == reads  # filtering never makes another broker read


def test_a_group_whose_retained_matches_are_already_named_is_not_asked_again(state, vision, ids, monkeypatch):
    from liberational import segment_log

    monkeypatch.setattr(server, "STATE", state)
    raw = next(r for r in vision.records[segment_log.FACE_MATCHES] if r["id"] == ids[20])
    raw["name"] = "Dana"  # the other retained match of Face 6 was already named Kara
    code, body = server._unknown_groups()
    assert code == 200
    assert 6 not in [g["id"] for g in body["groups"]]
    assert 6 in [g["id"] for g in fid.unknown_groups()]  # no mutation of the gallery


def test_a_group_resolved_by_owner_claims_is_not_asked_again(state, ids, monkeypatch):
    from liberational.domain.identity import Basis

    monkeypatch.setattr(server, "STATE", state)
    registry.at(state).claim(registry.detection(ids[20]), "Dana", Basis.OWNER_TAG, "test")
    code, body = server._unknown_groups()
    assert code == 200
    assert 6 not in [g["id"] for g in body["groups"]]


def test_mixed_named_and_low_confidence_unknown_evidence_still_counts_as_without_a_picture(
    state, vision, ids, monkeypatch
):
    from liberational import segment_log
    from liberational.people import face_shots

    monkeypatch.setattr(server, "STATE", state)
    monkeypatch.setattr(server, "_crop_jobs", face_shots.Pending(server._crop_process))
    raw = next(r for r in vision.records[segment_log.FACE_MATCHES] if r["id"] == ids[20])
    raw["det"] = 0.1  # not a crop candidate, but still unknown evidence in the mixed group
    face_shots.write(state, ids[1], "0", b"jpeg")
    face_shots.write(state, ids[1], "0", b"jpeg", "shots_grid")
    code, body = server._ready_face_groups(*server._unknown_groups())
    assert code == 200
    six = next(g for g in body["groups"] if g["id"] == 6)
    assert six["shots"] == []
    assert (six["shots_pending"], six["shots_unavailable"]) == (0, 0)
    assert body["groups_without_shots"] == 3  # Face 6 plus the two aggregates with no retained matches


def test_a_scan_limited_group_with_unknown_members_is_not_mistaken_for_resolved(state, monkeypatch):
    monkeypatch.setattr(server, "STATE", state)
    monkeypatch.setattr(G, "SCAN", 1)  # the newest Face 5 observation fills the picture scan
    code, body = server._unknown_groups()
    assert code == 200
    six = next(g for g in body["groups"] if g["id"] == 6)
    assert six["shots"] == []  # its older unknown member still keeps this group unresolved


def test_resolved_group_omission_requires_a_complete_snapshot(state):
    from liberational.vision import sightings

    rows = [{"id": 6, "shots": []}]
    incomplete = sightings.FaceBatch(0, (sightings.SHOT_DAYS + 2) * 86400, state)
    assert server._hide_resolved_face_groups(rows, incomplete) == rows
    assert server._hide_resolved_face_groups(rows, None) == rows


def test_ordinary_unknown_face_updates_during_warming_do_not_discard_the_kept_answer(state, monkeypatch):
    monkeypatch.setattr(server, "STATE", state)
    calls = []

    def compute():
        calls.append("read")
        ids, vecs, counts, nxt = fid._unk_load()
        fid._unk_save(ids, vecs, [n + 1 for n in counts], nxt)
        return 200, {"groups": [{"id": 5}]}

    kept = server._Fresh("face warm", compute, server._faces_owner_key, lambda: 0)
    assert kept.get() == (200, {"groups": [{"id": 5}]})
    assert kept.get() == (200, {"groups": [{"id": 5}]})
    assert calls == ["read"]


def test_group_removal_skips_and_identity_edits_invalidate_kept_face_rows(state, monkeypatch):
    monkeypatch.setattr(server, "STATE", state)
    before = server._faces_owner_key()
    ids, vecs, counts, nxt = fid._unk_load()
    fid._unk_save(ids[1:], vecs[1:], counts[1:], nxt)
    removed = server._faces_owner_key()
    assert removed != before
    server.INBOX_SKIPS.write_text(json.dumps({"face": [6]}))
    skipped = server._faces_owner_key()
    assert skipped != removed
    reg = registry.at(state)
    reg.ensure("Dana", "test")
    reg.rename("Dana", "Dee", "test")
    assert server._faces_owner_key() != skipped


def test_the_people_key_ignores_face_groups_the_worker_adds_and_evicts_but_not_naming(state, monkeypatch):
    monkeypatch.setattr(server, "STATE", state)
    before = server._faces_owner_key(groups=False)
    ids, vecs, counts, nxt = fid._unk_load()
    fid._unk_save([*ids[1:], nxt], np.vstack([vecs[1:], vecs[:1]]), [*counts[1:], 1], nxt + 1)  # one in, one out
    assert server._faces_owner_key(groups=False) == before
    reg = registry.at(state)
    reg.ensure("Dana", "test")
    reg.rename("Dana", "Dee", "test")
    assert server._faces_owner_key(groups=False) != before


def test_the_count_includes_groups_past_the_listed_ones(state, monkeypatch):
    monkeypatch.setattr(server, "STATE", state)
    baseline = server._unknown_groups()[1]["count"]
    ids, vecs, counts, _ = fid._unk_load()
    extra = list(range(100, 130))  # 30 more strangers, seen twice each, far from everyone
    axes = np.stack([at(0.0, 20 + k) for k in range(len(extra))])
    seen = dict.fromkeys(extra, time.time())  # seen just now
    fid._unk_save([*ids, *extra], np.vstack([vecs, axes]), [*counts, *[2] * len(extra)], 130, last_seen=seen)
    code, body = server._unknown_groups()
    assert code == 200
    assert len(body["groups"]) <= 24
    assert body["count"] == baseline + 30  # naming one of the listed groups must lower this, not pull in another


@pytest.fixture
def kara(state, monkeypatch):
    """The fixture's state with Kara known to the registry, as an owner tag of an earlier face left her."""
    from liberational.domain.identity import Basis

    monkeypatch.setattr(server, "STATE", state)
    reg = registry.at(state)
    reg.claim(registry.face_group(1), "Kara", Basis.OWNER_TAG, "test")
    return reg


def test_a_clear_match_is_named_automatically_without_teaching_the_gallery(kara, state):
    from liberational.domain.identity import Basis

    # Face 5 is 0.61 like Kara's only photo; Face 8 is too (0.58) but was put aside; Faces 6 and 7 are like nobody.
    named = fid.autoname(keep=[8], state=state)
    assert [(r["gid"], r["name"]) for r in named] == [(5, "Kara")]
    assert named[0]["undo"]
    assert fid.load_db()[0] == ["Kara"]  # no row from an automatic name, so a wrong one can't spread
    assert sorted(g["id"] for g in fid.unknown_groups()) == [6, 7, 8]
    decision = kara.person_of(registry.face_group(5))
    assert decision.person == kara.known_as("Kara")
    assert decision.winner is not None and decision.winner.basis == Basis.EMBEDDING_MATCH


def test_an_undone_automatic_name_brings_the_group_back_and_is_not_redone(kara, state):
    named = fid.autoname(keep=[8], state=state)
    code, body = server.undo_people({"undo": named[0]["undo"]})
    assert (code, body["ok"]) == (200, True)
    assert 5 in [g["id"] for g in fid.unknown_groups()]
    assert kara.person_of(registry.face_group(5)).person is None
    assert fid.autoname(keep=[8], state=state) == []  # the owner said no; it waits in Who's this


def test_a_group_marked_not_kara_is_never_named_kara_automatically(kara, state):
    """Face 5 clearly matches Kara, but the owner said it isn't her."""
    from liberational.people import face_nots

    face_nots.mark(state, "Kara", [5])
    assert fid.autoname(keep=[8], state=state) == []
    assert kara.person_of(registry.face_group(5)).person is None


def test_an_automatic_dry_run_names_nothing(kara, state):
    assert [r["gid"] for r in fid.autoname(keep=[8], apply=False, state=state)] == [5]
    assert 5 in [g["id"] for g in fid.unknown_groups()]
    assert kara.person_of(registry.face_group(5)).person is None


def track(vision, tid, kara_looks, group_looks=1, t=300.0, group=6):
    """Track `tid` on cam0: `group_looks` looks put in Face `group` (Face 6: like nobody), then `kara_looks` of Kara's
    face."""
    kara = at(0.61, 2)  # Kara's row in the gallery
    for k in range(group_looks + kara_looks):
        in_group = k < group_looks
        vision.add(
            segment_log.FACE_EMBEDDINGS,
            ts_start=B + t + k,
            cam="cam0",
            track_id=tid,
            group=subjects.group(f"Face {group}") if in_group else None,
            vector=(at(0.0, 3) if in_group else kara).tolist(),
        )


def test_a_group_whose_tracks_all_turned_out_to_be_one_person_is_named_them(kara, state, vision, caplog):
    # Face 6's faces are like nobody, but both tracks it was seen on were Kara once more of her face showed
    caplog.set_level("INFO")
    track(vision, "1", kara_looks=3)
    track(vision, "2", kara_looks=3, t=400.0)
    named = fid.autoname(keep=[8], state=state)
    assert [(r["gid"], r["name"], r.get("by")) for r in named] == [(5, "Kara", None), (6, "Kara", "tracks")]
    assert named[1]["undo"]
    assert kara.person_of(registry.face_group(6)).person == kara.known_as("Kara")
    lines = [r.getMessage() for r in caplog.records if "by its tracks" in r.getMessage()]
    assert len(lines) == 1 and "Kara" not in lines[0]


@pytest.mark.parametrize(
    ("tracks", "named"),
    [
        pytest.param([("1", 3)], False, id="one-short-track"),
        pytest.param([("1", 7)], True, id="one-track-of-eight-looks"),
        pytest.param([("1", 3), ("2", 0)], False, id="a-track-no-one"),
    ],
)
def test_a_groups_tracks_name_it_only_with_enough_and_unanimous_evidence(kara, state, vision, tracks, named):
    for k, (tid, looks) in enumerate(tracks):
        track(vision, tid, kara_looks=looks, t=300.0 + 100 * k)
    got = [r["gid"] for r in fid.autoname(keep=[8], state=state) if r.get("by") == "tracks"]
    assert got == ([6] if named else [])


def test_a_track_seen_under_a_group_folded_in_since_counts_for_the_group_it_went_into(kara, state, vision):
    # Face 40 was folded into Face 6 after its track was recorded (and that hour kept)
    track(vision, "1", kara_looks=3)
    track(vision, "2", kara_looks=3, t=400.0, group=40)
    assert [r for r in fid.autoname(keep=[8], apply=False, state=state) if r.get("by")] == []
    ids, vecs, counts, nxt, sums = fid._unk_load_sums()
    fid._unk_save(ids, vecs, counts, nxt, sums, aliases={40: 6})
    got = [r["gid"] for r in fid.autoname(keep=[8], apply=False, state=state) if r.get("by") == "tracks"]
    assert got == [6]


def test_a_track_across_an_hours_end_is_one_track(kara, state, vision):
    # Hours are summed apart; a track's looks either side of the hour are still its 8 looks
    hour_end = (B // 3600 + 1) * 3600
    track(vision, "1", kara_looks=7, t=hour_end - B - 1.5)
    got = [r["gid"] for r in fid.autoname(keep=[8], state=state) if r.get("by") == "tracks"]
    assert got == [6]


def test_a_pass_reads_again_only_the_hours_still_going(kara, state, vision):
    track(vision, "1", kara_looks=3)
    assert [r for r in fid.autoname(keep=[8], apply=False, state=state) if r.get("by")] == []
    first, vision.reads = vision.reads, 0
    assert [r for r in fid.autoname(keep=[8], apply=False, state=state) if r.get("by")] == []
    # the window's partial first hour and the hours not settled yet, of about 169
    assert vision.reads <= 3 < first
    since = time.time() - float(fid.cfg.get("faces.unnamed_days")) * 86400
    assert all(at >= since for s, at in fid._TRACK_HOURS if s == str(state))  # hours past the window are dropped


def test_a_look_written_late_into_a_kept_hour_counts_on_the_next_pass(kara, state, vision):
    track(vision, "1", kara_looks=3)
    assert [r for r in fid.autoname(keep=[8], apply=False, state=state) if r.get("by")] == []
    track(vision, "2", kara_looks=3, t=400.0)  # into an ended hour, after the pass kept it
    got = [r["gid"] for r in fid.autoname(keep=[8], apply=False, state=state) if r.get("by") == "tracks"]
    assert got == [6]


def test_a_match_to_a_name_the_registry_lacks_is_left_waiting(state):
    assert fid.autoname(keep=[8], state=state) == []
    assert 5 in [g["id"] for g in fid.unknown_groups()]


def test_groups_past_the_listed_ones_are_counted_only_while_still_asked(state, monkeypatch):
    monkeypatch.setattr(server, "STATE", state)
    baseline = server._unknown_groups()[1]["count"]
    ids, vecs, counts, _ = fid._unk_load()
    kara = fid.load_db()[1][0]
    extra = list(range(100, 130))  # 30 more groups, each Kara's own photo: a 100% guess, so never asked
    fid._unk_save([*ids, *extra], np.vstack([vecs, np.tile(kara, (len(extra), 1))]), [*counts, *[50] * 30], 130)
    code, body = server._unknown_groups()
    assert code == 200
    assert body["count"] == baseline == len(body["groups"])


@pytest.fixture
def taught(kara, state, vision):
    """Kara's photo is the first face of her tagged Face 1 (whose other faces lean to E1), she has a second tagged
    group (Face 3, all E1), and Dana's only tag, Face 2, is faces just like Kara's: a likely mis-tag."""
    from liberational.domain.identity import Basis

    row = fid.load_db()[1][0]
    dana = at(0.3, 9)
    fid.save_db(["Kara", "Dana"], np.stack([row, dana]))
    kara.claim(registry.face_group(2), "Dana", Basis.OWNER_TAG, "test")
    kara.claim(registry.face_group(3), "Kara", Basis.OWNER_TAG, "test")
    for gid, vectors in ((1, [row, E1, E1]), (2, [dana, E1, E1, E1]), (3, [E1, E1, at(0.9, 7)])):
        for k, v in enumerate(vectors):
            vision.add(segment_log.FACE_EMBEDDINGS, ts_start=B + gid * 10 + k, cam="cam0", group=_key(gid), vector=v)
    return kara


def _key(gid):
    return str(registry.subject_uuid(registry.face_group(gid)))


def test_the_held_out_check_reads_the_owners_tags_and_names_the_likely_mis_tag(taught, state):
    check = fid.refresh_check(state)
    assert (check["groups"], check["status"]) == (3, "unchecked")  # too few tagged groups to judge the gate yet
    assert [(s["gid"], s["tagged"], s["looks_like"]) for s in check["suspects"]] == [(2, "Dana", "Kara")]
    assert fid.trusted()


def test_a_failed_check_stops_naming_by_appearance_until_it_passes(kara, state):
    fid.CHECK.write_text(json.dumps({"status": "off"}))
    assert fid.match(fid.load_db()[1][0]) == (None, pytest.approx(1.0))  # Kara's own photo, but not trusted
    assert fid.autoname(keep=[8], state=state) == []
    assert 5 in [g["id"] for g in fid.unknown_groups()]
    fid.CHECK.write_text(json.dumps({"status": "on"}))
    assert fid.match(fid.load_db()[1][0])[0] == "Kara"
    assert [r["gid"] for r in fid.autoname(keep=[8], state=state)] == [5]


def test_a_rebuild_reenrolls_tagged_rows_as_centroids_and_names_clear_groups(taught, state):
    done = fid.rebuild(keep=[8], state=state)
    names, rows = fid.load_db()
    assert names == ["Kara"]  # Dana's row, from a group that looks like Kara, leaves the gallery
    assert float(rows[0] @ E1) > 0.95  # Kara's row is her group's centroid now, not its first face (0.61)
    assert [(s["gid"], s["tagged"], s["looks_like"]) for s in done["suspects"]] == [(2, "Dana", "Kara")]
    assert taught.person_of(registry.face_group(2)).person == taught.known_as("Dana")  # the owner's tag stays
    assert [(r["gid"], r["name"]) for r in done["named"]] == [(5, "Kara")]
    assert 5 not in [g["id"] for g in fid.unknown_groups()]


def test_a_rebuild_dry_run_changes_nothing(taught, state):
    before = fid.load_db()
    done = fid.rebuild(keep=[8], apply=False, state=state)
    assert [(r["gid"], r["name"]) for r in done["named"]] == [(5, "Kara")]
    assert fid.load_db()[0] == before[0] and np.array_equal(fid.load_db()[1], before[1])
    assert 5 in [g["id"] for g in fid.unknown_groups()]


def test_a_group_marked_not_someone_never_has_them_as_its_guess(state):
    """Face 5 looks like Kara, but the owner said it isn't her."""
    (five,) = G.groups([{"id": 5, "count": 12}], state)
    assert (five["guess"]["id"], five["guess"]["name"], five["not"]) == ("Kara", "Kara", [])
    (five,) = G.groups([{"id": 5, "count": 12}], state, nots={5: {"kara"}})
    assert (five["guess"], five["band"], five["not"]) == (None, "none", ["kara"]), "Kara was the only one named"


def test_rebuild_leaves_out_a_recorded_look_whose_signal_says_it_hardly_read_as_a_face(vision, state):
    # Recorded before head_faces set such looks aside: a turned-away look in Face 5 with a low norm.
    for k, (mid, norm, nose) in enumerate((("m-clear", 20.0, 0.9), ("m-away", 7.0, 0.9), ("m-blur", 20.0, 0.2))):
        vision.add(
            segment_log.FACE_EMBEDDINGS, ts_start=B + k, cam="cam0", group=_key(5), match_id=mid, vector=at(0.9, 2 + k)
        )
        vision.add(
            segment_log.FACE_SIGNALS,
            ts_start=B + k,
            cam="cam0",
            match_id=mid,
            head=[nose, 0.9, 0.9, 0.9, 0.9],
            norm=norm,
        )
    vision.add(segment_log.FACE_EMBEDDINGS, ts_start=B + 5, cam="cam0", group=_key(5), match_id="m-old", vector=E1)
    members = fid._members(state, fid.REBUILD_DAYS)[_key(5)]
    # Left: the clear look, and the old one with no signal (which can't be judged).
    assert members @ at(0.9, 2) == pytest.approx([1.0, 0.9], abs=1e-5)
