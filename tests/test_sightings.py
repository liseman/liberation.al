import numpy as np
import pytest

from liberational import segment_log
from liberational.domain.identity import Basis
from liberational.people import registry
from liberational.vision import faces, sightings

T = 1_800_000_000.0


@pytest.fixture
def now(monkeypatch):
    monkeypatch.setattr(sightings, "_now", lambda: T + 100)


def test_a_face_is_its_match_unless_the_registry_says_otherwise(tmp_path, vision, now):
    a = vision.face(T, name="Kara", sim=0.6)
    b = vision.face(T + 1, name="Kara", sim=0.6)
    c = vision.face(T + 2, name=None, cluster="Face 3")
    reg = registry.at(tmp_path)
    reg.claim(registry.detection(b["id"]), "Luca", Basis.OWNER_TAG, "dashboard")  # "Not Kara, it's Luca"
    sightings.correct([a["id"]], None, tmp_path)  # "Not Kara", nobody they know
    found = {f["id"]: f for f in sightings.faces(T - 1, T + 10, state=tmp_path)}
    assert (found[a["id"]]["name"], found[b["id"]]["name"], found[c["id"]]["name"]) == (None, "Luca", None)
    assert found[b["id"]]["sim"] == 1.0, "the owner's word is sure"
    assert found[c["id"]]["cluster"] == "Face 3"


def test_a_correction_and_its_undo(tmp_path, vision, now):
    d = vision.face(T, name="Kara", sim=0.6)
    prior = sightings.correct([d["id"]], "Luca", tmp_path)
    assert prior == [{"id": d["id"], "was": None}]
    assert sightings.faces(T - 1, T + 1, state=tmp_path)[0]["name"] == "Luca"
    sightings.uncorrect(prior, tmp_path)
    assert sightings.faces(T - 1, T + 1, state=tmp_path)[0]["name"] == "Kara", "back to the face worker's match"


def test_not_sure_takes_a_face_off_its_match(tmp_path, vision, now):
    d = vision.face(T, name="Kara", sim=0.6)
    sightings.correct([d["id"]], None, tmp_path)
    assert sightings.faces(T - 1, T + 1, state=tmp_path)[0]["name"] is None


def test_a_face_moved_to_someone_deleted_later_stays_off_its_match(tmp_path, vision, now):
    d = vision.face(T, name="Kara", sim=0.6)
    sightings.correct([d["id"]], "Luca", tmp_path)
    registry.at(tmp_path).forget("Luca", "dashboard")
    assert sightings.faces(T - 1, T + 1, state=tmp_path)[0]["name"] is None, "still not Kara, as v1 kept it"


def test_visits_who_was_there_and_when_each_was_last_seen(tmp_path, vision, now):
    vision.visit("Kara", T, T + 20, cam="cam0", best_sim=0.6)
    vision.visit("Kara", T + 50, T + 60, cam="cam1", best_sim=0.7)
    vision.visit("Luca", T + 10, T + 15, cam="cam0", best_sim=0.5)
    assert [(v["cam"], v["end"]) for v in sightings.visits("kara", state=tmp_path)] == [
        ("cam1", T + 60),
        ("cam0", T + 20),
    ]
    assert [(w["name"], w["cam"]) for w in sightings.who_at(T + 12, 5, tmp_path)] == [
        ("Kara", "cam0"),
        ("Luca", "cam0"),
    ]
    assert sightings.last_seen_where(tmp_path) == {"kara": (T + 60, "cam1"), "luca": (T + 15, "cam0")}
    assert sightings.last_seen("Nobody", tmp_path) is None


def test_shots_are_thinned_per_second_and_ranked_by_quality(tmp_path, vision, now):
    low = vision.face(T + 1.0, name="Kara", sim=0.5, quality=0.2)
    high = vision.face(T + 1.4, name="Kara", sim=0.7, quality=0.9)  # the same second: the better match is kept
    later = vision.face(T + 5, name="Kara", sim=0.6, quality=0.5)
    vision.face(T + 6, name="Kara", sim=0.6, quality=0.0)  # a profile: never a best shot
    for ts in (T + 1.0, T + 1.4, T + 5, T + 6):  # each its own visit, so each is a photo
        vision.visit("Kara", ts, ts)
    page = sightings.shots("Kara", state=tmp_path)
    assert page["total"] == 3
    assert [s["id"] for s in page["shots"]][:2] == [vision.records[segment_log.FACE_MATCHES][3]["id"], later["id"]]
    assert high["id"] in [s["id"] for s in page["shots"]] and low["id"] not in [s["id"] for s in page["shots"]]
    # Spread over days, and a minute apart on one camera within a day: these are seconds apart, so one.
    assert [r["id"] for r in sightings.best_spread("Kara", 6, tmp_path)] == [high["id"]]
    second = sightings.best("Kara", tmp_path, skip={high["id"]})
    assert second is not None and second["id"] == later["id"]


def test_unnamed_clips_are_split_by_face_group(tmp_path, vision, now):
    for i in range(3):
        vision.face(T + i, name=None, cluster="Face 3", quality=0.4)
        vision.face(T + i, name=None, cluster="Face 4", quality=0.6)
    vision.face(T + 40, name="Kara", sim=0.6)
    clips = sightings.unnamed_clips(10, state=tmp_path)
    assert sorted((len(c["ids"]), c["cam"]) for c in clips) == [(3, "cam0"), (3, "cam0")]
    assert len({c["id"] for c in clips}) == 2, "two faces starting together get their own ids"


def test_a_match_follows_its_person_through_a_rename_and_is_nobody_once_they_are_forgotten(tmp_path, vision, now):
    reg = registry.at(tmp_path)
    reg.ensure("Kara", "test")
    reg.ensure("Ann", "test")
    vision.face(T, name="Kara", sim=0.6)
    vision.face(T + 1, name="Ann", sim=0.6)
    vision.face(T + 2, name="Zed", sim=0.6)  # a name the registry never had: kept as the match gave it
    reg.rename("Kara", "Katherine", "test")
    reg.forget("Ann", "test")
    assert [f["name"] for f in sightings.faces(T - 1, T + 10, state=tmp_path)] == [
        "Katherine",
        None,
        "Zed",
    ]


def test_an_unnamed_face_is_who_its_group_was_named_since(tmp_path, vision, now):
    vision.face(T, name=None, cluster="Face 3")
    vision.face(T + 1, name=None, cluster="Face 4")
    registry.at(tmp_path).claim(registry.face_group(3), "Ann", Basis.INTRODUCTION, "identity")
    assert [f["name"] for f in sightings.faces(T - 1, T + 10, state=tmp_path)] == ["Ann", None]


def test_moving_a_visit_s_faces_moves_the_visit(tmp_path, vision, now):
    ids = [vision.face(T + s, name="Kara", sim=0.6)["id"] for s in (0, 1, 2)]
    vision.visit("Kara", T, T + 2, samples=3)
    vision.face(T + 50, name="Kara", sim=0.6)
    vision.visit("Kara", T + 50, T + 50)  # another visit, untouched
    sightings.correct(ids, "Eli", tmp_path)
    assert [(v["start"], v["end"]) for v in sightings.visits("Kara", state=tmp_path)] == [(T + 50, T + 50)]
    assert [(v["start"], v["end"], v["samples"]) for v in sightings.visits("Eli", state=tmp_path)] == [(T, T + 2, 3)]
    assert sightings.last_seen("Eli", tmp_path) == T + 2


def test_not_x_on_one_face_of_a_visit_keeps_the_rest_of_it(tmp_path, vision, now):
    ids = [vision.face(T + s, name="Kara", sim=0.6)["id"] for s in (0, 1, 2)]
    vision.visit("Kara", T, T + 2, samples=3)
    sightings.correct([ids[2]], None, tmp_path)
    assert [(v["start"], v["end"], v["samples"]) for v in sightings.visits("Kara", state=tmp_path)] == [(T, T + 1, 2)]


def test_removing_a_shot_named_by_its_face_group_takes_it_off_them(tmp_path, vision, now):
    d = vision.face(T, name=None, cluster="Face 3")
    registry.at(tmp_path).claim(registry.face_group(3), "Ann", Basis.INTRODUCTION, "identity")
    assert sightings.faces(T - 1, T + 1, state=tmp_path)[0]["name"] == "Ann"
    sightings.correct([d["id"]], None, tmp_path)
    assert sightings.faces(T - 1, T + 1, state=tmp_path)[0]["name"] is None


def test_a_shot_is_found_by_the_time_in_its_id(tmp_path, vision, now):
    d = vision.face(T - 40 * 86400, name="Kara", sim=0.6)  # older than the SHOT_DAYS a timeless lookup would search
    assert sightings.shot(d["id"], tmp_path) == {
        "id": d["id"],
        "ts": T - 40 * 86400,
        "cam": "cam0",
        "bbox_norm": d["bbox"],
    }


def test_a_read_looks_up_only_the_faces_the_registry_has_a_word_on(tmp_path, vision, now, monkeypatch):
    reg = registry.at(tmp_path)
    reg.ensure("Kara", "test")
    ids = [vision.face(T + s, name="Kara", sim=0.6)["id"] for s in range(20)]
    reg.claim(registry.detection(ids[3]), "Luca", Basis.OWNER_TAG, "dashboard")
    asked = []
    person_of = registry.Registry.person_of

    def counted(self, subject, via=None):
        asked.append(subject.id)
        return person_of(self, subject, via)

    monkeypatch.setattr(registry.Registry, "person_of", counted)
    names = [f["name"] for f in sightings.faces(T - 1, T + 30, state=tmp_path)]
    assert names == ["Kara"] * 3 + ["Luca"] + ["Kara"] * 16
    assert asked == [f"detection:{ids[3]}"]


def test_a_person_s_photos_are_their_visits_shots_without_reading_every_face(tmp_path, vision, now, monkeypatch):
    box = [0.1, 0.1, 0.3, 0.4]
    shots = [
        {"detection": f"d{n}", "ts": T + n, "bbox": box, "quality": q, "sim": 0.7} for n, q in ((1, 0.4), (2, 0.9))
    ]
    vision.add(
        segment_log.FACE_VISITS, ts_start=T, ts_end=T + 5, name="Kara", cam="cam0", best_sim=0.7, samples=9, shots=shots
    )
    ranges = []
    real = vision.range
    monkeypatch.setattr(vision, "range", lambda stream, *window: ranges.append(stream) or real(stream, *window))
    best = sightings.best("Kara", tmp_path)
    assert best is not None and (best["id"], best["score"]) == ("d2", 0.9)
    assert [s["id"] for s in sightings.shots("Kara", state=tmp_path)["shots"]] == ["d2", "d1"]
    assert segment_log.FACE_MATCHES not in ranges


def test_a_visit_from_before_shots_has_its_faces_read_once(tmp_path, vision, now, monkeypatch):
    monkeypatch.setattr(sightings, "_OLD_SHOTS", {})
    vision.face(T + 1, name="Kara", sim=0.6, quality=0.7, bbox=[0.1, 0.1, 0.2, 0.2])
    vision.visit("Kara", T, T + 5)  # no shots: committed before visits carried them
    ranges = []
    real = vision.range
    monkeypatch.setattr(vision, "range", lambda stream, *window: ranges.append(stream) or real(stream, *window))
    first = sightings.best("Kara", tmp_path)
    again = sightings.best("Kara", tmp_path)
    assert first == again and first is not None
    assert ranges.count(segment_log.FACE_MATCHES) == 1, "the second read takes them from the first"


def test_shared_clip_rows_keep_named_faces_and_known_empty_avoids_rereading(tmp_path, vision, now):
    vision.face(T, cluster="Face 3")
    vision.face(T + 1, name="Dana", sim=0.8)
    lone = vision.face(T + 100, cluster="Face 4")
    rows = sightings.faces(T - 1, T + 200, state=tmp_path)
    vision.reads = 0
    clips = sightings.unnamed_clips(10, state=tmp_path, rows=rows)
    assert [c["ids"] for c in clips] == [[lone["id"]]]
    assert sightings.unnamed_clips(10, state=tmp_path, rows=[]) == []
    assert vision.reads == 0


def test_faces_keep_exact_window_and_camera(tmp_path, vision):
    vision.face(T - 1, name="Before")
    first = vision.face(T, name=None)
    second = vision.face(T + 1, name="Dana", sim=0.9)
    vision.face(T + 1, cam="cam1", name="Other camera")
    vision.face(T + 2, name="At the excluded end")
    rows = sightings.faces(T, T + 2, "cam0", tmp_path)
    assert [(r["id"], r["ts"], r["name"], r["sim"]) for r in rows] == [
        (first["id"], T, None, None),
        (second["id"], T + 1, "Dana", 0.9),
    ]


def test_a_face_batch_slices_its_window_and_names_faces_afresh(tmp_path, vision):
    kept = vision.face(T + 1, name="Dana")
    vision.face(T + 11, name="Alice")  # in the batch, outside the window asked for
    rows = sightings.FaceBatch(T, T + 20, tmp_path).faces([(T, T + 10)])
    assert rows is not None
    assert [(row["id"], row["name"]) for row in rows] == [(kept["id"], "Dana")]
    sightings.correct([kept["id"]], "Alice", tmp_path)
    rows = sightings.FaceBatch(T, T + 20, tmp_path).faces([(T, T + 10)])
    assert rows is not None
    assert [(row["id"], row["name"]) for row in rows] == [(kept["id"], "Alice")]


def test_a_face_batch_window_keeps_half_open_spans_and_point_faces(tmp_path, vision):
    vision.face(T, ts_end=T + 2, name="Alice")  # ends where the window starts
    span = vision.face(T + 1, ts_end=T + 4, name="Alice")
    point = vision.face(T + 2, name="Dana")
    vision.face(T + 3, name="Dana")  # starts where the window ends
    rows = sightings.FaceBatch(T, T + 5, tmp_path).faces([(T + 2, T + 3)])
    assert rows is not None
    assert [row["id"] for row in rows] == [span["id"], point["id"]]


def _signal(vision, look, norm, nose=0.9):
    vision.add(
        segment_log.FACE_SIGNALS,
        ts_start=look["ts_start"],
        cam="cam0",
        match_id=look["id"],
        head=[nose, 0.9, 0.9, 0.9, 0.9],
        norm=norm,
    )


def test_shots_rank_by_how_much_they_read_as_a_face_and_backs_of_heads_are_left_out(tmp_path, vision, now):
    # Landmarks call a back of the head straight on, so quality can't tell (#2355): the norm ranks them.
    looks = {
        key: vision.face(T - 1000 + 70 * i, name="Kara", sim=0.6, quality=q)
        for i, (key, q) in enumerate(
            (
                ("face", 0.3),
                ("profile", 0.5),
                ("back", 0.9),
                ("blur", 0.7),
                ("old_face", 0.8),
                ("old_back", 0.95),
                ("unscored", 0.6),
            )
        )
    }
    for look in looks.values():
        vision.visit("Kara", look["ts_start"], look["ts_start"])
    _signal(vision, looks["face"], 20.0)
    _signal(vision, looks["profile"], 13.0)
    _signal(vision, looks["back"], 7.0)  # under the gate's norm
    _signal(vision, looks["blur"], 20.0, nose=0.2)  # under the gate's nose confidence
    (tmp_path / sightings.SHOT_NORMS).write_text(
        f'{{"{looks["old_face"]["id"]}": 18.0, "{looks["old_back"]["id"]}": 0.0}}'  # no face found on its crop
    )
    order = ["face", "old_face", "profile", "unscored"]
    assert [r["id"] for r in sightings.ranked("Kara", tmp_path)] == [looks[k]["id"] for k in order]
    assert [r["id"] for r in sightings.best_spread("Kara", 6, tmp_path)] == [looks[k]["id"] for k in order]


def test_newly_scored_crops_are_read_without_a_restart(tmp_path, vision, now):
    old = vision.face(T, name="Kara", sim=0.6, quality=0.9)
    other = vision.face(T + 70, name="Kara", sim=0.6, quality=0.2)
    for look in (old, other):
        vision.visit("Kara", look["ts_start"], look["ts_start"])
    assert [r["id"] for r in sightings.ranked("Kara", tmp_path)] == [old["id"], other["id"]]
    (tmp_path / sightings.SHOT_NORMS).write_text(f'{{"{old["id"]}": 0.0, "{other["id"]}": 19.0}}')
    assert [r["id"] for r in sightings.ranked("Kara", tmp_path)] == [other["id"]]


def test_an_owner_correction_teaches_the_gallery_and_its_undo_takes_it_back(tmp_path, vision, now, monkeypatch):
    # #3013: the owner's word on a face makes it one of the person's rows, as naming a group does
    monkeypatch.setattr(faces, "DB", tmp_path / "db.npz")
    monkeypatch.setattr(faces, "LEARNED_DB", tmp_path / "learned.npz")
    d = vision.face(T, name="Kara", sim=0.6)
    vec = np.zeros(512, np.float32)
    vec[7] = 1.0
    vision.add(segment_log.FACE_EMBEDDINGS, ts_start=T, cam="cam0", match_id=d["id"], vector=vec.tolist())
    prior = sightings.correct([d["id"]], "Luca", tmp_path)
    names, embs = faces.load_db()
    assert names == ["Luca"] and float(embs[0] @ vec) == pytest.approx(1.0)
    sightings.uncorrect(prior, tmp_path)
    assert faces.load_db()[0] == []


def test_not_kara_on_a_shot_makes_its_face_her_negative_and_its_undo_takes_it_back(tmp_path, vision, now, monkeypatch):
    # #3246: "Not <name>" teaches the matcher that this face isn't them
    monkeypatch.setattr(faces, "DB", tmp_path / "db.npz")
    monkeypatch.setattr(faces, "LEARNED_DB", tmp_path / "learned.npz")
    monkeypatch.setattr(faces, "NOTS_DB", tmp_path / "nots.npz")
    d = vision.face(T, name="Kara", sim=0.6)
    vec = np.zeros(512, np.float32)
    vec[7] = 1.0
    vision.add(segment_log.FACE_EMBEDDINGS, ts_start=T, cam="cam0", match_id=d["id"], vector=vec.tolist())
    prior = sightings.correct([d["id"]], None, tmp_path)
    names, embs, keys = faces.load_nots()
    assert (names, keys) == (["Kara"], [f"shot:{d['id']}"]) and float(embs[0] @ vec) == pytest.approx(1.0)
    sightings.uncorrect(prior, tmp_path)
    assert faces.load_nots()[0] == []
