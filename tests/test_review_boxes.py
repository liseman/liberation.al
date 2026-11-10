"""Review's boxes: the people and things one camera saw during a short stretch, from bounded reads."""

import pytest

from liberational import segment_log
from liberational.dashboard import review_boxes as rb
from liberational.vision import animal_embedder

T0 = 1_793_000_000.0


def face(vision, ts, name, box=(0.4, 0.2, 0.5, 0.35), cam="cam0"):
    vision.face(ts, cam, name, 0.6 if name else None, bbox=list(box))


def owner_crop(state, at: float, **fields) -> None:
    """An owner's object crop, written where object memory keeps them: the knowledge store."""
    import uuid

    from liberational import knowledge
    from liberational.people import registry

    ns = round(at * 1e9)
    value = {"ts_start": at, **fields}
    store = knowledge.Store(registry.knowledge_path(state))
    store.put([knowledge.Record(str(uuid.uuid7()), "object.sighting", str(fields["object_id"]), ns, ns, ns, value)])


@pytest.fixture
def state(tmp_path, vision):
    face(vision, T0 + 1, "Dana")
    face(vision, T0 + 2, None, box=(0.7, 0.1, 0.8, 0.25))  # a face nobody has named
    face(vision, T0 + 3, "Dana", cam="cam1")  # another camera
    face(vision, T0 + 500, "Dana")  # long after the window
    # object memory's sightings: one, and one whose box isn't a box
    owner_crop(tmp_path, T0 + 4, cam="cam0", object_id="o1", name="Coffee maker", bbox=[0.6, 0.5, 0.2, 0.3])
    owner_crop(tmp_path, T0 + 6, cam="cam0", object_id="o1", name="Coffee maker", bbox="not json")
    return tmp_path


def test_one_camera_s_faces_and_things_in_the_window_in_time_order(state):
    got = rb.boxes("cam0", T0, T0 + 60, state)["boxes"]
    assert [(b["ts"], b["kind"], b["label"], b["tone"]) for b in got] == [
        (T0 + 1, "face", "Dana", "known"),
        (T0 + 2, "face", "Who's this?", "unknown"),
        (T0 + 4, "object", "Coffee maker", "known"),
    ]
    assert got[0]["box"] == [0.4, 0.2, 0.5, 0.35]
    assert got[0]["id"] == "face:dana"  # one id per person, so their box moves rather than flickers
    assert got[2]["box"] == [0.6, 0.5, 0.8, 0.8]  # object memory's [x, y, w, h] as corners


def test_the_object_worker_s_matches_from_the_segment_store_are_boxes_beside_the_owner_s_crops(state, vision):
    vision.add(
        "object.match",
        ts_start=T0 + 3,
        cam="cam0",
        object_id="o2",
        name="Wallet",
        bbox=[0.1, 0.1, 0.2, 0.2],
        kind="visual_match",
        bbox_space="source",
    )
    got = [(b["ts"], b["label"]) for b in rb.boxes("cam0", T0, T0 + 60, state)["boxes"] if b["kind"] == "object"]
    assert got == [(T0 + 3, "Wallet"), (T0 + 4, "Coffee maker")]


def test_reads_at_most_the_longest_window(state):
    got = rb.boxes("cam0", T0, T0 + 10_000, state)
    assert got["end"] == T0 + rb.WINDOW_MAX_S
    assert all(b["ts"] < T0 + rb.WINDOW_MAX_S for b in got["boxes"])


def test_nothing_recorded_gives_no_boxes(tmp_path):
    assert rb.boxes("cam0", T0, T0 + 60, tmp_path)["boxes"] == []


@pytest.mark.parametrize("cam", ["entryway", "front-door", "FaceCam_A", "7North", "cam00", "CAM0", "0"])
def test_named_camera_reads_only_its_own_evidence(tmp_path, vision, cam):
    face(vision, T0 + 1, "Dana", cam=cam)
    face(vision, T0 + 2, "Alice", cam="cam0")
    got = rb.boxes(cam, T0, T0 + 60, tmp_path)
    assert got["cam"] == cam
    assert [b["label"] for b in got["boxes"]] == ["Dana"]


@pytest.mark.parametrize(
    "cam, start, end",
    [
        pytest.param("../etc", T0, T0 + 1, id="not-a-camera"),
        pytest.param("front.door", T0, T0 + 1, id="stream-suffix-not-camera"),
        pytest.param("front door", T0, T0 + 1, id="space"),
        pytest.param("cam0\n", T0, T0 + 1, id="newline"),
        pytest.param("cámara", T0, T0 + 1, id="non-ascii"),
        pytest.param("", T0, T0 + 1, id="missing-camera"),
        pytest.param("cam0", T0, T0, id="empty"),
        pytest.param("cam0", T0 + 5, T0, id="backwards"),
    ],
)
def test_refuses_a_bad_camera_or_times(tmp_path, cam, start, end):
    with pytest.raises(ValueError):
        rb.boxes(cam, start, end, tmp_path)


def test_an_unknown_face_that_stays_keeps_one_box(tmp_path, vision):
    face(vision, T0 + 1, None, box=(0.40, 0.20, 0.50, 0.35))
    face(vision, T0 + 1.5, None, box=(0.41, 0.21, 0.51, 0.36))  # moved a little: the same face
    face(vision, T0 + 2, None, box=(0.80, 0.60, 0.90, 0.75))  # somewhere else: another face
    ids = [b["id"] for b in rb.boxes("cam0", T0, T0 + 60, tmp_path)["boxes"]]
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]


@pytest.mark.parametrize(
    ("cluster", "want"),
    [
        pytest.param("Face 12", {"group": 12}, id="face-group"),
        pytest.param("Face 0", {"group": 0}, id="group-zero"),
        pytest.param("Dana", {}, id="a-name-is-not-a-group"),
        pytest.param("Face 12b", {}, id="not-a-number"),
        pytest.param(None, {}, id="no-cluster"),
        pytest.param(12, {}, id="not-a-string"),
    ],
)
def test_an_unknown_face_carries_its_face_group_for_naming_in_review(cluster: object, want: dict[str, int]) -> None:
    assert rb._group(cluster) == want


def test_spatially_grouped_unknown_boxes_keep_their_original_face_tracks(tmp_path, vision):
    vision.face(
        T0 + 1,
        cluster="Face 5",
        track_id="run:cam0:7",
        bbox=[0.40, 0.20, 0.50, 0.35],
    )
    vision.face(
        T0 + 1.5,
        cluster="Face 5",
        track_id="run:cam0:8",
        bbox=[0.41, 0.21, 0.51, 0.36],
    )
    got = rb.boxes("cam0", T0, T0 + 60, tmp_path)["boxes"]
    assert [b["id"] for b in got] == ["face:?1", "face:?1"]
    assert [(b["ts"], b["group"], b["track"]) for b in got] == [
        (T0 + 1, 5, "run:cam0:7"),
        (T0 + 1.5, 5, "run:cam0:8"),
    ]


@pytest.mark.parametrize("track", [None, "", " \t", 12, {}])
def test_faces_without_a_valid_original_track_never_invent_one(tmp_path, vision, track):
    vision.face(T0 + 1, cluster="Face 5", track_id=track)
    [box] = rb.boxes("cam0", T0, T0 + 60, tmp_path)["boxes"]
    assert box["group"] == 5
    assert "track" not in box


@pytest.mark.parametrize("provenance", [{"bbox_space": "decoded"}, {"kind": "manual_label"}])
def test_legacy_imported_decoded_boxes_are_not_projected_as_source_coordinates(state, vision, provenance):
    owner_crop(
        state,
        T0 + 7,
        cam="cam0",
        object_id="imported",
        name="Wallet",
        bbox=[0.2, 0.2, 0.3, 0.3],
        **provenance,
    )
    objects = [row for row in rb.boxes("cam0", T0, T0 + 60, state)["boxes"] if row["kind"] == "object"]
    assert [row["label"] for row in objects] == ["Coffee maker"]


@pytest.fixture
def animal_store(tmp_path, monkeypatch):
    log = segment_log.Log(segment_log.root(tmp_path))
    monkeypatch.setattr(segment_log, "at", lambda state=None: log)
    return tmp_path


def animal(state, ts, track="run:cat:1", species="cat", cam="cam0", box=(0.1, 0.2, 0.4, 0.6, 0.9)):
    import uuid

    writers = segment_log.Writers(segment_log.root(state))
    writers.append(
        segment_log.ANIMAL_DETECTIONS,
        cam,
        [
            {
                "ts_ns": round(ts * 1e9),
                "frame_id": str(uuid.uuid7()),
                "track_id": track,
                "species": species,
                "box": list(box),
                "verified": True,
                "model": "test",
            }
        ],
    )
    writers.close()


def pet_label(state, track, label, ts=T0, species="cat", active=True):
    import uuid

    from liberational import knowledge
    from liberational.people import pet_recognition, registry

    ns = round(ts * 1e9)
    knowledge.Store(registry.knowledge_path(state)).put(
        [
            knowledge.Record(
                str(uuid.uuid7()),
                pet_recognition.LABELS,
                track,
                ns,
                ns,
                ns,
                {
                    "track_id": track,
                    "label": label,
                    "active": active,
                    "source": "owner",
                    "species": species,
                    "cam": "cam0",
                    "sighting_start": T0,
                },
            )
        ]
    )


def test_pet_boxes_read_one_camera_window_with_source_coordinates_and_stable_track_ids(animal_store):
    animal(animal_store, T0 + 1)
    animal(animal_store, T0 + 1.5, box=(0.11, 0.2, 0.41, 0.6, 0.9))
    animal(animal_store, T0 + 2, track="run:cat:2")
    animal(animal_store, T0 + 3, cam="cam1")
    animal(animal_store, T0 + 120)
    response = rb.boxes("cam0", T0, T0 + 1000, animal_store)
    got = response["boxes"]
    assert response["end"] == T0 + 120
    assert [(b["ts"], b["id"], b["kind"], b["label"], b["tone"]) for b in got] == [
        (T0 + 1, "pet:run:cat:1", "pet", "cat", "unknown"),
        (T0 + 1.5, "pet:run:cat:1", "pet", "cat", "unknown"),
        (T0 + 2, "pet:run:cat:2", "pet", "cat", "unknown"),
    ]
    assert got[0]["box"] == [0.1, 0.2, 0.4, 0.6]
    assert got[1]["box"] == [0.11, 0.2, 0.41, 0.6]
    assert got[0]["species"] == "cat"
    assert got[0]["track"] == "run:cat:1"
    assert "pet_id" not in got[0]
    assert "name" not in got[0]
    assert rb.boxes("cam0", T0 + 1.1, T0 + 2, animal_store)["boxes"][0]["id"] == "pet:run:cat:1"


def test_owner_track_label_uses_current_pet_name_and_corrected_species_without_changing_track_identity(animal_store):
    from liberational.people import pets

    profile = pets.at(animal_store).save("Mia", "dog")
    animal(animal_store, T0 + 1)  # the detector called the owner's dog a cat
    pet_label(animal_store, "run:cat:1", profile["id"], species="dog")
    [before] = rb.boxes("cam0", T0, T0 + 60, animal_store)["boxes"]
    assert before == {
        "ts": T0 + 1,
        "id": "pet:run:cat:1",
        "kind": "pet",
        "box": [0.1, 0.2, 0.4, 0.6],
        "label": "Mia · dog",
        "tone": "known",
        "track": "run:cat:1",
        "species": "dog",
        "pet_id": profile["id"],
        "name": "Mia",
    }
    pets.at(animal_store).save("Pip", "dog", profile["id"])
    [after] = rb.boxes("cam0", T0, T0 + 60, animal_store)["boxes"]
    assert (after["name"], after["label"], after["id"]) == ("Pip", "Pip · dog", before["id"])


@pytest.mark.parametrize("status", ["unlabeled", "other", "ignored", "cleared", "retired"])
def test_a_pet_profile_does_not_turn_unconfirmed_or_retired_track_identity_into_a_name(animal_store, status):
    from liberational.people import pets

    profile = pets.at(animal_store).save("Mia", "cat")
    animal(animal_store, T0 + 1)
    labels = {
        "unlabeled": ("", False),
        "other": ("other", True),
        "ignored": ("ignored", True),
        "cleared": (profile["id"], False),
        "retired": ("no-longer-active-pet", True),
    }
    label, active = labels[status]
    pet_label(animal_store, "run:cat:1", label, active=active)
    [box] = rb.boxes("cam0", T0, T0 + 60, animal_store)["boxes"]
    assert (box["label"], box["species"], box["tone"]) == ("cat", "cat", "unknown")
    assert "name" not in box
    assert "pet_id" not in box


def test_newest_owner_track_label_wins_in_review(animal_store):
    from liberational.people import pets

    first = pets.at(animal_store).save("Mia", "cat")
    second = pets.at(animal_store).save("Pip", "cat")
    animal(animal_store, T0 + 1)
    pet_label(animal_store, "run:cat:1", first["id"])
    pet_label(animal_store, "run:cat:1", second["id"], ts=T0 + 2)
    [box] = rb.boxes("cam0", T0, T0 + 60, animal_store)["boxes"]
    assert box["name"] == "Pip"
    assert box["pet_id"] == second["id"]


@pytest.mark.parametrize(
    "box",
    [
        (0.4, 0.2, 0.1, 0.6, 0.9),
        (0.1, 0.2, 0.1, 0.6, 0.9),
        (float("nan"), 0.2, 0.4, 0.6, 0.9),
        (0.1, 0.2, float("inf"), 0.6, 0.9),
    ],
)
def test_invalid_pet_geometry_never_reaches_the_video_overlay(animal_store, box):
    animal(animal_store, T0 + 1, box=box)
    assert rb.boxes("cam0", T0, T0 + 60, animal_store)["boxes"] == []


def test_pet_boxes_are_bounded_after_camera_selection(animal_store, monkeypatch):
    monkeypatch.setattr(rb, "LIMIT", 2)
    animal(animal_store, T0 + 1, cam="cam1")
    animal(animal_store, T0 + 2)
    animal(animal_store, T0 + 3)
    animal(animal_store, T0 + 4)
    got = rb.boxes("cam0", T0, T0 + 60, animal_store)["boxes"]
    assert [b["ts"] for b in got] == [T0 + 2, T0 + 3]


def embedding(state, ts, track, identity, k, species="dog"):
    """One see-actions animal embedding: 0.95 along the animal's own axis, 0.31 along the sighting's."""
    import numpy as np

    v = np.zeros(384)
    v[identity], v[100 + k] = 0.95, 0.31
    writers = segment_log.Writers(segment_log.root(state))
    writers.append(
        segment_log.ANIMAL_EMBEDDINGS,
        "cam0",
        [
            {
                "id": f"emb-{track}-{k}",
                "ts_start": ts,
                "cam": "cam0",
                "track_id": track,
                "species": species,
                "box": [0.1, 0.2, 0.4, 0.6, 0.9],
                "vector": (v / np.linalg.norm(v)).tolist(),
                "model": animal_embedder.space(),
            }
        ],
    )
    writers.close()


def test_an_unlabeled_pet_recognized_by_appearance_is_named_in_review_and_a_stranger_is_not(animal_store):
    from liberational.people import pet_recognition, pets

    bess = pets.at(animal_store).save("Bess", "dog")
    for k in range(6):  # six visits an hour apart, labeled on the Pets page
        embedding(animal_store, T0 - 30000 + 3600 * k, f"bess{k}", 5, k)
    pet_recognition.label(animal_store, [f"bess{k}" for k in range(6)], bess["id"], [bess], now=T0)
    embedding(animal_store, T0 + 1, "run:dog:1", 5, 10)
    embedding(animal_store, T0 + 1, "run:dog:2", 6, 11)
    animal(animal_store, T0 + 1, track="run:dog:1", species="dog")
    animal(animal_store, T0 + 1, track="run:dog:2", species="dog")
    named, stranger = rb.boxes("cam0", T0, T0 + 60, animal_store)["boxes"]
    assert {k: named[k] for k in ("label", "tone", "pet_id", "name", "how")} == {
        "label": "Bess · dog",
        "tone": "known",
        "pet_id": bess["id"],
        "name": "Bess",
        "how": "recognized",
    }
    assert named["similarity"] >= pet_recognition.MIN_GALLERY_SIM
    assert (stranger["label"], stranger["tone"]) == ("dog", "unknown") and "name" not in stranger


def test_review_boxes_still_show_by_species_when_recognition_fails(animal_store, monkeypatch):
    from liberational.people import pet_recognition

    def broken(*args, **kwargs):
        raise TimeoutError("segment store unavailable")

    monkeypatch.setattr(pet_recognition, "resolve", broken)
    animal(animal_store, T0 + 1)
    [box] = rb.boxes("cam0", T0, T0 + 60, animal_store)["boxes"]
    assert (box["label"], box["tone"]) == ("cat", "unknown") and "name" not in box


@pytest.fixture
def rig(monkeypatch):
    from liberational.vision import views

    monkeypatch.setattr(views, "camera_ids", lambda active_only=False: ["cam0", "cam1"])
    monkeypatch.setattr(views, "names", lambda: {"cam1": "desk"})


def test_live_here_now_names_a_labeled_pet_in_view_with_its_camera_and_when_its_visit_began(animal_store, rig):
    from liberational.dashboard import pets as routes
    from liberational.people import pets

    mia = pets.at(animal_store).save("Mia", "cat")
    pet_label(animal_store, "run:cat:1", mia["id"])
    for t in (T0 - 300, T0 - 100, T0 - 80, T0 - 55, T0 - 30, T0 - 5):  # a gap of 200 s, then a visit from T0 - 100
        animal(animal_store, t, cam="cam1")
    animal(animal_store, T0 - 6, track="run:cat:9")  # a stranger on the other camera
    assert routes.here(animal_store, now=T0) == {
        mia["id"]: {"cam": "cam1", "place": "desk", "since": T0 - 100, "track": "run:cat:1"}
    }


def test_live_here_now_leaves_out_a_pet_last_seen_over_30_s_ago(animal_store, rig):
    from liberational.dashboard import pets as routes
    from liberational.people import pets

    mia = pets.at(animal_store).save("Mia", "cat")
    pet_label(animal_store, "run:cat:1", mia["id"])
    animal(animal_store, T0 - 31)
    assert routes.here(animal_store, now=T0) == {}


def test_live_here_now_says_nobody_when_the_boxes_fail(animal_store, rig, monkeypatch):
    from liberational.dashboard import pets as routes

    def broken(*args, **kwargs):
        raise TimeoutError("segment store unavailable")

    monkeypatch.setattr(rb, "pet_boxes", broken)
    assert routes.here(animal_store, now=T0) == {}


def grounding(state, ts, objects, cam="cam0"):
    """One caption-worker grounding record (camera.objects): its boxes as they stand at ts."""
    writers = segment_log.Writers(segment_log.root(state))
    writers.append(
        segment_log.CAMERA_OBJECTS,
        cam,
        [
            {
                "id": segment_log.new_id(),
                "ts_start": ts,
                "ts_end": ts,
                "frame_id": None,
                "frame_seq": 1,
                "source": "ollama",
                "observation": {"version": 1, "status": "cached", "observed_at": ts, "attempted_at": ts},
                "objects": [
                    {"label": label, "bbox": [0, 0, 1, 1], "ts": ts, "bbox_norm": list(norm) if norm else None}
                    for label, norm in objects
                ],
            }
        ],
    )
    writers.close()


SOFA = ("sofa", (0.6, 0.25, 1.0, 0.6))


def labels(state, start=T0, end=T0 + 60, cam="cam0"):
    return [
        (b["ts"], b["until"], b["label"], b["box"])
        for b in rb.boxes(cam, start, end, state)["boxes"]
        if b["kind"] == "label"
    ]


def test_a_groundings_labels_hold_until_the_cameras_next_record(tmp_path):
    grounding(tmp_path, T0 + 5, [SOFA, ("lamp", (0.3, 0.3, 0.7, 0.9))])
    grounding(tmp_path, T0 + 15, [SOFA])
    grounding(tmp_path, T0 + 20, [SOFA], cam="cam1")  # another camera's record doesn't end cam0's
    assert labels(tmp_path) == [
        (T0 + 5, T0 + 15, "lamp", [0.3, 0.3, 0.7, 0.9]),
        (T0 + 5, T0 + 15, "sofa", [0.6, 0.25, 1.0, 0.6]),
        (T0 + 15, T0 + 15 + rb.LABEL_HOLD_S, "sofa", [0.6, 0.25, 1.0, 0.6]),
    ]


def test_the_last_grounding_before_the_window_is_still_standing_at_its_start(tmp_path):
    grounding(tmp_path, T0 - 10, [SOFA])  # within the hold: still standing at T0
    grounding(tmp_path, T0 - 40, [("car", (0.1, 0.1, 0.2, 0.2))])  # long gone
    assert labels(tmp_path) == [(T0 - 10, T0 - 10 + rb.LABEL_HOLD_S, "sofa", [0.6, 0.25, 1.0, 0.6])]


def test_a_grounding_without_boxes_ends_the_ones_before_it(tmp_path):
    grounding(tmp_path, T0 + 5, [SOFA])
    grounding(tmp_path, T0 + 8, [])
    assert labels(tmp_path) == [(T0 + 5, T0 + 8, "sofa", [0.6, 0.25, 1.0, 0.6])]


def test_a_grounding_on_a_warped_tile_has_no_place_on_the_picture(tmp_path):
    # The store keeps normalized boxes only when every box in the record has one.
    grounding(tmp_path, T0 + 5, [("chair", None), SOFA])
    grounding(tmp_path, T0 + 10, [SOFA])
    assert [(ts, label) for ts, _, label, _ in labels(tmp_path)] == [(T0 + 10, "sofa")]


def poses(state, cam, rows, box=(0.1, 0.2, 0.3, 0.9)):
    """see-actions' `pose.track` rows: (time, pose track id) each, every one with `box`."""
    import numpy as np

    continuum_py = pytest.importorskip("continuum_py")
    from liberational.see.actions import JOINTS, POSE_SCHEMA
    from liberational.see.actions_worker import POSES

    writer = continuum_py.SegmentWriter(segment_log.root(state), POSES, cam, POSE_SCHEMA)
    writer.append(
        {
            "ts_ns": np.array([round(t * 1e9) for t, _ in rows], np.int64),
            "frame_id": [segment_log.new_id() for _ in rows],
            "track_id": np.array([tid for _, tid in rows], np.int64),
            "box": np.array([[*box, 0.8]] * len(rows), np.float32),
            "joints": np.zeros((len(rows), JOINTS * 3), np.int16),
            "model": ["test"] * len(rows),
        }
    )
    writer.close()


def people(state, cam="cam0"):
    return [b for b in rb.boxes(cam, T0, T0 + 60, state)["boxes"] if b["kind"] == "person"]


def test_a_tracked_person_s_body_comes_ten_times_a_second_whatever_the_camera_s_rate(tmp_path):
    poses(tmp_path, "cam0", [(T0 + i / 30, 7) for i in range(30)])  # cam0: 30 fps for a second
    poses(tmp_path, "kitchen", [(T0 + i / 10, 9) for i in range(10)])  # a 10 fps camera
    assert [round(b["ts"] - T0, 3) for b in people(tmp_path)] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert len(people(tmp_path, "kitchen")) == 10


def test_two_people_in_one_frame_each_get_their_own_body_box(tmp_path):
    poses(tmp_path, "cam0", [(T0, 7), (T0, 8), (T0 + 0.1, 7), (T0 + 0.1, 8)])
    assert sorted({b["id"] for b in people(tmp_path)}) == ["person:7", "person:8"]


def test_a_body_is_named_by_most_of_the_faces_seen_on_its_track(tmp_path, vision):
    poses(tmp_path, "cam0", [(T0 + 1, 7), (T0 + 1, 8)])
    vision.face(T0 + 0.5, "cam0", "Alice", 0.6, track_id="7")  # one stray look, the first
    vision.face(T0 + 1, "cam0", "Dana", 0.6, track_id="7")
    vision.face(T0 + 2, "cam0", "Dana", 0.6, track_id="7")
    vision.face(T0 + 2, "cam0", "Alice", 0.6, track_id="99")  # another track's face names nobody here
    by_id = {b["id"]: b for b in people(tmp_path)}
    assert (by_id["person:7"]["label"], by_id["person:7"]["tone"]) == ("Dana", "known")
    assert (by_id["person:8"]["label"], by_id["person:8"]["tone"]) == ("Who's this?", "unknown")


def test_an_unknown_body_carries_its_track_and_latest_face_group_for_naming(tmp_path, vision):
    poses(tmp_path, "cam0", [(T0 + 1, 7)])
    vision.face(T0 + 1, cluster="Face 4", track_id="7")
    vision.face(T0 + 2, cluster="Face 5", track_id="7")
    [body] = people(tmp_path)
    assert (body["track"], body["group"]) == ("7", 5)


def test_someone_facing_away_still_has_a_body_box(tmp_path):
    poses(tmp_path, "living-room", [(T0 + i / 10, 3) for i in range(5)], box=(0.1, 0.5, 0.35, 0.85))
    got = people(tmp_path, "living-room")
    assert [b["box"] for b in got] == [[0.1, 0.5, 0.35, 0.85]] * 5
    assert {b["label"] for b in got} == {"Who's this?"}


def test_bodies_are_read_from_one_camera_and_the_window_only(tmp_path):
    poses(tmp_path, "cam0", [(T0 - 1, 7), (T0 + 1, 7), (T0 + 61, 7)])
    poses(tmp_path, "cam1", [(T0 + 1, 8)])
    assert [(b["ts"], b["id"]) for b in people(tmp_path)] == [(T0 + 1, "person:7")]


def test_a_body_with_no_area_never_reaches_the_overlay(tmp_path):
    poses(tmp_path, "cam0", [(T0 + 1, 7)], box=(0.3, 0.2, 0.3, 0.9))
    assert people(tmp_path) == []
