"""Who's this?'s ignored pictures and blocked faces (people/face_hidden.py), the deselected pictures that move to a
group of their own (vision/face_rejects.py), and their routes."""

import numpy as np
import pytest

from liberational.dashboard import context, people_corrections, photo_files, unnamed_groups
from liberational.people import face_hidden, face_nots, face_photos, registry
from liberational.vision import face_rejects, faces


def face(axis: int, side: int | None = None, lean: float = 0.0) -> np.ndarray:
    """A unit embedding along `axis`, leaning `lean` toward `side`: it's sqrt(1 - lean^2) alike to face(axis)."""
    v = np.zeros(512, np.float32)
    v[axis] = np.sqrt(1 - lean**2)
    if side is not None:
        v[side] = lean
    return v


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A state directory with a people registry, and the face stores in it."""
    registry.at(tmp_path).ensure("Alice", "test")
    monkeypatch.setattr(context, "STATE", tmp_path)
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")
    monkeypatch.setattr(faces, "NOTS_DB", tmp_path / "nots.npz")
    monkeypatch.setattr(face_rejects, "path", lambda: tmp_path / "unknown_rejected.json")
    monkeypatch.setattr(photo_files, "_unknown_photo", lambda gid: tmp_path / f"{gid}.jpg")
    yield tmp_path
    registry.close_all()


@pytest.fixture
def group(home, monkeypatch):
    """Face group 5 with four recorded pictures: s1 and s2 of one face, s3 and s4 of another. The pictures and their
    embeddings are read as the dashboard reads them, through what the owner took out or moved."""
    looks = {"s1": face(10), "s2": face(10), "s3": face(20), "s4": face(20)}
    faces._unk_save([5], face(10)[None], [4], 6)

    def unnamed(state=None, *, rows=None):
        found = [
            {"id": sid, "ts": 100.0 - k, "cluster": face_rejects.group_of(sid, "Face 5")} for k, sid in enumerate(looks)
        ]
        return [f for f in found if f["cluster"]]

    monkeypatch.setattr(face_photos, "_unnamed", unnamed)
    monkeypatch.setattr(
        face_photos,
        "shot_embeddings",
        lambda shots, state=None: np.asarray([looks[s["id"]] for s in shots], np.float32).reshape(-1, 512),
    )
    return home


def group_face(gid: int) -> np.ndarray:
    vec = faces.group_vector(gid)
    assert vec is not None, f"no face group {gid}"
    return vec


def pictures(gid: int) -> list[str]:
    return [s["id"] for s in face_photos.page_of(gid, 0, 100, state=context.STATE)[0]]


# ---- the taught records ----


def test_an_ignored_picture_is_listed_until_it_is_taken_back(home):
    assert face_hidden.ignore(home, 5, ["s1", "s2"], now=10.0) == ["s1", "s2"]
    assert set(face_hidden.ignored(home)) == {"s1", "s2"}
    assert face_hidden.restore(home, "ignored", ["s1"], now=11.0) == [{"group": 5, "shots": ["s1"]}]
    listing = face_hidden.listing(home)
    assert [(r["id"], r["group"]) for r in listing["ignored"]] == [("s2", 5)]
    assert listing["ignored_count"] == 1


def test_a_block_keeps_the_mean_face_of_its_pictures_and_is_listed_until_it_is_taken_back(home):
    key = face_hidden.block(home, 5, ["s1", "s2"], np.stack([face(10, 11, 0.6), face(10, 12, 0.6)]), now=10.0)
    (row,) = face_hidden.blocks(home).values()
    assert np.linalg.norm(row["vector"]) == pytest.approx(1.0)
    assert np.argmax(row["vector"]) == 10  # what the two pictures share
    listing = face_hidden.listing(home)
    assert [(b["id"], b["shots"], b["count"]) for b in listing["blocked"]] == [(key, ["s1", "s2"], 2)]
    assert face_hidden.restore(home, "blocked", [key], now=11.0) == [{"group": 5, "shots": ["s1", "s2"]}]
    assert face_hidden.blocks(home) == {}


def test_ignored_and_blocked_are_taught_state_in_the_knowledge_store(home):
    from liberational import knowledge_cli

    face_hidden.ignore(home, 5, ["s1"], now=10.0)
    face_hidden.block(home, 5, ["s2"], face(10)[None], now=10.0)
    store = registry.knowledge_path(home)
    assert store.name == "knowledge.sqlite3" and store.exists()  # what a keep-taught reset keeps
    assert {face_hidden.IGNORES, face_hidden.BLOCKS} <= knowledge_cli._DOCUMENTS  # and an export carries


def test_reading_with_no_store_creates_none(tmp_path):
    assert face_hidden.listing(tmp_path)["ignored"] == []
    assert not registry.knowledge_path(tmp_path).exists()


@pytest.mark.parametrize(
    ("shots", "members"),
    [
        pytest.param([], face(10)[None], id="no-pictures"),
        pytest.param(["s1", 7], face(10)[None], id="not-a-shot-id"),
        pytest.param(["s1"], np.zeros((0, 512)), id="no-face-on-record"),
    ],
)
def test_a_block_that_cannot_be_made_writes_nothing(home, shots, members):
    with pytest.raises(ValueError):
        face_hidden.block(home, 5, shots, members)
    assert face_hidden.blocks(home) == {}


def test_restoring_what_is_not_hidden_restores_nothing(home):
    assert face_hidden.restore(home, "ignored", ["nope"]) == []
    with pytest.raises(ValueError):
        face_hidden.restore(home, "everything", ["s1"])


# ---- the block threshold ----


def test_the_block_threshold_is_the_one_that_folds_two_groups_as_one_person():
    assert face_hidden.BLOCK_SIM == faces.FOLD_SIM
    assert face_hidden.BLOCK_SIM > faces.THRESH  # well above what names one look


@pytest.mark.parametrize(
    ("alike", "hidden"),
    [
        pytest.param(0.99, True, id="the-same-face"),
        pytest.param(0.70, True, id="at-the-threshold"),
        pytest.param(0.69, False, id="just-under"),
        pytest.param(0.45, False, id="alike-enough-to-name-but-not-to-block"),
        pytest.param(0.0, False, id="someone-else"),
    ],
)
def test_a_group_is_blocked_only_at_high_confidence(home, alike, hidden):
    face_hidden.block(home, 5, ["s1"], face(10)[None], now=10.0)
    other = np.zeros(512, np.float32)
    other[[10, 30]] = alike + 1e-4, np.sqrt(max(0.0, 1 - alike**2))  # `alike` to the blocked face, a hair over
    assert face_hidden.blocked_groups(home, [7], other[None]) == ({7} if hidden else set())


def test_with_nothing_blocked_no_group_is(home):
    assert face_hidden.blocked_groups(home, [7], face(10)[None]) == set()


def test_who_s_this_leaves_out_a_group_whose_face_is_blocked_and_lists_it_again_once_unblocked(home):
    faces._unk_save([5, 6], np.stack([face(10), face(20)]), [4, 4], 7)
    rows = [{"id": 5}, {"id": 6}]
    assert unnamed_groups._without_blocked(rows) == rows
    key = face_hidden.block(home, 5, ["s1"], face(10, 11, 0.3)[None], now=10.0)  # 0.95 like group 5
    assert unnamed_groups._without_blocked(rows) == [{"id": 6}]
    face_hidden.restore(home, "blocked", [key], now=11.0)
    assert unnamed_groups._without_blocked(rows) == rows


def test_a_block_or_an_ignore_moves_the_face_rows_owner_key(home):
    before = unnamed_groups._face_nots_stamp()
    face_hidden.ignore(home, 5, ["s1"], now=10.0)
    ignored = unnamed_groups._face_nots_stamp()
    face_hidden.block(home, 5, ["s2"], face(10)[None], now=11.0)
    assert len({before, ignored, unnamed_groups._face_nots_stamp()}) == 3


# ---- pictures that leave a group, or move ----


def test_a_rejected_shot_has_no_group_a_moved_one_has_its_new_group_and_a_restored_one_its_own(home):
    face_rejects.reject(["s1"])
    face_rejects.move(["s2"], 9)
    assert face_rejects.group_of("s1", "Face 5") is None
    assert face_rejects.group_of("s2", "Face 5") == "Face 9"
    assert face_rejects.group_of("s3", "Face 5") == "Face 5"
    assert face_rejects.group_of("s2", None) is None  # a named face has no group to move
    assert face_rejects.restore(["s1"]) == 1
    assert face_rejects.group_of("s1", "Face 5") == "Face 5"
    assert face_rejects.moved() == {"s2": 9}  # a restore keeps the moves


def test_recorded_sightings_are_read_through_what_the_owner_took_out_or_moved(home):
    from liberational.vision import sightings

    face_rejects.reject(["s1"])
    face_rejects.move(["s2"], 9)
    found = [{"id": sid, "ts_start": 1.0} for sid in ("s1", "s2", "s3")]
    matches = {sid: {"cluster": "Face 5"} for sid in ("s1", "s2", "s3")}
    rows = sightings._face_rows(found, matches, home)
    assert [(r["id"], r["cluster"]) for r in rows] == [("s1", None), ("s2", "Face 9"), ("s3", "Face 5")]


def test_a_group_made_from_left_out_pictures_is_held_and_listed(home):
    faces._unk_save([5], face(10)[None], [4], 6)
    made = faces.new_group(np.stack([face(20)]))
    assert made == 6
    assert made in faces.unknown_held()
    assert {g["id"]: g["count"] for g in faces.unknown_groups()}[made] >= 2  # People lists from two sightings
    assert np.argmax(group_face(made)) == 20
    assert faces.new_group(np.zeros((0, 512))) is None


# ---- the routes ----


def test_deselected_pictures_move_to_a_group_of_their_own_and_the_rest_keep_theirs(group):
    code, reply = people_corrections.split_shots(5, ["s3", "s4"], True)
    assert (code, reply) == (200, {"ok": True, "group": 6, "moved": 2, "left": 2})
    assert pictures(5) == ["s1", "s2"]
    assert pictures(6) == ["s3", "s4"]  # still unnamed, together
    assert np.argmax(group_face(6)) == 20
    assert np.argmax(group_face(5)) == 10  # rebuilt from the pictures it kept


@pytest.mark.parametrize(
    ("gid", "ids", "control", "code"),
    [
        pytest.param(5, ["s3"], False, 403, id="not-this-machine"),
        pytest.param(99, ["s3"], True, 404, id="no-such-group"),
        pytest.param(5, [], True, 400, id="nothing-deselected"),
        pytest.param(5, ["elsewhere"], True, 400, id="not-its-picture"),
        pytest.param(5, ["s1", "s2", "s3", "s4"], True, 400, id="everything-deselected"),
        pytest.param(True, ["s3"], True, 400, id="a-bool-is-no-group"),
    ],
)
def test_a_split_that_cannot_be_made_changes_nothing(group, gid, ids, control, code):
    assert people_corrections.split_shots(gid, ids, control)[0] == code
    assert pictures(5) == ["s1", "s2", "s3", "s4"]
    assert [g["id"] for g in faces.unknown_groups()] == [5]


def test_ignore_hides_the_selected_pictures_and_leaves_the_deselected_in_the_group(group):
    code, reply = people_corrections.hide_shots(5, "ignore", ["s3", "s4"], True)
    assert (code, reply) == (200, {"ok": True, "kind": "ignored", "keys": ["s1", "s2"], "hidden": 2, "left": 2})
    assert pictures(5) == ["s3", "s4"]
    assert set(face_hidden.ignored(group)) == {"s1", "s2"}
    assert face_hidden.blocks(group) == {}  # an ignore blocks nothing: the same face seen again is asked about


def test_un_ignoring_a_picture_puts_it_back_in_its_group(group):
    people_corrections.hide_shots(5, "ignore", ["s3", "s4"], True)
    assert people_corrections.restore_hidden("ignored", ["s1"], True) == (200, {"ok": True, "restored": 1})
    assert pictures(5) == ["s1", "s3", "s4"]
    assert set(face_hidden.ignored(group)) == {"s2"}


def test_block_hides_the_pictures_and_any_group_with_that_face_until_unblocked(group):
    code, reply = people_corrections.hide_shots(5, "block", ["s3", "s4"], True)
    assert code == 200 and (reply["kind"], reply["hidden"], reply["left"]) == ("blocked", 2, 2)
    assert pictures(5) == ["s3", "s4"]
    again = np.stack([face(10, 11, 0.3), face(20)])  # the blocked face seen again, and the other one
    assert face_hidden.blocked_groups(group, [7, 8], again) == {7}
    assert people_corrections.restore_hidden("blocked", reply["keys"], True)[0] == 200
    assert pictures(5) == ["s1", "s2", "s3", "s4"]
    assert face_hidden.blocked_groups(group, [7, 8], again) == set()


def test_ignoring_every_picture_empties_the_group_without_putting_it_aside(group):
    from liberational.dashboard import inbox_state

    assert people_corrections.hide_shots(5, "ignore", None, True)[1]["left"] == 0
    assert pictures(5) == []
    assert inbox_state.inbox_skips()["face"] == []  # new pictures of the face are asked about


@pytest.mark.parametrize(
    ("gid", "action", "exclude", "control", "code"),
    [
        pytest.param(5, "ignore", [], False, 403, id="not-this-machine"),
        pytest.param(99, "ignore", [], True, 404, id="no-such-group"),
        pytest.param(5, "forget", [], True, 400, id="no-such-action"),
        pytest.param(5, "block", ["s1", "s2", "s3", "s4"], True, 400, id="nothing-selected"),
        pytest.param(5, "ignore", "s1", True, 400, id="exclude-is-a-list"),
    ],
)
def test_a_hide_that_cannot_be_made_changes_nothing(group, gid, action, exclude, control, code):
    assert people_corrections.hide_shots(gid, action, exclude, control)[0] == code
    assert pictures(5) == ["s1", "s2", "s3", "s4"]
    assert face_hidden.listing(group)["ignored"] == [] and face_hidden.listing(group)["blocked"] == []


@pytest.mark.parametrize(
    ("kind", "keys", "control", "code"),
    [
        pytest.param("ignored", ["s1"], False, 403, id="not-this-machine"),
        pytest.param("ignored", ["never-hidden"], True, 404, id="not-hidden"),
        pytest.param("everything", ["s1"], True, 400, id="no-such-kind"),
        pytest.param("ignored", [], True, 400, id="no-keys"),
    ],
)
def test_a_restore_that_cannot_be_made_is_refused(group, kind, keys, control, code):
    people_corrections.hide_shots(5, "ignore", ["s2", "s3", "s4"], True)
    assert people_corrections.restore_hidden(kind, keys, control)[0] == code
    assert set(face_hidden.ignored(group)) == {"s1"}


def test_the_hidden_listing_gives_each_picture_its_crop(group):
    people_corrections.hide_shots(5, "ignore", ["s2", "s3", "s4"], True)
    blocked = people_corrections.hide_shots(5, "block", ["s3", "s4"], True)[1]["keys"]
    code, reply = people_corrections.hidden_faces()
    assert code == 200
    assert [(r["id"], r["url"]) for r in reply["ignored"]] == [("s1", "/faces/shot/s1.jpg")]
    assert [(b["id"], b["shots"]) for b in reply["blocked"]] == [
        (blocked[0], [{"id": "s2", "url": "/faces/shot/s2.jpg"}])
    ]
    assert reply["block_similarity"] == face_hidden.BLOCK_SIM


# ---- "Not <name>" ----


def test_not_alice_is_recorded_for_the_group_so_alice_is_not_its_guess_again_and_undo_takes_it_back(group):
    assert people_corrections.faces_not({"name": "Alice", "groups": [5], "shots": []}, True)[0] == 200
    assert face_nots.names(group) == {5: {"Alice"}}  # what the guess leaves out (vision/face_guess.py `nots`)
    assert pictures(5) == ["s1", "s2", "s3", "s4"]  # the pictures stay, waiting for another name
    people_corrections.faces_not({"name": "Alice", "groups": [5], "shots": [], "not": False}, True)
    assert face_nots.names(group) == {}


def test_not_alice_after_deselecting_marks_only_the_selected_pictures_group(group):
    made = people_corrections.split_shots(5, ["s3", "s4"], True)[1]["group"]
    people_corrections.faces_not({"name": "Alice", "groups": [5], "shots": []}, True)
    assert face_nots.names(group) == {5: {"Alice"}}  # the left-out pictures' group can still be Alice
    assert pictures(made) == ["s3", "s4"]
