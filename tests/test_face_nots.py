"""The owner's "this face group is not <person>" (people/face_nots.py) and its routes."""

from unittest.mock import create_autospec

import numpy as np
import pytest

from liberational.dashboard import context, server
from liberational.people import face_nots, registry
from liberational.vision import faces


@pytest.fixture
def people(tmp_path, monkeypatch):
    reg = registry.at(tmp_path)
    reg.ensure("Kara", "test")
    reg.ensure("Sam", "test")
    monkeypatch.setattr(context, "STATE", tmp_path)
    yield tmp_path
    registry.close_all()


def test_a_mark_is_kept_by_person_and_taking_it_back_undoes_it(people):
    assert face_nots.mark(people, "Kara", [5, 6], now=1.0) == 2
    assert face_nots.names(people) == {5: {"Kara"}, 6: {"Kara"}}
    face_nots.mark(people, "Kara", [6], active=False, now=2.0)
    assert face_nots.names(people) == {5: {"Kara"}}


def test_a_mark_follows_a_rename(people):
    face_nots.mark(people, "Kara", [5], now=1.0)
    registry.at(people).rename("Kara", "Katherine", "test")
    assert face_nots.names(people) == {5: {"Katherine"}}


@pytest.mark.parametrize(
    ("name", "groups", "error"),
    [
        pytest.param("Nobody", [5], LookupError, id="no-such-person"),
        pytest.param("Kara", [], ValueError, id="no-groups"),
        pytest.param("Kara", ["5"], ValueError, id="not-a-group-number"),
        pytest.param("Kara", [True], ValueError, id="a-bool-is-no-group"),
        pytest.param("", [5], ValueError, id="no-name"),
    ],
)
def test_a_mark_that_cannot_be_true_writes_nothing(people, name, groups, error):
    with pytest.raises(error):
        face_nots.mark(people, name, groups)
    assert face_nots.read(people) == {}


def test_reading_with_no_store_creates_none(tmp_path):
    assert face_nots.read(tmp_path) == {}
    assert not registry.knowledge_path(tmp_path).exists()


def test_the_route_marks_groups_and_hands_shots_to_the_cards_correction(people, monkeypatch):
    correct = create_autospec(server.correct_card, return_value=(200, {"ok": True, "undo": "tok-1"}))
    monkeypatch.setattr(server, "correct_card", correct)
    code, reply = server.faces_not({"name": "Kara", "groups": [5], "shots": ["s1"]}, True)
    assert (code, reply) == (200, {"ok": True, "marked": 2, "undo": ["tok-1"]})
    correct.assert_called_once_with({"name": "Kara", "kind": "shot", "id": "s1", "action": "remove"})
    assert face_nots.names(people) == {5: {"Kara"}}


@pytest.mark.parametrize(
    ("body", "control", "code"),
    [
        pytest.param({"name": "Kara", "groups": [5]}, False, 403, id="not-this-machine"),
        pytest.param({"name": "Kara"}, True, 400, id="nothing-named"),
        pytest.param({"name": "Kara", "groups": [5], "not": "yes"}, True, 400, id="not-a-bool"),
        pytest.param({"name": "Kara", "shots": ["s1"], "not": False}, True, 400, id="a-shot-undoes-by-token"),
        pytest.param({"name": "Nobody", "groups": [5]}, True, 404, id="no-such-person"),
    ],
)
def test_the_route_refuses_what_it_cannot_do(people, body, control, code):
    assert server.faces_not(body, control)[0] == code
    assert face_nots.read(people) == {}


def test_the_people_photo_alias_takes_the_pets_verbs(people, monkeypatch):
    correct = create_autospec(server.correct_card, return_value=(200, {"ok": True, "undo": "tok"}))
    monkeypatch.setattr(server, "correct_card", correct)
    monkeypatch.setattr(server, "people_roster", lambda: [{"name": "Kara"}])
    monkeypatch.setattr(server, "person_card", lambda name: {"name": name})
    assert server.people_photo({"name": "Kara", "action": "main", "shot": "s1"}, True)[0] == 200
    correct.assert_called_once_with({"name": "Kara", "kind": "shot", "id": "s1", "action": "main"})
    server.card_prefs.set_main(people, "Kara", "s1")
    assert server.people_photo({"name": "kara", "action": "newest"}, True)[1]["card"] == {"name": "Kara"}
    assert server.card_prefs.get(people, "Kara")["main"] is None
    server.card_prefs.hide(people, "Kara", "shot", "s2")
    server.people_photo({"name": "Kara", "action": "show", "shot": "s2"}, True)
    assert "s2" not in server.card_prefs.get(people, "Kara")["hidden"]["shot"]
    assert server.people_photo({"name": "Kara", "action": "crop"}, True)[0] == 400
    assert server.people_photo({"name": "Kara", "action": "newest"}, False)[0] == 403
    assert server.people_photo({"name": "Nobody", "action": "newest"}, True)[0] == 404


def test_a_group_marked_not_kara_holds_the_matcher_off_its_next_look_until_taken_back(people, monkeypatch):
    # The mark teaches the face matcher too, not only the guesses
    monkeypatch.setattr(faces, "DB", people / "db.npz")
    monkeypatch.setattr(faces, "LEARNED_DB", people / "learned.npz")
    monkeypatch.setattr(faces, "NOTS_DB", people / "nots.npz")
    monkeypatch.setattr(faces, "UNK_DB", people / "unknown_clusters.npz")
    monkeypatch.setattr(faces, "CHECK", people / "check.json")
    kara, stranger, later = np.zeros((3, 512), np.float32)
    kara[0] = 1.0
    stranger[[0, 200]] = 0.6, 0.8
    later[[0, 200, 300]] = 0.58, 0.7, 0.4167
    faces.save_db(["Kara"], kara[None])
    faces._unk_save([5], stranger[None], [3], 6)
    assert faces.match(later)[0] == "Kara"
    face_nots.mark(people, "Kara", [5], now=1.0)
    assert faces.match(later)[0] is None
    face_nots.mark(people, "Kara", [5], active=False, now=2.0)
    assert faces.match(later)[0] == "Kara"
