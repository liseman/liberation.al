"""tools/drop_side_faces.py: which unnamed groups and learned rows came from the side or back of a head."""

import importlib.util
import json

import numpy as np
import pytest

from liberational import config as cfg

_spec = importlib.util.spec_from_file_location("drop_side_faces", cfg.ROOT / "tools" / "drop_side_faces.py")
assert _spec and _spec.loader
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

FACE = {"frontal": 0.6, "norm": 19.0}
SIDE = {"frontal": 0.0, "norm": 14.0}


def signal(match_id, look):
    return {"match_id": match_id, "head": [0.8] * 5, **look}


def test_a_group_goes_only_when_every_look_of_it_is_judged_and_none_shows_the_face():
    matches = [
        {"id": "a1", "cluster": "Face 1"},  # side only: goes
        {"id": "a2", "cluster": "Face 1"},
        {"id": "b1", "cluster": "Face 2"},  # one face look: stays
        {"id": "b2", "cluster": "Face 2"},
        {"id": "c1", "cluster": "Face 3"},  # a look with no signal: can't be judged, stays
        {"id": "c2", "cluster": "Face 3"},
        {"id": "d1", "cluster": "Face 4"},  # side only, but the owner touched it: stays
        {"id": "n1", "cluster": None},  # a named look
    ]
    shows = tool.facing(
        [
            signal("a1", SIDE),
            signal("a2", SIDE),
            signal("b1", SIDE),
            signal("b2", FACE),
            signal("c1", SIDE),
            signal("d1", SIDE),
            signal("n1", SIDE),
        ]
    )
    assert tool.side_only_groups(matches, shows, keep={4}) == {1}


def test_a_face_tipped_away_from_its_landmarks_but_with_a_face_norm_still_shows_the_face():
    shows = tool.facing([signal("a", {"frontal": 0.0, "norm": 19.0}), signal("b", {"frontal": 0.0, "norm": None})])
    assert shows == {"a": True, "b": False}


def test_a_learned_row_goes_only_when_it_is_a_side_looks_embedding():
    rows = np.eye(512, dtype=np.float32)[:3]
    embeddings = [
        {"match_id": "s", "vector": rows[0] * 2},  # a side look's embedding: row 0 goes
        {"match_id": "f", "vector": rows[1]},  # a face look's: row 1 stays
        {"match_id": "u", "vector": rows[2]},  # no signal: row 2 stays
    ]
    shows = tool.facing([signal("s", SIDE), signal("f", FACE)])
    (gone,) = tool.side_learned(rows, embeddings, shows)
    assert gone == pytest.approx(rows[0])


@pytest.fixture
def owner(tmp_path, monkeypatch):
    """A groups store and a state dir in tmp_path, nothing marked: the state dir."""
    from liberational.people import face_nots
    from liberational.vision import faces

    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")
    monkeypatch.setattr(face_nots, "read", lambda state: {})
    monkeypatch.setattr(tool.face_rejects, "rejected", lambda: frozenset())
    state = tmp_path / "state"
    state.mkdir()
    with cfg.override({"paths": {"state": str(state)}}):
        yield state


def test_every_group_the_owner_touched_is_kept(owner, monkeypatch):
    # The tool kept only put-aside groups and the window's rejected shots, so a group marked not someone or
    # with its name undone could go
    from liberational.people import face_nots
    from liberational.vision import faces

    (owner / "inbox_skips.json").write_text(json.dumps({"face": [7.0, 8], "voice": [9]}))
    monkeypatch.setattr(face_nots, "read", lambda state: {3: {"person-1"}})
    monkeypatch.setattr(tool.face_rejects, "rejected", lambda: frozenset({"y"}))
    faces.unname_cluster(4, np.eye(512, dtype=np.float32)[0], 2)  # the owner undid its name: held by the store
    matches = [{"id": "x", "cluster": "Face 5"}, {"id": "y", "cluster": "Face 6"}]
    assert tool.owner_kept(matches) == {3, 4, 6, 7, 8}


@pytest.mark.parametrize("skips", ["{not json", '{"face": [1'])
def test_unreadable_marks_stop_the_tool_rather_than_count_as_none(owner, skips):
    (owner / "inbox_skips.json").write_text(skips)
    with pytest.raises(ValueError):
        tool.owner_kept([{"id": "x", "cluster": "Face 1"}])
