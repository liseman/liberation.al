"""tools/restore_pet_references.py: which of the detector's animals in a photo is the track's, re-embedding
stranded labels from their kept photos, and carrying pets over from an earlier embedding space."""

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from liberational import config as cfg
from liberational import knowledge
from liberational.people import face_shots, pet_shots
from liberational.people import pet_recognition as R
from test_pet_recognition import MIA, PETS, POPPY, T, look, owner

_spec = importlib.util.spec_from_file_location(
    "restore_pet_references", pathlib.Path(__file__).resolve().parents[1] / "tools" / "restore_pet_references.py"
)
assert _spec and _spec.loader
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


@pytest.mark.parametrize(
    ("animals", "expected"),
    [
        pytest.param(
            [[0.0, 0.0, 0.2, 0.2, 0.99, 15], [0.1, 0.3, 0.9, 0.7, 0.5, 15]], [0.1, 0.3, 0.9, 0.7], id="central"
        ),
        pytest.param([], None, id="none-found"),
    ],
)
def test_the_photos_own_animal_is_the_one_nearest_its_centre(animals, expected):
    found = tool.central(np.array(animals, np.float32).reshape(-1, 6))
    assert found is None if expected is None else np.allclose(found, expected)


OLD = "old/model:tight"  # the space the pets were taught in
NEW_AXIS = 200  # the current model's vectors, in these tests, sit on other axes than the old model's


def taught(state, *examples):
    """Owner labels with their examples in the OLD space, as taught before a model change: (track, pet, identity)."""
    owner(state, *((track, pet, T + 60 * k) for k, (track, pet, _) in enumerate(examples)))
    records = []
    for k, (track, pet, identity) in enumerate(examples):
        label = {"track_id": track, "species": "cat", "label": pet, "cam": "cam0", "sighting_start": T + 60 * k}
        record = R.reference(np.array(look(identity, k)), label, int((T + 60 * k) * 1e9))
        record.value["model"] = OLD
        records.append(record)
    knowledge.Store(state / "knowledge.sqlite3").put(records)


def saved(state, track):
    face_shots.write(state, track, "1", f"jpg of {track}".encode(), pet_shots.KIND)


@pytest.fixture
def models(monkeypatch):
    """The detector and the two embedders, faked: `models.old[track]` is what the old model makes of a track's
    photo (an identity, as `look`), and the current model puts every photo of an identity on its own new axis."""
    fake = SimpleNamespace(old={}, embedded=[])

    def embed(crops, repo=None):
        fake.embedded.append((repo, sorted(crops)))
        out = {}
        for k, track in enumerate(sorted(crops)):
            v = np.array(look(fake.old[track], 50 + k), np.float32)
            out[track] = v if repo is not None else np.roll(v, NEW_AXIS)
        return out

    monkeypatch.setattr(
        tool, "cut", lambda photos: {track: jpg for track, jpg in photos.items() if b"empty" not in jpg}
    )
    monkeypatch.setattr(tool, "embed", embed)
    monkeypatch.setattr(tool.pets, "at", lambda state: SimpleNamespace(listing=lambda: PETS))
    return fake


def records(state, collection):
    return [r.record for r in knowledge.Store(state / "knowledge.sqlite3").query(collection)]


def test_carrying_over_labels_what_the_old_examples_surely_name_and_keeps_its_photo(tmp_path, models, capsys):
    taught(tmp_path, ("run0:cat:1", MIA, 0), ("run0:cat:2", POPPY, 1))
    for track, identity in (("run1:cat:1", 0), ("run1:cat:2", 1), ("run1:cat:3", 7)):  # the last is neither cat
        saved(tmp_path, track)
        models.old[track] = identity

    tool.carry(tmp_path, OLD, "old/model", tool.MIN_SIM, tool.MARGIN, tool.PER_PET, dry_run=False)

    labels = {r.key: r.value for r in records(tmp_path, R.LABELS) if r.value.get("carried_from")}
    assert {track: v["label"] for track, v in labels.items()} == {"run1:cat:1": MIA, "run1:cat:2": POPPY}
    assert all(v["carried_from"] == OLD and v["source"] == "owner" and v["active"] for v in labels.values())
    new = {r.key: r.value for r in records(tmp_path, R.REFERENCES) if r.value["model"] == R.embedding_model()}
    assert {track: v["label"] for track, v in new.items()} == {"run1:cat:1": MIA, "run1:cat:2": POPPY}
    assert np.argmax(new["run1:cat:1"]["vector"]) == NEW_AXIS, "the example is the current model's vector"
    assert R.photos(tmp_path) == {"run1:cat:1": b"jpg of run1:cat:1", "run1:cat:2": b"jpg of run1:cat:2"}
    said = capsys.readouterr().out
    assert "Mia (cat): 1 old examples, got 1 new, 1 in the current space  <- fewer than 6" in said

    tool.carry(tmp_path, OLD, "old/model", tool.MIN_SIM, tool.MARGIN, tool.PER_PET, dry_run=False)
    assert len(records(tmp_path, R.REFERENCES)) == 4, "a second run finds nothing more to do"
    assert "Mia (cat): 1 old examples, got 0 new, 1 in the current space" in capsys.readouterr().out


def test_a_dry_run_counts_and_writes_nothing_even_before_the_model_is_switched(tmp_path, models, capsys):
    taught(tmp_path, ("run0:cat:1", MIA, 0), ("run0:cat:2", POPPY, 1))
    saved(tmp_path, "run1:cat:1")
    models.old["run1:cat:1"] = 0
    with cfg.override({"models": {"hf": {"animal_embedding": {"space": "old/model"}}}}):
        tool.carry(tmp_path, OLD, "old/model", tool.MIN_SIM, tool.MARGIN, tool.PER_PET, dry_run=True)
        assert "Mia (cat): 1 old examples, would get 1 new, " in capsys.readouterr().out
        with pytest.raises(SystemExit, match="still the configured embedding space"):
            tool.carry(tmp_path, OLD, "old/model", tool.MIN_SIM, tool.MARGIN, tool.PER_PET, dry_run=False)
    assert [r for r in records(tmp_path, R.LABELS) if r.value.get("carried_from")] == []
    assert R.photos(tmp_path) == {}
    assert [repo for repo, _ in models.embedded] == ["old/model"], "the current model is never loaded for a dry run"


def test_without_the_old_model_photos_with_no_stored_vectors_are_passed_over(tmp_path, models, capsys):
    taught(tmp_path, ("run0:cat:1", MIA, 0), ("run0:cat:2", POPPY, 1))
    saved(tmp_path, "run1:cat:1")
    models.old["run1:cat:1"] = 0
    tool.carry(tmp_path, OLD, None, tool.MIN_SIM, tool.MARGIN, tool.PER_PET, dry_run=False)
    assert "the other 1 photos are passed over" in capsys.readouterr().out
    assert R.photos(tmp_path) == {}


def sighting(track, identity, k=90, species="cat"):
    return R.Sighting(track, species, "cam0", T, T, [np.array(look(identity, k), np.float32)])


def mixed(a, b, share):
    """A sighting between two identities: `share` of the way from `a` to `b`."""
    v = (1 - share) * np.array(look(a, 91)) + share * np.array(look(b, 92))
    return R.Sighting("q", "cat", "cam0", T, T, [(v / np.linalg.norm(v)).astype(np.float32)])


REFS = R.References([(sighting("a", 0, 1), MIA), (sighting("b", 1, 2), POPPY), (sighting("d", 2, 3, "dog"), "dog-pet")])


@pytest.mark.parametrize(
    ("query", "exclude", "expected"),
    [
        pytest.param(sighting("q", 0), (), MIA, id="clearly-one-pet"),
        pytest.param(mixed(0, 1, 0.45), (), None, id="too-close-to-the-other-pet"),
        pytest.param(mixed(0, 5, 0.5), (), None, id="not-alike-enough"),
        pytest.param(mixed(0, 2, 0.45), (), None, id="too-close-to-another-species"),
        pytest.param(sighting("q", 0), (MIA,), None, id="the-owner-said-not-this-pet"),
        pytest.param(sighting("q", 0, species="bird"), (), None, id="no-example-of-its-species"),
    ],
)
def test_a_sighting_is_named_only_when_the_examples_are_sure(query, exclude, expected):
    assert tool.sure(query, REFS, 0.75, 0.10, exclude) == expected


def test_a_stranded_label_is_embedded_again_from_its_kept_photo(tmp_path, models):
    track = "run0:cat:1"
    taught(tmp_path, (track, MIA, 0))  # its only example is of the old space
    label = {"track_id": track, "species": "cat", "label": MIA, "active": True, "source": "owner", "sighting_start": T}
    at = int((T + 2) * 1e9)
    store = knowledge.Store(tmp_path / "knowledge.sqlite3")
    store.put([knowledge.Record("019c0000-0000-7000-8000-00000000aaaa", R.LABELS, track, at, at, at, label)])
    store.put([R.photo(track, b"kept jpg", at)])
    models.old[track] = 0

    tool.restore(tmp_path, dry_run=False)

    (new,) = [r.value for r in records(tmp_path, R.REFERENCES) if r.value["model"] == R.embedding_model()]
    assert (new["track_id"], new["label"]) == (track, MIA)
    assert models.embedded == [(None, [track])]
    assert R.stranded(tmp_path) == []
