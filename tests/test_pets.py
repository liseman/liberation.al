"""Owner pet profiles in the knowledge store and their explicit, conservative species assumption."""

import sqlite3
from unittest.mock import create_autospec

import pytest

from liberational import knowledge
from liberational.dashboard import pets as routes
from liberational.people import pet_recognition, pets

ID = "019c0000-0000-7000-8000-000000000001"
OTHER = "019c0000-0000-7000-8000-000000000002"


@pytest.fixture
def registry(tmp_path):
    reg = pets.PetRegistry(tmp_path / "knowledge.sqlite3")
    return reg, reg.store


def revisions(store):
    return [row.record for row in store.query(pets.PET_PROFILES)]


def test_revisions_keep_the_pets_subject_and_retire_only_the_binding(registry):
    reg, store = registry
    first = reg.save("  Pip  ", "cat")
    second = reg.save("Miso", "cat", first["id"])
    assert second["id"] == first["id"]
    assert reg.listing() == [second]
    assert [r.subjects for r in revisions(store)] == [(first["id"],), (first["id"],)]
    reg.remove(first["id"])
    assert reg.listing() == []
    assert [r.value["active"] for r in revisions(store)] == [True, True, False]
    with pytest.raises(LookupError, match="pet not found"):
        reg.save("Pip", "cat", first["id"])


def test_profiles_outlast_the_process_that_wrote_them(tmp_path):
    first = pets.PetRegistry(tmp_path / "knowledge.sqlite3").save("Pip", "cat")
    assert pets.PetRegistry(tmp_path / "knowledge.sqlite3").listing() == [first]


def test_single_species_assumption_requires_one_named_pet_and_one_observed_animal(registry):
    reg, _ = registry
    first = reg.save("Pip", "cat")
    assert reg.resolve("cat") == {**first, "assumption": "single_named_pet_of_species"}
    assert reg.resolve("dog") is None
    assert reg.resolve("cat", observed_count=2) is None
    assert reg.resolve("cat", observed_count=True) is None
    second = reg.save("Miso", "cat")
    assert reg.resolve("cat") is None
    reg.remove(second["id"])
    assert reg.resolve("cat")["id"] == first["id"]


@pytest.mark.parametrize(
    "name,species,ident",
    [
        ("", "cat", None),
        ("Pip\nMiso", "cat", None),
        ("x" * 81, "cat", None),
        ("Pip", "person", None),
        ("Pip", "Cat", None),
        ("Pip", "cat", "not-a-uuid"),
        ("Pip", "cat", ID),
    ],
)
def test_invalid_or_unknown_pet_commands_do_not_write(registry, name, species, ident):
    reg, store = registry
    with pytest.raises((ValueError, LookupError)):
        reg.save(name, species, ident)
    assert revisions(store) == []


def test_revision_bound_is_actionable(registry, monkeypatch):
    reg, _ = registry
    reg.save("Pip", "cat")
    monkeypatch.setattr(pets, "MAX_REVISIONS", 0)
    with pytest.raises(RuntimeError, match="safe read bound"):
        reg.listing()


def test_a_malformed_profile_never_becomes_absence(registry):
    reg, store = registry
    reg.save("Pip", "cat")
    store.put([knowledge.Record(OTHER, pets.PET_PROFILES, OTHER, 1, 1, 1, {"active": True})])
    with pytest.raises(RuntimeError, match="malformed"):
        reg.listing()


def test_dashboard_local_control_and_unavailable_evidence_are_explicit(tmp_path, monkeypatch, registry):
    reg, _ = registry
    monkeypatch.setattr(pets, "at", lambda state: reg)
    assert routes.save({"name": "Pip", "species": "cat"}, False, tmp_path)[0] == 403
    code, reply = routes.save({"name": "Pip", "species": "cat"}, True, tmp_path)
    assert code == 200
    ident = reply["pet"]["id"]
    assert routes.listing(tmp_path)[1]["pets"][0]["id"] == ident
    assert routes.remove({"id": ident}, False, tmp_path)[0] == 403
    assert routes.remove({"id": ident}, True, tmp_path) == (200, {"ok": True, "pets": []})
    broken = create_autospec(pets.PetRegistry, instance=True)
    broken.listing.side_effect = TimeoutError("offline")
    monkeypatch.setattr(pets, "at", lambda state: broken)
    code, reply = routes.listing(tmp_path)
    assert code == 503
    assert "pets" not in reply
    assert "unavailable" in reply["error"]


def test_a_pet_s_sex_is_optional_and_kept_across_revisions(registry):
    reg, store = registry
    may = reg.save("May", "cat", sex="female")
    pip = reg.save("Pip", "cat")
    assert [(p["name"], p["sex"]) for p in reg.listing()] == [("May", "female"), ("Pip", None)]
    assert "sex" not in revisions(store)[1].value, "an unset sex isn't written"
    reg.save("May", "cat", may["id"], sex="")
    assert [(p["name"], p["sex"]) for p in reg.listing()] == [("May", None), ("Pip", None)]
    with pytest.raises(ValueError, match="female, male, or left unset"):
        reg.save("Pip", "cat", pip["id"], sex="tabby")
    assert len(revisions(store)) == 3


def test_reading_an_empty_roster_creates_no_store(tmp_path):
    reg = pets.PetRegistry(tmp_path / "knowledge.sqlite3")
    assert reg.listing() == []
    assert not (tmp_path / "knowledge.sqlite3").exists(), "a read must not create the owner's knowledge store"


def test_the_roster_reads_beside_another_reader(registry, tmp_path, monkeypatch):
    # Another connection mid-read, as the dashboard's warm-ups are after a start. In rollback journaling a
    # write can't commit until it ends, and every read used to open the store anew with a schema write: it waited out
    # the busy timeout (shortened here, so that fails at once instead of taking 10 s).
    reg, _ = registry
    pet = reg.save("Pip", "cat")
    monkeypatch.setattr(knowledge, "BUSY_TIMEOUT_S", 0.2)
    monkeypatch.setattr(knowledge, "LOCK_WAIT_S", 0)  # and no retries, so a lock it meets fails at once
    other = sqlite3.connect(tmp_path / "knowledge.sqlite3", isolation_level=None)
    other.execute("BEGIN")
    other.execute("SELECT count(*) FROM sqlite_schema").fetchone()
    try:
        assert reg.listing() == [pet]
        assert pet_recognition.labels(tmp_path) == {}
    finally:
        other.rollback()
        other.close()


TRACK = "a893c5e7290e:dog:1"
LATER = "409fe07240ef:dog:24"


def test_a_cards_photo_and_hidden_sightings_outlast_an_edit_and_the_process(registry, tmp_path):
    reg, _ = registry
    bess = reg.save("Bess", "dog")
    reg.set_photo(bess["id"], TRACK)
    reg.hide(bess["id"], LATER)
    reg.save("Bess B", "dog", bess["id"], sex="female")  # the Edit form sends neither
    (again,) = pets.PetRegistry(tmp_path / "knowledge.sqlite3").listing()
    assert (again["name"], again["photo"], again["hidden"]) == ("Bess B", TRACK, [LATER])


def test_hiding_the_chosen_photo_unchooses_it_and_choosing_a_hidden_one_shows_it(registry):
    reg, _ = registry
    bess = reg.save("Bess", "dog")
    reg.set_photo(bess["id"], TRACK)
    hidden = reg.hide(bess["id"], TRACK)
    assert (hidden["photo"], hidden["hidden"]) == (None, [TRACK])
    chosen = reg.set_photo(bess["id"], TRACK)
    assert (chosen["photo"], chosen["hidden"]) == (TRACK, [])
    shown = reg.hide(bess["id"], LATER, hidden=False)  # showing one that wasn't hidden changes nothing
    assert reg.listing() == [shown]
    assert (shown["photo"], shown["hidden"]) == (TRACK, [])


@pytest.mark.parametrize(
    "track",
    [
        pytest.param("", id="empty"),
        pytest.param("../etc/passwd", id="a-path"),
        pytest.param("a893c5e7290e:dog", id="no-number"),
        pytest.param(7, id="not-text"),
    ],
)
def test_a_card_choice_names_a_recorded_track_or_writes_nothing(registry, track):
    reg, store = registry
    bess = reg.save("Bess", "dog")
    with pytest.raises(ValueError, match="track"):
        reg.set_photo(bess["id"], track)
    with pytest.raises(ValueError, match="track"):
        reg.hide(bess["id"], track)
    assert len(revisions(store)) == 1


def test_the_card_route_takes_each_choice_and_refuses_the_rest(tmp_path, monkeypatch, registry):
    reg, _ = registry
    monkeypatch.setattr(pets, "at", lambda state: reg)
    bess = reg.save("Bess", "dog")

    def choose(action, track=TRACK, control=True, pet=bess["id"]):
        return routes.card({"id": pet, "track": track, "action": action}, control, tmp_path)

    assert choose("main")[1]["pets"][0]["photo"] == TRACK
    assert choose("newest", track=None)[1]["pets"][0]["photo"] is None
    assert choose("hide")[1]["pets"][0]["hidden"] == [TRACK]
    assert choose("show")[1]["pets"][0]["hidden"] == []
    assert choose("main", control=False)[0] == 403
    assert choose("crop")[0] == 400
    assert choose("main", track="nope")[0] == 400
    assert choose("main", pet=OTHER)[0] == 404
