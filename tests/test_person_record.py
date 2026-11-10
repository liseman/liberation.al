"""One person record: a person's phone number, added mark, and old names live on their `Person` in the
registry, so a rename or a merge carries them. contacts.json and people_added.json are read models, written beside
the registry and read only for a name the registry has nothing for."""

import json
from unittest.mock import patch

import pytest

from liberational import config as cfg
from liberational.dashboard import call_log, context, people_list, people_reads, people_writes, server
from liberational.people import contacts, profiles, registry
from liberational.phone import outbound

RERUN = {"people": 0, "assertions": 0, "phones": 0, "added": 0}


def bodies(state):
    log = registry.at(state).log
    return [json.dumps(s.event.to_dict()) for st in (registry.PEOPLE, registry.ASSERTIONS) for s in log.read(st)]


def fresh(state):
    """A registry rebuilt from the log alone."""
    return registry.Registry(registry.at(state).log)


def read(path):
    return json.loads(path.read_text())


def profile(name, *facts):
    return (
        "\n".join([profiles.heading(name), "", "Updated 2026-10-27T09:00 from 20 lines.", "", *facts, "", "Facts."])
        + "\n"
    )


def test_a_phone_and_the_added_mark_follow_a_rename(tmp_path):
    reg = registry.at(tmp_path)
    dana = reg.contact("Dana", "test", phone="+14155550123", added=True)
    reg.rename("Dana", "Dana Cole", "test")
    p = fresh(tmp_path).person(dana)
    assert p is not None
    assert (p.display_name, p.phone, p.added, p.aliases) == ("Dana Cole", "+14155550123", True, ("Dana",))


def test_contact_writes_only_changes_and_fill_never_replaces_a_number(tmp_path):
    reg = registry.at(tmp_path)
    dana = reg.contact("Dana", "test", phone="+14155550123")
    reg.contact("dana", "test", phone="+14155550123")  # nothing new
    reg.contact("Dana", "test", phone="+14155550999", fill=True)  # the backfill: she has one already
    reg.contact("Dana", "test", added=False)  # leaves the mark as it is
    assert len(bodies(tmp_path)) == 1
    reg.contact("Dana", "test", phone="")  # clears it
    p = fresh(tmp_path).person(dana)
    assert p is not None
    assert p.phone is None
    with pytest.raises(ValueError):
        reg.describe("Dana", "test", {"phone": "+14155550123"})  # a tag never sets a number


def test_a_merge_fills_a_missing_number_and_keeps_the_survivors_own(tmp_path):
    reg = registry.at(tmp_path)
    reg.contact("Dana", "test", phone="+14155550123", added=True)
    alice = reg.ensure("Alice", "test")
    reg.merge("Dana", "Alice", "test")  # Alice had no number: she gets Dana's
    p = fresh(tmp_path).person(alice)
    assert p is not None
    assert (p.phone, p.added, p.aliases) == ("+14155550123", True, ("Dana",))
    reg.contact("Erin", "test", phone="+14155550999")
    reg.rename("Erin", "Erin Lee", "test")
    reg.merge("Erin Lee", "Alice", "test")  # both have one: Alice keeps hers, like a role
    p = fresh(tmp_path).person(alice)
    assert p is not None
    assert (p.phone, p.aliases) == ("+14155550123", ("Dana", "Erin Lee", "Erin"))
    assert any("+14155550999" in b for b in bodies(tmp_path))  # Erin's number is still in the log


def test_forgetting_purges_the_number(tmp_path):
    reg = registry.at(tmp_path)
    reg.contact("Dana", "test", phone="+14155550123", added=True)
    reg.rename("Dana", "Dana Cole", "test")
    reg.forget("Dana Cole", "test")
    assert not any("+14155550123" in b or "Dana" in b for b in bodies(tmp_path))
    assert fresh(tmp_path).person_id("Dana Cole") is None


@pytest.fixture
def old_files(tmp_path):
    """contacts.json and people_added.json from before the backfill."""
    contacts_file = {"dana cole": "+14155550123", "Erin (phone)": "+14155550124", "_note": "x", "Gus": ""}
    (tmp_path / "contacts.json").write_text(json.dumps(contacts_file))
    (tmp_path / "people_added.json").write_text(json.dumps(["Eve", "dana cole"]))
    return tmp_path


def test_the_backfill_lands_numbers_and_added_names_in_the_registry_once(old_files):
    reg = registry.at(old_files)
    reg.ensure("Dana Cole", "test")  # already known from step 3, without a number
    first = registry.backfill(old_files, old_files / "faces" / "db.npz")
    assert (first["phones"], first["added"]) == (2, 2)
    count = len(bodies(old_files))
    assert registry.backfill(old_files, old_files / "faces" / "db.npz") == RERUN
    assert len(bodies(old_files)) == count
    people = {p.display_name: (p.phone, p.added) for p in reg.people()}
    assert people == {
        "Dana Cole": ("+14155550123", True),
        "Erin": ("+14155550124", False),
        "Eve": (None, True),
        "Gus": (None, False),  # listed with no number
    }


def test_a_backfill_rerun_never_replaces_a_number_set_since(old_files):
    registry.backfill(old_files, old_files / "faces" / "db.npz")
    book = contacts.Book.at(old_files)
    book.set_phone("Dana Cole", "+14155550999", "dashboard")
    (old_files / "contacts.json").write_text(json.dumps({"dana cole": "+14155550123"}))  # an old copy
    assert registry.backfill(old_files, old_files / "faces" / "db.npz") == RERUN
    assert book.number("Dana Cole") == "+14155550999"


def test_before_the_backfill_the_book_reads_the_files_and_makes_no_log(tmp_path):
    (tmp_path / "contacts.json").write_text(json.dumps({"dana": "+14155550123", "_note": "x"}))
    (tmp_path / "people_added.json").write_text(json.dumps(["Eve"]))
    book = contacts.Book.at(tmp_path)
    assert (book.phones(), book.added_names()) == ({"dana": "+14155550123"}, ["Eve"])
    assert not (tmp_path / "events.sqlite3").exists()


def test_adding_writes_the_registry_and_both_files(tmp_path):
    contacts.Book.at(tmp_path).add("Dana", "+14155550123", role="CEO")
    reg = registry.at(tmp_path)
    p = reg.person(reg.ensure("Dana", "test"))
    assert p is not None
    assert (p.phone, p.added, p.role) == ("+14155550123", True, "CEO")
    assert read(tmp_path / "contacts.json") == {"Dana": "+14155550123"}
    assert read(tmp_path / "people_added.json") == ["Dana"]
    assert (tmp_path / "contacts.json").stat().st_mode & 0o777 == 0o600


def test_the_registry_wins_and_the_file_fills_in_only_for_a_name_it_has_no_number_for(tmp_path):
    book = contacts.Book.at(tmp_path)
    book.add("Dana", "+14155550123")
    # Edited by hand after the backfill: a stale number for Dana, and someone the registry has no number for.
    (tmp_path / "contacts.json").write_text(json.dumps({"dana": "+14155550999", "Hana": "+14155550124"}))
    assert book.phones() == {"Dana": "+14155550123", "Hana": "+14155550124"}
    (tmp_path / "contacts.json").unlink()
    (tmp_path / "people_added.json").write_text("[]")
    assert (book.phones(), book.added_names()) == ({"Dana": "+14155550123"}, ["Dana"])  # from the registry


def test_a_rename_in_the_registry_carries_the_number_without_the_file(tmp_path):
    book = contacts.Book.at(tmp_path)
    book.add("Dana", "+14155550123")
    registry.at(tmp_path).rename("Dana", "Dana Cole", "dashboard")
    (tmp_path / "contacts.json").write_text("{}")  # the read model was never told
    assert book.phones() == {"Dana Cole": "+14155550123"}


def test_set_phone_and_clear_write_both(tmp_path):
    book = contacts.Book.at(tmp_path)
    book.add("Dana", "+14155550123")
    book.set_phone("dana", "+14155550999", "cli")
    assert read(tmp_path / "contacts.json") == {"dana": "+14155550999"}  # one entry per person
    book.set_phone("Dana", "", "cli")
    assert (read(tmp_path / "contacts.json"), book.number("Dana")) == ({}, None)


def test_follow_and_forget_update_the_files(tmp_path):
    (tmp_path / "contacts.json").write_text(json.dumps({"dana": "+1", "Alice": "+2", "_note": "x"}))
    (tmp_path / "people_added.json").write_text(json.dumps(["Dana", "Alice"]))
    book = contacts.Book.at(tmp_path)
    book.follow("Dana", "Alice")  # a merge: Alice keeps her own number
    assert read(tmp_path / "contacts.json") == {"Alice": "+2", "_note": "x"}
    assert read(tmp_path / "people_added.json") == ["Alice"]
    book.follow("Alice", "Alicia")
    book.forget("alicia")
    assert (read(tmp_path / "contacts.json"), read(tmp_path / "people_added.json")) == ({"_note": "x"}, [])


def test_calls_find_numbers_through_the_registry(tmp_path):
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        (tmp_path / "contacts.json").write_text(json.dumps({"Dana": "+14155550123"}))
        assert outbound.contacts() == {"dana": "+14155550123"}  # before the backfill: the file
        registry.backfill(tmp_path, tmp_path / "faces" / "db.npz")
        registry.at(tmp_path).rename("Dana", "Dana Cole", "dashboard")
        (tmp_path / "contacts.json").write_text("{}")
        assert outbound.contacts() == {"dana cole": "+14155550123"}
        with (
            patch.object(outbound, "public_url", side_effect=RuntimeError),
            patch.object(outbound.requests, "post", side_effect=AssertionError("must not dial")),
        ):
            assert "phone.public_host" in outbound.place("dana cole", "hi")[1]  # found, stopped before dialing
            assert outbound.place("dana", "hi")[1] == "I don't have a number for dana yet."


def test_the_phone_command_shows_sets_and_clears_a_number(tmp_path, capsys):
    registry.at(tmp_path).ensure("Dana", "test")

    def cli(*args):
        with pytest.raises(SystemExit) as done:
            registry.main(["--state-dir", str(tmp_path), "phone", *args])
        return done.value.code, capsys.readouterr().out.strip()

    assert cli("Dana") == (0, "Dana: no number")
    assert cli("dana", "+14155550123") == (0, "dana: +14155550123")
    assert read(tmp_path / "contacts.json") == {"dana": "+14155550123"}
    assert cli("Dana", "--clear") == (0, "Dana: no number")
    assert cli("Dana", "555")[0] == 1
    assert cli("Nobody", "+14155550123")[0] == 1


def test_a_rename_moves_the_profile_and_its_heading(tmp_path):
    (tmp_path / "dana.md").write_text(profile("dana", "- Dana runs a bakery."))
    assert profiles.rename(tmp_path, "Dana", "Dana Cole") == 1
    assert sorted(f.name for f in tmp_path.iterdir()) == ["dana cole.md"]
    text = (tmp_path / "dana cole.md").read_text()
    assert text.startswith("# Dana Cole\n")
    assert "- Dana runs a bakery." in text
    assert (tmp_path / "dana cole.md").stat().st_mode & 0o777 == 0o600
    assert profiles.path_for(tmp_path, "Dana Cole") == tmp_path / "dana cole.md"  # where main writes


def test_a_rename_that_finds_the_target_under_another_spelling_keeps_it(tmp_path):
    # On a case-insensitive disk (the Mac default) "Dana.md" and "dana.md" are one file. A second path to the same
    # entry, through a linked folder, is the same situation on any disk.
    folder = tmp_path / "people"
    folder.mkdir()
    (folder / "dana.md").write_text(profile("dana", "- Dana runs a bakery."))
    alias = tmp_path / "people-alias"
    alias.symlink_to(folder, target_is_directory=True)
    with patch.object(profiles, "files_of", return_value=[alias / "dana.md"]):
        assert profiles.rename(folder, "Dana", "dana") == 1
    assert "- Dana runs a bakery." in (folder / "dana.md").read_text()  # the profile is not deleted


def test_a_merge_adds_the_facts_the_survivor_lacks(tmp_path):
    (tmp_path / "dana.md").write_text(profile("dana", "- Dana runs a bakery.", "- Likes tea."))
    (tmp_path / "alice.md").write_text(profile("alice", "- Likes tea."))
    profiles.rename(tmp_path, "dana", "Alice")
    assert sorted(f.name for f in tmp_path.iterdir()) == ["alice.md"]
    lines = (tmp_path / "alice.md").read_text().splitlines()
    assert (lines[0], lines[-1]) == ("# Alice", "Facts.")
    assert [ln for ln in lines if ln.startswith("- ")] == ["- Likes tea.", "- Dana runs a bakery."]
    (tmp_path / "erin.md").write_text(profile("erin", "- Erin plays chess."))
    (tmp_path / "gus.md").write_text(profile("gus", profiles.UNCLEAR))
    profiles.rename(tmp_path, "Erin", "Gus")  # a fact makes "unclear" untrue
    assert [ln for ln in (tmp_path / "gus.md").read_text().splitlines() if ln.startswith("- ")] == [
        "- Erin plays chess."
    ]


def test_files_of_matches_any_spelling(tmp_path):
    for n in ("Dana Cole.md", "dana  cole.md", "dana.md", "index.json"):
        (tmp_path / n).write_text("x")
    assert [f.name for f in profiles.files_of(tmp_path, ["DANA COLE"])] == ["Dana Cole.md", "dana  cole.md"]
    assert profiles.files_of(tmp_path / "missing", ["Dana"]) == []
    assert profiles.rename(tmp_path, "Nobody", "Someone") == 0


def no_faces():
    raise RuntimeError("no face database here")


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    """The People tab on a state dir of its own, with no lines and no face database. Returns the state dir."""
    state = tmp_path / "state"
    (state / "people").mkdir(parents=True)
    for name, path in {}.items():
        monkeypatch.setattr(server, name, state / path)
    monkeypatch.setattr(people_list, "PEOPLE_ADDED", state / "people_added.json")
    monkeypatch.setattr(people_list, "CONTACTS_FILE", state / "contacts.json")
    monkeypatch.setattr(call_log, "CALLS_DIR", state / "calls")
    monkeypatch.setattr(context, "STATE", state)
    # conftest supplies an empty speech log; keep the real readers with this fixture's temporary call storage so
    # direct roster reads and mutation replies (which pass a shared raw/view snapshot) exercise the same path.
    monkeypatch.setattr(people_list, "_face_last_seen_at", dict)
    monkeypatch.setattr(people_list, "_faces_mod", no_faces)
    with cfg.override({"paths": {"state": str(state), "faces": str(tmp_path / "faces")}}):
        yield state


def roster():
    return {p["name"]: p for p in people_list.people_roster()}


def aliases(name):
    summary = people_reads.person_summary(name)
    assert summary is not None
    return summary["aliases"]


def test_the_roster_reads_the_number_from_the_registry(dashboard):
    people_writes.add_person("Dana", "+14155550123")
    (dashboard / "contacts.json").write_text(json.dumps({"Dana": "+14155550999"}))  # a stale hand edit
    (dashboard / "people_added.json").write_text("[]")
    assert roster()["Dana"]["number"] == "+14155550123"


def test_a_rename_carries_the_number_the_profile_and_the_old_name(dashboard):
    people = dashboard / "people"
    people_writes.add_person("Dana", "+14155550123")
    (people / "dana.md").write_text(profile("dana", "- Dana runs a bakery."))
    (people / "dana cole.md").write_text(profile("dana cole", "- Someone else."))  # left by a past rename
    code, body = people_writes.rename_person("Dana", "Dana Cole", control=True)
    assert (code, body["name"], body["notes"]) == (200, "Dana Cole", [])
    (dashboard / "contacts.json").unlink()  # the registry alone has it now
    assert roster()["Dana Cole"]["number"] == "+14155550123"
    assert aliases("Dana Cole") == ["Dana"]
    assert sorted(f.name for f in people.iterdir()) == ["dana cole.md"]
    assert "- Dana runs a bakery." in (people / "dana cole.md").read_text()
    parked = list((dashboard.parent / "premove" / "people").iterdir())
    assert [f.read_text() for f in parked] == [profile("dana cole", "- Someone else.")]  # moved aside
    assert read(dashboard / "people_added.json") == ["Dana Cole"]


def test_a_rename_reply_keeps_people_with_retained_room_transcripts(dashboard, speech):
    people_writes.add_person("Dana", "+14155550123")
    people_writes.add_person("Erin")
    speech.say({"ts_start": 1_700_000_000, "speaker": "Erin", "text": "I enjoy chess."})

    code, body = people_writes.rename_person("Dana", "Dana Cole", control=True)

    assert code == 200
    assert {p["name"]: (p["room"], p.get("number")) for p in body["people"]} == {
        "Dana Cole": (0, "+14155550123"),
        "Erin": (1, None),
    }
    assert body["unknown"]["voices"] == []


def test_a_long_new_name_is_cut_without_a_trailing_space(dashboard):
    people_writes.add_person("Dana", "+14155550123")
    code, body = people_writes.rename_person("Dana", "A" * 39 + " Bee", control=True)
    assert (code, body["name"]) == (200, "A" * 39)
    assert "A" * 39 in roster()


def test_a_merge_into_someone_off_the_roster_combines_their_profiles(dashboard):
    people = dashboard / "people"
    people_writes.add_person("Dana", "+14155550123")
    registry.at(dashboard).ensure("Erin", "test")  # known to the registry, but nothing puts Erin on the roster
    (people / "dana.md").write_text(profile("dana", "- Dana runs a bakery."))
    (people / "erin.md").write_text(profile("erin", "- Erin plays chess."))
    assert "Erin" not in roster()
    assert people_writes.rename_person("Dana", "Erin", control=True)[0] == 200
    text = (people / "erin.md").read_text()
    assert "- Erin plays chess." in text  # Erin's own profile is kept, not parked
    assert "- Dana runs a bakery." in text


def test_a_merge_combines_numbers_and_says_which_one_was_not_kept(dashboard):
    people = dashboard / "people"
    people_writes.add_person("Dana", "+14155550123")
    people_writes.add_person("Alice")
    (people / "dana.md").write_text(profile("dana", "- Dana runs a bakery."))
    (people / "alice.md").write_text(profile("alice", "- Alice likes tea."))
    code, body = people_writes.rename_person("Dana", "alice", control=True)
    assert (code, body["name"], body["merged"], body["notes"]) == (200, "Alice", True, [])
    (dashboard / "contacts.json").unlink()  # so the number below comes from the registry's merge
    assert roster()["Alice"]["number"] == "+14155550123"  # Alice had none: she gets Dana's
    assert "Dana" not in roster()
    facts = [ln for ln in (people / "alice.md").read_text().splitlines() if ln.startswith("- ")]
    assert facts == ["- Alice likes tea.", "- Dana runs a bakery."]
    people_writes.add_person("Erin", "+14155550999")
    _, body = people_writes.rename_person("Erin", "Alice", control=True)
    assert body["notes"] == ["Alice keeps the number +14155550123. Erin's number, +14155550999, was not kept."]
    assert roster()["Alice"]["number"] == "+14155550123"
    assert read(dashboard / "contacts.json") == {"Alice": "+14155550123"}
    assert aliases("Alice") == ["Dana", "Erin"]


def test_a_merge_keeps_a_number_the_survivor_has_only_in_the_file(dashboard):
    people_writes.add_person("Dana", "+14155550123")
    people_writes.add_person("Hana")
    contacts_file = dashboard / "contacts.json"
    contacts_file.write_text(json.dumps({**read(contacts_file), "Hana": "+14155550124"}))  # by hand
    _, body = people_writes.rename_person("Dana", "Hana", control=True)
    assert body["notes"] == ["Hana keeps the number +14155550124. Dana's number, +14155550123, was not kept."]
    contacts_file.unlink()
    assert roster()["Hana"]["number"] == "+14155550124"  # what the note said, from the registry


def test_before_the_backfill_the_files_still_list_everyone(dashboard):
    # An install that has not run the backfill: the registry has people but no numbers or added marks yet.
    registry.at(dashboard).ensure("Dana", "step-3-backfill")
    (dashboard / "contacts.json").write_text(json.dumps({"dana": "+14155550123"}))
    (dashboard / "people_added.json").write_text(json.dumps(["Eve"]))
    people_writes.add_person("Gus", "+14155550124")  # one new write must not hide the rest
    assert {k: v.get("number") for k, v in roster().items()} == {
        "dana": "+14155550123",
        "Eve": None,
        "Gus": "+14155550124",
    }
    assert outbound.contacts() == {"gus": "+14155550124", "dana": "+14155550123"}
