"""whisper_names: the facts it whispers come from the person's profile, however many words the name has."""

import pytest

from liberational.people import profiles, whisper_names


@pytest.fixture
def people(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper_names, "PEOPLE", tmp_path)
    return tmp_path


def test_a_name_of_two_words_finds_its_profile(people):
    profiles.path_for(people, "Dana Cole").write_text("# Dana Cole\n\n- Dana runs the bakery on Elm Street\n")
    assert whisper_names.facts_for("Dana Cole") == ["Dana runs the bakery on Elm Street"]


def test_no_profile_no_facts(people):
    assert whisper_names.facts_for("Erin") == []
