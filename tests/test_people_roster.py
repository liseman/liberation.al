"""A phone-call line keeps the label the bridge heard it under. People and profiles read it as of the call, as
profiles.load_calls does: the person's current name, and nothing for someone forgotten since."""

import json
from types import SimpleNamespace

import pytest

from liberational import config as cfg
from liberational.dashboard import call_log, context, server
from liberational.people import profiles, registry

NO_FACES = SimpleNamespace(counts=dict, unknown_groups=list, rename=lambda a, b: 0)


@pytest.fixture
def state(tmp_path, monkeypatch):
    (tmp_path / "speakers.json").write_text(json.dumps({"speakers": []}))
    for name, path in {
        "CONTACTS_FILE": "contacts.json",
        "PEOPLE_ADDED": "people_added.json",
    }.items():
        monkeypatch.setattr(server, name, tmp_path / path)
    monkeypatch.setattr(call_log, "CALLS_DIR", tmp_path / "calls")
    monkeypatch.setattr(context, "STATE", tmp_path)
    monkeypatch.setattr(server, "_room_lines", lambda *a, **k: [])
    monkeypatch.setattr(server, "_faces_mod", lambda: NO_FACES)
    monkeypatch.setattr(server, "_face_last_seen_at", dict)
    with cfg.override({"owner": {"name": "Olive"}, "paths": {"state": str(tmp_path)}}):
        yield tmp_path


@pytest.fixture
def no_forgotten_reply():
    """assistant.forgotten_reply off: no hash of a forgotten name is kept, so the registry decides alone."""
    with cfg.override({"assistant": {"forgotten_reply": False}}):
        yield


def call(state, label, folder="090000_CA1", started="2026-10-20T09:00:00"):
    d = state / "calls" / started[:10] / folder
    d.mkdir(parents=True)
    (d / "call.json").write_text(json.dumps({"started": started}))
    line = {"t": 1.5, "speaker": "caller", "speaker_label": label, "text": "I live in Boston with my sister"}
    (d / "transcript.jsonl").write_text(json.dumps(line) + "\n")


def roster():
    return sorted((p["name"], p["calls"]) for p in server.people_roster())


def test_a_forgotten_callers_lines_build_no_profile_with_the_forgotten_reply_off(
    state, no_forgotten_reply, monkeypatch
):
    # Finding 1: with assistant.forgotten_reply off no hash is kept, and the profile came back from calls.
    monkeypatch.setattr(profiles, "STATE", state)
    monkeypatch.setattr(profiles, "CALLS", state / "calls")
    reg = registry.at(state)
    reg.ensure("Dana", "test")
    call(state, "Dana")
    reg.forget("Dana", "test")
    assert profiles.load_calls() == []
    # pack-people runs in a process of its own, so it reads a projection rebuilt from the log
    registry.close_all()  # a later process, rebuilt from the log
    assert profiles.load_calls() == []


def test_a_renamed_caller_is_one_person_under_their_new_name(state):
    # Finding 7/17: the raw label made a second "Dana" entry, and Dee's profile had no calls.
    reg = registry.at(state)
    reg.ensure("Dana", "test")
    call(state, "Dana")
    reg.rename("Dana", "Dee", "test")
    assert roster() == [("Dee", 1)]
    dee = server.person_summary("Dee")
    assert dee is not None
    assert (dee["name"], dee["call_lines"]) == ("Dee", 1)
    reg.ensure("Dana", "test", added=True)  # someone new takes the old name, and calls after that are theirs
    call(state, "Dana", "120000_CA2", "2099-01-01T12:00:00")
    assert roster() == [("Dana", 1), ("Dee", 1)]


def test_a_deleted_caller_leaves_people(state, no_forgotten_reply):
    # Finding 7/17: a forgotten caller stayed on People for good, and Delete on the row did nothing.
    reg = registry.at(state)
    reg.ensure("Dana Cole", "test")
    call(state, "Dana Cole")
    call(state, "Zed", "090100_CA2")  # a label nobody in the registry has stays as it is
    assert roster() == [("Dana Cole", 1), ("Zed", 1)]
    reg.forget("Dana Cole", "test")
    assert roster() == [("Zed", 1)]
    assert server.person_summary("Dana Cole") == {"name": "Dana Cole", "lines": 0}
