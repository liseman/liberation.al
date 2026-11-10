"""Attributes from tags (a person's attributes, "Records" in docs/DOMAIN-MODEL.md): "that's Alice, CEO of Acme" parsed into a name, a role,
and an organization by fixed rules (people/tags.py), from speech and from the dashboard, and kept as person events.
Everything runs on temp dirs, with made-up people. No model is called."""

import functools
import http.client
import json
import time
import urllib.parse
from unittest.mock import patch

import pytest

from liberational import config as cfg
from liberational.assistant import voice
from liberational.assistant.voice_auth import Grant
from liberational.dashboard import (
    context,
    known_voices,
    people_cards,
    people_corrections,
    people_list,
    people_writes,
)
from liberational.people import identity, registry, tags
from speech_fake import fake
from voices import write_voices

ACME_CEO = {"role": "CEO", "organization": "Acme"}


def types(state):
    return [str(s.event.type) for s in registry.at(state).log.read(registry.PEOPLE)]


def fresh(state):
    """The registry rebuilt from the log, so a test reads what was written rather than what is cached."""
    return registry.Registry(registry.at(state).log)


def person(state, name):
    reg = registry.at(state)
    pid = reg.person_id(name)
    return reg.person(pid) if pid else None


def said(text):
    t = identity.spoken_tag(text)
    return (t.name, t.role, t.organization) if t else None


SPOKEN = {
    "that's Alice, CEO of Acme": ("Alice", "CEO", "Acme"),
    "That's Alice, CEO of Acme.": ("Alice", "CEO", "Acme"),
    "that is Alice, the CEO of Acme": ("Alice", "CEO", "Acme"),
    "that's Alice the CEO of Acme": ("Alice", "CEO", "Acme"),
    "this is Dana, the head of sales at Initech": ("Dana", "head of sales", "Initech"),
    "meet Dana, VP of Engineering at Initech": ("Dana", "VP of Engineering", "Initech"),
    "that's Alice, she's the CEO of Acme": ("Alice", "CEO", "Acme"),
    "that's Alice, who's a designer at Acme": ("Alice", "designer", "Acme"),
    "that's Alice, Acme's CEO": ("Alice", "CEO", "Acme"),
    "say hi to Dana Cole, senior engineer at Acme Labs": ("Dana Cole", "senior engineer", "Acme Labs"),
    "this is my friend Dana, a nurse at Mercy General": ("Dana", "nurse", "Mercy General"),
    "that's Alice, professor at the University of Michigan": ("Alice", "professor", "University of Michigan"),
    "that’s Alice, partner at Acme": ("Alice", "partner", "Acme"),
    "I'm Alice, CEO of Acme": ("Alice", "CEO", "Acme"),
    "my name is Alice and I'm a designer at Acme": ("Alice", "designer", "Acme"),
    "Hey everyone, this is Alice, CEO of Acme. She runs the place.": ("Alice", "CEO", "Acme"),
    # only a role, or only an organization
    "that's Alice, CEO": ("Alice", "CEO", None),
    "meet Dana, our new CTO": ("Dana", "new CTO", None),
    "that's Alice, VP of Engineering": ("Alice", "VP of Engineering", None),
    "I'm Alice, I work at Acme": ("Alice", None, "Acme"),
    "that's Alice, she works at Acme": ("Alice", None, "Acme"),
    "that's Alice, CTO for Acme": ("Alice", "CTO", "Acme"),
    # a plain name has no attributes
    "that's Alice": ("Alice", None, None),
    "that's Alice.": ("Alice", None, None),
    "this is Alice Smith": ("Alice Smith", None, None),
    "that's Mary Cook": ("Mary Cook", None, None),  # a surname, not a job
    "meet Dana": ("Dana", None, None),
}


@pytest.mark.parametrize(("text", "want"), SPOKEN.items(), ids=SPOKEN)
def test_a_spoken_tag_is_parsed(text, want):
    assert said(text) == want


@pytest.mark.parametrize(
    "text",
    [
        "that's Alice, she's here",
        "that's Alice and Dana",
        "that's Alice, a friend",
        "that's Alice, my partner",  # a relationship, not a role
        "that's Alice from Boston",  # a hometown, not an employer
        "that's Alice from Acme",
        "that's Alice with Dana",
        "that's Alice at Starbucks",  # a place
        "that's Alice at the door",
        "that's Alice, CEO of Acme, she's great",  # more than one clause
        "that's Alice from Acme and Dana from Initech",
        "that's Alice, the one I told you about",
        "that's Alice CEO of Acme",  # no comma and no article
        "that's Alice, she's the best",
        "that's great",
        "I think that's right",
        "that's alice, ceo of acme",  # a lowercase name is not a spoken name
        "that's Alice, Bob's lawyer",  # a client, not an employer
        "that's Alice, Dana's assistant",
        "that's Alice, Dana's doctor",
        "that's Alice, a lawyer for Bob Smith",
        "that's Alice, she works for Dana",
        "that's Alice, I work at Acme",  # the speaker, not Alice
        "that's Alice, I'm a nurse",
        "meet Dana, I'm the CTO of Acme",
    ],
)
def test_what_must_not_yield_attributes(text):
    t = identity.spoken_tag(text)
    assert (t.attributes() if t else {}) == {}


def test_the_assistants_command_takes_only_a_tag_with_attributes():
    t = identity.tag_command("that's Alice, CEO of Acme")
    assert t is not None
    assert t.attributes() == ACME_CEO
    assert identity.tag_command("that's Alice") is None  # INTRO_COMMAND's, as before
    assert identity.tag_command("that's Alice, she's here") is None
    assert identity.tag_command("that's Alice, I work at Acme") is None  # the wearer's job, not Alice's
    assert identity.tag_command("I'm Alice, CEO of Acme") is None  # not the wearer introducing someone
    assert identity.tag_command("so that's Alice, CEO of Acme") is None  # the command starts the request
    assert identity.INTRO_COMMAND.match("that's Alice, CEO of Acme") is None  # plain tags are unchanged


def test_the_tag_reads_back_as_a_phrase():
    assert tags.Tag("Alice", "CEO", "Acme").described() == "CEO at Acme"
    assert tags.Tag("Alice", "CEO").described() == "CEO"
    assert tags.Tag("Alice", organization="Acme").described() == "at Acme"
    assert tags.Tag("Alice").described() == ""


TYPED = {
    "Alice, CEO of Acme": ("Alice", "CEO", "Acme"),
    "José Ruiz, the head of sales at Initech": ("José Ruiz", "head of sales", "Initech"),
    # a plain typed name is left to _person_name
    "Dana Cole": None,
    "Mary Cook": None,
    "Alice from Acme": None,
    "alice, ceo of acme": None,
    "Alice and Dana": None,
    "Dana O'Neil": None,
    "Bob the plumber": None,
    "Sam the intern": None,
    "Dana the Cook": None,
    "Dana, Bob's lawyer": None,
    "Dana, I work at Acme": None,
}


@pytest.mark.parametrize(("text", "want"), TYPED.items(), ids=TYPED)
def test_a_typed_tag_is_parsed(text, want):
    t = people_writes._typed_tag(text)
    assert ((t.name, t.role, t.organization) if t else None) == want


# ---------------------------------------------------------------- registry ----
def test_a_new_person_is_registered_with_their_attributes(tmp_path):
    alice = registry.at(tmp_path).describe("Alice", "test", {"role": "CEO", "organization": " Acme  Corp "})
    assert types(tmp_path) == ["PersonRegistered"]
    p = fresh(tmp_path).person(alice)
    assert (p.display_name, p.role, p.organization) == ("Alice", "CEO", "Acme Corp")


def test_a_known_person_gets_a_change_and_an_unchanged_one_writes_nothing(tmp_path):
    reg = registry.at(tmp_path)
    alice = reg.ensure("Alice", "test")
    reg.describe("alice", "test", ACME_CEO)
    reg.describe("Alice", "test", {"role": "CEO"})  # nothing new
    assert types(tmp_path) == ["PersonRegistered", "PersonAttributesChanged"]
    reg.describe("Alice", "test", {"role": ""})  # clears the role, keeps the organization
    p = fresh(tmp_path).person(alice)
    assert (p.role, p.organization) == (None, "Acme")
    with pytest.raises(ValueError):
        reg.describe("Alice", "test", {"phone": "+14155550123"})


def test_attributes_follow_a_rename_and_survive_a_merge(tmp_path):
    reg = registry.at(tmp_path)
    alice = reg.describe("Alice", "test", ACME_CEO)
    reg.rename("Alice", "Alicia", "test")
    p = reg.person(alice)
    assert p is not None
    assert (p.display_name, p.role) == ("Alicia", "CEO")
    dana = reg.describe("Dana", "test", {"organization": "Initech"})
    reg.merge("Alicia", "Dana", "test")  # the survivor keeps her own organization, and gains the role
    p = fresh(tmp_path).person(dana)
    assert (p.role, p.organization) == ("CEO", "Initech")


def test_forgetting_purges_the_attributes(tmp_path):
    reg = registry.at(tmp_path)
    alice = reg.describe("Alice", "test", ACME_CEO)
    reg.describe("Alice", "test", {"role": "CTO"})
    reg.forget("Alice", "test")
    assert types(tmp_path) == ["PersonForgotten"]
    assert fresh(tmp_path).person(alice) is None
    bodies = [json.dumps(s.event.to_dict()) for s in registry.at(tmp_path).log.read("people")]
    assert not any("Acme" in b or "CTO" in b for b in bodies)


def test_the_cli_lists_role_and_organization(tmp_path, capsys):
    registry.at(tmp_path).describe("Alice", "test", ACME_CEO)
    registry.main(["--state-dir", str(tmp_path), "people"])
    assert "Alice  [CEO, Acme]" in capsys.readouterr().out


# ------------------------------------------------------- identity records ----
LINE = {"text": "This is Alice, CEO of Acme.", "ts_start": 990.0, "ts_end": 992.0, "speaker_id": 1}
NAMESAKES = "This is Alice Smith, CEO of Acme. And this is Alice Jones."


def heard(paths, line, confirmed):
    """The identity daemon's record for one line, with the model confirming `confirmed` as the name."""
    brain = identity.Identity(paths, clock=lambda: 1000.0)
    llm = {"intro": "third", "name": confirmed, "raw": "x"}
    with patch.object(identity, "llm_confirm", autospec=True, return_value=llm):
        return brain.on_line(line)


def test_an_introduction_keeps_its_attributes_until_it_is_applied(tmp_path):
    write_voices(tmp_path, {4: None})
    paths = identity.Paths(tmp_path)
    rec = heard(paths, LINE, "Alice")
    assert rec is not None
    assert rec["attributes"] == ACME_CEO
    assert person(tmp_path, "Alice") is None  # pending: nothing is said about anyone yet
    rec["speaker_id"] = 4
    identity.apply_record(paths, rec, "Alice")
    p = person(tmp_path, "Alice")
    assert (p.role, p.organization) == ("CEO", "Acme")


@pytest.mark.parametrize(
    ("text", "confirmed", "kept"),
    [
        pytest.param(LINE["text"], "Dana", False, id="another-name"),
        pytest.param(NAMESAKES, "Alice Jones", False, id="namesake-with-another-surname"),
        pytest.param(NAMESAKES, "Alice Smith", True, id="full-name"),
        pytest.param(NAMESAKES, "Alice", True, id="first-name"),
    ],
)
def test_the_attributes_belong_to_the_name_the_model_confirmed(tmp_path, text, confirmed, kept):
    rec = heard(identity.Paths(tmp_path), {**LINE, "text": text}, confirmed)
    assert rec is not None
    assert ("attributes" in rec) == kept


def test_an_overheard_tag_never_replaces_what_the_owner_set(tmp_path):
    write_voices(tmp_path, {4: None})
    paths = identity.Paths(tmp_path)
    registry.at(tmp_path).describe("Alice", "dashboard", {"role": "CTO"})
    rec = {"id": "i-1", "kind": "third_party", "speaker_id": 4, "attributes": ACME_CEO}
    identity.apply_record(paths, rec, "Alice")  # resolve_intros: a spoken introduction, not confirmed
    p = person(tmp_path, "Alice")
    assert (p.role, p.organization) == ("CTO", "Acme")  # the owner's role stands, the blank is filled
    rec = {"id": "i-2", "kind": "third_party", "speaker_id": 4, "attributes": {"role": "CEO"}}
    identity.apply_record(paths, rec, "Alice", basis=identity.Basis.OWNER_TAG)  # the owner confirms
    assert person(tmp_path, "Alice").role == "CEO"


def test_a_record_that_stays_pending_records_nothing(tmp_path):
    write_voices(tmp_path, {4: "Dana"})  # the voice names someone else, so the name is not applied
    rec = {"id": "i-1", "kind": "third_party", "speaker_id": 4, "attributes": {"role": "CEO"}}
    identity.apply_record(identity.Paths(tmp_path), rec, "Alice")
    assert rec["status"] == "pending"
    assert person(tmp_path, "Alice") is None


@pytest.mark.parametrize(
    ("on", "want"), [pytest.param(False, None, id="flag-off"), pytest.param(True, ("CEO", "Acme"), id="flag-on")]
)
def test_a_spoken_tag_gives_the_person_attributes_only_with_the_flag_on(tmp_path, monkeypatch, speech, on, want):
    write_voices(tmp_path, {4: None})  # the guest's voice, heard just now
    now = time.time()
    speech.say({"ts_start": now - 3, "ts_end": now - 1, "speaker": "Speaker 4", "speaker_id": 4, "text": "hello"})
    text = "that's Alice, CEO of Acme."
    grant = Grant(text, "gateway", True, True, False, False, True)
    # The real introduction on this test's state: its default state dir is bound when the module loads.
    monkeypatch.setattr(identity, "collar_intro", functools.partial(identity.collar_intro, state_dir=tmp_path))
    monkeypatch.setattr(voice.CONVERSATION, "accept", lambda *a: grant)
    monkeypatch.setattr(voice, "owner_answer", lambda *a, **k: "the agent's answer")
    monkeypatch.setattr(voice, "optional_module", lambda name: None)
    with cfg.override({"identity": {"command_attributes": on}}):
        reply = voice.handle("Hey Bob, " + text, None, {"speaker": "sam"})
    assert (reply == "the agent's answer") is not on  # off by default: the agent answers, as before step 6
    alice = person(tmp_path, "Alice")
    assert ((alice.role, alice.organization) if alice else None) == want


# --------------------------------------------------------------- dashboard ----
@pytest.fixture
def dash(tmp_path, monkeypatch):
    """The dashboard's people reads and writes on a temp state dir."""
    monkeypatch.setattr(people_list, "PEOPLE_ADDED", tmp_path / "people_added.json")
    monkeypatch.setattr(people_list, "CONTACTS_FILE", tmp_path / "contacts.json")
    monkeypatch.setattr(context, "STATE", tmp_path)
    return tmp_path


def roster():
    return {p["name"]: p for p in people_list.people_roster()}


def test_adding_a_person_typed_with_a_role(dash):
    tag = people_writes._typed_tag("Alice, CEO of Acme")
    assert tag is not None
    people_corrections.add_person("Alice", **tag.attributes())
    people_corrections.add_person("Dana Cole")
    people = roster()
    assert (people["Alice"]["role"], people["Alice"]["organization"]) == ("CEO", "Acme")
    assert "role" not in people["Dana Cole"]
    assert json.loads((dash / "people_added.json").read_text()) == ["Alice", "Dana Cole"]


def test_the_roster_opens_no_log_that_is_not_there(dash):
    (dash / "people_added.json").write_text(json.dumps(["Dana"]))
    assert "role" not in roster()["Dana"]
    assert not (dash / "events.sqlite3").exists()


def test_describing_a_person_sets_and_clears_role_and_organization(dash):
    people_corrections.add_person("Dana")
    code, body = people_writes.describe_person("dana", role="CTO", organization="Initech", control=True)
    assert (code, body["name"], body["role"], body["organization"]) == (200, "Dana", "CTO", "Initech")
    code, body = people_writes.describe_person("Dana", role="", control=True)
    assert (code, body.get("role"), body["organization"]) == (200, None, "Initech")
    assert people_writes.describe_person("Nobody", role="CEO", control=True)[0] == 404
    assert people_writes.describe_person("Dana", control=True)[0] == 400
    assert people_writes.describe_person("Dana", role="x" * 81, control=True)[0] == 400
    assert people_writes.describe_person("Dana", role=7, control=True)[0] == 400
    summary = people_cards.person_summary("Dana")
    assert summary is not None
    assert (summary.get("role"), summary["organization"]) == (None, "Initech")


@pytest.mark.parametrize(
    ("host", "code", "role"),
    [pytest.param("127.0.0.1", 200, "CTO", id="this-machine"), pytest.param("pack.example", 403, None, id="remote")],
)
def test_the_describe_route_writes_only_for_this_machine(dash, serve_dashboard, monkeypatch, host, code, role):
    people_corrections.add_person("Dana")
    c = http.client.HTTPConnection("127.0.0.1", urllib.parse.urlsplit(serve_dashboard()).port, timeout=5)
    body = json.dumps({"name": "Dana", "role": "CTO"})
    c.request("POST", "/people/describe", body, {"Content-Type": "application/json", "Host": host})
    r = c.getresponse()
    got = json.loads(r.read())
    c.close()
    assert (r.status, got["ok"]) == (code, code == 200)
    assert person(dash, "Dana").role == role


def test_a_failing_log_is_reported_when_describing(dash):
    (dash / "people_added.json").write_text(json.dumps(["Dana"]))
    with patch.object(registry, "at", autospec=True, side_effect=OSError("disk full")):
        code, body = people_writes.describe_person("Dana", role="CTO", control=True)
    assert code == 500
    assert "disk full" in body["error"]


def test_claiming_with_a_role_records_it_only_once_the_claim_stands(dash):
    write_voices(dash, {2: None})
    fake().say({"ts_start": 10.0, "text": "Hi.", "speaker_id": 2})  # heard, so on the voice list
    with patch.object(known_voices, "_speaker_cmds", autospec=True, return_value=(True, "")):
        assert people_writes.claim_identity("voice", 9, "Erin, CEO of Acme", control=True)[0] == 404
        assert person(dash, "Erin") is None
        code, body = people_writes.claim_identity("voice", 2, "Erin, CEO of Acme", control=True)
    assert (code, body["name"]) == (200, "Erin")
    p = person(dash, "Erin")
    assert (p.role, p.organization) == ("CEO", "Acme")
