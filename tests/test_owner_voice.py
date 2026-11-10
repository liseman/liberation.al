"""The owner check (assistant/voice.py owner_voice): only the owner's own word on a voice makes it the owner's."""

import contextlib
import http.client
import io
import json
import sys
import types
import urllib.parse

import numpy as np
import pytest
import soundfile as sf

from liberational import config as cfg
from liberational import segment_log
from liberational.assistant import voice
from liberational.dashboard import (
    context,
    known_voices,
    people_list,
    people_writes,
    roster_read,
    server,
    speaker_lines,
)
from liberational.domain import kernel
from liberational.domain.identity import MODEL_BASES, Basis, IdentityAssertion
from liberational.domain.kernel import AssertionId, EventType, PersonId
from liberational.people import registry, voice_edits
from voices import write_voices


@pytest.fixture(autouse=True)
def owner_alice():
    with cfg.override({"owner": {"name": "Alice"}}):
        yield


def unit(*v: float) -> list[float]:
    a = np.asarray(v, np.float32)
    return (a / np.linalg.norm(a)).tolist()


def voices(state, named=None, basis=Basis.OWNER_TAG):
    """Voice 1 (Alice's by `basis`), voice 2 (Dana's, an owner tag), and voice 3 (nobody's): voiceprints 1 to 3."""
    write_voices(state, {1: None, 2: "Dana", 3: None})
    producer = registry.voice_producer() if basis in MODEL_BASES else None
    registry.at(state).claim(registry.voiceprint(1), named or "Alice", basis, "test", producer=producer)


def test_an_owner_tag_on_the_voice_makes_it_the_owners(tmp_path):
    voices(tmp_path)
    assert voice.owner_voice(1, tmp_path)
    assert not voice.owner_voice(2, tmp_path)
    assert not voice.owner_voice(3, tmp_path)


@pytest.fixture
def enroll(tmp_path, monkeypatch):
    """enroll(name): voice_edits.enroll on a one-second recording, with the voice model and the broker stood in for.
    Returns the voiceprint it claimed for `name`."""
    wav = tmp_path / "ref.wav"
    sf.write(wav, np.zeros(16000, np.float32), 16000)

    def file(embedding, n, state, at, clone=False):
        return segment_log.new_id()

    monkeypatch.setattr(voice_edits, "_file", file)
    return lambda name: voice_edits.enroll(
        tmp_path, name, str(wav), embed=lambda audio: np.eye(segment_log.VOICE_DIM, dtype=np.float32)[0]
    )


def test_enrolling_the_owners_reference_makes_the_owners_voice(tmp_path, enroll):
    write_voices(tmp_path, {2: "Dana"})
    n = enroll("Alice")
    assert n == 3  # a voiceprint of its own, after the two handed out
    assert voice.owner_voice(n, tmp_path)
    assert not voice.owner_voice(2, tmp_path)


def test_recording_the_owners_reference_again_refreshes_the_same_voice(tmp_path, enroll):
    write_voices(tmp_path, {2: "Dana"})
    first = enroll("Alice")
    assert enroll("Alice") == first


@pytest.mark.parametrize(
    "basis",
    [
        Basis.INTRODUCTION,
        Basis.NAME_BADGE,
        Basis.CLUSTER_MERGE,
        Basis.EMBEDDING_MATCH,
        Basis.BACKFILL_MATCH,
        Basis.CORRECTION,
    ],
)
def test_no_other_basis_makes_a_voice_the_owners(tmp_path, basis):
    registry.at(tmp_path).ensure("Alice", "test")  # a model's match names only someone the registry has
    voices(tmp_path, basis=basis)
    assert registry.voice_names(tmp_path, [1])[1] == "Alice"  # the voice is Alice's, so only the basis says no
    assert not voice.owner_voice(1, tmp_path)


def test_the_owner_may_go_by_their_name_or_their_speaker_label(tmp_path):
    voices(tmp_path, named="Alice Cole")  # what "That voice is mine" names it: owner.name
    with cfg.override({"owner": {"name": "Alice Cole", "speaker_label": "alice"}}):
        assert voice.owner_voice(1, tmp_path)


def test_an_owners_name_and_label_naming_two_people_means_no_owner(tmp_path):
    voices(tmp_path, named="Alice Cole")
    registry.at(tmp_path).claim(registry.voiceprint(3), "Alice", Basis.OWNER_TAG, "test")
    with cfg.override({"owner": {"name": "Alice Cole", "speaker_label": "alice"}}):
        assert not voice.owner_voice(1, tmp_path)
        assert not voice.owner_voice(3, tmp_path)


def test_a_retraction_takes_the_owner_away(tmp_path):
    voices(tmp_path)
    registry.at(tmp_path).retract(registry.voiceprint(1), "test")
    assert not voice.owner_voice(1, tmp_path)


def test_naming_the_voice_someone_else_takes_the_owner_away(tmp_path):
    voices(tmp_path)
    registry.at(tmp_path).claim(registry.voiceprint(1), "Erin", Basis.OWNER_TAG, "test")
    assert not voice.owner_voice(1, tmp_path)


@pytest.mark.parametrize("sid", [0, -1, 4], ids=["zero", "negative", "next"])
def test_a_voiceprint_listen_never_handed_out_is_not_the_owners(tmp_path, sid):
    voices(tmp_path)  # voiceprints 1 to 3: the next one listen hands out is 4
    registry.at(tmp_path).claim(registry.voiceprint(sid), "Alice", Basis.OWNER_TAG, "test")
    assert registry.voice_names(tmp_path, [sid])[sid] == "Alice"  # the log names it, so only the number says no
    assert not voice.owner_voice(sid, tmp_path)


def test_no_voiceprint_count_means_no_owner(tmp_path):
    voices(tmp_path)
    (tmp_path / "voiceprints.json").unlink()
    assert not voice.owner_voice(1, tmp_path)


@pytest.mark.parametrize(
    "text", ["not json", '{"next": "4"}', '{"next": true}', "[4]"], ids=["garbage", "str", "bool", "list"]
)
def test_an_unreadable_voiceprint_count_means_no_owner(tmp_path, text):
    voices(tmp_path)
    (tmp_path / "voiceprints.json").write_text(text)
    assert not voice.owner_voice(1, tmp_path)


def test_a_clone_of_the_owners_voice_is_not_the_owner(tmp_path):
    voices(tmp_path)
    (tmp_path / "clones").mkdir()
    (tmp_path / "clones" / "1.json").write_text(json.dumps({"label": "Alice", "source": "recorded"}))
    assert not voice.owner_voice(1, tmp_path)


def test_no_log_means_no_owner_and_reading_never_creates_one(tmp_path):
    write_voices(tmp_path, {1: None})
    assert not voice.owner_voice(1, tmp_path)
    assert not (tmp_path / "events.sqlite3").exists()


def test_an_unreadable_log_means_no_owner(tmp_path):
    write_voices(tmp_path, {1: None})
    (tmp_path / "events.sqlite3").write_bytes(b"not a database at all, just bytes" * 64)
    assert not voice.owner_voice(1, tmp_path)


def test_no_owner_configured_means_nobody_is_the_owner(tmp_path):
    voices(tmp_path)
    with cfg.override({"owner": {"name": ""}}):
        assert not voice.owner_voice(1, tmp_path)


@pytest.mark.parametrize("sid", [None, True, "x", 1.5e300])
def test_a_line_without_a_usable_voice_id_is_not_the_owner(tmp_path, sid):
    voices(tmp_path)
    assert not voice.owner_voice(sid, tmp_path)


def test_a_name_stored_on_the_line_never_makes_the_owner(tmp_path, monkeypatch):
    voices(tmp_path)
    monkeypatch.setattr(voice, "STATE_DIR", tmp_path)
    assert not voice.owner_entry(voice.canonical_entry({"speaker": "alice", "speaker_sim": 0.95}))
    assert not voice.owner_entry(voice.canonical_entry({"speaker": "alice", "speaker_id": 3, "speaker_sim": 0.95}))
    assert not voice.owner_entry(voice.canonical_entry({"speaker_id": 3, "owner_voice": True, "speaker_sim": 0.95}))
    assert voice.owner_entry(voice.canonical_entry({"speaker": "Speaker 1", "speaker_id": 1, "speaker_sim": 0.95}))


def test_the_owners_voice_still_needs_a_good_match(tmp_path, monkeypatch):
    voices(tmp_path)
    monkeypatch.setattr(voice, "STATE_DIR", tmp_path)
    assert not voice.owner_entry(voice.canonical_entry({"speaker_id": 1, "speaker_sim": voice.OWNER_MIN_SIM - 0.01}))
    assert voice.owner_entry(voice.canonical_entry({"speaker_id": 1, "speaker_sim": voice.OWNER_MIN_SIM}))


# ---- holes an adversarial review found, each closed: none of these makes a voice the owner's ----


def guest_named_owner_by_introduction(state):
    """Voice 2 named Dana by an introduction, and Dana then merged into the owner (Alice) on People."""
    write_voices(state, {1: None, 2: None, 3: None})
    reg = registry.at(state)
    reg.claim(registry.voiceprint(2), "Dana", Basis.INTRODUCTION, "identity")
    reg.merge("Dana", "Alice", "dashboard")
    return reg


def test_merging_a_guest_into_the_owner_never_makes_their_owner_tagged_voice_the_owners(tmp_path):
    write_voices(tmp_path, {1: "Alice", 2: "Dana"})  # Dana's voice by an owner tag, as a dashboard label gives
    registry.at(tmp_path).merge("Dana", "Alice", "dashboard")
    assert registry.voice_names(tmp_path, [2])[2] == "Alice"  # the voice is Alice's person now
    assert not voice.owner_voice(2, tmp_path)  # but that owner tag was about Dana
    assert voice.owner_voice(1, tmp_path)


def test_renaming_a_guest_to_the_owners_name_in_the_registry_alone_is_refused_upstream(tmp_path):
    """The registry renames whom it is told to; the dashboard refuses this one (see the dashboard tests below)."""
    write_voices(tmp_path, {2: "Dana"})
    registry.at(tmp_path).rename("Dana", "Alice", "cli")
    assert voice.owner_voice(2, tmp_path)  # the same person, renamed: their own owner tag


def test_an_owner_tag_on_a_second_registration_of_the_owners_name_still_counts(tmp_path):
    """Two processes registering the owner's name at once leave two ids, the second merged into the first. An owner
    tag written by the second process names the second id, and is still the owner's."""
    write_voices(tmp_path, {1: None})
    reg = registry.at(tmp_path)
    reg.ensure("Alice", "dashboard")
    late = PersonId(kernel.new_id())  # what the other process registered, before it read the first registration
    payload = {"person": late, "display_name": "Alice"}
    tag = IdentityAssertion(
        AssertionId(kernel.new_id()), registry.voiceprint(1), late, Basis.OWNER_TAG, 1.0, kernel.now(), "ambient"
    )
    reg.log.append([registry._event(EventType.PERSON_REGISTERED, registry.PEOPLE, "ambient", payload, late, None)])
    reg.log.append([registry._assertion_event(tag)])
    assert reg.person_of(registry.voiceprint(1)).winner == tag
    assert voice.owner_voice(1, tmp_path)


def test_a_guest_merged_into_the_owner_by_introduction_is_not_the_owners_voice(tmp_path):
    guest_named_owner_by_introduction(tmp_path)
    assert registry.voice_names(tmp_path, [2])[2] == "Alice"  # the voice reads as Alice's person now
    assert not voice.owner_voice(2, tmp_path)


def test_enrolling_the_owner_never_takes_over_a_voice_that_names_them_some_weaker_way(tmp_path, enroll):
    guest_named_owner_by_introduction(tmp_path)
    n = enroll("Alice")
    assert not voice.owner_voice(2, tmp_path)
    assert n != 2 and voice.owner_voice(n, tmp_path)


def test_a_failed_clone_prepare_keeps_the_clone_mark(tmp_path, monkeypatch):
    from liberational.people import transcript_view
    from liberational.speech import clone_say

    write_voices(tmp_path, {7: {"label": "Alice", "centroid": unit(0, 1, 0, 0), "enrolled": True}})
    (tmp_path / "clones").mkdir()
    (tmp_path / "clones" / "7.json").write_text(json.dumps({"label": "Alice", "source": "recorded", "made_at": 1}))
    monkeypatch.setattr(clone_say, "OUT", tmp_path / "clones")
    monkeypatch.setattr(clone_say, "STATE", tmp_path)
    src = tmp_path / "up.wav"
    t = np.arange(24000 * 5) / 24000
    sf.write(src, (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 24000)
    asr = types.ModuleType("liberational.speech.asr")
    asr.ASR = lambda: lambda audio: ""  # ty: ignore[unresolved-attribute]  # a stand-in that hears no words
    monkeypatch.setitem(sys.modules, "liberational.speech.asr", asr)
    with pytest.raises(SystemExit):
        clone_say.prepare(7, src, "Ally")  # a new recording under another name, which fails
    assert 7 in transcript_view.clone_only_ids(tmp_path)
    assert not voice.owner_voice(7, tmp_path)


# ---- the dashboard routes that write an owner tag ----


@pytest.fixture
def dash(tmp_path, monkeypatch):
    """The dashboard on tmp_path, with people/voice_edits.py commands run in process. Returns the commands run."""
    ran = []

    def speaker_cmds(cmds):
        for c in cmds:
            c = [str(x) for x in c]
            ran.append(c)
            with contextlib.redirect_stdout(io.StringIO()):
                voice_edits.main(["--state-dir", str(tmp_path), *c])
        return True, ""

    class Faces:
        def counts(self):
            return {}

        def unknown_groups(self):
            return []

        def rename(self, old, new):
            return 0

    for name, value in {
        "DASH_PASSWORD": "",
    }.items():
        monkeypatch.setattr(server, name, value)
    monkeypatch.setattr(roster_read, "people_view", lambda *a, **k: {})
    monkeypatch.setattr(people_list, "_room_lines", lambda *a, **k: [])
    monkeypatch.setattr(people_list, "_faces_mod", lambda: Faces())
    monkeypatch.setattr(people_list, "_call_lines", lambda *a, **k: [])
    monkeypatch.setattr(speaker_lines, "VOICE_TAGS", tmp_path / "voice_tags.jsonl")
    monkeypatch.setattr(known_voices, "_speaker_cmds", speaker_cmds)
    monkeypatch.setattr(context, "STATE", tmp_path)
    return ran


def post(base, path, body, headers):
    c = http.client.HTTPConnection(urllib.parse.urlsplit(base).netloc, timeout=30)
    c.request("POST", path, json.dumps(body), headers)
    r = c.getresponse()
    r.read()
    c.close()
    return r.status


def test_a_cross_site_voice_label_is_refused(tmp_path, dash, serve):
    ran = dash
    base = serve(server.H)
    write_voices(tmp_path, {1: "Alice", 2: "Dana"})
    headers = {"Content-Type": "text/plain", "Origin": "http://evil.example"}  # a CORS simple request
    assert post(base, "/speakers/label", {"id": 2, "name": "Erin"}, headers) == 403
    headers = {"Content-Type": "application/json", "Origin": "https://evil.example", "Host": "pack.example"}
    assert post(base, "/speakers/label", {"id": 2, "name": "Erin"}, headers) == 403
    assert ran == []


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_the_dashboards_own_page_labels_a_voice_over_http_or_https(tmp_path, dash, serve, scheme):
    """The dashboard is served over HTTPS too, and Live's rename sends that Origin."""
    ran = dash
    base = serve(server.H)
    write_voices(tmp_path, {1: "Alice", 2: "Dana"})
    headers = {"Content-Type": "application/json", "Origin": f"{scheme}://pack.example", "Host": "pack.example"}
    assert post(base, "/speakers/label", {"id": 2, "name": "Erin"}, headers) == 200
    assert ran == [["label", "2", "Erin"]]


@pytest.mark.parametrize("control", [True, False])
def test_a_voice_is_never_labeled_with_the_owners_name(tmp_path, dash, control):
    ran = dash
    write_voices(tmp_path, {1: "Alice", 2: "Dana", 5: None})
    assert speaker_lines.label_speaker(5, "alice", control)[0] == 403
    assert speaker_lines.label_speaker(2, "ALICE", control)[0] == 403
    assert (ran, voice.owner_voice(5, tmp_path), voice.owner_voice(2, tmp_path)) == ([], False, False)


def test_the_owners_voice_is_renamed_only_from_this_machine(tmp_path, dash):
    ran = dash
    write_voices(tmp_path, {1: "Alice", 2: "Dana"})
    assert speaker_lines.label_speaker(1, "Erin", control=False)[0] == 403
    assert speaker_lines.label_speaker("1", "Erin", control=False)[0] == 403  # the id as a request sends it
    assert voice.owner_voice(1, tmp_path)
    assert not any(registry.owner_voice(tmp_path, v) for v in ("1", " 1", "+1", "\u0661", True))  # only a real int
    assert speaker_lines.label_speaker(2, "Erin", control=False)[0] == 200  # anyone else's voice is fine
    assert ran == [["label", "2", "Erin"]]


@pytest.mark.parametrize("owner_registered", [False, True], ids=["no-owner-yet", "owner-without-a-voice"])
def test_nobody_is_renamed_or_merged_into_the_owner(tmp_path, dash, owner_registered):
    write_voices(tmp_path, {2: "Dana"})
    if owner_registered:
        registry.at(tmp_path).ensure("Alice", "dashboard", added=True)
    assert people_writes.rename_person("Dana", "alice", control=True)[0] == 403
    assert registry.voice_names(tmp_path, [2])[2] == "Dana"
    assert not voice.owner_voice(2, tmp_path)


def test_a_casefold_twin_of_the_owners_name_is_the_owners_name(tmp_path, dash):
    ran = dash
    with cfg.override({"owner": {"name": "Groß"}}):
        write_voices(tmp_path, {1: "Groß", 2: None})
        assert people_writes.claim_identity("voice", 2, "GROSS", control=False)[0] == 403
        assert speaker_lines.label_speaker(2, "gross", control=False)[0] == 403
        assert (ran, voice.owner_voice(2, tmp_path)) == ([], False)


def test_lines_are_tagged_as_the_owner_only_from_this_machine(tmp_path, dash):
    write_voices(tmp_path, {1: "Alice"})
    assert speaker_lines.request_voice_tags([1000.0], "Alice", control=False)[0] == 403
    assert not (tmp_path / "voice_tags.jsonl").exists()
    assert speaker_lines.request_voice_tags([1000.0], "Alice", control=True)[0] == 200
    assert speaker_lines.request_voice_tags([1000.0], "Dana", control=False)[0] == 200


def test_the_owners_name_in_settings_never_takes_someone_elses(tmp_path, dash):
    write_voices(tmp_path, {2: "Dana"})  # the owner (Alice) is not on People
    assert (
        people_writes.rename_owner("Alice", "Dana")[0] == 409
    )  # so owner.name stays Alice, and Dana's voice is Dana's


@pytest.mark.parametrize("owner_voice_too", [False, True], ids=["no-owner-voice-yet", "with-an-owner-voice"])
def test_that_voice_is_mine_makes_a_voice_the_owners_when_its_name_only_looks_like_theirs(
    tmp_path, dash, owner_voice_too
):
    """A voice that names the owner through a person merge reads "Alice", so "That voice is mine" said
    it was already theirs and wrote nothing, while the owner check (rightly) refused it."""
    write_voices(tmp_path, {1: "Alice" if owner_voice_too else None, 2: "Dana"})
    registry.at(tmp_path).merge("Dana", "Alice", "dashboard")
    assert not voice.owner_voice(2, tmp_path)
    code, body = people_writes.confirm_owner_voice(2)
    assert code == 200 and "already" not in body["note"]
    assert voice.owner_voice(2, tmp_path)
    assert voice.owner_voice(1, tmp_path) == owner_voice_too
