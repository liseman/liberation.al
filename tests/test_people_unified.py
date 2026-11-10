"""People as one record: GET /people joins voices, faces, phone, added names, and lines; rename, merge, and claim
reach all of them. The voice edits (people/voice_edits.py) run in process on a temp state dir, so tests check the
voices they leave."""

import base64
import http.client
import io
import itertools
import json
import os
import pathlib
import subprocess
import threading
import time
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from liberational import config as cfg
from liberational import segment_log, vision_log
from liberational.dashboard import (
    call_log,
    context,
    inbox_state,
    known_voices,
    people_list,
    people_reads,
    people_writes,
    photo_files,
    server,
    speaker_lines,
    voice_clones,
)
from liberational.domain.identity import Basis
from liberational.people import registry, voice_edits
from liberational.see import face_boxes
from liberational.speech import clone_say
from liberational.vision import faces as face_db
from speech_fake import fake
from voices import write_voices

AUDIO = {"stream": "mic.audio", "start_ns": 0, "end_ns": 1000000000}
PEOPLE = ["Alice", "Dana Cole", "Bob", "Eve"]  # the owner first, then by lines, then by name


class FakeFaces:
    def __init__(self):
        self.rows = {"Alice": 2, "Bob": 1}
        self.groups = [{"id": 5, "count": 4}, {"id": 6, "count": 1}]
        self.renamed, self.named = [], []

    def counts(self):
        return dict(self.rows)

    def unknown_groups(self):
        return list(self.groups)

    def rename(self, old, new):
        self.renamed.append((old, new))
        return 0

    def name_cluster(self, cid, name, enrolled=None):
        if cid not in {g["id"] for g in self.groups}:
            raise ValueError(f"no face group {cid}")
        self.named.append((cid, name))
        if enrolled is not None:
            enrolled.update(vec=np.eye(1, 512, cid, dtype=np.float32)[0].tolist(), count=4, dropped=None)
        return name


CLIP = np.eye(1, segment_log.VOICE_DIM, 1, dtype=np.float32)[0]  # what an uploaded recording sounds like


def in_process_voice_edits(monkeypatch):
    """Puts people/voice_edits.py in place of the dashboard's child process, run on context.STATE with the embedding
    store and the voice model stood in for (an enroll files CLIP). Returns the commands it was given."""
    ran = []

    def no_store(*args, **kwargs):
        pass

    enroll = voice_edits.enroll
    monkeypatch.setattr(voice_edits, "_file", no_store)
    monkeypatch.setattr(voice_edits, "load_wav16", lambda path: np.zeros(16000, np.float32))
    monkeypatch.setattr(voice_edits, "enroll", lambda *a, **k: enroll(*a, **k, embed=lambda audio: CLIP))

    def speaker_cmds(cmds):
        outs = []
        for c in cmds:
            ran.append([*map(str, c)])
            with redirect_stdout(io.StringIO()) as out:
                try:
                    voice_edits.main(["--state-dir", str(context.STATE), *ran[-1]])
                except SystemExit as e:
                    return False, str(e)
            outs.append(out.getvalue().strip())
        return True, "\n".join(outs)

    monkeypatch.setattr(known_voices, "_speaker_cmds", speaker_cmds)
    return ran


@pytest.fixture
def ran(monkeypatch):
    """The voice edits, run in process on context.STATE. Returns the commands they were given."""
    return in_process_voice_edits(monkeypatch)


@pytest.fixture
def faces(monkeypatch):
    faces = FakeFaces()
    monkeypatch.setattr(people_list, "_faces_mod", lambda: faces)
    return faces


@pytest.fixture
def pack(tmp_path, monkeypatch):
    """pack.toml with Alice as the owner and an assistant called Pack (a Settings save writes here)."""
    pack = tmp_path / "pack.toml"
    pack.write_text(
        pathlib.Path(os.environ["PACK_CONFIG"]).read_text() + '[owner]\nname = "Alice"\n[assistant]\nname = "Pack"\n'
    )
    monkeypatch.setenv("PACK_CONFIG", str(pack))
    return pack


@pytest.fixture
def state(tmp_path, monkeypatch, pack, faces, ran):
    """The dashboard pointed at an empty state dir of its own."""
    state = tmp_path / "state"
    state.mkdir()
    files = {}
    for name, path in files.items():
        monkeypatch.setattr(server, name, state / path)
    monkeypatch.setattr(people_list, "PEOPLE_ADDED", state / "people_added.json")
    monkeypatch.setattr(inbox_state, "INBOX_SKIPS", state / "inbox_skips.json")
    monkeypatch.setattr(people_list, "CONTACTS_FILE", state / "contacts.json")
    monkeypatch.setattr(voice_clones, "_clones_dir", state / "clones")
    monkeypatch.setattr(call_log, "CALLS_DIR", state / "calls")
    monkeypatch.setattr(context, "STATE", state)
    monkeypatch.setattr(people_list, "_face_last_seen_at", lambda: {"alice": (300.0, "cam0")})
    with cfg.override({"paths": {"state": str(state), "faces": str(tmp_path / "faces")}}):
        yield state


def write_lines(state, *lines):
    for e in lines:
        fake().say(e)


@pytest.fixture
def people(state):
    """Alice (the owner) with voice 1 learned from a recording, Dana Cole with voice 3 and a number, Eve added by hand,
    Bob known by face, and unnamed voice 2 with a line to play."""
    write_voices(state, {1: "Alice"}, source=voice_edits.ENROLL_SOURCE)
    write_voices(state, {2: None, 3: "Dana Cole"})
    (state / "contacts.json").write_text(json.dumps({"dana cole": "+14155550123"}))
    (state / "people_added.json").write_text(json.dumps(["Eve"]))
    write_lines(
        state,
        {"ts_start": 60.0, "text": "Hi.", "speaker_id": 3, "audio": AUDIO},
        {"ts_start": 61.0, "speaker_id": 2, "audio": AUDIO},
    )
    return state


def label(sid):
    """What the dashboard calls voice `sid` now (GET /speakers)."""
    return {s["id"]: s["label"] for s in speaker_lines.speakers_public()["speakers"]}[sid]


def add_clone(state, sid, name):
    """A voice made from one uploaded recording: clones/<id>.json with source "recorded"."""
    write_voices(state, {sid: None})
    (state / "clones").mkdir(exist_ok=True)
    (state / "clones" / f"{sid}.json").write_text(json.dumps({"label": name, "source": "recorded"}))
    registry.at(state).claim(registry.voiceprint(sid), name, Basis.OWNER_TAG, "test", replace=True)


def test_roster_joins_voices_faces_phone_added_names_and_lines(people):
    view = people_reads.people_view()
    assert [p["name"] for p in view["people"]] == PEOPLE
    alice, dana, bob, _ = view["people"]
    assert (alice["voices"], alice["faces"], alice["owner"], alice["enrolled"], alice["last_seen"]) == (
        [1],
        2,
        True,
        True,
        300.0,
    )
    assert (bob["faces"], bob["voices"]) == (1, [])  # known by face only
    assert (dana["voices"], dana["number"], dana["lines"]) == ([3], "+14155550123", 1)
    assert [v["id"] for v in view["unknown"]["voices"]] == [2]
    assert view["unknown"]["faces"] == [{"id": 5, "count": 4, "label": "Face 5", "photo": None}]  # once is noise


def test_unnamed_voices_with_no_audio_left_are_not_listed(people):
    # 2 has a line with audio; 7 only a line without audio.
    write_voices(people, {7: None})
    write_lines(people, {"ts_start": 3, "speaker_id": 7})
    assert [v["id"] for v in people_reads.people_view()["unknown"]["voices"]] == [2]


def test_speakers_counts_each_voices_lines_and_names_it_by_its_person_newest_first(state):
    # GET /speakers keeps the shape the browser reads (frontend/src/dashboard/live/reduce.ts). A voice the event log
    # names with no line of its own is listed, never heard; nothing is ever merged.
    write_voices(state, {1: "Alice", 2: None, 3: "Dana Cole"})
    write_lines(
        state,
        {"ts_start": 50.0, "speaker_id": 3},
        {"ts_start": 70.0, "speaker_id": 3},
        {"ts_start": 61.0, "speaker_id": 2},
        {"ts_start": 65.0, "speaker_id": None},
    )
    out = speaker_lines.speakers_public()
    assert [(s["id"], s["label"], s["n"], s["first_seen"], s["last_seen"]) for s in out["speakers"]] == [
        (3, "Dana Cole", 2, 50.0, 70.0),
        (2, "Speaker 2", 1, 61.0, 61.0),
        (1, "Alice", 0, None, None),
    ]
    assert out["merged"] == {}


def test_speakers_marks_only_a_voice_learned_from_a_recording_as_enrolled(state):
    write_voices(state, {1: "Alice"}, source=voice_edits.ENROLL_SOURCE)
    write_voices(state, {3: "Dana Cole"})
    assert {s["id"]: s["enrolled"] for s in speaker_lines.speakers_public()["speakers"]} == {1: True, 3: False}


def test_rename_relabels_voices_faces_phone_without_a_restart(people, faces, ran):
    code, body = people_writes.rename_person("Dana Cole", "Dana Cole-Smith", control=True)
    assert (code, body["merged"]) == (200, False)
    assert ran == []  # her voice is named by her person: no voice command
    assert label(3) == "Dana Cole-Smith"
    assert faces.renamed == [("Dana Cole", "Dana Cole-Smith")]
    assert json.loads((people / "contacts.json").read_text()) == {"Dana Cole-Smith": "+14155550123"}


def test_renaming_into_someone_gives_them_both_voices(people, ran):
    # The people merge in the event log carries her voice along: no voice command.
    registry.at(people).claim(registry.voiceprint(2), "Erin", Basis.OWNER_TAG, "test")
    code, body = people_writes.rename_person("Dana Cole", "erin", control=True)
    assert (code, body["name"], body["merged"]) == (200, "Erin", True)
    assert (ran, label(2), label(3)) == ([], "Erin", "Erin")


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        # Their voices would answer as the owner: "That voice is mine" takes one voice at a time instead.
        pytest.param("Dana Cole", "alice", 403, id="into-the-owner"),
        # Critic 1: renaming the owner (or Same person as from their profile) relabeled their voice, so the owner
        # check stopped matching it, and the renamed owner could then be deleted. The owner is renamed in Settings.
        pytest.param("Alice", "Alicia", 403, id="the-owner"),
        pytest.param("Alice", "Dana Cole", 403, id="the-owner-into-someone"),
        pytest.param("Nobody", "Someone", 404, id="nobody"),
    ],
)
def test_a_refused_rename_changes_nothing(people, faces, ran, old, new, code):
    assert people_writes.rename_person(old, new, control=True)[0] == code
    assert (ran, faces.renamed, label(1)) == ([], [], "Alice")


def test_the_owner_can_change_only_the_capitals_of_their_name_here(people):
    code, body = people_writes.rename_person("alice", "ALICE", control=True)
    assert (code, body["name"]) == (200, "ALICE")


@pytest.mark.parametrize(
    "name",
    ["Speaker 2", "speaker 2", "Speaker 9", "Face 5", "face 5", "unknown", "Multiple speakers", "Caller x", "Pack"]
    + ["pack", "1 bad"],
)
def test_a_reserved_or_malformed_label_is_never_a_persons_name(people, ran, name):
    # Finding 10: Live's rename took "unknown" and the assistant's name, so the voice left People and its lines read as
    # nobody's, and "Speaker 9" gave two voices one automatic label.
    assert people_writes.rename_person("Dana Cole", name, control=True)[0] == 400
    assert people_writes.claim_identity("voice", 2, name, control=True)[0] == 400
    assert people_writes.claim_identity("face", 5, name, control=True)[0] == 400
    assert speaker_lines.request_voice_tags([60.0], name)[0] == 400
    assert speaker_lines.label_speaker(3, name)[0] == 400
    assert ran == []  # no voice was named
    assert [p["name"] for p in people_list.people_roster()] == PEOPLE


def test_the_live_voice_rename_takes_its_own_automatic_label_and_tidies_the_name(people):
    assert speaker_lines.label_speaker(3, "Speaker 3")[0] == 200
    assert label(3) == "Speaker 3"
    assert speaker_lines.label_speaker(3, " Dana  Cole-Smith ")[0] == 200
    assert label(3) == "Dana Cole-Smith"


def test_claiming_a_voice_for_someone_who_has_one_names_it_for_them_from_any_machine(people, ran):
    # Naming replaces merging: a person may have several voices.
    assert people_writes.claim_identity("voice", 2, "Dana Cole", control=False)[0] == 200
    assert ran == [["label", "2", "Dana Cole"]]
    assert (label(2), label(3)) == ("Dana Cole", "Dana Cole")


def test_claims_name_or_merge_unknown_voices_and_faces(people, faces):
    assert people_writes.claim_identity("voice", 2, "Zoe", control=False)[0] == 200
    assert label(2) == "Zoe"
    assert people_writes.claim_identity("face", 5, "dana cole", control=False)[0] == 200
    assert faces.named == [(5, "Dana Cole")]  # joins the existing person's spelling
    assert people_writes.claim_identity("face", 9, "Zoe", control=False)[0] == 404
    assert people_writes.claim_identity("hand", 2, "Zoe", control=True)[0] == 400


def test_claiming_a_voice_as_the_owner_names_it_beside_theirs(people):
    code, body = people_writes.claim_identity("voice", 2, "Alice", control=True)
    assert (code, body["ok"]) == (200, True)
    assert (label(1), label(2)) == ("Alice", "Alice")


def test_a_voice_that_names_someone_is_never_claimed_for_someone_else(people, ran):
    # A turn of the owner's that Live matched to Dana's voice: typing the owner's name claimed the voice, which merged
    # all of Dana's voice into the owner's, for good. Two people become one through Same person as on People.
    assert people_writes.claim_identity("voice", 3, "alice", control=True)[0] == 409
    assert people_writes.claim_identity("voice", 3, "Zoe", control=True)[0] == 409
    assert people_writes.confirm_owner_voice(3)[0] == 409
    assert ran == []
    assert people_writes.claim_identity("voice", 3, "dana cole", control=True)[0] == 200  # her own name is fine


def test_the_owners_voice_is_never_claimed_for_someone_else(people, ran):
    # Review 4, finding 6: Live's Turn rename sends /people/claim when the typed name exists, so typing Dana Cole on
    # one of the owner's own turns merged the owner's voice into Dana's, which can't be undone.
    assert people_writes.claim_identity("voice", 1, "Dana Cole", control=True)[0] == 403
    assert people_writes.claim_identity("voice", 1, "Zoe", control=True)[0] == 403
    assert ran == []
    assert people_writes.claim_identity("voice", 1, "alice", control=True)[0] == 200  # the owner's own voice is theirs
    add_clone(people, 1, "Alice")  # a clone of the owner's voice is not the owner's voice
    assert people_writes.claim_identity("voice", 1, "Zoe", control=True)[0] == 200
    assert label(1) == "Zoe"


@pytest.mark.parametrize(
    ("take", "sid", "cmds"),
    [
        pytest.param(
            lambda: people_writes.claim_identity("voice", 2, "Gus", control=False),
            2,
            [["label", "2", "Gus"]],
            id="claim",
        ),
        pytest.param(lambda: people_writes.rename_person("Dana Cole", "Gus", control=True), 3, [], id="rename"),
    ],
)
def test_taking_a_clones_name_leaves_the_clone_a_clone(people, ran, take, sid, cmds):
    # Finding 10: the clone's voice counted as the person's, so the real voice was merged into it.
    add_clone(people, 7, "Gus")
    code, _ = take()
    assert (code, ran) == (200, cmds)
    assert (label(sid), voice_clones._clone_only_ids()) == ("Gus", {7})


def test_a_clone_sharing_a_name_never_hides_the_person(people, faces):
    # Review 2, finding 10: Bob, known by face, left People once a clone was recorded as Bob.
    add_clone(people, 7, "Bob")
    bob = next(p for p in people_reads.people_view()["people"] if p["name"] == "Bob")
    assert (bob["faces"], bob["voices"]) == (1, [])  # the clone is not his voice
    assert people_writes.rename_person("Bob", "Robert", control=True)[0] == 200
    assert faces.renamed == [("Bob", "Robert")]


def test_a_name_only_a_clone_carries_stays_out_of_people(people):
    # A clone is listed under Voice clones, and a room line once filed under it does not make it a person.
    add_clone(people, 7, "Gus")
    write_lines(people, {"ts_start": 70.0, "text": "Hi.", "speaker_id": 7})
    assert [p["name"] for p in people_reads.people_view()["people"]] == PEOPLE


def test_claiming_a_voice_as_the_owner_skips_a_clone_of_the_owner(people):
    # Finding 10, the owner's branch of a claim: the only voice with the owner's name is a clone.
    add_clone(people, 1, "Alice")
    code, body = people_writes.claim_identity("voice", 2, "Alice", control=True)
    assert (code, body["ok"]) == (200, True)
    assert (label(2), voice_clones._clone_only_ids()) == ("Alice", {1})


def test_voice_commands_run_in_order_in_a_child_and_stop_at_a_failure(monkeypatch):
    ran = []

    def run(cmd, **kw):
        ran.append(cmd)
        return subprocess.CompletedProcess(cmd, int(cmd[-1] == "bad"), "", "")

    monkeypatch.setattr(known_voices, "subprocess", SimpleNamespace(run=run))
    ok, _ = known_voices._speaker_cmds([["label", 2, "bad"], ["label", 3, "Dana"]])
    voices = [context.PY, "-m", "liberational.people.voice_edits", "--state-dir", str(context.STATE)]
    assert ok is False
    assert ran == [[*voices, "label", "2", "bad"]]


@pytest.fixture
def restarted(monkeypatch):
    restarted = threading.Event()
    monkeypatch.setattr(
        server.settings,
        "restart",  # restart(log, names, dashboard, before): what a save restarts
        lambda log=print, names=(), dashboard=True, before=None: restarted.set(),
    )
    return restarted


@pytest.fixture
def save_owner(people, serve, restarted, monkeypatch):
    """POST /config with a new owner.name, as Settings saves it. Returns (status, body)."""
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    base = serve(server.H)

    def save(name):
        c = http.client.HTTPConnection(base.removeprefix("http://"), timeout=10)
        c.request("POST", "/config", json.dumps({"values": {"owner.name": name}}), {"Content-Type": "application/json"})
        r = c.getresponse()
        out = r.status, json.loads(r.read())
        c.close()
        return out

    return save


def test_a_settings_rename_carries_the_owner_along(people, pack, faces, ran, save_owner, restarted):
    # Review 4, finding 7: saving owner.name in Settings relabeled nothing. The owner's voice kept the old name, so the
    # owner check stopped matching it, its row became deletable, and the startup enroll made a second owner.
    reg = registry.at(people)
    alice, before = reg.person_id("Alice"), pack.read_text()
    code, body = save_owner("Dana Cole")  # someone else's name: that would merge the owner into Dana
    assert (code, body["field"]) == (409, "owner.name")
    assert (pack.read_text(), ran, restarted.is_set()) == (before, [], False)

    with patch.object(registry.Registry, "rename", autospec=True, side_effect=OSError(28, "No space left")):
        code, body = save_owner("Alicia")  # the log can't be written: nothing changes
    assert (code, body["field"]) == (500, "owner.name")
    assert (pack.read_text(), restarted.is_set()) == (before, False)
    assert (reg.person_id("Alice"), reg.person_id("Alicia")) == (alice, None)

    code, body = save_owner("Alicia")
    assert code == 200, body
    assert ran == []  # no voice command: the owner's voice is named by their person
    assert (label(1), reg.person_id("Alicia")) == ("Alicia", alice)  # so the owner check matches it
    assert faces.renamed == [("Alice", "Alicia")]
    assert restarted.wait(5)


def test_a_settings_rename_that_raises_still_answers_and_keeps_one_owner(people, pack, ran, save_owner, restarted):
    # An exception in the owner rename escaped the save with no reply and no restart. A failure before
    # anything moved puts pack.toml back; one after the voice moved keeps the new name, since putting it back would
    # split the owner again.
    alice, before = registry.at(people).person_id("Alice"), pack.read_text()
    with patch.object(people_list, "people_roster", autospec=True, side_effect=OSError(13, "Permission denied")):
        code, body = save_owner("Alicia")
    assert (code, body["field"]) == (500, "owner.name")
    assert (pack.read_text(), restarted.is_set(), ran) == (before, False, [])

    with patch.object(people_list.contacts.Book, "follow", autospec=True, side_effect=OSError(13, "Permission denied")):
        code, body = save_owner("Alicia")
    assert code == 200, body
    assert (label(1), registry.at(people).person_id("Alicia")) == ("Alicia", alice)
    assert "Alicia" in pack.read_text()
    assert restarted.wait(5)


@pytest.fixture
def dana_recorded(state, monkeypatch):
    """Voice 3 is Dana's, heard in the room. Recording a clone enrolls it in process (the `ran` fixture)."""
    write_voices(state, {3: "Dana"})

    def clone_run(args, timeout):  # clone_say prepare: the reference, marked as a recorded clone
        sid, name = args[1], args[3]
        (state / "clones" / f"{sid}.json").write_text(json.dumps({"label": name, "source": "recorded"}))
        return 0, {"seconds": 12.0, "ref_text": "hello"}, []

    monkeypatch.setattr(voice_clones, "_clone_run", clone_run)
    monkeypatch.setattr(clone_say, "voiced_seconds", lambda *a, **k: 12.0)
    monkeypatch.setattr(clone_say, "STATE", state)
    monkeypatch.setattr(clone_say, "OUT", state / "clones")


def test_a_clone_under_an_existing_name_is_a_voice_of_its_own(state, dana_recorded):
    code, body = voice_clones.clone_record(b"RIFF", "Dana")
    assert code == 200, body
    assert body["id"] != 3
    assert voice_clones._clone_only_ids() == {body["id"]}
    assert (label(3), label(body["id"])) == ("Dana", "Dana")


def test_same_person_as_leaves_a_clone_a_clone(state, dana_recorded, ran):
    # Review 2, findings 13 and 18: merging Dana into Eve folded Dana's clone into Eve's voice too. The people merge
    # in the event log carries every voice of Dana's along, and runs no voice command.
    write_voices(state, {7: "Eve"})
    clone = voice_clones.clone_record(b"RIFF", "Dana")[1]["id"]
    code, body = people_writes.rename_person("Dana", "Eve", control=True)
    assert code == 200, body
    assert [c[:2] for c in ran] == [["enroll-clone", "Dana"]]  # the recording, and nothing since
    assert (label(3), label(7), label(clone)) == ("Eve", "Eve", "Eve")
    assert voice_clones._clone_only_ids() == {clone}


def test_a_renamed_persons_clone_stays_a_clone_when_spoken(state, dana_recorded):
    # Review 2, finding 18: after a rename, Say it moved the clone's mark aside, so it was matched in the room.
    clone = voice_clones.clone_record(b"RIFF", "Dana")[1]["id"]
    ref = state / "clones" / f"{clone}.wav"
    ref.write_bytes(b"RIFF")
    (state / "clones" / f"{clone}.txt").write_text("hello")
    assert people_writes.rename_person("Dana", "Danielle", control=True)[0] == 200
    assert (label(3), label(clone)) == ("Danielle", "Danielle")
    assert clone_say.ref_for(clone, make=False) == ref  # still listed under Voice clones
    with patch.object(clone_say, "build", side_effect=AssertionError("a clone is never rebuilt from the room")):
        assert clone_say.ref_for(clone) == ref
    meta = json.loads((state / "clones" / f"{clone}.json").read_text())
    assert (meta["label"], meta["source"]) == ("Danielle", "recorded")
    assert voice_clones._clone_only_ids() == {clone}


def test_removing_a_recorded_clone_keeps_it_out_of_the_room(state, dana_recorded):
    # Review 2, finding 19: Remove clone deleted clones/<id>.json, the only mark that keeps a clone unmatched.
    clone = voice_clones.clone_record(b"RIFF", "Gus")[1]["id"]
    for ext in ("wav", "txt"):
        (state / "clones" / f"{clone}.{ext}").write_bytes(b"x")
    code, body = voice_clones.clone_remove(clone)
    assert code == 200, body
    assert sorted(body["removed"]) == [f"{clone}.txt", f"{clone}.wav"]
    assert voice_clones._clone_only_ids() == {clone}
    assert clone_say.ref_for(clone, make=False) is None  # gone from Voice clones until recorded again
    assert voice_clones.clone_record(b"RIFF", "Gus")[1]["id"] == clone  # recording it again reuses the voice


def test_speaking_a_clone_whose_clip_is_gone_keeps_its_mark(state, dana_recorded):
    # Review 2, finding 19: with its clip missing, ref_for moved a clone's note aside, so it lost its mark.
    clone = voice_clones.clone_record(b"RIFF", "Gus")[1]["id"]
    with (
        patch.object(clone_say, "build", side_effect=SystemExit("only 0.0s of audio verified")),
        pytest.raises(SystemExit, match="record it again"),
    ):
        clone_say.ref_for(clone)
    assert voice_clones._clone_only_ids() == {clone}


@pytest.fixture
def face_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(face_db, "DB", tmp_path / "db.npz")
    monkeypatch.setattr(face_db, "UNK_DB", tmp_path / "unknown.npz")


def test_face_counts_rename_and_unknown_groups(face_rows):
    v = np.eye(512, dtype=np.float32)
    face_db.save_db(["Alice", "alice", "Bob"], v[:3])
    assert face_db.counts() == {"Alice": 1, "alice": 1, "Bob": 1}
    assert face_db.rename("ALICE", "Alicia") == 2
    assert face_db.counts() == {"Alicia": 2, "Bob": 1}
    face_db._unk_save([3, 4], v[3:5], [1, 9], 5)
    assert [(g["id"], g["count"]) for g in face_db.unknown_groups()] == [(4, 9), (3, 1)]


def test_a_face_rename_keeps_the_newest_rows_per_name(face_rows):
    n = face_db.MAX_PER_NAME
    face_db.save_db(["Bob"] * n + ["Rob"], np.eye(512, dtype=np.float32)[: n + 1])
    face_db.rename("Rob", "Bob")
    names, embs = face_db.load_db()
    assert list(names) == ["Bob"] * n
    assert int(np.argmax(embs[-1])) == n  # Rob's row (the newest) survived


SCENE_TS = 1792816000


@pytest.fixture
def see(state, monkeypatch, vision):
    """A camera frame on the vision log, and a clock that starts at its time and steps 1 ms per read. Returns a
    function that puts faces in view in that frame: see((label, box as shares of the frame), ...), a "Face N" label
    being an unnamed face in that face group."""
    vision.frames[("cam0", float(SCENE_TS))] = np.full((500, 1000, 3), (150, 120, 90), np.uint8)
    clock = itertools.count(SCENE_TS * 1000)
    monkeypatch.setattr(time, "time", lambda: next(clock) / 1000)
    monkeypatch.setitem(people_reads._scene_photos, "t", 0.0)

    def see(*faces):
        # Only these faces now, as see-actions matched them in that frame.
        vision.records.pop(segment_log.FACE_MATCHES, None)
        for label, box in faces:
            group = label.startswith("Face ")
            vision.face(float(SCENE_TS), "cam0", None if group else label, bbox=box, cluster=label if group else None)

    return see


BOX = [0.1, 0.1, 0.2, 0.3]  # 100 by 100 px of the 1000 by 500 frame


def test_tagging_saves_a_square_crop_and_keeps_the_newest_eight(see):
    see(("Face 7", BOX))
    for _ in range(people_writes.PHOTOS_PER_PERSON + 2):
        assert people_writes.save_person_photo("Dana Cole", "Face 7", 7)
    photos = photo_files._photos("Dana Cole")
    assert len(photos) == people_writes.PHOTOS_PER_PERSON
    assert photos[0].parent.name == "dana-cole"
    assert Image.open(photos[0]).size == (192, 192)
    assert Image.open(photos[0]).getpixel((96, 96)) == pytest.approx((90, 120, 150), abs=2)  # BGR frame, RGB photo


def test_tiny_faces_and_gone_frames_are_not_photos(see, tmp_path):
    assert not people_reads._crop(
        "cam0", float(SCENE_TS), [0.01, 0.01, 0.03, 0.05], tmp_path / "x.jpg"
    )  # 20 px: on a wall
    assert not people_reads._crop("cam1", float(SCENE_TS), BOX, tmp_path / "x.jpg")  # no frame kept
    assert not (tmp_path / "x.jpg").exists()


def test_a_stale_scene_never_crops_a_photo_under_a_name_nobody_has_now(state, see):
    # Finding 19: faces in view keep the name they were matched to, so right after People: Delete (or a rename) a new
    # photo was cropped under the old name, which someone new who takes the name would show as theirs.
    reg = registry.at(state)
    for name in ("Dana Cole", "Dee", "Eve"):
        reg.ensure(name, "test")
    reg.forget("Dana Cole", "test")
    reg.rename("Dee", "Dana", "test")
    see(*[(n, BOX) for n in ("Dana Cole", "Dee", "Eve", "Zed")])
    people_reads._photos_from_scene()
    got = {n: len(photo_files._photos(n)) for n in ("Dana Cole", "Dee", "Dana", "Eve", "Zed")}
    # A face still matched as Dee is the person renamed Dana (the face worker hasn't read the rename yet), so the photo
    # is Dana's; none goes under the old name, which someone new could take. Zed is a name the registry never had: it
    # gets a photo as before.
    assert got == {"Dana Cole": 0, "Dee": 0, "Dana": 1, "Eve": 1, "Zed": 1}


def test_faces_in_view_get_pictures_served_read_only_and_kept_on_rename(state, see, serve, monkeypatch):
    see(("Face 3", BOX))
    people_reads._photos_from_scene()
    assert photo_files._unknown_photo(3).exists()
    see(("Dana Cole", BOX))
    monkeypatch.setitem(people_reads._scene_photos, "t", 0.0)
    people_reads._photos_from_scene()  # a named person with no photo yet gets one
    assert len(photo_files._photos("Dana Cole")) == 1

    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    host = serve(server.H).removeprefix("http://")

    def get(path):
        c = http.client.HTTPConnection(host, timeout=5)
        c.request("GET", path)
        r = c.getresponse()
        out = r.status, r.getheader("Content-Type"), r.read()
        c.close()
        return out

    (photo,) = photo_files._photos("Dana Cole")
    assert get(f"/people/Dana%20Cole/photos/{photo.name}")[:2] == (200, "image/jpeg")  # the card's photo URL
    assert get("/faces/unknown/3.jpg")[:2] == (200, "image/jpeg")
    assert get("/faces/unknown/4.jpg")[0] == 404
    assert get(f"/people/Nobody/photos/{photo.name}")[0] == 404
    assert get("/faces/unknown/..%2Fx.jpg")[0] == 404

    (state / "people_added.json").write_text(json.dumps(["Dana Cole"]))
    assert people_writes.rename_person("Dana Cole", "Dana", control=True)[0] == 200
    assert (len(photo_files._photos("Dana")), photo_files._photos("Dana Cole")) == (1, [])


@pytest.mark.parametrize("changed", ["missing", "camera", "track", "cluster", "named"])
def test_live_face_claim_rejects_missing_or_changed_source_before_enrolling(people, faces, monkeypatch, changed):
    source = {"camera": "cam0", "track": "run:cam0:7", "at": 100.5, "group": 5}
    row = {"cam": "cam0", "track_id": "run:cam0:7", "cluster": "Face 5", "name": None, "ts_start": 100.5}
    rows = [row]
    if changed == "missing":
        rows = []
    elif changed == "named":
        row["name"] = "Dana Cole"
    else:
        row[{"camera": "cam", "track": "track_id", "cluster": "cluster"}[changed]] = "different"
    monkeypatch.setattr(
        segment_log,
        "at",
        lambda state=None: SimpleNamespace(
            range=lambda stream, *args: rows if stream == segment_log.FACE_MATCHES else []
        ),
    )
    code, _body = people_writes.claim_identity("face", 5, "Dana Cole", True, source)
    assert code == 409
    assert faces.named == []


def test_live_face_claim_uses_selected_source_time_and_group(people, faces, monkeypatch, selected_face_jpeg):
    reads = []

    def near(state, lo, hi):
        reads.append((lo, hi))
        return [
            {"cam": "cam0", "start_ns": 100_500_000_000, "faces": [{"track_id": "run:cam0:7", "cluster": "Face 5"}]}
        ]

    monkeypatch.setattr(face_boxes, "near", near)
    monkeypatch.setattr(vision_log, "at", lambda state: SimpleNamespace(range=lambda *a: []))
    code, _body = people_writes.claim_identity(
        "face",
        5,
        "Dana Cole",
        True,
        {"camera": "cam0", "track": "run:cam0:7", "at": 100.5, "group": 5},
        selected_face_jpeg,
    )
    assert code == 200
    assert reads[0] == (100_499_000_000, 100_501_000_000), "the selected frame's own time"
    assert faces.named == [(5, "Dana Cole")]
    saved = photo_files._photos("Dana Cole")
    assert len(saved) == 1
    photo = people_list.main_photo("Dana Cole")
    assert photo is not None
    photo_url = photo["url"]
    assert photo_url is not None and photo_url.startswith("/people/Dana%20Cole/photos/")
    assert saved[0].name in photo_url
    with Image.open(saved[0]) as photo:
        assert photo.size == (32, 48)
        pixel = photo.getpixel((10, 10))
        assert isinstance(pixel, tuple)
        assert all(abs(a - b) < 4 for a, b in zip(pixel, (200, 60, 10)))


@pytest.mark.parametrize(
    ("stream", "record"),
    [
        pytest.param(
            None,
            {
                "start_ns": 100_500_000_000,
                "cam": "cam0",
                "faces": [{"track_id": "selected-track", "cluster": "Face 5", "name": None}],
            },
            id="live-frame",
        ),
        pytest.param(
            segment_log.FACE_MATCHES,
            {"ts_start": 100.0, "cam": "cam0", "track_id": "selected-track", "cluster": "Face 5", "name": None},
            id="recorded-frame",
        ),
    ],
)
def test_selected_face_claim_creates_new_person_and_enrolls_the_group(
    people, face_rows, monkeypatch, selected_face_jpeg, vision, live_face_boxes, stream, record
):
    monkeypatch.setattr(people_list, "_faces_mod", lambda: face_db)
    face_db.save_db([], np.empty((0, 512), np.float32))
    face_db._unk_save([5], np.eye(1, 512, dtype=np.float32), [4], 6)
    if stream is None:  # the live frame, from the face.boxes ring
        live_face_boxes.append(record)
    else:
        vision.add(stream, **record)
    reg = registry.at(people)
    assert reg.person_id("Taylor Vale") is None
    assert not any(person["name"] == "Taylor Vale" for person in people_list.people_roster())

    code, body = people_writes.claim_identity(
        "face",
        5,
        "Taylor Vale",
        True,
        {"camera": "cam0", "track": "selected-track", "at": 100.5, "group": 5},
        selected_face_jpeg,
    )

    assert code == 200
    pid = reg.person_id("Taylor Vale")
    assert pid is not None
    assert reg.person_of(registry.face_group(5)).person == pid
    assert face_db.counts() == {"Taylor Vale": 1}
    assert face_db.unknown_groups() == []
    person = next(person for person in body["people"] if person["name"] == "Taylor Vale")
    assert person["faces"] == 1
    saved = photo_files._photos("Taylor Vale")
    assert len(saved) == 1
    assert reg.person_of(registry.photo(saved[0].stem)).person == pid
    with Image.open(saved[0]) as photo:
        assert photo.size == (32, 48)


def test_live_face_claim_resolves_prior_recorded_match_when_overlay_has_expired(
    people, faces, monkeypatch, selected_face_jpeg
):
    rows = [
        {"cam": "cam0", "track_id": "selected-track", "cluster": "Face 5", "name": None, "ts_start": 100.0},
        {"cam": "cam0", "track_id": "different-track", "cluster": "Face 6", "name": None, "ts_start": 100.49},
        {"cam": "cam0", "track_id": "selected-track", "cluster": "Face 6", "name": None, "ts_start": 100.7},
    ]
    monkeypatch.setattr(
        segment_log,
        "at",
        lambda state=None: SimpleNamespace(
            range=lambda stream, *args: rows if stream == segment_log.FACE_MATCHES else []
        ),
    )
    code, _ = people_writes.claim_identity(
        "face",
        5,
        "Dana Cole",
        True,
        {"camera": "cam0", "track": "selected-track", "at": 100.5, "group": 5},
        selected_face_jpeg,
    )
    assert code == 200
    assert faces.named == [(5, "Dana Cole")]


@pytest.fixture
def selected_face_jpeg():
    data = io.BytesIO()
    Image.new("RGB", (32, 48), (200, 60, 10)).save(data, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()


def test_selected_face_removed_from_gallery_does_not_create_person_or_photo(
    people, faces, monkeypatch, selected_face_jpeg, live_face_boxes
):
    faces.groups = []
    live_face_boxes.append(
        {"cam": "cam0", "start_ns": 100_500_000_000, "faces": [{"track_id": "selected-track", "cluster": "Face 5"}]}
    )
    code, _ = people_writes.claim_identity(
        "face",
        5,
        "Taylor",
        True,
        {"camera": "cam0", "track": "selected-track", "at": 100.5, "group": 5},
        selected_face_jpeg,
    )
    assert code == 409
    assert faces.named == []
    assert photo_files._photos("Taylor") == []
    assert not any(p["name"] == "Taylor" for p in people_list.people_roster())


def test_selected_photo_registry_failure_preserves_unknown_group_for_retry(
    people, faces, monkeypatch, selected_face_jpeg, live_face_boxes
):
    record = people_writes._record_person_photo
    calls = 0

    def fail_once(name, dest, *, prune=True):
        nonlocal calls
        calls += 1
        if calls == 1:
            dest.unlink()
            raise registry.WriteFailed("photo registry unavailable")
        record(name, dest, prune=prune)

    monkeypatch.setattr(people_writes, "_record_person_photo", fail_once)
    live_face_boxes.append(
        {"cam": "cam0", "start_ns": 100_500_000_000, "faces": [{"track_id": "selected-track", "cluster": "Face 5"}]}
    )
    source = {"camera": "cam0", "track": "selected-track", "at": 100.5, "group": 5}
    with pytest.raises(registry.WriteFailed, match="photo registry unavailable"):
        people_writes.claim_identity("face", 5, "Taylor", True, source, selected_face_jpeg)
    assert faces.named == []
    assert faces.groups == [{"id": 5, "count": 4}, {"id": 6, "count": 1}]
    assert photo_files._photos("Taylor") == []
    code, _ = people_writes.claim_identity("face", 5, "Taylor", True, source, selected_face_jpeg)
    assert code == 200
    assert faces.named == [(5, "Taylor")]
    saved = photo_files._photos("Taylor")
    assert len(saved) == 1
    with Image.open(saved[0]) as image:
        assert image.size == (32, 48)


def test_selected_face_claim_failure_withdraws_only_its_new_photo(people, faces, monkeypatch, selected_face_jpeg):
    folder = photo_files._photo_dir("Dana Cole")
    folder.mkdir(parents=True, exist_ok=True)
    old = {f"{1000000000000 + i}.jpg": f"existing {i}".encode() for i in range(8)}
    for name, data in old.items():
        (folder / name).write_bytes(data)
    old_ref = registry.photo("1000000000000")
    registry.at(context.STATE).claim(old_ref, "Dana Cole", Basis.OWNER_TAG, "dashboard")
    old_person = registry.at(context.STATE).person_of(old_ref).person
    recorded = []
    record = people_writes._record_person_photo

    def capture(name, dest, *, prune=True):
        record(name, dest, prune=prune)
        recorded.append(dest.stem)

    def fail_claim(*args):
        raise ValueError("face claim failed")

    monkeypatch.setattr(people_writes, "_record_person_photo", capture)
    monkeypatch.setattr(faces, "name_cluster", fail_claim)
    code, _ = people_writes._claim("face", 5, "Dana Cole", True, people_writes._selected_face_image(selected_face_jpeg))
    assert code == 404
    assert {p.name: p.read_bytes() for p in folder.glob("*.jpg")} == old
    assert len(recorded) == 1
    assert registry.at(context.STATE).person_of(registry.photo(recorded[0])).person is None
    assert registry.at(context.STATE).person_of(old_ref).person == old_person
    assert faces.named == []


def test_this_is_me_requires_current_voice_confirmation_before_owner_authority(people, faces):
    reg = registry.at(people)
    reg.contact("Alice", "test", phone="+15550001")
    owner = reg.person_id("Alice")
    assert owner is not None
    source = reg.person_id("Dana Cole")
    before = reg.mark()
    code, body = people_writes.merge_into_owner("Dana Cole", None, control=True)
    assert code == 409
    assert body["voices"] == [3]
    assert reg.mark() == before
    assert label(3) == "Dana Cole"
    assert not registry.owner_voice(people, 3)
    write_voices(people, {4: "Dana Cole"})
    assert people_writes.merge_into_owner("Dana Cole", [3], control=True)[0] == 409
    assert label(3) == "Dana Cole"
    code, body = people_writes.merge_into_owner("Dana Cole", [3, 4], control=True)
    assert (code, body["name"], body["merged"]) == (200, "Alice", True)
    assert reg.person_id("Alice") == owner
    person = reg.person(owner)
    assert person is not None
    assert person.phone == "+15550001"
    fresh = registry.Registry(reg.log)
    fresh.refresh()
    assert fresh.resolve(source) == owner
    winner = fresh.person_of(registry.voiceprint(3)).winner
    assert winner is not None
    assert winner.person == owner
    assert registry.owner_voice(people, 3)
    assert registry.owner_voice(people, 4)
    assert faces.renamed == [("Dana Cole", "Alice")]


def test_this_is_me_without_voices_needs_no_extra_confirmation(people, faces):
    registry.at(people).ensure("Bob", "test")
    code, body = people_writes.merge_into_owner("Bob", None, control=True)
    assert (code, body["name"]) == (200, "Alice")
    assert faces.renamed == [("Bob", "Alice")]


@pytest.mark.parametrize("confirmation", [None, [3]])
def test_this_is_me_requires_local_control(people, faces, confirmation):
    assert people_writes.merge_into_owner("Dana Cole", confirmation, control=False)[0] == 403
    assert label(3) == "Dana Cole"
    assert faces.renamed == []


def test_this_is_me_failed_single_commit_changes_neither_voice_nor_person(people, monkeypatch):
    (people / "contacts.json").write_text("{}")
    reg = registry.at(people)
    source = reg.person_id("Dana Cole")
    before = reg.mark()
    append = reg.log.append

    def failed_commit(events):
        events = list(events)
        assert len(events) == 1
        assert len(events[0].payload["owner_voice_assertions"]) == 1
        raise OSError("synthetic failed commit")

    monkeypatch.setattr(reg.log, "append", failed_commit)
    assert people_writes.merge_into_owner("Dana Cole", [3], control=True)[0] == 500
    assert reg.mark() == before
    assert label(3) == "Dana Cole"
    assert not registry.owner_voice(people, 3)
    monkeypatch.setattr(reg.log, "append", append)
    assert people_writes.merge_into_owner("Dana Cole", [3], control=True)[0] == 200
    fresh = registry.Registry(reg.log)
    owner = fresh.owner()
    assert fresh.resolve(source) == owner
    winner = fresh.person_of(registry.voiceprint(3)).winner
    assert winner is not None
    assert winner.person == owner
    assert registry.owner_voice(people, 3)


def test_ordinary_merge_never_grants_owner_voice_authority(people):
    registry.at(people).merge("Dana Cole", "Alice", "test")
    assert label(3) == "Alice"
    assert not registry.owner_voice(people, 3)


@pytest.mark.parametrize("local,origin", [(False, None), (True, "http://elsewhere.invalid")])
def test_this_is_me_http_preserves_local_and_same_origin_controls(people, serve, monkeypatch, local, origin):
    monkeypatch.setattr(server.H, "_can_control", lambda self: local)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    host = serve(server.H).removeprefix("http://")
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    connection = http.client.HTTPConnection(host, timeout=5)
    connection.request("POST", "/people/me", json.dumps({"name": "Dana Cole", "voices": [3]}), headers)
    response = connection.getresponse()
    assert response.status == 403
    response.read()
    connection.close()
    assert label(3) == "Dana Cole"
    assert not registry.owner_voice(people, 3)


@pytest.mark.parametrize(
    "headers,status,name",
    [
        pytest.param({}, 200, "Dana Gray", id="https"),
        pytest.param({"Origin": "http://pack.example.test:8443"}, 200, "Dana Gray", id="http"),
        pytest.param({"Origin": "https://elsewhere.invalid:8443"}, 403, "Dana Cole", id="other-host"),
        pytest.param({"Origin": "https://pack.example.test:9443"}, 403, "Dana Cole", id="other-port"),
        pytest.param({"Origin": "null"}, 403, "Dana Cole", id="opaque-origin"),
        pytest.param({"Content-Type": "text/plain"}, 403, "Dana Cole", id="non-json"),
        pytest.param({"Authorization": ""}, 401, "Dana Cole", id="unauthenticated"),
    ],
)
def test_known_rename_through_https_proxy_preserves_write_gates(people, serve, monkeypatch, headers, status, name):
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_teaching", lambda: True)
    request_headers = {
        "Host": "pack.example.test:8443",
        "Origin": "https://pack.example.test:8443",
        "Content-Type": "application/json",
        "X-Forwarded-For": "100.64.0.9",
        "Authorization": "Bas" + "ic " + base64.b64encode(b"viewer:pw").decode(),
        **headers,
    }
    host = serve(server.H).removeprefix("http://")
    connection = http.client.HTTPConnection(host, timeout=5)
    try:
        connection.request(
            "POST", "/people/rename", json.dumps({"from": "Dana Cole", "to": "Dana Gray"}), request_headers
        )
        response = connection.getresponse()
        body = response.read()
        assert response.status == status, body
    finally:
        connection.close()
    assert label(3) == name


@pytest.mark.parametrize(
    "remote,teaching,status,name",
    [
        pytest.param(True, False, 403, "Dana Cole", id="remote-off-refused"),
        pytest.param(True, True, 200, "Dana Gray", id="remote-on-allowed"),
        pytest.param(False, False, 200, "Dana Gray", id="local-off-allowed"),
        pytest.param(False, True, 200, "Dana Gray", id="local-on-allowed"),
    ],
)
def test_known_rename_follows_remote_teaching_like_inbox_skip(
    people, serve, monkeypatch, remote, teaching, status, name
):
    # A rename is teaching, so a password-checked remote page may rename only with dashboard.remote_teaching on.
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_teaching", lambda: teaching)
    host = serve(server.H).removeprefix("http://")
    headers = {"Host": host, "Content-Type": "application/json"}
    if remote:
        headers |= {
            "X-Forwarded-For": "100.64.0.9",
            "Authorization": "Bas" + "ic " + base64.b64encode(b"viewer:pw").decode(),
        }
    connection = http.client.HTTPConnection(host, timeout=120)  # generous: the Linux machine can be loaded
    try:
        connection.request("POST", "/people/rename", json.dumps({"from": "Dana Cole", "to": "Dana Gray"}), headers)
        response = connection.getresponse()
        body = json.loads(response.read())
    finally:
        connection.close()
    assert response.status == status, body
    assert body["ok"] is (status == 200)
    assert label(3) == name


def test_a_refused_rename_answers_like_a_refused_inbox_skip(people):
    code, body = people_writes.rename_person("Dana Cole", "Dana Gray", control=False)
    skip_code, skip_body = inbox_state.inbox_skip("face", 3, control=False)
    assert (code, body.keys(), body["ok"]) == (skip_code, skip_body.keys(), False)
    assert "from this machine" in body["error"]
    assert label(3) == "Dana Cole"


def test_owner_target_rename_refusal_points_to_person_merge_not_voice_action(people):
    code, body = people_writes.rename_person("Bob", "Alice", control=True)
    assert code == 403
    assert "This is me" in body["error"]
    assert "That voice is mine" not in body["error"]


def test_this_is_me_after_reset_without_voices_uses_existing_person(state, faces):
    reg = registry.at(state)
    source = reg.ensure("Dana", "test", role="Gardener")
    faces.rows = {"Dana": 1}
    assert reg.owner() is None
    code, body = people_writes.merge_into_owner("Dana", None, control=True)
    assert (code, body["name"]) == (200, "Alice")
    assert reg.owner() == source
    person = reg.person(source)
    assert person is not None
    assert person.role == "Gardener"
    assert faces.renamed == [("Dana", "Alice")]
    fresh = registry.Registry(reg.log)
    assert fresh.owner() == source


def test_this_is_me_after_reset_confirms_current_voices_before_atomic_owner_rename(state, faces):
    write_voices(state, {3: "Dana"})
    faces.rows = {"Dana": 1}
    reg = registry.at(state)
    source = reg.person_id("Dana")
    before = reg.mark()
    code, body = people_writes.merge_into_owner("Dana", None, control=True)
    assert (code, body["voices"]) == (409, [3])
    assert reg.mark() == before
    assert reg.owner() is None
    write_voices(state, {4: "Dana"})
    assert people_writes.merge_into_owner("Dana", [3], control=True)[0] == 409
    assert reg.owner() is None
    code, body = people_writes.merge_into_owner("Dana", [3, 4], control=True)
    assert (code, body["name"]) == (200, "Alice")
    assert reg.owner() == source
    fresh = registry.Registry(reg.log)
    assert fresh.owner() == source
    assert registry.owner_voice(state, 3)
    assert registry.owner_voice(state, 4)


def test_this_is_me_after_reset_failed_commit_leaves_no_owner_or_authority(state, faces, monkeypatch):
    write_voices(state, {3: "Dana"})
    faces.rows = {"Dana": 1}
    reg = registry.at(state)
    source = reg.person_id("Dana")
    before = reg.mark()
    append = reg.log.append

    def fail(events):
        events = list(events)
        assert len(events) == 1
        assert events[0].payload["changes"] == {"display_name": "Alice"}
        assert len(events[0].payload["owner_voice_assertions"]) == 1
        raise OSError("synthetic failed commit")

    monkeypatch.setattr(reg.log, "append", fail)
    assert people_writes.merge_into_owner("Dana", [3], control=True)[0] == 500
    assert reg.mark() == before
    assert reg.owner() is None
    assert not registry.owner_voice(state, 3)
    assert faces.renamed == []
    monkeypatch.setattr(reg.log, "append", append)
    assert people_writes.merge_into_owner("Dana", [3], control=True)[0] == 200
    assert registry.Registry(reg.log).owner() == source
    assert registry.owner_voice(state, 3)


def test_this_is_me_without_configured_owner_name_does_not_write(state, faces):
    reg = registry.at(state)
    reg.ensure("Dana", "test")
    before = reg.mark()
    with cfg.override({"owner": {"name": "", "speaker_label": ""}}):
        code, body = people_writes.merge_into_owner("Dana", None, control=True)
    assert code == 400
    assert "Settings" in body["error"]
    assert reg.mark() == before
    assert faces.renamed == []


def test_this_is_me_with_conflicting_owner_names_does_not_write(state, faces):
    reg = registry.at(state)
    reg.ensure("Dana", "test")
    reg.ensure("Alice", "test")
    reg.ensure("Eli", "test")
    before = reg.mark()
    with cfg.override({"owner": {"name": "Alice", "speaker_label": "Eli"}}):
        assert reg.owner() is None
        code, body = people_writes.merge_into_owner("Dana", None, control=True)
    assert code == 409
    assert "different people" in body["error"]
    assert reg.mark() == before
    assert faces.renamed == []


def settle():
    """Wait out the people reads running in the background."""
    with people_reads._PEOPLE_READS.lock:
        pending = list(people_reads._PEOPLE_READS.pending.values())
    for future in pending:
        future.exception(timeout=60)


def settled_people():
    """people_view after a refresh started now (not one already running) has finished."""
    settle()
    people_reads.people_view()
    settle()
    return people_reads.people_view()


def test_people_shares_one_speech_read_and_refreshes_after_new_lines(people):
    log = fake()
    with patch.object(log, "all_lines", autospec=True, side_effect=log.all_lines) as reads:
        first = people_reads.people_view()
        second = settled_people()
        assert reads.call_count == 1
        assert first == second
        log.say({"ts_start": 65.0, "text": "Another line.", "speaker_id": 3, "audio": AUDIO})
        updated = settled_people()
        assert reads.call_count == 2
    dana = next(p for p in updated["people"] if p["name"] == "Dana Cole")
    assert dana["room"] == 2


def test_people_speech_cache_keeps_identity_edits_and_purges(people):
    log = fake()
    next(line for line in log.lines["speech.utterance"] if line.get("speaker_id") == 2)["device"] = "ingest-upload"
    people_reads.people_view()
    registry.at(people).rename("Dana Cole", "Dee", "test")
    renamed = people_reads.people_view()
    assert "Dana Cole" not in [p["name"] for p in renamed["people"]]
    assert next(p for p in renamed["people"] if p["name"] == "Dee")["room"] == 1
    before = log.mark()
    assert log.purge("speech.utterance", lambda e: e.get("speaker_id") == 2) == 1
    assert log.mark()[0] == before[0], "purging imported speech must invalidate without a new commit"
    assert log.mark()[1] != before[1]
    after = people_reads.people_view()
    assert after["unknown"]["voices"] == []
    assert next(p for p in after["people"] if p["name"] == "Dee")["room"] == 1
