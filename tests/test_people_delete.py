"""Deleting a person (POST /people/delete): Liberational forgets their face, voice, phone, photos, the events about
them in owner knowledge, native observations, and the journal entries written from their lines, queues their transcript lines for the microphone
service, and keeps the recordings. The owner can't be deleted, and deleting someone unknown does nothing."""

import io
import json
import time
from types import SimpleNamespace
from unittest.mock import create_autospec, patch

import pytest

from liberational import config as cfg
from liberational import speech_log
from liberational.dashboard import context, server
from liberational.domain.identity import Basis
from liberational.memory import ask, history_qa, journal
from liberational.people import contacts, face_shots, forgotten, registry
from liberational.speech import clone_say
from speech_fake import fake
from test_people_unified import in_process_voice_edits
from voices import write_voices


class FakeFaces:
    def __init__(self):
        self.rows = {"Alice": 2, "Dana Cole": 3}
        self.forgotten = []

    def counts(self):
        return dict(self.rows)

    def unknown_groups(self):
        return []

    def forget(self, name):
        self.forgotten.append(name)
        return self.rows.pop(name, 0)


def journal_entry(state, sources, start, end, level=30):
    """A journal entry on the memory log, written from `sources`: a finest entry's are line ids (kept in
    `line_speakers`, since lines are in the segment store and can't be broker parents), a rollup's its child entries
    (its parents)."""
    from liberational import memory_log

    body = {"ts_start": start, "ts_end": end, "level": level, "text": "an entry"}
    if level == 30:
        body["line_speakers"] = {line: {} for line in sources}
        return memory_log.at(state).append(memory_log.JOURNAL, body, parents=[])
    return memory_log.at(state).append(memory_log.JOURNAL, body, parents=list(sources))


def journal_status(state, *ids):
    from liberational import memory_log

    present = {r["id"] for r in memory_log.at(state).range(memory_log.JOURNAL, 0.0, 1e12)}
    return ["active" if i in present else "purged" for i in ids]


def add_lines(state, *lines):
    for ts, speaker, sid in lines:
        fake().say({"ts_start": ts, "text": "Hi.", "speaker": speaker, "speaker_id": sid})


def named_voices(state):
    """{voice: its name} for each voice the event log names now."""
    return {s["id"]: s["label"] for s in server._speakers()["speakers"] if s["label"] != f"Speaker {s['id']}"}


def requests(state):
    """The requests queued for the microphone service."""
    return [json.loads(ln) for ln in (state / "voice_tags.jsonl").read_text().splitlines()]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Alice (the owner, voice 1) and Dana Cole (voices 3 and 4, a phone, a profile, a photo, a clone clip, a line,
    and a journal entry with its rollup), plus Eve added by hand and a journal entry of Alice's."""
    state, faces_dir = tmp_path / "state", tmp_path / "faces"
    write_voices(state, {1: "Alice", 3: "Dana Cole", 4: "dana cole"})
    (state / "contacts.json").write_text(json.dumps({"dana cole": "+14155550123", "Eve": "+14155550124"}))
    (state / "people_added.json").write_text(json.dumps(["Eve", "Dana Cole"]))
    (state / "people").mkdir()
    (state / "people" / "Dana Cole.md").write_text("# Dana\n")
    (faces_dir / "dana-cole").mkdir(parents=True)
    (faces_dir / "dana-cole" / "1700000000000.jpg").write_bytes(b"jpg")
    (state / "clones").mkdir()
    (state / "clones" / "3.wav").write_bytes(b"wav")
    (state / "clones" / "3.json").write_text(json.dumps({"label": "Dana Cole"}))
    add_lines(state, (2000.0, "Dana Cole", 3), (2011.0, "Alice", 1))
    hers, alices = (line["id"] for line in fake().lines[speech_log.UTTERANCES])
    entry = journal_entry(state, [hers], 1980, 2010)
    rollup = journal_entry(state, [entry], 1980, 2160, level=180)
    kept = journal_entry(state, [alices], 2010, 2040)
    faces = FakeFaces()
    files = {
        "CONTACTS_FILE": "contacts.json",
        "PEOPLE_ADDED": "people_added.json",
        "VOICE_TAGS": "voice_tags.jsonl",
        "CALLS_DIR": "calls",
    }
    for name, path in files.items():
        monkeypatch.setattr(server, name, state / path)
    monkeypatch.setattr(context, "STATE", state)
    ran = in_process_voice_edits(monkeypatch)
    monkeypatch.setattr(server, "_listening", lambda: False)  # the microphone service isn't running
    monkeypatch.setattr(server, "_faces_mod", lambda: faces)
    monkeypatch.setattr(server, "_face_last_seen_at", dict)
    monkeypatch.setattr(server, "_room_lines", create_autospec(server._room_lines, return_value=[]))
    monkeypatch.setattr(clone_say, "STATE", state)
    monkeypatch.setattr(clone_say, "OUT", state / "clones")
    with cfg.override({"owner": {"name": "Alice"}, "paths": {"state": str(state), "faces": str(faces_dir)}}):
        yield SimpleNamespace(state=state, faces=faces, ran=ran, journal=(entry, rollup, kept))


def test_forgets_everything_but_the_recordings(world):
    state = world.state
    reg = registry.at(state)
    dana = reg.person_id("Dana Cole")
    assert dana is not None
    before = {
        row.event.event_id
        for stream in (registry.PEOPLE, registry.ASSERTIONS)
        for row in reg.log.read(stream)
        if row.event.subject == dana
    }
    assert len(before) == 3  # registration and both voice assertions, stored in owner knowledge
    code, body = server.delete_person("dana cole", control=True)
    assert (code, body["ok"], body["name"]) == (200, True, "Dana Cole")
    assert body["removed"] == {
        "voices": 2,
        "faces": 3,
        "photos": 1,
        "clones": 2,
        "journal": 2,
        # This fixture has in-memory speech, not native observations. Registry events are counted separately
        # from the broker purge and must still be erased from owner knowledge below.
        "continuum": 0,
    }
    after = [row.event for stream in (registry.PEOPLE, registry.ASSERTIONS) for row in reg.log.read(stream)]
    assert not before & {event.event_id for event in after}
    assert not any(event.subject == dana for event in after)
    rebuilt = registry.Registry(reg.log)
    assert rebuilt.person_id("Dana Cole") is None
    assert rebuilt.person_of(registry.voiceprint(3)).person is None
    assert rebuilt.person_id("Alice") is not None
    assert world.ran == [["forget", "3"], ["forget", "4"]]
    assert named_voices(state) == {1: "Alice"}
    assert [(r["forget"], r["name"]) for r in requests(state)] == [([3, 4], "Dana Cole")]
    assert "isn't running" in body["notes"][0]
    assert world.faces.forgotten == ["Dana Cole"]
    assert json.loads((state / "contacts.json").read_text()) == {"Eve": "+14155550124"}
    assert json.loads((state / "people_added.json").read_text()) == ["Eve"]
    assert "Dana Cole" not in {p["name"] for p in body["people"]}
    premove = state.parent / "premove"
    assert sorted(f.suffix for f in (premove / "clones").iterdir()) == [".json", ".wav"]  # parked, not deleted
    assert sorted(f.name.rsplit("-", 1)[0] for f in (premove / "people").iterdir()) == ["Dana Cole", "dana-cole"]
    assert not (state / "clones" / "3.wav").exists()
    assert journal_status(state, *world.journal) == ["purged", "purged", "active"]


def test_the_journal_forgets_before_their_lines_are_put_back_to_unnamed(world):
    calls = []
    forget = journal.forget

    def spy(name, ids, state):
        calls.append((name, sorted(ids), (state / "voice_tags.jsonl").exists()))
        return forget(name, ids, state)

    with patch.object(journal, "forget", spy):
        server.delete_person("Dana Cole", control=True)
    assert calls == [("Dana Cole", ["3", "4"], False)]  # voice_tags not written yet


def test_the_journal_forgets_a_line_labeled_as_them_only_in_the_log(world):
    # Alice's voice, labeled Dana Cole by hand after step 4: the line itself still says Alice.
    registry.at(world.state).claim(registry.utterance(2011.0), "Dana Cole", Basis.OWNER_TAG, "dashboard")
    code, body = server.delete_person("Dana Cole", control=True)
    assert (code, body["removed"]["journal"]) == (200, 3)
    assert journal_status(world.state, world.journal[2]) == ["purged"]
    assert requests(world.state)[0]["lines"] == [
        "2000.000",
        "2011.000",
    ]  # their own voice's line, and the one in the log


def test_an_old_label_on_two_microphones_lines_forgets_only_the_line_that_is_still_theirs(world):
    for mic in ("Mic A", "Mic B"):
        fake().say({"ts_start": 2050.0, "text": "Hi.", "speaker": "Alice", "speaker_id": 1, "device": mic})
    reg = registry.at(world.state)
    reg.claim(registry.utterance(2050.0), "Dana Cole", Basis.OWNER_TAG, "dashboard")  # before lines had devices
    reg.claim(registry.utterance(2050.0, "Mic B"), "Alice", Basis.OWNER_TAG, "dashboard")  # since: Mic B's is Alice's
    assert server.delete_person("Dana Cole", control=True)[0] == 200
    assert "mic-a@2050.000" in requests(world.state)[0]["lines"]
    assert not {"2050.000", "mic-b@2050.000"} & set(requests(world.state)[0]["lines"])


def test_a_journal_failure_deletes_nothing_so_deleting_again_finishes(world):
    reg = registry.at(world.state)
    with patch.object(journal, "forget", side_effect=OSError("disk full")):
        code, body = server.delete_person("Dana Cole", control=True)
    assert (code, body["ok"]) == (500, False)
    assert "nothing was deleted" in body["error"]
    # The registry still knows her, so a second delete finds her and her journal entries under every name.
    assert (world.ran, (world.state / "voice_tags.jsonl").exists(), world.faces.forgotten) == ([], False, [])
    assert reg.person_id("Dana Cole") is not None
    code, body = server.delete_person("Dana Cole", control=True)
    assert (code, body["removed"]["journal"], body["removed"]["voices"]) == (200, 2, 2)
    assert reg.person_id("Dana Cole") is None


def test_deleting_again_or_someone_unknown_is_a_no_op(world):
    server.delete_person("Dana Cole", control=True)
    world.ran.clear()
    code, body = server.delete_person("Dana Cole", control=True)
    assert (code, body["ok"], world.ran, sum(body["removed"].values())) == (200, True, [], 0)


def test_forgetting_also_purges_their_events(world):
    reg = registry.at(world.state)
    dana = reg.person_id("Dana Cole")
    server.delete_person("dana cole", control=True)
    assert (reg.person_id("Dana Cole"), reg.person_of(registry.voiceprint(3)).person) == (None, None)
    assert reg.person_id("Alice") is not None
    left = [s.event for st in (registry.PEOPLE, registry.ASSERTIONS) for s in reg.log.read(st)]
    assert not [e for e in left if e.subject == dana]
    assert dana in [e.payload.get("person") for e in left if str(e.type) == "PersonForgotten"]


def test_forgetting_moves_their_profile_aside_under_every_name_they_had(world):
    state = world.state
    reg = registry.at(state)
    reg.ensure("Dee", "test")
    reg.rename("Dee", "Dana", "test")
    reg.rename("Dana", "Dana Cole", "test")  # renamed twice before profiles moved with a rename
    reg.ensure("Dee", "test")  # someone else has her first name now
    contacts.Book.at(state).set_phone("Dana Cole", "+14155550123", "test")
    c = json.loads((state / "contacts.json").read_text())
    (state / "contacts.json").write_text(json.dumps({**c, "Dana": "+14155550199"}))  # left under a name she had
    for n in ("dana.md", "dee.md", "Erin.md"):
        (state / "people" / n).write_text("# x\n")
    server.delete_person("Dana Cole", control=True)
    parked = sorted(f.name.rsplit("-", 1)[0] for f in (state.parent / "premove" / "people").glob("*.md"))
    assert parked == ["Dana Cole", "dana"]
    assert sorted(f.name for f in (state / "people").iterdir()) == ["Erin.md", "dee.md"]
    left = [json.dumps(s.event.to_dict()) for s in reg.log.read(registry.PEOPLE)]
    assert not any("+14155550123" in b for b in left)  # their number went with their events
    assert json.loads((state / "contacts.json").read_text()) == {"Eve": "+14155550124"}


@pytest.mark.parametrize("question", ["What did Dana Cole say yesterday?", "When was Dana last seen?", "who's dana"])
def test_asking_about_them_afterwards_doesnt_look_like_anything(world, question):
    reg = registry.at(world.state)
    reg.claim(registry.voiceprint(3), "Dana", Basis.OWNER_TAG, "test")
    reg.rename("Dana", "Dana Cole", "test")  # renamed before she was forgotten
    server.delete_person("Dana Cole", control=True)
    with (
        patch.object(history_qa, "collect", side_effect=AssertionError("looked for evidence")),
        patch.object(ask, "local_answer", side_effect=AssertionError("asked the model")),
    ):
        assert ask.execute(question, state_dir=world.state)["answer"] == forgotten.LINE
    assert forgotten.store().has("dana") and forgotten.store().has("dana cole")


def dana_renamed_dee(state):
    """Voice 5 was Dana's and said two lines as Dana (one through a voice that is unnamed now), then she was renamed
    Dee. The lines keep the name they were heard under."""
    reg = registry.at(state)
    reg.claim(registry.voiceprint(5), "Dana", Basis.OWNER_TAG, "test", at=100.0)
    add_lines(state, (3000.0, "Dana", 5), (3050.0, "Dana", 7))
    reg.rename("Dana", "Dee", "test")
    return reg


def test_a_leftover_row_under_a_renamed_persons_old_name_is_refused(world, monkeypatch):
    state = world.state
    reg = dana_renamed_dee(state)
    hers = journal_entry(state, [], 2990, 3020)  # an entry of hers from before the rename, its line long gone
    call = [{"who": "Dana", "text": "Hi, it's Dana.", "ts": 2990.0, "src": "call:x"}]  # heard before the rename
    monkeypatch.setattr(server, "_call_lines", lambda: call)
    assert "Dana" in {p["name"] for p in server.people_roster()}
    code, body = server.delete_person("Dana", control=True)
    assert (code, body["ok"]) == (409, False)
    assert "Dee" in body["error"]
    assert (world.ran, (state / "voice_tags.jsonl").exists(), world.faces.forgotten) == ([], False, [])
    assert journal_status(state, hers) == ["active"]
    assert reg.known_as("Dana") == reg.person_id("Dee") is not None


def test_a_voice_clone_that_shares_their_name_stays(world):
    # A clone recorded from one uploaded clip, named like her: not her voice.
    state = world.state
    write_voices(state, {9: "Dana Cole"})
    (state / "clones" / "9.wav").write_bytes(b"wav")
    (state / "clones" / "9.json").write_text(json.dumps({"label": "Dana Cole", "source": "recorded"}))
    code, body = server.delete_person("Dana Cole", control=True)
    assert (code, body["removed"]["voices"], body["removed"]["clones"]) == (200, 2, 2)
    assert world.ran == [["forget", "3"], ["forget", "4"]]  # never the clone's voice
    assert requests(state)[0]["forget"] == [3, 4]
    assert (state / "clones" / "9.wav").exists()


def test_owner_and_remote_requests_are_refused(world):
    assert server.delete_person("alice", control=True)[0] == 403
    assert server.delete_person("Dana Cole", control=False)[0] == 403
    assert server.delete_person("", control=True)[0] == 400
    assert world.ran == []
    assert named_voices(world.state) == {1: "Alice", 3: "Dana Cole", 4: "Dana Cole"}


def test_concurrent_reference_attach_cannot_survive_deletion(world, serve, monkeypatch):
    import concurrent.futures
    import threading

    import requests as http_requests
    from PIL import Image

    from liberational.people import references

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buf, "PNG")
    item = references.stage(world.state, buf.getvalue(), "synthetic")
    entered, release, deleted = threading.Event(), threading.Event(), threading.Event()
    attach = references.attach

    def paused_attach(state, token, person):
        entered.set()
        assert release.wait(5)
        return attach(state, token, person)

    def delete():
        result = server.delete_person("Dana Cole", control=True)
        deleted.set()
        return result

    monkeypatch.setattr(references, "attach", paused_attach)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    url = serve(server.H)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        saving = pool.submit(
            http_requests.post,
            url + "/people/reference-attach",
            json={"id": item["id"], "person": "Dana Cole"},
            timeout=5,
        )
        assert entered.wait(3)
        deleting = pool.submit(delete)
        deleted.wait(0.15)
        release.set()
        assert saving.result(timeout=5).status_code == 200
        assert deleting.result(timeout=5)[0] == 200
    assert references.listing(world.state, "Dana Cole") == []


def test_their_continuum_subjects_include_the_voices_the_delete_forgets(world, monkeypatch):
    # Read before the voice forget retracts their voiceprints, or those would no longer resolve to them.
    asked = []
    monkeypatch.setattr(server, "_purge_continuum", lambda subjects: (asked.append(set(subjects)), (0, 0))[1])
    pid = registry.at(world.state).person_id("Dana Cole")
    assert server.delete_person("dana cole", control=True)[0] == 200
    assert asked == [{pid, *(registry.subject_uuid(registry.voiceprint(i)) for i in (3, 4))}]


def test_a_failed_continuum_purge_keeps_them_in_the_registry_to_delete_again(world, monkeypatch):
    monkeypatch.setattr(server, "_purge_continuum", lambda subjects: (0, 1))
    code, body = server.delete_person("dana cole", control=True)
    assert (code, body["ok"]) == (500, False) and "delete them again" in body["error"]
    assert registry.at(world.state).person_id("Dana Cole") is not None


def test_deleting_someone_clears_their_face_crops_and_keeps_everyone_elses(world, vision):
    # Clearing every crop made the dashboard cut every face again from video, which starved the pets' photos.
    hers, alices = (vision.face(time.time() - 60, "cam0", who, 0.8)["id"] for who in ("Dana Cole", "Alice"))
    for sid in (hers, alices):
        face_shots.write(world.state, sid, "0", b"jpg")
        face_shots.write(world.state, sid, "0", b"jpg", "shots_grid")
    assert server.delete_person("dana cole", control=True)[0] == 200
    assert sorted(p.name for p in (world.state / "shots").rglob("*.jpg")) == sorted(
        face_shots.path(world.state, alices, "0", kind).name for kind in ("shots", "shots_grid")
    )


def test_a_delete_stopped_by_the_continuum_purge_finishes_when_run_again(world, monkeypatch):
    reg = registry.at(world.state)
    pid = reg.person_id("Dana Cole")
    asked: list[set[str]] = []
    results = iter([(0, 1), (3, 0)])  # the first purge fails partway; the second one works

    def purge(subjects):
        asked.append(set(subjects))
        return next(results)

    monkeypatch.setattr(server, "_purge_continuum", purge)
    assert server.delete_person("dana cole", control=True)[0] == 500
    code, body = server.delete_person("dana cole", control=True)
    assert (code, body["ok"]) == (200, True)
    # The second run still found everything of hers on the broker, her voiceprints included.
    prints = {registry.subject_uuid(registry.voiceprint(i)) for i in (3, 4)}
    assert asked[1] >= {pid, *prints}
    assert reg.person_id("Dana Cole") is None and named_voices(world.state) == {1: "Alice"}
    assert "dana cole" not in json.loads((world.state / "contacts.json").read_text())


def test_a_failed_forget_in_the_registry_is_not_reported_as_done(world, monkeypatch):
    real = registry.Registry.forget
    calls = []

    def flaky(self, name, source):
        calls.append(name)
        if len(calls) == 1:
            raise OSError("the event log is unavailable")
        return real(self, name, source)

    monkeypatch.setattr(registry.Registry, "forget", flaky)
    code, body = server.delete_person("dana cole", control=True)
    assert (code, body["ok"]) == (500, False)
    assert "still in the person registry" in body["error"] and "delete them again" in body["error"]
    assert registry.at(world.state).person_id("Dana Cole") is not None
    code, body = server.delete_person("dana cole", control=True)
    assert (code, body["ok"]) == (200, True)
    assert registry.at(world.state).person_id("Dana Cole") is None
