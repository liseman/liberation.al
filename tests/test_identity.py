"""The identity worker (people/identity.py): names from introductions, badges, and the owner's word, over the
broker's face matches and speech lines; its records in the knowledge store."""

import json
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from liberational import config as cfg
from liberational import knowledge
from liberational.domain.identity import Basis
from liberational.domain.kernel import EventType
from liberational.people import identity, registry
from liberational.vision import faces
from voices import write_voices


def voice_name(state, sid):
    return registry.voice_names(state, [sid])[sid]


@pytest.fixture(autouse=True)
def identity_state(tmp_path):
    """Ambient face-gallery calls and explicit identity.Paths address this same isolated installation."""
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        yield


@pytest.fixture
def paths(tmp_path):
    return identity.Paths(tmp_path)


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    """An empty face gallery under the test's directory, with its unknown groups."""
    monkeypatch.setattr(faces, "DB", tmp_path / "faces" / "db.npz")
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "faces" / "unknown.npz")
    (tmp_path / "faces").mkdir(exist_ok=True)


def unit(i: int) -> np.ndarray:
    """A face embedding along one axis (the gallery's vectors are 512-d)."""
    v = np.zeros(512, np.float32)
    v[i] = 1.0
    return v


STRANGER = unit(1)


def unknown_group(vec=STRANGER) -> int:
    """A face group the face worker made for a stranger: its number."""
    return faces.group_unknown(vec)


def sighting(cluster: int, ts: float, cam: str = "cam1", detection: str | None = None) -> dict[str, Any]:
    return {
        "ts": ts,
        "cam": cam,
        "cluster": cluster,
        "detection": detection or f"d-{cam}-{ts:.0f}",
        "track_id": f"r:{cam}:1",
        "sim": 0.2,
    }


# label_speaker reads a voice's name from the event log: speakers.json holds no names.
@pytest.mark.parametrize("owner, said", [("Alice", "Alice"), ("Mary Ann", "mary  ann")], ids=["exact", "respaced"])
def test_an_introduction_never_names_a_voice_as_the_owner(tmp_path, paths, owner, said):
    write_voices(tmp_path, {7: None})
    with cfg.override({"owner": {"name": owner}}):
        assert identity.label_speaker(paths, 7, said)[0] is False
    assert voice_name(tmp_path, 7) is None


def test_an_introduction_names_an_unnamed_voice(tmp_path, paths):
    write_voices(tmp_path, {7: None})
    assert identity.label_speaker(paths, 7, "Dana") == (True, "Speaker 7 -> Dana")
    assert voice_name(tmp_path, 7) == "Dana"
    winner = registry.at(tmp_path).person_of(registry.voiceprint(7)).winner
    assert winner is not None
    assert winner.basis is Basis.INTRODUCTION


def test_a_voice_that_names_someone_keeps_its_name(tmp_path, paths):
    write_voices(tmp_path, {4: "Dana"}, forwards={9: 4})
    assert identity.label_speaker(paths, 4, " dana ") == (True, "already")
    assert identity.label_speaker(paths, 9, "Dana") == (True, "already")  # merged into Dana's voice
    ok, why = identity.label_speaker(paths, 4, "Priya")
    assert not ok
    assert "'Dana'" in why
    assert voice_name(tmp_path, 4) == "Dana"


def test_a_voice_whose_name_was_withdrawn_may_be_named(tmp_path, paths):
    write_voices(tmp_path, {4: "Dana"})
    registry.at(tmp_path).retract(registry.voiceprint(4), "test")  # named back to "Speaker 4" in the log only
    assert identity.label_speaker(paths, 4, "Priya")[0]
    assert voice_name(tmp_path, 4) == "Priya"


def test_records_live_in_the_knowledge_store_and_outlast_the_broker(tmp_path, paths, monkeypatch):
    monkeypatch.setattr(registry, "_socket", lambda state: pytest.fail("identity records reached for the broker"))
    identity.append_record(paths, {"id": "first", "name": "Dana", "status": "pending"})
    identity.append_record(paths, {"id": "second", "name": "Alice", "status": "pending"})
    identity.append_record(paths, {"id": "first", "name": "Dana", "status": "applied"})
    # Another process, after the broker's catalog was reset: the records are still there.
    again = identity.Records(knowledge.Store(tmp_path / "knowledge.sqlite3"))
    assert list(again.all().values()) == [
        {"id": "first", "name": "Dana", "status": "applied"},
        {"id": "second", "name": "Alice", "status": "pending"},
    ]


def test_records_imported_from_the_broker_are_read_and_revised(tmp_path, paths):
    # What `pack-knowledge import` writes for an old broker payload: the payload's id, keyed by the record's id.
    old = {"id": "i-1", "name": "Dana", "status": "pending", "faces": [{"cluster": 3}]}
    imported = knowledge.Record("019a0000-0000-7000-8000-000000000001", "identity.record", "i-1", 5, 5, 5, old)
    knowledge.Store(tmp_path / "knowledge.sqlite3").restore([imported])
    assert identity.load_records(paths) == {"i-1": old}
    identity.append_record(paths, {**old, "status": "rejected"})
    assert identity.load_records(paths)["i-1"]["status"] == "rejected"


# Review 4, finding 1: an overheard "I'm Alice" (the owner's name) or a badge reading it blended a stranger's face into
# the owner's face row, and wrote a face assertion naming the owner. Only the owner may give their own name.
OWNER_FACE = unit(0)[None]


@pytest.fixture
def owner_alice(tmp_path, gallery):
    """The owner is Alice, with her face row and her voice (3); voice 7 is a guest's, and the guest's face is an
    unknown group."""
    faces.save_db(["Alice"], OWNER_FACE)
    write_voices(tmp_path, {3: "Alice", 7: None})
    with cfg.override({"owner": {"name": "Alice"}}):
        yield unknown_group()


def face_owner(state, rec):
    reg = registry.at(state)
    pid = reg.person_of(registry.face_group(rec["faces"][0]["cluster"])).person
    person = reg.person(pid) if pid else None
    return person.display_name if person else None


def assert_owner_untouched(state, rec):
    assert (rec["status"], rec["applied"]["face"]) == ("pending", False)
    names, embs = faces.load_db()
    assert names == ["Alice"]
    np.testing.assert_array_equal(embs, OWNER_FACE)
    assert face_owner(state, rec) is None
    assert registry.voice_names(state, [3, 7]) == {3: "Alice", 7: None}


def test_the_owner_introducing_himself_to_a_visitor_keeps_the_visitor_off_his_face(
    tmp_path, paths, owner_alice, monkeypatch
):
    """The owner's voice already carries his name, so label_speaker said "already", and the one unknown face in the
    window (the visitor's) was enrolled as the owner."""
    now = SimpleNamespace(t=1000.0)
    brain = identity.Identity(paths, clock=lambda: now.t)
    brain.sightings.append(sighting(owner_alice, 999.0))
    monkeypatch.setattr(identity, "llm_confirm", lambda *a: {"intro": "self", "name": "Alice", "raw": "{}"})
    line = {"text": "Hi, I'm Alice, welcome in.", "speaker": "Alice", "speaker_id": 3, "ts_start": 999.0}
    assert brain.on_line({**line, "ts_end": 1000.0}) is not None
    now.t = 2000.0
    (rec,) = brain.resolve_intros()
    assert rec["confidence"] >= identity.APPLY_CONF
    assert_owner_untouched(tmp_path, rec)
    assert any("owner" in n for n in rec["applied"]["notes"])


@pytest.mark.parametrize("kind, speaker_id", [("self_intro", 7), ("third_party", 7), ("badge", None)])
def test_an_overheard_or_badge_owner_name_is_left_for_the_owner_to_confirm(
    tmp_path, paths, owner_alice, kind, speaker_id
):
    rec = {
        "id": f"i-{kind}",
        "kind": kind,
        "speaker_id": speaker_id,
        "confidence": 0.9,
        "attributes": {"role": "CEO"},
        "faces": [{"cluster": owner_alice, "detections": ["d-1"]}],
    }
    identity.apply_record(paths, rec, " alice ")
    assert_owner_untouched(tmp_path, rec)
    reg = registry.at(tmp_path)
    alice = reg.person_id("Alice")
    assert alice is not None
    assert reg.person(alice).role is None  # ty: ignore[unresolved-attribute]  # Alice is registered (her voice)


@pytest.mark.parametrize("how", [{"basis": Basis.OWNER_TAG}, {"owner": True}], ids=["confirm", "command"])
def test_the_owner_may_still_name_himself(tmp_path, paths, owner_alice, how):
    rec: dict[str, Any] = {
        "id": "i-1",
        "kind": "self_intro",
        "confidence": 0.9,
        "faces": [{"cluster": owner_alice, "detections": ["d-1", "d-2"]}],
    }
    identity.apply_record(paths, rec, "Alice", **how)
    assert (rec["status"], rec["applied"]["face"]) == ("applied", True)
    assert face_owner(tmp_path, rec) == "Alice"
    names, _embs = faces.load_db()
    assert names == ["Alice", "Alice"]  # the group's face joined her row
    reg = registry.at(tmp_path)
    alice = reg.person_id("Alice")
    # The detections the group was seen in are hers too: a reader asking about one of them is answered.
    assert [reg.person_of(registry.detection(d)).person for d in ("d-1", "d-2")] == [alice, alice]


def test_a_spoken_introduction_names_the_one_unknown_face_group_with_its_basis(tmp_path, paths, gallery, monkeypatch):
    """One stranger in view during "I'm Dana": the group is enrolled as Dana (the face worker names her from now on),
    the group and the window's detections are claimed as a spoken introduction, and her voice is named."""
    write_voices(tmp_path, {7: None})
    group = unknown_group()
    now = SimpleNamespace(t=1000.0)
    brain = identity.Identity(paths, clock=lambda: now.t)
    for ts in (995.0, 997.0, 999.0):
        brain.on_match(
            {"cam": "cam0", "ts_start": ts, "cluster": f"Face {group}", "detection": f"d-{ts:.0f}", "sim": 0.3}
        )
    brain.on_match({"cam": "cam0", "ts_start": 900.0, "name": "Erin", "sim": 0.8, "detection": "d-erin"})  # named
    monkeypatch.setattr(identity, "llm_confirm", lambda *a: {"intro": "self", "name": "Dana", "raw": "{}"})
    line = {"text": "Hi, I'm Dana.", "speaker": "Speaker 7", "speaker_id": 7, "ts_start": 998.0, "ts_end": 999.0}
    assert brain.on_line(line) is not None
    assert brain.resolve_intros() == []  # the window is still open
    now.t = 1020.0
    (rec,) = brain.resolve_intros()
    assert (rec["status"], rec["applied"]) == (
        "applied",
        {"speaker": True, "face": True, "notes": rec["applied"]["notes"]},
    )
    assert rec["faces"] == [
        {
            "cluster": group,
            "cam": "cam0",
            "cams": ["cam0"],
            "ts": 999.0,
            "sim": 0.3,
            "n": 3,
            "detections": ["d-995", "d-997", "d-999"],
        }
    ]
    assert faces.load_db()[0] == ["Dana"]
    reg = registry.at(tmp_path)
    w = reg.person_of(registry.face_group(group)).winner
    assert w is not None and (w.basis, w.source) == (Basis.INTRODUCTION, "identity")
    seen = reg.person_of(registry.detection("d-997")).winner
    assert seen is not None and seen.basis is Basis.INTRODUCTION
    assert reg.person_of(registry.detection("d-erin")).person is None  # Erin's face is not Dana's
    assert voice_name(tmp_path, 7) == "Dana"
    assert identity.load_records(paths)[rec["id"]]["status"] == "applied"


def test_two_unknown_people_in_the_window_leave_the_introduction_ambiguous(tmp_path, paths, gallery, monkeypatch):
    write_voices(tmp_path, {7: None})
    a, b = unknown_group(STRANGER), unknown_group(unit(2))
    now = SimpleNamespace(t=1000.0)
    brain = identity.Identity(paths, clock=lambda: now.t)
    brain.sightings.extend([sighting(a, 998.0, "cam0"), sighting(b, 998.5, "cam1")])
    monkeypatch.setattr(identity, "llm_confirm", lambda *a: {"intro": "third", "name": "Dana", "raw": "{}"})
    brain.recent_lines.append({"text": "hello", "speaker_id": 7, "ts_end": 990.0})
    line = {"text": "This is Dana.", "speaker": "Alice", "speaker_id": 3, "ts_start": 998.0, "ts_end": 999.0}
    heard = brain.on_line(line)
    assert heard is not None and heard["speaker_id"] == 7  # the most recent other speaker
    now.t = 1020.0
    (rec,) = brain.resolve_intros()
    assert rec["status"] == "ambiguous"
    assert [f["cluster"] for f in rec["faces"]] == [a, b]
    assert faces.load_db()[0] == []
    assert voice_name(tmp_path, 7) is None  # nothing applied until the owner picks a face


def test_confirm_refuses_a_face_the_record_does_not_have(tmp_path, paths, gallery):
    group = unknown_group()
    rec = {"id": "i-1", "kind": "badge", "speaker_id": None, "status": "pending", "faces": [{"cluster": group}]}
    identity.append_record(paths, rec)
    for face_idx in (5, -1):
        with pytest.raises(SystemExit, match=f"no face candidate {face_idx}"):
            identity.cmd_confirm(paths, "i-1", "Dana", face_idx=face_idx)
    assert faces.load_db()[0] == []  # the one face was not enrolled as Dana
    assert identity.load_records(paths)["i-1"]["status"] == "pending"


def test_records_keep_the_latest_version_per_id_and_refuse_one_without_an_id(paths):
    identity.append_record(paths, {"id": "i-1", "name": "Dana", "status": "pending"})
    identity.append_record(paths, {"id": "i-2", "name": "Erin", "status": "pending"})
    identity.append_record(paths, {"id": "i-1", "name": "Dana", "status": "rejected"})
    with pytest.raises(ValueError, match="needs an id"):
        identity.append_record(paths, {"name": "nobody"})
    got = identity.load_records(paths)
    assert list(got) == ["i-1", "i-2"]
    assert got["i-1"]["status"] == "rejected"


@pytest.mark.parametrize(
    "label, auto",
    [
        ("Speaker 4", True),
        ("speaker 4", True),
        ("SPEAKER 4", True),
        ("Unknown", True),
        (None, True),
        ("Dana", False),
        ("Speaker Dana", False),
    ],
)
def test_auto_label_ignores_capitals(label, auto):
    assert identity.auto_label(label) is auto


def daemon(state):
    return identity.Identity(identity.Paths(state), clock=lambda: 1000.0)


def measurement_events(state):
    return [s.event for s in registry.at(state).log.read("measurements")]


def test_new_quantities_record_person_time_and_line_evidence_without_context(tmp_path):
    write_voices(tmp_path, {7: "Dana"})
    brain = daemon(tmp_path)
    line = {
        "ts_start": 1100.0,
        "ts_end": 1102.0,
        "speaker_id": 7,
        "speaker": "Speaker 7",
        "text": "putting on 135; five kilometers",
    }
    assert len(brain.on_lines([line])) == 1
    events = measurement_events(tmp_path)
    assert [(e.payload["value"], e.payload["unit"]) for e in events] == [(135.0, None), (5.0, "km")]
    person = registry.at(tmp_path).person_of(registry.voiceprint(7)).winner
    assert person is not None
    assert all(e.subject == person.person for e in events)
    assert all(e.occurred_at == 1100.0 for e in events)
    assert all(e.evidence == (registry.utterance_evidence(1100.0),) for e in events)
    assert "Dana" not in json.dumps([e.to_dict() for e in events])
    assert all("activity" not in e.payload for e in events)
    assert daemon(tmp_path).on_lines([]) == []
    # The same line again (a restart before it was acknowledged): the same measurement ids, so nothing doubles.
    assert len(daemon(tmp_path).on_lines([line])) == 1
    assert len(measurement_events(tmp_path)) == 2


def test_a_failed_measurement_write_leaves_the_line_unhandled(tmp_path, monkeypatch):
    """The worker acknowledges a line only once `on_lines` returns, so a failure leaves it to come again."""
    from liberational.knowledge_eventlog import SQLiteEventLog

    brain = daemon(tmp_path)
    line = {"ts_start": 1100.0, "text": "five kilometers"}
    original = SQLiteEventLog.append
    with monkeypatch.context() as m:

        def fail(self, events):
            raise OSError("knowledge write unavailable")

        m.setattr(SQLiteEventLog, "append", fail)
        with pytest.raises(OSError, match="knowledge write unavailable"):
            brain.on_lines([line])
    assert SQLiteEventLog.append is original
    assert measurement_events(tmp_path) == []
    assert len(brain.on_lines([line])) == 1
    assert len(measurement_events(tmp_path)) == 1


def test_quantities_skip_assistant_malformed_time_and_nontext(tmp_path):
    brain = daemon(tmp_path)
    handled = brain.on_lines(
        [
            {"ts_start": 1100.0, "text": "five kilometers", "assistant": True},
            {"ts_start": "bad", "text": "five kilometers"},
            {"ts_start": 1102.0, "text": ["five kilometers"]},
            {"ts_start": 1103.0, "text": "that's ten", "speaker_id": 8},
            "not a line",
        ]
    )
    assert len(handled) == 4
    events = measurement_events(tmp_path)
    assert len(events) == 1
    assert events[0].subject is None
    assert events[0].payload["unit"] is None
    assert events[0].payload["voice_id"] == "speaker:8"


def test_forget_between_speaker_snapshot_and_measurement_append_does_not_resurrect(tmp_path, monkeypatch):
    from liberational.knowledge_eventlog import SQLiteEventLog

    write_voices(tmp_path, {7: "Dana"})
    reg = registry.at(tmp_path)
    person = reg.person_of(registry.voiceprint(7)).winner
    assert person is not None
    brain = daemon(tmp_path)
    append = SQLiteEventLog.append

    def forget_then_append(log, events):
        # Dana is forgotten after her line was attributed and before the measurement is written: the registry no
        # longer knows her, so the measurement about her is not written (nothing resurrects a forgotten person).
        if any(e.type is EventType.MEASUREMENT_RECORDED for e in events):
            monkeypatch.setattr(SQLiteEventLog, "append", append)
            reg.forget("Dana", "test")
            return append(log, events)
        return append(log, events)

    monkeypatch.setattr(SQLiteEventLog, "append", forget_then_append)
    assert len(brain.on_lines([{"ts_start": 1100.0, "speaker_id": 7, "text": "five kilometers"}])) == 1
    assert measurement_events(tmp_path) == []


def test_the_collar_names_the_one_unknown_face_group_in_view(tmp_path, gallery, vision, speech, monkeypatch):
    """The owner says "that's Dana": the unknown group matched just now is named as the owner's word, and so is the
    most recent other voice."""
    write_voices(tmp_path, {2: None})
    group = unknown_group()
    now = time.time()
    vision.add("face.match", ts_start=now - 2, cam="cam0", cluster=f"Face {group}", sim=0.3, detection="d-1")
    vision.add("face.match", ts_start=now - 1, cam="cam0", name="Erin", sim=0.9, detection="d-2")
    speech.say({"ts_start": now - 5, "ts_end": now - 4, "speaker_id": 2, "speaker": "Speaker 2", "text": "hi"})
    with cfg.override({"owner": {"name": "Alice"}}):
        r = identity.collar_intro("Dana", tmp_path)
    assert r["reply"] == "Got it, that's Dana. I've got their voice too."
    assert r["record"]["faces"][0]["detections"] == ["d-1"]
    assert faces.load_db()[0] == ["Dana"]
    reg = registry.at(tmp_path)
    for subject in (registry.face_group(group), registry.detection("d-1"), registry.voiceprint(2)):
        w = reg.person_of(subject).winner
        assert w is not None and w.basis is Basis.OWNER_TAG, subject
    assert reg.person_of(registry.detection("d-2")).person is None


def test_the_worker_follows_the_speech_log_and_resumes_after_the_last_line_handled(tmp_path, gallery, monkeypatch):
    """The worker on a real broker and segment store: a stranger matched on cam0, then "I'm Dana" with a measurement
    in it. The line's measurement is written, the one unknown group is named Dana, and the worker's saved place moves
    past the line, so the next run starts after it (the first run started from now: history is never replayed)."""
    import asyncio
    import threading

    import continuum_py

    from liberational import segment_log, speech_log

    write_voices(tmp_path, {7: None})
    group = unknown_group()
    monkeypatch.setattr(identity, "WINDOW_S", 0.5)
    monkeypatch.setattr(identity, "llm_confirm", lambda *a: {"intro": "self", "name": "Dana", "raw": "{}"})
    socket = registry._socket(tmp_path)  # the test's broker
    paths = identity.Paths(tmp_path)
    brain = identity.Identity(paths)
    old = speech_log.Writer("mic", segment_log.root(tmp_path))
    old.write(speech_log.UTTERANCES, {"ts_start": time.time() - 60, "speaker_id": 7, "text": "I'm Erin"})
    old.close()  # before the worker's first run: never replayed
    worker = threading.Thread(target=brain.run, args=(6.0,), daemon=True)
    worker.start()
    # The first run saves its place before it follows anything (history is never replayed), so waiting for that file
    # rather than a fixed time is what says the line below is new to the worker; a fixed sleep flaked when loaded.
    place = paths.state / identity.SPEECH_CURSOR
    deadline = time.monotonic() + 15.0
    while not place.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert place.exists(), "the first run saves its place before it follows"

    async def feed() -> None:
        now = time.time()  # after the wait: both followers are following, and the line is past the saved place
        client = await continuum_py.Client.connect(str(socket), client_name="test-feed")
        try:
            faces_out = segment_log.Writers(segment_log.root(tmp_path))
            match = {
                "id": segment_log.new_id(),
                "ts_start": now,
                "cam": "cam0",
                "track_id": "1",
                "frame_id": segment_log.new_id(),
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "det": 0.9,
                "name": None,
                "cluster": f"Face {group}",
                "sim": 0.3,
                "quality": 0.5,
            }
            faces_out.append(segment_log.FACE_MATCHES, "cam0", [match])
            faces_out.sync()
            faces_out.close()
            line = {
                "ts_start": now,
                "ts_end": now + 1,
                "speaker_id": 7,
                "speaker": "Speaker 7",
                "text": "I'm Dana, five kilometers",
            }
            writer = speech_log.Writer("mic", segment_log.root(tmp_path))
            writer.write(speech_log.UTTERANCES, line)
            writer.close()
        finally:
            await client.close()

    asyncio.run(feed())
    worker.join(timeout=15)
    assert not worker.is_alive()
    events = measurement_events(tmp_path)
    assert [(e.payload["value"], e.payload["unit"]) for e in events] == [(5.0, "km")]
    (rec,) = identity.load_records(paths).values()
    assert (rec["kind"], rec["name"], rec["status"]) == ("self_intro", "Dana", "applied")
    assert rec["faces"][0]["cluster"] == group
    assert faces.load_db()[0] == ["Dana"]
    assert voice_name(tmp_path, 7) == "Dana"
    saved = json.loads((tmp_path / identity.SPEECH_CURSOR).read_text())
    assert saved == {speech_log.UTTERANCES: {"mic": 2}}, "the saved place is just past the line handled"


class _Loop:
    """A stand-in for the worker's event loop: `call_soon_threadsafe` runs the call at once."""

    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


class _Acked:
    def __init__(self):
        self.positions = []

    def put_nowait(self, position):
        self.positions.append(position)


def consume(brain, items, settle=0.0):
    """Runs the brain's consumer over `items`, lets it tick for `settle` seconds, then stops it; returns the positions
    it acknowledged."""
    import queue
    import threading

    work = queue.Queue()
    for item in items:
        work.put(item)
    threading.Timer(settle, work.put, [None]).start()
    acked = _Acked()
    brain._consume(work, _Loop(), acked)
    return acked.positions


def test_the_consumer_ticks_while_records_keep_coming(tmp_path, monkeypatch):
    """Matches arrive about once a second per camera, so a tick that waited for the queue to go quiet
    never came, and introductions were never resolved."""
    monkeypatch.setattr(identity, "POLL_S", 0.05)
    brain = daemon(tmp_path)
    ticks = []
    monkeypatch.setattr(brain, "tick", lambda: ticks.append(1) or [])
    slow = []

    def on_match(m):
        slow.append(m)
        time.sleep(0.02)

    monkeypatch.setattr(brain, "on_match", on_match)
    match = {"cam": "cam0", "ts_start": 1000.0, "cluster": "Face 1", "detection": "d", "sim": 0.3}
    consume(brain, [(identity.segment_log.FACE_MATCHES, match)] * 10)
    assert len(slow) == 10
    assert len(ticks) >= 2


def test_a_failed_line_holds_the_acknowledgments_back_until_it_goes_through(tmp_path, monkeypatch):
    """An acknowledgment covers every position before it, so acknowledging a later line would have
    skipped a line whose measurements could not be written. The failed line is tried again each tick and nothing
    after it is acknowledged until it is handled."""
    monkeypatch.setattr(identity, "POLL_S", 0.05)
    brain = daemon(tmp_path)
    handled, attempts = [], {"bad": 0}

    def on_lines(lines):
        (line,) = lines
        if line["text"] == "bad" and attempts["bad"] < 2:
            attempts["bad"] += 1
            raise OSError("knowledge write unavailable")
        handled.append(line["text"])
        return lines

    monkeypatch.setattr(brain, "on_lines", on_lines)
    lines = [("line", (1, {"text": "ok"})), ("line", (2, {"text": "bad"})), ("line", (3, {"text": "later"}))]
    acked = consume(brain, lines, settle=0.5)
    # "later" was handled at once, but acknowledged only once "bad" went through, on a later tick; each position once.
    assert handled == ["ok", "later", "bad"]
    assert acked == [1, 3]
    assert attempts["bad"] == 2
