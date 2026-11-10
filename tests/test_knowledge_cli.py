import dataclasses
import hashlib
import json
import shutil
from unittest.mock import create_autospec

import pytest

from liberational import config as cfg
from liberational import knowledge, knowledge_backup, knowledge_cli
from liberational.knowledge import Record, Store
from liberational.knowledge_eventlog import record_of
from liberational.people import registry


def bundle(records=(), attachments=None):
    return knowledge_backup.Bundle(tuple(records), (), attachments or {})


@pytest.mark.parametrize("replacement", [0, True, float("inf"), float("nan"), []])
def test_invalid_enrollment_vectors_are_rejected_before_import(replacement):
    row = Record(
        "voice",
        "voice.reference",
        "speaker:1",
        1,
        1,
        1,
        {
            "dims": 1,
            "vector": [replacement],
            "model": "campp",
            "source_id": "source",
            "voiceprint": 1,
            "person": "alice",
        },
        ("alice",),
    )
    with pytest.raises(ValueError, match="vector"):
        knowledge_cli.validate(bundle([row]))


def test_face_binding_and_native_dimension_must_match():
    row = Record(
        "face",
        "face.reference",
        "alice",
        1,
        1,
        1,
        {"dims": 1, "vector": [1.0], "model": "arcface", "source_id": "source", "person": "alice", "name": "Alice"},
        ("alice",),
    )
    with pytest.raises(ValueError, match="face"):
        knowledge_cli.validate(bundle([row]))


def test_event_index_metadata_cannot_disagree_with_its_envelope(tmp_path):
    reg = registry.at(tmp_path)
    reg.ensure("Alice", "test")
    (event,) = reg.log.read("people")
    row = dataclasses.replace(record_of(event.event), subjects=("another person",))
    with pytest.raises(ValueError, match="metadata"):
        knowledge_cli.validate(bundle([row]))


def test_suppression_hashes_never_import_a_raw_name():
    row = Record("alice", "forgotten.names", "alice", 0, 0, 0, {})
    with pytest.raises(ValueError, match="suppression"):
        knowledge_cli.validate(bundle([row]))


def test_attachment_metadata_must_match_the_verified_blob():
    row = Record(
        "attachment",
        "knowledge.attachment",
        "config/pack.toml",
        0,
        0,
        0,
        {"sha256": hashlib.sha256(b"different").hexdigest(), "size": 4},
    )
    with pytest.raises(ValueError, match="metadata"):
        knowledge_cli.validate(bundle([row], {"config/pack.toml": b"real"}))


def test_sqlite_export_verify_and_import_survive_source_removal(tmp_path, monkeypatch, capsys):
    source, destination = tmp_path / "source", tmp_path / "destination"
    reg = registry.at(source)
    reg.claim(registry.voiceprint(1), "Alice", registry.Basis.OWNER_TAG, "test")
    person = reg.person_id("Alice")
    artifact = tmp_path / "taught.zip"
    monkeypatch.setattr(knowledge_cli, "_files", create_autospec(knowledge_cli._files, return_value={}))
    knowledge_cli.main(["export", "--source", "sqlite", "--state", str(source), "--output", str(artifact)])
    registry.close_all()
    (source / "knowledge.sqlite3").unlink()
    knowledge_cli.main(["verify", "--input", str(artifact)])
    knowledge_cli.main(["import", "--state", str(destination), "--input", str(artifact)])
    restored = registry.at(destination)
    assert restored.person_id("Alice") == person
    assert restored.person_of(registry.voiceprint(1)).person == person
    before = Store(destination / "knowledge.sqlite3").revision()
    knowledge_cli.main(["import", "--state", str(destination), "--input", str(artifact)])
    assert Store(destination / "knowledge.sqlite3").revision() == before
    assert "Alice" not in capsys.readouterr().out


def unit(dims):
    return [1.0] + [0.0] * (dims - 1)


def enrollment(dims):
    return {"dims": dims, "vector": unit(dims), "model": "test-model", "source_id": "source-1"}


# One record of each collection the owner teaches, as its writer stores it (#3262: pet.not and transcript.correction
# made the live export fail, and face.not, voice.not, and activity.kind are newer).
OWNER_MEDIA_MARK = Record(
    "tc",
    "transcript.correction",
    "tc",
    7,
    8,
    9,
    {"action": "media", "source": "owner", "lines": [{"id": 1, "line": "k1"}]},
    ("utterance:k1",),
)
OWNER_ROWS = (
    Record("idr", "identity.record", "7", 1, 1, 1, {"id": 7, "name": "Alice"}),
    Record("pp", "pet.profile", "pet-1", 1, 1, 1, {"name": "Rex", "species": "dog"}, ("pet-1",)),
    Record("ps", "pet.sighting", "41", 1, 1, 1, {"pet": "pet-1", "cam": "cam0", "source": "owner"}, ("pet-1",)),
    Record(
        "pn",
        "pet.not",
        "42:pet-1",
        2,
        2,
        2,
        {"track_id": "42", "pet": "pet-1", "cam": "cam0", "active": True, "source": "owner"},
        ("pet-1",),
    ),
    Record(
        "pr",
        "pet.reference",
        "41",
        1,
        1,
        1,
        {**enrollment(4), "track_id": "41", "species": "dog", "label": "Rex", "owner_label": {"pet": "pet-1"}},
        ("pet-1",),
    ),
    Record(
        "fn",
        "face.not",
        "9:alice",
        3,
        3,
        3,
        {"group": 9, "person": "alice", "active": True, "source": "owner"},
        ("alice",),
    ),
    Record(
        "fr", "face.reference", "alice", 1, 1, 1, {**enrollment(512), "person": "alice", "name": "Alice"}, ("alice",)
    ),
    Record("vn", "voice.not", "line-1:alice", 4, 5, 6, {"vector": unit(4), "source": "owner"}, ("alice",)),
    Record(
        "vr", "voice.reference", "speaker:3", 1, 1, 1, {**enrollment(4), "voiceprint": 3, "person": "alice"}, ("alice",)
    ),
    OWNER_MEDIA_MARK,
    Record("ak", "activity.kind", "squat", 1, 1, 1, {"label": "Squat", "active": True}),
    Record("an", "activity.named", "ev-1", 1, 1, 1, {"name": "Stretching"}),
    Record("af", "activity.feedback", "ev-1", 1, 1, 1, {"verdict": "right"}),
    Record("pl", "place.plan", "place-1", 1, 1, 1, {"plan": "room Kitchen 0 0 3 4", "source": "owner"}),
    Record("pna", "place.named", "place-1", 1, 1, 1, {"name": "Home"}),
    Record("zd", "zone.defined", "cam0", 1, 1, 1, {"zones": [{"name": "Couch", "points": [[0, 0], [1, 1]]}]}),
    Record("dl", "device.link", "link-1", 1, 1, 1, {"device": "aa:bb", "person": "alice"}),
    Record("dloc", "device.location", "location-1", 1, 1, 1, {"place": "place-1", "room": "Kitchen"}),
    Record("os", "object.sighting", "mug", 1, 1, 1, {"label": "Mug", "crop": "c.jpg"}),
    Record(
        hashlib.sha256(b"forgotten").hexdigest(),
        "forgotten.names",
        hashlib.sha256(b"forgotten").hexdigest(),
        0,
        0,
        0,
        {},
    ),
)


def test_every_owner_taught_record_and_file_survives_export_wipe_and_import(tmp_path):
    state, faces, voices = tmp_path / "state", tmp_path / "faces", tmp_path / "voices"
    reg = registry.at(state)
    reg.claim(registry.voiceprint(1), "Alice", registry.Basis.OWNER_TAG, "test")
    knowledge.at(state / "knowledge.sqlite3").put(OWNER_ROWS)
    token = "d" * 32
    taught = {
        state / "voiceprints.json": b'{"1": "alice"}',
        state / "person_references" / token / "meta.json": json.dumps(
            {"id": token, "person": "alice", "file": "media.jpg"}
        ).encode(),
        state / "person_references" / token / "media.jpg": b"photo",
        faces / "db.npz": b"owner rows",
        faces / "nots.npz": b"owner negatives",
        voices / "owner_ref.wav": b"sample",
    }
    for path, content in taught.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    before = {item.record.id: item.record for item in knowledge.at(state / "knowledge.sqlite3").snapshot().records}
    artifact = tmp_path / "taught.zip"
    paths = ["--state", str(state), "--faces", str(faces)]
    with cfg.override({"owner": {"voice_ref": str(voices / "owner_ref.wav")}}):
        knowledge_cli.main(["export", "--source", "sqlite", *paths, "--output", str(artifact)])
        registry.close_all()
        knowledge.close_all()
        for root in (state, faces, voices):
            shutil.rmtree(root)
        knowledge_cli.main(["import", "--input", str(artifact), *paths])
    after = {
        item.record.id: item.record
        for item in Store(state / "knowledge.sqlite3").snapshot().records
        if item.record.collection != "knowledge.attachment"
    }
    assert after == before
    assert {path: path.read_bytes() for path in taught} == taught


def test_export_leaves_out_the_audio_taggers_marks_and_keeps_the_owners(tmp_path):
    state = tmp_path / "state"
    tagger = Record(
        "tag",
        "transcript.correction",
        "tag",
        1,
        1,
        1,
        {"action": "media", "source": "classifier", "lines": [{"id": 2, "line": "k2"}]},
        ("utterance:k2",),
    )
    undo = Record("undo", "transcript.correction", "tc", 9, 9, 9, {"action": "undo", "undo": "tc"})
    knowledge.at(state / "knowledge.sqlite3").put([OWNER_MEDIA_MARK, tagger, undo])
    with cfg.override({"owner": {"voice_ref": str(tmp_path / "none.wav")}}):
        bundle = knowledge_cli.export("sqlite", state, tmp_path / "faces", tmp_path / "taught.zip")
    assert {r.id for r in bundle.records} == {"tc", "undo"}


def test_validation_names_every_unknown_collection_at_once():
    rows = [Record("a", "later.thing", None, 0, 0, 0, {}), Record("b", "another.thing", None, 0, 0, 0, {})]
    with pytest.raises(ValueError, match="another.thing, later.thing"):
        knowledge_cli.validate(bundle(rows))


def test_invalid_application_bundle_cannot_create_destination_state(tmp_path):
    artifact = tmp_path / "invalid.zip"
    knowledge_backup.write_bundle(bundle([Record("unknown", "unsupported", None, 0, 0, 0, {})]), artifact)
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="unsupported"):
        knowledge_cli.main(["import", "--state", str(destination), "--input", str(artifact)])
    assert not (destination / "knowledge.sqlite3").exists()


def test_missing_sqlite_export_does_not_initialize_empty_state(tmp_path, monkeypatch):
    source = tmp_path / "source"
    monkeypatch.setattr(knowledge_cli, "_files", create_autospec(knowledge_cli._files, return_value={}))
    with pytest.raises(FileNotFoundError):
        knowledge_cli.export("sqlite", source, tmp_path / "faces", tmp_path / "taught.zip")
    assert not source.exists()
