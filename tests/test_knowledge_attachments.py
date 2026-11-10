"""Only taught files enter portable bundles; logical attachment names never become extraction paths."""

import hashlib
import json

import pytest

from liberational import knowledge
from liberational import knowledge_attachments as A
from liberational.knowledge_backup import Bundle


def put(path, content=b"synthetic"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def reference(state, token, *, person="Dana", filename="media.jpg", **extra):
    folder = state / "person_references" / token
    put(folder / "meta.json", json.dumps({"id": token, "person": person, "file": filename, **extra}).encode())
    put(folder / filename)
    return folder


def row(id="owner-event", value=None):
    return knowledge.Record(id, "events:people", None, 1, 1, 1, {"name": "Dana"} if value is None else value)


def test_collection_reads_only_exact_taught_files_and_confirmed_reference_media(tmp_path):
    state, faces = tmp_path / "state", tmp_path / "faces"
    for name in (
        "voiceprints.json",
        "voice_enrollments.json",
        "card_prefs.json",
        "people_undo.json",
        "contacts.json",
        "people_added.json",
    ):
        put(state / name, b"{}")
    for name in ("12.wav", "12.txt", "12.json", "12.checks.json"):
        put(state / "clones" / name)
    attached = reference(state, "a" * 32)
    reference(state, "b" * 32, person=None)
    put(attached / "input", b"raw upload excluded")
    put(state / "people" / "dana.md", b"generated transcript profile excluded")
    put(state / "clones" / "raw.wav")
    put(state / "clones" / "12.wav.tmp")
    put(state / "clones" / "12.checks.wav")
    put(state / "clones" / "nested" / "13.wav")
    put(state / "recordings" / "12.wav")
    put(state / ".env", b"SECRET=excluded")
    put(state / "models" / "weights.bin")
    put(faces / "db.npz", b"synthetic named enrollment archive")
    put(faces / "nots.npz", b"synthetic owner negatives")
    put(faces / "learned.npz", b"machine-learned rows excluded")
    put(faces / "unknown" / "cluster.jpg")
    files = A.collect_files(state, faces)
    assert set(files) == {
        *(
            f"state/{name}"
            for name in (
                "voiceprints.json",
                "voice_enrollments.json",
                "card_prefs.json",
                "people_undo.json",
                "contacts.json",
                "people_added.json",
            )
        ),
        *(f"state/clones/{name}" for name in ("12.wav", "12.txt", "12.json", "12.checks.json")),
        f"state/person_references/{'a' * 32}/meta.json",
        f"state/person_references/{'a' * 32}/media.jpg",
        "faces/db.npz",
        "faces/nots.npz",
    }
    assert all(str(tmp_path).encode() not in content for content in files.values())


def test_explicit_configuration_and_owner_voice_are_named_without_source_paths(tmp_path):
    config = put(tmp_path / "private" / "settings.toml", b"[owner]\nname='Alice'")
    devices = put(tmp_path / "private" / "devices.toml", b"[devices]")
    voice = put(tmp_path / "private" / "sample.wav", b"synthetic wave")
    put(voice.with_suffix(".txt"), b"synthetic transcript")
    assert A.collect_files(
        tmp_path / "state", tmp_path / "faces", config_file=config, devices_file=devices, voice_reference=voice
    ) == {
        "config/pack.toml": b"[owner]\nname='Alice'",
        "config/devices.toml": b"[devices]",
        "voices/owner_ref.wav": b"synthetic wave",
        "voices/owner_ref.txt": b"synthetic transcript",
    }


def test_missing_optional_state_is_empty_but_explicit_missing_file_fails(tmp_path):
    assert A.collect_files(tmp_path / "state", tmp_path / "faces") == {}
    with pytest.raises(FileNotFoundError):
        A.collect_files(tmp_path / "state", tmp_path / "faces", config_file=tmp_path / "missing.toml")


@pytest.mark.parametrize("filename", ["media.jpg", "media.wav", "media.mp4"])
def test_attached_reference_uses_its_declared_supported_media_only(tmp_path, filename):
    state = tmp_path / "state"
    reference(state, "a" * 32, filename=filename)
    assert set(A.collect_files(state, tmp_path / "faces")) == {
        f"state/person_references/{'a' * 32}/meta.json",
        f"state/person_references/{'a' * 32}/{filename}",
    }


@pytest.mark.parametrize(
    "metadata",
    [
        {"id": "b" * 32, "person": "Dana", "file": "media.jpg"},
        {"id": "a" * 32, "person": "", "file": "media.jpg"},
        {"id": "a" * 32, "person": 42, "file": "media.jpg"},
        {"id": "a" * 32, "person": "Dana", "file": "../../secret"},
        {"id": "a" * 32, "person": "Dana", "file": "input"},
    ],
)
def test_attached_reference_metadata_cannot_redirect_collection(tmp_path, metadata):
    state = tmp_path / "state"
    put(state / "person_references" / ("a" * 32) / "meta.json", json.dumps(metadata).encode())
    with pytest.raises(ValueError, match="reference"):
        A.collect_files(state, tmp_path / "faces")


def test_preview_media_is_never_opened_even_if_it_is_a_symlink(tmp_path):
    state = tmp_path / "state"
    folder = reference(state, "a" * 32, person=None)
    (folder / "media.jpg").unlink()
    (folder / "media.jpg").symlink_to(tmp_path / "secret")
    assert A.collect_files(state, tmp_path / "faces") == {}


@pytest.mark.parametrize(
    "location", ["state", "clones", "clone-file", "reference-folder", "metadata", "media", "faces", "enrollment"]
)
def test_symlinks_at_each_allowlisted_boundary_fail_closed(tmp_path, location):
    state, faces = tmp_path / "state", tmp_path / "faces"
    folder = reference(state, "a" * 32)
    clone = put(state / "clones" / "1.wav")
    enrollment = put(faces / "db.npz")
    target = {
        "state": state,
        "clones": clone.parent,
        "clone-file": clone,
        "reference-folder": folder,
        "metadata": folder / "meta.json",
        "media": folder / "media.jpg",
        "faces": faces,
        "enrollment": enrollment,
    }[location]
    saved = target.with_name(target.name + "-original")
    target.rename(saved)
    target.symlink_to(saved)
    with pytest.raises(ValueError, match="symlink"):
        A.collect_files(state, faces)


@pytest.mark.parametrize("mutation", ["rewrite", "unlink"])
def test_file_mutated_during_read_is_refused(tmp_path, monkeypatch, mutation):
    state = tmp_path / "state"
    target = put(state / "contacts.json", b"original")
    fstat = A.os.fstat
    calls = []

    def mutate(descriptor):
        calls.append(descriptor)
        if len(calls) == 2:
            if mutation == "rewrite":
                target.write_bytes(b"changed content")
            else:
                target.unlink()
        return fstat(descriptor)

    monkeypatch.setattr(A.os, "fstat", mutate)
    with pytest.raises(ValueError, match="changed|disappeared"):
        A.collect_files(state, tmp_path / "faces")


@pytest.mark.parametrize("limit", ["MAX_BLOB_BYTES", "MAX_TOTAL_BYTES", "MAX_ATTACHMENTS"])
def test_collection_refuses_bounds_instead_of_dropping_files(tmp_path, monkeypatch, limit):
    put(tmp_path / "state" / "contacts.json")
    monkeypatch.setattr(A.backup, limit, 0)
    with pytest.raises(ValueError, match="limit"):
        A.collect_files(tmp_path / "state", tmp_path / "faces")


def test_managed_import_survives_source_removal_without_extracting_logical_paths(tmp_path):
    source = put(tmp_path / "source.toml", b"owner settings")
    bundle = Bundle((row(),), (), {"config/pack.toml": source.read_bytes()})
    store = knowledge.Store(tmp_path / "new-install" / "knowledge.sqlite3")
    result = A.import_bundle(store, bundle)
    source.unlink()
    assert result == {"records": 1, "attachments": 1, "blobs": 1}
    assert A.read_attachment(store, "config/pack.toml") == b"owner settings"
    assert A.read_attachment(store, "missing") is None
    assert not (store.path.parent / "config").exists()
    assert A.exportattachments(store) == {"config/pack.toml": b"owner settings"}
    refs = store.query(A.COLLECTION)
    assert len(refs) == 1
    assert refs[0].record.key == "config/pack.toml"
    assert refs[0].record.start_ns == refs[0].record.end_ns == refs[0].record.recorded_ns == 0
    before = store.snapshot()
    assert A.import_bundle(store, bundle) == {"records": 1, "attachments": 1, "blobs": 0}
    assert store.snapshot() == before
    exported = Bundle(tuple(saved.record for saved in store.snapshot().records), (), A.exportattachments(store))
    restored = knowledge.Store(tmp_path / "third-install" / "knowledge.sqlite3")
    A.import_bundle(restored, exported)
    assert A.read_attachment(restored, "config/pack.toml") == b"owner settings"


def test_latest_content_uses_a_new_immutable_reference_and_shared_content_deduplicates(tmp_path):
    store = knowledge.Store(tmp_path / "knowledge.sqlite3")
    A.import_bundle(store, Bundle((), (), {"config/pack.toml": b"first"}))
    first = store.query(A.COLLECTION)[0]
    A.import_bundle(store, Bundle((), (), {"config/pack.toml": b"second", "alias.toml": b"second"}))
    assert store.get(first.record.id) == first
    assert A.exportattachments(store) == {"alias.toml": b"second", "config/pack.toml": b"second"}
    assert len(list((tmp_path / "knowledge_blobs").iterdir())) == 2
    assert len(store.query(A.COLLECTION, key="config/pack.toml")) == 2


def test_whole_bundle_conflict_keeps_old_logical_refs_and_rolls_back_erasures(tmp_path):
    store = knowledge.Store(tmp_path / "knowledge.sqlite3")
    A.import_bundle(store, Bundle((row(), row("victim")), (), {"config/pack.toml": b"old"}))
    before = store.snapshot()
    with pytest.raises(knowledge.Conflict):
        A.import_bundle(
            store, Bundle((row("new-first"), row(value={"different": True})), ("victim",), {"config/pack.toml": b"new"})
        )
    assert store.snapshot() == before
    assert A.read_attachment(store, "config/pack.toml") == b"old"
    # Orphaned immutable bytes are safe: another concurrent importer could already be using them.
    assert (tmp_path / "knowledge_blobs" / hashlib.sha256(b"new").hexdigest()).read_bytes() == b"new"


def test_corrupted_existing_content_hash_file_is_never_overwritten_or_referenced(tmp_path):
    store = knowledge.Store(tmp_path / "knowledge.sqlite3")
    root = tmp_path / "knowledge_blobs"
    digest = hashlib.sha256(b"new").hexdigest()
    put(root / digest, b"bad")
    before = store.snapshot()
    with pytest.raises(ValueError, match="hash"):
        A.import_bundle(store, Bundle((row(),), (), {"config/pack.toml": b"new"}))
    assert (root / digest).read_bytes() == b"bad"
    assert store.snapshot() == before


def test_reading_corrupted_managed_bytes_never_returns_partial_or_empty_data(tmp_path):
    store = knowledge.Store(tmp_path / "knowledge.sqlite3")
    A.import_bundle(store, Bundle((), (), {"photo.bin": b"image"}))
    (tmp_path / "knowledge_blobs" / hashlib.sha256(b"image").hexdigest()).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash"):
        A.read_attachment(store, "photo.bin")
    with pytest.raises(ValueError, match="hash"):
        A.exportattachments(store)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a\\escape", "."])
def test_managed_import_never_interprets_untrusted_names_as_paths(tmp_path, name):
    store = knowledge.Store(tmp_path / "knowledge.sqlite3")
    with pytest.raises(ValueError, match="relative"):
        A.import_bundle(store, Bundle((row(),), (), {name: b"content"}))
    assert store.snapshot().records == ()
    assert not (tmp_path / "knowledge_blobs").exists()


@pytest.mark.parametrize("location", ["directory", "blob"])
def test_managed_storage_refuses_symlinks(tmp_path, location):
    store = knowledge.Store(tmp_path / "knowledge.sqlite3")
    outside = put(tmp_path / "outside.bin", b"image")
    root = tmp_path / "knowledge_blobs"
    if location == "directory":
        root.symlink_to(tmp_path)
    else:
        root.mkdir()
        (root / hashlib.sha256(b"image").hexdigest()).symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        A.import_bundle(store, Bundle((), (), {"photo.bin": b"image"}))


def test_invalid_attachment_reference_cannot_redirect_a_typed_reader(tmp_path):
    store = knowledge.Store(tmp_path / "knowledge.sqlite3")
    invalid = knowledge.Record("bad", A.COLLECTION, "photo.bin", 0, 0, 0, {"sha256": "../outside", "size": 1})
    store.put([invalid])
    with pytest.raises(ValueError, match="reference"):
        A.read_attachment(store, "photo.bin")


def test_erased_reference_id_cannot_be_resurrected_by_a_late_import(tmp_path):
    store = knowledge.Store(tmp_path / "knowledge.sqlite3")
    bundle = Bundle((), (), {"photo.bin": b"image"})
    A.import_bundle(store, bundle)
    (reference,) = store.query(A.COLLECTION)
    store.remove([reference.record.id])
    before = store.snapshot()
    with pytest.raises(knowledge.Deleted):
        A.import_bundle(store, bundle)
    assert store.snapshot() == before


def homes(tmp_path):
    return tmp_path / "state", tmp_path / "faces", tmp_path / "voices" / "owner_ref.wav"


def test_restore_puts_taught_files_home_and_leaves_settings_in_the_bundle(tmp_path):
    state, faces, voice = homes(tmp_path)
    token = "c" * 32
    files = {
        "state/voiceprints.json": b"{}",
        "state/clones/3.wav": b"clone",
        f"state/person_references/{token}/meta.json": b"{}",
        f"state/person_references/{token}/media.jpg": b"photo",
        "faces/db.npz": b"rows",
        "faces/nots.npz": b"negatives",
        "voices/owner_ref.wav": b"sample",
        "voices/owner_ref.txt": b"what the sample says",
        "config/pack.toml": b"settings",
    }
    placed = A.restore_files(files, state, faces, voice)
    assert placed == A.Placed(restored=8, same=0, bundle=1, differ=())
    assert (state / "clones" / "3.wav").read_bytes() == b"clone"
    assert (state / "person_references" / token / "media.jpg").read_bytes() == b"photo"
    assert (faces / "nots.npz").read_bytes() == b"negatives"
    assert voice.read_bytes() == b"sample"
    assert voice.with_suffix(".txt").read_bytes() == b"what the sample says"
    assert not (tmp_path / "config").exists()
    assert sorted(p.name for p in (state / "clones").iterdir()) == ["3.wav"]  # no temporary left behind


def test_restore_never_overwrites_a_different_file(tmp_path):
    state, faces, voice = homes(tmp_path)
    put(faces / "db.npz", b"written since the reset")
    put(state / "voiceprints.json", b"{}")
    placed = A.restore_files({"faces/db.npz": b"exported rows", "state/voiceprints.json": b"{}"}, state, faces, voice)
    assert placed == A.Placed(restored=0, same=1, bundle=0, differ=("faces/db.npz",))
    assert (faces / "db.npz").read_bytes() == b"written since the reset"


@pytest.mark.parametrize(
    "name", ["faces/learned.npz", "state/people/dana.md", "state/clones/raw.wav", "voices/other.wav"]
)
def test_restore_writes_nothing_for_names_outside_the_allowlist(tmp_path, name):
    state, faces, voice = homes(tmp_path)
    assert A.restore_files({name: b"x"}, state, faces, voice) == A.Placed(0, 0, 1, ())
    assert not any(tmp_path.iterdir())
