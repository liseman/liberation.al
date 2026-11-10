"""Portable knowledge bundles validate completely before atomic database restoration."""

import dataclasses
import hashlib
import json
import stat
import zipfile

import pytest
from hypothesis import strategies as st

from liberational import knowledge
from liberational import knowledge_backup as B


def record(id="event-1", *, value=None):
    return knowledge.Record(
        id, "events:people", "Dana", 10, 20, 30, {"name": "Dana"} if value is None else value, ("person:dana",)
    )


def manifest(*, records=(), erased=(), attachments=()):
    return {
        "version": 1,
        "records": [dataclasses.asdict(row) for row in records],
        "erased": list(erased),
        "attachments": list(attachments),
    }


def write_bundle(path, value, blobs=(), *, raw=None, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.writestr("manifest.json", json.dumps(value) if raw is None else raw)
        for name, content in blobs:
            archive.writestr(name, content)
    return path


def test_import_cannot_resurrect_an_id_erased_by_the_destination(tmp_path):
    path = write_bundle(tmp_path / "bundle.zip", manifest(records=[record("fresh"), record("deleted")]))
    target = knowledge.Store(tmp_path / "target.sqlite3")
    target.put([record("deleted")])
    target.remove(["deleted"])
    before = target.snapshot()
    with pytest.raises(knowledge.Deleted):
        B.import_bundle(target, path)
    assert target.snapshot() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("version", True),
        ("records", {}),
        ("erased", "event-1"),
        ("attachments", {}),
        ("unknown", None),
    ],
)
def test_unsupported_manifest_shape_never_changes_the_store(tmp_path, field, value):
    data = manifest(records=[record()])
    data[field] = value
    path = write_bundle(tmp_path / "bundle.zip", data)
    target = knowledge.Store(tmp_path / "target.sqlite3")
    target.put([record("sentinel")])
    before = target.snapshot()
    with pytest.raises(ValueError):
        B.import_bundle(target, path)
    assert target.snapshot() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", ""),
        ("collection", " "),
        ("key", 1),
        ("start_ns", True),
        ("end_ns", 9),
        ("recorded_ns", -1),
        ("value", []),
        ("subjects", ["a", "a"]),
        ("subjects", [None]),
        ("extra", 1),
    ],
)
def test_malformed_later_record_is_rejected_before_any_restore_call(tmp_path, monkeypatch, field, value):
    data = manifest(records=[record("valid-first"), record("bad-second")])
    data["records"][1][field] = value
    path = write_bundle(tmp_path / "bundle.zip", data)
    target = knowledge.Store(tmp_path / "target.sqlite3")

    def must_not_write(*args, **kwargs):
        pytest.fail("malformed bundle reached Store.restore")

    monkeypatch.setattr(target, "restore", must_not_write)
    with pytest.raises(ValueError):
        B.import_bundle(target, path)


@pytest.mark.parametrize(
    "raw",
    [
        '{"version":1,"version":1,"records":[],"erased":[],"attachments":[]}',
        '{"version":1,"records":[],"erased":[],"attachments":[],"x":NaN}',
        '{"version":1,"records":[],"erased":[],"attachments":[],"x":1e9999}',
        '{"version":1,"records":[],"erased":[],"attachments":[],"x":Infinity}',
        b"\xff",
    ],
)
def test_ambiguous_or_nonfinite_json_is_rejected(tmp_path, raw):
    path = write_bundle(tmp_path / "bundle.zip", {}, raw=raw)
    with pytest.raises(ValueError):
        B.read_bundle(path)


@pytest.mark.parametrize("kind", ["duplicate-record", "duplicate-erasure", "erased-record"])
def test_record_and_erasure_identities_must_be_unambiguous(tmp_path, kind):
    data = {
        "duplicate-record": manifest(records=[record(), record()]),
        "duplicate-erasure": manifest(erased=["gone", "gone"]),
        "erased-record": manifest(records=[record()], erased=["event-1"]),
    }[kind]
    with pytest.raises(ValueError):
        B.read_bundle(write_bundle(tmp_path / "bundle.zip", data))


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a/../escape", "a\\escape", "C:/escape", "a//b", "a/./b", "."]
)
def test_attachment_names_cannot_escape_a_portable_logical_namespace(tmp_path, name):
    digest = hashlib.sha256(b"x").hexdigest()
    data = manifest(attachments=[{"name": name, "sha256": digest, "size": 1}])
    with pytest.raises(ValueError):
        B.read_bundle(write_bundle(tmp_path / "bundle.zip", data, [(f"blobs/{digest}", b"x")]))


@pytest.mark.parametrize("kind", ["hash", "size", "missing", "extra", "duplicate-ref", "invalid-hash"])
def test_every_blob_must_match_its_exact_manifest_reference(tmp_path, kind):
    digest = hashlib.sha256(b"x").hexdigest()
    ref = {"name": "photo.bin", "sha256": digest, "size": 1}
    blobs = [(f"blobs/{digest}", b"y" if kind == "hash" else b"x")]
    data = manifest(attachments=[ref])
    if kind == "size":
        ref["size"] = 2
    elif kind == "missing":
        blobs = []
    elif kind == "extra":
        blobs.append((f"blobs/{'a' * 64}", b"unused"))
    elif kind == "duplicate-ref":
        data["attachments"].append(dict(ref))
    elif kind == "invalid-hash":
        ref["sha256"] = "../outside"
    with pytest.raises(ValueError, match="SHA256" if kind == "hash" else ".*"):
        B.read_bundle(write_bundle(tmp_path / "bundle.zip", data, blobs))


def test_composed_bundle_can_be_published_without_a_staging_database(tmp_path):
    original = B.Bundle((record("second-id"), record("first-id")), ("deleted",), {"photo.bin": b"image"})
    B.write_bundle(original, tmp_path / "bundle.zip")
    assert B.read_bundle(tmp_path / "bundle.zip") == original
    assert list(tmp_path.iterdir()) == [tmp_path / "bundle.zip"]


def test_composed_bundle_is_fully_validated_before_publication(tmp_path):
    invalid = B.Bundle((record(), record()), (), {"photo.bin": b"image"})
    with pytest.raises(ValueError, match="identities"):
        B.write_bundle(invalid, tmp_path / "bundle.zip")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [{1: "integer key"}, {"bad": float("nan")}])
def test_composed_bundle_does_not_coerce_invalid_json_values(tmp_path, value):
    with pytest.raises(ValueError, match="JSON"):
        B.write_bundle(B.Bundle((record(value=value),), (), {}), tmp_path / "bundle.zip")
    assert list(tmp_path.iterdir()) == []


def test_deflated_blob_is_validated_without_extracting_it(tmp_path):
    content = b"synthetic attachment" * 100
    digest = hashlib.sha256(content).hexdigest()
    path = write_bundle(
        tmp_path / "bundle.zip",
        manifest(attachments=[{"name": "image.bin", "sha256": digest, "size": len(content)}]),
        [(f"blobs/{digest}", content)],
        compression=zipfile.ZIP_DEFLATED,
    )
    assert dict(B.read_bundle(path).attachments) == {"image.bin": content}
    assert not (tmp_path / "image.bin").exists()


@pytest.mark.parametrize("name", ["../manifest.json", "/outside", "blobs/../outside", "extra.txt", "blobs/"])
def test_archive_members_are_never_extracted_and_unknown_paths_are_refused(tmp_path, name):
    path = write_bundle(tmp_path / "bundle.zip", manifest(), [(name, b"bad")])
    with pytest.raises(ValueError):
        B.read_bundle(path)
    assert not (tmp_path.parent / "manifest.json").exists()


def test_duplicate_archive_member_is_not_silently_selected(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        path = write_bundle(tmp_path / "bundle.zip", manifest(), [("manifest.json", b"{}")])
    with pytest.raises(ValueError, match="duplicate"):
        B.read_bundle(path)


def test_zip_symlink_members_and_symlink_containers_are_rejected(tmp_path):
    path = tmp_path / "linked.zip"
    info = zipfile.ZipInfo(f"blobs/{'a' * 64}")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest()))
        archive.writestr(info, "../../outside")
    with pytest.raises(ValueError):
        B.read_bundle(path)
    link = tmp_path / "bundle-link.zip"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular file"):
        B.read_bundle(link)


def test_compressed_members_are_bounded_by_uncompressed_size_before_read(tmp_path, monkeypatch):
    content = b"x" * 4096
    digest = hashlib.sha256(content).hexdigest()
    path = write_bundle(
        tmp_path / "bundle.zip",
        manifest(attachments=[{"name": "blob", "sha256": digest, "size": len(content)}]),
        [(f"blobs/{digest}", content)],
        compression=zipfile.ZIP_DEFLATED,
    )
    monkeypatch.setattr(B, "MAX_BLOB_BYTES", 32)
    with pytest.raises(ValueError, match="limit"):
        B.read_bundle(path)


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: st.lists(children, max_size=4) | st.dictionaries(st.text(), children, max_size=4),
    max_leaves=12,
)
