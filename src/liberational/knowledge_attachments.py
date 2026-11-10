"""Explicit taught-file collection and immutable managed blobs for owner-knowledge bundles.

Collection is a small file allowlist plus numeric clone files and confirmed person-reference entries. It never
walks raw recordings, unknown face clusters, pending previews, generated profiles, models, or environment files. Of
the face gallery it takes the owner's rows (`db.npz`, which reference photos also teach) and the owner's
negatives (`nots.npz`), not the rows the matcher learned from confident tracks (`learned.npz`), which are
machine-made and grow back. Each owner row's picture and source (`faces/examples/`, vision/face_examples.py) go with
the rows, so Training images still shows them after an import.
Managed import stores only content hashes on disk; logical names never become filesystem extraction paths. Putting
taught files back (`restore_files`) maps each allowlisted logical name to its one home under the given state, faces,
and voice-reference paths, writes only where nothing is, and leaves settings TOML in the bundle.
References and the bundle's knowledge records commit together. Failed imports may leave unreferenced immutable
blobs, which are deliberately retained because another concurrent import may already reference them.
"""

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from liberational import knowledge
from liberational import knowledge_backup as backup

COLLECTION = "knowledge.attachment"
_STATE_FILES = (
    "voiceprints.json",
    "voice_enrollments.json",
    "card_prefs.json",
    "people_undo.json",
    "contacts.json",
    "people_added.json",
)
_CLONE = re.compile(r"[0-9]+(?:\.(?:wav|txt|json)|\.checks\.json)")
_TOKEN = re.compile(r"[0-9a-f]{32}")
_HASH = re.compile(r"[0-9a-f]{64}")
_MEDIA = {"media.jpg", "media.wav", "media.mp4"}
_FACE_FILES = ("db.npz", "nots.npz")  # the owner's face rows and negatives; learned.npz is the matcher's own
_FACE_EXAMPLE = re.compile(r"[0-9a-f]{24}\.(?:json|jpg)")  # a face row's kept source and picture, by its key


def _name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("attachment name must be nonempty text")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or str(path) != value
        or any(part in {".", ".."} for part in path.parts)
        or "\\" in value
        or ":" in value
        or "\x00" in value
    ):
        raise ValueError("attachment name must be a portable relative path")
    return value


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read(path: Path) -> tuple[bytes, tuple[int, ...]]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("attachment must be a regular file without symlinks")
    if before.st_size > backup.MAX_BLOB_BYTES:
        raise ValueError("attachment exceeds individual byte limit")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError as error:
        raise ValueError("attachment disappeared before reading") from error
    with os.fdopen(descriptor, "rb") as source:
        if _signature(os.fstat(source.fileno())) != _signature(before):
            raise ValueError("attachment changed before reading")
        content = source.read(backup.MAX_BLOB_BYTES + 1)
        after = os.fstat(source.fileno())
    if len(content) > backup.MAX_BLOB_BYTES:
        raise ValueError("attachment exceeds individual byte limit")
    try:
        current = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("attachment disappeared while reading") from error
    if (
        len(content) != before.st_size
        or _signature(after) != _signature(before)
        or _signature(current) != _signature(before)
    ):
        raise ValueError("attachment changed while reading")
    return content, _signature(before)


class _Collector:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.watched: dict[Path, tuple[int, ...]] = {}
        self.read_bytes = 0

    def directory(self, path: Path) -> bool:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("attachment directory must not be a symlink or file")
        self.watched[path] = _signature(info)
        return True

    def children(self, path: Path) -> list[Path]:
        if not self.directory(path):
            return []
        found: list[Path] = []
        for child in path.iterdir():
            found.append(child)
            if len(found) > backup.MAX_ATTACHMENTS:
                raise ValueError("attachment directory exceeds entry limit")
        return sorted(found)

    def read(self, path: Path, *, optional: bool = False) -> bytes | None:
        try:
            content, signature = _read(path)
        except FileNotFoundError:
            if optional:
                return None
            raise
        self.read_bytes += len(content)
        if self.read_bytes > backup.MAX_TOTAL_BYTES:
            raise ValueError("attachment collection exceeds total byte limit")
        self.watched[path] = signature
        return content

    def add(self, name: str, path: Path, *, optional: bool = True) -> None:
        content = self.read(path, optional=optional)
        if content is not None:
            self.keep(name, content)

    def keep(self, name: str, content: bytes) -> None:
        _name(name)
        if name in self.files or len(self.files) >= backup.MAX_ATTACHMENTS:
            raise ValueError("duplicate attachment name or count exceeds limit")
        self.files[name] = content

    def finish(self) -> dict[str, bytes]:
        for path, signature in self.watched.items():
            try:
                current = path.lstat()
            except FileNotFoundError as error:
                raise ValueError("attachment collection disappeared during reading") from error
            if _signature(current) != signature:
                raise ValueError("attachment collection changed during reading")
        return self.files


def _metadata(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate reference metadata key")
            result[key] = value
        return result

    def number(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("nonfinite reference metadata")
        return value

    def constant(raw: str) -> None:
        raise ValueError(f"invalid reference metadata constant {raw}")

    try:
        value = json.loads(raw.decode(), object_pairs_hook=pairs, parse_float=number, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid person reference metadata") from error
    if not isinstance(value, dict):
        raise ValueError("person reference metadata must be an object")
    return value


def collect_files(
    state: Path,
    faces: Path,
    *,
    config_file: Path | None = None,
    devices_file: Path | None = None,
    voice_reference: Path | None = None,
) -> dict[str, bytes]:
    """Read only taught-file candidates; caller-specific enrollment validation remains mandatory."""
    collected = _Collector()
    if collected.directory(state):
        for filename in _STATE_FILES:
            collected.add(f"state/{filename}", state / filename)
        for path in collected.children(state / "clones"):
            if _CLONE.fullmatch(path.name):
                collected.add(f"state/clones/{path.name}", path, optional=False)
        for folder in collected.children(state / "person_references"):
            if not _TOKEN.fullmatch(folder.name):
                continue
            if not collected.directory(folder):
                raise ValueError("person reference directory disappeared")
            raw = collected.read(folder / "meta.json", optional=True)
            if raw is None:
                continue
            metadata = _metadata(raw)
            person = metadata.get("person")
            if person is None:
                continue  # Unconfirmed preview media is deliberately never opened.
            if (
                not isinstance(person, str)
                or not person.strip()
                or metadata.get("id") != folder.name
                or metadata.get("file") not in _MEDIA
            ):
                raise ValueError("attached person reference has invalid person, token, or media")
            collected.keep(f"state/person_references/{folder.name}/meta.json", raw)
            filename = metadata["file"]
            collected.add(f"state/person_references/{folder.name}/{filename}", folder / filename, optional=False)
    if collected.directory(faces):
        for filename in _FACE_FILES:
            collected.add(f"faces/{filename}", faces / filename)
        for path in collected.children(faces / "examples"):
            if _FACE_EXAMPLE.fullmatch(path.name):
                collected.add(f"faces/examples/{path.name}", path)
    for name, path in (("config/pack.toml", config_file), ("config/devices.toml", devices_file)):
        if path is not None:
            if path.suffix.lower() != ".toml":
                raise ValueError("explicit configuration attachment must be a TOML file")
            collected.add(name, path, optional=False)
    if voice_reference is not None:
        if voice_reference.suffix.lower() != ".wav":
            raise ValueError("explicit owner voice reference must be a WAV file")
        collected.add("voices/owner_ref.wav", voice_reference, optional=False)
        collected.add("voices/owner_ref.txt", voice_reference.with_suffix(".txt"))
    return collected.finish()


def _home(name: str, state: Path, faces: Path, voice_reference: Path) -> tuple[Path, Path] | None:
    """Where a collected logical name lives, as (root, file); None for settings and anything not collected."""
    parts = PurePosixPath(_name(name)).parts
    match parts:
        case ("state", filename) if filename in _STATE_FILES:
            return state, state / filename
        case ("state", "clones", filename) if _CLONE.fullmatch(filename):
            return state, state / "clones" / filename
        case ("state", "person_references", token, filename) if _TOKEN.fullmatch(token) and (
            filename == "meta.json" or filename in _MEDIA
        ):
            return state, state / "person_references" / token / filename
        case ("faces", filename) if filename in _FACE_FILES:
            return faces, faces / filename
        case ("faces", "examples", filename) if _FACE_EXAMPLE.fullmatch(filename):
            return faces, faces / "examples" / filename
        case ("voices", "owner_ref.wav"):
            return voice_reference.parent, voice_reference
        case ("voices", "owner_ref.txt"):
            return voice_reference.parent, voice_reference.with_suffix(".txt")
    return None


@dataclass(frozen=True)
class Placed:
    """What `restore_files` did: files put back, already identical, left only in the bundle, and differing names."""

    restored: int
    same: int
    bundle: int
    differ: tuple[str, ...]


def restore_files(files: Mapping[str, bytes], state: Path, faces: Path, voice_reference: Path) -> Placed:
    """Put each taught file back at its home when nothing is there (an import after a fresh reset or on a new install).

    An identical file is left alone; a different one is never overwritten, only named, so a service that wrote its own
    file first loses nothing and the owner decides. Settings (`config/`) stay in the bundle. Each write is a whole file
    linked into place, so a crash never leaves half of one, and a home that resolves outside its root is refused."""
    restored = same = bundle = 0
    differ: list[str] = []
    for name, content in sorted(files.items()):
        home = _home(name, state, faces, voice_reference)
        if home is None:
            bundle += 1
            continue
        root, target = home
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.resolve().is_relative_to(root.resolve()):
            raise ValueError("taught file home resolves outside its root")
        if os.path.lexists(target):
            existing, _ = _read(target)
            if existing == content:
                same += 1
            else:
                differ.append(name)
            continue
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                differ.append(name)  # written meanwhile by someone else: theirs stays
                continue
            restored += 1
        finally:
            os.unlink(temporary)
        _sync_directory(target.parent)
    return Placed(restored, same, bundle, tuple(differ))


def _reference(name: str, digest: str, size: int) -> knowledge.Record:
    ident = hashlib.sha256(
        b"liberational.knowledge.attachment.v1\0" + name.encode() + b"\0" + digest.encode()
    ).hexdigest()
    return knowledge.Record(ident, COLLECTION, name, 0, 0, 0, {"sha256": digest, "size": size})


def _blob_root(store: knowledge.Store, *, create: bool = False) -> Path:
    root = store.path.parent / "knowledge_blobs"
    if create:
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("managed blob directory must not be a symlink or file")
    return root


def _verify(path: Path, digest: str, size: int) -> bytes:
    content, _ = _read(path)
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("managed attachment blob hash or size mismatch")
    return content


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install(root: Path, digest: str, content: bytes) -> bool:
    target = root / digest
    if os.path.lexists(target):
        _verify(target, digest, len(content))
        return False
    descriptor, temporary = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            _verify(target, digest, len(content))
            return False
        return True
    finally:
        os.unlink(temporary)


def import_bundle(store: knowledge.Store, bundle: backup.Bundle) -> dict[str, int]:
    """Install immutable blobs, then atomically commit logical references with all records and erasures."""
    if len(bundle.attachments) > backup.MAX_ATTACHMENTS:
        raise ValueError("attachment count exceeds limit")
    refs: list[knowledge.Record] = []
    blobs: dict[str, bytes] = {}
    total = 0
    for name, content in sorted(bundle.attachments.items()):
        _name(name)
        if not isinstance(content, bytes) or len(content) > backup.MAX_BLOB_BYTES:
            raise ValueError("attachment must contain immutable bytes within the individual limit")
        total += len(content)
        if total > backup.MAX_TOTAL_BYTES:
            raise ValueError("attachment total exceeds byte limit")
        digest = hashlib.sha256(content).hexdigest()
        refs.append(_reference(name, digest, len(content)))
        blobs[digest] = content
    records = {row.id: row for row in bundle.records}
    if len(records) != len(bundle.records):
        raise ValueError("duplicate bundle record identities")
    for ref in refs:
        if ref.id in records and records[ref.id] != ref:
            raise knowledge.Conflict("attachment identity conflicts with an imported record")
        records[ref.id] = ref
    installed = 0
    if blobs:
        root = _blob_root(store, create=True)
        for digest, content in blobs.items():
            installed += _install(root, digest, content)
        _sync_directory(root)
        _sync_directory(root.parent)
    store.restore(records.values(), bundle.erased)
    return {"records": len(bundle.records), "attachments": len(refs), "blobs": installed}


def read_attachment(store: knowledge.Store, logicalname: str) -> bytes | None:
    """Read the latest immutable logical reference; missing or corrupt bytes never count as an empty file."""
    name = _name(logicalname)
    rows = store.query(COLLECTION, key=name)
    if not rows:
        return None
    ref = rows[-1].record
    value = ref.value
    digest, size = value.get("sha256"), value.get("size")
    if (
        set(value) != {"sha256", "size"}
        or not isinstance(digest, str)
        or _HASH.fullmatch(digest) is None
        or type(size) is not int
        or not 0 <= size <= backup.MAX_BLOB_BYTES
        or ref != _reference(name, digest, size)
    ):
        raise ValueError("invalid managed attachment reference")
    return _verify(_blob_root(store) / digest, digest, size)


def exportattachments(store: knowledge.Store) -> dict[str, bytes]:
    """Return the currently visible logical attachment values, never scan the managed blob directory."""
    revision = store.revision((COLLECTION,))
    names = {row.record.key for row in store.query(COLLECTION)}
    if None in names or len(names) > backup.MAX_ATTACHMENTS:
        raise ValueError("invalid attachment keys or count exceeds limit")
    result: dict[str, bytes] = {}
    total = 0
    for name in sorted(name for name in names if name is not None):
        content = read_attachment(store, name)
        if content is None:
            raise ValueError("managed attachment reference disappeared during export")
        total += len(content)
        if total > backup.MAX_TOTAL_BYTES:
            raise ValueError("attachment export exceeds total byte limit")
        result[name] = content
    if store.revision((COLLECTION,)) != revision:
        raise ValueError("managed attachments changed during export")
    return result
