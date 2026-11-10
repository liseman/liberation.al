"""Portable, bounded owner-knowledge ZIP bundles with explicit optional attachments.

The version-1 manifest contains full records in their original append order, erased IDs, and named SHA256 blob
references. Native broker payloads and arbitrary state directories are never scanned. Import validates every
member before one atomic Store.restore call; returned attachment bytes are for the caller to install explicitly.

Export fsyncs a private sibling file and publishes it with an exclusive hard link, never replacing a destination.
Filesystems without hard links fail rather than weakening publication. POSIX also fsyncs the containing directory;
Windows has no portable directory-fsync operation, so only the complete container's file data is explicitly synced.
"""

import dataclasses
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from liberational import knowledge

VERSION = 1
# Sized for a real home's taught state (2026-11-03: the first live export passed 32 MiB of manifest).
MAX_RECORDS = 2_000_000
MAX_ATTACHMENTS = 4096
MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MAX_BLOB_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024 * 1024
_HASH = re.compile(r"[0-9a-f]{64}")
_RECORD_FIELDS = {field.name for field in dataclasses.fields(knowledge.Record)}


@dataclass(frozen=True)
class Bundle:
    records: tuple[knowledge.Record, ...]
    erased: tuple[str, ...]
    attachments: Mapping[str, bytes]


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("bundle text must be nonempty")
    return value


def _name(value: object) -> str:
    name = _text(value)
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or str(path) != name
        or any(part in {".", ".."} for part in path.parts)
        or "\\" in name
        or ":" in name
        or "\x00" in name
    ):
        raise ValueError("attachment name must be a portable relative path without traversal")
    return name


def _integer(value: object) -> int:
    if type(value) is not int or not 0 <= value <= knowledge.MAX_TIME_NS:
        raise ValueError("bundle clock must be a nonnegative SQLite integer")
    return value


def _encode(value: Any, maximum: int | None = None) -> bytes:
    def keys(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("bundle JSON object keys must be strings")
            for child in item.values():
                keys(child)
        elif isinstance(item, list | tuple):
            for child in item:
                keys(child)

    try:
        keys(value)
        encoded = bytearray()
        for chunk in json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":")).iterencode(value):
            part = chunk.encode()
            if maximum is not None and len(encoded) + len(part) > maximum:
                raise ValueError("bundle JSON exceeds byte limit")
            encoded.extend(part)
        return bytes(encoded)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError(f"bundle contains invalid JSON: {error}") from error


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("bundle JSON contains duplicate keys")
        result[key] = value
    return result


def _float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("bundle JSON contains a nonfinite number")
    return value


def _constant(raw: str) -> None:
    raise ValueError(f"bundle JSON contains forbidden constant {raw}")


def _record(value: object) -> knowledge.Record:
    if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
        raise ValueError("bundle record has unsupported fields")
    key = value["key"]
    if key is not None:
        key = _text(key)
    subjects = value["subjects"]
    if not isinstance(subjects, list) or any(
        not isinstance(subject, str) or not subject.strip() for subject in subjects
    ):
        raise ValueError("bundle record subjects must be nonempty strings")
    if len(set(subjects)) != len(subjects):
        raise ValueError("bundle record has duplicate subjects")
    start, end, recorded = (_integer(value[name]) for name in ("start_ns", "end_ns", "recorded_ns"))
    body = value["value"]
    if end < start or not isinstance(body, dict):
        raise ValueError("bundle record has invalid span or value")
    _encode(body, knowledge.MAX_JSON_BYTES)
    return knowledge.Record(
        _text(value["id"]), _text(value["collection"]), key, start, end, recorded, body, tuple(subjects)
    )


def _manifest(raw: bytes) -> tuple[tuple[knowledge.Record, ...], tuple[str, ...], dict[str, tuple[str, int]]]:
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("bundle manifest exceeds byte limit")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_float=_float, parse_constant=_constant)
    except (UnicodeError, RecursionError, ValueError) as error:
        raise ValueError(f"invalid bundle JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != {"version", "records", "erased", "attachments"}:
        raise ValueError("unsupported bundle manifest fields")
    if type(value["version"]) is not int or value["version"] != VERSION:
        raise ValueError("unsupported bundle version")
    rows, erased, refs = (value[name] for name in ("records", "erased", "attachments"))
    if not isinstance(rows, list) or not isinstance(erased, list) or len(rows) + len(erased) > MAX_RECORDS:
        raise ValueError("bundle record count exceeds limit or has invalid shape")
    records = tuple(_record(row) for row in rows)
    deleted = tuple(_text(id) for id in erased)
    ids = {record.id for record in records}
    if len(ids) != len(records) or len(set(deleted)) != len(deleted) or ids.intersection(deleted):
        raise ValueError("bundle record/erased identities overlap or repeat")
    if not isinstance(refs, list) or len(refs) > MAX_ATTACHMENTS:
        raise ValueError("bundle attachment count exceeds limit or has invalid shape")
    attachments: dict[str, tuple[str, int]] = {}
    sizes: dict[str, int] = {}
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"name", "sha256", "size"}:
            raise ValueError("invalid bundle attachment reference")
        name, digest, size = _name(ref["name"]), ref["sha256"], ref["size"]
        if not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
            raise ValueError("invalid bundle attachment hash")
        if type(size) is not int or not 0 <= size <= MAX_BLOB_BYTES:
            raise ValueError("bundle attachment exceeds individual byte limit")
        if name in attachments or (digest in sizes and sizes[digest] != size):
            raise ValueError("duplicate attachment name or contradictory blob size")
        attachments[name], sizes[digest] = (digest, size), size
    if len(raw) + sum(size for _, size in attachments.values()) > MAX_TOTAL_BYTES:
        raise ValueError("bundle exceeds total byte limit")
    return records, deleted, attachments


def _read_regular(path: Path, maximum: int) -> bytes:
    # Never follow a final symlink. Comparing the opened inode also detects replacement between lstat and open
    # on platforms without O_NOFOLLOW. Only the caller's explicitly supplied attachment files are opened.
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("bundle input must be a bounded regular file without symlinks")
    if before.st_size > maximum:
        raise ValueError("bundle input exceeds individual byte limit")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("bundle input changed while opening")
        content = source.read(maximum + 1)
    if len(content) > maximum:
        raise ValueError("bundle input exceeds byte limit")
    return content


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bundle(bundle: Bundle, destination: str | Path) -> None:
    """Validate and atomically publish an explicitly composed bundle without a staging database."""
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"bundle destination already exists: {destination}")
    if len(bundle.records) + len(bundle.erased) > MAX_RECORDS:
        raise ValueError("bundle record count exceeds limit")
    if len(bundle.attachments) > MAX_ATTACHMENTS:
        raise ValueError("bundle attachment count exceeds limit")
    refs: list[dict[str, Any]] = []
    blobs: dict[str, bytes] = {}
    total = 0
    for name, content in sorted(bundle.attachments.items()):
        _name(name)
        if not isinstance(content, bytes):
            raise ValueError("bundle attachments must contain immutable bytes")
        if len(content) > MAX_BLOB_BYTES:
            raise ValueError("bundle attachment exceeds individual byte limit")
        total += len(content)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("bundle exceeds total byte limit")
        digest = hashlib.sha256(content).hexdigest()
        refs.append({"name": name, "sha256": digest, "size": len(content)})
        blobs[digest] = content
    manifest = _encode(
        {
            "version": VERSION,
            "records": [dataclasses.asdict(row) for row in bundle.records],
            "erased": list(bundle.erased),
            "attachments": refs,
        },
        MAX_MANIFEST_BYTES,
    )
    _manifest(manifest)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("manifest.json", manifest)
                for digest, content in sorted(blobs.items()):
                    archive.writestr(f"blobs/{digest}", content)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, destination)
        os.unlink(temporary)
        _sync_directory(destination.parent)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def read_bundle(source: str | Path) -> Bundle:
    """Validate the complete container without extracting or writing any attachment path."""
    # ZIP headers and directory metadata are separately bounded in addition to logical uncompressed bytes.
    raw = _read_regular(Path(source), MAX_TOTAL_BYTES + 8 * 1024 * 1024)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if not 1 <= len(infos) <= MAX_ATTACHMENTS + 1:
                raise ValueError("bundle member count exceeds limit")
            members: dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                mode = info.external_attr >> 16
                if (
                    info.filename != info.orig_filename
                    or info.filename in members
                    or info.is_dir()
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                    or info.flag_bits & 1
                    or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                ):
                    raise ValueError("bundle contains duplicate, linked, encrypted, or unsupported members")
                if info.filename != "manifest.json" and not (
                    info.filename.startswith("blobs/") and _HASH.fullmatch(info.filename[6:])
                ):
                    raise ValueError("bundle member path is not allowed")
                maximum = MAX_MANIFEST_BYTES if info.filename == "manifest.json" else MAX_BLOB_BYTES
                if not 0 <= info.file_size <= maximum:
                    raise ValueError("bundle member exceeds individual byte limit")
                members[info.filename] = info
            if "manifest.json" not in members:
                raise ValueError("bundle manifest is missing")
            if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
                raise ValueError("bundle exceeds total byte limit")
            manifest = archive.read(members["manifest.json"])
            records, erased, refs = _manifest(manifest)
            hashes = {digest for digest, _ in refs.values()}
            if set(members) != {"manifest.json", *(f"blobs/{digest}" for digest in hashes)}:
                raise ValueError("bundle has missing or unreferenced blobs")
            if any(members[f"blobs/{digest}"].file_size != size for digest, size in refs.values()):
                raise ValueError("bundle attachment size mismatch")
            content: dict[str, bytes] = {}
            for digest in hashes:
                blob = archive.read(members[f"blobs/{digest}"])
                if hashlib.sha256(blob).hexdigest() != digest:
                    raise ValueError("bundle blob SHA256 mismatch")
                content[digest] = blob
            if any(len(content[digest]) != size for digest, size in refs.values()):
                raise ValueError("bundle attachment size mismatch")
            return Bundle(
                records, erased, MappingProxyType({name: content[digest] for name, (digest, _) in refs.items()})
            )
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, RecursionError) as error:
        raise ValueError(f"invalid knowledge bundle: {error}") from error


def import_bundle(store: knowledge.Store, source: str | Path) -> Bundle:
    """Atomically restore validated records/tombstones; return attachments without installing them."""
    bundle = read_bundle(source)
    store.restore(bundle.records, bundle.erased)
    return bundle
