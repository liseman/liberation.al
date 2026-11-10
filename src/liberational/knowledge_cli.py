"""Explicit export, verification, and import of owner knowledge (`knowledge.sqlite3` and the taught files).

`pack-knowledge export --source sqlite --output taught.zip` writes a bundle, `verify` checks one, and `import` loads
one into `knowledge.sqlite3`. None of them resets a catalog or restarts services. A portable import stores attachment
bytes under content hashes, then puts each taught file back where it lives when nothing is there yet
(`knowledge_attachments.restore_files`); it never overwrites a file, config TOML included, or writes outside the
allowlisted taught paths.

Every collection the knowledge store holds must be listed here (`_DOCUMENTS`, or one `validate` checks by shape): an
export refuses a store with any other, naming every unknown collection at once, rather than leave out what
the owner taught. Machine-made rows written into an owner collection stay out of the bundle (`_machine`). Owner state
kept in files is the allowlist in `knowledge_attachments.collect_files`; what is deliberately left out (the face
matcher's learned rows, place revisions and maps, hand-made measurements) is listed in docs/PEOPLE.md.
"""

import argparse
import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path

from liberational import config as cfg
from liberational import knowledge, knowledge_backup
from liberational.domain.kernel import Event
from liberational.knowledge_eventlog import record_of

_HASH = re.compile(r"[0-9a-f]{64}")
_DOCUMENTS = frozenset(
    {
        "identity.record",
        "pet.profile",
        "pet.sighting",
        "pet.photo",  # a labeled sighting's saved photo, to embed its reference again (people/pet_recognition.py)
        "zone.defined",
        "place.named",
        "place.plan",
        "activity.named",
        "activity.feedback",
        "device.link",
        "device.location",  # where the owner said a device lives (proximity/devices.py)
        "object.sighting",
        "journal.entry",
        "person.fact",
        "journal.page",
        "review.summary",
        "ask.question",
        "ask.state",
        "ask.answer",
        "voice.not",  # the owner's "this line isn't them" (listen/voices.deny)
        "face.not",  # the owner's "this face group isn't them" (people/face_nots.py)
        "pet.not",  # the owner's "this sighting isn't that pet" (people/pet_recognition.py)
        "face.ignore",  # pictures the owner ignored on Who's this? (people/face_hidden.py)
        "face.block",  # faces the owner blocked from Who's this? (people/face_hidden.py)
        "pet.block",  # sightings the owner blocked from Sightings to name (people/pet_recognition.py)
        "transcript.correction",  # the owner's rejections and TV marks of transcript lines (listen/revisions.py)
        "activity.kind",  # the activities the owner added or deleted (memory/activity_kinds.py)
        "camera.intrinsics",  # each camera's held focal and lens (spatial/lens.py)
    }
)


def _machine(record: knowledge.Record) -> bool:
    """A machine-made row in an owner collection, which an export leaves out: the audio tagger's TV marks."""
    return record.collection == "transcript.correction" and record.value.get("source") == "classifier"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def validate(bundle: knowledge_backup.Bundle) -> None:
    """Validate application records before any imported bytes or durable state are installed."""
    unsupported: set[str] = set()
    for record in bundle.records:
        if record.collection.startswith("event:"):
            event = Event.from_dict(record.value)
            if record_of(event) != record:
                raise ValueError("event envelope and indexed knowledge metadata disagree")
        elif record.collection == "forgotten.names":
            if not _HASH.fullmatch(record.id) or record.key != record.id or record.value or record.subjects:
                raise ValueError("invalid forgotten-name suppression record")
            if (record.start_ns, record.end_ns, record.recorded_ns) != (0, 0, 0):
                raise ValueError("suppression records do not carry observation clocks")
        elif record.collection in {"voice.reference", "face.reference", "pet.reference"}:
            _reference(record)
        elif record.collection == "knowledge.attachment":
            value = record.value
            if (
                set(value) != {"sha256", "size"}
                or not isinstance(value["sha256"], str)
                or not _HASH.fullmatch(value["sha256"])
            ):
                raise ValueError("invalid managed attachment reference")
            if type(value["size"]) is not int or not 0 <= value["size"] <= knowledge_backup.MAX_BLOB_BYTES:
                raise ValueError("invalid managed attachment size")
            name = _text(record.key, "attachment key")
            content = bundle.attachments.get(name)
            if content is None:
                raise ValueError("managed attachment reference lacks its bytes")
            if len(content) != value["size"] or hashlib.sha256(content).hexdigest() != value["sha256"]:
                raise ValueError("managed attachment metadata does not match its bytes")
        elif record.collection not in _DOCUMENTS and record.collection != "audit:eventlog":
            unsupported.add(record.collection)
    if unsupported:
        raise ValueError(f"unsupported knowledge collection: {', '.join(sorted(unsupported))}")


def _reference(record: knowledge.Record) -> None:
    value = record.value
    dims, vector = value.get("dims"), value.get("vector")
    if type(dims) is not int or not 0 < dims <= 65_536 or not isinstance(vector, list) or len(vector) != dims:
        raise ValueError("enrollment vector dimensions do not match")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector) or not any(vector):
        raise ValueError("enrollment vector must be finite and nonzero")
    _text(value.get("model"), "enrollment model")
    _text(value.get("source_id"), "enrollment source")
    if record.collection == "voice.reference":
        if type(value.get("voiceprint")) is not int or value["voiceprint"] <= 0:
            raise ValueError("invalid voiceprint reference")
        person = _text(value.get("person"), "voiceprint person")
        if person not in record.subjects or record.key != f"speaker:{value['voiceprint']}":
            raise ValueError("voiceprint reference has no matching person/print binding")
    elif record.collection == "face.reference":
        person = _text(value.get("person"), "face person")
        _text(value.get("name"), "face name")
        if dims != 512 or record.key != person or record.subjects != (person,):
            raise ValueError("face reference has no matching person binding")
    else:
        if record.key != _text(value.get("track_id"), "pet track"):
            raise ValueError("pet reference track mismatch")
        _text(value.get("species"), "pet species")
        _text(value.get("label"), "pet owner label")
        if not isinstance(value.get("owner_label"), dict):
            raise ValueError("pet reference lacks the owner label")


def _voice_reference() -> Path:
    return cfg.path("owner.voice_ref")


def _files(state: Path, faces: Path) -> dict[str, bytes]:
    """The taught files; settings and the owner's voice sample only when this install has them (a new one may not)."""
    from liberational import knowledge_attachments

    config, devices, reference = cfg.local_config(), cfg.devices_file(), _voice_reference()
    return knowledge_attachments.collect_files(
        state,
        faces,
        config_file=config if config.exists() else None,
        devices_file=devices if devices.exists() else None,
        voice_reference=reference if reference.exists() else None,
    )


def counts(bundle: knowledge_backup.Bundle) -> dict[str, int]:
    """How many records of each collection and taught files of each folder a bundle holds (no names)."""
    out: dict[str, int] = {}
    for record in bundle.records:
        out[record.collection] = out.get(record.collection, 0) + 1
    for name in bundle.attachments:
        folder = f"files:{name.split('/', 1)[0]}"
        out[folder] = out.get(folder, 0) + 1
    return dict(sorted(out.items()))


def export(source: str, state: Path, faces: Path, destination: Path) -> knowledge_backup.Bundle:
    """Prepare a complete explicit bundle; never create an empty SQLite source as an export side effect."""
    path = state / "knowledge.sqlite3"
    snapshot = knowledge.Store(path).snapshot() if path.exists() else None
    files = _files(state, faces)
    if source == "sqlite":
        if snapshot is None:
            raise FileNotFoundError("no SQLite knowledge exists to export")
        from liberational import knowledge_attachments

        managed = knowledge_attachments.exportattachments(knowledge.Store(path))
        files = {**managed, **files}
        records = tuple(item.record for item in snapshot.records if not _machine(item.record))
        bundle = knowledge_backup.Bundle(_current_attachments(records, files), snapshot.erased, files)
    else:
        raise ValueError("source must be sqlite")
    validate(bundle)
    knowledge_backup.write_bundle(bundle, destination)
    return bundle


def _current_attachments(
    records: tuple[knowledge.Record, ...], files: dict[str, bytes]
) -> tuple[knowledge.Record, ...]:
    latest = {record.key: record.id for record in records if record.collection == "knowledge.attachment"}
    return tuple(
        record
        for record in records
        if record.collection != "knowledge.attachment"
        or (
            latest[record.key] == record.id
            and record.key in files
            and hashlib.sha256(files[record.key]).hexdigest() == record.value.get("sha256")
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export")
    exporting.add_argument("--source", choices=("sqlite",), required=True)
    exporting.add_argument("--output", type=Path, required=True)
    exporting.add_argument("--faces", type=Path, default=None)
    importing = commands.add_parser("import")
    importing.add_argument("--input", type=Path, required=True)
    importing.add_argument("--faces", type=Path, default=None)
    checking = commands.add_parser("verify")
    checking.add_argument("--input", type=Path, required=True)
    for command in (exporting, importing):
        command.add_argument("--state", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "verify":
        bundle = knowledge_backup.read_bundle(args.input)
        validate(bundle)
        print(f"Verified {len(bundle.records)} records and {len(bundle.attachments)} attachments.")
    elif args.command == "export":
        bundle = export(
            args.source,
            args.state or cfg.path("paths.state"),
            args.faces or cfg.path("paths.faces"),
            args.output,
        )
        print(f"Owner knowledge exported: {len(bundle.records)} records and {len(bundle.attachments)} files.")
        for name, n in counts(bundle).items():
            print(f"  {name}: {n}")
    else:
        from liberational import knowledge_attachments

        bundle = knowledge_backup.read_bundle(args.input)
        validate(bundle)
        state = args.state or cfg.path("paths.state")
        imported = knowledge_attachments.import_bundle(knowledge.Store(state / "knowledge.sqlite3"), bundle)
        print(f"Imported {imported['records']} records and {imported['attachments']} managed attachments.")
        placed = knowledge_attachments.restore_files(
            bundle.attachments, state, args.faces or cfg.path("paths.faces"), _voice_reference()
        )
        print(
            f"Taught files: {placed.restored} put back, {placed.same} already there, {len(placed.differ)} left as "
            f"they are (a different file is there), {placed.bundle} kept only in the bundle (settings)."
        )
        for name in placed.differ:
            print(f"  differs, not overwritten: {name}")


if __name__ == "__main__":
    main()
