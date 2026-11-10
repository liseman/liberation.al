"""Rename cameras (#3277): `pack-rename-camera OLD=NEW [OLD=NEW ...] [--dry-run]`, with the services stopped.

A camera's name is its id everywhere: the device manifest, Continuum's streams, the segment store's slots, places,
zones, settings, and the owner's labels. This moves what the owner taught and the machine state that follows the
segment store to the new names, and prints the config edits to make by hand (it never writes config). It is
idempotent: a second run finds nothing under the old names and changes nothing. `--dry-run` says what it would do
and writes nothing, and may run while the services are up; the real run refuses while a segment writer or see-zones
holds a camera it would move.

What carries across, migrated in place:

- Owner-taught, in the knowledge store (`knowledge.sqlite3`): every record outside the domain event log with a
  camera at a `cam`, `camera`, `cams`, or `cameras` field, or keyed by one (`zone.defined`'s key is its camera;
  `pet.sighting`, `pet.reference`, `pet.not`, `activity.feedback`, and `identity.record`'s faces carry one). Each is
  rewritten in place (`knowledge.Store.amend`), keeping its id, so whatever refers to it still does.
- Owner-taught: owner plans' camera spots (`x-camera <name>` in every `place.plan` record, so restoring an earlier
  plan brings the new names too), and the Reports tab's layout (a walk-by card per camera, `walkbys:<name>`, and
  the walk-by card's camera pins, in `reports.sqlite3`).
- Machine-made, moved because a rename is cheap and keeps the history whole: the segment store's camera slots
  (`segments/<stream>/<name>/`, whose rows take their camera from the slot), and the state that follows those slots
  or the cameras' registrations: place revisions (`places/<place>/<revision>.json`'s `cameras` and `signals`, and
  `x-camera` in their `.plan` files), `places/footprints.json` and `seats.json`, `see-zones.json`'s cursors and
  pipelines, the vector worker's `continuum/vector/segments.<stream>.json` cursors, `screens_on.json`, and
  `continuum/capture_paused.json`. Cached report days that name an old camera are dropped and recompute.

Printed for the owner to edit (`config_edits`): `config/devices.toml`'s `[[device]] name`, and every setting in the
effective config (defaults.toml and pack.toml) keyed by a camera, listing one, or naming one (`camera_heights_m`,
`actions_camera_gpus`, `actions_camera_fps`, `caption_every_s`, `vision.places`, and any added later). A camera's
lens and masks sit under its `[[device]]`, so they move with the name.

Left under the old names to age out: Continuum's broker streams (`<name>.video` and the rest, in the catalog and
the tier). Recordings stay readable as cameras of their old names until the 30-day video window purges them; a
pre-rename segment row looked up under its new name finds no video meanwhile. The same goes for text and caches
that mention a camera: journal evidence clips, video thumbnails, research jobs, summaries, and the learned face
rows' track sources.
"""

import argparse
import fcntl
import json
import os
import re
import sqlite3
import sys
import tempfile
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from liberational import config as cfg
from liberational import knowledge, locking

# A camera name as the segment store takes it (continuum-segments' check_name).
NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")
CAMERA_KEYS = frozenset({"cam", "camera", "cams", "cameras"})
# Knowledge collections this leaves alone: the domain event log replays from its envelopes (knowledge_cli.validate).
SKIP_PREFIXES = ("event:", "audit:")
PLAN_COLLECTION = "place.plan"
# Machine state files under `paths.state` keyed by camera, beside the places and vector cursors found by glob.
STATE_FILES = ("see-zones.json", "screens_on.json", "continuum/capture_paused.json")
PLACE_FILES = ("footprints.json", "seats.json")
_X_CAMERA = re.compile(r"^([ \t]*x-camera[ \t]+)(\S+)", re.MULTILINE)


def parse_names(pairs: Sequence[str]) -> dict[str, str]:
    """`OLD=NEW` pairs as {old: new}. Raises ValueError for a malformed name, a camera renamed twice, two cameras
    given one name, or a new name that is also an old one (a chain or a swap, which a re-run couldn't tell apart from
    one half done)."""
    names: dict[str, str] = {}
    for pair in pairs:
        old, sep, new = pair.partition("=")
        if not sep or not NAME.fullmatch(old) or not NAME.fullmatch(new):
            raise ValueError(f"{pair!r}: want OLD=NEW, each letters, digits, '.', '_', or '-'")
        if old == new or old in names:
            raise ValueError(f"{pair!r}: rename a camera once, to another name")
        names[old] = new
    if len(set(names.values())) != len(names):
        raise ValueError("two cameras can't take one name")
    if set(names) & set(names.values()):
        raise ValueError("a new name can't also be an old one; rename in two runs")
    if not names:
        raise ValueError("name at least one OLD=NEW")
    return names


def rename_json(value: Any, names: Mapping[str, str]) -> Any:
    """`value` with every camera renamed: dict keys that are an old name, and the text (or each text of a list) at a
    `cam`, `camera`, `cams`, or `cameras` field."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k in CAMERA_KEYS and isinstance(v, str):
                v = names.get(v, v)
            elif k in CAMERA_KEYS and isinstance(v, list):
                v = [names.get(x, x) if isinstance(x, str) else rename_json(x, names) for x in v]
            else:
                v = rename_json(v, names)
            out[names.get(k, k)] = v
        return out
    if isinstance(value, list):
        return [rename_json(x, names) for x in value]
    return value


def rename_plan(text: str, names: Mapping[str, str]) -> str:
    """A plan's DSL with each `x-camera <name>` spot's camera renamed (spatial/plan_import.py)."""
    return _X_CAMERA.sub(lambda m: m.group(1) + names.get(m.group(2), m.group(2)), text)


def rename_setting(value: Any, names: Mapping[str, str]) -> Any:
    """A setting with its cameras renamed: a table keyed by camera, a list of names or of `name: display name`
    entries (`vision.places`), or a name."""
    if isinstance(value, dict):
        return {names.get(k, k): rename_setting(v, names) for k, v in value.items()}
    if isinstance(value, list):
        return [rename_setting(x, names) for x in value]
    if isinstance(value, str):
        head, sep, rest = value.partition(":")
        if sep and head.strip() in names:
            return names[head.strip()] + sep + rest
        return names.get(value, value)
    return value


@dataclass
class Result:
    """What a run did, or would do: one line per change, and the conflicts that stop a real run."""

    changes: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _write_json(path: Path, value: Any) -> None:
    """`path` replaced whole, so a reader sees the old file or the new one."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, separators=(",", ":"))
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _renamed_record(record: knowledge.Record, names: Mapping[str, str]) -> knowledge.Record:
    value = rename_json(record.value, names)
    if record.collection == PLAN_COLLECTION and isinstance(value.get("plan"), str):
        value = {**value, "plan": rename_plan(value["plan"], names)}
    key = names.get(record.key, record.key) if record.key is not None else None
    return replace(record, key=key, value=value)


def _knowledge(state: Path) -> knowledge.Store | None:
    path = state / "knowledge.sqlite3"
    return knowledge.at(path) if path.exists() else None


def rename_knowledge(state: Path, names: Mapping[str, str], dry: bool, result: Result) -> None:
    """Every owner record naming an old camera, rewritten in place (one transaction per collection)."""
    store = _knowledge(state)
    if store is None:
        return
    for collection in store.collections():
        rows = store.query(collection)
        if collection.startswith(SKIP_PREFIXES):
            named = sum(1 for r in rows if any(f'"{old}"' in json.dumps(r.record.value) for old in names))
            if named:
                result.warnings.append(f"{collection}: {named} event records mention an old camera; left as they are")
            continue
        changed = [new for r in rows if (new := _renamed_record(r.record, names)) != r.record]
        if changed:
            result.changes.append(f"knowledge {collection}: {len(changed)} records")
            if not dry:
                store.amend(changed)


def rename_reports(state: Path, names: Mapping[str, str], dry: bool, result: Result) -> None:
    """The Reports tab's per-camera walk-by cards and pins, and the cached report days that name an old camera."""
    path = state / "reports.sqlite3"
    if not path.exists():
        return
    db = sqlite3.connect(path)
    try:
        with db:
            row = db.execute("SELECT value FROM meta WHERE key='layout'").fetchone()
            if row is not None:
                layout = json.loads(row[0])
                renamed = {**layout, "reports": [_renamed_report(r, names) for r in layout.get("reports", [])]}
                if renamed != layout:
                    result.changes.append("reports layout: per-camera walk-by cards and pins")
                    if not dry:
                        db.execute("UPDATE meta SET value=? WHERE key='layout'", (json.dumps(renamed),))
            where = " OR ".join("body LIKE ?" for _ in names)
            likes = [f'%"{old}"%' for old in names]
            stale = db.execute(f"SELECT COUNT(*) FROM days WHERE {where}", likes).fetchone()[0]
            if stale:
                result.changes.append(f"reports: {stale} cached days dropped, to recompute")
                if not dry:
                    db.execute(f"DELETE FROM days WHERE {where}", likes)
    except sqlite3.OperationalError as error:  # an older or empty store has no such table
        result.warnings.append(f"reports.sqlite3: {error}")
    finally:
        db.close()


def _renamed_report(report: Any, names: Mapping[str, str]) -> Any:
    if not isinstance(report, dict):
        return report
    out = rename_json(report, names)
    arg = out.get("arg")
    if out.get("kind") == "walkbys" and isinstance(arg, str) and arg in names:
        new = names[arg]
        out["arg"] = new
        if out.get("id") == f"walkbys:{arg}":
            out["id"] = f"walkbys:{new}"
        if isinstance(out.get("title"), str):
            out["title"] = re.sub(rf"(?<![\w.-]){re.escape(arg)}(?![\w.-])", new, out["title"])
    return out


def _json_file(path: Path, names: Mapping[str, str], dry: bool, result: Result, label: str) -> None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        result.warnings.append(f"{label}: unreadable, left alone ({error})")
        return
    renamed = rename_json(value, names)
    if renamed != value:
        result.changes.append(label)
        if not dry:
            _write_json(path, renamed)


def rename_places(state: Path, names: Mapping[str, str], dry: bool, result: Result) -> None:
    """Place revisions' cameras and signals, their plans' camera spots, and the per-camera place marks."""
    root = state / "places"
    if not root.is_dir():
        return
    with locking.locked(root):
        revisions = plans = 0
        for path in sorted(root.glob("*/*.json")):
            value = json.loads(path.read_text())
            renamed = rename_json(value, names)
            if renamed != value:
                revisions += 1
                if not dry:
                    _write_json(path, renamed)
        for path in sorted(root.glob("*/*.plan")):
            text = path.read_text()
            if (renamed_text := rename_plan(text, names)) != text:
                plans += 1
                if not dry:
                    _write_text(path, renamed_text)
        if revisions:
            result.changes.append(f"places: {revisions} revisions' cameras")
        if plans:
            result.changes.append(f"places: {plans} plan files' camera spots")
        for name in PLACE_FILES:
            if (root / name).exists():
                _json_file(root / name, names, dry, result, f"places/{name}")


def rename_state_files(state: Path, names: Mapping[str, str], dry: bool, result: Result) -> None:
    """Worker state keyed by camera: see-zones' checkpoint, the vector worker's cursors, and the rest."""
    files = [state / name for name in STATE_FILES] + sorted((state / "continuum" / "vector").glob("segments.*.json"))
    for path in files:
        if path.exists():
            _json_file(path, names, dry, result, str(path.relative_to(state)))


def _slots(state: Path, names: Mapping[str, str]) -> Iterator[tuple[Path, Path]]:
    """(old slot, new slot) for each old camera's directory in each segment stream."""
    root = state / "segments"
    if not root.is_dir():
        return
    for stream in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for old, new in names.items():
            if (stream / old).is_dir():
                yield stream / old, stream / new


def slot_conflicts(state: Path, names: Mapping[str, str]) -> list[str]:
    """Segment streams that hold both a camera's old slot and its new one: a writer ran under the new name first."""
    return [f"{old} and {new} both exist" for old, new in _slots(state, names) if new.exists()]


def rename_slots(state: Path, names: Mapping[str, str], dry: bool, result: Result) -> None:
    moved = 0
    for old, new in _slots(state, names):
        if new.exists():
            continue  # a conflict, refused before anything ran
        moved += 1
        if not dry:
            old.rename(new)
    if moved:
        result.changes.append(f"segments: {moved} camera slots renamed")
    deletions = state / "segments" / ".deletions"
    if deletions.exists() and any(f'"{old}"' in line for line in deletions.read_text().splitlines() for old in names):
        result.warnings.append("segments/.deletions names an old camera; those forgets keep the old name")


def _held(lock: Path) -> bool:
    try:
        with lock.open("a") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        return False
    return False


def busy(state: Path, names: Mapping[str, str]) -> list[str]:
    """What's still running on the state this would move: a segment writer on an old camera's slot, or see-zones."""
    locks = [old / ".writer.lock" for old, _ in _slots(state, names)] + [state / "see-zones.lock"]
    return [str(lock.relative_to(state)) for lock in locks if lock.exists() and _held(lock)]


def leftovers(state: Path, names: Mapping[str, str]) -> list[str]:
    """Owner-taught state still under an old name: knowledge records (with plans' camera spots) and the reports
    layout. Empty once a run has finished."""
    out: list[str] = []
    store = _knowledge(state)
    if store is not None:
        for collection in store.collections():
            if collection.startswith(SKIP_PREFIXES):
                continue
            left = sum(1 for r in store.query(collection) if _renamed_record(r.record, names) != r.record)
            if left:
                out.append(f"knowledge {collection}: {left} records")
    path = state / "reports.sqlite3"
    if path.exists():
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = db.execute("SELECT value FROM meta WHERE key='layout'").fetchone()
        except sqlite3.OperationalError:
            row = None
        finally:
            db.close()
        if row is not None:
            layout = json.loads(row[0])
            if [_renamed_report(r, names) for r in layout.get("reports", [])] != layout.get("reports", []):
                out.append("reports layout")
    return out


def _toml(value: Any) -> str:
    """`value` as TOML for one line of pack.toml: a table inline."""
    import tomlkit

    if isinstance(value, dict):
        table = tomlkit.inline_table()
        table.update(value)
        return table.as_string()
    return tomlkit.item(value).as_string()


def _settings(table: Mapping[str, Any], names: Mapping[str, str], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every setting as (dotted path, value). A table keyed by an old camera is one setting (`camera_heights_m`,
    `[vision.actions_camera_fps]`); any other table is a section, walked into."""
    for key, value in table.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and not any(k in names for k in value):
            yield from _settings(value, names, path)
        else:
            yield path, value


def config_edits(names: Mapping[str, str], devices: Path, settings: Mapping[str, Any] | None = None) -> list[str]:
    """The config edits the rename needs, as lines to read: each device to rename in `devices` (the manifest), then
    each setting in the effective config (`settings`, default `config.load()`) that names an old camera, with its
    new value. Prints; never writes."""
    out: list[str] = []
    try:
        manifest = tomllib.loads(devices.read_text()) if devices.exists() else {}
    except tomllib.TOMLDecodeError as error:
        out.append(f"{devices}: unreadable ({error})")
        manifest = {}
    present = {str(d.get("name")) for d in manifest.get("device", []) if isinstance(d, dict)}
    for old, new in names.items():
        if old in present:
            out.append(f'{devices}: [[device]] name = "{old}"  ->  name = "{new}"')
        elif new in present:
            out.append(f"{devices}: {new} already renamed")
        else:
            out.append(f"{devices}: no device named {old}")
    local_path = cfg.local_config()
    local = tomllib.loads(local_path.read_text()) if local_path.exists() else {}
    effective = cfg.load() if settings is None else settings
    for path, value in _settings(effective, names):
        new_value = rename_setting(value, names)
        if new_value == value:
            continue
        section, _, key = path.rpartition(".")
        where = "pack.toml" if _lookup(local, path) is not None else "defaults.toml; add it to pack.toml"
        out.append(f"{local_path.name}: [{section}] {key} = {_toml(new_value)}   (now from {where})")
    return out


def _lookup(table: Mapping[str, Any], path: str) -> Any:
    node: Any = table
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


STEPS: tuple[Callable[[Path, Mapping[str, str], bool, Result], None], ...] = (
    rename_knowledge,
    rename_reports,
    rename_places,
    rename_state_files,
    rename_slots,
)


def run(state: Path, names: Mapping[str, str], dry: bool) -> Result:
    """Rename the cameras in `state` (or, `dry`, say what would change). Refuses, changing nothing, while a writer
    holds a camera's slot or a slot's new name is taken."""
    result = Result(conflicts=slot_conflicts(state, names))
    if not dry:
        result.conflicts += [f"{what} is held: stop the services first" for what in busy(state, names)]
    if result.conflicts and not dry:
        return result
    for step in STEPS:
        step(state, names, dry, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pack-rename-camera", description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="+", metavar="OLD=NEW", help="a camera's name and its new one")
    parser.add_argument("--dry-run", action="store_true", help="say what would change; write nothing")
    parser.add_argument("--state", type=Path, default=None, help="the state directory (default paths.state)")
    parser.add_argument(
        "--devices", type=Path, default=cfg.ROOT / "config" / "devices.toml", help="the device manifest"
    )
    args = parser.parse_args(argv)
    try:
        names = parse_names(args.names)
    except ValueError as error:
        parser.error(str(error))
    state = args.state or cfg.path("paths.state")
    result = run(state, names, args.dry_run)
    verb = "Would change" if args.dry_run else "Changed"
    for line in result.conflicts:
        print(f"CONFLICT: {line}")
    if result.conflicts and not args.dry_run:
        print("Nothing changed.")
        return 1
    print(f"{verb}:" if result.changes else "Nothing under the old names to change.")
    for line in result.changes:
        print(f"  {line}")
    for line in result.warnings:
        print(f"WARN: {line}")
    if not args.dry_run:
        left = leftovers(state, names)
        print(
            "Owner-taught state left under an old name:"
            if left
            else "Nothing owner-taught is left under the old names."
        )
        for line in left:
            print(f"  {line}")
    print("Config edits to make by hand:")
    for line in config_edits(names, args.devices):
        print(f"  {line}")
    return 1 if result.conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
