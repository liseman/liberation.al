#!/usr/bin/env python3
"""Export saved Ask exchanges as draft retrieval labels, without running a model or measurement.

Run `uv run python tools/export_ask_history.py --surface review --day YYYY-MM-DD` for JSONL on stdout.
`--output <paths.measurement>/retrieval.jsonl` atomically upserts exported drafts, retaining manual labels.
Every exported row remains draft:true until a person confirms it. Questions without available canonical
timestamped citations are skipped and counted; no answer expectation is invented from a prior model answer.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from liberational import config as cfg
from liberational.memory import ask_history


def output_path(path: Path) -> Path:
    """Allow only untracked JSONL labels inside the configured runtime measurement directory."""
    target = path.expanduser().resolve()
    measurement = cfg.path("paths.measurement").resolve()
    if target.suffix != ".jsonl" or not target.is_relative_to(measurement) or target == measurement:
        raise ValueError("output must be a .jsonl file under paths.measurement")
    root = cfg.ROOT.resolve()
    if any(
        target.is_relative_to(root / name)
        for name in (
            "src",
            "tests",
            "tools",
            "frontend",
            "continuum",
            "docs",
            "config",
            ".git",
        )
    ):
        raise ValueError("output cannot be in tracked or source directories")
    if target.is_relative_to(root):
        checked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(target.relative_to(root))],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if checked.returncode == 0:
            raise ValueError("output cannot overwrite a tracked file")
        if checked.returncode != 1:
            raise RuntimeError("could not verify that output is untracked")
    return target


def merge_drafts(
    existing: Sequence[Mapping[str, Any]], drafts: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], int, int]:
    """Replace matching drafts only, preserve confirmed/manual rows, and append new export IDs."""
    pending = {str(row["id"]): dict(row) for row in drafts}
    rows = []
    updated = protected = 0
    for old in existing:
        key = str(old.get("id", ""))
        proposed = pending.pop(key, None)
        if proposed is not None and old.get("draft") is True:
            rows.append(proposed)
            updated += 1
        else:
            rows.append(dict(old))
            protected += proposed is not None
    rows.extend(pending.values())
    return rows, updated, protected


def write_drafts(path: Path, drafts: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
    """Validate every old row before writing, then replace one runtime label file atomically."""
    target = output_path(path)
    existing = []
    if target.exists():
        for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"existing labels have invalid JSON on line {number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"existing labels must contain JSON objects (line {number})")
            existing.append(row)
    merged, updated, protected = merge_drafts(existing, drafts)
    added = len(merged) - len(existing)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".ask-drafts-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in merged:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return added, updated, protected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-dir", type=Path, help="state directory of the existing local broker")
    parser.add_argument("--surface", choices=("live", "review", "call", "voice"))
    parser.add_argument("--day", help="selected Review day, YYYY-MM-DD")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="runtime .jsonl file under paths.measurement")
    destination.add_argument("--stdout", action="store_true", help="write JSONL to stdout (the default)")
    args = parser.parse_args(argv)
    if args.day is not None:
        try:
            if dt.date.fromisoformat(args.day).isoformat() != args.day:
                raise ValueError()
        except ValueError:
            parser.error("--day must be YYYY-MM-DD")
    try:
        # Reject an unsafe destination before connecting or reading history.
        if args.output is not None:
            output_path(args.output)
        history = ask_history.at(args.state_dir)
        drafts = history.export_drafts(surface=args.surface, day=args.day)
        if args.output is None:
            for row in drafts:
                print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            print(
                f"exported {len(drafts)} draft labels; skipped {history.export_summary['skipped']} unscorable exchanges",
                file=sys.stderr,
            )
        else:
            added, updated, protected = write_drafts(args.output, drafts)
            print(
                f"draft labels: {added} added, {updated} updated, {protected} manual labels preserved; skipped {history.export_summary['skipped']} unscorable exchanges",
                file=sys.stderr,
            )
    except Exception as error:
        print(f"Ask history export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
