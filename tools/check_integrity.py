"""Check that the stores agree with each other: references that point nowhere, read-only.

    uv run tools/check_integrity.py                     # the state in paths.state
    uv run tools/check_integrity.py --state-dir DIR     # another state (a copy, say)
    uv run tools/check_integrity.py --json

Each check compares what one store says with another and counts the references that resolve to nothing:

- transcript.jsonl: lines it can't read, two lines with one key, and a speaker_id no voiceprint has (the
  numbers listen handed out, state/voiceprints.json);
- the event log (on the broker): identity assertions about a line or voice that doesn't exist, or for a person nobody
  registered;
- sightings.sqlite3 (v1's, which holds speech's voice_lines): a cached line embedding for a line the transcript
  doesn't have. Faces are on the vision log now, with nothing here to cross-check;
- the face store (paths.faces): names in db.npz, and photo folders, that resolve to no person;
- interpretations.sqlite3 (the journal): an active entry citing a line, an entry, or an assistant reply that's gone.

It prints counts, and record ids (line keys, voice and sighting ids) for a few of each, never a person's name. It
opens every database read-only and writes nothing.
"""

import argparse
import collections
import json
import pathlib
import sqlite3
import sys
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational.people import registry

SAMPLE = 6  # record ids shown per check


def _ro(path: pathlib.Path) -> sqlite3.Connection | None:
    """A read-only connection, or None when the store isn't there."""
    if not path.is_file():
        return None
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)


def _tables(c: sqlite3.Connection) -> set[str]:
    return {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}


class Report:
    """Each check's count of bad references, out of how many it looked at, with a few record ids."""

    def __init__(self) -> None:
        self.checks: dict[str, dict[str, Any]] = {}

    def add(self, check: str, bad: Iterable[object], total: int, note: str = "", show: bool = True) -> None:
        bad = list(bad)
        self.checks[check] = {
            "bad": len(bad),
            "of": total,
            "sample": [str(b)[:100] for b in bad[:SAMPLE]] if show else [],
            "note": note,
        }

    def text(self) -> str:
        out = []
        for check, r in self.checks.items():
            mark = "  " if not r["bad"] else "!!"
            out.append(f"{mark} {check}: {r['bad']} of {r['of']}" + (f" ({r['note']})" if r["note"] else ""))
            out += [f"     {s}" for s in r["sample"]]
        problems = sum(1 for r in self.checks.values() if r["bad"])
        out.append(f"{problems} of {len(self.checks)} checks found something" if problems else "Everything agrees.")
        return "\n".join(out) + "\n"


class Lines:
    """The transcript's lines, and what a reference to one may be: its key, or (in older rows) its
    millisecond alone, which names it while no other line starts in that millisecond."""

    def __init__(self, path: pathlib.Path, report: Report) -> None:
        self.lines: list[dict[str, Any]] = []
        unreadable = []
        text = path.read_text(errors="replace") if path.exists() else ""
        for n, raw in enumerate(text.splitlines(), 1):
            if not raw.strip():
                continue
            try:
                e = json.loads(raw)
                if not isinstance(e, dict) or not isinstance(e.get("ts_start"), int | float):
                    raise ValueError
                self.lines.append(e)
            except ValueError:
                unreadable.append(f"line {n}")
        report.add(
            "transcript: a line that can't be read, or has no ts_start", unreadable, len(self.lines) + len(unreadable)
        )
        counts = collections.Counter(registry.key_of(e) for e in self.lines)
        report.add("transcript: two lines with one key", [k for k, n in counts.items() if n > 1], len(counts))
        self.keys = set(counts)
        self.ms = collections.Counter(registry.line_key(e["ts_start"]) for e in self.lines)

    def has(self, ref: str) -> bool:
        """Whether a line key, or an older millisecond-only reference, names a line here."""
        return ref in self.keys or self.ms.get(ref) == 1


def check(state: pathlib.Path, faces: pathlib.Path) -> Report:
    """Every check over one state. Read-only."""
    report = Report()
    lines = Lines(state / "transcript.jsonl", report)

    handed_out = registry.handed_out(state)  # the voiceprints listen handed out (listen/voices.py)

    def voice(sid: int) -> int | None:
        return sid if 1 <= sid < handed_out else None

    sids = {
        e["speaker_id"]
        for e in lines.lines
        if isinstance(e.get("speaker_id"), int) and not isinstance(e.get("speaker_id"), bool)
    }
    report.add("transcript: a speaker_id no voiceprint has", sorted(s for s in sids if voice(s) is None), len(sids))

    _events(state, report, lines, voice)

    reg = registry.at(state) if registry.log_exists(state) else None
    names: set[str] = set()
    if reg is not None:
        for p in reg.people():
            names |= {registry.norm(n) for n in (p.display_name, *p.aliases)}

    def known(name: str) -> bool:
        return registry.norm(name) in names or (reg is not None and reg.known_as(name) is not None)

    sg = _ro(state / "sightings.sqlite3")
    if sg is not None:
        _voice_lines(sg, report, lines)
        sg.close()
    _faces(faces, report, known)
    _journal(state, report, lines)
    return report


def _events(state: pathlib.Path, report: Report, lines: Lines, voice: Callable[[int], int | None]) -> None:
    if not registry.log_exists(state):
        return
    log = registry.at(state).log
    people = set()
    for stored in log.read(registry.PEOPLE):
        p = stored.event.payload.get("person")
        if isinstance(p, str):
            people.add(p)
    bad: dict[str, list[str]] = collections.defaultdict(list)
    seen: collections.Counter[str] = collections.Counter()
    for stream in (registry.ASSERTIONS, registry.SIGHTINGS):
        for stored in log.read(stream):
            kind, payload = str(stored.event.type), stored.event.payload
            a = payload.get("assertion") or {}
            subject = str((a.get("subject") or {}).get("id") or payload.get("utterance") or "")
            what, _, ref = subject.partition(":")
            if what == "speaker":
                seen["voice"] += 1
                if not ref.isdigit() or voice(int(ref)) is None:
                    bad["voice"].append(subject)
            elif what == "utterance":
                seen["line"] += 1
                if not lines.has(ref):
                    bad["line"].append(f"{kind} {subject}")
            if (person := a.get("person")) and isinstance(person, str):
                seen["person"] += 1
                if person not in people:
                    bad["person"].append(f"{kind} {person}")
    for k in ("line", "voice"):
        report.add(f"events: an assertion about a {k} that doesn't exist", bad[k], seen[k])
    report.add("events: an assertion for a person nobody registered", bad["person"], seen["person"])


def _voice_lines(sg: sqlite3.Connection, report: Report, lines: Lines) -> None:
    if "voice_lines" in _tables(sg) and "key" in {r[1] for r in sg.execute("PRAGMA table_info(voice_lines)")}:
        cached = [k for (k,) in sg.execute("SELECT key FROM voice_lines")]
        report.add("sightings: a cached line embedding for a line that's gone", [k for k in cached if not lines.has(k)],
                   len(cached))  # fmt: skip


def _faces(faces: pathlib.Path, report: Report, known: Callable[[str], bool]) -> None:
    db = faces / "db.npz"
    if db.is_file():
        with np.load(db, allow_pickle=False) as d:
            stored = {str(n) for n in d["names"]} if "names" in d.files else set()
        report.add("faces: a name in db.npz that resolves to no person", [n for n in stored if not known(n)], len(stored),
                   "names not shown", show=False)  # fmt: skip
    if faces.is_dir():
        folders = [p.name for p in faces.iterdir() if p.is_dir() and not p.name.startswith("_")]
        report.add("faces: a photo folder for no person", [n for n in folders if not known(n)], len(folders),
                   "names not shown", show=False)  # fmt: skip


def _journal(state: pathlib.Path, report: Report, lines: Lines) -> None:
    j = _ro(state / "interpretations.sqlite3")
    if j is None or not {"interpretations", "sources"} <= _tables(j):
        return
    ids = {i for (i,) in j.execute("SELECT id FROM interpretations")}
    cited = j.execute(
        "SELECT s.interpretation, s.kind, s.ref FROM sources s JOIN interpretations i ON i.id = s.interpretation "
        "WHERE i.status = 'active'"
    ).fetchall()
    j.close()
    replies: set[str] = set()
    said = state / "assistant.jsonl"
    for raw in said.read_text(errors="replace").splitlines() if said.exists() else []:
        try:
            e = json.loads(raw)
        except ValueError:
            continue
        if isinstance(e, dict) and isinstance(e.get("ts_start"), int | float):
            replies.add(f"assistant@{float(e['ts_start']):.3f}")
    for kind, what, gone in (
        ("utterance", "line", lambda r: not lines.has(r)),
        ("interpretation", "entry", lambda r: r not in ids),
        ("event", "assistant reply", lambda r: r.startswith("assistant@") and r not in replies),
    ):
        refs = [(i, r) for i, k, r in cited if k == kind]
        report.add(f"journal: an active entry citing a{'n' if what[0] in 'ae' else ''} {what} that's gone",
                   [f"{i[:13]} {r}" for i, r in refs if gone(r)], len(refs))  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state-dir", help="state to check (default: paths.state)")
    ap.add_argument("--faces-dir", help="face store to check (default: paths.faces)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args(argv)
    state = pathlib.Path(a.state_dir) if a.state_dir else cfg.path("paths.state")
    faces = pathlib.Path(a.faces_dir) if a.faces_dir else cfg.path("paths.faces")
    report = check(state.resolve(), faces.resolve())
    print(json.dumps(report.checks, indent=2) if a.json else report.text(), end="")
    return 1 if any(r["bad"] for r in report.checks.values() if "often retention" not in r["note"]) else 0


if __name__ == "__main__":
    sys.exit(main())
