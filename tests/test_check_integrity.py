"""tools/check_integrity.py: references between the stores that point nowhere, on a made-up state."""

import importlib.util
import json
import sqlite3

import pytest

from liberational import config as cfg
from liberational.domain.identity import Basis
from liberational.people import registry
from voices import write_voices

_spec = importlib.util.spec_from_file_location("check_integrity", cfg.ROOT / "tools" / "check_integrity.py")
assert _spec and _spec.loader
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)

T = 1793237000.0


def line(i, device="Mic", sid=1):
    return {"ts_start": T + i, "ts_end": T + i + 1, "device": device, "speaker": "Speaker 1", "speaker_id": sid,
            "speaker_sim": 0.7, "text": f"line {i}"}  # fmt: skip


@pytest.fixture
def state(tmp_path):
    """A state that agrees with itself: two voices, three lines, a cached line embedding, and a journal entry citing a
    line."""
    write_voices(tmp_path, {1: "Alice", 2: None})
    lines = [line(0), line(1), line(2, device=None)]  # the last one has no device: its key is its millisecond
    (tmp_path / "transcript.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
    c = sqlite3.connect(tmp_path / "sightings.sqlite3")  # v1's, which holds speech's voice_lines
    c.execute("CREATE TABLE voice_lines (key TEXT PRIMARY KEY, ts_start REAL NOT NULL, dur REAL, emb BLOB)")
    old = registry.line_key(T + 1)  # a legacy cached embedding stored under its millisecond alone
    c.execute("INSERT INTO voice_lines VALUES (?, ?, 1.0, NULL)", (old, T + 1))
    c.commit()
    c.close()
    j = sqlite3.connect(tmp_path / "interpretations.sqlite3")
    j.executescript(
        "CREATE TABLE interpretations (id TEXT PRIMARY KEY, status TEXT NOT NULL);"
        "CREATE TABLE sources (interpretation TEXT NOT NULL, kind TEXT NOT NULL, ref TEXT NOT NULL);"
    )
    j.execute("INSERT INTO interpretations VALUES ('e1', 'active')")
    j.execute("INSERT INTO sources VALUES ('e1', 'utterance', ?)", (registry.key_of(lines[0]),))
    j.commit()
    j.close()
    return tmp_path


def run(state):
    return {k: r["bad"] for k, r in ci.check(state, state / "faces").checks.items()}


def test_a_state_that_agrees_with_itself_has_nothing_to_report(state):
    bad = run(state)
    assert sum(bad.values()) == 0, {k: v for k, v in bad.items() if v}
    assert {"transcript: two lines with one key", "events: an assertion about a line that doesn't exist"} <= set(bad)


def test_each_reference_that_points_nowhere_is_counted(state):
    with (state / "transcript.jsonl").open("a") as f:
        f.write(json.dumps(line(5, sid=9)) + "\n")  # a voice the list doesn't hold
        f.write(json.dumps(line(0)) + "\n")  # the first line again: one key, two lines
        f.write("{not json\n")
    reg = registry.at(state)
    reg.claim(registry.utterance(T + 50, "Mic"), "Alice", Basis.CORRECTION, "test")  # no line then
    c = sqlite3.connect(state / "sightings.sqlite3")
    c.execute("INSERT INTO voice_lines VALUES ('mic@1793237077.000', ?, 1.0, NULL)", (T + 77,))  # no such line
    c.commit()
    c.close()
    j = sqlite3.connect(state / "interpretations.sqlite3")
    j.execute("INSERT INTO sources VALUES ('e1', 'interpretation', 'gone')")
    j.commit()
    j.close()
    bad = run(state)
    assert {k: v for k, v in bad.items() if v} == {
        "transcript: a line that can't be read, or has no ts_start": 1,
        "transcript: two lines with one key": 1,
        "transcript: a speaker_id no voiceprint has": 1,
        "events: an assertion about a line that doesn't exist": 1,
        "sightings: a cached line embedding for a line that's gone": 1,
        "journal: an active entry citing an entry that's gone": 1,
    }


def test_an_old_millisecond_reference_names_a_line_only_while_it_is_alone_in_it(state):
    with (state / "transcript.jsonl").open("a") as f:
        f.write(json.dumps(line(1, device="Other mic")) + "\n")  # now two lines start in that millisecond
    assert run(state)["sightings: a cached line embedding for a line that's gone"] == 1


def test_it_never_prints_a_persons_name(state, capsys):
    (state / "faces" / "Zelda").mkdir(parents=True)  # a photo folder for nobody the registry knows
    code = ci.main(["--state-dir", str(state), "--faces-dir", str(state / "faces")])
    out = capsys.readouterr().out
    assert code == 1 and "Zelda" not in out and "a photo folder for no person: 1 of 1" in out


def test_it_writes_nothing(state):
    before = {p: p.read_bytes() for p in state.rglob("*") if p.is_file() and not p.name.endswith(("-shm", "-wal"))}
    run(state)
    after = {p: p.read_bytes() for p in state.rglob("*") if p.is_file() and not p.name.endswith(("-shm", "-wal"))}
    assert after == before
