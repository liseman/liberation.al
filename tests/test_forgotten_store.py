"""Forgotten-name hashes persist independently of the broker and the registry's erasure scope."""

import subprocess
import sys

from liberational import config as cfg
from liberational import knowledge
from liberational.people import forgotten


def test_sqlite_suppression_keeps_only_hashes_and_counts_new_normalized_names(tmp_path):
    database = tmp_path / "knowledge.sqlite3"
    backend = knowledge.Store(database)
    kept = forgotten.SQLiteStore(backend)
    assert kept.remember({"Dana", " dana ", "Dana White"}) == 2
    revision = backend.revision()
    assert kept.remember({"DANA", "dana white"}) == 0
    assert backend.revision() == revision
    assert kept.has("Dana") and kept.has("DANA WHITE")
    assert not kept.has("Erin")
    rows = backend.query("forgotten.names")
    assert len(rows) == 2
    assert {row.record.id for row in rows} == {
        forgotten.broker_digest("Dana").hex(),
        forgotten.broker_digest("Dana White").hex(),
    }
    assert all(row.record.key == row.record.id for row in rows)
    assert all(row.record.value == {} and row.record.subjects == () for row in rows)
    assert b"dana" not in database.read_bytes().lower()
    assert forgotten.broker_digest("Dana") == forgotten.broker_digest(" dana ")


def test_reopened_store_reuses_the_same_suppression_without_new_records(tmp_path):
    database = tmp_path / "knowledge.sqlite3"
    original = forgotten.SQLiteStore(knowledge.Store(database))
    assert original.remember({"Dana"}) == 1
    reopened = forgotten.SQLiteStore(knowledge.Store(database))
    assert reopened.has("DANA")
    assert reopened.remember({" dana "}) == 0
    assert len(reopened.backend.query("forgotten.names")) == 1


def test_state_selection_is_independent_and_reset_does_not_erase_suppression(tmp_path):
    default, other = tmp_path / "default", tmp_path / "other"
    with cfg.override({"paths": {"state": str(default)}}):
        kept = forgotten.store()
        assert kept.backend.path == default / "knowledge.sqlite3"
        assert forgotten.store(default) is kept
        assert kept.remember({"Dana"}) == 1
        assert not forgotten.store(other).has("Dana")
        assert forgotten.store(other).remember({"Erin"}) == 1
        forgotten.reset()
        assert forgotten.store().has("Dana")
        assert not forgotten.store().has("Erin")
        assert forgotten.store(other).has("Erin")


def test_store_wraps_the_process_wide_knowledge_store(tmp_path):
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        assert forgotten.store().backend is knowledge.at(tmp_path / "knowledge.sqlite3")


def test_disabled_suppression_does_not_create_a_knowledge_database(tmp_path):
    with cfg.override({"paths": {"state": str(tmp_path)}, "assistant": {"forgotten_reply": False}}):
        assert forgotten.remember(["Dana"], tmp_path) == 0
        assert not forgotten.named("Who's Dana?", state=tmp_path)
    assert not (tmp_path / "knowledge.sqlite3").exists()


def test_another_process_reads_and_extends_suppression_without_a_broker(tmp_path):
    database = tmp_path / "knowledge.sqlite3"
    kept = forgotten.SQLiteStore(knowledge.Store(database))
    assert kept.remember({"Dana"}) == 1
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from liberational import knowledge; from liberational.people import forgotten; "
                "store = forgotten.SQLiteStore(knowledge.Store(sys.argv[1])); "
                "assert store.has('DANA'); assert store.remember({' dana '}) == 0; "
                "assert store.remember({'Erin'}) == 1"
            ),
            str(database),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert kept.has("Erin")
    assert len(kept.backend.query("forgotten.names")) == 2
