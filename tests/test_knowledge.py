"""Durability, indexed reads, and transaction/change semantics of the independent knowledge store."""

import dataclasses
import gc
import json
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from liberational import knowledge
from liberational.knowledge import Conflict, Deleted, Record, StorageError, Store


def record(id="one", *, collection="notes", key="label", start=10, end=20, subjects=("alice",), **value):
    return Record(id, collection, key, start, end, 30, value or {"text": "hello"}, subjects)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "knowledge.sqlite3")


@pytest.fixture
def tracked_connections(monkeypatch):
    opened, closed = [], []
    connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        def close(self):
            super().close()
            closed.append(self)

    def tracked(*args, **kwargs):
        db = connect(*args, **kwargs, factory=TrackedConnection)
        opened.append(db)
        return db

    monkeypatch.setattr(sqlite3, "connect", tracked)
    return opened, closed


@pytest.fixture
def read_in_fresh_thread():
    def read(store):
        results = []
        thread = threading.Thread(target=lambda: results.append(store.revision()))
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert results == [0]

    return read


def test_exited_request_threads_close_their_cached_connections(store, tracked_connections, read_in_fresh_thread):
    store.close()
    read_in_fresh_thread(store)
    read_in_fresh_thread(store)
    opened, closed = tracked_connections
    assert len(opened) == 2
    assert closed == opened
    first, second = opened
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        first.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        second.execute("SELECT 1")


def test_persistent_worker_reuses_handle_and_reconnects_after_store_close(store, tracked_connections):
    store.close()
    opened, closed = tracked_connections
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(store.revision).result(timeout=5) == 0
        assert worker.submit(store.revision).result(timeout=5) == 0
        assert len(opened) == 1
        assert closed == []
        store.close()
        assert closed == opened
        assert worker.submit(store.revision).result(timeout=5) == 0
        assert len(opened) == 2
        assert len(closed) == 1
    assert closed == opened


def test_close_waits_for_a_transaction_in_progress_on_another_thread(store):
    inside, release = threading.Event(), threading.Event()

    def write():
        with store.transaction():
            store.put([record("first", key="a")])
            inside.set()
            release.wait(5)
            store.put([record("second", key="b")])

    with ThreadPoolExecutor(max_workers=2) as pool:
        writing = pool.submit(write)
        assert inside.wait(5)
        closing = pool.submit(store.close)
        with pytest.raises(TimeoutError):
            closing.result(timeout=0.2)
        release.set()
        writing.result(timeout=5)
        closing.result(timeout=5)
    assert [store.get(id).record.id for id in ("first", "second")] == ["first", "second"]


def test_close_inside_its_own_transaction_raises_and_keeps_the_transaction(store):
    with store.transaction():
        store.put([record()])
        with pytest.raises(StorageError, match="inside its own transaction"):
            store.close()
    assert store.get("one").record == record()


def test_new_owner_database_is_private(store):
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_exact_retries_preserve_position_without_extra_changes(store):
    first = record()
    (position,) = store.put([first])
    assert store.put([first, first]) == [position, position]
    assert store.get("one").record == first
    assert [(c.operation, c.id) for c in store.changes()] == [("put", "one")]


def test_conflict_rolls_back_the_entire_batch_and_its_changes(store):
    store.put([record()])
    before = store.revision()
    with pytest.raises(Conflict):
        store.put([record("new"), record(text="different")])
    assert store.get("new") is None
    assert store.revision() == before
    assert store.get("one").record.value == {"text": "hello"}


@pytest.mark.parametrize("replacement", [True, 1.0])
def test_json_type_changes_are_conflicts_even_when_python_equality_agrees(store, replacement):
    store.put([record(number=1)])
    before = store.revision()
    with pytest.raises(Conflict):
        store.put([record(number=replacement)])
    assert type(store.get("one").record.value["number"]) is int
    assert store.revision() == before


def test_mutating_input_or_read_result_cannot_change_saved_content(store):
    original = record(items=[{"nested": [1]}])
    store.put([original])
    original.value["items"][0]["nested"].append(2)
    read = store.get("one")
    read.record.value["items"].clear()
    assert store.get("one").record.value == {"items": [{"nested": [1]}]}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1: "non-string key"}, object()])
def test_bad_json_anywhere_in_a_batch_writes_nothing(store, bad):
    with pytest.raises(ValueError):
        store.put([record(), record("bad", value=bad)])
    assert store.query("notes") == []
    assert store.revision() == 0


@pytest.mark.parametrize("field,value", [("start_ns", True), ("end_ns", -1), ("recorded_ns", 2**63)])
def test_invalid_native_times_are_rejected_before_writing(store, field, value):
    with pytest.raises(ValueError):
        store.put([dataclasses.replace(record(), **{field: value})])
    assert store.revision() == 0


def test_time_window_keeps_overlapping_spans_and_points_at_lower_bound_only(store):
    store.put(
        [
            record("left", start=0, end=10),
            record("overlap", start=9, end=11),
            record("point", start=10, end=10),
            record("right", start=20, end=20),
        ]
    )
    assert [s.record.id for s in store.query("notes", start_ns=10, end_ns=20)] == ["overlap", "point"]


def test_subject_key_collection_and_position_filters_compose_without_duplicates(store):
    positions = store.put(
        [
            record("one", subjects=("alice", "dana")),
            record("two", subjects=("dana",)),
            record("other-key", key="different", subjects=("dana",)),
            record("other-collection", collection="facts", subjects=("dana",)),
        ]
    )
    assert [s.record.id for s in store.query("notes", subjects=("alice", "dana"), key="label")] == ["one", "two"]
    assert [s.record.id for s in store.query("notes", after=positions[0], through=positions[1], limit=1)] == ["two"]
    assert store.collections(prefix="fac") == ["facts"]


@pytest.fixture
def statements(monkeypatch):
    seen = []
    connect = sqlite3.connect

    def traced(*args, **kwargs):
        db = connect(*args, **kwargs)
        db.set_trace_callback(seen.append)
        return db

    monkeypatch.setattr(sqlite3, "connect", traced)
    return seen


def test_a_thousand_row_query_reads_subjects_in_a_constant_number_of_statements(statements, tmp_path):
    store = Store(tmp_path / "knowledge.sqlite3")  # opened after the trace is installed
    store.put(
        [record("lone", subjects=())]
        + [record(f"r{i:04}", subjects=("zed", f"pet{i % 3}", "alice")) for i in range(1000)]
    )
    store.query("notes", limit=1)
    statements.clear()
    one = store.query("notes", limit=1)
    small = len(statements)
    statements.clear()
    rows = store.query("notes")
    assert 0 < len(statements) == small
    assert one[0].record.subjects == ()
    assert len(rows) == 1001
    assert rows[1].record.subjects == ("alice", "pet0", "zed")
    assert rows[1000].record == record("r0999", subjects=("alice", "pet0", "zed"))


def test_get_snapshot_and_a_subject_filter_return_every_subject_of_each_record(store):
    store.put([record("one", subjects=("dana", "alice")), record("two", subjects=())])
    assert store.get("one").record.subjects == ("alice", "dana")
    assert [s.record.subjects for s in store.snapshot().records] == [("alice", "dana"), ()]
    assert [s.record.subjects for s in store.query("notes", subjects=("dana",))] == [("alice", "dana")]


def test_erasure_changes_survive_reopen_and_ids_cannot_resurrect(store):
    (position,) = store.put([record()])
    assert store.remove(["one"]) == 1
    reopened = Store(store.path)
    assert reopened.get("one") is None
    assert [(c.operation, c.id) for c in reopened.changes(position)] == [("remove", "one")]
    with pytest.raises(Deleted):
        reopened.put([record()])
    (later,) = reopened.put([record("later")])
    assert later > position + 1


def on_disk(store):
    """The database file and its journals (a WAL store's -wal and -shm) as one blob."""
    return b"".join(p.read_bytes() for p in store.path.parent.glob(store.path.name + "*"))


def test_erased_record_content_is_not_left_in_database_free_space(store):
    secret = "private-fixture-value-that-must-not-remain"
    store.put([record(text=secret)])
    assert secret.encode() in on_disk(store)
    store.remove(["one"])
    assert secret.encode() not in on_disk(store)


def test_audit_failure_rolls_back_erasure_and_change_feed(store):
    store.put([record(), record("audit", collection="audit")])
    before = store.revision()
    with pytest.raises(Conflict):
        store.remove(["one"], audit=record("audit", collection="audit", text="conflicting audit"))
    assert store.get("one") is not None
    assert store.revision() == before


def test_audit_uses_actual_removed_count_and_zero_removal_publishes_nothing(store):
    store.put([record()])
    audit = record("audit", collection="audit", removed=999)
    assert store.remove(["one", "one", "missing"], audit=audit) == 1
    assert store.get("audit").record.value == {"removed": 1}
    before = store.revision()
    assert store.remove(["missing"], audit=record("unused", collection="audit")) == 0
    assert store.revision() == before
    assert store.get("unused") is None


def test_collection_revision_includes_removals_and_cursor_never_moves_back(store):
    (position,) = store.put([record()])
    store.ack("facts", position)
    store.ack("facts", 0)
    store.remove(["one"])
    assert store.revision(["notes"]) > position
    assert store.revision(["other"]) == 0
    assert Store(store.path).cursor("facts") == position


def test_independent_writers_serialize_positions_without_losing_content(store):
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(Store(store.path).put, [record("a")])
        b = pool.submit(Store(store.path).put, [record("b")])
        positions = [a.result(timeout=10)[0], b.result(timeout=10)[0]]
    assert len(set(positions)) == 2
    assert {r.record.id for r in store.query("notes")} == {"a", "b"}


def test_wait_observes_another_process_without_reopening_the_reader(store):
    script = (
        "import sys; from liberational.knowledge import Store,Record; "
        "Store(sys.argv[1]).put([Record('child','notes',None,1,1,1,{'text':'child'})])"
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter = pool.submit(store.wait, 0, timeout_s=10)
        child = subprocess.run(
            [sys.executable, "-c", script, str(store.path)], capture_output=True, text=True, timeout=10
        )
        assert child.returncode == 0, child.stderr
        assert [(c.operation, c.id) for c in waiter.result(timeout=10)] == [("put", "child")]
    assert store.get("child").record.value == {"text": "child"}


def test_wait_has_a_bounded_empty_result(store):
    assert store.wait(0, timeout_s=0) == []


@pytest.mark.parametrize("contents", [b"not a database", b"SQLite format 3\x00broken"])
def test_corrupt_file_never_becomes_an_empty_store(tmp_path, contents):
    path = tmp_path / "knowledge.sqlite3"
    path.write_bytes(contents)
    with pytest.raises(StorageError):
        Store(path)
    assert path.read_bytes() == contents


def test_unrelated_or_newer_schema_is_not_reinitialized(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE precious (value TEXT)")
        db.execute("INSERT INTO precious VALUES ('keep')")
    with pytest.raises(StorageError, match="format"):
        Store(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT value FROM precious").fetchone()[0] == "keep"


def test_matching_version_markers_do_not_hide_missing_indexes(store):
    store.put([record()])
    with sqlite3.connect(store.path) as db:
        db.execute("DROP INDEX records_collection_time")
    with pytest.raises(StorageError, match="schema"):
        Store(store.path)


def test_matching_erasure_rolls_back_when_selection_fails(store):
    store.put([record("a"), record("b")])
    before = store.revision()

    def broken(_record):
        raise ValueError("cannot decide")

    with pytest.raises(ValueError, match="cannot decide"):
        store.remove_matching(["notes"], predicate=broken)
    assert {r.record.id for r in store.query("notes")} == {"a", "b"}
    assert store.revision() == before


def test_matching_erasure_is_bounded_to_its_collection_and_subject(store):
    store.put(
        [
            record("a", collection="event:a"),
            record("b", collection="event:b", subjects=("dana",)),
            record("precious", collection="enrollment"),
        ]
    )
    assert store.remove_matching(collection_prefix="event:", subjects=["alice"]) == 1
    assert store.get("a") is None
    assert store.get("b") is not None
    assert store.get("precious") is not None
    with pytest.raises(ValueError, match="boundary"):
        store.remove_matching()


def test_crash_inside_a_write_leaves_no_content_or_change_commit(store):
    script = (
        "import os,sys; from liberational.knowledge import Store,Record; "
        "s=Store(sys.argv[1]); original=s._put; "
        "s._put=lambda *a: (original(*a),os._exit(7))[0]; "
        "s.put([Record('crash','notes',None,1,1,1,{'text':'never committed'})])"
    )
    child = subprocess.run([sys.executable, "-c", script, str(store.path)], capture_output=True, text=True, timeout=10)
    assert child.returncode == 7, child.stderr
    reopened = Store(store.path)
    assert reopened.get("crash") is None
    assert reopened.revision() == 0


def test_actual_filtered_query_uses_collection_and_subject_indexes(store, monkeypatch):
    store.put([record(), record("other", subjects=("dana",))])
    statements = []
    connect = sqlite3.connect
    store.close()  # Trace the fresh reusable handle, rather than the already-open fixture connection.

    def traced(*args, **kwargs):
        db = connect(*args, **kwargs)
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(sqlite3, "connect", traced)
    assert [r.record.id for r in store.query("notes", subjects=["alice"], start_ns=10, end_ns=20)] == ["one"]
    (actual,) = [sql for sql in statements if sql.startswith("SELECT r.*")]
    with connect(store.path) as db:
        plan = json.dumps(db.execute("EXPLAIN QUERY PLAN " + actual).fetchall())
    assert "subjects_subject_record" in plan
    assert "records_collection_" in plan


def test_nineteen_nested_writes_reuse_one_connection_and_one_commit(store, monkeypatch):
    store.close()
    statements, connections = [], []
    connect = sqlite3.connect

    def traced(*args, **kwargs):
        db = connect(*args, **kwargs)
        connections.append(db)
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(sqlite3, "connect", traced)
    with store.transaction():
        positions = [store.put([record(str(n))])[0] for n in range(19)]
        assert len(store.query("notes")) == 19
    assert len(set(positions)) == 19
    assert len(connections) == 1
    assert statements.count("BEGIN IMMEDIATE") == 1
    assert statements.count("COMMIT") == 1
    assert len(store.query("notes")) == 19
    assert len(connections) == 1


def test_nested_transaction_failure_rolls_back_all_operations_and_reconnects_after_close(store):
    with pytest.raises(Conflict), store.transaction():
        store.put([record("before")])
        with store.transaction():
            store.put([record("nested")])
        store.put([record("before", text="conflict")])
    assert store.query("notes") == []
    assert store.revision() == 0
    store.close()
    store.put([record("after")])
    assert [item.record.id for item in store.query("notes")] == ["after"]


def test_cached_handle_reads_replacement_database(store, tmp_path):
    store.put([record("old")])
    replacement = Store(tmp_path / "replacement.sqlite3")
    replacement.put([record("new")])
    replacement.close()
    replacement.path.replace(store.path)
    assert store.get("old") is None
    assert store.get("new").record.id == "new"


def test_reference_generation_changes_for_same_incarnation_equal_revision_fork(store, tmp_path):
    with sqlite3.connect(store.path) as original, sqlite3.connect(tmp_path / "fork.sqlite3") as fork:
        original.backup(fork)
    store.put([record("old", collection="voice.reference")])
    before = store.generation(["voice.reference"])
    fork = Store(tmp_path / "fork.sqlite3")
    fork.put([record("new", collection="voice.reference")])
    fork.close()
    fork.path.replace(store.path)
    after = store.generation(["voice.reference"])
    assert before[1:] == after[1:]
    assert before[0] != after[0]
    assert [item.record.id for item in store.query("voice.reference")] == ["new"]


def test_generation_uses_opened_file_epoch_when_replacement_races_connect(store, tmp_path, monkeypatch):
    with sqlite3.connect(store.path) as original, sqlite3.connect(tmp_path / "fork.sqlite3") as fork:
        original.backup(fork)
    store.put([record("old", collection="voice.reference")])
    before = store.generation(["voice.reference"])
    store.close()
    replacement = Store(tmp_path / "fork.sqlite3")
    replacement.put([record("new", collection="voice.reference")])
    replacement.close()
    connect = sqlite3.connect
    replaced = False

    def replace_before_connect(*args, **kwargs):
        nonlocal replaced
        if not replaced:
            replacement.path.replace(store.path)
            replaced = True
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", replace_before_connect)
    after = store.generation(["voice.reference"])
    assert before[1:] == after[1:]
    assert before[0] != after[0]
    assert [item.record.id for item in store.query("voice.reference")] == ["new"]


def test_cached_handle_does_not_serve_deleted_database(store):
    store.put([record()])
    store.path.unlink()
    with pytest.raises(StorageError, match="removed"):
        store.query("notes")
    assert not store.path.exists()


def test_reconnecting_to_a_corrupted_file_fails_closed(store):
    # A live WAL connection keeps serving the committed pages it cached; the next connection must not read garbage
    store.put([record()])
    store.path.write_bytes(b"not a database")
    store.close()
    with pytest.raises(StorageError):
        store.query("notes")


@pytest.mark.parametrize(
    "sql,args,index",
    [
        ("SELECT * FROM records WHERE id=?", ("one",), "sqlite_autoindex_records_1"),
        ("SELECT * FROM records WHERE collection=? AND start_ns>=?", ("notes", 10), "records_collection_time"),
        ("SELECT record_id FROM subjects WHERE subject=?", ("alice",), "subjects_subject_record"),
        ("SELECT * FROM records WHERE collection=? AND logical_key=?", ("notes", "label"), "records_collection_key"),
    ],
)
def test_lookup_shapes_have_real_indexes(store, sql, args, index):
    store.put([record()])
    with sqlite3.connect(store.path) as db:
        plan = db.execute("EXPLAIN QUERY PLAN " + sql, args).fetchall()
    assert index in json.dumps(plan)


# A lock another process holds is waited out, not raised (#2898). The backoff's pause releases the lock, so each
# test knows the store met it and retried.


@pytest.fixture
def quick_locks(monkeypatch):
    monkeypatch.setattr(knowledge, "BUSY_TIMEOUT_S", 0.01)


def hold(path, how="BEGIN EXCLUSIVE"):
    other = sqlite3.connect(path, isolation_level=None, timeout=0)
    other.execute(how)
    return other


def release_on_pause(monkeypatch, other):
    pauses = []

    def pause(seconds):
        pauses.append(seconds)
        if other.in_transaction:
            other.rollback()

    monkeypatch.setattr(knowledge, "_sleep", pause)
    return pauses


def test_a_store_opened_while_another_process_holds_the_lock_waits_for_it(store, quick_locks, monkeypatch):
    other = hold(store.path)
    pauses = release_on_pause(monkeypatch, other)
    assert Store(store.path).revision() == 0
    assert pauses == [knowledge.FIRST_RETRY_S]


def test_a_write_waits_out_another_process_lock_and_lands(store, quick_locks, monkeypatch):
    writer = Store(store.path)
    other = hold(store.path)
    pauses = release_on_pause(monkeypatch, other)
    (position,) = writer.put([record()])
    assert pauses
    assert store.get("one").position == position


def test_a_rollback_journal_commit_waits_for_a_reader_to_finish(tmp_path, quick_locks, monkeypatch):
    monkeypatch.setattr(knowledge, "WAL", False)
    writer = Store(tmp_path / "knowledge.sqlite3")
    reader = hold(writer.path, "BEGIN")
    reader.execute("SELECT count(*) FROM records").fetchone()  # a reader's lock: a commit can't finish under it
    pauses = release_on_pause(monkeypatch, reader)
    writer.put([record()])
    assert pauses
    assert writer.get("one") is not None


def test_a_lock_held_past_the_wait_fails_as_a_storage_error(store, quick_locks, monkeypatch):
    writer = Store(store.path)
    monkeypatch.setattr(knowledge, "LOCK_WAIT_S", 0)
    other = hold(store.path)
    try:
        with pytest.raises(StorageError, match="locked"):
            writer.put([record()])
    finally:
        other.rollback()
    assert store.get("one") is None


def test_wal_converts_a_rollback_store_and_writes_beside_a_reader(tmp_path, monkeypatch):
    path = tmp_path / "knowledge.sqlite3"
    monkeypatch.setattr(knowledge, "WAL", False)
    old = Store(path)
    old.put([record()])
    old.close()
    monkeypatch.setattr(knowledge, "WAL", True)
    store = Store(path)
    reader = hold(path, "BEGIN")
    assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert reader.execute("SELECT count(*) FROM records").fetchone()[0] == 1
    monkeypatch.setattr(knowledge, "_sleep", lambda seconds: pytest.fail("a WAL writer waited for a reader"))
    store.put([record("two")])
    assert reader.execute("SELECT count(*) FROM records").fetchone()[0] == 1  # the reader keeps its snapshot
    reader.rollback()
    assert {r.record.id for r in store.query("notes")} == {"one", "two"}


def test_a_long_held_transaction_is_logged_with_its_caller(store, monkeypatch, caplog):
    monkeypatch.setattr(knowledge, "LONG_HOLD_S", -1.0)  # every transaction counts as long
    with caplog.at_level("WARNING", logger="liberational.knowledge"), store.transaction():
        store.put([record()])
    (line,) = [r.getMessage() for r in caplog.records if "was held" in r.getMessage()]
    assert "write transaction" in line
    assert "test_knowledge.py" in line
    assert "test_a_long_held_transaction_is_logged_with_its_caller" in line


def test_a_rollback_commit_gives_up_after_its_own_shorter_wait(tmp_path, quick_locks, monkeypatch):
    # A writer waiting to commit holds new readers out, so it can't wait as long as an open (LOCK_WAIT_S).
    monkeypatch.setattr(knowledge, "WAL", False)
    writer = Store(tmp_path / "knowledge.sqlite3")
    monkeypatch.setattr(knowledge, "COMMIT_WAIT_S", 0)
    reader = hold(writer.path, "BEGIN")
    reader.execute("SELECT count(*) FROM records").fetchone()
    monkeypatch.setattr(knowledge, "_sleep", lambda seconds: pytest.fail("the commit retried past its wait"))
    try:
        with pytest.raises(StorageError, match="locked"):
            writer.put([record()])
    finally:
        reader.rollback()
    assert writer.get("one") is None


def test_lock_wait_caps_every_wait_in_its_context(store, quick_locks, monkeypatch):
    writer = Store(store.path)
    other = hold(store.path)
    monkeypatch.setattr(knowledge, "_sleep", lambda seconds: pytest.fail("a capped wait retried"))
    try:
        with knowledge.lock_wait(0), pytest.raises(StorageError, match="locked"):
            writer.put([record()])
    finally:
        other.rollback()
    writer.put([record()])  # outside the context the long wait is back, and nothing holds the lock now
    assert store.get("one") is not None


def test_a_commit_wait_is_a_hard_bound_even_with_a_long_busy_timeout(tmp_path, monkeypatch):
    # Each attempt waits inside SQLite at most the time its window has left, not a whole busy timeout past it.
    monkeypatch.setattr(knowledge, "WAL", False)
    writer = Store(tmp_path / "knowledge.sqlite3")  # its connection's own timeout stays the 30 s default
    monkeypatch.setattr(knowledge, "COMMIT_WAIT_S", 0.3)
    reader = hold(writer.path, "BEGIN")
    reader.execute("SELECT count(*) FROM records").fetchone()
    began = time.monotonic()
    try:
        with pytest.raises(StorageError, match="locked"):
            writer.put([record()])
    finally:
        reader.rollback()
    assert time.monotonic() - began < 5  # one 30 s busy wait would blow straight through this


def test_at_reuses_a_store_no_caller_holds(tmp_path, tracked_connections):
    # Callers like see/zones._store() keep no reference; each call must not open and validate the file again (#2940)
    path = tmp_path / "knowledge.sqlite3"
    knowledge.at(path).put([record()])
    gc.collect()
    opened, _ = tracked_connections
    before = len(opened)

    assert [item.record.id for item in knowledge.at(path).query("notes")] == ["one"]
    assert len(opened) == before


def test_at_rebuilds_a_store_whose_file_was_removed(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    knowledge.at(path).put([record()])
    for leftover in tmp_path.iterdir():
        leftover.unlink()

    assert knowledge.at(path).query("notes") == []
    knowledge.at(path).put([record("two")])
    assert [item.record.id for item in knowledge.at(path).query("notes")] == ["two"]


def test_rebuilding_a_removed_store_never_holds_up_other_stores_while_it_waits_for_a_transaction(tmp_path):
    # #3176: at() closed the removed file's store under the lock every at() takes, and close waits for transactions
    # in progress (#3170): until the transaction ended no other store could be opened, a deadlock when the
    # transaction itself opened one
    path, other = tmp_path / "a" / "knowledge.sqlite3", tmp_path / "b" / "knowledge.sqlite3"
    knowledge.at(path).put([record()])
    knowledge.at(other).put([record("elsewhere")])
    inside, done = threading.Event(), threading.Event()
    got: dict[str, object] = {}

    def write():
        with knowledge.at(path).transaction() as s:
            s.put([record("two")])
            inside.set()
            done.wait(10)

    def rebuild():
        got["rebuilt"] = knowledge.at(path)

    def open_other():
        got["other"] = [item.record.id for item in knowledge.at(other).query("notes")]

    # Daemon threads, and the transaction always let go: a regression fails here instead of hanging the run
    writing, rebuilding, opening = (threading.Thread(target=f, daemon=True) for f in (write, rebuild, open_other))
    writing.start()
    assert inside.wait(5)
    for leftover in path.parent.iterdir():
        leftover.unlink()
    rebuilding.start()
    time.sleep(0.3)  # the rebuild is closing the old store now, waiting for the transaction
    opening.start()
    opening.join(timeout=3)
    other_opened = not opening.is_alive()
    done.set()
    for t in (writing, rebuilding, opening):
        t.join(timeout=10)
    assert other_opened, "another store waited for the removed store's transaction"
    assert got["other"] == ["elsewhere"]
    assert not rebuilding.is_alive() and got["rebuilt"] is knowledge.at(path)
    assert knowledge.at(path).query("notes") == []


def test_a_file_s_journaling_is_logged_once_a_process_however_often_it_is_opened(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(knowledge, "_announced", set())
    with caplog.at_level("INFO", logger="liberational.knowledge"):
        Store(tmp_path / "a.sqlite3").close()
        Store(tmp_path / "a.sqlite3").close()  # each Store opened said it again, 26 lines an hour (#3147)
        Store(tmp_path / "b.sqlite3").close()
    lines = [r.getMessage() for r in caplog.records if "journaling" in r.getMessage()]
    assert [line.split(":")[0] for line in lines] == [
        f"knowledge store {tmp_path / 'a.sqlite3'}",
        f"knowledge store {tmp_path / 'b.sqlite3'}",
    ]
