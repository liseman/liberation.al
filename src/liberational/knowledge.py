"""Indexed, low-rate knowledge independent of the broker catalog.

Records are immutable. Exact retries retain their position; conflicting IDs and erased IDs cannot be reused.
Every content write and erasure advances the durable change feed in the same transaction. This module stores
neither sensor samples nor native broker headers. Domain adapters own their records' meaning and validation.

The database lives directly under paths.state, outside continuum/. `at` keeps one Store per path for the process,
and each Store reuses one connection per thread. Transactions remain short-lived; `transaction` groups a worker's
related writes and nested reads into one commit. Connections detect file replacement before reuse. The store
journals in WAL, so readers never block the writer, when SQLite is 3.51.3 or newer; older SQLite has the WAL-reset
defect (a checkpoint racing a writer can corrupt the file), so there it keeps rollback journaling. Both use FULL
synchronization. A WAL store checkpoints and truncates its WAL after each write (`_fold`).
A lock held by another process is waited out: SQLite's busy timeout, then retries with backoff for up to
LOCK_WAIT_S, so a slow peer delays a service rather than stopping it. A COMMIT waits at most COMMIT_WAIT_S,
because a rollback-journal writer waiting to commit holds new readers out. `lock_wait` caps all of it for a context,
so a dashboard request fails fast instead of piling up behind a lock.
Thread-local holders close connections when their thread exits; the Store tracks holders weakly so short-lived
dashboard request threads cannot accumulate open databases. Persistent workers continue to reuse their handle.
"""

import contextvars
import json
import logging
import math
import os
import sqlite3
import threading
import time
import traceback
import uuid
import weakref
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

FORMAT = 1
APPLICATION_ID = 0x4C504B4E
BUSY_TIMEOUT_S = 30.0  # one wait inside SQLite for another connection's lock
LOCK_WAIT_S = 300.0  # how long a locked open or BEGIN keeps retrying before it fails
REQUEST_WAIT_S = 10.0  # a dashboard request's whole wait (`lock_wait`): it answers with an error, not a queue
COMMIT_WAIT_S = 60.0  # a COMMIT's whole wait: in rollback journaling, readers can't start while it waits
FIRST_RETRY_S = 0.1  # the backoff doubles from here up to MAX_RETRY_S
MAX_RETRY_S = 5.0
CHECKPOINT_WAIT_S = 1.0  # a checkpoint after a write waits this long for readers, then leaves it for the next write
LONG_HOLD_S = 5.0  # a transaction held longer than this is logged with its caller, to find who blocks the others
WAL = sqlite3.sqlite_version_info >= (3, 51, 3)  # the first release without the WAL-reset defect
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_TIME_NS = 2**63 - 1


class Conflict(ValueError):
    """An ID already identifies different immutable content."""


class Deleted(ValueError):
    """An erased ID cannot be written again, including by a late producer."""


class StorageError(RuntimeError):
    """The knowledge database cannot be read safely."""


_log = logging.getLogger("liberational.knowledge")
_announced: set[str] = set()  # the files whose journaling this process has logged
_announced_lock = threading.Lock()
_sleep = time.sleep  # the backoff's pause (tests release a held lock here)
_wait_cap: contextvars.ContextVar[float | None] = contextvars.ContextVar("knowledge_lock_wait", default=None)


@contextmanager
def lock_wait(seconds: float) -> Generator[None]:
    """Cap every knowledge lock wait in this context (busy timeout and retries together) at `seconds`. For request
    threads, which should answer with an error rather than queue behind a lock."""
    token = _wait_cap.set(seconds)
    try:
        yield
    finally:
        _wait_cap.reset(token)


def _budget(longest: float) -> float:
    """`longest`, or this context's `lock_wait` cap when that is shorter."""
    cap = _wait_cap.get()
    return longest if cap is None else max(0.0, min(longest, cap))


def _locked(error: sqlite3.Error) -> bool:
    """Whether `error` is another connection's lock (SQLITE_BUSY or SQLITE_LOCKED, extended codes included)."""
    code = getattr(error, "sqlite_errorcode", None)
    if not isinstance(error, sqlite3.OperationalError) or code is None:
        return False
    return code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


def _retry[T](attempt: Callable[[float], T], wait_s: float, what: str) -> T:
    """`attempt(left)`, again with backoff while another connection holds the lock, for up to `wait_s`; `left` is
    the time remaining, which the attempt caps its own busy timeout at, so `wait_s` is a hard bound. Logs the first
    wait and how it ended, not every retry."""
    began = time.monotonic()
    pause = FIRST_RETRY_S
    waited = False
    while True:
        try:
            result = attempt(max(0.0, began + wait_s - time.monotonic()))
        except sqlite3.OperationalError as error:
            remaining = began + wait_s - time.monotonic()
            if not _locked(error):
                raise
            if remaining <= 0:
                if waited:
                    _log.warning("knowledge %s gave up on the lock after %.1f s", what, time.monotonic() - began)
                raise
            if not waited:
                _log.warning("knowledge database is locked (%s); the %s waits up to %.0f s", error, what, wait_s)
                waited = True
            _sleep(min(pause, remaining))
            pause = min(pause * 2, MAX_RETRY_S)
        else:
            if waited:
                _log.info("knowledge %s got the lock after %.1f s", what, time.monotonic() - began)
            return result


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def _integer(value: object, field: str, *, maximum: int = MAX_TIME_NS) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [0, {maximum}]")
    return value


def _json(value: Any) -> str:
    """Validate JSON before a transaction; never normalize nonfinite values or non-string object keys."""

    def check(item: Any) -> None:
        if item is None or isinstance(item, str | bool | int):
            return
        if isinstance(item, float) and math.isfinite(item):
            return
        if isinstance(item, list | tuple):
            for child in item:
                check(child)
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError("knowledge values must be finite JSON with string object keys")

    check(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    if len(encoded.encode()) > MAX_JSON_BYTES:
        raise ValueError("knowledge record exceeds JSON byte limit")
    return encoded


@dataclass(frozen=True)
class Record:
    id: str
    collection: str
    key: str | None
    start_ns: int
    end_ns: int
    recorded_ns: int
    value: dict[str, Any]
    subjects: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoredRecord:
    position: int
    record: Record


@dataclass(frozen=True)
class Change:
    position: int
    operation: str
    collection: str
    id: str


@dataclass(frozen=True)
class Snapshot:
    revision: int
    records: tuple[StoredRecord, ...]
    erased: tuple[str, ...]


_SCHEMA = (
    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT",
    (
        "CREATE TABLE changes (position INTEGER PRIMARY KEY AUTOINCREMENT, operation TEXT NOT NULL "
        "CHECK(operation IN ('put','remove')), collection TEXT NOT NULL, id TEXT NOT NULL) STRICT"
    ),
    "CREATE INDEX changes_collection_position ON changes(collection, position)",
    (
        "CREATE TABLE records (id TEXT PRIMARY KEY, collection TEXT NOT NULL, logical_key TEXT, "
        "start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL, recorded_ns INTEGER NOT NULL, value TEXT NOT NULL, "
        "position INTEGER NOT NULL UNIQUE REFERENCES changes(position)) STRICT"
    ),
    "CREATE INDEX records_collection_time ON records(collection, start_ns, end_ns, position)",
    "CREATE INDEX records_collection_position ON records(collection, position)",
    "CREATE INDEX records_collection_key ON records(collection, logical_key, position)",
    (
        "CREATE TABLE subjects (record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE, "
        "subject TEXT NOT NULL, PRIMARY KEY(record_id, subject)) WITHOUT ROWID, STRICT"
    ),
    "CREATE INDEX subjects_subject_record ON subjects(subject, record_id)",
    "CREATE TABLE erased (id TEXT PRIMARY KEY) STRICT",
    "CREATE TABLE cursors (consumer TEXT PRIMARY KEY, position INTEGER NOT NULL) STRICT",
)


class _Handle:
    """TLS owns this holder; its finalizer retains only the connection, never the holder or its Store."""

    def __init__(self, identity: tuple[int, int], connection: sqlite3.Connection) -> None:
        self.identity = identity
        self.connection = connection
        self._finalizer = weakref.finalize(self, connection.close)

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        self._finalizer()


class Store:
    """A small durable store. Missing files initialize; existing incompatible or corrupt files fail explicitly."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._handles: weakref.WeakSet[_Handle] = weakref.WeakSet()
        self._handles_lock = threading.Lock()
        # Threads inside a transaction, and closes waiting for them: closing a connection another thread is using
        # crashes the process inside SQLite, so close waits for them and new transactions wait for close.
        self._use = threading.Condition()
        self._users = 0
        self._closing = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with self._connection(write=True, initialize=True) as db:
            application = db.execute("PRAGMA application_id").fetchone()[0]
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if application == APPLICATION_ID and version == FORMAT:
                self._check_schema(db)
            else:
                tables = db.execute("SELECT name FROM sqlite_schema WHERE type='table'").fetchall()
                if application or version or tables:
                    raise StorageError("unrecognized knowledge database format")
                for statement in _SCHEMA:
                    db.execute(statement)
                db.execute("INSERT INTO metadata VALUES ('incarnation', ?)", (str(uuid.uuid4()),))
                db.execute(f"PRAGMA application_id={APPLICATION_ID}")
                db.execute(f"PRAGMA user_version={FORMAT}")
        with _announced_lock:  # once a process per file: each Store opened said it again, 26 lines an hour
            first = str(self.path) not in _announced
            _announced.add(str(self.path))
        if first:
            _log.info("knowledge store %s: %s journaling, SQLite %s", self.path, _journal(), sqlite3.sqlite_version)

    @staticmethod
    def _check_schema(db: sqlite3.Connection) -> None:
        actual = {
            row[0]: " ".join(row[1].split()).casefold()
            for row in db.execute(
                "SELECT name,sql FROM sqlite_schema WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
            )
        }
        expected = {statement.split()[2]: " ".join(statement.split()).casefold() for statement in _SCHEMA}
        if actual != expected:
            raise StorageError("knowledge database schema does not match its format")

    @contextmanager
    def _connection(self, *, write: bool = False, initialize: bool = False) -> Generator[sqlite3.Connection]:
        try:
            active = getattr(self._local, "active", None)
            if active is not None:
                if write and not self._local.write:
                    raise StorageError("cannot write inside a knowledge read transaction")
                yield active
                return
            self._enter()
            try:
                db = _retry(
                    lambda left: self._begin(left, write=write, initialize=initialize), _budget(LOCK_WAIT_S), "open"
                )
            except BaseException:
                self._leave()
                raise
            self._local.active, self._local.write = db, write
            began, committing = time.monotonic(), None
            try:
                yield db
                if initialize:
                    self._local.schema = db.execute("PRAGMA schema_version").fetchone()[0]
                committing = time.monotonic()
                # A rollback-journal commit waits for readers; a busy one stays open, so retrying it applies it once
                _retry(lambda left: _commit(db, left), _budget(COMMIT_WAIT_S), "commit")
                if write and WAL:
                    _fold(db)
            except BaseException:
                db.rollback()
                self._local.generations = {}
                raise
            finally:
                self._local.active = None
                self._leave()
                ended = time.monotonic()
                held = (committing or ended) - began  # the commit's own wait is reported apart
                commit_wait = ended - committing if committing is not None else 0.0
                if held > LONG_HOLD_S or commit_wait > LONG_HOLD_S:
                    caller = next(
                        (
                            f
                            for f in reversed(traceback.extract_stack(limit=12))
                            if f.filename != __file__ and not f.filename.endswith("contextlib.py")
                        ),
                        traceback.extract_stack(limit=1)[0],
                    )
                    _log.warning(
                        "a knowledge %s transaction was held %.1f s and its commit waited %.1f s (thread %s, from %s:%d %s)",
                        "write" if write else "read",
                        held,
                        commit_wait,
                        threading.current_thread().name,
                        caller.filename,
                        caller.lineno or 0,
                        caller.name,
                    )
        except sqlite3.DatabaseError as error:
            raise StorageError(f"knowledge database operation failed: {error}") from error

    def _begin(self, left: float, *, write: bool, initialize: bool) -> sqlite3.Connection:
        """This thread's connection inside a new transaction, its lock taken (one attempt; `_retry` repeats it), waiting
        inside SQLite at most `left` seconds."""
        db = self._handle(left)
        _busy(db, left)
        mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        if mode != _journal():
            raise StorageError(f"knowledge database requires {_journal()} journaling, not {mode}")
        db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            if not initialize:  # the first read takes a reader's lock, so a locked store fails here, not later
                application = db.execute("PRAGMA application_id").fetchone()[0]
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if application != APPLICATION_ID or version != FORMAT:
                    raise StorageError("knowledge database format changed")
                schema = db.execute("PRAGMA schema_version").fetchone()[0]
                if schema != getattr(self._local, "schema", None):
                    self._check_schema(db)
                    self._local.schema = schema
        except BaseException:
            db.rollback()
            raise
        return db

    def _handle(self, left: float) -> sqlite3.Connection:
        try:
            stat = self.path.stat()
        except FileNotFoundError as error:
            raise StorageError("knowledge database was removed") from error
        identity = (stat.st_dev, stat.st_ino)
        previous = getattr(self._local, "handle", None)
        with self._handles_lock:
            if previous is not None:
                if previous.identity == identity and not previous.closed:
                    return previous.connection
                previous.close()
                self._handles.discard(previous)
            while True:
                db = sqlite3.connect(
                    self.path, timeout=min(_budget(BUSY_TIMEOUT_S), left), isolation_level=None, check_same_thread=False
                )
                try:
                    opened = self.path.stat()
                except FileNotFoundError as error:
                    db.close()
                    raise StorageError("knowledge database was removed while opening") from error
                current = (opened.st_dev, opened.st_ino)
                if current == identity:
                    break
                # A replacement raced connection creation; do not attach the previous file's epoch to this handle.
                db.close()
                identity = current
            try:
                # Persistent in the file; switching takes a brief exclusive lock, so a busy store retries this open.
                mode = db.execute(f"PRAGMA journal_mode={_journal()}").fetchone()[0]
            except BaseException:
                db.close()
                raise
            if mode != _journal():  # a WAL store another connection holds open, met by a pre-3.51.3 SQLite
                db.close()
                raise StorageError(f"knowledge database requires {_journal()} journaling, not {mode}")
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA secure_delete=ON")
            db.execute("PRAGMA synchronous=FULL")
            handle = _Handle(identity, db)
            self._handles.add(handle)
            self._local.handle = handle
            self._local.schema = None
            self._local.generations = {}
            return db

    @contextmanager
    def transaction(self) -> Generator[Store]:
        """Group this thread's related operations in one atomic write; let failures escape to roll back the batch.

        Never retain this context across an await or share its connection with another thread. Nested Store calls
        use the same transaction. Other threads and processes see only its committed result.
        """
        with self._connection(write=True):
            yield self

    @staticmethod
    def _prepare(record: Record) -> tuple[Record, str]:
        if not isinstance(record, Record):
            raise TypeError("knowledge holds Record values")
        _text(record.id, "id")
        _text(record.collection, "collection")
        if record.key is not None:
            _text(record.key, "key")
        _integer(record.start_ns, "start_ns")
        _integer(record.end_ns, "end_ns")
        _integer(record.recorded_ns, "recorded_ns")
        if record.end_ns < record.start_ns:
            raise ValueError("end_ns precedes start_ns")
        if not isinstance(record.value, dict):
            raise ValueError("record value must be an object")
        subjects = tuple(sorted({_text(subject, "subject") for subject in record.subjects}))
        encoded = _json(record.value)
        return replace(record, value=json.loads(encoded), subjects=subjects), encoded

    @staticmethod
    def _stored(db: sqlite3.Connection, row: sqlite3.Row) -> StoredRecord:
        subjects = tuple(
            r[0] for r in db.execute("SELECT subject FROM subjects WHERE record_id=? ORDER BY subject", (row["id"],))
        )
        return StoredRecord(
            row["position"],
            Record(
                row["id"],
                row["collection"],
                row["logical_key"],
                row["start_ns"],
                row["end_ns"],
                row["recorded_ns"],
                json.loads(row["value"]),
                subjects,
            ),
        )

    def _put(self, db: sqlite3.Connection, record: Record, encoded: str) -> int:
        if db.execute("SELECT 1 FROM erased WHERE id=?", (record.id,)).fetchone():
            raise Deleted("knowledge record was erased")
        old = db.execute("SELECT * FROM records WHERE id=?", (record.id,)).fetchone()
        if old is not None:
            if old["value"] != encoded or self._stored(db, old).record != record:
                raise Conflict("knowledge ID already identifies different content")
            return old["position"]
        change = db.execute(
            "INSERT INTO changes(operation,collection,id) VALUES ('put',?,?)", (record.collection, record.id)
        )
        position = int(change.lastrowid or 0)
        db.execute(
            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?)",
            (
                record.id,
                record.collection,
                record.key,
                record.start_ns,
                record.end_ns,
                record.recorded_ns,
                encoded,
                position,
            ),
        )
        db.executemany("INSERT INTO subjects VALUES (?,?)", [(record.id, subject) for subject in record.subjects])
        return position

    def put(self, records: Iterable[Record]) -> list[int]:
        """Atomically append a validated batch; an exact retry retains its original position."""
        prepared = [self._prepare(record) for record in records]
        with self._connection(write=True) as db:
            return [self._put(db, record, encoded) for record, encoded in prepared]

    def amend(self, records: Iterable[Record]) -> list[int]:
        """Atomically rewrite existing records' key and value in place, each as a new `put` change, keeping their ids,
        times, and subjects, so whatever refers to a record by id still finds it. Only for an owner-run migration
        with the services stopped (a camera rename, camera_rename.py): records are otherwise immutable, and `put`
        refuses new content under an existing id. A record that isn't stored raises KeyError and amends nothing."""
        prepared = [self._prepare(record) for record in records]
        positions: list[int] = []
        with self._connection(write=True) as db:
            for record, encoded in prepared:
                old = db.execute("SELECT * FROM records WHERE id=?", (record.id,)).fetchone()
                if old is None:
                    raise KeyError(f"no knowledge record {record.id}")
                was = self._stored(db, old).record
                if (record.collection, record.start_ns, record.end_ns, record.recorded_ns, record.subjects) != (
                    was.collection,
                    was.start_ns,
                    was.end_ns,
                    was.recorded_ns,
                    was.subjects,
                ):
                    raise ValueError("an amendment changes only a record's key and value")
                change = db.execute(
                    "INSERT INTO changes(operation,collection,id) VALUES ('put',?,?)", (record.collection, record.id)
                )
                position = int(change.lastrowid or 0)
                db.execute(
                    "UPDATE records SET logical_key=?, value=?, position=? WHERE id=?",
                    (record.key, encoded, position, record.id),
                )
                positions.append(position)
        return positions

    def get(self, id: str) -> StoredRecord | None:
        with self._connection() as db:
            row = db.execute("SELECT * FROM records WHERE id=?", (_text(id, "id"),)).fetchone()
            return self._stored(db, row) if row is not None else None

    def snapshot(self) -> Snapshot:
        """One consistent low-rate record cut, including erasure barriers, for an explicit portable export."""
        with self._connection() as db:
            revision = int(db.execute("SELECT COALESCE(MAX(position),0) FROM changes").fetchone()[0])
            records = tuple(
                self._stored(db, row) for row in db.execute("SELECT * FROM records ORDER BY position").fetchall()
            )
            erased = tuple(row[0] for row in db.execute("SELECT id FROM erased ORDER BY id"))
            return Snapshot(revision, records, erased)

    def restore(self, records: Iterable[Record], erased: Iterable[str] = ()) -> list[int]:
        """Atomically import validated knowledge. Erasures win and conflicting immutable IDs abort the import."""
        prepared = [self._prepare(record) for record in records]
        tombstones = sorted({_text(id, "erased id") for id in erased})
        if {record.id for record, _ in prepared} & set(tombstones):
            raise ValueError("an import cannot contain a record and its erasure")
        with self._connection(write=True) as db:
            self._remove(db, tombstones, None)
            for id in tombstones:
                inserted = db.execute("INSERT OR IGNORE INTO erased VALUES (?)", (id,)).rowcount
                if inserted:
                    db.execute("INSERT INTO changes(operation,collection,id) VALUES ('remove','_erased',?)", (id,))
            return [self._put(db, record, encoded) for record, encoded in prepared]

    def query(
        self,
        collection: str,
        *,
        after: int = 0,
        through: int | None = None,
        limit: int | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
        subjects: Iterable[str] = (),
        key: str | None = None,
    ) -> list[StoredRecord]:
        """Records in append order, optionally intersecting a half-open time window or any given subject."""
        terms = ["r.collection=?", "r.position>?"]
        parameters: list[Any] = [_text(collection, "collection"), _integer(after, "after")]
        if through is not None:
            terms.append("r.position<=?")
            parameters.append(_integer(through, "through"))
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be a positive integer")
        if key is not None:
            terms.append("r.logical_key=?")
            parameters.append(_text(key, "key"))
        if start_ns is not None:
            terms.append("(r.end_ns>? OR (r.start_ns=r.end_ns AND r.start_ns>=?))")
            parameters.extend([_integer(start_ns, "start_ns")] * 2)
        if end_ns is not None:
            terms.append("r.start_ns<?")
            parameters.append(_integer(end_ns, "end_ns"))
        if start_ns is not None and end_ns is not None and end_ns <= start_ns:
            raise ValueError("query end must follow start")
        wanted = sorted({_text(subject, "subject") for subject in subjects})
        if wanted:
            terms.append(f"r.id IN (SELECT record_id FROM subjects WHERE subject IN ({','.join('?' for _ in wanted)}))")
            parameters.extend(wanted)
        sql = "SELECT r.* FROM records r WHERE " + " AND ".join(terms) + " ORDER BY r.position"
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        with self._connection() as db:
            return [self._stored(db, row) for row in db.execute(sql, parameters).fetchall()]

    def collections(self, prefix: str = "") -> list[str]:
        if not isinstance(prefix, str):
            raise ValueError("prefix must be text")
        with self._connection() as db:
            rows = db.execute("SELECT DISTINCT collection FROM records ORDER BY collection").fetchall()
            return [row[0] for row in rows if row[0].startswith(prefix)]

    def _remove(self, db: sqlite3.Connection, wanted: Iterable[str], audit: Record | None) -> int:
        removed = 0
        for id in wanted:
            row = db.execute("SELECT collection FROM records WHERE id=?", (id,)).fetchone()
            if row is None:
                continue
            db.execute("INSERT INTO changes(operation,collection,id) VALUES ('remove',?,?)", (row[0], id))
            db.execute("INSERT INTO erased VALUES (?)", (id,))
            db.execute("DELETE FROM records WHERE id=?", (id,))
            removed += 1
        if removed and audit is not None:
            record, encoded = self._prepare(replace(audit, value={**audit.value, "removed": removed}))
            self._put(db, record, encoded)
        return removed

    def remove(self, ids: Iterable[str], *, audit: Record | None = None) -> int:
        """Erase records and prevent late ID reuse, recording actual removal count in an optional safe audit."""
        wanted = sorted({_text(id, "id") for id in ids})
        prepared = self._prepare(audit)[0] if audit is not None else None
        with self._connection(write=True) as db:
            return self._remove(db, wanted, prepared)

    def remove_matching(
        self,
        collections: Iterable[str] = (),
        *,
        collection_prefix: str | None = None,
        subjects: Iterable[str] = (),
        ids: Iterable[str] = (),
        before_ns: int | None = None,
        predicate: Callable[[Record], bool] | None = None,
        audit: Record | None = None,
    ) -> int:
        """Select and erase under one write lock, including collections created by concurrent producers.

        The optional predicate must inspect its supplied record only; it must not call the store or perform I/O.
        At least one collection or a nonempty collection prefix is required to bound the deletion domain.
        """
        names = sorted({_text(name, "collection") for name in collections})
        terms: list[str] = []
        parameters: list[Any] = []
        if names:
            terms.append(f"r.collection IN ({','.join('?' for _ in names)})")
            parameters.extend(names)
        if collection_prefix is not None:
            _text(collection_prefix, "collection_prefix")
            terms.append("substr(r.collection,1,length(?))=?")
            parameters.extend([collection_prefix, collection_prefix])
        if not terms:
            raise ValueError("matching removal requires a collection boundary")
        wanted_subjects = sorted({_text(subject, "subject") for subject in subjects})
        if wanted_subjects:
            terms.append(
                f"r.id IN (SELECT record_id FROM subjects WHERE subject IN ({','.join('?' for _ in wanted_subjects)}))"
            )
            parameters.extend(wanted_subjects)
        wanted_ids = sorted({_text(id, "id") for id in ids})
        if wanted_ids:
            terms.append(f"r.id IN ({','.join('?' for _ in wanted_ids)})")
            parameters.extend(wanted_ids)
        if before_ns is not None:
            terms.append("r.start_ns<?")
            parameters.append(_integer(before_ns, "before_ns"))
        prepared = self._prepare(audit)[0] if audit is not None else None
        with self._connection(write=True) as db:
            rows = db.execute("SELECT r.* FROM records r WHERE " + " AND ".join(terms), parameters).fetchall()
            selected = [row["id"] for row in rows if predicate is None or predicate(self._stored(db, row).record)]
            return self._remove(db, selected, prepared)

    def changes(self, after: int = 0, *, limit: int = 1000, collections: Iterable[str] = ()) -> list[Change]:
        _integer(after, "after")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be positive")
        wanted = sorted({_text(name, "collection") for name in collections})
        terms = ["position>?"]
        parameters: list[Any] = [after]
        if wanted:
            terms.append(f"collection IN ({','.join('?' for _ in wanted)})")
            parameters.extend(wanted)
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM changes WHERE " + " AND ".join(terms) + " ORDER BY position LIMIT ?",
                [*parameters, limit],
            ).fetchall()
            return [Change(row["position"], row["operation"], row["collection"], row["id"]) for row in rows]

    def revision(self, collections: Iterable[str] = ()) -> int:
        wanted = sorted({_text(name, "collection") for name in collections})
        suffix = f" WHERE collection IN ({','.join('?' for _ in wanted)})" if wanted else ""
        with self._connection() as db:
            return int(db.execute("SELECT COALESCE(MAX(position),0) FROM changes" + suffix, wanted).fetchone()[0])

    def generation(self, collections: Iterable[str]) -> tuple[str, int, int]:
        """Physical database epoch, relevant change revision, and deletion revision in one consistent read.

        Forks and backups retain metadata incarnation. Include the connection's captured file identity so an
        atomic replacement with equal counters still invalidates projections and reference galleries. Never stat
        the path after reading: it might already name a different database from the one supplying this snapshot.
        """
        wanted = sorted({_text(name, "collection") for name in collections})
        if not wanted:
            raise ValueError("generation requires a collection boundary")
        with self._connection() as db:
            metadata = db.execute("SELECT value FROM metadata WHERE key='incarnation'").fetchone()[0]
            device, inode = self._local.handle.identity
            incarnation = str(uuid.uuid5(uuid.NAMESPACE_URL, f"knowledge:{metadata}:{device}:{inode}"))
            revision = max(
                db.execute("SELECT COALESCE(MAX(position),0) FROM changes WHERE collection=?", (name,)).fetchone()[0]
                for name in wanted
            )
            cache = getattr(self._local, "generations", {})
            previous = cache.get(tuple(wanted), (incarnation, 0, 0))
            if previous[0] != incarnation or previous[1] > revision:
                previous = (incarnation, 0, 0)
            removed = previous[2]
            if revision > previous[1]:
                # Inspect only new relevant changes; stable refreshes never rescan the assertion history.
                removed = max(
                    removed,
                    db.execute(
                        "SELECT COALESCE(MAX(position),0) FROM changes WHERE position>? AND operation='remove' "
                        f"AND collection IN ({','.join('?' for _ in wanted)})",
                        [previous[1], *wanted],
                    ).fetchone()[0],
                )
            self._local.generations = cache
            cache[tuple(wanted)] = (incarnation, revision, removed)
            return incarnation, revision, removed

    def head(self) -> int:
        """A shared inclusive position cut, including removals, for finite cross-collection drains."""
        return self.revision()

    def wait(self, after: int, *, timeout_s: float = 1.0, collections: Iterable[str] = ()) -> list[Change]:
        """Wait for committed cross-process changes without retaining a read transaction or missing erasures."""
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("timeout_s must be finite and nonnegative")
        names = tuple(collections)
        deadline = time.monotonic() + timeout_s
        pause = threading.Event()
        while True:
            found = self.changes(after, collections=names)
            if found:
                return found
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            pause.wait(min(0.05, remaining))

    def ack(self, consumer: str, position: int) -> None:
        with self._connection(write=True) as db:
            db.execute(
                "INSERT INTO cursors VALUES (?,?) ON CONFLICT(consumer) DO UPDATE SET "
                "position=MAX(position,excluded.position)",
                (_text(consumer, "consumer"), _integer(position, "position")),
            )

    def cursor(self, consumer: str) -> int:
        with self._connection() as db:
            row = db.execute("SELECT position FROM cursors WHERE consumer=?", (_text(consumer, "consumer"),)).fetchone()
            return row[0] if row is not None else 0

    def _enter(self) -> None:
        with self._use:
            self._use.wait_for(lambda: not self._closing)
            self._users += 1

    def _leave(self) -> None:
        with self._use:
            self._users -= 1
            self._use.notify_all()

    def close(self) -> None:
        """Release cached handles once every transaction in progress has finished. Later operations reconnect.

        A transaction another thread starts meanwhile waits for the close. Closing inside this thread's own
        transaction would wait for itself, so it raises instead.
        """
        if getattr(self._local, "active", None) is not None:
            raise StorageError("cannot close a knowledge store inside its own transaction")
        with self._use:
            self._closing += 1
            try:
                self._use.wait_for(lambda: not self._users)
                with self._handles_lock:
                    for handle in list(self._handles):
                        handle.close()
                    self._handles.clear()
            finally:
                self._closing -= 1
                self._use.notify_all()


def _busy(db: sqlite3.Connection, left: float) -> None:
    """Wait inside SQLite at most the busy timeout, this context's cap, and the `left` of the caller's retry window."""
    db.execute(f"PRAGMA busy_timeout={round(min(_budget(BUSY_TIMEOUT_S), left) * 1000)}")


def _commit(db: sqlite3.Connection, left: float) -> None:
    _busy(db, left)
    db.commit()


def _fold(db: sqlite3.Connection) -> None:
    """Checkpoint a committed write into the database file and truncate the WAL. Erased content then stays nowhere but
    zeroed pages, and a WAL left behind never holds frames to replay onto a file put in this one's place (SQLite won't
    checkpoint a WAL whose database was moved). A reader still on an older snapshot leaves it for the next write."""
    try:
        db.execute(f"PRAGMA busy_timeout={round(_budget(CHECKPOINT_WAIT_S) * 1000)}")
        busy, _, _ = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.DatabaseError as error:  # the write committed; the next one checkpoints again
        _log.warning("knowledge WAL checkpoint failed: %s", error)
    else:
        if busy:
            _log.debug("knowledge WAL checkpoint deferred: a reader holds an older snapshot")


def _journal() -> str:
    return "wal" if WAL else "delete"


# Strong, so a caller that drops its Store between calls still reuses it: a weak cache rebuilt it on every such call,
# each rebuild taking the write lock to check the schema.
_stores: dict[str, Store] = {}
_stores_lock = threading.Lock()


def at(path: str | Path) -> Store:
    """Reuse the validated store for a database path for the life of the process; its connections and transactions
    remain thread-local. A store whose file was removed (a state reset) is rebuilt, which initializes a new file. The
    old one is closed after the lock is let go: its close waits for its transactions in progress, and one of those
    may be opening another store here."""
    key = str(Path(path).resolve())
    stale = None
    with _stores_lock:
        store = _stores.get(key)
        if store is None or not store.path.exists():
            stale = store
            store = Store(key)
            _stores[key] = store
    if stale is not None:
        stale.close()
    return store


def close_all() -> None:
    """Close and forget every cached store, so the next `at` validates its file again (tests start each one fresh)."""
    with _stores_lock:
        stores = list(_stores.values())
        _stores.clear()
    for store in stores:
        store.close()
