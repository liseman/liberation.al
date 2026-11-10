"""The ingest state store: one SQLite database (`ingest.sqlite3` under `paths.state`) of every item ingested, deduped
by the SHA-256 of its original bytes, plus each source's record of what it has already handed over.

- `items`: one row per distinct original: where it came from (`source`, and the source's own `source_key`), its
  file name, kind (photo, video, audio), original time and where that time came from (`exif`, `ffprobe`, `source`,
  or `upload`), and how far it got (`queued`, `converting`, `registering`, `done`, `failed`) with a `detail`.
  `outputs` is JSON: each converted file with its stream and chunk id.
- `seen`: (source, key) pairs a source has handed over, each pointing at its item, so a poll never fetches the
  same thing twice even when its bytes turned out to be a duplicate of something else.
- `source_state`: a JSON blob per source for cursors and the like.

Every call opens its own connection, so the dashboard's request threads, its ingest worker, and `pack-ingest poll`
can share the file.
"""

import fcntl
import json
import sqlite3
import time
from collections.abc import Generator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    source_key TEXT,
    name TEXT NOT NULL,
    kind TEXT,
    original_ts REAL,
    time_from TEXT,
    added_ts REAL NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    original TEXT NOT NULL,
    stream TEXT NOT NULL,
    outputs TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS items_added ON items (added_ts);
CREATE TABLE IF NOT EXISTS seen (
    source TEXT NOT NULL,
    key TEXT NOT NULL,
    item_id INTEGER,
    at REAL NOT NULL,
    PRIMARY KEY (source, key)
);
CREATE TABLE IF NOT EXISTS source_state (
    source TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deleted_items (
    item_id INTEGER PRIMARY KEY,
    requested_at REAL NOT NULL,
    pending INTEGER NOT NULL DEFAULT 1,
    purge_id TEXT,
    blocked INTEGER NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT ''
);
"""

# The statuses an item can have; PENDING are the ones still being worked on.
PENDING = ("queued", "converting", "registering")
STATUSES = (*PENDING, "done", "failed")


class ItemGone(LookupError):
    """The item isn't there: never added, deleted, or removed while it was being worked on. Its own type, so a stray
    KeyError from a bug is never taken for this."""


@dataclass
class Item:
    """One ingested original."""

    id: int
    sha256: str
    source: str
    name: str
    added_ts: float
    status: str
    original: str
    stream: str
    source_key: str | None = None
    kind: str | None = None
    original_ts: float | None = None
    time_from: str | None = None
    detail: str = ""
    outputs: list[dict[str, Any]] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        """What the Ingest tab shows: everything but the original's path on disk."""
        return {
            "id": self.id,
            "sha256": self.sha256,
            "source": self.source,
            "source_key": self.source_key,
            "name": self.name,
            "kind": self.kind,
            "original_ts": self.original_ts,
            "time_from": self.time_from,
            "added_ts": self.added_ts,
            "status": self.status,
            "detail": self.detail,
            "stream": self.stream,
            "outputs": [{k: v for k, v in o.items() if k != "path"} for o in self.outputs],
        }


def _item(row: sqlite3.Row) -> Item:
    d = dict(row)
    d["outputs"] = json.loads(d["outputs"] or "[]")
    return Item(**d)


class Store:
    """The ingest database at `path` (made on first use)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self) -> Generator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db

    def by_hash(self, sha256: str) -> Item | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM items WHERE sha256 = ?", (sha256,)).fetchone()
        return _item(row) if row else None

    def get(self, item_id: int) -> Item | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return _item(row) if row else None

    def removed(self, item_id: int) -> bool:
        """A durable fence: removed items must never be processed or analyzed again."""
        with self._db() as db:
            return db.execute("SELECT 1 FROM deleted_items WHERE item_id = ?", (item_id,)).fetchone() is not None

    @contextmanager
    def item_lock(self, item_id: int, *, blocking: bool = True) -> Generator[None]:
        """Serialize conversion, analysis, and deletion across processes. Lock files stay to keep one inode."""
        directory = self.path.parent / "ingest-locks"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{item_id}.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def request_delete(self, item_id: int) -> None:
        """Fence an existing item before waiting for its active work to stop; repeat requests are harmless."""
        with self._db() as db:
            db.execute(
                "INSERT OR IGNORE INTO deleted_items(item_id, requested_at) SELECT id, ? FROM items WHERE id = ?",
                (time.time(), item_id),
            )

    def deletion(self, item_id: int) -> dict[str, Any] | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM deleted_items WHERE item_id = ?", (item_id,)).fetchone()
        return dict(row) if row else None

    def deletion_progress(
        self, item_id: int, *, purge_id: str | None = None, detail: str = "", blocked: bool = False
    ) -> None:
        with self._db() as db:
            db.execute(
                "UPDATE deleted_items SET purge_id = COALESCE(?, purge_id), detail = ?, blocked = ? WHERE item_id = ?",
                (purge_id, detail, blocked, item_id),
            )

    def reset_purge(self, item_id: int) -> None:
        """The broker no longer knows a receipt; the next cleanup poll must issue the purge again."""
        with self._db() as db:
            db.execute("UPDATE deleted_items SET purge_id = NULL WHERE item_id = ?", (item_id,))

    def finish_delete(self, item_id: int) -> None:
        """Drop the item, retaining only its opaque id fence so a stale retry cannot delete a later item."""
        with self._db() as db:
            db.execute("DELETE FROM items WHERE id = ?", (item_id,))
            db.execute(
                "UPDATE deleted_items SET pending = 0, purge_id = NULL, detail = '', blocked = 0 WHERE item_id = ?",
                (item_id,),
            )

    def deletions(self) -> list[dict[str, Any]]:
        """Pending cleanup notices, separate from the visible imported items."""
        with self._db() as db:
            rows = db.execute(
                "SELECT d.item_id AS id, i.name, d.detail, d.blocked FROM deleted_items d JOIN items i ON i.id = d.item_id"
                " WHERE d.pending = 1 ORDER BY d.requested_at"
            ).fetchall()
        return [{**dict(r), "blocked": bool(r["blocked"])} for r in rows]

    def add(
        self,
        *,
        sha256: str,
        source: str,
        name: str,
        original: str,
        stream: str,
        source_key: str | None = None,
        added_ts: float | None = None,
    ) -> tuple[Item, bool]:
        """Records a new original as `queued`, or returns the item already holding these bytes: (item, new)."""
        with self._db() as db:
            cur = db.execute(
                "INSERT INTO items (id, sha256, source, source_key, name, added_ts, status, original, stream)"
                " VALUES ((SELECT COALESCE(MAX(id), 0) + 1 FROM"
                " (SELECT MAX(id) AS id FROM items UNION ALL SELECT MAX(item_id) AS id FROM deleted_items)),"
                " ?, ?, ?, ?, ?, 'queued', ?, ?) ON CONFLICT (sha256) DO NOTHING",
                (sha256, source, source_key, name, time.time() if added_ts is None else added_ts, original, stream),
            )
            new = cur.rowcount == 1
            row = db.execute("SELECT * FROM items WHERE sha256 = ?", (sha256,)).fetchone()
        return _item(row), new

    def update(self, item_id: int, **fields: Any) -> Item:
        """Sets some of an item's fields (`outputs` as a list) and returns it as it is now; ItemGone when it isn't there."""
        allowed = {"kind", "original_ts", "time_from", "status", "detail", "outputs", "original"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"not an item field: {sorted(unknown)}")
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"not a status: {fields['status']!r}")
        if "outputs" in fields:
            fields["outputs"] = json.dumps(fields["outputs"])
        with self._db() as db:
            if fields:
                sets = ", ".join(f"{k} = ?" for k in fields)
                db.execute(f"UPDATE items SET {sets} WHERE id = ?", (*fields.values(), item_id))
            row = db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise ItemGone(item_id)
        return _item(row)

    def recent(self, limit: int = 100) -> list[Item]:
        """The newest items first."""
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM items WHERE id NOT IN (SELECT item_id FROM deleted_items)"
                " ORDER BY added_ts DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_item(r) for r in rows]

    def finished(self, *, after_id: int = 0, limit: int = 500) -> list[Item]:
        """Done items with an id above `after_id`, oldest first, at most `limit`.

        Callers page with the last id they saw. `recent` is the newest rows of any status, which hides a done
        item once enough newer rows exist.
        """
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM items WHERE status = 'done' AND id > ?"
                " AND id NOT IN (SELECT item_id FROM deleted_items) ORDER BY id LIMIT ?",
                (after_id, limit),
            ).fetchall()
        return [_item(r) for r in rows]

    def pending(self) -> list[Item]:
        """Items not finished yet (queued, or cut off mid-way by a restart), oldest first."""
        marks = ",".join("?" * len(PENDING))
        with self._db() as db:
            rows = db.execute(
                f"SELECT * FROM items WHERE status IN ({marks})"
                " AND id NOT IN (SELECT item_id FROM deleted_items) ORDER BY id",
                PENDING,
            ).fetchall()
        return [_item(r) for r in rows]

    # ---- per-source state

    def seen(self, source: str, key: str) -> bool:
        with self._db() as db:
            return db.execute("SELECT 1 FROM seen WHERE source = ? AND key = ?", (source, key)).fetchone() is not None

    def mark_seen(self, source: str, key: str, item_id: int | None) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO seen (source, key, item_id, at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT (source, key) DO UPDATE SET item_id = excluded.item_id, at = excluded.at",
                (source, key, item_id, time.time()),
            )

    def seen_count(self, source: str) -> int:
        with self._db() as db:
            return int(db.execute("SELECT COUNT(*) FROM seen WHERE source = ?", (source,)).fetchone()[0])

    def source_state(self, source: str) -> dict[str, Any]:
        with self._db() as db:
            row = db.execute("SELECT data FROM source_state WHERE source = ?", (source,)).fetchone()
        return json.loads(row[0]) if row else {}

    def put_source_state(self, source: str, data: dict[str, Any]) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO source_state (source, data) VALUES (?, ?)"
                " ON CONFLICT (source) DO UPDATE SET data = excluded.data",
                (source, json.dumps(data)),
            )
