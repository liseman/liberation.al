"""An in-memory memory log for tests of `memory_log`'s readers and writers: `memory_log.at` returns it (the `memory`
fixture in conftest.py installs it), and a test adds the records the broker would hold, or reads what code wrote."""

import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from liberational import memory_log


class FakeMemoryLog:
    """Records by stream, written by `append` and read back by `range` as `MemoryLog` does: overlapping a window, in
    time order, every record (superseded ones too)."""

    def __init__(self) -> None:
        self.records: dict[str, list[dict[str, Any]]] = {}
        self.cursors: dict[str, int] = {}  # durable name -> how many records of its stream were handled
        self.state: Any = None  # the state directory the code last asked for its log by, which `page` reviews against
        # As the broker: a parent it never had (a row from the hourly segment files) is refused too. Off by default,
        # since most tests name sources from other fakes without writing them here.
        self.known_parents_only = False

    def append(
        self,
        stream: str,
        record: Mapping[str, Any],
        *,
        parents: Sequence[str] = (),
        subjects: Sequence[str] = (),
        producer: Any = None,
        payload_id: str | None = None,
        span_ns: Any = None,
    ) -> str:
        if stream not in memory_log.STREAMS:
            raise ValueError(f"no stream {stream}")
        extra = set(record) - set(memory_log.STREAMS[stream].keys) - {"ts_start", "ts_end"}
        if extra:
            raise ValueError(f"{stream} has no {sorted(extra)}")  # the codec refuses unknown keys
        purged = {r["id"] for rows in self.records.values() for r in rows if r.get("_purged")}
        known = {r["id"] for rows in self.records.values() for r in rows} if self.known_parents_only else None
        for parent in parents:
            if parent in purged or (known is not None and parent not in known):
                # the catalog refuses a payload naming a parent it no longer has, or never had
                raise ValueError(f"payload names unknown parent {parent}")
        pid = payload_id or str(uuid.uuid7())
        r = {"ts_end": record["ts_start"], **record, "id": pid, "parents": list(parents), "subjects": list(subjects)}
        r["producer"] = producer
        self.records.setdefault(stream, []).append(r)
        return pid

    def range(self, stream: str, start: float, end: float, *, subjects: Iterable[str] = ()) -> list[dict[str, Any]]:
        wanted = set(subjects)
        found = [
            {k: v for k, v in r.items() if k != "_purged"}
            for r in self.records.get(stream, [])
            if not r.get("_purged")
            and r["ts_start"] < end
            and max(r["ts_end"], r["ts_start"]) >= start
            and (not wanted or wanted & set(r["subjects"]))
        ]
        return sorted(found, key=lambda r: (r["ts_start"], r["id"]))

    def append_bytes(
        self,
        stream: str,
        shape: str,
        data: bytes,
        *,
        span: tuple[float, float],
        parents: Sequence[str] = (),
        subjects: Sequence[str] = (),
        producer: Any = None,
    ) -> str:
        pid = str(uuid.uuid7())
        r = {"ts_start": span[0], "ts_end": span[1], "shape": shape, "data": data, "id": pid}
        r.update(parents=list(parents), subjects=list(subjects))
        self.records.setdefault(stream, []).append(r)
        return pid

    def parents_in(self, stream: str, start: float, end: float) -> set[str]:
        return {p for r in self.range(stream, start, end) for p in r["parents"]}

    def entries(
        self,
        start: float,
        end: float,
        level: int | None = None,
        person: str | None = None,
        *,
        include_quiet: bool = False,
        review: Any = None,
    ) -> list[dict[str, Any]]:
        found = memory_log.current(self.range(memory_log.JOURNAL, start, end))
        if not include_quiet:
            found = [r for r in found if not r.get("status")]
        if level is not None:
            found = [r for r in found if r.get("level") == level]
        if person is not None:
            found = [r for r in found if person in r["subjects"]]
        return sorted(found, key=lambda r: (r["ts_start"], r.get("level") or 0, r["id"]))

    def facts(self, person: str) -> list[dict[str, Any]]:
        return self.range(memory_log.FACTS, 0.0, 1e12, subjects=[person])

    def page(self, level: str, period: str) -> dict[str, Any] | None:
        a, _ = memory_log.day_bounds(period)
        found = [
            r
            for r in memory_log.reviewed(memory_log.current(self.range(memory_log.PAGES, a, a + 1.0)), self.state)
            if r.get("level") == level and r.get("period") == period and not r.get("status")
        ]
        return max(found, key=lambda r: str(r["id"])) if found else None

    def mark(self, streams: Iterable[str] = (memory_log.JOURNAL,)) -> int:
        return sum(len(self.records.get(s, [])) for s in streams)

    def drain(self, stream: str, durable: str, handle: Any, *, limit: int, idle_s: float = 0.5) -> int:
        """The records on `stream` after the durable name's position, in the order written, handled on this thread."""
        position = self.cursors.get(durable, 0)
        handled = 0
        rows = self.records.get(stream, [])
        while position < len(rows) and handled < limit:
            if rows[position].get("_purged"):
                position += 1
                self.cursors[durable] = position
                continue
            record = {k: v for k, v in rows[position].items() if k != "_purged"}
            record["position"] = position + 1
            if not handle(record):
                break
            handled += 1
            position += 1
            self.cursors[durable] = position
        return handled

    def purge(self, ids: Iterable[str], reason: str, detail: str = "") -> int:
        """Removes the records `ids` and everything derived from them, as the broker's cascade does."""
        gone = set(ids)
        while True:
            more = {
                r["id"]
                for rows in self.records.values()
                for r in rows
                if r["id"] not in gone and not r.get("_purged") and gone & set(r["parents"])
            }
            if not more:
                break
            gone |= more
        removed = 0
        for rows in self.records.values():
            for r in rows:
                if r["id"] in gone and not r.get("_purged"):
                    r["_purged"] = True  # a tombstone keeps the stream's positions where they were
                    removed += 1
        return removed

    def close(self) -> None:
        pass
