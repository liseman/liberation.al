"""Local Ask exchanges on Continuum, with answer provenance and no second history store.

An independent question root survives a cited source purge. Final metadata, also parented only by that root,
keeps safe moment references and a manifest. Answer JSON is split into bounded inline chunks parented by the
question and every available canonical source ID. A source purge therefore removes the copied answer, while
small per-citation markers identify which references were removed. Metadata never copies source quotations.
Deleting the root cascades through every version; a late answer cannot re-create the missing parent.
"""

import base64
import binascii
import json
import math
import re
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from liberational import memory_log
from liberational.listen import revisions

if TYPE_CHECKING:
    from continuum_py import Client

_END = 18_446_744_073.0  # within the catalog's u64 nanosecond range
_WINDOW_S = 86_400.0  # list_entries reads back a day at a time, then by widening multiples of it
_CHUNK = 32 * 1024  # base64 plus the JSON envelope remains below the 64 KiB inline limit
_SURFACES = frozenset({"live", "review", "call", "voice"})
_SOURCES = frozenset({"transcript", "archive", "camera", "call", "journal"})


def _canonical(value: object) -> str | None:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError, TypeError, AttributeError:
        return None


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value < _END:
        return float(value)
    return None


def _source_ids(evidence: Mapping[str, Any]) -> list[str]:
    values = [evidence.get("payload_id"), evidence.get("interpretation")]
    interpretation = evidence.get("interpretation")
    if isinstance(interpretation, Mapping):
        values.extend((interpretation.get("payload_id"), interpretation.get("id")))
    ids = evidence.get("payload_ids")
    if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
        values.extend(ids)
    return list(dict.fromkeys(pid for value in values if (pid := _canonical(value)) is not None))


def _citations(result: Mapping[str, Any]) -> list[tuple[dict[str, Any], list[str]]]:
    rows = result.get("evidence") or result.get("citations") or []
    if not isinstance(rows, list):
        return []
    cited = set(re.findall(r"\[([A-Za-z][A-Za-z0-9_-]*)\]", str(result.get("answer") or "")))
    citations = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        label = str(row.get("label") or row.get("id") or "")
        camera = str(row.get("camera") or row.get("cam") or "")
        source = str(row.get("source") or "")
        citation = {
            "label": label if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", label) else "",
            "ts": _number(row.get("ts")),
            "camera": camera if re.fullmatch(r"[A-Za-z0-9_.:-]{0,128}", camera) else "",
            "source": source if source in _SOURCES else "unknown",
        }
        if citation["label"] in cited:
            ids = _source_ids(row)
            citation["verified"] = bool(ids)
            citations.append((citation, ids))
    return citations


def _gone(error: BaseException) -> bool:
    """Only missing provenance is a removed answer; storage/connection failures remain actionable."""
    import continuum_py

    return (
        (
            isinstance(error, continuum_py.BrokerError)
            and bool(error.args)
            and error.args[0] in {"NotFound", "BadRequest"}
            and "parent" in str(error).lower()
        )
        or isinstance(error, ValueError)
        and "unknown parent" in str(error).lower()
    )


UNKNOWN_PARENT = re.compile(r"unknown parent ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")


def _unknown_parent(error: BaseException) -> str | None:
    """The source id the broker refused as a parent because it doesn't have it, or None for any other error."""
    found = UNKNOWN_PARENT.search(str(error)) if _gone(error) else None
    return found[1] if found else None


def _uses_history(root: Mapping[str, Any], result: Mapping[str, Any] | None) -> bool:
    """History scope depends on retrieved inputs even when an answer omits citations."""
    return root.get("scope") == "history" or result is None or bool(result.get("evidence") or result.get("citations"))


class History:
    """A restart-safe view of Ask records in the state directory's existing memory log."""

    def __init__(self, state: str | Path | None = None) -> None:
        from liberational import config

        self.state = Path(state) if state is not None else config.path("paths.state")
        self.log = memory_log.at(state)
        self._lock = threading.RLock()
        self.export_summary = {"exported": 0, "skipped": 0}

    def start(
        self,
        question: str,
        *,
        exchange_id: str | None = None,
        surface: str = "live",
        day: str | None = None,
        scope: str = "general",
        mode: str = "local",
        call_id: str | None = None,
        created_at: float | None = None,
    ) -> str:
        """Persist the question before work is queued. The public exchange ID is the root UUID's hex form."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        if surface not in _SURFACES:
            raise ValueError("unknown Ask surface")
        if day is not None:
            memory_log.day_bounds(day)
        now = time.time() if created_at is None else created_at
        if _number(now) is None:
            raise ValueError("created_at must be a finite epoch timestamp")
        payload = _canonical(exchange_id) if exchange_id is not None else str(uuid.uuid7())
        if payload is None:
            raise ValueError("exchange_id must be a UUID")
        with self._lock:
            if self.get(payload) is not None:
                raise ValueError("exchange_id already exists")
            self.log.append(
                memory_log.ASK_QUESTION,
                {
                    "ts_start": now,
                    "question": question,
                    "surface": surface,
                    "day": day,
                    "scope": scope,
                    "mode": mode,
                    "call_id": call_id,
                    "created_at": now,
                    "revisions": self._revisions(),
                    "transcript_review": revisions.review_revision(self.state),
                },
                payload_id=payload,
            )
        return uuid.UUID(payload).hex

    def _roots(self) -> list[dict[str, Any]]:
        return self.log.range(memory_log.ASK_QUESTION, 0.0, _END)

    def _revisions(self) -> dict[str, int] | None:
        """A conservative deletion checkpoint for legacy evidence without canonical source IDs."""
        link = getattr(self.log, "link", None)
        if link is None:
            return None

        async def go(client: Client) -> dict[str, int] | None:
            rows = await client.list_streams()
            if any(not isinstance(row.get("invalidation_revision"), int) for row in rows):
                return None
            return {str(row["name"]): int(row["invalidation_revision"]) for row in rows}

        return link.call(go)

    def _near(self, stream: str, root: Mapping[str, Any]) -> list[dict[str, Any]]:
        timestamp = float(root["created_at"])
        return [
            row
            for row in self.log.range(stream, max(0.0, timestamp - 0.000001), timestamp + 0.000001)
            if row.get("exchange") == root["id"]
        ]

    def finish(
        self,
        exchange_id: str,
        result: Mapping[str, Any],
        *,
        started_at: float | None,
        finished_at: float,
        status: str,
    ) -> bool:
        """Append a complete version. False means the question was deleted during or before completion."""
        if _number(finished_at) is None or started_at is not None and _number(started_at) is None:
            raise ValueError("answer timestamps must be finite epoch timestamps")
        payload = _canonical(exchange_id)
        with self._lock:
            root = next((row for row in self._roots() if row["id"] == payload), None)
            if root is None:
                return False
            root_id = str(root["id"])
            version = str(uuid.uuid7())
            citations = _citations(result)
            evidence = result.get("evidence") or result.get("citations") or []
            evidence = [row for row in evidence if isinstance(row, Mapping)] if isinstance(evidence, list) else []
            all_ids = [pid for row in evidence for pid in _source_ids(row)]
            parents = list(dict.fromkeys([root_id, *all_ids]))
            unverified = any(not _source_ids(row) for row in evidence)
            metadata: dict[str, Any] = {
                "ts_start": root["created_at"],
                "exchange": root_id,
                "version": version,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "model": str(result["model"])[:256] if isinstance(result.get("model"), str) else None,
                "chunks": [],
                "citations": [],
                "answer_unverified": unverified,
                "revisions": root.get("revisions") if unverified else None,
                "transcript_review": root.get("transcript_review", 0) if _uses_history(root, result) else None,
            }
            baseline = metadata["revisions"]
            current = self._revisions() if unverified and isinstance(baseline, dict) else None
            changed = isinstance(baseline, dict) and (
                current is None or any(current.get(name) != revision for name, revision in baseline.items())
            )
            changed |= metadata["transcript_review"] is not None and metadata[
                "transcript_review"
            ] != revisions.review_revision(self.state)
            encoded = json.dumps(dict(result), ensure_ascii=False, separators=(",", ":")).encode()
            try:
                for index, start in enumerate(range(0, 0 if changed else len(encoded), _CHUNK)):
                    chunk = {
                        "ts_start": root["created_at"],
                        "exchange": root_id,
                        "version": version,
                        "kind": "answer",
                        "index": index,
                        "data": base64.b64encode(encoded[start : start + _CHUNK]).decode("ascii"),
                    }
                    try:
                        chunk_id, kept = self._append_known(memory_log.ASK_ANSWER, chunk, parents, root_id)
                        if len(kept) < len(parents):
                            # Only a payload the broker has can be a parent. A transcript line read from the hourly
                            # segment files has a row id, not a broker payload: it is left out, and the answer is
                            # kept as unverified, so the revision checkpoint hides it once the transcript changes
                            # (a forget, a correction), as for any source without canonical provenance.
                            parents = kept
                            metadata.update(answer_unverified=True, revisions=root.get("revisions"))
                        metadata["chunks"].append(chunk_id)
                    except Exception as error:
                        if not _gone(error):
                            raise
                        break
                # Even a source removed during writing must leave a manifest that cannot expose a partial answer.
                metadata["chunk_count"] = math.ceil(len(encoded) / _CHUNK)
                for index, (citation, ids) in enumerate(citations):
                    marker = {
                        "ts_start": root["created_at"],
                        "exchange": root_id,
                        "version": version,
                        "kind": "citation",
                        "index": index,
                    }
                    try:
                        cited = [pid for pid in ids if pid in parents]
                        marker_id = self.log.append(memory_log.ASK_ANSWER, marker, parents=[root_id, *cited])
                    except Exception as error:
                        if not _gone(error):
                            raise
                        marker_id = None
                    metadata["citations"].append({**citation, "marker": marker_id})
                self.log.append(memory_log.ASK_STATE, metadata, parents=[root_id])
            except Exception as error:
                exception = error
            else:
                exception = None
            # One read after the write answers both "purged during the write" and "written": a record removed
            # mid-write and a failed write used to need their own read each.
            purged = not any(row["id"] == root_id for row in self._roots())
            if exception is not None:
                if _gone(exception) and purged:
                    return False
                raise exception
            return not purged

    def _append_known(
        self, stream: str, record: Mapping[str, Any], parents: list[str], root_id: str
    ) -> tuple[str, list[str]]:
        """Append `record`, leaving out each parent the broker reports it doesn't have (a refused append writes
        nothing), and return its id and the parents it was written with. The root must be known; an unknown root
        means the question was deleted, which the caller handles."""
        kept = list(parents)
        while True:
            try:
                return self.log.append(stream, record, parents=kept), kept
            except Exception as error:
                missing = _unknown_parent(error)
                if missing is None or missing == root_id or missing not in kept:
                    raise
                kept.remove(missing)

    def _entry(self, root: Mapping[str, Any]) -> dict[str, Any]:
        entry = {key: root.get(key) for key in ("question", "surface", "day", "scope", "mode", "call_id", "created_at")}
        entry.update(
            id=uuid.UUID(str(root["id"])).hex,
            status="queued",
            started_at=None,
            finished_at=None,
            model=None,
            result=None,
            citations=[],
            answer_removed=False,
            answer_unverified=False,
            warnings=[],
        )
        states = self._near(memory_log.ASK_STATE, root)
        if not states:
            return entry
        state = max(states, key=lambda row: (row.get("commit_seq", 0), row["id"]))
        for key in ("status", "started_at", "finished_at", "model"):
            entry[key] = state.get(key)
        parts = {row["id"]: row for row in self._near(memory_log.ASK_ANSWER, root)}
        ids = state.get("chunks") or []
        unverified = bool(state.get("answer_unverified"))
        removed = len(ids) != state.get("chunk_count") or any(pid not in parts for pid in ids)
        review = revisions.review_revision(self.state)
        baseline = state.get(
            "transcript_review", 0 if root.get("scope") == "history" or state.get("citations") else None
        )
        review_changed = baseline is not None and baseline != review
        removed |= review_changed
        checkpoint = state.get("revisions")
        checkpoint_changed = False
        if unverified and isinstance(checkpoint, dict):
            current = self._revisions()
            checkpoint_changed = current is None or any(
                current.get(name) != revision for name, revision in checkpoint.items()
            )
            removed |= checkpoint_changed
        if not removed:
            try:
                data = b"".join(base64.b64decode(parts[pid]["data"], validate=True) for pid in ids)
                result = json.loads(data)
                if not isinstance(result, dict):
                    raise ValueError("answer is not an object")
                if "transcript_review" not in state and review and _uses_history(root, result):
                    removed = review_changed = True
                else:
                    entry["result"] = result
            except (ValueError, KeyError, TypeError, binascii.Error) as error:
                raise ValueError("corrupt saved Ask answer") from error
        entry["answer_removed"] = removed
        entry["answer_unverified"] = unverified
        if unverified:
            entry["warnings"] = [
                "Some source references have no canonical provenance; their availability is unverified."
            ]
        entry["citations"] = [
            {key: row.get(key) for key in ("label", "ts", "camera", "source", "verified")}
            | {
                "removed": review_changed
                or row.get("marker") not in parts
                or checkpoint_changed
                and not row.get("verified")
            }
            for row in state.get("citations") or []
        ]
        return entry

    def get(self, exchange_id: str) -> dict[str, Any] | None:
        """One saved exchange, or None after deletion."""
        payload = _canonical(exchange_id)
        with self._lock:
            root = next((row for row in self._roots() if row["id"] == payload), None)
            return self._entry(root) if root is not None else None

    def can_reuse_unsaved(self, exchange_id: str, result: Mapping[str, Any] | None = None) -> bool:
        """A live answer whose write failed is reusable only while its pre-retrieval deletion checkpoint holds."""
        payload = _canonical(exchange_id)
        with self._lock:
            root = next((row for row in self._roots() if row["id"] == payload), None)
            baseline = root.get("revisions") if root is not None else None
            if root is None:
                return False
            if not _uses_history(root, result):
                return True
            if root.get("transcript_review", 0) != revisions.review_revision(self.state):
                return False
            if not isinstance(baseline, dict):
                return False
            current = self._revisions()
            return current is not None and all(current.get(name) == revision for name, revision in baseline.items())

    def list_entries(
        self,
        *,
        surface: str | None = None,
        day: str | None = None,
        limit: int = 20,
        before: str | None = None,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """Newest first, with an exclusive exchange-ID cursor. Reads retain complete answers.

        The read starts at the newest exchange and goes back in widening windows until `limit` roots follow the
        cursor, rather than sorting every exchange ever saved; `since` bounds it to exchanges created at or after
        that time, which is what Ask's pronoun question passes.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        number = None if since is None else _number(since)
        if since is not None and number is None:
            raise ValueError("since must be a finite epoch timestamp")
        floor = max(0.0, number) if number is not None else 0.0
        cursor = _canonical(before) if before is not None else None
        with self._lock:
            roots: list[dict[str, Any]] = []
            end = _END
            start = max(floor, time.time() - _WINDOW_S)
            span = _WINDOW_S
            while True:
                window = [
                    row
                    for row in self.log.range(memory_log.ASK_QUESTION, start, end)
                    if (surface is None or row.get("surface") == surface) and (day is None or row.get("day") == day)
                ]
                roots = sorted(window, key=lambda row: (row["created_at"], row["id"]), reverse=True) + roots
                if start <= floor or (len(roots) > limit and (cursor is None or any(r["id"] == cursor for r in roots))):
                    break
                end = start
                span *= 4
                start = max(floor, end - span)
            if cursor is not None:
                position = next((n for n, row in enumerate(roots) if row["id"] == cursor), None)
                if position is None:
                    return []
                roots = roots[position + 1 :]
            return [self._entry(root) for root in roots[:limit]]

    def delete(self, exchange_id: str | None = None, *, surface: str | None = None, day: str | None = None) -> int:
        """Purge selected question roots and all their versions; return the number of exchanges removed."""
        payload = _canonical(exchange_id) if exchange_id is not None else None
        if exchange_id is not None and payload is None:
            return 0
        with self._lock:
            roots = [
                row["id"]
                for row in self._roots()
                if (payload is None or row["id"] == payload)
                and (surface is None or row.get("surface") == surface)
                and (day is None or row.get("day") == day)
            ]
            self.log.purge(roots, "Ask history deleted")
            return len(roots)

    def export_drafts(self, *, surface: str | None = None, day: str | None = None) -> list[dict[str, Any]]:
        """Draft retrieval labels for manual review, with asked-at and only available timestamped moments."""
        out = []
        skipped = 0
        before = None
        while True:
            entries = self.list_entries(surface=surface, day=day, limit=100, before=before)
            for entry in entries:
                evidence = [
                    {"at": row["ts"], "within_s": 1, "source": row["source"]}
                    for row in entry["citations"]
                    if row["ts"] is not None
                    and not row["removed"]
                    and row.get("verified")
                    and row["source"] in _SOURCES
                ]
                if entry["scope"] != "history" or not evidence:
                    skipped += 1
                    continue
                out.append(
                    {
                        "id": entry["id"],
                        "question": entry["question"],
                        "asked_at": entry["started_at"] if entry["started_at"] is not None else entry["created_at"],
                        "surface": entry["surface"],
                        "day": entry["day"],
                        "scope": entry["scope"],
                        "mode": entry["mode"],
                        "draft": True,
                        "evidence": evidence,
                        "note": "Draft prior citations. A person must verify the expected evidence before accepting this label.",
                    }
                )
            if len(entries) < 100:
                self.export_summary = {"exported": len(out), "skipped": skipped}
                return out
            before = entries[-1]["id"]

    def close(self) -> None:
        """The shared memory log owns its connection; closing this view releases no other caller's state."""


_HISTORIES: dict[str, History] = {}
_LOCK = threading.Lock()


def at(state: str | Path | None = None) -> History:
    """Shared history view for a configured state directory; no independent persistent state."""
    from liberational import config as cfg

    key = str(Path(state) if state is not None else cfg.path("paths.state"))
    with _LOCK:
        if key not in _HISTORIES:
            _HISTORIES[key] = History(state)
        return _HISTORIES[key]


def close_all() -> None:
    """Drop cached views between brokers or tests; memory_log.close_all owns connection teardown."""
    with _LOCK:
        _HISTORIES.clear()
