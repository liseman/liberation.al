"""Owner names and ratings on activity spans, written to and read from the knowledge store as memory/events.py keeps
them, for tests that set them up directly instead of through events.name_activity and feedback_activity."""

import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from liberational import knowledge, segment_log
from liberational.memory import events


def _store(state: Path) -> knowledge.Store:
    return events._store(state)


def _put(state: Path, collection: str, span: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    start = float(span["ts_start"])
    end = float(span.get("ts_end", start))
    value = {"activity": span["id"], **fields, "source": fields.get("source", "owner")}
    record = knowledge.Record(
        str(uuid.uuid7()), collection, str(span["id"]), round(start * 1e9), round(end * 1e9), time.time_ns(), value
    )
    _store(state).put([record])
    return {**value, "id": record.id}


def name(state: Path, span: Mapping[str, Any], label: str, **fields: Any) -> dict[str, Any]:
    """The owner names `span` (a record with `id`, `ts_start`, and optionally `ts_end`)."""
    # The kind the span had when it was named, as events.name_activity stores it: a name without one reads as a
    # retired kind's, which no one asks about.
    return _put(state, events.NAMED, span, {"name": label, "kind": span.get("kind"), **fields})


def rate(
    state: Path,
    span: Mapping[str, Any],
    verdict: str,
    *,
    stream: str | None = segment_log.WALKBY_SPANS,
    **fields: Any,
) -> dict[str, Any]:
    """The owner rates `span` Right or Wrong, with its kind and camera as events writes them. `stream=None` writes a
    legacy rating without a stream, which calibration leaves out."""
    value = {"verdict": verdict, "kind": span.get("kind"), "cam": span.get("cam"), **fields}
    if stream is not None:
        value["stream"] = stream
    return _put(state, events.FEEDBACK, span, value)


def values(state: Path, collection: str) -> list[dict[str, Any]]:
    """Every annotation in `collection` (events.NAMED or events.FEEDBACK), in the order written."""
    return [row.record.value for row in _store(state).query(collection)]
