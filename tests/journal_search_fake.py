"""The vector worker's searches over the journal, in memory, for tests of `recall`: tests/conftest.py puts `hits` in
place of `recall._hits` for every test, so no test reaches a vector worker. It searches the in-memory memory log's
`journal.entry` records as the worker's indexes would: by words in the entry's rendered `search` text, else its text
(an entry matching more of the question's words first, then the earlier), within the window and about any of the subjects asked; and by meaning, whatever entry ids a
test puts in `near`, in that order.
"""

import re
from collections.abc import Iterable
from typing import Any, cast

from liberational import memory_log
from memory_fake import FakeMemoryLog

#: Entry ids the meaning search finds, best first (a test sets it; empty means none).
near: list[str] = []


def _hit(r: dict[str, Any]) -> dict[str, Any]:
    start = int(float(r["ts_start"]) * 1e9)
    end = int(float(r.get("ts_end") or r["ts_start"]) * 1e9)
    return {
        "payload_id": str(r["id"]),
        "entity_id": None,
        "parent_id": None,
        "start_ns": start,
        "end_ns": end,
        "score": 0,
    }


def _in(r: dict[str, Any], lo: float, hi: float, subjects: Iterable[str] | None) -> bool:
    start, end = float(r["ts_start"]), float(r.get("ts_end") or r["ts_start"])
    about = subjects is None or bool(set(subjects) & {str(s) for s in r.get("subjects") or []})
    return start < hi and max(end, start) >= lo and about


def hits(
    words: str, vector: Any, lo: float, hi: float, subjects: set[str] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """recall._hits over the in-memory memory log (see the module's docs)."""
    records = cast(FakeMemoryLog, memory_log.at()).records.get(memory_log.JOURNAL, [])
    asked = set(words.split())
    scored = []
    for r in records:
        text = set(re.findall(r"\w+", str(r.get("search") or r.get("text") or "").casefold()))
        if (score := len(asked & text)) and _in(r, lo, hi, subjects):
            scored.append((-score, float(r["ts_start"]), r))
    lexical = [_hit(r) for *_, r in sorted(scored, key=lambda x: (x[0], x[1]))]
    by_id = {str(r["id"]): r for r in records}
    meaning = [_hit(by_id[i]) for i in near if i in by_id and _in(by_id[i], lo, hi, subjects)]
    return lexical, meaning
