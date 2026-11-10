"""Review's reads of the journal: the summary blocks of a stretch of time, and search within a day (moved out of the
server, #61).

GET /review/summary  ?day= or ?start=&end=: what happened over that stretch, as timeline blocks
GET /review/search   ?day=&q=: the day's lines and events that match
"""

import math
import threading
import time
from typing import TYPE_CHECKING, Any

from liberational import speech_log
from liberational.dashboard import context, routes
from liberational.dashboard.context import LOG, Reply

if TYPE_CHECKING:
    from liberational.dashboard.server import H


REVIEW_SUMMARY_MAX_S = 31 * 86400


def review_search(day: str, query: str) -> Reply:
    """Search the local meaning index inside one canonical local day; never run a ranking/chat model."""
    from liberational.memory import ask, journal, recall

    try:
        if ask.validate_day(day) is None:
            raise ValueError("day must be YYYY-MM-DD")
        start, end = journal.day_bounds(day)
    except ValueError, OverflowError:
        return 400, {"error": "day must be YYYY-MM-DD"}
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        return 400, {"error": "Search must be 1 to 500 characters."}
    try:
        found = recall.search(
            query.strip(), start, end, state=context.STATE, rank=False, limit=24, require_semantic=True
        )
    except recall.SemanticUnavailableError as error:
        LOG.warning("review meaning search unavailable: %s", type(error).__name__)
        return 503, {"error": str(error)}
    except Exception as error:
        LOG.warning("review meaning search failed: %s", type(error).__name__)
        return 503, {"error": "The local meaning index could not answer. Try again."}
    results = []
    seen: set[str] = set()
    for item in found:
        ts, stop, text = item.get("ts"), item.get("end"), item.get("text")
        if (
            isinstance(ts, bool)
            or not isinstance(ts, int | float)
            or not math.isfinite(ts)
            or not start <= ts < end
            or not isinstance(text, str)
            or not text.strip()
        ):
            continue
        stop = (
            min(end, max(ts, stop))
            if isinstance(stop, int | float) and not isinstance(stop, bool) and math.isfinite(stop)
            else ts
        )
        key = str(item.get("interpretation") or item.get("id") or f"{ts}:{len(results)}")
        if key in seen:
            continue
        seen.add(key)
        results.append({"id": key, "ts": ts, "end": stop, "text": text, "kind": "semantic"})
        if len(results) == 24:
            break
    return 200, {"results": results}


# Review's summaries, kept: {(start, end): (stamp, answer)}. Reading a day's journal off the broker took 2 to 5 s
# a request, every request.
_summaries: dict[tuple[float, float], tuple[Any, dict[str, Any]]] = {}


_summaries_lock = threading.Lock()


SUMMARY_KEEP_S = 60.0  # how long a kept summary is used, ready or not


ENDED_SUMMARY_KEEP_S = 900.0  # and of an ended one, whose ready answer only a purge or a rare rebuild changes


def review_summary_kept(start: float, end: float) -> dict[str, Any]:
    """GET /review/summary's answer for [start, end), kept for SUMMARY_KEEP_S, and a ready answer of an ended stretch
    for ENDED_SUMMARY_KEEP_S, or until a purge. The journal's own position can't be the stamp: it's the broker's
    commit sequence, which moves every few seconds whatever stretch is asked about. A pending or partial answer (today's
    always is, while the journal writes) is kept a minute too: working it out queues the rewrites still missing, and
    once a minute is enough for that, where every request was 3 to 5 s."""
    from liberational.memory import review_summary

    now = time.time()
    purge = speech_log.shared().newest_purge()
    with _summaries_lock:
        hit = _summaries.get((start, end))
        if hit is not None:
            (kept_purge, until), answer = hit
            if kept_purge == purge and now < until:
                return answer
    answer = review_summary.summary(start, end, context.STATE)
    keep = ENDED_SUMMARY_KEEP_S if end <= now and answer.get("status") == "ready" else SUMMARY_KEEP_S
    with _summaries_lock:
        if len(_summaries) >= 64:
            _summaries.clear()
        _summaries[(start, end)] = ((purge, now + keep), answer)
    return answer


@routes.get("/review/search")
def _get_review_search(h: H) -> None:
    q = h._query()
    code, reply = review_search(q.get("day", ""), q.get("q", ""))
    return h._json(reply, code)


@routes.get("/review/summary")
def _get_review_summary(h: H) -> None:
    q = h._query()
    if "day" in q:
        from liberational.memory import ask, journal

        try:
            if ask.validate_day(q["day"]) is None:
                raise ValueError("day must be YYYY-MM-DD")
            start, end = journal.day_bounds(q["day"])
        except ValueError, OverflowError:
            return h._json({"error": "day must be YYYY-MM-DD"}, 400)
    else:
        try:
            start, end = float(q.get("start", "")), float(q.get("end", ""))
        except ValueError:
            return h._json({"error": "start and end are times in seconds"}, 400)
    if not (math.isfinite(start) and math.isfinite(end) and start < end <= start + REVIEW_SUMMARY_MAX_S):
        return h._json({"error": "end must come after start, and within 31 days of it"}, 400)
    try:
        return h._json(review_summary_kept(start, end))
    except Exception as ex:
        LOG.error("review summary failed: %s", repr(ex))
        return h._json({"error": "Couldn't read the journal for that stretch."}, 500)
