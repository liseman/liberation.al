"""Review HTTP wiring, with fake inference and temporary state only."""

import datetime as dt
import http.client
import json
import urllib.parse
from unittest.mock import create_autospec
from zoneinfo import ZoneInfo

import pytest

from liberational.dashboard import context, review_reads, server
from liberational.memory import history_qa, recall, review_summary


@pytest.fixture
def request_(serve, monkeypatch, tmp_path):
    monkeypatch.setattr(context, "STATE", tmp_path)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    netloc = urllib.parse.urlsplit(serve(server.H)).netloc

    def send(method, path, body=None):
        conn = http.client.HTTPConnection(netloc, timeout=5)
        data = json.dumps(body) if body is not None else None
        conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        value = response.status, json.loads(response.read())
        conn.close()
        return value

    return send


@pytest.fixture
def meaning_search(monkeypatch):
    search = create_autospec(recall.search, return_value=[])
    monkeypatch.setattr(recall, "search", search)
    monkeypatch.setattr(history_qa, "TZ", dt.UTC)
    return search


def test_review_search_passes_one_day_and_requires_meaning_without_chat_ranking(request_, meaning_search):
    meaning_search.return_value = [
        {"id": "J1", "interpretation": "journal-1", "ts": 1792454405.0, "end": 1792454430.0, "text": "A quiet lunch."},
    ]
    code, body = request_("GET", "/review/search?day=2026-10-20&q=quiet%20lunch")
    assert code == 200
    assert body == {
        "results": [
            {"id": "journal-1", "ts": 1792454405.0, "end": 1792454430.0, "text": "A quiet lunch.", "kind": "semantic"},
        ]
    }
    meaning_search.assert_called_once_with(
        "quiet lunch",
        1792454400.0,
        1792540800.0,
        state=context.STATE,
        rank=False,
        limit=24,
        require_semantic=True,
    )


@pytest.mark.parametrize("query", ["", " ", "x" * 501])
def test_review_search_rejects_empty_and_oversized_queries_without_search(request_, meaning_search, query):
    assert request_("GET", "/review/search?day=2026-10-20&q=" + urllib.parse.quote(query))[0] == 400
    meaning_search.assert_not_called()


@pytest.mark.parametrize("day", ["", "2026-02-30", "2026-2-03", "2026-10-20T00:00:00", "../private"])
def test_review_search_requires_an_explicit_canonical_day_without_search(request_, meaning_search, day):
    assert request_("GET", "/review/search?day=" + urllib.parse.quote(day) + "&q=lunch")[0] == 400
    meaning_search.assert_not_called()


@pytest.mark.parametrize(
    ("day", "start", "end"),
    [("2026-03-08", 1772946000.0, 1773028800.0), ("2026-11-01", 1793505600.0, 1793595600.0)],
)
def test_review_search_uses_next_local_midnight_on_clock_change_days(
    request_,
    meaning_search,
    monkeypatch,
    day,
    start,
    end,
):
    monkeypatch.setattr(history_qa, "TZ", ZoneInfo("America/New_York"))
    assert request_("GET", f"/review/search?day={day}&q=lunch")[0] == 200
    meaning_search.assert_called_once_with(
        "lunch",
        start,
        end,
        state=context.STATE,
        rank=False,
        limit=24,
        require_semantic=True,
    )


def test_review_search_drops_bad_or_outside_day_hits_and_clamps_end(request_, meaning_search):
    meaning_search.return_value = [
        {"id": "old", "ts": 1792454399.0, "text": "Before the day."},
        {"id": "next", "ts": 1792540800.0, "text": "After the day."},
        {"id": "nan", "ts": float("nan"), "text": "Invalid time."},
        {"id": "bool", "ts": True, "text": "Invalid time."},
        {"id": "missing", "text": "No time."},
        {"id": "empty", "ts": 1792454401.0, "text": " "},
        {"id": "valid", "ts": 1792540799.0, "end": 1792540900.0, "text": "Last moment."},
        {"id": "valid", "ts": 1792540799.0, "text": "Duplicate."},
    ]
    code, body = request_("GET", "/review/search?day=2026-10-20&q=moment")
    assert (code, body) == (
        200,
        {
            "results": [
                {"id": "valid", "ts": 1792540799.0, "end": 1792540800.0, "text": "Last moment.", "kind": "semantic"},
            ]
        },
    )


def test_review_search_caps_results_even_if_index_exceeds_its_requested_limit(request_, meaning_search):
    meaning_search.return_value = [{"id": str(i), "ts": 1792454400.0 + i, "text": "Lunch."} for i in range(30)]
    code, body = request_("GET", "/review/search?day=2026-10-20&q=lunch")
    assert code == 200
    assert len(body["results"]) == 24
    assert body["results"][-1]["id"] == "23"


@pytest.mark.parametrize(
    "error", [recall.SemanticUnavailableError("Local meaning search is unavailable."), RuntimeError("private query")]
)
def test_review_search_reports_unavailability_instead_of_no_results(request_, meaning_search, error):
    meaning_search.side_effect = error
    code, body = request_("GET", "/review/search?day=2026-10-20&q=lunch")
    assert code == 503
    assert "results" not in body
    assert "private query" not in body["error"]


@pytest.fixture
def timeline_summary(monkeypatch):
    summary = create_autospec(review_summary.summary, return_value={"blocks": [], "status": "ready"})
    monkeypatch.setattr(review_summary, "summary", summary)
    monkeypatch.setattr(history_qa, "TZ", ZoneInfo("America/New_York"))
    return summary


@pytest.mark.parametrize(
    ("day", "start", "end"),
    [
        ("2026-10-20", 1792468800.0, 1792555200.0),
        ("2026-03-08", 1772946000.0, 1773028800.0),
        ("2026-11-01", 1793505600.0, 1793595600.0),
    ],
    ids=["recording-timezone", "23-hour-day", "25-hour-day"],
)
def test_day_summary_uses_recording_timezone_instead_of_browser_bounds(request_, timeline_summary, day, start, end):
    code, body = request_("GET", f"/review/summary?day={day}")
    assert (code, body) == (200, {"blocks": [], "status": "ready"})
    timeline_summary.assert_called_once_with(start, end, context.STATE)


@pytest.mark.parametrize("day", ["", "2026-02-30", "2026-2-03", "2026-10-20T00:00:00", "../private", "9999-12-31"])
def test_day_summary_rejects_invalid_or_noncanonical_day_without_reading(request_, timeline_summary, day):
    assert request_("GET", "/review/summary?day=" + urllib.parse.quote(day))[0] == 400
    timeline_summary.assert_not_called()


@pytest.mark.parametrize(
    ("status", "kept_s"),
    [pytest.param("ready", review_reads.ENDED_SUMMARY_KEEP_S, id="ready"), pytest.param("partial", 60, id="partial")],
)
def test_an_ended_stretchs_summary_is_kept_longest_once_ready(request_, monkeypatch, status, kept_s):
    """A day's summary read the journal off the broker every request (2 to 5 s); today's is always partial."""
    clock = {"now": 1_800_000_000.0}
    monkeypatch.setattr(server.time, "time", lambda: clock["now"])
    monkeypatch.setattr(review_reads, "_summaries", {})
    answer = {"start": 0, "end": 1, "level": 1, "status": status, "blocks": []}
    summary = create_autospec(review_summary.summary, return_value=answer)
    monkeypatch.setattr(review_summary, "summary", summary)
    path = "/review/summary?start=1700000000&end=1700003600"  # an ended hour
    assert request_("GET", path) == (200, answer)
    clock["now"] += kept_s - 1  # the journal moves meanwhile: the kept answer stands
    assert request_("GET", path) == (200, answer)
    assert summary.call_count == 1
    clock["now"] += 2
    request_("GET", path)
    assert summary.call_count == 2, "worked out again once its time is up"
