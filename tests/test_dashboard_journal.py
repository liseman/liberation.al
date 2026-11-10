"""The Journal tab's reads (dashboard/journal.py, #2811): months down to events, pages, and evidence."""

import datetime as dt
import http.client
import json
from urllib.parse import urlsplit

import pytest

from liberational import memory_log, speech_log
from liberational.dashboard import journal as view
from liberational.dashboard import server
from liberational.listen import revisions
from liberational.memory import journal

DAY = dt.date(2026, 11, 7)  # a Saturday
NOW = journal.day_bounds(DAY)[0] + 12 * 3600  # noon


@pytest.fixture
def state(tmp_path):
    return tmp_path


def at(day: dt.date, hour: float) -> float:
    return journal.day_bounds(day)[0] + hour * 3600


def page(memory, level, period: dt.date, md, end: dt.date | None = None, status=None):
    end = end or period + dt.timedelta(days=1)
    memory.append(
        memory_log.PAGES,
        {
            "ts_start": journal.day_bounds(period)[0],
            "ts_end": journal.day_bounds(end)[0],
            "level": level,
            "period": period.isoformat(),
            **({"status": status} if status else {"md": md, "sections": {}}),
        },
    )


def entry(memory, start, length, text, sources=()):
    memory.append(
        memory_log.JOURNAL,
        {
            "ts_start": start,
            "ts_end": start + length,
            "level": length,
            "text": text,
            "evidence": {"status": "verified", "sources": list(sources)},
        },
    )


def source(stream, start, camera=None, kind=None):
    return {
        "id": "s",
        "stream": stream,
        "ts_start": start,
        "ts_end": start + 5,
        "people": [{"id": "p1", "name": "Dana"}],
        "camera": camera,
        "kind": kind,
        "text": "said hello",
    }


def test_months_list_the_last_twelve_newest_first_and_mark_written_pages(memory, state):
    page(memory, "month", dt.date(2026, 10, 1), "# October", end=dt.date(2026, 11, 1))
    code, reply = view.listing({}, state, NOW)
    assert code == 200
    keys = [m["key"] for m in reply["items"]]
    assert keys[0] == "2026-11-01" and keys[-1] == "2025-12-01" and len(keys) == 12
    assert [m["key"] for m in reply["items"] if m["page"]] == ["2026-10-01"]
    assert reply["items"][1]["label"] == "October 2026"


def test_a_month_lists_the_weeks_that_overlap_it_up_to_today_with_its_page(memory, state):
    page(memory, "month", dt.date(2026, 11, 1), "Lunch [h1](#t=1762500000)", end=dt.date(2026, 12, 1))
    page(memory, "week", dt.date(2026, 10, 26), "# Week", end=dt.date(2026, 11, 2))
    code, reply = view.listing({"month": "2026-11-01"}, state, NOW)
    assert code == 200
    # Nov 1 is a Sunday, so the week of Mon Oct 26 overlaps it; the week of Nov 9 hasn't started.
    assert [(w["key"], w["label"], w["page"]) for w in reply["items"]] == [
        ("2026-11-02", "Nov 2 to 8", False),
        ("2026-10-26", "Oct 26 to Nov 1", True),
    ]
    assert reply["page"]["md"] == "Lunch [h1](/review#t=1762500000)"  # a citation opens Review there


def test_a_week_lists_its_days_so_far_with_their_hour_counts(memory, state):
    page(memory, "day", dt.date(2026, 11, 5), "# Thursday")
    for h in (9, 10, 14):
        entry(memory, at(dt.date(2026, 11, 5), h), 3600, "worked")
    entry(memory, at(dt.date(2026, 11, 5), 9), 180, "a finer entry, not an hour")
    code, reply = view.listing({"week": "2026-11-04"}, state, NOW)  # any day of the week finds its Monday
    assert code == 200
    assert [d["key"] for d in reply["items"]] == [f"2026-11-0{d}" for d in (7, 6, 5, 4, 3, 2)]
    thursday = reply["items"][2]
    assert (thursday["label"], thursday["page"], thursday["hours"]) == ("Thu Nov 5", True, 3)
    assert reply["items"][0]["hours"] == 0
    assert reply["page"] is None


def test_a_day_lists_its_hours_in_time_order_with_the_day_page(memory, state):
    page(memory, "day", DAY, "# Saturday")
    entry(memory, at(DAY, 10), 3600, "ten")
    entry(memory, at(DAY, 9), 3600, "nine")
    entry(memory, at(DAY - dt.timedelta(days=1), 23), 3600, "the day before")
    code, reply = view.listing({"day": DAY.isoformat()}, state, NOW)
    assert code == 200
    assert [(h["label"], h["text"]) for h in reply["items"]] == [("09:00", "nine"), ("10:00", "ten")]
    assert reply["items"][0]["key"] == str(int(at(DAY, 9)))
    assert reply["page"]["md"] == "# Saturday"
    assert reply["why"] is None


LAST_WEEK, OLD_WEEK = dt.date(2026, 10, 26), dt.date(2026, 10, 12)


@pytest.mark.parametrize(
    ("week", "status", "corrected", "why"),
    [
        pytest.param(dt.date(2026, 11, 2), None, False, "pending", id="this-week-not-over"),
        pytest.param(LAST_WEEK, None, False, "pending", id="last-week-the-next-batch-writes"),
        pytest.param(OLD_WEEK, None, False, "old", id="too-old-to-be-written"),
        pytest.param(LAST_WEEK, "quiet", False, "quiet", id="nothing-recorded"),
        pytest.param(LAST_WEEK, "failed", False, "failed", id="the-model-failed"),
        pytest.param(LAST_WEEK, "page", True, "rewriting", id="hidden-by-a-correction"),
    ],
)
def test_a_period_with_no_page_to_show_says_why(memory, state, week, status, corrected, why):
    if status:
        page(
            memory, "week", week, "# Week", end=week + dt.timedelta(days=7), status=None if status == "page" else status
        )
    if corrected:  # the owner marks a line of that week as a TV's after its page was written
        t = at(week + dt.timedelta(days=2), 10)
        revisions.mark_media(state, [{"id": "u1", "ts_start": t, "ts_end": t + 4, "device": "wave3"}])
    code, reply = view.listing({"week": week.isoformat()}, state, NOW)
    assert code == 200
    assert (reply["page"], reply["why"]) == (None, why)


def test_an_hour_lists_its_events_and_nothing_outside_it(memory, state):
    start = at(DAY, 9)
    level = view.event_level()
    entry(memory, start + 60, level, "made coffee")
    entry(memory, start + 3600, level, "the next hour")
    entry(memory, start, 3600, "the hour itself")
    code, reply = view.listing({"hour": str(int(start))}, state, NOW)
    assert code == 200
    assert [e["text"] for e in reply["items"]] == ["made coffee"]


def test_an_entry_lists_its_rows_without_resolving_evidence(memory, state, monkeypatch):
    monkeypatch.setattr(view.journal_evidence, "resolve_sources", lambda *a, **k: pytest.fail("resolved for a listing"))
    entry(memory, at(DAY, 9), 3600, "nine", [source(speech_log.UTTERANCES, at(DAY, 9))])
    code, reply = view.listing({"day": DAY.isoformat()}, state, NOW)
    assert code == 200
    assert reply["items"][0]["sources"] == 1 and reply["items"][0]["id"]


def test_an_entrys_evidence_is_resolved_in_one_batch_with_who_and_when(memory, state, monkeypatch):
    start = at(DAY, 9)
    calls = []

    def resolve_sources(state_, day, sources, **kw):
        calls.append((day, [s["stream"] for s in sources]))
        return [
            {"stage": "expired", "note": "footage expired", "caption": s["text"], "camera": s["camera"]}
            if s.get("camera")
            else {"stage": "audio", "t": s["ts_start"], "caption": s["text"]}
            for s in sources
        ]

    monkeypatch.setattr(view.journal_evidence, "resolve_sources", resolve_sources)
    entry(
        memory,
        start,
        180,
        "x",
        [source(speech_log.UTTERANCES, start + 10), source("camera.caption", start + 20, "porch")],
    )
    eid = memory.records[memory_log.JOURNAL][0]["id"]
    code, reply = view.listing({"entry": eid, "start": str(start)}, state, NOW)
    assert code == 200
    assert calls == [(DAY.isoformat(), ["speech.utterance", "camera.caption"])]  # one batch, not one call per source
    speech, gone = reply["sources"]
    assert (speech["stage"], speech["people"], speech["review"]) == ("audio", ["Dana"], f"/review#t={start + 10:.1f}")
    assert speech["audio"].startswith("/audio/range?start=")
    assert (gone["stage"], gone["note"]) == ("expired", "footage expired") and "audio" not in gone
    assert reply["status"] == "verified"


def test_an_entry_that_isnt_there_is_a_404(memory, state):
    assert view.listing({"entry": "nope", "start": str(at(DAY, 9))}, state, NOW)[0] == 404


@pytest.mark.parametrize(
    "query",
    [
        {"day": "2026-13-01"},
        {"week": "yesterday"},
        {"month": "2026-11"},
        {"hour": "soon"},
        {"hour": "nan"},
        {"hour": "inf"},
        {"entry": "x", "start": "later"},
        {"entry": "x", "start": "1e300"},
        {"hour": "1e300"},
    ],
)
def test_a_bad_period_is_a_400(state, query):
    code, reply = view.listing(query, state, NOW)
    assert code == 400 and "error" in reply


def test_the_route_answers_json_and_a_page_load_gets_the_app(serve, memory):
    base = serve(server.H)
    host = urlsplit(base).netloc

    def get(path, accept):
        conn = http.client.HTTPConnection(host, timeout=120)  # generous: a loaded machine is slow, not broken
        conn.request("GET", path, headers={"Accept": accept})
        r = conn.getresponse()
        body = r.read()
        conn.close()
        return r.status, r.getheader("Content-Type") or "", body

    status, kind, body = get("/journal?day=2026-11-07", "application/json")
    assert (status, kind.split(";")[0]) == (200, "application/json")
    assert json.loads(body) == {"items": [], "page": None, "why": "pending"}
    status, kind, _ = get("/journal", "text/html")
    assert kind.startswith("text/html")
