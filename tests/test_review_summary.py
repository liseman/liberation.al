"""Review's summary column (memory/review_summary.py) and the plain lines it reads (memory/plain_text.py):
the journal's entries at the level that fits the stretch, each rewritten as a short neutral line, never first person,
cached, with chip tags. Synthetic journal entries and a fake model in a temp state dir; never the real data/."""

import http.client
import json
import time
import urllib.parse
from unittest.mock import create_autospec

import pytest

from liberational.domain.identity import Basis
from liberational.memory import journal, plain_text, review_summary
from liberational.people import registry
from voices import write_voices

T0 = 1_793_000_000.0 - (1_793_000_000.0 % 3600)  # on the hour, so every level's windows line up from it
DAY = 86400.0


def entry(state, start, level, text, **payload):
    """A journal entry on the memory log, as the journal worker writes one."""
    from liberational import memory_log

    return memory_log.at(state).append(
        memory_log.JOURNAL, {"ts_start": start, "ts_end": start + level, "level": level, "text": text, **payload}
    )


def fake(replies):
    """A model stand-in: the JSON reply for each rendered entry text, else a first-person one (always dropped)."""
    calls = []

    def rewrite(text):
        calls.append(text)
        return replies.get(text, json.dumps({"text": "I sat at my desk", "topics": ["work"]}))

    return calls, rewrite


ROLLED = json.dumps({"text": "Worked at the desk, then lunch", "topics": ["work"]})


class StandInRollups:
    """review_summary.rollups_at with a stand-in model and no background writer: `calls` holds what each sentence was
    asked to sum up, and `replies` the answer for a name ("Morning", "day"), else ROLLED."""

    def __init__(self) -> None:
        self.made: dict[str, review_summary.Rollups] = {}
        self.calls: list[dict] = []
        self.sources: list[str] = []  # as the model got them, a retry's broken rule included
        self.replies: dict[str, str | list[str]] = {}  # a list answers each ask in turn

    def rewrite(self, source: str) -> str:
        data = json.loads(source.split("\n\nYour last line")[0])
        self.calls.append(data)
        self.sources.append(source)
        reply = self.replies.get(data["name"], ROLLED)
        return reply.pop(0) if isinstance(reply, list) else reply

    def __call__(self, state) -> review_summary.Rollups:
        return self.made.setdefault(str(state), review_summary.Rollups(state, self.rewrite, background=False))


@pytest.fixture(autouse=True)
def rollups(monkeypatch):
    """The part-of-day and day sentences, never the real model."""
    stand_in = StandInRollups()
    monkeypatch.setattr(review_summary, "rollups_at", stand_in)
    return stand_in


@pytest.fixture
def route(state, monkeypatch, serve_dashboard):
    """The dashboard serving `state`, with a route cache of its own: `review_reads.review_summary_kept` keeps each window's
    answer for a minute keyed by the window alone, so without this a test asking for the same window as an earlier
    one, from another state, gets that test's answer."""
    from liberational.dashboard import context, review_reads

    monkeypatch.setattr(context, "STATE", state)
    monkeypatch.setattr(review_reads, "_summaries", {})
    return urllib.parse.urlsplit(serve_dashboard())


@pytest.fixture
def state(tmp_path):
    write_voices(tmp_path, {2: "Dana"})
    registry.at(tmp_path).claim(registry.faceprint("enroll:x"), "Sam", Basis.OWNER_TAG, "test")
    return tmp_path


def run(state, start, end, replies, now=T0 + 10 * DAY):
    calls, rewrite = fake(replies)
    cache = review_summary.Lines(state, rewrite, background=False)
    first = review_summary.summary(start, end, state, now=now, rewrites=cache)
    cache.drain()
    for _ in range(2):  # a day's view: its parts' sentences, then the day's, so the status can settle
        review_summary.summary(start, end, state, now=now, rewrites=cache)
        review_summary.rollups_at(state).drain()
    return first, review_summary.summary(start, end, state, now=now, rewrites=cache), calls


@pytest.mark.parametrize(
    ("span", "level"),
    [
        pytest.param(DAY, 3600, id="a-day-in-hours"),
        pytest.param(3600, 600, id="an-hour-in-ten-minutes"),
        pytest.param(1200, 180, id="twenty-minutes-in-three"),
        pytest.param(120, 30, id="two-minutes-at-the-finest"),
    ],
)
def test_the_level_fits_the_stretch(span, level):
    assert review_summary.level_for(span, [30, 180, 600, 3600]) == level


def test_blocks_are_plain_lines_with_chip_tags_in_time_order(state):
    entry(state, T0 + 3600, 3600, "I had lunch with [S2] in the kitchen.", speakers=["2"], faces=["Sam"])
    entry(state, T0, 3600, "I worked at my desk for the whole hour.")
    entry(state, T0, 600, "A finer entry that the hour level never reads.")
    replies = {
        "I had lunch with Dana in the kitchen.": json.dumps(
            {"text": "Lunch with Dana in the kitchen.", "topics": ["meal", "nap"]}
        ),
        "I worked at my desk for the whole hour.": json.dumps({"text": "worked at the desk", "topics": ["work"]}),
    }
    first, second, calls = run(state, T0, T0 + DAY, replies)
    assert (first["status"], first["blocks"]) == ("pending", [])  # nothing written yet: it never waits
    assert first["level"] == 3600
    assert second["status"] == "ready"
    assert second["blocks"] == [
        {
            "start": T0,
            "end": T0 + 3600,
            "text": "Worked at the desk",
            "tags": ["topic:work"],
            "evidence": {"status": "unverified", "sources": []},
        },
        {
            "start": T0 + 3600,
            "end": T0 + 7200,
            "text": "Lunch with Dana in the kitchen",  # a timeline row: no full stop
            "tags": ["person:dana", "topic:meal"],  # the line never mentions Sam
            "evidence": {"status": "unverified", "sources": []},
        },
    ]
    assert sorted(calls) == sorted(replies)  # each entry rewritten once, with the names it has now


def test_a_first_person_line_never_reaches_the_column(state):
    entry(state, T0, 3600, "I sat at my desk all hour.")
    _, second, _ = run(state, T0, T0 + DAY, {})  # the fake answers "I sat at my desk"
    assert (second["status"], second["blocks"]) == ("ready", [])  # dropped, and not asked again


def test_lines_are_cached_per_entry_until_it_changes(state):
    entry(state, T0, 3600, "I read a book.")
    replies = {"I read a book.": json.dumps({"text": "Read a book", "topics": ["rest"]})}
    calls, rewrite = fake(replies)
    cache = review_summary.Lines(state, rewrite, background=False)
    for _ in range(3):
        review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
        cache.drain()
    assert calls == ["I read a book."]
    # Kept on the memory log: the next process finds the line without asking the model.
    fresh = review_summary.Lines(state, lambda t: pytest.fail("asked again"), background=False)
    got = review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=fresh)
    assert [b["text"] for b in got["blocks"]] == ["Read a book"]


def test_a_stretch_reaching_now_is_partial(state):
    entry(state, T0, 3600, "I worked at my desk for the whole hour.")
    replies = {"I worked at my desk for the whole hour.": json.dumps({"text": "Worked at the desk", "topics": []})}
    _, second, _ = run(state, T0, T0 + DAY, replies, now=T0 + DAY - 60)  # today's last hour isn't written yet
    assert (second["status"], [b["text"] for b in second["blocks"]]) == ("partial", ["Worked at the desk"])


def test_a_stretch_with_nothing_recorded_is_ready_and_empty(state):
    _, second, calls = run(state, T0, T0 + DAY, {})
    assert (second["status"], second["blocks"], calls) == ("ready", [], [])


def test_a_model_that_is_down_is_tried_again_later(state):
    entry(state, T0, 3600, "I read a book.")
    now = [T0]

    def down(text):
        raise ConnectionError("ollama down")

    cache = review_summary.Lines(state, down, background=False, clock=lambda: now[0])
    review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    cache.drain()
    again = review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    assert again["status"] == "pending"
    assert not cache.todo  # not asked again on every read
    now[0] += plain_text.RETRY_S + 1
    review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    assert cache.todo


@pytest.mark.parametrize(
    ("reply", "want"),
    [
        pytest.param(
            '{"text": "Worked at the desk.", "topics": ["work"]}',
            {"text": "Worked at the desk", "topics": ["work"]},
            id="kept",
        ),
        pytest.param('Sure! {"text": "Lunch", "topics": []}', {"text": "Lunch", "topics": []}, id="json-among-words"),
        pytest.param('{"text": "I worked at my desk", "topics": []}', None, id="first-person"),
        pytest.param('{"text": "She worked at the desk", "topics": []}', None, id="pronoun"),
        pytest.param('{"text": "The person worked at the desk", "topics": []}', None, id="the-person"),
        pytest.param('{"text": "' + "x" * 91 + '", "topics": []}', None, id="too-long"),
        pytest.param("Worked at the desk", None, id="not-json"),
        pytest.param('{"topics": ["work"]}', None, id="no-text"),
    ],
)
def test_clean_keeps_only_plain_lines(reply, want):
    assert plain_text.clean(reply) == want


def test_the_route_answers_and_refuses_bad_ranges(state, monkeypatch, route):
    fake_cache = review_summary.Lines(state, fake({})[1], background=False)  # never the model
    monkeypatch.setitem(review_summary._lines, str(state), fake_cache)
    entry(state, T0, 3600, "I read a book.")
    base = route

    def get(query):
        c = http.client.HTTPConnection("127.0.0.1", base.port, timeout=10)
        try:
            c.request("GET", "/review/summary?" + query)
            r = c.getresponse()
            return r.status, json.loads(r.read())
        finally:
            c.close()

    code, body = get(f"start={T0}&end={T0 + DAY}")
    assert (code, body["level"], body["start"], body["end"]) == (200, 3600, T0, T0 + DAY)
    assert body["status"] in ("pending", "partial", "ready")
    assert get(f"start={T0 + DAY}&end={T0}")[0] == 400  # backwards
    assert get(f"start={T0}&end={T0 + 40 * DAY}")[0] == 400  # too long
    assert get("start=x&end=y")[0] == 400


# ---- from the local review ----
def markers(state, done, alive=True, now=T0 + 10 * DAY):
    """Quiet markers on the memory log for every hour window of [T0, done), as the journal leaves them, and one near
    `now` when the journal is alive (it writes a marker or an entry every pass)."""
    from liberational import memory_log

    log = memory_log.at(state)
    for start in journal.window_starts(3600, T0, done):
        log.append(memory_log.JOURNAL, {"ts_start": start, "ts_end": start + 3600, "level": 3600, "status": "quiet"})
    if alive:  # a marker written within STALE_S: the journal is running
        log.append(memory_log.JOURNAL, {"ts_start": now - 700, "ts_end": now - 670, "level": 30, "status": "quiet"})


@pytest.mark.parametrize(
    ("done", "alive", "status"),
    [
        pytest.param(T0 + 3600, True, "partial", id="journal-behind-a-past-stretch"),
        pytest.param(T0 + DAY, True, "ready", id="journal-done"),
        pytest.param(T0 + 3600, False, "ready", id="journal-stopped"),
    ],
)
def test_status_follows_the_journals_markers(state, done, alive, status):
    entry(state, T0, 3600, "I read a book.")
    markers(state, done, alive)
    replies = {"I read a book.": json.dumps({"text": "Read a book", "topics": []})}
    _, second, _ = run(state, T0, T0 + DAY, replies)
    assert second["status"] == status


def test_some_lines_written_and_some_not_is_partial(state):
    entry(state, T0, 3600, "I read a book.")
    entry(state, T0 + 3600, 3600, "I cooked.")
    markers(state, T0 + DAY)
    _, rewrite = fake({"I read a book.": json.dumps({"text": "Read a book", "topics": []})})
    cache = review_summary.Lines(state, rewrite, background=False)
    review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)  # notes both entries' texts
    cache.write_one("I read a book.")
    out = review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    assert (out["status"], [b["text"] for b in out["blocks"]]) == ("partial", ["Read a book"])


@pytest.mark.parametrize(
    ("reply", "ok"),
    [
        pytest.param("Talked with Dana about her new job", True, id="her-for-someone-else"),
        pytest.param("Watched a documentary about the US", True, id="the-us"),
        pytest.param("Watched a World War I documentary", True, id="world-war-i"),
        pytest.param("Listened to They Might Be Giants", True, id="a-band"),
        pytest.param("Talked with the owner of the cafe", True, id="an-owner-mid-line"),
        pytest.param("She worked at the desk", False, id="a-pronoun-subject"),
        pytest.param("A person sat at the desk", False, id="a-person"),
        pytest.param("Someone cooked dinner", False, id="someone"),
        pytest.param("Man cooked dinner", False, id="man"),
        pytest.param("Owner had lunch in the kitchen", False, id="owner"),
        pytest.param("The narrator read a book", False, id="narrator"),
        pytest.param("You worked at the desk", False, id="you"),
        pytest.param("Worked at the desk by ourselves", False, id="ourselves"),
        pytest.param("Let's say lunch happened", False, id="lets"),
        pytest.param("Worked at my desk", False, id="my"),
        pytest.param("Then I ate lunch", False, id="i-after-the-first-word"),
        pytest.param("Lunch, then I napped", False, id="i-mid-line"),
    ],
)
def test_the_rules_keep_lines_about_others_and_drop_any_subject(reply, ok):
    assert (plain_text.clean(json.dumps({"text": reply, "topics": []})) is not None) == ok


def test_a_line_that_breaks_a_rule_is_asked_for_once_more(state):
    replies = iter(
        [json.dumps({"text": "I read a book", "topics": []}), json.dumps({"text": "Read a book", "topics": []})]
    )
    asked = []

    def rewrite(text):
        asked.append(text)
        return next(replies)

    cache = plain_text.Rewrites(state, rewrite, background=False)
    assert cache.write_one("I read a book.") == {"text": "Read a book", "topics": []}
    assert "broke a rule: no I, me, my, we, or our" in asked[1]


def test_the_first_json_object_is_read_among_words():
    reply = 'Here is {the line}: {"text": "Lunch", "topics": ["meal"]} (and {more})'
    assert plain_text.clean(reply) == {"text": "Lunch", "topics": ["meal"]}


def test_unmentioned_window_people_do_not_tag_the_line(state):
    pid = registry.at(state).person_id("Sam")
    entry(state, T0, 3600, "I saw someone.", faces=["Gus"], heard_at=[["Bob", T0 + 5, str(pid)]])
    replies = {"I saw someone.": json.dumps({"text": "Saw a visitor", "topics": ["visit"]})}
    _, second, _ = run(state, T0, T0 + DAY, replies)
    assert second["blocks"][0]["tags"] == ["topic:visit"]


def test_object_summary_does_not_inherit_alice_from_window_evidence(state):
    write_voices(state, {3: "Alice"})
    pid = registry.at(state).person_id("Alice")
    source = "I saw a black container and an orange toy vehicle on the patio."
    line = "Saw black container and orange toy vehicle on patio"
    entry(
        state,
        T0,
        3600,
        source,
        speakers=["3"],
        faces=["Alice"],
        faces_at=[["Alice", T0 + 5]],
        heard_at=[["Alice", T0 + 6, str(pid)]],
    )
    _, second, _ = run(state, T0, T0 + DAY, {source: json.dumps({"text": line, "topics": []})})
    assert second["blocks"] == [
        {"start": T0, "end": T0 + 3600, "text": line, "tags": [], "evidence": {"status": "unverified", "sources": []}}
    ]


@pytest.mark.parametrize(
    ("line", "tags"),
    [
        ("Talked with SAM about lunch", ["person:sam"]),
        ("Read about samplers", []),
        ("Talked with Sam's friend", ["person:sam"]),
    ],
)
def test_named_person_matches_without_window_presence_and_not_inside_a_word(state, line, tags):
    source = "I talked about lunch."
    entry(state, T0, 3600, source)
    _, second, _ = run(state, T0, T0 + DAY, {source: json.dumps({"text": line, "topics": []})})
    assert second["blocks"][0]["tags"] == tags


def test_named_payload_person_keeps_current_name_when_not_in_registry_people(state):
    source = "I met Gus."
    entry(state, T0, 3600, source, faces=["Gus"])
    _, second, _ = run(state, T0, T0 + DAY, {source: json.dumps({"text": "Met Gus", "topics": []})})
    assert second["blocks"][0]["tags"] == ["person:gus"]


def test_a_stretch_with_only_finer_entries_falls_back_to_them(state):
    entry(state, T0, 600, "I read a book.")
    replies = {"I read a book.": json.dumps({"text": "Read a book", "topics": []})}
    _, second, _ = run(state, T0, T0 + DAY, replies)
    assert (second["level"], [b["text"] for b in second["blocks"]]) == (600, ["Read a book"])


def test_the_same_words_are_rewritten_once_however_many_entries_have_them(state):
    entry(state, T0, 3600, "I read a book.")
    entry(state, T0 + 3600, 3600, "I read a book.")
    _, second, calls = run(state, T0, T0 + DAY, {"I read a book.": json.dumps({"text": "Read a book", "topics": []})})
    assert (calls, len(second["blocks"])) == (["I read a book."], 2)


def test_the_newest_request_is_written_first(state):
    cache = plain_text.Rewrites(state, lambda t: "{}", background=False)
    cache.many(["a", "b"])
    cache.many(["c", "a"])
    assert list(cache.todo) == ["c", "a", "b"]


def test_the_background_writer_waits_for_a_voice_turn_then_writes(state, monkeypatch):
    turns = [True, True, False]
    cache = plain_text.Rewrites(
        state, lambda t: json.dumps({"text": "Read a book", "topics": []}), idle_s=0.05,
        busy=lambda: turns.pop(0) if turns else False,
    )  # fmt: skip
    monkeypatch.setattr(plain_text.time, "sleep", lambda s: None)  # no real waiting in a test
    cache.many(["x"])
    worker = cache.worker
    assert worker is not None
    worker.join(10)
    assert turns == []  # it asked until the turn was over
    assert cache.many(["x"]) == {"x": ("ready", {"text": "Read a book", "topics": []})}


# ---- written ahead ----
NOW = T0 + 10 * DAY


def line_for(text):
    """A clean line for any entry text (the diary's "I" taken out)."""
    return json.dumps({"text": "Line " + text.replace("I ", ""), "topics": []})


def ahead(state, texts, **kw):
    """precompute() with a model that answers every entry, and the texts it was given, in order."""
    calls = []

    def rewrite(text):
        calls.append(text)
        return line_for(text)

    # One writer across passes, as the long-lived process that writes ahead has (its backfill bookmark is its own).
    cache = _writers.setdefault(str(state), review_summary.Lines(state, rewrite, background=False))
    cache.rewrite = rewrite
    kw.setdefault("busy", lambda: False)
    return review_summary.precompute(state, now=NOW, rewrites=cache, **kw), calls


_writers: dict[str, review_summary.Lines] = {}


def test_recent_lines_are_written_coarse_first_then_newest_first(state):
    entry(state, NOW - 7200, 3600, "Hour A.")
    entry(state, NOW - 3600, 3600, "Hour B.")
    entry(state, NOW - 3600, 600, "Ten C.")
    entry(state, NOW - 1200, 600, "Ten D.")
    entry(state, NOW - 600, 30, "Thirty E.")
    out, calls = ahead(state, [])
    assert calls == ["Hour B.", "Hour A.", "Ten D.", "Ten C.", "Thirty E."]
    assert out == {"written": 5, "rolled": 2, "backfilled": 0}  # then the hours' part of the day, and the day
    again, calls = ahead(state, [])
    assert (again["written"], calls) == (0, [])  # each line once


def test_the_column_finds_lines_written_ahead_without_asking_the_model(state):
    entry(state, NOW - 7200, 3600, "Hour A.")
    entry(state, NOW - 3600, 3600, "Hour B.")
    ahead(state, [])
    dashboard = review_summary.Lines(state, lambda t: pytest.fail("asked"), background=False)
    got = review_summary.summary(NOW - 8 * 3600, NOW, state, now=NOW + DAY, rewrites=dashboard)  # the journal is done
    assert (got["status"], [b["text"] for b in got["blocks"]]) == ("ready", ["Line Hour A", "Line Hour B"])


def test_a_renamed_voice_gets_its_line_written_again(state):
    entry(state, NOW - 3600, 3600, "I talked with [S2].", speakers=["2"])
    _, calls = ahead(state, [])
    write_voices(state, {2: "Danielle"})
    _, again = ahead(state, [])
    assert (calls, again) == (["I talked with Dana."], ["I talked with Danielle."])


def test_writing_ahead_stops_for_a_voice_turn_and_after_its_budget(state):
    for k in range(5):
        entry(state, NOW - 3600 + 600 * k, 600, f"Ten {k}.")
    out, calls = ahead(state, [], busy=lambda: True)
    assert (out, calls) == ({"written": 0, "rolled": 0, "backfilled": 0, "stopped": "voice"}, [])
    ticks = iter(range(0, 1000, 10))  # each look at the clock is 10 s later
    out, calls = ahead(state, [], budget_s=25, clock=lambda: next(ticks))
    assert (out["written"], out["stopped"]) == (2, "budget")


def test_older_days_are_written_a_few_a_pass_back_to_the_first_entry(state):
    for k in range(4):
        entry(state, NOW - 3 * DAY + 30 * k, 30, f"Old {k}.")
    entry(state, NOW - 5 * DAY, 30, "Older.")
    passes = [ahead(state, []) for _ in range(5)]
    assert all(o["written"] == 0 for o, _ in passes)
    # at most 3 lines a pass, a day at a time going back, at most 2 days looked at a pass
    assert [c for _, c in passes] == [["Old 3.", "Old 2.", "Old 1."], ["Old 0."], [], ["Older."], []]


def test_a_day_the_sweep_went_past_is_reached_on_its_next_round(state, monkeypatch):
    # The sweep goes back BACKFILL_DAYS, then round again from yesterday (4 here, so a round is a few passes).
    monkeypatch.setattr(review_summary, "BACKFILL_DAYS", 4)
    entry(state, NOW - 3 * DAY, 30, "Old.")
    for _ in range(3):
        ahead(state, [])
    entry(state, NOW - 1.5 * DAY, 30, "Missed.")  # behind the bookmark: written while the journal was off, say
    calls = [c for _ in range(4) for c in ahead(state, [])[1]]
    assert calls == ["Missed."]


def test_a_line_that_cant_be_saved_isnt_asked_again_every_pass(state, monkeypatch):
    entry(state, NOW - 3600, 3600, "Hour A.")
    calls = []
    cache = review_summary.Lines(state, lambda t: calls.append(t) or line_for(t), background=False)

    def full(*a):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "_save", full)
    for _ in range(3):
        review_summary.precompute(state, now=NOW, rewrites=cache, busy=lambda: False)
    assert calls == ["Hour A."]


@pytest.mark.parametrize(("enabled", "written"), [(True, 1), (False, 0)])
def test_the_journal_writes_lines_ahead_after_a_pass_unless_off(state, monkeypatch, enabled, written):
    real = journal.cfg.get
    monkeypatch.setattr(journal.cfg, "get", lambda k, d=None: enabled if k == "journal.enabled" else real(k, d))
    entry(state, time.time() - 3600, 3600, "I read a book.")
    calls = []
    lines = review_summary.Lines(state, lambda t: calls.append(t) or line_for(t), background=False)
    monkeypatch.setitem(review_summary._lines, str(state), lines)
    journal.summaries(state)
    assert calls == ["I read a book."][:written]


@pytest.fixture
def block_sources(state):
    reg = registry.at(state)
    return [
        {
            "id": "speech-1",
            "stream": "speech.utterance",
            "ts_start": T0,
            "ts_end": T0 + 2,
            "people": [{"id": str(reg.person_id("Dana")), "name": "Dana"}],
            "camera": None,
            "kind": None,
            "text": "Dana discussed the trip",
        },
        {
            "id": "caption-1",
            "stream": "camera.caption",
            "ts_start": T0,
            "ts_end": T0 + 1,
            "people": [],
            "camera": "cam1",
            "kind": None,
            "text": "A mug on a table",
        },
        {
            "id": "face-1",
            "stream": "face.match",
            "ts_start": T0,
            "ts_end": T0 + 1,
            "people": [{"id": str(reg.person_id("Sam")), "name": "Sam"}],
            "camera": "cam2",
            "kind": None,
            "text": "Sam was visible",
        },
        {
            "id": "action-1",
            "stream": "action.span",
            "ts_start": T0,
            "ts_end": T0 + 2,
            "people": [{"id": str(reg.person_id("Sam")), "name": "Sam"}],
            "camera": "cam3",
            "kind": "action",
            "activity_kind": "drink",
            "text": "Sam drank",
        },
        {
            "id": "walk-1",
            "stream": "walkby.span",
            "ts_start": T0,
            "ts_end": T0 + 2,
            "people": [],
            "camera": "cam3",
            "kind": "walkby",
            "activity_kind": "crossing",
            "text": "A walk-by",
        },
        {
            "id": "zone-1",
            "stream": "zone.visit",
            "ts_start": T0,
            "ts_end": T0 + 2,
            "people": [],
            "camera": "cam3",
            "kind": "zone",
            "text": "A zone visit",
        },
    ]


@pytest.mark.parametrize(
    ("source_id", "tags"),
    [
        ("caption-1", ["camera:cam1", "topic:rest"]),
        ("speech-1", ["person:dana", "topic:rest"]),
        ("face-1", ["camera:cam2", "person:sam", "topic:rest"]),
        ("action-1", ["camera:cam3", "event:activity:drink", "person:sam", "topic:rest"]),
        ("walk-1", ["camera:cam3", "event:activity:crossing", "topic:rest"]),
        ("zone-1", ["camera:cam3", "topic:rest"]),
    ],
)
def test_new_block_facets_use_only_its_selected_atomic_sources(state, block_sources, source_id, tags):
    entry(
        state,
        T0,
        3600,
        "I noticed a mug, heard Dana, and saw Sam.",
        speakers=["2"],
        faces=["Sam"],
        evidence={"status": "verified", "sources": block_sources},
    )
    cache = review_summary.Lines(
        state,
        lambda _: json.dumps({"text": "A quiet moment", "topics": ["rest"], "source_ids": [source_id]}),
        background=False,
    )
    review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    cache.drain()
    fresh = review_summary.Lines(state, lambda _: pytest.fail("rewrote a cached attributed line"), background=False)
    [block] = review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=fresh)["blocks"]
    assert block["tags"] == tags
    assert block["evidence"] == {
        "status": "verified",
        "sources": [next(s for s in block_sources if s["id"] == source_id)],
    }


@pytest.mark.parametrize("selection", [None, [], ["invented"], ["caption-1", "invented"], [3]])
def test_missing_or_invented_source_selection_is_explicitly_unverified(state, block_sources, selection):
    entry(
        state,
        T0,
        3600,
        "I saw a mug and Sam.",
        faces=["Sam"],
        evidence={"status": "verified", "sources": block_sources},
    )
    cache = review_summary.Lines(
        state,
        lambda _: json.dumps({"text": "A mug on a table", "topics": [], "source_ids": selection}),
        background=False,
    )
    review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    cache.drain()
    [block] = review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)["blocks"]
    assert block["evidence"]["status"] == "unverified"
    assert block["tags"] == []


def test_identical_journal_words_keep_each_windows_own_evidence_and_parent(state, block_sources):
    from liberational import memory_log

    caption = block_sources[1]
    other = {**caption, "id": "caption-2", "camera": "cam9"}
    first = entry(state, T0, 3600, "I saw a mug.", evidence={"status": "verified", "sources": [caption]})
    second = entry(state, T0 + 3600, 3600, "I saw a mug.", evidence={"status": "verified", "sources": [other]})
    calls = []

    def rewrite(source):
        data = json.loads(source)
        calls.append(data["entry_id"])
        return json.dumps(
            {"text": "A mug on a table", "topics": [], "source_ids": [data["evidence"]["sources"][0]["id"]]}
        )

    cache = review_summary.Lines(state, rewrite, background=False)
    review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    cache.drain()
    blocks = review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)["blocks"]
    assert [b["tags"] for b in blocks] == [["camera:cam1"], ["camera:cam9"]]
    assert sorted(calls) == sorted([first, second])
    rows = memory_log.at(state).range(memory_log.REVIEW_SUMMARY, T0, T0 + DAY)
    assert sorted(r["parents"] for r in rows) == sorted([[first], [second]])


def test_legacy_cached_block_stays_unverified_without_a_new_model_call(state):
    entry(state, T0, 3600, "I read a book.")
    reply = {"I read a book.": json.dumps({"text": "Read a book", "topics": ["rest"]})}
    run(state, T0, T0 + DAY, reply)
    cache = review_summary.Lines(state, lambda _: pytest.fail("regenerated legacy summary"), background=False)
    [block] = review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)["blocks"]
    assert block["evidence"] == {"status": "unverified", "sources": []}


def test_rewrite_keeps_explicit_selection_after_retrying_a_prose_rule(state, block_sources, monkeypatch):
    entry(state, T0, 3600, "I saw a mug.", evidence={"status": "verified", "sources": [block_sources[1]]})
    chat = create_autospec(
        journal.chat,
        side_effect=[
            json.dumps({"text": "I saw a mug", "topics": [], "source_ids": ["caption-1"]}),
            json.dumps({"text": "A mug on a table", "topics": [], "source_ids": ["caption-1"]}),
        ],
    )
    monkeypatch.setattr(journal, "chat", chat)
    cache = review_summary.Lines(state, background=False)
    review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    cache.drain()
    [block] = review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)["blocks"]
    assert block["tags"] == ["camera:cam1"]
    assert block["evidence"]["status"] == "verified"
    assert chat.call_count == 2
    assert all("source_ids" in call.args[0] for call in chat.call_args_list)


def test_sources_trimmed_out_of_the_rewrite_prompt_cannot_be_claimed(state, block_sources):
    sources = [{**block_sources[1], "id": f"caption-{n}", "text": "椅" * 160} for n in range(32)]
    entry(state, T0, 3600, "I saw furniture.", evidence={"status": "verified", "sources": sources})
    inputs = []

    def rewrite(source):
        inputs.append(source)
        return json.dumps({"text": "Furniture in view", "topics": [], "source_ids": ["caption-31"]})

    cache = review_summary.Lines(state, rewrite, background=False)
    review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    cache.drain()
    [block] = review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)["blocks"]
    [source] = inputs
    assert len(source.encode()) <= journal.MAX_INPUT
    assert "caption-31" not in {s["id"] for s in json.loads(source)["evidence"]["sources"]}
    assert block["evidence"] == {"status": "unverified", "sources": []}
    assert block["tags"] == []


def test_blocks_http_response_retains_selected_source_evidence(state, block_sources, monkeypatch, route):
    entry(state, T0, 3600, "I saw a mug.", evidence={"status": "verified", "sources": [block_sources[1]]})
    cache = review_summary.Lines(
        state,
        lambda _: json.dumps({"text": "A mug on a table", "topics": [], "source_ids": ["caption-1"]}),
        background=False,
    )
    review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    cache.drain()
    monkeypatch.setitem(review_summary._lines, str(state), cache)
    base = route
    conn = http.client.HTTPConnection("127.0.0.1", base.port, timeout=10)
    conn.request("GET", f"/review/summary?start={T0}&end={T0 + DAY}")
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    assert response.status == 200
    [block] = body["blocks"]
    assert block["tags"] == ["camera:cam1"]
    assert block["evidence"] == {"status": "verified", "sources": [block_sources[1]]}


def test_cached_evidence_people_follow_renames_without_rewriting_stored_sources(state, block_sources):
    from liberational import memory_log

    entry(state, T0, 3600, "I saw a familiar face.", evidence={"status": "verified", "sources": [block_sources[2]]})
    cache = review_summary.Lines(
        state, lambda _: json.dumps({"text": "A visitor", "topics": [], "source_ids": ["face-1"]}), background=False
    )
    review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    cache.drain()
    registry.at(state).rename("Sam", "Samuel", "test")
    fresh = review_summary.Lines(state, lambda _: pytest.fail("rewrote after a metadata rename"), background=False)
    [block] = review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=fresh)["blocks"]
    assert block["tags"] == ["camera:cam2", "person:samuel"]
    assert block["evidence"]["sources"][0]["people"] == [{"id": block_sources[2]["people"][0]["id"], "name": "Samuel"}]
    [stored] = memory_log.at(state).range(memory_log.REVIEW_SUMMARY, T0, T0 + DAY)
    assert stored["evidence"]["sources"][0]["people"][0]["name"] == "Sam"


def test_bound_unresolved_source_identity_does_not_reuse_a_current_display_name(state, block_sources):
    source = {**block_sources[2], "people": [{"id": "00000000-0000-4000-8000-000000000123", "name": "Sam"}]}
    entry(state, T0, 3600, "I saw a visitor.", evidence={"status": "verified", "sources": [source]})
    cache = review_summary.Lines(
        state, lambda _: json.dumps({"text": "A visitor", "topics": [], "source_ids": ["face-1"]}), background=False
    )
    review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)
    cache.drain()
    [block] = review_summary.summary(T0, T0 + DAY, state, now=T0 + 10 * DAY, rewrites=cache)["blocks"]
    assert block["tags"] == ["camera:cam2"]
    assert block["evidence"]["sources"][0]["people"] == [{"id": source["people"][0]["id"], "name": ""}]


# ---- Part-of-day and day sentences ----


@pytest.fixture
def utc(monkeypatch):
    """The local clock pinned to UTC, so the parts of the day fall on known hours."""
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


DAY0 = T0 - (T0 % DAY)  # a UTC midnight
HOURS = {
    8: ("I worked at my desk.", "Worked at the desk"),
    9: ("I had a call with [S2].", "Call with Dana"),
    14: ("I cooked lunch.", "Cooked lunch"),
}


def hours(state, which=HOURS):
    replies = {}
    for h, (text, line) in which.items():
        entry(state, DAY0 + h * 3600, 3600, text, speakers=["2"] if "[S2]" in text else [])
        replies[journal.render(text, state, journal.speakers(state))] = json.dumps({"text": line, "topics": []})
    return replies


def settle(state, replies, rollups, start=DAY0, end=DAY0 + DAY):
    """Read the day's summary until nothing more gets written, writing what each read asked for."""
    _calls, rewrite = fake(replies)
    lines = review_summary.Lines(state, rewrite, background=False)
    got = None
    for _ in range(4):
        got = review_summary.summary(start, end, state, now=end + DAY, rewrites=lines)
        lines.drain()
        rollups(state).drain()
    return got


def test_a_days_view_reads_in_parts_under_the_days_sentence(state, utc, rollups):
    rollups.replies.update(
        Morning=json.dumps({"text": "Worked at the desk, then a call with Dana", "topics": ["work", "call"]}),
        Afternoon=json.dumps({"text": "Cooked lunch", "topics": ["meal"]}),
        day=json.dumps({"text": "A work morning and lunch at home", "topics": ["work"]}),
    )
    got = settle(state, hours(state), rollups)
    assert got["days"] == [
        {
            "day": "2026-10-26",
            "start": DAY0,
            "end": DAY0 + DAY,
            "text": "A work morning and lunch at home",
            "status": "ready",
            "parts": [
                {
                    "name": "Morning",
                    "start": DAY0 + 5 * 3600,
                    "end": DAY0 + 12 * 3600,
                    "text": "Worked at the desk, then a call with Dana",
                    "status": "ready",
                },
                {
                    "name": "Afternoon",
                    "start": DAY0 + 12 * 3600,
                    "end": DAY0 + 17 * 3600,
                    "text": "Cooked lunch",
                    "status": "ready",
                },
            ],
        }
    ]
    asked = {c["name"]: (c["kind"], c["lines"]) for c in rollups.calls}
    assert len(rollups.calls) == len(asked)  # each once
    assert asked == {
        "Morning": ("part", ["08:00 Worked at the desk", "09:00 Call with Dana"]),
        "Afternoon": ("part", ["14:00 Cooked lunch"]),
        "day": ("day", ["05:00 Morning: Worked at the desk, then a call with Dana", "12:00 Afternoon: Cooked lunch"]),
    }


def test_a_part_waits_for_its_hours_still_being_written(state, utc, rollups):
    replies = hours(state)
    replies.pop(journal.render("I had a call with [S2].", state, journal.speakers(state)))
    _calls, rewrite = fake({**replies})
    down = {"n": 0}

    def flaky(text):  # the call's hour can't be written yet: the model is down for it
        if "Dana" in text:
            down["n"] += 1
            raise TimeoutError
        return rewrite(text)

    lines = review_summary.Lines(state, flaky, background=False)
    for _ in range(3):
        got = review_summary.summary(DAY0, DAY0 + DAY, state, now=DAY0 + 2 * DAY, rewrites=lines)
        lines.drain()
        rollups(state).drain()
    morning = got["days"][0]["parts"][0]
    assert (morning["name"], morning["text"], morning["status"]) == ("Morning", None, "pending")
    assert [c["name"] for c in rollups.calls] == ["Afternoon"]  # no Morning sentence thrown away a moment later
    assert got["days"][0]["text"] is None


def test_a_part_the_model_cant_sum_up_is_left_out(state, utc, rollups):
    rollups.replies["Morning"] = json.dumps({"text": "I worked all morning", "topics": []})
    got = settle(state, hours(state), rollups)
    day = got["days"][0]
    assert [p["name"] for p in day["parts"]] == ["Afternoon"]  # the feed shows the morning's hours instead
    assert sorted(c["name"] for c in rollups.calls) == ["Afternoon", "Morning", "Morning", "day"]  # once more
    assert day["status"] == "ready"


def test_a_new_hour_gives_its_part_and_day_a_new_sentence(state, utc, rollups):
    replies = hours(state)
    settle(state, replies, rollups)
    replies.update(hours(state, {10: ("I read a book.", "Read a book")}))
    rollups.calls.clear()
    rollups.replies["Morning"] = json.dumps({"text": "Worked, a call, and a book", "topics": []})
    settle(state, replies, rollups)
    assert [(c["name"], len(c["lines"])) for c in rollups.calls] == [("Morning", 3), ("day", 2)]


def test_a_part_summed_up_the_same_way_leaves_the_days_sentence_alone(state, utc, rollups):
    replies = hours(state)
    settle(state, replies, rollups)
    replies.update(hours(state, {10: ("I read a book.", "Read a book")}))
    rollups.calls.clear()
    settle(state, replies, rollups)  # the stand-in sums the new morning up in the same words
    assert [c["name"] for c in rollups.calls] == ["Morning"]


def test_finer_views_have_no_parts(state, utc):
    got = settle(state, hours(state), review_summary.rollups_at, DAY0 + 8 * 3600, DAY0 + 10 * 3600)
    assert got["level"] < 3600
    assert "days" not in got


def test_parts_and_days_are_written_ahead_and_kept(state, utc, rollups):
    now = DAY0 + 20 * 3600
    for h in (8, 14):
        entry(state, DAY0 + h * 3600, 3600, f"Hour {h}.")
    out, _ = ahead_at(state, now)
    assert out["rolled"] == 3
    assert [(c["kind"], c["name"]) for c in rollups.calls] == [
        ("part", "Morning"),
        ("part", "Afternoon"),
        ("day", "day"),
    ]
    fresh = review_summary.Rollups(state, lambda s: pytest.fail("asked again"), background=False)
    lines = review_summary.Lines(state, lambda t: pytest.fail("asked again"), background=False)
    got = review_summary.summary(DAY0, DAY0 + DAY, state, now=now, rewrites=lines, rollups=fresh)
    assert [p["text"] for p in got["days"][0]["parts"]] == [json.loads(ROLLED)["text"]] * 2
    assert got["days"][0]["text"] == json.loads(ROLLED)["text"]


def ahead_at(state, now):
    calls = []

    def rewrite(text):
        calls.append(text)
        return line_for(text)

    cache = review_summary.Lines(state, rewrite, background=False)
    return review_summary.precompute(state, now=now, rewrites=cache), calls


def test_the_days_sentence_waits_for_a_part_whose_hours_are_all_on_their_way(state, utc, rollups):
    _calls, rewrite = fake(hours(state))

    def flaky(text):  # the afternoon's only hour can't be written yet: the model is down for it
        if "lunch" in text:
            raise TimeoutError
        return rewrite(text)

    lines = review_summary.Lines(state, flaky, background=False)
    for _ in range(4):
        got = review_summary.summary(DAY0, DAY0 + DAY, state, now=DAY0 + 2 * DAY, rewrites=lines)
        lines.drain()
        rollups(state).drain()
    day = got["days"][0]
    assert [(p["name"], p["status"]) for p in day["parts"]] == [("Morning", "ready"), ("Afternoon", "pending")]
    assert [c["name"] for c in rollups.calls] == ["Morning"]  # no day sentence from the morning alone
    assert (day["text"], got["status"]) == (None, "partial")


def test_a_past_day_keeps_the_client_asking_until_its_sentences_are_written(state, utc, rollups):
    _calls, rewrite = fake(hours(state))
    lines = review_summary.Lines(state, rewrite, background=False)
    review_summary.summary(DAY0, DAY0 + DAY, state, now=DAY0 + 10 * DAY, rewrites=lines)
    lines.drain()  # every hour written, no sentence yet
    statuses = []
    for _ in range(4):
        statuses.append(review_summary.summary(DAY0, DAY0 + DAY, state, now=DAY0 + 10 * DAY, rewrites=lines)["status"])
        rollups(state).drain()
    assert statuses == ["partial", "partial", "ready", "ready"]  # the parts, then the day, then nothing more


# ---- A gist, never the diarist's own name, plain words ----


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr(review_summary.cfg, "owner_name", lambda: "Ann")


def reply(text):
    return json.dumps({"text": text, "topics": []})


@pytest.mark.parametrize(
    ("first", "rule"),
    [
        pytest.param("Talked with Ann and Dana all morning", "never write Ann", id="names-the-diarist"),
        pytest.param(
            "Worked at the desk, then a call with Dana about the trip, then more desk work, then lunch, then a walk "
            "outside, then emails, then a nap",
            "the gist rather than a list",
            id="a-list",
        ),
    ],
)
def test_a_rollup_that_names_the_diarist_or_lists_everything_is_asked_for_again(
    state, utc, owner, rollups, first, rule
):
    rollups.replies["Morning"] = [reply(first), reply("A desk-heavy morning, with a call with Dana")]
    got = settle(state, hours(state), rollups)
    assert got["days"][0]["parts"][0]["text"] == "A desk-heavy morning, with a call with Dana"
    morning = [src for src, c in zip(rollups.sources, rollups.calls, strict=True) if c["name"] == "Morning"]
    assert len(morning) == 2
    assert rule in morning[1]


def test_a_name_that_only_starts_with_the_diarists_is_kept(state, utc, owner, rollups):
    rollups.replies["Morning"] = reply("A morning of calls with Annabel")
    got = settle(state, hours(state), rollups)
    assert got["days"][0]["parts"][0]["text"] == "A morning of calls with Annabel"


def test_the_hour_lines_are_told_the_diarists_name_and_plain_words(owner, monkeypatch):
    asked = []
    monkeypatch.setattr(journal, "chat", lambda system, *a, **k: asked.append(system) or "{}")
    source = json.dumps({"format": "review-evidence-v1", "entry_id": "x", "text": "I worked.", "evidence": {}})
    review_summary.Lines._rewrite(source)
    review_summary.Lines._rewrite("I worked on my laptop.")
    assert all("never write Ann" in system and '"on the laptop"' in system for system in asked)
    assert len(asked) == 2


def test_the_rollup_prompt_describes_the_shape_for_each_kind(owner, monkeypatch):
    asked = []
    monkeypatch.setattr(journal, "chat", lambda system, *a, **k: asked.append(system) or "{}")
    for kind, name in (("part", "Afternoon"), ("day", "day")):
        review_summary.Rollups._rewrite_rollup(json.dumps({"format": "review-rollup-v3", "kind": kind, "name": name}))
    part, day = asked
    assert ("At most 20 words" in part, "At most 25 words" in day) == (True, True)
    assert "the afternoon was mostly about" in part and "the day was mostly about" in day
    assert all("(Ann)" in system for system in asked)  # never named


@pytest.mark.parametrize(
    ("line", "read"),
    [
        pytest.param("Worked on moving plans with Ann", "Worked on moving plans", id="with"),
        pytest.param("Talked with Ann and Dana about the trip", "Talked with Dana about the trip", id="name-and"),
        pytest.param("Cooked with Dana and Ann", "Cooked with Dana", id="and-name"),
        pytest.param("Dinner with Annabel", "Dinner with Annabel", id="a-longer-name-stays"),
    ],
)
def test_a_rollup_reads_its_hours_without_the_diarists_name(owner, line, read):
    assert review_summary.without_owner(line) == read


def test_an_hour_naming_the_diarist_no_longer_costs_its_part(state, utc, owner, rollups):
    got = settle(
        state, hours(state, {15: ("I worked on moving plans with [S2].", "Worked on moving plans with Ann")}), rollups
    )
    assert [p["name"] for p in got["days"][0]["parts"]] == ["Afternoon"]
    assert [c["lines"] for c in rollups.calls if c["name"] == "Afternoon"] == [["15:00 Worked on moving plans"]]
