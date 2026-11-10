"""Journal search across levels and the search-or-answer loop, with a fake Ollama and a temp state dir."""

import contextvars
import datetime as dt
import io
import json
from unittest.mock import create_autospec

import pytest

import speech_fake
from liberational import config as cfg
from liberational import memory_log
from liberational import ollama_health as oh
from liberational.domain import kernel
from liberational.memory import embedding, question, recall, video_evidence
from liberational.memory import history_qa as h
from liberational.people import registry
from voices import write_voices

_REAL_HITS = recall._hits

# A fixed test offset keeps the model strings and explicit UTC timestamps comparable on any runner.
TZ = dt.timezone(dt.timedelta(hours=-7))
NOW = dt.datetime(2026, 10, 23, 12, tzinfo=TZ).timestamp()
YESTERDAY = NOW - 86400  # 2026-10-22 12:00
WINDOW = (YESTERDAY - 12 * 3600, YESTERDAY + 12 * 3600)  # all of 2026-10-22
NEAR = (YESTERDAY - 3600, YESTERDAY + 3600)
PRODUCER = {"model": "test-llm", "version": "1", "prompt_id": "journal.v1", "prompt_hash": "abc"}
Q = "what did Dana say yesterday"


@pytest.fixture(autouse=True)
def fixed_timezone(monkeypatch):
    monkeypatch.setattr(h, "TZ", TZ)


@pytest.fixture
def state(tmp_path):
    s = tmp_path / "state"
    write_voices(s, {3: "Dana", 4: "Alice"}, forwards={7: 3})
    return s


class Fake:
    """recall's model calls, answered in order: decisions from replies (then "answer"), rankings from ranks (then
    junk, so the BM25 order stands). A reply is a dict (sent as JSON), a str, or an exception to raise."""

    def __init__(self):
        self.calls, self.replies, self.ranks = [], [], []

    def __call__(self, payload):
        self.calls.append(payload)
        ranking = payload["format"] == recall.RANK_SCHEMA
        queue = self.ranks if ranking else self.replies
        reply = queue.pop(0) if queue else ("junk" if ranking else {"decision": "answer"})
        if isinstance(reply, Exception):
            raise reply
        return {"message": {"content": reply if isinstance(reply, str) else json.dumps(reply)}}

    def decisions(self):
        return [c for c in self.calls if c["format"] != recall.RANK_SCHEMA]


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    monkeypatch.setattr(recall, "_post", f)
    monkeypatch.setattr(recall, "settings", lambda: ("test-llm", 16384))
    with cfg.override({"recall": {"max_rounds": 3}}):
        yield f


@pytest.mark.parametrize(
    "recall_model, ask_model, num_ctx, expected",
    [
        pytest.param("", "", 0, ("synthetic", 4096), id="defaults-share-the-voice-model"),
        pytest.param("", "big", 0, ("big", 16384), id="follows-ask-model"),
        pytest.param("", "big", 32768, ("big", 32768), id="follows-ask-context"),
        pytest.param("other", "big", 32768, ("other", 16384), id="own-recall-model"),
        pytest.param("synthetic", "big", 32768, ("synthetic", 4096), id="shared-planner-keeps-voice-context"),
    ],
)
def test_the_loop_uses_asks_model_and_context_unless_recall_model_is_set(recall_model, ask_model, num_ctx, expected):
    overrides = {
        "recall": {"model": recall_model},
        "ask": {"model": ask_model, "num_ctx": num_ctx},
        "models": {"ollama": {"llm": "synthetic"}},
    }
    with cfg.override(overrides):
        assert recall.settings() == expected


@pytest.mark.parametrize("ask_model, keep_alive", [("", -1), ("big", "30m")])
def test_the_loop_lets_an_own_ask_model_go_when_idle(monkeypatch, ask_model, keep_alive):
    sent = []
    monkeypatch.setattr(recall, "_post", lambda payload: sent.append(payload) or {"message": {"content": "{}"}})
    with cfg.override({"ask": {"model": ask_model}, "models": {"ollama": {"llm": "synthetic"}}}):
        recall._chat("system", {}, {}, 10)
    assert (sent[0]["model"], sent[0]["keep_alive"]) == (ask_model or "synthetic", keep_alive)


@pytest.mark.parametrize(
    "recall_model, url",
    [("", "http://127.0.0.1:11435"), ("other", "http://127.0.0.1:11434")],
    ids=["follows-ask-to-its-ollama", "own-model-stays-on-the-shared-ollama"],
)
def test_the_loop_calls_asks_ollama_only_with_asks_model(recall_model, url):
    overrides = {
        "recall": {"model": recall_model},
        "ask": {"model": "big", "ollama_url": "http://127.0.0.1:11435"},
        "ollama": {"url": "http://127.0.0.1:11434"},
    }
    with cfg.override(overrides):
        assert recall.url(recall.settings()[0]) == url


@pytest.mark.parametrize("context,max_bytes", [(4096, 11000), (8192, 22000)])
def test_planner_compacts_old_snippets_but_keeps_search_decisions_and_recent_evidence(monkeypatch, context, max_bytes):
    sent = Fake()
    monkeypatch.setattr(recall, "_post", create_autospec(recall._post, side_effect=sent))
    monkeypatch.setattr(recall, "settings", lambda: ("compact-planner", context))
    rounds = [
        {"round": n, "tool": "transcript", "args": {"text": f"topic {n}"}, "items": [f"round {n}: " + "x" * 400] * 8}
        for n in range(1, 9)
    ]
    prompt = {"question": "What was decided?", "earlier_rounds": rounds, "evidence_so_far": {"first": [], "count": 64}}

    assert recall._chat("Search instructions.", prompt, {}, 300) == {"decision": "answer"}

    request = sent.calls[0]
    content = request["messages"][1]["content"]
    fitted = json.loads(content)
    assert len(content.encode()) < max_bytes
    assert [(r["tool"], r["args"]) for r in fitted["earlier_rounds"]] == [
        ("transcript", {"text": f"topic {n}"}) for n in range(1, 9)
    ]
    assert fitted["earlier_rounds"][-1]["items"] == rounds[-1]["items"]
    assert fitted["earlier_rounds"][0]["items"] == []
    assert rounds[0]["items"] == ["round 1: " + "x" * 400] * 8  # fitting does not alter retained history
    assert request["think"] is False and request["options"]["num_ctx"] == context


def test_planner_does_not_send_oversized_instructions_or_drop_search_decisions(monkeypatch):
    post = create_autospec(recall._post, return_value={"message": {"content": "{}"}})
    monkeypatch.setattr(recall, "_post", post)
    monkeypatch.setattr(recall, "settings", lambda: ("compact-planner", 4096))
    with pytest.raises(ValueError, match="planner context"):
        recall._chat("Instructions " * 2000, {"earlier_rounds": []}, {}, 300)
    post.assert_not_called()


def test_planner_learns_token_density_and_compacts_the_next_call_without_mutating_evidence(monkeypatch):
    requests = []

    def post(payload):
        requests.append(payload)
        return {
            "model": "planner-density-test",
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 3400,
            "eval_count": 9,
            "message": {"role": "assistant", "content": '{"decision":"answer"}'},
        }

    monkeypatch.setattr(recall, "_post", create_autospec(recall._post, side_effect=post))
    monkeypatch.setattr(recall, "settings", lambda: ("planner-density-test", 4096))
    old_items, recent_items = ["Old evidence. " * 40] * 4, ["Recent evidence. " * 40] * 4
    prompt = {
        "question": "What was decided?",
        "earlier_rounds": [
            {"round": 1, "tool": "transcript", "args": {"text": "old topic"}, "items": old_items},
            {"round": 2, "tool": "camera", "args": {"text": "recent topic"}, "items": recent_items},
        ],
    }

    assert recall._chat("Search instructions.", prompt, {}, 300) == {"decision": "answer"}
    assert recall._chat("Search instructions.", prompt, {}, 300) == {"decision": "answer"}

    first = json.loads(requests[0]["messages"][1]["content"])
    second = json.loads(requests[1]["messages"][1]["content"])
    assert len(first["earlier_rounds"][0]["items"]) == 4
    assert len(second["earlier_rounds"][0]["items"]) < 4
    assert second["earlier_rounds"][1]["items"] == recent_items
    assert second["earlier_rounds"][0]["args"] == {"text": "old topic"}
    assert old_items == ["Old evidence. " * 40] * 4


def put(state, text, start, level=30, people=(), **extra):
    """A journal entry on the (in-memory) memory log, about `people` (by name, registered when new), as the journal
    worker writes one: its people are its subjects."""
    reg = registry.at(state)
    subjects = [str(reg.ensure(n, "test")) for n in people]
    record = {"ts_start": start, "ts_end": start + level, "text": text, "level": level, **extra}
    return memory_log.at(state).append(memory_log.JOURNAL, record, subjects=subjects)


def found(state, person, text="garden", window=NEAR):
    return sorted(r["text"] for r in recall.search(text, *window, person=person, rank=False, state=state))


def test_overview_keeps_early_context_without_keyword_index_or_model_calls(state, monkeypatch):
    first = put(state, "Dana arranged chairs beside the entrance.", YESTERDAY, 30)
    put(state, "Later, Alice checked the garden.", YESTERDAY + 600, 30)

    def unavailable(*args, **kwargs):
        pytest.fail("overview must not call a search index or model")

    monkeypatch.setattr(recall, "_hits", unavailable)
    monkeypatch.setattr(recall, "_query_vector", unavailable)
    monkeypatch.setattr(recall, "_post", unavailable)
    items = recall.overview(*NEAR, person=None, state=state)
    assert [r["interpretation"] for r in items] == [first]
    assert items[0]["id"] == "J1"
    assert items[0]["source"] == "journal"


def test_overview_prefers_longest_summary_at_earliest_start_and_obeys_limit(state):
    fine = put(state, "A fine detail.", YESTERDAY, 30)
    coarse = put(state, "The whole gathering.", YESTERDAY, 180)
    put(state, "An hour later.", YESTERDAY + 600, 180)
    assert [r["interpretation"] for r in recall.overview(*NEAR, None, state)] == [coarse]
    assert [r["interpretation"] for r in recall.overview(*NEAR, None, state, limit=2)] == [coarse, fine]


def test_overview_excludes_summaries_extending_outside_requested_window(state):
    put(state, "Starts before the window.", YESTERDAY - 10, 30)
    put(state, "Ends after the window.", YESTERDAY, 180)
    inside = put(state, "Contained exactly through the end.", YESTERDAY + 30, 30)
    assert [r["interpretation"] for r in recall.overview(YESTERDAY, YESTERDAY + 60, None, state)] == [inside]


def test_overview_honors_merged_person_identity_and_unknown_people(state):
    put(state, "Alice arrived first.", YESTERDAY, 30, people=["Alice"])
    selected = put(state, "Dee arrived later.", YESTERDAY + 30, 30, people=["Dee"])
    registry.at(state).merge("Dee", "Dana", "test")
    assert [r["interpretation"] for r in recall.overview(*NEAR, "Dana", state)] == [selected]
    assert recall.overview(*NEAR, "Unknown", state) == []


def test_overview_uses_current_entries_without_quiet_or_failed_markers(state):
    put(state, "Quiet marker.", YESTERDAY - 60, 30, status=memory_log.QUIET)
    put(state, "Failed marker.", YESTERDAY - 30, 30, status="failed")
    old = put(state, "An outdated summary.", YESTERDAY, 30)
    corrected = put(state, "A corrected summary.", YESTERDAY, 30, supersedes=old)
    assert [r["interpretation"] for r in recall.overview(*NEAR, None, state)] == [corrected]


@pytest.mark.parametrize(
    "start, end, limit, enabled",
    [
        (0, 172801, 1, True),
        (0, float("inf"), 1, True),
        (float("nan"), 100, 1, True),
        (100, 100, 1, True),
        (101, 100, 1, True),
        (0, 60, 0, True),
        (0, 60, 1, False),
    ],
)
def test_overview_skips_unbounded_disabled_or_empty_requests_before_reading(
    state, monkeypatch, start, end, limit, enabled
):
    def unavailable(*args, **kwargs):
        pytest.fail("ineligible overview must not read the journal")

    monkeypatch.setattr(memory_log, "at", unavailable)
    with cfg.override({"journal": {"enabled": enabled}}):
        assert recall.overview(start, end, None, state, limit=limit) == []


def test_search_without_a_journal_is_empty(state, fake):
    assert recall.search("venue", state=state) == []
    assert fake.calls == []


def test_search_per_level_time_and_person(state, fake):
    a = put(state, "[S3] asked about the venue for the party.", YESTERDAY, 30, people=["Dana"])
    put(state, "[S4] talked about the venue deposit.", YESTERDAY + 60, 30, people=["Alice"])
    b = put(state, "An hour of planning the venue with [S7].", YESTERDAY, 3600, people=["Dee"])
    registry.at(state).merge("Dee", "Dana", "test")  # Dee turned out to be Dana
    put(state, "[S3] mentioned the venue last week.", YESTERDAY - 7 * 86400, 30, people=["Dana"])
    put(state, "Lunch with someone at the venue.", YESTERDAY + 120, 180, people=["Dana"])
    put(state, "[S3] watered the garden.", YESTERDAY + 300, 30, people=["Dana"])

    items = recall.search("what about the venue", *WINDOW, rank=False, state=state)
    # The week-old entry is outside the window, the garden one doesn't match.
    assert [r["id"] for r in items] == ["J1", "J2", "J3", "J4"]
    assert {r["kind"] for r in items} == {"journal"}
    first = next(r for r in items if r["interpretation"] == a)
    assert first["text"] == "Dana asked about the venue for the party."
    assert (first["level"], first["start"], first["end"]) == (30, YESTERDAY, YESTERDAY + 30)
    assert first["window"] == "2026-10-22 12:00 to 12:00"
    assert first["href"] == f"/review#day=2026-10-22&t={YESTERDAY}"
    assert first["timestamp"] == "2026-10-22T12:00:00-07:00"

    dana = recall.search("venue", *WINDOW, person="dana", rank=False, state=state)
    # Hers, and the one about Dee, merged into her; never Alice's.
    assert len(dana) == 3 and b in {r["interpretation"] for r in dana}
    assert all("deposit" not in r["text"] for r in dana)
    assert found(state, "Bob", "venue", WINDOW) == []
    assert fake.calls == []  # rank=False makes no model call

    with cfg.override({"recall": {"per_level": 1}}):
        per = recall.search("venue", *WINDOW, rank=False, state=state)
    assert sorted(r["level"] for r in per) == [30, 180, 3600]


def test_the_person_filter_follows_a_rename_until_someone_new_takes_the_old_name(state, fake):
    reg = registry.at(state)
    put(state, "[S3] talked about the garden.", YESTERDAY, 30, people=["Dana"])
    put(state, "Someone watered the garden.", YESTERDAY + 60, 30, people=["Dana"])
    reg.rename("Dana", "Dee", "dashboard")
    want = ["Dee talked about the garden.", "Someone watered the garden."]
    # While nobody else has it, the former name finds them too.
    assert found(state, "Dee") == found(state, "Dana") == want
    reg.ensure("Dana", "dashboard")  # later, someone new is called Dana
    assert found(state, "Dee") == want
    assert found(state, "Dana") == []


@pytest.mark.parametrize("asked", ["what did Dee book", "what did Dana book"], ids=["new-name", "former-name"])
def test_a_keyword_search_finds_a_renamed_person_under_every_name_they_had(state, fake, asked):
    reg = registry.at(state)
    put(state, "[S3] booked the venue.", YESTERDAY, 30, people=["Dana"], search="Dana booked the venue.")
    reg.rename("Dana", "Dee", "dashboard")
    put(state, "[S3] booked the caterer.", YESTERDAY + 60, 30, people=["Dee"], search="Dee booked the caterer.")
    put(state, "Alice booked a table.", YESTERDAY + 120, 30, people=["Alice"])
    assert sorted(r["quote"] for r in recall.search(asked, *NEAR, rank=False, state=state)) == [
        "Dee booked the caterer.",
        "Dee booked the venue.",
    ]


def test_a_person_named_with_a_stop_word_is_searched_for_but_the_word_is_not(state, fake):
    put(state, "Will fixed the gate.", YESTERDAY, 30, people=["Will"])
    assert [r["quote"] for r in recall.search("What did Will do", *NEAR, rank=False, state=state)] == [
        "Will fixed the gate."
    ]
    assert not recall.searchable("what will we do", state)


def test_someone_new_who_takes_a_forgotten_persons_name_doesnt_get_their_entries(state, fake, monkeypatch):
    reg = registry.at(state)
    put(state, "Someone watered the garden.", YESTERDAY, 30, people=["Gus"])
    monkeypatch.setattr(kernel, "now", lambda: YESTERDAY + 600)
    reg.forget("Gus", "dashboard")
    reg.ensure("Gus", "dashboard")  # someone new is called Gus
    assert found(state, "Gus") == []


def test_a_superseded_entry_or_a_quiet_marker_is_never_evidence(state, fake):
    old = put(state, "We planned the garden.", YESTERDAY, 30)
    put(state, "We planned the garden and the shed.", YESTERDAY, 30, supersedes=old)
    put(state, "garden", YESTERDAY + 60, 30, status=memory_log.QUIET)
    assert found(state, None) == ["We planned the garden and the shed."]


def test_meaning_hits_join_the_word_hits(state, fake, journal_search):
    words = put(state, "We planned the garden.", YESTERDAY, 30)
    meaning = put(state, "Tomatoes went in by the fence.", YESTERDAY + 60, 30)  # no word in common
    journal_search.near[:] = [meaning]
    got = [r["interpretation"] for r in recall.search("garden", *NEAR, rank=False, state=state)]
    assert sorted(got) == sorted([words, meaning])


def test_hybrid_fuses_candidates_before_per_level_limit(state, monkeypatch):
    monkeypatch.setattr(
        recall, "_query_vector", create_autospec(recall._query_vector, return_value=("synthetic", "v1", [1.0]))
    )
    lexical = put(state, "Dana discussed the garden.", YESTERDAY)
    semantic = put(state, "Seedlings beside the fence.", YESTERDAY + 60)
    shared = put(state, "Dana planted tomatoes in the garden.", YESTERDAY + 120)
    hits = create_autospec(
        _REAL_HITS,
        return_value=(
            [indexed_hit(lexical), indexed_hit(shared, start=YESTERDAY + 120, end=YESTERDAY + 150)],
            [
                indexed_hit(semantic, start=YESTERDAY + 60, end=YESTERDAY + 90),
                indexed_hit(shared, start=YESTERDAY + 120, end=YESTERDAY + 150),
            ],
        ),
    )
    monkeypatch.setattr(recall, "_hits", hits)
    with cfg.override({"recall": {"per_level": 1}}):
        got = recall.search("Dana garden", *NEAR, rank=False, state=state)
    assert [item["interpretation"] for item in got] == [shared]


def indexed_hit(payload_id, parent_id=None, start=YESTERDAY, end=YESTERDAY + 30):
    return {
        "payload_id": payload_id,
        "entity_id": None,
        "parent_id": parent_id,
        "start_ns": int(start * 1e9),
        "end_ns": int(end * 1e9),
        "score": 0.5,
    }


def test_semantic_embedding_hits_hydrate_journal_parent_but_lexical_hits_keep_own_identity(state, monkeypatch):
    meaning = put(state, "Tomatoes by the fence.", YESTERDAY)
    words = put(state, "Planning the garden.", YESTERDAY + 60)
    unrelated = put(state, "An unrelated source.", YESTERDAY + 60)
    embedding = "10000000-0000-4000-8000-000000000001"
    hits = create_autospec(
        recall._hits,
        return_value=(
            [indexed_hit(words, unrelated, YESTERDAY + 60, YESTERDAY + 90)],
            [indexed_hit(embedding, meaning), indexed_hit("10000000-0000-4000-8000-000000000002", meaning)],
        ),
    )
    monkeypatch.setattr(recall, "_hits", hits)
    monkeypatch.setattr(
        recall, "_query_vector", create_autospec(recall._query_vector, return_value=("test", "v1", [1.0]))
    )
    results = recall.search("garden", *NEAR, state=state, rank=False)
    assert {r["interpretation"] for r in results} == {meaning, words}
    assert len(results) == 2
    assert embedding not in {r["interpretation"] for r in results}


@pytest.mark.parametrize("excluded", ["superseded", "quiet", "other_person", "outside_hit_span", "missing"])
def test_semantic_parent_hydration_never_revives_ineligible_journal_entries(state, monkeypatch, excluded):
    meaning = put(state, "Tomatoes by the fence.", YESTERDAY, people=["Dana"])
    subject = None
    hit = indexed_hit("10000000-0000-4000-8000-000000000001", meaning)
    if excluded == "superseded":
        put(state, "A corrected summary.", YESTERDAY, supersedes=meaning)
    elif excluded == "quiet":
        meaning = put(state, "Quiet marker.", YESTERDAY, status=memory_log.QUIET)
        hit["parent_id"] = meaning
    elif excluded == "other_person":
        registry.at(state).ensure("Alice", "test")
        subject = "Alice"
    elif excluded == "outside_hit_span":
        hit = indexed_hit(hit["payload_id"], meaning, YESTERDAY + 60, YESTERDAY + 90)
    else:
        hit["parent_id"] = "10000000-0000-4000-8000-000000000003"
    monkeypatch.setattr(recall, "_hits", create_autospec(recall._hits, return_value=([], [hit])))
    monkeypatch.setattr(
        recall, "_query_vector", create_autospec(recall._query_vector, return_value=("test", "v1", [1.0]))
    )
    assert recall.search("garden", *NEAR, person=subject, state=state, rank=False) == []


def test_the_facts_tool_reads_the_persons_facts_and_their_merged_ones(state, fake):
    reg = registry.at(state)
    dana, dee = str(reg.person_id("Dana")), str(reg.ensure("Dee", "test"))
    log = memory_log.at(state)
    for who, fact in ((dana, "Dana walks the dog."), (dee, "Dee runs Acme.")):
        record = {"ts_start": YESTERDAY, "ts_end": YESTERDAY + 3600, "fact": fact, "kind": "habit", "person": who}
        log.append(memory_log.FACTS, record, subjects=[who])
    reg.merge("Dee", "Dana", "test")
    items, note = recall.run("facts", {"start": WINDOW[0], "end": WINDOW[1], "person": "Dana"}, Q, state, [])
    assert note is None
    assert [i["quote"] for i in items] == ["Dana walks the dog.", "Dee runs Acme."]
    assert {i["kind"] for i in items} == {"fact"}


@pytest.fixture
def plain(state, fake):
    """Two entries at different levels, and their BM25 order."""
    put(state, "Dana booked the venue.", YESTERDAY, 30)
    put(state, "A long afternoon about the venue and venue costs.", YESTERDAY, 3600)
    return [r["interpretation"] for r in recall.search("venue", state=state, rank=False)]


def ranked(state):
    return [r["interpretation"] for r in recall.search("venue", state=state)]


def test_rank_orders_by_the_models_pick_and_may_leave_entries_out(state, fake, plain):
    fake.ranks += [{"ids": ["2", "1"]}, {"ids": ["2"]}]
    assert ranked(state) == plain[::-1]
    entries = json.loads(fake.calls[-1]["messages"][1]["content"])["entries"]
    assert sorted(e["level"] for e in entries) == ["1 h", "30 s"]
    assert ranked(state) == plain[1:]


@pytest.mark.parametrize(
    "junk",
    [RuntimeError("down"), "not json", {"ids": ["99"]}, {"ids": []}, {"other": 1}, {"ids": ["²"]}, {"ids": ["--1"]}],
    ids=["model-down", "not-json", "unknown-id", "none-picked", "wrong-shape", "superscript-digit", "double-minus"],
)
def test_rank_falls_back_to_bm25(state, fake, plain, junk):
    fake.ranks.append(junk)
    assert ranked(state) == plain


def test_search_ignores_question_words(state, fake):
    put(state, "What did they do? Nothing.", YESTERDAY, 30)
    assert recall.search("what did you do", state=state, rank=False) == []


def test_the_journal_switch_turns_its_search_and_tool_off(state, fake):
    put(state, "We planned the venue with Dana", YESTERDAY)
    assert recall.search("venue", state=state, rank=False)
    assert "journal" in recall.tools()
    with cfg.override({"journal": {"enabled": False}}):
        assert recall.search("venue", state=state, rank=False) == []
        assert "journal" not in recall.tools()


def test_when_reads_the_times_the_model_writes():
    half_past_two = YESTERDAY + 2.5 * 3600
    assert recall.when("2026-10-22 14:30", WINDOW) == half_past_two
    assert recall.when("2026-10-22T14:30:15", WINDOW) == half_past_two + 15
    assert recall.when("14:30", WINDOW) == half_past_two  # on the window's last day
    assert recall.when("2:30 pm", WINDOW) == half_past_two
    assert recall.when("2026-10-22", WINDOW) == WINDOW[0]
    assert recall.when("2026-10-22", WINDOW, end=True) == WINDOW[1]
    assert recall.when("2026-10-22T21:30:00+00:00", WINDOW) == half_past_two


@pytest.mark.parametrize("text", ["noon", "25:00", "13:00 pm", "yesterday"])
def test_when_refuses_what_is_not_a_time(text):
    with pytest.raises(ValueError):
        recall.when(text, WINDOW)


def test_arguments_clamp_times_and_normalize_values():
    args, shown = recall.arguments("transcript", {"start": "2026-10-21 20:00", "end": "13:00"}, WINDOW)
    assert (args["start"], args["end"]) == (WINDOW[0], YESTERDAY + 3600)
    assert shown == {"start": "2026-10-22 00:00", "end": "2026-10-22 13:00"}
    args, _ = recall.arguments("camera", {"camera": "camera 3", "bogus": "x"}, WINDOW)
    assert args == {"start": WINDOW[0], "end": WINDOW[1], "camera": "cam3"}
    assert recall.arguments("visits", {"person": "me"}, WINDOW)[0]["person"] == cfg.owner_or()


@pytest.mark.parametrize(
    ("tool", "raw", "match"),
    [
        pytest.param("transcript", {"start": "2026-10-20 10:00", "end": "2026-10-20 11:00"}, "outside", id="outside"),
        pytest.param("journal", {"text": "the"}, "keywords", id="no-keywords"),
        pytest.param("visits", {}, "needs a person", id="no-person"),
        pytest.param("camera", {"camera": "lobby/other"}, "camera id", id="bad-camera"),
        pytest.param("transcript", {"person": ["Dana"]}, "short string", id="not-a-string"),
        pytest.param("transcript", "Dana", "object", id="not-an-object"),
    ],
)
def test_arguments_refuse(tool, raw, match):
    with pytest.raises(ValueError, match=match):
        recall.arguments(tool, raw, WINDOW)


def history(state, question=Q):
    return h.collect(question, state, NOW)


def run_loop(state, question=Q):
    return recall.loop(question, history(state, question), state)


def test_the_planner_is_told_how_the_question_was_read(state, fake, readings):
    asked = "what did Dana tell me about the garden yesterday"
    readings[asked] = question.Reading(people=("Dana", "me"), day="yesterday", text="ignored")
    fake.replies += [{"decision": "answer"}]
    recall.loop(asked, history(state, asked), state, NOW)
    prompt = json.loads(fake.decisions()[0]["messages"][1]["content"])
    assert prompt["read_as"] == {"source": "speech", "measure": "describe", "people": ["Dana", "me"]}


def test_journal_only_loop_rejects_other_sources_and_defaults_to_focused_search(state, fake):
    meaning = put(state, "Tomatoes in the garden.", YESTERDAY)
    lines(state, [(YESTERDAY, "Dana", 3, "Unrelated speech must stay out.")])
    start = history(state)
    start["evidence"] = []
    start["query"].update(kind="journal", search_text="garden")
    fake.replies += [
        {"decision": "search", "tool": "transcript"},
        {"decision": "search", "tool": "journal"},
        {"decision": "answer"},
    ]
    out = recall.loop("What did the journal describe yesterday?", start, state, NOW)
    assert {r["source"] for r in out["evidence"]} == {"journal"}
    assert [r["interpretation"] for r in out["evidence"]] == [meaning]
    assert out["rounds"][0]["decision"] == "error"
    assert out["rounds"][1]["args"]["text"] == "garden"
    assert fake.decisions()[0]["format"]["properties"]["tool"]["enum"] == ["journal"]


def test_journal_only_loop_with_disabled_journal_never_requests_search_decisions(state, fake):
    start = history(state)
    start["evidence"] = []
    start["query"].update(kind="journal", search_text="garden")
    with cfg.override({"journal": {"enabled": False}}):
        out = recall.loop("What did the journal describe yesterday?", start, state, NOW)
    assert out["rounds"] == []
    assert out["evidence"] == []
    assert fake.decisions() == []


@pytest.mark.parametrize("other_tool", ["journal", "transcript", "encounters"])
def test_named_camera_loop_cannot_expand_to_other_sources(state, fake, vision, other_tool):
    put(state, "A journal entry about the garden.", YESTERDAY)
    lines(state, [(YESTERDAY, "Dana", 3, "Unrelated garden speech.")])
    vision.caption(YESTERDAY, "study", "A book on a desk.")
    vision.caption(YESTERDAY, "elsewhere", "A bicycle.")
    start = history(state)
    start["evidence"] = []
    start["query"]["cameras"] = ["study"]
    fake.replies += [
        {"decision": "search", "tool": other_tool, "args": {"text": "garden"}},
        {"decision": "search", "tool": "camera"},
        {"decision": "answer"},
    ]
    out = recall.loop("What was visible on the requested camera?", start, state, NOW)
    assert out["rounds"][0]["decision"] == "error"
    assert [r["camera"] for r in out["evidence"]] == ["study"]
    assert out["rounds"][1]["args"]["camera"] == "study"
    assert fake.decisions()[0]["format"]["properties"]["tool"]["enum"] == ["camera"]


@pytest.mark.parametrize("camera_id", ["cam00", "CAM0"])
def test_named_camera_loop_defaults_preserve_exact_ids(state, fake, vision, camera_id):
    vision.caption(YESTERDAY, camera_id, "A book.")
    vision.caption(YESTERDAY, "cam0", "A bicycle.")
    start = history(state)
    start["evidence"] = []
    start["query"]["cameras"] = [camera_id]
    fake.replies.append({"decision": "search", "tool": "camera"})
    out = recall.loop("What was visible?", start, state, NOW)
    assert [r["camera"] for r in out["evidence"]] == [camera_id]
    assert out["rounds"][0]["args"]["camera"] == camera_id


@pytest.mark.parametrize("invalid_args", [None, {"camera": "elsewhere"}])
def test_multiple_camera_scope_requires_one_of_the_requested_exact_ids(state, fake, vision, invalid_args):
    vision.caption(YESTERDAY, "study", "A book.")
    vision.caption(YESTERDAY, "cam00", "A desk.")
    vision.caption(YESTERDAY, "elsewhere", "A bicycle.")
    start = history(state)
    start["evidence"] = []
    start["query"]["cameras"] = ["study", "cam00"]
    fake.replies += [
        {"decision": "search", "tool": "camera", "args": invalid_args},
        {"decision": "search", "tool": "camera", "args": {"camera": "cam00"}},
        {"decision": "answer"},
    ]
    out = recall.loop("What was visible on the requested cameras?", start, state, NOW)
    assert out["rounds"][0]["decision"] == "error"
    assert [r["camera"] for r in out["evidence"]] == ["cam00"]


def test_named_camera_frames_keep_opt_in_and_default_to_the_requested_exact_id(state, fake, monkeypatch):
    def frames(state_, query):
        assert query["cameras"] == ["cam00"]
        return [dict(ts=YESTERDAY, camera="cam00", quote="A book.", source="camera", link="/review#t=1")], {}, []

    collect = create_autospec(video_evidence.collect, side_effect=frames)
    monkeypatch.setattr(video_evidence, "collect", collect)
    start = history(state)
    start["evidence"] = []
    start["query"]["cameras"] = ["cam00"]
    fake.replies.append({"decision": "search", "tool": "frames"})
    out = recall.loop("What was visible?", start, state, NOW)
    assert out["rounds"][0]["decision"] == "error"
    collect.assert_not_called()
    fake.replies.append({"decision": "search", "tool": "frames"})
    with cfg.override({"review": {"recorded_frame_analysis": True}}):
        out = recall.loop("What was visible?", start, state, NOW)
    assert [r["camera"] for r in out["evidence"]] == ["cam00"]
    assert fake.decisions()[-2]["format"]["properties"]["tool"]["enum"] == ["camera", "frames"]


def lines(state, rows):
    speech_fake.fake().write([dict(ts_start=t, speaker=s, speaker_id=i, text=x) for t, s, i, x in rows])


def six_lines_and_five_searches(state, fake):
    lines(state, [(YESTERDAY + i * 600, "Dana", 3, f"Line {i}") for i in range(6)])
    fake.replies += [
        {"decision": "search", "tool": "transcript", "args": {"start": f"2026-10-22 {12 + i}:00"}} for i in range(5)
    ]


def test_loop_picks_tools_dedupes_and_stops(state, fake):
    put(state, "[S3] said the venue is booked for Friday.", YESTERDAY + 600, 180, people=["Dana"])
    lines(
        state,
        [
            (YESTERDAY + 610, "Dana", 3, "The venue is booked."),
            (YESTERDAY + 620, "Dana", 3, "Friday at six."),
            (YESTERDAY + 4000, "Alice", 4, "I will bring cake."),
        ],
    )
    start = history(state, "what did Dana say about the venue yesterday")
    assert [r["id"] for r in start["evidence"]] == ["J1", "T2", "T3"]
    fake.replies += [
        {"decision": "search", "tool": "journal", "args": {"text": "venue booked", "person": "Dana"}},
        {"decision": "search", "tool": "transcript", "args": {"start": "2026-10-22 13:00", "end": "2026-10-22 14:00"}},
        {"decision": "answer"},
    ]
    out = recall.loop(start["query"]["question"], start, state, NOW)
    assert [r["decision"] for r in out["rounds"]] == ["search", "search", "answer"]
    assert out["rounds"][0] == {
        "round": 1,
        "decision": "search",
        "tool": "journal",
        # A range the model left out is the question's window, shown as searched.
        "args": {"start": "2026-10-22 00:00", "end": "2026-10-23 00:00", "text": "venue booked", "person": "Dana"},
        "found": 1,
        "new": [],  # the same entry collect already found
    }
    assert out["rounds"][1]["new"] == ["T4"] and out["rounds"][1]["found"] == 1
    assert [r["id"] for r in out["evidence"]] == ["J1", "T2", "T3", "T4"]
    assert out["evidence"][3]["text"] == "I will bring cake."
    # The answering model sees what the rounds found first.
    assert [r["id"] for r in json.loads(out["context"])["evidence"]] == ["T4", "J1", "T2", "T3"]
    assert json.loads(out["context"])["evidence"][1]["window"] == "2026-10-22 12:10 to 12:13"
    # Each round shows the earlier ones compactly and gives the fixed now.
    second = json.loads(fake.decisions()[1]["messages"][1]["content"])
    assert second["now"] == "2026-10-23 12:00"
    assert second["earlier_rounds"][0]["result"] == "1 found, 0 new"
    assert second["evidence_so_far"]["first"][0].startswith("[J1] 2026-10-22 12:10 to 12:13 3 min journal: Dana said")
    assert out["warnings"] == []  # collect's warnings aren't repeated


def test_loop_stops_at_max_rounds(state, fake):
    six_lines_and_five_searches(state, fake)
    with cfg.override({"recall": {"max_rounds": 3}}):
        out = run_loop(state)
    assert [r["round"] for r in out["rounds"]] == [1, 2, 3]
    assert len(fake.decisions()) == 3


def test_loop_stops_after_two_errors(state, fake):
    fake.replies += ["not json", {"decision": "search", "tool": "web"}, {"decision": "search", "tool": "transcript"}]
    with cfg.override({"recall": {"max_rounds": 5}}):
        out = run_loop(state)
    assert [r["decision"] for r in out["rounds"]] == ["error", "error"]
    assert "unknown tool 'web'" in out["rounds"][1]["error"]
    assert len(fake.decisions()) == 2


def test_loop_counts_bad_arguments_repeats_and_tool_failures(state, fake, monkeypatch):
    def unreadable(*a):
        raise OSError("unreadable")

    monkeypatch.setattr(h, "_camera", unreadable)
    same = {"decision": "search", "tool": "visits", "args": {"person": "Dana"}}
    fake.replies += [same, same, {"decision": "search", "tool": "camera", "args": {"camera": "cam1"}}]
    with cfg.override({"recall": {"max_rounds": 5}}):
        out = run_loop(state)
    assert [r["decision"] for r in out["rounds"]] == ["search", "error", "error"]
    assert "repeat" in out["rounds"][1]["error"]
    assert out["rounds"][2]["error"] == "OSError" and out["rounds"][2]["tool"] == "camera"


def test_loop_keeps_a_round_bounded(state, fake):
    lines(state, [(YESTERDAY + i, "Dana", 3, f"Line number {i}") for i in range(50)])
    start = history(state, "what did Alice say yesterday")  # Alice said nothing: no starting evidence
    assert start["evidence"] == []
    fake.replies.append({"decision": "search", "tool": "transcript", "args": {"person": "Dana"}})
    out = recall.loop("what did Alice say yesterday", start, state)
    assert len(out["evidence"]) == 20
    assert out["evidence"][0]["text"] == "Line number 0" and out["evidence"][-1]["text"] == "Line number 49"
    assert "kept 20 of 50" in out["rounds"][0]["note"]


def test_frames_only_with_the_opt_in(state, fake, monkeypatch):
    seen = []

    def frames(state_, query):
        seen.append(query)
        item = dict(ts=YESTERDAY + 5, camera="cam1", quote="A person at the door", source="camera", link="/review#t=1")
        return [item], {"samples": 1}, ["Recorded frames are sparse."]

    monkeypatch.setattr(video_evidence, "collect", frames)
    ask = {"decision": "search", "tool": "frames", "args": {"start": "12:00", "end": "12:01", "camera": "cam1"}}
    fake.replies.append(ask)
    out = run_loop(state)
    assert out["rounds"][0]["decision"] == "error" and seen == []
    assert "frames" not in fake.decisions()[0]["format"]["properties"]["tool"]["enum"]

    fake.replies.append(ask)
    with cfg.override({"review": {"recorded_frame_analysis": True}}):
        out = run_loop(state)
    assert out["rounds"][0]["decision"] == "search"
    assert seen[0]["cameras"] == ["cam1"] and seen[0]["end"] - seen[0]["start"] == 60
    assert out["evidence"][-1]["id"].startswith("C") and "Recorded frames are sparse." in out["warnings"]


@pytest.fixture
def clock(monkeypatch, fake):
    """A fake monotonic clock (the one time module, video_evidence's too) that every model call moves on by
    clock.step seconds."""

    class Clock:
        now, step = 1000.0, 0.0

    def slow(payload):
        Clock.now += Clock.step
        return fake(payload)

    monkeypatch.setattr(recall.time, "monotonic", lambda: Clock.now)
    monkeypatch.setattr(recall, "_post", slow)
    return Clock


def test_loop_leaves_time_for_the_answer(state, fake, clock):
    six_lines_and_five_searches(state, fake)
    clock.step = 40
    with cfg.override({"recall": {"max_rounds": 5}}):
        out = run_loop(state)
    assert len(fake.decisions()) == 2  # 80 s of the budget gone, so no third round
    assert any("stopped early" in w for w in out["warnings"])
    assert recall._deadline.get() is None


def test_frames_never_outlast_the_loop(state, fake, clock, monkeypatch, vision):
    """A frames round that starts late stops at the loop's deadline, not review.video_timeout_s later."""
    lines(state, [(YESTERDAY + i * 600, "Dana", 3, f"Line {i}") for i in range(6)])
    vision.video.append(("cam1", YESTERDAY - 3600, YESTERDAY + 3600))
    clock.step = 25

    def frame(log, cam, ts):
        clock.now += 1
        return b"jpg"

    def caption(image, timeout, question="x"):
        clock.now += min(7, timeout)
        return "A person at the door"

    monkeypatch.setattr(video_evidence, "_frame", frame)
    monkeypatch.setattr(video_evidence, "_caption", caption)
    fake.replies += [
        {"decision": "search", "tool": "transcript", "args": {"start": "2026-10-22 12:00"}},
        {"decision": "search", "tool": "transcript", "args": {"start": "2026-10-22 13:00"}},
        {"decision": "search", "tool": "frames", "args": {"start": "12:00", "end": "12:05", "camera": "cam1"}},
    ]
    with cfg.override({"recall": {"max_rounds": 5}, "review": {"recorded_frame_analysis": True}}):
        hist = history(state)
        began = clock.now
        out = recall.loop(Q, hist, state)
    assert [r.get("tool") for r in out["rounds"]] == ["transcript", "transcript", "frames"]
    assert out["rounds"][2]["decision"] == "search"
    assert clock.now - began <= 80


def test_calls_inside_the_loop_never_outlast_its_deadline(monkeypatch):
    timeouts = []

    class Opener:
        def open(self, request, timeout):
            timeouts.append(timeout)
            return io.BytesIO(b'{"message": {"content": "{}"}}')

    def post_by(deadline):
        recall._deadline.set(deadline)  # run in a fresh context, so it never leaks into other tests
        recall._post({})

    monkeypatch.setattr(recall.urllib.request, "build_opener", lambda *a: Opener())
    monkeypatch.setattr(recall.time, "monotonic", lambda: 100.0)
    recall._post({})
    contextvars.Context().run(post_by, 130.0)
    with pytest.raises(TimeoutError):
        contextvars.Context().run(post_by, 100.5)
    assert timeouts == [90, 30.0]


# ---- encounters: who was here, last seen, and how long, from the timeline, citing what's behind them ----

DANA, ALICE = kernel.PersonId("p-dana"), kernel.PersonId("p-alice")


@pytest.fixture
def met(monkeypatch):
    """Synthetic encounters on 2026-10-22: Dana seen on cam1 and heard at 12:10 to 12:15, Dana heard alone at 14:00,
    and Alice seen on cam0 at 12:11 to 12:13."""
    from liberational.memory import timeline as tl

    seen = [
        tl.Observed(DANA, tl.Visit("Dana", "cam1", YESTERDAY + 600, YESTERDAY + 900, 0.81)),
        tl.Observed(ALICE, tl.Visit("Alice", "cam0", YESTERDAY + 660, YESTERDAY + 780, 0.77)),
    ]
    heard = [
        tl.Observed(DANA, tl.Line(YESTERDAY + 700, YESTERDAY + 705, "Let's plan the garden.")),
        tl.Observed(DANA, tl.Line(YESTERDAY + 7200, YESTERDAY + 7210, "See you tomorrow.")),
    ]
    built = tl.Timeline(tl.project(seen + heard, 120.0), {DANA: "Dana", ALICE: "Alice"})
    monkeypatch.setattr(tl, "load", lambda state=None: built)
    monkeypatch.setattr(tl, "person", lambda name, state=None: {"dana": DANA, "alice": ALICE}.get(name.casefold()))
    with cfg.override({"timeline": {"enabled": True, "gap_s": 120.0}}):
        yield built


def encounters(state, start=WINDOW[0], end=WINDOW[1], person=None):
    return recall.run("encounters", dict(start=start, end=end, person=person), Q, state, [])


def test_encounters_say_how_and_how_long_and_cite_the_visits_and_lines_behind_them(state, met):
    items, note = encounters(state, person="Dana")
    assert note is None
    assert [(r["source"], r["quote"]) for r in items] == [
        ("encounters", "Dana was seen and heard from 2026-10-22 12:10 to 12:15 (5 minutes) on cam1"),
        ("transcript", "Let's plan the garden."),
        ("camera", items[2]["quote"]),
        ("encounters", "Dana was heard from 2026-10-22 14:00 to 14:00 (under a minute)"),
        ("transcript", "See you tomorrow."),
    ]
    assert items[2]["quote"].startswith("Dana seen on cam1 from ")
    assert items[0]["method"].endswith("Face and voice matching are uncertain, not proof of identity.")


def test_encounters_without_a_person_say_who_was_there(state, met):
    items, _ = encounters(state, YESTERDAY + 500, YESTERDAY + 1000)
    assert [r["quote"] for r in items if r["source"] == "encounters"] == [
        "Dana was seen and heard from 2026-10-22 12:10 to 12:15 (5 minutes) on cam1",
        "Alice was seen from 2026-10-22 12:11 to 12:13 (2 minutes) on cam0",
    ]


def test_an_encounter_is_cut_to_the_range_asked_about(state, met):
    items, _ = encounters(state, YESTERDAY + 800, YESTERDAY + 1000, person="Dana")
    assert (
        items[0]["quote"] == "Dana was seen from 2026-10-22 12:13 to 12:15 (2 minutes) on cam1"
    )  # the line was before


@pytest.mark.parametrize(
    ("load", "person", "note"),
    [
        pytest.param(True, "Zed", "nobody is called Zed", id="no-such-person"),
        pytest.param(False, "Dana", "no encounters: there is no event log yet", id="no-event-log"),
    ],
)
def test_encounters_say_why_they_found_nothing(state, met, monkeypatch, load, person, note):
    from liberational.memory import timeline as tl

    if not load:
        monkeypatch.setattr(tl, "load", lambda state=None: None)
    assert encounters(state, person=person) == ([], note)


def test_encounters_are_offered_only_with_the_timeline_on():
    with cfg.override({"timeline": {"enabled": False}}):
        assert "encounters" not in recall.tools()
    with cfg.override({"timeline": {"enabled": True}}):
        assert "encounters" in recall.tools()


def test_a_mixed_question_answers_its_presence_half_from_encounters(state, fake, met):
    """ "When did I last see Dana, and what did we talk about?": the loop reads encounters for when, then the
    transcript for what, and the answering model gets the encounter with the line behind it."""
    lines(
        state,
        [(YESTERDAY + 700, "Dana", 3, "Let's plan the garden."), (YESTERDAY + 7200, "Dana", 3, "See you tomorrow.")],
    )
    question = "when did I last see Dana, and what did we talk about?"
    fake.replies += [
        {"decision": "search", "tool": "encounters", "args": {"person": "Dana"}},
        {"decision": "search", "tool": "transcript", "args": {"person": "Dana"}},
        {"decision": "answer"},
    ]
    start = history(state, question)
    out = recall.loop(question, start, state, NOW)
    assert [r.get("tool") for r in out["rounds"]] == ["encounters", "transcript", None]
    assert "encounters" in fake.decisions()[0]["messages"][0]["content"]  # offered, with its help text
    context = json.loads(out["context"])["evidence"]
    quotes = [r.get("text") or r.get("quote") for r in context]
    assert "Dana was heard from 2026-10-22 14:00 to 14:00 (under a minute)" in quotes
    assert "See you tomorrow." in quotes


def test_required_meaning_search_reports_disabled_journal_without_embedding(state, monkeypatch):
    from unittest.mock import create_autospec

    vector = create_autospec(recall._query_vector)
    monkeypatch.setattr(recall, "_query_vector", vector)
    with cfg.override({"journal": {"enabled": False}}), pytest.raises(recall.SemanticUnavailableError):
        recall.search("garden", state=state, rank=False, require_semantic=True)
    vector.assert_not_called()


def test_required_meaning_search_does_not_silently_fall_back_to_words(state, monkeypatch):
    from unittest.mock import create_autospec

    vector = create_autospec(recall._query_vector, return_value=None)
    monkeypatch.setattr(recall, "_query_vector", vector)
    with pytest.raises(recall.SemanticUnavailableError, match="embedding model"):
        recall.search("garden", state=state, rank=False, require_semantic=True)
    assert recall.search("garden", state=state, rank=False) == []


@pytest.mark.parametrize("query", ["garden", "to be"])
def test_required_meaning_search_embeds_once_and_does_not_rerank(state, fake, journal_search, monkeypatch, query):
    from unittest.mock import create_autospec

    meaning = put(state, "Tomatoes by the fence.", YESTERDAY, 30)
    journal_search.near[:] = [meaning]
    vector = create_autospec(recall._query_vector, return_value=("test-model", "v1", [1.0]))
    monkeypatch.setattr(recall, "_query_vector", vector)

    def indexed(words, vector, lo, hi, subjects, *, strict=False):
        assert strict
        return journal_search.hits(words, vector, lo, hi, subjects)

    monkeypatch.setattr(recall, "_hits", create_autospec(_REAL_HITS, side_effect=indexed))
    results = recall.search(query, *NEAR, state=state, rank=False, require_semantic=True)
    assert [r["interpretation"] for r in results] == [meaning]
    vector.assert_called_once_with(query)
    assert fake.calls == []


def test_required_meaning_search_reports_worker_failure_but_legacy_stays_empty(monkeypatch):
    import continuum_py

    class UnavailableClient:
        @staticmethod
        async def connect(*args, **kwargs):
            raise OSError("unavailable broker")

    monkeypatch.setattr(continuum_py, "Client", UnavailableClient)
    assert _REAL_HITS("garden", ("test-model", "v1", [1.0]), *NEAR, None) == ([], [])
    with pytest.raises(recall.SemanticUnavailableError, match="meaning index"):
        _REAL_HITS("garden", ("test-model", "v1", [1.0]), *NEAR, None, strict=True)


def test_required_meaning_search_reports_nearest_timeout_but_legacy_keeps_lexical(monkeypatch):
    import continuum_py
    import continuum_py.search

    class ClientContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    class Client:
        @staticmethod
        async def connect(*args, **kwargs):
            return ClientContext()

    class Searcher(ClientContext):
        def __init__(self, client):
            pass

        async def text_search(self, *args, **kwargs):
            return [{"payload_id": "lexical-entry", "start_ns": 1, "end_ns": 2, "score": 1.0}]

        async def nearest(self, *args, **kwargs):
            raise TimeoutError("nearest unavailable")

    monkeypatch.setattr(continuum_py, "Client", Client)
    monkeypatch.setattr(continuum_py.search, "Searcher", Searcher)
    assert _REAL_HITS("garden", ("test-model", "v1", [1.0]), *NEAR, None) == (
        [{"payload_id": "lexical-entry", "start_ns": 1, "end_ns": 2, "score": 1.0}],
        [],
    )
    with pytest.raises(recall.SemanticUnavailableError, match="meaning index"):
        _REAL_HITS("garden", ("test-model", "v1", [1.0]), *NEAR, None, strict=True)


@pytest.mark.parametrize("camera_id", ["entryway", "retired-yard", "lobby", "0", "cam00", "CAM0", "camera3", "camzero"])
def test_direct_tool_camera_ids_are_exact_even_without_current_configuration(camera_id):
    args, shown = recall.arguments("camera", {"camera": camera_id}, WINDOW)
    assert args["camera"] == camera_id
    assert shown["camera"] == camera_id


@pytest.mark.parametrize(
    "camera,match",
    [
        ({}, "short string"),
        ([], "short string"),
        (True, "short string"),
        ("../entryway", "camera id"),
        ("front door", "camera id"),
        ("cam0.video", "camera id"),
    ],
)
def test_direct_tool_camera_argument_rejects_non_ids(camera, match):
    with pytest.raises(ValueError, match=match):
        recall.arguments("camera", {"camera": camera}, WINDOW)


@pytest.mark.parametrize("camera", [None, ""])
def test_optional_camera_argument_remains_omitted(camera):
    args, shown = recall.arguments("camera", {"camera": camera}, WINDOW)
    assert "camera" not in args and "camera" not in shown


@pytest.mark.parametrize("camera,want", [("camera one", "cam1"), ("CAM 00", "cam0"), ("camera 3", "cam3")])
def test_direct_tool_camera_argument_retains_spaced_legacy_aliases(camera, want):
    assert recall.arguments("camera", {"camera": camera}, WINDOW)[0]["camera"] == want


def test_planner_serialized_prefix_keeps_initial_evidence_and_scope_before_round_changes(state, fake, vision):
    start = history(state)
    start["query"]["cameras"] = ["garden"]
    start["evidence"] = [
        dict(
            id="C1",
            ts=YESTERDAY,
            quote="Stable initial garden evidence.",
            source="camera",
            camera="garden",
            link="/review#t=1",
        )
    ]
    vision.caption(YESTERDAY + 30, "garden", "New tool evidence.")
    fake.replies += [{"decision": "search", "tool": "camera"}, {"decision": "answer"}]
    out = recall.loop("What was visible on the garden camera?", start, state, NOW)
    first, second = [call["messages"][1]["content"].encode() for call in fake.decisions()]
    prefix_length = next(i for i, (a, b) in enumerate(zip(first, second, strict=False)) if a != b)
    shared = first[:prefix_length]
    assert b"Stable initial garden evidence." in shared
    assert b'"cameras": ["garden"]' in shared
    assert b'"window"' in shared
    before, after = json.loads(first), json.loads(second)
    assert before["evidence_so_far"]["first"] == after["evidence_so_far"]["first"]
    assert before["evidence_so_far"]["count"] == 1
    assert after["evidence_so_far"]["count"] == 2
    assert before["earlier_rounds"] == []
    assert after["earlier_rounds"][0]["items"][0].endswith("New tool evidence.")
    assert (before["round"], after["round"]) == (1, 2)
    assert (before["rounds_left"], after["rounds_left"]) == (3, 2)
    assert before["now"] == after["now"] == "2026-10-23 12:00"
    assert before["question"] == after["question"] == "What was visible on the garden camera?"
    assert before["window"] == after["window"]
    assert [row["quote"] for row in out["evidence"]] == ["Stable initial garden evidence.", "New tool evidence."]


class EmbeddingOllama:
    """A fake /api/embed and /api/tags whose tag can change while a question is embedded."""

    def __init__(self, repull: bool):
        self.digest, self.repull = "sha256:aaaaaaaaaaaa1111", repull
        self.tags_calls = 0

    def post(self, payload, timeout):
        if self.repull:
            self.digest = "sha256:bbbbbbbbbbbb2222"
        return {"embeddings": [[1.0, 0.0, 0.0, 0.0] for _ in payload["input"]]}

    def tags(self, timeout=5):
        self.tags_calls += 1
        return [{"name": "qwen3-embedding:0.6b", "digest": self.digest}]

    def get(self, url, timeout):
        self.tags_calls += 1

        class Response:
            def __init__(self, digest):
                self._digest = digest

            def raise_for_status(self):
                return None

            def json(self):
                return {"models": [{"name": "qwen3-embedding:0.6b", "digest": self._digest}]}

        return Response(self.digest)


@pytest.mark.parametrize(
    ("repull", "expected"),
    [
        pytest.param(False, ("qwen3-embedding:0.6b", "aaaaaaaaaaaa", [1.0, 0.0, 0.0, 0.0]), id="same-weights"),
        pytest.param(
            True, ("qwen3-embedding:0.6b", "aaaaaaaaaaaa", [1.0, 0.0, 0.0, 0.0]), id="tag-changed-after-check"
        ),
    ],
)
def test_a_question_vector_checks_tags_once_per_search(monkeypatch, repull, expected):
    ollama = EmbeddingOllama(repull)
    monkeypatch.setattr(oh, "_TAGS_CACHE", None)
    monkeypatch.setattr(oh, "_get", ollama.get)
    monkeypatch.setattr(embedding, "_post", ollama.post)
    monkeypatch.setattr(embedding, "_trouble", {"why": None})
    overrides = {
        "ollama": {"url": "http://127.0.0.1:11434"},
        "semantic": {"enabled": True, "dimensions": 4},
        "models": {"ollama": {"embedding": "qwen3-embedding:0.6b"}},
    }
    with cfg.override(overrides):
        assert recall._query_vector("where was the party") == expected
    assert ollama.tags_calls == 1
