"""Ask history keeps complete local exchanges while following Continuum provenance deletion."""

import json
import uuid

import pytest

from liberational import continuum_link, memory_log
from liberational.listen import revisions
from liberational.memory import ask, ask_history

T0 = 1_700_000_000.0


@pytest.fixture(autouse=True)
def history_views():
    ask_history.close_all()
    yield
    ask_history.close_all()


def source(memory, text="Private source words.", at=T0 - 60):
    return memory.append(memory_log.JOURNAL, {"ts_start": at, "text": text, "level": 30})


def answer(payload_id, text="The event happened [J1].", ts=T0 - 60):
    return {
        "status": "ok",
        "answer": text,
        "model": "synthetic-model",
        "evidence": [
            {
                "id": "J1",
                "label": "J1",
                "ts": ts,
                "source": "journal",
                "interpretation": payload_id,
                "quote": "Private source words.",
            }
        ],
    }


def complete(history, exchange_id, result):
    return history.finish(exchange_id, result, started_at=T0 + 5, finished_at=T0 + 10, status="completed")


@pytest.fixture(params=["canonical", "uncited"])
def review_answer(request, memory):
    return answer(source(memory)) if request.param == "canonical" else {"answer": "Misheard words"}


def test_transcript_review_invalidates_saved_answers(memory, tmp_path, review_answer):
    history = ask_history.History(tmp_path)
    ident = history.start("What was said?", scope="history", created_at=T0)
    assert complete(history, ident, review_answer)
    token = revisions.reject_lines(tmp_path, [{"id": "utterance-one", "ts_start": T0 - 60, "device": "mic"}])
    saved = ask_history.History(tmp_path).get(ident)
    assert saved is not None
    assert saved["question"] == "What was said?"
    assert saved["result"] is None
    assert saved["answer_removed"]
    assert all(row["removed"] for row in saved["citations"])
    # Undo changes the reviewed evidence again; it cannot revive an answer made against an older checkpoint.
    revisions.undo_correction(tmp_path, token)
    undone = history.get(ident)
    assert undone is not None
    assert undone["result"] is None
    fresh = history.start("Ask again", created_at=T0 + 20)
    assert complete(history, fresh, {"answer": "Reviewed answer"})
    updated = history.get(fresh)
    assert updated is not None
    assert updated["result"] == {"answer": "Reviewed answer"}


def test_transcript_review_while_answer_runs_prevents_stale_completion(memory, tmp_path, review_answer):
    history = ask_history.History(tmp_path)
    ident = history.start("What was said?", scope="history", created_at=T0)
    revisions.reject_lines(tmp_path, [{"id": "utterance-one", "ts_start": T0 - 60, "device": "mic"}])
    assert complete(history, ident, review_answer)
    saved = ask_history.History(tmp_path).get(ident)
    assert saved is not None
    assert saved["result"] is None
    assert saved["answer_removed"]
    assert all(row["removed"] for row in saved["citations"])


def test_transcript_review_prevents_reusing_an_unsaved_answer(memory, tmp_path, monkeypatch):
    history = ask_history.History(tmp_path)
    monkeypatch.setattr(history, "_revisions", lambda: {"speech.utterance": 0})
    ident = history.start("What was said?", scope="history", created_at=T0)
    assert history.can_reuse_unsaved(ident)
    revisions.reject_lines(tmp_path, [{"ts_start": T0 - 60, "device": "mic"}])
    assert not history.can_reuse_unsaved(ident)


@pytest.mark.parametrize("surface", ["live", "call"])
def test_general_answer_without_evidence_survives_unrelated_transcript_review(memory, tmp_path, surface):
    history = ask_history.History(tmp_path)
    ident = history.start("What is two plus two?", surface=surface, scope="general", created_at=T0)
    result = {"answer": "Four", "evidence": []}
    assert complete(history, ident, result)
    token = revisions.reject_lines(tmp_path, [{"ts_start": T0 - 60, "device": "mic"}])
    saved = history.get(ident)
    assert saved is not None and saved["result"] == result and not saved["answer_removed"]
    assert history.can_reuse_unsaved(ident, result)
    revisions.undo_correction(tmp_path, token)
    restored = history.get(ident)
    assert restored is not None and restored["result"] == result


def test_general_answer_finishing_after_a_correction_remains_available(memory, tmp_path):
    history = ask_history.History(tmp_path)
    ident = history.start("A general question", created_at=T0)
    revisions.reject_lines(tmp_path, [{"ts_start": T0 - 60, "device": "mic"}])
    result = {"answer": "An unrelated answer"}
    assert complete(history, ident, result)
    saved = history.get(ident)
    assert saved is not None and saved["result"] == result


def test_general_answer_that_used_evidence_is_invalidated_by_transcript_review(memory, tmp_path):
    history = ask_history.History(tmp_path)
    ident = history.start("What was said?", created_at=T0)
    result = answer(source(memory))
    assert complete(history, ident, result)
    revisions.reject_lines(tmp_path, [{"ts_start": T0 - 60, "device": "mic"}])
    saved = history.get(ident)
    assert saved is not None and saved["result"] is None and saved["answer_removed"]
    assert not history.can_reuse_unsaved(ident, result)


@pytest.mark.parametrize(
    ("scope", "expected_status", "expected_answer"), [("general", "completed", "Four"), ("history", "failed", None)]
)
def test_unsaved_job_retrieval_keeps_general_answers_and_blocks_stale_uncited_history(
    memory, tmp_path, monkeypatch, scope, expected_status, expected_answer
):
    monkeypatch.setattr(ask, "STATE", tmp_path)
    history = ask_history.at(tmp_path)
    monkeypatch.setattr(history, "_revisions", lambda: {"speech.utterance": 0})
    ident = history.start("Question", scope=scope, created_at=T0)
    live = {"status": "completed", "answer_saved": False, "result": {"answer": "Four", "evidence": []}}
    revisions.reject_lines(tmp_path, [{"ts_start": T0 - 60, "device": "mic"}])
    entry = ask._saved_job(ident, live)
    assert entry is not None
    assert entry["status"] == expected_status
    assert (entry.get("result") or {}).get("answer") == expected_answer


def test_interrupted_question_and_finished_answer_survive_a_fresh_history_instance(memory, tmp_path):
    history = ask_history.History(tmp_path)
    exchange_id = history.start("What happened?", surface="review", day="2023-11-14", scope="history", created_at=T0)
    restarted = ask_history.History(tmp_path)
    queued = restarted.get(exchange_id)
    assert queued is not None
    assert (queued["question"], queued["status"], queued["result"]) == ("What happened?", "queued", None)
    result = answer(source(memory))
    assert complete(restarted, exchange_id, result)
    saved = ask_history.History(tmp_path).get(exchange_id)
    assert saved is not None
    assert saved["result"] == result
    assert (saved["status"], saved["started_at"], saved["finished_at"], saved["model"]) == (
        "completed",
        T0 + 5,
        T0 + 10,
        "synthetic-model",
    )


def test_delete_removes_every_version_and_late_finish_cannot_restore_the_question(memory):
    history = ask_history.History()
    exchange_id = history.start("Question", created_at=T0)
    assert complete(history, exchange_id, {"answer": "first"})
    assert complete(history, exchange_id, {"answer": "second"})
    assert history.delete(exchange_id) == 1
    assert history.get(exchange_id) is None
    assert history.list_entries() == []
    assert not complete(ask_history.History(), exchange_id, {"answer": "late"})
    assert memory.range(memory_log.ASK_ANSWER, 0, T0 + 100) == []
    assert memory.range(memory_log.ASK_STATE, 0, T0 + 100) == []


def test_delete_racing_with_first_answer_write_cannot_resurrect_metadata(memory, monkeypatch):
    history = ask_history.History()
    exchange_id = history.start("Question", created_at=T0)
    append = memory.append

    def racing_append(stream, record, **kwargs):
        if stream == memory_log.ASK_ANSWER:
            memory.purge([str(uuid.UUID(exchange_id))], "test deletion")
        return append(stream, record, **kwargs)

    monkeypatch.setattr(memory, "append", racing_append)
    assert not complete(history, exchange_id, {"answer": "late"})
    assert ask_history.History().get(exchange_id) is None


def test_source_purge_removes_all_copied_answer_data_but_keeps_question_and_safe_moment(memory):
    history = ask_history.History()
    evidence_id = source(memory)
    exchange_id = history.start("What happened?", scope="history", created_at=T0)
    assert complete(history, exchange_id, answer(evidence_id))
    memory.purge([evidence_id], "test source removal")
    saved = ask_history.History().get(exchange_id)
    assert saved is not None
    assert saved["question"] == "What happened?"
    assert saved["result"] is None
    assert saved["answer_removed"]
    assert saved["citations"] == [
        {"label": "J1", "ts": T0 - 60, "camera": "", "source": "journal", "verified": True, "removed": True}
    ]
    metadata = memory.range(memory_log.ASK_STATE, 0, T0 + 100)
    assert "Private source words" not in json.dumps(metadata)
    assert "The event happened" not in json.dumps(metadata)


def test_only_removed_citation_marker_is_marked_removed(memory):
    history = ask_history.History()
    first = source(memory)
    second = source(memory, "Another source.", T0 - 30)
    result = answer(first, "Both [J1] and [J2].")
    result["evidence"].append({"label": "J2", "ts": T0 - 30, "source": "journal", "interpretation": second})
    exchange_id = history.start("Question", created_at=T0)
    assert complete(history, exchange_id, result)
    memory.purge([first], "test one source removal")
    saved = history.get(exchange_id)
    assert saved is not None
    assert saved["result"] is None
    assert [(row["label"], row["removed"]) for row in saved["citations"]] == [("J1", True), ("J2", False)]


def test_uncited_copied_evidence_still_parents_the_answer(memory):
    history = ask_history.History()
    first = source(memory)
    uncited = source(memory, "Uncited copied source.", T0 - 30)
    result = answer(first)
    result["evidence"].append(
        {
            "label": "J2",
            "ts": T0 - 30,
            "source": "journal",
            "interpretation": uncited,
            "quote": "Uncited copied source.",
        }
    )
    exchange_id = history.start("Question", created_at=T0)
    assert complete(history, exchange_id, result)
    memory.purge([uncited], "test uncited source removal")
    saved = history.get(exchange_id)
    assert saved is not None
    assert saved["result"] is None
    assert [(row["label"], row["removed"]) for row in saved["citations"]] == [("J1", False)]


def test_long_unicode_result_round_trips_without_oversized_inline_payloads(memory):
    history = ask_history.History()
    result = {
        "answer": "A detailed answer 🐈. " * 15_000,
        "status": "ok",
        "evidence": [],
        "nested": {"values": [1, 2, 3]},
    }
    exchange_id = history.start("Question", created_at=T0)
    assert complete(history, exchange_id, result)
    saved = ask_history.History().get(exchange_id)
    assert saved is not None
    assert saved["result"] == result
    chunks = memory.range(memory_log.ASK_ANSWER, 0, T0 + 100)
    assert len(chunks) > 1
    assert max(len(memory_log.body_bytes(memory_log.ASK_ANSWER, chunk)) for chunk in chunks) <= memory_log.MAX_BODY


def test_untracked_cited_sources_persist_with_explicit_unverified_availability(memory):
    history = ask_history.History()
    exchange_id = history.start("Question", created_at=T0)
    result = answer("not-a-canonical-payload")
    assert complete(history, exchange_id, result)
    saved = history.get(exchange_id)
    assert saved is not None
    assert saved["answer_unverified"]
    assert saved["result"] == result
    assert saved["citations"][0]["verified"] is False
    assert saved["warnings"]
    assert not saved["answer_removed"]


def test_filtered_pagination_and_delete_are_newest_first_and_review_day_scoped(memory):
    history = ask_history.History()
    old = history.start("Old", surface="review", day="2023-11-14", created_at=T0)
    latest = history.start("Latest", surface="review", day="2023-11-14", created_at=T0 + 20)
    other_day = history.start("Other day", surface="review", day="2023-11-15", created_at=T0 + 40)
    live = history.start("Live", surface="live", created_at=T0 + 60)
    assert [row["id"] for row in history.list_entries(surface="review", day="2023-11-14", limit=1)] == [latest]
    assert [row["id"] for row in history.list_entries(surface="review", day="2023-11-14", before=latest)] == [old]
    assert history.delete(surface="review", day="2023-11-14") == 2
    assert [row["id"] for row in history.list_entries()] == [live, other_day]


def test_finish_returns_false_when_the_exchange_is_purged_during_the_write(memory, monkeypatch):
    history = ask_history.History()
    ident = history.start("Question", created_at=T0)
    written = history.log.append

    def purge_first(stream, record, **kwargs):
        if stream == memory_log.ASK_ANSWER and not memory.records.get(stream):
            memory.purge([str(uuid.UUID(ident))], "deleted while the answer was written")
        return written(stream, record, **kwargs)

    monkeypatch.setattr(history.log, "append", purge_first)
    assert not complete(history, ident, {"answer": "Four"})


def test_list_entries_reads_a_bounded_window_and_pages_from_the_newest(memory):
    history = ask_history.History()
    old = history.start("Old", created_at=T0)
    recent = history.start("Recent", created_at=T0 + 3600)
    now = T0 + 3600
    assert [row["id"] for row in history.list_entries(since=now - 60)] == [recent]
    assert [row["id"] for row in history.list_entries(since=T0 - 1)] == [recent, old]
    assert [row["id"] for row in history.list_entries(since=T0 - 1, before=recent)] == [old]
    assert [row["id"] for row in history.list_entries(since=T0 - 1, limit=1)] == [recent]
    with pytest.raises(ValueError, match="since"):
        history.list_entries(since=float("nan"))


def test_export_uses_retrieval_drafts_with_execution_time_and_only_available_canonical_moments(memory):
    history = ask_history.History()
    evidence = source(memory)
    exchange_id = history.start(
        "What happened yesterday?", surface="review", day="2023-11-14", scope="history", created_at=T0
    )
    assert complete(history, exchange_id, answer(evidence))
    history.start("General advice", scope="general", created_at=T0 + 20)
    assert history.export_drafts() == [
        {
            "id": exchange_id,
            "question": "What happened yesterday?",
            "asked_at": T0 + 5,
            "surface": "review",
            "day": "2023-11-14",
            "scope": "history",
            "mode": "local",
            "draft": True,
            "evidence": [{"at": T0 - 60, "within_s": 1, "source": "journal"}],
            "note": "Draft prior citations. A person must verify the expected evidence before accepting this label.",
        }
    ]
    assert history.export_summary == {"exported": 1, "skipped": 1}
    memory.purge([evidence], "test source removal")
    assert history.export_drafts() == []
    assert history.export_summary == {"exported": 0, "skipped": 2}


@pytest.mark.parametrize("created_at", [float("nan"), float("inf"), -1, True])
def test_invalid_creation_timestamp_is_refused_without_persisting(created_at, memory):
    history = ask_history.History()
    with pytest.raises(ValueError, match="created_at"):
        history.start("Question", created_at=created_at)
    assert history.list_entries() == []


def test_empty_broker_is_an_empty_history_and_a_new_connection_reads_saved_result(tmp_path, monkeypatch):
    log = memory_log.MemoryLog(continuum_link.socket_path(), client_name="ask-history-test")
    monkeypatch.setattr(memory_log, "at", lambda state=None: log)
    try:
        first = ask_history.History(tmp_path)
        assert first.list_entries() == []
        exchange_id = first.start("Offline local question", surface="voice", created_at=T0)
        assert complete(first, exchange_id, {"answer": "Saved locally", "evidence": []})
    finally:
        log.close()
    restarted = memory_log.MemoryLog(continuum_link.socket_path(), client_name="ask-history-restarted")
    monkeypatch.setattr(memory_log, "at", lambda state=None: restarted)
    try:
        saved = ask_history.History(tmp_path).get(exchange_id)
        assert saved is not None
        assert saved["result"] == {"answer": "Saved locally", "evidence": []}
    finally:
        restarted.close()


def test_real_broker_source_and_root_purges_remove_answers_and_refuse_late_finish(tmp_path, monkeypatch):
    log = memory_log.MemoryLog(continuum_link.socket_path(), client_name="ask-history-purge-test")
    monkeypatch.setattr(memory_log, "at", lambda state=None: log)
    try:
        evidence_id = log.append(
            memory_log.JOURNAL, {"ts_start": T0 - 60, "text": "Private source words.", "level": 30}
        )
        history = ask_history.History(tmp_path)
        exchange_id = history.start("Question", scope="history", created_at=T0)
        assert complete(history, exchange_id, answer(evidence_id))
        legacy_id = history.start("Legacy question", scope="history", created_at=T0 + 1)
        legacy_answer = answer("legacy-reference")
        assert complete(history, legacy_id, legacy_answer)
        legacy_saved = history.get(legacy_id)
        assert legacy_saved is not None
        assert legacy_saved["result"] == legacy_answer
        log.purge([evidence_id], "test source purge")
        saved = ask_history.History(tmp_path).get(exchange_id)
        assert saved is not None
        assert saved["question"] == "Question"
        assert saved["result"] is None
        assert saved["answer_removed"]
        assert saved["citations"][0]["removed"]
        legacy = history.get(legacy_id)
        assert legacy is not None
        assert legacy["answer_removed"]
        assert legacy["result"] is None
        assert legacy["citations"][0]["removed"]
        # A source gone before the answer lands also cannot be copied into root-only history.
        assert complete(history, exchange_id, answer(evidence_id))
        unavailable = history.get(exchange_id)
        assert unavailable is not None
        assert unavailable["answer_removed"]
        assert history.delete(exchange_id) == 1
        assert not complete(history, exchange_id, {"answer": "late"})
        assert history.get(exchange_id) is None
    finally:
        log.close()


def test_legacy_source_purged_while_answer_runs_cannot_be_saved_as_available(memory, monkeypatch):
    revisions = {"speech.utterance": 0}
    history = ask_history.History()
    monkeypatch.setattr(history, "_revisions", lambda: dict(revisions))
    ident = history.start("What was said?")
    # The model already read this legacy row when a source purge commits.
    revisions["speech.utterance"] = 1
    complete(
        history,
        ident,
        {
            "status": "ok",
            "answer": "Deleted words [T1]",
            "evidence": [
                {"id": "T1", "ts": T0, "source": "transcript", "text": "Deleted words"},
            ],
        },
    )
    saved = history.get(ident)
    assert saved is not None
    assert saved["result"] is None
    assert saved["answer_removed"]
    assert saved["citations"][0]["removed"]
