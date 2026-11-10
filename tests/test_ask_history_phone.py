"""Only actual phone model turns enter Ask history; authentication/control text never does."""

import asyncio
import threading
from typing import cast
from unittest.mock import create_autospec

import numpy as np
import pytest

from liberational.memory import ask_history
from liberational.phone import bridge

QUESTION = "What time is it?"
TRANSCRIBE = bridge.transcribe
OWNER_TURN = bridge.owner_turn


@pytest.fixture
def phone(monkeypatch):
    ask_history.close_all()
    call = bridge.Call(cast(bridge.WebSocket, None))
    call.call_sid = "synthetic-call"
    speak = create_autospec(call.speak, return_value=None)
    monkeypatch.setattr(call, "speak", speak)
    monkeypatch.setattr(call, "system_for_turn", create_autospec(call.system_for_turn, return_value="Synthetic system"))
    monkeypatch.setattr(bridge, "owner_turn", create_autospec(bridge.owner_turn, return_value=(None, None)))
    monkeypatch.setattr(bridge, "transcribe", create_autospec(bridge.transcribe, return_value=QUESTION))
    monkeypatch.setattr(bridge, "identify_speaker", create_autospec(bridge.identify_speaker, return_value=(None, None)))

    async def fake_mlx(fn, *args):
        return fn(*args)

    monkeypatch.setattr(bridge, "run_mlx", create_autospec(bridge.run_mlx, side_effect=fake_mlx))
    yield call, speak
    ask_history.close_all()


def say(call):
    asyncio.run(call.handle_utterance([np.zeros(160, dtype=np.int16)]))


def test_owner_agent_turn_is_persisted_without_claiming_gateway_agent_id_is_an_actual_model(phone, monkeypatch):
    call, speak = phone
    call.owner = True
    agent = create_autospec(call.owner_agent_reply, return_value="Owner reply.")
    local = create_autospec(bridge.ask_llm, side_effect=AssertionError("must not call local model"))
    monkeypatch.setattr(call, "owner_agent_reply", agent)
    monkeypatch.setattr(bridge, "ask_llm", local)
    say(call)
    rows = ask_history.at(bridge.STATE_DIR).list_entries(surface="call")
    assert len(rows) == 1
    assert (rows[0]["question"], rows[0]["call_id"], rows[0]["mode"], rows[0]["model"]) == (
        QUESTION,
        "synthetic-call",
        "agent",
        None,
    )
    assert rows[0]["result"]["answer"] == "Owner reply."
    assert rows[0]["status"] == "completed"
    agent.assert_awaited_once_with(QUESTION, None)
    speak.assert_awaited_once_with("Owner reply.")


def test_guest_route_persists_the_final_sanitized_spoken_reply_and_storage_runs_off_event_loop(
    phone, monkeypatch, memory
):
    call, speak = phone
    local = create_autospec(bridge.ask_llm, return_value="I can send messages.")
    monkeypatch.setattr(bridge, "ask_llm", local)
    main_thread = threading.get_ident()
    writes = []
    append = memory.append

    def recording_append(*args, **kwargs):
        writes.append(threading.get_ident())
        return append(*args, **kwargs)

    monkeypatch.setattr(memory, "append", recording_append)
    say(call)
    row = ask_history.at(bridge.STATE_DIR).list_entries(surface="call")[0]
    assert row["result"]["answer"] == bridge.CAPABILITY_FALLBACK
    assert (row["model"], row["mode"], row["surface"]) == (bridge.LLM, "local", "call")
    assert call.history == [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": bridge.CAPABILITY_FALLBACK},
    ]
    assert writes and main_thread not in writes
    speak.assert_awaited_once_with(bridge.CAPABILITY_FALLBACK)
    local.assert_called_once_with([{"role": "user", "content": QUESTION}], "Synthetic system")


def test_owner_gateway_failure_records_the_actual_local_fallback_model(phone, monkeypatch):
    call, speak = phone
    call.owner = True
    monkeypatch.setattr(bridge, "build_notes", create_autospec(bridge.build_notes, return_value=None))
    monkeypatch.setattr(
        bridge,
        "phone_agent_reply",
        create_autospec(bridge.phone_agent_reply, side_effect=ConnectionError("gateway unavailable")),
    )
    monkeypatch.setattr(bridge, "ask_llm", create_autospec(bridge.ask_llm, return_value="Local fallback reply."))
    say(call)
    row = ask_history.at(bridge.STATE_DIR).list_entries(surface="call")[0]
    assert row["model"] == bridge.LLM
    assert row["result"]["answer"] == "Local fallback reply."
    speak.assert_awaited_once_with("Local fallback reply.")


def test_failed_model_turn_keeps_question_and_failed_status_without_exception_details(phone, monkeypatch):
    call, speak = phone
    monkeypatch.setattr(
        bridge, "ask_llm", create_autospec(bridge.ask_llm, side_effect=RuntimeError("private transport detail"))
    )
    say(call)
    row = ask_history.at(bridge.STATE_DIR).list_entries(surface="call")[0]
    assert row["status"] == "failed"
    assert row["question"] == QUESTION
    assert row["result"]["answer"] == "Call reply failed."
    assert call.state == "listening"
    speak.assert_not_awaited()


def test_history_start_failure_warns_but_does_not_break_the_call(phone, monkeypatch, caplog):
    call, speak = phone
    monkeypatch.setattr(
        bridge.ask_history, "at", create_autospec(ask_history.at, side_effect=ConnectionError("broker offline"))
    )
    monkeypatch.setattr(bridge, "ask_llm", create_autospec(bridge.ask_llm, return_value="Still answered."))
    say(call)
    speak.assert_awaited_once_with("Still answered.")
    assert "Ask history unavailable" in caplog.text


def test_history_finish_failure_warns_but_preserves_reply_and_original_question(phone, monkeypatch, caplog):
    call, speak = phone
    history = ask_history.at(bridge.STATE_DIR)
    monkeypatch.setattr(
        history, "finish", create_autospec(history.finish, side_effect=ConnectionError("broker disconnected"))
    )
    monkeypatch.setattr(bridge, "ask_llm", create_autospec(bridge.ask_llm, return_value="Still answered."))
    say(call)
    speak.assert_awaited_once_with("Still answered.")
    assert history.list_entries(surface="call")[0]["question"] == QUESTION
    assert "Ask history completion unavailable" in caplog.text


def test_password_attempt_is_not_persisted(phone, monkeypatch):
    call, speak = phone
    local = create_autospec(bridge.ask_llm, side_effect=AssertionError("control must not reach model"))
    monkeypatch.setattr(bridge, "ask_llm", local)
    monkeypatch.setattr(bridge, "transcribe", create_autospec(TRANSCRIBE, return_value="password is guess"))
    monkeypatch.setattr(bridge, "password_matches", create_autospec(bridge.password_matches, return_value=False))
    monkeypatch.setattr(call, "try_password", create_autospec(call.try_password, return_value="Denied."))
    say(call)
    assert ask_history.at(bridge.STATE_DIR).list_entries(surface="call") == []
    local.assert_not_called()
    speak.assert_awaited_once_with("Denied.")


def test_direct_control_reply_is_not_persisted(phone, monkeypatch):
    call, speak = phone
    local = create_autospec(bridge.ask_llm, side_effect=AssertionError("control must not reach model"))
    monkeypatch.setattr(bridge, "ask_llm", local)
    monkeypatch.setattr(bridge, "owner_turn", create_autospec(OWNER_TURN, return_value=("Control complete.", None)))
    say(call)
    assert ask_history.at(bridge.STATE_DIR).list_entries(surface="call") == []
    local.assert_not_called()
    speak.assert_awaited_once_with("Control complete.")


def test_recall_control_reply_is_not_persisted(phone, monkeypatch):
    call, speak = phone
    local = create_autospec(bridge.ask_llm, side_effect=AssertionError("control must not reach model"))
    monkeypatch.setattr(bridge, "ask_llm", local)
    monkeypatch.setattr(bridge, "transcribe", create_autospec(TRANSCRIBE, return_value="What did I just ask?"))
    say(call)
    assert ask_history.at(bridge.STATE_DIR).list_entries(surface="call") == []
    local.assert_not_called()
    speak.assert_awaited_once_with("You haven't asked me anything yet.")


def test_password_phrase_after_owner_authentication_is_excluded_without_changing_the_existing_model_route(
    phone, monkeypatch
):
    call, speak = phone
    call.owner = True
    monkeypatch.setattr(bridge, "transcribe", create_autospec(TRANSCRIBE, return_value="password is guess"))
    monkeypatch.setattr(bridge, "password_matches", create_autospec(bridge.password_matches, return_value=False))
    agent = create_autospec(call.owner_agent_reply, return_value="Already authenticated.")
    monkeypatch.setattr(call, "owner_agent_reply", agent)
    say(call)
    assert ask_history.at(bridge.STATE_DIR).list_entries(surface="call") == []
    agent.assert_awaited_once_with("password is guess", None)
    speak.assert_awaited_once_with("Already authenticated.")
