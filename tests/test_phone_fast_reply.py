"""Owner calls: everyday turns get the fast local model, tool requests and deferrals the phone agent."""

import asyncio
from typing import cast

import pytest

from liberational.phone import bridge


@pytest.fixture
def owner(monkeypatch):
    call = bridge.Call(cast(bridge.WebSocket, None))
    call.call_sid = "synthetic-call"
    call.owner = True
    monkeypatch.setattr(call, "system_for_turn", lambda hint=None: "Synthetic system")
    monkeypatch.setattr(bridge, "build_notes", lambda hint: None)
    monkeypatch.setattr(bridge, "fast_replies", lambda: True)
    asked = {"llm": [], "agent": []}

    def llm(history, system=None):
        asked["llm"].append(history)
        return asked.get("llm_reply", "Sure. Autumn leaves drift down.")

    def agent(call_sid, text, notes=None, timeout=60):
        asked["agent"].append((text, notes))
        return "It will be sunny tomorrow."

    monkeypatch.setattr(bridge, "ask_llm", llm)
    monkeypatch.setattr(bridge, "phone_agent_reply", agent)
    return call, asked


def turn(call, text):
    return asyncio.run(call.reply_turn(text, None, record_history=False))


def test_chat_gets_the_fast_model(owner):
    call, asked = owner
    assert turn(call, "write me a haiku about fall") == "Sure. Autumn leaves drift down."
    assert call.route == "fast" and not asked["agent"]
    assert [m["role"] for m in call.history] == ["user", "assistant"]
    assert call.timing["route"] == "fast"


def test_a_tool_request_goes_to_the_agent_with_what_it_missed(owner):
    call, asked = owner
    turn(call, "write me a haiku about fall")
    assert turn(call, "what's the weather tomorrow") == "It will be sunny tomorrow."
    assert call.route == "agent" and len(asked["llm"]) == 1
    notes = asked["agent"][0][1]
    assert "haiku" in notes and "Autumn leaves" in notes
    assert call.unseen == []


def test_a_deferring_fast_reply_falls_back_to_the_agent(owner):
    call, asked = owner
    asked["llm_reply"] = "I don't have real-time access to that."
    assert turn(call, "tell me something fun") == "It will be sunny tomorrow."
    assert call.route == "agent" and asked["agent"]


def test_timing_is_logged_once_at_first_audio(owner):
    call, _ = owner
    call.timing = {"end": 100.0, "asr": 100.3, "reply": 100.9, "route": "fast"}
    call.log_timing(101.4)
    assert call.timings == [{"route": "fast", "asr_s": 0.3, "reply_s": 0.6, "voice_s": 0.5, "total_s": 1.4}]
    assert call.timing == {}


@pytest.mark.parametrize(
    "promise", ["Got it. I'll end the call.", "I'll grab the map and meet you there in 20 minutes."]
)
def test_a_fast_reply_that_promises_an_action_goes_to_the_agent(owner, promise):
    call, asked = owner
    asked["llm_reply"] = promise
    assert turn(call, "sounds good") == "It will be sunny tomorrow."
    assert call.route == "agent"


def test_the_fast_model_is_told_to_talk_to_the_owner(owner):
    call, _ = owner
    turn(call, "tell me a joke")
    system = bridge.fast_system("Base")
    assert system.startswith("Base") and "say 'you'" in system


def test_let_me_know_is_a_fine_fast_reply(owner):
    call, asked = owner
    asked["llm_reply"] = "You're welcome. Let me know if you need anything else."
    assert turn(call, "okay thanks") == "You're welcome. Let me know if you need anything else."
    assert call.route == "fast" and not asked["agent"]
