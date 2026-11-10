"""The profile fact pass's model call (people/profiles.py): on the shared small model it asks for the context the
model is kept loaded at, and its lines fit it (#917)."""

from unittest.mock import patch

from liberational import config as cfg
from liberational.memory import ask
from liberational.people import profiles


def test_the_shared_model_is_asked_at_its_loaded_context_and_a_separate_one_at_its_own(monkeypatch):
    monkeypatch.setattr(profiles, "LLM", cfg.get("models.ollama.llm"))
    assert profiles.num_ctx() == ask.SHARED_NUM_CTX
    monkeypatch.setattr(profiles, "LLM", "some-bigger-model:30b")
    assert profiles.num_ctx() == 98304


def test_the_sample_keeps_the_newest_lines_that_fit():
    lines = [f"line {i} " + "x" * 200 for i in range(80)]  # about 16 000 characters in all
    sample = profiles.sample_of(lines, ask.SHARED_NUM_CTX)
    budget = (ask.SHARED_NUM_CTX - profiles.REPLY_TOKENS - profiles.PROMPT_TOKENS) * 3
    assert len(sample) <= budget
    assert sample.endswith(lines[-1]) and "line 0 " not in sample, "the newest kept, oldest first"
    assert profiles.sample_of(lines[:3], ask.SHARED_NUM_CTX).count("\n") == 2, "a short list is kept whole"


def test_the_call_sends_the_shared_context(monkeypatch):
    monkeypatch.setattr(profiles, "LLM", cfg.get("models.ollama.llm"))
    sent = {}

    class Reply:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"message": {"content": "[]"}}

    def post(url, timeout, json):
        sent.update(json)
        return Reply()

    with patch.object(profiles.requests, "post", side_effect=post):
        assert profiles.ask_llm("Dana", ["I live in Leeds."]) == []
    assert sent["options"]["num_ctx"] == ask.SHARED_NUM_CTX
