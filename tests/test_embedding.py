"""Embeddings from a local model (memory/embedding.py): the vectors' shape, the query instruction, and which model
stays loaded. The journal's embedding pass is tested in test_journal.py, recall's search in test_recall.py."""

import pytest

from liberational import config as cfg
from liberational import ollama_health as oh
from liberational.memory import embedding, journal

MODEL = "qwen3-embedding:4b"


class Ollama:
    """A fake /api/embed and /api/tags: every text gets the same unit vector; the calls are kept."""

    def __init__(self):
        self.calls = []
        self.digest = "sha-1"

    def post(self, payload, timeout):
        self.calls.append(payload)
        return {"embeddings": [[1.0, 0.0, 0.0, 0.0] for _ in payload["input"]]}

    def tags(self, timeout=5, refresh=False):
        return [{"name": MODEL, "digest": self.digest}]


@pytest.fixture
def ollama(monkeypatch):
    o = Ollama()
    monkeypatch.setattr(oh, "_TAGS_CACHE", None)
    monkeypatch.setattr(oh, "tags", o.tags)
    monkeypatch.setattr(embedding, "_post", o.post)
    monkeypatch.setattr(embedding, "_trouble", {"why": None})
    with cfg.override({"semantic": {"enabled": True, "dimensions": 4}, "models": {"ollama": {"embedding": MODEL}}}):
        yield o


def test_unit_truncates_and_normalizes():
    assert embedding.unit([3, 4, 100], 2) == pytest.approx([0.6, 0.8])
    with pytest.raises(ValueError, match="fewer"):
        embedding.unit([1, 2], 3)
    with pytest.raises(ValueError, match="empty"):
        embedding.unit([0, 0, 1], 2)


def test_unit_normalizes_large_finite_values_without_overflow():
    assert embedding.unit([1e200, 1e200], 2) == pytest.approx([2**-0.5, 2**-0.5])


@pytest.mark.parametrize("bad", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_unit_refuses_a_vector_that_is_not_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        embedding.unit([1.0, bad], 2)


def test_questions_get_the_instruction_and_entries_dont(ollama):
    embedding.embed(["Dana booked the venue"], MODEL, 4, 5)
    embedding.embed(["where was the party"], MODEL, 4, 5, query=True)
    doc, query = (c["input"][0] for c in ollama.calls)
    assert doc == "Dana booked the venue"
    assert query.startswith("Instruct: ") and query.endswith("\nQuery:where was the party")
    assert all(c["dimensions"] == 4 for c in ollama.calls)
    embedding.embed(["where"], "nomic-embed-text", 4, 5, query=True)
    assert ollama.calls[-1]["input"] == ["where"]  # the prefix is Qwen3-Embedding's


def test_only_the_configured_model_is_kept_loaded(ollama):
    small = "qwen3-embedding:0.6b"
    embedding.embed(["venue"], MODEL, 4, 5)
    with cfg.override({"models": {"ollama": {"embedding": small}}}):
        embedding.embed(["venue"], MODEL, 4, 5)  # an older space's model, still asked for a question
        embedding.embed(["venue"], small, 4, 5)
    assert [(c["model"], c.get("keep_alive")) for c in ollama.calls] == [(MODEL, -1), (MODEL, None), (small, -1)]


def test_every_call_asks_for_one_context_so_ollama_keeps_one_runner(ollama):
    embedding.embed(["Dana booked the venue", "lunch"], MODEL, 4, 5)
    embedding.embed(["where was the party"], MODEL, 4, 5, query=True)
    embedding.embed(["venue"], "qwen3-embedding:0.6b", 4, 5)  # an older space's model loads small too
    contexts = {c.get("options", {}).get("num_ctx") for c in ollama.calls}
    assert len(contexts) == 1 and None not in contexts


def test_the_digest_reads_tags_and_says_why_it_cant_be(ollama, monkeypatch):
    assert embedding.digest(MODEL) == ("sha-1", None)
    ollama.digest = "sha-2"  # a re-pulled tag: the next read sees the new weights
    assert embedding.digest(MODEL) == ("sha-2", None)
    _, why = embedding.digest("nomic-embed-text")
    assert why and why.startswith("the embedding model nomic-embed-text isn't pulled")
    monkeypatch.setattr(journal, "_tags", lambda timeout=5, fresh=False: None)
    _, why = embedding.digest(MODEL)
    assert why and why.startswith("Ollama isn't answering")


def test_trouble_is_logged_once_per_kind_and_cleared_when_fine(ollama, capsys):
    embedding.trouble("Ollama is down")
    embedding.trouble("Ollama is down")
    embedding.fine()
    embedding.fine()
    out = capsys.readouterr().err
    assert out.count("Ollama is down; searching by words alone") == 1
    assert out.count("embeddings work again") == 1
