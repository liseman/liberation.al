import json
import os
import tempfile
import types
from collections import defaultdict
from typing import Any
from unittest.mock import patch

import pytest

from liberational.assistant import ask_friends as af

MODEL: dict[str, Any] = {"model": "qwen3:4b-instruct", "backend": "ollama", "capabilities": ["completion", "thinking"]}
# The agent backend is opt-in (empty by default); these tests run with one configured.
AGENT = {"url": "http://127.0.0.1:18789/v1/chat/completions", "friends_agent": "openclaw/friends-advisor"}


def cloud(key):
    return {"key": key, "local": False, "input": "text", "available": True}


@pytest.fixture(autouse=True)
def configured(tmp_path, monkeypatch):
    """An agent configured, the panel's lock under tmp_path, and a fresh performance log."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(af, "FRIENDS_LOG", tmp_path / "friends_performance.json")
    with af.cfg.override({"agent": AGENT}):
        yield


class Models:
    """The OpenClaw inventory, Ollama (af.request), and the agent gateway (af.cloud_request). A model answers "4"
    unless `replies` queues something else (an exception is raised); every chat payload is kept in `chats`."""

    def __init__(self):
        self.inventory = [{"key": "ollama/" + MODEL["model"], "local": True, "tags": ["default"]}]
        self.replies, self.chats = defaultdict(list), []

    def request(self, base, path, payload: Any = None, timeout=0):
        if path != "/api/chat":  # /api/show and /api/ps
            return {"capabilities": ["completion"], "models": []}
        return self.chat(payload)

    def cloud_request(self, model, messages):
        return self.chat({"model": model, "messages": messages, "cloud": True})

    def chat(self, payload):
        self.chats.append(payload)
        reply = self.replies[payload["model"]].pop(0) if self.replies[payload["model"]] else "4"
        if isinstance(reply, Exception):
            raise reply
        return {"message": {"content": reply}}


@pytest.fixture
def models(monkeypatch):
    fake = Models()
    monkeypatch.setattr(af, "configured_inventory", lambda: fake.inventory)
    monkeypatch.setattr(af, "request", fake.request)
    monkeypatch.setattr(af, "cloud_request", fake.cloud_request)
    return fake


@pytest.mark.parametrize(
    ("question", "status"),
    [
        pytest.param("My password is dummy", "blocked_sensitive", id="sensitive"),
        pytest.param("x" * 1201, "input_too_long", id="too-long"),
    ],
)
def test_refused_before_any_request(models, question, status):
    assert af.ask(question)["status"] == status
    assert models.chats == []


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("my passwords are hunter2", id="passwords"),
        pytest.param("our api keys leaked", id="api-keys"),
        pytest.param("our API-keys leaked", id="api-keys-hyphen"),
        pytest.param("two access tokens", id="access-tokens"),
        pytest.param("my secrets", id="secrets"),
        pytest.param("the passphrases", id="passphrases"),
        pytest.param("two verification codes came in", id="verification-codes"),
        pytest.param("my one-time codes", id="one-time-codes"),
        pytest.param("My password is dummy", id="password"),
    ],
)
def test_sensitive_catches_singular_and_plural_credential_words(text):
    assert af.sensitive(text)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("ask my secretary", id="secretary"),
        pytest.param("a secretive plan", id="secretive"),
        pytest.param("the keys are on the hook", id="keys"),
        pytest.param("access the tokens page", id="access-tokens-apart"),
    ],
)
def test_sensitive_ignores_words_that_merely_contain_a_stem(text):
    assert not af.sensitive(text)


def test_pair_mode_asks_one_anthropic_one_openai_and_main_and_a_failure_makes_it_partial(models):
    models.inventory += [cloud("google/example"), cloud("anthropic/example"), cloud("openai/example")]
    models.replies["anthropic/example"] = [TimeoutError()]
    models.replies[MODEL["model"]] = ["4", "4; one participant failed."]
    result = af.ask("Two plus two?")
    assert [p["model"] for p in result["participants"]] == ["anthropic/example", "openai/example", MODEL["model"]]
    assert (result["status"], result["answer"]) == ("partial", "4; one participant failed.")
    assert (result["successful_opinions"], result["attempted"]) == (2, 3)
    asked, referee = models.chats[:3], models.chats[3]
    assert asked[0]["messages"] == asked[1]["messages"] == asked[2]["messages"]  # every participant asked alike
    assert json.loads(referee["messages"][1]["content"])["failed_models"] == ["anthropic/example"]


def test_a_failed_referee_is_synthesis_failed(models):
    models.replies[MODEL["model"]] = ["4", TimeoutError()]
    assert af.ask("Two plus two?")["status"] == "synthesis_failed"


@pytest.mark.parametrize(
    ("name", "think"),
    [pytest.param("qwen3:4b-instruct", False, id="instruct"), pytest.param("qwen3:8b", True, id="hybrid")],
)
def test_a_helper_model_is_bounded_and_thinks_unless_instruct(models, name, think):
    af.query(dict(MODEL, model=name), [], set())
    [payload] = models.chats
    assert (payload["options"]["num_ctx"], payload["keep_alive"], payload["think"]) == (4096, 0, think)


def test_an_error_never_leaks_input(models):
    models.replies[MODEL["model"]] = [RuntimeError("private input")]
    result = af.query(MODEL, [], set())
    assert result["error"] == "RuntimeError"
    assert "private input" not in str(result)


def test_discovery_takes_the_main_local_model_and_explicit_cloud_ones(models):
    models.inventory = [
        {"key": "ollama/main", "local": True, "tags": ["default"], "contextWindow": 65536},
        {"key": "ollama/helper", "local": True},
        cloud("cloud/advisor"),
    ]
    found, excluded, unavailable, errors = af.discover()
    assert [(m["model"], m.get("context_length")) for m in found] == [("main", 65536), ("cloud/advisor", None)]
    assert [m["model"] for m in excluded] == ["ollama/helper"]
    assert unavailable == errors == []


@pytest.mark.parametrize(
    "tags", [pytest.param(["configured"], id="configured"), pytest.param(["fallback#1"], id="fallback")]
)
def test_pair_mode_prefers_the_inventory_tagged_cloud_model(models, tags):
    models.inventory += [cloud("anthropic/plain"), dict(cloud("anthropic/tagged"), tags=tags)]
    result = af.ask("Two plus two?")
    assert [p["model"] for p in result["participants"]] == ["anthropic/tagged", MODEL["model"]]


OPENCLAW_PARAMS = {
    "agents": {
        "defaults": {
            "model": {"primary": "ollama/" + MODEL["model"]},
            "models": {"ollama/" + MODEL["model"]: {"params": {"num_ctx": 98304}}},
        }
    }
}


@pytest.mark.parametrize(
    ("model", "config", "num_ctx"),
    [
        pytest.param(dict(MODEL, context_length=65536, primary=True), {}, 65536, id="own-context"),
        pytest.param(dict(MODEL, primary=True), OPENCLAW_PARAMS, 98304, id="openclaw-params"),
    ],
)
def test_the_primary_model_keeps_its_context_and_stays_loaded(models, monkeypatch, model, config, num_ctx):
    monkeypatch.setattr(af.cfg, "openclaw_config", lambda: config)
    af.query(model, [], set())
    [payload] = models.chats
    assert (payload["options"]["num_ctx"], payload["keep_alive"]) == (num_ctx, -1)


def test_a_cloud_model_goes_through_the_agent_gateway(models):
    assert af.query({"model": "openai/example", "backend": "cloud"}, [], set())["status"] == "ok"
    assert models.chats == [{"model": "openai/example", "messages": [], "cloud": True}]


def test_cloud_fails_closed_without_isolation():
    with (
        patch.object(af.cfg, "openclaw_config", autospec=True, return_value={"agents": {"entries": {}}}),
        patch.object(af.urllib.request, "build_opener", autospec=True) as net,
        pytest.raises(ValueError, match="advisor isolation unavailable"),
    ):
        af.cloud_request("openai/example", [])
    net.assert_not_called()


def test_inventory_under_launchd_path_resolves_cli_and_node(monkeypatch):
    minimal = "/usr/bin:/bin:/usr/sbin:/sbin"
    monkeypatch.setenv("PATH", minimal)
    monkeypatch.setenv("PACK_TEST_MARKER", "retained")
    payload = {"models": [{"key": "ollama/synthetic", "local": True}]}
    with (
        patch.object(af.shutil, "which", autospec=True, return_value="/opt/homebrew/bin/openclaw") as which,
        patch.object(
            af.subprocess, "run", autospec=True, return_value=types.SimpleNamespace(stdout=json.dumps(payload))
        ) as run,
    ):
        assert af.configured_inventory() == payload["models"]
    env = run.call_args.kwargs["env"]
    assert env["PATH"].startswith(minimal + os.pathsep)
    assert env["PACK_TEST_MARKER"] == "retained"
    assert os.environ["PATH"] == minimal  # caller env unchanged
    which.assert_called_once_with("openclaw", path=env["PATH"])
    assert run.call_args.args[0] == ["/opt/homebrew/bin/openclaw", "models", "list", "--json"]
    assert run.call_args.kwargs["check"] is True


def test_inventory_preserves_inherited_path_without_duplicates(monkeypatch):
    path = "/custom/bin:/opt/homebrew/bin:/usr/bin"
    monkeypatch.setenv("PATH", path)
    directories = af.inventory_environment()["PATH"].split(os.pathsep)
    assert directories[:3] == path.split(os.pathsep)
    assert directories.count("/opt/homebrew/bin") == 1


def test_inventory_missing_cli_never_spawns():
    with (
        patch.object(af.shutil, "which", autospec=True, return_value=None),
        patch.object(af.subprocess, "run", autospec=True) as run,
        pytest.raises(FileNotFoundError),
    ):
        af.configured_inventory()
    run.assert_not_called()


def test_remote_endpoint_refused():
    with pytest.raises(ValueError, match="non-local endpoint"):
        af.request("https://example.com", "/api/chat")


@pytest.mark.parametrize("subset", ["pair", "all"])
def test_synthesizer_receives_answer_first_and_panel_after_instructions(models, subset):
    models.inventory += [cloud("anthropic/example"), cloud("openai/example")]
    models.replies[MODEL["model"]] = ["4", "4. All agreed."]
    result = af.ask("Two plus two?", subset=subset)
    assert result["answer"] == "4. All agreed."
    referee = models.chats[-1]
    assert referee["messages"][0]["role"] == "system"
    text = referee["messages"][0]["content"]
    assert "Start with the answer itself" in text
    assert "never open with the models" in text
    assert "one short final sentence" in text
    assert "Mention a failed model only if it changes your confidence" in text
    request = json.loads(referee["messages"][1]["content"])
    assert request["request"]["question"] == "Two plus two?"
    assert len(request["opinions"]) == 3
