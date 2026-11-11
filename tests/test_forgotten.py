"""Asked about a person who was forgotten, the pack says "Doesn't look like anything to me." (people/forgotten.py).
Only HMAC hashes of their names are kept, independently of person event erasure."""

import json
import subprocess
from unittest.mock import create_autospec

import pytest

from liberational import config as cfg
from liberational.assistant import voice
from liberational.domain.identity import Basis
from liberational.memory import ask, history_qa
from liberational.people import forgotten, registry

NORMAL = {"is_history": True, "answer": None, "evidence": [], "coverage": {}, "warnings": [], "query": {"k": 1}}


@pytest.fixture
def state(tmp_path):
    with cfg.override({"paths": {"state": str(tmp_path)}, "owner": {"name": "Alice"}, "assistant": {"name": "Bob"}}):
        yield tmp_path


@pytest.fixture
def reg(state):
    return registry.at(state)


def forget(state, name):
    """What People: Delete does: keep the hashes of every name they had, then forget them in the registry."""
    reg = registry.at(state)
    forgotten.remember({name, *reg.names(name)}, state)
    reg.forget(name, "dashboard")


@pytest.fixture
def execute(state, monkeypatch):
    """ask.execute in local mode, with the evidence and the model stubbed so a normal answer says "normal". Returns
    the result and whether either stub was reached."""
    collect = create_autospec(history_qa.collect, return_value=NORMAL)
    model = create_autospec(ask.local_answer, return_value=("normal", "m"))
    monkeypatch.setattr(history_qa, "collect", collect)
    monkeypatch.setattr(ask, "local_answer", model)

    def run(question):
        return ask.execute(question, state_dir=state), collect.called or model.called

    return run


@pytest.mark.parametrize(
    "question",
    [
        "What did Dana say yesterday?",
        "When was Dana last seen?",
        "who's dana",
        "What about Dana?",
        "Did Dana's sister call today?",
    ],
)
def test_asking_about_a_forgotten_person_gets_the_line_and_nothing_else(state, reg, execute, question):
    reg.claim(registry.voiceprint(3), "Dana", Basis.OWNER_TAG, "test")
    forget(state, "Dana")
    result, looked = execute(question)
    assert result["answer"] == "Doesn't look like anything to me."
    assert (result["evidence"], result["status"], looked) == ([], "ok", False)
    assert result["paths"] == ["forgotten"]  # Ask's "New" tag for its switch


@pytest.mark.parametrize("question", ["What did Dana say yesterday?", "Who's Dana Cole?", "what about Dee?"])
def test_an_earlier_name_an_alias_and_someone_merged_in_fire_too(state, reg, execute, question):
    reg.ensure("Dana", "test")
    reg.rename("Dana", "Dana Cole", "test")
    reg.ensure("Dee", "test")
    reg.merge("Dee", "Dana Cole", "test")
    forget(state, "Dana Cole")
    assert execute(question)[0]["answer"] == forgotten.LINE


@pytest.mark.parametrize("question", ["What did Erin say yesterday?", "Who's Danny?", "What happened today?"])
def test_a_stranger_gets_the_normal_answer(state, reg, execute, question):
    reg.ensure("Dana", "test")
    forget(state, "Dana")
    result, looked = execute(question)
    assert (result["answer"], looked) == ("normal", True)


@pytest.mark.parametrize(
    "question, fires",
    [
        ("What about Dana?", True),
        ("Who's Dana?", True),
        pytest.param("who's dana and why", True, id="lowercase-speech"),
        pytest.param("Who is Dana Marie Cole?", True, id="three-words"),
        pytest.param("what did will say yesterday", True, id="a-common-word-the-parser-names"),
        pytest.param("Who's Dana White?", False, id="a-full-name-not-forgotten"),
        # The owner's and the assistant's names never fire, though someone forgotten had them
        ("What did Alice say yesterday?", False),
        ("who's Bob?", False),
        ("What about Alice?", False),
        ("when did I last see bob", False),
        # Nor do common words
        ("Will it rain today?", False),
        ("May I see what happened today?", False),
        ("What happened on May 3?", False),
        ("What did we talk about yesterday?", False),
        ("Did I mark the boxes yesterday?", False),
    ],
)
def test_only_a_forgotten_name_fires(state, question, fires):
    forgotten.remember({"Dana", "Dana Marie Cole", "Alice", "Bob", "Will", "May", "Mark"}, state)
    assert forgotten.named(question, history_qa.people_named(question)) is fires


def test_off_keeps_no_hash_and_answers_as_before(state, reg, execute):
    reg.ensure("Eve", "test")
    forget(state, "Eve")  # kept while on
    reg.ensure("Dana", "test")
    with cfg.override({"assistant": {"forgotten_reply": False}}):
        forget(state, "Dana")
        assert forgotten.store(state).has("eve") and not forgotten.store(state).has("dana")  # Eve's alone
        assert execute("What did Eve say yesterday?")[0]["answer"] == "normal"


def test_someone_new_with_the_name_is_answered_normally(state, reg, execute):
    reg.ensure("Dana", "test")
    forget(state, "Dana")
    reg.ensure("Dana", "test")  # a new Dana is someone new
    assert execute("What did Dana say yesterday?")[0]["answer"] == "normal"


def test_the_hashes_outlive_the_log_purge(state, reg, monkeypatch):
    reg.claim(registry.voiceprint(3), "Dana", Basis.OWNER_TAG, "test")
    dana = reg.person_id("Dana")
    forget(state, "Dana")
    log = registry.at(state).log
    left = [s.event for stream in (registry.PEOPLE, registry.ASSERTIONS) for s in log.read(stream)]
    assert [e for e in left if e.subject == dana] == []  # forgetting stays complete
    assert log.purges()
    registry.close_all()  # a later process rebuilds the registry; suppression remains in its own collection
    forgotten.reset()
    assert forgotten.named("What about Dana?")


@pytest.mark.parametrize(
    "said, fires",
    [
        ("who's Dana?", True),
        ("What did Dana say yesterday?", True),
        ("remind me who Dana is", True),
        pytest.param("Tell Dana hello", False, id="not-a-question"),
    ],
)
def test_voice_knows_a_question_about_a_forgotten_person(state, reg, said, fires):
    reg.ensure("Dana", "test")
    forget(state, "Dana")
    assert voice.asks_about_forgotten(said) is fires


def test_a_voice_history_question_reaches_the_line_through_execute(state, reg, monkeypatch):
    """The voice and camera routes ask history questions through ask.ask, whose worker runs execute."""
    reg.ensure("Dana", "test")
    forget(state, "Dana")

    def worker(cmd, *, input, **kwargs):  # the worker, run here: execute on the request it was sent
        data = json.loads(input)
        result = ask.execute(data["question"], data["state_dir"])
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(result), stderr="")

    monkeypatch.setattr(ask.subprocess, "run", worker)
    assert ask.spoken(ask.ask("Dana said what yesterday", state_dir=state)) == forgotten.LINE
