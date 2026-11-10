import json

import pytest

import speech_fake
from liberational import config as cfg
from liberational.memory import history_qa as h
from liberational.memory.conversation_summary import _llm as conversation_llm
from voices import write_voices


def ask(tmp_path, rows):
    speech_fake.fake().write(rows)
    return h.collect("what did Alice and Dana talk about most in last 24 hours?", tmp_path, now=86400)


def test_recurring_late_topic_beats_early_fillers(tmp_path):
    rows = [dict(ts_start=i, speaker="Alice", text="Yeah okay") for i in range(150)]
    rows += [
        dict(ts_start=1000, speaker="Alice", text="Garden irrigation needs repair"),
        dict(ts_start=1001, speaker="Dana", text="Garden tomatoes need water"),
        dict(ts_start=1002, speaker="Alice", text="Garden soil needs compost"),
    ]
    r = ask(tmp_path, rows)
    assert r["coverage"]["topics"]["themes"][0]["term"] == "garden"
    assert r["coverage"]["topics"]["themes"][0]["utterances"] == 3
    assert r["coverage"]["topics"]["rows_scanned"] == 153
    assert r["coverage"]["topics"]["joint_exchange_supported"]
    assert all("Yeah" not in e["quote"] for e in r["evidence"])


def test_cooccurring_words_do_not_claim_a_global_frequency_ranking(tmp_path):
    rows = [dict(ts_start=10 + i, speaker="Alice", text="apple banana") for i in range(5)]
    rows += [dict(ts_start=20 + i, speaker="Dana", text="carrot donut") for i in range(2)]
    result = ask(tmp_path, rows)
    assert result["answer"].startswith("apple, carrot are recurring words")
    assert "recur most" not in result["answer"]
    assert result["evidence"]


def test_pronouns_and_noise_are_not_ranked_as_topics(tmp_path):
    r = ask(
        tmp_path,
        [
            dict(ts_start=10, speaker="Alice", text="I'm doing that, they're doing it"),
            dict(ts_start=11, speaker="Dana", text="I'm doing that, they're doing it"),
            dict(ts_start=12, speaker="Alice", text="(keyboard clicking)"),
            dict(ts_start=13, speaker="Dana", text="(keyboard clicking)"),
        ],
    )
    assert r["coverage"]["topics"]["themes"] == []


def test_unknown_partner_never_becomes_requested_name(tmp_path):
    r = ask(
        tmp_path,
        [
            dict(ts_start=1000, speaker="Alice", text="Garden irrigation needs repair"),
            dict(ts_start=1001, speaker="unknown", text="Garden tomatoes need water"),
        ],
    )
    assert r["coverage"]["topics"]["missing_participants"] == ["Dana"]
    assert not r["coverage"]["topics"]["joint_exchange_supported"]
    assert "cannot establish" in r["answer"]
    assert not r["coverage"]["topics"]["themes"]


def test_independent_speeches_are_not_shared_conversation(tmp_path):
    r = ask(
        tmp_path,
        [
            dict(ts_start=1000, speaker="Alice", text="Garden irrigation needs repair"),
            dict(ts_start=10000, speaker="Dana", text="Garden tomatoes need water"),
        ],
    )
    assert not r["coverage"]["topics"]["joint_exchange_supported"]
    assert "not proof of a shared conversation" in r["answer"]
    assert not r["coverage"]["topics"]["themes"]
    assert "Garden" not in r["answer"]


def test_named_voice_alias_is_resolved_before_ranking(tmp_path):
    write_voices(tmp_path, {7: "Dana"})
    r = ask(
        tmp_path,
        [
            dict(ts_start=1000, speaker="Alice", text="Garden irrigation needs repair"),
            dict(ts_start=1001, speaker="Speaker 7", speaker_id=7, text="Garden tomatoes need water"),
        ],
    )
    assert r["coverage"]["topics"]["joint_exchange_supported"]
    assert r["coverage"]["topics"]["participant_utterances"]["Dana"] == 1


def test_conversation_llm_posts_to_configured_ollama_url(model_server):
    model_server.says(json.dumps({"kind": "CONVERSATION", "topic": "Gardening", "summary": "They discussed tomatoes."}))
    with cfg.override({"ollama": {"url": "http://127.0.0.1:22434"}}):
        assert conversation_llm("Alice: tomatoes", background=False) == {
            "kind": "CONVERSATION",
            "topic": "Gardening",
            "summary": "They discussed tomatoes.",
        }
    assert [sent.url for sent in model_server.sent] == ["http://127.0.0.1:22434/api/chat"]


@pytest.mark.parametrize("background", [True, False])
def test_conversation_model_yields_only_for_background_work(monkeypatch, model_server, background):
    from liberational import voice_turn

    events = []
    monkeypatch.setattr(voice_turn, "wait_quiet", lambda: events.append("quiet"))

    def infer(sent):
        events.append("inference")
        content = {"kind": "CONVERSATION", "topic": "Gardening", "summary": "They discussed tomatoes."}
        return {"message": {"content": json.dumps(content)}}

    model_server.reply = infer
    assert conversation_llm("Alice: tomatoes", background=background) == {
        "kind": "CONVERSATION",
        "topic": "Gardening",
        "summary": "They discussed tomatoes.",
    }
    assert events == (["quiet", "inference"] if background else ["inference"])


def test_spoken_conversation_question_does_not_wait_for_its_own_turn(tmp_path, monkeypatch, model_server):
    from liberational import voice_turn
    from liberational.memory import conversation_summary

    speech_fake.fake().write(
        [
            {"ts_start": 1, "speaker": "Alice", "text": "Water the tomatoes."},
            {"ts_start": 2, "speaker": "Dana", "text": "The tomatoes look dry."},
            {"ts_start": 3, "speaker": "Alice", "text": "The garden needs water."},
            {"ts_start": 4, "speaker": "Dana", "text": "I will water it."},
        ]
    )

    def wait():
        pytest.fail("A foreground conversation answer waited for the voice turn it belongs to")

    content = {"kind": "CONVERSATION", "topic": "Gardening", "summary": "They discussed watering tomatoes."}
    monkeypatch.setattr(voice_turn, "wait_quiet", wait)
    monkeypatch.setattr(conversation_summary, "_llm", conversation_llm)
    model_server.says(json.dumps(content))
    result = h._conversation_summary(tmp_path, {"participants": ["Alice", "Dana"], "start": 0, "end": 100})
    assert result is not None
    assert "Gardening" in result[0]
