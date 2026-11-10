"""The overnight Parakeet/Qwen pass and its local inference boundary; never real recordings or models."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from liberational import config as cfg
from liberational.listen import deep


def never(*a):
    raise AssertionError("the merge model must not be asked")


def test_ollama_choose_posts_to_configured_ollama_url(monkeypatch):
    seen: list[str] = []

    def urlopen(req, **kwargs):
        seen.append(req.full_url)
        return io.BytesIO(json.dumps({"message": {"content": json.dumps({"text": "a b c f"})}}).encode())

    monkeypatch.setattr(deep.urllib.request, "urlopen", urlopen)
    with cfg.override({"ollama": {"url": "http://127.0.0.1:22434"}}):
        choose = deep.ollama_choose()
        assert choose("a b c d", "a b c e", "a b c f") == "a b c f"
    assert seen == ["http://127.0.0.1:22434/api/chat"]


def test_agreeing_passes_win_without_the_model():
    r = deep.merge(
        "Take the Roman for example", "Take the Romans for example.", qwen="take the romans, for example", choose=never
    )
    assert (r.text, r.method, r.uncertain) == ("Take the Romans for example.", "agree", False)


@pytest.mark.parametrize(
    ("parakeet", "qwen", "chosen"),
    [
        pytest.param("a b c e", "a b c f", "a b c f", id="close"),
        pytest.param("rain falls before dawn", "the package arrived today", "the package arrived today", id="wide"),
    ],
)
def test_constrained_choice_can_resolve_every_disagreement(parakeet, qwen, chosen):
    result = deep.merge("a b c d", parakeet, qwen=qwen, choose=lambda *args: chosen)
    assert (result.text, result.method, result.uncertain) == (chosen, "merged", False)


def test_chooser_may_use_a_known_name_but_not_invent_a_word():
    r = deep.merge(
        "hi Alex there",
        "hi Ally there",
        qwen="hi Alise there",
        choose=lambda *args: "hi Alice there",
        names=["Alice"],
    )
    assert (r.text, r.method) == ("hi Alice there", "merged")
    refused = deep.merge("a b c d", "a b c e", qwen="a b c f", choose=lambda *args: "a b c invented")
    assert (refused.text, refused.method, refused.uncertain) == ("a b c e", "parakeet", False)


@pytest.mark.parametrize(
    ("live", "parakeet", "qwen", "unheard"),
    [
        pytest.param("猫が来た", "犬が来た", "鳥が来た", "魚が来た", id="japanese"),
        pytest.param("猫在院子", "狗在院子", "鸟在院子", "鱼在院子", id="chinese"),
    ],
)
def test_cjk_disagreement_is_not_false_agreement_or_permission_to_invent(live, parakeet, qwen, unheard):
    result = deep.merge(live, parakeet, qwen=qwen, choose=lambda *args: unheard)
    assert (result.text, result.method, result.uncertain) == (live, "uncertain", True)


@pytest.mark.parametrize(
    ("live", "parakeet", "qwen", "heard", "unheard", "fallback", "method", "uncertain"),
    [
        pytest.param("a кот", "a кот", "b пёс", "b кот", "a лев", "a кот", "uncertain", True, id="cyrillic"),
        pytest.param(
            "a été choisi",
            "a été choisi",
            "a ete choisi",
            "a été choisi",
            "a étè choisi",
            "a été choisi",
            "parakeet",
            False,
            id="accents",
        ),
    ],
)
def test_unicode_heard_words_remain_available_but_unheard_words_are_rejected(
    live, parakeet, qwen, heard, unheard, fallback, method, uncertain
):
    accepted = deep.merge(live, parakeet, qwen=qwen, choose=lambda *args: heard)
    assert (accepted.text, accepted.method, accepted.uncertain) == (heard, "merged", False)
    rejected = deep.merge(live, parakeet, qwen=qwen, choose=lambda *args: unheard)
    assert (rejected.text, rejected.method, rejected.uncertain) == (fallback, method, uncertain)


@pytest.mark.parametrize("chosen", [None, "a completely invented sentence"])
def test_wide_disagreement_keeps_live_uncertain_when_choice_is_unavailable_or_invalid(chosen):
    r = deep.merge(
        "Leave the front door open",
        "rain falls before dawn",
        qwen="the package arrived today",
        choose=lambda *args: chosen,
    )
    assert (r.text, r.method, r.uncertain) == ("Leave the front door open", "uncertain", True)


@pytest.mark.parametrize(
    ("live", "qwen", "text", "method", "uncertain"),
    [
        ("Take the Roman for example", "Take the Romans for example.", "Take the Romans for example.", "qwen", False),
        ("Open door", "Open window", "Open window", "qwen", False),  # one substitution in two words: exactly CLOSE
        ("Hello Dana.", "hello Dana", "Hello Dana.", "kept", False),
        ("We can do it.", "Mm-hmm.", "We can do it.", "uncertain", True),
        ("hm", None, "hm", "kept", False),
    ],
)
def test_qwen_alone_must_be_a_near_revision_of_live(live, qwen, text, method, uncertain):
    r = deep.merge(live, "", qwen=qwen, choose=never)
    assert (r.text, r.method, r.uncertain) == (text, method, uncertain)


def test_no_qwen_speech_leaves_parakeets_pass_only_as_a_near_revision():
    assert (
        deep.merge("Take the Roman for example", "Take the Romans for example.", qwen=None, choose=never).method
        == "parakeet"
    )
    r = deep.merge(
        "I mean if it's a X will be more costly", "these are exciting costs for a punch", qwen=None, choose=never
    )
    assert (r.text, r.method, r.uncertain) == ("I mean if it's a X will be more costly", "uncertain", True)


def test_run_records_each_line_once_and_the_view_reads_the_revision(tmp_path: Path):
    store = deep.Revisions(tmp_path / "rev.sqlite3")
    lines = [
        {"id": "u1", "ts_start": 1.0, "ts_end": 3.0, "text": "Take the Roman", "audio": {"stream": "s"}},
        {"id": "u2", "ts_start": 4.0, "ts_end": 4.2, "text": "short", "audio": {"stream": "s"}},
        {"id": "u3", "ts_start": 5.0, "ts_end": 9.0, "text": "secret", "password": True, "audio": {"stream": "s"}},
    ]
    one = np.ones(16000, np.float32)
    counts = deep.run(
        lines,
        lambda e: one,
        lambda x: "Take the Romans",
        qwen=lambda x: "take the romans",
        store=store,
        choose=lambda ctx: never,
    )
    assert counts == {"agree": 1} and store.done() == {"u1"}
    assert store.all()["u1"]["qwen"] == "take the romans"
    assert deep.run(lines, lambda e: one, never, qwen=never, store=store, choose=lambda ctx: never) == {}
    e = store.apply({"id": "u1", "text": "Take the Roman"})
    assert e["text"] == "Take the Romans" and e["text_live"] == "Take the Roman" and e["revised"] == "agree"
    assert store.apply({"id": "u9", "text": "x"}) == {"id": "u9", "text": "x"}


def test_unavailable_qwen_aborts_without_marking_the_line_done_and_can_retry(tmp_path: Path):
    store = deep.Revisions(tmp_path / "rev.sqlite3")
    lines = [{"id": "u1", "ts_start": 1.0, "ts_end": 3.0, "text": "Open the door", "audio": {"stream": "s"}}]
    one = np.ones(16000, np.float32)

    def unavailable(audio):
        raise OSError("Qwen unavailable")

    with pytest.raises(OSError, match="Qwen unavailable"):
        deep.run(
            lines, lambda e: one, lambda x: "Open the door", qwen=unavailable, store=store, choose=lambda ctx: never
        )
    assert store.done() == set()
    assert store.all() == {}
    assert deep.run(
        lines,
        lambda e: one,
        lambda x: "Open the door",
        qwen=lambda x: "Open the door",
        store=store,
        choose=lambda ctx: never,
    ) == {"agree": 1}
    assert store.done() == {"u1"}


class ProcessorInputs(dict):
    def to(self, device, dtype):
        return self


class Processor:
    def __init__(self, parsed):
        self.parsed = parsed
        self.requests = []
        self.decoded = None
        self.decode_format = None

    def apply_transcription_request(self, **request):
        import torch

        self.requests.append(request)
        return ProcessorInputs(input_ids=torch.tensor([[100, 101]], dtype=torch.int64))

    def decode(self, generated, *, return_format):
        self.decoded = generated.tolist()
        self.decode_format = return_format
        return self.parsed


class Model:
    device = "cpu"

    def __init__(self, tokens, eos):
        import torch

        self.dtype = torch.float32
        self.tokens = tokens
        self.generation_config = SimpleNamespace(eos_token_id=eos)
        self.calls = 0
        self.options = None

    def generate(self, input_ids, *, do_sample, max_new_tokens):
        import torch

        self.calls += 1
        self.options = {"do_sample": do_sample, "max_new_tokens": max_new_tokens}
        return torch.cat((input_ids, torch.tensor([self.tokens], dtype=torch.int64)), dim=1)


@pytest.fixture
def qwen_adapter():
    pytest.importorskip("torch")

    def build(parsed=None, tokens=None, eos=None):
        adapter = object.__new__(
            deep.Qwen
        )  # Loading is the model boundary; exercise the real call without downloading weights.
        adapter.processor = Processor(parsed or {"language": "English", "transcription": "Real text"})
        adapter.model = Model(tokens if tokens is not None else [7, 2], eos if eos is not None else [2, 3])
        return adapter

    return build


@pytest.mark.parametrize("audio", [np.ones((2, 3), np.float32), np.array([np.nan]), np.array([np.inf])])
def test_qwen_rejects_nonmono_or_nonfinite_audio_before_inference(qwen_adapter, audio):
    adapter = qwen_adapter()
    with pytest.raises(ValueError, match="finite mono audio"):
        adapter(audio)
    assert adapter.processor.requests == []
    assert adapter.model.calls == 0


@pytest.mark.parametrize("audio", [np.array([], dtype=np.float32), np.zeros(16000, np.float32)])
def test_qwen_empty_or_silent_audio_does_not_reach_the_processor_or_model(qwen_adapter, audio):
    adapter = qwen_adapter()
    assert adapter(audio) is None
    assert adapter.processor.requests == []
    assert adapter.model.calls == 0


def test_qwen_keeps_text_when_language_is_unknown_and_decodes_only_generated_tokens(qwen_adapter):
    adapter = qwen_adapter(parsed={"language": None, "transcription": " Real text "})
    assert adapter(np.ones(16000, np.float32)) == "Real text"
    assert adapter.processor.decoded == [7, 2]
    assert adapter.processor.decode_format == "parsed"
    assert adapter.model.options == {"do_sample": False, "max_new_tokens": 512}
    request = adapter.processor.requests[0]
    assert request["processor_kwargs"] == {"sampling_rate": 16000, "return_tensors": "pt"}
    assert request["language"] is None and request["prompt"] is None


def test_qwen_blank_transcription_is_no_speech(qwen_adapter):
    adapter = qwen_adapter(parsed={"language": None, "transcription": " \n "})
    assert adapter(np.ones(16000, np.float32)) is None


def test_qwen_refuses_output_that_hits_the_generation_limit_without_eos(qwen_adapter):
    adapter = qwen_adapter(tokens=[7] * 512)
    with pytest.raises(RuntimeError, match="token limit"):
        adapter(np.ones(16000, np.float32))
    assert adapter.processor.decoded is None


@pytest.mark.parametrize(("token", "eos"), [(2, [2, 3]), (3, 3)])
def test_qwen_accepts_output_that_ends_at_the_generation_limit_with_eos(qwen_adapter, token, eos):
    adapter = qwen_adapter(tokens=[7] * 511 + [token], eos=eos)
    assert adapter(np.ones(16000, np.float32)) == "Real text"
