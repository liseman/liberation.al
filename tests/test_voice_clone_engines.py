"""Voice cloning engine selection (MLX, CUDA, Kokoro fallback) and the CUDA backend's streaming, with models faked."""

import json
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from liberational import config as cfg
from liberational import model_runtime
from liberational.speech import qwen3_cuda, tts, voice_clone


class FakeCloneModel:
    """Yields three chunks of 0.1 s each, like a streaming clone engine."""

    def __init__(self, name="fake clone"):
        self.calls = []
        self.name = name

    def generate(self, **kw):
        self.calls.append(kw)
        for i in range(3):
            yield SimpleNamespace(audio=np.full(2400, i, np.float32), sample_rate=24000)


class FakeKokoro:
    def __init__(self, voice, speed):
        self.texts = []

    def generate(self, text):
        self.texts.append(text)
        yield np.zeros(240, np.float32), 24000


@pytest.fixture(autouse=True)
def unloaded(monkeypatch):
    """No engine loaded yet, and whatever a test loads (or selects) is forgotten after it."""
    monkeypatch.setattr(voice_clone, "_m", None)
    monkeypatch.setattr(voice_clone, "KIND", None)
    monkeypatch.setattr(voice_clone, "ENGINE", voice_clone.ENGINE)


@pytest.fixture
def no_mlx(monkeypatch):
    monkeypatch.setattr(cfg, "has_mlx", lambda: False)


@pytest.fixture
def cuda(monkeypatch, no_mlx):
    """CUDA cloning available; returns the model qwen3_cuda.load hands out."""
    model = FakeCloneModel("fake CUDA clone")
    monkeypatch.setattr(qwen3_cuda, "available", lambda: (True, ""))
    monkeypatch.setattr(qwen3_cuda, "load", lambda: model)
    return model


def test_mlx_on_apple_silicon(monkeypatch):
    utils = types.ModuleType("mlx_audio.tts.utils")
    vars(utils).update(load=lambda repo: ("mlx model", repo))
    monkeypatch.setattr(cfg, "has_mlx", lambda: True)
    for name, mod in {
        "mlx_audio": types.ModuleType("mlx_audio"),
        "mlx_audio.tts": types.ModuleType("mlx_audio.tts"),
        "mlx_audio.tts.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    assert voice_clone.load()[0] == "mlx model"
    assert voice_clone.KIND == "mlx"


def test_cuda_when_available(cuda):
    assert voice_clone.load() is cuda
    assert voice_clone.can_clone()
    assert voice_clone.KIND == "cuda"


def test_kokoro_when_neither_speaks_without_a_reference(monkeypatch, no_mlx):
    monkeypatch.setattr(qwen3_cuda, "available", lambda: (False, "no CUDA GPU"))
    monkeypatch.setitem(sys.modules, "liberational.speech.kokoro", SimpleNamespace(Kokoro=FakeKokoro))
    audio, sr = voice_clone.say("Hello.", max_tokens=10)
    assert not voice_clone.can_clone()
    assert voice_clone.KIND == "kokoro"
    assert (len(audio), sr) == (240, 24000)
    assert voice_clone._m.texts == ["Hello."]  # no clone-only arguments reach Kokoro


def test_say_on_cuda_assembles_chunks_with_the_reference_transcript(tmp_path, cuda):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"")
    ref.with_suffix(".txt").write_text("What I read aloud.\n")
    audio, sr = voice_clone.say("Hello there.", ref=ref)
    assert (sr, len(audio)) == (24000, 7200)
    assert (cuda.calls[0]["ref_text"], cuda.calls[0]["ref_audio"]) == ("What I read aloud.", str(ref))


def test_reference_transcript_falls_back_to_the_prompt(tmp_path):
    assert voice_clone.ref_text_for(tmp_path / "none.wav") == voice_clone.PROMPT


def test_tts_clones_on_cuda_without_mlx(tmp_path, monkeypatch, cuda):
    ref = tmp_path / "owner_ref.wav"
    sf.write(str(ref), np.zeros(1600, np.float32), 16000)  # a reference the voice can read (#301)
    monkeypatch.setattr(tts, "REF_WAV", ref)
    t = tts.TTS(clone=True)
    assert t.clone
    assert [len(a) for a, _ in t.generate("Hello.")] == [2400, 2400, 2400]
    assert cuda.calls[0]["ref_text"] == voice_clone.PROMPT  # no .txt next to the reference


@pytest.fixture
def backend():
    """A Qwen3Cuda without the real model; returns it and the keyword arguments each stream call got."""
    b = object.__new__(qwen3_cuda.Qwen3Cuda)
    b._ref_mtimes = {}
    seen = []

    def stream(**kw):
        seen.append(kw)
        for i in range(3):
            yield np.full(1200, i, np.float32), 24000, {}

    b.m = SimpleNamespace(generate_voice_clone_streaming=stream, _voice_prompt_cache={})
    return b, seen


def test_cuda_streams_chunks_and_maps_mlx_style_arguments(backend):
    b, seen = backend
    out = list(
        b.generate("Hi.", "ref.wav", "ref text", stream=True, streaming_interval=0.5, max_tokens=99, temperature=0.7)
    )
    assert [len(r.audio) for r in out] == [1200, 1200, 1200]
    assert (seen[0]["chunk_size"], seen[0]["max_new_tokens"], seen[0]["temperature"]) == (6, 99, 0.7)


def test_cuda_without_streaming_yields_one_result(backend):
    b, _ = backend
    (r,) = b.generate("Hi.", "ref.wav", "ref text", stream=False)
    assert (len(r.audio), r.sample_rate) == (3600, 24000)


def test_a_rerecorded_reference_drops_the_cached_encoding(backend, tmp_path):
    b, _ = backend
    (tmp_path / "owner_ref.wav").touch()
    ref = str(tmp_path / "owner_ref.wav")
    key = (ref, "t", False, True)
    list(b.generate("Hi.", ref, "t"))
    b.m._voice_prompt_cache[key] = "old encoding"
    list(b.generate("Hi.", ref, "t"))
    assert key in b.m._voice_prompt_cache  # unchanged file: keep it
    os.utime(ref, ns=(1, 1))
    list(b.generate("Hi.", ref, "t"))
    assert key not in b.m._voice_prompt_cache


def test_mlx_load_reports_the_actual_model_only_after_success(tmp_path, monkeypatch):
    model = FakeCloneModel()
    utils = types.ModuleType("mlx_audio.tts.utils")
    vars(utils).update(load=lambda repo: model)
    monkeypatch.setattr(cfg, "has_mlx", lambda: True)
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", types.ModuleType("mlx_audio.tts"))
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils)
    with cfg.override(
        {"paths": {"state": str(tmp_path)}, "models": {"hf": {"tts_clone": {"repo": "example/test-voice"}}}}
    ):
        monkeypatch.setitem(voice_clone.ENGINES, "qwen3", ("example/test-voice", True))
        with model_runtime.reporting("voice"):
            assert voice_clone.load() is model
            assert voice_clone.load() is model
            (report,) = (tmp_path / "model_loads").glob("*.json")
            (loaded,) = json.loads(report.read_text())["models"]
            assert (loaded["key"], loaded["name"], loaded["purpose"], loaded["device"]) == (
                "models.hf.tts_clone",
                "example/test-voice",
                "voice",
                "Apple GPU",
            )


def test_a_failed_mlx_load_does_not_report_a_loaded_model(tmp_path, monkeypatch):
    def fail(repo):
        raise RuntimeError("model unavailable")

    utils = types.ModuleType("mlx_audio.tts.utils")
    vars(utils).update(load=fail)
    monkeypatch.setattr(cfg, "has_mlx", lambda: True)
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", types.ModuleType("mlx_audio.tts"))
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils)
    with cfg.override({"paths": {"state": str(tmp_path)}}), model_runtime.reporting("voice"):
        with pytest.raises(RuntimeError, match="model unavailable"):
            voice_clone.load()
        (report,) = (tmp_path / "model_loads").glob("*.json")
        assert json.loads(report.read_text())["models"] == []
