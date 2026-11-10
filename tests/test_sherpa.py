"""Parakeet's portable backend contract, without loading model weights."""

import sys
import types

import numpy as np
import pytest

from liberational.speech import sherpa


class FakeStream:
    def __init__(self, text):
        self.result = types.SimpleNamespace(text=text)
        self.sample_rate = None
        self.samples = None

    def accept_waveform(self, sample_rate, samples):
        self.sample_rate = sample_rate
        self.samples = samples


class FakeRecognizer:
    def __init__(self):
        self.options = {}
        self.streams = []
        self.decoded = []
        self.text = ""

    def create_stream(self):
        stream = FakeStream(self.text)
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        self.decoded.append(stream)


@pytest.fixture
def backend(monkeypatch, tmp_path):
    recognizer = FakeRecognizer()

    def from_transducer(**options):
        recognizer.options = options
        return recognizer

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(OfflineRecognizer=types.SimpleNamespace(from_transducer=from_transducer)),
    )
    model_dir = tmp_path / "parakeet-v3-int8"
    model_dir.mkdir()
    with sherpa.cfg.override({"models": {"files": {"asr_parakeet": {"path": str(model_dir)}}}}):
        yield recognizer, model_dir


def test_v3_int8_constructor_uses_configured_components_and_128_mel_features(backend, monkeypatch):
    recognizer, model_dir = backend
    monkeypatch.setattr(sherpa.os, "cpu_count", lambda: 16)
    sherpa.SherpaParakeet()

    assert recognizer.options == {
        "encoder": str(model_dir / "encoder.int8.onnx"),
        "decoder": str(model_dir / "decoder.int8.onnx"),
        "joiner": str(model_dir / "joiner.int8.onnx"),
        "tokens": str(model_dir / "tokens.txt"),
        "num_threads": 4,
        "sample_rate": 16000,
        "feature_dim": 128,
        "decoding_method": "greedy_search",
        "model_type": "nemo_transducer",
        "provider": "cpu",
    }


def test_explicit_model_directory_and_provider_override_defaults(backend, tmp_path):
    recognizer, _ = backend
    sherpa.SherpaParakeet(tmp_path, provider="cuda")
    assert recognizer.options["encoder"] == str(tmp_path / "encoder.int8.onnx")
    assert recognizer.options["provider"] == "cuda"


def test_missing_model_directory_explains_how_to_fetch_it(backend, tmp_path):
    recognizer, _ = backend
    with pytest.raises(FileNotFoundError, match="run `mise run models`"):
        sherpa.SherpaParakeet(tmp_path / "missing")
    assert recognizer.options == {}


def test_sherpa_decodes_each_utterance_in_a_fresh_stream_as_contiguous_float32(backend):
    recognizer, _ = backend
    recognizer.text = "  Bonjour Dana.  "
    transcribe = sherpa.SherpaParakeet()
    audio = np.arange(3200, dtype=np.float64)[::2]

    assert transcribe(audio) == "Bonjour Dana."
    assert transcribe(audio) == "Bonjour Dana."

    assert len(recognizer.streams) == 2
    assert recognizer.streams[0] is not recognizer.streams[1]
    assert recognizer.decoded == recognizer.streams
    stream = recognizer.streams[0]
    assert stream.sample_rate == 16000
    assert stream.samples.dtype == np.float32
    assert stream.samples.flags.c_contiguous
    np.testing.assert_array_equal(stream.samples, audio)


@pytest.mark.parametrize("size", [0, 1599])
def test_sherpa_skips_audio_shorter_than_one_tenth_second(backend, size):
    recognizer, _ = backend
    assert sherpa.SherpaParakeet()(np.zeros(size)) == ""
    assert recognizer.streams == []
    assert recognizer.decoded == []


@pytest.mark.parametrize("text, expected", [(" Hmm!? ", ""), ("  Alice said hmm.  ", "Alice said hmm.")])
def test_sherpa_preserves_transcripts_except_standalone_fillers(backend, text, expected):
    recognizer, _ = backend
    recognizer.text = text
    assert sherpa.SherpaParakeet()(np.zeros(1600)) == expected
