"""The Parakeet ASR caps MLX's buffer cache, which otherwise grows with every new utterance length."""

import sys
import types

import numpy as np
import pytest

from liberational.speech import asr


def test_parakeet_asr_caps_the_mlx_cache(monkeypatch):
    limits = []
    mx = types.SimpleNamespace(set_cache_limit=limits.append, array=lambda x: x)
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "parakeet_mlx", types.SimpleNamespace(from_pretrained=lambda repo: object()))
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", types.SimpleNamespace(get_logmel=None))
    monkeypatch.setattr(asr, "ASR_ENGINE", "parakeet")
    asr.ASR()
    assert limits == [asr.MLX_CACHE_LIMIT]
    assert 0 < asr.MLX_CACHE_LIMIT <= 2 << 30


@pytest.fixture
def mlx_asr(monkeypatch):
    loaded = []
    preprocessed = []
    generated = []
    preprocessor = types.SimpleNamespace(sample_rate=16000, features=128)
    mel = object()
    model = types.SimpleNamespace(preprocessor_config=preprocessor, text="")

    def from_pretrained(repo):
        loaded.append(repo)
        return model

    def get_logmel(audio, config):
        preprocessed.append((audio, config))
        return mel

    def generate(features):
        generated.append(features)
        return [types.SimpleNamespace(text=model.text)]

    model.generate = generate
    mx = types.SimpleNamespace(set_cache_limit=lambda limit: None, array=lambda audio: audio)
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "parakeet_mlx", types.SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", types.SimpleNamespace(get_logmel=get_logmel))
    monkeypatch.setattr(asr, "ASR_ENGINE", "parakeet")
    with asr.cfg.override({"models": {"hf": {"asr_parakeet": {"repo": "mlx-community/parakeet-tdt-0.6b-v3"}}}}):
        yield types.SimpleNamespace(
            asr=asr.ASR(), model=model, loaded=loaded, preprocessed=preprocessed, generated=generated, mel=mel
        )


def test_mlx_v3_uses_configured_repo_and_model_preprocessor(mlx_asr):
    audio = np.arange(3200, dtype=np.float64)[::2]
    mlx_asr.model.text = "  Bonjour Alice.  "

    assert mlx_asr.asr(audio) == "Bonjour Alice."

    assert mlx_asr.loaded == ["mlx-community/parakeet-tdt-0.6b-v3"]
    samples, config = mlx_asr.preprocessed[0]
    assert config is mlx_asr.model.preprocessor_config
    assert samples.dtype == np.float32
    assert samples.flags.c_contiguous
    np.testing.assert_array_equal(samples, audio)
    assert mlx_asr.generated == [mlx_asr.mel]


@pytest.mark.parametrize("size", [0, 1599])
def test_mlx_skips_audio_shorter_than_one_tenth_second(mlx_asr, size):
    assert mlx_asr.asr(np.zeros(size)) == ""
    assert mlx_asr.preprocessed == []
    assert mlx_asr.generated == []


@pytest.mark.parametrize("text, expected", [(" Um!? ", ""), ("  Dana said um.  ", "Dana said um.")])
def test_mlx_preserves_transcripts_except_standalone_fillers(mlx_asr, text, expected):
    mlx_asr.model.text = text
    assert mlx_asr.asr(np.zeros(1600)) == expected


if __name__ == "__main__":
    pytest.main([__file__])
