"""tools/phase1_harness.py: a turn the model leaves unanswered still reports its latencies."""

import importlib.util
from unittest.mock import create_autospec

import numpy as np

from liberational import config as cfg

_spec = importlib.util.spec_from_file_location("phase1_harness", cfg.ROOT / "tools" / "phase1_harness.py")
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


def test_an_empty_reply_still_prints_its_latencies(monkeypatch, capsys):
    tts = create_autospec(harness.tts)
    monkeypatch.setattr(harness, "asr", lambda audio16k: "hello")
    monkeypatch.setattr(harness, "llm_stream", lambda prompt: iter([]))
    monkeypatch.setattr(harness, "tts", tts)
    monkeypatch.setattr(harness, "t", iter([0.0, 0.25, 0.75]).__next__)  # start, after ASR, end
    assert harness.run_once(np.zeros(16000, np.float32)) == 0.75
    tts.assert_not_called()
    out = capsys.readouterr().out
    assert "ASR 250 ms" in out
    assert "LLM+TTS->first audio 500 ms" in out
