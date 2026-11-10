"""The speech benchmark (tools/asr_bench.py): its scoring, and the shipped clip set it runs on from a fresh clone."""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "asr_bench.py"
spec = importlib.util.spec_from_file_location("asr_bench", PATH)
assert spec and spec.loader
bench = importlib.util.module_from_spec(spec)
sys.modules["asr_bench"] = bench
spec.loader.exec_module(bench)


@pytest.mark.parametrize(
    ("hyp", "ref", "expected"),
    [
        pytest.param("Set a timer for 25 minutes.", "set a timer for twenty five minutes", (0, 7), id="digits"),
        pytest.param("Twenty-five, um, minutes", "twenty five minutes", (0, 3), id="hyphen-and-filler"),
        pytest.param("I paid 1,200", "i paid one thousand two hundred", (0, 6), id="thousands-comma"),
        pytest.param("turn the light off", "turn the lights off", (1, 4), id="substitution"),
        pytest.param("turn off", "turn the lights off", (2, 4), id="deletions"),
        pytest.param("", "", (0, 0), id="empty"),
    ],
)
def test_edits_ignore_formatting_and_count_word_errors(hyp, ref, expected):
    assert bench.edits(hyp, ref) == expected


def test_wer_is_pooled_over_the_corpus_not_averaged_per_clip():
    # 1 edit in 1 word and 0 in 3: pooled 1/4, where a per-clip mean would say 50%.
    assert bench.wer([("b", "a"), ("x y z", "x y z")]) == pytest.approx(0.25)


def test_every_shipped_clip_is_present_and_speech_length():
    manifest = json.loads((bench.CLIPS / "clips.json").read_text())
    durations = [len(bench.load16k(bench.CLIPS / c["file"])) / 16000 for c in manifest["clips"]]
    assert len(durations) >= 10
    assert all(1.0 < d < 15.0 for d in durations)
    assert all(c["text"].strip() for c in manifest["clips"])


class FakeASR:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, audio16k: np.ndarray) -> str:
        self.calls += 1
        return "hello there" if self.calls > 1 else ""  # the first call is the warm-up


def test_bench_reports_accuracy_and_agreement_and_saves(tmp_path, monkeypatch, capsys):
    from liberational.speech import asr

    sf.write(tmp_path / "a.wav", np.zeros(8000, np.float32), 8000)  # 1 s at 8 kHz, resampled to 16 kHz on load
    manifest = {
        "clips": [{"file": "a.wav", "text": "hello world", "engines": {"Other": "hello there friend", "Fake": "stale"}}]
    }
    (tmp_path / "clips.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(asr, "ASR", FakeASR)
    monkeypatch.setattr(asr, "ASR_MODEL_NAME", "Fake")

    bench.bench(tmp_path, verbose=False, save=True)

    out = capsys.readouterr().out
    assert "1 clips, 1 s of audio" in out
    assert "accuracy vs reference transcripts: 50.0% corpus WER" in out
    assert "disagreement vs Other: 33.3% corpus WER (1 clips)" in out
    assert "vs Fake" not in out  # never scored against its own earlier run
    saved = json.loads((tmp_path / "clips.json").read_text())["clips"][0]["engines"]
    assert saved == {"Other": "hello there friend", "Fake": "hello there"}
