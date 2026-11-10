"""Automatic voice clone references require individual human-confirmed transcript lines, never inferred matches."""

import json
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

import speech_fake
from liberational import config as cfg
from liberational import model_runtime
from liberational.domain.identity import Basis
from liberational.listen import revisions
from liberational.people import registry, transcript_view
from liberational.speech import clone_say
from voices import write_voices

SR = clone_say.SR
DANA = np.array([1.0, 0.0, 0.0, 0.0], np.float32)  # the voice's centroid


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Voice 3, named Dana by the owner; voice 5, unnamed; voice 7, merged into Dana's (a cluster merge)."""
    for name, path in (
        ("STATE", tmp_path),
        ("OUT", tmp_path / "clones"),
    ):
        monkeypatch.setattr(clone_say, name, path)
    write_voices(tmp_path, {3: {"label": "Dana", "centroid": DANA.tolist()}, 5: None}, forwards={7: 3})
    return tmp_path


def line(ts, sid=3, sim=0.7, dur=3.0):
    return {
        "ts_start": ts, "ts_end": ts + dur, "speaker_id": sid, "speaker_sim": sim, "device": "kitchen",
        "text": "this is a line with seven words", "audio": {"stream": "kitchen.audio", "start_ns": int(ts * 1e9), "end_ns": int((ts + dur) * 1e9)},
    }  # fmt: skip


def write(state, lines):
    speech_fake.fake().write(lines)


def confirm(state, lines, basis=Basis.OWNER_TAG):
    registry.at(state).claim_many(
        [registry.Claim(registry.utterance(e["ts_start"], e["device"]), "Dana") for e in lines],
        basis,
        transcript_view.VOICE_TAG,
    )


@pytest.fixture
def fake_models(monkeypatch):
    """Synthetic audio and a fixed transcription, without loading models or touching services."""
    monkeypatch.setattr(clone_say, "cut", lambda e, pad=0.15: np.full(int(clone_say.dur(e) * SR), 0.1, np.float32))
    monkeypatch.setattr("liberational.speech.asr.ASR", lambda: lambda audio: "what was said")


def test_only_individually_confirmed_lines_enter_a_reference(state, fake_models):
    named = [line(1000.0 + 10 * k, sim=0.2) for k in range(4)]
    automatic = [line(2000.0 + 10 * k, sim=0.99) for k in range(5)]
    confirm(state, named)
    write(state, named + automatic)
    with patch.object(clone_say, "cut", autospec=True, side_effect=clone_say.cut) as cut:
        ref = clone_say.build(3)
    assert [call.args[0]["ts_start"] for call in cut.call_args_list] == [1000.0, 1010.0, 1020.0, 1030.0]
    assert [call.kwargs for call in cut.call_args_list] == [{"pad": 0.0}] * 4
    assert ref.exists() and (state / "clones" / "3.txt").read_text() == "what was said"
    assert not (state / "clones" / "3.checks.json").exists()
    assert clone_say.ref_for(3, make=False) == ref
    assert json.loads((state / "clones" / "3.json").read_text())["reference_policy"] == "confirmed-lines-v1"


def test_naming_a_voice_does_not_confirm_its_later_auto_matches(state):
    raw = [line(1000.0, sim=0.99), line(1010.0, sid=7, sim=0.99)]
    assert clone_say.confirmed(transcript_view.View(state), raw) == [False, False]


def test_rejection_during_clone_build_cannot_stamp_stale_audio_as_current(state, fake_models, monkeypatch):
    raw = [line(1000.0 + 10 * k) for k in range(4)]
    confirm(state, raw)
    write(state, raw)

    def transcribe(audio):
        revisions.reject_lines(state, raw[:1])
        return "words before the correction"

    monkeypatch.setattr("liberational.speech.asr.ASR", lambda: transcribe)
    clone_say.build(3)
    assert clone_say.ref_for(3, make=False) is None


def test_build_refuses_auto_matches_even_to_a_named_voice(state, fake_models):
    write(state, [line(1000.0 + 10 * k, sim=0.99) for k in range(5)])
    with pytest.raises(SystemExit, match="Name more of their lines"):
        clone_say.build(3)
    assert not (state / "clones" / "3.wav").exists()


def test_backfill_matches_never_enter_an_automatic_reference(state, fake_models):
    reg = registry.at(state)
    lines = [line(2000.0 + 10 * k, sid=5) for k in range(5)]  # voice 5, unnamed; each line matched to Dana
    for e in lines:
        reg.claim(registry.utterance(e["ts_start"], e["device"]), "Dana", Basis.BACKFILL_MATCH, "backfill",
                  confidence=0.8, producer=registry.voice_producer())  # fmt: skip
    write(state, lines)
    with pytest.raises(SystemExit, match=r"only 0\.0 s of Dana's speech is usable for cloning; it needs 12 s"):
        clone_say.build(3)
    assert clone_say.candidates() == []
    assert not (state / "clones" / "3.wav").exists()


def test_a_line_the_owner_labeled_counts_even_on_another_voice(state):
    reg = registry.at(state)
    e = line(3000.0, sid=5, sim=0.3)  # heard on voice 5, labeled Dana by the owner in Live
    reg.claim(registry.utterance(e["ts_start"], e["device"]), "Dana", Basis.OWNER_TAG, transcript_view.VOICE_TAG)
    view = transcript_view.View(state)
    assert clone_say.confirmed(view, view.lines([e, line(3010.0, sid=5)])) == [True, False]


def test_the_list_says_heard_usable_and_needed(state, fake_models):
    reg = registry.at(state)
    named = [line(4000.0 + 10 * k) for k in range(2)]
    confirm(state, named)
    matched = [line(5000.0 + 10 * k, sid=5) for k in range(3)]
    for e in matched:
        reg.claim(registry.utterance(e["ts_start"], e["device"]), "Dana", Basis.BACKFILL_MATCH, "backfill",
                  confidence=0.8, producer=registry.voice_producer())  # fmt: skip
    write(state, named + matched)
    (dana,) = [c for c in clone_say.candidates() if c["id"] == 3]
    assert (dana["clean_s"], dana["usable_s"], dana["confirmed_s"], dana["unchecked_s"], dana["needs_s"]) == (
        6.0, 6.0, 6.0, 0.0, 12.0,
    )  # fmt: skip
    assert (dana["ready"], dana["rejected_s"]) == (False, 0.0)
    (state / "clones").mkdir()
    (state / "clones" / "3.checks.json").write_text(
        json.dumps({registry.line_key(e["ts_start"], e["device"]): 1.0 for e in matched})
    )
    with pytest.raises(SystemExit):
        clone_say.build(3)
    (dana,) = [c for c in clone_say.candidates() if c["id"] == 3]
    assert (dana["usable_s"], dana["unchecked_s"], dana["rejected_s"]) == (6.0, 0.0, 0.0)
    assert not dana["ready"]


def test_voice_names_merges_and_match_scores_cannot_confirm_individual_lines(state):
    merged = line(6000.0, sid=7)  # heard on voice 7, which a cluster merge folded into Dana's
    weak = line(6010.0, sim=0.3)  # a weak live match on Dana's own voice
    strong = line(6020.0)
    view = transcript_view.View(state)
    assert clone_say.confirmed(view, [merged, weak, strong]) == [False, False, False]


def test_lines_tagged_with_no_match_score_count_as_heard_and_usable(state):
    reg = registry.at(state)
    starts = [7000.0 + 10 * k for k in range(4)]
    tagged = [{**line(ts, sid=5), "speaker_sim": None} for ts in starts]  # 12 s tagged in Live, no match score
    for ts in starts:
        reg.claim(registry.utterance(ts, "kitchen"), "Dana", Basis.OWNER_TAG, transcript_view.VOICE_TAG)
    write(state, tagged)
    (dana,) = [c for c in clone_say.candidates() if c["id"] == 3]
    assert (dana["clean_s"], dana["usable_s"], dana["ready"]) == (12.0, 12.0, True)


def test_a_clip_is_read_from_the_broker_padded_and_at_the_clone_rate(state, monkeypatch):
    asked = []

    def audio(stream, start_ns, end_ns, channel=0):
        asked.append((stream, start_ns, end_ns, channel))
        return 16_000, np.full(int((end_ns - start_ns) / 1e9 * 16_000), 0.2, np.float32)

    monkeypatch.setattr(speech_fake.fake(), "audio", audio, raising=False)
    x = clone_say.cut(line(100.0, dur=2.0))
    assert asked == [("kitchen.audio", 99_850_000_000, 102_150_000_000, 0)]  # 0.15 s each side
    assert len(x) == round(2.3 * SR)  # resampled to the clone rate


def test_a_clip_with_no_audio_kept_is_refused(state, monkeypatch):
    monkeypatch.setattr(speech_fake.fake(), "audio", lambda *a, **k: (16_000, np.zeros(100, np.float32)), raising=False)
    with pytest.raises(ValueError, match="no audio was kept"):
        clone_say.cut(line(100.0))
    with pytest.raises(ValueError, match="no audio"):
        clone_say.cut({**line(100.0), "audio": None})


@pytest.mark.parametrize("basis", [Basis.OWNER_TAG, Basis.CORRECTION])
def test_individual_confirmation_ignores_the_auto_match_score(state, basis):
    raw = [line(8000.0, sid=5, sim=0.01), line(8010.0, sim=0.99)]
    confirm(state, raw[:1], basis)
    assert clone_say.confirmed(transcript_view.View(state), raw) == [True, False]


def test_candidate_totals_share_complete_history_over_one_broker_page(state):
    log = speech_fake.fake()
    raw = [line(10000.0 + 10 * k) for k in range(1030)]
    confirm(state, raw)
    write(state, raw)
    with patch.object(log, "all_lines", autospec=True, side_effect=log.all_lines) as reads:
        candidates = clone_say.candidates()
        assert reads.call_count == 1
        assert next(c for c in candidates if c["id"] == 3)["clean_s"] == 3090.0
        clone_say.candidates()
        assert reads.call_count == 1
        extra = line(30000.0)
        confirm(state, [extra])
        log.say(extra)
        updated = clone_say.candidates()
        assert reads.call_count == 2
        assert next(c for c in updated if c["id"] == 3)["clean_s"] == 3093.0


def test_clone_subprocess_entrypoint_reports_its_own_successful_load_and_cleans_up(tmp_path, monkeypatch):
    seen = []

    def speak(sid, text, out):
        token = model_runtime.loaded("models.hf.tts_clone", "example/voice", "voice", "CPU")
        assert token is not None, "clone subprocess entrypoint must activate model reporting"
        (report,) = (tmp_path / "model_loads").glob("*.json")
        seen.append(json.loads(report.read_text()))

    monkeypatch.setattr(clone_say, "say", speak)
    monkeypatch.setattr(sys, "argv", ["clone_say", "say", "3", "Hello", str(tmp_path / "out.wav")])
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        clone_say.main()
    assert [(r["pid"], r["service"], r["models"][0]["name"]) for r in seen] == [
        (os.getpid(), "clone-say", "example/voice")
    ]
    assert list((tmp_path / "model_loads").glob("*.json")) == []
