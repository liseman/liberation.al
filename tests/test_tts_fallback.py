"""Speaking when the owner's voice reference goes missing or can't be read under a running voice service:
that reply is Kokoro, with one warning, and cloning resumes once the reference is back."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from liberational.speech import kokoro, qwen3_cuda, tts

CLONED, PLAIN = 0.5, -0.5  # what each fake voice says, so a reply shows which spoke it


class FakeClone:
    """The CUDA clone at its boundary: like faster-qwen3-tts, it reads the reference file for each reply."""

    def __init__(self) -> None:
        self.fail: str | None = None  # "before" any audio, or "after" the first chunk
        self.calls = 0

    def generate(self, text, ref_audio, ref_text, stream=True, streaming_interval=0.5):
        self.calls += 1
        if self.fail == "before":
            raise RuntimeError("CUDA out of memory")
        sf.read(str(ref_audio))  # a missing or broken file fails here, as in the real model
        yield SimpleNamespace(audio=np.full(4, CLONED, np.float32), sample_rate=24000)
        if self.fail == "after":
            raise RuntimeError("CUDA out of memory")


class FakeKokoro:
    def __init__(self, voice, speed):
        self.name = f"Kokoro ({voice}, kokoro-onnx)"

    def generate(self, text):
        yield np.full(4, PLAIN, np.float32), 24000


@pytest.fixture
def ref(tmp_path, monkeypatch):
    """A recorded reference, the CUDA clone and Kokoro faked, and the clone's loads counted."""
    path = tmp_path / "owner_ref.wav"
    record(path)
    model, loads = FakeClone(), []
    broken = SimpleNamespace(on=False)  # the model fails to load while on

    def load():
        loads.append(1)
        if broken.on:
            raise RuntimeError("CUDA out of memory")
        return model

    monkeypatch.setattr(tts, "REF_WAV", path)
    monkeypatch.setattr(tts, "REF_TXT", path.with_suffix(".txt"))
    monkeypatch.setattr(tts.cfg, "has_mlx", lambda: False)
    monkeypatch.setattr(qwen3_cuda, "available", lambda: (True, ""))
    monkeypatch.setattr(qwen3_cuda, "load", load)
    clock = SimpleNamespace(t=1000.0)
    monkeypatch.setattr(tts, "_now", lambda: clock.t)
    monkeypatch.setattr(kokoro, "Kokoro", FakeKokoro)
    return SimpleNamespace(path=path, model=model, loads=loads, broken=broken, clock=clock)


class Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        (self.warnings if record.levelno >= logging.WARNING else self.infos).append(record.getMessage())


@pytest.fixture
def logged():
    """What the voice logs (the tts logger doesn't reach pytest's root handler)."""
    handler = Collect()
    level = tts.LOG.level
    tts.LOG.addHandler(handler)
    tts.LOG.setLevel(logging.INFO)
    yield handler
    tts.LOG.removeHandler(handler)
    tts.LOG.setLevel(level)


@pytest.fixture
def warnings(logged):
    return logged.warnings


def record(path, seconds=1.0):
    sf.write(str(path), np.zeros(int(16000 * seconds), np.float32), 16000)


def voice(engine: tts.TTS) -> float:
    """Which voice spoke a reply: CLONED or PLAIN."""
    audio, _sr = engine.synth("Hello there.")
    return float(audio[0])


def test_a_reply_is_cloned_while_the_reference_is_there(ref, warnings):
    engine = tts.TTS(clone=True)
    assert (voice(engine), voice(engine)) == (CLONED, CLONED)
    assert warnings == []


def test_a_missing_reference_is_kokoro_with_one_warning_until_it_is_back(ref, warnings):
    engine = tts.TTS(clone=True)
    ref.path.unlink()
    assert (voice(engine), voice(engine)) == (PLAIN, PLAIN)
    assert len(warnings) == 1 and "missing" in warnings[0]
    record(ref.path)
    assert voice(engine) == CLONED
    assert ref.loads == [1]  # the clone model stays loaded throughout


def test_an_unreadable_reference_is_kokoro_until_it_is_recorded_again(ref, warnings):
    engine = tts.TTS(clone=True)
    ref.path.write_bytes(b"not a wav at all")
    assert (voice(engine), voice(engine)) == (PLAIN, PLAIN)
    assert len(warnings) == 1 and "can't be read" in warnings[0]
    record(ref.path, seconds=2.0)
    assert voice(engine) == CLONED


def test_a_service_started_without_a_reference_clones_once_one_is_recorded(ref, warnings):
    ref.path.unlink()
    engine = tts.TTS(clone=True)
    assert voice(engine) == PLAIN
    assert ref.loads == []  # no clone model until there is something to clone
    record(ref.path)
    assert voice(engine) == CLONED
    assert len(warnings) == 1


def test_the_reference_coming_back_is_logged(ref, logged):
    engine = tts.TTS(clone=True)
    ref.path.unlink()
    voice(engine)
    record(ref.path)
    voice(engine)
    assert [m for m in logged.infos if "is back" in m] != []


def test_a_clone_that_keeps_failing_is_kokoro_with_one_warning_and_tried_again_later(ref, logged):
    engine = tts.TTS(clone=True)
    ref.model.fail = "before"
    assert [voice(engine) for _ in range(3)] == [PLAIN] * 3
    assert len(logged.warnings) == 1 and "cloning failed" in logged.warnings[0]
    assert ref.model.calls == 1  # not tried again on every reply
    assert not [m for m in logged.infos if "is back" in m]  # the reference never left
    ref.model.fail = None
    ref.clock.t += tts.RETRY_S
    assert voice(engine) == CLONED


def test_a_clone_model_that_wont_load_is_kokoro_and_doesnt_blame_the_reference(ref, warnings):
    ref.broken.on = True
    engine = tts.TTS(clone=True)
    assert [voice(engine) for _ in range(3)] == [PLAIN] * 3
    assert ref.loads == [1]
    assert len(warnings) == 1 and "model failed to load" in warnings[0] and "can't be read" not in warnings[0]
    ref.broken.on = False
    ref.clock.t += tts.RETRY_S
    assert voice(engine) == CLONED


def test_a_clone_that_fails_partway_is_not_finished_in_another_voice(ref, warnings):
    engine = tts.TTS(clone=True)
    ref.model.fail = "after"
    with pytest.raises(RuntimeError):
        engine.synth("Hello there.")


def test_kokoro_is_used_without_asking_for_a_clone(ref, warnings):
    engine = tts.TTS(clone=False)
    assert voice(engine) == PLAIN
    assert ref.loads == []


def test_voice_speaker_starts_without_enrollment_and_clones_only_after_recording(ref, warnings, monkeypatch):
    from liberational.assistant import voice as assistant_voice
    from liberational.speech import voice_clone

    ref.path.unlink()
    # The old Speaker bypassed TTS and passed the missing path straight to the clone engine.
    monkeypatch.setattr(voice_clone, "load", lambda: ref.model)

    def direct_clone(text, **kwargs):
        sf.read(str(ref.path))
        return np.full(4, CLONED, np.float32), 24000

    monkeypatch.setattr(voice_clone, "say", direct_clone)
    played = []
    monkeypatch.setattr(voice_clone, "play_chunks", lambda chunks: played.extend((a.copy(), sr) for a, sr in chunks))
    speaker = assistant_voice.Speaker()
    assert ref.loads == [], "startup without enrollment must not load a clone model"
    assert len(warnings) == 1 and "missing" in warnings[0]
    assert played == [], "startup warmup must not play audio"
    speaker.speak("Hello there.")
    assert len(played) == 1
    np.testing.assert_array_equal(played[0][0], np.full(4, PLAIN, np.float32))
    assert played[0][1] == 24000
    record(ref.path)
    speaker.speak("Hello again.")
    assert ref.loads == [1]
    assert len(played) == 2
    np.testing.assert_array_equal(played[1][0], np.full(4, CLONED, np.float32))
    assert played[1][1] == 24000
