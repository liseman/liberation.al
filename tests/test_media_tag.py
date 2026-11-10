"""Lines heard from a TV, tagged by ear (listen/media_tag.py), and the owner's Not TV."""

import json
import threading
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from liberational import config as cfg
from liberational.dashboard import context, server
from liberational.listen import media_tag, revisions
from liberational.people import registry

T = 1_800_000_000.0


# ---- a clip as the tagger takes it ----


def test_a_quiet_short_clip_is_made_loud_resampled_and_repeated_to_five_seconds():
    rng = np.random.default_rng(1)
    pcm = (rng.normal(0, 0.005, 48000)).astype(np.float32)  # 1 s at 48 kHz, a quiet room
    wave = media_tag.prepared(pcm, 48000)
    assert wave is not None and wave.dtype == np.float32
    assert len(wave) == int(media_tag.MIN_S * media_tag.RATE)
    assert float(np.sqrt(np.mean(wave[:32000].astype(np.float64) ** 2))) == pytest.approx(
        media_tag.TARGET_RMS, rel=0.05
    )
    np.testing.assert_array_equal(wave[:32000], wave[32000:64000])  # repeated, not padded


def test_a_long_clip_is_kept_whole():
    pcm = np.full(32000 * 7, 0.05, np.float32)
    wave = media_tag.prepared(pcm, 32000)
    assert wave is not None and len(wave) == 32000 * 7


@pytest.mark.parametrize(
    "pcm",
    [
        pytest.param(np.zeros(16000, np.float32), id="silence-fill"),
        pytest.param(np.concatenate([np.zeros(8000), np.full(8000, 0.1)]).astype(np.float32), id="half-not-kept"),
        pytest.param(np.full(16000, 1e-7, np.float32), id="silent"),
        pytest.param(np.zeros(0, np.float32), id="empty"),
    ],
)
def test_a_clip_mostly_not_kept_or_silent_is_not_scored(pcm):
    assert media_tag.prepared(pcm, 16000) is None


def scores(**kw: float) -> dict[str, float]:
    return {name: 0.0 for name in (*media_tag.BROADCAST, *media_tag.ROOM)} | kw


@pytest.mark.parametrize(
    ("got", "is_media"),
    [
        pytest.param(scores(Music=0.23), True, id="music-bed"),
        pytest.param(scores(Music=0.22), False, id="music-under"),
        pytest.param(scores(Television=0.2, Radio=0.05, **{"Speech synthesizer": 0.06}), True, id="broadcast"),
        pytest.param(
            scores(Television=0.2, Radio=0.05, **{"Speech synthesizer": 0.06, "Conversation": 0.1}),
            False,
            id="a-room-conversation-takes-it-down",
        ),
        # One voice close to the desk microphone reads as a monologue: it isn't a broadcast.
        pytest.param(scores(**{"Narration, monologue": 0.39, "Music": 0.03}), False, id="close-mic-monologue"),
        pytest.param(scores(Speech=0.9), False, id="plain-speech"),
    ],
)
def test_what_sounds_like_a_broadcast(got, is_media):
    assert media_tag.media(got) is is_media


@pytest.mark.parametrize(
    ("speaker", "theirs"),
    [("Dana", True), ("Speaker 9", False), ("unknown", False), ("multiple speakers", False), ("", False)],
)
def test_a_line_is_a_household_voice_when_the_transcript_names_someone(speaker, theirs):
    assert media_tag.household({"speaker": speaker}) is theirs


# ---- the tagger following the speech log ----


def line(ts: float, text: str, speaker: str = "Speaker 9", **kw) -> dict:
    return {
        "id": f"id-{ts}",
        "ts_start": ts,
        "ts_end": ts + 3,
        "device": "wave-3",
        "text": text,
        "speaker": speaker,
        "speaker_id": 9,
        "audio": {"stream": "wave-3.audio", "start_ns": int(ts * 1e9), "end_ns": int((ts + 3) * 1e9)},
        **kw,
    }


class Speech:
    """The speech log as the tagger reads it: lines appended, `since` and `mark` by count."""

    def __init__(self, state) -> None:
        self.state, self.rows = state, []

    def mark(self):
        return len(self.rows), None

    def since(self, after):
        return revisions.filter_lines(self.state, self.rows[after:]), len(self.rows)


class Ears:
    """A classifier that says a line is a TV's by its text."""

    def scores(self, pcm, rate):
        return scores(Music=0.9) if "news" in str(pcm) else scores(Speech=0.9)


@pytest.fixture
def tagger(tmp_path):
    (tmp_path / "screens_on.json").write_text(json.dumps({"cams": {"cam0": [[T, T + 600]]}, "at": T}))
    speech = Speech(tmp_path)
    t = media_tag.MediaTagger(
        tmp_path,
        cast(Any, Ears()),
        speech=speech,
        audio=lambda e: (16000, e["text"]),  # the "audio" is the text, which Ears reads
        view=lambda lines: list(lines),
        now=lambda: T + 3600,  # an hour on: every test line has settled
    )
    return t, speech, tmp_path


def test_a_line_that_sounds_like_tv_while_a_screen_is_on_is_marked_by_the_tagger(tagger):
    t, speech, state = tagger
    speech.rows += [line(T + 10, "the news at ten"), line(T + 20, "dinner's ready")]
    assert t.step() == [registry.key_of(speech.rows[0])]
    got = revisions.filter_lines(state, speech.rows)
    assert [e.get("media") for e in got] == [True, None]
    (mark,) = revisions._media(state)
    assert mark.value["source"] == "classifier" and len(mark.value["lines"]) == 1, "one line a mark"
    assert t.step() == [], "each line is looked at once"


def test_an_error_mid_batch_leaves_that_line_and_the_rest_for_the_next_pass(tagger):
    """The cursor moved past the whole batch before scoring, so one error lost the rest for good."""
    t, speech, _state = tagger
    speech.rows += [line(T + 10, "the news one"), line(T + 20, "the news two"), line(T + 30, "the news three")]
    hiccups = ["the news two"]
    audio = t.audio

    def flaky(e):
        if e["text"] in hiccups:
            hiccups.remove(e["text"])
            raise TimeoutError("broker")
        return audio(e)

    t.audio = flaky
    with pytest.raises(TimeoutError):
        t.step()
    assert t.step() == [registry.key_of(e) for e in speech.rows[1:]]
    assert t.step() == []


def test_a_line_that_always_fails_is_passed_over_after_a_few_tries(tagger):
    t, speech, _state = tagger
    speech.rows += [line(T + 10, "the news, broken"), line(T + 20, "the news after")]
    audio = t.audio

    def broken(e):
        if "broken" in e["text"]:
            raise ValueError("bad audio")
        return audio(e)

    t.audio = broken
    for _ in range(media_tag.TRIES - 1):
        with pytest.raises(ValueError):
            t.step()
    assert t.step() == [registry.key_of(speech.rows[1])]


def test_a_missing_module_never_passes_a_line_over(tagger):
    # While continuum_py was rebuilt in a deploy, every line failed to import it and was given up on.
    t, speech, _state = tagger
    speech.rows += [line(T + 10, "the news"), line(T + 20, "the news after")]
    audio = t.audio

    def missing(e):
        raise ModuleNotFoundError("No module named 'continuum_py.read'")

    t.audio = missing
    for _ in range(media_tag.TRIES + 2):
        with pytest.raises(ModuleNotFoundError):
            t.step()
    t.audio = audio  # rebuilt
    assert t.step() == [registry.key_of(e) for e in speech.rows]


def test_a_line_is_judged_once_its_screen_span_could_be_written(tmp_path):
    """A show's span is written after it has run 30 s, so a line from its first moments wasn't covered
    when it arrived, and was never looked at again."""
    speech = Speech(tmp_path)
    wall = [T + 10.0]
    t = media_tag.MediaTagger(
        tmp_path, cast(Any, Ears()), speech=speech, audio=lambda e: (16000, e["text"]), view=list, now=lambda: wall[0]
    )
    speech.rows.append(line(T + 5, "the news at ten"))  # heard before any span was written
    assert t.step() == []
    (tmp_path / "screens_on.json").write_text(json.dumps({"cams": {"cam0": [[T, T + 600]]}, "at": T + 40}))
    wall[0] = T + 8 + media_tag.SETTLE_S  # it ended at T + 8
    assert t.step() == [registry.key_of(speech.rows[0])]


def test_a_line_whose_audio_isnt_readable_yet_waits_and_then_is_passed_over(tmp_path):
    (tmp_path / "screens_on.json").write_text(json.dumps({"cams": {"cam0": [[T, T + 600]]}, "at": T}))
    speech = Speech(tmp_path)
    wall = [T + 9 + media_tag.SETTLE_S]  # both lines (ending T + 8 and T + 9) have settled
    stored: dict[str, Any] = {}
    t = media_tag.MediaTagger(
        tmp_path,
        cast(Any, Ears()),
        speech=speech,
        audio=lambda e: stored.get(e["text"]),
        view=list,
        now=lambda: wall[0],
    )
    speech.rows += [line(T + 5, "the news at ten"), line(T + 6, "the news at eleven")]
    assert t.step() == [] and len(t.pending) == 2  # the first line's audio isn't readable yet: both wait
    stored["the news at ten"] = (16000, "the news at ten")
    stored["the news at eleven"] = (16000, "the news at eleven")
    assert t.step() == [registry.key_of(e) for e in speech.rows]
    speech.rows.append(line(T + 20, "the news, never stored"))
    wall[0] = T + 23 + media_tag.AUDIO_WAIT_S  # long gone: passed over
    assert t.step() == [] and t.pending == []


def test_no_screen_on_a_household_voice_or_a_marked_line_is_left_alone(tagger):
    t, speech, state = tagger
    speech.rows += [
        line(T + 900, "the news at ten"),  # no screen on then
        line(T + 30, "the news at ten", speaker="Dana"),  # someone the transcript names
        line(T + 40, "the news at ten", media=True),  # marked already
    ]
    assert t.step() == []
    assert revisions._media(state) == []


def test_not_tv_takes_back_exactly_the_tagger_line_and_it_is_never_marked_again(tagger):
    t, speech, state = tagger
    speech.rows += [line(T + 10, "the news at ten"), line(T + 20, "more news")]
    first, _ = t.step()
    token, lines = revisions.unmark_media(state, [first])
    assert token is not None and lines == [first]
    assert [e.get("media") for e in revisions.filter_lines(state, speech.rows)] == [None, True]
    speech.rows.append(line(T + 30, "dinner's ready"))
    t.step()
    assert [e.get("media") for e in revisions.filter_lines(state, speech.rows)] == [None, True, None]


def test_a_start_looks_only_at_lines_heard_from_then_on(tmp_path):
    (tmp_path / "screens_on.json").write_text(json.dumps({"cams": {"cam0": [[T, T + 600]]}, "at": T}))
    speech = Speech(tmp_path)
    speech.rows.append(line(T + 10, "the news at ten"))
    t = media_tag.MediaTagger(
        tmp_path, cast(Any, Ears()), speech=speech, audio=lambda e: (16000, e["text"]), view=list, now=lambda: T + 3600
    )
    assert t.step() == []


def test_not_tv_takes_back_only_its_lines_of_a_mark_and_comes_undone(tmp_path):
    """Not TV on one line took back the owner's whole mark, with no Undo, and raced its own Undo."""
    rows = [line(T, "one"), line(T + 5, "two"), line(T + 10, "three")]
    revisions.mark_media(tmp_path, rows)
    token, lines = revisions.unmark_media(tmp_path, [registry.key_of(rows[1])])
    assert lines == [registry.key_of(rows[1])]
    assert [e.get("media") for e in revisions.filter_lines(tmp_path, rows)] == [True, None, True]
    assert revisions.unmark_media(tmp_path, [registry.key_of(rows[1])]) == (None, [])  # a double click: nothing left
    assert token is not None
    assert revisions.undo_correction(tmp_path, token) == [registry.key_of(rows[1])]
    assert [e.get("media") for e in revisions.filter_lines(tmp_path, rows)] == [True, True, True]
    with pytest.raises(ValueError, match="already"):
        revisions.undo_correction(tmp_path, token)  # undone twice: a clear refusal, not a half-done write


def test_this_is_tv_again_after_not_tv_marks_the_line_anew(tmp_path):
    rows = [line(T, "one")]
    revisions.mark_media(tmp_path, rows, source="classifier")
    revisions.unmark_media(tmp_path, [registry.key_of(rows[0])])
    token = revisions.mark_media(tmp_path, rows)
    assert [(e.get("media"), e.get("media_by")) for e in revisions.filter_lines(tmp_path, rows)] == [(True, "owner")]
    revisions.undo_correction(tmp_path, token)
    assert revisions.filter_lines(tmp_path, rows)[0].get("media") is None  # the tagger's mark stays taken back


def test_this_is_tv_on_a_line_the_tagger_marked_never_hands_over_the_taggers_token(tmp_path):
    """The owner's Undo then took back the tagger's mark."""
    rows = [line(T, "one")]
    tagger = revisions.mark_media(tmp_path, rows, source="classifier")
    assert revisions.current_media(tmp_path, [registry.key_of(rows[0])]) is None
    with pytest.raises(ValueError, match="already"):
        revisions.mark_media(tmp_path, rows)
    assert revisions.current_media(tmp_path, [registry.key_of(rows[0])], "classifier") == tagger


# ---- the owner's Not TV over HTTP ----


@pytest.fixture
def heard(tmp_path, monkeypatch):
    rows = [line(T, "the news"), line(T + 5, "dinner")]
    monkeypatch.setattr(context, "STATE", tmp_path)
    return rows


def test_not_tv_unmarks_the_lines_and_says_when_none_was_marked(heard, tmp_path):
    revisions.mark_media(tmp_path, heard[:1], source="classifier")
    key = registry.key_of(heard[0])
    code, body = server.not_media({"lines": [key]})
    assert (code, body["lines"]) == (200, [key])
    assert server.not_media({"lines": [key]})[0] == 404
    assert server.undo_transcript({"undo": body["undo"]})[0] == 200  # its Undo marks it a TV's again
    assert revisions.filter_lines(tmp_path, heard[:1])[0].get("media") is True
    assert server.not_media({"lines": []})[0] == 400


# ---- the model ----


@pytest.mark.parametrize(("drawn", "idle"), [pytest.param([], True, id="none"), pytest.param(["a"], False, id="one")])
def test_the_tagger_says_once_at_start_when_no_screen_zone_is_drawn(tmp_path, monkeypatch, caplog, drawn, idle):
    from liberational.see import zones

    definition = SimpleNamespace(
        zones=[SimpleNamespace(kind="screen", polygon=((0, 0), (1, 0), (0, 1)))] if drawn else []
    )
    monkeypatch.setattr(zones, "current", lambda state: {"cam0": definition})
    with caplog.at_level("INFO"):
        media_tag._idle_notice(tmp_path)
    assert ("media tag: idle, no screen zones drawn" in caplog.text) is idle


def test_without_its_model_the_tagger_stays_off(tmp_path, monkeypatch):
    with cfg.override({"paths": {"models": str(tmp_path)}}):
        assert media_tag.Classifier.configured() is None
        thread = media_tag.start(tmp_path, threading.Event())
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_a_tagger_that_cant_start_never_stops_identity_and_is_tried_again(tmp_path, monkeypatch):
    """A corrupt model or a broker mid-restart: start returns, and the tagger is tried again later."""
    stop, tries = threading.Event(), []

    def broken():
        tries.append(1)
        if len(tries) == 2:
            stop.set()
        raise RuntimeError("broker restarting")

    monkeypatch.setattr(media_tag.Classifier, "configured", staticmethod(broken))
    monkeypatch.setattr(media_tag, "START_RETRY_S", 0.0)
    thread = media_tag.start(tmp_path, stop)
    thread.join(timeout=5)
    assert len(tries) == 2


def test_the_built_model_scores_a_clip_when_this_machine_has_it():
    classifier = media_tag.Classifier.configured()
    if classifier is None:
        pytest.skip("the audio tagger is built on each machine (tools/export_audio_tagger.py)")
    t = np.arange(32000 * 2) / 32000
    got = classifier.scores((0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 32000)
    assert got is not None and set(got) == {*media_tag.BROADCAST, *media_tag.ROOM}
    assert all(0.0 <= v <= 1.0 for v in got.values())


def test_a_repeated_failure_is_said_once_and_the_recovery_once(tagger, caplog, monkeypatch):
    # An outage used to log the same warning every POLL_S, about 1,100 times.
    t, _speech, _state = tagger
    # the same kind of failure with a message that changes each pass is still one outage
    outcomes: list[Exception | None] = [ModuleNotFoundError(f"No module named 'x' (pass {n})") for n in range(5)]
    outcomes += [None, RuntimeError("other"), None]
    stop = threading.Event()

    def step():
        ex = outcomes.pop(0)
        if not outcomes:
            stop.set()
        if ex is not None:
            raise ex
        return []

    monkeypatch.setattr(t, "step", step)
    monkeypatch.setattr(media_tag, "POLL_S", 0)
    with caplog.at_level("INFO", logger=media_tag.LOG.name):
        t.run(stop)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("media tag:")]
    assert lines == [
        "media tag: ModuleNotFoundError: No module named 'x' (pass 0)",
        "media tag: working again after 5 failed passes",
        "media tag: RuntimeError: other",
        "media tag: working again after 1 failed pass",
    ]
