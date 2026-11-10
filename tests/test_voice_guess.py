""" "Who's this?" rows for unnamed voices: people/voice_guess.py and GET /voices/unknown/groups."""

import json
import math
import threading
from unittest.mock import patch

import numpy as np
import pytest

from liberational.dashboard import context, inbox_state, known_voices, people_reads, unnamed_groups
from liberational.people import registry
from liberational.people import voice_guess as G
from speech_fake import fake
from voices import write_voices


def at(sim, axis=1):
    """A 4-d voiceprint whose similarity with [1, 0, 0, 0] is sim."""
    v = [0.0] * 4
    v[0], v[axis] = sim, math.sqrt(1 - sim * sim)
    return v


E1 = [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(("sim", "expected"), [(0.25, 0), (0.45, 50), (0.65, 100), (0.1, 0)])
def test_a_similarity_becomes_a_likeness_with_the_voice_threshold_at_50(sim, expected):
    assert G.likeness(sim) == expected


@pytest.mark.parametrize(
    ("voices", "guess"),
    [
        pytest.param(
            [("Kara", 0.61), ("Kara", 0.2), ("Luca", 0.4)],
            {"name": "Kara", "score": 90, "band": "probable", "close": None},
            id="probable-by-the-closest-voice",
        ),
        pytest.param(
            [("Kara", 0.61), ("Luca", 0.60)],
            {"name": "Kara", "score": 56, "band": "maybe", "close": "Luca"},
            id="a-close-runner-up-holds-the-score",
        ),
        pytest.param(
            [("Luca", 0.35)], {"name": "Luca", "score": 25, "band": "none", "close": None}, id="no-good-guess"
        ),
    ],
)
def test_the_best_guess_is_the_closest_named_voice(voices, guess):
    names = [n for n, _ in voices]
    cents = [at(s, axis=k % 3 + 1) for k, (_, s) in enumerate(voices)]
    assert G.best_guess(E1, names, cents) == guess


@pytest.mark.parametrize(
    ("centroid", "names", "cents"),
    [
        pytest.param(E1, [], [], id="nobody-named"),
        pytest.param([0.0] * 4, ["Kara"], [E1], id="no-voiceprint-yet"),
        pytest.param(E1, ["Kara"], [[1.0, 0.0]], id="voiceprints-of-another-size"),
        pytest.param(None, ["Kara"], [E1], id="no-centroid"),
    ],
)
def test_there_is_no_guess_without_something_to_compare(centroid, names, cents):
    assert G.best_guess(centroid, names, cents) is None


def test_a_named_voiceprint_of_another_size_doesnt_spoil_the_others():
    guess = G.best_guess(E1, ["Luca", "Kara"], [[1.0, 0.0], at(0.61)])
    assert guess is not None
    assert (guess["name"], guess["band"]) == ("Kara", "probable")


PROBABLE = {"name": "Kara", "score": 90, "band": "probable", "close": None}


@pytest.mark.parametrize(
    ("guess", "count", "dur", "match", "expected"),
    [
        pytest.param(PROBABLE, 40, 3.0, 0.53, "A clear voice, close to Kara's", id="clear"),
        pytest.param({**PROBABLE, "band": "maybe", "score": 56, "close": "Luca"}, 40, 3.0, 0.8,
                     "Sounds like Luca too", id="held-by-the-runner-up"),
        pytest.param(PROBABLE, 40, 3.0, 0.51, "Its lines don't all sound alike", id="mixed"),
        pytest.param(PROBABLE, 2, 1.5, 0.8, "Only 2 lines heard and short clips", id="little-to-go-on"),
        pytest.param(PROBABLE, 1, None, None, "Only 1 line heard", id="one-line"),
        pytest.param(None, 40, 3.0, 0.8, "Nobody is named yet, so there's no one to compare with", id="nobody"),
    ],
)  # fmt: skip
def test_the_reason_names_what_holds_the_guess_back(guess, count, dur, match, expected):
    assert G.reason(guess, count, dur, match, anyone=guess is not None) == expected


def test_a_voice_with_no_voiceprint_says_so_when_people_are_named():
    assert G.reason(None, 40, 3.0, 0.8, anyone=True) == "No voiceprint to compare yet"


@pytest.fixture
def prints(monkeypatch):
    """Each voice's stored embedding, as the broker would hand it to the dashboard (listen/voices.py prints)."""
    prints = {1: at(0.61), 2: E1, 3: [0.0, 0.0, 1.0, 0.0], 4: E1, 5: E1}
    monkeypatch.setattr(
        unnamed_groups, "_voice_prints", lambda ns: {n: [np.asarray(prints[n], np.float32)] for n in ns if n in prints}
    )
    return prints


@pytest.fixture
def state(tmp_path, monkeypatch, prints):
    """Kara named (voice 1); voice 2 sounds like her (five lines, two left to play), voice 3 like nobody, voice 4 was
    put aside, and voice 5 has no line left to play."""
    write_voices(tmp_path, {1: "Kara", 2: None, 3: None, 4: None, 5: None})
    audio = {"stream": "mic.audio", "start_ns": 0, "end_ns": 3000000000}
    lines = [
        {"ts_start": 90.5, "speaker_id": 2, "text": "One."},
        {"ts_start": 91.5, "speaker_id": 2, "text": "Two."},
        {"ts_start": 92.5, "speaker_id": 2, "text": "Three."},
        {"ts_start": 100.5, "speaker_id": 2, "speaker_sim": 0.8, "text": "Morning.", "audio": audio},
        {"ts_start": 101.5, "speaker_id": 2, "speaker_sim": 0.7, "text": "Coffee?", "audio": audio},
        {
            "ts_start": 102.5,
            "speaker_id": 3,
            "speaker_sim": 0.5,
            "text": "Hi.",
            "audio": {**audio, "end_ns": 1_200_000_000},
        },
        {"ts_start": 103.5, "speaker_id": 4, "speaker_sim": 0.9, "text": "Aside.", "audio": audio},
    ]
    fake().write(lines)
    monkeypatch.setattr(context, "STATE", tmp_path)
    monkeypatch.setattr(inbox_state, "INBOX_SKIPS", tmp_path / "inbox_skips.json")
    monkeypatch.setattr(unnamed_groups._VOICE_GROUPS, "value", None)  # nothing kept from another test
    inbox_state.inbox_skip("voice", 4, True)
    yield tmp_path
    settle()  # a background refresh must not outlive the state it reads


def settle():
    """Wait for the kept voice rows' background refresh, so the next read gets the full answer."""
    for thread in threading.enumerate():
        if thread.name == "fresh-voice groups":
            thread.join(timeout=30)


def test_unnamed_voices_come_best_guess_first_with_their_clips(state):
    code, body = people_reads.unknown_voices_view()
    assert (code, body["count"], body["skipped"]) == (200, 2, [4])
    two, three = body["groups"]
    assert (two["label"], two["guess"]["name"], two["band"], two["score"]) == ("Speaker 2", "Kara", "probable", 90)
    assert two["guess"]["id"] == "Kara", "the name to send back, as faces' and pets' guesses"
    assert [c["ts"] for c in two["clips"]] == [101.5, 100.5]  # the less certain line first: it teaches more
    assert two["clips"][0] == {
        "ts": 101.5,
        "line": "101.500",
        "text": "Coffee?",
        "dur": 3.0,
        "url": "/clip?line=101.500",
    }
    assert two["reason"] == "A clear voice, close to Kara's"
    assert (three["guess"], three["band"]) == (None, "none")
    assert three["reason"] == "Its lines don't all sound alike and only 1 line heard"


def test_two_microphones_lines_in_one_millisecond_are_two_clips_by_key(state):
    audio = {"stream": "mic.audio", "start_ns": 0, "end_ns": 3000000000}
    twins = [
        {
            "ts_start": 110.25,
            "device": "Kitchen Mic",
            "speaker_id": 3,
            "speaker_sim": 0.6,
            "text": "Tea?",
            "audio": audio,
        },
        {
            "ts_start": 110.25,
            "device": "Hall",
            "speaker_id": 3,
            "speaker_sim": 0.61,
            "text": "Yes please.",
            "audio": audio,
        },
    ]
    for e in twins:
        fake().say(e)
    three = next(g for g in people_reads.unknown_voices_view()[1]["groups"] if g["id"] == 3)
    twin = {c["line"]: c["url"] for c in three["clips"] if c["ts"] == 110.25}
    assert twin == {
        "kitchen-mic@110.250": "/clip?line=kitchen-mic%40110.250",
        "hall@110.250": "/clip?line=hall%40110.250",
    }


def test_the_route_says_when_the_voices_cannot_be_read(state, monkeypatch):
    def broken():
        raise ValueError("a half-written file")

    monkeypatch.setattr(known_voices, "_speakers", broken)
    code, body = people_reads.unknown_voices_view()
    assert (code, body) == (500, {"error": "Could not read the voices waiting for a name."})


def test_a_voice_whose_lines_are_all_too_short_to_play_isnt_listed(state):
    write_voices(state, {6: None})
    fake().say(
        {
            "ts_start": 104.5,
            "speaker_id": 6,
            "text": "Yeah.",
            "audio": {"stream": "mic.audio", "start_ns": 0, "end_ns": 500000000},
        }
    )
    assert unnamed_groups.speakers_with_audio() >= {2, 3, 6}  # it has a line with audio; it's only too short to play
    assert [g["id"] for g in people_reads.unknown_voices_view()[1]["groups"]] == [2, 3]


def test_a_named_voice_counts_by_its_refs():
    kara = {"label": "Kara", "centroid": [0.0, 0.0, 1.0, 0.0], "refs": [at(0.61)]}  # one of her stored embeddings
    voice = {"id": 2, "label": "Speaker 2", "centroid": E1, "n": 12, "last_seen": 1}
    clip = {"id": 2, "ts": 1.0, "line": "1.000", "text": "Hi.", "dur": 3.0, "sim": 0.8}
    (row,) = G.rows([voice], [kara], [clip])
    assert (row["guess"]["name"], row["band"]) == ("Kara", "probable")


def test_a_voice_clone_made_from_a_recording_is_never_a_guess(state, prints):
    prints.update({1: [0.0, 0.0, 1.0, 0.0], 7: E1})  # Kara sounds nothing like voice 2
    write_voices(state, {7: "Star"})
    (state / "clones").mkdir()
    (state / "clones" / "7.json").write_text(json.dumps({"label": "Star", "source": "recorded"}))
    two = next(g for g in people_reads.unknown_voices_view()[1]["groups"] if g["id"] == 2)
    assert two["guess"] is None  # Star sounds exactly like it, but Star is a clone


def test_the_rows_are_kept_until_something_they_read_changes(state):
    with patch.object(
        unnamed_groups, "speakers_with_audio", autospec=True, side_effect=unnamed_groups.speakers_with_audio
    ) as audible:
        first = people_reads.unknown_voices_view()
        assert people_reads.unknown_voices_view() == first
        assert audible.call_count == 1
        inbox_state.inbox_skip("voice", 3, True)
        patched = people_reads.unknown_voices_view()[1]
        assert ([g["id"] for g in patched["groups"]], patched["refreshing"]) == ([2], True)
        settle()
        assert audible.call_count == 2
        full = people_reads.unknown_voices_view()[1]
        assert ([g["id"] for g in full["groups"]], "refreshing" in full) == ([2], False)


def test_a_voice_heard_while_a_named_face_was_in_view_suggests_that_name(state, vision):
    """Voice 3 speaks three times while Kara's face is in view."""
    audio = {"stream": "mic.audio", "start_ns": 0, "end_ns": 3000000000}
    for ts in (500.5, 1000.5):
        fake().say(
            {"ts_start": ts, "ts_end": ts + 3, "speaker_id": 3, "speaker_sim": 0.5, "text": "Hi.", "audio": audio}
        )
    for ts in (103.0, 501.0, 1002.0, 2000.0):  # the last is far from any line
        vision.face(ts, "cam0", "Kara", 0.7)
    three = next(g for g in people_reads.unknown_voices_view()[1]["groups"] if g["id"] == 3)
    assert three["together"] is not None
    assert (three["together"]["name"], three["together"]["moments"]) == ("Kara", 3)
    two = next(g for g in people_reads.unknown_voices_view()[1]["groups"] if g["id"] == 2)
    assert two["together"] is None  # nobody named was in view while voice 2 spoke


def test_voice_recomputation_reuses_retained_speech_instead_of_reading_three_times(state):
    log = fake()
    with patch.object(log, "all_lines", autospec=True, side_effect=log.all_lines) as reads:
        first = unnamed_groups._unknown_voices()
        second = unnamed_groups._unknown_voices()
        assert reads.call_count == 1
    assert first == second
    assert first[0] == 200
    assert [g["id"] for g in first[1]["groups"]] == [2, 3]


def test_kept_voice_groups_refresh_identity_names_and_drop_purged_audio_immediately(state):
    assert people_reads.unknown_voices_view()[1]["groups"][0]["guess"]["name"] == "Kara"
    registry.at(state).rename("Kara", "Katherine", "test")
    assert (
        people_reads.unknown_voices_view()[1]["refreshing"] is True
    )  # the kept rows, while the guesses are worked out
    settle()
    renamed = people_reads.unknown_voices_view()
    assert renamed[1]["groups"][0]["guess"]["name"] == "Katherine"
    fake().purge("speech.utterance", lambda e: e.get("speaker_id") == 3)
    assert [g["id"] for g in people_reads.unknown_voices_view()[1]["groups"]] == [2]


def test_voice_recomputation_reads_new_speech_after_warming_retained_rows(state):
    assert [g["id"] for g in unnamed_groups._unknown_voices()[1]["groups"]] == [2, 3]
    fake().say(
        {
            "ts_start": 110.5,
            "speaker_id": 5,
            "speaker_sim": 0.7,
            "text": "A new voice sample.",
            "audio": {"stream": "mic.audio", "start_ns": 0, "end_ns": 3_000_000_000},
        }
    )
    assert {g["id"] for g in unnamed_groups._unknown_voices()[1]["groups"]} == {2, 3, 5}


def test_audio_availability_and_newest_samples_do_not_borrow_another_microphones_label(state):
    from liberational.domain.identity import Basis

    audio = {"stream": "mic.audio", "start_ns": 0, "end_ns": 3_000_000_000}
    fake().write(
        [
            {
                "ts_start": 120.0,
                "ts_end": 123.0,
                "device": "Desk",
                "text": "The same speech.",
                "speaker_id": 5,
                "speaker_sim": 0.7,
                "audio": audio,
            },
            {
                "ts_start": 120.4,
                "ts_end": 123.4,
                "device": "Hall",
                "text": "The same speech.",
                "speaker_id": 5,
                "speaker_sim": 0.7,
                "audio": audio,
            },
        ]
    )
    registry.at(state).claim(registry.utterance(120.0, "Desk"), "Dana", Basis.OWNER_TAG, "test")
    assert unnamed_groups.speakers_with_audio() == {5}
    assert people_reads._newest_lines()[5]["device"] == "Hall"
