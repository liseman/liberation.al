"""Voices made from one uploaded recording (clones/<id>.json with source "recorded") stay apart from people: a person
who shares a clone's name keeps their own voice, and enrolling one never lands on the other (people/voice_edits.py)."""

import json

import numpy as np
import pytest
import soundfile as sf

from liberational import segment_log
from liberational.domain.identity import Basis
from liberational.people import registry, transcript_view, voice_edits
from voices import write_voices


def unit(*v):
    a = np.asarray(v, np.float32)
    return a / np.linalg.norm(a)


DANA, OTHER, CLIP = unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)
SILENCE = np.zeros(16000, np.float32)


@pytest.fixture
def enroll(tmp_path, monkeypatch):
    """enroll(name, clone=False): voice_edits.enroll on a one-second recording, with the voice model and the store
    stood in for. Returns (voiceprint, the voiceprint it filed under, and whether the clone mark was there then)."""
    wav = tmp_path / "ref.wav"
    sf.write(wav, SILENCE, 16000)
    filed = []

    def file(embedding, n, state, at, clone=False):
        filed.append((n, (state / "clones" / f"{n}.json").exists()))
        return segment_log.new_id()

    monkeypatch.setattr(voice_edits, "_file", file)

    def run(name, clone=False):
        n = voice_edits.enroll(
            tmp_path,
            name,
            str(wav),
            clone=clone,
            embed=lambda audio: np.eye(segment_log.VOICE_DIM, dtype=np.float32)[0],
        )
        return n, filed[-1]

    return run


def mark_clones(state, *ids, label=None):
    (state / "clones").mkdir(exist_ok=True)
    for i in ids:
        (state / "clones" / f"{i}.json").write_text(json.dumps({"source": "recorded", "label": label}))


def test_recording_a_clone_under_a_persons_name_keeps_their_room_voice(tmp_path, enroll):
    # Finding 13: the clone reused the person's voice by label, so their room voice became clone-only.
    write_voices(tmp_path, {3: "Dana"})
    n, (filed_under, marked) = enroll("Dana", clone=True)
    assert n != 3 and filed_under == n  # a voice of its own
    assert marked  # a clone from the moment anything is filed under it
    assert transcript_view.clone_only_ids(tmp_path) == frozenset({n})
    assert registry.voice_names(tmp_path, [3])[3] == "Dana"
    assert enroll("Dana", clone=True)[0] == n  # recording the clone again replaces that clone


def test_enrolling_a_person_by_name_skips_a_clone_of_the_same_name(tmp_path, enroll):
    # Finding 13, the other way round: enrolling a person by name reached the clone first.
    write_voices(tmp_path, {1: "Dana"}, source=voice_edits.ENROLL_SOURCE)  # the clone, enrolled under her name
    mark_clones(tmp_path, 1, label="Dana")
    n, _ = enroll("Dana")
    assert n != 1
    assert enroll("Dana")[0] == n  # her own enrolled voice, refreshed


def test_someone_new_may_take_a_clones_name(tmp_path, enroll):
    write_voices(tmp_path, {7: "Gus"})
    mark_clones(tmp_path, 7, label="Gus")
    n, _ = enroll("Gus")
    assert n not in (7,) and n not in transcript_view.clone_only_ids(tmp_path)
    assert enroll("Gus")[0] == n


def test_a_voice_tagged_line_reads_with_the_voice_tag_taught_not_a_clone(tmp_path):
    # Review 2, finding 5: a clone of Alice came first in the voice list, so a line tagged Alice read as the clone's.
    write_voices(tmp_path, {2: "Alice", 5: "Alice", 9: None})
    mark_clones(tmp_path, 2, label="Alice")
    real = {"id": 5}  # voice 2 is the clone
    reg = registry.at(tmp_path)
    tagged = {"ts_start": 1.0e9 + 5, "speaker": "Speaker 9", "speaker_id": 9, "speaker_sim": 0.5, "text": "hi"}
    reg.claim(registry.utterance(tagged["ts_start"]), "Alice", Basis.OWNER_TAG, transcript_view.VOICE_TAG)
    matched = {"ts_start": 1.0e9 + 9, "speaker": "Speaker 9", "speaker_id": 9, "speaker_sim": 0.5, "text": "yo"}
    reg.claim(
        registry.utterance(matched["ts_start"]),
        "Alice",
        Basis.BACKFILL_MATCH,
        "backfill",
        confidence=0.8,
        producer=registry.voice_producer(),
    )
    view = transcript_view.View(tmp_path)
    assert view.voice_of(reg.person_id("Alice")) == real["id"]
    got = view.lines([tagged, matched])
    assert [(e["speaker"], e["speaker_id"]) for e in got] == [("Alice", real["id"])] * 2
