"""Whose voice a clip is (voice_lookup.py): the nearest stored voice embedding, resolved to its voiceprint, named
through the registry, with the audio worker's thresholds. The vector worker's hits are given here; no model runs."""

import json

import numpy as np
import pytest

from liberational import segment_log
from liberational.listen import voices
from liberational.people import registry
from liberational.voice_lookup import VoiceLookup
from voices import write_voices


@pytest.fixture
def lookup(tmp_path):
    """Voiceprint 4 is Dana, voiceprint 5 names nobody yet. The audio worker has handed out both
    (state/voiceprints.json), so both are known even though only one has a claim in the registry."""
    write_voices(tmp_path, {4: "Dana", 5: None})
    (tmp_path / "voiceprints.json").write_text(json.dumps({"next": 6}))
    return VoiceLookup(tmp_path, embed=lambda audio: audio)


def hit(entity, sim):
    return {"entity_id": entity, "score": 1.0 - sim}


def confirmed(lookup):
    person = str(registry.at(lookup.state).person_of(registry.voiceprint(4)).person)
    vector = np.eye(segment_log.VOICE_DIM, dtype=np.float32)[0]
    lookup.references = lambda: [voices.Reference(person, 4, str(i), vector) for i in range(3)]
    return vector


def test_a_named_index_hit_does_not_identify_without_confirmed_references(lookup):
    assert lookup.judge([hit(voices.subject(4), 1.0)], seconds=2.0).name is None


def test_a_clip_is_named_only_from_confirmed_references(lookup):
    vector = confirmed(lookup)
    heard = lookup.judge([hit(voices.subject(4), 0.8)], seconds=2.0, embedding=vector)
    assert (heard.number, heard.name) == (4, "Dana")
    assert heard.sim == pytest.approx(1.0)


def test_a_voice_that_names_nobody_is_its_speaker_number(lookup):
    assert lookup.judge([hit(voices.subject(5), 0.7)], seconds=2.0).name == "Speaker 5"


def test_a_renamed_voice_names_the_clip_by_the_new_name(lookup, tmp_path):
    registry.at(tmp_path).rename("Dana", "Dee", "test")  # in the event log only
    assert lookup.judge([hit(voices.subject(4), 0.8)], seconds=2.0, embedding=confirmed(lookup)).name == "Dee"


def test_a_persons_embedding_is_their_voiceprint(lookup, tmp_path):
    dana = registry.at(tmp_path).person_of(registry.voiceprint(4)).person
    assert lookup.judge([hit(str(dana), 0.8)], seconds=2.0, embedding=confirmed(lookup)).number == 4


@pytest.mark.parametrize(
    ("sim", "seconds", "number"),
    [
        pytest.param(0.46, 2.0, 5, id="a-second-or-more-at-045"),
        pytest.param(0.44, 2.0, None, id="under-045"),
        pytest.param(0.50, 0.5, None, id="a-short-clip-needs-055"),
        pytest.param(0.56, 0.5, 5, id="a-short-clip-at-055"),
    ],
)
def test_too_far_from_every_voice_is_nobody(lookup, sim, seconds, number):
    heard = lookup.judge([hit(voices.subject(5), sim)], seconds=seconds)
    assert heard.number == number
    assert heard.sim == pytest.approx(sim)


def test_hits_no_voiceprint_holds_are_passed_over(lookup):
    heard = lookup.judge([hit("not-a-voice", 0.99), hit(voices.subject(5), 0.7)], seconds=2.0)
    assert heard.number == 5
    assert lookup.judge([], seconds=2.0).number is None


def test_a_voice_clone_never_names_a_clip(lookup, tmp_path):
    # v1 never matched room speech to a clone (otherwise room lines were filed under a clone's name).
    (tmp_path / "clones").mkdir()
    (tmp_path / "clones" / "4.json").write_text(json.dumps({"source": "recorded"}))
    assert lookup.judge([hit(voices.subject(4), 0.9), hit(voices.subject(5), 0.7)], seconds=2.0).number == 5
