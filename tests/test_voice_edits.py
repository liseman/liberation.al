"""Voice edits and explicit recording enrollment, including confirmed reference windows."""

import json

import numpy as np
import pytest
import soundfile as sf

from liberational import knowledge, segment_log
from liberational.listen import voices
from liberational.people import registry, voice_edits
from voices import write_voices


def test_naming_a_voice_makes_it_read_as_the_person(tmp_path):
    write_voices(tmp_path, {2: None})
    assert voice_edits.label(tmp_path, 2, "Dana") == "Dana"
    assert registry.voice_names(tmp_path, [2]) == {2: "Dana"}


def test_a_person_may_have_several_voices(tmp_path):
    write_voices(tmp_path, {1: "Dana", 2: None})
    voice_edits.label(tmp_path, 2, "Dana")
    assert registry.voice_names(tmp_path, [1, 2]) == {1: "Dana", 2: "Dana"}


def test_naming_a_voice_back_to_its_own_label_withdraws_its_person(tmp_path):
    write_voices(tmp_path, {2: "Dana"})
    assert voice_edits.label(tmp_path, 2, "Speaker 2") == "Speaker 2"
    assert registry.voice_names(tmp_path, [2]) == {2: None}


def test_a_voice_never_handed_out_cant_be_named(tmp_path):
    write_voices(tmp_path, {2: None})
    with pytest.raises(SystemExit, match="no voice 3"):
        voice_edits.label(tmp_path, 3, "Dana")


def file_print(state, n):
    m = voices.VoiceMatcher(
        voice_edits.voiceprints(state),
        voices.RegistryPeople(state),
        voice_edits._NoSearch(),
        segments=segment_log.root(state),
    )
    m.file(np.eye(segment_log.VOICE_DIM, dtype=np.float32)[0], n, None, 10**18, 10**18)
    m.close()


def test_forgetting_a_voice_withdraws_its_person_and_forgets_its_prints(tmp_path):
    write_voices(tmp_path, {2: "Dana", 3: None})
    file_print(tmp_path, 2)
    file_print(tmp_path, 3)
    assert voice_edits.forget(tmp_path, 2)
    assert registry.voice_names(tmp_path, [2]) == {2: None}
    assert [p for p, _ in stored_prints(tmp_path, 2)] == []
    assert len(stored_prints(tmp_path, 3)) == 1, "another voice's are kept"


def test_forgetting_a_voice_never_handed_out_does_nothing(tmp_path):
    write_voices(tmp_path, {2: None})
    file_print(tmp_path, 2)
    assert not voice_edits.forget(tmp_path, 7)
    assert len(stored_prints(tmp_path, 2)) == 1


def test_the_command_line_prints_what_the_dashboard_reads(tmp_path, capsys):
    write_voices(tmp_path, {2: None})
    voice_edits.main(["--state-dir", str(tmp_path), "label", "2", "Dana"])
    assert capsys.readouterr().out == "2 -> Dana\n"


def stored_prints(state, n):
    """The embeddings in `state`'s segment store under voiceprint n, as (id, person)."""
    rows = segment_log.Log(segment_log.root(state)).range(voices.EMBEDDINGS, 0.0, 2.0**32)
    return [(r["id"], r["person"]) for r in rows if r["voiceprint"] == voices.subject(n)]


@pytest.fixture
def enrolling(tmp_path):
    """enroll(name, clone=False) into the test's segment store, the voice model stood in for."""
    wav = tmp_path / "ref.wav"
    sf.write(wav, np.zeros(16000, np.float32), 16000)
    return lambda name, clone=False: voice_edits.enroll(
        tmp_path, name, str(wav), clone=clone, embed=lambda audio: np.eye(segment_log.VOICE_DIM, dtype=np.float32)[0]
    )


def test_recording_a_reference_again_retires_the_old_one(tmp_path, enrolling):
    # A bad first reference (someone else caught in it) must not keep matching the voice.
    write_voices(tmp_path, {1: None})
    n = enrolling("Alice")
    (first,) = [pid for pid, _ in stored_prints(tmp_path, n)]
    assert enrolling("Alice") == n
    (second,) = [pid for pid, _ in stored_prints(tmp_path, n)]
    assert second != first


def test_a_clone_is_filed_under_its_own_voiceprint_alone_even_once_named(tmp_path, enrolling):
    # The vector index keys on the person: under the person's, the clone's recording would be their voice.
    write_voices(tmp_path, {1: "Alice"})
    n = enrolling("Alice", clone=True)
    enrolling("Alice", clone=True)  # uploaded again, when the clone already reads as Alice
    assert [person for _, person in stored_prints(tmp_path, n)] == [None]


def test_recording_enrollment_saves_distinct_confirmed_windows(tmp_path):
    wav = tmp_path / "reference.wav"
    audio = np.repeat(np.array([-0.25, 0.0, 0.25], np.float32), 3 * 16000)
    sf.write(wav, audio, 16000, subtype="FLOAT")
    original = wav.read_bytes()

    def embed(window):
        vector = np.zeros(segment_log.VOICE_DIM, np.float32)
        vector[0] = 1.0
        vector[1] = float(window.mean())
        return vector

    n = voice_edits.enroll(tmp_path, "Alice", str(wav), embed=embed)
    stored = knowledge.Store(registry.knowledge_path(tmp_path)).query(voices.REFERENCES)
    records = [r.record for r in stored]
    assert len(records) == 3
    assert [r.value["vector"][1] for r in records] == [-0.25, 0.0, 0.25]
    assert [r.end_ns - r.start_ns for r in records] == [3_000_000_000] * 3
    assert [r.start_ns - records[0].start_ns for r in records] == [0, 3_000_000_000, 6_000_000_000]
    assert len({r.value["source_id"] for r in records}) == 3
    enrollment = json.loads((tmp_path / voice_edits.ENROLLMENTS).read_text())[str(n)]
    person = str(registry.at(tmp_path).person_id("Alice"))
    assert [r.key for r in records] == [f"speaker:{n}"] * 3
    assert [r.subjects for r in records] == [(person, voices.subject(n))] * 3
    assert [
        (
            r.value["voiceprint"],
            r.value["person"],
            r.value["dims"],
            r.value["model"],
            r.value["version"],
            r.value["source_kind"],
            r.value["enrollment_id"],
        )
        for r in records
    ] == [(n, person, segment_log.VOICE_DIM, voices.MODEL, voices.VERSION, "voice.enrollment", enrollment)] * 3
    assert len(stored_prints(tmp_path, n)) == 1
    references = voices.confirmed_references(tmp_path)
    assert len(references) == 3
    assert voices.calibrate(references)[person]["references"] == 1
    assert not voices.calibrate(references)[person]["eligible"]
    assert wav.read_bytes() == original


@pytest.mark.parametrize("seconds, expected", [(0.5, []), (1.0, [1.0]), (3.5, [3.0]), (4.0, [3.0, 1.0])])
def test_enrollment_windows_require_one_second_without_copying_short_clips(tmp_path, seconds, expected):
    wav = tmp_path / "reference.wav"
    sf.write(wav, np.zeros(int(seconds * 16000), np.float32), 16000)
    n = voice_edits.enroll(
        tmp_path, "Alice", str(wav), embed=lambda audio: np.eye(segment_log.VOICE_DIM, dtype=np.float32)[0]
    )
    records = [r.record for r in knowledge.Store(registry.knowledge_path(tmp_path)).query(voices.REFERENCES)]
    assert [(r.end_ns - r.start_ns) / 1e9 for r in records] == expected
    assert len(stored_prints(tmp_path, n)) == 1


def test_replaced_enrollment_windows_stop_teaching_and_clones_never_teach(tmp_path, enrolling):
    n = enrolling("Alice")
    first = voices.confirmed_references(tmp_path)
    assert len(first) == 1
    assert not voices.calibrate(first)[first[0].person]["eligible"]
    assert enrolling("Alice") == n
    second = voices.confirmed_references(tmp_path)
    assert len(second) == 1
    assert second[0].key != first[0].key
    enrolling("Alice", clone=True)
    assert [r.key for r in voices.confirmed_references(tmp_path)] == [second[0].key]
