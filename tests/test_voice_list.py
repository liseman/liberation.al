"""The voice list on Continuum (people/voice_list.py): each voiceprint with how often and when it was heard,
counted from the speech log's lines."""

import pytest

from liberational import speech_log
from liberational.people import voice_list
from speech_fake import FakeSpeechLog


def heard(sid, ts):
    return {"ts_start": ts, "ts_end": ts + 2, "text": "hello there", "speaker_id": sid, "speaker": f"Speaker {sid}"}


def test_each_voice_counts_its_lines_and_when_it_was_first_and_last_heard():
    tally = voice_list.Tally()
    for line in (heard(3, 50.0), heard(1, 10.0), heard(3, 20.0), heard(3, 40.0)):
        tally.add(line)
    assert tally.rows() == [
        {"id": 1, "n": 1, "first_seen": 10.0, "last_seen": 10.0},
        {"id": 3, "n": 3, "first_seen": 20.0, "last_seen": 50.0},
    ]


@pytest.mark.parametrize(
    "line",
    [
        pytest.param({"ts_start": 1.0, "speaker_id": None}, id="unmatched"),
        pytest.param({"ts_start": 1.0}, id="no-voice"),
        pytest.param({"ts_start": 1.0, "speaker_id": True}, id="bool-id"),
        pytest.param({"ts_start": 1.0, "speaker_id": "3"}, id="string-id"),
        pytest.param({"ts_start": 1.0, "speaker_id": 0}, id="zero"),
        pytest.param({"ts_start": None, "speaker_id": 3}, id="no-time"),
    ],
)
def test_a_line_without_a_real_voiceprint_counts_for_no_voice(line):
    tally = voice_list.Tally()
    tally.add(line)
    assert tally.rows() == []


def test_a_voice_the_log_knows_with_no_line_yet_is_listed_unheard():
    tally = voice_list.Tally()
    tally.add(heard(2, 5.0))
    assert tally.rows(known=[4, 2]) == [
        {"id": 2, "n": 1, "first_seen": 5.0, "last_seen": 5.0},
        {"id": 4, "n": 0, "first_seen": None, "last_seen": None},
    ]


class Counting(FakeSpeechLog):
    """The fake log, remembering where each read started."""

    def __init__(self):
        super().__init__()
        self.reads: list[int] = []

    def since(self, after_seq, streams=None):
        self.reads.append(after_seq)
        return super().since(after_seq, streams)


def test_the_list_reads_only_the_lines_committed_since_it_last_read():
    log = Counting()
    voices = voice_list.VoiceList(lambda: log)
    log.say(heard(1, 10.0))
    assert [v["n"] for v in voices.voices()] == [1]
    log.say(heard(1, 20.0))
    log.say(heard(2, 30.0))
    assert [(v["id"], v["n"]) for v in voices.voices()] == [(1, 2), (2, 1)]
    assert log.reads == [0, 1]


def test_a_purge_recounts_from_the_start():
    log = FakeSpeechLog()
    voices = voice_list.VoiceList(lambda: log)
    log.say(heard(1, 10.0))
    log.say(heard(2, 20.0))
    voices.voices()
    log.purge(speech_log.UTTERANCES, lambda e: e["speaker_id"] == 2)  # someone forgotten
    assert [v["id"] for v in voices.voices()] == [1]


def test_a_different_log_is_read_afresh():
    first, second = FakeSpeechLog(), FakeSpeechLog()
    current = [first]
    voices = voice_list.VoiceList(lambda: current[0])
    first.say(heard(1, 10.0))
    voices.voices()
    current[0] = second
    second.say(heard(5, 10.0))
    assert [v["id"] for v in voices.voices()] == [5]
