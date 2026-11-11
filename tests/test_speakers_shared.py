"""The voice list is worked out once per change and shared by every page's event stream and GET /speakers."""

import threading
import time
from types import SimpleNamespace

import pytest

from liberational.dashboard import known_voices, speaker_lines


@pytest.fixture
def voices(monkeypatch):
    """A counted `_speakers`, and a key a test moves."""
    state = SimpleNamespace(calls=0, key=("mark", 1))

    def speakers():
        state.calls += 1
        time.sleep(0.01)  # long enough that callers arriving together overlap it
        return {
            "speakers": [
                {"id": 4, "label": "Dana", "last_seen": 2.0},
                {"id": 9, "label": "Speaker 9", "last_seen": 5.0},
            ],
            "merged": {},
        }

    monkeypatch.setattr(known_voices, "_speakers", speakers)
    monkeypatch.setattr(speaker_lines, "_speakers_key", lambda: state.key)
    return state


def test_the_voice_list_is_worked_out_once_while_nothing_it_reads_changed(voices):
    first = speaker_lines.speakers_public()
    assert [s["id"] for s in first["speakers"]] == [9, 4], "most recently heard first"
    assert speaker_lines.speakers_public() == first
    assert voices.calls == 1


def test_a_label_or_a_new_line_works_it_out_again(voices):
    speaker_lines.speakers_public()
    voices.key = ("mark", 2)
    speaker_lines.speakers_public()
    assert voices.calls == 2


def test_each_caller_gets_its_own_copy(voices):
    first = speaker_lines.speakers_public()
    first["speakers"][0]["label"] = "changed by a caller"
    assert speaker_lines.speakers_public()["speakers"][0]["label"] == "Speaker 9"


def test_a_dozen_streams_asking_at_once_share_one_working_out(voices):
    got = []
    threads = [threading.Thread(target=lambda: got.append(speaker_lines.speakers_public())) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 12 and all(g == got[0] for g in got)
    assert voices.calls == 1


def test_every_live_feed_shares_one_log_mark_read_within_the_cadence(monkeypatch):
    """The live feed asked for the mark twice a second per open page, and each read takes the registry's lock and a
    knowledge-store transaction. Feeds open together now share one answer within LOG_MARK_S."""
    from liberational.dashboard import shell_status

    clock = [100.0]
    monkeypatch.setattr(shell_status.time, "monotonic", lambda: clock[0])
    reads = []

    def mark():
        reads.append(clock[0])
        return (len(reads),)

    monkeypatch.setattr(speaker_lines, "log_mark", mark)
    shell_status._kept.clear()
    assert speaker_lines.log_mark_shared() == (1,)
    clock[0] = 101.5
    assert speaker_lines.log_mark_shared() == (1,), "a second feed's read, inside the cadence, is the one kept"
    clock[0] = 100.0 + speaker_lines.LOG_MARK_S
    assert speaker_lines.log_mark_shared() == (2,)
    assert len(reads) == 2
