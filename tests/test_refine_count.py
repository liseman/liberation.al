"""The "Lines to check" list in People: a page of uncertain lines plus how many there are in all."""

from liberational.dashboard import context, inbox_state, speaker_lines
from liberational.people import voice_edits
from speech_fake import fake
from voices import write_voices


def _line(i, sid, sim):
    return {"ts_start": 1000.0 + i, "speaker_id": sid, "speaker_sim": sim, "text": f"line {i}", "audio": {"file": "a"}}


def test_counts_every_uncertain_line_but_sends_one_page(tmp_path, monkeypatch):
    write_voices(tmp_path, {1: "Sam"}, source=voice_edits.ENROLL_SOURCE)  # Sam's voice, learned from a recording
    write_voices(tmp_path, {2: None})
    lines = [_line(i, 1, 0.5) for i in range(5)]  # uncertain Sam lines
    lines += [_line(900, 1, 0.9), _line(901, 2, 0.4)]  # a confident one, and one from a voice that isn't enrolled
    fake().write(lines)
    monkeypatch.setattr(speaker_lines, "REFINE_PAGE", 3)
    monkeypatch.setattr(inbox_state, "INBOX_SKIPS", tmp_path / "inbox_skips.json")
    monkeypatch.setattr(context, "STATE", tmp_path)
    d = speaker_lines.speakers_refine_list()
    assert d["remaining"] == 5
    assert [e["ts_start"] for e in d["items"]] == [1000.0, 1001.0, 1002.0]
    assert d["items"][0]["sim_hint"] == "50% match to Sam"
