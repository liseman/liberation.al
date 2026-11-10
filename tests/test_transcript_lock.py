"""A day's transcript/<day>.md is rebuilt under its lock, so a line the microphone service appends meanwhile is never
lost or doubled."""

import json
import threading
import time

from liberational import locking
from liberational.people import transcript_view
from voices import write_voices


def test_a_rebuild_waits_for_the_appender_and_keeps_its_line(tmp_path, monkeypatch):
    t0 = time.mktime((2026, 10, 20, 12, 0, 0, 0, 0, -1))
    old = {"ts_start": t0, "speaker": "Speaker 3", "speaker_id": 3, "speaker_sim": 0.4, "text": "hi"}
    new = {"ts_start": t0 + 5, "speaker": "Speaker 4", "speaker_id": 4, "speaker_sim": 0.9, "text": "late"}
    jsonl, md = tmp_path / "transcript.jsonl", tmp_path / "transcript" / "2026-10-20.md"
    md.parent.mkdir()
    jsonl.write_text(json.dumps(old) + "\n")
    md.write_text(transcript_view.md_line(old) + "\n")
    write_voices(tmp_path, {3: "Alice"})
    holding, asked = threading.Event(), threading.Event()

    def appender():
        """The microphone service: holds the day's lock and the transcript's while it appends to both. It finishes
        only once the rebuild asks for the day's lock; a rebuild that never does has already written the day file,
        and the line appended after that is doubled."""
        with locking.locked(md), locking.locked(jsonl):
            with jsonl.open("a") as f:
                f.write(json.dumps(new) + "\n")
            holding.set()
            asked.wait(5)
            with md.open("a") as f:
                f.write(transcript_view.md_line(new) + "\n")

    t = threading.Thread(target=appender)
    t.start()
    assert holding.wait(5)
    real = locking.locked

    def locked(path):
        if path == md:
            asked.set()  # the rebuild asks for the day's lock, and waits for the appender to let go
        return real(path)

    monkeypatch.setattr(locking, "locked", locked)
    assert transcript_view.rebuild_markdown(tmp_path, ["2026-10-20"]) == ["2026-10-20"]
    t.join()
    assert md.read_text().splitlines() == ["[12:00:00] Alice: hi", "[12:00:05] Speaker 4: late"]
