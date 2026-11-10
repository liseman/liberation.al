"""The assistant's spoken replies are logged (assistant/voice.py) and shown in the dashboard's live transcript."""

from liberational.dashboard import server
from speech_fake import fake


def test_live_transcript_merges_replies_in_time_order(tmp_path, monkeypatch):
    fake().write([{"ts_start": t, "text": f"heard {t}"} for t in (1.0, 3.0, 5.0)])
    [fake().reply(x) for x in [{"ts_start": 4.0, "text": "said 4", "assistant": True}]]
    entries = server.merged_tail(3)
    assert [e["ts_start"] for e in entries] == [3.0, 4.0, 5.0]  # newest three, replies interleaved
    assert entries[1]["assistant"] is True
