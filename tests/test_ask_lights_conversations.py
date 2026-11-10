"""Lights from colour cast, and conversation summaries with media set aside (2026-10-29)."""

import re

from liberational.memory import conversation_summary as cs
from liberational.memory import lights_color as lc


def _rows(spec):
    return [{"ts": i * 300.0, "mean": m, "gap": g} for i, (m, g) in enumerate(spec)]


def test_warm_cast_is_lamps_neutral_bright_is_daylight_neutral_dim_is_dark():
    night = [(90, -3)] * 10
    lamps = [(108, -19)] * 10
    day = [(122, -2)] * 5
    labels = lc.classify(_rows(night + lamps + day))
    assert labels[:10] == ["dark"] * 10
    assert labels[10:20] == ["on"] * 10
    assert labels[20:] == ["daylight"] * 5


def test_a_day_without_a_warm_cast_never_reports_lamps():
    labels = lc.classify(_rows([(90, -3)] * 10 + [(120, -4)] * 10))
    assert "on" not in labels


def test_spans_join_adjacent_slots():
    slots = [(0.0, "on"), (300.0, "on"), (600.0, "dark"), (900.0, "on")]
    assert lc._spans(slots, "on", 300) == [(0.0, 600.0), (900.0, 1200.0)]


def _line(ts, speaker, text):
    return {"ts_start": ts, "speaker": speaker, "text": text}


def test_media_is_set_aside_and_conversations_ranked_by_time(tmp_path, monkeypatch, speech):
    rows = [_line(100 + i * 30, s, "hi") for i, s in enumerate(["Luca", "Kara", "Luca", "Kara"])]
    rows += [_line(5000 + i * 60, s, "and so the khan") for i, s in enumerate(["Luca", "Kara"] * 15)]
    for row in rows:
        speech.say(row)
    verdicts = iter(
        [
            {"kind": "CONVERSATION", "topic": "Fixing the router", "summary": "They reset the router."},
            {"kind": "MEDIA", "topic": "Genghis Khan", "summary": "A history audiobook."},
        ]
    )
    monkeypatch.setattr(cs, "_llm", lambda *a, **k: next(verdicts))
    res = cs.analyse(tmp_path, ["Luca", "Kara"], 0, 10_000)
    text = cs.answer_text(res, "yesterday")
    assert text.startswith("Luca and Kara talked most about Fixing the router")
    assert "Not counted" in text and "Genghis Khan" in text
    # Spoken answers: a stretch of time reads "from to", never a dash a voice would skip or misread
    assert re.search(r"\(\d+ min, \d\d:\d\d to \d\d:\d\d\)", text)
    assert re.search(r"\d\d:\d\d to \d\d:\d\d \(Genghis Khan\)", text)


def test_the_lights_answer_says_each_stretch_from_to():
    res = {"measured_s": 3600, "window_s": 3600, "on_s": 600, "on": [(0.0, 600.0)], "daylight": [(1200.0, 1800.0)]}
    text = lc.answer_text(res)
    assert f"{lc._clock(0.0)} to {lc._clock(600.0)}." in text
    assert f"lit by daylight {lc._clock(1200.0)} to {lc._clock(1800.0)};" in text


def test_a_stretch_where_only_one_named_person_speaks_is_not_their_conversation(tmp_path, speech):
    for i in range(6):
        speech.say(_line(100 + i * 30, "Luca", "hi"))
    assert cs.analyse(tmp_path, ["Luca", "Kara"], 0, 10_000)["conversations"] == []
