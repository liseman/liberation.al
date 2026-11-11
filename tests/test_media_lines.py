"""Lines heard from a TV: marked, not dropped (listen/revisions.py), and left out of the readers that speak
for people (unknown voices, people counts, voice learning), with the owner's "This is TV" and its Undo."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from liberational import speech_log
from liberational.dashboard import context, people_list, speaker_lines, transcript_lines, unnamed_groups
from liberational.listen import revisions
from liberational.people import registry

T = 1_800_000_000.0


def line(ts: float, text: str, sid: int | None = 9, **kw) -> dict:
    return {
        "id": f"id-{ts}",
        "ts_start": ts,
        "ts_end": ts + 2,
        "device": "wave-3",
        "text": text,
        "speaker_id": sid,
        "speaker": f"Speaker {sid}",
        "audio": {"stream": "wave-3.audio", "start_ns": int(ts * 1e9), "end_ns": int((ts + 2) * 1e9)},
        **kw,
    }


def test_a_marked_line_stays_marked_and_undo_unmarks_it(tmp_path):
    rows = [line(T, "Tonight on the news"), line(T + 5, "Dinner's ready")]
    before = revisions.review_revision(tmp_path, T, T + 10)
    token = revisions.mark_media(tmp_path, rows[:1])
    assert revisions.review_revision(tmp_path, T, T + 10) > before, "every cache keyed on it refreshes"
    got = revisions.filter_lines(tmp_path, rows)
    assert [(r["text"], r.get("media")) for r in got] == [("Tonight on the news", True), ("Dinner's ready", None)]
    assert rows[0].get("media") is None, "the raw row is untouched"
    assert revisions.media_subjects(tmp_path) == {f"utterance:{registry.key_of(rows[0])}"}
    assert revisions.mark_media(tmp_path, rows[:1]) == token, "a retry is harmless"
    assert revisions.current_media(tmp_path, [registry.key_of(rows[0])]) == token
    assert revisions.undo_correction(tmp_path, token) == [registry.key_of(rows[0])]
    assert revisions.filter_lines(tmp_path, rows) == rows
    with pytest.raises(ValueError, match="already"):
        revisions.undo_correction(tmp_path, token)


def test_the_corrections_are_read_once_per_change_however_often_lines_are_read(tmp_path, monkeypatch):
    """The tagger writes a correction per TV line, and filter_lines and review_revision (polled by every
    Live page) read and parsed all of them on every call."""
    from liberational import knowledge

    rows = [line(T + i, f"line {i}") for i in range(20)]
    for row in rows[:10]:
        revisions.mark_media(tmp_path, [row], source="classifier")
    reads = []
    query = knowledge.Store.query
    monkeypatch.setattr(knowledge.Store, "query", lambda self, *a, **k: reads.append(a) or query(self, *a, **k))
    for _ in range(50):
        assert sum(1 for e in revisions.filter_lines(tmp_path, rows) if e.get("media")) == 10
        revisions.review_revision(tmp_path)
        revisions.review_revision(tmp_path, T, T + 30)
    assert len(reads) == 1
    token = revisions.mark_media(tmp_path, rows[10:11])  # a change: read again, once, and it shows
    marked = [e.get("media_by") for e in revisions.filter_lines(tmp_path, rows)]
    assert marked[10] == "owner" and len(reads) == 2  # the mark reads them before it writes; the next read after
    revisions.undo_correction(tmp_path, token)
    assert revisions.filter_lines(tmp_path, rows)[10].get("media") is None


def test_lines_marked_already_are_left_out_of_a_new_mark(tmp_path):
    rows = [line(T, "one"), line(T + 5, "two")]
    first = revisions.mark_media(tmp_path, rows[:1])
    token = revisions.mark_media(tmp_path, rows)
    assert token != first
    assert revisions.mark_media(tmp_path, rows[1:]) == token, "exactly the second mark's lines: a retry of it"
    with pytest.raises(ValueError, match="already"):  # both marked, by two marks: nothing new
        revisions.mark_media(tmp_path, rows)
    assert revisions.undo_correction(tmp_path, token) == [registry.key_of(rows[1])]
    assert [r.get("media") for r in revisions.filter_lines(tmp_path, rows)] == [True, None]


def test_a_rejected_line_is_hidden_even_when_marked(tmp_path):
    rows = [line(T, "garbled"), line(T + 5, "fine")]
    revisions.mark_media(tmp_path, rows[:1])
    revisions.reject_lines(tmp_path, rows[:1])
    assert [r["text"] for r in revisions.filter_lines(tmp_path, rows)] == ["fine"]


def test_only_the_owner_or_the_tagger_marks_a_line(tmp_path):
    with pytest.raises(ValueError, match="marked by"):
        revisions.mark_media(tmp_path, [line(T, "x")], source="someone")


def test_a_tvs_line_counts_for_nobody():
    rows = people_list._room_lines(
        named_lines=[
            {**line(T, "Tonight on the news"), "speaker": "Alice", "media": True},
            {**line(T + 5, "Dinner's ready"), "speaker": "Alice"},
        ]
    )
    assert [r["text"] for r in rows] == ["Dinner's ready"]


def test_a_voice_heard_only_from_the_tv_is_nobody_to_name():
    view = SimpleNamespace(individual_lines=lambda items: list(items))
    raw = [line(T, "news", 9, media=True), line(T + 5, "more news", 9, media=True), line(T + 10, "hi", 4)]
    assert unnamed_groups.speakers_with_audio(raw, cast(Any, view)) == {4}


@pytest.fixture
def heard(tmp_path, monkeypatch):
    rows = [line(T, "Tonight on the news", 9), line(T + 5, "and the weather", 9), line(T + 10, "Hi", 4)]
    monkeypatch.setattr(context, "STATE", tmp_path)
    monkeypatch.setattr(transcript_lines, "heard_lines", lambda *a, **k: revisions.filter_lines(tmp_path, rows))
    monkeypatch.setattr(speech_log, "retained", lambda: rows)
    monkeypatch.setattr(
        transcript_lines, "_view", lambda: SimpleNamespace(lines=lambda raw: revisions.filter_lines(tmp_path, raw))
    )
    return rows


def test_this_is_tv_on_transcript_lines_marks_them_and_undo_takes_it_back(heard):
    keys = [registry.key_of(heard[0])]
    code, reply = speaker_lines.media_transcript({"lines": keys})
    assert code == 200 and reply["lines"] == keys
    assert speaker_lines.media_transcript({"lines": keys}) == (200, reply), "a retry gets the same Undo"
    assert [r.get("media") for r in transcript_lines.heard_lines()] == [True, None, None]
    assert speaker_lines.undo_transcript({"undo": reply["undo"]}) == (200, {"ok": True, "lines": keys})
    assert [r.get("media") for r in transcript_lines.heard_lines()] == [None, None, None]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        pytest.param({}, 400, id="no-lines"),
        pytest.param({"lines": []}, 400, id="empty"),
        pytest.param({"lines": ["wave-3@1.000"]}, 404, id="not-heard"),
    ],
)
def test_this_is_tv_needs_lines_that_were_heard(heard, body, status):
    assert speaker_lines.media_transcript(body)[0] == status


def test_a_batch_of_keys_is_found_from_the_hours_they_fall_in(tmp_path, monkeypatch):
    """GET /transcript's corrections take up to 1000 line keys at once, including two microphones' copies starting
    in the same millisecond. The route reads the hours those keys fall in, not the whole speech history."""
    from speech_fake import fake

    monkeypatch.setattr(context, "STATE", tmp_path)
    both = [line(T, "Front door", device="Mic A"), line(T, "Front door", device="Mic B")]
    rest = [line(T + 7 * i, f"line {i}") for i in range(1, 999)]  # one screen of lines, a couple of hours apart
    fake().write([*both, *rest])
    keys = [registry.key_of(e) for e in [*both, *rest]]
    code, reply = speaker_lines.media_transcript({"lines": keys})
    assert (code, len(reply["lines"])) == (200, 1000)
    marked = {registry.key_of(r) for r in revisions.filter_lines(tmp_path, [*both, *rest]) if r.get("media")}
    assert marked == set(keys), "every key the batch named, on both microphones"


def test_this_is_tv_on_an_unnamed_voice_marks_every_line_it_has(heard, monkeypatch):
    monkeypatch.setattr(registry, "voice_names", lambda state, sids: dict.fromkeys(sids))
    code, reply = speaker_lines.media_voice({"group": 9})
    assert code == 200
    assert reply["lines"] == [registry.key_of(heard[0]), registry.key_of(heard[1])]
    assert [r.get("media") for r in transcript_lines.heard_lines()] == [True, True, None]
    assert speaker_lines.media_voice({"group": 9})[0] == 404, "nothing left to mark"


def test_this_is_tv_is_refused_for_a_named_voice(heard, monkeypatch):
    monkeypatch.setattr(registry, "voice_names", lambda state, sids: dict.fromkeys(sids, "Alice"))
    assert speaker_lines.media_voice({"group": 9})[0] == 409
    assert speaker_lines.media_voice({"group": "9"})[0] == 400


def test_a_marked_line_says_who_marked_it(tmp_path):
    rows = [line(T, "the news"), line(T + 5, "more news"), line(T + 10, "dinner")]
    revisions.mark_media(tmp_path, rows[:1], source="classifier")
    revisions.mark_media(tmp_path, rows[1:2])
    got = revisions.filter_lines(tmp_path, rows)
    assert [(r.get("media"), r.get("media_by")) for r in got] == [(True, "classifier"), (True, "owner"), (None, None)]
