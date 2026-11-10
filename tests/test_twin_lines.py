"""The same speech heard on two microphones, written as two lines (#284): a label on one copy applies to the other,
readers show it once, and "Why this name" explains the copy by the label it borrowed. Each line keeps its own key."""

from unittest.mock import create_autospec

import pytest

from liberational.dashboard import server
from liberational.domain.identity import Basis
from liberational.people import registry, transcript_view
from speech_fake import fake
from voices import write_voices


def line(ts, text, device, sid=1):
    return {"ts_start": ts, "ts_end": ts + 2, "text": text, "speaker": "Speaker 1", "speaker_id": sid,
            "speaker_sim": 0.6, "device": device}  # fmt: skip


LINES = [
    line(1000.0, "Lights on, please.", "Kitchen"),
    line(1000.4, "lights on please", "Hall"),  # the same speech on the other mic
    line(1100.0, "Is dinner ready?", "Hall"),  # said once
]


@pytest.fixture
def state(tmp_path, monkeypatch):
    write_voices(tmp_path, {1: None})  # an unnamed voice heard all three
    fake().write(LINES)
    monkeypatch.setattr(server, "STATE", tmp_path)
    return tmp_path


def label(state, ts, device, name):
    registry.at(state).claim(registry.utterance(ts, device), name, Basis.CORRECTION, "dashboard")


@pytest.mark.parametrize("labeled", [(1000.0, "Kitchen"), (1000.4, "Hall")], ids=["the-shown-copy", "the-other"])
def test_a_label_on_either_copy_names_both(state, labeled):
    label(state, *labeled, "Eve")
    read = transcript_view.View(state).lines(LINES)
    assert [e["speaker"] for e in read] == ["Eve", "Eve", "Speaker 1"]
    assert [registry.key_of(e) for e in read] == [registry.key_of(e) for e in LINES]  # each keeps its own key


def test_a_copy_labeled_on_its_own_keeps_its_own_label(state):
    label(state, 1000.0, "Kitchen", "Eve")
    label(state, 1000.4, "Hall", "Dana")  # the owner said otherwise about this copy: that stands
    assert [e["speaker"] for e in transcript_view.View(state).lines(LINES)][:2] == ["Eve", "Dana"]


def test_readers_show_one_thing_said_once(state):
    read = transcript_view.View(state).lines(LINES)
    assert [e.get("twin_of") for e in read] == [None, "kitchen@1000.000", None]
    assert [e["text"] for e in transcript_view.once(read)] == ["Lights on, please.", "Is dinner ready?"]


def test_one_line_read_with_the_lines_around_it_borrows_its_copys_label(state):
    label(state, 1000.0, "Kitchen", "Eve")
    view = transcript_view.View(state)
    hall = LINES[1]
    assert view.line(hall)["speaker"] == "Speaker 1"  # on its own it can't know its copy
    assert view.line(hall, LINES)["speaker"] == "Eve"
    s = view.explain(hall, LINES)
    assert s is not None and s.own.winner is not None and s.own.winner.basis == Basis.CORRECTION


def test_why_this_name_explains_a_copy_by_the_label_it_borrowed(state):
    label(state, 1000.0, "Kitchen", "Eve")
    body, code = server.why_name_view({"line": "hall@1000.400"})
    assert code == 200
    assert (body["name"], body["via"], body["basis"]) == ("Eve", "line", "correction")


def test_the_days_transcript_shows_each_thing_said_once(state, monkeypatch):
    entries = server.merged_tail(80)
    assert [e["text"] for e in entries] == ["Lights on, please.", "Is dinner ready?"]


def test_a_withdrawn_label_isnt_lent_to_a_copy_on_another_voice(state):
    write_voices(state, {1: "Alice", 2: "Carol"})
    kitchen, hall = {**LINES[0], "speaker_id": 1}, {**LINES[1], "speaker_id": 2}
    label(state, 1000.0, "Kitchen", "Eve")
    registry.at(state).retract(registry.utterance(1000.0, "Kitchen"), "dashboard")  # Kitchen reads as Alice again
    read = transcript_view.View(state).lines([kitchen, hall])
    assert [e["speaker"] for e in read] == ["Alice", "Carol"]  # Hall follows its own voice, not Alice's


def test_the_live_feed_resends_a_labeled_line_even_when_it_reads_as_a_copy(state):
    seen = server.label_marks(transcript_view.View(state).lines(LINES))
    label(state, 1000.4, "Hall", "Eve")  # the owner labels the copy the client was showing
    changed, _ = server.relabeled_tail(seen)
    assert "hall@1000.400" in {registry.key_of(e) for e in changed}


# ---- readers that show or count speech, once (#476) ----


def with_audio(state):
    """The same lines, each with a clip worth playing, so the People inbox would offer them."""
    lines = [{**e, "audio": {"stream": "mic.audio", "start_ns": 0, "end_ns": 2000000000}} for e in LINES]
    fake().write(lines)


def test_the_people_inbox_asks_about_speech_heard_on_two_mics_once(state, monkeypatch):
    with_audio(state)
    monkeypatch.setattr(server, "inbox_skips", lambda: {"voice": [], "line": [], "face": []})
    clips, total = server.voice_clips([{"id": 1, "label": "Speaker 1"}])
    assert total == 2  # "Lights on, please." once, and "Is dinner ready?"
    assert sorted(c["text"] for c in clips) == ["Is dinner ready?", "Lights on, please."]


def test_who_was_seen_together_counts_speech_heard_on_two_mics_once(state):
    write_voices(state, {1: "Dana"})  # the voice is Dana's now
    server._named_cache.update(key=None, value=[])
    spoken = server._named_lines(0)
    assert [(t, who) for t, _end, who in spoken] == [(1000.0, "Dana"), (1100.0, "Dana")]


@pytest.mark.parametrize(
    ("labelled", "speakers"),
    [(0, ["Eve", "Speaker 1", "Speaker 1"]), (1, ["Speaker 1", "Eve", "Speaker 1"])],
    ids=["shown-mic-labelled", "copy-mic-labelled"],
)
def test_independent_line_projection_does_not_borrow_a_twins_label(state, labelled, speakers):
    selected = LINES[labelled]
    label(state, selected["ts_start"], selected["device"], "Eve")
    view = transcript_view.View(state)
    read = view.individual_lines(iter(LINES))
    assert read == [view.line(e) for e in LINES]
    assert [e["speaker"] for e in read] == speakers
    assert [e.get("twin_of") for e in read] == [None, None, None]
    assert [e["speaker"] for e in view.lines(LINES)] == ["Eve", "Eve", "Speaker 1"]
    assert [e.get("twin_of") for e in view.lines(LINES)] == [None, "kitchen@1000.000", None]


def test_independent_line_projection_keeps_multiple_speakers_and_copy_flags_local(state):
    reg = registry.at(state)
    reg.detach(registry.utterance(1000.0, "Kitchen"), transcript_view.MULTI, "test")
    rows = [LINES[0], {**LINES[1], "twin_of": "stored-copy"}, LINES[2]]
    view = transcript_view.View(state)
    read = view.individual_lines(rows)
    assert read == [view.line(e) for e in rows]
    assert [(e["speaker"], e.get("multi_speaker"), e.get("twin_of")) for e in read] == [
        ("multiple speakers", True, None),
        ("Speaker 1", None, "stored-copy"),
        ("Speaker 1", None, None),
    ]
    assert [e["speaker"] for e in view.lines(rows)] == ["multiple speakers", "multiple speakers", "Speaker 1"]


def test_independent_line_projection_recovers_other_lines_when_one_decision_fails(state, monkeypatch):
    write_voices(state, {1: "Alice"})
    label(state, 1000.0, "Kitchen", "Eve")
    view = transcript_view.View(state)
    reg = view.reg
    assert reg is not None
    real = reg.said
    bad = registry.utterance(1000.4, "Hall")

    def failed_on_hall(lines, refresh=True):
        lines = list(lines)
        if any(subject == bad for subject, _sid in lines):
            raise RuntimeError("one line decision unavailable")
        return real(lines, refresh=refresh)

    monkeypatch.setattr(reg, "said", create_autospec(real, side_effect=failed_on_hall))
    read = view.individual_lines(iter(LINES))
    assert read == [view.line(e) for e in LINES]
    assert [e["speaker"] for e in read] == ["Eve", "Speaker 1", "Alice"]
    assert [e["speaker"] for e in view.lines(LINES)] == ["Speaker 1", "Speaker 1", "Speaker 1"]


def test_independent_line_projection_preserves_nonheard_input_and_reads_without_a_log(tmp_path):
    rows = [None, "not a line", {"assistant": True, "ts_start": 10.0, "text": "reply"}, {"text": "no time"}]
    assert transcript_view.View(tmp_path).individual_lines(rows) == rows
    heard = line(1200.0, "a fresh recording", "Kitchen")
    [read] = transcript_view.View(tmp_path).individual_lines([heard])
    assert (read["speaker"], read["line"], read.get("twin_of")) == ("Speaker 1", "kitchen@1200.000", None)


def test_the_same_lines_read_twice_are_decided_once_and_each_reader_gets_its_own(state, monkeypatch):
    label(state, 1000.0, "Kitchen", "Eve")
    batches = []
    said = registry.Registry.said

    def counted(self, lines, refresh=True):
        batches.append(len(lines := list(lines)))
        return said(self, lines, refresh)

    monkeypatch.setattr(registry.Registry, "said", counted)
    view = transcript_view.View(state)
    first = view.lines(LINES)
    first[0]["speaker"] = "changed"
    assert [e["speaker"] for e in view.named(LINES)] == ["Eve", "Eve", "Speaker 1"]  # #2440: lines, then named
    assert batches == [3]
    view.lines(LINES[:2])  # another batch
    view.individual_lines(LINES)  # the same lines, read without borrowing a copy's label
    assert batches == [3, 2, 3]
