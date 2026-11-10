"""Labels given in "Lines to check" are logged with what they replaced, and can be undone one line at a time or
everything since a moment, including any voice reference a label added. A label and its undo are events in the log;
transcript.jsonl is never rewritten, and lines read through people/transcript_view.py."""

import json

import pytest

from liberational import speech_log
from liberational.dashboard import server
from liberational.people import registry, transcript_view
from speech_fake import fake
from voices import write_voices


def _line(ts, text, speaker="Sam", sid=1, sim=0.4, **extra):
    return {"ts_start": ts, "text": text, "speaker": speaker, "speaker_id": sid, "speaker_sim": sim, **extra}


class Transcript:
    """The lines on the speech log, and how readers see them."""

    def __init__(self, state):
        self.state = state

    def write(self, lines):
        fake().write(lines)
        self.stored = [dict(e) for e in fake().lines[speech_log.UTTERANCES]]

    def read(self):
        """Each line as readers see it. The lines themselves never change."""
        assert [dict(e) for e in fake().lines[speech_log.UTTERANCES]] == self.stored
        lines = transcript_view.View(self.state).lines(fake().range(0, 2**40))
        return {e["ts_start"]: e for e in lines}


@pytest.fixture
def lines(tmp_path, monkeypatch):
    write_voices(tmp_path, {1: {"label": "Sam", "enrolled": True, "refs": []}})
    t = Transcript(tmp_path)
    t.write(
        [
            _line(1000.0, "first"),
            _line(1001.0, "second", sim=0.5, tagged=True),
            _line(1002.0, "third", speaker="multiple speakers", sid=None, sim=None, multi_speaker=True),
        ]
    )
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "REFINE_LOG", tmp_path / "refine_log.jsonl")
    return t


def label_fields(e):
    return {k: e.get(k) for k in server.REFINE_FIELDS}


def logged():
    return [json.loads(ln) for ln in server.REFINE_LOG.read_text().splitlines()]


def test_a_label_is_logged_with_what_it_replaced(lines):
    code, o = server.speakers_refine("assign", {"ts_start": 1000.0, "name": "Carly"})
    assert (code, o["entry"]["speaker"], o["entry"]["speaker_id"]) == (200, "Carly", None)
    (entry,) = logged()
    assert entry["action"] == "assign"
    assert entry["old"] == {"speaker": "Sam", "speaker_id": 1, "speaker_sim": 0.4}
    assert entry["new"] == {"speaker": "Carly", "speaker_id": None, "speaker_sim": None, "tagged": True}
    assert entry["was"] == "voice"
    assert server.refine_log_list()["labels"][0]["text"] == "first"
    assert lines.read()[1000.0]["speaker"] == "Carly"


def test_undo_one_line_restores_it_exactly(lines):
    before = {ts: label_fields(e) for ts, e in lines.read().items()}
    server.speakers_refine("assign", {"ts_start": 1001.0, "name": "Carly"})
    server.speakers_refine("multi", {"ts_start": 1000.0})
    server.speakers_refine("assign", {"ts_start": 1002.0, "name": "Carly"})  # was "multiple speakers"
    code, o = server.speakers_refine_undo({"ts_starts": [1001.0, 1002.0]})
    assert (code, o["ok"], len(o["undone"])) == (200, True, 2)
    after = lines.read()
    assert label_fields(after[1001.0]) == before[1001.0]  # tagged flag and sim are back
    assert label_fields(after[1002.0]) == before[1002.0]  # multi_speaker is back
    assert after[1000.0]["speaker"] == "multiple speakers"  # not asked for, still labeled
    assert [e["ts_start"] for e in server.refine_log_list()["labels"]] == [1000.0]


def test_undo_never_brings_a_deleted_person_back(lines):
    reg = registry.at(server.STATE)
    for ts in (1000.0, 1001.0):
        server.speakers_refine("assign", {"ts_start": ts, "name": "Dana"})
        server.speakers_refine("assign", {"ts_start": ts, "name": "Carly"})
    reg.forget("Dana", "dashboard")  # People: Delete
    code, o = server.speakers_refine_undo({"ts_starts": [1000.0]})
    assert (code, len(o["undone"])) == (200, 1)
    assert reg.person_id("Dana") is None
    assert lines.read()[1000.0]["speaker"] == "unknown"  # unnamed, as forgetting leaves a line
    new = reg.ensure("Dana", "dashboard")  # someone new is called Dana
    server.speakers_refine_undo({"ts_starts": [1001.0]})
    assert lines.read()[1001.0]["speaker"] == "unknown"  # the old Dana's line is not the new Dana's
    assert reg.utterances_of("Dana") == []
    assert reg.person_id("Dana") == new


def test_undo_follows_a_rename(lines):
    reg = registry.at(server.STATE)
    server.speakers_refine("assign", {"ts_start": 1000.0, "name": "Dana"})
    server.speakers_refine("assign", {"ts_start": 1000.0, "name": "Carly"})
    reg.rename("Dana", "Dana Cole", "dashboard")
    server.speakers_refine_undo({"ts_starts": [1000.0]})
    assert lines.read()[1000.0]["speaker"] == "Dana Cole"
    assert reg.person_id("Dana") is None  # the undo registered nobody under her old name


def test_undo_everything_since_a_moment(lines, monkeypatch):
    before = {ts: label_fields(e) for ts, e in lines.read().items()}
    server.speakers_refine("assign", {"ts_start": 1000.0, "name": "Carly"})
    cut = logged()[-1]["ts"] + 0.001
    with monkeypatch.context() as m:
        m.setattr(server.time, "time", lambda: cut + 10)
        server.speakers_refine("assign", {"ts_start": 1001.0, "name": "Carly"})
        server.speakers_refine("assign", {"ts_start": 1001.0, "name": "Dana"})  # labeled twice: both unwind
    code, o = server.speakers_refine_undo({"since": cut})
    assert (code, len(o["undone"])) == (200, 2)
    after = lines.read()
    assert label_fields(after[1001.0]) == before[1001.0]
    assert after[1000.0]["speaker"] == "Carly"  # before the cut, kept
    assert server.refine_log_list()["total"] == 1


def undo_old_label(lines, old, written):
    """Undo a label logged before step 4, when it was written onto the line itself (`written`); `old` is what the log
    says it replaced. Returns the undone line, as the undo reports it and as readers see it."""
    lines.write([_line(1000.0, "first", speaker=written, sid=None, sim=None, tagged=True)])
    entry = {"ts": 5.0, "action": "assign", "ts_start": 1000.0, "text": "first", "old": old}
    server.REFINE_LOG.write_text(json.dumps(entry) + "\n")
    code, o = server.speakers_refine_undo({"ts_starts": [1000.0]})
    assert code == 200
    return o["undone"][0], lines.read()[1000.0]


def test_undo_of_a_label_given_before_step_4(lines):
    undone, line = undo_old_label(lines, {"speaker": "Sam", "speaker_id": 1, "speaker_sim": 0.4}, "Carly")
    assert (undone["speaker"], line["speaker"]) == ("Sam", "Sam")


def test_undo_before_step_4_of_a_label_over_an_automatic_name(tmp_path, lines):
    """The old label erased the line's voice id. Its voice was "Speaker 7" then and is Gus now: the line reads as
    Gus, his own, not a detached "Speaker 7"."""
    write_voices(tmp_path, {7: {"label": "Gus", "refs": []}})
    undone, line = undo_old_label(lines, {"speaker": "Speaker 7", "speaker_id": 7, "speaker_sim": 0.6}, "Dana")
    assert (undone["speaker"], line["speaker"], line.get("unlabeled")) == ("Gus", "Gus", None)


def test_bad_requests(lines):
    assert server.speakers_refine_undo({})[0] == 400
    assert server.speakers_refine_undo({"ts_starts": "x"})[0] == 400
    code, o = server.speakers_refine_undo({"ts_starts": [5.0]})  # never labeled: nothing to do
    assert (code, o["undone"]) == (200, [])


# ------------------------------------------------------- two microphones ----
A, B = "mic-a@2000.000", "mic-b@2000.000"


@pytest.fixture
def twins(lines):
    """Two microphones' lines that start in the same millisecond, beside the fixture's three."""
    stored = lines.stored
    lines.write([*stored, _line(2000.0, "from a", device="Mic A"), _line(2000.0, "from b", device="Mic B")])
    return lines


def twin(t, key):
    return next(e for e in transcript_view.View(t.state).lines(fake().range(0, 2**40)) if e["line"] == key)


def test_a_label_by_line_key_labels_only_that_microphones_line(twins):
    code, o = server.speakers_refine("assign", {"line": B, "name": "Carly"})
    assert (code, o["entry"]["line"]) == (200, B)
    assert (twin(twins, A)["speaker"], twin(twins, B)["speaker"]) == ("Sam", "Carly")
    assert logged()[-1]["line"] == B


def test_a_time_two_microphones_share_is_refused_and_labels_nothing(twins):
    assert server.speakers_refine("assign", {"ts_start": 2000.0, "name": "Carly"}) == (
        409,
        {"ok": False, "error": server.AMBIGUOUS},
    )
    assert (twin(twins, A)["speaker"], twin(twins, B)["speaker"]) == ("Sam", "Sam")


def test_undo_by_line_key_puts_back_only_that_line(twins):
    server.speakers_refine("assign", {"line": A, "name": "Carly"})
    server.speakers_refine("assign", {"line": B, "name": "Carly"})
    code, o = server.speakers_refine_undo({"lines": [A]})
    assert (code, [u["line"] for u in o["undone"]]) == (200, [A])
    assert (twin(twins, A)["speaker"], twin(twins, B)["speaker"]) == ("Sam", "Carly")


def test_undo_by_line_key_finds_a_label_logged_before_keys(lines):
    server.speakers_refine("assign", {"ts_start": 1000.0, "name": "Carly"})
    rows = logged()
    del rows[-1]["line"]  # legacy entry without a line key
    server.REFINE_LOG.write_text("".join(json.dumps(r) + "\n" for r in rows))
    code, o = server.speakers_refine_undo({"lines": ["1000.000"]})
    assert (code, [u["ts_start"] for u in o["undone"]]) == (200, [1000.0])


def test_a_clip_by_line_key_is_that_microphones_line(twins):
    text = {k: (e or {}).get("text") for k in (A, B, "mic-c@2000.000") for e in [server.find_line(k)]}
    assert text == {A: "from a", B: "from b", "mic-c@2000.000": None}


def test_undo_takes_back_one_label_of_a_line_labeled_before_and_after_keys(lines):
    server.speakers_refine("assign", {"ts_start": 1000.0, "name": "Carly"})
    rows = logged()
    del rows[-1]["line"]  # the first label, written without a line key
    server.REFINE_LOG.write_text("".join(json.dumps(r) + "\n" for r in rows))
    server.speakers_refine("assign", {"ts_start": 1000.0, "name": "Dana"})  # labeled again since
    code, o = server.speakers_refine_undo({"ts_starts": [1000.0]})
    assert (code, len(o["undone"]), lines.read()[1000.0]["speaker"]) == (200, 1, "Carly")


def test_the_live_feed_sends_only_the_microphones_line_that_was_labeled(twins):
    _, seen = server.relabeled_tail({})
    server.speakers_refine("assign", {"line": B, "name": "Carly"})
    changed, seen = server.relabeled_tail(seen)
    assert [e["line"] for e in changed] == [B]
    server.speakers_refine("assign", {"line": A, "name": "Dana"})
    changed, _ = server.relabeled_tail(seen)
    assert [e["line"] for e in changed] == [A]


@pytest.mark.parametrize(
    "body",
    [pytest.param({"lines": ["bogus"]}, id="undo-bad-key"), pytest.param({"lines": [5]}, id="undo-not-text")],
)
def test_undo_refuses_lines_that_are_not_keys(lines, body):
    assert server.speakers_refine_undo(body)[0] == 400


def test_a_label_refuses_a_line_that_is_not_a_key(lines):
    assert server.speakers_refine("assign", {"line": 5, "name": "Carly"})[0] == 400


def test_undo_takes_back_one_label_of_a_microphones_line_labeled_before_and_after_keys(lines):
    stored = lines.stored
    lines.write([*stored, _line(3000.0, "alone", device="Mic A")])  # one microphone's line, alone in its millisecond
    key = "mic-a@3000.000"
    server.speakers_refine("assign", {"line": key, "name": "Carly"})
    rows = logged()
    del rows[-1]["line"]  # the first label, written without a line key
    server.REFINE_LOG.write_text("".join(json.dumps(r) + "\n" for r in rows))
    server.speakers_refine("assign", {"line": key, "name": "Dana"})  # labeled again since, by its key
    code, o = server.speakers_refine_undo({"lines": [key]})
    assert (code, len(o["undone"]), twin(lines, key)["speaker"]) == (200, 1, "Carly")
