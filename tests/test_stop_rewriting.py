"""Rules 1 and 2 of docs/DOMAIN-MODEL.md: relabels and merges are assertions, transcript.jsonl is never rewritten for them,
and every reader resolves a line's speaker through the projection (people/transcript_view.py).

The migration test builds a transcript the way the old rewriting code left it, with relabels and merges, and checks
that every line shows the same name before and after `pack-registry backfill`. Everything runs on temp dirs, with
made-up people."""

import datetime as dt
import http.client
import io
import json
import threading
import time
import urllib.parse
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as hs

import speech_fake
from liberational import speech_log
from liberational.dashboard import (
    context,
    people_list,
    presence,
    server,
    shell_status,
    speaker_lines,
    transcript_lines,
    unnamed_groups,
)
from liberational.domain.identity import Basis
from liberational.domain.kernel import EventType
from liberational.memory import history_qa, recall, summarize
from liberational.people import forgotten, identity, profiles, registry, transcript_view, voice_edits, whisper_names
from liberational.phone import bridge
from liberational.speech import clone_say
from voices import write_voices

T0 = time.mktime((2026, 10, 20, 9, 0, 0, 0, 0, -1))
MULTI, UNKNOWN = transcript_view.MULTI, transcript_view.UNKNOWN


def unit(*v):
    a = np.asarray(v, np.float32)
    return (a / np.linalg.norm(a)).tolist()


def line(i, speaker, sid, sim, **extra):
    return {"ts_start": T0 + i, "ts_end": T0 + i + 2, "speaker": speaker, "speaker_id": sid, "speaker_sim": sim,
            "text": f"line {i}", **extra}  # fmt: skip


def write_lines(state, lines):
    """The lines, on the speech log, and in transcript.jsonl for the readers still on the file (the registry's day
    files, CLI, and backfill)."""
    (state / "transcript.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in lines))
    speech = speech_fake.fake()
    if speech.lines.get(speech_log.UTTERANCES):
        speech.purge(speech_log.UTTERANCES, lambda e: True)
    for e in lines:
        speech.say(e)


def stored(state):
    return [dict(e) for e in speech_fake.fake().lines.get(speech_log.UTTERANCES, [])]


def ts(state, i):
    return stored(state)[i]["ts_start"]


def viewed(state, i):
    """Stored line i as readers see it."""
    return transcript_view.View(state).line(stored(state)[i])


def tag(state, i, name, source=transcript_view.VOICE_TAG):
    registry.at(state).claim(registry.utterance(ts(state, i)), name, Basis.OWNER_TAG, source)


def names(state):
    return [e["speaker"] for e in transcript_view.view(stored(state), state)]


def legacy_state(state):
    """A state directory with lines as the code before step 4 left them: rewritten by merges, dashboard labels, voice
    tags, unlabels, and backfill relabels. The voices v1 merged (3 into Alice's, 6 into Dana's, 7 into 2) are their
    own voiceprints now, each named for whoever it was folded into (naming replaces merging)."""
    write_voices(state, {1: "Alice", 2: None, 3: "Alice", 4: "Dana", 5: "Gus", 6: "Dana", 7: None})
    write_lines(
        state,
        [
            line(0, "Alice", 1, 0.8),  # a voice line
            line(1, "Alice", 1, 0.7),  # was voice 3: the merge rewrote its id and label
            line(2, "Speaker 6", 6, 0.66),  # voice 6 folded into Dana by a label, which never rewrote lines
            line(3, "Erin", None, None, tagged=True),  # labeled in Lines to check
            line(4, "Dana", 4, None, tagged=True),  # a voice tag: relabeled onto Dana's voice
            line(5, MULTI, None, None, multi_speaker=True),
            line(6, UNKNOWN, None, 0.0, unlabeled=True),
            line(7, "Alice", 1, 0.81),  # a backfill relabel
            line(8, "Speaker 2", 2, 0.6),  # an unnamed voice
            line(9, "Speaker 2", 2, 0.62),  # was voice 7, merged into 2
            line(10, "Speaker 5", 5, 0.7),  # voice 5 was named Gus after this line was heard
            line(11, UNKNOWN, None, 0.2),  # too short to open a voice
        ],
    )
    return state


def migrated(state):
    """The legacy state brought up to date: the backfill (step 3) records the hand labels on its lines."""
    registry.backfill(state, state / "faces" / "db.npz")
    return state


@pytest.fixture
def state(tmp_path):
    return migrated(legacy_state(tmp_path))


@pytest.fixture
def dashboard(monkeypatch):
    """Point the dashboard server at a state directory, with no password."""

    def at(state):
        monkeypatch.setattr(speaker_lines, "REFINE_LOG", state / "refine_log.jsonl")
        monkeypatch.setattr(context, "STATE", state)
        monkeypatch.setattr(server, "DASH_PASSWORD", "")
        return state

    return at


class Feed(server.H):
    """The handler for one GET /events, without a socket."""

    def __init__(self, out):
        self.wfile = out

    def _shell_status(self):
        return shell_status.Status({})  # these feeds are about the transcript: no pauses, services, or research

    def send_response(self, code, message=None):
        pass

    def send_header(self, keyword, value):
        pass

    def end_headers(self):
        pass


def feed(monkeypatch, during=lambda: None):
    """Run one live feed through three polls and return the transcript events it sent. `during` runs in the first
    poll, where the feed would otherwise sleep."""
    me, ticks = threading.get_ident(), [0]

    def sleep(s):
        if threading.get_ident() != me:  # another test's feed, still open
            return time.sleep(s)
        ticks[0] += 1
        if ticks[0] == 1:
            during()
        elif ticks[0] > 3:
            raise BrokenPipeError

    class Clock:
        def __getattr__(self, name):
            return sleep if name == "sleep" else getattr(time, name)

    monkeypatch.setattr(server, "time", Clock())
    out = io.BytesIO()
    Feed(out).sse()
    blocks = out.getvalue().decode().split("\n\n")
    return [json.loads(b.split("data: ", 1)[1]) for b in blocks if b.startswith("event: transcript")]


def post(base, path, body):
    c = http.client.HTTPConnection("127.0.0.1", urllib.parse.urlsplit(base).port, timeout=5)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    got = r.status, json.loads(r.read())
    c.close()
    return got


# ---------------------------------------------------------------- migration ----
def test_every_line_shows_the_same_name_before_and_after(tmp_path):
    state = legacy_state(tmp_path)
    # what each line showed before step 4
    before = ["Alice", "Alice", "Dana", "Erin", "Dana", MULTI, UNKNOWN, "Alice",
              "Speaker 2", "Speaker 2", "Gus", UNKNOWN]  # fmt: skip
    migrated(state)
    after = transcript_view.view(stored(state), state)
    v = transcript_view.View(state)
    assert [v.shown(e) for e in after] == before
    assert [e["speaker"] for e in after] == before  # what server-side readers use
    assert transcript_view.changed_names(state) == []
    flags = lambda e: {k: e.get(k) for k in ("multi_speaker", "unlabeled", "tagged")}
    assert [flags(e) for e in after] == [flags(e) for e in stored(state)]  # the backfill and the dashboard read them


def test_a_voice_renamed_after_a_line_was_heard(tmp_path):
    state = legacy_state(tmp_path)
    write_lines(state, [line(0, "Hal", 5, 0.7)])  # voice 5 was Hal then, and is Gus now
    migrated(state)
    (e,) = transcript_view.view(stored(state), state)
    assert (e["speaker"], transcript_view.View(state).shown(e)) == ("Hal", "Gus")  # as before step 4
    assert transcript_view.changed_names(state) == []


def test_a_rebuild_matches_a_line_stamped_from_its_unrounded_time(tmp_path):
    state = legacy_state(tmp_path)
    write_lines(state, [line(1, "Alice", 1, 0.8)])  # heard at T0 + 0.9996, stored rounded to T0 + 1.0
    md = state / "transcript" / f"{transcript_view.day_of(T0)}.md"
    md.parent.mkdir()
    md.write_text("[09:00:00] Alice: line 1\n")
    transcript_view.rebuild_markdown(state)
    assert md.read_text() == "[09:00:01] Alice: line 1\n"


def test_a_rerun_of_the_backfill_never_undoes_a_later_label(state):
    tag(state, 3, "Dana", source="dashboard")  # the line labeled Erin
    assert registry.backfill(state, state / "faces" / "db.npz")["assertions"] == 0
    assert transcript_view.View(state).name(stored(state)[3]) == "Dana"


def test_the_cli_checks_and_rebuilds(state, capsys):
    with pytest.raises(SystemExit) as done:
        registry.main(["--state-dir", str(state), "lines"])
    assert done.value.code == 0
    assert "0 line(s)" in capsys.readouterr().out
    registry.main(["--state-dir", str(state), "transcript"])
    md = (state / "transcript" / f"{transcript_view.day_of(T0)}.md").read_text().splitlines()
    assert md[2] == "[09:00:02] Dana: line 2"
    assert len(md) == 12


# ------------------------------------------------------------------ writers ----
@pytest.fixture
def written(state, dashboard):
    """The migrated state behind the dashboard. Every former rewriter leaves the lines alone."""
    before = stored(state)
    yield dashboard(state)
    assert stored(state) == before  # never rewritten


def test_naming_a_voice_moves_its_lines_without_rewriting_them(written):
    voice_edits.label(written, 2, "Alice")  # the unnamed voice 2 is Alice's
    assert names(written)[8:10] == ["Alice", "Alice"]
    e = viewed(written, 9)
    assert (e["speaker_id"], e["speaker_sim"]) == (2, 0.62)  # its own voice, as heard


def test_labels_unlabels_and_undo_from_the_dashboard(written):
    hand, multi = ts(written, 8), ts(written, 0)
    assert speaker_lines.speakers_refine("assign", {"ts_start": hand, "name": "Hana"})[0] == 200
    assert names(written)[8] == "Hana"
    speaker_lines.speakers_refine("multi", {"ts_start": multi})
    assert names(written)[0] == MULTI
    speaker_lines.speakers_refine_undo({"ts_starts": [hand, multi]})
    assert (names(written)[0], names(written)[8]) == ("Alice", "Speaker 2")
    md = (written / "transcript" / f"{transcript_view.day_of(T0)}.md").read_text()
    assert "[09:00:00] Alice: line 0" in md  # the day's file was rebuilt


def test_unlabel_route(written, serve_dashboard):
    code, got = post(serve_dashboard(), "/transcript/unlabel", {"ts_start": ts(written, 0)})
    assert (code, got["entry"]["speaker"], got["entry"]["unlabeled"]) == (200, UNKNOWN, True)
    assert names(written)[0] == UNKNOWN
    registry.at(written).claim(registry.voiceprint(1), "Erin", Basis.OWNER_TAG, "test")
    assert names(written)[0] == UNKNOWN  # the voice renamed later: still unknown


def test_the_live_feed_sends_a_line_again_when_a_label_changes_it(written, monkeypatch):
    when = ts(written, 11)  # an unknown line with no voice: a tag changes only the log
    first, got, *_ = feed(monkeypatch, lambda: tag(written, 11, "Erin", source="test"))
    assert first["reset"]
    assert (got["reset"], got["replace"]) == (False, True)
    assert [(e["ts_start"], e["speaker"], e["tagged"]) for e in got["entries"]] == [(when, "Erin", True)]


# ------------------------------------------------------- forget and the feed ----
@pytest.fixture
def forgetting(tmp_path, dashboard):
    """Forgetting a person is the one rewrite of transcript.jsonl left ("Forgetting" in docs/DOMAIN-MODEL.md). The rewrite usually makes
    the file larger ("Dana" becomes "Speaker 4"), so the live feed must notice a new file, not only a smaller one."""
    write_voices(tmp_path, {4: {"label": "Dana", "centroid": unit(1, 0), "n": 6}, 9: {"centroid": unit(0, 1), "n": 6}})
    write_lines(tmp_path, [line(i, *(("Dana", 4) if i % 2 == 0 else ("Speaker 9", 9)), 0.8) for i in range(6)])
    return dashboard(tmp_path)


REPLY = {"ts_start": T0 + 100, "ts_end": T0 + 101, "text": "hi Dana", "assistant": True, "speaker": "Pack"}


def reply(state, text):
    speech_fake.fake().reply({**REPLY, "ts_start": T0 + 102, "text": text})


def test_a_reply_while_the_feed_connects_reaches_it(forgetting, monkeypatch):
    """Finding 11: on connect the feed read the tail, then the assistant's offset, so a reply in between was in
    neither the first batch nor any later read."""
    speech_fake.fake().reply({**REPLY, "text": "first"})
    real_tail, replied = transcript_lines.merged_tail, []

    def tail(n=80):
        got = real_tail(n)
        if not replied:  # the voice loop replies right after the tail read the file
            replied.append(True)
            reply(forgetting, "REPLY-X")
        return got

    monkeypatch.setattr(transcript_lines, "merged_tail", tail)
    sent = [e["text"] for ev in feed(monkeypatch) for e in ev["entries"]]
    assert sent.count("REPLY-X") == 1


# ------------------------------------------------------------- tagged voice ----
def voice_of(state, i):
    """The name a labeled line shows, and the voice it goes with: what a voice tag taught, else the line's own voice
    when it names the same person, else none (a label from Lines to check)."""
    e = viewed(state, i)
    return e["speaker"], e["speaker_id"]


def test_a_voice_tag_goes_with_the_voice_labeled_with_the_name(state):
    tag(state, 8, "Dana")  # heard as unnamed voice 2: the tag taught Dana's voice (4)
    assert voice_of(state, 8) == ("Dana", 4)
    registry.at(state).rename("Dana", "Dana Cole", "test")
    assert voice_of(state, 8) == ("Dana Cole", 4)


def test_a_voice_tag_goes_with_its_persons_voice_wherever_the_line_was_heard(state):
    registry.at(state).claim(registry.voiceprint(2), "Hana", Basis.OWNER_TAG, "test")  # was unnamed
    tag(state, 8, "Hana")
    assert voice_of(state, 8) == ("Hana", 2)
    tag(state, 0, "Hana")  # heard as Alice's voice: SpeakerDB.tag taught Hana's voice, found by her person
    assert voice_of(state, 0) == ("Hana", 2)
    tag(state, 1, "Ivy")  # nobody's voice is Ivy's, and the line's own voice is Alice's
    assert voice_of(state, 1) == ("Ivy", None)


def test_a_label_from_lines_to_check_has_no_voice(state):
    tag(state, 8, "Dana", source="dashboard")
    tag(state, 0, "Alice", source="dashboard")  # even the voice's own person
    assert [voice_of(state, i)[1] for i in (8, 0)] == [None, None]


# -------------------------------------------------------------- line labels ----
@pytest.fixture
def reg(tmp_path):
    """The registry side: line labels, and an undo that leaves nothing behind. Voice 1 is Alice's."""
    r = registry.at(tmp_path)
    r.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    return r


def who(reg, ts, sid=1):
    (s,) = reg.said([(ts, sid)])
    return s.label if s.detached else reg.person(s.person).display_name if s.person else None


def test_undoing_a_label_that_named_the_voices_own_person(reg, tmp_path):
    reg.claim(registry.utterance(5.0), "Alice", Basis.OWNER_TAG, "test")
    reg.release(registry.utterance(5.0), "test")
    assert who(reg, 5.0) == "Alice"  # back to following the voice, not withdrawn from it
    reg.claim(registry.voiceprint(1), "Dana", Basis.OWNER_TAG, "test")
    assert who(reg, 5.0) == "Dana"
    assert not reg.person_of(registry.utterance(5.0)).retractions
    fresh = registry.Registry(registry.at(tmp_path).log)
    assert fresh.said([(5.0, 1)]) == reg.said([(5.0, 1)])  # rebuilt from the log, the same


def test_a_line_from_an_unnamed_voice_can_be_several_speakers(reg, tmp_path):
    reg.detach(6.0, MULTI, "test", via=registry.voiceprint(9))  # voice 9 names nobody: nothing to withdraw
    assert who(reg, 6.0, 9) == MULTI
    log = registry.at(tmp_path).log
    labeled = [s.event for s in log.read(registry.ASSERTIONS) if s.event.type is EventType.UTTERANCE_LABELED]
    assert (labeled[0].subject, labeled[0].payload["label"]) == (None, MULTI)
    reg.claim(registry.utterance(6.0), "Dana", Basis.OWNER_TAG, "test")  # a person replaces the label
    assert who(reg, 6.0, 9) == "Dana"
    reg.release(registry.utterance(6.0), "test")
    assert who(reg, 6.0, 9) is None  # follows its voice again, which names nobody


def test_detach_withdraws_both_the_lines_own_person_and_its_voices(reg):
    reg.claim(registry.utterance(7.0), "Dana", Basis.OWNER_TAG, "test")
    reg.detach(7.0, UNKNOWN, "test", via=registry.voiceprint(1))
    assert who(reg, 7.0) == UNKNOWN
    assert reg.person_of(registry.utterance(7.0), via=registry.voiceprint(1)).person is None


def test_only_the_owner_gives_a_detached_line_a_person(reg):
    reg.detach(10.0, MULTI, "test", via=registry.voiceprint(1))
    made = reg.claim(
        registry.utterance(10.0), "Gus", Basis.BACKFILL_MATCH, "backfill", confidence=0.8,
        producer=registry.voice_producer(),
    )  # fmt: skip
    assert made is None
    assert who(reg, 10.0) == MULTI
    reg.claim(registry.utterance(10.0), "Gus", Basis.OWNER_TAG, "test")
    assert who(reg, 10.0) == "Gus"


def test_the_lines_labeled_as_someone_are_found_for_forgetting(reg):
    reg.claim(registry.utterance(8.0), "Dana", Basis.OWNER_TAG, "test")
    reg.claim(registry.utterance(9.0), "Dana Cole", Basis.OWNER_TAG, "test")
    reg.merge("Dana Cole", "Dana", "test")
    assert reg.utterances_of("dana") == ["8.000", "9.000"]
    assert reg.utterances_of("Nobody") == []


# ------------------------------------------------------- two microphones ----
A, B = registry.utterance(20.0, "Mic A"), registry.utterance(20.0, "Mic B")  # one millisecond, two microphones


def test_a_label_on_one_microphones_line_leaves_the_other_microphones(reg):
    reg.claim(A, "Dana", Basis.OWNER_TAG, "test")
    reg.detach(registry.utterance(21.0, "Mic A"), UNKNOWN, "test", via=registry.voiceprint(1))
    assert (who(reg, A), who(reg, B)) == ("Dana", "Alice")  # B follows its voice
    assert (who(reg, registry.utterance(21.0, "Mic A")), who(reg, registry.utterance(21.0, "Mic B"))) == (
        UNKNOWN,
        "Alice",
    )
    assert reg.utterances_of("Dana") == ["mic-a@20.000"]


def test_a_label_given_before_line_keys_had_devices_still_names_its_line(reg, tmp_path):
    reg.claim(registry.utterance(20.0), "Dana", Basis.OWNER_TAG, "test")  # the id every line had before
    assert who(reg, A) == "Dana"
    fresh = registry.Registry(registry.at(tmp_path).log)
    assert fresh.said([(A, 1)]) == reg.said([(A, 1)])
    reg.release(A, "test")
    assert who(reg, A) == "Alice"  # the old label is withdrawn, not left under the old id
    assert registry.Registry(registry.at(tmp_path).log).said([(A, 1)]) == reg.said([(A, 1)])


def test_an_old_line_label_is_cleared_for_good(reg):
    reg.detach(20.0, MULTI, "test", via=registry.voiceprint(1))  # "multiple speakers", under the old id
    assert who(reg, A) == MULTI
    reg.claim(A, "Gus", Basis.OWNER_TAG, "test")
    assert who(reg, A) == "Gus"
    reg.release(A, "test")
    assert who(reg, A) == "Alice"  # the old label doesn't show through once the line's own is gone


def test_a_new_label_outranks_an_old_one_on_its_line(reg):
    reg.claim(registry.utterance(20.0), "Dana", Basis.OWNER_TAG, "test")
    reg.claim(A, "Gus", Basis.OWNER_TAG, "test")
    assert who(reg, A) == "Gus"


def test_an_old_label_on_a_millisecond_two_microphones_share_counts_for_both_until_one_changes(reg, tmp_path):
    # What was said under the legacy millisecond id names every line that starts in that millisecond. What is
    # said from now on is per line, so undoing or replacing it on one line leaves the other.
    reg.claim(registry.utterance(20.0), "Dana", Basis.OWNER_TAG, "test")
    assert (who(reg, A), who(reg, B)) == ("Dana", "Dana")
    reg.release(A, "test")
    assert (who(reg, A), who(reg, B)) == ("Alice", "Dana")
    reg.claim(B, "Gus", Basis.OWNER_TAG, "test")
    assert (who(reg, A), who(reg, B)) == ("Alice", "Gus")
    fresh = registry.Registry(registry.at(tmp_path).log)
    assert fresh.said([(A, 1), (B, 1)]) == reg.said([(A, 1), (B, 1)])


def test_an_old_line_label_on_a_shared_millisecond_is_cleared_for_one_line_alone(reg):
    reg.detach(20.0, MULTI, "test")  # before lines had devices
    reg.claim(A, "Gus", Basis.OWNER_TAG, "test")  # the owner names one of them
    assert (who(reg, A), who(reg, B)) == ("Gus", MULTI)
    reg.release(A, "test")
    assert (who(reg, A), who(reg, B)) == ("Alice", MULTI)


def test_ambiguous_lists_old_ids_that_more_than_one_microphones_line_starts_in(reg, tmp_path):
    reg.claim(registry.utterance(20.0), "Dana", Basis.OWNER_TAG, "test")
    reg.detach(30.0, UNKNOWN, "test")
    reg.claim(registry.utterance(40.0, "Mic A"), "Dana", Basis.OWNER_TAG, "test")  # a new id is never ambiguous
    write_lines(
        tmp_path,
        [
            {"ts_start": 20.0, "device": "Mic A"},
            {"ts_start": 20.0004, "device": "Mic B"},
            {"ts_start": 30.0, "device": "Mic A"},  # one line in its millisecond
            {"ts_start": 40.0, "device": "Mic A"},
            {"ts_start": 40.0, "device": "Mic B"},
        ],
    )
    assert registry.ambiguous(tmp_path) == {"utterance:20.000": ["mic-a@20.000", "mic-b@20.000"]}


def test_ambiguous_never_makes_the_event_log(tmp_path):
    write_lines(tmp_path, [{"ts_start": 20.0, "device": "Mic A"}, {"ts_start": 20.0, "device": "Mic B"}])
    assert registry.ambiguous(tmp_path) == {}
    assert not (tmp_path / "events.sqlite3").exists()


def test_readers_see_a_label_on_only_its_microphones_line_unless_the_other_is_the_same_speech(tmp_path):
    write_voices(tmp_path, {1: {"label": "Alice", "centroid": unit(1, 0), "n": 3}})
    registry.at(tmp_path).claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    apart = [{**line(0, "Alice", 1, 0.9), "device": "Mic A"}, {**line(0, "Alice", 1, 0.9), "device": "Mic B",
             "text": "something else entirely"}]  # fmt: skip
    write_lines(tmp_path, apart)
    registry.at(tmp_path).claim(registry.utterance_of(apart[0]), "Dana", Basis.OWNER_TAG, "dashboard")
    assert [e["speaker"] for e in transcript_view.View(tmp_path).lines(apart)] == ["Dana", "Alice"]
    # The same words on both mics at once are one thing said: the label reaches its copy, each line keeps its key
    twins = [{**line(0, "Alice", 1, 0.9), "device": d} for d in ("Mic A", "Mic B")]
    assert [e["speaker"] for e in transcript_view.View(tmp_path).lines(twins)] == ["Dana", "Dana"]


def test_a_backfill_after_a_label_on_a_microphones_line_keeps_the_label(tmp_path):
    state = legacy_state(tmp_path)
    write_lines(state, [line(3, "Erin", None, None, tagged=True, device="Mic A")])  # hand-tagged before step 4
    e = stored(state)[0]
    registry.at(state).claim(registry.utterance_of(e), "Dana", Basis.OWNER_TAG, "dashboard")
    migrated(state)  # the first pack-registry backfill comes after the label
    assert transcript_view.View(state).name(e) == "Dana"


def test_a_backfilled_hand_tag_stays_on_its_microphones_line(tmp_path):
    state = legacy_state(tmp_path)
    tagged, other = (
        line(3, "Erin", None, None, tagged=True, device="Mic A"),
        line(3, "Speaker 2", 2, 0.6, device="Mic B"),
    )
    write_lines(state, [tagged, other])
    migrated(state)
    assert [transcript_view.View(state).name(e) for e in stored(state)] == ["Erin", "Speaker 2"]


@given(slug=hs.from_regex(r"[a-z0-9]+(-[a-z0-9]+)*", fullmatch=True), ms=hs.integers(0, 4_000_000_000_000))
def test_a_line_key_gives_back_its_device_and_millisecond(slug, ms):
    assert registry.split_key(registry.utterance(ms / 1000, slug).id) == (slug, float(f"{ms / 1000:.3f}"))


def test_a_line_with_no_device_keeps_the_id_lines_had_before():
    assert registry.utterance(1700000000.1234).id == registry.utterance(1700000000.1234, "").id
    assert registry.utterance(1700000000.1234, None).id == "utterance:1700000000.123"
    assert registry.legacy_of(registry.utterance(1700000000.1234, "Mic A")) == registry.utterance(1700000000.1234)
    assert registry.legacy_of(registry.utterance(1700000000.1234)) is None


# ------------------------------------------------------------------ readers ----
AUDIO = {"stream": "mic.audio", "start_ns": 0, "end_ns": 4000000000}
FENCE, GATE = "please fix the garden fence today", "and the gate too, please"
HEARD = [
    {**line(0, "Alice", 1, 0.8, audio=AUDIO), "ts_end": T0 + 4, "text": FENCE},
    {**line(5, "Alice", 1, 0.9, audio=AUDIO), "ts_end": T0 + 9, "text": GATE},
]
NOW = T0 + 10


@pytest.fixture
def heard(tmp_path, dashboard):
    """Every reader that shows or uses a speaker name sees a line's label from the log. The line at T0 was heard as
    Alice's voice; the owner said it was Dana. No file says so: only the log. The state is a subdirectory, so the
    calls directory history answers read beside it (history_qa.calls_dir) is this test's own."""
    s = tmp_path / "state"
    s.mkdir()
    write_voices(s, {1: {"label": "Alice", "centroid": unit(1, 0), "n": 3}, 2: None})
    write_lines(s, HEARD)
    registry.at(s).claim(registry.utterance(T0), "Dana", Basis.OWNER_TAG, "dashboard")
    return dashboard(s)


def rename_alice(state, newcomer=True):
    """People: Rename Alice to Alicia (the registry: voice 1 follows its person), and later People: Add a new
    Alice."""
    reg = registry.at(state)
    reg.rename("Alice", "Alicia", "test")
    write_lines(state, [*HEARD, line(7, "Alicia", 1, 0.9)])
    if newcomer:
        reg.ensure("Alice", "test")


def test_dashboard(heard):
    assert [e["speaker"] for e in transcript_lines.merged_tail(10)] == ["Dana", "Alice"]
    assert [e["speaker"] for e in transcript_lines.transcript_around(T0, 4)["entries"]] == ["Dana", "Alice"]
    assert [r["who"] for r in people_list._room_lines()] == ["Dana", "Alice"]
    latest = speaker_lines.latest_for_speaker(1)
    assert latest is not None
    assert latest["ts_start"] == T0 + 5  # Dana's line is not voice 1's
    clips, _ = unnamed_groups.voice_clips([{"id": 1, "label": "Alice"}])
    assert [c["ts"] for c in clips] == [T0 + 5]


def test_a_voices_latest_line_is_found_past_a_newer_line_with_no_voice(heard):
    # A newer line with audio but no voice of its own raised KeyError, so the People tab couldn't play the voice.
    unvoiced = {"ts_start": T0 + 8, "ts_end": T0 + 9, "speaker": "", "text": "line 8", "audio": AUDIO}
    write_lines(heard, [*HEARD, unvoiced])
    latest = speaker_lines.latest_for_speaker(1)
    assert latest is not None
    assert latest["ts_start"] == T0 + 5


def test_a_voices_newest_clip_heard_recently_is_read_from_the_last_hour(heard, monkeypatch):
    """A voice still being heard costs an hour's read, not a read of every line: the whole log is only opened when
    the last hour holds nothing of it."""
    now = time.time()
    fresh = {
        "ts_start": now - 60,
        "ts_end": now - 58,
        "speaker": "Alice",
        "speaker_id": 1,
        "speaker_sim": 0.9,
        "text": "heard a minute ago",
        "audio": AUDIO,
    }
    write_lines(heard, [*HEARD, fresh])
    real = transcript_lines.heard_lines

    def counted(start: float = 0.0, end: float | None = None):
        assert start != 0.0, "a voice heard a minute ago must not cost a read of every line"
        return real(start, end)

    monkeypatch.setattr(transcript_lines, "heard_lines", counted)
    latest = speaker_lines.latest_for_speaker(1)
    assert latest is not None and latest["ts_start"] == now - 60


def test_a_voices_newest_clip_is_found_in_the_whole_log_when_it_was_long_ago(heard, monkeypatch):
    """And a voice unheard for longer still answers: the hour is tried first, then every line."""
    monkeypatch.setattr(speaker_lines, "LATEST_WINDOW_S", 1.0)  # the two lines in HEARD are hours old
    latest = speaker_lines.latest_for_speaker(1)
    assert latest is not None and latest["ts_start"] == T0 + 5


def test_people_views_keep_a_renamed_persons_lines_from_a_newcomer(heard):
    """The roster, a profile's recent lines, and Evidence read a line heard as Alice as of its time: it stays
    Alicia's after someone new is called Alice."""
    rename_alice(heard)
    assert [r["who"] for r in people_list._room_lines()] == ["Dana", "Alicia", "Alicia"]


def test_presence_follows_a_rename(heard):
    rename_alice(heard)
    assert sorted(c["label"] for c in presence.people(heard, T0 - 60, T0 + 60)) == ["Alicia", "Dana"]


def test_a_visit_named_before_a_rename_follows_it(heard, vision):
    vision.visit("Alice", T0 + 20, T0 + 30, best_sim=0.8, samples=4)
    rename_alice(heard, newcomer=False)  # the visit keeps the name it was recorded under
    chips = {c["label"]: c for c in presence.people(heard, T0 - 60, T0 + 60)}
    assert sorted(chips) == ["Alicia", "Dana"]
    assert "cam0" in chips["Alicia"]["cams"]  # the visit itself, not only her heard line


def test_profiles_follow_a_rename(heard, monkeypatch):
    monkeypatch.setattr(profiles, "STATE", heard)
    rename_alice(heard)
    assert [r["name"] for r in profiles.load_inperson()] == ["dana", "alicia", "alicia"]


def write_call(calls, day, folder, label, text, started=None):
    d = calls / day / folder
    d.mkdir(parents=True)
    (d / "transcript.jsonl").write_text(json.dumps({"t": 1.5, "speaker": "caller", "speaker_label": label,
                                                    "text": text}) + "\n")  # fmt: skip
    if started:
        (d / "call.json").write_text(json.dumps({"started": started}))


def test_profiles_read_call_lines_as_of_the_call(heard, monkeypatch):
    """A caller line keeps the name the bridge heard then. A rebuild reads it as of the call: a call heard as
    Alice before the rename is Alicia's, one after a newcomer took the name is the newcomer's, a name nobody has
    stays as it is, and a forgotten person's calls rebuild no profile."""
    calls = heard / "calls"
    for folder, label, started in (
        ("090100_CA1", "Alice", "2026-10-20T09:01:00"),  # before the rename, call.json written
        ("090200_CA2", "Alice", None),  # still in progress: no call.json yet
        ("090300_CA4", "Zed", None),  # a voice label nobody in the registry has
        ("090400_CA5", "Gus", None),  # someone deleted in People since
    ):
        write_call(calls, "2026-10-20", folder, label, "on the phone", started)
    write_call(calls, "2099-01-01", "120000_CA3", "Alice", "on the phone", "2099-01-01T12:00:00")  # a newcomer's
    reg = registry.at(heard)
    reg.ensure("Gus", "test")
    forgotten.remember(["Gus"], heard)
    reg.forget("Gus", "test")
    rename_alice(heard)
    monkeypatch.setattr(profiles, "STATE", heard)
    monkeypatch.setattr(profiles, "CALLS", calls)
    assert sorted((r["src"], r["name"]) for r in profiles.load_calls()) == [
        ("call:090100_CA1", "alicia"), ("call:090200_CA2", "alicia"), ("call:090300_CA4", "zed"),
        ("call:120000_CA3", "alice"),
    ]  # fmt: skip


def history(state, person, participants=()):
    """'What did X say': the rows found, and the warnings."""
    q = {"person": person, "participants": list(participants), "start": 0, "end": 2e9}
    warnings: list[str] = []
    rows, _ = history_qa._transcripts(state, q, warnings)
    return rows, warnings


def transcript_tool(state, person):
    """Ask's transcript tool: the quotes it found."""
    rows, _ = recall.run("transcript", {"start": 0, "end": 2e9, "person": person}, "q", state, [])
    return [r["quote"] for r in rows]


def test_history_answers(heard):
    rows, _ = history(heard, "Dana")
    assert [(r["speaker"], r["recorded_speaker"]) for r in rows] == [("Dana", "Alice")]


def test_history_answers_follow_a_rename(heard):
    """ "What did X say", and Ask's transcript tool, find a person's lines by person: a renamed person keeps the
    lines heard under the old name, and someone new who takes the old name does not get them."""
    said = lambda person: [(r["speaker"], r["quote"]) for r in history(heard, person)[0]]
    alicia = [("Alicia", GATE), ("Alicia", "line 7")]
    rename_alice(heard, newcomer=False)
    assert said("Alicia") == alicia
    assert said("Alice") == alicia  # the old name still finds her while nobody else has it
    rename_alice(heard)  # People: Add a new Alice
    assert said("Alicia") == alicia
    assert said("Alice") == []
    assert transcript_tool(heard, "Alicia") == [GATE, "line 7"]
    assert transcript_tool(heard, "Alice") == []
    no_exchange = lambda people: any(w.startswith("No planning exchange") for w in history(heard, None, people)[1])
    assert no_exchange(["Alice", "Dana"])  # the new Alice never spoke with Dana
    assert not no_exchange(["Alicia", "Dana"])


def test_history_answers_read_call_lines_by_person(heard):
    """A caller line keeps the name the bridge heard then. "What did X say" and Ask's transcript tool read it as
    of the line's time, as profiles do: a call heard as Alice before the rename is Alicia's, and one after a
    newcomer took the name is the newcomer's."""
    calls = history_qa.calls_dir(heard)

    def call(folder, started, text):
        day = dt.date.fromtimestamp(started).isoformat()
        write_call(calls, day, folder, "Alice", text, dt.datetime.fromtimestamp(started).isoformat())

    def said(person):
        rows = [(r["speaker"], r["quote"]) for r in history(heard, person)[0] if r["source"] == "call"]
        return rows, transcript_tool(heard, person)

    call("090100_CA1", T0 + 60, "fix the fence")  # heard as Alice, before the rename
    rename_alice(heard)  # Alice is Alicia now, and someone new is called Alice
    call("120000_CA2", time.time() + 60, "new here")  # heard as Alice, after the newcomer took the name
    alicia, tool = said("Alicia")
    assert alicia == [("Alicia", "fix the fence")]
    assert "fix the fence" in tool
    assert "new here" not in tool
    alice, tool = said("Alice")
    assert alice == [("Alice", "new here")]
    assert "fix the fence" not in tool


def test_presence_and_summaries(heard):
    assert sorted(c["label"] for c in presence.people(heard, T0 - 60, T0 + 60)) == ["Alice", "Dana"]
    entries = summarize.Transcript(heard).read()
    assert [e["label"] for e in entries] == ["Dana", "Alice"]


def test_whisper_names_identity_bridge_and_clone_say(heard, monkeypatch):
    monkeypatch.setattr(whisper_names, "STATE", heard)
    tail = whisper_names.TranscriptTail()
    tail.seq = 0  # from the start, not from now
    assert [e["name"] for e in tail.events(NOW)] == ["Dana", "Alice"]
    monkeypatch.setattr(bridge, "STATE_DIR", heard)
    with patch.object(time, "time", return_value=NOW):
        other = identity.recent_other_speaker(identity.Paths(heard), exclude_labels=("alice",))
        room = bridge.room_transcript_line()
    assert other is None  # Dana's line has no voice of its own, and Alice is excluded
    assert "Dana: please fix" in room
    monkeypatch.setattr(clone_say, "STATE", heard)
    assert [e["ts_start"] for e in clone_say.lines() if e["speaker_id"] == 1] == [T0 + 5]


def test_lines_read_as_stored_without_a_log(heard, monkeypatch):
    # No event log to ask (the broker isn't there): a reader shows each line as it was stored, and connects to nothing.
    monkeypatch.setattr(registry, "log_exists", lambda state=None: False)
    monkeypatch.setattr(registry, "at", lambda state=None: pytest.fail("a reader never opens the log on its own"))
    assert [e["speaker"] for e in transcript_view.view(HEARD, heard)] == ["Alice", "Alice"]


def test_every_line_a_reader_sees_carries_its_key(tmp_path):
    lines = [{"ts_start": 20.0, "device": "Mic A", "text": "a"}, {"ts_start": 21.0, "text": "b"}]
    assert [e["line"] for e in transcript_view.View(tmp_path).lines(lines)] == ["mic-a@20.000", "21.000"]


@pytest.mark.parametrize("device", [None, "", "  ", [], {"unusable": "device"}])
def test_transcript_subject_reuse_accepts_devices_without_a_string_name(tmp_path, device):
    write_voices(tmp_path, {1: "Alice"})
    sample = line(70, "Speaker 1", 1, 0.8, device=device)
    view = transcript_view.View(tmp_path)
    assert view.line(sample)["speaker"] == "Alice"
    assert view.explain(sample) is not None
    assert view.line(sample)["line"] == f"{T0 + 70:.3f}"


@pytest.mark.parametrize("method", ["lines", "individual_lines"])
def test_reused_transcript_subjects_keep_new_input_and_identity_projections_fresh(tmp_path, method):
    def read(items):
        return getattr(transcript_view.View(tmp_path), method)(items)

    write_voices(tmp_path, {1: "Alice", 2: "Dana", 3: "Eve"})
    sample = line(80, "Speaker 1", 1, 0.8, device="Mic A")
    speech = speech_fake.fake()
    speech.say(sample)
    reg = registry.at(tmp_path)
    assert read([sample])[0]["speaker"] == "Alice"
    reg.rename("Alice", "Erin", "test")
    assert read([sample])[0]["speaker"] == "Erin"
    subject = registry.utterance(T0 + 80, "Mic A")
    reg.claim(subject, "Dana", Basis.CORRECTION, "test")
    assert read([sample])[0]["speaker"] == "Dana"
    reg.retract(subject, "test")
    assert read([sample])[0]["speaker"] == "Erin"
    speech.purge(speech_log.UTTERANCES, lambda e: True)
    assert read(speech.range(0, T0 + 100)) == []
    speech.say({**sample, "speaker_id": 3, "text": "new recording"})
    [projected] = read(speech.range(0, T0 + 100))
    assert (projected["speaker"], projected["speaker_id"], projected["text"]) == ("Eve", 3, "new recording")
