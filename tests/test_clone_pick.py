"""Pick clips (#312): the owner picks a voice clone's reference from the person's recent lines. The reference is
built from exactly those lines, with their own words and no voice check, the old one is moved aside only once the
new one is written, and each picked line becomes the owner's word that it is theirs. Synthetic audio and a temp
state dir; never the real data/."""

import json

import numpy as np
import pytest

import speech_fake
from liberational import config as cfg
from liberational.dashboard import server
from liberational.domain.identity import Basis
from liberational.listen import revisions
from liberational.people import registry
from liberational.speech import clone_say
from voices import write_voices

SR = clone_say.SR


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Voice 3, Carly; voice 5, unnamed; and audio that cuts as a tone as long as each line."""
    for name, path in (
        ("STATE", tmp_path),
        ("OUT", tmp_path / "clones"),
    ):
        monkeypatch.setattr(clone_say, name, path)
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(clone_say, "cut", lambda e: np.full(int(clone_say.dur(e) * SR), 0.1, np.float32))
    write_voices(tmp_path, {3: {"label": "Carly", "centroid": [1.0, 0, 0, 0]}, 5: None})
    return tmp_path


def line(ts, sid=3, dur=4.0, words="this is one of her lines here", sim=0.4, **kw):
    return {
        "ts_start": ts, "ts_end": ts + dur, "speaker_id": sid, "speaker_sim": sim, "device": "kitchen", "text": words,
        "audio": {"stream": "kitchen.audio", "start_ns": int(ts * 1e9), "end_ns": int((ts + dur) * 1e9)}, **kw,
    }  # fmt: skip


def write(state, lines):
    speech_fake.fake().write(lines)


def key(ts):
    return registry.line_key(ts, "kitchen")


def test_her_recent_lines_are_offered_newest_first_weak_matches_too(state):
    write(
        state,
        [
            line(100.0),
            line(200.0, sim=0.1),  # a weak match: still hers to pick, she hears it and decides
            line(300.0, dur=1.0),  # too short to carry a voice
            line(400.0, dur=20.0),  # too long to be one person
            line(500.0, words="hi"),  # too few words
            line(600.0, sid=5),  # someone else's voice
        ],
    )
    offered = clone_say.pickable(3)
    assert [o["line"] for o in offered] == [key(200.0), key(100.0)]
    assert offered[0] == {
        "line": key(200.0),
        "ts_start": 200.0,
        "text": "this is one of her lines here",
        "dur_s": 4.0,
        "confirmed": False,
    }


def test_picking_clips_makes_the_reference_from_exactly_them(state):
    write(
        state, [line(100.0, words="first one of the clips"), line(200.0), line(300.0, words="third of the clips here")]
    )
    old = state / "clones" / "3.wav"
    old.parent.mkdir()
    old.write_bytes(b"RIFF old")
    done = clone_say.pick(3, [key(300.0), key(100.0), key(200.0)])
    # three 4 s clips: 12 s of speech, the pauses between not counted
    assert (done["seconds"], done["clips"]) == (12.0, 3)
    wav = state / "clones" / "3.wav"
    assert wav.read_bytes() != b"RIFF old"
    assert (
        state / "clones" / "3.txt"
    ).read_text() == "third of the clips here first one of the clips this is one of her lines here"
    meta = json.loads((state / "clones" / "3.json").read_text())
    assert (meta["label"], meta["source"], meta["lines"]) == ("Carly", "picked", [key(300.0), key(100.0), key(200.0)])
    assert list((state.parent / "premove" / "clones").glob("3-*.wav"))  # the old reference is kept aside
    assert clone_say.ref_for(3, make=False) == wav  # and it is the reference the warm voice reads next


def test_it_stops_once_the_reference_is_long_enough(state):
    write(state, [line(100.0 + 10 * k, dur=6.0) for k in range(4)])
    done = clone_say.pick(3, [key(100.0 + 10 * k) for k in range(4)])
    assert done["clips"] == 2  # 12 s after two 6 s clips: the rest aren't needed


@pytest.mark.parametrize(
    ("picked", "error"),
    [
        pytest.param([key(100.0)], "a reference needs 12 s", id="too-little"),
        pytest.param([key(100.0), key(600.0)], "isn't Carly's", id="someone-elses-line"),
        pytest.param([key(100.0), "kitchen@1.000"], "isn't Carly's", id="no-such-line"),
    ],
)
def test_what_cant_make_a_reference_changes_nothing(state, picked, error):
    write(state, [line(100.0), line(600.0, sid=5)])
    with pytest.raises(ValueError, match=error):
        clone_say.pick(3, picked)
    assert not (state / "clones" / "3.wav").exists()


def test_a_voice_made_from_a_recording_is_never_replaced_by_picks(state):
    write(state, [line(100.0 + 10 * k) for k in range(4)])
    (state / "clones").mkdir()
    (state / "clones" / "3.json").write_text(json.dumps({"label": "Carly", "source": "recorded"}))
    with pytest.raises(ValueError, match="made from a recording"):
        clone_say.pick(3, [key(100.0 + 10 * k) for k in range(4)])


def test_the_route_makes_the_reference_and_records_each_pick_as_hers(state):
    write(state, [line(100.0 + 10 * k) for k in range(4)])
    code, out = server.clone_pick(3, [key(100.0 + 10 * k) for k in range(4)])
    assert (code, out["ok"], out["clips"]) == (200, True, 3)
    reg = registry.at(state)
    said = reg.person_of(registry.utterance(100.0, "kitchen"))
    assert (said.person, said.winner.basis if said.winner else None) == (reg.person_id("Carly"), Basis.OWNER_TAG)
    assert reg.person_of(registry.utterance(130.0, "kitchen")).person is None  # not used, so not claimed
    assert server.clone_pick(3, ["kitchen@1.000"])[0] == 409
    assert server.clone_pick(3, [])[0] == 400
    assert server.clone_pick(True, [key(100.0)])[0] == 400


# ---- from the local review ----
def test_a_pick_that_fails_keeps_the_old_reference_where_it_is(state):
    write(state, [line(100.0)])
    (state / "clones").mkdir()
    (state / "clones" / "3.wav").write_bytes(b"RIFF old")
    with pytest.raises(ValueError, match="needs 12 s"):
        clone_say.pick(3, [key(100.0)])
    assert (state / "clones" / "3.wav").read_bytes() == b"RIFF old"  # never moved aside for nothing
    assert not list((state / "clones").glob(".*"))  # and no temp files left


def test_a_line_labeled_as_them_from_another_voice_is_offered(state):
    write(state, [line(100.0, sid=5)])
    registry.at(state).claim(registry.utterance(100.0, "kitchen"), "Carly", Basis.OWNER_TAG, "dashboard")
    assert [o["line"] for o in clone_say.pickable(3)] == [key(100.0)]
    assert [o["line"] for o in clone_say.pickable(5)] == []  # and not to the voice it was heard on


def test_the_reference_never_runs_past_fifteen_seconds(state):
    write(state, [line(100.0, dur=11.9), line(200.0, dur=15.0)])
    done = clone_say.pick(3, [key(100.0), key(200.0)])
    assert done["seconds"] == clone_say.PREP_MAX_S  # the second clip cut to fit, not 26.9 s


def test_the_owners_own_voice_is_never_picked_for(state, tmp_path):
    ref = tmp_path / "owner.wav"
    ref.write_bytes(b"RIFF")
    write(state, [line(100.0 + 10 * k) for k in range(3)])
    with (
        cfg.override({"owner": {"name": "Carly", "voice_ref": str(ref)}}),
        pytest.raises(ValueError, match="recorded reference"),
    ):
        clone_say.pick(3, [key(100.0 + 10 * k) for k in range(3)])


def test_a_rename_keeps_the_picked_reference(state):
    write(state, [line(100.0 + 10 * k) for k in range(3)])
    clone_say.pick(3, [key(100.0 + 10 * k) for k in range(3)])
    registry.at(state).rename("Carly", "Caitlin", "test")
    assert clone_say.ref_for(3, make=True) == state / "clones" / "3.wav"  # not parked and rebuilt from room speech
    assert json.loads((state / "clones" / "3.json").read_text())["label"] == "Caitlin"


def test_picked_lines_still_count_as_the_voices_for_a_build(state):
    write(state, [line(100.0 + 10 * k) for k in range(3)])  # weak matches: not usable for a build before
    assert clone_say.utterances(3) == []
    assert server.clone_pick(3, [key(100.0 + 10 * k) for k in range(3)])[0] == 200
    assert len(clone_say.utterances(3)) == 3  # the owner's word, and still on the voice


def test_rejecting_picked_transcript_hides_reference_until_undo_without_deleting_it(state):
    raw = [line(100.0 + 10 * k) for k in range(3)]
    write(state, raw)
    clone_say.pick(3, [key(e["ts_start"]) for e in raw])
    reference = state / "clones" / "3.wav"
    original = reference.read_bytes()
    token = revisions.reject_lines(state, raw[:1])
    assert clone_say.ref_for(3, make=False) is None
    with pytest.raises(SystemExit, match="rejected transcript"):
        clone_say.ref_for(3)
    assert reference.read_bytes() == original
    revisions.undo_correction(state, token)
    assert clone_say.ref_for(3, make=False) == reference


def test_a_pick_waits_its_turn_behind_a_build(state):
    write(state, [line(100.0 + 10 * k) for k in range(3)])
    with server.CLONE_LOCK:
        code, out = server.clone_pick(3, [key(100.0 + 10 * k) for k in range(3)])
    assert (code, out["error"]) == (409, "A voice is being built right now; try again in a moment.")
    assert not (state / "clones" / "3.wav").exists()


def test_an_unnamed_voice_is_picked_for_and_nobody_is_named(state):
    write(state, [line(100.0 + 10 * k, sid=5) for k in range(3)])
    assert server.clone_pick(5, [key(100.0 + 10 * k) for k in range(3)])[0] == 200
    assert registry.at(state).person_of(registry.utterance(100.0, "kitchen")).person is None
