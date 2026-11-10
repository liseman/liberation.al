"""Which clip a voice is cloned from: the owner's own recording, else a reference made for that exact voice."""

import json
import os
from pathlib import Path

import pytest

from liberational import config as cfg
from liberational.dashboard import server
from liberational.people import registry
from liberational.speech import clone_say
from voices import write_voices


def test_missing_speakers_means_no_clone_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(clone_say, "STATE", tmp_path)
    monkeypatch.setattr(clone_say, "OUT", tmp_path / "clones")
    voices, canon = clone_say.speakers()
    assert voices == {}
    assert canon(7) == 7
    assert clone_say.candidates() == []
    assert server.clones_list() == {"clones": [], "error": None}


@pytest.fixture
def owner_ref(tmp_path):
    ref = tmp_path / "voices" / "owner_ref.wav"
    ref.parent.mkdir()
    ref.write_bytes(b"RIFF")
    return ref


@pytest.fixture
def out(tmp_path, monkeypatch, owner_ref):
    state = tmp_path / "state"
    (state / "clones").mkdir(parents=True)
    write_voices(state, {1: "Alice", 4: "Dana"})
    monkeypatch.setattr(clone_say, "STATE", state)
    monkeypatch.setattr(clone_say, "OUT", state / "clones")
    with cfg.override({"owner": {"name": "Alice", "voice_ref": str(owner_ref)}}):
        yield state / "clones"


@pytest.fixture
def built(monkeypatch, out):
    """The ids a reference is built for, without reading audio or loading speech recognition."""
    built = []
    monkeypatch.setattr(clone_say, "build", lambda sid: built.append(sid) or out / f"{sid}.wav")
    return built


def test_the_owner_clones_from_their_own_recording(out, owner_ref, built):
    (out / "1.wav").write_bytes(b"someone else")  # an old clip at the owner's id is ignored
    assert clone_say.ref_for(1) == owner_ref
    assert built == []


def test_a_clip_made_for_someone_else_is_moved_aside_and_rebuilt(tmp_path, out, built):
    (out / "4.wav").write_bytes(b"old")
    (out / "4.json").write_text(json.dumps({"label": "Eli"}))
    assert clone_say.ref_for(4, make=False) is None
    clone_say.ref_for(4)
    assert built == [4]
    assert not (out / "4.wav").exists()
    assert len(list((tmp_path / "premove" / "clones").glob("4-*"))) == 2


def test_parking_the_same_id_twice_in_one_second_keeps_both_clips(tmp_path, out, built):
    for clip in (b"first", b"second"):
        (out / "4.wav").write_bytes(clip)
        (out / "4.json").write_text(json.dumps({"label": "Eli"}))
        for f in (out / "4.wav", out / "4.json"):
            os.utime(f, (1_700_000_000, 1_700_000_000))
        clone_say.ref_for(4)
    parked = tmp_path / "premove" / "clones"
    assert sorted(f.read_bytes() for f in parked.glob("4-*.wav")) == [b"first", b"second"]


def test_a_confirmed_only_reference_for_this_voice_is_reused(out, built):
    (out / "4.wav").write_bytes(b"dana")
    clone_say.write_meta(4, "Dana", "transcript")
    assert clone_say.ref_for(4, make=False) == out / "4.wav"
    assert clone_say.ref_for(4) == out / "4.wav"
    assert built == []


@pytest.mark.parametrize("source", ["recorded", "picked"])
def test_explicit_reference_needs_no_automatic_selection_policy(out, built, source):
    (out / "4.wav").write_bytes(b"selected recording")
    (out / "4.txt").write_text("Words the owner selected.")
    (out / "4.json").write_text(json.dumps({"label": "Dana", "source": source}))
    assert clone_say.ref_for(4, make=False) == out / "4.wav"
    assert clone_say.ref_for(4, make=True) == out / "4.wav"
    assert built == []


@pytest.mark.parametrize(
    "meta",
    [
        {"label": "Dana", "source": "transcript"},
        {"label": "Dana", "source": "transcript", "reference_policy": "automatic-matches"},
        {"label": "Dana"},
        None,
        [],
    ],
)
def test_legacy_automatic_references_are_not_offered(out, built, meta):
    (out / "4.wav").write_bytes(b"unreviewed speech")
    (out / "4.json").write_text(json.dumps(meta))
    assert clone_say.ref_for(4, make=False) is None
    assert built == []
    assert (out / "4.wav").read_bytes() == b"unreviewed speech"


def test_legacy_automatic_reference_is_parked_before_a_confirmed_rebuild(tmp_path, out, owner_ref):
    (out / "4.wav").write_bytes(b"unreviewed speech")
    (out / "4.json").write_text(json.dumps({"label": "Dana", "source": "transcript"}))
    with pytest.raises(SystemExit, match="Name more of their lines"):
        clone_say.ref_for(4, make=True)
    assert not (out / "4.wav").exists()
    assert [p.read_bytes() for p in (tmp_path / "premove" / "clones").glob("4-*.wav")] == [b"unreviewed speech"]
    assert owner_ref.read_bytes() == b"RIFF"


def test_candidates_count_a_voice_with_a_reference_as_ready(out, built, monkeypatch):
    """The owner's recording makes them cloneable with no clean lines; a voice with no lines and no clip is left out."""
    got = {c["id"]: c for c in clone_say.candidates()}
    assert list(got) == [1]
    assert (got[1]["ready"], got[1]["built"], got[1]["clean_s"]) == (True, True, 0.0)


def test_clone_render_cleanup_keeps_new_low_id_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CLONES", tmp_path)
    for i in range(20):
        old = tmp_path / f"say-99-{1000 + i}.wav"
        old.write_bytes(b"old audio")
        os.utime(old, (1000 + i, 1000 + i))
    reference = tmp_path / "1.wav"
    reference.write_bytes(b"reference")

    def render(args, timeout):
        Path(args[-1]).write_bytes(b"new audio")
        return 0, {"ms": 1}, []

    monkeypatch.setattr(server, "_clone_run", render)
    code, result = server.clone_say(1, "A short test.")
    assert code == 200
    assert (tmp_path / result["url"].split("f=")[1]).read_bytes() == b"new audio"
    assert len(list(tmp_path.glob("say-*.wav"))) == 20
    assert not (tmp_path / "say-99-1000.wav").exists()
    assert (tmp_path / "say-99-1019.wav").exists()
    assert reference.read_bytes() == b"reference"


def test_clone_render_cleanup_survives_a_render_removed_meanwhile(tmp_path, monkeypatch):
    # Two renders finishing together: the other one's cleanup can remove a file this one has listed but not read.
    monkeypatch.setattr(server, "CLONES", tmp_path)
    (tmp_path / "say-5-1.wav").symlink_to(tmp_path / "gone.wav")  # listed by the glob, but stat() fails

    def render(args, timeout):
        Path(args[-1]).write_bytes(b"new audio")
        return 0, {"ms": 1}, []

    monkeypatch.setattr(server, "_clone_run", render)
    code, result = server.clone_say(1, "A short test.")
    assert code == 200
    assert (tmp_path / result["url"].split("f=")[1]).read_bytes() == b"new audio"


@pytest.mark.parametrize("shared", [True, False])
def test_clone_choices_deduplicate_the_resolved_reference_not_the_name(out, owner_ref, monkeypatch, shared):
    speakers = {1: {"id": 1, "label": "Alice"}, 2: {"id": 2, "label": "Alice"}}
    other = out / "other.wav"
    other.write_bytes(b"RIFF distinct reference")
    alias = out / "alias.wav"
    alias.symlink_to(owner_ref)
    monkeypatch.setattr(clone_say, "speakers", lambda raw=None, view=None: (speakers, lambda sid: sid))
    monkeypatch.setattr(clone_say, "lines", lambda raw=None, view=None: [])
    monkeypatch.setattr(
        clone_say, "ref_for", lambda sid, make=False, known=None: owner_ref if sid == 1 else alias if shared else other
    )
    assert [c["id"] for c in clone_say.candidates()] == ([1] if shared else [1, 2])


@pytest.fixture
def retained_clone(out):
    def save(label="Eli", source="recorded"):
        ref = out / "9.wav"
        ref.write_bytes(b"saved clone audio")
        ref.with_suffix(".txt").write_text("Words from the saved recording.")
        meta = ref.with_suffix(".json")
        meta.write_text(json.dumps({"label": label, "source": source}))
        return ref, meta

    return save


@pytest.mark.parametrize("source", ["recorded", "picked"])
def test_clone_list_keeps_saved_name_without_voiceprint_binding(retained_clone, source):
    retained_clone(source=source)
    assert registry.voice_names(clone_say.STATE, [9])[9] is None

    response = server.clones_list()
    assert response["error"] is None
    choices = {c["id"]: c for c in response["clones"]}
    assert choices[9]["label"] == "Eli"
    assert choices[9]["ready"] and choices[9]["built"]
    assert registry.voice_names(clone_say.STATE, [9])[9] is None


@pytest.mark.parametrize("source", ["recorded", "picked"])
def test_using_unbound_clone_does_not_erase_its_saved_name(retained_clone, source):
    ref, meta = retained_clone(source=source)
    before = meta.read_bytes()

    assert clone_say.ref_for(9) == ref

    assert meta.read_bytes() == before
    assert ref.read_bytes() == b"saved clone audio"
    assert ref.with_suffix(".txt").read_text() == "Words from the saved recording."
    assert registry.voice_names(clone_say.STATE, [9])[9] is None


@pytest.mark.parametrize("label", [None, 12, "", "   ", []])
def test_missing_or_invalid_saved_clone_name_stays_anonymous(retained_clone, label):
    retained_clone(label=label)
    choices = {c["id"]: c for c in clone_say.candidates()}
    assert choices[9]["label"] == "Speaker 9"


@pytest.mark.parametrize("source", ["recorded", "picked"])
def test_current_voiceprint_name_wins_over_saved_clone_name(retained_clone, source):
    ref, meta = retained_clone(source=source)
    write_voices(clone_say.STATE, {9: "Dana"})
    choices = {c["id"]: c for c in clone_say.candidates()}
    assert choices[9]["label"] == "Dana"
    assert clone_say.ref_for(9) == ref
    assert json.loads(meta.read_text())["label"] == "Dana"


def test_preparing_unbound_recording_keeps_its_saved_name(retained_clone, monkeypatch):
    ref, _meta = retained_clone()
    ref.with_suffix(".txt").unlink()
    prepared = []

    def prepare(sid, src, label=None, source="recorded"):
        prepared.append((sid, src, label))
        return src

    monkeypatch.setattr(clone_say, "prepare", prepare)
    assert clone_say.ref_for(9) == ref
    assert prepared == [(9, ref, "Eli")]


def test_saved_owner_name_does_not_replace_clone_audio_with_owner_reference(retained_clone, owner_ref):
    ref, _meta = retained_clone(label="Alice")
    choices = {c["id"]: c for c in clone_say.candidates()}
    assert choices[1]["label"] == choices[9]["label"] == "Alice"
    assert clone_say.ref_for(1) == owner_ref
    assert clone_say.ref_for(9) == ref
