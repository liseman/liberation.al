"""People's inbox: Skip keeps a face, voice, or line out for good, and "clear" brings them all back."""

from types import SimpleNamespace

import pytest

from liberational.dashboard import (
    context,
    inbox_state,
    people_list,
    roster_read,
    unnamed_groups,
)
from liberational.people import voice_edits
from speech_fake import fake
from voices import write_voices


@pytest.fixture(autouse=True)
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(inbox_state, "INBOX_SKIPS", tmp_path / "inbox_skips.json")
    return tmp_path


@pytest.mark.parametrize(
    "kind,ident,control,code",
    [
        pytest.param("face", 3, False, 403, id="not-from-this-machine"),
        pytest.param("hat", 3, True, 400, id="unknown-kind"),
        pytest.param("line", "soon", True, 400, id="id-not-a-number"),
        pytest.param("line", True, True, 400, id="id-a-bool"),
    ],
)
def test_bad_skips_are_refused(kind, ident, control, code):
    assert inbox_state.inbox_skip(kind, ident, control)[0] == code
    assert inbox_state.inbox_skips() == {"face": [], "voice": [], "line": []}


def test_skips_persist_once_each_and_clear_brings_them_back():
    assert inbox_state.inbox_skip("face", 3, True) == (200, {"ok": True, "skipped": 1})
    assert inbox_state.inbox_skip("face", 3, True)[1]["skipped"] == 1
    assert inbox_state.inbox_skip("line", 1792816328.9, True)[1]["skipped"] == 2
    assert inbox_state.inbox_skips() == {"face": [3.0], "voice": [], "line": [1792816328.9]}
    assert inbox_state.inbox_skip("clear", None, True) == (200, {"ok": True, "skipped": 0})
    assert inbox_state.inbox_skips() == {"face": [], "voice": [], "line": []}


def test_skipped_things_stay_out_of_the_inbox(state, monkeypatch):
    write_voices(state, {1: "Alice"}, source=voice_edits.ENROLL_SOURCE)  # learned from a recording
    write_voices(state, {2: None, 3: None})
    audio = {"stream": "mic.audio", "start_ns": 0, "end_ns": 1000000000}
    lines = [
        {"ts_start": 100.5, "speaker_id": 2, "speaker_sim": 0.9, "text": "kept", "audio": audio},
        {"ts_start": 101.5, "speaker_id": 2, "speaker_sim": 0.9, "text": "skipped", "audio": audio},
        {"ts_start": 102.5, "speaker_id": 1, "speaker_sim": 0.4, "text": "a named voice", "audio": audio},
        {"ts_start": 103.5, "speaker_id": 3, "speaker_sim": 0.9, "text": "voice 3", "audio": audio},
    ]
    fake().write(lines)
    groups = [{"id": 7, "count": 5}, {"id": 8, "count": 4}]
    faces = SimpleNamespace(counts=dict, rename=lambda a, b: 0, unknown_groups=lambda: groups)
    inbox_state.inbox_skip("line", 101.5, True)
    inbox_state.inbox_skip("voice", 3, True)
    inbox_state.inbox_skip("face", 8, True)
    monkeypatch.setattr(context, "STATE", state)
    monkeypatch.setattr(people_list, "_faces_mod", lambda: faces)  # the face model
    view = roster_read.people_view()
    assert [v["id"] for v in view["unknown"]["voices"]] == [2]
    assert [f["id"] for f in view["unknown"]["faces"]] == [7]
    # The line answered "Nobody" is not offered as a clip of voice 2; its neighbour is.
    assert [c["text"] for c in view["unknown"]["voice_clips"]] == ["kept"]


def test_a_line_put_aside_by_key_leaves_the_other_microphones_line_and_a_time_reaches_both():
    a = {"ts_start": 2000.0, "device": "Mic A"}
    b = {"ts_start": 2000.0, "device": "Mic B"}
    assert inbox_state.inbox_skip("line", "mic-a@2000.000", True)[0] == 200
    skipped = set(inbox_state.inbox_skips()["line"])
    assert (unnamed_groups.put_aside(a, skipped), unnamed_groups.put_aside(b, skipped)) == (True, False)
    assert inbox_state.inbox_skip("line", 2000.0, True)[0] == 200  # as Inbox sent it before per-device line keys
    skipped = set(inbox_state.inbox_skips()["line"])
    assert (unnamed_groups.put_aside(a, skipped), unnamed_groups.put_aside(b, skipped)) == (True, True)


def test_a_line_with_no_device_is_put_aside_by_its_key():
    assert inbox_state.inbox_skip("line", "2000.000", True)[0] == 200
    assert unnamed_groups.put_aside({"ts_start": 2000.0}, set(inbox_state.inbox_skips()["line"]))
    assert inbox_state.inbox_skip("line", "not a key", True)[0] == 400
