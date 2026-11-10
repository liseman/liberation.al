"""POST /speakers/me ("that was me"): name a voice as the owner's, beside any voice of theirs (naming replaces
merging, #772), with the microphone service left running."""

import subprocess
from unittest.mock import create_autospec

import pytest

from liberational import config as cfg
from liberational.dashboard import server
from speech_fake import fake
from voices import write_voices


@pytest.fixture
def ran(tmp_path, monkeypatch):
    """The voice and service commands the dashboard runs, each from the module's name on (none really run)."""
    ran = []

    def run(cmd, **kw):
        ran.append(cmd[cmd.index("-m") + 2 :])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    fake = create_autospec(subprocess)  # scoped to the server module, unlike patching subprocess.run
    fake.run.side_effect = run
    monkeypatch.setattr(server, "subprocess", fake)
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server.services, "installed", lambda: ["ambient", "vision"])
    monkeypatch.setattr(server.services, "processes", dict)
    with cfg.override({"owner": {"name": "Alice"}}):
        yield ran


@pytest.mark.parametrize(
    "owners_voice", [pytest.param("Alice", id="beside-the-owners"), pytest.param(None, id="the-owner-has-none-yet")]
)
def test_names_the_voice_as_the_owner_with_the_mic_service_running(tmp_path, ran, owners_voice):
    write_voices(tmp_path, {1: owners_voice, 2: None})
    fake().say({"ts_start": 10.0, "text": "Hi.", "speaker_id": 2})
    code, body = server.confirm_owner_voice(2)
    assert (code, body["ok"]) == (200, True)
    (label,) = ran  # no service stopped or started around it
    assert label[-3:] == ["label", "2", "Alice"]


@pytest.mark.parametrize(
    ("sid", "owner", "code"),
    [
        pytest.param(1, "Alice", 200, id="already-the-owners"),
        pytest.param(9, "Alice", 404, id="no-such-voice"),
        pytest.param("x", "Alice", 400, id="bad-id"),
        pytest.param(2, "", 400, id="no-owner-set"),
    ],
)
def test_changes_nothing_for_the_owners_own_voice_or_a_bad_request(tmp_path, ran, sid, owner, code):
    write_voices(tmp_path, {1: "Alice", 2: None})
    with cfg.override({"owner": {"name": owner}}):
        assert server.confirm_owner_voice(sid)[0] == code
    assert ran == []
