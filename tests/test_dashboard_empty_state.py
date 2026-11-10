"""Page data endpoints on a fresh install, before any voice or recording exists."""

import http.client
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest

from liberational import config as cfg
from liberational.assistant import voice
from liberational.dashboard import context, server
from liberational.memory import timeline
from liberational.speech import clone_say


@pytest.fixture
def empty_api(tmp_path, monkeypatch, serve):
    state = tmp_path / "state"
    old_state = context.STATE
    for name, value in vars(server).copy().items():
        if isinstance(value, Path) and value.is_relative_to(old_state):
            monkeypatch.setattr(server, name, state / value.relative_to(old_state))
    monkeypatch.setattr(clone_say, "STATE", state)
    monkeypatch.setattr(clone_say, "OUT", state / "clones")
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    # A stand-in for Kokoro's downloaded voices file (an .npz of one array per voice), so /config/voices never opens
    # the real one under data/models.
    kokoro_voices = tmp_path / "voices.npz"
    np.savez(kokoro_voices, af_heart=np.zeros(2))
    overrides = {
        "paths": {"state": str(state)},
        "summary": {"auto": False},
        "models": {"files": {"tts_kokoro_voices": {"path": str(kokoro_voices)}}},
    }
    with cfg.override(overrides):
        port = urlsplit(serve(server.H)).port

        def get(path):
            # Generous, so a loaded machine can't fail a request that is only slow; a hang still ends the test.
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            conn.request("GET", path)
            response = conn.getresponse()
            body = json.loads(response.read())
            conn.close()
            return response.status, body

        yield get


@pytest.mark.parametrize(
    "path,key,empty",
    [
        ("/scene", None, []),
        ("/ambient", None, {"devices": {}}),  # no microphone stream on the broker yet
        ("/vision", "cameras", {}),  # no camera stream on the broker yet
        ("/transcript", "entries", []),
        ("/transcript/day?day=2026-01-01", "entries", []),
        ("/audio/spans?day=2026-01-01", "spans", []),
        ("/clones", "clones", []),
        ("/calls", None, []),
        ("/speakers", "speakers", []),
        ("/speakers/refine", "items", []),
        ("/journal/evidence?day=2026-01-01", "sources", []),  # no day page: nothing cited yet
    ],
)
def test_page_data_is_empty_before_first_capture(empty_api, path, key, empty):
    status, body = empty_api(path)
    assert status == 200
    assert (body[key] if key else body) == empty
    assert not isinstance(body, dict) or not body.get("error")


@pytest.mark.parametrize(
    "path",
    [
        "/people",
        "/inbox/skips",
        "/speakers/refine/log",
        "/health",
        "/config",
        "/config/voice_ref",
        "/config/voices",
        "/review/presence?day=2026-01-01",
    ],
)
def test_page_queries_do_not_require_existing_state(empty_api, path):
    status, body = empty_api(path)
    assert status == 200
    assert not isinstance(body, dict) or not body.get("error")


def test_voice_reader_keeps_unknown_utterance_without_speaker_database(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "STATE_DIR", tmp_path)
    with cfg.override({"owner": {"name": "Alice"}}):
        assert voice._voiceprints() == []
        entry = voice.canonical_entry({"speaker_id": 1, "speaker": "Speaker 1", "text": "Hello"})
    assert entry == {"speaker_id": 1, "speaker": "Speaker 1", "text": "Hello", "owner_voice": False}


def test_timeline_queries_are_empty_without_voice_or_event_files(tmp_path):
    assert timeline.encounters(state=tmp_path) == []
    assert timeline.present(0, 100, state=tmp_path) == []
