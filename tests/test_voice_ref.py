"""Recording the owner's voice reference from the settings page: GET and POST /config/voice_ref."""

import http.client
import io
import json
import subprocess
import sys
import urllib.parse

import numpy as np
import pytest
import soundfile as sf

from liberational import config as cfg
from liberational.dashboard import server, settings

ENROLL = [sys.executable, "-m", "liberational.people.voice_edits"]


def wav(seconds, rate=16000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * rate), np.float32), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def ref(tmp_path):
    """Where the owner's reference is saved, for an owner named Alice."""
    path = tmp_path / "voices" / "owner_ref.wav"
    with cfg.override({"owner": {"name": "Alice", "voice_ref": str(path)}}):
        yield path


@pytest.fixture
def enrolled(monkeypatch):
    """The (name, clip) of every enroll the handler starts; any other command runs as usual."""
    calls = []
    real = subprocess.run

    def run(cmd, *args, **kwargs):
        if cmd[:3] != ENROLL:
            return real(cmd, *args, **kwargs)
        calls.append(cmd[-2:])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(settings.subprocess, "run", run)
    return calls


@pytest.fixture
def call(serve, monkeypatch, ref, enrolled):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    port = urllib.parse.urlsplit(serve(server.H)).port

    def request(method, body=None, path="/config/voice_ref", **headers):
        headers.setdefault("Host", f"127.0.0.1:{port}")
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request(method, path, body, headers)
        r = c.getresponse()
        out = r.status, r.read()
        c.close()
        return out

    return request


def test_saves_the_clip_with_its_transcript_and_enrolls_the_owner(call, ref, enrolled):
    code, body = call("POST", wav(10), **{"Content-Type": "audio/wav"})
    assert code == 200, body
    assert json.loads(body)["enrolled"]
    assert sf.info(str(ref)).duration == pytest.approx(10, abs=0.05)
    assert "Liberational" in ref.with_suffix(".txt").read_text()
    assert enrolled == [["Alice", str(ref)]]
    info = json.loads(call("GET")[1])
    assert (info["exists"], info["seconds"]) == (True, 10.0)
    assert call("GET", path="/config/voice_ref?audio=1")[0] == 200


def test_says_whether_the_reference_is_missing_or_cant_be_read(call, ref):
    assert json.loads(call("GET")[1])["problem"] == "missing"
    ref.parent.mkdir(parents=True)
    ref.write_bytes(b"not a wav at all")
    assert (json.loads(call("GET")[1])["exists"], json.loads(call("GET")[1])["problem"]) == (False, "unreadable")
    ref.write_bytes(wav(10))
    assert json.loads(call("GET")[1])["problem"] is None


OK = {"Content-Type": "audio/wav"}
# These gates reject from headers, before reading an upload. Sending a large WAV concurrently with the early close
# races the socket and can reset the client before it reads the valid 403 response. Advertise the same valid upload
# length, but send headers only: rejection must not wait for audio.
UPLOAD = {**OK, "Content-Length": str(len(wav(10)))}


@pytest.mark.parametrize(
    "body, headers, code",
    [
        pytest.param(None, {**UPLOAD, "X-Forwarded-For": "100.64.0.9"}, 403, id="remote"),
        pytest.param(None, {**UPLOAD, "Host": "evil.example"}, 403, id="rebound-host"),
        pytest.param(None, {**UPLOAD, "Origin": "http://evil.example"}, 403, id="cross-origin"),
        pytest.param(None, {**UPLOAD, "Content-Type": "text/plain"}, 403, id="not-audio"),
        pytest.param(b"not a wav", OK, 400, id="not-a-wav"),
        pytest.param(wav(3), OK, 400, id="too-short-to-clone-from"),
    ],
)
def test_refuses_remote_rebound_cross_origin_and_bad_audio(call, ref, enrolled, body, headers, code):
    assert call("POST", body, **headers)[0] == code
    assert not ref.exists()
    assert enrolled == []
