"""Clone speech from the voice service's warm model: speech/clone_server.py in the voice service, and
dashboard/clone_stream.py with POST /clones/say and GET /clones/render in the dashboard. A fake model stands in for
Qwen3-TTS; it yields half-second chunks as the real one does."""

import http.client
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
import soundfile as sf

from liberational import config as cfg
from liberational.dashboard import clone_stream, server
from liberational.speech import clone_say, clone_server

SR = 24000
CHUNKS = [np.full(SR // 2, 0.25, np.float32), np.full(SR // 2, -0.5, np.float32)]


def fake_synth(text, ref, ref_text, max_tokens):
    assert (text, ref_text) == ("Hello there.", "the reference's words")
    yield from ((c, SR) for c in CHUNKS)


def broken_synth(*_):
    raise RuntimeError("out of memory")
    yield  # a generator, as the real one is


@pytest.fixture
def clones(tmp_path):
    folder = tmp_path / "state" / "clones"
    folder.mkdir(parents=True)
    (folder / "7.wav").write_bytes(b"RIFF")
    with cfg.override({"paths": {"state": str(tmp_path / "state")}, "owner": {"voice_ref": str(tmp_path / "me.wav")}}):
        yield folder


@pytest.fixture
def voice(clones, monkeypatch):
    """The voice service's endpoint on a free port, with its loop thread making clone jobs with synth."""
    synth = {"fn": fake_synth}
    monkeypatch.setattr(clone_server, "_engine", lambda: "cuda")
    srv = clone_server.serve(0)
    assert srv is not None
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            clone_server.run_pending(0.05, lambda *a: synth["fn"](*a))  # the model as the test has it now

    looping = threading.Thread(target=loop, daemon=True)
    looping.start()
    with cfg.override({"ports": {"voice": srv.server_port}}):
        yield synth
    stop.set()
    looping.join(5)  # or it could take the next test's job with this test's model
    srv.shutdown()
    srv.server_close()


def post_clone(body):
    c = http.client.HTTPConnection("127.0.0.1", cfg.get("ports.voice"), timeout=10)
    c.request("POST", "/clone", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read()


GOOD = {"text": "Hello there.", "ref_text": "the reference's words", "max_tokens": 100}


def test_the_voice_service_streams_the_line_as_16_bit_pcm(voice, clones):
    status, headers, body = post_clone({**GOOD, "ref": str(clones / "7.wav")})
    assert (status, headers["Content-Type"], headers["X-Sample-Rate"]) == (200, "audio/L16", "24000")
    pcm = np.frombuffer(body, "<i2")
    assert len(pcm) == SR
    assert (pcm[0], pcm[-1]) == (8191, -16383)  # 0.25 and -0.5 of full scale


@pytest.mark.parametrize(
    "ref",
    [
        pytest.param("{clones}/8.wav", id="no-such-clip"),
        pytest.param("{clones}/../clones/7.txt", id="not-a-wav"),
        pytest.param("/etc/hostname", id="outside-the-clones"),
    ],
)
def test_the_voice_service_refuses_a_reference_that_isnt_a_clone_clip(voice, clones, ref):
    status, _, body = post_clone({**GOOD, "ref": ref.format(clones=clones)})
    assert status == 400
    assert "ref" in json.loads(body)["error"]


def test_the_owners_own_recording_is_a_reference_too(clones):
    me = cfg.path("owner.voice_ref")
    me.write_bytes(b"RIFF")
    assert clone_server.allowed_ref(str(me)) == me.resolve()


def test_a_failure_before_any_audio_is_an_error_not_silence(voice, clones):
    voice["fn"] = broken_synth
    status, _, body = post_clone({**GOOD, "ref": str(clones / "7.wav")})
    assert (status, json.loads(body)["error"]) == (500, "out of memory")


def test_a_job_whose_listener_left_stops_making_audio():
    made = []

    def synth(*_):
        for c in CHUNKS:
            made.append(c)
            yield c, SR

    job = clone_server.Job("Hi.", "ref.wav", "words", 100)
    job.cancelled.set()
    clone_server.run_job(job, synth)
    assert len(made) == 1  # the first chunk is made, then the job sees nobody is listening
    assert job.out.get_nowait() is None


def floats(b):
    return np.frombuffer(b, "<f4").tolist()


def test_a_render_streams_the_same_from_the_start_to_a_late_listener_and_ends_as_a_file(tmp_path):
    r = clone_stream.Render(tmp_path / "say-7-1.wav")
    early = r.stream()
    r.add(np.array([16384, 16384], "<i2").tobytes(), SR)  # 0.5, 0.5
    assert next(early) == clone_stream.stream_header(2 * SR)
    assert floats(next(early)) == [0.5, 0.5, 0.5, 0.5]
    r.add(np.array([-16384], "<i2").tobytes(), SR)  # -0.5: the midpoint from 0.5 comes first
    r.finish()
    assert floats(b"".join(early)) == [0.0, -0.5]
    late = b"".join(r.stream())
    assert late[:44] == clone_stream.stream_header(2 * SR)
    assert floats(late[44:]) == [0.5, 0.5, 0.5, 0.5, 0.0, -0.5]
    audio, sr = sf.read(r.path, dtype="int16")  # the file keeps the model's own rate and 16 bits
    assert (sr, audio.tolist()) == (SR, [16384, 16384, -16384])


@pytest.mark.parametrize(
    ("agent", "waits"),
    [
        pytest.param("Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15", True, id="safari"),
        pytest.param("Mozilla/5.0 (X11) AppleWebKit/537.36 Chrome/141.0 Safari/537.36", False, id="chrome"),
        pytest.param("Mozilla/5.0 (Macintosh) Gecko/20100101 Firefox/144.0", False, id="firefox"),
    ],
)
def test_only_safari_waits_for_the_finished_file(agent, waits):
    assert clone_stream.waits(agent) is waits


def test_the_dashboard_says_a_line_from_the_warm_voice_and_streams_it(voice, clones, serve, monkeypatch):
    more = threading.Event()  # the second half second is made only once the browser is listening

    def synth(*_):
        yield CHUNKS[0], SR
        assert more.wait(5)
        yield CHUNKS[1], SR

    voice["fn"] = synth
    monkeypatch.setattr(server, "CLONES", clones)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    monkeypatch.setattr(clone_stream, "_reference", lambda sid: (clones / f"{sid}.wav", "the reference's words"))
    monkeypatch.setattr(server, "_clone_run", lambda *a: pytest.fail("the old way ran"))
    base = serve(server.H)
    req = urllib.request.Request(
        base + "/clones/say",
        json.dumps({"id": 7, "text": "Hello there."}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        said = json.loads(r.read())
    assert said["ok"] and said["streamed"]
    assert said["url"].startswith("/clones/render?f=say-7-")
    with urllib.request.urlopen(base + said["url"], timeout=10) as r:
        assert r.read(44) == clone_stream.stream_header(2 * SR)  # playing before the line is finished
        more.set()
        audio = np.frombuffer(r.read(), "<f4")
    assert len(audio) == 2 * SR  # a second of speech at 48 kHz
    assert (audio[0], audio[-1]) == (pytest.approx(0.25, abs=1e-4), pytest.approx(-0.5, abs=1e-4))  # 16-bit steps
    name = said["url"].split("f=")[1]
    deadline = time.monotonic() + 5
    while not (clones / name).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sf.info(clones / name).frames == SR  # and kept as a render, as before


def test_without_the_warm_voice_the_line_is_made_the_old_way(clones, serve, monkeypatch):
    monkeypatch.setattr(server, "CLONES", clones)
    monkeypatch.setattr(clone_stream, "_reference", lambda sid: (clones / f"{sid}.wav", "words"))
    ran = []
    monkeypatch.setattr(server, "_clone_run", lambda args, timeout: ran.append(args[0]) or (0, {"ms": 1}, []))
    with cfg.override({"ports": {"voice": 9}}):  # nothing listens there
        code, body = server.clone_say(7, "Hello there.")
    assert (code, ran, body.get("streamed")) == (200, ["say"], None)


def test_a_voice_whose_clip_isnt_ready_is_made_the_old_way(voice, clones, monkeypatch):
    monkeypatch.setattr(clone_stream, "_reference", lambda sid: None)
    assert not clone_stream.start(7, "Hello there.", clones / "say-7-2.wav")


@pytest.fixture
def windows(tmp_path, monkeypatch):
    from liberational.speech import playback_guard

    path = tmp_path / "playback_windows.json"
    monkeypatch.setattr(playback_guard, "PATH", path)
    return lambda: playback_guard._read(path)


def test_the_playback_window_covers_the_longest_line_then_its_real_length(voice, clones, windows, monkeypatch):
    monkeypatch.setattr(clone_stream, "_reference", lambda sid: (clones / f"{sid}.wav", "the reference's words"))
    out = clones / "say-7-3.wav"
    assert clone_stream.start(7, "Hello there.", out)
    r = clone_stream.live(out.name)
    assert r is not None
    r.wait(5)
    # Once the line is done, a window covers its real length (a second of speech, the lag, the tail), and none still
    # runs for the longest it could have been. One window or two, depending on whether the line finished before the
    # first was opened.
    longest = clone_say.token_budget("Hello there.") / 12 + clone_stream.PLAY_LAG_S
    now = time.time()
    rows = windows()
    assert rows and max(w["end"] for w in rows) <= now + 1.0 + clone_stream.PLAY_LAG_S + 0.6
    assert all(w["end"] - w["start"] < longest for w in rows)


def test_a_line_that_fails_partway_is_an_error_and_isnt_kept(voice, clones, windows, serve, monkeypatch):
    def half(*_):
        yield CHUNKS[0], SR
        raise RuntimeError("out of memory")

    voice["fn"] = half
    monkeypatch.setattr(server, "CLONES", clones)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    monkeypatch.setattr(clone_stream, "_reference", lambda sid: (clones / f"{sid}.wav", "the reference's words"))
    out = clones / "say-7-4.wav"
    assert clone_stream.start(7, "Hello there.", out)
    r = clone_stream.live(out.name)
    assert r is not None
    r.wait(5)
    assert (r.error is not None, out.exists()) == (True, False)
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(serve(server.H) + f"/clones/render?f={out.name}", timeout=5)
    assert e.value.code == 502


def test_a_line_the_dashboard_gave_up_on_stops_being_made(voice, clones, monkeypatch):
    made, go = [], threading.Event()

    def slow(*_):
        assert go.wait(5)  # the first audio comes only after the dashboard gave up
        for c in CHUNKS * 4:
            made.append(c)
            yield c, SR
            time.sleep(0.05)

    voice["fn"] = slow
    monkeypatch.setattr(clone_stream, "FIRST_AUDIO_S", 0.2)
    monkeypatch.setattr(clone_stream, "_reference", lambda sid: (clones / f"{sid}.wav", "the reference's words"))
    out = clones / "say-7-5.wav"
    assert not clone_stream.start(7, "Hello there.", out)
    go.set()
    time.sleep(1.0)
    assert len(made) < len(CHUNKS * 4)  # stopped once nobody read it
    assert not out.exists()


def test_a_render_whose_file_cant_be_written_still_lets_its_listeners_go(tmp_path):
    r = clone_stream.Render(tmp_path / "gone" / "say-7-6.wav")  # no such folder
    r.add(b"\x01\x00", SR)
    r.finish()
    assert r.done and "couldn't save" in (r.error or "")
    assert b"".join(r.stream())  # ends instead of waiting forever
    assert not list(tmp_path.rglob("*.part"))


def test_the_voice_loop_makes_one_line_between_transcript_reads(monkeypatch):
    monkeypatch.setattr(clone_server, "JOBS", clone_server.queue.Queue(maxsize=4))
    jobs = [clone_server.Job("Hi.", "r.wav", "w", 100) for _ in range(2)]
    for j in jobs:
        clone_server.JOBS.put(j)
    assert clone_server.run_pending(0.01, lambda *a: iter([(CHUNKS[0], SR)]))
    assert clone_server.JOBS.qsize() == 1


@pytest.mark.parametrize(
    ("headers", "code"),
    [
        pytest.param({"Host": "evil.example:8776", "Content-Type": "application/json"}, 403, id="rebound-host"),
        pytest.param({"Content-Type": "text/plain"}, 403, id="a-simple-cross-site-post"),
    ],
)
def test_the_voice_endpoint_only_takes_local_json(voice, clones, headers, code):
    c = http.client.HTTPConnection("127.0.0.1", cfg.get("ports.voice"), timeout=10)
    c.request("POST", "/clone", json.dumps({**GOOD, "ref": str(clones / "7.wav")}), headers)
    assert c.getresponse().status == code


def test_a_full_queue_is_busy_not_a_pile_up(voice, clones, monkeypatch):
    full = clone_server.queue.Queue(maxsize=1)
    full.put(clone_server.Job("Hi.", "r.wav", "w", 100))
    monkeypatch.setattr(clone_server, "JOBS", full)
    status, _, body = post_clone({**GOOD, "ref": str(clones / "7.wav")})
    assert (status, json.loads(body)["error"]) == (503, "the voice service is busy; try again")
