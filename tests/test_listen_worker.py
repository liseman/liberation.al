import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import continuum_py
import numpy as np
import pytest

import continuum_broker
from liberational import speech_log
from liberational.listen import voices
from liberational.listen.gate import VAD
from liberational.listen.pcm import Format
from liberational.listen.worker import (
    Candidate,
    Heard,
    Mic,
    MicPipeline,
    StateChecks,
    Transcriber,
    Twin,
    Worker,
    mics_from_manifest,
    ring_batches,
)

NS = 1_000_000_000
RATE = 16_000
BLOCK = RATE // 100  # 10 ms
FMT = Format.of("PCM_S16LEx1", 16 + BLOCK * 2, block_ms=10)


class FixedClock:
    """Host monotonic ns to UTC by a fixed offset, so expected times can be worked out by hand."""

    offset = 1_800_000_000_000_000_000

    def utc_ns(self, host_ns: int) -> int:
        return host_ns + self.offset


def speech(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (amplitude * rng.standard_normal(int(seconds * RATE))).astype(np.float32)


def blocks(audio: np.ndarray) -> list[np.ndarray]:
    return [audio[i : i + BLOCK] for i in range(0, len(audio) - BLOCK + 1, BLOCK)]


def pipeline() -> MicPipeline:
    return MicPipeline(Mic("testmic"), FMT, FixedClock(), VAD)  # ty: ignore[invalid-argument-type]


def test_a_burst_of_speech_becomes_one_utterance_at_its_utc_time() -> None:
    quiet = np.zeros(RATE, np.float32) + 1e-4
    audio = np.concatenate([quiet, speech(1.0), quiet])
    p = pipeline()
    found = []
    for i, block in enumerate(blocks(audio)):
        found += p.feed(host_ns=i * 10_000_000, first=i * BLOCK, x=block)
    (c,) = found
    assert c.mic == "testmic.audio"
    # The speech runs from 1.0 s to 2.0 s; v1's VAD pads 300 ms each side and needs a few frames to open and close.
    assert c.start_ns - FixedClock.offset == pytest.approx(0.7e9, abs=0.1e9)
    assert c.end_ns - FixedClock.offset == pytest.approx(2.3e9, abs=0.35e9)
    assert len(c.audio) == pytest.approx((c.end_ns - c.start_ns) * RATE / 1e9, abs=RATE // 50)


def test_a_gap_mid_speech_drops_that_speech_instead_of_splicing_it() -> None:
    quiet = np.zeros(RATE, np.float32) + 1e-4
    audio = np.concatenate([quiet, speech(1.0), quiet])
    p = pipeline()
    found = []
    for i, block in enumerate(blocks(audio)):
        skip = 150 if i >= 150 else 0  # 1.5 s in, mid-speech, a whole 1.5 s of records goes missing
        found += p.feed(host_ns=(i + skip) * 10_000_000, first=(i + skip) * BLOCK, x=block)
    assert p.breaks == 1
    assert all(c.start_ns - FixedClock.offset >= 2.9e9 for c in found), "nothing spans the gap"


def test_mics_come_from_capture_s_manifest_with_their_block_and_channel(tmp_path: Path) -> None:
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        '[[device]]\nname = "facecam"\nkind = "camera"\nmatch = { usb = "0fd9:0078" }\n'
        '[[device]]\nname = "wave3"\nkind = "microphone"\nmatch = { name = "Wave 3" }\n'
        '[[device]]\nname = "xvf"\nkind = "microphone"\nmatch = { name = "XVF" }\naudio = { block_ms = 20 }\n'
    )
    mics = mics_from_manifest(manifest, {"xvf": 1})
    assert mics == [Mic("wave3", 10, 0), Mic("xvf", 20, 1)]
    assert mics[0].match == {"name": "Wave 3"}  # how auto gain finds its PipeWire node


# ---- End to end: a broker, capture's microphone path playing a file, and the worker


def _binary(name: str) -> Path | None:
    if name == "continuum-broker":
        return Path(continuum_broker.broker_binary())
    repo = Path(__file__).resolve().parents[1]
    common = Path(
        subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=repo, capture_output=True, text=True
        ).stdout.strip()
        or ".git"
    )
    for target in (repo / "continuum" / "target", (repo / common).resolve() / "continuum-target"):
        path = target / "debug" / name
        if path.is_file():
            return path
    return None


@pytest.fixture
def binary_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    monkeypatch.setitem(globals(), "__file__", str(repo / "tests" / "test_listen_worker.py"))
    monkeypatch.setattr(continuum_broker, "__file__", str(repo / "tests" / "continuum_broker.py"))
    run = subprocess.run

    def git_or_help(args, **kwargs):
        if args[0] == "git":
            return subprocess.CompletedProcess(args, 0, stdout=".git\n", stderr="")
        return run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", git_or_help)
    return repo


def test_compatible_release_is_not_shadowed_by_stale_debug_without_override(
    binary_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = binary_tree / ".git/continuum-target/debug/continuum-broker"
    release = binary_tree / "continuum/target/release/continuum-broker"
    shared.parent.mkdir(parents=True)
    release.parent.mkdir(parents=True)
    shared.write_text(f"#!{sys.executable}\nprint('--socket')\n")
    release.write_text(f"#!{sys.executable}\nprint('--socket --tier0-classes')\n")
    shared.chmod(0o755)
    release.chmod(0o755)
    monkeypatch.delenv("CONTINUUM_BROKER", raising=False)
    try:
        selected = _binary("continuum-broker")
    except pytest.skip.Exception as error:
        pytest.fail(f"compatible release was skipped: {error}")
    assert selected == release


def test_explicit_broker_wins_over_stale_local_and_shared_builds(
    binary_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = binary_tree / "continuum" / "target" / "debug" / "continuum-broker"
    shared = binary_tree / ".git" / "continuum-target" / "debug" / "continuum-broker"
    release = binary_tree / "release" / "continuum-broker"
    local.parent.mkdir(parents=True)
    shared.parent.mkdir(parents=True)
    release.parent.mkdir(parents=True)
    local.write_bytes(b"stale local build")
    shared.write_bytes(b"stale shared build")
    release.write_text(f"#!{sys.executable}\nprint('--socket --tier0-classes')\n")
    release.chmod(0o755)
    monkeypatch.setenv("CONTINUUM_BROKER", str(release))
    assert _binary("continuum-broker") == release


def test_broker_override_does_not_replace_the_microphone_example(
    binary_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mic = binary_tree / "continuum" / "target" / "debug" / "examples" / "pcm_file_mic"
    release = binary_tree / "continuum-broker"
    mic.parent.mkdir(parents=True)
    mic.write_bytes(b"microphone example")
    release.write_bytes(b"broker")
    monkeypatch.setenv("CONTINUUM_BROKER", str(release))
    assert _binary("examples/pcm_file_mic") == mic


@pytest.mark.parametrize("override", ["missing", "."])
def test_missing_or_non_file_override_keeps_the_shared_build_fallback(
    binary_tree: Path, monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    shared = binary_tree / ".git" / "continuum-target" / "debug" / "continuum-broker"
    shared.parent.mkdir(parents=True)
    shared.write_text(f"#!{sys.executable}\nprint('--socket --tier0-classes')\n")
    shared.chmod(0o755)
    monkeypatch.setenv("CONTINUUM_BROKER", str(binary_tree / override))
    assert _binary("continuum-broker") == shared


DIE_WITH_PARENT = ["setpriv", "--pdeathsig", "KILL"] if sys.platform == "linux" and shutil.which("setpriv") else []


@pytest.fixture
def broker(tmp_path: Path) -> Iterator[Path]:
    yield from continuum_broker.start_broker(tmp_path)


@pytest.fixture
def mic_binary() -> Path:
    binary = _binary("examples/pcm_file_mic")
    if binary is None:
        pytest.skip("needs pcm_file_mic example built (cargo build -p continuum-capture --examples)")
    return binary


def test_speech_played_through_a_microphone_is_committed_as_an_utterance(
    broker: Path, tmp_path: Path, mic_binary: Path
) -> None:
    pcm, store = tmp_path / "speech.s16", tmp_path / "segments"
    quiet = np.zeros(RATE // 2, np.float32)
    audio = np.concatenate([quiet, speech(1.2), quiet])
    pcm.write_bytes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())

    heard: list[int] = []

    def fake_asr() -> object:
        def transcribe(audio16k: np.ndarray) -> str:
            heard.append(len(audio16k))
            return "hello there"

        return transcribe

    async def run() -> tuple[list[dict], int]:
        client = await continuum_py.Client.connect(str(broker), client_name="listen-test")
        transcriber = Transcriber(fake_asr)  # ty: ignore[invalid-argument-type]
        worker = Worker(client, [Mic("testmic")], transcriber, vad=lambda mic: VAD(), segments=store)
        stop = asyncio.Event()
        try:
            running = asyncio.create_task(worker.run(stop))
            deadline = time.monotonic() + 15
            while worker.committed == 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            stop.set()
            await asyncio.wait_for(running, 10)
        finally:
            transcriber.close()
        lines = await speech_log.read_range(0, time.time() + 60, base=store)
        await client.close()
        return lines, worker.committed

    # The microphone registers its stream first (it prints "playing" once it has), so the worker can subscribe to it.
    # Its stderr goes to a file (a pipe nobody reads could fill), which says why when it doesn't start.
    errors = tmp_path / "mic.stderr"
    with errors.open("wb") as stderr:
        player = subprocess.Popen(
            [*DIE_WITH_PARENT, mic_binary, "--socket", broker, "--name", "testmic", "--rate", "16000", pcm],
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
    try:
        assert player.stdout is not None
        if player.stdout.readline().strip() != b"playing":
            player.wait(timeout=10)
            why = errors.read_text(errors="replace").strip()
            if "IncompatibleSchema" in why:
                # A build artifact older than the broker, like an unbuilt one: an environment to refresh, not a bug.
                pytest.skip(
                    f"{mic_binary} predates the broker's schema; rebuild it (cargo build -p continuum-capture --examples): {why}"
                )
            pytest.fail(f"pcm_file_mic didn't start: {why or f'exit {player.returncode}, no stderr'}")
        lines, committed = asyncio.run(run())
    finally:
        player.terminate()
        player.wait(timeout=10)
    assert committed == 1 and len(heard) == 1
    (line,) = lines
    assert line["text"] == "hello there" and line["device"] == "testmic"
    assert line["audio"]["stream"] == "testmic.audio"
    assert abs(line["ts_start"] - time.time()) < 60, "a UTC time, now-ish"
    assert line["ts_end"] - line["ts_start"] == pytest.approx(1.2 + 0.6, abs=0.5)


def state_checks(state: Path) -> StateChecks:
    salt, iterations = b"test salt only", 100_000
    digest = hashlib.pbkdf2_hmac("sha256", b"open sesame", salt, iterations).hex()
    verifier = state / "voice_auth.json"
    verifier.write_text(json.dumps({"version": 1, "salt": salt.hex(), "iterations": iterations, "hash": digest}))
    verifier.chmod(0o600)
    return StateChecks(state)


def test_a_span_overlapping_playback_is_echo(tmp_path: Path) -> None:
    from liberational.speech import playback_guard

    checks = state_checks(tmp_path)
    now = time.time()
    token = playback_guard.open_window(2.0)
    try:
        assert checks.echo(now + 0.5, now + 1.0)
        assert not checks.echo(now - 100, now - 99)
    finally:
        playback_guard.close_window(token)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("(keyboard clicking)", None, id="a-caption-alone"),
        pytest.param("[BLANK_AUDIO]", None, id="blank-audio"),
        pytest.param("(laughs) okay then", ("okay then", False), id="caption-stripped"),
    ],
)
def test_captions_are_not_speech(tmp_path: Path, text: str, expected: tuple[str, bool] | None) -> None:
    assert state_checks(tmp_path).heard("wave3", 10.0, 11.0, text) == expected


def test_another_mics_copy_of_a_line_just_heard_is_dropped(tmp_path: Path) -> None:
    checks = state_checks(tmp_path)
    assert checks.heard("wave3", 100.0, 102.0, "turn the lights off") == ("turn the lights off", False)
    assert checks.heard("xvf", 100.1, 102.1, "turn the lights off") == Twin("wave3", 100.0), "names the line it copies"
    assert checks.heard("xvf", 200.0, 202.0, "turn the lights off") == ("turn the lights off", False), "much later"


def test_the_voice_password_is_written_as_password_and_marked(tmp_path: Path) -> None:
    checks = state_checks(tmp_path)
    assert checks.heard("wave3", 10.0, 11.0, "Password is open sesame.") == ("[password]", True)
    assert checks.heard("wave3", 20.0, 21.0, "password is wrong guess") == ("password is wrong guess", False)


def test_every_setting_pack_listen_reads_has_a_default() -> None:
    # pack-listen crash-looped on a setting read at startup that config/defaults.toml lacked (2026-10-30): a key
    # missing from the defaults raises KeyError, whatever pack.toml holds.
    import re

    from liberational import config as cfg
    from liberational.audio import ambient, sensitivity
    from liberational.listen import worker
    from liberational.speech import asr

    sources = [Path(m.__file__ or "") for m in (worker, ambient, sensitivity, asr)]
    keys = {k for s in sources for k in re.findall(r'cfg\.get\("([a-z_.]+)"', s.read_text())}
    assert "audio.mic_channel" in keys
    for key in sorted(keys):
        cfg.get(key)


# ---- Levels: a quiet microphone's speech reaches the gate and ASR at a normal level


def test_a_quiet_candidate_is_brought_to_a_normal_level_with_bounded_gain_and_no_clipping() -> None:
    from liberational.listen import worker as w

    quiet = speech(1.0, amplitude=10 ** (-59 / 20))  # the Wave 3's speech, about -59 dBFS
    assert w.dbfs(w.level(quiet)) == pytest.approx(w.TARGET_DBFS, abs=0.5)
    hum = speech(1.0, amplitude=1e-6)  # -120 dBFS: raised by no more than MAX_GAIN_DB
    assert w.dbfs(w.level(hum)) == pytest.approx(w.dbfs(hum) + w.MAX_GAIN_DB, abs=0.5)
    spiky = np.zeros(16000, np.float32)
    spiky[100] = 0.5  # one click in silence: levelling it to -23 dBFS would clip, so the peak is held
    assert float(np.max(np.abs(w.level(spiky)))) <= w.PEAK + 1e-6


def test_quiet_speech_gets_past_a_level_sensitive_gate_and_is_transcribed(caplog: pytest.LogCaptureFixture) -> None:
    # On devbox nothing was transcribed: Silero found no speech in -59 dBFS audio until it was levelled.
    from liberational.listen import worker as w

    heard = []

    def gate(audio: np.ndarray) -> bool:
        return w.dbfs(audio) > -40  # like Silero on near-silent audio

    t = Transcriber(lambda: lambda audio: heard.append(audio) or "hello", gate=gate)
    try:
        got = asyncio.run(t(speech(1.0, amplitude=10 ** (-59 / 20)), "wave3.audio"))
        noise = asyncio.run(t(speech(1.0, amplitude=1e-7), "wave3.audio"))
    finally:
        t.close()
    assert got is not None and got.text == "hello"
    assert noise is None, "near-silence stays below the gate"
    assert "wave3.audio: 1.0 s at -59.0 dBFS: heard" in caplog.text
    assert "dBFS: not speech" in caplog.text


def test_each_microphone_logs_its_levels_every_so_often(caplog: pytest.LogCaptureFixture, monkeypatch) -> None:
    from liberational.listen import worker as w

    monkeypatch.setattr(w, "LEVELS_S", 1)
    caplog.set_level("INFO")
    p = pipeline()
    for i, block in enumerate(blocks(speech(1.5, amplitude=0.01))):
        p.feed(host_ns=i * 10_000_000, first=i * BLOCK, x=block)
    (line,) = [r.getMessage() for r in caplog.records if "levels p50" in r.getMessage()]
    assert line.startswith("testmic.audio: levels p50 -40.") and "candidates" in line


# ---- Auto gain (#798): each microphone's gain rides along in its pipeline


class FakeControl:
    """A mic's own capture volume, 0 to 40 dB, held here."""

    kind, lo, hi, step_db = "hardware", 0.0, 40.0, 0.0

    def __init__(self, db: float) -> None:
        self.db = db

    def get_db(self) -> float:
        return self.db

    def set_db(self, db: float) -> None:
        self.db = db


@pytest.fixture
def pack_toml(tmp_path, monkeypatch):
    from liberational import config as cfg

    path = tmp_path / "pack.toml"
    monkeypatch.setattr(cfg, "local_config", lambda: path)
    return path


def test_unverified_candidates_do_not_change_microphone_gain(pack_toml) -> None:
    from liberational.audio import gain

    control = FakeControl(10.0)
    now = [0.0]
    mic_gain = gain.MicGain("testmic", lambda: "testmic-node", lambda device: control)
    p = MicPipeline(Mic("testmic"), FMT, FixedClock(), VAD, mic_gain, monotonic=lambda: now[0])  # ty: ignore[invalid-argument-type]
    quiet = np.zeros(RATE, np.float32) + 1e-4
    audio = np.concatenate([quiet, speech(1.0), quiet, quiet, quiet, quiet, quiet])
    for i, block in enumerate(blocks(audio)):
        now[0] = i / 100
        p.feed(host_ns=i * 10_000_000, first=i * BLOCK, x=block)
    assert p.candidates == 1
    assert control.db == 10.0, "an energy candidate is not confirmed speech"


def test_software_gain_reaches_the_samples_the_pipeline_hears(pack_toml, monkeypatch, caplog) -> None:
    from liberational.audio import gain
    from liberational.listen import worker as w

    pack_toml.write_text("[audio.mic_auto_gain]\ntestmic = true\n")
    monkeypatch.setattr(w, "LEVELS_S", 1)
    caplog.set_level("INFO")
    mic_gain = gain.MicGain("testmic", lambda: "testmic-node", lambda device: gain.SoftwareGain(db=20.0))
    p = MicPipeline(Mic("testmic"), FMT, FixedClock(), VAD, mic_gain)  # ty: ignore[invalid-argument-type]
    p.feed(host_ns=0, first=0, x=np.zeros(BLOCK, np.float32))  # its first look finds the stage
    p.levels.clear()
    for i, block in enumerate(blocks(speech(1.5, amplitude=0.001)), start=1):
        p.feed(host_ns=i * 10_000_000, first=i * BLOCK, x=block)
    (line,) = [r.getMessage() for r in caplog.records if "levels p50" in r.getMessage()]
    assert line.startswith("testmic.audio: levels p50 -40.")  # -60 dBFS as captured, +20 dB
    assert line.endswith("software gain +20.0 dB")


def test_the_vads_floor_moves_with_the_gain_so_a_step_up_is_not_heard_as_speech(pack_toml) -> None:
    # devbox: the Wave 3 went from 0 to 30 dB, every frame read as loud, and the VAD stayed open in 20 s stretches.
    from liberational.audio import gain

    control = FakeControl(0.0)
    now = [0.0]
    mic_gain = gain.MicGain("testmic", lambda: "testmic-node", lambda device: control)
    p = MicPipeline(Mic("testmic"), FMT, FixedClock(), VAD, mic_gain, monotonic=lambda: now[0])  # ty: ignore[invalid-argument-type]
    room = speech(20.0, amplitude=0.001)  # about -60 dBFS of room noise
    found = []
    for i, block in enumerate(blocks(room)):
        now[0] = i / 100
        if i == 300:
            control.db = 30.0  # moved by hand, 3 s in; read back at the next look (every REREAD_S)
        found += p.feed(host_ns=i * 10_000_000, first=i * BLOCK, x=block * (31.62 if i >= 300 else 1.0))
    assert p.vad.floor == pytest.approx(-30.0, abs=2.0)  # ty: ignore[unresolved-attribute]
    assert not p.vad.in_speech  # ty: ignore[unresolved-attribute]
    late = [c for c in found if c.start_ns - FixedClock.offset > (gain.REREAD_S + 1) * 1e9]
    assert late == [], "room noise at the new gain is not speech"


def test_a_gain_change_moves_the_vads_floor_by_as_much_and_says_so(caplog: pytest.LogCaptureFixture) -> None:
    class OneStep:  # a MicGain (hardware: no software factor) that changed by -3 dB once
        factor = 1.0

        def __init__(self) -> None:
            self.changes = [-3.0]

        def heard(self, audio, sr, ended) -> None: ...
        def tick(self, now, in_speech) -> None: ...
        def take_change(self) -> float:
            return self.changes.pop() if self.changes else 0.0

    from liberational.listen.gate import VAD as GateVAD

    caplog.set_level("INFO")
    p = MicPipeline(Mic("testmic"), FMT, FixedClock(), GateVAD, gain=OneStep())  # ty: ignore[invalid-argument-type]
    before = p.vad.floor  # ty: ignore[unresolved-attribute]
    p.feed(host_ns=0, first=0, x=np.zeros(BLOCK, np.float32) + 1e-4)
    assert p.vad.floor == pytest.approx(before - 3.0, abs=0.1)  # ty: ignore[unresolved-attribute]
    assert "testmic.audio: gain -3.0 dB: VAD floor" in caplog.text


def header(stream_id: int, parents: list[str] | None = None) -> bytes:
    import uuid

    h: continuum_py.PayloadHeader = {
        "id": str(uuid.uuid7()),
        "stream_id": stream_id,
        "start_ns": NS,
        "end_ns": 2 * NS,
        "layer": "Layer1Raw",
        "format": "RawBytes",
        "shape_name": "embedding/campp/v1",
        "parents": parents or [],
        "subjects": [],
        "seq": 0,
    }
    return continuum_py.encode_header(h)


def test_microphone_waits_for_capture_to_register_its_ring(monkeypatch):
    notice = header(7)

    class Subscription:
        def __init__(self, rings):
            self.rings = rings

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def __aiter__(self):
            yield {"type": "Event", "item": {"header": notice}}
            await asyncio.Event().wait()

    class Client:
        def __init__(self):
            self.subscriptions = 0

        async def subscribe(self, streams, *, pinned=False):
            self.subscriptions += 1
            rings = [] if self.subscriptions == 1 else [{"stream_id": 7, "lease": {"ring_stride": 336}}]
            return Subscription(rings)

        def ring_reader(self, lease):
            return object()

    async def run():
        client = Client()
        w = Worker(
            cast(continuum_py.Client, client),
            [Mic("testmic")],
            Transcriber(lambda: lambda a: ""),
            vad=lambda mic: VAD(),
        )
        reading = asyncio.Event()

        async def read(*args):
            reading.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(w, "_read", read)
        mic = asyncio.create_task(w._mic(Mic("testmic")))
        try:
            await asyncio.wait_for(reading.wait(), 2)
            assert client.subscriptions == 2
        finally:
            mic.cancel()
            await asyncio.gather(mic, return_exceptions=True)
            w.transcriber.close()

    asyncio.run(run())


def test_microphone_failure_stops_the_worker_instead_of_leaving_it_healthy(monkeypatch):
    async def run():
        w = Worker(
            cast(continuum_py.Client, None), [Mic("testmic")], Transcriber(lambda: lambda a: ""), vad=lambda mic: VAD()
        )

        async def broken(mic):
            raise ValueError("capture ring failed")

        monkeypatch.setattr(w, "_mic", broken)
        try:
            with pytest.raises(ValueError, match="capture ring failed"):
                await asyncio.wait_for(w.run(asyncio.Event()), 1)
        finally:
            w.transcriber.close()

    asyncio.run(run())


def test_energy_activity_reports_transitions_and_heartbeats_per_microphone(tmp_path: Path) -> None:
    store = tmp_path / "segments"
    transcriber = Transcriber(lambda: lambda a: "")
    worker = Worker(cast(continuum_py.Client, None), [], transcriber, vad=lambda mic: VAD(), segments=store)
    t = 1_800_000_000 * NS
    try:
        worker._activity(Mic("alpha"), t + 1_000_000_000, False, -55.0, -40.0)
        worker._activity(Mic("alpha"), t + 1_100_000_000, False, -54.0, -40.0)  # no change within a second
        worker._activity(Mic("alpha"), t + 1_200_000_000, True, -25.0, -40.0)  # a transition: at once
        worker._activity(Mic("alpha"), t + 2_200_000_000, True, -26.0, -40.0)  # the heartbeat
        worker._activity(Mic("beta"), t + 2_200_000_000, False, float("nan"), float("nan"))
        for writer in worker.writers.values():
            writer.sync()
    finally:
        transcriber.close()
    rows = speech_log.lines(t / NS, t / NS + 10, (speech_log.ACTIVITY,), base=store)
    assert [(round(r["ts_start"] - t / NS, 3), r["device"], r["in_speech"]) for r in rows] == [
        (1.0, "alpha", False),
        (1.2, "alpha", True),
        (2.2, "alpha", True),
        (2.2, "beta", False),
    ]
    assert {k: v for k, v in rows[1].items() if k != "ts_start"} == {
        "device": "alpha",
        "stream": "alpha.audio",
        "in_speech": True,
        "level_dbfs": -25.0,
        "threshold_dbfs": -40.0,
    }
    assert "level_dbfs" not in rows[-1] and "threshold_dbfs" not in rows[-1]


@pytest.mark.parametrize("text,expected_db", [("", 10.0), ("hello", 11.5)])
def test_gain_learns_only_from_confirmed_transcribed_speech(pack_toml, tmp_path, text, expected_db):
    from liberational.audio import gain

    async def run():
        control = FakeControl(10.0)
        mic_gain = gain.MicGain("testmic", lambda: "testmic-node", lambda device: control)
        transcriber = Transcriber(lambda: lambda audio: text)
        worker = Worker(
            cast(continuum_py.Client, None),
            [Mic("testmic")],
            transcriber,
            vad=lambda mic: VAD(),
            gains={"testmic": mic_gain},
            segments=tmp_path / "segments",
        )
        candidate = Candidate("testmic.audio", 1_000_000_000, 2_000_000_000, speech(1.0, amplitude=0.001))
        mic_gain.tick(time.monotonic(), False)  # the PCM pipeline initializes its control before any candidate
        try:
            await worker._hear(candidate)
            mic_gain.tick(time.monotonic() + gain.QUIET_S + 1, False)
        finally:
            transcriber.close()
        assert control.db == expected_db
        assert worker.committed == bool(text)

    asyncio.run(run())


def test_speaker_embedding_feeds_mono_waveform_and_reuses_cpu_session(tmp_path, monkeypatch):
    from unittest.mock import create_autospec

    import onnxruntime as ort

    from liberational import model_runtime, segment_log
    from liberational.listen.worker import SpeakerEmbedding

    session = create_autospec(ort.InferenceSession, instance=True)
    vector = np.zeros((1, segment_log.VOICE_DIM), np.float32)
    vector[0, :2] = [3, 4]
    session.run.return_value = [vector]
    factory = create_autospec(ort.InferenceSession, return_value=session)
    monkeypatch.setattr(ort, "InferenceSession", factory)
    monkeypatch.setattr(model_runtime, "loaded", lambda *args, **kwargs: None)
    embed = SpeakerEmbedding(tmp_path / "model.onnx")
    audio = np.arange(2048, dtype=np.float32)[::2]

    first, second = embed(audio), embed(audio)

    np.testing.assert_allclose(first[:2], [0.6, 0.8])
    np.testing.assert_array_equal(first, second)
    assert factory.call_count == 1
    assert factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    assert factory.call_args.kwargs["sess_options"].intra_op_num_threads == 2
    output, inputs = session.run.call_args.args
    assert output == ["embedding"]
    np.testing.assert_array_equal(inputs["waveform"], audio[None, :])
    assert inputs["waveform"].flags.c_contiguous


@pytest.mark.parametrize("audio", [np.zeros(511), np.zeros((1, 1024)), np.full(1024, np.nan)])
def test_speaker_embedding_rejects_invalid_waveform_before_loading(tmp_path, audio):
    from liberational.listen.worker import SpeakerEmbedding

    with pytest.raises(ValueError, match="finite mono"):
        SpeakerEmbedding(tmp_path / "missing.onnx")(audio)


@pytest.mark.parametrize("kind", ["wrong-space", "zero", "nonfinite"])
def test_speaker_embedding_rejects_invalid_model_output(tmp_path, monkeypatch, kind):
    from unittest.mock import create_autospec

    import onnxruntime as ort

    from liberational import model_runtime, segment_log
    from liberational.listen.worker import SpeakerEmbedding

    vectors = {
        "wrong-space": np.ones((1, 512), np.float32),
        "zero": np.zeros((1, segment_log.VOICE_DIM), np.float32),
        "nonfinite": np.full((1, segment_log.VOICE_DIM), np.inf, np.float32),
    }
    session = create_autospec(ort.InferenceSession, instance=True)
    session.run.return_value = [vectors[kind]]
    monkeypatch.setattr(ort, "InferenceSession", create_autospec(ort.InferenceSession, return_value=session))
    monkeypatch.setattr(model_runtime, "loaded", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="invalid ReDimNet2"):
        SpeakerEmbedding(tmp_path / "model.onnx")(np.zeros(1024, np.float32))


class NoHits:
    async def nearest(self, *_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return []


class NoOne:
    def person_of(self, n: int) -> str | None:
        return None

    def voice_of(self, person: str) -> int | None:
        return None


@pytest.mark.parametrize(("text", "labelled"), [("Yeah.", False), ("Yeah, sure.", True)])
def test_a_one_word_line_is_written_without_an_unknown_speaker(tmp_path, text, labelled):
    prints = voices.Voiceprints(tmp_path / "voiceprints.json", lambda: 0)
    matcher = voices.VoiceMatcher(prints, NoOne(), NoHits(), segments=tmp_path / "segments")
    worker = Worker(
        cast(continuum_py.Client, None),
        [Mic("testmic")],
        cast(Transcriber, None),
        vad=lambda mic: VAD(),
        matcher=matcher,
        segments=tmp_path / "segments",
    )
    embedding = np.zeros(192, np.float32)
    embedding[0] = 1.0
    candidate = Candidate("testmic.audio", 1 * NS, 3 * NS, speech(2.0))
    asyncio.run(worker._write(candidate, "testmic", Heard(text, embedding), text, False))
    for writer in worker.writers.values():
        writer.close()
    matcher.close()
    (line,) = speech_log.lines(0, 10, base=tmp_path / "segments")
    assert ("speaker" in line) is labelled


class _Ring:
    """A ring reader over a list: next_record waits for a record when caught up, as the native one does."""

    def __init__(self, records: list[tuple[int, bytes, int]]) -> None:
        self.records = list(records)
        self.waits = 0
        self.landed = asyncio.Event()

    def land(self, *records: tuple[int, bytes, int]) -> None:
        self.records.extend(records)
        self.landed.set()

    def read_record(self) -> tuple[int, bytes, int] | None:
        return self.records.pop(0) if self.records else None

    async def next_record(self, timeout_ms: int | None = None) -> tuple[int, bytes, int] | None:
        while not self.records:
            self.waits += 1
            self.landed.clear()
            await self.landed.wait()
        return self.records.pop(0)


def test_ring_batches_takes_everything_landed_per_wake_and_waits_when_caught_up():
    async def run() -> tuple[list[list[tuple[int, bytes, int]]], int]:
        ring = _Ring([(1, b"a", 0), (2, b"b", 0), (3, b"c", 2)])
        batches = ring_batches(cast(continuum_py.RingReader, ring), pace_s=0)
        got = [await anext(batches)]
        ring.land((4, b"d", 0), (5, b"e", 0))
        got.append(await anext(batches))
        later = asyncio.ensure_future(anext(batches))
        await asyncio.sleep(0.01)
        assert not later.done()  # caught up: it waits for the ring, it doesn't hand back an empty batch
        ring.land((6, b"f", 0))
        got.append(await asyncio.wait_for(later, 2))
        return got, ring.waits

    got, waits = asyncio.run(run())
    assert got == [
        [(1, b"a", 0), (2, b"b", 0), (3, b"c", 2)],
        [(4, b"d", 0), (5, b"e", 0)],
        [(6, b"f", 0)],
    ]
    assert waits == 1
