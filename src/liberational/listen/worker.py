"""The P1 audio worker on Continuum: each microphone's PCM ring followed live, gated, transcribed, and written
as utterances on `speech.utterance` in the segment store.

Per microphone, a `MicPipeline` takes ring records as they land through a continuity check, a resample to 16 kHz, and
v1's energy VAD, and puts out candidate utterances with their UTC span. The reader drains whatever has landed, then
sleeps `READ_PACE_S` (`ring_batches`), so the loop wakes about 20 times a second instead of once per 10 ms block;
caught up, it waits on the ring's sequence word, never a poll. Each record keeps its own stamp, so the gate's
timing doesn't move; a candidate only reaches ASR up to `READ_PACE_S` later. A `Transcriber` on one dedicated thread (MLX models are thread-affine) runs each
candidate through v1's Silero speech gate, ASR, and a ReDimNet2 B3 embedding, and a `Matcher` names the speaker. The
utterance is then written through a `speech_log.Writer` under the mic's device name, which also gets the energy
gate's state on `speech.activity` (transitions at once, else once a second). The line names its audio (`audio`, the
mic stream and span); it has no broker parents. Its embedding goes to the segment store too (listen/voices.py), so
the worker writes nothing to the broker.

Capture can register its stream and ring after this service starts, especially after a broker restart. A microphone
waits for that ring before reading. A failed reader or batch subscription ends the worker, so the service manager
restarts it instead of reporting a running service with no transcription.

Each microphone's auto gain (audio/gain.py's `MicGain`) rides along in its `MicPipeline`: it learns from
confirmed transcriptions, moves the mic's own capture volume between utterances (or scales the samples before the
VAD, for a mic without one), and `Worker` writes where each gain is to `listen.gain.json` for Settings.

Around the transcription, `Checks` keeps v1's rules: a candidate overlapping playback is dropped before ASR (echo and
barge-in over the assistant's own voice), a sound-effect caption is not speech, another microphone's copy of a line
just committed is dropped, and a line that is the voice password is written "[password]" with `password: true`.
`V1Checks` does that exactly as v1's ambient service does, through its playback windows and password window, now
moved onto broker streams. The service's `Matcher` is `voices.VoiceMatcher`: each line's
voiceprint from the vector worker's search, and its embedding filed on `voice.embedding` (listen/voices.py, which
also handles the dashboard's voice tags).
"""

import asyncio
import bisect
import collections
import concurrent.futures
import functools
import logging
import math
import re
import signal
import time
import tomllib
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import continuum_py
import numpy as np
from continuum_py import Client

from liberational import continuum_link, model_runtime, speech_log
from liberational.audio import gain as auto_gain
from liberational.listen.pcm import SR, Continuity, Format, HostClock, Resampler
from liberational.listen.voices import Match, word_count

log = logging.getLogger(__name__)

NS = 1_000_000_000
CAPTURE_RETRY_S = 1.0  # capture may register its stream and ring after listen starts
ACTIVITY_INTERVAL_NS = NS  # operational gate state heartbeat; transitions are sent at once
GAIN_STATUS_S = 5.0  # how often each mic's gain is written for Settings (it calls a status stale after 30 s)
# How long the ring reader sleeps after draining what had landed: five 10 ms blocks a wake instead of one,
# well inside the ring's ten seconds and the Live meters' quarter second.
READ_PACE_S = 0.05


@dataclass(frozen=True)
class Mic:
    """A microphone the worker reads: its device name in the capture manifest (its stream is `<name>.audio`)."""

    name: str
    block_ms: int = 10
    channel: int = 0
    # The manifest entry's `match` ({"name": "Wave 3"}), which finds its PipeWire node (and so its mixer) for gain.
    match: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def stream(self) -> str:
        return f"{self.name}.audio"


def mics_from_manifest(path: Path, channels: dict[str, int] | None = None) -> list[Mic]:
    """The microphones in capture's device manifest (`config/devices.toml`), with their block length, and the
    channel to transcribe from `channels` (pack.toml's `[audio.mic_channel]`, by device name; 0 by default, as v1)."""
    manifest = tomllib.loads(path.read_text())
    channels = channels or {}
    out = []
    for d in manifest.get("device", []):
        if d.get("kind") != "microphone" and "audio" not in d:
            continue
        block_ms = int(d.get("audio", {}).get("block_ms", 10))
        out.append(Mic(d["name"], block_ms, int(channels.get(d["name"], 0)), dict(d.get("match") or {})))
    return out


@dataclass(frozen=True)
class Candidate:
    """A stretch of speech the VAD found: its microphone, UTC span, and audio at 16 kHz."""

    mic: str
    start_ns: int
    end_ns: int
    audio: np.ndarray


class Vad(Protocol):
    def feed(self, block: np.ndarray) -> Any: ...


async def ring_batches(
    reader: continuum_py.RingReader, pace_s: float = READ_PACE_S
) -> AsyncIterator[list[tuple[int, bytes, int]]]:
    """Each wake's records, oldest first: whatever has landed, then a `pace_s` sleep before the next look. Caught up,
    it waits on the ring's sequence word (next_record), so an idle ring costs no wakeups."""
    while True:
        first = await reader.next_record()
        assert first is not None  # no timeout, so it waits until there is one
        batch = [first]
        while (more := reader.read_record()) is not None:
            batch.append(more)
        yield batch
        await asyncio.sleep(pace_s)


class MicPipeline:
    """One microphone's live path, synchronous so it can be tested without a broker or a device: `feed` each ring
    record (its stamp, its first sample, the decoded channel, and how many records were lost before it) and get back
    the utterances that ended in it.

    A break in continuity (lost records, a sample count or stamp that doesn't follow) drops any speech in progress and
    starts afresh: the times on either side don't join, and splicing them would make one utterance of two.
    """

    def __init__(
        self,
        mic: Mic,
        fmt: Format,
        clock: HostClock,
        vad: Callable[[], Vad],
        gain: auto_gain.MicGain | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.mic, self.fmt, self.clock, self.new_vad = mic, fmt, clock, vad
        self.gain, self.monotonic = gain, monotonic
        self.breaks = 0
        self.levels: list[float] = []  # each record's dBFS since the last levels line
        self.candidates = 0
        self._restart()

    def _restart(self) -> None:
        self.continuity = Continuity(self.fmt)
        self.resampler = Resampler(self.fmt.rate_hz)
        self.vad = self.new_vad()
        self.idx = 0  # 16 kHz samples fed to this VAD
        # (16 kHz index, UTC ns) at each record's start, for the last minute: a VAD index maps through the nearest
        # earlier mark, so the sample clock's drift against UTC never accumulates past one record.
        self.marks: collections.deque[tuple[int, int]] = collections.deque(maxlen=60 * 1000 // self.mic.block_ms)
        self.last_utc: int | None = None  # the last record's start, UTC ns

    def utc_of(self, idx: int) -> int:
        """UTC nanoseconds of a 16 kHz sample index of the current VAD."""
        keys = [m[0] for m in self.marks]
        i = max(0, bisect.bisect_right(keys, idx) - 1)
        at, utc = self.marks[i]
        return utc + (idx - at) * NS // SR

    def feed(self, host_ns: int, first: int, x: np.ndarray, lost: int = 0) -> list[Candidate]:
        why = self.continuity.check(host_ns, first, lost)
        if why is not None:
            self.breaks += 1
            log.info("%s: %s; starting the VAD afresh", self.mic.stream, why)
            self._restart()
            self.continuity.check(host_ns, first)
        if self.gain is not None and (factor := self.gain.factor) != 1.0:
            x = x * np.float32(factor)  # software gain, before everything that hears the mic
        y = self.resampler(x)
        self.last_utc = self.clock.utc_ns(host_ns)
        self.marks.append((self.idx, self.last_utc))
        self.idx += len(y)
        out = []
        for s0, s1, audio in self.vad.feed(y):
            out.append(Candidate(self.mic.stream, self.utc_of(s0), self.utc_of(s1), np.asarray(audio, np.float32)))
        self.candidates += len(out)
        if self.gain is not None:
            # Gain learns only from confirmed transcriptions in Worker._commit, never raw noise candidates.
            now = self.monotonic()
            self.gain.tick(now, bool(getattr(self.vad, "in_speech", False)))
            if (changed := self.gain.take_change()) and (shift := getattr(self.vad, "shift", None)) is not None:
                before = getattr(self.vad, "floor", float("nan"))
                shift(changed)  # the VAD's floor moves with the gain, or every frame reads as loud after a step up
                log.info(
                    "%s: gain %+.1f dB: VAD floor %.1f -> %.1f dBFS",
                    self.mic.stream,
                    changed,
                    before,
                    getattr(self.vad, "floor", float("nan")),
                )
        self.levels.append(dbfs(x))
        if len(self.levels) >= LEVELS_S * 1000 // self.mic.block_ms:
            self.log_levels()
        return out

    def log_levels(self) -> None:
        """One line on how loud the mic has been and how the VAD sees it, then start counting again."""
        d = np.asarray(self.levels)
        g = self.gain.status() if self.gain is not None else {}
        gain_now = (
            f"; {g['kind']} gain {g['db']:+.1f} dB{'' if g['auto'] else ' (auto off)'}"
            if g.get("db") is not None
            else ""
        )
        log.info(
            "%s: levels p50 %.1f p90 %.1f max %.1f dBFS; VAD floor %.1f, opens at %.1f; %d candidates%s",
            self.mic.stream,
            float(np.percentile(d, 50)),
            float(np.percentile(d, 90)),
            float(d.max()),
            getattr(self.vad, "floor", float("nan")),
            getattr(self.vad, "threshold", float("nan")),
            self.candidates,
            gain_now,
        )
        self.levels, self.candidates = [], 0


class Matcher(Protocol):
    """Which voiceprint said a line (`match`), and filing its embedding once the line is committed (`learn`)."""

    async def match(
        self, embedding: np.ndarray, start_ns: int, end_ns: int, *, words: int | None = None
    ) -> Match | None: ...
    def learn(
        self,
        embedding: np.ndarray,
        match: Match | None,
        utterance_id: str,
        start_ns: int,
        end_ns: int,
        *,
        payload_id: str | None = None,
        slot: str | None = None,
    ) -> None: ...


class NoMatcher:
    """Names nobody and files nothing (tests, and a worker run without the vector worker)."""

    async def match(
        self, embedding: np.ndarray, start_ns: int, end_ns: int, *, words: int | None = None
    ) -> Match | None:
        return None

    def learn(
        self,
        embedding: np.ndarray,
        match: Match | None,
        utterance_id: str,
        start_ns: int,
        end_ns: int,
        *,
        payload_id: str | None = None,
        slot: str | None = None,
    ) -> None:
        return None


TARGET_DBFS = -23.0  # a candidate's level before the gate and ASR: normal speech
MAX_GAIN_DB = 45.0  # the most a candidate is raised, so near-silence stays near-silent
PEAK = 0.95
LEVELS_S = 60  # how often each microphone's levels are logged


def dbfs(audio: np.ndarray) -> float:
    """RMS level in dBFS (float samples in [-1, 1])."""
    return float(20 * np.log10(np.sqrt(np.mean(np.square(audio, dtype=np.float64))) + 1e-12)) if len(audio) else -240.0


def level(audio: np.ndarray) -> np.ndarray:
    """`audio` brought to TARGET_DBFS, raised by at most MAX_GAIN_DB and kept under PEAK. Auto gain (MicGain) raises
    a quiet mic itself now, as v1 did, but slowly: until it has (and for a mic it is off for), Silero's gate found
    no speech in such quiet audio (on a Wave 3 at 0 dB, a second of speech at -59 dBFS read as 0 s raw,
    1.0 s levelled), so nothing was transcribed. With the gain right, this moves a candidate a few dB at most."""
    gain_db = min(MAX_GAIN_DB, TARGET_DBFS - dbfs(audio))
    out = audio.astype(np.float32) * np.float32(10 ** (gain_db / 20))
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    return out * np.float32(PEAK / peak) if peak > PEAK else out


@dataclass
class Heard:
    """What the transcriber made of a candidate: its text, its embedding, and how long each step took."""

    text: str
    embedding: np.ndarray | None
    lat_ms: dict[str, int] = field(default_factory=dict)


class Transcriber:
    """The gate, ASR, and embedding for every microphone, on one dedicated thread (MLX models must be used on the
    thread that made them). `gate` (v1's Silero speech gate) drops a candidate with too little speech; `asr` is
    `speech.asr.ASR` or a test's stand-in; `embed` makes the ReDimNet2 B3 voiceprint (None to skip it)."""

    def __init__(
        self,
        asr: Callable[[], Callable[[np.ndarray], str]],
        *,
        gate: Callable[[np.ndarray], bool] | None = None,
        embed: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self._make_asr, self.gate, self.embed = asr, gate, embed
        self._asr: Callable[[np.ndarray], str] | None = None
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="listen-asr")

    def _run(self, audio: np.ndarray, mic: str = "") -> Heard | None:
        lat: dict[str, int] = {}
        was = dbfs(audio)
        audio = level(audio)
        t = time.perf_counter()
        if self.gate is not None and not self.gate(audio):
            found = getattr(self.gate, "last_speech_s", None)
            log.info(
                "%s: %.1f s at %.1f dBFS: not speech%s",
                mic or "a mic",
                len(audio) / SR,
                was,
                f" ({found:.2f} s found)" if isinstance(found, float) else "",
            )
            return None
        lat["gate"] = round((time.perf_counter() - t) * 1000)
        if self._asr is None:
            self._asr = self._make_asr()
        t = time.perf_counter()
        text = self._asr(audio).strip()
        lat["asr"] = round((time.perf_counter() - t) * 1000)
        log.info("%s: %.1f s at %.1f dBFS: %s", mic or "a mic", len(audio) / SR, was, "heard" if text else "no words")
        if not text:
            return None
        emb = None
        if self.embed is not None:
            t = time.perf_counter()
            emb = self.embed(audio)
            lat["emb"] = round((time.perf_counter() - t) * 1000)
        return Heard(text, emb, lat)

    async def __call__(self, audio: np.ndarray, mic: str = "") -> Heard | None:
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._run, audio, mic)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


@dataclass(frozen=True)
class Twin:
    """A line dropped as another microphone's copy of a line just committed: that line's device and start."""

    device: str
    start_s: float


class Checks(Protocol):
    def echo(self, start_s: float, end_s: float) -> bool:
        """Whether the span overlaps playback (the assistant's own voice, or audio the dashboard played)."""
        ...

    def heard(self, device: str, start_s: float, end_s: float, text: str) -> tuple[str, bool] | Twin | None:
        """What to commit for a transcribed line: its text and whether it's the voice password; a `Twin` to drop it
        as another microphone's copy of a line just committed; or None to drop it as not speech."""
        ...


class NoChecks:
    """Commits everything as heard: for tests of the pipeline alone."""

    def echo(self, start_s: float, end_s: float) -> bool:
        return False

    def heard(self, device: str, start_s: float, end_s: float, text: str) -> tuple[str, bool] | Twin | None:
        return text, False


_CAPTION = re.compile(r"[\(\[][^\)\]]*[\)\]]")


class V1Checks:
    """v1's rules around a transcription (audio/ambient.py's Transcriber.handle), with its state files: playback
    windows (speech/playback_guard), the password verifier and window (assistant/voice_auth), and two-mic dedup
    (audio/twins)."""

    def __init__(self, state_dir: Path) -> None:
        from liberational.audio import twins

        self.verifier = state_dir / "voice_auth.json"
        self.recent = twins.Recent()

    def echo(self, start_s: float, end_s: float) -> bool:
        from liberational.speech.playback_guard import overlaps

        return overlaps(start_s, end_s)

    def heard(self, device: str, start_s: float, end_s: float, text: str) -> tuple[str, bool] | Twin | None:
        import dataclasses

        from liberational.assistant import voice_auth
        from liberational.audio import twins
        from liberational.people import registry

        # Whisper captions sounds "(keyboard clicking)" or "[BLANK_AUDIO]": never speech.
        spoken = _CAPTION.sub("", text).strip()
        if not spoken or not re.search(r"[A-Za-z0-9]", spoken):
            return None
        window, alike, primary = twins.settings()
        h = twins.Heard(
            registry.key_of({"device": device, "ts_start": round(start_s, 3)}), device, start_s, end_s, spoken
        )
        pending = {"ts_start": start_s, "ts_end": end_s, "device": device}
        maybe_password = voice_auth.may_be_password(self.verifier, pending, spoken)
        # Never drop a line that may be the password as a twin: a worse copy written first would fail it.
        if not maybe_password and (written := self.recent.twin_of(h, window, alike, primary)) is not None:
            return Twin(written.device, written.start)
        password = maybe_password and voice_auth.said_password(self.verifier, spoken)
        shown = voice_auth.PASSWORD_SHOWN if password else spoken
        self.recent.add(dataclasses.replace(h, shown=shown))
        return shown, password


class Worker:
    """The audio worker: `run` reads every microphone in `mics` until `stop` is set."""

    def __init__(
        self,
        client: Client,
        mics: Sequence[Mic],
        transcriber: Transcriber,
        *,
        vad: Callable[[Mic], Vad],
        matcher: Matcher | None = None,
        checks: Checks | None = None,
        gains: dict[str, auto_gain.MicGain] | None = None,
        gain_status: Path | None = None,
        segments: Path | None = None,
    ) -> None:
        self.client, self.mics, self.transcriber, self.vad = client, list(mics), transcriber, vad
        self.matcher = matcher or NoMatcher()
        self.checks = checks or NoChecks()
        self.segments = segments  # the segment store (None: paths.state/segments)
        self.clock = HostClock()
        self.gains = gains or {}  # by mic name
        self.gain_status = gain_status
        self.committed = 0
        self.writers: dict[str, speech_log.Writer] = {}  # by mic name: each mic's lines and gate state
        self._activity_last: dict[str, tuple[int, bool]] = {}
        self._pending: set[asyncio.Task[None]] = set()

    async def run(self, stop: asyncio.Event) -> None:
        # One subscription per microphone: its rings name just that stream, and its events (the stream's batch
        # notices) give the PCM format and the parents.
        tasks = [asyncio.create_task(self._mic(mic)) for mic in self.mics]
        if self.gain_status is not None and self.gains:
            tasks.append(asyncio.create_task(self._gain_status(self.gain_status)))
        stopping = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait([*tasks, stopping], return_when=asyncio.FIRST_COMPLETED)
            for task in done - {stopping}:
                task.result()
                raise RuntimeError("a listen task ended while the worker was running")
        finally:
            stopping.cancel()
            tasks.append(stopping)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._pending:
                await asyncio.gather(*self._pending, return_exceptions=True)
            for writer in self.writers.values():
                writer.close()
            self.writers.clear()

    async def _gain_status(self, path: Path) -> None:
        """Where each mic's gain is, every GAIN_STATUS_S, for Settings."""
        while True:
            try:
                auto_gain.write_status(path, self.gains.values())
            except OSError as ex:
                log.warning("couldn't write %s: %s", path, ex)
            await asyncio.sleep(GAIN_STATUS_S)

    async def _mic(self, mic: Mic) -> None:
        waiting = False
        while True:
            try:
                # Pinned: a notice carries its ring batch's locator (first_seq, count) only then, and speech coverage
                # needs it to know which records a batch holds. A ring batch is never actually pinned (capture
                # never waits); a notice the ring has already lapped is left out, and its batch couldn't be read.
                sub = await self.client.subscribe([mic.stream], pinned=True)
            except continuum_py.BrokerError as ex:
                if not ex.args or ex.args[0] not in {"NotFound", "Unavailable"}:
                    raise
                sub = None
            if sub is not None:
                async with sub:
                    if sub.rings:
                        await self._follow_ring(mic, sub)
                        raise RuntimeError(f"{mic.stream}: broker audio reader ended")
            if not waiting:
                log.warning("%s: waiting for capture to register its broker audio ring", mic.stream)
                waiting = True
            await asyncio.sleep(CAPTURE_RETRY_S)

    async def _follow_ring(self, mic: Mic, sub: continuum_py.Subscription) -> None:
        (ring,) = sub.rings
        stream_id, lease = ring["stream_id"], ring["lease"]
        # Fix the reader's place before awaiting the first batch, which brings its PCM format and parents.
        reader = self.client.ring_reader(lease)
        shape: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        notices = asyncio.create_task(self._notices(sub, shape))

        async def read() -> None:
            await self._read(mic, stream_id, lease, reader, await shape)

        reading = asyncio.create_task(read())
        try:
            done, _ = await asyncio.wait([notices, reading], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            notices.cancel()
            reading.cancel()
            await asyncio.gather(notices, reading, return_exceptions=True)

    async def _notices(
        self,
        sub: continuum_py.Subscription,
        shape: asyncio.Future[str],
    ) -> None:
        async for event in sub:
            if event["type"] != "Event" or "item" not in event:
                continue
            h = continuum_py.decode_header(event["item"]["header"])
            if not shape.done():
                shape.set_result(h["shape_name"])

    async def _read(
        self, mic: Mic, stream_id: int, lease: continuum_py.StreamLease, reader: continuum_py.RingReader, shape: str
    ) -> None:
        # The first batch notice names the stream's PCM format; the ring's stride and the block length give the rest.
        fmt = Format.of(shape, lease["ring_stride"], mic.block_ms)
        pipeline = MicPipeline(mic, fmt, self.clock, lambda: self.vad(mic), self.gains.get(mic.name))
        log.info("%s: %d Hz, %d channels, reading channel %d", mic.stream, fmt.rate_hz, fmt.channels, mic.channel)
        await self._records(mic, stream_id, reader, fmt, pipeline)

    async def _records(
        self,
        mic: Mic,
        stream_id: int,
        reader: continuum_py.RingReader,
        fmt: Format,
        pipeline: MicPipeline,
    ) -> None:
        async for batch in ring_batches(reader):
            for _seq, record, lost in batch:
                host_ns, first, x = fmt.decode(record, mic.channel)
                candidates = pipeline.feed(host_ns, first, x, lost)
                self._activity(
                    mic,
                    self.clock.utc_ns(host_ns),
                    bool(getattr(pipeline.vad, "in_speech", False)),
                    float(getattr(pipeline.vad, "level", float("nan"))),
                    float(getattr(pipeline.vad, "threshold", float("nan"))),
                )
                for c in candidates:
                    task = asyncio.create_task(self._hear(c))
                    self._pending.add(task)
                    task.add_done_callback(self._pending.discard)

    def _activity(
        self,
        mic: Mic,
        now_ns: int,
        in_speech: bool,
        level_dbfs: float,
        threshold_dbfs: float,
    ) -> None:
        """The energy gate's current state, not verified speech. Stop updating if the audio reader stops, so
        consumers can distinguish stale processing from quiet. No media or transcript content is copied."""
        previous = self._activity_last.get(mic.stream)
        if previous is not None and previous[1] == in_speech and now_ns - previous[0] < ACTIVITY_INTERVAL_NS:
            return
        line: dict[str, Any] = {
            "device": mic.name,
            "stream": mic.stream,
            "ts_start": now_ns / NS,
            "in_speech": in_speech,
        }
        if math.isfinite(level_dbfs):
            line["level_dbfs"] = level_dbfs
        if math.isfinite(threshold_dbfs):
            line["threshold_dbfs"] = threshold_dbfs
        self._writer(mic.name).write(speech_log.ACTIVITY, line, span_ns=(now_ns, now_ns))
        self._activity_last[mic.stream] = (now_ns, in_speech)

    async def _hear(self, c: Candidate) -> None:
        """Hears a candidate and commits its line, unless it's playback, not speech, a sound caption, or a twin of a
        line another mic already has."""
        start_s, end_s = c.start_ns / NS, c.end_ns / NS
        device = c.mic.removesuffix(".audio")
        if self.checks.echo(start_s, end_s):
            log.info("%s: playback overlap excluded before ASR", c.mic)
            return
        heard = await self.transcriber(c.audio, c.mic)
        if heard is None:
            return  # the speech gate said not speech, or the ASR heard no words
        kept = self.checks.heard(device, start_s, end_s, heard.text)
        if kept is None or isinstance(kept, Twin):
            return  # a sound caption, or a line another mic already has
        text, password = kept
        try:
            await self._write(c, device, heard, text, password)
        except Exception as error:
            log.error("%s: a line wasn't written: %s", c.mic, error)

    async def _write(
        self,
        c: Candidate,
        device: str,
        heard: Heard,
        text: str,
        password: bool,
    ) -> str:
        """Writes a heard line and returns its id."""
        if gain := self.gains.get(device):
            gain.heard(c.audio, SR, time.monotonic())
        match = None
        if heard.embedding is not None:
            match = await self.matcher.match(heard.embedding, c.start_ns, c.end_ns, words=word_count(text))
        line: dict[str, Any] = {
            "ts_start": c.start_ns / NS,
            "ts_end": c.end_ns / NS,
            "device": device,
            "text": text,
            "lat_ms": heard.lat_ms,
            "audio": {"stream": c.mic, "start_ns": c.start_ns, "end_ns": c.end_ns},
        }
        if password:
            line["password"] = True
        if match is not None:
            line |= {"speaker": match.speaker, "speaker_id": match.speaker_id, "speaker_sim": round(match.sim, 3)}
        writer = self._writer(device)
        payload_id = writer.write(speech_log.UTTERANCES, line, span_ns=(c.start_ns, c.end_ns))
        writer.sync()  # durable before anything reacts to it
        self.committed += 1
        if heard.embedding is not None:
            self.matcher.learn(heard.embedding, match, payload_id, c.start_ns, c.end_ns)
        return payload_id

    def _writer(self, mic: str) -> speech_log.Writer:
        if mic not in self.writers:
            self.writers[mic] = speech_log.Writer(mic, self.segments)
        return self.writers[mic]


class SpeakerEmbedding:
    """ReDimNet2 B3 VoxCeleb2 LM, with its waveform frontend in ONNX, on two CPU threads.

    The pinned export is checked against the upstream checkpoint. Its 192-dimensional space is incompatible
    with CAM++: cutover discards previous voice vectors and confirmed references, without conversion.
    """

    def __init__(self, model: Path) -> None:
        self.model = model
        self._session: Any = None

    def __call__(self, audio16k: np.ndarray) -> np.ndarray:
        from liberational import segment_log

        audio = np.asarray(audio16k, dtype=np.float32)
        if audio.ndim != 1 or len(audio) < 512 or not np.isfinite(audio).all():
            raise ValueError("speaker embedding needs finite mono 16 kHz audio of at least 512 samples")
        if self._session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.model), sess_options=options, providers=["CPUExecutionProvider"]
            )
            model_runtime.loaded("models.files.speaker_embedding", self.model.name, "speaker ID", "CPU", owner=self)
        embedding = np.asarray(
            self._session.run(["embedding"], {"waveform": np.ascontiguousarray(audio[None, :])})[0],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(embedding))
        if (
            embedding.shape != (1, segment_log.VOICE_DIM)
            or not np.isfinite(embedding).all()
            or not math.isfinite(norm)
            or norm <= 0
        ):
            raise ValueError("speaker model returned an invalid ReDimNet2 B3 embedding")
        return embedding[0] / norm


@model_runtime.reporting("listen")
def main(argv: Sequence[str] | None = None) -> None:
    """`pack-listen`: the audio worker as a service. It reads the microphones in capture's manifest from the broker
    (run continuum-broker and continuum-capture with it), with v1's per-mic sensitivity, speech gate, and ASR engine."""
    import argparse

    from continuum_py.search import Searcher

    from liberational import config as cfg
    from liberational.audio import sensitivity
    from liberational.listen import gate, voices
    from liberational.people import registry
    from liberational.speech.asr import ASR

    parser = argparse.ArgumentParser(prog="pack-listen", description=main.__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("config/devices.toml"), help="capture's device manifest")
    parser.add_argument("--socket", type=Path, default=None, help="the broker's socket (default: under paths.state)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    mics = mics_from_manifest(args.manifest, cfg.get("audio.mic_channel") or {})
    if not mics:
        raise SystemExit(f"no microphones in {args.manifest}")
    exclude = tomllib.loads(args.manifest.read_text()).get("audio", {}).get("exclude", auto_gain.CAPTURE_EXCLUDE)
    gains = {
        m.name: auto_gain.MicGain(m.name, functools.partial(auto_gain.device_name, m.match, list(exclude)))
        for m in mics
    }

    def vad(mic: Mic) -> Vad:
        t = sensitivity.tuning(sensitivity.value(mic.name))
        return gate.VAD(margin_db=t.margin_db, min_dbfs=t.min_dbfs)

    transcriber = Transcriber(
        ASR, gate=gate.speech_gate(), embed=SpeakerEmbedding(Path(cfg.model_path("speaker_embedding")))
    )

    async def run() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        state = Path(cfg.path("paths.state"))
        from liberational.people import transcript_view

        people = voices.RegistryPeople(state)
        if seeded := voices.seed_taught(state, people):  # a fresh segment store: the taught voices, from knowledge
            log.info("wrote %d taught voice vectors for the vector worker to index", seeded)
        prints = voices.Voiceprints(
            state / "voiceprints.json", lambda: max(registry.at(state).voiceprints(), default=0)
        )
        try:
            async with (
                continuum_link.connected(args.socket or speech_log.socket_path(), client_name="listen") as client,
                Searcher(client) as search,
            ):
                matcher = voices.VoiceMatcher(
                    prints,
                    people,
                    search,
                    clones=lambda: transcript_view.clone_only_ids(state),
                    slot="listen",
                    references=voices.ReferenceGallery(state),
                )
                worker = Worker(
                    client,
                    mics,
                    transcriber,
                    vad=vad,
                    checks=V1Checks(state),
                    matcher=matcher,
                    gains=gains,
                    gain_status=state / auto_gain.STATUS,
                )
                tags = voices.VoiceTags(state, matcher)
                log.info("subscribing to broker audio streams: %s", ", ".join(m.stream for m in mics))
                await asyncio.gather(worker.run(stop), tags.run(stop))
        finally:
            transcriber.close()

    continuum_link.run_service(run())
