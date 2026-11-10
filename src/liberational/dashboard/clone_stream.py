"""The dashboard's clone speech, streamed from the voice service's warm model.

POST /clones/say used to run clone_say.py in a new process for every line, which loaded the ~1.5 GB model each time
(about 20 s). Now, when the voice service is running with its clone model (speech/clone_server.py), the line is made
there: /clones/say asks for it, waits for the first half second of audio, and answers with the render's usual URL
(/clones/render?f=say-<id>-<ms>.wav). While the line is still being made that URL streams it as a WAV of open
length, so the browser starts playing at once; when it's done the file is written and served as before. Without the
voice service, or for a voice whose reference clip still has to be built, the old process runs instead.

The stream is 32-bit float at twice the model's rate (48 kHz), not the file's 16-bit 24 kHz: Chrome starts playing a
stream once it has about 200 KB, whatever its format. Measured on the Linux machine with audio arriving like the warm model's
(first half second at 1.5 s, then twice real time), playback started at 3.8 s for 16-bit 24 kHz, 2.7 s for float
24 kHz, and 2.1 s for float 48 kHz. Safari is sent the finished file instead (it doesn't play an open-length WAV).

The browser can't tell how long an open-length line is, so the playback window (speech/playback_guard.py, which makes
the room microphone drop what it hears of the speakers) is opened here: from the first audio for the longest the line
could be, then, once it's done, for the rest of its real length.
"""

import http.client
import json
import struct
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from liberational import config as cfg
from liberational import logs

LOG = logs.get("clone-stream")
FIRST_AUDIO_S = 30.0  # longer than this (the voice is busy with a long reply) and the old way is used instead
KEEP_S = 120.0  # a finished render stays streamable this long; after that its file is read
READ = 4800  # bytes per read from the voice service: 0.1 s of 24 kHz 16-bit audio
OPEN_LENGTH = 0x7FFFFFFF  # the RIFF and data sizes of a WAV still being made
PLAY_LAG_S = 1.5  # from the first audio to the browser playing it, with room to spare


def wav_header(sr: int, size: int = OPEN_LENGTH) -> bytes:
    """A 44-byte header for 16-bit mono PCM; size is the data's length, or an open length while it's being made."""
    riff = min(0xFFFFFFFF, size + 36)
    return (
        b"RIFF"
        + struct.pack("<I", riff)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
        + (b"data" + struct.pack("<I", size))
    )


def stream_header(sr: int) -> bytes:
    """A 44-byte header for 32-bit float mono of open length (format 3, IEEE float)."""
    return (
        b"RIFF"
        + struct.pack("<I", OPEN_LENGTH)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 3, 1, sr, sr * 4, 4, 32)
        + (b"data" + struct.pack("<I", OPEN_LENGTH))
    )


def upsample2(x: NDArray[np.float32], before: float | None) -> NDArray[np.float32]:
    """x at twice its rate: each sample, then the midpoint to the next. The first sample also gets the midpoint from
    before (the previous piece's last sample), so the pieces join smoothly; the very first one repeats itself."""
    y = np.empty(2 * len(x), np.float32)
    prev = np.concatenate([[x[0] if before is None else before], x[:-1]]).astype(np.float32)
    y[0::2] = (prev + x) / 2
    y[1::2] = x
    return y


class Render:
    """One line being made: its audio so far, for any number of readers, and its file once it's done."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.sr: int | None = None
        self.chunks: list[bytes] = []
        self.done = False
        self.error: str | None = None
        self.ended = 0.0
        self.abandoned = False  # /clones/say gave up waiting and made the line the old way: stop reading
        self.first_at: float | None = None
        self.guard_token: str | None = None  # the playback window's token
        self.cond = threading.Condition()

    def add(self, b: bytes, sr: int) -> None:
        with self.cond:
            self.sr = self.sr or sr
            if self.first_at is None:
                self.first_at = time.monotonic()
            self.chunks.append(b)
            self.cond.notify_all()

    def seconds(self) -> float:
        with self.cond:
            return sum(len(c) for c in self.chunks) / 2 / (self.sr or 24000)

    def finish(self, error: str | None = None) -> None:
        """The line is done, or failed (error): a complete line is written as its render file. Readers are always let
        go, even when the file can't be written."""
        tmp = self.path.with_suffix(".part")
        try:
            if not error and self.chunks and self.sr:
                pcm = b"".join(self.chunks)
                tmp.write_bytes(wav_header(self.sr, len(pcm)) + pcm)
                tmp.replace(self.path)
        except OSError as ex:
            error = f"couldn't save the line: {ex}"
            tmp.unlink(missing_ok=True)
        finally:
            with self.cond:
                self.error = error or (None if self.chunks else "no audio was made")
                self.done, self.ended = True, time.monotonic()
                self.cond.notify_all()
                self._reguard()

    def guard(self, longest: float) -> None:
        """Open the playback window for a line that has started: for the longest it could be, or, when it's already
        done, for the rest of its real length. Under the lock, so it and `finish` never both open one."""
        from liberational.speech import playback_guard

        with self.cond:
            self.guard_token = playback_guard.open_window(longest)
            if self.done:
                self._reguard()

    def _reguard(self) -> None:
        """The line's real length is known now: the window covers what's left of it, instead of the longest guess.
        Called with the lock held."""
        if self.guard_token is None or self.first_at is None:
            return
        from liberational.speech import playback_guard

        playback_guard.close_window(self.guard_token)
        left = sum(len(c) for c in self.chunks) / 2 / (self.sr or 24000) - (time.monotonic() - self.first_at)
        left += PLAY_LAG_S
        self.guard_token = playback_guard.open_window(left) if left > 0 and not self.error else None

    def wait(self, timeout: float = 300.0) -> None:
        """Until the render is done (its file written), or timeout seconds."""
        with self.cond:
            self.cond.wait_for(lambda: self.done, timeout)

    def first(self, timeout: float) -> bool:
        """Wait for the first audio (True), or for the render to end without any (False)."""
        with self.cond:
            self.cond.wait_for(lambda: self.chunks or self.done, timeout)
            return bool(self.chunks)

    def stream(self) -> Iterator[bytes]:
        """The stream from the start: its header, then each piece of audio as it comes (float at twice the rate),
        until the render is done."""
        i, last = 0, None
        while True:
            with self.cond:
                self.cond.wait_for(lambda i=i: len(self.chunks) > i or self.done)
                new, done, sr = self.chunks[i:], self.done, self.sr
            if i == 0 and new and sr:
                yield stream_header(2 * sr)
            for b in new:
                x = (np.frombuffer(b, "<i2") / 32768).astype(np.float32)
                if len(x):
                    yield upsample2(x, last).astype("<f4").tobytes()
                    last = float(x[-1])
            i += len(new)
            if done and i >= len(self.chunks):
                return


def waits(user_agent: str) -> bool:
    """Whether this browser gets the finished file rather than the stream: Safari, which doesn't play a WAV of open
    length. Chrome, Edge, and Firefox say "Chrome", "Chromium", or "Firefox" in theirs."""
    return "Safari" in user_agent and not any(k in user_agent for k in ("Chrome", "Chromium", "Firefox", "Android"))


RENDERS: dict[str, Render] = {}  # by file name, while being made and for KEEP_S after
_lock = threading.Lock()


def live(name: str) -> Render | None:
    """The render streaming as name, or None to read its file."""
    with _lock:
        for k in [k for k, r in RENDERS.items() if r.done and time.monotonic() - r.ended > KEEP_S]:
            del RENDERS[k]
        return RENDERS.get(name)


def warm(timeout: float = 0.5) -> bool:
    """Whether the voice service is up with a clone model to speak with."""
    try:
        with urllib.request.urlopen(cfg.local_url("voice", "/health"), timeout=timeout) as r:
            return bool(json.loads(r.read()).get("ok"))
    except OSError, ValueError:
        return False


def _fetch(r: Render, body: dict[str, object]) -> None:
    req = urllib.request.Request(
        cfg.local_url("voice", "/clone"), json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    error = None
    try:
        with urllib.request.urlopen(req, timeout=FIRST_AUDIO_S + 5) as resp:
            sr = int(resp.headers.get("X-Sample-Rate") or 24000)
            carry = b""
            while not r.abandoned and (b := resp.read1(READ)):
                b = carry + b
                cut = len(b) - len(b) % 2  # whole 16-bit samples only
                b, carry = b[:cut], b[cut:]
                if b:
                    r.add(b, sr)
    except urllib.error.HTTPError as ex:
        try:
            error = json.loads(ex.read()).get("error")
        except OSError, ValueError:
            error = None
        error = error or f"the voice service answered {ex.code}"
    except (OSError, http.client.HTTPException) as ex:  # IncompleteRead: the line failed partway
        error = (
            f"the voice service: {ex!r}" if isinstance(ex, http.client.HTTPException) else f"the voice service: {ex}"
        )
    finally:
        r.finish(error)
        if error:
            LOG.warning(f"clone speech for {r.path.name}: {error}")


def start(sid: int, text: str, out: Path, resolve: Callable[[int], tuple[Path, str] | None] | None = None) -> bool:
    """Make sid's line in the voice service, streamed as out. True once its first audio has come; False when the old
    way should be used (no warm voice service, no reference clip yet, or no audio in time)."""
    try:
        got = (resolve or _reference)(sid)
    except (Exception, SystemExit) as ex:  # a recorded clone with its clip missing: the old way says why
        LOG.info(f"no warm clone speech for voice {sid}: {ex}")
        return False
    if got is None or not warm():
        return False
    from liberational.speech import clone_say

    ref, ref_text = got
    r = Render(out)
    with _lock:
        RENDERS[out.name] = r
    body = {"text": text, "ref": str(ref), "ref_text": ref_text, "max_tokens": clone_say.token_budget(text)}
    threading.Thread(target=_fetch, args=(r, body), name="clone-stream", daemon=True).start()
    if r.first(FIRST_AUDIO_S):
        r.guard(body["max_tokens"] / 12 + PLAY_LAG_S)  # the token budget, at the codec's 12 a second
        return True
    r.abandoned = True  # stop reading; the voice service stops making it when the connection goes
    with _lock:
        RENDERS.pop(out.name, None)
    return False


def _reference(sid: int) -> tuple[Path, str] | None:
    """sid's reference clip and its words, when one is ready (building one takes the old way: it needs ASR)."""
    from liberational.speech import clone_say, voice_clone

    ref = clone_say.ref_for(sid, make=False)
    if ref is None:
        return None
    if not ref.with_suffix(".txt").exists() and ref != cfg.path("owner.voice_ref"):
        return None  # a clip uploaded before references were transcribed: the old way transcribes it first
    return ref, voice_clone.ref_text_for(ref)
