"""Clone speech from the voice service's warm model, for the dashboard (issue #333).

The voice service already holds the clone model (Qwen3-TTS on CUDA or MLX), loaded and warmed at start. This module
lets the dashboard use it instead of starting a process that loads the model again for every line (about 20 s):

    POST http://127.0.0.1:<ports.voice>/clone  {"text", "ref", "ref_text", "max_tokens"}
      -> 200, Content-Type audio/L16 (16-bit little-endian PCM, mono), X-Sample-Rate: 24000, the audio streamed in
         chunks of about half a second as they are made; or 4xx/5xx with {"error"} before any audio
    GET  http://127.0.0.1:<ports.voice>/health  -> {"ok": true, "engine": "cuda" | "mlx"}

The HTTP server only listens on 127.0.0.1, and only answers requests addressed to 127.0.0.1 or localhost with a JSON
body, so a web page in a browser on this machine can't use it (no simple cross-site POST, no DNS rebinding). Synthesis
runs on the voice service's own loop thread (run_pending), one line between transcript reads, never on a request
thread: MLX models are thread-affine, and it's one model copy, shared with the voice's own replies. A reference must
be a clone clip in state/clones or the owner's own recording. A line that fails partway ends its stream without the
closing chunk, so the dashboard sees it as failed, not as a short line.
"""

import json
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import logs

LOG = logs.get("clone-server")
FIRST_AUDIO_S = 25.0  # no audio by then (the voice is busy): give up, before the dashboard does (30 s)
MAX_TEXT = 600  # the dashboard's limit too
QUEUE = 4  # lines waiting at once; more are refused as busy

Synth = Callable[[str, str, str, int], Iterator[tuple[np.ndarray, int]]]


@dataclass
class Job:
    text: str
    ref: str
    ref_text: str
    max_tokens: int
    out: queue.Queue[tuple[bytes, int] | BaseException | None] = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)


JOBS: queue.Queue[Job] = queue.Queue(maxsize=QUEUE)


def allowed_ref(ref: str) -> Path | None:
    """ref as a path, when it's a clone clip under state/clones or the owner's own recording; else None."""
    try:
        p = Path(ref).resolve(strict=True)
    except OSError, RuntimeError, TypeError, ValueError:
        return None
    clones = (cfg.path("paths.state") / "clones").resolve()
    owner = cfg.path("owner.voice_ref")
    if p.suffix.lower() == ".wav" and (p.parent == clones or (owner.exists() and p == owner.resolve())):
        return p
    return None


def model_synth(text: str, ref: str, ref_text: str, max_tokens: int) -> Iterator[tuple[np.ndarray, int]]:
    """The loaded clone model's streamed chunks: (float32 mono audio, sample rate)."""
    from liberational.speech import voice_clone

    m = voice_clone.load()
    if voice_clone.KIND not in ("cuda", "mlx"):
        raise RuntimeError("voice cloning isn't available on this machine")
    for r in m.generate(
        text=text, ref_audio=ref, ref_text=ref_text, stream=True, streaming_interval=0.5, max_tokens=max_tokens,
        temperature=0.7,
    ):  # fmt: skip
        yield np.asarray(r.audio, dtype=np.float32).reshape(-1), int(r.sample_rate)


def pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def run_job(job: Job, synth: Synth = model_synth) -> None:
    """Make one job's audio, handing each chunk to its request as it comes. Never raises."""
    try:
        for audio, sr in synth(job.text, job.ref, job.ref_text, job.max_tokens):
            if job.cancelled.is_set():  # the browser left: stop making audio nobody hears
                break
            job.out.put((pcm16(audio), sr))
    except Exception as ex:
        LOG.warning(f"clone speech failed: {ex}")
        job.out.put(ex)
    finally:
        job.out.put(None)


def run_pending(timeout: float, synth: Synth = model_synth) -> bool:
    """Run one waiting clone job on this thread, waiting up to timeout for it. True when one ran. The voice service's
    loop calls this in place of its pause between transcript reads, so a wake phrase never waits behind more than one
    dashboard line. A job whose request already gave up is skipped."""
    try:
        job = JOBS.get(timeout=timeout)
    except queue.Empty:
        return False
    if job.cancelled.is_set():
        job.out.put(None)
        return False
    run_job(job, synth)
    return True


def _engine() -> str | None:
    from liberational.speech import voice_clone

    return voice_clone.KIND if voice_clone.KIND in ("cuda", "mlx") else None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _json(self, code: int, body: dict[str, Any]) -> None:
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:
        if self.path != "/health":
            return self._json(404, {"error": "not found"})
        engine = _engine()
        self._json(200 if engine else 503, {"ok": bool(engine), "engine": engine})

    def _local(self) -> bool:
        """Addressed to this machine by name or address, so a page served from elsewhere (DNS rebinding) is refused."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost")

    def do_POST(self) -> None:
        if self.path != "/clone":
            return self._json(404, {"error": "not found"})
        if not self._local() or self.headers.get_content_type() != "application/json":
            return self._json(403, {"error": "local JSON requests only"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n)) if 0 < n <= 64_000 else None
        except ValueError, OSError:
            body = None
        if not isinstance(body, dict):
            return self._json(400, {"error": "send a JSON object"})
        text, ref_text = str(body.get("text") or "").strip(), str(body.get("ref_text") or "").strip()
        ref = allowed_ref(str(body.get("ref") or ""))
        try:
            max_tokens = int(body.get("max_tokens") or 0)
        except TypeError, ValueError:
            max_tokens = 0
        if not text or len(text) > MAX_TEXT or not ref_text or ref is None or not 0 < max_tokens <= 4096:
            return self._json(400, {"error": "text, ref (a clone clip), ref_text, and max_tokens are needed"})
        job = Job(text, str(ref), ref_text, max_tokens)
        try:
            JOBS.put_nowait(job)
        except queue.Full:
            return self._json(503, {"error": "the voice service is busy; try again"})
        try:
            self._stream(job)
        finally:
            job.cancelled.set()

    def _stream(self, job: Job) -> None:
        try:
            first = job.out.get(timeout=FIRST_AUDIO_S)
        except queue.Empty:
            return self._json(503, {"error": "the voice service is busy; try again"})
        if first is None or isinstance(first, BaseException):
            return self._json(500, {"error": str(first) if first else "no audio was made"})
        self.send_response(200)
        self.send_header("Content-Type", "audio/L16")
        self.send_header("X-Sample-Rate", str(first[1]))
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        item: tuple[bytes, int] | BaseException | None = first
        try:
            while isinstance(item, tuple):
                self.wfile.write(f"{len(item[0]):x}\r\n".encode() + item[0] + b"\r\n")
                self.wfile.flush()
                item = job.out.get()
            if item is None:
                self.wfile.write(b"0\r\n\r\n")
            else:
                self.close_connection = True  # failed partway: no closing chunk, so it reads as cut off
        except OSError:
            pass  # the dashboard went away; the job is cancelled


def serve(port: int | None = None) -> ThreadingHTTPServer | None:
    """Start the endpoint on 127.0.0.1 in a background thread. None when the port is taken (logged)."""
    port = int(port if port is not None else cfg.get("ports.voice"))
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as ex:
        LOG.warning(f"clone speech endpoint not started on port {port}: {ex}")
        return None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="clone-server", daemon=True).start()
    LOG.info(f"clone speech on 127.0.0.1:{port}")
    return srv
