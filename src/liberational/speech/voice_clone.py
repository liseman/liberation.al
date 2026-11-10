"""Local zero-shot voice cloning (no training): Qwen3-TTS on MLX (Apple Silicon) or on CUDA (Linux, NVIDIA).

  python voice_clone.py record            # record 20 s of your voice -> owner.voice_ref
  python voice_clone.py say "text" [--ref data/voices/owner_ref.wav] [--out x.wav] [--play]
  python voice_clone.py bench [--engine qwen3|chatterbox]
After recording, the .txt next to the reference holds the transcript (auto-written from the prompt).
Engines, in order: MLX on Apple Silicon (mlx-audio); CUDA on Linux with an NVIDIA GPU and the voice-clone extra
(speech/qwen3_cuda.py, the same Qwen3-TTS model); otherwise say() speaks with Kokoro (kokoro-onnx), without cloning.

Tips for the reference clip: quiet room, normal speaking pace, 10-20 s, one
continuous take, no music. Read a paragraph rather than isolated words.
"""

import argparse
import contextlib
import os
import pathlib
import queue
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import soundfile as sf

from liberational import config as cfg
from liberational import logs, model_runtime

DEFAULT_REF = cfg.path("owner.voice_ref")
ENGINES = {
    # engine: (repo, needs_ref_text). Benched 2026-10-07 on M5 Pro, warm, 3.8 s of speech:
    "qwen3": (cfg.hf_repo("tts_clone"), True),  # ~150 ms to first audio
    "chatterbox": ("mlx-community/chatterbox-turbo-fp16", False),  # ~600 ms to first audio
}
ENGINE = "qwen3"
PROMPT = (
    "The quick brown fox jumps over the lazy dog. I'm recording a short reference clip "
    "so Liberational can speak in my voice. It should sound natural, relaxed, and clear, "
    "the way I'd actually talk to someone standing next to me."
)

_m: Any = None  # an mlx-audio model, a Qwen3Cuda, or a Kokoro
KIND: str | None = None  # "mlx", "cuda", or "kokoro" once load() has run


LOG = logs.get("voice-clone")


def can_clone() -> bool:
    """Whether this machine can clone a voice: MLX on Apple Silicon, or CUDA with the voice-clone extra."""
    if cfg.has_mlx():
        return True
    from liberational.speech import qwen3_cuda

    return qwen3_cuda.available()[0]


def load(engine: str | None = None) -> Any:
    global _m, ENGINE, KIND
    if engine:
        ENGINE = engine
    if _m is None:
        if cfg.has_mlx():
            from mlx_audio.tts.utils import load as load_tts

            repo = ENGINES[ENGINE][0]
            _m, KIND = load_tts(repo), "mlx"
            key = "models.hf.tts_clone" if ENGINE == "qwen3" else f"models.runtime.voice_clone.{ENGINE}"
            model_runtime.loaded(key, repo, "voice", "Apple GPU", owner=_m)
            LOG.info(f"engine: {ENGINE} on MLX")
        else:
            from liberational.speech import qwen3_cuda

            ok, why = qwen3_cuda.available()
            if ok:
                _m, KIND = qwen3_cuda.load(), "cuda"
                LOG.info(f"engine: {_m.name}")
            else:
                from liberational.speech.kokoro import Kokoro

                LOG.warning(f"no voice cloning ({why}); speaking with Kokoro (kokoro-onnx)")
                _m, KIND = Kokoro(voice=cfg.get("tts.voice"), speed=1.1), "kokoro"
    return _m


def ref_text_for(ref: str | os.PathLike[str]) -> str:
    """Qwen3-TTS needs the transcript of the reference clip. Use <ref>.txt if present, else PROMPT."""
    t = pathlib.Path(ref).with_suffix(".txt")
    return t.read_text().strip() if t.exists() else PROMPT


def record(path: str | os.PathLike[str] | None = None, secs: float = 20) -> None:
    import sounddevice as sd

    path = pathlib.Path(path or DEFAULT_REF)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Read this aloud ({secs}s):\n\n  {PROMPT}\n")
    input("Press Enter, then speak...")
    a = sd.rec(int(secs * 24000), samplerate=24000, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, a[:, 0], 24000)
    path.with_suffix(".txt").write_text(PROMPT)
    print(f"saved {path}")


def say(
    text: str,
    ref: str | os.PathLike[str] | None = None,
    out: str | os.PathLike[str] | None = None,
    play: bool = False,
    **kw: object,
) -> tuple[np.ndarray, int]:
    ref = ref or DEFAULT_REF
    m = load()
    t0 = time.perf_counter()
    first = None
    parts = []
    sr = 24000
    if KIND == "kokoro":  # no reference, no clone-only kwargs
        for a, sr in m.generate(text):
            if first is None:
                first = time.perf_counter() - t0
            parts.append(a)
    else:
        if KIND == "cuda" or ENGINES[ENGINE][1]:
            kw["ref_text"] = ref_text_for(ref)
        for r in m.generate(text=text, ref_audio=str(ref), stream=True, streaming_interval=0.5, **kw):
            if first is None:
                first = time.perf_counter() - t0
            parts.append(np.array(r.audio))
            sr = r.sample_rate
    if not parts:
        raise RuntimeError("TTS returned no audio")
    assert first is not None  # set with the first part
    audio = np.concatenate(parts)
    dur = len(audio) / sr
    LOG.info(
        f"first synthesized chunk {1000 * first:.0f} ms | synthesis total {1000 * (time.perf_counter() - t0):.0f} ms | {dur:.1f}s speech"
    )
    if out:
        sf.write(out, audio, sr)
    if play:
        play_audio(audio, sr)
    return audio, sr


def play_audio(audio: np.ndarray, sr: int) -> None:
    """On macOS use a fresh CoreAudio player so long-lived workers follow output changes.

    PortAudio can cache the default device at initialization. afplay resolves the
    current macOS output for each utterance; no input device is opened. Elsewhere
    play through sounddevice.
    """
    from liberational.speech.playback_guard import playback_window

    if sys.platform != "darwin":
        import sounddevice as sd

        with playback_window(len(audio) / sr):
            sd.play(np.asarray(audio, dtype=np.float32), sr)
            sd.wait()
        return
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="pack-speech-") as tmp:
        path = pathlib.Path(tmp) / "speech.wav"
        sf.write(path, audio, sr)
        with playback_window(len(audio) / sr):
            subprocess.run(["/usr/bin/afplay", str(path)], check=True, timeout=max(15, len(audio) / sr + 10))


def play_chunks(chunks: Iterable[tuple[np.ndarray, int]]) -> None:
    """Play synthesized chunks as they arrive, with bounded buffering and one output stream per reply.

    Model generation stays on the calling thread (required by MLX). A writer drains audio concurrently so
    blocking playback does not serialize synthesis with speech. macOS keeps its fresh CoreAudio player path.
    Each write and the final device drain remain covered by the playback exclusion window.
    """
    if sys.platform == "darwin":
        parts: list[np.ndarray] = []
        sample_rate = 24000
        for audio, sample_rate in chunks:
            parts.append(audio)
        if parts:
            play_audio(np.concatenate(parts), sample_rate)
        return

    import sounddevice as sd

    from liberational.speech.dsp import resample
    from liberational.speech.playback_guard import playback_window

    pending: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(maxsize=4)
    stopped = threading.Event()
    errors: list[BaseException] = []

    def write() -> None:
        try:
            first = pending.get()
            if first is None:
                return
            rate = first[1]
            output_context = contextlib.ExitStack()
            try:
                with playback_window(0):
                    output = output_context.enter_context(sd.OutputStream(samplerate=rate, channels=1, dtype="float32"))
                item = first
                while item is not None:
                    audio, sr = item
                    if sr != rate:
                        audio = resample(audio, sr, rate)
                    with playback_window(len(audio) / rate):
                        output.write(np.ascontiguousarray(audio, dtype=np.float32).reshape(-1, 1))
                        # The device can still have buffered audio while generation makes the next chunk.
                        item = pending.get()
            finally:
                # Context exit drains the device, including when a write or synthesis failed.
                try:
                    with playback_window(0):
                        output_context.close()
                finally:
                    # A guard-storage failure must not leave the device open. close() is idempotent.
                    output_context.close()
        except BaseException as error:
            errors.append(error)
        finally:
            stopped.set()

    def offer(item: tuple[np.ndarray, int] | None) -> None:
        while not stopped.is_set():
            try:
                pending.put(item, timeout=0.1)
                return
            except queue.Full:
                pass
        if errors:
            raise errors[0]

    iterator = iter(chunks)
    writer = threading.Thread(target=write, name="voice-playback")
    writer.start()
    try:
        for audio, sr in iterator:
            # A model may reuse its output buffer when generation resumes on this thread.
            offer((np.array(audio, dtype=np.float32, copy=True), sr))
    finally:
        try:
            # A writer failure can leave generation paused at a yield while holding a CUDA/MLX lock.
            # Close it on the model's own thread, even when a retained traceback keeps the iterator alive.
            if close := getattr(iterator, "close", None):
                close()
        finally:
            try:
                offer(None)
            finally:
                writer.join()
    if errors:
        raise errors[0]


def bench(ref: str | os.PathLike[str] | None = None) -> None:
    ref = pathlib.Path(ref or DEFAULT_REF)
    if not ref.exists():
        print("no reference clip; synthesizing a stand-in with Kokoro (run 'record' for your real voice)")
        # Never at owner.voice_ref: the ambient daemon would enroll this synthetic voice as the owner.
        ref = pathlib.Path(tempfile.mkdtemp(prefix="pack-bench-")) / "standin.wav"
        if cfg.has_mlx():
            from mlx_audio.tts.utils import load as load_tts

            k = load_tts(cfg.hf_repo("tts_kokoro"))
            a = np.concatenate([np.array(o.audio) for o in k.generate(text=PROMPT, voice="am_adam")])
        else:
            from liberational.speech.kokoro import Kokoro

            a, _ = Kokoro(voice="am_adam").synth(PROMPT)
        sf.write(ref, a, 24000)
    load()
    voices = cfg.path("paths.voices")
    voices.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        print(f"run {i + 1}:")
        say("Sure. The coffee place on the left is open until nine tonight.", ref=ref, out=voices / "test_out.wav")


def main() -> None:
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "say", "bench"])
    ap.add_argument("text", nargs="?")
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--out")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--secs", type=int, default=20)
    ap.add_argument("--engine", choices=list(ENGINES), default=ENGINE)
    a = ap.parse_args()
    ENGINE = a.engine
    if a.cmd == "record":
        record(pathlib.Path(a.ref), a.secs)
    elif a.cmd == "say":
        say(a.text or "Hello, this is a test.", ref=a.ref, out=a.out, play=a.play)
    else:
        bench(a.ref)


if __name__ == "__main__":
    main()
