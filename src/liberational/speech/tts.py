"""Speech synthesis: Kokoro or a Qwen3-TTS zero-shot clone, on MLX (Apple Silicon) or CUDA (Linux, NVIDIA);
Kokoro (kokoro-onnx) where neither is available.

tts = TTS(clone=False)
for audio, sr in tts.generate("Hello."): ...   # float32 mono chunks
"""

import contextlib
import math
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from liberational import config as cfg
from liberational import logs, model_runtime
from liberational.speech.dsp import resample

SR_TTS = 24000
KOKORO = cfg.hf_repo("tts_kokoro")
KOKORO_VOICE = cfg.get("tts.voice")
QWEN3_TTS = cfg.hf_repo("tts_clone")
REF_WAV = cfg.path("owner.voice_ref")
REF_TXT = REF_WAV.with_suffix(".txt")
mlx_lock = threading.RLock()  # serialize MLX work across threads (MLX is thread-affine)
_NOLOCK = contextlib.nullcontext()
RETRY_S = 300.0  # after the clone fails (its model, or a reply), Kokoro speaks this long before cloning is tried again
_now = time.monotonic


LOG = logs.get("tts")
Stamp = tuple[int, int, int, int]


def log(*a: object, level: int = logs.INFO) -> None:
    LOG.log(level, " ".join(str(x) for x in a))


class TTS:
    """Kokoro (fast, fixed voice) or Qwen3-TTS zero-shot clone. generate() yields (audio_f32, sr).
    Without MLX, cloning runs on CUDA (speech/qwen3_cuda.py) when available, and Kokoro runs on kokoro-onnx.

    The clone needs the owner's voice reference (owner.voice_ref), which can be deleted, re-recorded, or left unreadable
    while the service runs. Each reply checks it: while it's missing or can't be read, replies are spoken with
    Kokoro and one warning is logged; once it's back (or recorded again), cloning resumes. When the clone itself fails
    (its model won't load, or a reply fails before any audio), Kokoro speaks for RETRY_S, or until the reference
    changes, with one warning. The models load on first use, so a service started without a reference loads the
    clone when one appears."""

    def __init__(self, clone: bool = False) -> None:
        self.cuda = not cfg.has_mlx()
        self.want_clone = clone
        self.clone = False  # whether the reply being spoken (or the next) is cloned
        self._clone_m: Any = None  # a Qwen3Cuda or an mlx-audio model, loaded once there is a reference
        self._kokoro: Any = None
        self._lock = threading.Lock()
        self._ref_stamp: Stamp | None = None  # the reference loaded now
        self._warned: object = None  # the state last warned about, so each is warned about once
        self._lost = False  # the reference was missing or unreadable: say so when it's back
        self._down: tuple[Stamp, float] | None = None  # cloning is off for this reference until this time
        if clone and self.cuda:
            from liberational.speech import qwen3_cuda

            ok, why = qwen3_cuda.available()
            if not ok:
                log(f"no voice cloning ({why}); using Kokoro", level=logs.WARN)
                self.want_clone = False
        if not self._refresh():
            self._kokoro_model()  # load it now, not on the first reply
        self.name = self._name()

    # ---- the reference ----

    @staticmethod
    def _stamp() -> Stamp | None:
        """The reference's version: the wav's inode, mtime, and size (a re-recording is replaced, a new inode), and its
        transcript's mtime. None when there's no wav."""
        try:
            st = REF_WAV.stat()
        except OSError:
            return None
        try:
            txt = REF_TXT.stat().st_mtime_ns
        except OSError:
            txt = 0
        return st.st_ino, st.st_mtime_ns, st.st_size, txt

    def _warn(self, state: object, msg: str) -> None:
        """One warning for each state (the reference missing, a version of it unreadable, or the clone failing)."""
        if self._warned != state:
            self._warned = state
            log(msg, level=logs.WARN)

    def _refresh(self) -> bool:
        """Check the reference before a reply: load it when it's new or changed. True when this reply is cloned."""
        if not self.want_clone:
            return False
        with self._lock:
            stamp = self._stamp()
            if stamp is not None and stamp == self._ref_stamp:
                return self.clone
            self._set_clone(False)
            self._ref_stamp = None
            if stamp is None:
                self._lost = True
                self._warn("missing", f"voice reference {REF_WAV} missing; speaking with Kokoro until it's back")
                return False
            if self._down and self._down[0] == stamp and _now() < self._down[1]:
                return False  # this reference already failed; wait for it to change, or for the retry
            try:
                self._load_reference()
            except Exception as ex:
                self._lost = True
                self._down = (stamp, math.inf)  # until the file changes
                self._warn(stamp, f"voice reference {REF_WAV} can't be read ({ex!r}); speaking with Kokoro")
                return False
            if self._clone_m is None:
                try:
                    self._clone_m = self._load_clone()
                except Exception as ex:
                    self._down = (stamp, _now() + RETRY_S)
                    self._warn("model", f"the voice clone model failed to load ({ex!r}); speaking with Kokoro")
                    return False
            if self._lost:
                log(f"voice reference {REF_WAV} is back; cloning again")
            self._lost, self._down, self._warned, self._ref_stamp = False, None, None, stamp
            self._set_clone(True)
            return True

    def _failed(self, ex: Exception) -> None:
        """A cloned reply failed before any audio: Kokoro speaks until RETRY_S passes or the reference changes."""
        with self._lock:
            if self._ref_stamp is not None:
                self._down = (self._ref_stamp, _now() + RETRY_S)
            self._ref_stamp = None
            self._set_clone(False)
            self._warn("clone", f"cloning failed ({ex!r}); speaking with Kokoro")

    def _set_clone(self, on: bool) -> None:
        if self.clone != on:
            self.clone = on
            self.name = self._name()

    def _load_reference(self) -> None:
        """Read the reference and its words. A file that can't be decoded raises here, not halfway into a reply."""
        import soundfile as sf

        a, sr = sf.read(str(REF_WAV), dtype="float32")
        if self.cuda:
            from liberational.speech.voice_clone import ref_text_for

            # faster-qwen3-tts reads the file itself (and notices a re-recording by its mtime)
            self.ref_audio: Any = str(REF_WAV)
            self.ref_text = ref_text_for(REF_WAV)
            return
        # Load the reference once; passing a path makes mlx-audio re-read/decode the wav per sentence.
        import mlx.core as mx

        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != SR_TTS:
            a = resample(a, sr, SR_TTS)
        self.ref_audio = mx.array(a)
        if REF_TXT.exists():
            self.ref_text = REF_TXT.read_text().strip()
        else:
            from liberational.speech.voice_clone import PROMPT

            self.ref_text = PROMPT
            log(f"{REF_TXT.name} missing; assuming the reference clip reads voice_clone.PROMPT", level=logs.WARN)

    # ---- the models ----

    def _load_clone(self) -> Any:
        if self.cuda:
            from liberational.speech import qwen3_cuda

            return qwen3_cuda.load()
        from mlx_audio.tts.utils import load as load_tts

        model = load_tts(QWEN3_TTS)
        model_runtime.loaded("models.hf.tts_clone", QWEN3_TTS, "voice", "Apple GPU", owner=model)
        return model

    def _kokoro_model(self) -> Any:
        if self._kokoro is None:
            if self.cuda:
                from liberational.speech.kokoro import Kokoro

                self._kokoro = Kokoro(voice=KOKORO_VOICE, speed=1.1)
            else:
                from mlx_audio.tts.utils import load as load_tts

                self._kokoro = load_tts(KOKORO)
                model_runtime.loaded("models.hf.tts_kokoro", KOKORO, "voice", "Apple GPU", owner=self._kokoro)
        return self._kokoro

    def _name(self) -> str:
        if self.clone:
            return f"Qwen3-TTS clone ({REF_WAV.name})" + (" on CUDA" if self.cuda else "")
        k = self._kokoro
        return str(k.name) if self.cuda and k is not None else f"Kokoro ({KOKORO_VOICE})"

    # ---- speech ----

    def _cloned(self, text: str) -> Iterator[tuple[NDArray[np.float32], int]]:
        for r in self._clone_m.generate(
            text=text, ref_audio=self.ref_audio, ref_text=self.ref_text, stream=True, streaming_interval=0.5
        ):
            yield np.asarray(r.audio, dtype=np.float32), r.sample_rate

    def _plain(self, text: str) -> Iterator[tuple[NDArray[np.float32], int]]:
        k = self._kokoro_model()
        if self.cuda:
            yield from k.generate(text)
            return
        for r in k.generate(text=text, voice=KOKORO_VOICE, speed=1.1):
            yield np.asarray(r.audio, dtype=np.float32), r.sample_rate

    def generate(self, text: str) -> Iterator[tuple[NDArray[np.float32], int]]:
        with mlx_lock if not self.cuda else _NOLOCK:  # the reference's MLX array is made under it too
            if self._refresh():
                started = False
                try:
                    for chunk in self._cloned(text):
                        started = True
                        yield chunk
                    return
                except Exception as ex:
                    if started:
                        raise  # half a reply in the cloned voice: don't finish it in another
                    self._failed(ex)
            yield from self._plain(text)

    def synth(self, text: str) -> tuple[NDArray[np.float32], int]:
        parts: list[NDArray[np.float32]] = []
        sr = SR_TTS
        for a, sr in self.generate(text):
            parts.append(a)
        return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), sr
