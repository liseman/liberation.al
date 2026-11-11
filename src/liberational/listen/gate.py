"""Speech detection for the listen worker: an adaptive energy VAD cuts candidate utterances, and Silero drops
non-speech before ASR. Per-mic tuning comes from audio/sensitivity.py. A rolling lower-decile level estimate keeps
learning during an open candidate, so amplified room noise cannot freeze a low startup floor until every clip
reaches the duration cap. The window spans syllables, and upward changes are bounded; ordinary speech modulation
continues opening the gate while sentence pauses close it."""

from __future__ import annotations

import collections
import logging
import pathlib
import subprocess
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from liberational import config as cfg
from liberational import model_runtime
from liberational.audio import sensitivity

if TYPE_CHECKING:
    import sherpa_onnx

log = logging.getLogger(__name__)

SR = 16000
FRAME = SR // 50  # 20 ms
PAD_MS = 300
GATE_MODEL = cfg.model_path("vad_silero")
GATE_MODEL_URL = cfg.get("models.files.vad_silero.url")
NEAR_DB = 6.0  # a stretch this close to the VAD's threshold that never opens is logged as a near miss
NOISE_WINDOW = 100  # two seconds of 20 ms frame levels, spanning speech syllables and their quiet gaps
NOISE_READY = 50  # wait for one second before adapting an open candidate's floor
NOISE_UPDATE = 10  # refresh the lower decile every 200 ms, not every frame
NOISE_RISE_DB = 0.2  # at most 10 dB/s while open, so one loud frame cannot raise the floor


class VAD:
    """Energy VAD, 20 ms frames, adaptive floor. feed(block16k) -> yields (start_idx, end_idx, audio) in absolute 16 kHz samples,
    padded PAD_MS on both sides."""

    def __init__(
        self,
        start_ms: int = 60,
        end_ms: int = 600,
        min_ms: int = 300,
        max_s: float = 20,
        margin_db: float = 9,
        min_dbfs: float = -45,
        pad_ms: int = PAD_MS,
    ) -> None:
        self.start_n, self.end_n, self.min_n, self.pad_n = start_ms // 20, end_ms // 20, min_ms // 20, pad_ms // 20
        self.max_frames = max_s * 50
        self.margin, self.min_dbfs, self.floor = margin_db, min_dbfs, -60.0
        self.pre = collections.deque(maxlen=self.pad_n)
        self.buf, self.in_speech, self.run, self.sil = [], False, 0, 0
        self.rem, self.idx, self.start_idx = np.zeros(0, np.float32), 0, 0
        self.level = -90.0
        # Stretches that came within NEAR_DB of opening but never did: (peak dBFS, threshold dBFS), for the log.
        self.near_misses: collections.deque[tuple[float, float]] = collections.deque(maxlen=100)
        self.near, self.near_peak = 0, -120.0
        self.noise_levels: collections.deque[float] = collections.deque(maxlen=NOISE_WINDOW)
        self.noise_floor: float | None = None

    @property
    def threshold(self) -> float:
        """The level a frame has to pass to count as loud now (dBFS)."""
        return max(self.floor + self.margin, self.min_dbfs)

    def tune(self, margin_db: float, min_dbfs: float) -> None:
        """Change how loud speech has to be, from the next frame on (audio/sensitivity.py)."""
        self.margin, self.min_dbfs = float(margin_db), float(min_dbfs)

    def shift(self, db: float) -> None:
        """The mic's gain changed by `db` (auto gain): move what the VAD has learned of its levels with it. The
        floor rises at most 0.05 dB a frame, and only while the VAD is closed, so after a gain jump every frame read
        as loud and the VAD stayed open in max_s stretches (on the Linux machine, the Wave 3 from 0 to 30 dB)."""
        self.floor += db
        self.level += db
        self.noise_levels = collections.deque((level + db for level in self.noise_levels), maxlen=NOISE_WINDOW)
        if self.noise_floor is not None:
            self.noise_floor += db
        if self.near:
            self.near_peak += db

    def feed(self, block: np.ndarray) -> Iterator[tuple[int, int, np.ndarray]]:
        x = np.concatenate([self.rem, block.astype(np.float32)])
        n = len(x) // FRAME
        self.rem = x[n * FRAME :]
        for i in range(n):
            f = x[i * FRAME : (i + 1) * FRAME]
            fidx = self.idx
            self.idx += FRAME
            db = 20 * np.log10(np.sqrt(np.mean(f * f)) + 1e-9)
            self.level = db
            self.noise_levels.append(float(db))
            if len(self.noise_levels) >= NOISE_READY and self.idx // FRAME % NOISE_UPDATE == 0:
                self.noise_floor = float(np.percentile(self.noise_levels, 10))
            if not self.in_speech:
                self.floor = min(db, self.floor + 0.05) if db > self.floor else 0.9 * self.floor + 0.1 * db
            elif self.noise_floor is not None:
                # The old estimator froze here, so an elevated room floor could hold the VAD open forever.
                # Lower-tail levels leave room above the quiet parts of speech; a brief loud syllable is not noise.
                target = self.noise_floor
                self.floor = min(target, self.floor + NOISE_RISE_DB) if target > self.floor else target
            threshold = max(self.floor + self.margin, self.min_dbfs)
            loud = db > threshold
            if not self.in_speech:
                # Near: within NEAR_DB of the threshold, and at least halfway up from the room floor to it, so
                # the floor's own wobble isn't a miss when the threshold sits close to it.
                if db > max(threshold - NEAR_DB, self.floor + self.margin / 2):
                    self.near, self.near_peak = self.near + 1, max(self.near_peak, db)
                else:
                    if self.near >= self.start_n:
                        self.near_misses.append((round(self.near_peak, 1), round(threshold, 1)))
                    self.near, self.near_peak = 0, -120.0
                self.pre.append(f)
                self.run = self.run + 1 if loud else 0
                if self.run >= self.start_n:
                    self.near, self.near_peak = 0, -120.0  # it opened: not a miss
                    self.in_speech, self.buf, self.sil = True, list(self.pre), 0
                    self.start_idx = fidx + FRAME - len(self.buf) * FRAME
            else:
                self.buf.append(f)
                self.sil = 0 if loud else self.sil + 1
                if self.sil >= self.end_n or len(self.buf) >= self.max_frames:
                    keep = len(self.buf) - max(0, self.sil - self.pad_n)
                    utt = np.concatenate(self.buf[:keep])
                    s0 = self.start_idx
                    self.in_speech, self.run, self.buf = False, 0, []
                    self.pre.clear()
                    if len(utt) >= (self.min_n + 2 * self.pad_n) * FRAME:
                        yield s0, s0 + len(utt), utt


class SpeechGate:
    """Second opinion on each energy-VAD utterance, before ASR: Silero VAD (sherpa-onnx, CPU).

    The energy VAD opens on anything 9 dB over the room floor, so knocks, clicks, typing and scrapes become
    utterances. Parakeet then often decodes them as "Yeah." or "Okay.", and CAM++ assigns those to the nearest
    voice (usually the owner's: noise clips scored 0.58-0.89 against it, the same range as their real speech), so
    the transcript fills with one-word lines nobody said. The gate drops an utterance when Silero finds less than
    min_speech_s of speech in it. Measured on a private clip set (45 reSpeaker clips, 13 of them non-speech
    that whisper had captioned "you" / "Thank you."): threshold 0.5 kept 32/32 speech clips (least speech found:
    0.46 s) and dropped 13/13 noise clips. On ESC-50 household sounds it dropped about 9 in 10 knocks, clicks, typing
    and pours; coughs, laughs, sneezes and breathing mostly pass. About 12 ms per utterance on one CPU thread, which
    the skipped ASR call more than pays back.

    gate = SpeechGate(); gate(audio16k) -> True to transcribe, False to drop. gate.last_speech_s is the measurement."""

    WINDOW = 512  # Silero's 32 ms window at 16 kHz

    def __init__(
        self, threshold: float = 0.5, min_speech_s: float = 0.25, model: str | pathlib.Path = GATE_MODEL
    ) -> None:
        self.threshold, self.min_speech_s, self.model_path = float(threshold), float(min_speech_s), pathlib.Path(model)
        self._vad, self.last_speech_s = None, None

    def detector(self) -> sherpa_onnx.VoiceActivityDetector:
        if self._vad is None:
            import sherpa_onnx

            if not self.model_path.exists():
                log.info("speech gate model missing, downloading %s ...", self.model_path.name)
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["curl", "-sSL", "-o", str(self.model_path), GATE_MODEL_URL], check=True)
            c = sherpa_onnx.VadModelConfig()
            c.sample_rate, c.num_threads, c.provider = SR, 1, "cpu"
            c.silero_vad.model, c.silero_vad.threshold = str(self.model_path), self.threshold
            c.silero_vad.min_silence_duration, c.silero_vad.min_speech_duration = 0.1, 0.05
            c.silero_vad.window_size = self.WINDOW
            self._vad = sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=60)
            model_runtime.loaded("models.files.vad_silero", self.model_path.name, "speech detection", "CPU", owner=self)
        return self._vad

    def speech_s(self, audio16k: np.ndarray) -> float:
        """Seconds of speech Silero finds in the clip (fresh state per clip)."""
        v = self.detector()
        v.reset()
        a = np.ascontiguousarray(audio16k, dtype=np.float32)
        for i in range(0, len(a), self.WINDOW):  # one whole-clip accept_waveform yields nothing in sherpa-onnx 2.0
            v.accept_waveform(a[i : i + self.WINDOW])
        v.flush()
        n = 0
        while not v.empty():
            n += len(v.front.samples)
            v.pop()
        return n / SR

    def __call__(self, audio16k: np.ndarray) -> bool:
        self.last_speech_s = self.speech_s(audio16k)
        return self.last_speech_s >= self.min_speech_s


def speech_gate(threshold: float | None = None) -> SpeechGate | None:
    """The configured gate at threshold (audio.sensitivity's by default), or None when audio.speech_gate is off."""
    if not cfg.get("audio.speech_gate"):
        return None
    if threshold is None:
        threshold = sensitivity.tuning(sensitivity.default()).gate_threshold
    return SpeechGate(threshold, cfg.get("audio.speech_gate_min_s"))


class Gates:
    """One speech gate per device, tuned to that device's sensitivity, rebuilt when the sensitivity's stamp moves.

    The transcriber runs on one pool thread and its gate is stateless between calls except last_speech_s, so a
    separate SpeechGate per microphone is safe and keeps a loud mic from setting the threshold for every other."""

    def __init__(self) -> None:
        self._stamp: Any = None
        self._gates: dict[str, SpeechGate | None] = {}

    def __call__(self, device: str) -> SpeechGate | None:
        """This device's gate now. `device` is the microphone's name or its stream's `<name>.audio`."""
        name = device.removesuffix(".audio")
        stamp = sensitivity.stamp()
        if stamp != self._stamp:
            self._stamp = stamp
            self._gates.clear()
        if name not in self._gates:
            self._gates[name] = speech_gate(sensitivity.tuning(sensitivity.value(name)).gate_threshold)
        return self._gates[name]
