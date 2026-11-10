"""Kokoro v1.0 TTS through kokoro-onnx, for platforms without MLX.

  Kokoro(voice="af_heart").generate(text) -> yields (float32 audio, sample_rate)

Same model and voices the Apple Silicon path runs on MLX. Unknown words (names,
jargon) go through espeak-ng, which must be installed system-wide (apt install
espeak-ng, or brew install espeak-ng). The model files are declared in
config/defaults.toml ([models.files.tts_kokoro*]); fetch them with `mise run models`.
"""

import ctypes.util
import glob
import os
import pathlib
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from liberational import config as cfg
from liberational import model_runtime

if TYPE_CHECKING:
    from kokoro_onnx import EspeakConfig

ESPEAK_HINT = "espeak-ng not found: install it (apt install espeak-ng, or brew install espeak-ng)"


def _words(n: float, **kw: str) -> str:
    from num2words import num2words

    return num2words(n, **kw).replace("-", " ").replace(",", "")


def _clock(m: re.Match[str]) -> str:
    h, mm = int(m[1]), int(m[2])
    hour = _words(h) if h else "twelve"
    if m[3]:
        mins = "" if mm == 0 else f" oh {_words(mm)}" if mm < 10 else f" {_words(mm)}"
        return f"{hour}{mins} {m[3].lower()} m"
    if mm == 0:
        return f"{hour} hundred" if h > 12 else f"{hour} o'clock"
    return f"{hour} oh {_words(mm)}" if mm < 10 else f"{hour} {_words(mm)}"


def _digits(run: str) -> str:
    """Each digit of run read on its own ("1,203" -> "one two zero three")."""
    return " ".join(_words(int(d)) for d in run if d.isdigit())


# num2words reads English numbers below 10**306 and raises OverflowError past that, so a longer run is read digit by
# digit before any other rule sees it.
_MAX_WORD_DIGITS = 306


def _long_run(m: re.Match[str]) -> str:
    return _digits(m[0]) if sum(c.isdigit() for c in m[0]) > _MAX_WORD_DIGITS else m[0]


def _version(m: re.Match[str]) -> str:
    """A version number part by part, a lone zero as "oh": "v2.0.1" -> "v two point oh point one"."""
    parts = " point ".join("oh" if p == "0" else _words(int(p)) for p in m[2].split("."))
    return f"{m[1]} {parts}" if m[1] else parts


def _money(m: re.Match[str]) -> str:
    if m[2] and len(m[2]) > 2:  # finer than a cent ("$1.005"): read the amount as a decimal
        return f"{_words(float(m[1].replace(',', '') + '.' + m[2]))} dollars"
    whole = int(m[1].replace(",", ""))
    cents = int(m[2].ljust(2, "0")) if m[2] else 0
    out = f"{_words(whole)} dollar{'' if whole == 1 else 's'}"
    return f"{out} {_words(cents)}" if cents else out


def _decade(m: re.Match[str]) -> str:
    year = _words(int(m[1]), to="year")
    return year[:-1] + "ies" if year.endswith("y") else year + "s"


def speakable(text: str) -> str:
    """Spell out numbers and number symbols the way a person would say them."""
    t = re.sub(r"(?<![\d.])\d+(?:,\d{3})*", _long_run, text)
    t = re.sub(r"\b(\d{1,2}):(\d{2})(?:\s*([AaPp])\.?[Mm](?:\.(?=\s+[a-z]))?)?(?!\w)", _clock, t)
    t = re.sub(r"\$(\d+(?:,\d{3})*)(?:\.(\d+))?(?!\d)", _money, t)
    t = re.sub(r"\b([vV](?=\d)|(?=\d+(?:\.\d+){2,}\b))(\d+(?:\.\d+)*)\b", _version, t)
    t = re.sub(
        r"\b(\d{3})-(\d{4})\b|\b(\d{3})-(\d{3})-(\d{4})\b",
        lambda m: " ".join(" ".join(_words(int(d)) for d in g) for g in m.groups() if g),
        t,
    )
    t = re.sub(r"(?:(?<=\s)|^)([+-])(?=\d)", lambda m: "plus " if m[1] == "+" else "minus ", t)
    t = re.sub(r"\s?°\s?F\b", " degrees Fahrenheit", t)
    t = re.sub(r"\s?°\s?C\b", " degrees Celsius", t)
    t = re.sub(r"\s?°", " degrees", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: f"{_words(float(m[1]) if '.' in m[1] else int(m[1]))} percent", t)
    t = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: _words(int(m[1]), to="ordinal"), t)
    t = re.sub(r"\b(1[1-9]\d0|20\d0)s\b", _decade, t)
    t = re.sub(
        r"\b(in|since|by|from|until|of|around|year)\s+(1[1-9]\d\d|20\d\d)\b(?![.,]\d)",
        lambda m: f"{m[1]} {_words(int(m[2]), to='year')}",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\d+(?:,\d{3})*(?:\.\d+)?",
        lambda m: _words(float(m[0].replace(",", ""))) if "." in m[0] else _words(int(m[0].replace(",", ""))),
        t,
    )
    return t


def espeak_config() -> EspeakConfig:
    """Where the system espeak-ng library and its data are (kokoro-onnx's bundled copy doesn't load)."""
    from kokoro_onnx import EspeakConfig

    lib = (
        os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
        or ctypes.util.find_library("espeak-ng")
        or next(
            (
                p
                for p in ("/opt/homebrew/lib/libespeak-ng.dylib", "/usr/local/lib/libespeak-ng.dylib")
                if os.path.exists(p)
            ),
            None,
        )
    )
    data = [os.environ.get("ESPEAK_DATA_PATH", "")] + glob.glob("/usr/lib/*/espeak-ng-data")
    data += ["/usr/share/espeak-ng-data", "/usr/local/share/espeak-ng-data", "/opt/homebrew/share/espeak-ng-data"]
    data_path = next((d for d in data if d and os.path.exists(os.path.join(d, "phontab"))), None)
    if not lib or not data_path:
        raise RuntimeError(ESPEAK_HINT)
    return EspeakConfig(lib_path=lib, data_path=data_path)


class Kokoro:
    """Kokoro 82M v1.0. Output is 24 kHz mono float32, one sentence at a time."""

    def __init__(self, voice: str = "af_heart", speed: float = 1.1) -> None:
        from kokoro_onnx import Kokoro as KokoroOnnx

        model, voices = cfg.model_path("tts_kokoro"), cfg.model_path("tts_kokoro_voices")
        for p in (model, voices):
            if not pathlib.Path(p).exists():
                raise FileNotFoundError(f"Kokoro model not found at {p}; run `mise run models`")
        import onnxruntime as ort

        # CPU, as with the plain onnxruntime build: given the CUDA build, kokoro-onnx would ask for every provider it
        # lists, TensorRT included, which fails without the TensorRT libraries.
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.tts = KokoroOnnx.from_session(session, str(voices), espeak_config=espeak_config())
        if voice not in self.tts.get_voices():
            raise ValueError(f"unknown Kokoro voice {voice!r}; available: {', '.join(self.tts.get_voices())}")
        self.voice, self.speed, self.sample_rate = voice, speed, 24000
        self.name = f"Kokoro ({voice}, kokoro-onnx)"
        model_runtime.loaded("models.files.tts_kokoro", model.name, "voice", "CPU", owner=self)

    def generate(self, text: str) -> Iterator[tuple[NDArray[np.float32], int]]:
        # One sentence at a time so playback can start before the whole reply is synthesized.
        for sentence in (s for s in re.split(r"(?<=[.!?])\s+", speakable(text).strip()) if s):
            audio, sr = self.tts.create(sentence, voice=self.voice, speed=self.speed, lang="en-us")
            if len(audio):
                yield np.asarray(audio, dtype=np.float32), sr

    def synth(self, text: str) -> tuple[NDArray[np.float32], int]:
        parts: list[NDArray[np.float32]] = []
        sr = self.sample_rate
        for a, sr in self.generate(text):
            parts.append(a)
        return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), sr
