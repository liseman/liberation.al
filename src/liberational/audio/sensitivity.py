"""How sensitive each microphone is: one 0 to 100 setting that tunes the energy VAD and the speech gate.

Ambient drops speech at two gates before ASR: the energy VAD opens on a frame louder than
max(floor + margin_db, min_dbfs), and the Silero speech gate then drops an utterance with too little speech at its
threshold. One sensitivity moves all three together, linearly: 0 is strict (12 dB over the room floor, never below
-40 dBFS, Silero threshold 0.6) and 100 is loose (4 dB, -90 dBFS, 0.25).

The default, 70 (6.4 dB, -75 dBFS, 0.355), comes from 2.6 hours of the Linux machine's Elgato Wave 3, scored
against Silero run over the whole stream (468 speech stretches of at least 0.25 s, 24 minutes). The old fixed
values (9 dB, -45 dBFS, 0.5) caught 5 of them end to end, because this mic's speech peaks near -55 dBFS. At
6 dB, -75 dBFS, and 0.35 they caught 429, with 49 kept utterances holding no Silero speech (as many as at 0.4).

audio.sensitivity is the default for every mic, and [audio.mic_sensitivity] sets one mic by its device name (the
Sources drawer writes it). Ambient reads them again whenever pack.toml changes, so a change applies within about a
second without a restart.
"""

import dataclasses
import math
import os
import threading
import tomllib
from typing import Any

from liberational import config as cfg

TABLE = "mic_sensitivity"  # [audio.mic_sensitivity]: device name -> 0..100


@dataclasses.dataclass(frozen=True)
class Tuning:
    margin_db: float
    min_dbfs: float
    gate_threshold: float


def tuning(sensitivity: float) -> Tuning:
    s = min(100.0, max(0.0, float(sensitivity)))
    return Tuning(margin_db=12 - 0.08 * s, min_dbfs=-40 - 0.5 * s, gate_threshold=0.6 - 0.0035 * s)


def _checked(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{where} must be a number from 0 to 100")
    if not 0 <= value <= 100:
        raise ValueError(f"{where} must be from 0 to 100")
    return round(value)


_lock = threading.Lock()
_seen: tuple[Any, dict[str, Any]] | None = None  # (pack.toml's stat, its [audio] table)


def _local_audio() -> dict[str, Any]:
    """pack.toml's [audio] table, parsed again only when the file changes."""
    global _seen
    path = cfg.local_config()
    try:
        st = path.stat()
        stamp: Any = (st.st_mtime_ns, st.st_size, st.st_ino)
    except FileNotFoundError:
        stamp = None
    with _lock:
        if _seen is not None and _seen[0] == stamp:
            return _seen[1]
        table = tomllib.loads(path.read_text()).get("audio", {}) if stamp is not None else {}
        _seen = (stamp, table if isinstance(table, dict) else {})
        return _seen[1]


def stamp() -> Any:
    """The stamp of pack.toml's [audio] table as `_local_audio` last read it (None when there is no file), so a
    caller that caches a tuning can tell when a rewrite moved it."""
    _local_audio()
    return _seen[0] if _seen is not None else None


def shipped() -> int:
    """audio.sensitivity as defaults.toml ships it."""
    return _checked(tomllib.loads(cfg.DEFAULTS.read_text())["audio"]["sensitivity"], "audio.sensitivity")


def default() -> int:
    """audio.sensitivity now: from the environment, else pack.toml as it is now, else defaults.toml. (Not the
    process's cached config, which would keep a line removed from pack.toml since it started.)"""
    if os.environ.get("PACK_AUDIO_SENSITIVITY"):
        return _checked(cfg.get("audio.sensitivity"), "audio.sensitivity")
    local = _local_audio()
    return _checked(local["sensitivity"], "audio.sensitivity") if "sensitivity" in local else shipped()


def value(device: str) -> int:
    """This mic's sensitivity now: its own entry in [audio.mic_sensitivity], else audio.sensitivity."""
    table = _local_audio().get(TABLE)
    if not isinstance(table, dict):
        table = {}
    if device in table:
        return _checked(table[device], f"audio.{TABLE}.{device}")
    return default()


def save(device: str, sensitivity: int | None) -> None:
    """Set one mic's sensitivity in pack.toml (None: back to audio.sensitivity), keeping the rest of the file."""
    import tomlkit

    if sensitivity is not None:
        sensitivity = _checked(sensitivity, "sensitivity")
    with cfg.LOCAL_WRITE:  # the same lock as cfg.save, so two saves can't lose each other's change
        path = cfg.local_config()
        doc = tomlkit.parse(path.read_text()) if path.exists() else tomlkit.document()
        audio = doc.setdefault("audio", tomlkit.table())
        table = audio.setdefault(TABLE, tomlkit.table())
        if sensitivity is None:
            table.pop(device, None)
        else:
            table[device] = sensitivity
        if not table:
            del audio[TABLE]
        if not audio:
            del doc["audio"]
        cfg.write_local(tomlkit.dumps(doc))
