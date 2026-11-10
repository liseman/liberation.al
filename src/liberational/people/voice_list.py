"""The voice list on Continuum: every voiceprint with how often and when it was heard.

v1 kept the list in `voices.sqlite3`, a row per voice with its centroid, references, line count, and first and last
time heard. Listen (listen/voices.py) now gives each line a voiceprint number and files the line's embedding on the
broker, so the list is read from those:

- **Voices.** A voiceprint is listed when a line on the speech log carries its number (`speaker_id`), or when the
  event log holds a claim about it (a name, a voice tag's voiceprint for a person) even with no line of its own yet.
- **Counts.** `n` is the lines heard in that voice, and `first_seen` and `last_seen` the start of its first and last
  line (UTC seconds). A voiceprint known only to the event log has none of these: `n` is 0 and the times are None.
- **Prints.** A voiceprint's embeddings are in the segment store (`voice.embedding`, each with its voiceprint and
  person), so the list holds none. Whatever compares a voice against another reads or searches them there.
- **Names.** The list holds no names, as in v1: the dashboard names each voice through the registry.

`Tally` counts lines; `VoiceList` keeps one over the speech log, reading only the lines committed since it last
read, and recounting from the start after a purge (which may have taken lines it counted).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol


class Tally:
    """Lines per voiceprint, and when each was first and last heard."""

    def __init__(self) -> None:
        self.voices: dict[int, dict[str, Any]] = {}

    def add(self, line: Mapping[str, Any]) -> None:
        sid, ts = line.get("speaker_id"), line.get("ts_start")
        if not isinstance(sid, int) or isinstance(sid, bool) or sid < 1:
            return  # an unmatched line (no voiceprint), or a malformed one
        if not isinstance(ts, (int, float)) or isinstance(ts, bool):
            return
        v = self.voices.get(sid)
        if v is None:
            self.voices[sid] = {"id": sid, "n": 1, "first_seen": float(ts), "last_seen": float(ts)}
            return
        v["n"] += 1
        v["first_seen"] = min(v["first_seen"], float(ts))
        v["last_seen"] = max(v["last_seen"], float(ts))

    def rows(self, known: Iterable[int] = ()) -> list[dict[str, Any]]:
        """Every voice, lowest number first; `known` adds voiceprints with no line (n 0, never seen)."""
        out = {sid: dict(v) for sid, v in self.voices.items()}
        for sid in known:
            if sid not in out:
                out[sid] = {"id": sid, "n": 0, "first_seen": None, "last_seen": None}
        return [out[sid] for sid in sorted(out)]


class Log(Protocol):
    """The part of speech_log.SpeechLog the list reads."""

    def since(self, after_seq: int) -> tuple[list[dict[str, Any]], int]: ...
    def newest_purge(self, after: str | None = None) -> str | None: ...


class VoiceList:
    """A `Tally` kept current with the speech log. Safe to share between threads."""

    def __init__(self, log: Callable[[], Log]) -> None:
        self._log = log
        self._lock = threading.Lock()
        self._tally = Tally()
        self._seq = 0
        self._purge: str | None = None
        self._read: Log | None = None  # the log last read, so a different one (a reconnect, a test's) is read afresh

    def voices(self, known: Iterable[int] = ()) -> list[dict[str, Any]]:
        with self._lock:
            log = self._log()
            if log is not self._read:
                self._tally, self._seq, self._purge = Tally(), 0, None
            purge = log.newest_purge(self._purge)
            if self._read is not None and purge != self._purge:
                self._tally, self._seq = Tally(), 0  # a purge may have taken counted lines: count again
            self._purge = purge
            lines, self._seq = log.since(self._seq)
            for line in lines:
                self._tally.add(line)
            self._read = log
            return self._tally.rows(known)
