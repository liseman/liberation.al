"""The speech log in memory, for tests: tests/conftest.py puts one in place of `speech_log.shared()` for every test,
and `fake()` hands it back typed as what it is."""

from typing import cast

from liberational import speech_log


class FakeSpeechLog:
    """The speech log in memory, in place of `speech_log.shared()` for every test: the same calls and the same line
    dicts (ts_start, ts_end, id, and the body's keys; replies get assistant=True), with no broker."""

    def __init__(self) -> None:
        self.lines: dict[str, list[dict]] = {}
        self.seq = 0  # the last commit sequence given out
        self.purges: list[str] = []
        self.sound: dict[str, list] = {}  # a microphone stream's kept audio: (start_ns, rate, samples)
        self.mics: dict[str, int | None] = {}  # a microphone stream and the end of its newest audio (UTC ns)

    def stamp(self, start: float, end: float) -> tuple:
        """As `SpeechLog.stamp`: moves when a line of the window is added or purged."""
        lines = tuple(
            (stream, tuple(line["seq"] for line in rows if start <= line["ts_start"] < end))
            for stream, rows in sorted(self.lines.items())
        )
        return lines, tuple(self.purges)

    def add(self, stream: str, line: dict) -> dict:
        """Puts a line on a stream as the broker would hand it back, and returns it."""
        import uuid

        from liberational import speech_log

        got = dict(line)
        got.setdefault("id", str(uuid.uuid7()))
        got["ts_start"] = float(got["ts_start"])
        got["ts_end"] = float(got.get("ts_end") or got["ts_start"])
        if stream == speech_log.REPLIES:
            got["assistant"] = True
        self.seq += 1
        got["seq"] = self.seq
        self.lines.setdefault(stream, []).append(got)
        return got

    def purge(self, stream: str, which) -> int:
        """Takes the lines `which(line)` picks off a stream, as a purge would; returns how many."""
        import uuid

        kept = [line for line in self.lines.get(stream, []) if not which(line)]
        gone = len(self.lines.get(stream, [])) - len(kept)
        self.lines[stream] = kept
        self.purges.append(str(uuid.uuid7()))
        return gone

    def say(self, line: dict) -> dict:
        """An utterance."""
        from liberational import speech_log

        return self.add(speech_log.UTTERANCES, line)

    def write(self, rows) -> None:
        """Replaces every utterance with `rows`, as a v1 test wrote transcript.jsonl."""
        from liberational import speech_log

        if self.lines.get(speech_log.UTTERANCES):
            self.purge(speech_log.UTTERANCES, lambda e: True)
        for row in rows:
            self.say(row)

    def reply(self, line: dict) -> dict:
        """An assistant reply."""
        from liberational import speech_log

        return self.add(speech_log.REPLIES, line)

    def range(self, start: float, end: float, streams=None) -> list[dict]:
        from liberational import speech_log

        streams = tuple(streams or (speech_log.UTTERANCES,))
        out = [
            dict(line)
            for stream in streams
            for line in self.lines.get(stream, [])
            # A line overlaps [start, end): it starts before the end and ends after the start (an instant covers 1 ns).
            if line["ts_start"] < end and max(line["ts_end"], line["ts_start"] + 1e-9) > start
        ]
        return sorted(out, key=lambda line: (line["ts_start"], line["id"]))

    def all_lines(self, streams=None) -> list[dict]:
        out = [line for line in self.range(float("-inf"), float("inf"), streams)]
        for line in out:
            line.pop("seq", None)
        return out

    def since(self, after_seq: int, streams=None) -> tuple[list[dict], int]:
        from liberational import speech_log

        streams = tuple(streams or (speech_log.UTTERANCES,))
        out = [dict(line) for stream in streams for line in self.lines.get(stream, []) if line["seq"] > after_seq]
        return sorted(out, key=lambda line: line["seq"]), max(after_seq, self.seq)

    def newest_purge(self, after: str | None = None) -> str | None:
        return self.purges[-1] if self.purges else after

    def newest_removal(self) -> str | None:
        return self.purges[-1] if self.purges else None  # the fake's purges are all deliberate

    def newest_voice_purge(self) -> str | None:
        return self.purges[-1] if self.purges else None  # the fake's purges are all of speech

    def mark(self, streams=None) -> tuple[int, str | None]:
        return self.seq, self.newest_purge()

    def recent(self, seconds: float, streams=None, *, now: float | None = None) -> list[dict]:
        import time

        end = now if now is not None else time.time()
        return self.range(end - seconds, end + 1.0, streams)

    def put_audio(self, stream: str, start_ns: int, samples, rate: int = 16_000) -> None:
        """Keeps a stretch of a microphone's audio, as the tier worker's chunks would."""
        import numpy as np

        self.sound.setdefault(stream, []).append((start_ns, rate, np.asarray(samples, np.float32)))

    def audio(self, stream: str, start_ns: int, end_ns: int, channel: int = 0):
        import numpy as np

        if not 0 < end_ns - start_ns <= 330 * 10**9:
            raise ValueError("a span of audio is 1 ns to 330 s")
        kept = self.sound.get(stream, [])
        rate = kept[0][1] if kept else 16_000
        out = np.zeros(int((end_ns - start_ns) * rate // 10**9), np.float32)
        for t0, r, samples in kept:
            at = (t0 - start_ns) * r // 10**9
            lo, hi = max(0, at), min(len(out), at + len(samples))
            if hi > lo:
                out[lo:hi] = samples[lo - at : hi - at]
        return rate, out

    def audio_spans(self, stream: str, start_ns: int, end_ns: int) -> list[tuple[int, int]]:
        out = []
        for t0, r, samples in sorted(self.sound.get(stream, []), key=lambda k: k[0]):
            t1 = t0 + len(samples) * 10**9 // r
            if t0 < end_ns and t1 > start_ns:
                out.append((t0, t1))
        return out

    def mic_freshness(self, window_s: float = 30.0) -> dict:
        """Each microphone stream and the end of its newest audio (tests set `self.mics`)."""
        return dict(self.mics)

    def append(self, stream: str, line: dict, **kwargs) -> str:
        return self.add(stream, line)["id"]


def fake() -> FakeSpeechLog:
    """This test's in-memory speech log (the `speech` fixture's)."""
    return cast(FakeSpeechLog, speech_log.shared())
