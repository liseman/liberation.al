"""The transcript as the dashboard reads it: the lines heard, a day's lines, a line's audio, and the room's kept
audio (moved out of the server, #61).

GET /transcript       the newest lines, or the lines around a moment
GET /transcript/day   one day's lines
GET /clip             one line's audio
GET /audio/range      the room's kept audio between two times
GET /audio/spans      each microphone's kept audio for a day
GET /review/days      the days with anything recorded
"""

import datetime as dt
import io
import math
import pathlib
import re
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import speech_log, vision_log
from liberational.audio import loudness
from liberational.dashboard import context, routes
from liberational.dashboard.context import LOG
from liberational.people import registry, transcript_view

try:
    import soundfile as sf
except ImportError:
    sf = None

if TYPE_CHECKING:
    from liberational.dashboard.server import H

PAD_S, MAX_CLIP_S = 0.4, 60.0


# What was heard and what the assistant said come from the broker's speech log (speech_log.shared()): utterances
# on speech.utterance, replies on assistant.reply, each a transcript line (a dict) with its payload `id`. A line names
# its voice (speaker_id, "Speaker n"), never a person; who said it is read through the event log (_view).
TAIL_S = 7 * 86400  # how far back a tail looks first; the whole log only when that holds too few lines


END_OF_TIME = float(2**32)  # 2106: "to the end" for a read of every line, whatever the clock says


def heard_lines(start: float = 0.0, end: float | None = None) -> list[dict[str, Any]]:
    """The utterances overlapping [start, end) (seconds; to the end by default), oldest first."""
    from liberational.listen import revisions

    return revisions.filter_lines(
        context.STATE, speech_log.shared().range(start, end if end is not None else END_OF_TIME)
    )


def said_lines(start: float = 0.0, end: float | None = None) -> list[dict[str, Any]]:
    """The assistant's spoken replies overlapping [start, end), oldest first."""
    return speech_log.shared().range(start, end if end is not None else END_OF_TIME, [speech_log.REPLIES])


def heard_tail(n: int) -> list[dict[str, Any]]:
    """The newest n utterances, oldest first."""
    lines = heard_lines(time.time() - TAIL_S)
    return (lines if len(lines) >= n else heard_lines())[-n:]


def said_tail(n: int) -> list[dict[str, Any]]:
    """The newest n replies, oldest first."""
    lines = said_lines(time.time() - TAIL_S)
    return (lines if len(lines) >= n else said_lines())[-n:]


def _view() -> transcript_view.View:
    """Who said each line: the speech log's lines through the event log (people/transcript_view.py). Every route that
    shows or uses a line's speaker reads lines through one of these, made per request."""
    return transcript_view.View(context.STATE)


def merged_tail(n: int = 80) -> list[dict[str, Any]]:
    """The last n lines of what was heard and what the assistant said, in time order."""
    heard_once = transcript_view.once(_view().lines(heard_tail(n)))  # another mic's copy of a line shows once
    return sorted(heard_once + said_tail(n), key=lambda e: e.get("ts_start") or 0)[-n:]


REVIEW_VIDEO_DAYS_S = 600.0  # the days with video are asked of the broker again this often (old days age out)


REVIEW_TODAY_VIDEO_S = 60.0  # and whether today has video yet, at most this often until it has


@dataclass
class _Days:
    """What recorded_days keeps between requests: each day's speech kinds, read in full once per state, purge,
    and review revision and then followed line by line, and the days with video, asked again every
    REVIEW_VIDEO_DAYS_S (today's every REVIEW_TODAY_VIDEO_S until it has some)."""

    key: tuple[Any, ...] | None = None
    seq: int = 0  # the speech history's position the days were read to
    speech: dict[str, dict[str, bool]] = field(default_factory=dict)  # day: {"audio", "transcript"}
    video: set[str] = field(default_factory=set)
    video_key: tuple[str, str] | None = None  # (state, the local day) the video days were asked on
    video_at: float = -math.inf
    today_at: float = -math.inf  # when today's video was last asked about
    lock: threading.Lock = field(default_factory=threading.Lock)


_DAYS = _Days()


def _note_speech(days: dict[str, dict[str, bool]], e: dict[str, Any]) -> None:
    kinds = days.setdefault(time.strftime("%Y-%m-%d", time.localtime(e["ts_start"])), {})
    kinds["transcript"] = True
    if has_audio(e):  # a line keeps its audio as long as the line is kept
        kinds["audio"] = True


def recorded_days(state: pathlib.Path) -> list[dict[str, Any]]:
    """Every day with something Review can show, oldest first: [{"day", "video", "audio", "transcript"}], each kind
    true when the day has it: a camera's video on the broker (`<cam>.video`, any tier), and the speech log's lines by
    local day, with audio when a line has it.

    Reading every line and asking the broker for every day each time cost about 0.4 s of CPU a request. The
    speech part is read in full only when the state, a purge, or a review correction could have taken lines; otherwise
    only the lines committed since are added, so a line for any day shows at once. The days with video are asked again
    every REVIEW_VIDEO_DAYS_S or when the local day changes, and today's every REVIEW_TODAY_VIDEO_S until it has video.
    The lock is held across the broker's answers, so requests arriving together share one read."""
    from liberational.listen import revisions

    speech = speech_log.shared()
    key = (str(state), speech.newest_purge(), revisions.review_revision(state))
    today = time.strftime("%Y-%m-%d")
    with _DAYS.lock:
        if _DAYS.key != key:
            seq = speech.mark((speech_log.UTTERANCES,))[0]  # before the read: a line committed in between comes twice
            _DAYS.speech = {}
            for e in heard_lines():
                _note_speech(_DAYS.speech, e)
            _DAYS.key, _DAYS.seq = key, seq
        else:
            new, _DAYS.seq = speech.since(_DAYS.seq, (speech_log.UTTERANCES,))
            for e in new:
                _note_speech(_DAYS.speech, e)
        if _DAYS.video_key is None or _DAYS.video_key[0] != str(state):
            _DAYS.video, _DAYS.today_at = set(), -math.inf  # another state's days never show
        try:
            log_of = vision_log.at(state)
            if _DAYS.video_key != (str(state), today) or time.monotonic() - _DAYS.video_at >= REVIEW_VIDEO_DAYS_S:
                _DAYS.video = set(log_of.video_days())
                _DAYS.video_key, _DAYS.video_at = (str(state), today), time.monotonic()
            elif today not in _DAYS.video and time.monotonic() - _DAYS.today_at >= REVIEW_TODAY_VIDEO_S:
                _DAYS.today_at = time.monotonic()
                midnight = time.mktime(time.strptime(today, "%Y-%m-%d"))
                if log_of.has_video(midnight, time.time() + 1):
                    _DAYS.video.add(today)
        except vision_log.continuum_py.ContinuumError as e:  # no broker: the days with speech still show
            LOG.warning("review days: no video days from the broker: %s", repr(e))
        days = sorted(set(_DAYS.speech) | _DAYS.video)
        return [
            {
                "day": d,
                "video": d in _DAYS.video,
                "audio": _DAYS.speech.get(d, {}).get("audio", False),
                "transcript": _DAYS.speech.get(d, {}).get("transcript", False),
            }
            for d in days
        ]


def iter_back() -> Iterator[dict[str, Any]]:
    """The utterances, newest first."""
    yield from reversed(heard_lines())


# A line's key carries its millisecond, so a key always names one line exactly. A time alone names the newest line
# starting within LINE_MATCH_S of it; every route that takes a time uses this one tolerance.
LINE_MATCH_S = 0.002
WINDOW_EDGE_S = 1e-3  # so a line starting exactly at the edge is read too (a read of [start, end) leaves it out)


def line_window(ts: float, tol: float = LINE_MATCH_S) -> list[dict[str, Any]]:
    """The stored lines starting within `tol` of ts, oldest first: a read of the moments around it, not of the log."""
    near = heard_lines(ts - tol, ts + tol + WINDOW_EDGE_S)
    return [
        e
        for e in near
        if isinstance(e.get("ts_start"), int | float)
        and not isinstance(e["ts_start"], bool)
        and abs(float(e["ts_start"]) - ts) <= tol
    ]


def line_at(ts: object = None, key: object = None, tol: float = LINE_MATCH_S) -> dict[str, Any] | None:
    """The stored line a key or a ts_start names: by its exact key when one is given (its time is read from the key),
    else the newest line starting within `tol` of the time. Reads the moments around it, not the whole log.
    Raises ValueError or TypeError for a time that isn't one."""
    if isinstance(key, str) and key:
        try:
            _slug, when = registry.split_key(key)
        except ValueError:
            return None
        return next((e for e in reversed(line_window(when, tol)) if registry.key_of(e) == key), None)
    if ts is None:
        return None
    hits = line_window(float(ts), tol)  # ty: ignore[invalid-argument-type]  # a bad value is the caller's TypeError
    return hits[-1] if hits else None  # the newest, as the walk from the newest line gave


BATCH_READ_S = 3600.0  # a batch's keys are read an hour at a time: a store read opens the hour parts it spans


def lines_by_key(keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    """{key: the stored line} for those of `keys` that are there: the batch write routes read the hours their keys
    fall in instead of the whole history. A key that doesn't parse is simply absent."""
    wanted = set(keys)
    moments = [when for key in wanted if key and (when := _key_time(key)) is not None]
    out: dict[str, dict[str, Any]] = {}
    for hour in sorted({math.floor(when / BATCH_READ_S) for when in moments}):
        for e in heard_lines(hour * BATCH_READ_S, (hour + 1) * BATCH_READ_S):
            if isinstance(e.get("ts_start"), int | float) and (key := registry.key_of(e)) in wanted:
                out[key] = e
    return out


def _key_time(key: str) -> float | None:
    """The ts_start a line key names, or None when it isn't a line key."""
    try:
        return registry.split_key(key)[1]
    except ValueError:
        return None


def find_entry(ts_start: float, tol: float = LINE_MATCH_S) -> dict[str, Any] | None:
    """The stored line nearest after ts_start (the newest within `tol`), or None."""
    return line_at(ts=ts_start, tol=tol)


def find_line(line: str) -> dict[str, Any] | None:
    """The transcript line with this key (registry.key_of)."""
    return line_at(key=line)


# A line's audio is a stretch of its microphone's stream on the broker ("audio": {stream, start_ns, end_ns}), read
# back through the speech log (room_audio.py): the tier worker's flushed batches, silence where nothing was kept.
# Audio still in the microphone's ring (the last couple of seconds) isn't flushed yet, so a clip that new is empty.
RANGE_MAX_S = 330.0  # the longest stretch of room audio served at once (a video segment is five minutes)


def has_audio(e: dict[str, Any]) -> bool:
    """Whether a line says where its audio is: a microphone stream and a span on the broker."""
    a = e.get("audio")
    return isinstance(a, dict) and isinstance(a.get("stream"), str) and isinstance(a.get("start_ns"), int)


def _channel(stream: str) -> int:
    """The channel a microphone's lines are transcribed from ([audio.mic_channel], 0 by default)."""
    return int((cfg.get("audio.mic_channel") or {}).get(stream.removesuffix(".audio"), 0))


def wav(rate: int, samples: Any, loud: bool = True) -> bytes:
    """Float samples as a 16-bit mono WAV, brought up to a listening level unless `loud` is False (audio/loudness.py:
    some mics record speech near -55 dBFS, which plays as silence)."""
    import numpy

    pcm = (numpy.clip(numpy.asarray(samples, numpy.float32), -1.0, 1.0) * 32767).astype(numpy.int16)
    if loud and len(pcm):
        pcm = loudness.normalize(pcm, rate)
    buf = io.BytesIO()
    sf.write(buf, pcm, rate, format="WAV", subtype="PCM_16")  # ty: ignore[unresolved-attribute]  # soundfile is a dependency here
    return buf.getvalue()


def read_audio(stream: str, start: float, end: float) -> tuple[int, Any]:
    """(rate, float samples) of `stream` over [start, end) seconds, silence where nothing was kept."""
    return speech_log.shared().audio(stream, int(start * 1e9), int(end * 1e9), _channel(stream))


def line_clip(e: dict[str, Any], pad: float = PAD_S) -> tuple[bytes, dict[str, Any]]:
    """A line's audio as a WAV, padded `pad` seconds each side and at most MAX_CLIP_S long, and what it holds:
    {"sr", "dur_s", "frames" (0 when nothing was kept: none yet, or purged), "stream", "start_s"}."""
    import numpy

    a = e["audio"]
    start = max(0.0, a["start_ns"] / 1e9 - pad)
    end = min(start + MAX_CLIP_S + 2 * pad, int(a.get("end_ns") or a["start_ns"]) / 1e9 + pad)
    rate, x = read_audio(a["stream"], start, max(end, start + 0.01))
    kept = int(numpy.count_nonzero(x))
    return wav(rate, x), {"sr": rate, "dur_s": round(len(x) / rate, 3), "frames": len(x) if kept else 0,
                          "stream": a["stream"], "start_s": round(start, 3)}  # fmt: skip


def audio_range(start: Any, end: Any) -> tuple[bytes, dict[str, Any]]:
    """-> (wav_bytes, info) of the room between two timestamps, for Review to play under recorded video (the
    recordings carry no sound). Reads the microphone the newest line with audio came from, else each microphone in
    turn until one kept audio there; silence where none did."""
    import numpy

    start, end = float(start), float(end)
    if not end > start:
        raise ValueError("end must be after start")
    end = min(end, start + RANGE_MAX_S)
    mics = sorted(speech_log.shared().mic_freshness())
    newest = _newest_audio_stream()
    if newest in mics:
        mics.remove(newest)
        mics.insert(0, newest)
    rate, x = 16_000, numpy.zeros(int((end - start) * 16_000), numpy.float32)
    for stream in mics:
        r, y = read_audio(stream, start, end)
        if numpy.count_nonzero(y):
            rate, x = r, y
            return wav(rate, x), {"stream": stream, "sr": rate, "dur_s": round(len(x) / rate, 3)}
    if not mics:
        raise FileNotFoundError("no microphone has kept any audio")
    return wav(rate, x, loud=False), {"stream": None, "sr": rate, "dur_s": round(len(x) / rate, 3)}


def audio_spans(day: str) -> list[dict[str, Any]]:
    """Each microphone's kept audio over a local day: [{"device", "start", "end"}] (seconds), for Review's timeline."""
    t0 = time.mktime(time.strptime(day, "%Y-%m-%d"))
    end = dt.datetime.strptime(day, "%Y-%m-%d") + dt.timedelta(days=1)
    t1 = time.mktime(end.timetuple())
    out = []
    for stream in sorted(speech_log.shared().mic_freshness()):
        for a, b in speech_log.shared().audio_spans(stream, int(t0 * 1e9), int(t1 * 1e9)):
            out.append({"device": stream.removesuffix(".audio"), "start": a / 1e9, "end": b / 1e9})
    return out


def transcript_around(ts: float, n: int = 12) -> dict[str, Any]:
    """Utterances closest to ts (by ts_start), n in total, in time order, from the day around it."""
    es = heard_lines(ts - 86400, ts + 86400)
    if not es:
        return {"entries": [], "target": None}
    i = min(range(len(es)), key=lambda k: abs(es[k].get("ts_start", 0) - ts))
    lo, hi = max(0, i - n // 2), min(len(es), i + n // 2 + 1)
    return {"entries": _view().lines(es[lo:hi]), "target": es[i].get("ts_start")}


@routes.get("/transcript")
def _get_transcript(h: H) -> None:
    q = h._query()
    try:
        n = max(1, min(1000, int(q.get("n", "80"))))
    except ValueError:
        n = 80
    if q.get("around"):  # the moment a summary link points at: n/2 utterances either side of ts
        try:
            ts = float(q["around"])
        except ValueError:
            return h._json({"error": "bad around"}, 400)
        return h._json(transcript_around(ts, n))
    return h._json({"entries": merged_tail(n), "reset": True})


@routes.get("/clip")
def _get_clip(h: H) -> None:
    q = h._query()
    info = q.get("info") in ("1", "true")
    if q.get("line"):  # the line's key: one microphone's line, never another's in the same millisecond
        e = find_line(q["line"])
        if e is None:
            return h._json({"ok": False, "error": "no transcript line with that key"}, 404)
        return h.clip(entry=e, info=info)
    if "ts_start" in q:
        try:
            ts = float(q["ts_start"])
        except ValueError:
            return h._json({"ok": False, "error": "bad ts_start"}, 400)
        e = find_entry(ts)
        if e is None:
            return h._json({"ok": False, "error": "no transcript entry at that ts_start"}, 404)
        return h.clip(entry=e, info=info)
    return h._json({"ok": False, "error": "need line=<key> or ts_start=<unix seconds>"}, 400)


@routes.get("/review/days")
def _get_review_days(h: H) -> None:
    return h._json({"days": recorded_days(context.STATE), "today": time.strftime("%Y-%m-%d")})


@routes.get("/transcript/day")
def _get_transcript_day(h: H) -> None:
    day = h._query().get("day") or time.strftime("%Y-%m-%d")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        return h._json({"error": "bad day"}, 400)
    t0 = time.mktime(time.strptime(day, "%Y-%m-%d"))
    es = [
        e for e in heard_lines(t0 - 3600, t0 + 90000) if time.strftime("%Y-%m-%d", time.localtime(e["ts_start"])) == day
    ]
    return h._json({"day": day, "entries": transcript_view.once(_view().lines(es))})  # once each


def _newest_audio_stream() -> str | None:
    """The microphone the newest line with audio came from: read from the last two minutes, then the last hour, and
    only then from every line. The common case answers from a small read."""
    now = time.time()
    for start in (now - 120.0, now - 3600.0, 0.0):
        found = next(
            (e["audio"]["stream"] for e in reversed(heard_lines(start)) if has_audio(e)),
            None,
        )
        if found is not None:
            return found
    return None


@routes.get("/audio/range")
def _get_audio_range(h: H) -> None:
    q = h._query()
    try:
        wav, _info = audio_range(q.get("start"), q.get("end"))
    except TypeError, ValueError:
        return h._json({"error": "start and end must be timestamps, end after start"}, 400)
    except FileNotFoundError as ex:
        return h._json({"error": str(ex)}, 404)
    except Exception as ex:
        return h._json({"error": str(ex)}, 500)
    # Browsers fetch media with Range requests and give up on a plain 200 (Review's room audio never loaded,
    # with networkState stuck at 2); _media answers 206 like /clip does.
    return h._media(wav, "audio/wav", {"Cache-Control": "no-store"})


@routes.get("/audio/spans")
def _get_audio_spans(h: H) -> None:
    day = h._query().get("day") or time.strftime("%Y-%m-%d")
    spans = audio_spans(day) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) else []
    return h._json({"day": day, "spans": spans})
