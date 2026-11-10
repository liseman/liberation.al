"""Journal evidence that degrades in steps instead of breaking (#2808, goal #2805).

A day page (memory/journal_pages.py) cites hour entries (`[h3](#t=...)`, `h3` being the page's third parent), and
each hour entry lists the records behind it (`evidence.sources`, memory/journal.py). A source with a camera is a
moment of video; one without (speech, the assistant's replies) is audio. What a citation can show ages in steps:

1. While the camera's video is kept (the 30-day window, video_window.py), the moment plays from the recording.
2. Once the video is gone, a promoted clip plays instead: CLIP_S seconds around the moment, re-encoded small
   (CLIP_WIDTH px, CLIP_FPS frames a second, H.264), with a still and the caption as the fallback when no clip
   could be made. Only the moments a day page cites are promoted, at most PER_HOUR per cited hour (detected spans
   before plain captions), which keeps it to a few MB a day (docs/STORAGE.md).
3. A speech source plays its audio for as long as any microphone kept audio there. Since retention (#2833) ages out
   audio no speech touches past the window, a cited source that isn't an utterance (an assistant reply, say) can
   lose its audio then, and says "audio expired".
4. Once the source is gone with nothing promoted, the citation says so ("footage expired", from the broker's record
   of what it aged out, or "no footage was kept" when there was never any), with the caption. It never vanishes.

Promotion runs from `pack-video-window` (video_window.py), before it ages anything out, in the same process: every
moment cited by a day page in the window is promoted the night it first appears, about 28 days before the purge
reaches it, and a page written after its video has already gone (a late rebuild, the disk floor's early age-out)
is promoted from whatever is left: fewer frames make a shorter clip, none leave the still or the caption. Each clip
is one read and one decode pass over its seconds of video (vision_log.clip_frames), and its still is taken from the
frames already decoded. A run promotes at most MAX_PER_RUN moments within MAX_RUN_S, oldest day first (the one the
purge reaches next), and leaves the rest for the next night, so a backlog (the first night, a long outage) can't hold
up the purge for long.

Files live under `paths.state/journal_evidence/<YYYY-MM-DD>/`: `<camera>-<ms>.mp4` and `.jpg`, and `index.json`,
the outcome for each moment (clip, still, or nothing left), so a moment is never worked on twice. They are durable,
like the journal, and kept for good.
"""

import dataclasses
import datetime as dt
import json
import pathlib
import re
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from liberational import logs, memory_log, segment_log, vision_log
from liberational.memory import history_qa, journal, journal_pages

log = logs.get("journal-evidence")

ROOT = "journal_evidence"
INDEX = "index.json"
CLIP_S = 6.0  # seconds around a moment
CLIP_FPS = 8
CLIP_WIDTH = 480
STILL_WIDTH = 640
CRF = 32  # x264 quality: small files for a reminder of the moment, not a recording
MIN_FRAMES = 4  # fewer frames than this make no clip, only a still
PER_HOUR = 3  # camera moments promoted per cited hour
SAME_MOMENT_S = 10.0  # one camera's sources this close are one moment
CITED = re.compile(r"\[(h\d+)\]\(#t=")
MAX_PER_RUN = 300  # moments one night promotes; the rest wait for the next night
MAX_RUN_S = 600.0  # seconds one night spends promoting before it lets the purge go on

#: Whether any microphone kept audio over [start, end) (seconds).
AudioKept = Callable[[float, float], bool]


@dataclasses.dataclass(frozen=True)
class Moment:
    day: str  # YYYY-MM-DD, the day page that cites it
    label: str  # the citation, "h3"
    camera: str
    ts: float  # the moment's centre, seconds
    caption: str
    stream: str

    @property
    def key(self) -> str:
        return f"{_safe(self.camera)}-{round(self.ts * 1000)}"


def _safe(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]", "_", name)[:64] or "camera"


def moment_time(source: Mapping[str, Any]) -> float:
    """Where in a source to centre its clip: its start for a point, else its middle, at most CLIP_S / 2 in."""
    a = float(source.get("ts_start") or 0.0)
    b = float(source.get("ts_end") or a)
    return a if b <= a else min((a + b) / 2, a + CLIP_S / 2)


def folder(state: pathlib.Path, day: str) -> pathlib.Path:
    return pathlib.Path(state) / ROOT / day


def index(state: pathlib.Path, day: str) -> dict[str, dict[str, Any]]:
    """What has been promoted for a day page: each moment's key to its outcome."""
    path = folder(state, day) / INDEX
    try:
        found = json.loads(path.read_text())
    except FileNotFoundError, ValueError:
        return {}
    return found if isinstance(found, dict) else {}


def _save_index(state: pathlib.Path, day: str, found: Mapping[str, Any]) -> None:
    where = folder(state, day)
    where.mkdir(parents=True, exist_ok=True)
    tmp = where / (INDEX + ".tmp")
    tmp.write_text(json.dumps(found, indent=1, sort_keys=True))
    tmp.replace(where / INDEX)


def cited_sources(state: pathlib.Path, day: str) -> list[tuple[str, dict[str, Any]]]:
    """(label, source) for every source behind the hours a day page cites, in citation order."""
    page = memory_log.at(state).page(journal_pages.DAY, day)
    if not page:
        return []
    parents = list(page.get("parents") or [])
    labels = list(dict.fromkeys(CITED.findall(str(page.get("md") or ""))))
    a, b = journal.day_bounds(day)
    entries = {str(r.get("id")): r for r in memory_log.at(state).range(memory_log.JOURNAL, a, b)}
    out = []
    for label in labels:
        n = int(label[1:])
        if not 1 <= n <= len(parents):
            continue
        entry = entries.get(str(parents[n - 1])) or {}
        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        out += [(label, dict(s)) for s in (evidence or {}).get("sources") or [] if isinstance(s, Mapping)]
    return out


def cited_moments(state: pathlib.Path, day: str) -> list[Moment]:
    """The camera moments a day page cites, at most PER_HOUR per cited hour: detected spans (actions, zone visits,
    walk-bys, face matches) before plain captions, then by time; a camera's sources within SAME_MOMENT_S are one."""
    by_label: dict[str, list[Moment]] = {}
    for label, s in cited_sources(state, day):
        if not s.get("camera"):
            continue
        by_label.setdefault(label, []).append(
            Moment(day, label, str(s["camera"]), moment_time(s), str(s.get("text") or ""), str(s.get("stream") or ""))
        )
    out: list[Moment] = []
    for found in by_label.values():
        picked: list[Moment] = []
        for m in sorted(found, key=lambda m: (m.stream == segment_log.CAPTIONS, m.ts)):
            if len(picked) == PER_HOUR:
                break
            if not any(p.camera == m.camera and abs(p.ts - m.ts) < SAME_MOMENT_S for p in picked):
                picked.append(m)
        out += sorted(picked, key=lambda m: m.ts)
    return out


def _scaled(image: Any, width: int) -> Any:
    import cv2

    h, w = image.shape[:2]
    if w <= width:
        return image
    return cv2.resize(image, (width, max(2, round(h * width / w / 2) * 2)), interpolation=cv2.INTER_AREA)


def _frames(vision: Any, m: Moment) -> list[Any]:
    """The camera's frames over the clip, CLIP_FPS a second, each the one nearest its tick (None where gone), read
    and decoded in one pass."""
    start = m.ts - CLIP_S / 2
    return list(vision.clip(m.camera, start, start + CLIP_S, CLIP_FPS))


def encode_clip(frames: Sequence[Any], dst: pathlib.Path, fps: int = CLIP_FPS) -> bool:
    """Write frames (BGR images, all one size) as a small H.264 MP4 with ffmpeg; False without ffmpeg or on failure."""
    from liberational.ingest import media

    try:
        ffmpeg = media.tool("ffmpeg")
    except media.MediaError:
        return False
    h, w = frames[0].shape[:2]
    tmp = dst.with_suffix(".tmp.mp4")
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(tmp),
    ]  # fmt: skip
    try:
        done = subprocess.run(command, input=b"".join(f.tobytes() for f in frames), capture_output=True, timeout=60)
    except subprocess.TimeoutExpired, OSError:
        tmp.unlink(missing_ok=True)
        return False
    if done.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dst)
    return True


Encoder = Callable[[Sequence[Any], pathlib.Path], bool]


def promote(
    state: pathlib.Path,
    moments: Iterable[Moment],
    *,
    vision: Any = None,
    encode: Encoder = encode_clip,
    limit: int | None = None,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, int]:
    """Promote each moment not promoted yet, in the order given: a clip when MIN_FRAMES or more of its frames are
    left, a still from the decoded frame nearest its centre, and the caption, recorded in its day's index. At most
    `limit` moments, and none once `clock()` passes `deadline`: the rest are counted as `deferred` and left for the
    next run. Counts by outcome."""
    import cv2

    log_ = vision if vision is not None else vision_log.at(state)
    counts = {"clip": 0, "still": 0, "gone": 0, "bytes": 0, "deferred": 0}
    done = 0
    by_day: dict[str, list[Moment]] = {}
    for m in moments:
        by_day.setdefault(m.day, []).append(m)
    for day, todo in by_day.items():
        found = index(state, day)
        where = folder(state, day)
        for m in todo:
            if m.key in found:
                continue
            if (limit is not None and done >= limit) or (deadline is not None and clock() >= deadline):
                counts["deferred"] += 1
                continue
            done += 1
            # Ticks with no picture (a dropped frame, video gone part way) are left out, so a clip with gaps plays a
            # little short and fast: it is a reminder of the moment, not a recording of it.
            frames = [f for f in _frames(log_, m) if f is not None]
            ticks = round(CLIP_S * CLIP_FPS)
            if 0 < len(frames) < ticks // 2:
                # Video was there but most ticks got no picture: the moment's video is going (the purge got there
                # first), or the camera's stream decodes in a way clip_frames can't follow. Name it, to look into.
                log.warning("journal evidence: only %d of %d frames for %s at %.0f", len(frames), ticks, m.camera, m.ts)
            outcome: dict[str, Any] = dict(camera=m.camera, ts=m.ts, caption=m.caption, label=m.label)
            if frames:
                where.mkdir(parents=True, exist_ok=True)
                small = [_scaled(f, CLIP_WIDTH) for f in frames]
                if len(small) >= MIN_FRAMES and len({f.shape for f in small}) == 1:
                    clip = where / f"{m.key}.mp4"
                    if encode(small, clip):
                        outcome["clip"] = clip.name
                        counts["bytes"] += clip.stat().st_size
                middle = frames[len(frames) // 2]
                ok, jpeg = cv2.imencode(".jpg", _scaled(middle, STILL_WIDTH), [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    still = where / f"{m.key}.jpg"
                    still.write_bytes(jpeg.tobytes())
                    outcome["still"] = still.name
                    counts["bytes"] += len(jpeg)
            kind = "clip" if "clip" in outcome else "still" if "still" in outcome else "gone"
            counts[kind] += 1
            found[m.key] = outcome
            _save_index(state, day, found)  # after each moment, so an interrupted run keeps what it made
    return counts


def promote_window(
    state: pathlib.Path,
    now: float,
    days: float,
    *,
    vision: Any = None,
    limit: int | None = MAX_PER_RUN,
    budget_s: float | None = MAX_RUN_S,
) -> dict[str, int]:
    """Promote what every day page in the video window cites, oldest day first (`pack-video-window` calls this
    before aging anything out), at most `limit` moments in `budget_s` seconds. Each day costs a page read and a read
    of its hour entries every night, done or not; only the moments not in its index are decoded."""
    today = dt.datetime.fromtimestamp(now, history_qa.TZ).date()
    moments: list[Moment] = []
    for back in range(int(days) + 2, 0, -1):
        day = (today - dt.timedelta(days=back)).isoformat()
        moments += cited_moments(state, day)
    deadline = None if budget_s is None else time.monotonic() + budget_s
    return promote(state, moments, vision=vision, limit=limit, deadline=deadline)


# --------------------------------------------------------------- resolving ----
def _audio_spans(start: float, end: float) -> list[tuple[float, float]]:
    """Every microphone's kept audio over [start, end), seconds: every `*.audio` stream on the broker, so a
    microphone since renamed or unplugged still counts."""
    from liberational import speech_log

    shared = speech_log.shared()
    lo, hi = int(start * 1e9), int(end * 1e9)
    return [(a / 1e9, b / 1e9) for stream in shared.audio_streams() for a, b in shared.audio_spans(stream, lo, hi)]


def _audio_kept(start: float, end: float) -> bool:
    return bool(_audio_spans(start, end))


class _DayView:
    """One day's video, aged-out, and audio spans, each read from the broker once, answering resolve's questions
    for every source of the day (resolve_day) instead of a broker call per source."""

    def __init__(self, vision: Any, cameras: Iterable[str], start: float, end: float, audio: bool) -> None:
        self.video = {c: vision.video_spans(c, start, end) for c in cameras}
        self.aged = {c: vision.aged_out(c, start, end) for c in cameras}
        self.audio = _audio_spans(start, end) if audio else []

    @staticmethod
    def _any(spans: Iterable[Sequence[float]], a: float, b: float) -> list[list[float]]:
        return [[max(x, a), min(y, b)] for x, y in spans if y > a and x < b]

    def video_spans(self, cam: str, a: float, b: float) -> list[list[float]]:
        return self._any(self.video.get(cam, []), a, b)

    def aged_out(self, cam: str, a: float, b: float) -> list[list[float]]:
        return self._any(self.aged.get(cam, []), a, b)

    def audio_kept(self, a: float, b: float) -> bool:
        return bool(self._any(self.audio, a, b))


def _promoted(found: Mapping[str, Mapping[str, Any]], camera: str, t: float) -> Mapping[str, Any]:
    """The promoted moment of `camera` within SAME_MOMENT_S of `t` with the most kept (a clip, then a still, then
    nothing left), the nearest of those: cited_moments folds a camera's sources this close into one promoted moment,
    so each of them finds it, and a nearer moment whose video was already gone never hides a clip."""
    near = [
        (not v.get("clip"), not v.get("still"), abs(float(v.get("ts") or 0.0) - t), v)
        for v in found.values()
        if isinstance(v, Mapping) and v.get("camera") == camera and abs(float(v.get("ts") or 0.0) - t) < SAME_MOMENT_S
    ]
    return min(near, key=lambda row: row[:3])[3] if near else {}


def resolve(
    state: pathlib.Path,
    day: str,
    source: Mapping[str, Any],
    *,
    vision: Any = None,
    audio_kept: AudioKept | None = None,
    found: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """What a cited source can show now. `stage` is "video" (play the recording at `t` on `camera`), "clip" or
    "still" (a promoted file under /journal/evidence/<day>/<file>), "audio" (play the room's audio at `t`), or
    "expired" (`note` says why). `caption` is always the source's text."""
    caption = str(source.get("text") or "")
    camera = source.get("camera")
    a = float(source.get("ts_start") or 0.0)
    b = max(a, float(source.get("ts_end") or a))
    if not camera:
        kept = (audio_kept or _audio_kept)(a, b + 0.5)
        if kept:
            return dict(stage="audio", t=a, caption=caption)
        return dict(stage="expired", note="audio expired", caption=caption)
    log_ = vision if vision is not None else vision_log.at(state)
    t = moment_time(source)
    if log_.video_spans(str(camera), t - 0.5, t + 0.5):
        return dict(stage="video", camera=camera, t=t, caption=caption)
    promoted = _promoted(index(state, day) if found is None else found, str(camera), t)
    for stage in ("clip", "still"):
        if promoted.get(stage):
            out = dict(
                stage=stage, camera=camera, t=t, caption=caption, url=f"/journal/evidence/{day}/{promoted[stage]}"
            )
            if stage == "clip" and promoted.get("still"):
                out["still"] = f"/journal/evidence/{day}/{promoted['still']}"
            return out
    aged = log_.aged_out(str(camera), t - 0.5, t + 0.5)
    return dict(
        stage="expired", camera=camera, t=t, caption=caption, note="footage expired" if aged else "no footage was kept"
    )


def resolve_sources(
    state: pathlib.Path,
    day: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    vision: Any = None,
    audio_kept: AudioKept | None = None,
) -> list[dict[str, Any]]:
    """`resolve` for every source of `sources`, all from `day`, in order: the broker is read once per camera (video
    and aged-out spans) and once for audio over the sources' whole span, and the index once, however many there are.
    A day page's citations (resolve_day) and the Journal tab's entries (dashboard/journal.py) both resolve this way."""
    if not sources:
        return []
    a = min(float(s.get("ts_start") or 0.0) for s in sources) - CLIP_S
    b = max(float(s.get("ts_end") or s.get("ts_start") or 0.0) for s in sources) + CLIP_S
    cameras = sorted({str(s["camera"]) for s in sources if s.get("camera")})
    speech = any(not s.get("camera") for s in sources)
    view = _DayView(
        vision if vision is not None else vision_log.at(state), cameras, a, b, speech and audio_kept is None
    )
    found = index(state, day)
    return [resolve(state, day, s, vision=view, audio_kept=audio_kept or view.audio_kept, found=found) for s in sources]


def resolve_day(
    state: pathlib.Path, day: str, *, vision: Any = None, audio_kept: AudioKept | None = None
) -> list[dict[str, Any]]:
    """Every source behind a day page's citations, resolved: [{label, stream, ts_start, ts_end, and resolve's
    fields}], in citation order, read in one batch (resolve_sources)."""
    cited = cited_sources(state, day)
    resolved = resolve_sources(state, day, [s for _, s in cited], vision=vision, audio_kept=audio_kept)
    return [
        dict(label=label, stream=s.get("stream"), ts_start=s.get("ts_start"), ts_end=s.get("ts_end"), **r)
        for (label, s), r in zip(cited, resolved, strict=True)
    ]


def file(state: pathlib.Path, day: str, name: str) -> pathlib.Path | None:
    """A promoted file by its day and name, or None for anything else (no path leaves the day's folder)."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) or not re.fullmatch(r"[0-9A-Za-z_.-]{1,96}\.(?:mp4|jpg)", name):
        return None
    path = folder(state, day) / name
    return path if path.is_file() else None
