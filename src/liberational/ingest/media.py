"""What an ingested file is, when it was made, and ffmpeg conversions to what Continuum's recording streams take.

Continuum registers recordings in place on two kinds of stream (continuum-tier's inplace_audio and inplace_video):
FLAC files on `<name>.audio_v1`, all of one layout per stream,
and H.264 MP4 files (with their `moov`) on `<name>.video_v1`. So:

- audio, and a video's sound, become 16 kHz mono 16-bit FLAC (`to_flac`), one layout for every ingest stream;
- video becomes H.264 MP4 without sound (`to_h264`), copied without re-encoding when it already is H.264;
- a photo becomes a short H.264 clip of the still (`still_to_h264`, `STILL_S` seconds at `STILL_FPS`), turned
  upright by its EXIF orientation first, so it plays and is read like any recording.

An item's original time is its EXIF DateTimeOriginal for a photo (local wall-clock time unless EXIF gives an
offset), else the container's `creation_time` from ffprobe; with neither the caller falls back to the source's
date or the upload time.
"""

import datetime as dt
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image, ImageOps

AUDIO_RATE = 16000
STILL_S = 2.0
STILL_FPS = 2
# Long enough for a several-hour recording on a slow machine; a hung ffmpeg doesn't hold the worker forever.
FFMPEG_TIMEOUT_S = 6 * 3600


class MediaError(Exception):
    """A file that can't be read as a photo, video, or audio, or a conversion that failed."""


@dataclass(frozen=True)
class Probe:
    """What ffprobe (or Pillow, for a still) says about a file."""

    kind: str  # "photo", "video", or "audio"
    has_audio: bool = False
    video_codec: str | None = None
    format_name: str = ""
    duration_s: float | None = None
    fps: float | None = None
    creation_ts: float | None = None


def tool(name: str) -> str:
    """The path of ffmpeg or ffprobe on PATH, or a MediaError saying how to get it."""
    found = shutil.which(name)
    if not found:
        raise MediaError(f"{name} isn't installed (mise install puts ffmpeg on PATH)")
    return found


def parse_time(raw: object) -> float | None:
    """An ISO 8601 creation time (ffprobe's `creation_time`, `2026-10-31T16:12:33.000000Z`) as Unix seconds, or None.
    A time without a zone is UTC, as QuickTime and ffmpeg write it. The epoch, which some cameras write when they
    don't know, is no time."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        t = dt.datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.UTC)
    ts = t.timestamp()
    return ts if ts > 86400 else None


def exif_time(raw: object, offset: object = None) -> float | None:
    """An EXIF date (`2026:10:31 16:12:33`) as Unix seconds: at `offset` (`-07:00`) when EXIF gives one, else in
    this machine's local time, as the camera's clock showed it. None for a blank or malformed date, and for one no
    camera could have taken a photo at (year 1, or the epoch a camera with a reset clock writes), as parse_time."""
    if not isinstance(raw, str):
        return None
    try:
        t = dt.datetime.strptime(raw.strip().rstrip("\x00"), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    if isinstance(offset, str) and offset.strip():
        try:
            t = t.replace(tzinfo=dt.datetime.strptime(offset.strip().rstrip("\x00"), "%z").tzinfo)
        except ValueError:
            pass
    try:
        ts = t.timestamp()
    except ValueError, OverflowError, OSError:  # a local time before year 1 or past 9999
        return None
    return ts if ts > 86400 else None


def photo(path: Path) -> tuple[bool, float | None]:
    """(whether Pillow reads `path` as a still image, its EXIF original time or None)."""
    try:
        with Image.open(path) as im:
            if getattr(im, "n_frames", 1) > 1 and im.format not in ("MPO",):  # an animated GIF or WebP is a video
                return False, None
            exif = im.getexif()
    except OSError, ValueError, Image.DecompressionBombError:
        return False, None
    sub = exif.get_ifd(ExifTags.IFD.Exif)
    taken = exif_time(sub.get(ExifTags.Base.DateTimeOriginal), sub.get(ExifTags.Base.OffsetTimeOriginal))
    if taken is None:
        taken = exif_time(exif.get(ExifTags.Base.DateTime), sub.get(ExifTags.Base.OffsetTime))
    return True, taken


def _rate(raw: object) -> float | None:
    if not isinstance(raw, str) or "/" not in raw:
        return None
    num, _, den = raw.partition("/")
    try:
        r = float(num) / float(den)
    except ValueError, ZeroDivisionError:
        return None
    return r if 0 < r < 1000 else None


def from_ffprobe(info: dict[str, Any]) -> Probe:
    """A Probe from ffprobe's `-show_format -show_streams` JSON. A single-frame image format is a photo; a file with
    no video and no audio is a MediaError."""
    streams = info.get("streams") or []
    fmt = info.get("format") or {}
    video = [
        s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
    ]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    name = str(fmt.get("format_name") or "")
    created = parse_time((fmt.get("tags") or {}).get("creation_time"))
    for s in video + audio:
        created = created or parse_time((s.get("tags") or {}).get("creation_time"))
    try:
        duration = float(fmt["duration"]) if fmt.get("duration") not in (None, "N/A") else None
    except ValueError:
        duration = None
    if video:
        v = video[0]
        still = name.startswith("image2") or name.endswith("_pipe")
        return Probe(
            kind="photo" if still else "video",
            has_audio=bool(audio),
            video_codec=v.get("codec_name"),
            format_name=name,
            duration_s=duration,
            fps=_rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate")),
            creation_ts=created,
        )
    if audio:
        return Probe(kind="audio", has_audio=True, format_name=name, duration_s=duration, creation_ts=created)
    raise MediaError("no picture or sound in it")


def probe(path: Path) -> Probe:
    """What `path` is: a still Pillow reads is a photo (with its EXIF time); anything else is what ffprobe says."""
    still, taken = photo(path)
    if still:
        return Probe(kind="photo", creation_ts=taken)
    try:
        out = subprocess.run(
            [tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as e:
        raise MediaError("ffprobe took too long") from e
    if out.returncode != 0:
        raise MediaError(f"not a photo, video, or audio file ({_last(out.stderr)})")
    try:
        info = json.loads(out.stdout or "{}")
    except ValueError as e:
        raise MediaError("ffprobe's answer wasn't JSON") from e
    return from_ffprobe(info)


def _last(stderr: str) -> str:
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    return lines[-1][:300] if lines else "no reason given"


def _ffmpeg(args: list[str], dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.part{dst.suffix}")
    try:
        out = subprocess.run(
            [tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args, str(tmp)],
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        tmp.unlink(missing_ok=True)
        raise MediaError("ffmpeg took too long") from e
    if out.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise MediaError(f"ffmpeg couldn't convert it: {_last(out.stderr)}")
    tmp.replace(dst)
    return dst


def to_flac(src: Path, dst: Path) -> Path:
    """`src`'s sound as 16 kHz mono 16-bit FLAC: every ingest audio stream has this one layout."""
    return _ffmpeg(
        ["-i", str(src), "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-sample_fmt", "s16", "-c:a", "flac"], dst
    )


# Even dimensions (H.264 in 4:2:0 needs them) in the pixel format every decoder takes.
_EVEN = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p"]


def to_h264(src: Path, dst: Path, codec: str | None) -> Path:
    """`src`'s picture as H.264 MP4 without sound, its index at the front. Already H.264: copied, not re-encoded."""
    if codec == "h264":
        video = ["-map", "0:v:0", "-c:v", "copy"]
    else:
        video = ["-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", *_EVEN]
    return _ffmpeg(["-i", str(src), *video, "-an", "-movflags", "+faststart"], dst)


def still_to_h264(src: Path, dst: Path) -> Path:
    """A photo as a STILL_S second H.264 MP4 at STILL_FPS, upright by its EXIF orientation."""
    upright = dst.with_name(f".{dst.stem}.upright.png")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            ImageOps.exif_transpose(im).convert("RGB").save(upright)
    except Image.DecompressionBombError as e:
        raise MediaError(f"the photo is too large to decode ({e})") from e
    except (OSError, ValueError) as e:
        raise MediaError(f"couldn't read the photo: {e}") from e
    try:
        return _ffmpeg(
            [
                "-loop", "1", "-framerate", str(STILL_FPS), "-i", str(upright), "-t", str(STILL_S),
                "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-g", str(STILL_FPS), *_EVEN,
                "-movflags", "+faststart",
            ],
            dst,
        )  # fmt: skip
    finally:
        upright.unlink(missing_ok=True)
