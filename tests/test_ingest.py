"""The ingest pipeline, its state store, media handling, and sources (ffmpeg for real; a fake registrar in place of
the broker, which test_ingest_broker.py covers)."""

import datetime as dt
import http.server
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import create_autospec

import pytest
from continuum_py import BrokerError
from PIL import Image

from liberational.ingest import media, runner
from liberational.ingest import sources as src
from liberational.ingest.broker import Refused
from liberational.ingest.pipeline import Incoming, Pipeline, safe_suffix, stream_name
from liberational.ingest.sources import google_photos, link, stubs, wyze
from liberational.ingest.store import ItemGone, Store

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


class FakeRegistrar:
    """Records each registration and answers with a chunk id; a stream in `refuse` is refused."""

    def __init__(self, refuse: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple] = []
        self.refuse = refuse

    def video(self, stream, path, start, fps):
        if f"{stream}.video_v1" in self.refuse:
            raise Refused("layout")
        self.calls.append(("video", stream, Path(path), start, fps))
        return f"chunk-{len(self.calls)}"

    def audio(self, stream, path, start):
        if f"{stream}.audio_v1" in self.refuse:
            raise Refused("layout")
        self.calls.append(("audio", stream, Path(path), start))
        return f"chunk-{len(self.calls)}"


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "state" / "ingest.sqlite3")


@pytest.fixture
def fake():
    return FakeRegistrar()


@pytest.fixture
def pipe(tmp_path, store, fake):
    return Pipeline(store, tmp_path / "ingest", fake, now=lambda: 1_700_000_000.0)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def make_photo(path: Path, taken: str | None = None, orientation: int | None = None) -> Path:
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    if taken:
        exif.get_ifd(0x8769)[36867] = taken
    Image.new("RGB", (64, 48), (200, 10, 10)).save(path, exif=exif)
    return path


def make_clip(path: Path, created: str | None = "2026-05-01T12:00:00Z", audio: bool = True) -> Path:
    args = ["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440"]
    args += ["-t", "2", "-c:v", "mpeg4"]  # not H.264, so the pipeline has to re-encode it
    if created:
        args += ["-metadata", f"creation_time={created}"]
    ffmpeg(*args, str(path))
    return path


# ---- media


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-05-01T12:00:00.000000Z", 1777636800.0),
        ("2026-05-01T12:00:00", 1777636800.0),  # no zone: UTC, as containers write it
        ("2026-05-01T14:00:00+02:00", 1777636800.0),
        pytest.param("1970-01-01T00:00:00Z", None, id="the-epoch-is-no-time"),
        ("not a date", None),
        ("", None),
        (None, None),
    ],
)
def test_container_creation_times(raw, expected):
    assert media.parse_time(raw) == expected


def test_an_exif_time_is_local_unless_exif_gives_an_offset():
    assert media.exif_time("2025:12:24 18:30:00") == dt.datetime(2025, 12, 24, 18, 30).timestamp()
    assert media.exif_time("2025:12:24 18:30:00", "-08:00") == 1766629800.0
    assert media.exif_time("0000:00:00 00:00:00") is None
    assert media.exif_time(None) is None


@pytest.mark.parametrize(
    ("raw", "offset"),
    [
        pytest.param("0001:01:01 00:00:00", None, id="year-one-local"),
        pytest.param("0001:01:01 00:00:00", "+05:00", id="year-one-with-offset"),
        pytest.param("1970:01:01 00:00:00", "+00:00", id="the-epoch"),
        pytest.param("1970:01:01 00:00:00", None, id="the-epoch-local"),
    ],
)
def test_an_exif_time_a_camera_couldnt_have_is_no_time(raw, offset):
    assert media.exif_time(raw, offset) is None


def test_ffprobe_kinds():
    still = {"format": {"format_name": "image2"}, "streams": [{"codec_type": "video", "codec_name": "mjpeg"}]}
    assert media.from_ffprobe(still).kind == "photo"
    song = {
        "format": {"format_name": "mp3", "duration": "12.5"},
        "streams": [
            {"codec_type": "audio", "codec_name": "mp3"},
            {"codec_type": "video", "codec_name": "png", "disposition": {"attached_pic": 1}},  # cover art
        ],
    }
    assert media.from_ffprobe(song) == media.Probe(kind="audio", has_audio=True, format_name="mp3", duration_s=12.5)
    video = {
        "format": {"format_name": "mov,mp4", "tags": {"creation_time": "2026-05-01T12:00:00Z"}},
        "streams": [{"codec_type": "video", "codec_name": "hevc", "avg_frame_rate": "30000/1001"}],
    }
    got = media.from_ffprobe(video)
    assert (got.kind, got.video_codec, got.has_audio, got.creation_ts) == ("video", "hevc", False, 1777636800.0)
    assert got.fps == pytest.approx(29.97, abs=0.01)
    with pytest.raises(media.MediaError, match="no picture or sound"):
        media.from_ffprobe({"format": {}, "streams": [{"codec_type": "data"}]})


@pytest.mark.parametrize(
    ("name", "suffix"), [("a.JPG", ".jpg"), ("clip.tar.mp4", ".mp4"), ("noext", ""), ("x.<script>", "")]
)
def test_only_plain_suffixes_are_kept(name, suffix):
    assert safe_suffix(name) == suffix


def test_stream_names():
    assert stream_name("Front Door!") == "front-door"
    assert stream_name("ingest-google_photos") == "ingest-google-photos"
    assert stream_name("***") == "unnamed"


# ---- the pipeline


@needs_ffmpeg
def test_a_photo_becomes_an_upright_clip_at_its_exif_time(tmp_path, pipe, fake):
    photo = make_photo(tmp_path / "IMG_1.jpg", "2025:12:24 18:30:00", orientation=6)
    item = pipe.ingest(Incoming(photo, "upload", "IMG_1.jpg"))
    assert (item.status, item.kind, item.time_from) == ("done", "photo", "exif")
    assert item.original_ts == dt.datetime(2025, 12, 24, 18, 30).timestamp()
    assert not photo.exists()  # moved into the originals, not copied
    assert Path(item.original).read_bytes()[:2] == b"\xff\xd8"
    [(kind, stream, clip, start, fps)] = fake.calls
    assert (kind, stream, start, fps) == ("video", "ingest-upload", item.original_ts, media.STILL_FPS)
    info = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(clip)],
            capture_output=True,
            check=True,
        ).stdout
    )["streams"][0]
    assert (info["codec_name"], info["width"], info["height"]) == ("h264", 48, 64)  # turned upright
    assert item.outputs == [
        {"kind": "video", "stream": "ingest-upload.video_v1", "chunk_id": "chunk-1", "path": str(clip)}
    ]


@needs_ffmpeg
def test_a_video_registers_its_picture_and_its_sound_at_its_creation_time(tmp_path, pipe, fake):
    item = pipe.ingest(Incoming(make_clip(tmp_path / "v.mov"), "link", "v.mov", keep=True))
    assert (item.status, item.kind, item.time_from, item.original_ts) == ("done", "video", "ffprobe", 1777636800.0)
    assert [(c[0], c[1], c[2].suffix, c[3]) for c in fake.calls] == [
        ("video", "ingest-link", ".mp4", 1777636800.0),
        ("audio", "ingest-link", ".flac", 1777636800.0),
    ]
    assert (tmp_path / "v.mov").exists()  # keep: the caller's file stays
    flac = media.probe(fake.calls[1][2])
    assert flac.kind == "audio"
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0",
            str(fake.calls[1][2]),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "16000,1"


@needs_ffmpeg
def test_without_a_time_in_the_file_the_sources_time_then_the_upload_time_is_used(tmp_path, pipe):
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=300", "-t", "1", str(tmp_path / "a.wav"))
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=500", "-t", "1", str(tmp_path / "b.wav"))
    sourced = pipe.ingest(Incoming(tmp_path / "a.wav", "wyze", "a.wav", taken_at=1_600_000_000.0))
    uploaded = pipe.ingest(Incoming(tmp_path / "b.wav", "upload", "b.wav"))
    assert (sourced.time_from, sourced.original_ts) == ("source", 1_600_000_000.0)
    assert (uploaded.time_from, uploaded.original_ts) == ("upload", 1_700_000_000.0)


@needs_ffmpeg
def test_the_same_bytes_twice_are_one_item(tmp_path, pipe, fake, store):
    make_photo(tmp_path / "a.jpg", "2025:12:24 18:30:00")
    shutil.copy(tmp_path / "a.jpg", tmp_path / "copy.jpg")
    first = pipe.ingest(Incoming(tmp_path / "a.jpg", "upload", "a.jpg"))
    again = pipe.ingest(Incoming(tmp_path / "copy.jpg", "google_photos", "copy.jpg", key="gp-1"))
    assert again.id == first.id and again.source == "upload"
    assert len(fake.calls) == 1
    assert not (tmp_path / "copy.jpg").exists()  # the duplicate is dropped
    assert store.seen("google_photos", "gp-1")  # and the source won't hand it over again
    assert [i.id for i in store.recent()] == [first.id]


@needs_ffmpeg
def test_a_refused_sound_leaves_the_picture_registered_and_says_so(tmp_path, store):
    fake = FakeRegistrar(refuse=("ingest-upload.audio_v1",))
    pipe = Pipeline(store, tmp_path / "ingest", fake)
    item = pipe.ingest(Incoming(make_clip(tmp_path / "v.mov"), "upload", "v.mov"))
    assert item.status == "done"
    assert item.detail == "not registered: audio: layout"
    assert [o["kind"] for o in item.outputs] == ["video"]


def test_a_file_that_isnt_media_fails_with_why(tmp_path, pipe, fake):
    (tmp_path / "notes.txt").write_text("hello")
    item = pipe.ingest(Incoming(tmp_path / "notes.txt", "upload", "notes.txt"))
    assert item.status == "failed"
    assert "not a photo, video, or audio file" in item.detail
    assert fake.calls == []


@needs_ffmpeg
def test_items_a_restart_cut_off_are_finished_on_resume(tmp_path, pipe, store):
    make_photo(tmp_path / "a.jpg")
    item, new = pipe.add(Incoming(tmp_path / "a.jpg", "upload", "a.jpg"))
    store.update(item.id, status="converting")  # the process died here
    assert new and [i.id for i in store.pending()] == [item.id]
    assert [i.status for i in pipe.resume()] == ["done"]
    assert store.pending() == []


class BrokenOnceRegistrar(FakeRegistrar):
    """The broker failing in a way the pipeline has no name for, on the first registration only."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error: Exception | None = error

    def video(self, stream, path, start, fps):
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        return super().video(stream, path, start, fps)


@needs_ffmpeg
@pytest.mark.parametrize(
    ("error", "why"),
    [
        pytest.param(BrokerError("Internal", "worker registration failed"), "worker registration failed", id="broker"),
        # A KeyError from a bug is not "the item is gone" (#3164): it fails the item rather than escaping with it
        # left converting, which every resume then skipped
        pytest.param(KeyError("fps"), "KeyError: 'fps'", id="stray-keyerror"),
    ],
)
def test_an_unexpected_error_fails_its_item_and_resume_goes_on_to_the_next(tmp_path, store, error, why):
    pipe = Pipeline(store, tmp_path / "ingest", BrokenOnceRegistrar(error), now=lambda: 1_700_000_000.0)
    first, _ = pipe.add(Incoming(make_photo(tmp_path / "a.jpg"), "upload", "a.jpg"))
    second, _ = pipe.add(Incoming(make_photo(tmp_path / "b.png"), "upload", "b.png"))
    resumed = pipe.resume()
    assert [(i.id, i.status) for i in resumed] == [(first.id, "failed"), (second.id, "done")]
    assert why in resumed[0].detail
    assert store.pending() == []


@needs_ffmpeg
def test_a_photo_too_big_to_decode_fails_with_why(tmp_path, pipe, fake, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # 64x48 is past twice this: a decompression bomb
    item = pipe.ingest(Incoming(make_photo(tmp_path / "big.jpg"), "upload", "big.jpg"))
    assert item.status == "failed"
    assert "too large" in item.detail
    assert fake.calls == []


# ---- the store


def test_store_rejects_unknown_fields_and_statuses(store):
    item, _ = store.add(sha256="ab", source="upload", name="x", original="/x", stream="s")
    with pytest.raises(ValueError, match="not an item field"):
        store.update(item.id, sha256="cd")
    with pytest.raises(ValueError, match="not a status"):
        store.update(item.id, status="lost")


def test_source_state_is_kept_per_source(store):
    a, b = src.SourceState(store, "wyze"), src.SourceState(store, "google_photos")
    a.mark("front/1.mp4")
    a.put({"cursor": 3})
    assert a.seen("front/1.mp4") and not b.seen("front/1.mp4")
    assert (a.count(), b.count()) == (1, 0)
    assert (a.get(), b.get()) == ({"cursor": 3}, {})


# ---- sources


@pytest.mark.parametrize("name", ["icloud_photos", "allsound", "thinking_hat"])
def test_stub_sources_say_they_arent_set_up(name, store):
    s = src.make(name)
    assert not s.status().configured
    assert s.status().later  # Settings lists it under Coming later (#2728)
    with pytest.raises(src.NotConfigured, match=stubs.STUBS[name].label):
        s.list_new(src.SourceState(store, name))


def test_every_source_has_a_status():
    for name in src.SOURCES:
        s = src.make(name)
        assert s.name == name
        assert isinstance(s.status().message, str) and s.status().message


def test_google_photos_without_its_package_says_so_plainly_and_holds_its_sign_in(monkeypatch, capsys):
    # The owner reads what's missing in plain words, and the sign-in waits; the command is for whoever runs the machine
    # (#2695): `pack-ingest sources` still prints it.
    monkeypatch.setitem(sys.modules, "playwright", None)  # importing it now raises ImportError
    st = src.make("google_photos").status()
    assert (st.configured, st.blocked) == (False, True)
    assert "extra package" in st.message and "uv" not in st.message and "Playwright" not in st.message
    assert st.fix is not None and "playwright install chromium" in st.fix
    runner.main(["sources"])
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("google_photos"))
    assert line.endswith(f"({st.fix})")


def _seg(d: Path, name: str, mtime: float) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"x")
    os.utime(p, (mtime, mtime))
    return p


def test_a_wyze_segment_is_new_once_the_next_one_starts_or_its_recorder_stopped(tmp_path):
    d = tmp_path / "front"
    _seg(d, "20261031-100000.mp4", 1000)
    _seg(d, "20261031-100500.mp4", 1300)
    _seg(d, "notes.mp4", 0)
    assert [p.name for p in wyze.finished_segments(d)] == ["20261031-100000.mp4"]
    assert [p.name for p in wyze.finished_segments(d, recorder_stopped=True)] == [
        "20261031-100000.mp4",
        "20261031-100500.mp4",
    ]
    assert wyze.finished_segments(tmp_path / "none") == []


def test_wyze_lists_each_cameras_finished_segments_once(tmp_path, store):
    root = tmp_path / "wyze"
    _seg(root / "front-door", "20261031-100000.mp4", 1000)
    _seg(root / "front-door", "20261031-100500.mp4", 1300)
    _seg(root / "garage", "20261031-090000.mp4", 1000)
    _seg(root / "garage", "20261031-090500.mp4", 1300)
    source = wyze.WyzeSource(["Front Door", "garage"], "rtsp://bridge:8554", root, record=False)
    state = src.SourceState(store, "wyze")
    items = source.list_new(state)
    assert [(i.key, i.stream) for i in items] == [
        ("front-door/20261031-100000.mp4", "wyze-front-door"),
        ("garage/20261031-090000.mp4", "wyze-garage"),
    ]
    assert items[0].taken_at == dt.datetime(2026, 10, 31, 10, 0).timestamp()
    assert source.fetch(items[0], tmp_path) == root / "front-door" / "20261031-100000.mp4"
    state.mark(items[0].key)
    assert [i.key for i in source.list_new(state)] == ["garage/20261031-090000.mp4"]


def test_wyze_without_cameras_isnt_set_up_and_names_what_the_bridge_lacks(tmp_path, store):
    source = wyze.WyzeSource([], "rtsp://x", tmp_path, credentials={"email": True, "password": False})
    assert source.status() == src.Status(
        False, "No cameras listed (ingest.wyze_cameras); the bridge still needs password"
    )
    with pytest.raises(src.NotConfigured):
        source.list_new(src.SourceState(store, "wyze"))


class FakeProc:
    def __init__(self, cmd):
        self.cmd, self.returncode, self.terminated = cmd, None, False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.terminated = True


def test_recorders_start_each_camera_once_and_restart_one_that_exited_after_a_pause():
    spawned: list[FakeProc] = []
    now = [0.0]
    rec = wyze.Recorders(spawn=lambda cmd: spawned.append(FakeProc(cmd)) or spawned[-1], clock=lambda: now[0])
    cams = {f"cam{i}": ["ffmpeg", f"cam{i}"] for i in range(20)}
    assert len(rec.ensure(cams)) == 20
    assert rec.ensure(cams) == []
    spawned[3].returncode = 1
    now[0] = 30.0
    assert rec.ensure(cams) == []  # too soon after its start
    now[0] = 61.0
    assert rec.ensure(cams) == ["cam3"]
    assert rec.ensure({"cam0": cams["cam0"]}) == []
    assert sum(p.terminated for p in spawned) == 19  # the cameras taken out of the config are stopped


def test_the_record_command_copies_rtsp_into_named_segments(tmp_path):
    cmd = wyze.record_command("ffmpeg", "rtsp://bridge:8554/garage", tmp_path, 300)
    assert cmd[cmd.index("-i") + 1] == "rtsp://bridge:8554/garage"
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert cmd[cmd.index("-segment_time") + 1] == "300"
    assert cmd[-1] == str(tmp_path / "%Y%m%d-%H%M%SZ.mp4")


def test_wyze_recorders_name_their_segments_in_utc(monkeypatch):
    popen = create_autospec(subprocess.Popen)
    monkeypatch.setattr(wyze.subprocess, "Popen", popen)
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    wyze._spawn(["ffmpeg", "-i", "rtsp://bridge:8554/garage"])
    assert popen.call_args.kwargs["env"]["TZ"] == "UTC0"  # ffmpeg's -strftime fills the names in its local time


def test_wyze_segments_in_the_hour_clocks_repeat_are_each_their_own(tmp_path, store):
    root = tmp_path / "wyze"
    # 01:30 Pacific on 2026-11-01 happens twice: at 08:30 UTC (PDT), then at 09:30 UTC (PST)
    _seg(root / "front", "20261101-083000Z.mp4", 1000)
    _seg(root / "front", "20261101-093000Z.mp4", 1300)
    _seg(root / "front", "20261101-093500Z.mp4", 1600)  # the open one
    source = wyze.WyzeSource(["front"], "rtsp://bridge:8554", root, record=False)
    items = source.list_new(src.SourceState(store, "wyze"))
    assert [(i.key, i.taken_at) for i in items] == [
        ("front/20261101-083000Z.mp4", 1793521800.0),
        ("front/20261101-093000Z.mp4", 1793525400.0),
    ]
    assert source.fetch(items[1], tmp_path) == root / "front" / "20261101-093000Z.mp4"


@pytest.fixture
def berlin(monkeypatch):
    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_wyze_orders_old_local_segments_before_new_utc_ones_by_time(tmp_path, store, berlin):
    # Berlin is ahead of UTC, so by name alone the newer UTC segment would sort first and pass as closed
    root = tmp_path / "wyze"
    _seg(root / "front", "20261031-090000.mp4", 1000)  # 08:00 UTC (CET), recorded before the upgrade
    _seg(root / "front", "20261031-083000Z.mp4", 1300)  # the open one
    source = wyze.WyzeSource(["front"], "rtsp://bridge:8554", root, record=False)
    assert [i.key for i in source.list_new(src.SourceState(store, "wyze"))] == ["front/20261031-090000.mp4"]


def test_wyze_keeps_a_stalled_recorders_newest_segment_until_the_process_exits(tmp_path, store):
    root = tmp_path / "wyze"
    _seg(root / "front", "20261031-100000.mp4", 1000)
    newest = _seg(root / "front", "20261031-100500.mp4", 1300)
    proc = FakeProc(["ffmpeg"])
    source = wyze.WyzeSource(
        ["front"],
        "rtsp://bridge:8554",
        root,
        record=False,
        recorders=wyze.Recorders(procs={"front": proc}),
    )
    state = src.SourceState(store, "wyze")
    assert [item.key for item in source.list_new(state)] == ["front/20261031-100000.mp4"]
    assert newest.read_bytes() == b"x"
    proc.returncode = 0
    assert [item.key for item in source.list_new(state)] == [
        "front/20261031-100000.mp4",
        "front/20261031-100500.mp4",
    ]


def test_a_dashboard_wyze_poll_defers_an_unknown_recorders_stale_newest_segment(tmp_path, store):
    root = tmp_path / "wyze"
    _seg(root / "front", "20261031-100000.mp4", 1000)
    _seg(root / "front", "20261031-100500.mp4", 1300)
    source = wyze.WyzeSource(["front"], "rtsp://bridge:8554", root, record=False)
    assert [item.key for item in source.list_new(src.SourceState(store, "wyze"))] == ["front/20261031-100000.mp4"]


def test_a_restarted_wyze_recorder_does_not_publish_its_new_open_segment(tmp_path, store, monkeypatch):
    root = tmp_path / "wyze"
    _seg(root / "front", "20261031-100000.mp4", 1000)
    _seg(root / "front", "20261031-100500.mp4", 1300)
    old = FakeProc(["ffmpeg"])
    old.returncode = 0

    def start(_cmd):
        _seg(root / "front", "20261031-101000.mp4", 1900)
        return FakeProc(["ffmpeg"])

    monkeypatch.setattr(wyze.shutil, "which", lambda _name: "ffmpeg")
    recorders = wyze.Recorders(procs={"front": old}, spawn=start)
    source = wyze.WyzeSource(["front"], "rtsp://bridge:8554", root, recorders=recorders)
    assert [item.key for item in source.list_new(src.SourceState(store, "wyze"))] == [
        "front/20261031-100000.mp4",
        "front/20261031-100500.mp4",
    ]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Photo - Landscape - Oct 31, 2026, 4:12:33 PM", dt.datetime(2026, 10, 31, 16, 12, 33)),
        ("Video - Portrait - Jan 2, 2025, 9:05 AM", dt.datetime(2025, 1, 2, 9, 5)),
        ("Photo - Landscape", None),
    ],
)
def test_google_photos_tile_times(label, expected):
    assert google_photos.label_time(label) == (expected.timestamp() if expected else None)


def test_google_photos_tiles_and_albums():
    thumb = 'url("https://lh3.googleusercontent.com/pw/AbC-12_x=w300-h200-no")'
    item = google_photos.parse_tile("./photo/AF1QipXyz", "Video - Landscape - Jan 2, 2025, 9:05 AM", thumb)
    assert item is not None
    assert (item.key, item.name, item.url) == (
        "AF1QipXyz",
        "AF1QipXyz.mp4",
        "https://lh3.googleusercontent.com/pw/AbC-12_x=dv",
    )
    assert item.extra["kind"] == "video"
    photo = google_photos.parse_tile("./album/A1/photo/P2", "Photo - Oct 31, 2026, 4:12:33 PM", thumb)
    assert photo is not None and photo.url is not None and photo.url.endswith("=d")
    assert google_photos.parse_tile("./photo/X", "Photo", 'url("https://evil.example/x=w1")') is None
    assert google_photos.parse_tile("./search", "Photo", thumb) is None
    assert google_photos.parse_album("./album/AF1Qalb", "Trip to Lisbon\n42 items") == {
        "id": "AF1Qalb",
        "title": "Trip to Lisbon",
    }
    assert google_photos.parse_album("./sharing", "Sharing") is None


def test_google_photos_downloads_only_from_googles_media_host(tmp_path):
    source = google_photos.GooglePhotosSource(tmp_path / "profile")
    bad = src.RemoteItem("k", "k.jpg", url="https://example.com/x=d")
    with pytest.raises(ValueError, match="not a Google Photos download"):
        list(source.fetch_many([bad], tmp_path))


class _GooglePage:
    """The page a signed-in profile would open to read its user agent."""

    def evaluate(self, script):
        return "test-agent"

    def close(self):
        return None


class _GoogleBrowser:
    """A signed-in profile with no Playwright attached."""

    def cookies(self):
        return []

    def new_page(self):
        return _GooglePage()


class _GoogleResponse:
    """A streaming HTTP response: `parts` arrive in order, then `error` if the connection drops."""

    def __init__(self, parts, error=None, status=200, headers=None):
        self._parts = parts
        self._error = error
        self.status_code = status
        self.ok = 200 <= status < 300
        self.headers = headers or {}
        self.is_redirect = status in (301, 302, 303, 307, 308)
        self.is_permanent_redirect = status in (301, 308)
        self.urls = []

    def iter_content(self, chunk_size=1 << 20):
        yield from self._parts
        if self._error is not None:
            raise self._error

    def close(self):
        return None


def _install_google_download(monkeypatch, response):
    """Make fetch_many download `response` instead of opening the real profile."""

    @contextmanager
    def context(self):
        yield _GoogleBrowser()

    def get(self, url, stream=True, timeout=None, allow_redirects=True):
        response.urls.append(url)
        return response

    monkeypatch.setattr(google_photos.GooglePhotosSource, "_context", context)
    monkeypatch.setattr(google_photos.requests.Session, "get", get)


def test_google_photos_streams_an_original_to_disk(monkeypatch, tmp_path):
    response = _GoogleResponse((b"orig", b"inal-bytes"))
    _install_google_download(monkeypatch, response)
    source = google_photos.GooglePhotosSource(tmp_path / "profile", max_bytes=len(b"original-bytes"))
    item = src.RemoteItem("k", "k.jpg", url="https://lh3.googleusercontent.com/pw/AbC=d")
    got = list(source.fetch_many([item], tmp_path))
    assert [(got[0][0], got[0][1].read_bytes(), response.urls)] == [(item, b"original-bytes", [item.url])]
    assert list(tmp_path.iterdir()) == [got[0][1]]


def test_google_photos_drops_a_download_over_the_cap(monkeypatch, tmp_path):
    _install_google_download(monkeypatch, _GoogleResponse((b"x" * 1000, b"yyyy")))
    source = google_photos.GooglePhotosSource(tmp_path / "profile", max_bytes=1000)
    item = src.RemoteItem("k", "big.mp4", url="https://lh3.googleusercontent.com/pw/AbC=dv")
    with pytest.raises(OSError, match="bigger than"):
        list(source.fetch_many([item], tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_google_photos_removes_the_temp_file_when_the_stream_stops(monkeypatch, tmp_path):
    _install_google_download(monkeypatch, _GoogleResponse((b"partial",), error=ConnectionError("reset")))
    source = google_photos.GooglePhotosSource(tmp_path / "profile", max_bytes=10_000)
    item = src.RemoteItem("k", "clip.mp4", url="https://lh3.googleusercontent.com/pw/AbC=dv")
    with pytest.raises(ConnectionError, match="reset"):
        list(source.fetch_many([item], tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_google_photos_stops_when_a_download_redirects_off_the_media_host(monkeypatch, tmp_path):
    response = _GoogleResponse((), status=302, headers={"Location": "https://example.com/taken"})
    _install_google_download(monkeypatch, response)
    item = src.RemoteItem("k", "k.jpg", url="https://lh3.googleusercontent.com/pw/AbC=d")
    with pytest.raises(ValueError, match="not a Google Photos download"):
        list(google_photos.GooglePhotosSource(tmp_path / "profile").fetch_many([item], tmp_path))
    assert response.urls == [item.url]
    assert list(tmp_path.iterdir()) == []


def test_google_photos_isnt_set_up_before_the_sign_in(tmp_path):
    source = google_photos.GooglePhotosSource(tmp_path / "profile")
    assert not source.status().configured
    with pytest.raises(src.NotConfigured):
        source.albums()


# ---- links


class Files(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body, kind = {
            "/pic.jpg": (b"\xff\xd8jpeg", "image/jpeg"),
            "/clip.mp4": (b"mp4bytes", "application/octet-stream"),
            "/page": (b"<html></html>", "text/html"),
            "/big.mp3": (b"x" * 3000, "audio/mpeg"),
        }.get(self.path, (b"", ""))
        if not kind:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def fake_yt_dlp(tmp_path):
    """A yt-dlp stand-in that "downloads" the page into the output template's directory and prints as yt-dlp does."""
    script = tmp_path / "yt-dlp"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys, pathlib\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).parent / 'vid123.webm'\n"
        "out.write_bytes(b'webm')\n"
        'print(\'{"id": "vid123", "title": "A talk", "timestamp": 1700000000}\')\n'
        "print(out)\n"
    )
    script.chmod(0o755)
    return str(script)


def test_a_link_to_a_file_is_downloaded_as_is(serve, tmp_path):
    base = serve(Files)
    source = link.LinkSource(yt_dlp="no-such-yt-dlp")
    into = tmp_path / "in"
    into.mkdir()
    path, taken = source.fetch_with_time(source.item(f"{base}/pic.jpg"), into)
    assert (path.name, path.read_bytes(), taken) == ("pic.jpg", b"\xff\xd8jpeg", None)
    path, _ = source.fetch_with_time(source.item(f"{base}/clip.mp4"), into)  # octet-stream, but a media name
    assert path.read_bytes() == b"mp4bytes"


def test_a_link_to_a_page_goes_to_yt_dlp(serve, tmp_path, fake_yt_dlp):
    base = serve(Files)
    into = tmp_path / "in"
    into.mkdir()
    path, taken = link.LinkSource(yt_dlp=fake_yt_dlp).fetch_with_time(
        link.RemoteItem("k", "page", url=f"{base}/page"), into
    )
    assert (path, path.read_bytes(), taken) == (into / "vid123.webm", b"webm", 1700000000.0)


def test_a_page_without_yt_dlp_says_so(serve, tmp_path):
    base = serve(Files)
    with pytest.raises(src.NotConfigured, match="isn't installed"):
        link.LinkSource(yt_dlp="no-such-yt-dlp").fetch(link.RemoteItem("k", "page", url=f"{base}/page"), tmp_path)


def test_a_download_over_the_limit_stops_and_leaves_nothing(serve, tmp_path):
    base = serve(Files)
    with pytest.raises(link.LinkError, match="bigger than"):
        link.LinkSource(max_bytes=1000).fetch(link.RemoteItem("k", "big.mp3", url=f"{base}/big.mp3"), tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x/y.jpg", "https://", "javascript:alert(1)"])
def test_only_http_links_are_taken(url):
    with pytest.raises(link.LinkError):
        link.check_url(url)


def test_file_names_come_from_the_disposition_or_the_path():
    assert link.file_name("https://x/a/b%20c.jpg?x=1") == "b c.jpg"
    assert link.file_name("https://x/dl", 'attachment; filename="../../etc/IMG_9.HEIC"') == "IMG_9.HEIC"
    assert link.file_name("https://x/") == "download"


# ---- the runner


class ListSource:
    """An ongoing source with a fixed list of local files; one key can't be fetched."""

    name, label, ongoing = "fake", "Fake", True

    def __init__(self, files: dict[str, Path], broken: str | None = None):
        self.files, self.broken = files, broken

    def status(self):
        return src.Status(True, "ok")

    def list_new(self, state):
        return [src.RemoteItem(k, p.name, taken_at=1_600_000_000.0) for k, p in self.files.items() if not state.seen(k)]

    def fetch(self, item, into):
        if item.key == self.broken:
            raise OSError("network down")
        dest = into / item.name
        shutil.copy(self.files[item.key], dest)
        return dest


@needs_ffmpeg
def test_a_poll_ingests_what_is_new_counts_duplicates_and_retries_what_it_couldnt_fetch(tmp_path, pipe, store):
    a = make_photo(tmp_path / "a.jpg", "2025:12:24 18:30:00")
    shutil.copy(a, tmp_path / "same.jpg")
    b = make_photo(tmp_path / "b.jpg", "2025:12:25 09:00:00", orientation=3)
    source = ListSource({"1": a, "2": tmp_path / "same.jpg", "3": b}, broken="3")
    first = runner.poll_source(pipe, source)
    assert (first.new, first.done, first.duplicate, first.failed, first.unfetched) == (3, 1, 1, 0, ["3"])
    source.broken = None
    second = runner.poll_source(pipe, source)
    assert (second.new, second.done) == (1, 1)
    assert runner.poll_source(pipe, source).new == 0
    assert list((pipe.root / "incoming").iterdir()) == []  # every download directory is cleaned up


def test_a_poll_of_a_source_that_isnt_set_up_reports_it(pipe):
    result = runner.poll_source(pipe, src.make("allsound"))
    assert result.not_configured is not None and "AllSound" in result.not_configured


class PollSource(ListSource):
    def __init__(self, error=None, stop=None, recorders=None):
        super().__init__({})
        self.error, self.stop, self.recorders = error, stop, recorders
        self.calls = 0

    def list_new(self, state):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.stop is not None:
            self.stop.set()
        return []


def test_a_source_listing_failure_has_an_actionable_poll_result(pipe):
    result = runner.poll_source(pipe, PollSource(error=OSError("network down")))
    assert result.error == "network down"
    assert "network down" in str(result)
    assert result.new == result.done == 0


def test_an_offline_source_does_not_stop_polling_other_sources(pipe):
    stop = threading.Event()
    offline = PollSource(error=OSError("network down"))
    available = PollSource(stop=stop)
    runner.poll(pipe, [offline, available], every_s=0, stop=stop)
    assert offline.calls == available.calls == 1


def test_a_store_failure_still_propagates_and_stops_recorders(pipe):
    proc = FakeProc(["ffmpeg"])
    recorder = wyze.Recorders(procs={"front": proc})
    broken = PollSource(error=sqlite3.DatabaseError("database disk image is malformed"), recorders=recorder)
    with pytest.raises(sqlite3.DatabaseError, match="malformed"):
        runner.poll(pipe, [broken], every_s=0, stop=threading.Event())
    assert proc.terminated
    assert recorder.procs == {}


class DeleteLedger:
    def __init__(self, item_id):
        self.rows = {item_id: {"frames": 2}}

    def remove(self, item_id):
        self.rows.pop(item_id, None)


class DeletePurger:
    def __init__(self, finished=True):
        self.finished = finished
        self.items = []
        self.checked = []

    def purge_item(self, item):
        self.items.append(item.id)
        return "purge-1" if item.outputs else None

    def purge_finished(self, purge_id):
        self.checked.append(purge_id)
        return self.finished


def deletion_item(pipe, tmp_path):
    original = tmp_path / "original.wav"
    original.write_bytes(b"an imported soundtrack")
    item, _ = pipe.add(Incoming(original, "upload", "original.wav", key="remote-key"))
    converted = pipe.root / "media" / "converted.flac"
    converted.parent.mkdir(parents=True)
    converted.write_bytes(b"converted")
    return pipe.store.update(
        item.id,
        status="done",
        kind="audio",
        outputs=[{"kind": "audio", "stream": "ingest-upload.audio_v1", "chunk_id": "chunk-1", "path": str(converted)}],
    )


def test_delete_removes_files_ledger_and_item_but_does_not_reuse_the_id(pipe, tmp_path):
    item = deletion_item(pipe, tmp_path)
    ledger, purger = DeleteLedger(item.id), DeletePurger()
    assert pipe.delete(item.id, ledger, purger) == {"ok": True, "deleted": True, "pending": False}
    assert pipe.store.get(item.id) is None
    assert pipe.store.removed(item.id)
    assert pipe.store.recent() == []
    assert ledger.rows == {}
    assert not Path(item.original).exists()
    assert not Path(item.outputs[0]["path"]).exists()
    assert pipe.store.seen("upload", "remote-key")  # a source must not silently import deleted media again
    newer = tmp_path / "new.wav"
    newer.write_bytes(b"another soundtrack")
    new_item, _ = pipe.add(Incoming(newer, "upload", "new.wav"))
    assert new_item.id > item.id
    assert pipe.delete(item.id, ledger, purger) == {"ok": True, "deleted": True, "pending": False}
    assert pipe.store.get(new_item.id) == new_item
    assert purger.items == [item.id]


def test_delete_waits_for_storage_ack_and_resumes_the_same_receipt_after_restart(pipe, tmp_path):
    item = deletion_item(pipe, tmp_path)
    ledger, purger = DeleteLedger(item.id), DeletePurger(finished=False)
    assert pipe.delete(item.id, ledger, purger, wait_s=0) == {"ok": True, "deleted": True, "pending": True}
    assert pipe.store.recent() == []
    assert pipe.store.finished() == []
    assert pipe.store.deletions() == [
        {
            "id": item.id,
            "name": "original.wav",
            "detail": "Waiting for storage cleanup acknowledgments.",
            "blocked": False,
        }
    ]
    assert Path(item.original).exists()  # keep bytes available to keepers until they acknowledge
    assert ledger.rows == {}
    restarted = Pipeline(Store(pipe.store.path), pipe.root, pipe.registrar)
    purger.finished = True
    assert restarted.delete(item.id, ledger, purger, wait_s=0) == {"ok": True, "deleted": True, "pending": False}
    assert purger.items == [item.id]
    assert purger.checked == ["purge-1", "purge-1"]
    assert restarted.store.deletions() == []


def test_delete_fences_active_work_before_purging_and_rejects_queued_retries(pipe, tmp_path):
    item = deletion_item(pipe, tmp_path)
    ledger, purger = DeleteLedger(item.id), DeletePurger()
    with pipe.store.item_lock(item.id):
        assert pipe.delete(item.id, ledger, purger, wait_s=0) == {"ok": True, "deleted": True, "pending": True}
        assert pipe.store.removed(item.id)
        assert purger.items == []
        assert ledger.rows == {item.id: {"frames": 2}}
    with pytest.raises(ItemGone):
        pipe.process(item.id)
    assert pipe.delete(item.id, ledger, purger, wait_s=0)["pending"] is False
    assert purger.items == [item.id]


def test_a_missing_item_delete_is_safe_and_does_not_reserve_an_id(pipe):
    assert pipe.delete(100, DeleteLedger(100), DeletePurger()) == {"ok": True, "deleted": False, "pending": False}
    assert pipe.store.deletion(100) is None


def test_item_guard_excludes_another_process_until_active_work_releases_it(store):
    probe = """
import sys
from pathlib import Path
from liberational.ingest.store import ItemGone, Store
try:
    with Store(Path(sys.argv[1])).item_lock(1, blocking=False):
        print('acquired')
except BlockingIOError:
    print('busy')
"""
    with store.item_lock(1):
        waiting = subprocess.run(
            [sys.executable, "-c", probe, str(store.path)], check=True, capture_output=True, text=True, timeout=5
        )
    assert waiting.stdout.strip() == "busy"
    released = subprocess.run(
        [sys.executable, "-c", probe, str(store.path)], check=True, capture_output=True, text=True, timeout=5
    )
    assert released.stdout.strip() == "acquired"


def test_delete_recreates_a_missing_purge_receipt_instead_of_stalling_forever(pipe, tmp_path):
    from continuum_py import BrokerError

    item = deletion_item(pipe, tmp_path)
    ledger = DeleteLedger(item.id)

    class MissingReceipt(DeletePurger):
        def purge_item(self, item):
            self.items.append(item.id)
            return f"purge-{len(self.items)}"

        def purge_finished(self, purge_id):
            if purge_id == "purge-1":
                raise BrokerError("NotFound", "purge receipt is no longer present")
            return True

    purger = MissingReceipt()
    assert pipe.delete(item.id, ledger, purger, wait_s=0)["pending"] is True
    assert pipe.store.deletion(item.id)["purge_id"] is None
    assert pipe.delete(item.id, ledger, purger, wait_s=0)["pending"] is False
    assert purger.items == [item.id, item.id]


@pytest.mark.parametrize("refused", [False, True])
def test_retry_retains_previously_registered_chunks_for_later_deletion(pipe, tmp_path, monkeypatch, refused):
    item = deletion_item(pipe, tmp_path)
    item = pipe.store.update(item.id, outputs=[{**item.outputs[0], "chunk_id": "older-chunk"}])
    monkeypatch.setattr(media, "probe", lambda path: media.Probe("audio", has_audio=True))
    monkeypatch.setattr(media, "to_flac", lambda original, dest: Path(item.outputs[0]["path"]))
    if refused:
        pipe.registrar = FakeRegistrar(("ingest-upload.audio_v1",))
    result = pipe.process(item.id)
    assert "older-chunk" in [o["chunk_id"] for o in result.outputs]
    assert result.status == ("failed" if refused else "done")


def test_deleting_an_item_forgets_its_segment_slot_and_nothing_else() -> None:
    import numpy as np

    from liberational import segment_log, speech_log
    from liberational.ingest import pipeline

    store = segment_log.root()
    item, other = 900_001, 900_002  # slots no other test writes
    for n in (item, other):
        writer = speech_log.Writer(pipeline.slot(n), store)
        writer.write(speech_log.UTTERANCES, {"ts_start": 1_800_000_000.0 + n, "text": f"line {n}"})
        writer.close()
        voice = segment_log.Writers(store)
        voice.append(
            segment_log.VOICE_EMBEDDINGS,
            pipeline.slot(n),
            [
                {
                    "start_ns": n * 10**9,
                    "end_ns": n * 10**9,
                    "voiceprint": "0199a000-0000-7000-8000-000000000001",
                    "vector": np.zeros(segment_log.VOICE_DIM, np.float32),
                }
            ],
        )
        voice.close()

    pipeline.forget_slot(item)

    texts = {line["text"] for line in speech_log.lines(1_800_000_000.0, 1_800_000_000.0 + 10**6, base=store)}
    assert texts >= {f"line {other}"} and f"line {item}" not in texts
    left = {r["start_ns"] // 10**9 for r in segment_log.Log(store).range(segment_log.VOICE_EMBEDDINGS, 0.0, 2.0**32)}
    assert other in left and item not in left
