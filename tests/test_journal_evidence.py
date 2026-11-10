"""Journal evidence that degrades in steps (#2808): which moments a day page's citations promote, what promotion keeps
while video is going, and what a citation resolves to at each stage (video, clip, still, audio, expired).

A day page and its hour entries go in the suite's in-memory memory log; frames and video stretches in the in-memory
vision log (tests/vision_fake.py); clips are encoded by a stand-in unless a test is about real encoding.
"""

import shutil

import numpy as np
import pytest

from liberational import memory_log, segment_log, speech_log
from liberational.memory import journal
from liberational.memory import journal_evidence as E

DAY = "2026-10-20"
START, _ = journal.day_bounds(DAY)
H = 3600.0


def source(stream, ts, text, camera=None, end=None):
    return {"id": f"{stream}-{ts}", "stream": stream, "ts_start": ts, "ts_end": ts if end is None else end,
            "people": [], "camera": camera, "kind": None, "text": text}  # fmt: skip


def hour(memory, hour_of_day, sources):
    a = START + hour_of_day * H
    record = {"ts_start": a, "ts_end": a + H, "level": 3600, "text": "an hour",
              "evidence": {"status": "verified", "sources": sources}}  # fmt: skip
    return memory.append(memory_log.JOURNAL, record)


def page(memory, parents, md):
    record = {"ts_start": memory_log.day_bounds(DAY)[0], "ts_end": START + 24 * H, "level": "day", "period": DAY,
              "md": md}  # fmt: skip
    return memory.append(memory_log.PAGES, record, parents=parents)


def frames_for(vision, camera, centre, *, only=None):
    """The camera's frames over a clip around `centre`, at the clip's ticks (or only the first `only` of them)."""
    n = round(E.CLIP_S * E.CLIP_FPS)
    ticks = [centre - E.CLIP_S / 2 + i / E.CLIP_FPS for i in range(n)][:only]
    for t in ticks:
        vision.frames[(camera, t)] = np.full((360, 640, 3), 90, np.uint8)
    return ticks


class Encoder:
    """Stands in for ffmpeg: writes a small file and remembers what it was given."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[int, ...]]] = []

    def __call__(self, frames, dst):
        self.calls.append((len(frames), frames[0].shape))
        dst.write_bytes(b"mp4" * 100)
        return True


@pytest.fixture
def cited(memory):
    """A day page citing h1 and h3 (not h2). h1 has five camera sources on one camera and a speech line; h3 one
    caption on another camera."""
    h1 = hour(
        memory,
        9,
        [
            source(segment_log.CAPTIONS, START + 9 * H + 60, "a cat on the couch", "den"),
            source(segment_log.CAPTIONS, START + 9 * H + 600, "an empty room", "den"),
            source("action.span", START + 9 * H + 300, "cat jumped down", "den", end=START + 9 * H + 330),
            source("action.span", START + 9 * H + 304, "cat jumped down", "den", end=START + 9 * H + 306),
            source("zone.visit", START + 9 * H + 1200, "cat at the bowl", "den", end=START + 9 * H + 1260),
            source(speech_log.UTTERANCES, START + 9 * H + 90, "Good morning"),
        ],
    )
    h2 = hour(memory, 12, [source(segment_log.CAPTIONS, START + 12 * H, "lunch", "kitchen")])
    h3 = hour(memory, 18, [source(segment_log.CAPTIONS, START + 18 * H + 5, "a dog at the door", "porch")])
    page(memory, [h1, h2, h3], "# Monday\n\n## In short\nThe cat was busy [h1](#t=1) and a dog came by [h3](#t=3).\n")
    return memory


def test_a_day_page_promotes_only_the_cited_hours_detected_spans_first(cited, tmp_path):
    moments = E.cited_moments(tmp_path, DAY)
    assert [(m.label, m.camera, m.ts - START, m.stream) for m in moments] == [
        # h1: three of its five camera sources, detected spans before captions; 9:05:04 is the same moment as
        # 9:05:00 (within SAME_MOMENT_S) and the span is centred CLIP_S / 2 in.
        ("h1", "den", 9 * H + 60, segment_log.CAPTIONS),
        ("h1", "den", 9 * H + 303, "action.span"),
        ("h1", "den", 9 * H + 1203, "zone.visit"),
        ("h3", "porch", 18 * H + 5, segment_log.CAPTIONS),
    ]


def test_promotion_keeps_a_small_clip_a_still_and_the_caption_once(cited, tmp_path, vision):
    moments = E.cited_moments(tmp_path, DAY)
    for m in moments:
        frames_for(vision, m.camera, m.ts)
    encode = Encoder()
    counts = E.promote(tmp_path, moments, vision=vision, encode=encode)
    assert (counts["clip"], counts["still"], counts["gone"]) == (4, 0, 0)
    # Every clip is the whole CLIP_S at CLIP_FPS, scaled down to CLIP_WIDTH.
    assert encode.calls == [(48, (270, 480, 3))] * 4
    found = E.index(tmp_path, DAY)
    porch = found[moments[3].key]
    assert (porch["caption"], porch["clip"], porch["still"]) == (
        "a dog at the door",
        f"{moments[3].key}.mp4",
        f"{moments[3].key}.jpg",
    )
    assert (E.folder(tmp_path, DAY) / porch["still"]).stat().st_size > 0
    assert E.promote(tmp_path, moments, vision=vision, encode=encode)["clip"] == 0  # done once, never again
    assert len(encode.calls) == 4


def test_a_moment_whose_video_is_already_going_keeps_what_is_left(cited, tmp_path, vision):
    # The purge got there first (a page written late, or the disk floor's early hours): a few frames left make a
    # still and no clip, none leave only the caption.
    moments = E.cited_moments(tmp_path, DAY)
    frames_for(vision, moments[0].camera, moments[0].ts, only=E.MIN_FRAMES - 1)
    encode = Encoder()
    counts = E.promote(tmp_path, moments[:2], vision=vision, encode=encode)
    assert (counts["clip"], counts["still"], counts["gone"], encode.calls) == (0, 1, 1, [])
    found = E.index(tmp_path, DAY)
    assert "clip" not in found[moments[0].key] and found[moments[0].key]["still"]
    assert found[moments[1].key] == {"camera": "den", "ts": moments[1].ts, "caption": "cat jumped down", "label": "h1"}


def test_an_interrupted_promotion_keeps_what_it_made(cited, tmp_path, vision):
    moments = E.cited_moments(tmp_path, DAY)
    for m in moments:
        frames_for(vision, m.camera, m.ts)
    made = []

    def encode(frames, dst):
        if made:
            raise RuntimeError("killed")
        made.append(dst)
        dst.write_bytes(b"mp4")
        return True

    with pytest.raises(RuntimeError):
        E.promote(tmp_path, moments, vision=vision, encode=encode)
    assert list(E.index(tmp_path, DAY)) == [moments[0].key]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_real_clip_is_a_small_h264_mp4(tmp_path):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (270, 480, 3), dtype=np.uint8) for _ in range(48)]
    dst = tmp_path / "clip.mp4"
    assert E.encode_clip(frames, dst)
    data = dst.read_bytes()
    assert data[4:8] == b"ftyp" and b"avc1" in data  # an MP4 holding H.264
    assert len(data) < 2_000_000  # noise is the worst case for size; a real moment is far smaller


def resolved(tmp_path, vision, src, audio=False):
    return E.resolve(tmp_path, DAY, src, vision=vision, audio_kept=lambda a, b: audio)


def test_a_camera_moment_resolves_video_then_clip_then_still_then_expired(tmp_path, vision):
    t = START + 18 * H + 5
    src = source(segment_log.CAPTIONS, t, "a dog at the door", "porch")
    vision.video.append(("porch", t - 60, t + 60))
    assert resolved(tmp_path, vision, src) == {
        "stage": "video",
        "camera": "porch",
        "t": t,
        "caption": "a dog at the door",
    }

    vision.video.clear()
    vision.aged.append(("porch", t - 60, t + 60))
    assert resolved(tmp_path, vision, src)["note"] == "footage expired"
    assert resolved(tmp_path, vision, src)["stage"] == "expired"

    m = E.Moment(DAY, "h3", "porch", t, "a dog at the door", segment_log.CAPTIONS)
    E._save_index(tmp_path, DAY, {m.key: {"camera": "porch", "ts": t, "still": f"{m.key}.jpg", "caption": m.caption}})
    assert resolved(tmp_path, vision, src)["url"] == f"/journal/evidence/{DAY}/{m.key}.jpg"
    assert resolved(tmp_path, vision, src)["stage"] == "still"

    E._save_index(tmp_path, DAY, {m.key: {"camera": "porch", "ts": t, "clip": f"{m.key}.mp4", "still": f"{m.key}.jpg"}})
    got = resolved(tmp_path, vision, src)
    assert (got["stage"], got["url"], got["still"]) == (
        "clip",
        f"/journal/evidence/{DAY}/{m.key}.mp4",
        f"/journal/evidence/{DAY}/{m.key}.jpg",
    )


def test_a_moment_with_no_video_ever_says_so_rather_than_expired(tmp_path, vision):
    got = resolved(tmp_path, vision, source(segment_log.CAPTIONS, START + 5, "a room", "attic"))
    assert (got["stage"], got["note"], got["caption"]) == ("expired", "no footage was kept", "a room")


def test_speech_plays_its_audio_while_it_is_kept_and_then_says_it_expired(tmp_path, vision):
    line = source(speech_log.UTTERANCES, START + 90, "Good morning")
    assert resolved(tmp_path, vision, line, audio=True) == {
        "stage": "audio",
        "t": START + 90,
        "caption": "Good morning",
    }
    assert resolved(tmp_path, vision, line, audio=False) == {
        "stage": "expired",
        "note": "audio expired",
        "caption": "Good morning",
    }


def test_resolve_day_lists_every_cited_source_with_its_stage(cited, tmp_path, vision):
    vision.video.append(("den", START, START + 24 * H))
    got = E.resolve_day(tmp_path, DAY, vision=vision, audio_kept=lambda a, b: True)
    assert [(r["label"], r["stage"]) for r in got] == [("h1", "video")] * 5 + [("h1", "audio"), ("h3", "expired")]


@pytest.mark.parametrize("name", ["../index.json", "index.json", "x.mp4/../../a.jpg", "a.png"])
def test_only_promoted_media_files_are_served(tmp_path, name):
    (E.folder(tmp_path, DAY)).mkdir(parents=True)
    (E.folder(tmp_path, DAY) / "index.json").write_text("{}")
    assert E.file(tmp_path, DAY, name) is None


def test_a_promoted_file_is_served_by_day_and_name(tmp_path):
    E.folder(tmp_path, DAY).mkdir(parents=True)
    (E.folder(tmp_path, DAY) / "den-1.jpg").write_bytes(b"jpeg")
    assert E.file(tmp_path, DAY, "den-1.jpg") == E.folder(tmp_path, DAY) / "den-1.jpg"
    assert E.file(tmp_path, "2026-10-21", "den-1.jpg") is None


def test_a_source_folded_into_a_promoted_moment_resolves_to_its_clip(tmp_path, vision):
    # cited_moments folds a camera's sources within SAME_MOMENT_S into one promoted moment (here at t); a source 4 s
    # later in that group finds the same clip instead of reading as expired.
    t = START + 9 * H + 303
    m = E.Moment(DAY, "h1", "den", t, "cat jumped down", "action.span")
    E._save_index(tmp_path, DAY, {m.key: {"camera": "den", "ts": t, "clip": f"{m.key}.mp4", "still": f"{m.key}.jpg"}})
    vision.aged.append(("den", START, START + 24 * H))
    near = resolved(tmp_path, vision, source("action.span", t + 4, "cat jumped down", "den"))
    assert (near["stage"], near["url"]) == ("clip", f"/journal/evidence/{DAY}/{m.key}.mp4")
    far = resolved(tmp_path, vision, source("action.span", t + E.SAME_MOMENT_S + 1, "later", "den"))
    assert (far["stage"], far["note"]) == ("expired", "footage expired")
    other = resolved(tmp_path, vision, source("action.span", t + 4, "elsewhere", "porch"))
    assert other["stage"] == "expired"  # another camera's moment never stands in


def test_resolve_day_reads_the_broker_once_per_camera_not_per_source(cited, tmp_path, vision, monkeypatch):
    calls = []
    for name in ("video_spans", "aged_out"):
        real = getattr(vision, name)
        monkeypatch.setattr(
            vision, name, lambda cam, a, b, real=real, name=name: calls.append((name, cam)) or real(cam, a, b)
        )
    vision.video.append(("den", START, START + 24 * H))
    got = E.resolve_day(tmp_path, DAY, vision=vision, audio_kept=lambda a, b: True)
    assert len(got) == 7  # six sources in h1, one in h3
    assert sorted(calls) == [
        ("aged_out", "den"),
        ("aged_out", "porch"),
        ("video_spans", "den"),
        ("video_spans", "porch"),
    ]


def test_a_run_promotes_at_most_its_limit_oldest_first_and_leaves_the_rest_for_the_next(cited, tmp_path, vision):
    moments = E.cited_moments(tmp_path, DAY)
    for m in moments:
        frames_for(vision, m.camera, m.ts)
    first = E.promote(tmp_path, moments, vision=vision, encode=Encoder(), limit=3)
    assert (first["clip"], first["deferred"]) == (3, 1)
    assert set(E.index(tmp_path, DAY)) == {m.key for m in moments[:3]}
    second = E.promote(tmp_path, moments, vision=vision, encode=Encoder(), limit=3)
    assert (second["clip"], second["deferred"]) == (1, 0)


def test_a_run_stops_promoting_when_its_time_is_up(cited, tmp_path, vision):
    moments = E.cited_moments(tmp_path, DAY)
    for m in moments:
        frames_for(vision, m.camera, m.ts)
    ticks = iter([0.0, 5.0, 11.0, 12.0])  # the clock read before each moment: past the deadline at the third
    got = E.promote(tmp_path, moments, vision=vision, encode=Encoder(), deadline=10.0, clock=lambda: next(ticks))
    assert (got["clip"], got["deferred"]) == (2, 2)


def test_each_clip_is_one_read_of_its_video(cited, tmp_path, vision):
    moments = E.cited_moments(tmp_path, DAY)
    for m in moments:
        frames_for(vision, m.camera, m.ts)
    E.promote(tmp_path, moments, vision=vision, encode=Encoder())
    assert vision.clips == len(moments)


def test_a_promoted_clip_is_served_in_byte_ranges(tmp_path, monkeypatch, serve):
    import http.client
    from urllib.parse import urlsplit

    from liberational.dashboard import server

    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    E.folder(tmp_path, DAY).mkdir(parents=True)
    (E.folder(tmp_path, DAY) / "den-1.mp4").write_bytes(bytes(range(100)))
    port = urlsplit(serve(server.H)).port
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    conn.request("GET", f"/journal/evidence/{DAY}/den-1.mp4", headers={"Range": "bytes=10-19"})
    reply = conn.getresponse()
    body = reply.read()
    conn.close()
    assert (reply.status, reply.getheader("Content-Range"), reply.getheader("Accept-Ranges")) == (
        206,
        "bytes 10-19/100",
        "bytes",
    )
    assert (reply.getheader("Content-Type"), body) == ("video/mp4", bytes(range(10, 20)))


def test_a_nearer_moment_with_nothing_left_never_hides_a_clip(tmp_path, vision):
    t = START + 9 * H + 300
    gone = E.Moment(DAY, "h1", "den", t + 1, "late page", "action.span")
    kept = E.Moment(DAY, "h1", "den", t + 8, "cat jumped down", "action.span")
    E._save_index(
        tmp_path,
        DAY,
        {
            gone.key: {"camera": "den", "ts": t + 1, "caption": "late page"},
            kept.key: {"camera": "den", "ts": t + 8, "clip": f"{kept.key}.mp4", "still": f"{kept.key}.jpg"},
        },
    )
    got = resolved(tmp_path, vision, source("action.span", t, "cat jumped down", "den"))
    assert (got["stage"], got["url"]) == ("clip", f"/journal/evidence/{DAY}/{kept.key}.mp4")


def test_a_clip_that_comes_out_mostly_empty_names_its_camera(cited, tmp_path, vision, caplog):
    m = E.cited_moments(tmp_path, DAY)[3]
    frames_for(vision, m.camera, m.ts, only=10)  # 10 of 48 ticks have a picture
    caplog.set_level("WARNING")
    got = E.promote(tmp_path, [m], vision=vision, encode=Encoder())
    assert got["clip"] == 1  # still made: 10 frames is enough for a short clip
    assert "only 10 of 48 frames for porch" in caplog.text
