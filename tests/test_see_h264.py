"""Synthetic decoder outputs verify header association without cameras or codec installs."""

import logging
from typing import cast

import pytest

from liberational.see.frames import H264Frames, cameras_from_manifest


class Decoder:
    waiting_for_keyframe = True

    def __init__(self):
        self.tags = []
        self.dropped_tags = []
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.waiting_for_keyframe = True
        self.tags.clear()

    def push(self, data, tag):
        self.tags.append(tag)
        self.waiting_for_keyframe = False
        if len(self.tags) == 3:
            # Delayed output belongs to the first AU, even with regressing input PTS.
            return [(self.tags[0], 2, 2, bytes([128] * 6))]
        return []


def header(seq, pts):
    from continuum_py import PayloadHeader

    return cast(
        PayloadHeader,
        {
            "shape_name": "H264AnnexB",
            "start_ns": pts,
            "seq": seq,
            "capture": {"width": 2, "height": 2, "flip": False, "device_seq": seq},
        },
    )


def test_every_au_decodes_before_sampling_and_delayed_output_keeps_original_header(monkeypatch):
    decoder = Decoder()
    monkeypatch.setattr("liberational.see.frames.continuum_py.H264Decoder", lambda: decoder, raising=False)
    frames = H264Frames()
    first = header(10, 100)
    assert frames.push(first, b"idr") == []
    assert frames.push(header(11, 400), b"p") == []
    [(image, original)] = frames.push(header(12, 200), b"b")
    assert decoder.tags == [1, 2, 3]
    assert original is first
    assert image.shape == (2, 2, 3)
    assert len(frames.pending) == 2


def test_decode_sequence_gap_resets_pending_headers(monkeypatch):
    decoder = Decoder()
    monkeypatch.setattr("liberational.see.frames.continuum_py.H264Decoder", lambda: decoder, raising=False)
    frames = H264Frames()
    frames.push(header(1, 100), b"idr")
    frames.push(header(3, 300), b"idr")
    assert decoder.resets == 2  # initial layout, then missing packet
    assert list(frames.pending) == [2]


def test_optional_decoder_error_is_actionable(monkeypatch):
    def unsupported():
        raise NotImplementedError("optional ffmpeg feature required")

    monkeypatch.setattr("liberational.see.frames.continuum_py.H264Decoder", unsupported, raising=False)
    with pytest.raises(NotImplementedError, match="ffmpeg"):
        H264Frames()


def test_rtsp_devices_are_vision_cameras(tmp_path):
    path = tmp_path / "devices.toml"
    path.write_text('[[device]]\nname="network-camera"\nkind="rtsp"\n')
    assert [camera.stream for camera in cameras_from_manifest(path)] == ["network-camera.video"]


def test_newest_feeds_all_h264_units_while_consumer_is_busy(monkeypatch):
    import asyncio

    import numpy as np

    from liberational.see import frames

    class Subscription:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def __aiter__(self):
            for sequence in range(8):
                yield {"type": "Event", "item": {"header": sequence}}
                await asyncio.sleep(0)

    class Client:
        async def subscribe(self, *args, **kwargs):
            return Subscription()

    fed = []

    def read(client, item, decoder, want=None):
        sequence = item["header"]
        fed.append(sequence)
        return [(np.zeros((2, 2, 3), np.uint8), header(sequence, sequence))]

    monkeypatch.setattr(frames.continuum_py, "decode_header", lambda value: header(value, value))
    monkeypatch.setattr(frames, "H264Frames", lambda **_: object())
    monkeypatch.setattr(frames, "read_h264", read)

    async def consume():
        stream = frames.newest(cast(frames.Client, Client()), frames.Camera("synthetic"))
        first = await anext(stream)
        assert first[1]["seq"] == 0
        for _ in range(16):
            await asyncio.sleep(0)
        latest = await anext(stream)
        assert latest[1]["seq"] == 7
        assert fed == list(range(8))
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    asyncio.run(consume())


def test_corrupt_au_resets_only_its_decoder_and_recovers_at_next_idr(monkeypatch, caplog):
    class DamagedDecoder(Decoder):
        def push(self, data, tag):
            if data == b"corrupt":
                raise ValueError("decode send: Invalid data found when processing input")
            if self.waiting_for_keyframe and data != b"idr":
                return []
            self.waiting_for_keyframe = False
            return [(tag, 2, 2, bytes([128] * 6))]

    decoder = DamagedDecoder()
    monkeypatch.setattr("liberational.see.frames.continuum_py.H264Decoder", lambda: decoder, raising=False)
    frames = H264Frames(name="kitchen")
    assert len(frames.push(header(1, 100), b"idr")) == 1
    with caplog.at_level(logging.WARNING, logger="liberational.see-frames"):
        assert frames.push(header(2, 200), b"corrupt") == []
    assert frames.pending == {} and decoder.waiting_for_keyframe
    # The line names the camera and the decoder's reason (246 of them in 8 hours said neither).
    reason = "decode send: Invalid data found when processing input"
    assert caplog.messages == [f"kitchen: an H264 access unit could not be decoded ({reason}); waiting for a keyframe"]
    assert frames.push(header(3, 300), b"delta") == []
    next_idr = header(4, 400)
    [(image, original)] = frames.push(next_idr, b"idr")
    assert original is next_idr and image.shape == (2, 2, 3)


def test_a_lost_access_unit_is_a_break_but_the_first_units_reset_is_not(monkeypatch):
    decoder = Decoder()
    monkeypatch.setattr("liberational.see.frames.continuum_py.H264Decoder", lambda: decoder, raising=False)
    breaks = []
    frames = H264Frames(on_break=lambda: breaks.append(len(decoder.tags)))
    frames.push(header(1, 100), b"idr")
    frames.push(header(2, 200), b"p")
    assert breaks == []
    frames.push(header(4, 400), b"idr")  # 3 never came
    assert breaks == [0], "the break comes before the unit after the gap is decoded"


def test_a_frame_lost_before_it_is_read_is_a_break(monkeypatch):
    import asyncio

    import numpy as np

    from liberational.see import frames

    asked = asyncio.Event()

    class Subscription:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def __aiter__(self):
            for sequence in range(3):
                yield {"type": "Event", "item": {"header": sequence}}
                await asked.wait()
                asked.clear()

    class Client:
        async def subscribe(self, *args, **kwargs):
            return Subscription()

    def read(client, item):
        sequence = item["header"]
        return None if sequence == 1 else (np.zeros((2, 2, 3), np.uint8), header(sequence, sequence))

    monkeypatch.setattr(frames.continuum_py, "decode_header", lambda value: {"shape_name": "MJPEG"})
    monkeypatch.setattr(frames, "read_frame", read)
    events = []

    def broke():
        events.append("break")
        asked.set()

    async def consume():
        stream = frames.newest(cast(frames.Client, Client()), frames.Camera("synthetic"), on_break=broke)
        events.append((await anext(stream))[1]["seq"])
        asked.set()
        events.append((await anext(stream))[1]["seq"])
        await stream.aclose()

    asyncio.run(asyncio.wait_for(consume(), 5))  # without the break, the camera would wait forever
    assert events == [0, "break", 2]


@pytest.mark.parametrize("loss", ["missing_locator", "wrong_locator", "arena", "lease", "capture"])
def test_unreadable_h264_input_breaks_and_discards_delayed_pictures(monkeypatch, loss):
    from liberational.see import frames as module

    decoder = Decoder()
    monkeypatch.setattr(module.continuum_py, "H264Decoder", lambda: decoder, raising=False)
    breaks = []
    reader = H264Frames(on_break=lambda: breaks.append("break"))
    reader.push(header(1, 100), b"idr")
    reader.push(header(2, 200), b"p")
    assert reader.pending and breaks == []  # normal buffering is not loss
    lost = header(3, 300)
    if loss == "capture":
        lost.pop("capture")
        assert reader.push(lost, b"p") == []
    else:
        monkeypatch.setattr(module.continuum_py, "decode_header", lambda _: lost)

        class Client:
            def arena(self, _):
                if loss == "arena":
                    raise module.continuum_py.ContinuumError("lost arena")
                return self

            def lease(self, *args):
                raise module.continuum_py.ContinuumError("lost lease")

        locator = None
        if loss == "wrong_locator":
            locator = {"kind": "File"}
        elif loss in {"arena", "lease"}:
            locator = {"kind": "Shm", "arena": "synthetic", "slab_class": 0, "slot": 0, "slot_generation": 0}
        assert (
            module.read_h264(
                cast(module.Client, Client()), cast(module.Item, {"header": b"", "locator": locator}), reader
            )
            == []
        )
    assert breaks == ["break"]
    assert reader.pending == {}
    assert decoder.waiting_for_keyframe and decoder.tags == []
    assert reader.push(header(4, 400), b"idr") == []
    assert breaks == ["break"], "recovery must not report the same loss twice"
    assert decoder.tags == [3], "pre-break delayed pictures cannot escape after recovery"


def test_a_picture_the_caller_turns_down_still_decodes_but_is_not_returned(monkeypatch):
    decoder = Decoder()
    monkeypatch.setattr("liberational.see.frames.continuum_py.H264Decoder", lambda: decoder, raising=False)
    frames = H264Frames()
    asked = []

    def want(original):
        asked.append(original["seq"])
        return False

    frames.push(header(10, 100), b"idr", want)
    frames.push(header(11, 400), b"p", want)
    assert frames.push(header(12, 200), b"b", want) == []
    assert decoder.tags == [1, 2, 3]
    assert asked == [10]
    assert list(frames.pending) == [2, 3]  # the turned-down picture's header is released


@pytest.mark.parametrize("codec", ["H264AnnexB", "MJPEG"])
@pytest.mark.parametrize(
    "clocks,expected,breaks",
    [
        # Slots of 200 ms at about 17 fps: the first frame of each. The restarted clock is a break, and its frames
        # count from the slot they're in.
        pytest.param(
            [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 90, 150, 210],
            [0, 240, 420, 600, 90, 210],
            1,
            id="slots-then-restart",
        ),
        # A clock that goes back within the slot just taken, and recovers before the next, still reaches the
        # caller with a break, so its progress fences it.
        pytest.param([100, 90, 210], [100, 90, 210], 1, id="regression-inside-the-taken-slot"),
        pytest.param([100, 100, 210], [100, 100, 210], 1, id="repeated-clock"),
    ],
)
def test_capped_cameras_reach_the_model_once_per_capture_slot(monkeypatch, codec, clocks, expected, breaks):
    import asyncio

    import numpy as np

    from liberational.see import frames

    ms = 1_000_000
    clocks = [c * ms for c in clocks]

    class Subscription:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def __aiter__(self):
            for i in range(len(clocks)):
                yield {"type": "Event", "item": {"header": i}}
                for _ in range(4):
                    await asyncio.sleep(0)

    class Client:
        async def subscribe(self, *args, **kwargs):
            return Subscription()

    read = []

    def picture(i):
        return (np.zeros((2, 2, 3), np.uint8), {**header(i, clocks[i]), "shape_name": codec})

    def read_h264(client, item, decoder, want=None):
        value = picture(item["header"])
        if want is not None and not want(value[1]):
            return []
        read.append(item["header"])
        return [value]

    def read_frame(client, item):
        read.append(item["header"])
        return picture(item["header"])

    monkeypatch.setattr(frames.continuum_py, "decode_header", lambda i: picture(i)[1])
    monkeypatch.setattr(frames, "H264Frames", lambda **_: object())
    monkeypatch.setattr(frames, "read_h264", read_h264)
    monkeypatch.setattr(frames, "read_frame", read_frame)

    broken = []

    async def consume():
        stream = frames.newest(
            cast(frames.Client, Client()),
            frames.Camera("patio", "rtsp", 200 * ms),
            on_break=lambda: broken.append(True),
        )
        return [frame[1]["start_ns"] // ms async for frame in stream]

    assert asyncio.run(consume()) == expected
    assert len(broken) == breaks
    assert len(read) == len(expected)  # the skipped frames were never read or converted


def decoder_errors(monkeypatch, caplog, kinds, available):
    from liberational.see import frames

    monkeypatch.setattr("liberational.see.frames.continuum_py.h264_decoder_available", lambda: available)
    with caplog.at_level("ERROR"):
        frames.check_h264_decoder([frames.Camera(f"cam{i}", kind) for i, kind in enumerate(kinds)])
    return [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]


def test_a_worker_start_errors_loudly_naming_network_cameras_it_cannot_decode(monkeypatch, caplog):
    (error,) = decoder_errors(monkeypatch, caplog, ("camera", "rtsp", "rtsp"), available=False)
    assert "cam1, cam2 get NO vision" in error
    assert "mise run continuum-py" in error


@pytest.mark.parametrize(
    ("kinds", "available"),
    [
        pytest.param(("camera", "rtsp"), True, id="decoder-present"),
        pytest.param(("camera",), False, id="no-network-cameras"),
    ],
)
def test_no_decoder_error_when_every_camera_can_be_decoded(monkeypatch, caplog, kinds, available):
    assert decoder_errors(monkeypatch, caplog, kinds, available) == []


def test_default_settings_read_network_cameras_through_the_shared_decoder(tmp_path):
    import tomllib
    from pathlib import Path

    defaults = tomllib.loads((Path(__file__).parents[1] / "config" / "defaults.toml").read_text())
    manifest = tmp_path / "devices.toml"
    manifest.write_text("[[device]]\nname='synthetic'\nkind='rtsp'\n[[device]]\nname='usb'\nkind='camera'\n")
    cameras = cameras_from_manifest(manifest, vision=defaults["vision"])
    assert [(camera.name, camera.shared_decode) for camera in cameras] == [("synthetic", True), ("usb", False)]


def test_shared_decode_wildcard_preserves_native_readers_and_encoding_overrides(tmp_path):
    manifest = tmp_path / "devices.toml"
    manifest.write_text("""[video.encoding]
enabled=true
[[device]]
name='network'
kind='rtsp'
[[device]]
name='encoded'
kind='camera'
[[device]]
name='native'
kind='camera'
video={encoding={enabled=false}}
""")
    cameras = cameras_from_manifest(manifest, vision={"shared_decode_cameras": ["*"]})
    assert [(camera.name, camera.shared_decode) for camera in cameras] == [
        ("network", True),
        ("encoded", True),
        ("native", False),
    ]


def test_native_nv12_preserves_delayed_header_without_cpu_conversion(monkeypatch):
    from unittest.mock import create_autospec

    import cv2

    decoder = Decoder()
    monkeypatch.setattr(
        "liberational.see.frames.continuum_py.H264Decoder",
        create_autospec(Decoder, return_value=decoder),
        raising=False,
    )
    convert = create_autospec(cv2.cvtColor, side_effect=AssertionError("NV12 must not convert to BGR"))
    monkeypatch.setattr(cv2, "cvtColor", convert)
    frames = H264Frames()
    first = header(10, 100)
    assert frames.push_nv12(first, b"idr") == []
    assert frames.push_nv12(header(11, 400), b"p") == []
    [(pixels, width, height, original)] = frames.push_nv12(header(12, 200), b"b")
    assert (pixels, width, height) == (bytes([128] * 6), 2, 2)
    assert original is first
    assert decoder.tags == [1, 2, 3]
    convert.assert_not_called()


def test_capture_epoch_change_discards_old_references_even_when_sequence_is_contiguous(monkeypatch):
    from unittest.mock import create_autospec

    decoder = Decoder()
    monkeypatch.setattr(
        "liberational.see.frames.continuum_py.H264Decoder",
        create_autospec(Decoder, return_value=decoder),
        raising=False,
    )
    breaks = []
    frames = H264Frames(on_break=lambda: breaks.append("gap"))
    first = header(1, 100)
    first["capture"]["capture_epoch"] = 5
    frames.push_nv12(first, b"old-idr")
    next_epoch = header(2, 200)
    next_epoch["capture"]["capture_epoch"] = 6
    assert frames.push_nv12(next_epoch, b"new-idr") == []
    assert decoder.resets == 2
    assert breaks == ["gap"]
    assert list(frames.pending.values()) == [next_epoch]
    assert first not in frames.pending.values()


def test_shared_newest_reads_at_the_cameras_rate_and_keeps_gaps_without_private_decoder(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import create_autospec

    import numpy as np

    from liberational.see import frames, shared_frames

    async def run():
        release = asyncio.Event()
        finished = asyncio.Event()
        gap_sent = asyncio.Event()
        next_picture = asyncio.Event()
        clock = [0.0]
        originals = [header(n, n * 1_000_000_000) for n in range(5)]
        pixels = np.zeros((2, 2, 3), np.uint8)

        async def shared(camera, *, format, on_break=None, every_ns=0):
            assert (camera, format, every_ns) == ("synthetic", "bgr", 2_000_000_000)
            yield shared_frames.DecodedFrame(originals[0], pixels, "synthetic-epoch", False)
            await release.wait()
            clock[0] = 1.1
            yield shared_frames.DecodedFrame(originals[1], pixels, "synthetic-epoch", False)
            clock[0] = 2.2
            yield shared_frames.DecodedFrame(originals[2], pixels, "synthetic-epoch", False)
            assert on_break is not None
            on_break()
            gap_sent.set()
            await next_picture.wait()
            clock[0] = 3.3
            yield shared_frames.DecodedFrame(originals[3], pixels, "synthetic-epoch", True)
            clock[0] = 4.4
            yield shared_frames.DecodedFrame(originals[4], pixels, "synthetic-epoch", False)
            finished.set()

        private = create_autospec(frames.H264Frames, side_effect=AssertionError("private decoder constructed"))
        monkeypatch.setattr(frames, "H264Frames", private)
        monkeypatch.setattr(shared_frames, "decoded_frames", shared)
        monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        gaps = []
        stream = frames.newest(
            cast(frames.Client, object()),
            frames.Camera("synthetic", every_ns=2_000_000_000, shared_decode=True),
            on_break=lambda: gaps.append("gap"),
        )
        try:
            first = await anext(stream)
            release.set()
            await asyncio.wait_for(gap_sent.wait(), 2)
            assert gaps == ["gap"], "standalone loss is delivered before the next picture"
            next_picture.set()
            await asyncio.wait_for(finished.wait(), 2)
            last = await anext(stream)
            assert first[1] is originals[0]
            assert last[1] is originals[4]
            private.assert_not_called()
        finally:
            await stream.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("every_ns", "min_interval_s", "asked_ns"),
    [
        pytest.param(0, 1.5, 1_500_000_000, id="the-model-interval"),
        pytest.param(2_000_000_000, 0.0, 2_000_000_000, id="the-cameras-slots"),
    ],
)
def test_shared_newest_asks_the_decoder_for_its_rate_and_takes_what_it_sends(
    monkeypatch, every_ns, min_interval_s, asked_ns
):
    # The decoder paces a shared reader, so a picture it sends a moment after the last is the next one, not
    # one to drop for the model's interval or slot.
    import asyncio
    from types import SimpleNamespace

    import numpy as np

    from liberational.see import frames, shared_frames

    async def run():
        clock = [0.0]
        pixels = np.zeros((2, 2, 3), np.uint8)
        asked = []
        taken = asyncio.Event()

        async def shared(camera, *, format, on_break=None, every_ns=0):
            asked.append(every_ns)
            yield shared_frames.DecodedFrame(header(1, 100), pixels, "epoch", False)
            await taken.wait()
            clock[0] = 0.25  # a moment later, well inside the model's interval
            yield shared_frames.DecodedFrame(header(2, 200), pixels, "epoch", False)

        monkeypatch.setattr(shared_frames, "decoded_frames", shared)
        monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        stream = frames.newest(
            cast(frames.Client, object()),
            frames.Camera("synthetic", every_ns=every_ns, shared_decode=True),
            min_interval_s=min_interval_s,
        )
        try:
            first = (await anext(stream))[1]["start_ns"]
            taken.set()
            second = (await asyncio.wait_for(anext(stream), 2))[1]["start_ns"]
            return asked, [first, second]
        finally:
            await stream.aclose()

    assert asyncio.run(run()) == ([asked_ns], [100, 200])


def test_standalone_shared_gap_drops_pending_latest_before_any_following_picture(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import numpy as np

    from liberational.see import frames, shared_frames

    async def run():
        release = asyncio.Event()
        gap_sent = asyncio.Event()
        next_picture = asyncio.Event()
        hold_open = asyncio.Event()
        clock = [0.0]
        original = header(1, 0)
        lost_buffer = header(2, 2_000_000_000)
        replacement = header(3, 2_100_000_000)
        pixels = np.zeros((2, 2, 3), np.uint8)

        async def shared(camera, *, format, on_break=None, every_ns=0):
            yield shared_frames.DecodedFrame(original, pixels, "epoch", False)
            await release.wait()
            clock[0] = 1.1
            yield shared_frames.DecodedFrame(lost_buffer, pixels, "epoch", False)
            assert on_break is not None
            on_break()
            gap_sent.set()
            await next_picture.wait()
            clock[0] = 2.2
            yield shared_frames.DecodedFrame(replacement, pixels, "epoch", True)
            await hold_open.wait()

        monkeypatch.setattr(shared_frames, "decoded_frames", shared)
        monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        gaps = []
        stream = frames.newest(
            cast(frames.Client, object()),
            frames.Camera("synthetic", every_ns=2_000_000_000, shared_decode=True),
            min_interval_s=1.0,
            on_break=lambda: gaps.append("gap"),
        )
        try:
            assert (await anext(stream))[1] is original
            release.set()
            await asyncio.wait_for(gap_sent.wait(), 2)
            assert gaps == ["gap"]
            pending = asyncio.create_task(anext(stream))
            try:
                await asyncio.sleep(0)  # Run the pull against the already-ready, now-invalidated buffer.
                assert not pending.done(), "pre-loss buffered pixels must not reach a model after DecodedGap"
                next_picture.set()
                assert (await asyncio.wait_for(pending, 2))[1] is replacement
            finally:
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
        finally:
            await stream.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("rich", [False, True])
def test_decoder_drops_release_original_headers_and_report_a_gap(monkeypatch, rich):
    class DroppingDecoder(Decoder):
        def push(self, data, tag):
            self.waiting_for_keyframe = False
            self.dropped_tags = [1, 2] if tag == 3 else []
            return [(tag, 2, 2, bytes([128] * 6))] if tag >= 3 else []

        def push_outputs(self, data, tag, presentation_ns, want_bgr, keep=True):
            return [(t, w, h, p, "limited", "bt601", None) for t, w, h, p in self.push(data, tag)]

    decoder = DroppingDecoder()
    monkeypatch.setattr("liberational.see.frames.continuum_py.H264Decoder", lambda: decoder)
    breaks = []
    frames = H264Frames(on_break=lambda: breaks.append("gap"))
    frames.push_nv12(header(1, 100), b"idr", rich=rich)
    frames.push_nv12(header(2, 200), b"p", rich=rich)
    third = header(3, 300)
    third["id"] = "third"
    [(pixels, width, height, original)] = frames.push_nv12(third, b"p", rich=rich)
    assert original is third and (len(pixels), width, height) == (6, 2, 2)
    assert frames.pending == {}, "the lost pictures' source headers must not accumulate"
    assert breaks == ["gap"]
    fourth = header(4, 400)
    fourth["id"] = "fourth"
    assert frames.push_nv12(fourth, b"p", rich=rich)[0][3] is fourth
    assert frames.pending == {} and breaks == ["gap"]
    assert decoder.resets == 1, "reporting a codec's discarded picture does not reset healthy decoding"
