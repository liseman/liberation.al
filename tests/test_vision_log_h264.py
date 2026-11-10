"""Recorded H.264 crops decode from the GOP's keyframe (vision_log.h264_picture)."""

import asyncio
from typing import Any, cast

import pytest

from liberational import vision_log

S = 10**9
T0 = 1_800_000_000 * S


class Decoder:
    """The native decoder's rich contract: nothing until a keyframe (b"K"), then each access unit's picture as
    (tag, width, height, NV12, range, matrix, BGR), or with `hold`, each picture one unit late, the last one only
    from finish_outputs(). A `signaled` stream gives its own BGR (every byte the tag); one that isn't gives none
    and mid-grey NV12."""

    def __init__(self, hold=False, signaled=True):
        self.waiting, self.hold, self.held, self.pushed, self.signaled = True, hold, None, [], signaled

    def _picture(self, tag):
        if self.signaled:
            return (tag, 2, 2, bytes([128] * 6), "full", "bt601", bytes([tag % 256] * 12))
        return (tag, 2, 2, bytes([128] * 6), "unspecified", "unspecified", None)

    def push_outputs(self, data, tag, presentation_ns, want_bgr):
        assert want_bgr, "crops take the decoder's own colour conversion"
        self.pushed.append(tag)
        if self.waiting and data != b"K":
            return []
        self.waiting = False
        if not self.hold:
            return [self._picture(tag)]
        out, self.held = ([] if self.held is None else [self._picture(self.held)]), tag
        return out

    def finish_outputs(self, want_bgr):
        out, self.held = ([] if self.held is None else [self._picture(self.held)]), None
        return out


def frames(start_s, kinds):
    """One record per second from start_s, its bytes saying keyframe (K) or delta (P)."""
    return [(T0 + (start_s + i) * S, kind.encode()) for i, kind in enumerate(kinds)]


@pytest.fixture
def stretches(monkeypatch):
    """The broker's records as stretches, recording each [lo, hi) asked for."""
    asked, decoders = [], []
    stored: dict[str, Any] = {"records": [], "hold": False, "signaled": True}

    async def stretch(client, stream, lo, hi):
        asked.append((lo, hi))
        records = [r for r in stored["records"] if lo <= r[0] < hi]
        decoders.append(Decoder(stored["hold"], stored["signaled"]))
        return vision_log._Gop(records, False, decoders[-1]) if records else None

    monkeypatch.setattr(vision_log, "_h264_stretch", stretch)
    monkeypatch.setattr(vision_log, "_OPEN", type(vision_log._OPEN)())
    stored["asked"], stored["decoders"] = asked, decoders
    return stored


def picture(at_s, tolerance_ns=1000):
    return asyncio.run(vision_log.h264_picture(cast(Any, None), "cam0", T0 + at_s * S, tolerance_ns))


def test_a_delta_picture_decodes_from_the_keyframe_before_it(stretches):
    stretches["records"] = frames(0, "KPPPP")
    image, at_ns, flip = picture(3)
    assert (image.shape, at_ns, flip) == ((2, 2, 3), T0 + 3 * S, False)
    assert stretches["decoders"][0].pushed == [0, 1, 2, 3]  # from the keyframe, and no further


def test_a_keyframe_further_back_than_the_first_try_is_found_by_the_second(stretches):
    stretches["records"] = frames(0, "K" + "P" * 9)  # the keyframe is 9 s before the wanted picture
    assert picture(9) is not None
    assert len(stretches["asked"]) == 2
    assert stretches["asked"][1][0] == T0 + 9 * S - vision_log.to_ns(vision_log.GOP_LOOKBACK_S[1])


def test_with_no_keyframe_before_it_there_is_no_picture(stretches):
    stretches["records"] = frames(0, "PPPPK")
    assert picture(2) is None


def test_a_later_picture_in_the_same_stretch_decodes_on_from_the_open_decoder(stretches):
    stretches["records"] = frames(0, "KPPPPP")
    assert picture(1) is not None and picture(3) is not None
    assert len(stretches["asked"]) == 1
    assert stretches["decoders"][0].pushed == [0, 1, 2, 3]


def test_an_earlier_picture_than_the_decoder_reached_starts_a_new_stretch(stretches):
    stretches["records"] = frames(0, "KPPPP")
    assert picture(3) is not None and picture(1) is not None
    assert len(stretches["asked"]) == 2
    assert stretches["decoders"][1].pushed == [0, 1]


def test_a_picture_the_decoder_holds_back_comes_from_the_next_units_without_ending_the_decoder(stretches):
    stretches["records"], stretches["hold"] = frames(0, "KPPPP"), True
    image, at_ns, _ = picture(2)
    assert (at_ns, int(image[0, 0, 0])) == (T0 + 2 * S, 2)
    assert stretches["decoders"][0].pushed == [0, 1, 2, 3]  # one unit on, not drained
    assert picture(3) is not None and len(stretches["asked"]) == 1  # the same decoder goes on


def test_a_picture_held_back_at_the_stretchs_end_is_drained(stretches):
    stretches["records"], stretches["hold"] = frames(0, "KPP"), True
    _image, at_ns, _ = picture(2)
    assert at_ns == T0 + 2 * S


def test_the_decoders_own_colours_are_kept(stretches):
    stretches["records"] = frames(0, "KPP")
    image, _at, _flip = picture(2)
    assert image.tolist() == [[[2, 2, 2]] * 2] * 2  # the decoder's BGR, not a reconversion of its NV12


def test_a_stream_that_signals_no_colours_falls_back_to_limited_bt601(stretches):
    stretches["records"], stretches["signaled"] = frames(0, "KPP"), False
    image, _at, _flip = picture(2)
    # NV12 all 128, as limited-range BT.601: Y 128 -> (128 - 16) * 255 / 219 = 130.4, chroma neutral
    assert abs(int(image.min()) - 130) <= 1 and abs(int(image.max()) - 130) <= 1


def test_a_time_with_no_picture_near_it_gives_none(stretches):
    stretches["records"] = frames(0, "KPP")
    assert picture(1.5, tolerance_ns=S // 10) is None


def test_records_go_in_decode_order_not_time_order(monkeypatch):
    """A chunk keeps its stored (decode) order; Tier 0 units follow by capture epoch, then device sequence, those
    capture didn't attest by time; a unit in both tiers is decoded once."""
    chunk = {"shape_name": "H264AnnexB", "start_ns": T0, "capture": {}}
    items = [{"header": chunk, "locator": {"kind": "Chunk"}}]
    stored = {id(chunk): [("a", 0), ("b", 3), ("c", 1)]}  # a B-frame shown before the P it was decoded after
    for payload, at_s, epoch, sequence in [("c", 1, 5, 3), ("d", 4, 5, 5), ("e", 5, 5, 4), ("f", 6, 0, None)]:
        header = {"shape_name": "H264AnnexB", "start_ns": T0 + at_s * S, "capture": {"capture_epoch": epoch}}
        if sequence is not None:
            header["capture"]["device_seq"] = sequence
        items.append({"header": header, "locator": {"kind": "Shm"}})
        stored[id(header)] = [(payload, at_s)]

    class Client:
        async def query(self, stream, lo, hi):
            return {"items": items}

    async def records(client, stream, item, *, root, start_ns, end_ns):
        return [
            vision_log.read.Record(payload, T0 + at_s * S, T0 + at_s * S, payload.encode(), None)
            for payload, at_s in stored[id(item["header"])]
        ]

    monkeypatch.setattr(vision_log.continuum_py, "decode_header", lambda header: header)
    monkeypatch.setattr(vision_log.read, "records", records)
    gop = asyncio.run(vision_log._h264_stretch(cast(Any, Client()), "cam0.video", 0, 2**62))
    assert gop is not None
    assert [data for _start, data in gop.records] == [b"a", b"b", b"c", b"e", b"d", b"f"]


def unit_record(start_ns, unit, key=True):
    """continuum-tier's one-unit record, as a Tier 1 chunk stores an access unit."""
    row = start_ns.to_bytes(8, "little") * 2 + bytes([int(key)]) + (0).to_bytes(4, "little")
    return (1).to_bytes(4, "little") + row + len(unit).to_bytes(4, "little") + unit


@pytest.mark.parametrize(
    ("data", "unit"),
    [
        pytest.param(unit_record(T0, b"\0\0\0\x01\x65\x88"), b"\0\0\0\x01\x65\x88", id="chunk-h264"),
        pytest.param(unit_record(T0, b"\x12\x00\x0a\x01\x00"), b"\x12\x00\x0a\x01\x00", id="chunk-av1"),
        pytest.param(b"\0\0\0\x01\x65\x88", b"\0\0\0\x01\x65\x88", id="tier0-annexb"),
        pytest.param(unit_record(T0, b"abc")[:-1], unit_record(T0, b"abc")[:-1], id="truncated-kept-whole"),
    ],
)
def test_a_chunk_records_access_unit_is_unwrapped_for_the_decoder(data, unit):
    assert vision_log._unit(data) == unit


def test_a_stretch_from_h264_into_an_archived_av1_chunk_switches_decoders(stretches, monkeypatch):
    """The archive pass leaves some chunks H.264 next to AV1 ones: the H.264 decoder is drained, then an AV1
    decoder starts at the AV1 chunk's keyframe."""
    opened = []

    def decoder(codec="h264"):
        opened.append(codec)
        return Decoder()

    monkeypatch.setattr(vision_log.continuum_py, "H264Decoder", decoder)
    records = frames(0, "KPKP")
    first = Decoder()
    gop = vision_log._Gop(records, False, first, ["h264", "h264", "av1", "av1"], "h264")
    assert vision_log._advance(gop, 3) is not None
    assert (opened, first.pushed, gop.codec, gop.decoder.pushed) == (["av1"], [0, 1], "av1", [2, 3])
    assert vision_log._codec(b"\x12\x00\x0a") == "av1"
    assert vision_log._codec(b"\0\0\0\x01\x67") == "h264"


def test_a_clip_is_one_stretch_decoded_once_from_its_keyframe(stretches):
    stretches["records"] = frames(0, "KPPPPPPP")
    got = asyncio.run(vision_log.clip_frames(cast(Any, None), "cam0", 1_800_000_002.0, 1_800_000_005.0, 1.0))
    assert [None if f is None else f.shape for f in got] == [(2, 2, 3)] * 3
    assert len(stretches["asked"]) == 1  # one read for the whole clip, not one per frame
    assert stretches["decoders"][0].pushed == [0, 1, 2, 3, 4]  # from the keyframe through the last tick, once


def test_a_slow_cameras_picture_serves_every_tick_near_it(stretches):
    stretches["records"] = frames(0, "KPPP")
    # Ticks every 0.5 s within 0.25 s of a picture: 2.0 has one, 2.5 doesn't, 3.0 does.
    got = asyncio.run(vision_log.clip_frames(cast(Any, None), "cam0", 1_800_000_002.0, 1_800_000_003.5, 2.0))
    assert [f is not None for f in got] == [True, False, True]


def test_a_clip_from_a_b_frame_stream_fills_every_tick_with_its_own_picture(stretches):
    # Decode order isn't time order: the picture at 2 s is decoded before the one at 1 s, 4 s before 3 s.
    stretches["records"] = [(T0 + s * S, kind) for s, kind in ((0, b"K"), (2, b"P"), (1, b"P"), (4, b"P"), (3, b"P"))]
    got = asyncio.run(vision_log.clip_frames(cast(Any, None), "cam0", 1_800_000_000.0, 1_800_000_005.0, 1.0))
    # Each signaled picture is filled with its decode index, so the tick at 1 s must show index 2, and so on.
    assert [None if f is None else int(f[0, 0, 0]) for f in got] == [0, 2, 1, 4, 3]
