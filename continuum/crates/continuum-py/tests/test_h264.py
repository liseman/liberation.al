"""Optional decoder API works in both lightweight and FFmpeg SDK wheels."""

from pathlib import Path

import pytest


def test_decoder_capability_matches_constructor() -> None:
    from continuum_py import H264Decoder, h264_decoder_available

    assert h264_decoder_available() == H264Decoder.supported()
    if not h264_decoder_available():
        with pytest.raises(NotImplementedError, match="ffmpeg"):
            H264Decoder()
        return
    decoder = H264Decoder()
    assert decoder.waiting_for_keyframe
    with pytest.raises(ValueError, match="access unit"):
        decoder.push(b"", 0)
    decoder.reset()
    assert decoder.waiting_for_keyframe


def test_delayed_b_frames_return_their_original_input_tags() -> None:
    from continuum_py import H264Decoder, gop_units, h264_decoder_available

    if not h264_decoder_available():
        pytest.skip("optional ffmpeg decoder is not built")
    fixture = Path(__file__).resolve().parents[2] / "continuum-tier/tests/data/bframes.mp4"
    units = gop_units(str(fixture), 0, 0)
    decoder = H264Decoder()
    pictures = []
    for tag, unit in enumerate(units):
        pictures.extend(decoder.push(unit["bytes"], tag))
    pictures.extend(decoder.finish())
    assert len(pictures) == len(units)
    assert sorted(tag for tag, _, _, _ in pictures) == list(range(len(units)))
    presentation = [units[tag]["pts_ns"] for tag, _, _, _ in pictures]
    assert presentation == sorted(presentation)
    assert all((width, height, len(data)) == (64, 48, 4608) for _, width, height, data in pictures)
