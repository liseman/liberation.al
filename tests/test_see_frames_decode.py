"""Camera frames as the vision workers decode them (see/frames.py): MJPEG and UYVY, flipped when capture says so,
and the manifest's cameras."""

from pathlib import Path

import cv2
import numpy as np

from camera_fake import capture
from liberational.see import frames
from liberational.see.frames import decode_frame


def test_an_mjpeg_frame_decodes_and_flips_when_capture_says_so() -> None:
    img = np.zeros((48, 64, 3), np.uint8)
    img[:24] = 255  # the top half white
    ok, jpeg = cv2.imencode(".jpg", img)
    assert ok
    plain = decode_frame("MJPEG", jpeg.tobytes(), capture(64, 48))
    flipped = decode_frame("MJPEG", jpeg.tobytes(), capture(64, 48, flip=True))
    assert plain is not None and flipped is not None
    assert plain.shape == (48, 64, 3)
    assert plain[0, 0, 0] > 200 and plain[47, 0, 0] < 50
    assert flipped[0, 0, 0] < 50 and flipped[47, 0, 0] > 200


def test_a_uyvy_frame_decodes_at_the_headers_size_and_bad_bytes_dont() -> None:
    uyvy = np.full((2, 4, 2), 128, np.uint8)  # mid grey: U=V=128, Y=128
    img = decode_frame("UYVY", uyvy.tobytes(), capture(4, 2))
    assert img is not None and img.shape == (2, 4, 3)
    assert abs(int(img[0, 0, 0]) - 128) < 3
    assert decode_frame("UYVY", uyvy.tobytes()[:8], capture(4, 2)) is None, "too few bytes for the size"
    assert decode_frame("NV12", b"", capture(4, 2)) is None, "not a format the worker reads"
    assert decode_frame("MJPEG", b"not a jpeg", None) is None


def test_the_manifests_cameras(tmp_path: Path) -> None:
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        '[[device]]\nname = "facecam"\nkind = "camera"\n\n[[device]]\nname = "wave3"\nkind = "microphone"\n'
    )
    assert [c.stream for c in frames.cameras_from_manifest(manifest)] == ["facecam.video"]
