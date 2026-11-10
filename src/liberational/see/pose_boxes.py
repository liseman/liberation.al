"""`pose.boxes`: see-actions' tracked people for the Live overlay, in a Tier 0 ring that never reaches the catalog, as `face.boxes` is for faces.

`pose.track` reaches the segment store in 1 s batches, and a page following it got its skeletons 0.4 to 0.6 s late,
in bursts. Every analyzed frame with a tracked person gets one record here as it is analyzed: the camera, the
frame's start, and each person's track, box (x1, y1, x2, y2 normalized to the frame, and the detector's score), and
halpe26 joints ([x, y, confidence] normalized to the frame; a joint not placed is [-1, -1, 0]). That is the shape
the gateway reads `pose.track` in, so its overlay takes either. Recorded video's skeletons still read `pose.track`.

A record is a little-endian u32 length and then that many bytes of JSON, zero-padded to `STRIDE`, as `face.boxes`'
are (`face_boxes.decode` reads it). A frame whose people don't fit keeps its largest.
"""

import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from continuum_py import Client, RingWriter

STREAM = "pose.boxes"
STRIDE = 8192  # about 650 bytes a person: a dozen people a frame
RATE_HZ = 128.0  # records a second, across every camera: cam0 at 30 and the rest at 10
SECONDS = 10.0  # live only; nothing looks back
DIGITS = 4


def record(cam: str, start_ns: int, people: list[tuple[int, Any, Any]]) -> dict[str, Any]:
    """The record for one frame: `people` is (track, box, joints) per person, the box as the detector gives it and
    the joints K x 3 normalized to the frame (x, y, confidence; -1, -1, 0 where not placed)."""
    return {
        "cam": cam,
        "start_ns": start_ns,
        "people": [
            {
                "track_id": str(track),
                "box": np.round(np.asarray(box[:5], np.float64), DIGITS).tolist(),
                "keypoints": np.round(np.asarray(joints, np.float64), DIGITS).tolist(),
            }
            for track, box, joints in people
        ],
    }


def encode(rec: dict[str, Any]) -> bytes:
    """One ring record for `rec`; people that don't fit go, smallest box first."""
    people = sorted(rec.get("people", []), key=lambda p: -_area(p.get("box")))
    while True:
        body = json.dumps({**rec, "people": people}, separators=(",", ":")).encode()
        if 4 + len(body) <= STRIDE or not people:
            return len(body).to_bytes(4, "little") + body
        people = people[:-1]


def decode(data: bytes) -> dict[str, Any] | None:
    """The record a ring slot holds, or None for an empty, torn, or foreign one."""
    from liberational.see import face_boxes

    return face_boxes.decode(data)


def _area(box: Any) -> float:
    if not isinstance(box, list) or len(box) < 4:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


async def writer(client: Client) -> RingWriter:
    """Registers the ring stream (the same ring on every registration) and takes its writer."""
    registered = await client.register_stream(
        STREAM,
        kind="derived",
        format="Text",
        shape_name=STREAM,
        nominal_rate_hz=RATE_HZ,
        ring_stride_bytes=STRIDE,
        ring_seconds=SECONDS,
        layer="Layer2Decision",
    )
    lease = registered["lease"]
    if lease is None:
        raise RuntimeError(f"the broker gave {STREAM} no ring")
    return client.ring_writer(lease)
