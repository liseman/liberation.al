"""The skeleton playback test page's reads (/skeletons, #1987): recent movement spans, and one span's poses.

`spans` lists see-actions' `action.span` and `walkby.span` rows, newest first. `poses` gives one track's `pose.track`
rows in a window (the span plus its lead-in and lead-out) as frames of stick-figure keypoints. Pose rows outlive the
30-day video window, so a span plays as skeletons after its video is gone.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from liberational import segment_log
from liberational.see.actions import POSE_SCHEMA, SCALE
from liberational.see.actions_worker import POSES

NS = 1_000_000_000
MAX_SPANS = 300
MAX_WINDOW_S = 600.0  # one span's playback, lead-in and lead-out included


def spans(state: str | Path, *, hours: float = 24.0, now: float | None = None) -> list[dict[str, Any]]:
    """The last `hours` of action and walk-by spans, newest first, at most MAX_SPANS."""
    end = time.time() if now is None else now
    log = segment_log.at(state)
    out: list[dict[str, Any]] = []
    for stream, source in ((segment_log.ACTION_SPANS, "action"), (segment_log.WALKBY_SPANS, "walk-by")):
        for r in log.range(stream, end - hours * 3600, end):
            direction = (r.get("attributes") or {}).get("direction")
            kind = r["kind"] if source == "action" else f"walk-by {direction}" if direction else "walk-by"
            out.append(
                {
                    "id": r["id"],
                    "source": source,
                    "cam": r["cam"],
                    "kind": kind,
                    "start": r["ts_start"],
                    "end": r["ts_end"],
                    "track_id": str(r["track_id"]),
                    "person": r.get("person_name"),
                }
            )
    out.sort(key=lambda s: float(s["start"]), reverse=True)
    return out[:MAX_SPANS]


def poses(state: str | Path, *, cam: str, track_id: str, start: float, end: float) -> dict[str, Any]:
    """One track's poses on `cam` in [start, end), as frames in time order: each frame's time, box (x1, y1, x2, y2,
    score), and 26 keypoints [x, y, confidence], 0 to 1 of the frame (-1, -1, 0 where the model placed none)."""
    from continuum_py import read_segments

    if not end > start or end - start > MAX_WINDOW_S:
        raise ValueError(f"the window must be positive and at most {MAX_WINDOW_S:.0f} s")
    rows = read_segments(segment_log.root(state), POSES, POSE_SCHEMA, round(start * NS), round(end * NS), [cam])
    frames = []
    for i, tid in enumerate(rows["track_id"]):
        if str(int(tid)) != track_id:
            continue
        joints = [int(v) / SCALE for v in rows["joints"][i]]
        frames.append(
            {
                "ts": int(rows["ts_ns"][i]) / NS,
                "box": [float(v) for v in rows["box"][i]],
                "keypoints": [joints[k : k + 3] for k in range(0, len(joints), 3)],
            }
        )
    frames.sort(key=lambda f: f["ts"])
    return {"cam": cam, "track_id": track_id, "start": start, "end": end, "frames": frames}
