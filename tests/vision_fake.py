"""An in-memory vision log for tests of `vision_log`'s readers: `vision_log.at` returns it (the `vision` fixture in
conftest.py installs it), and a test adds the records the broker would hold."""

import copy
import uuid
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from liberational import segment_log, vision_log

_DTYPES = {
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "u8": np.uint8,
    "f32": np.float32,
    "f64": np.float64,
    "bool": np.bool_,
}


def _uuid7(ts: float, n: int) -> str:
    """A uuid7 whose time is `ts` (seconds) and whose random bits are the counter `n`."""
    ms = max(0, round(ts * 1000))
    return str(uuid.UUID(int=(ms << 80) | (0x7 << 76) | (0b10 << 62) | n))


class FakeVisionLog:
    """Records by stream, read back as `VisionLog` reads them: overlapping a window, in time order."""

    def __init__(self) -> None:
        self.records: dict[str, list[dict[str, Any]]] = {}
        self.newest_frames: dict[str, tuple[Any, float, str]] = {}
        self.frames: dict[tuple[str, float], Any] = {}
        self.video: list[tuple[str, float, float]] = []  # (cam, start, end): where a camera has video
        self.aged: list[tuple[str, float, float]] = []  # (cam, start, end): where its video was aged out
        self.reads = 0  # range reads, for tests of what presence reads again

    def stamp(self, start: float, end: float, streams: Any = None) -> tuple[Any, ...]:
        """As `segment_log.Log.stamp`: moves when a record of the window is added or taken out."""
        return tuple(
            (stream, tuple(r["id"] for r in rows if r["ts_start"] < end and r.get("ts_end", r["ts_start"]) >= start))
            for stream, rows in sorted(self.records.items())
            if streams is None or stream in streams
        )

    def add(self, stream: str, **record: Any) -> dict[str, Any]:
        """Adds one record (`ts_start` in seconds, optionally `ts_end`, and its fields), with an id, and returns it."""
        # A uuid7 made at the record's time, as the workers' are (readers find a record near the time in its id), its
        # counter the order records were added (presence names an episode's evidence in id order)
        r = {
            "ts_end": record["ts_start"],
            "id": _uuid7(record["ts_start"], sum(map(len, self.records.values()))),
            **record,
        }
        self.records.setdefault(stream, []).append(r)
        return r

    def retained(self, stream: str, *, window: tuple[float, float] | None = None) -> list[dict[str, Any]]:
        """The current retained raw records, copied as the real log's incremental reader returns them."""
        rows = self.records.get(stream, [])
        if window is not None:
            rows = [
                r
                for r in rows
                if vision_log.to_ns(r["ts_start"]) < vision_log.to_ns(window[1])
                and max(vision_log.to_ns(r["ts_end"]), vision_log.to_ns(r["ts_start"]) + 1)
                > vision_log.to_ns(window[0])
            ]
        return copy.deepcopy(sorted(rows, key=lambda r: (r["ts_start"], r["id"])))

    def warm_retained(self, stream: str, *, window: tuple[float, float]) -> bool:
        """The in-memory fixture can publish every complete window immediately."""
        self.retained(stream, window=window)
        return True

    def bounded(
        self, stream: str, start: float, end: float, *, max_items: int, max_records: int
    ) -> list[dict[str, Any]] | None:
        """One fixture record is one catalog item; overflowing either budget requests a fallback."""
        rows = self.range(stream, start, end)
        return rows if len(rows) <= min(max_items, max_records) else None

    def caption(self, ts: float, cam: str = "cam0", caption: str | None = "a room", **fields: Any) -> dict[str, Any]:
        """Adds a `camera.caption` record (a fresh one unless `fields` say otherwise)."""
        fields.setdefault("observation", {"version": 1, "status": "fresh" if caption else "not_observed"})
        if isinstance(fields.get("objects"), list):  # a box, as every grounded object has, for a test that names none
            fields["objects"] = [{"bbox": [0, 0, 1, 1], **o} if isinstance(o, dict) else o for o in fields["objects"]]
        return self.add(vision_log.CAPTIONS, ts_start=ts, cam=cam, yaw_deg=0.0, caption=caption, **fields)

    def face(
        self,
        ts: float,
        cam: str = "cam0",
        name: str | None = None,
        sim: float | None = None,
        *,
        bbox: list[float] | None = None,
        quality: float = 0.5,
        cluster: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Adds a `face.match` as see-actions writes it: one look at a tracked head, its box as shares of the
        frame, and who it is; its `detection` is its own id. Returns it."""
        m = self.add(
            segment_log.FACE_MATCHES,
            ts_start=ts,
            cam=cam,
            bbox=bbox or [0.4, 0.2, 0.5, 0.4],
            quality=quality,
            det=0.8,
            name=name,
            sim=sim,
            cluster=cluster,
            **fields,
        )
        m["detection"] = m["id"]
        return m

    def visit(
        self, name: str, start: float, end: float, cam: str = "cam0", best_sim: float = 0.6, samples: int = 1
    ) -> dict[str, Any]:
        """Adds a `face.visit` (in the segment store)."""
        return self.add(
            segment_log.FACE_VISITS,
            ts_start=start,
            ts_end=end,
            name=name,
            cam=cam,
            best_sim=best_sim,
            samples=samples,
            shots=[],
        )

    def frame_evidence(self, cam: str, ts: float) -> vision_log.FrameEvidence | None:
        image = self.frames.get((cam, ts))
        if image is None:
            return None
        height, width = image.shape[:2]
        return vision_log.FrameEvidence(
            image,
            round(ts * 1e9),
            {
                "capturedAtUs": round(ts * 1e6),
                "width": width,
                "height": height,
                "flip": False,
                "profileStatus": "absent",
                "profile": None,
                "nativeGeometry": {"width": width, "height": height},
            },
        )

    def frame(self, cam: str, ts: float, within_s: float = 0.001) -> Any:
        """The frame a test put in `frames` for `cam` nearest `ts`, within `within_s`."""
        near = [(abs(t - ts), t) for c, t in self.frames if c == cam and abs(t - ts) <= within_s]
        return self.frames[(cam, min(near)[1])] if near else None

    def newest(self, cam: str, within_s: float = 5.0) -> tuple[Any, float, str] | None:
        """The frame a test set for the camera with `frame`, as `VisionLog.newest` gives one."""
        return self.newest_frames.get(cam)

    def video_spans(self, cam: str, start: float, end: float) -> list[list[float]]:
        """The stretches a test put in `video` for `cam`, clipped to [start, end)."""
        return [[max(a, start), min(b, end)] for c, a, b in self.video if c == cam and b > start and a < end]

    def clip(self, cam: str, start: float, end: float, fps: float) -> list[Any]:
        """One `frame` per tick over [start, end), as `VisionLog.clip` gives them; `clips` counts the calls."""
        self.clips = getattr(self, "clips", 0) + 1
        n = max(0, round((end - start) * fps))
        return [self.frame(cam, start + i / fps, 0.5 / fps) for i in range(n)]

    def aged_out(self, cam: str, start: float, end: float) -> list[list[float]]:
        """The stretches a test put in `aged` for `cam`, clipped to [start, end)."""
        return [[max(a, start), min(b, end)] for c, a, b in self.aged if c == cam and b > start and a < end]

    def video_days(self) -> list[str]:
        """The local days the stretches in `video` touch."""
        import datetime as dt

        self.video_days_asks = getattr(self, "video_days_asks", 0) + 1
        return sorted({dt.date.fromtimestamp(t).isoformat() for _, a, b in self.video for t in (a, b)})

    def video_cameras(self) -> list[str]:
        return sorted({c for c, _, _ in self.video} | {c for c, _ in self.frames})

    def has_video(self, start: float, end: float) -> bool:
        """Whether a stretch in `video` overlaps [start, end); `video_asks` counts the calls."""
        self.video_asks = getattr(self, "video_asks", 0) + 1
        return any(b > start and a < end for _, a, b in self.video)

    def frame_now(self, cam: str, image: Any, ts: float, frame_id: str = "frame-1") -> None:
        """Sets the camera's newest frame (an image), for `newest`."""
        self.newest_frames[cam] = (image, ts, frame_id)

    def range(
        self, stream: str, start: float, end: float, cameras: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        self.reads += 1
        found = [
            dict(r)
            for r in self.records.get(stream, [])
            if r["ts_start"] < end
            and max(r["ts_end"], r["ts_start"]) >= start
            and (cameras is None or r.get("cam") in cameras)
        ]
        return sorted(found, key=lambda r: (r["ts_start"], r["id"]))

    def table(
        self, stream: str, start: float, end: float, names: Sequence[str], cameras: Sequence[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """As `segment_log.Log.table`: the window's columns by camera, typed as the store types them."""
        spec = segment_log.STREAMS[stream]
        kinds = dict(spec.schema)
        by_cam: dict[str, list[dict[str, Any]]] = {}
        for r in self.range(stream, start, end, cameras):
            by_cam.setdefault(r["cam"], []).append(spec.to_row(r))
        return {
            cam: {
                name: [row[name] for row in rows]
                if kinds[name].startswith("str") or kinds[name].endswith("[]")
                else np.asarray([row[name] for row in rows], dtype=_DTYPES[kinds[name].split("[")[0]])
                for name in names
            }
            for cam, rows in by_cam.items()
        }

    def covered(
        self, stream: str, start: float, end: float, cameras: Sequence[str] | None = None, **how: Any
    ) -> dict[str, segment_log.Covered]:
        return segment_log.Log.covered(self, stream, start, end, cameras, **how)  # ty: ignore[invalid-argument-type]  # the real join, over this log's table

    def captions(self, start: float, end: float, cams: Iterable[str] = ()) -> list[dict[str, Any]]:
        wanted = set(cams)
        return [
            vision_log.as_v1_row(r)
            for r in self.range(vision_log.CAPTIONS, start, end)
            if not wanted or r.get("cam") in wanted
        ]

    def scene(self, now: float, limit: int = 30) -> list[dict[str, Any]]:
        window = vision_log.VisionLog.SCENE_WINDOW_S
        return vision_log.scene_entries(self.range(vision_log.CAPTIONS, max(0.0, now - window), now + 1.0), limit)

    def latest(self, now: float) -> dict[str, dict[str, Any]]:
        window = vision_log.VisionLog.SCENE_WINDOW_S
        return vision_log.latest_by_camera(self.range(vision_log.CAPTIONS, max(0.0, now - window), now + 1.0))

    def close(self) -> None:
        pass
