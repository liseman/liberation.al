"""Measure the heart rate this system would read from a recorded stretch of one camera's video.

This runs the live path's own maths over recorded video: the same skin regions (`vitals.face_regions`, from the
pose rows see-actions recorded for that stretch), the same patch averages, and the same `vitals.Trace` (POS with
CHROM as the agreement check, the in-band SNR, and the confidence). Nothing here is a second implementation, so the
rate it prints is what Live would have shown had you been sitting there.

It reads no models and decodes no more than the video itself: the face geometry comes from `pose.track`, the rows
see-actions already wrote, and the colour from the recorded H.264 through Continuum's store (`continuum_py`). Both must cover the
stretch, so it is a check on a camera see-actions has been following (the close, full-rate one), not on an arbitrary
file.

    # the six-minute check, with a watch or a finger oximeter (see the module's "The six-minute check" below)
    .venv/bin/python tools/measure_vitals.py --camera cam0 --from "2026-02-03T19:40" --minutes 6

    # a stretch by epoch seconds, one line a reading and a JSON summary
    .venv/bin/python tools/measure_vitals.py --camera cam0 --start 1770147600 --end 1770147960 --json out.json

The output is a line a reading (the time, the track, and either a rate or why there is none), then a table a
minute, then a summary: how many readings were confident, their mean and spread, and how far the minute-by-minute
means are from each other. Compare a minute's rate with what the watch or oximeter said at that minute.

The six-minute check: sit still facing the desk camera, sitting as you usually do when you read, and look at your
watch (or the oximeter's screen) once a minute for six minutes, writing down each minute's number and the
wall-clock time you read it. Keep your head and hands out of the way of the camera's view of your face, and don't
talk. Then run the command above for that stretch, and compare its minute table with your six numbers: a still,
well-lit close camera should be within a few beats a minute, and a reading that is off by more than ten is a
reading the tool should mostly have called no reading at all. Run it once at the camera's current capture quality
and once after lowering it (`video.encoding.cq` in `config/devices.toml` for that camera; the capture service
reloads its manifest) to see what compression costs on this camera.

Nothing here writes to the repository, the store, or a running install: it only reads.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import segment_log, vision_log, vitals
from liberational.see.actions import POSE_SCHEMA

NS = 1_000_000_000
SLICE_S = 10.0  # how much video is decoded at a time: a 30 fps 960x540 slice is about half a gigabyte
JOINT_EDGE = 10_000.0  # pose joints are stored times this, normalized to the frame (see.actions.SCALE)


def parse_time(text: str) -> float:
    """`text` as epoch seconds: a number, or a local time ("2026-02-03T19:40", "2026-02-03 19:40:00")."""
    try:
        return float(text)
    except ValueError:
        pass
    for pattern in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return time.mktime(time.strptime(text, pattern))
        except ValueError:
            continue
    raise SystemExit(f"{text!r} isn't a time: give epoch seconds, or like 2026-02-03T19:40")


def poses(root: Path, camera: str, start: float, end: float) -> dict[str, np.ndarray]:
    """The camera's recorded pose rows in [start, end), as the arrays `read_segments` gives: ts_ns, track_id, and
    the quantized joints."""
    from continuum_py import read_segments

    got = read_segments(root, segment_log.POSE_TRACKS, POSE_SCHEMA, round(start * NS), round(end * NS), [camera])
    if not len(got["ts_ns"]):
        return {}
    return {
        "ts_ns": np.asarray(got["ts_ns"], np.int64),
        "track_id": np.asarray(got["track_id"], np.int64),
        "joints": np.asarray(got["joints"], np.int16).reshape(len(got["ts_ns"]), -1, 3).astype(np.float64),
    }


def nearest(index: np.ndarray, target: int) -> int | None:
    """The position of `index` (sorted) nearest `target`, or None when it holds nothing."""
    if not len(index):
        return None
    at = int(np.searchsorted(index, target))
    best = min((p for p in (at - 1, at) if 0 <= p < len(index)), key=lambda p: abs(int(index[p]) - target))
    return best


def joints_at(
    rows: dict[str, np.ndarray], order: np.ndarray, tick_ns: int, width: int, height: int, within_ns: int = NS // 10
) -> list[tuple[int, np.ndarray]]:
    """Each tracked person's joints (frame pixels) in the recorded frame nearest `tick_ns`, within `within_ns`:
    (track_id, K x 3) per person. Half a tick is what a caller passes, so one recorded frame serves one tick."""
    at = nearest(order, tick_ns)
    if at is None or abs(int(order[at]) - tick_ns) > within_ns:
        return []
    wanted = int(order[at])
    out: list[tuple[int, np.ndarray]] = []
    for i in np.flatnonzero(rows["ts_ns"] == wanted):
        joints = rows["joints"][i] / JOINT_EDGE
        joints[:, 0] *= width
        joints[:, 1] *= height
        out.append((int(rows["track_id"][i]), joints))
    return out


def measure(camera: str, start: float, end: float, *, fps: float, root: Path) -> list[dict[str, Any]]:
    """Every reading over [start, end) from the camera's recorded video and pose rows, oldest first."""
    from continuum_py import Client

    from liberational import continuum_link

    rows = poses(root, camera, start, end)
    if not rows:
        raise SystemExit(
            f"no pose.track rows for {camera} between {start} and {end}: see-actions has to have followed this "
            "camera over that stretch (it writes them); try a window it was running in"
        )
    order = np.sort(rows["ts_ns"])
    hearts: dict[int, vitals.Trace] = {}
    readings: list[dict[str, Any]] = []

    step = max(1, round(fps))  # a reading a track a second of video, as see/hearts.py takes one on live frames

    def read(tick: int) -> None:
        """Take every track's reading at `tick`, keeping the ones whose window ended within the last second."""
        for track_id, trace in sorted(hearts.items()):
            if trace.last_ns < tick - NS:  # no sample in the last second: nothing new to read
                continue
            reading = trace.reading()
            if reading is None:  # the window does not hold enough of a trace yet
                continue
            readings.append(
                {
                    "at": trace.last_ns / NS,
                    "track_id": track_id,
                    "bpm": None if not reading.trusted else round(reading.bpm, 1),
                    "confidence": round(reading.confidence, 3),
                    "snr_db": round(reading.snr_db, 1),
                    "coverage": round(reading.coverage, 3),
                    "face_px": reading.face_px,
                    "fps": round(reading.fps, 2),
                }
            )

    async def walk(client: Any) -> None:
        at = start
        while at < end - 0.5:
            upto = min(end, at + SLICE_S)
            frames = await vision_log.clip_frames(client, camera, at, upto, fps)
            for i, image in enumerate(frames):
                if image is None:  # a picture the recording doesn't have: nothing sampled
                    continue
                tick = round((at + i / fps) * NS)
                height, width = int(image.shape[0]), int(image.shape[1])
                half = NS // (2 * max(1, round(fps)))
                for track_id, joints in joints_at(rows, order, tick, width, height, half):
                    regions = vitals.face_regions(joints, width, height)
                    if regions is None:
                        continue
                    trace = hearts.get(track_id)
                    if trace is None:
                        trace = hearts[track_id] = vitals.Trace()
                    x1, y1, x2, y2 = regions.box
                    trace.add(tick, vitals.region_means(image[y1:y2, x1:x2], regions.patches), regions.side)
                if i % step == 0:
                    read(tick)
            at = upto

    async def run() -> None:
        client = await Client.connect(str(continuum_link.socket_path()), client_name="measure-vitals")
        try:
            await walk(client)
        finally:
            await continuum_link.close_quietly(client)

    asyncio.run(run())
    return readings


def minutes(readings: list[dict[str, Any]]) -> dict[int, list[float]]:
    """The confident rates by whole minute from the first reading."""
    if not readings:
        return {}
    began = readings[0]["at"]
    out: dict[int, list[float]] = defaultdict(list)
    for r in readings:
        if r["bpm"] is not None:
            out[int((r["at"] - began) // 60)].append(float(r["bpm"]))
    return dict(sorted(out.items()))


def stamp(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch).strftime("%H:%M:%S")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="the camera's name in config/devices.toml, like cam0")
    parser.add_argument("--from", dest="since", default=None, help="when the stretch starts: epoch seconds or a time")
    parser.add_argument("--start", type=float, default=None, help="the same, as epoch seconds")
    parser.add_argument("--end", type=float, default=None, help="when it ends, as epoch seconds (default: now)")
    parser.add_argument("--minutes", type=float, default=None, help="how long the stretch is, from its start")
    parser.add_argument("--fps", type=float, default=30.0, help="the rate to sample the video at (default 30)")
    parser.add_argument("--json", type=Path, default=None, help="write the readings and the summary here")
    args = parser.parse_args(argv)

    since = args.start if args.start is not None else parse_time(args.since) if args.since else None
    if since is None:
        raise SystemExit("say when the stretch starts: --from <time> or --start <epoch seconds>")
    end = args.end if args.end is not None else (since + args.minutes * 60.0 if args.minutes else time.time())
    if end <= since:
        raise SystemExit("the stretch ends before it starts")
    readings = measure(args.camera, since, end, fps=args.fps, root=segment_log.root(cfg.path("paths.state")))

    for r in readings:
        rate = f"{r['bpm']:.0f} bpm" if r["bpm"] is not None else "no reading"
        print(
            f"{stamp(r['at'])}  track {r['track_id']:<6} {rate:<10} confidence {r['confidence']:.2f}"
            f"  snr {r['snr_db']:.1f} dB  coverage {r['coverage']:.2f}  face {r['face_px']} px"
            f"  {r['fps']:.1f} fps"
        )

    by_minute = minutes(readings)
    print("\nminute  confident  mean  min  max")
    for minute, rates in by_minute.items():
        print(f"{minute:>6}  {len(rates):>9}  {np.mean(rates):4.0f}  {min(rates):3.0f}  {max(rates):3.0f}")
    trusted = [r["bpm"] for r in readings if r["bpm"] is not None]
    summary = {
        "camera": args.camera,
        "start": since,
        "end": end,
        "readings": len(readings),
        "confident": len(trusted),
        "mean_bpm": round(float(np.mean(trusted)), 1) if trusted else None,
        "spread_bpm": round(float(np.std(trusted)), 1) if trusted else None,
        "minutes": {str(k): [round(float(np.mean(v)), 1), len(v)] for k, v in by_minute.items()},
        "by_reading": readings,
    }
    print(
        f"\n{len(trusted)} of {len(readings)} readings were confident"
        + (f", mean {summary['mean_bpm']:.1f} bpm, spread {summary['spread_bpm']:.1f}" if trusted else "")
        + ". Compare each minute's mean with your watch."
    )
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"written to {args.json}")


if __name__ == "__main__":
    sys.exit(main())
