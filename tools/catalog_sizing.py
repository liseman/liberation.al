"""Estimate what the Continuum catalog holds, per stream, from a running broker (read-only).

The broker owns catalog.redb and redb refuses a second open, and a file copy taken while it runs is torn (its roots
are rewritten within a second). So this asks the broker instead: it lists the streams, queries a few evenly spaced
sample windows on each at the background band, and extrapolates rows and bytes a day from what the windows hold.
A stream with few payloads is read whole, so its numbers are exact.

Per payload it measures the header bytes and the inline payload bytes as the catalog stores them, counts parents
and subjects (each a row in PROV_PARENTS, PROV_CHILDREN, and SUBJECTS), and adds the fixed key and value sizes of
the index rows every payload has (TIME_INDEX, PAYLOAD_SEQ, STREAM_SEQ, and LOCATIONS keys), from
continuum-catalog/src/tables.rs. Raw video frames absorbed into chunks keep a 25-byte entry in their chunk's frame
index and, while they have children or subjects, a FRAMES and a CHUNK_FRAMES row; give a camera's frame rate with
--fps to have those estimated too.

It writes nothing to the broker and never opens the catalog file:

    .venv/bin/python tools/catalog_sizing.py --since 2026-10-31T00:10:00Z --windows 12 --window-s 120
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import pathlib
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

DAY_NS = 86_400 * 1_000_000_000

# Fixed index rows per payload (keys + values, bytes), from continuum-catalog/src/tables.rs.
TIME_INDEX = 16 + 16  # (stream u32, start u64, seq u32) -> id
PAYLOAD_SEQ = 16 + 8  # id -> commit sequence
STREAM_SEQ = 12 + 16  # (stream u32, seq u64) -> id
PAYLOAD_KEY = 16  # PAYLOADS key
LOCATION_KEY = 17  # (id, tier u8)
EDGE = 32  # PROV_CHILDREN (parent, child) or SUBJECTS (entity, id); value empty
PARENTS_ROW = 16  # PROV_PARENTS key, plus 16 per parent in its value
CHUNK_FRAME_ENTRY = 25  # start u64, offset u64, length u64, cut u8
FRAMES_ROW = 16 + 16 + 4 + 4  # frame id -> (chunk id, index u32, subject count), before subject ids
CHUNK_FRAMES_ROW = 32  # (chunk id, frame id) -> ()
# redb stores two u32 end offsets per leaf entry.
ENTRY_OVERHEAD = 8


@dataclass
class Sample:
    """What the sample windows on one stream held."""

    name: str
    kind: str
    exact: bool = False
    seconds: float = 0.0
    payloads: int = 0
    header_bytes: int = 0
    inline_bytes: int = 0
    inline_locator_bytes: int = 0
    other_locator_bytes: int = 0
    parents: int = 0
    subjects: int = 0
    rows: int = 0
    locators: Counter[str] = field(default_factory=Counter)
    keys: Counter[str] = field(default_factory=Counter)
    span_ns: int = 0

    def index_bytes(self) -> int:
        """Key and fixed-index bytes the sample's payloads cost, beyond the header and the inline payload."""
        per = TIME_INDEX + PAYLOAD_SEQ + STREAM_SEQ + PAYLOAD_KEY + LOCATION_KEY
        rows = 5 * self.payloads + self.parents + self.subjects + (self.payloads if self.parents else 0)
        edges = (self.parents + self.subjects) * EDGE + self.parents * 16 + (PARENTS_ROW if self.parents else 0)
        return per * self.payloads + edges + rows * ENTRY_OVERHEAD

    def total_bytes(self) -> int:
        return self.header_bytes + self.inline_locator_bytes + self.other_locator_bytes + self.index_bytes()

    def scale(self, recorded_s: float) -> float:
        """Multiplier from the sample to a day of recording."""
        if self.exact:
            return 86_400 / recorded_s if recorded_s else 0.0
        return 86_400 / self.seconds if self.seconds else 0.0


def parse_time(text: str) -> int:
    return int(dt.datetime.fromisoformat(text).timestamp() * 1e9)


def json_keys(raw: bytes) -> list[str]:
    """The top-level keys (and the keys of a list of objects under them) of a JSON payload, for describing its shape."""
    try:
        value = json.loads(raw)
    except ValueError:
        return ["(not JSON)"]
    if not isinstance(value, dict):
        return [f"({type(value).__name__})"]
    keys = []
    for k, v in value.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            keys.append(f"{k}[{','.join(sorted(v[0]))}]")
        else:
            keys.append(k)
    return keys


async def measure(client: Any, name: str, start_ns: int, end_ns: int, sample: Sample) -> None:
    from continuum_py import decode_header

    result = await client.query(name, start_ns, end_ns, all=True, band="P3Background")
    sample.seconds += (end_ns - start_ns) / 1e9
    for item in result["items"]:
        header = item["header"]
        h = decode_header(header)
        if h["start_ns"] < start_ns:
            continue  # began before the window: counted in the window it starts in
        sample.payloads += 1
        sample.header_bytes += len(header)
        sample.parents += len(h.get("parents") or [])
        sample.subjects += len(h.get("subjects") or [])
        sample.span_ns += max(0, h["end_ns"] - h["start_ns"])
        loc = item["locator"]
        sample.locators[loc["kind"] if loc else "None"] += 1
        if loc and loc["kind"] == "Inline":
            raw = loc.get("bytes") or b""
            sample.inline_bytes += len(raw)
            # The FlatBuffer locator around the bytes: root, union, vector length, and padding.
            sample.inline_locator_bytes += len(raw) + 40
            if len(sample.keys) < 64:
                sample.keys.update(json_keys(raw))
        elif loc:
            sample.other_locator_bytes += 64 + len(loc.get("uri") or loc.get("arena") or "")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from continuum_py import Client

    client = await Client.connect(args.socket, client_name="catalog-sizing", request_timeout_s=60.0)
    try:
        # Fifteen minutes back, so streams their producers trim within minutes (worker checkpoints)
        # count only what they keep.
        now = time.time_ns() - 900 * 1_000_000_000
        since = parse_time(args.since)
        recorded_s = (now - since) / 1e9
        streams = await client.list_streams()
        ids = {s["stream_id"]: s["name"] for s in streams}
        width = int(args.window_s * 1e9)
        step = (now - since - width) // max(1, args.windows - 1)
        windows = [(since + i * step, since + i * step + width) for i in range(args.windows)]
        samples: dict[str, Sample] = {}
        for s in streams:
            name = s["name"]
            sample = Sample(name, s["kind"])
            for a, b in windows:
                await measure(client, name, a, b, sample)
            estimate = sample.payloads * recorded_s / max(1.0, sample.seconds)
            if estimate <= args.exact_below:
                sample = Sample(name, s["kind"], exact=True)
                await measure(client, name, since, now, sample)
            samples[name] = sample
            print(f"{name}: {sample.payloads} payloads sampled", file=sys.stderr)
        return {
            "recorded_s": recorded_s,
            "since": since,
            "now": now,
            "windows": len(windows),
            "window_s": args.window_s,
            "ids": ids,
            "samples": samples,
        }
    finally:
        await client.close("measured")


def report(out: dict[str, Any], rate_hz: dict[str, float]) -> str:
    samples: dict[str, Sample] = out["samples"]
    recorded_s: float = out["recorded_s"]
    lines = [
        (
            f"Recorded {recorded_s / 3600:.1f} h; {out['windows']} windows of {out['window_s']} s per stream "
            "(streams marked exact were read whole)."
        ),
        "",
        "| stream | sampled | payloads/day | header B | inline B | parents | subjects | MiB/day | of which index |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for s in samples.values():
        f = s.scale(recorded_s)
        n = max(1, s.payloads)
        rows.append((s.total_bytes() * f, s, f, n))
    rows.sort(key=lambda r: -r[0])
    total = 0.0
    for per_day, s, f, n in rows:
        total += per_day
        lines.append(
            f"| {s.name} | {'exact' if s.exact else 'sampled'} | {s.payloads * f:,.0f} | {s.header_bytes / n:.0f} | "
            f"{s.inline_bytes / n:.0f} | {s.parents / n:.2f} | {s.subjects / n:.2f} | {per_day / 2**20:,.1f} | "
            f"{s.index_bytes() * f / 2**20:,.1f} |"
        )
    lines.append(f"| total | | | | | | | {total / 2**20:,.1f} | |")
    for name, hz in rate_hz.items():
        frames = hz * 86_400
        lines.append(
            f"\n{name}: chunk frame indexes about {frames * CHUNK_FRAME_ENTRY / 2**20:,.1f} MiB/day at {hz:g} fps; "
            f"FRAMES + CHUNK_FRAMES for frames with children about "
            f"{frames * (FRAMES_ROW + CHUNK_FRAMES_ROW + 2 * ENTRY_OVERHEAD) / 2**20:,.1f} MiB/day if every frame has one."
        )
    lines.append("\n## Payload shapes (top-level JSON keys seen)\n")
    for _, s, _, _ in rows:
        if s.keys:
            lines.append(f"- {s.name}: {', '.join(k for k, _ in s.keys.most_common(24))}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", default="data/state/continuum/broker.sock", type=pathlib.Path)
    parser.add_argument("--since", required=True, help="when recording started, ISO 8601 UTC")
    parser.add_argument("--windows", type=int, default=12, help="sample windows per stream, evenly spaced")
    parser.add_argument("--window-s", type=float, default=120.0, help="each window's length in seconds")
    parser.add_argument("--exact-below", type=int, default=20_000, help="read a stream whole below this many payloads")
    parser.add_argument("--fps", action="append", default=[], help="NAME=HZ for a camera's frame rate")
    args = parser.parse_args(argv)
    rate_hz = {k: float(v) for k, v in (x.split("=", 1) for x in args.fps)}
    print(report(asyncio.run(run(args)), rate_hz))


if __name__ == "__main__":
    main()
