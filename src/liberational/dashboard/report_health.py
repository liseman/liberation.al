"""System health for the Reports tab: storage against the plan, memory and GPU use, and service uptime.

Most of this has no history of its own on disk, so the dashboard keeps one: `Sampler` takes a sample every SAMPLE_S
while the dashboard runs and adds it to `state/reports.sqlite3` (memory/reports.py's Store). A sample holds memory
use (psutil), GPU use (nvidia-smi, when there is one), and which installed services are up (the service manager's
states). Hours with no sample (the dashboard was down, or before it started sampling) are gaps in the charts, never
zeros.

Storage is measured, not sampled: every STORAGE_S the sampler adds up the Tier 1 and Tier 2 chunk files by the local
day their recording started (each chunk's name holds its stream and start time, as docs/STORAGE.md measured them),
and keeps that with the catalog's and the search vectors' sizes. Each recorded day is drawn against the plan in
docs/STORAGE.md's footprint under the video window: PLAN_FRESH_GB a day (video and audio) while a day is inside the
window (`video_window.days`), then PLAN_AGED_GB (audio only) once its video has been deleted.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from liberational import config as cfg
from liberational import logs, services
from liberational.memory import reports

LOG = logs.get("reports")

SAMPLE_S = 300.0
STORAGE_S = 1800.0
HOURS_SHOWN = 14 * 24  # the hourly charts show the last two weeks
UPTIME_DAYS = 30
KEEP_S = 400 * 86400.0  # samples older than this are dropped
# docs/STORAGE.md, "How much disk": six cameras' video (about 38 GB a day) for the window, and
# one microphone's Opus audio (about 0.76 GB a day), which is kept.
PLAN_FRESH_GB = 38.76
PLAN_AGED_GB = 0.76
CHUNK = re.compile(r"^\d+-(\d+)-\d+")
GB = 1e9


def gpu() -> dict[str, float] | None:
    """Utilization and memory summed over the GPUs nvidia-smi sees, or None without one."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        rows = [[float(x) for x in line.split(",")] for line in out.splitlines() if line.strip()]
    except OSError, ValueError, subprocess.SubprocessError:
        return None
    if not rows:
        return None
    return {
        "util": round(statistics.fmean(r[0] for r in rows), 1),
        "mem_used_mb": sum(r[1] for r in rows),
        "mem_total_mb": sum(r[2] for r in rows),
    }


def take() -> dict[str, Any]:
    """One sample of this machine now."""
    import psutil

    vm = psutil.virtual_memory()
    up = {name: state in services.UP_STATES for name, state in services.states().items()}
    return {
        "mem_pct": vm.percent,
        "mem_used_gb": round((vm.total - vm.available) / GB, 2),
        "gpu": gpu(),
        "services": up,
    }


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _chunk_entries(tier: Path) -> list[os.DirEntry[str]]:
    """The chunk files of a tier directory, all in `streams/<stream>/<hour>/`."""
    out: list[os.DirEntry[str]] = []
    dirs: list[Path] = []
    try:
        with os.scandir(tier / "streams") as streams:
            for stream in streams:
                if stream.is_dir():
                    with os.scandir(stream.path) as hours:
                        dirs.extend(Path(h.path) for h in hours if h.is_dir())
    except OSError:
        pass
    for d in dirs:
        try:
            with os.scandir(d) as entries:
                out.extend(e for e in entries if e.is_file())
        except OSError:
            continue
    return out


def storage(continuum: Path) -> dict[str, Any]:
    """Bytes held per recorded day ({day: {"tier1", "tier2"}}), and the catalog's and vectors' sizes."""
    days: dict[str, dict[str, int]] = {}
    for tier in ("tier", "tier2"):
        for e in _chunk_entries(continuum / tier):
            m = CHUNK.match(e.name)
            if not m:
                continue
            try:
                size = e.stat().st_size
            except OSError:
                continue
            day = reports.day_of(int(m.group(1)) / 1e9)
            slot = days.setdefault(day, {"tier1": 0, "tier2": 0})
            slot["tier1" if tier == "tier" else "tier2"] += size
    return {
        "at": time.time(),
        "days": days,
        "catalog": _size(continuum / "catalog.redb") if (continuum / "catalog.redb").exists() else 0,
        "vectors": _size(continuum / "vector") if (continuum / "vector").exists() else 0,
    }


def plan_gb(day: str, today: str) -> float:
    """What the plan holds for a recorded day of this age: video and audio inside the video window, audio after it.
    With the window off, video is kept, so every day holds both."""
    age = len(reports.days_between(day, today)) - 1 if day <= today else 0
    if not cfg.get("video_window.enabled"):
        return PLAN_FRESH_GB
    return PLAN_FRESH_GB if age < int(cfg.get("video_window.days")) else PLAN_AGED_GB


def _hourly(samples: Sequence[tuple[float, Mapping[str, Any]]], now: float) -> list[dict[str, Any]]:
    """The last HOURS_SHOWN clock hours: mean memory use, peak GPU memory, and mean GPU use.
    An hour with no sample is all None."""
    end = now - now % 3600 + 3600
    start = end - HOURS_SHOWN * 3600
    buckets: dict[int, list[Mapping[str, Any]]] = {}
    for ts, body in samples:
        if start <= ts < end:
            buckets.setdefault(int((ts - start) // 3600), []).append(body)
    out = []
    for i in range(HOURS_SHOWN):
        rows = buckets.get(i, [])
        gpus = [r["gpu"] for r in rows if r.get("gpu")]
        out.append(
            {
                "t": start + i * 3600,
                "mem_pct": round(statistics.fmean(r["mem_pct"] for r in rows), 1) if rows else None,
                "gpu_mem_pct": round(max(100 * g["mem_used_mb"] / g["mem_total_mb"] for g in gpus), 1)
                if gpus
                else None,
                "gpu_util": round(statistics.fmean(g["util"] for g in gpus), 1) if gpus else None,
            }
        )
    return out


def view(store: reports.Store, now: float | None = None) -> dict[str, Any]:
    """The System health report's figures."""
    now = time.time() if now is None else now
    samples = store.samples(now - UPTIME_DAYS * 86400)
    today = reports.day_of(now)
    up: dict[str, list[bool]] = {}
    for _, body in samples:
        for name, is_up in (body.get("services") or {}).items():
            up.setdefault(name, []).append(bool(is_up))
    latest = samples[-1][1] if samples else {}
    held = store.meta("storage")
    stored = None
    if isinstance(held, Mapping):
        days = sorted(held["days"])
        rows = [
            {
                "day": d,
                "gb": round((held["days"][d]["tier1"] + held["days"][d]["tier2"]) / GB, 2),
                "tier2_gb": round(held["days"][d]["tier2"] / GB, 2),
                "plan_gb": plan_gb(d, today),
            }
            for d in days
        ]
        stored = {
            "at": held["at"],
            "days": rows,
            "media_gb": round(sum(r["gb"] for r in rows), 1),
            "plan_gb": round(sum(r["plan_gb"] for r in rows), 1),
            "catalog_gb": round(held["catalog"] / GB, 1),
            "vectors_gb": round(held["vectors"] / GB, 2),
        }
    return {
        "report": "health",
        "first": reports.day_of(samples[0][0]) if samples else today,
        "last": today,
        "pending": [],
        "refreshing": [],
        "sampled_from": samples[0][0] if samples else None,
        "sample_s": SAMPLE_S,
        "hours": _hourly(samples, now),
        "services": sorted(
            (
                {
                    "name": name,
                    "uptime_pct": round(100 * sum(v) / len(v), 1),
                    "samples": len(v),
                    "up_now": bool((latest.get("services") or {}).get(name)),
                }
                for name, v in up.items()
            ),
            key=lambda s: (s["uptime_pct"], s["name"]),
        ),
        "storage": stored,
    }


def service_history(state: Path, names: Sequence[str], start: float, end: float) -> services.HistoryView:
    """Read only existing health samples. Each observation covers at most SAMPLE_S; longer gaps stay unknown."""
    path = state / "reports.sqlite3"
    unreadable = False
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2) as db:
            rows = db.execute(
                "SELECT ts, body FROM samples WHERE ts >= ? AND ts <= ? ORDER BY ts", (start, end)
            ).fetchall()
        samples = [(float(ts), json.loads(body)) for ts, body in rows]
        if any(not isinstance(body, dict) or not isinstance(body.get("services", {}), dict) for _, body in samples):
            raise ValueError("Invalid health samples")
    except OSError, ValueError, sqlite3.Error:
        samples = []
        unreadable = path.exists()
    services_out: list[services.ServiceHistory] = []
    available = False
    for name in sorted(names):
        spans: list[services.HistorySpan] = []
        previous = start
        observed = [
            (ts, body["services"][name]) for ts, body in samples if isinstance(body.get("services", {}).get(name), bool)
        ]
        for i, (ts, value) in enumerate(observed):
            available = True
            services._history_span(spans, previous, ts, "unknown")
            next_at = min(end, ts + SAMPLE_S, observed[i + 1][0] if i + 1 < len(observed) else end)
            services._history_span(spans, ts, next_at, "up" if value else "stopped")
            previous = next_at
        services._history_span(spans, previous, end, "unknown")
        services_out.append({"name": name, "spans": spans, "events": []})
    return {
        "start": start,
        "end": end,
        "source": "sampled" if available else "unavailable",
        "available": available,
        "sample_s": SAMPLE_S if available else None,
        "truncated": False,
        "note": "Exact start, restart, and exit details are unavailable."
        if available
        else "Service history could not be read."
        if unreadable
        else "No service history is available for this window.",
        "services": services_out,
    }


class Sampler:
    """The background thread that takes a sample every SAMPLE_S and measures storage every STORAGE_S."""

    def __init__(
        self,
        store: reports.Store,
        continuum: Path,
        *,
        sample: Callable[[], dict[str, Any]] = take,
        measure: Callable[[Path], dict[str, Any]] = storage,
    ) -> None:
        self.store, self.continuum, self.sample, self.measure = store, continuum, sample, measure
        self.stop = threading.Event()
        self._measured = 0.0

    def once(self, now: float) -> None:
        try:
            self.store.add_sample(now, self.sample())
        except Exception as e:
            LOG.warning(f"health sample not taken: {type(e).__name__}: {e}")
        if now - self._measured >= STORAGE_S:
            self._measured = now
            try:
                self.store.set_meta("storage", self.measure(self.continuum))
            except Exception as e:
                LOG.warning(f"storage not measured: {type(e).__name__}: {e}")
            self.store.drop_samples(now - KEEP_S)

    def run(self) -> None:
        while not self.stop.is_set():
            self.once(time.time())
            self.stop.wait(SAMPLE_S)


def start(state: Path) -> threading.Thread:
    """Start sampling for the Reports tab (the dashboard calls this once)."""
    sampler = Sampler(reports.at(state).store, state / "continuum")
    thread = threading.Thread(target=sampler.run, daemon=True, name="reports-health")
    thread.start()
    return thread
