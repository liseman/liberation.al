"""This machine's numbers for the Resource monitor and Settings: CPU, memory, the pack's daemons, and the models
Ollama has loaded (moved out of the server, #61).

GET /stats          one snapshot (`Stats.snapshot`), which also feeds the usage history
GET /stats/history  per-service CPU and memory over the last half hour (dashboard/machine_usage.py); `SAMPLER`
                    keeps it going from dashboard start
"""

import itertools
import json
import math
import os
import re
import threading
import time
import urllib.request
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import model_runtime, ollama_health, services
from liberational.dashboard import machine_usage, routes
from liberational.dashboard.context import LOG, SSE_CLIENTS

try:
    import psutil
except ImportError:
    psutil = None

if TYPE_CHECKING:
    from liberational.dashboard.server import H


# role by module; a daemon runs as `pack-<command>` or `python -m liberational.<module>`
DAEMONS = {
    "listen.worker": ("listen", "pack-listen"),
    "dashboard.server": ("dashboard", "pack-dashboard"),
}


def daemon_role(cmd: list[str]) -> str | None:
    for mod, (role, script) in DAEMONS.items():
        if any(a == "liberational." + mod or a == script or a.endswith("/" + script) for a in cmd):
            return role
    return None


SYSTEM_S = 300.0  # how often /stats checks for missing system packages again


# CPU is measured "since the last call", so two pages polling a moment apart read nearly nothing: a /stats within this
# long of the last one gets that one again, and every measurement spans at least this long.
STATS_MIN_S = 1.0


OLLAMA_BLOB = re.compile(r"sha256-[0-9a-f]{64}")  # the file an Ollama model's weights are in, as a runner loads it


def runner_blob(cmd: list[str]) -> str | None:
    """The weights an Ollama runner process loaded ("sha256-<hex>"), from its --model argument, or None."""
    for flag, value in itertools.pairwise(cmd):
        if flag in ("--model", "-m") and (m := OLLAMA_BLOB.search(value)):
            return m[0]
    return None


def ollama_blob(name: str) -> str | None:
    """The weights a model loads from ("sha256-<hex>"): Ollama's /api/show names the file on its modelfile's FROM
    line. None when it doesn't say."""
    ask = urllib.request.Request(
        cfg.ollama_url() + "/api/show",
        data=json.dumps({"model": name}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(ask, timeout=1.5) as r:
        modelfile = json.loads(r.read()).get("modelfile") or ""
    for line in modelfile.splitlines():
        if line.startswith("FROM ") and (m := OLLAMA_BLOB.search(line)):
            return m[0]
    return None


def cpu_busy(before: Any, after: Any) -> float:
    """The busy share, in percent (one decimal), between two psutil.cpu_times() readings, as psutil.cpu_percent works
    it out: guest time is already inside user time on Linux, and idle and I/O wait aren't busy. No time passed: 0."""

    def split(t: Any) -> tuple[float, float]:
        f = t._asdict()
        total = sum(f.values()) - f.get("guest", 0.0) - f.get("guest_nice", 0.0)
        return total, total - f.get("idle", 0.0) - f.get("iowait", 0.0)

    total0, busy0 = split(before)
    total1, busy1 = split(after)
    if total1 <= total0:
        return 0.0
    return round(min(100.0, max(0.0, 100 * (busy1 - busy0) / (total1 - total0))), 1)


class Stats:
    """psutil per-daemon CPU/RSS (cpu_percent is 'since last call', so poll at a steady 2 s) + Ollama /api/ps (cached 2 s).
    Each service's processes are named for it (services.processes()), and each Ollama runner for the model it runs."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model_scan_known = False
        self.procs: dict[int, tuple[str, Any, str | None]] = {}  # pid: (role, psutil.Process, a runner's weights)
        self.scan_t = 0
        self.ollama = {"models": [], "error": None}
        self.ollama_t = 0
        self.blobs: dict[str, str | None] = {}  # a loaded model's digest: its weights, from /api/show, asked once
        self.model_of: dict[str, str] = {}  # weights: the loaded model's name
        self.last: dict[str, Any] | None = None  # the last snapshot, given again within STATS_MIN_S
        self.system: dict[str, Any] = {}
        self.system_t = 0.0
        # The machine's CPU times at the last snapshot, for the busy share since. Kept here, not psutil.cpu_percent's own:
        # psutil 7 keeps that per thread, and each /stats request is a new server thread, so it read 0%.
        self.cpu_t: Any = psutil.cpu_times() if psutil else None

    def cpu_since(self) -> float:
        """The whole machine's busy share, in percent, since the last snapshot. Called under the lock."""
        if psutil is None:
            return 0.0
        now = psutil.cpu_times()
        before, self.cpu_t = self.cpu_t, now
        return cpu_busy(before, now) if before is not None else 0.0

    def scan(self) -> None:
        if psutil is None or time.time() - self.scan_t < 10:
            return
        self.scan_t = time.time()
        self.model_scan_known = True
        try:
            owner = {pid: name for name, pids in services.processes().items() for pid in pids}
        except (Exception, SystemExit) as e:  # pack.toml names a service that doesn't exist: the other roles still show
            LOG.warning(f"stats: services' processes unreadable: {e}")
            owner = {}
            self.model_scan_known = False
        from liberational import proc_table  # psutil's, and psutil is optional here

        found = {}
        # Every process, each one's command line read once (proc_table); only the ones named here are measured.
        for entry in proc_table.table().values():
            cmd, name = list(entry.cmdline), entry.name
            role = owner.get(entry.pid) or daemon_role(cmd)
            if role is None and name == "ollama":
                role = "ollama runner" if "runner" in cmd else "ollama"  # macOS runs `ollama runner --model ...`
            elif role is None and (name == "llama-server" or (cmd and cmd[0].endswith("/llama-server"))):
                role = "ollama runner"
            if not role:
                continue
            blob = runner_blob(cmd) if role == "ollama runner" else None
            known = self.procs.get(entry.pid)
            if known is not None:
                found[entry.pid] = (role, known[1], blob)
                continue
            try:
                proc = psutil.Process(entry.pid)
                proc.cpu_percent(None)  # its first reading, so the next one is a rate
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                self.model_scan_known = False
                continue
            found[entry.pid] = (role, proc, blob)
        self.procs = found

    def ollama_ps(self) -> dict[str, Any]:
        if time.time() - self.ollama_t < 2:
            return self.ollama
        self.ollama_t = time.time()
        try:
            with urllib.request.urlopen(cfg.ollama_url() + "/api/ps", timeout=1.5) as r:
                d = json.loads(r.read())
            self.ollama = {
                "models": [
                    {
                        "name": m.get("name"),
                        "purpose": model_runtime.ollama_purpose(str(m.get("name") or "")),
                        "size": m.get("size"),
                        "size_vram": m.get("size_vram"),
                        "ctx": m.get("context_length"),
                        "params": (m.get("details") or {}).get("parameter_size"),
                        "expires_at": m.get("expires_at"),
                    }
                    for m in d.get("models", [])
                ],
                "error": None,
            }
            self.model_of = self.weights(d.get("models", []))
        except Exception as e:
            self.ollama = {"models": [], "error": str(e)[:120]}
        # Up, and the configured models it lacks, in words the owner can act on. /api/tags is cached in
        # ollama_health for 5 minutes while Ollama answers, and /api/ps failures force a fresh check.
        health = ollama_health.status(refresh=bool(self.ollama["error"]))
        self.ollama |= {"up": health["up"], "missing": health["missing"], "message": health["message"]}
        return self.ollama

    def weights(self, models: list[dict[str, Any]]) -> dict[str, str]:
        """{weights: model name} for the models /api/ps lists, so each runner process can say which model it runs. A
        model's weights are asked of /api/show once per digest (a pull that changes the model changes its digest)."""
        out: dict[str, str] = {}
        for m in models:
            name = str(m.get("name") or "")
            digest = str(m.get("digest") or name)
            if not name:
                continue
            if digest not in self.blobs:
                try:
                    self.blobs[digest] = ollama_blob(name)
                except Exception:  # an Ollama that doesn't say: its runners go by their pids
                    self.blobs[digest] = None
            if blob := self.blobs[digest]:
                out[blob] = name
        return out

    def system_status(self) -> dict[str, Any]:
        """The system packages missing here (config/system-packages.toml) and the commands that install them. Checked
        every SYSTEM_S, since finding a library can run ldconfig or a compiler."""
        if self.system and time.time() - self.system_t < SYSTEM_S:
            return self.system
        self.system_t = time.time()
        from liberational import services, system_packages

        try:
            # Only what the services selected for this machine use: a package for a service it doesn't run, or an
            # optional one, would show here forever.
            self.system = system_packages.status(services=list(services.selected()))
        except (Exception, SystemExit) as e:  # a bad declaration or [services] list mustn't break /stats
            self.system = {"missing": [], "commands": [], "error": str(e)[:200]}
        return self.system

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            if self.last is not None and time.time() - self.last["ts"] < STATS_MIN_S:
                return self.last
            out = {
                "ts": time.time(),
                "procs": [],
                "clients": SSE_CLIENTS[0],
                "ollama": self.ollama_ps(),
                "system": self.system_status(),
                "model_runtime": {"processes": [], "configured": [], "errors": []},
            }
            if psutil is None:
                out["model_runtime"] = model_runtime.snapshot(ollama=out["ollama"])
                out["error"] = "psutil not installed"
                return out
            self.scan()
            out["model_runtime"] = model_runtime.snapshot(
                {pid: role for pid, (role, _process, _blob) in self.procs.items()} if self.model_scan_known else None,
                out["ollama"],
            )
            vm = psutil.virtual_memory()
            out.update(
                {
                    "cpu": self.cpu_since(),
                    "cpu_count": psutil.cpu_count(),
                    "mem_total": vm.total,
                    "mem_used": vm.total - vm.available,
                    "swap_used": psutil.swap_memory().used,
                    "mem_pct": vm.percent,
                    "load": list(os.getloadavg()),
                }
            )
            for pid, (role, p, blob) in list(self.procs.items()):
                try:
                    out["procs"].append(
                        {
                            "pid": pid,
                            "role": role,
                            "model": self.model_of.get(blob) if blob else None,
                            "cpu": p.cpu_percent(None),
                            "rss": p.memory_info().rss,
                            "threads": p.num_threads(),
                            "uptime": time.time() - p.create_time(),
                        }
                    )
                except psutil.NoSuchProcess, psutil.AccessDenied:
                    self.procs.pop(pid, None)
                    self.scan_t = 0
            out["procs"].sort(key=lambda d: (d["role"] != "ambient", d["role"] != "vision", d["role"]))
            self.last = out
            return out


STATS = Stats()


USAGE = machine_usage.History()  # This machine's per-service usage over time


SAMPLER = machine_usage.Sampler(USAGE, STATS.snapshot)  # the history's samples from dashboard start, always


@routes.get("/stats")
def _get_stats(h: H) -> None:
    snapshot = STATS.snapshot()
    USAGE.record(snapshot)
    return h._json(snapshot)


# SAMPLER keeps the history going from dashboard start
@routes.get("/stats/history")
def _get_stats_history(h: H) -> None:
    try:
        since = float(h._query().get("since") or 0)
    except ValueError:
        since = 0.0
    return h._json(USAGE.view(since if math.isfinite(since) else 0.0))
