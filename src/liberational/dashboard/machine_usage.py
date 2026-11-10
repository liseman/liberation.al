"""What each service uses over time, for the Resource monitor's stacked usage chart (#2245, #2277, GET
/stats/history?since=).

A `Sampler` thread started with the dashboard hands a /stats snapshot (server.Stats.snapshot) to `History.record`
every `STEP_S`, which keeps one sample a step for the last `WINDOW_S`, whether or not anyone is looking; /stats polls
record too. A sample costs about 0.4% of a core on devbox (36 samples at 5 s, niced): the process scan reads each process's command line once
(proc_table.py) and the GPUs come from NVML in-process (nvml.py), where a full scan and three nvidia-smi calls a sample
cost 4.3% (#2263). A gap in the samples is a gap in the chart, never a zero.

A sample is {"ts", "cpu": {group: % of the whole machine}, "mem": {group: bytes}, "gpu": [{"index", "used", "util",
"apps": {group: bytes}}]}. A group is a service (the role /stats names a process by). An Ollama runner counts toward the
Ollama server that started it, so the main Ollama and Ask's own `ask-ollama` stay apart. `OTHER` is the rest of the
machine: its total less every listed process, never below zero.

GPU memory is per device: `nvidia-smi` lists each GPU and each compute process by GPU uuid, read only when a sample is
due. Without nvidia-smi (a Mac) `gpu` is empty and `gpus` says so. Per-process GPU utilization isn't sampled (it needs
`nvidia-smi pmon`, which takes a second a call); each GPU's overall utilization is, as `util`.
"""

import logging
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any

STEP_S = 5.0  # one sample at most this often
WINDOW_S = 1800.0  # kept this long
OTHER = "other"  # the rest of the machine
RUNNER = "ollama runner"  # /stats' role for an Ollama runner process
NVIDIA_TIMEOUT_S = 1.0
LOG = logging.getLogger(__name__)
MIB = 1024 * 1024

Gpus = list[dict[str, Any]]  # [{"index", "uuid", "name", "total", "used", "util", "apps": {pid: bytes}}]


def nvidia() -> Gpus | None:
    """Each NVIDIA GPU with its memory, utilization, and compute processes' memory by pid: from NVML in this process
    (liberational/nvml.py), else nvidia-smi; None without either, or when both fail."""
    from liberational import nvml

    found = nvml.gpus()
    return found if found is not None else _smi()


def _smi() -> Gpus | None:
    """nvidia-smi's answer: two calls, each a new process (30 to 60 ms of CPU on devbox)."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None

    def query(*args: str) -> list[list[str]]:
        reply = subprocess.run(
            [executable, *args, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=NVIDIA_TIMEOUT_S,
            check=True,
        )
        return [[cell.strip() for cell in line.split(",")] for line in reply.stdout.splitlines() if line.strip()]

    try:
        devices = query("--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu")
        apps: dict[str, dict[int, int]] = {row[1]: {} for row in devices}
        for uuid, pid, used in query("--query-compute-apps=gpu_uuid,pid,used_gpu_memory"):
            if uuid in apps and used.isdigit():
                apps[uuid][int(pid)] = apps[uuid].get(int(pid), 0) + int(used) * MIB
        gpus = [
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "total": int(total) * MIB,
                "used": int(used) * MIB,
                "util": int(util) if util.isdigit() else None,
                "apps": apps[uuid],
            }
            for index, uuid, name, used, total, util in devices
        ]
    except OSError, IndexError, ValueError, subprocess.SubprocessError:  # missing, failed, or a reply it can't read
        return None
    return sorted(gpus, key=lambda g: g["index"])


def groups(procs: Iterable[Mapping[str, Any]], parent: Callable[[int], int | None]) -> dict[int, str]:
    """Each listed process's group by pid: its role, except that an Ollama runner joins the server that started it
    (the plain "ollama" server when its parent isn't listed)."""
    roles = {int(p["pid"]): str(p["role"]) for p in procs if p.get("pid") is not None}
    out: dict[int, str] = {}
    for pid, role in roles.items():
        if role == RUNNER:
            owner = roles.get(ppid) if (ppid := parent(pid)) is not None else None
            role = owner if owner and owner != RUNNER else "ollama"
        out[pid] = role
    return out


def sample(snapshot: Mapping[str, Any], group: Mapping[int, str], gpus: Gpus | None, ts: float) -> dict[str, Any]:
    """One chart sample from a /stats snapshot, its processes' groups, and the GPUs (see the module's docs)."""
    cores = snapshot.get("cpu_count") or 1
    cpu: dict[str, float] = {}
    mem: dict[str, int] = {}
    for p in snapshot.get("procs") or []:
        g = group.get(int(p["pid"]), str(p["role"]))
        cpu[g] = cpu.get(g, 0.0) + float(p.get("cpu") or 0.0) / cores
        mem[g] = mem.get(g, 0) + int(p.get("rss") or 0)
    cpu[OTHER] = max(0.0, float(snapshot.get("cpu") or 0.0) - sum(cpu.values()))
    mem[OTHER] = max(0, int(snapshot.get("mem_used") or 0) - sum(mem.values()))
    gpu = []
    for g in gpus or []:
        apps: dict[str, int] = {}
        for pid, used in g["apps"].items():
            name = group.get(pid, OTHER)
            apps[name] = apps.get(name, 0) + used
        apps[OTHER] = apps.get(OTHER, 0) + max(0, g["used"] - sum(g["apps"].values()))
        gpu.append({"index": g["index"], "used": g["used"], "util": g["util"], "apps": apps})
    return {
        "ts": ts,
        "cpu": {k: round(v, 2) for k, v in cpu.items()},
        "mem": mem,
        "gpu": gpu,
    }


class History:
    """The kept samples, and what the chart needs to draw them (see the module's docs)."""

    def __init__(
        self,
        gpu_reader: Callable[[], Gpus | None] = nvidia,
        parent: Callable[[int], int | None] | None = None,
        step_s: float = STEP_S,
        window_s: float = WINDOW_S,
    ) -> None:
        self.lock = threading.Lock()
        self.samples: deque[dict[str, Any]] = deque(maxlen=int(window_s / step_s) + 1)
        self.step_s, self.window_s = step_s, window_s
        self.gpu_reader = gpu_reader
        self.parent = parent or _parent
        self.gpus: list[dict[str, Any]] = []  # [{"index", "name", "total"}], from the newest sample
        self.cores = 1
        self.mem_total = 0

    def record(self, snapshot: Mapping[str, Any], now: float | None = None) -> None:
        """Keep a sample of this snapshot if one is due. A snapshot without machine totals (no psutil) is skipped."""
        now = time.time() if now is None else now
        if "cpu" not in snapshot:
            return
        with self.lock:
            if self.samples and now - self.samples[-1]["ts"] < self.step_s:
                return
            gpus = self.gpu_reader()
            taken = sample(snapshot, groups(snapshot.get("procs") or [], self.parent), gpus, now)
            self.samples.append(taken)
            while self.samples and self.samples[0]["ts"] < now - self.window_s:
                self.samples.popleft()
            self.gpus = [{"index": g["index"], "name": g["name"], "total": g["total"]} for g in gpus or []]
            self.cores = int(snapshot.get("cpu_count") or 1)
            self.mem_total = int(snapshot.get("mem_total") or 0)

    def view(self, since: float = 0.0) -> dict[str, Any]:
        """The samples newer than `since`, with the totals each resource stacks up to."""
        with self.lock:
            return {
                "step": self.step_s,
                "window": self.window_s,
                "cpu_count": self.cores,
                "mem_total": self.mem_total,
                "gpus": list(self.gpus),
                "samples": [s for s in self.samples if s["ts"] > since],
            }


class Sampler:
    """Samples `snapshot()` into `history` every `step_s` on a daemon thread of its own, from `start` for the dashboard's
    whole life (#2277), so the chart always has the last half hour whether or not anyone is looking."""

    def __init__(self, history: History, snapshot: Callable[[], Mapping[str, Any]], step_s: float = STEP_S) -> None:
        self.history, self.snapshot, self.step_s = history, snapshot, step_s
        self.thread: threading.Thread | None = None
        self.stopping = threading.Event()

    def start(self) -> None:
        if self.thread is None or not self.thread.is_alive():
            self.stopping.clear()
            self.thread = threading.Thread(target=self._run, name="usage-sampler", daemon=True)
            self.thread.start()

    def stop(self) -> None:
        self.stopping.set()

    def _run(self) -> None:
        while not self.stopping.is_set():
            try:
                self.history.record(self.snapshot())
            except Exception as error:  # one bad sample mustn't stop the sampling
                LOG.warning("usage sampler: a sample failed: %s", error)
            self.stopping.wait(self.step_s)


def _parent(pid: int) -> int | None:
    try:
        import psutil

        return psutil.Process(pid).ppid()
    except Exception:  # gone, not allowed, or no psutil: the runner counts toward the plain Ollama server
        return None
