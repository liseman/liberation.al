"""Actual in-process model loads, reported only during an explicitly activated service lifetime.

Each process atomically replaces its own file under state/model_loads. Readers validate its PID and creation
time before trusting it. Files describe successful loads, never configured intent; memory belongs to the process,
not each model sharing it. No library import or ordinary loader test activates reporting.
"""

import atexit
import contextlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
import weakref
from collections.abc import Generator
from pathlib import Path
from typing import Any, Self

import psutil

from liberational import config as cfg
from liberational import json_state, logs

LOG = logs.get("model-runtime")
_lock = threading.RLock()
_service: str | None = None
_depth = 0
_models: dict[str, dict[str, Any]] = {}
_started = 0.0
_written: dict[Path, tuple[int, int, float, str]] = {}
_writer = uuid.uuid4().hex
MAX_STATUS_FILES = 256
MAX_STATUS_BYTES = 256 * 1024
MAX_LIVE_REPORTERS = 256  # priority/remembered file reads, separate from the reserved discovery quota
MAX_STALE_PRUNES = 64  # deletions per five-second cleanup window; discovery remains independently bounded
STALE_PRUNE_INTERVAL_S = 5.0
_prune_at = -math.inf
_prune_left = 0
_reader_directory: Path | None = None
_reader_entries: Generator[Path] | None = None
_verified_paths: set[Path] = set()

_gpu_cache: tuple[float, dict[int, int] | None] = (0.0, None)


def _directory() -> Path:
    return cfg.path("paths.state") / "model_loads"


def _path() -> Path:
    return _directory() / f"{os.getpid()}.json"


def _discard_owned(paths: list[Path]) -> None:
    """Invalidate only files this reporter wrote, never a replacement from another PID incarnation/writer."""
    for path in paths:
        expected = _written.get(path)
        if expected is None:
            continue
        try:
            current = path.stat()
            record = _record(path)
            if (
                (current.st_dev, current.st_ino) == expected[:2]
                and record["created"] == expected[2]
                and record.get("writer") == expected[3]
            ):
                latest = path.stat()
                if (latest.st_dev, latest.st_ino) == expected[:2]:
                    path.unlink(missing_ok=True)
            _written.pop(path, None)
        except FileNotFoundError, ValueError:
            _written.pop(path, None)  # Missing/corrupt evidence already cannot claim a verified load.
        except OSError as error:
            LOG.warning(f"could not invalidate model load status: {error}")


def _exit_reporting() -> None:
    """Clean only this reporter's exact owned files, including externally held model reports."""
    with _lock:
        _discard_owned(list(_written))


atexit.register(_exit_reporting)


def _write() -> None:
    """Replace only this process's evidence; a reporting failure must not stop its model."""
    if _service is None:
        return
    try:
        holders: dict[int, list[dict[str, Any]]] = {os.getpid(): []}
        for model in _models.values():
            holders.setdefault(model["_pid"], []).append({k: v for k, v in model.items() if not k.startswith("_")})
        current: set[Path] = set()
        for pid, models in holders.items():
            path = _directory() / f"{pid}.json"
            body = {
                "version": 1,
                "writer": _writer,
                "pid": pid,
                "created": next(
                    (m["_created"] for m in _models.values() if m["_pid"] == pid), psutil.Process().create_time()
                ),
                "service": _service,
                "since": _started,
                "models": models,
            }
            json_state.write(path, body)
            status = path.stat()
            _written[path] = (status.st_dev, status.st_ino, body["created"], _writer)
            current.add(path)
        _discard_owned([path for path in _written if path not in current])
    except (OSError, psutil.Error) as error:
        LOG.warning(f"could not report model loads: {error}")
        _discard_owned(list(_written))


def _forked() -> None:
    global _lock, _models, _started, _written, _service, _depth, _writer
    _lock = threading.RLock()
    _models = {}
    _written = {}
    _writer = uuid.uuid4().hex
    _service, _depth = None, 0
    _started = time.time()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forked)


class reporting(contextlib.ContextDecorator):
    """Enable reports for this real service and loader threads; fork children activate their own lifetime."""

    def __init__(self, service: str) -> None:
        self.service = service

    def __enter__(self) -> Self:
        global _service, _depth, _started, _writer
        with _lock:
            if _service is not None and _service != self.service:
                raise RuntimeError("one process cannot report as two services")
            if _depth == 0:
                _service, _started = self.service, time.time()
                _writer = uuid.uuid4().hex
                _models.clear()
            _depth += 1
            _write()
        return self

    def __exit__(self, *exc: object) -> None:
        global _service, _depth
        with _lock:
            _depth -= 1
            if _depth == 0:
                _models.clear()
                _discard_owned(list(_written))
                _service = None


class LoadToken:
    """One successful load; close on explicit unload, or attach an owner for collection cleanup."""

    def __init__(self, ident: str) -> None:
        self.ident, self.pid = ident, os.getpid()
        self.finalizer: weakref.finalize | None = None

    def close(self) -> None:
        with _lock:
            if self.pid == os.getpid() and _models.pop(self.ident, None) is not None:
                _write()
            if self.finalizer is not None:
                self.finalizer.detach()


def _device(device: str) -> str:
    if device in ("CPUExecutionProvider", "cpu"):
        return "CPU"
    if device in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        return "GPU"
    if device.startswith("cuda"):
        return "GPU" + (f" {device.split(':', 1)[1]}" if ":" in device else "")
    if device in ("mps", "metal"):
        return "Apple GPU"
    if device == "CoreMLExecutionProvider":
        return "Apple GPU" if cfg.platform_key() == "darwin-arm64" else "Core ML"
    return device or "unknown"


def ort_device(session: Any) -> str:
    """Where an ONNX Runtime session really runs: its first bound provider, with the CUDA device it bound ("cuda:1").

    ONNX Runtime 1.2x logs "No registered plugin EP device found for 'CUDAExecutionProvider'" when it looks for a
    plugin build of the provider first; it then binds the built-in one, which `get_providers` reports."""
    provider = str(session.get_providers()[0])
    if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        index = session.get_provider_options().get(provider, {}).get("device_id")
        return f"cuda:{index}" if index not in (None, "") else provider
    return provider


def loaded(
    key: str, name: str, purpose: str, device: str, owner: object | None = None, *, external_pid: int | None = None
) -> LoadToken | None:
    """Report only after a successful load. Inactive library calls are deliberately inert."""
    with _lock:
        if _service is None:
            return None
        pid = external_pid if external_pid is not None else os.getpid()
        try:
            created = psutil.Process(pid).create_time()
        except psutil.Error as error:
            LOG.warning(f"could not identify loaded model process: {error}")
            return None
        ident = uuid.uuid4().hex
        key = key.removesuffix(".path").removesuffix(".repo")
        _models[ident] = {
            "key": key,
            "name": name,
            "purpose": purpose,
            "device": _device(device),
            "since": time.time(),
            "_pid": pid,
            "_created": created,
        }
        token = LoadToken(ident)
        if owner is not None:
            try:
                token.finalizer = weakref.finalize(owner, token.close)
            except TypeError:
                pass  # Some native model objects do not support weak references; explicit close still works.
        _write()
        return token


_PURPOSES = {
    "asr_parakeet": "speech to text",
    "asr_whisper": "speech to text",
    "asr_deep": "speech to text",
    "tts_kokoro": "voice",
    "tts_clone": "voice",
    "tts_clone_cuda": "voice",
    "object_segmentation": "object outlines",
    "vlm": "camera captions",
    "speaker_embedding": "speaker recognition",
    "vad_silero": "speech detection",
    "person_detector": "people detection",
    "pose_live": "body poses",
    "animal_verifier": "animal check",
    "animal_embedding": "pet embeddings",
}


def configured() -> list[dict[str, str]]:
    """The current platform's configured application models; Ollama is listed by its own API."""
    out: list[dict[str, str]] = []
    platform = cfg.platform_key()
    models = cfg.get("models")
    for family in ("hf", "files"):
        for key, item in models.get(family, {}).items():
            if key not in _PURPOSES:
                continue
            if item.get("only_on") and platform not in item["only_on"] or platform in item.get("skip_on", []):
                continue
            if item.get("needs") == "cuda" and not shutil.which("nvidia-smi"):
                continue
            name = item.get("repo") if family == "hf" else Path(item.get("path", "")).name
            if not name:
                continue
            device = "Apple GPU" if platform == "darwin-arm64" and family == "hf" else "unknown"
            out.append({"key": f"models.{family}.{key}", "name": name, "purpose": _PURPOSES[key], "device": device})
    for family, role, purpose in (("insightface", "detector", "face detection"), ("adaface", "recognition", "faces")):
        name = models.get(family, {}).get(role)
        if name:
            out.append({"key": f"models.{family}.{role}", "name": name, "purpose": purpose, "device": "unknown"})
    for key, name in models.get("ollama", {}).items():
        if name:
            purpose = {
                "llm": "assistant replies",
                "big_llm": "profiles",
                "vlm": "camera captions",
                "embedding": "semantic search",
            }.get(key, key)
            out.append({"key": f"models.ollama.{key}", "name": name, "purpose": purpose, "device": "Ollama"})
    return out


def ollama_purpose(name: str) -> str:
    """Configured uses of a loaded Ollama model, including aliases sharing one model name."""
    purposes = [
        row["purpose"] for row in configured() if row["key"].startswith("models.ollama.") and row["name"] == name
    ]
    return ", ".join(dict.fromkeys(purposes)) or "Purpose not reported"


_SERVICES: dict[str, tuple[str, ...]] = {
    "asr_parakeet": ("listen", "ingest-analyze", "phone-bridge"),
    "asr_whisper": ("whisper",),
    "asr_deep": ("deep-transcribe",),
    "speaker_embedding": ("listen", "voice", "ingest-analyze", "clone-say"),
    "vad_silero": ("listen", "ingest-analyze"),
    "tts_kokoro": ("voice", "clone-tts", "clone-say", "phone-bridge"),
    "tts_clone": ("voice", "clone-tts", "clone-say", "phone-bridge"),
    "tts_clone_cuda": ("voice", "clone-tts", "clone-say", "phone-bridge"),
    "object_segmentation": ("dashboard", "objects"),
    "person_detector": ("see-actions", "dashboard"),
    "pose_live": ("see-actions", "dashboard"),
    "animal_embedding": ("see-actions", "see-captions"),
    "animal_verifier": ("see-actions", "see-captions"),
    "vlm": ("vlm", "see-captions"),
    "detector": ("see-actions", "dashboard"),
    "recognition": ("see-actions", "dashboard"),
}


def _gpu_memory() -> dict[int, int] | None:
    global _gpu_cache
    with _lock:
        now = time.monotonic()
        if now - _gpu_cache[0] < 2:
            return _gpu_cache[1]
        value = _query_gpu_memory()
        _gpu_cache = (now, value)
        return value


def _query_gpu_memory() -> dict[int, int] | None:
    """Each compute process's GPU memory, summed over devices: NVML in this process (nvml.py), else nvidia-smi."""
    from liberational import nvml

    found = nvml.gpus()
    if found is not None:
        result: dict[int, int] = {}
        for gpu in found:
            for pid, used in gpu["apps"].items():
                result[pid] = result.get(pid, 0) + used
        return result
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        reply = subprocess.run(
            [executable, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        )
        result: dict[int, int] = {}
        for line in reply.stdout.splitlines():
            pid, amount = line.split(",")
            result[int(pid.strip())] = result.get(int(pid.strip()), 0) + int(amount.strip()) * 1024 * 1024
        return result
    except OSError, ValueError, subprocess.SubprocessError:
        return None


def _valid_time(value: object) -> bool:
    try:
        return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _record(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        data = source.read(MAX_STATUS_BYTES + 1)
    if len(data) > MAX_STATUS_BYTES:
        raise ValueError("oversized model status")
    value = json.loads(data)
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("invalid model status")
    pid, created = value.get("pid"), value.get("created")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1 or path.stem != str(pid):
        raise ValueError("invalid process identity")
    if not _valid_time(created) or not _valid_time(value.get("since")):
        raise ValueError("invalid process time")
    if (
        not isinstance(value.get("service"), str)
        or not value["service"]
        or not isinstance(value.get("models"), list)
        or len(value["models"]) > 256
    ):
        raise ValueError("invalid model status")
    for model in value["models"]:
        if (
            not isinstance(model, dict)
            or any(not isinstance(model.get(k), str) or not model[k] for k in ("key", "name", "purpose", "device"))
            or not _valid_time(model.get("since"))
        ):
            raise ValueError("invalid loaded model")
    return value


def _prune_stale(path: Path, record: dict[str, Any], observed: os.stat_result) -> None:
    """Remove abandoned evidence only if its file and process lifetime are still the ones just checked."""
    global _prune_at, _prune_left
    with _lock:
        now = time.monotonic()
        if now - _prune_at >= STALE_PRUNE_INTERVAL_S:
            _prune_at, _prune_left = now, MAX_STALE_PRUNES
        if _prune_left <= 0:
            return
        _prune_left -= 1

    def identity(status: os.stat_result) -> tuple[int, int, int, int]:
        return status.st_dev, status.st_ino, status.st_mtime_ns, status.st_size

    try:
        if identity(path.stat()) != identity(observed):
            return  # another reporter replaced or updated it
        try:
            process = psutil.Process(record["pid"])
            if (
                abs(process.create_time() - record["created"]) <= 0.001
                and process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                return
        except psutil.NoSuchProcess:
            pass
        if identity(path.stat()) == identity(observed):
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    except (OSError, psutil.Error) as error:
        LOG.debug("could not prune stale model load status: %s", error)


def _directory_entries(directory: Path) -> Generator[Path]:
    # scandir is lazy: even a directory full of abandoned temporary files has a bounded per-poll discovery cost.
    with os.scandir(directory) as entries:
        for entry in entries:
            yield Path(entry.path)


def _select_paths(directory: Path, active: dict[int, str] | None, errors: list[str]) -> list[Path]:
    """Read known live reporters first, but always reserve independent discovery work for unlisted children.

    Only paths are remembered. Their files and PID lifetimes are re-read every poll, so unload/removal is immediate.
    The open discovery cursor advances across polls instead of repeatedly reading a stale directory prefix.
    """
    global _reader_directory, _reader_entries
    with _lock:
        if directory != _reader_directory:
            if _reader_entries is not None:
                _reader_entries.close()
            _reader_directory, _reader_entries = directory, None
            _verified_paths.clear()
        active_pids = sorted(active or {})
        if len(active_pids) > MAX_LIVE_REPORTERS:
            errors.append("Too many active processes to verify all model reporters")
        paths = [directory / f"{pid}.json" for pid in active_pids[:MAX_LIVE_REPORTERS]]
        paths.extend(sorted(_verified_paths))
        if _reader_entries is None:
            _reader_entries = _directory_entries(directory)
        for _ in range(MAX_STATUS_FILES):
            entry = next(_reader_entries, None)
            if entry is None:
                _reader_entries.close()
                _reader_entries = None
                break
            if entry.suffix == ".json":
                paths.append(entry)
        else:
            errors.append("Model reporter discovery is continuing on the next poll")
        return list(dict.fromkeys(paths))


def snapshot(active_processes: dict[int, str] | None = None, ollama: dict[str, Any] | None = None) -> dict[str, Any]:
    """Process-grouped verified loads and configured intent, with unknown evidence kept explicit."""
    processes: list[dict[str, Any]] = []
    errors: list[str] = []
    directory = _directory()
    try:
        paths = _select_paths(directory, active_processes, errors)
    except FileNotFoundError:
        paths = []
    except OSError:
        paths = []
        errors.append("Model load status is unavailable")
    verified: set[Path] = set()
    gpu = _gpu_memory() if paths else None
    for path in paths:
        try:
            observed = path.stat()
            record = _record(path)
            process = psutil.Process(record["pid"])
            if (
                abs(process.create_time() - record["created"]) > 0.001
                or not process.is_running()
                or process.status() == psutil.STATUS_ZOMBIE
            ):
                _prune_stale(path, record, observed)
                continue
            verified.add(path)
            try:
                rss = process.memory_info().rss
            except psutil.AccessDenied:
                rss = None
            processes.append(
                {
                    "pid": record["pid"],
                    "service": record["service"],
                    "since": record["since"],
                    "rss_bytes": rss,
                    "gpu_bytes": gpu.get(record["pid"], 0) if gpu is not None else None,
                    "models": record["models"],
                }
            )
        except FileNotFoundError:
            continue
        except psutil.NoSuchProcess:
            _prune_stale(path, record, observed)
        except OSError, ValueError, TypeError, psutil.Error:
            errors.append(f"Model load status could not be verified ({path.stem})")
    with _lock:
        if directory == _reader_directory:
            _verified_paths.clear()
            _verified_paths.update(sorted(verified)[:MAX_LIVE_REPORTERS])
            if len(verified) > MAX_LIVE_REPORTERS:
                errors.append("Too many live model reporters to retain all priority paths")
    return aggregate(processes, configured(), _service_roots(active_processes), ollama, errors)


def _service_roots(active: dict[int, str] | None) -> dict[int, str] | None:
    """The active processes less each one whose parent is the same service: a service's multiprocessing helpers (the
    resource tracker, a spawn pool's workers) never report, and would otherwise make every model it serves unknown.
    A helper that loads a model opens its own reporting lifetime, and its file is still read."""
    if active is None:
        return None
    roots: dict[int, str] = {}
    for pid, role in active.items():
        try:
            parent = psutil.Process(pid).ppid()
        except psutil.Error:
            parent = None
        if parent is None or active.get(parent) != role:
            roots[pid] = role
    return roots


def aggregate(
    records: list[dict[str, Any]],
    catalog: list[dict[str, str]],
    active_processes: dict[int, str] | None,
    ollama: dict[str, Any] | None,
    errors: list[str],
) -> dict[str, Any]:
    """Pure presentation aggregation of verified process facts; memory is counted once per process.

    Empty valid reporters prove a service has no current load. Missing evidence on an active service means unknown.
    Multiple live instances of one model share a row until their last token closes; different devices remain separate.
    """
    reporters = {record["pid"] for record in records}
    by_pid: dict[int, dict[str, Any]] = {}
    for record in records:
        process = by_pid.setdefault(record["pid"], {**record, "models": []})
        by_model = {(m["key"], m["name"], m["device"]): m for m in process["models"]}
        for model in record["models"]:
            identity = (model["key"], model["name"], model["device"])
            previous = by_model.get(identity)
            if previous is None or model["since"] < previous["since"]:
                by_model[identity] = dict(model)
        process["models"] = list(by_model.values())
    processes = [p for p in by_pid.values() if p["models"]]
    loaded_names = {(model["key"], model["name"]) for process in processes for model in process["models"]}
    configured_rows = []
    ollama_names = {model["name"] for model in (ollama or {}).get("models", [])}
    for model in catalog:
        if (
            (model["key"], model["name"]) in loaded_names
            or model["key"].startswith("models.ollama.")
            and model["name"] in ollama_names
        ):
            continue
        services = _SERVICES.get(model["key"].split(".")[-1], ())
        unknown = (
            bool(errors)
            or active_processes is None
            or any(role in services and pid not in reporters for pid, role in (active_processes or {}).items())
        )
        if model["key"].startswith("models.ollama."):
            unknown = bool((ollama or {}).get("error")) or ollama is None
        configured_rows.append({**model, "state": "unknown" if unknown else "unloaded"})
    return {
        "processes": sorted(processes, key=lambda p: (p["service"], p["pid"])),
        "configured": configured_rows,
        "errors": list(errors),
    }
