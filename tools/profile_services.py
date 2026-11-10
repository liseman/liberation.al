"""Passively survey Linux pack-service CPU use, without attaching to processes.

Run with the deployed checkout's Python: python -m tools.profile_services --output /tmp/service-cpu.json
Reports core equivalents (one core = 100% CPU), not stack/function attribution. No command lines are saved.
Processes born or exiting between polls have incomplete CPU coverage; short-lived processes can be missed entirely.
Offline: --profiles DIR [--survey saved.json] --output report.json; add --seconds 60 for a new survey.
"""

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liberational import services


@dataclass(frozen=True)
class Process:
    pid: int
    parent: int
    start: int
    user: int
    system: int
    service: str | None = None


def parse_stat(pid: int, text: str, service: str | None = None) -> Process:
    """The comm field may contain spaces and parentheses; numeric fields start after its last closing parenthesis."""
    fields = text[text.rindex(")") + 2 :].split()
    return Process(pid, int(fields[1]), int(fields[19]), int(fields[11]), int(fields[12]), service)


def snapshot(proc: Path, commands: tuple[Mapping[str, str], Mapping[str, str]]) -> dict[int, Process]:
    by_name, by_path = commands
    result: dict[int, Process] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode().split("\0")
            service = next(
                (name for arg in args if (name := by_name.get(arg.rsplit("/", 1)[-1]) or by_path.get(arg))), None
            )
            pid = int(entry.name)
            result[pid] = parse_stat(pid, (entry / "stat").read_text(), service)
        except OSError, ValueError, IndexError, UnicodeError:
            continue  # raced exit or unreadable process: never substitute zero counters
    return result


def assign_owners(rows: Mapping[int, Process]) -> dict[int, str]:
    """Each process belongs to its nearest named ancestor, including all descendant levels."""
    owners: dict[int, str] = {}
    for pid in rows:
        current, seen = pid, set()
        while current in rows and current not in seen:
            seen.add(current)
            row = rows[current]
            if row.service:
                owners[pid] = row.service
                break
            current = row.parent
    return owners


def survey(
    seconds: float,
    read: Callable[[], dict[int, Process]],
    *,
    ticks: int,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if not math.isfinite(seconds) or not 0 < seconds <= 3600:
        raise ValueError("seconds must be finite, positive, and at most 3600")
    if ticks <= 0:
        raise ValueError("clock ticks must be positive")
    started = clock()
    previous = read()
    previous_at = clock()
    initial = {(r.pid, r.start) for r in previous.values()}
    records: dict[tuple[int, int], dict[str, Any]] = {}
    owners = assign_owners(previous)
    while previous_at < started + seconds:
        sleep(min(1.0, started + seconds - previous_at))
        current = read()
        at = clock()
        current_owners = assign_owners(current)
        for pid, row in previous.items():
            key = (pid, row.start)
            name = owners.get(pid)
            if not name:
                continue
            rec = records.setdefault(
                key,
                {
                    "pid": pid,
                    "start_ticks": row.start,
                    "service": name,
                    "user_seconds": 0.0,
                    "system_seconds": 0.0,
                    "observed_seconds": 0.0,
                    "partial": key not in initial,
                },
            )
            next_row = current.get(pid)
            if next_row is None or next_row.start != row.start:
                rec["partial"] = True
                continue
            if next_row.user < row.user or next_row.system < row.system:
                rec["partial"] = True
                continue
            rec["user_seconds"] += (next_row.user - row.user) / ticks
            rec["system_seconds"] += (next_row.system - row.system) / ticks
            rec["observed_seconds"] += at - previous_at
        # Preserve ownership of orphaned children when their named root exits.
        owners = {
            pid: current_owners.get(pid) or owners[pid]
            for pid, row in current.items()
            if pid in current_owners or (pid in previous and row.start == previous[pid].start and pid in owners)
        }
        previous, previous_at = current, at
    duration = previous_at - started
    for pid, row in previous.items():
        if pid in owners:
            key = (pid, row.start)
            records.setdefault(
                key,
                {
                    "pid": pid,
                    "start_ticks": row.start,
                    "service": owners[pid],
                    "user_seconds": 0.0,
                    "system_seconds": 0.0,
                    "observed_seconds": 0.0,
                    "partial": True,
                },
            )
    grouped: dict[str, dict[str, Any]] = {}
    for rec in records.values():
        name = rec.pop("service")
        group = grouped.setdefault(name, {"service": name, "user_seconds": 0.0, "system_seconds": 0.0, "processes": []})
        group["user_seconds"] += rec["user_seconds"]
        group["system_seconds"] += rec["system_seconds"]
        group["processes"].append(rec)
    for group in grouped.values():
        group["cores"] = (group["user_seconds"] + group["system_seconds"]) / duration
    return {
        "schema_version": 1,
        "duration_seconds": duration,
        "clock_ticks_per_second": ticks,
        "logical_cpus": os.cpu_count(),
        "services": sorted(grouped.values(), key=lambda g: -g["cores"]),
        "limitations": [
            "CPU deltas exclude unobserved process lifetime tails and processes between polls.",
            "Counters include native threads but do not attribute functions or GPU work.",
        ],
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Create a private report; refuse symlinks and fix permissions before writing existing files."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(report, stream, indent=2)
        stream.write("\n")


def _number(value: Any, name: str, *, positive: bool = False, signed: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0 or not signed and value < 0:
        raise ValueError(f"{name} is outside its allowed range")
    return float(value)


def _wait_summary(row: Mapping[str, Any], hz: float, elapsed: float) -> dict[str, Any] | str:
    """Top wait sites and per-thread wakeup rates; captures from before #3180 have no wait table."""
    sites = row.get("wait_sites")
    if sites is None:
        return "unavailable"
    if not isinstance(sites, list):
        raise ValueError("invalid wait sites")
    ranked = []
    for site in sites:
        if not isinstance(site, dict) or not isinstance(site.get("stack"), list):
            raise ValueError("invalid wait site")
        if not all(isinstance(location, str) for location in site["stack"]):
            raise ValueError("invalid wait site stack")
        ranked.append(
            {
                "stack": site["stack"],
                "wakeups_per_s": _number(site.get("wakeups"), "wakeups") / elapsed,
                "nonrunning_cores": _number(site.get("nonrunning_ticks"), "nonrunning_ticks") / hz / elapsed,
                "threads": int(_number(site.get("threads"), "threads")),
            }
        )
    threads = []
    for thread in row.get("threads", []):
        if not isinstance(thread, dict):
            raise ValueError("invalid thread row")
        switches = _number(thread.get("voluntary_switches", 0), "voluntary_switches")
        if switches:
            threads.append(
                {
                    "native_tid": thread.get("native_tid"),
                    "wakeups_per_s": switches / elapsed,
                    "cores": _number(thread.get("cpu_ticks", 0), "cpu_ticks") / hz / elapsed,
                }
            )
    totals = {
        key: _number(row.get(key, 0), key) / elapsed
        for key in ("voluntary_switches", "wait_site_wakeups", "running_endpoint_wakeups", "native_wakeups")
    }
    return {
        "method": row.get("wait_method"),
        "wakeups_per_s": totals,
        "top_sites": sorted(ranked, key=lambda r: (-r["wakeups_per_s"], -r["nonrunning_cores"]))[:10],
        "top_threads": sorted(threads, key=lambda r: -r["wakeups_per_s"])[:10],
    }


def _site(stack: Sequence[str], frames: int = 4) -> str:
    """Innermost frames first, so the blocking call and its callers lead the cell."""
    shown = " < ".join(reversed(stack[-frames:]))
    return shown.replace("|", "\\|").replace("\n", " ")


def read_profiles(directory: Path) -> list[dict[str, Any]]:
    """Read final cumulative snapshots, selecting the newest capture per process identity, never summing snapshots."""
    if not directory.is_dir():
        raise ValueError("--profiles must name an existing directory")
    latest: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            row = json.loads(path.read_text())
            if not isinstance(row, dict) or type(row.get("version")) is not int or row["version"] != 1:
                raise ValueError("unsupported profile version; expected 1")
            if type(row.get("final")) is not bool:
                raise ValueError("missing boolean final field")
            if not row["final"]:
                continue
            identity = row.get("identity")
            if not isinstance(identity, dict):
                raise ValueError("missing identity")
            name = identity.get("service")
            if not isinstance(name, str) or not name or not isinstance(identity.get("capture_id"), str):
                raise ValueError("missing service/capture identity")
            for key in ("pid", "process_start_ticks", "started_at_ns"):
                _number(identity.get(key), key, positive=True)
            hz = _number(row.get("tick_hz"), "tick_hz", positive=True)
            elapsed = _number(row.get("elapsed_s"), "elapsed_s", positive=True)
            counts = {}
            for key in (
                "process_ticks",
                "observed_thread_ticks",
                "excluded_sampler_ticks",
                "python_attributed_ticks",
                "native_unattributed_ticks",
                "process_minus_observed_thread_ticks",
            ):
                counts[key] = _number(row.get(key), key, signed=key == "process_minus_observed_thread_ticks")
            # Earlier captures did not filter idle endpoints; retain their method instead of mixing rankings.
            counts["nonrunning_unattributed_ticks"] = _number(
                row.get("nonrunning_unattributed_ticks", 0), "nonrunning_unattributed_ticks"
            )
            stack_filter = row.get("stack_filter", "all-endpoint-states")
            if stack_filter not in {"all-endpoint-states", "running-only"}:
                raise ValueError("unknown stack filter")
            functions = row.get("functions")
            if not isinstance(functions, dict):
                raise ValueError("missing function estimates")
            ranked = []
            for location, estimate in functions.items():
                if not isinstance(location, str) or not isinstance(estimate, dict):
                    raise ValueError("invalid function estimate")
                own = _number(estimate.get("self_ticks_estimate"), "self_ticks_estimate")
                inclusive = _number(estimate.get("inclusive_ticks_estimate"), "inclusive_ticks_estimate")
                ranked.append(
                    {
                        "location": location,
                        "self_cores_estimate": own / hz / elapsed,
                        "inclusive_cores_estimate": inclusive / hz / elapsed,
                    }
                )
            result = {
                "service": name,
                "identity": identity,
                "elapsed_s": elapsed,
                "cores": counts["process_ticks"] / hz / elapsed,
                "accounting_ticks": counts,
                "tick_hz": hz,
                "top_functions": sorted(
                    ranked, key=lambda f: (-f["self_cores_estimate"], -f["inclusive_cores_estimate"], f["location"])
                )[:10],
                "native_functions": "unavailable",
                "stack_filter": stack_filter,
                "attribution_method": row.get("attribution_method", "thread-cpu-delta-endpoint-v1"),
                "method": row.get("method"),
                "limits": row.get("limits"),
                "waits": _wait_summary(row, hz, elapsed),
            }
            key = (name, identity["pid"], identity["process_start_ticks"])
            prior = latest.get(key)
            if prior is None or (identity["started_at_ns"], elapsed) > (
                prior["identity"]["started_at_ns"],
                prior["elapsed_s"],
            ):
                latest[key] = result
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"Invalid profile {path.name}: {error}") from error
    if not latest:
        raise ValueError("No final version-1 profiles found; wait for captures to finish")
    return sorted(latest.values(), key=lambda r: -r["cores"])


def markdown(report: Mapping[str, Any]) -> str:
    lines = ["| Service | Survey cores | Profile cores |", "|---|---:|---:|"]
    survey_rows = {r["service"]: r for r in report.get("services", [])}
    profiles: dict[str, list[dict[str, Any]]] = {}
    for row in report.get("profiles", []):
        profiles.setdefault(row["service"], []).append(row)
    for name in sorted(survey_rows.keys() | profiles.keys()):
        measured = f"{survey_rows[name]['cores']:.3f}" if name in survey_rows else "unavailable"
        rows = profiles.get(name, [])
        profiled = "unavailable"
        if len(rows) == 1:
            profiled = f"{rows[0]['cores']:.3f}"
        elif rows:
            profiled = "per-PID rates: " + "; ".join(
                f"PID {r['identity']['pid']} start {r['identity']['process_start_ticks']}: {r['cores']:.3f}"
                for r in rows
            )
        lines.append(f"| {name} | {measured} | {profiled} |")
    lines.append(
        "\nSurvey and profile windows are independent; their core estimates must not be added. Per-PID profile rates may cover different windows and are not summed. Native functions unavailable. Python function weights are current-stack estimates; native work may be assigned to its visible Python caller. Inclusive estimates overlap."
    )
    for row in report.get("profiles", []):
        name = row["service"]
        identity = row["identity"]
        lines.append(
            f"\n{name} PID {identity['pid']} start {identity['process_start_ticks']}, capture {identity['capture_id']}: profile window {row['elapsed_s']:.3f}s, stack filter {row['stack_filter']}. CPU accounting (OS ticks): {row['accounting_ticks']}."
        )
        lines.extend(["", "| Python location (estimate) | Self cores | Inclusive cores |", "|---|---:|---:|"])
        for function in row["top_functions"]:
            location = function["location"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {location} | {function['self_cores_estimate']:.4f} | {function['inclusive_cores_estimate']:.4f} |"
            )
        waits = row.get("waits", "unavailable")
        if not isinstance(waits, dict):
            lines.append("\nWait sites between wakeups: unavailable (capture predates wait-site sampling).")
            continue
        rates = waits["wakeups_per_s"]
        lines.append(
            f"\nWait sites between wakeups (not CPU attribution; non-running CPU stays in the unattributed total above): {rates['voluntary_switches']:.1f} wakeups/s in all, {rates['wait_site_wakeups']:.1f} at Python wait sites, {rates['running_endpoint_wakeups']:.1f} at running endpoints, {rates['native_wakeups']:.1f} on native threads."
        )
        lines.extend(
            ["", "| Wait site, innermost first | Wakeups/s | Non-running cores | Threads |", "|---|---:|---:|---:|"]
        )
        for site in waits["top_sites"]:
            lines.append(
                f"| {_site(site['stack'])} | {site['wakeups_per_s']:.1f} | {site['nonrunning_cores']:.4f} | {site['threads']} |"
            )
        lines.extend(["", "| Native thread | Wakeups/s | CPU cores |", "|---|---:|---:|"])
        for thread in waits["top_threads"]:
            lines.append(f"| {thread['native_tid']} | {thread['wakeups_per_s']:.1f} | {thread['cores']:.4f} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Survey duration (default 60); with --profiles, explicitly starts a new survey",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profiles", type=Path, help="Summarize final sampler JSON snapshots without attaching")
    parser.add_argument("--survey", type=Path, help="Use an existing survey JSON instead of reading /proc")
    args = parser.parse_args(argv)
    try:
        if args.survey and args.seconds is not None:
            raise ValueError("--survey and --seconds cannot be combined")
        report: dict[str, Any] = {"schema_version": 1, "services": []}
        if args.survey:
            report = json.loads(args.survey.read_text())
            if (
                not isinstance(report, dict)
                or report.get("schema_version") != 1
                or not isinstance(report.get("services"), list)
            ):
                raise ValueError("--survey requires a version-1 service survey JSON")
            for row in report["services"]:
                if not isinstance(row, dict) or not isinstance(row.get("service"), str):
                    raise ValueError("invalid survey service")
                _number(row.get("cores"), "survey cores")
        elif args.seconds is not None or not args.profiles:
            if not sys.platform.startswith("linux"):
                raise ValueError("passive process CPU survey requires Linux /proc")
            commands = services._commands()
            report = survey(
                60.0 if args.seconds is None else args.seconds,
                lambda: snapshot(Path("/proc"), commands),
                ticks=os.sysconf("SC_CLK_TCK"),
            )
        if args.profiles:
            report["profiles"] = read_profiles(args.profiles)
        write_report(args.output, report)
        print(markdown(report))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
