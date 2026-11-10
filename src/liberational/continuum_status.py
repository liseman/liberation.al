"""Is Continuum up and healthy on this machine? (`pack-continuum-status`, `mise run continuum-status`.)

One screen for turning Continuum on (docs/CONTINUUM.md), and exit 1 if anything's wrong:

- each Continuum service: installed, and its service manager's state (`pack-services status`);
- the broker: its socket accepts a connection, and the client handshake works;
- every registered stream: whether a producer is connected, and its newest payload in the last `--window` seconds;
- the Tier 0 arena: per size class, slots in use and slots held (being written, pinned, or not in Tier 1 yet),
  failing on held, since a capture class runs full by design. A network (RTSP) camera's encoded ring fills its byte
  budget whatever it holds, so a class only those rings use is judged by each ring's last health line instead (its
  pictures dropped under pressure or overwritten unflushed), matched to the broker's own Tier 0 plan;
- the tier worker's flush lag: the oldest recent payload of a capture stream that's still only in Tier 0;
- the vector worker: its search streams are registered;
- the gateway: its port accepts a connection.

With `--port audio` or `--port vision` it checks the stage of the switch the machine is at (the ports switch one at a
time): every service that stage needs (up to `--through`, the last one turned on so far) must be installed and running, and every device in the manifest
(`config/devices.toml`, only the microphones at the audio stage) must have a live, fresh stream. Its exit code alone
then says go on (0) or roll back (1); `--settle N` checks again for up to N seconds before deciding, for a service
just started.

It only reads: it starts, stops, and changes nothing.
"""

import argparse
import asyncio
import pathlib
import re
import socket as py_socket
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from liberational import config as cfg
from liberational import logs, services

AUDIO = (
    "continuum-broker",
    "continuum-capture",
    "continuum-tier-worker",
    "continuum-vector-worker",
    "listen",
    "voice",
)
"""What the audio port's switch runs, in the order it's turned on."""
VISION = (*AUDIO, "see-decode", "see-actions", "see-captions", "continuum-gateway")
"""What the vision port's switch adds."""
PORTS = {"audio": AUDIO, "vision": VISION}
SERVICES = VISION
"""Every Continuum service this checks."""
GATEWAY_PORT = 8766


@dataclass
class Report:
    """What the check found: lines to print, and the problems among them."""

    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def ok(self, line: str) -> None:
        self.lines.append(f"  ok    {line}")

    def note(self, line: str) -> None:
        self.lines.append(f"        {line}")

    def bad(self, line: str) -> None:
        self.lines.append(f"  FAIL  {line}")
        self.problems.append(line)

    def section(self, title: str) -> None:
        self.lines.append(title)


def check_services(
    report: Report,
    installed: list[str],
    state: dict[str, str],
    expected: Iterable[str] = (),
    *,
    only_expected: bool = False,
) -> set[str]:
    """Each Continuum service's state. One the stage expects must be installed and running; any other that isn't
    installed is only noted. Returns the running ones."""
    report.section("Services")
    expected = set(expected)
    running = set()
    for name in SERVICES:
        if only_expected and name not in expected:
            continue
        if name not in installed:
            if name in expected:
                report.bad(f"{name}: not installed, and this stage needs it")
            else:
                report.note(f"{name}: not installed")
        elif state.get(name) in ("active", "running"):
            report.ok(f"{name}: {state[name]}")
            running.add(name)
        else:
            report.bad(f"{name}: {state.get(name, 'unknown')}")
    return running


def manifest_streams(path: pathlib.Path) -> set[str]:
    """The capture streams a device manifest makes: `<name>.video` for a camera and `<name>.audio` for a
    microphone (continuum-capture's names)."""
    devices = tomllib.loads(path.read_text()).get("device", [])
    suffix = {"camera": "video", "rtsp": "video", "microphone": "audio"}
    return {f"{d['name']}.{suffix[d['kind']]}" for d in devices if d.get("kind") in suffix and d.get("name")}


def accepts(address: Any, family: int, timeout: float = 2.0) -> bool:
    """Whether a socket at `address` accepts a connection."""
    probe = py_socket.socket(family)
    probe.settimeout(timeout)
    try:
        probe.connect(address)
    except OSError:
        return False
    finally:
        probe.close()
    return True


def arena_name(socket: pathlib.Path) -> str | None:
    """The Tier 0 arena's name, the second line of `<socket>.arena`, which the broker writes."""
    state = pathlib.Path(f"{socket}.arena")
    try:
        lines = state.read_text().splitlines()
    except OSError:
        return None
    return lines[1] if len(lines) > 1 else None


@dataclass(frozen=True)
class Tier0Plan:
    """The broker's Tier 0 plan (`continuum-broker --print-tier0-plan`): each class's slot count by slot size, the
    slots network cameras' encoded rings reserve by slot size, and those cameras."""

    classes: dict[int, int]
    network: dict[int, int]
    cameras: tuple[str, ...]


_PLAN_CLASS = re.compile(r"class=\d+ configured_slot_bytes=(\d+) count=(\d+)")
_PLAN_NETWORK = re.compile(r"camera=(\S+) .*\bbyte_budget_limited=true class_slots=\[([^\]]*)\]")
_PLAN_SLOTS = re.compile(r"\((\d+), (\d+)\)")
_HEALTH = re.compile(
    r"\[capture-worker\] [A-Z]+ (\S+): RTSP frames=(\d+) reader_dropped=\d+ pressure_dropped=(\d+) unflushed_overwrites=(\d+)"
)


def parse_plan(text: str) -> Tier0Plan | None:
    """The plan in the broker's `--print-tier0-plan` output, None when it lists no classes."""
    classes: dict[int, int] = {}
    network: dict[int, int] = {}
    cameras = []
    for line in text.splitlines():
        if found := _PLAN_CLASS.fullmatch(line.strip()):
            classes[int(found[1])] = int(found[2])
        elif found := _PLAN_NETWORK.fullmatch(line.strip()):
            cameras.append(found[1])
            for size, count in _PLAN_SLOTS.findall(found[2]):
                network[int(size)] = network.get(int(size), 0) + int(count)
    return Tier0Plan(classes, network, tuple(cameras)) if classes else None


def tier0_plan() -> Tier0Plan | None:
    """The plan the broker's own command line sizes Tier 0 to. It prints and exits, creating nothing; None when the
    broker isn't declared or can't say."""
    svc = services.declared().get("continuum-broker")
    if svc is None:
        return None
    try:
        done = subprocess.run(
            [*services.argv(svc), "--print-tier0-plan"], capture_output=True, text=True, timeout=30, cwd=cfg.workdir()
        )
    except OSError, subprocess.SubprocessError:
        return None
    return parse_plan(done.stdout) if done.returncode == 0 else None


@dataclass(frozen=True)
class RingHealth:
    """One network camera's encoded ring over its last health interval: pictures it published, and pictures it
    dropped under pressure or overwrote before they were flushed. The worker writes the line every 10 s, but at INFO
    only once every 10 minutes (#59), so at the default level the interval is 10 minutes."""

    frames: int
    dropped: int
    overwritten: int


def ring_health(lines: Iterable[str]) -> dict[str, RingHealth]:
    """Each RTSP worker's last interval, from its two newest health lines (their counts run since its connection).
    A camera whose counts went down reconnected in between: its newest line alone is that connection's so far."""
    newest: dict[str, tuple[tuple[int, int, int], ...]] = {}
    for line in lines:
        if found := _HEALTH.search(line):
            counts = (int(found[2]), int(found[3]), int(found[4]))
            newest[found[1]] = (*newest.get(found[1], ())[-1:], counts)
    result = {}
    for name, seen in newest.items():
        last = seen[-1]
        first = seen[0] if len(seen) == 2 and all(a <= b for a, b in zip(seen[0], last, strict=True)) else (0, 0, 0)
        result[name] = RingHealth(*(b - a for a, b in zip(first, last, strict=True)))
    return result


def log_tail(path: pathlib.Path, limit: int = 1 << 20) -> list[str]:
    """The last `limit` bytes of a log and its older copy (logs.tail), as lines (a partial first line dropped);
    empty when it can't be read."""
    parts = logs.tail(path, limit)
    lines = b"".join(p.data for p in parts).decode("utf-8", "replace").splitlines()
    return lines[1:] if parts and parts[0].offset > 0 else lines


def check_arena(
    report: Report,
    socket: pathlib.Path,
    full: float,
    plan: Tier0Plan | None = None,
    health: Mapping[str, RingHealth] | None = None,
) -> None:
    from continuum_py import Arena

    name = arena_name(socket)
    if name is None:
        report.bad(f"Tier 0 arena: no {socket}.arena naming it")
        return
    try:
        arena = Arena.open(name)
    except Exception as e:  # any failure to map it is the finding
        report.bad(f"Tier 0 arena {name}: can't open it ({e})")
        return
    rows = [(*arena.slab_class(c), arena.used(c), arena.held(c)) for c in range(arena.class_count)]
    judge_arena(report, rows, full, plan, health or {})


def judge_arena(
    report: Report,
    rows: list[tuple[int, int, int, int]],
    full: float,
    plan: Tier0Plan | None,
    health: Mapping[str, RingHealth],
) -> None:
    """Judges each class's (slot bytes, count, used, held). A capture class runs full by design (its producer
    overwrites the oldest slot once its frame is archived), so what counts is what can't be taken back: being written,
    pinned by a reader, or not in Tier 1 yet. A network camera's ring holds its byte budget whatever that is, so its
    slots count only through its own health: did it drop or overwrite anything in its last interval."""
    matched = plan is not None and plan.classes == {size: count for size, count, _, _ in rows}
    if plan is not None and plan.network and not matched:
        report.note("Tier 0: the broker's plan doesn't match the live arena (restart pending?); judging by held alone")
    network = plan.network if matched and plan is not None else {}
    for c, (size, count, used, held) in enumerate(rows):
        line = f"Tier 0 class {c} ({size // 1024} KiB slots): {used} of {count} in use, {held} held"
        ring = min(network.get(size, 0), count)
        if count and ring == count:
            report.ok(f"{line}, network camera rings (judged by their health below)")
            continue
        # A shared class: its ring slots may all be held; the rest must keep room.
        held, count = max(0, held - ring), count - ring
        if count and held / count >= full:
            report.bad(f"{line}, at or above {full:.0%} held" + (f" outside its {ring} ring slots" if ring else ""))
        else:
            report.ok(line)
    if not network or plan is None:
        return
    for name in plan.cameras:
        state = health.get(name)
        if state is None:
            report.note(f"{name} ring: no health line in capture's log yet")
        elif state.dropped or state.overwritten:
            report.bad(
                f"{name} ring, last interval: {state.dropped} dropped under pressure, {state.overwritten} "
                f"overwritten unflushed, {state.frames} kept"
            )
        elif not state.frames:
            report.bad(f"{name} ring: no pictures in its last interval")
        else:
            report.ok(f"{name} ring: {state.frames} pictures in its last interval, none dropped")


async def check_broker(
    report: Report,
    socket: pathlib.Path,
    args: argparse.Namespace,
    expected_streams: Iterable[str] = (),
    expect_vector: bool = False,
) -> None:
    from continuum_py import Client

    report.section(f"Broker ({socket})")
    if not accepts(str(socket), py_socket.AF_UNIX):
        report.bad("its socket doesn't accept connections")
        return
    try:
        client = await Client.connect(str(socket), client_name="continuum-status")
    except Exception as e:  # any failure to connect is the finding
        report.bad(f"connecting failed: {e}")
        return
    try:
        report.ok(f"connected (schema {'.'.join(map(str, client.broker_schema))})")
        archive_checks = not bool(
            getattr(args, "selected_only", False)
        ) or "continuum-tier-worker" in _expected_services(args)
        if archive_checks:
            check_arena(
                report, socket, args.full, getattr(args, "tier0_plan", None), getattr(args, "ring_health", None)
            )
        else:
            report.note("Tier worker unselected: archive-held pressure and flush lag are not checked")
        streams = await client.list_streams()
        await check_streams(report, client, streams, args, set(expected_streams), archive_checks=archive_checks)
        names = {s["name"] for s in streams}
        report.section("Vector worker")
        if {"vector.search", "text.search"} <= names:
            report.ok("its search streams are registered")
        elif expect_vector:
            report.bad("its search streams aren't registered")
        else:
            report.note("its search streams aren't registered (it hasn't run yet)")
    finally:
        await client.close("status checked")


async def check_streams(
    report: Report,
    client: Any,
    streams: list[dict[str, Any]],
    args: argparse.Namespace,
    expected: set[str],
    *,
    archive_checks: bool = True,
) -> None:
    """Each stream's newest payload in the window, and for a capture stream that's live, how fresh it is and how far
    the tier worker is behind. Every stream in `expected` (the manifest's devices) must be registered and live."""
    report.section(f"Streams (newest in the last {args.window} s)")
    registered = {s["name"] for s in streams}
    for name in sorted(expected - registered):
        report.bad(f"{name}: the manifest names its device, but capture hasn't registered it")
    now = time.time_ns()
    since = now - args.window * 1_000_000_000
    for s in streams:
        try:
            result = await client.query(s["name"], since, now)
        except Exception as e:  # a stream that can't be read is the finding
            report.bad(f"{s['name']}: can't query it ({e})")
            continue
        items = result["items"]
        live = "live" if s["live"] else "no producer"
        sensor = s["kind"] == "Sensor"
        if s["name"] in expected and not s["live"]:
            report.bad(f"{s['name']}: no producer connected (capture isn't reading its device)")
        if not items:
            if sensor and s["live"]:
                report.bad(f"{s['name']} ({live}): nothing in the window, though its producer is connected")
            else:
                report.note(f"{s['name']} ({live}): nothing in the window")
            continue
        from continuum_py import decode_header

        headers = [decode_header(i["header"]) for i in items]
        age = (now - max(h["end_ns"] for h in headers)) / 1e9
        line = f"{s['name']} ({live}): {len(items)} payloads, newest {age:.1f} s ago"
        if sensor and s["live"] and age > args.stale:
            report.bad(f"{line}, older than {args.stale} s")
        else:
            report.ok(line)
        if sensor and archive_checks:
            # The oldest payload still only in Tier 0: how far the tier worker is behind on this stream.
            hot = [
                h["start_ns"] for h, i in zip(headers, items, strict=True) if (i["locator"] or {}).get("kind") == "Shm"
            ]
            if hot:
                lag = (now - min(hot)) / 1e9
                flush = f"{s['name']}: tier worker flush lag {lag:.1f} s"
                if lag > args.lag:
                    report.bad(f"{flush}, more than {args.lag} s")
                else:
                    report.ok(flush)


def check_gateway(report: Report) -> None:
    report.section("Gateway")
    if accepts(("127.0.0.1", GATEWAY_PORT), py_socket.AF_INET):
        report.ok(f"port {GATEWAY_PORT} accepts connections")
    else:
        report.bad(f"port {GATEWAY_PORT} doesn't accept connections")


def stage(port: str | None, through: str | None) -> tuple[str, ...]:
    """The services a stage of the switch needs: the port's, in the order they're turned on, up to and including
    `through` (all of them when it's None). No port, no expectations."""
    if port is None:
        return ()
    order = PORTS[port]
    if through is None:
        return order
    if through not in order:
        raise SystemExit(f"{through} isn't one of the {port} port's services: {', '.join(order)}")
    return order[: order.index(through) + 1]


def _expected_services(args: argparse.Namespace) -> tuple[str, ...]:
    expected = stage(args.port, args.through)
    explicit = getattr(args, "expected_service", None)
    selected_only = bool(getattr(args, "selected_only", False))
    if explicit is not None:
        if not selected_only or not args.port:
            raise ValueError("--expected-service requires --selected-only and --port")
        invalid = set(explicit) - set(expected)
        if invalid:
            raise ValueError(f"expected services outside the requested port stage: {', '.join(sorted(invalid))}")
        return tuple(name for name in expected if name in explicit)
    if selected_only:
        chosen = services.selected()
        return tuple(name for name in expected if name in chosen)
    return expected


def run(args: argparse.Namespace) -> Report:
    report = Report()
    args.tier0_plan = tier0_plan()
    args.ring_health = ring_health(log_tail(services.log_path("continuum-capture")))
    expected = _expected_services(args)
    selected_only = bool(getattr(args, "selected_only", False))
    running = check_services(report, services.installed(), services.states(), expected, only_expected=selected_only)
    try:
        import continuum_py  # noqa: F401  # the SDK, built by uv sync (MIGRATING.md, section 46)
    except ImportError:
        report.bad("continuum_py isn't installed in this environment: run uv sync")
        return report
    streams: set[str] = set()
    if args.port and "continuum-capture" in expected:
        if args.manifest.exists():
            streams = manifest_streams(args.manifest)
        else:
            report.bad(f"{args.manifest} doesn't exist: capture has no devices to read")
    socket = cfg.path("paths.state") / "continuum" / "broker.sock"
    asyncio.run(check_broker(report, socket, args, streams, "continuum-vector-worker" in expected))
    if "continuum-gateway" in running or "continuum-gateway" in expected:
        check_gateway(report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pack-continuum-status", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--port",
        choices=sorted(PORTS),
        help="the switch stage: every service it needs must run, and every manifest device must stream",
    )
    parser.add_argument("--through", help="the last of the port's services turned on so far (default: all of them)")
    parser.add_argument(
        "--selected-only", action="store_true", help="check only selected services within the requested port stage"
    )
    parser.add_argument(
        "--expected-service",
        action="append",
        help="check this service within the port stage (repeatable; requires --selected-only)",
    )
    parser.add_argument("--manifest", type=pathlib.Path, default=cfg.devices_file(), help="capture's device manifest")
    parser.add_argument("--settle", type=float, default=0.0, help="check again for up to this many seconds first")
    parser.add_argument("--window", type=int, default=60, help="seconds back to look for each stream's newest")
    parser.add_argument("--stale", type=float, default=10.0, help="a live capture stream older than this fails")
    parser.add_argument("--lag", type=float, default=30.0, help="a tier worker flush lag over this fails")
    parser.add_argument(
        "--full", type=float, default=0.9, help="a Tier 0 class with at least this share held (not reclaimable) fails"
    )
    args = parser.parse_args(argv)
    if args.selected_only and not args.port:
        parser.error("--selected-only requires --port")
    if args.expected_service is not None:
        try:
            _expected_services(args)
        except ValueError as error:
            parser.error(str(error))
    deadline = time.monotonic() + args.settle
    report = run(args)
    while report.problems and time.monotonic() < deadline:
        time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
        report = run(args)
    print("\n".join(report.lines))
    if report.problems:
        print(f"\n{len(report.problems)} problem(s): roll back (docs/CONTINUUM.md, Running and checking it)")
        sys.exit(1)
    print("\nall good")


if __name__ == "__main__":
    main()
