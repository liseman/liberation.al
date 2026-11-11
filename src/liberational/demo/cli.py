"""pack-demo: the dashboard on the made-up demo household (#17), apart from any real data.

    pack-demo [--dir DIR] [--port PORT]

Makes a demo directory (DIR, or a new one under the system's temporary directory), points every `paths.*` setting
and the pack's config file into it, starts a private Continuum broker on its state directory, seeds the dataset in
demo/dataset.py, and runs the dashboard on PORT. Without --port it is 8780, a port no service of the pack uses, so the
demo runs beside an open application; when 8780 is taken, a free port the pack doesn't use instead, and the address
line says which. That line is printed once the dashboard answers, so a script may open it at once. A --port that is
taken is refused at once. Ctrl-C stops the dashboard, broker, and private playback gateway. The demo runs no models:
Ask uses prepared answers, and the demo HTTP process refuses outbound network connections and subprocesses.
Room illustrations, recordings, portraits, and point clouds are generated locally at each start.

Each start restores a fresh household in DIR, discarding the previous demo edits. The real `data/` directory is never read or written: the
launcher refuses a DIR inside it.
"""

import argparse
import datetime as dt
import http.client
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from liberational import config as cfg

# The dashboard's default port: none of the pack's own ([ports] in config/defaults.toml, and the gateway's).
PORT = 8780
# The broker's Tier 0 arena, kept tiny: generated video is read from disk, not live capture.
TIER0 = "16384x64"
# Where Ollama would be: the discard port, which nothing listens on, so the demo can't reach a real model.
NO_OLLAMA = "http://127.0.0.1:9"
# Where the page looks for the camera gateway: the discard port too, which browsers refuse to open, so the demo's page
# never reaches a real install's gateway on this machine (it would show that install's cameras and recordings).
NO_GATEWAY = 9
SEEDED = "seeded"  # the marker a seeded demo directory carries
SOCKET_MAX = 100  # bytes a unix socket path may hold, with room to spare (108 on Linux, 104 on macOS)
CONFIG = """# The demo household's settings (pack-demo, #17). Everyone here is made up.
[owner]
name = "Alice"

[assistant]
name = "Bob"
"""


def path_keys() -> list[str]:
    """Every `paths.*` setting config/defaults.toml has."""
    return sorted(tomllib.loads(cfg.DEFAULTS.read_text())["paths"])


def environment(root: Path, port: int, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment the demo's broker, seeding, and dashboard run in: `base` (os.environ by default) with every
    `paths.*` setting under `root`, the config file `root/pack.toml`, the broker's socket in `root`'s state directory
    (or a short temporary one when that path is too long for a socket), and Ollama and the camera gateway nowhere."""
    inherited = os.environ if base is None else base
    aliases = {name for names in cfg.ALIASES.values() for name in names}
    env = {
        key: value
        for key, value in inherited.items()
        if not key.startswith(("PACK_", "OLLAMA_", "CONTINUUM_")) and key not in aliases
    }
    root = root.resolve()
    env["PACK_ROOT"] = str(root)
    env["TMPDIR"] = str(root / "tmp")
    for key in path_keys():
        env[f"PACK_PATHS_{key.upper()}"] = str(root / key)
    env["PACK_STATE"] = env["PACK_PATHS_STATE"]  # the older name, which a shell may still set
    env["PACK_CONFIG"] = str(root / "pack.toml")
    env["PACK_PORTS_DASHBOARD"] = str(port)
    env["PACK_PORTS_GATEWAY"] = str(NO_GATEWAY)
    env["PACK_OLLAMA_URL"] = NO_OLLAMA
    env.pop("OLLAMA_URL", None)
    env.pop("OLLAMA_HOST", None)
    sock = root / "state" / "continuum" / "broker.sock"
    if len(str(sock).encode()) > SOCKET_MAX:
        sock = Path(tempfile.mkdtemp(prefix="pack-demo-")) / "b.sock"
    env["CONTINUUM_SOCKET"] = str(sock)
    return env


def prepare(root: Path) -> None:
    """Write only the demo's settings and device manifest, including when reusing an older demo."""
    from liberational.demo.dataset import MIC

    root.mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(exist_ok=True)
    (root / "pack.toml").write_text(CONFIG)
    folder = root / "config"
    folder.mkdir(exist_ok=True)
    cameras = "".join(
        f'[[device]]\nname = "{name}"\nkind = "camera"\n\n' for name in ("front-room", "kitchen", "porch", "study")
    )
    (folder / "devices.toml").write_text(cameras + f'[[device]]\nname = "{MIC}"\nkind = "microphone"\n')


def check_dir(root: Path) -> None:
    """Refuses a demo directory inside the real data directory, or one that holds something other than a demo."""
    real = cfg.data_dir().resolve()
    here = root.resolve()
    if here == real or real in here.parents:
        raise SystemExit(f"pack-demo: {root} is inside the real data directory ({real}); pick another --dir")
    if here.exists() and any(here.iterdir()) and not (here / SEEDED).exists():
        raise SystemExit(f"pack-demo: {root} isn't empty and isn't a demo directory; pick an empty or new --dir")


def reset(root: Path) -> None:
    """Discard only this marked demo's scratch state before starting any child process."""
    check_dir(root)
    for path in root.iterdir() if root.exists() else ():
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def reserved_ports() -> set[int]:
    """The ports the pack's own services use: every `ports.*` setting, as config/defaults.toml has it and as this
    install sets it, and the one above the camera gateway's, where a TLS proxy may sit in front of it."""
    defaults = tomllib.loads(cfg.DEFAULTS.read_text())["ports"]
    here = cfg.load().get("ports", {})
    taken = {v for v in (*defaults.values(), *here.values()) if isinstance(v, int)}
    return taken | {port + 1 for port in (defaults["gateway"], here.get("gateway")) if isinstance(port, int)}


def port_free(port: int) -> bool:
    """Whether the dashboard could listen on `port` now, binding as it does (every address, address reuse on)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def any_free_port(reserved: set[int]) -> int:
    """A port the system says is free now and the pack doesn't use."""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("0.0.0.0", 0))
            port = probe.getsockname()[1]
        if port not in reserved:
            return port


def choose_port(asked: int | None) -> int:
    """The dashboard's port: `asked` when it is free (SystemExit when it isn't, before anything starts); without it,
    PORT when it is free and this install hasn't given it to a service, else a free port the pack doesn't use."""
    if asked is not None:
        if not port_free(asked):
            raise SystemExit(
                f"pack-demo: port {asked} is in use by another program; pick another with --port, or leave --port out "
                "for a free one"
            )
        return asked
    reserved = reserved_ports()
    if PORT not in reserved and port_free(PORT):
        return PORT
    port = any_free_port(reserved)
    print(f"pack-demo: port {PORT} is taken, so the demo dashboard takes port {port}", flush=True)
    return port


def broker_binary() -> Path:
    """The continuum-broker to run: CONTINUUM_BROKER, else the one in this environment (in a checkout, where
    `mise run continuum-release` links the release build)."""
    given = os.environ.get("CONTINUUM_BROKER")
    found = Path(given) if given else cfg.program("continuum-broker")
    if not found.is_file():
        raise SystemExit(f"pack-demo: no broker at {found}: in a checkout, build it with `mise run continuum-release`")
    return found


def wait_for_socket(path: Path, process: subprocess.Popen[bytes], timeout: float = 15.0) -> None:
    """Waits until the broker listens on `path`; raises SystemExit if it exits or takes longer than `timeout`."""
    deadline = time.monotonic() + timeout
    while True:
        with socket.socket(socket.AF_UNIX) as probe:
            try:
                probe.connect(str(path))
                return
            except OSError:
                pass
        if process.poll() is not None:
            raise SystemExit(f"pack-demo: the broker exited ({process.returncode}) before it listened")
        if time.monotonic() > deadline:
            raise SystemExit(f"pack-demo: the broker didn't listen on {path} within {timeout:.0f} s")
        time.sleep(0.05)


def start_broker(root: Path, env: Mapping[str, str], output: IO[bytes] | None = None) -> subprocess.Popen[bytes]:
    """Starts the demo's own broker on `root`'s state directory, its catalog there too, its log to `output` (by
    default this process's)."""
    state = Path(env["PACK_PATHS_STATE"])
    sock = Path(env["CONTINUUM_SOCKET"])
    (state / "continuum").mkdir(parents=True, exist_ok=True)
    sock.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(broker_binary()),
        "--state-dir",
        str(state),
        "--socket",
        str(sock),
        "--catalog",
        str(state / "continuum" / "catalog.redb"),
        "--tier0-classes",
        TIER0,
        "--tier0-ring-mib",
        "1",
    ]
    process = subprocess.Popen(command, cwd=root, env=dict(env), stdout=output, stderr=output)
    try:
        wait_for_socket(sock, process)
    except BaseException:
        stop(process)
        raise
    return process


def gateway_command(env: Mapping[str, str], port: int) -> list[str]:
    """The private playback gateway's command, listening on `port`: it is given the broker's socket explicitly,
    because it would otherwise look for one under the state directory, where a long demo directory's socket path is
    too long for a socket and the broker listens in a short temporary one instead."""
    return [
        str(cfg.program("continuum-gateway")),
        "--state-dir",
        env["PACK_PATHS_STATE"],
        "--broker-socket",
        env["CONTINUUM_SOCKET"],
        "--listen",
        f"127.0.0.1:{port}",
    ]


def start_vector_worker(env: Mapping[str, str], output: IO[bytes] | None = None) -> subprocess.Popen[bytes]:
    """Starts a vector worker on the demo's broker, which indexes the journal's text as it is written, so a journal
    search finds the demo's entries (the meaning index needs embeddings, which the demo doesn't make)."""
    worker = cfg.program("continuum-vector-worker")
    if not worker.is_file():
        raise SystemExit(f"pack-demo: no vector worker at {worker}: build it with `mise run continuum-release`")
    command = [str(worker), "--socket", env["CONTINUUM_SOCKET"], "--state-dir", env["PACK_PATHS_STATE"]]
    return subprocess.Popen(command, env=dict(env), stdout=output, stderr=output)


def seed_here(today: dt.date | None = None, *, media: bool = False) -> dt.date:
    """Seeds the demo into the configured state directory (run in the demo's environment, by `run`), its day the one
    before `today` (by default the real today)."""
    from liberational import memory_log, speech_log
    from liberational.demo import dataset

    state = cfg.path("paths.state")
    mic = speech_log.Writer(dataset.MIC)
    try:
        day = dataset.seed(state, memory_log.at(state), today, say=lambda line: mic.write(speech_log.UTTERANCES, line))
        mic.sync()
    finally:
        mic.close()
    dataset.seed_faces(cfg.path("paths.faces"), time.time())
    if media:
        from liberational.demo import household

        household.seed(state, day)
    print(f"pack-demo: seeded the demo household, with a journal and a transcript for {day.isoformat()}")
    return day


def wait_for_index(day: dt.date, timeout: float = 60.0) -> None:
    """Waits until a vector worker has indexed the demo journal's text: until a word search finds the day's last hour
    entry, written after the rest. Raises SystemExit after `timeout` seconds."""
    from liberational.demo import dataset
    from liberational.memory import journal, recall

    state = cfg.path("paths.state")
    start, end = journal.day_bounds(day)
    deadline = time.monotonic() + timeout
    while not any(
        r.get("text") == dataset.DAY[-1].text for r in recall.search("lights", start, end, None, state, rank=False)
    ):
        if time.monotonic() > deadline:
            raise SystemExit(f"pack-demo: the journal wasn't indexed for search within {timeout:.0f} s")
        time.sleep(0.2)


def wait_for_http(port: int, process: subprocess.Popen[bytes], timeout: float = 60.0) -> None:
    """Waits until the dashboard answers `/health` on `port`; raises SystemExit if it exits or takes longer than
    `timeout`. The address line is printed only after this, so a script that opens it at once is never refused."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.request("GET", "/health")
                if connection.getresponse().status == 200:
                    return
            finally:
                connection.close()
        except (OSError, http.client.HTTPException):
            pass
        if process.poll() is not None:
            raise SystemExit(f"pack-demo: the dashboard exited ({process.returncode}) before it listened")
        if time.monotonic() > deadline:
            raise SystemExit(f"pack-demo: the dashboard didn't listen on port {port} within {timeout:.0f} s")
        time.sleep(0.05)


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def stop(process: subprocess.Popen[Any]) -> None:
    """Stops a child of this launcher, whether it was started with byte streams or text ones."""
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run(root: Path, port: int, dashboard_args: Sequence[str] = ()) -> int:
    """Prepares `root`, starts the broker, seeds a new demo, and runs the dashboard until it stops; returns its exit
    status."""
    check_dir(root)
    # A stop (SIGTERM) unwinds like Ctrl-C, so the dashboard and the broker are stopped too, never left running.
    signal.signal(signal.SIGTERM, _interrupt)
    root = root.resolve()
    reset(root)
    prepare(root)
    env = environment(root, port)
    gateway_port = any_free_port(reserved_ports() | {port})
    env["PACK_PORTS_GATEWAY"] = str(gateway_port)
    broker = start_broker(root, env)
    try:
        subprocess.run([sys.executable, "-m", "liberational.demo.cli", "--seed-here", "--media"], env=env, check=True)
        (root / SEEDED).write_text("pack-demo\n")
        gateway = subprocess.Popen(gateway_command(env, gateway_port), env=env, cwd=root)
        try:
            dashboard = subprocess.Popen(
                [sys.executable, "-m", "liberational.demo.server", str(port), *dashboard_args], env=env
            )
            try:
                wait_for_http(port, dashboard)
                print(
                    f"pack-demo: the demo dashboard is on http://localhost:{port}/ (its files are in {root})",
                    flush=True,
                )
                return dashboard.wait()
            except KeyboardInterrupt:
                return 0
            finally:
                stop(dashboard)
        finally:
            stop(gateway)
    finally:
        stop(broker)
        sock = Path(env["CONTINUUM_SOCKET"])
        if not sock.is_relative_to(root):
            shutil.rmtree(sock.parent, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> None:
    """pack-demo [--dir DIR] [--port PORT]"""
    parser = argparse.ArgumentParser(prog="pack-demo", description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", type=Path, help="the demo directory (default: a new temporary one)")
    parser.add_argument(
        "--port", type=int, help=f"the dashboard's port (default {PORT}, or a free one when {PORT} is taken)"
    )
    parser.add_argument("--dev", action="store_true", help="run the dashboard with --dev (pages from Vite)")
    parser.add_argument("--media", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed-here", action="store_true", help=argparse.SUPPRESS)
    # With --seed-here: the day after the demo's, and waiting for a vector worker to index it (tools/measure.py --demo)
    parser.add_argument("--today", type=dt.date.fromisoformat, help=argparse.SUPPRESS)
    parser.add_argument("--wait-index", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.seed_here:
        day = seed_here(args.today, media=args.media)
        if args.wait_index:
            wait_for_index(day)
        return
    if args.media:
        from liberational.demo import household

        household.seed(cfg.path("paths.state"), (args.today or dt.date.today()) - dt.timedelta(days=1))
        return
    port = choose_port(args.port)
    root = args.dir or Path(tempfile.mkdtemp(prefix="pack-demo-"))
    sys.exit(run(root, port, ["--dev"] if args.dev else []))


if __name__ == "__main__":
    main()
