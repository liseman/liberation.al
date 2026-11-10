"""pack-demo: the dashboard on the made-up demo household (#17), apart from any real data.

    pack-demo [--dir DIR] [--port PORT]

Makes a demo directory (DIR, or a new one under the system's temporary directory), points every `paths.*` setting
and the pack's config file into it, starts a private Continuum broker on its state directory, seeds the dataset in
demo/dataset.py, and runs the dashboard on PORT (8766 by default, so it doesn't collide with a real dashboard on
8765). Ctrl-C stops the dashboard and the broker. The demo runs no models: its Ollama URL points nowhere, so asking a
question in it fails rather than reaching a real model.

A DIR that already holds a seeded demo is reused as it is. The real `data/` directory is never read or written: the
launcher refuses a DIR inside it.
"""

import argparse
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

from liberational import config as cfg

PORT = 8766
# The broker's Tier 0 arena, kept tiny: the demo has no video or audio, only the journal's text payloads.
TIER0 = "16384x64"
# Where Ollama would be: the discard port, which nothing listens on, so the demo can't reach a real model.
NO_OLLAMA = "http://127.0.0.1:9"
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
    (or a short temporary one when that path is too long for a socket), and Ollama nowhere."""
    env = dict(os.environ if base is None else base)
    for key in path_keys():
        env[f"PACK_PATHS_{key.upper()}"] = str(root / key)
    env["PACK_STATE"] = env["PACK_PATHS_STATE"]  # the older name, which a shell may still set
    env["PACK_CONFIG"] = str(root / "pack.toml")
    env["PACK_PORTS_DASHBOARD"] = str(port)
    env["PACK_OLLAMA_URL"] = NO_OLLAMA
    env.pop("OLLAMA_URL", None)
    env.pop("OLLAMA_HOST", None)
    sock = root / "state" / "continuum" / "broker.sock"
    if len(str(sock).encode()) > SOCKET_MAX:
        sock = Path(tempfile.mkdtemp(prefix="pack-demo-")) / "b.sock"
    env["CONTINUUM_SOCKET"] = str(sock)
    return env


def check_dir(root: Path) -> None:
    """Refuses a demo directory inside the real data directory, or one that holds something other than a demo."""
    real = cfg.data_dir().resolve()
    here = root.resolve()
    if here == real or real in here.parents:
        raise SystemExit(f"pack-demo: {root} is inside the real data directory ({real}); pick another --dir")
    if here.exists() and any(here.iterdir()) and not (here / SEEDED).exists():
        raise SystemExit(f"pack-demo: {root} isn't empty and isn't a demo directory; pick an empty or new --dir")


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


def start_broker(root: Path, env: Mapping[str, str]) -> subprocess.Popen[bytes]:
    """Starts the demo's own broker on `root`'s state directory, its catalog there too."""
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
    process = subprocess.Popen(command, cwd=root, env=dict(env))
    try:
        wait_for_socket(sock, process)
    except BaseException:
        stop(process)
        raise
    return process


def seed_here() -> None:
    """Seeds the demo into the configured state directory (run in the demo's environment, by `run`)."""
    from liberational import memory_log
    from liberational.demo import dataset

    state = cfg.path("paths.state")
    day = dataset.seed(state, memory_log.at(state))
    print(f"pack-demo: seeded the demo household, with a journal for {day.isoformat()}")


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def stop(process: subprocess.Popen[bytes]) -> None:
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
    root.mkdir(parents=True, exist_ok=True)
    env = environment(root, port)
    if not Path(env["PACK_CONFIG"]).exists():
        Path(env["PACK_CONFIG"]).write_text(CONFIG)
    broker = start_broker(root, env)
    try:
        if not (root / SEEDED).exists():
            subprocess.run([sys.executable, "-m", "liberational.demo.cli", "--seed-here"], env=env, check=True)
            (root / SEEDED).write_text("pack-demo\n")
        print(f"pack-demo: the demo dashboard is on http://localhost:{port}/ (its files are in {root})")
        dashboard = subprocess.Popen(
            [sys.executable, "-m", "liberational.dashboard.server", str(port), *dashboard_args], env=env
        )
        try:
            return dashboard.wait()
        except KeyboardInterrupt:
            stop(dashboard)
            return 0
    finally:
        stop(broker)
        sock = Path(env["CONTINUUM_SOCKET"])
        if not sock.is_relative_to(root):
            shutil.rmtree(sock.parent, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> None:
    """pack-demo [--dir DIR] [--port PORT]"""
    parser = argparse.ArgumentParser(prog="pack-demo", description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", type=Path, help="the demo directory (default: a new temporary one)")
    parser.add_argument("--port", type=int, default=PORT, help=f"the dashboard's port (default {PORT})")
    parser.add_argument("--dev", action="store_true", help="run the dashboard with --dev (pages from Vite)")
    parser.add_argument("--seed-here", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.seed_here:
        seed_here()
        return
    root = args.dir or Path(tempfile.mkdtemp(prefix="pack-demo-"))
    sys.exit(run(root, args.port, ["--dev"] if args.dev else []))


if __name__ == "__main__":
    main()
