#!/usr/bin/env python3
"""Take the documentation's screenshots again, from the made-up demo household.

  .venv/bin/python tools/doc_screenshots.py                  # every picture, into docs/images/
  .venv/bin/python tools/doc_screenshots.py demo-journal-day      # only the ones named
  .venv/bin/python tools/doc_screenshots.py --out /tmp/shots # somewhere else, to look before replacing any

It starts `pack-demo` on a free port in a temporary directory, runs tools/doc_screenshots.mjs against it in headless
Chromium, stops the demo, and makes each PNG small (256 colours, which a page of flat colour and text loses nothing
to). tools/doc_screenshots.mjs lists the pictures: its `SHOTS` table has each one's name, colour scheme, and the
clicks that bring the page to what it shows. To add or change a picture, edit that table and run this again.

It needs the frontend built (`mise run frontend`), the frontend's packages (they bring Playwright), and Playwright's
browser (`mise run playwright`, once per machine). Run it on the machine itself, never against a real install: the
pictures are public, so they show the demo household only. Look at every picture before committing it.
"""

import argparse
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
SCRIPT = pathlib.Path(__file__).with_suffix(".mjs")
READY = "the demo dashboard is on "  # pack-demo's address line
START_S = 600.0  # how long the demo may take to seed and start: it draws its own pictures and clips first
COLOURS = 256


def free_port() -> int:
    """A port nothing listens on, for the demo's dashboard."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def listening(port: int) -> bool:
    """Whether something accepts connections on `port` of this machine."""
    with socket.socket() as probe:
        probe.settimeout(1.0)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def wait_ready(demo: subprocess.Popen[bytes], log: pathlib.Path, port: int) -> None:
    """Wait for the demo's address line, which it prints once its dashboard answers on `port`, and for that answer.
    Raises with the log's tail if the demo stops or takes too long."""
    deadline = time.monotonic() + START_S
    while time.monotonic() < deadline:
        if READY in log.read_text(errors="replace") and listening(port):
            return
        if demo.poll() is not None:
            break
        time.sleep(0.5)
    tail = "\n".join(log.read_text(errors="replace").splitlines()[-15:])
    raise SystemExit(f"the demo didn't start:\n{tail}")


def shrink(path: pathlib.Path) -> int:
    """Rewrite a PNG with an adaptive palette, and return its size in bytes."""
    with Image.open(path) as picture:
        small = picture.convert("RGB").quantize(
            colors=COLOURS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
    small.save(path, optimize=True)
    return path.stat().st_size


def take(names: Sequence[str], out: pathlib.Path, frontend: pathlib.Path) -> int:
    """Start the demo, take the pictures, and stop it. Returns the screenshot script's exit code."""
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="doc-shots-") as scratch:
        log = pathlib.Path(scratch) / "demo.log"
        with log.open("wb") as sink:
            demo = subprocess.Popen(
                [sys.executable, "-m", "liberational.demo.cli", "--dir", f"{scratch}/demo", "--port", str(port)],
                stdout=sink,
                stderr=subprocess.STDOUT,
            )
        try:
            wait_ready(demo, log, port)
            shots = subprocess.run(
                ["node", str(SCRIPT), f"http://localhost:{port}", str(out), str(frontend), *names], check=False
            )
        finally:
            demo.terminate()
            try:
                demo.wait(timeout=30)
            except subprocess.TimeoutExpired:
                demo.kill()
    return shots.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="the pictures to take (default: all of them)")
    parser.add_argument("--out", type=pathlib.Path, default=OUT, help="where the PNGs go (default docs/images)")
    parser.add_argument(
        "--frontend",
        type=pathlib.Path,
        default=ROOT / "frontend",
        help="the directory whose node_modules has Playwright",
    )
    args = parser.parse_args(argv)
    before = {p: p.stat().st_mtime_ns for p in args.out.glob("*.png")} if args.out.is_dir() else {}
    code = take(args.names, args.out, args.frontend)
    total = 0
    for path in sorted(args.out.glob("*.png")):
        if before.get(path) == path.stat().st_mtime_ns:
            continue  # not one this run wrote
        size = shrink(path)
        total += size
        print(f"{path.name}: {size / 1024:.0f} kB")
    print(f"total: {total / 1024:.0f} kB")
    return code


if __name__ == "__main__":
    sys.exit(main())
