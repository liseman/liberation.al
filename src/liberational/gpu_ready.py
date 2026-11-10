"""Wait for the NVIDIA driver before execing a GPU service: `python -m liberational.gpu_ready -- <command>`.

Service renderers opt in through services.toml's `wait_for_gpu`. A program that picks its device once at startup,
like `ollama serve`, falls back to the CPU for good when it starts before the driver is up: on the Linux machine
the user services can start at boot before the driver loads ten minutes later, leaving Ollama running every model on the CPU until
it is restarted. This waits until `nvidia-smi -L` lists a GPU, then replaces itself with the service so signals and
exit status still belong to the service manager.

A machine with no `nvidia-smi` (a Mac, a CPU-only box) execs at once: there's no driver to wait for. When the wait
runs out, the service starts anyway with a warning, since a CPU server beats none; a later restart picks the GPU up.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence

from liberational import logs

LOG = logs.get("gpu-ready")
WAIT_S = 900.0  # the driver can take 10 minutes to load after boot on the Linux machine
POLL_S = 5.0


def gpus(smi: str) -> list[str]:
    """The GPUs `nvidia-smi -L` lists, or none while the driver isn't up."""
    try:
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=30, check=False)
    except OSError, subprocess.TimeoutExpired:
        return []
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line.startswith("GPU ")]


def wait_for_gpu(
    *,
    which: Callable[[str], str | None] = shutil.which,
    probe: Callable[[str], list[str]] = gpus,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    wait_s: float = WAIT_S,
) -> bool:
    """True once a GPU answers, or when there's no NVIDIA tooling to wait for; False when the wait ran out."""
    smi = which("nvidia-smi")
    if smi is None:
        return True
    deadline = clock() + wait_s
    waited = False
    while True:
        if found := probe(smi):
            if waited:
                LOG.info("the NVIDIA driver is up (%d GPUs); starting", len(found))
            return True
        if clock() >= deadline:
            LOG.warning("no GPU after %g s; starting anyway, so it may run on the CPU until restarted", wait_s)
            return False
        if not waited:
            LOG.info("waiting for the NVIDIA driver before starting")
            waited = True
        sleep(POLL_S)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a service command is required")
    wait_for_gpu()
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
