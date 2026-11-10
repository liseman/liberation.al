"""Set up an installed package: what a new machine needs after the package is on it, with nothing built.

  pack-setup               # check the system, install this machine's extras, fetch the models
  pack-setup --check       # only say what is missing; install and download nothing
  pack-setup --no-models   # everything but the models (about 12 GB on a full install)
  pack-setup --no-ollama   # fetch the model files, but pull nothing through Ollama

It does, in order:

1. System packages: lists what this machine is missing and the command that installs it (system_packages.py). It
   installs none of them itself: that takes sudo, and `pack-system install` asks first.
2. Python extras for this machine (extras()): face recognition, object masks, and deep transcription everywhere,
   and voice cloning on Linux with an NVIDIA GPU. MLX on Apple Silicon and CUDA's ONNX Runtime on Linux come with
   the package itself. They are installed with uv into the environment this command runs from, as the extras of the
   package that is already installed, so no index has to carry the package. When the release ships its lock
   (LOCK, a constraints file the release build writes into the package), every version is pinned to it.
3. Continuum: says whether the SDK and the daemons are in this environment. They come with the package as wheels;
   nothing here builds or fetches them.
4. Ollama: says whether it is there and answering, and how to get it. Its models are pulled in step 5 when it is.
5. Models (fetch_models.py): fetches the ones this machine wants and doesn't have. A download that was cut off
   carries on where it stopped, and each says how far it is.

It ends by saying how to open the application. Run again, it installs and downloads nothing it already has. The exit
status is 1 while something the application needs is still missing (a needed system package, the extras, Continuum,
or a model), and 0 otherwise; Ollama alone doesn't fail it, since the application opens without it.

In a checkout, `mise run setup` is the path: it also builds what a checkout builds (whisper-server, the frontend)
and syncs the extras from uv.lock. Here step 2 only says so; the other steps work the same.

LOCK is written by the release workflow (tools/release.py, docs/RELEASES.md). uv is the one beside the interpreter,
else the one on PATH: the .deb's launcher (packaging/deb/launch) builds the environment and links the uv the package
carries there before it runs this, so an apt install needs no uv of its own.
"""

import argparse
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable

from liberational import config as cfg
from liberational import fetch_models, system_packages

DIST = "liberational-pack"
# Every dependency's pinned version, written into the package by the release build. A build without it (a wheel made
# by hand) resolves the extras to the newest versions that fit.
LOCK = cfg.RESOURCES / "constraints.txt"
EVERYWHERE = ("faces", "segmentation", "deep-asr")
DAEMONS = ("continuum-broker", "continuum-capture", "continuum-tier-worker", "continuum-vector-worker")
OPEN = "pack-app"

type Run = Callable[[list[str]], int]


def extras(platform: str | None = None, nvidia: bool | None = None) -> list[str]:
    """The optional extras this machine gets, as `mise run setup` picks them: voice cloning needs CUDA, so it is
    only for Linux x86-64 with an NVIDIA GPU (nvidia-smi on PATH)."""
    platform = platform or cfg.platform_key()
    nvidia = shutil.which("nvidia-smi") is not None if nvidia is None else nvidia
    return [*EVERYWHERE, *(["voice-clone"] if nvidia and platform == "linux-x86_64" else [])]


def uv() -> str | None:
    """The uv to install with: beside the interpreter (a package that bundles it), else on PATH."""
    beside = cfg.program("uv")
    return str(beside) if beside.is_file() else shutil.which("uv")


def extras_command(tool: str, chosen: list[str], version: str | None = None) -> list[str]:
    """The command that installs the extras into this environment: the installed package again, at the version it
    has, with the extras named, so uv reads their requirements from the installed package and fetches only them."""
    version = version or importlib.metadata.version(DIST)
    command = [tool, "pip", "install", "--python", sys.executable, f"{DIST}[{','.join(chosen)}]=={version}"]
    return [*command, "--constraint", str(LOCK)] if LOCK.is_file() else command


def _run(command: list[str]) -> int:
    try:
        return subprocess.run(command).returncode
    except OSError as error:
        print(f"  can't run {command[0]}: {error}")
        return 127


def system_step() -> bool:
    """Lists the missing system packages and how to install them. True when no needed one is missing."""
    print("System packages:")
    gone = system_packages.missing(system_packages.load())
    if not gone:
        print("  nothing missing")
        return True
    print(system_packages.describe(gone))
    return all(p.optional for p in gone)


def extras_step(check: bool, run: Run = _run) -> bool:
    """Installs this machine's extras (or, with `check`, says which). True when they are in, or when there is
    nothing for this command to do (a checkout)."""
    chosen = extras()
    print(f"Python extras for this machine: {', '.join(chosen)}")
    if cfg.CHECKOUT is not None:
        print("  a checkout syncs them from uv.lock: `mise run setup`")
        return True
    tool = uv()
    command = extras_command(tool or "uv", chosen)
    if check:
        print(f"  would run: {' '.join(command)}")
        return True
    if tool is None:
        print(f"  uv isn't installed (https://docs.astral.sh/uv/). With it, run: {' '.join(command)}")
        return False
    print(f"  $ {' '.join(command)}", flush=True)
    if code := run(command):
        print(f"  that failed (exit {code}); run pack-setup again once it's fixed")
        return False
    return True


def continuum_step() -> bool:
    """Says whether Continuum's SDK and daemons are in this environment. True when they all are."""
    print("Continuum:")
    gone = [name for name in DAEMONS if not cfg.program(name).is_file()]
    if importlib.util.find_spec("continuum_py") is None:
        gone.insert(0, "the Python SDK (continuum_py)")
    if not gone:
        print("  the SDK and the daemons are here")
        return True
    how = (
        "build them with `mise run setup`, then `mise run continuum-release`"
        if cfg.CHECKOUT is not None
        else "they come with the package's Continuum wheels: install the release again"
    )
    print(f"  missing from {cfg.BIN}: {', '.join(gone)}\n  {how}")
    return False


def ollama_answers(timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(cfg.ollama_url() + "/api/version", timeout=timeout):
            return True
    except urllib.error.URLError, OSError, ValueError:
        return False


def ollama_step() -> bool:
    """Says whether Ollama is installed and answering, and how to get it. True when it is installed."""
    print("Ollama (the local language and vision models):")
    if not shutil.which("ollama"):
        print(f"  not installed: {fetch_models.ollama_hint()}, then run pack-setup again to pull its models")
        return False
    if ollama_answers():
        print(f"  installed, and answering at {cfg.ollama_url()}")
    else:
        print(
            f"  installed, but not answering at {cfg.ollama_url()}: the application starts it. To pull its models "
            "before that, start it (`ollama serve`) and run pack-setup again."
        )
    return True


def models_step(check: bool, ollama: bool) -> bool:
    """Fetches the models (or, with `check`, lists them). True when every one this machine wants is here: not when
    a gated one was skipped, or when a download stopped (what arrived is kept, and the next run carries on)."""
    print("Models:")
    try:
        gated = fetch_models.fetch(ollama=ollama, listing=check)
    except OSError as error:  # the network, or a full disk
        print(f"  stopped: {error}\n  What was downloaded is kept: run pack-setup again to carry on.")
        return False
    if gated:
        print(fetch_models.gated_help(gated))
    return not gated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only say what is missing; install and download nothing")
    parser.add_argument("--no-models", action="store_true", help="skip the models")
    parser.add_argument("--no-ollama", action="store_true", help="pull nothing through Ollama")
    a = parser.parse_args(argv)
    left = []
    if not system_step():
        left.append("install the system packages above")
    if not extras_step(a.check):
        left.append("install the Python extras")
    if not continuum_step():
        left.append("get Continuum into this environment")
    ollama_step()
    if not a.no_models and not models_step(a.check, ollama=not a.no_ollama):
        left.append("fetch the rest of the models")
    print()
    if left:
        print("Still to do: " + "; ".join(left) + ". Then run pack-setup again.")
    else:
        print("Checked: nothing was installed or downloaded." if a.check else "Set up.")
    print(f"Open the application with: {OPEN}")
    raise SystemExit(1 if left else 0)


if __name__ == "__main__":
    main()
