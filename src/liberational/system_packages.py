"""The programs and libraries Liberational needs from the operating system, as config/system-packages.toml declares
them: which are missing here, and the commands that install them.

  uv run pack-system check              # what's missing, and the commands that install it (exit 1 if a needed one is)
  uv run pack-system check --all        # every package this machine could need, and whether it's here
  uv run pack-system install            # install the needed ones that are missing, after asking (mise run setup)
  uv run pack-system install --optional # the optional ones too
  uv run pack-system install --yes      # without asking

Nothing is installed without asking: without --yes and without a terminal to ask on, install only prints the commands.
It never fails setup for something it can't install itself (an unknown package manager): it says what to install and
returns 0. A package is found by its command on PATH, by its library loading (ctypes.util.find_library, plus
Homebrew's folders on macOS, which find_library doesn't search), or by `pkg-config --exists` for a development
package whose headers matter (a runtime library can load while its headers are missing). On Linux the installer is
apt-get (through sudo, which asks for the password itself); on macOS it's Homebrew, when it's installed. Packages mise.toml provides are
fixed with `mise install` instead.
"""

import argparse
import ctypes.util
import dataclasses
import pathlib
import shlex
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from liberational import config as cfg

BREW_LIBS = ("/opt/homebrew/lib", "/usr/local/lib")  # where Homebrew puts libraries; find_library skips them
DETECT = "detect"  # commands(): use the system's own package manager
WHEN = {"nvidia": lambda: shutil.which("nvidia-smi") is not None}  # conditions a package can be declared under


@dataclasses.dataclass(frozen=True)
class Package:
    name: str
    why: str
    command: str | None = None
    library: str | None = None
    pkgconfig: str | None = None  # a pkg-config module: present when `pkg-config --exists` says so
    apt: str | None = None
    brew: str | None = None
    mise: bool = False
    services: tuple[str, ...] = ()
    optional: bool = False
    when: str | None = None  # a key of WHEN: only on machines where that holds


def _text(d: dict[str, Any], key: str) -> str | None:
    v = d.get(key)
    if v is not None and (not isinstance(v, str) or not v.strip()):
        raise ValueError(f"{key} must be text")
    return v


def load(path: pathlib.Path | None = None) -> list[Package]:
    """The declared packages, resolving the current root on each call unless a path is supplied.

    Raises ValueError naming the package when a table doesn't follow the format. Do not capture ROOT in a default:
    importing this module while configuration is overridden must not pin later reads to that old checkout.
    """
    if path is None:
        path = cfg.ROOT / "config" / "system-packages.toml"
    known = {f.name for f in dataclasses.fields(Package)} - {"name"}
    out = []
    for name, d in tomllib.loads(path.read_text()).items():
        try:
            if not isinstance(d, dict):
                raise ValueError("a package is a table")
            if unknown := sorted(set(d) - known):
                raise ValueError(f"unknown fields {unknown}")
            why, command, library = _text(d, "why"), _text(d, "command"), _text(d, "library")
            pkgconfig = _text(d, "pkgconfig")
            apt, brew, when = _text(d, "apt"), _text(d, "brew"), _text(d, "when")
            mise, optional, services = d.get("mise", False), d.get("optional", False), d.get("services", [])
            if not why:
                raise ValueError("say why it's needed")
            if int(bool(command)) + int(bool(library)) + int(bool(pkgconfig)) != 1:
                raise ValueError("give one of a command, a library, or a pkg-config name to check")
            if not isinstance(mise, bool) or not isinstance(optional, bool):
                raise ValueError("mise and optional are true or false")
            if not (apt or brew or mise):
                raise ValueError("say where it comes from (apt, brew, or mise)")
            if not isinstance(services, list) or not all(isinstance(s, str) for s in services):
                raise ValueError("services is a list of service names")
            if when is not None and when not in WHEN:
                raise ValueError(f"when must be one of {sorted(WHEN)}")
        except ValueError as e:
            raise ValueError(f"{name}: {e}") from None
        out.append(Package(name, why, command, library, pkgconfig, apt, brew, mise, tuple(services), optional, when))
    return out


def _platform() -> str:
    return sys.platform


def manager(platform: str | None = None, which: Callable[[str], str | None] | None = None) -> str | None:
    """The package manager this knows and the system has: 'brew' on macOS, 'apt' on Linux, else None."""
    platform, which = platform or _platform(), which or shutil.which
    if platform == "darwin":
        return "brew" if which("brew") else None
    if platform.startswith("linux") and which("apt-get"):
        return "apt"
    return None


def applies(p: Package, platform: str | None = None) -> bool:
    """Whether this machine needs the package: mise ones everywhere, others where they have a name for its installer,
    and only where its `when` holds."""
    platform = platform or _platform()
    if p.when and not WHEN[p.when]():
        return False
    return p.mise or bool(p.brew if platform == "darwin" else p.apt)


def _library(name: str, platform: str) -> bool:
    if ctypes.util.find_library(name):
        return True
    return platform == "darwin" and any(pathlib.Path(d, f"lib{name}.dylib").exists() for d in BREW_LIBS)


def _pkgconfig(name: str) -> bool:
    if not shutil.which("pkg-config"):
        return False
    return subprocess.run(["pkg-config", "--exists", name], check=False, capture_output=True).returncode == 0


def present(
    p: Package,
    platform: str | None = None,
    which: Callable[[str], str | None] | None = None,
    library: Callable[[str, str], bool] | None = None,
    pkgconfig: Callable[[str], bool] | None = None,
) -> bool:
    if p.command:
        return (which or shutil.which)(p.command) is not None
    if p.pkgconfig:
        return (pkgconfig or _pkgconfig)(p.pkgconfig)
    return (library or _library)(p.library or "", platform or _platform())


def missing(
    packages: Iterable[Package],
    services: Sequence[str] | None = None,
    platform: str | None = None,
    which: Callable[[str], str | None] | None = None,
    library: Callable[[str, str], bool] | None = None,
    optional: bool = True,
    pkgconfig: Callable[[str], bool] | None = None,
) -> list[Package]:
    """The packages this machine needs and doesn't have: only those some of `services` use, when it's given, and
    only the required ones unless optional."""
    return [
        p
        for p in packages
        if applies(p, platform)
        and (optional or not p.optional)
        and (services is None or set(p.services) & set(services))
        and not present(p, platform, which, library, pkgconfig)
    ]


def commands(packages: Sequence[Package], platform: str | None = None, installer: str | None = DETECT) -> list[str]:
    """The commands that install these packages: for the package manager (the system's own, unless installer names
    one, or None for none), and `mise install` for the ones mise provides."""
    installer = manager(platform) if installer == DETECT else installer
    out = []
    names = [(p.brew if installer == "brew" else p.apt) for p in packages if not p.mise]
    names = [n for n in dict.fromkeys(names) if n]
    if names and installer == "apt":
        out += ["sudo apt-get update", "sudo apt-get install -y " + " ".join(names)]  # a fresh image has no lists yet
    elif names and installer == "brew":
        out.append("brew install " + " ".join(names))
    if any(p.mise for p in packages):
        out.append("mise install")
    return out


def describe(packages: Sequence[Package], platform: str | None = None, installer: str | None = DETECT) -> str:
    """Lines naming each missing package and why, then the commands that install them."""
    lines = [f"  {p.name}{' (optional)' if p.optional else ''}: {p.why}" for p in packages]
    fix = commands(packages, platform, installer)
    if fix:
        lines.append("Install with: " + " && ".join(fix))
    if any(not p.mise for p in packages) and not any(not c.startswith("mise") for c in fix):
        lines.append("Install them with your system's package manager (Liberational knows apt-get and Homebrew).")
    return "\n".join(lines)


def status(
    packages: Iterable[Package] | None = None, services: Sequence[str] | None = None, platform: str | None = None
) -> dict[str, Any]:
    """For the dashboard: the needed packages the given services (all, when None) use and this machine is missing,
    each with the command that installs it alone, and the commands that install them all. Optional ones are left out:
    they'd show forever where they don't apply."""
    gone = missing(load() if packages is None else packages, services=services, platform=platform, optional=False)
    return {
        "missing": [
            {
                "name": p.name,
                "why": p.why,
                "optional": p.optional,
                "services": list(p.services),
                "command": " && ".join(commands([p], platform)),
            }
            for p in gone
        ],
        "commands": commands(gone, platform),
    }


def _run(argv: list[str]) -> int:
    try:
        return subprocess.run(argv).returncode
    except OSError as e:
        print(f"pack-system: can't run {argv[0]}: {e}", file=sys.stderr)
        return 127


def install(
    packages: Sequence[Package],
    yes: bool = False,
    platform: str | None = None,
    interactive: Callable[[], bool] = lambda: sys.stdin.isatty(),
    ask: Callable[[str], str] = input,
    run: Callable[[list[str]], int] = _run,
) -> int:
    """Install these packages: show the commands, ask (unless yes), and run them. Returns the first failing exit code,
    else 0. Without a terminal to ask on (and without yes), or with no installer it knows, it only says what to do."""
    if not packages:
        print("System packages: nothing missing.")
        return 0
    print("Missing system packages:\n" + describe(packages, platform))
    fix = commands(packages, platform)
    if not fix:
        return 0  # nothing it can run here; it has said what to install, and setup goes on
    if not yes:
        if not interactive():
            print("Not installing without a terminal to ask on. Run the commands above, or pack-system install --yes.")
            return 0
        if ask("Run them now? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Skipped. Run the commands above when you're ready.")
            return 0
    for command in fix:
        print(f"$ {command}", flush=True)
        if code := run(shlex.split(command)):
            print(f"pack-system: that failed (exit {code})", file=sys.stderr)
            return code
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="list what's missing and how to install it")
    c.add_argument("--all", action="store_true", help="list every package this machine could need, found or not")
    i = sub.add_parser("install", help="install the needed packages that are missing, after asking")
    i.add_argument("--yes", action="store_true", help="don't ask")
    i.add_argument("--optional", action="store_true", help="the optional packages too")
    a = ap.parse_args(argv)
    packages = load()
    if a.cmd == "install":
        raise SystemExit(install(missing(packages, optional=a.optional), yes=a.yes))
    gone = missing(packages)
    if a.all:
        for p in packages:
            if applies(p):
                print(f"  {'missing' if p in gone else 'found  '}  {p.name}{' (optional)' if p.optional else ''}")
    print(("Missing system packages:\n" + describe(gone)) if gone else "System packages: nothing missing.")
    raise SystemExit(1 if any(not p.optional for p in gone) else 0)


if __name__ == "__main__":
    main()
