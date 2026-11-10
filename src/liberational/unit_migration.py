"""Move an install from one OS service per daemon to the one application (#165, part of #151).

  pack-services migrate --dry-run   # print exactly what would be done here, and do nothing
  pack-services migrate             # stop the per-service units, in order, and remove them

Earlier installs ran each service as a systemd user unit (`~/.config/systemd/user/pack-<name>.service`, and a
`.timer` for a scheduled job) or a launchd agent (`~/Library/LaunchAgents/pack.<name>.plist`). The application
(pack-app, or pack-supervisor run) runs the same services as its own children and registers nothing with the OS, and
it refuses to start while the old ones run. This removes them, once:

1. Stop them in the order a quit uses, so what capture recorded reaches the store: the readers, then capture, then
   the tier worker once Tier 0 is flushed (asked of the broker, for at most the cascade's DRAIN_TIMEOUT; a check that
   can't be made is skipped), then the broker, then the dashboard and the gateway.
2. Disable them and delete their unit files, and have the service manager forget them.
3. Say whether linger is on (`loginctl enable-linger` made the units start at boot). Nothing of the application
   needs it. `--disable-linger` turns it off; without the flag it is only reported, since other user services may
   want it.

Nothing under the state directory, `pack.toml`, or `config/devices.toml` is touched: the application reads the same
files from the same checkout. It knows the units by their file names alone, so it works after pack-services has
stopped rendering them. Running it again when nothing is left does nothing.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from liberational import config as cfg
from liberational import logs, service_cascade
from liberational.service_cascade import BROKER, CAPTURE, DIAGNOSTICS, TIER

type Run = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def systemd_dir() -> Path:
    return Path.home() / ".config/systemd/user"


def launchd_dir() -> Path:
    return Path.home() / "Library/LaunchAgents"


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), capture_output=True, text=True, timeout=120, check=False)


@dataclass
class Plan:
    """What a migration would do here: the commands, in order, and the files it would delete."""

    darwin: bool
    units: dict[str, list[Path]] = field(default_factory=dict)  # service name -> its unit files
    dropins: list[Path] = field(default_factory=list)  # `pack-<name>.service.d` override directories beside them
    linger: bool | None = None  # whether linger is on for this user; None where there is no such thing
    disable_linger: bool = False

    def stages(self) -> list[tuple[str, list[str]]]:
        """The stop order: readers, capture, (drain), the tier worker, the broker, then the diagnostics."""
        names = sorted(self.units)
        readers = [n for n in names if n not in DIAGNOSTICS and n not in {BROKER, CAPTURE, TIER}]
        return [
            ("readers", readers),
            ("capture", [n for n in names if n == CAPTURE]),
            ("the tier worker", [n for n in names if n == TIER]),
            ("the broker", [n for n in names if n == BROKER]),
            ("the dashboard and the gateway", [n for n in names if n in DIAGNOSTICS]),
        ]

    def stop_command(self, names: Sequence[str]) -> list[list[str]]:
        if self.darwin:
            return [["launchctl", "bootout", f"gui/{os.getuid()}/pack.{name}"] for name in names]
        units = [path.name for name in names for path in self.units[name]]
        return [["systemctl", "--user", "stop", *units]]

    def after_stop(self) -> list[list[str]]:
        """Forget the units: disable them (systemd), then reload once their files are gone."""
        if self.darwin:
            return []
        units = [path.name for paths in self.units.values() for path in paths]
        return [["systemctl", "--user", "disable", *units]] if units else []

    def reload(self) -> list[list[str]]:
        return [] if self.darwin or not self.units else [["systemctl", "--user", "daemon-reload"]]

    def linger_command(self) -> list[str] | None:
        return ["loginctl", "disable-linger", getpass.getuser()] if self.linger and self.disable_linger else None

    def lines(self) -> list[str]:
        """The plan in words and exact commands, as --dry-run prints it."""
        if not self.units:
            out = ["No per-service units are installed here: nothing to migrate."]
        else:
            files = [path for paths in self.units.values() for path in paths]
            out = [f"{len(self.units)} services run as {len(files)} unit files here. In order:"]
            for stage, names in self.stages():
                if stage == "the tier worker" and names and CAPTURE in self.units:
                    out.append("  wait for the tier worker to flush what capture left (at most 20 s)")
                for command in self.stop_command(names) if names else []:
                    out.append(f"  stop {stage}: {' '.join(command)}")
            out += [f"  {' '.join(command)}" for command in self.after_stop()]
            out += [f"  delete {path}" for path in [*files, *self.dropins]]
            out += [f"  {' '.join(command)}" for command in self.reload()]
        if self.linger:
            command = self.linger_command()
            out.append(
                f"  {' '.join(command)}"
                if command
                else "Linger is on for this user (units started at boot). The application doesn't need it; "
                "`pack-services migrate --disable-linger` turns it off, or leave it for other user services."
            )
        out.append("Nothing under the state directory, pack.toml, or config/devices.toml is touched.")
        out.append("Then open the application: pack-app (or pack-supervisor run, without the window).")
        return out


def find(disable_linger: bool = False, run: Run | None = None) -> Plan:
    """The per-service units installed for this user, by their file names."""
    run = run or _run
    darwin = sys.platform == "darwin"
    units: dict[str, list[Path]] = {}
    if darwin:
        for path in sorted(launchd_dir().glob("pack.*.plist")):
            units.setdefault(path.name.removeprefix("pack.").removesuffix(".plist"), []).append(path)
        return Plan(darwin, units)
    for path in sorted([*systemd_dir().glob("pack-*.service"), *systemd_dir().glob("pack-*.timer")]):
        units.setdefault(path.stem.removeprefix("pack-"), []).append(path)
    dropins = sorted(path for path in systemd_dir().glob("pack-*.service.d") if path.is_dir())
    try:
        shown = run(["loginctl", "show-user", getpass.getuser(), "--property=Linger"])
        linger: bool | None = shown.stdout.strip() == "Linger=yes" if shown.returncode == 0 else None
    except OSError, subprocess.SubprocessError:
        linger = None
    return Plan(darwin, units, dropins, linger, disable_linger)


def migrate(
    plan: Plan,
    run: Run | None = None,
    say: Callable[[str], None] = print,
    drain: Callable[[float], int] | None = None,
) -> bool:
    """Carry the plan out. Returns whether every step worked; a step that fails is said, and the rest still run, so
    a second run finishes what is left."""
    ok = True
    run = run or _run

    def step(command: Sequence[str], tolerate: bool = False) -> None:
        nonlocal ok
        try:
            done = run(command)
            failed = done.returncode != 0 and not tolerate
            detail = (done.stderr or done.stdout).strip()[-300:]
        except (OSError, subprocess.SubprocessError) as error:
            failed, detail = True, logs.reason(error)
        say(f"{' '.join(command)}: {'FAILED ' + detail if failed else 'done'}")
        ok = ok and not failed

    for stage, names in plan.stages():
        if stage == "the tier worker" and names and CAPTURE in plan.units and drain is not None:
            try:
                left = drain(time.monotonic() + service_cascade.DRAIN_TIMEOUT)
                say("Tier 0: " + (f"{left} payloads were not flushed; going on" if left else "flushed"))
            except Exception as error:
                say(f"Tier 0: couldn't check ({logs.reason(error)}); going on")
        for command in plan.stop_command(names) if names else []:
            # launchd reports a job that isn't loaded as an error: it is stopped either way
            step(command, tolerate=plan.darwin)
    for command in plan.after_stop():
        step(command)
    for paths in plan.units.values():
        for path in paths:
            try:
                path.unlink(missing_ok=True)
                say(f"deleted {path}")
            except OSError as error:
                say(f"delete {path}: FAILED {error}")
                ok = False
    for directory in plan.dropins:
        shutil.rmtree(directory, ignore_errors=True)
        say(f"deleted {directory}")
    for command in plan.reload():
        step(command)
    if (command := plan.linger_command()) is not None:
        step(command)
    return ok


def main(dry_run: bool, disable_linger: bool) -> None:
    """`pack-services migrate`."""
    plan = find(disable_linger)
    if dry_run:
        print("Dry run: nothing is changed.")
        print(*plan.lines(), sep="\n")
        return
    if not plan.units and plan.linger_command() is None:
        print(*plan.lines(), sep="\n")
        return
    drain = None
    try:
        ready = service_cascade.SDKReadiness(cfg.devices_file(), cfg.path("paths.state") / "continuum/broker.sock")
        drain = ready.drain
    except Exception as error:  # no manifest or SDK here: stop without the flush check
        print(f"Tier 0 can't be checked here ({logs.reason(error)}); stopping without the wait")
    if not migrate(plan, drain=drain):
        raise SystemExit("some steps failed (above); fix what they name and run `pack-services migrate` again")
    print("The per-service units are gone. Open the application: pack-app")
