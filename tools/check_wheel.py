#!/usr/bin/env python3
"""Check that a built wheel carries everything the running system reads that isn't the owner's.

  python3 tools/check_wheel.py dist                                   # the newest wheel in a directory
  python3 tools/check_wheel.py dist/liberational_pack-0.1.0-py3-none-any.whl

`mise run wheel` builds the frontend and the wheel and then runs this, and so does the hosted `wheel` check. It
needs only the standard library, so it runs before anything is installed.

The wheel build takes whatever is under src/liberational/, and the frontend build (dashboard/static/app/) is
gitignored: a wheel built in a clone that never ran `mise run frontend` is complete Python with no dashboard pages.
This is what notices. It checks that the wheel has the settings and service declarations the code reads
(config.RESOURCES), the frontend's manifest, its page, and every file the manifest names, and the commands a
packaged install is set up with. It prints what is missing and exits 1, or says the wheel is complete.
"""

import argparse
import json
import pathlib
import sys
import zipfile
from collections.abc import Collection, Mapping
from typing import Any

PACKAGE = "liberational"
RESOURCES = ("defaults.toml", "labels.toml", "services.toml", "system-packages.toml")
APP = f"{PACKAGE}/dashboard/static/app"
MANIFEST = f"{APP}/.vite/manifest.json"
PAGES = ("index.html",)
COMMANDS = ("pack-config", "pack-dashboard", "pack-models", "pack-services", "pack-supervisor", "pack-system")


def built_files(manifest: Mapping[str, Any]) -> set[str]:
    """Every file Vite's manifest says the build wrote, as paths in the wheel: each entry's own file, its
    stylesheets, and its other assets."""
    out: set[str] = set()
    for entry in manifest.values():
        out.update(f"{APP}/{name}" for name in (entry["file"], *entry.get("css", ()), *entry.get("assets", ())))
    return out


def commands(entry_points: str) -> set[str]:
    """The console script names in a wheel's entry_points.txt."""
    names: set[str] = set()
    section = ""
    for line in entry_points.splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
        elif section == "[console_scripts]" and "=" in line:
            names.add(line.split("=", 1)[0].strip())
    return names


def missing(names: Collection[str], manifest: Mapping[str, Any] | None, scripts: Collection[str]) -> list[str]:
    """What a wheel with these file names, this frontend manifest (None when it has none), and these console
    scripts lacks, one line each; empty when it is complete."""
    out = [f"resource {name}" for name in RESOURCES if f"{PACKAGE}/resources/{name}" not in names]
    if manifest is None:
        out.append(f"the frontend build ({MANIFEST}): run `mise run frontend` before building the wheel")
    else:
        out += [f"frontend page {page}" for page in PAGES if f"{APP}/{page}" not in names]
        out += [f"frontend file {name}" for name in sorted(built_files(manifest) - set(names))]
    out += [f"command {name}" for name in COMMANDS if name not in scripts]
    return out


def newest(path: pathlib.Path) -> pathlib.Path:
    """The wheel to check: `path` itself, or the newest wheel in it when it is a directory."""
    if not path.is_dir():
        return path
    wheels = sorted(path.glob("*.whl"), key=lambda wheel: wheel.stat().st_mtime_ns)
    if not wheels:
        raise SystemExit(f"check_wheel: no wheel in {path}")
    return wheels[-1]


def check(wheel: pathlib.Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read(MANIFEST)) if MANIFEST in names else None
        points = next((name for name in names if name.endswith(".dist-info/entry_points.txt")), None)
        scripts = commands(archive.read(points).decode()) if points else set()
    return missing(names, manifest, scripts)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("wheel", type=pathlib.Path, help="a wheel, or a directory to take the newest wheel from")
    wheel = newest(parser.parse_args().wheel)
    lacking = check(wheel)
    for line in lacking:
        print(f"missing from {wheel.name}: {line}")
    if lacking:
        sys.exit(1)
    with zipfile.ZipFile(wheel) as archive:
        frontend = sum(1 for name in archive.namelist() if name.startswith(APP + "/") and not name.endswith("/"))
    megabytes = wheel.stat().st_size / 1e6
    print(f"{wheel.name}: complete ({len(RESOURCES)} resources, {frontend} frontend files, {megabytes:.1f} MB)")


if __name__ == "__main__":
    main()
