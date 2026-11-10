# /// script
# requires-python = ">=3.14"
# dependencies = ["packaging"]
# ///
"""What the release workflow (.github/workflows/release.yml) needs worked out: the version a tag names, the locks, and
the release's notes. `uv run tools/release.py <command>` brings its one dependency with it.

    release.py version v0.2.0                       # the tag's version, if pyproject.toml has the same one
    release.py constraints universal.txt out.txt    # the lock the package carries, for pack-setup's extras
    release.py lock universal.txt linux-x86_64 out  # one platform's lock, with hashes
    release.py notes 0.2.0                          # that version's section of docs/RELEASES.md

The locks come from `uv export` of uv.lock with every extra: one requirements file for every platform, each line
with the marker that says where it applies, and each package's hashes.

- `constraints` drops the hashes and keeps the markers. The release build writes it into the package as
  `resources/constraints.txt`, where `pack-setup` pins this machine's extras to it.
- `lock` keeps one platform's lines (PLATFORMS) and their hashes, and drops the markers it has just answered. It is
  a release asset: `uv tool install --constraints` pins and checks every dependency against it.

`notes` refuses a version with no section in docs/RELEASES.md, or one whose section doesn't say what an upgrade
resets: this project resets machine-made data now and then, and a release has to say so.
"""

import argparse
import dataclasses
import re
import sys
import tomllib
from pathlib import Path

from packaging.markers import Marker, default_environment

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "docs" / "RELEASES.md"
TAG = re.compile(r"v(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)")
RESETS = "resets"

# The marker values that tell the release's platforms apart; every other one is the same on all three.
PLATFORMS: dict[str, dict[str, str]] = {
    "linux-x86_64": {"sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64"},
    "linux-aarch64": {"sys_platform": "linux", "platform_system": "Linux", "platform_machine": "aarch64"},
    "macos-arm64": {"sys_platform": "darwin", "platform_system": "Darwin", "platform_machine": "arm64"},
}
PYTHON = {
    "os_name": "posix",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "python_version": "3.14",
    "python_full_version": "3.14.0",
    "implementation_version": "3.14.0",
}


@dataclasses.dataclass(frozen=True)
class Pin:
    """One package of an exported lock: `name==version`, where it applies, and its files' hashes."""

    requirement: str
    marker: str | None
    hashes: tuple[str, ...]


def pins(exported: str) -> list[Pin]:
    """The packages of a `uv export` requirements file, in its order. Comments are dropped."""
    out: list[Pin] = []
    for raw in exported.splitlines():
        line = raw.strip().removesuffix("\\").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--hash="):
            if not out:
                raise ValueError(f"a hash before any package: {line}")
            out[-1] = dataclasses.replace(out[-1], hashes=(*out[-1].hashes, line.removeprefix("--hash=")))
        elif "==" in line:
            requirement, _, marker = line.partition(";")
            out.append(Pin(requirement.strip(), marker.strip() or None, ()))
        else:
            raise ValueError(f"not a pinned package or a hash: {line}")
    return out


def constraints(exported: str) -> str:
    """Every platform's pins with their markers and no hashes: what the package carries."""
    return "".join(f"{p.requirement} ; {p.marker}\n" if p.marker else f"{p.requirement}\n" for p in pins(exported))


def lock(exported: str, platform: str) -> str:
    """One platform's pins, with hashes. A package with no hashes is refused: the lock exists to check them."""
    environment = {**default_environment(), **PYTHON, **PLATFORMS[platform]}
    lines: list[str] = []
    for pin in pins(exported):
        if pin.marker and not Marker(pin.marker).evaluate(environment):
            continue
        if not pin.hashes:
            raise ValueError(f"{pin.requirement} has no hashes in the export")
        lines.append(" \\\n".join([pin.requirement, *(f"    --hash={h}" for h in pin.hashes)]))
    return "\n".join(lines) + "\n"


def version(tag: str, pyproject: str) -> str:
    """The version `tag` names (`v0.2.0` is 0.2.0), which must be the one pyproject.toml declares."""
    found = TAG.fullmatch(tag)
    if found is None:
        raise ValueError(f"{tag} isn't a release tag: it should look like v0.2.0 or v0.2.0rc1")
    declared = tomllib.loads(pyproject)["project"]["version"]
    if found[1] != declared:
        raise ValueError(f"{tag} names version {found[1]}, but pyproject.toml says {declared}: set it, then tag")
    return declared


def notes(wanted: str, releases: str) -> str:
    """The section of docs/RELEASES.md for one version: the text under its `## <version>` heading, which has to say
    what an upgrade resets (a paragraph that uses the word, even to say "nothing")."""
    sections = re.split(r"^## +", releases, flags=re.MULTILINE)[1:]
    for section in sections:
        heading, _, body = section.partition("\n")
        if heading.strip().split()[0].removeprefix("v") == wanted:
            if RESETS not in body.lower():
                raise ValueError(f"the notes for {wanted} don't say what an upgrade {RESETS}")
            return body.strip() + "\n"
    raise ValueError(f"docs/RELEASES.md has no section for {wanted}: add `## {wanted}` with its notes")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="the version a tag names").add_argument("tag")
    for name, text in (("constraints", "the lock the package carries"), ("lock", "one platform's lock")):
        command = commands.add_parser(name, help=text)
        command.add_argument("exported", type=Path, help="uv export's requirements file")
        if name == "lock":
            command.add_argument("platform", choices=sorted(PLATFORMS))
        command.add_argument("out", type=Path)
    commands.add_parser("notes", help="a version's release notes").add_argument("version")
    args = parser.parse_args()
    try:
        if args.command == "version":
            print(version(args.tag, (ROOT / "pyproject.toml").read_text()))
        elif args.command == "constraints":
            args.out.write_text(constraints(args.exported.read_text()))
        elif args.command == "lock":
            args.out.write_text(lock(args.exported.read_text(), args.platform))
        else:
            sys.stdout.write(notes(args.version, NOTES.read_text()))
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
