"""Packs Continuum's daemons into a wheel, and adds license files to a wheel that's already built.

A wheel's `<name>-<version>.data/scripts/` files are installed into the environment's `bin/`, so the daemons land on
the `PATH` beside `python`, with no Rust toolchain on the machine. maturin can only do that for one crate's binaries,
and the daemons come from five crates, so `packaging/build.sh` builds them with Cargo and this writes the archive:

    pack_wheel.py pack --bin-dir target/release --out dist --version 0.1.0 --platform manylinux_2_36_x86_64 \
        --licenses <prefix>/licenses continuum-broker continuum-capture ...
    pack_wheel.py licenses dist/continuum_py-0.1.0-cp314-cp314-manylinux_2_36_x86_64.whl <prefix>/licenses

A `manylinux_X_Y` platform is a promise that every binary runs on glibc X.Y, so `pack` reads each binary's symbol
versions (objdump) and refuses a binary that needs a newer one. Both wheels link FFmpeg statically, so both carry its
license and the ones of what it was built with (`licenses`, for the SDK wheel maturin made).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

NAME = "continuum_daemons"
SUMMARY = "Continuum Broker's daemons (broker, capture, tier worker, vector worker, gateway, export) as programs"
# The repository's code is Apache-2.0; the binaries statically link FFmpeg (LGPL), OpenH264 (BSD), and NVIDIA's
# codec headers (MIT).
LICENSE = "Apache-2.0 AND LGPL-2.1-or-later AND BSD-2-Clause AND MIT"
EXECUTABLE = 0o100755 << 16
REGULAR = 0o100644 << 16
GLIBC_SYMBOL = re.compile(r"GLIBC_(\d+)\.(\d+)")
MANYLINUX = re.compile(r"manylinux_(\d+)_(\d+)_")


def glibc_needed(dynamic_symbols: str) -> tuple[int, int]:
    """The newest glibc version among `objdump -T`'s symbol versions, or (0, 0) when there's none."""
    return max(((int(major), int(minor)) for major, minor in GLIBC_SYMBOL.findall(dynamic_symbols)), default=(0, 0))


def glibc_promised(platform: str) -> tuple[int, int] | None:
    """The glibc a `manylinux_X_Y_<arch>` platform tag promises, or None for another platform (macOS)."""
    found = MANYLINUX.match(platform)
    return (int(found[1]), int(found[2])) if found else None


def check_glibc(binary: Path, platform: str) -> None:
    """Exit with a message when `binary` needs a newer glibc than `platform` promises."""
    promised = glibc_promised(platform)
    if promised is None:
        return
    symbols = subprocess.run(["objdump", "-T", str(binary)], capture_output=True, text=True, check=True).stdout
    needed = glibc_needed(symbols)
    if needed > promised:
        sys.exit(f"{binary.name} needs glibc {needed[0]}.{needed[1]}, newer than {platform} promises")


def record_line(name: str, data: bytes) -> str:
    """One RECORD line: the path, its SHA-256 as unpadded URL-safe base64, and its size."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def metadata(version: str, license_files: list[str]) -> str:
    """The daemons wheel's METADATA."""
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {NAME.replace('_', '-')}",
        f"Version: {version}",
        f"Summary: {SUMMARY}",
        f"License-Expression: {LICENSE}",
        *(f"License-File: {name}" for name in license_files),
    ]
    return "\n".join(lines) + "\n"


def write_wheel(path: Path, files: list[tuple[str, bytes, int]], dist_info: str) -> None:
    """Write `files` (name, bytes, zip mode bits) and their RECORD as the wheel at `path`. Entries carry no build
    time, so the same input gives the same archive."""
    record = f"{dist_info}/RECORD"
    lines = [record_line(name, data) for name, data, _ in files] + [f"{record},,"]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data, mode in [*files, (record, ("\n".join(lines) + "\n").encode(), REGULAR)]:
            info = zipfile.ZipInfo(name)
            info.external_attr = mode
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)


def license_entries(dist_info: str, licenses: Path | None) -> list[tuple[str, bytes, int]]:
    """The files of `licenses`, as entries under the wheel's `<dist-info>/licenses/`."""
    if licenses is None:
        return []
    return [
        (f"{dist_info}/licenses/{path.name}", path.read_bytes(), REGULAR)
        for path in sorted(licenses.iterdir())
        if path.is_file()
    ]


def pack(bin_dir: Path, out: Path, version: str, platform: str, licenses: Path | None, names: list[str]) -> Path:
    """Write the daemons wheel for `platform` from the binaries `names` in `bin_dir`, and return its path."""
    dist_info = f"{NAME}-{version}.dist-info"
    files: list[tuple[str, bytes, int]] = []
    for name in names:
        binary = bin_dir / name
        check_glibc(binary, platform)
        files.append((f"{NAME}-{version}.data/scripts/{name}", binary.read_bytes(), EXECUTABLE))
    notices = license_entries(dist_info, licenses)
    tag = f"py3-none-{platform}"
    wheel = f"Wheel-Version: 1.0\nGenerator: continuum-packaging\nRoot-Is-Purelib: false\nTag: {tag}\n"
    files += [
        (
            f"{dist_info}/METADATA",
            metadata(version, [name.rsplit("/", 1)[1] for name, _, _ in notices]).encode(),
            REGULAR,
        ),
        (f"{dist_info}/WHEEL", wheel.encode(), REGULAR),
        *notices,
    ]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{NAME}-{version}-{tag}.whl"
    write_wheel(path, files, dist_info)
    return path


def add_licenses(wheel: Path, licenses: Path) -> None:
    """Rewrite `wheel` in place with the files of `licenses` under its `<dist-info>/licenses/`."""
    with zipfile.ZipFile(wheel) as archive:
        infos = archive.infolist()
        dist_info = next(info.filename.split("/")[0] for info in infos if info.filename.endswith(".dist-info/RECORD"))
        added = license_entries(dist_info, licenses)
        replaced = {f"{dist_info}/RECORD", *(name for name, _, _ in added)}
        kept = [
            (info.filename, archive.read(info), info.external_attr or REGULAR)
            for info in infos
            if info.filename not in replaced and not info.is_dir()
        ]
    write_wheel(wheel, kept + added, dist_info)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", maxsplit=1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack", help="write the daemons wheel")
    packing.add_argument("--bin-dir", type=Path, required=True)
    packing.add_argument("--out", type=Path, required=True)
    packing.add_argument("--version", required=True)
    packing.add_argument("--platform", required=True, help="the wheel's platform tag, like manylinux_2_36_x86_64")
    packing.add_argument("--licenses", type=Path)
    packing.add_argument("names", nargs="+")
    adding = commands.add_parser("licenses", help="add license files to a built wheel")
    adding.add_argument("wheel", type=Path)
    adding.add_argument("licenses", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        print(pack(args.bin_dir, args.out, args.version, args.platform, args.licenses, args.names))
    else:
        add_licenses(args.wheel, args.licenses)
        print(args.wheel)


if __name__ == "__main__":
    main()
