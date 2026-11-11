# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Builds the .deb and the apt repository's index from a release's files (docs/RELEASES.md, "The apt repository").

    apt_package.py build dist --arch amd64 --out debs     # liberational_<version>-1_amd64.deb from dist/'s wheels
    apt_package.py index debs --key <fingerprint>         # Packages, Release, InRelease, and the public key
    apt_package.py index debs --key <f> --keep Packages   # the same, keeping the published versions listed there

`dist` is a directory with a release's files, however it got them: `gh release download`, the release workflow's
artifacts, or a local build (`mise run wheel`, `mise run continuum-wheels`, and the locks from tools/release.py).
`uv run tools/apt_package.py` runs it; it needs `dpkg-deb`, and `index` needs `gpg`.

The package is a bootstrap, not the system: it holds the three wheels, this platform's lock, a pinned uv (no
distribution ships Python 3.14, so uv fetches it), a desktop entry, an icon, and one launcher that every pack-*
command in /usr/bin links to (packaging/deb/launch). The environment itself is built per user by `pack-setup`. So
one .deb per architecture serves every supported release (Ubuntu 24.04 and 26.04, Debian 12 and 13): nothing in it
links a distribution's libraries, and its Depends are the apt names in config/system-packages.toml.

The repository is flat (`deb [signed-by=...] <url> ./`): the .deb files, `Packages`, and a signed `Release` side by
side, which is what a GitHub Release's assets can hold. A .deb is about 120 MB, over GitHub Pages' 100 MB a file.
Asset names can't carry a `~`, so a pre-release's file is named with the wheel's version (0.2.0rc1) while its
control file has Debian's (0.2.0~rc1, which sorts before 0.2.0).
"""

import argparse
import datetime as dt
import gzip
import hashlib
import io
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "packaging" / "deb"
PACKAGE = "liberational"
DIST = "liberational_pack"
LIB = f"usr/lib/{PACKAGE}"
HOMEPAGE = "https://github.com/liseman/liberation.al"
MAINTAINER = "Liberational contributors <liberational@users.noreply.github.com>"
KEYRING = "liberational-archive-keyring.gpg"

# Debian's architecture: (the release's platform, the wheels' machine tag, uv's archive).
ARCHITECTURES: dict[str, tuple[str, str, str]] = {
    "amd64": ("linux-x86_64", "x86_64", "uv-x86_64-unknown-linux-gnu"),
    "arm64": ("linux-aarch64", "aarch64", "uv-aarch64-unknown-linux-gnu"),
}
# The uv the package carries, checked against its release's SHA-256. Raise both together.
UV_VERSION = "0.13.7"
UV_SHA256 = {
    "amd64": "c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83",
    "arm64": "71279f8d3c67241e807b9d8db872d1bf541579fb9f08b7d1eb1682196d1e57dc",
}
# What the launcher and the maintainer scripts run that isn't Essential: uv fetches over TLS.
ALWAYS = ("ca-certificates",)
# Nothing is recommended beyond the system packages a running system declares: Ollama has no apt package, so naming
# it only ever showed a recommendation apt could not resolve. pack-setup and the docs say how to get it instead.
DESCRIPTION = """Ambient assistant that sees, hears, and remembers, on your own machine
 Liberational records the cameras and microphones you give it, works out who and
 what is there, and answers questions about what happened, all on this machine.
 .
 This package installs the application and its commands. It builds nothing as
 root and registers nothing with systemd: after installing it, run `pack-setup`
 as yourself to build the Python environment for this machine and fetch the
 models, then open Liberational. It records only while it is open."""


def debian_version(wheel: str, revision: int = 1) -> str:
    """A wheel's version as Debian orders it: a pre-release (0.2.0rc1) sorts before its release with a tilde."""
    if not (m := re.fullmatch(r"(\d+(?:\.\d+)*)((?:a|b|rc)\d+)?", wheel)):
        raise ValueError(f"not a release version: {wheel}")
    return m[1] + (f"~{m[2]}" if m[2] else "") + f"-{revision}"


def relations(system_packages: str) -> dict[str, list[str]]:
    """Depends, Recommends, and Suggests from config/system-packages.toml: every apt name a running system needs is
    a Depends, one only an NVIDIA machine needs is a Recommends, and an optional one is a Suggests. A package only a
    build from source needs (`build`) is left out: the .deb builds nothing."""
    out: dict[str, set[str]] = {"Depends": set(ALWAYS), "Recommends": set(), "Suggests": set()}
    for entry in tomllib.loads(system_packages).values():
        if not (apt := entry.get("apt")) or entry.get("build"):
            continue
        field = "Suggests" if entry.get("optional") else "Recommends" if entry.get("when") else "Depends"
        out[field].add(apt)
    out["Recommends"] -= out["Depends"]
    out["Suggests"] -= out["Depends"] | out["Recommends"]
    return {field: sorted(names) for field, names in out.items()}


def control(version: str, arch: str, system_packages: str, installed_kib: int) -> str:
    fields = {
        "Package": PACKAGE,
        "Version": version,
        "Architecture": arch,
        "Maintainer": MAINTAINER,
        "Installed-Size": str(installed_kib),
        **{field: ", ".join(names) for field, names in relations(system_packages).items() if names},
        "Section": "misc",
        "Priority": "optional",
        "Homepage": HOMEPAGE,
        "Description": DESCRIPTION,
    }
    return "".join(f"{field}: {value}\n" for field, value in fields.items())


def commands(wheel: Path) -> list[str]:
    """The pack-* console scripts a wheel declares: each gets a link in /usr/bin."""
    with zipfile.ZipFile(wheel) as z:
        name = next(n for n in z.namelist() if n.endswith(".dist-info/entry_points.txt"))
        text = z.read(name).decode()
    section, out = "", []
    for line in text.splitlines():
        if line.startswith("["):
            section = line.strip()
        elif section == "[console_scripts]" and "=" in line:
            out.append(line.partition("=")[0].strip())
    return sorted(c for c in out if c.startswith("pack-"))


def release_files(dist: Path, arch: str) -> tuple[str, list[Path], Path]:
    """(the wheel's version, the three wheels, this platform's lock) from a release's directory. Exactly one of
    each has to be there: a directory with two releases in it is refused rather than guessed at."""
    platform, machine, _ = ARCHITECTURES[arch]

    def one(pattern: str) -> Path:
        found = sorted(dist.glob(pattern))
        if len(found) != 1:
            raise SystemExit(f"{dist}: expected one {pattern}, found {len(found)}")
        return found[0]

    app = one(f"{DIST}-*-py3-none-any.whl")
    wheels = [app, one(f"continuum_py-*_{machine}.whl"), one(f"continuum_daemons-*_{machine}.whl")]
    return app.name.split("-")[1], wheels, one(f"*-lock-{platform}.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def release_stamp(version: str, wheel_version: str, files: list[Path]) -> str:
    """/usr/lib/liberational/release: what the launcher compares an environment against, so a rebuilt package with
    different files is a different release even at the same version."""
    lines = [f"package: {version}", f"wheel: {wheel_version}", *(f"{sha256(p)}  {p.name}" for p in files)]
    return "\n".join(lines) + "\n"


def fetch_uv(arch: str) -> bytes:
    """The pinned uv's binary, from its release, checked against UV_SHA256."""
    archive = ARCHITECTURES[arch][2]
    url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{archive}.tar.gz"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if (got := hashlib.sha256(data).hexdigest()) != UV_SHA256[arch]:
        raise SystemExit(f"{url}: SHA-256 is {got}, not the pinned {UV_SHA256[arch]}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        member = tar.extractfile(f"{archive}/uv")
        if member is None:
            raise SystemExit(f"{url}: no uv in it")
        return member.read()


def maintainer_script(name: str) -> str:
    return "#!/bin/sh\n" + (FILES / "common.sh").read_text() + "\n" + (FILES / name).read_text()


def tree(root: Path, arch: str, version: str, wheel_version: str, wheels: list[Path], lock: Path, uv: bytes) -> None:
    """Lay the package out under `root`, DEBIAN/ included."""
    lib = root / LIB
    (lib / "wheels").mkdir(parents=True)
    (lib / "bin").mkdir()
    for wheel in wheels:
        shutil.copyfile(wheel, lib / "wheels" / wheel.name)
    shutil.copyfile(lock, lib / "wheels" / "lock.txt")
    (lib / "bin" / "uv").write_bytes(uv)
    (lib / "bin" / "uv").chmod(0o755)
    shutil.copyfile(FILES / "launch", lib / "launch")
    (lib / "launch").chmod(0o755)
    (lib / "release").write_text(release_stamp(version, wheel_version, [*wheels, lock]))
    (root / "usr/bin").mkdir()
    for command in commands(wheels[0]):
        (root / "usr/bin" / command).symlink_to(f"../lib/{PACKAGE}/launch")
    for source, target in (
        (FILES / "liberational.desktop", "usr/share/applications/liberational.desktop"),
        (FILES / "liberational.svg", "usr/share/icons/hicolor/scalable/apps/liberational.svg"),
        (ROOT / "LICENSE", f"usr/share/doc/{PACKAGE}/copyright"),
    ):
        (root / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, root / target)
    kib = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink()) // 1024
    debian = root / "DEBIAN"
    debian.mkdir()
    system_packages = (ROOT / "config" / "system-packages.toml").read_text()
    (debian / "control").write_text(control(version, arch, system_packages, kib))
    for name in ("prerm", "postinst", "postrm"):
        (debian / name).write_text(maintainer_script(name))
        (debian / name).chmod(0o755)
    for path in root.rglob("*"):  # whatever the umask was
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() or path.stat().st_mode & 0o100 else 0o644)


def build(dist: Path, arch: str, out: Path, revision: int = 1, uv: Path | None = None) -> Path:
    wheel_version, wheels, lock = release_files(dist, arch)
    version = debian_version(wheel_version, revision)
    out.mkdir(parents=True, exist_ok=True)
    deb = out / f"{PACKAGE}_{wheel_version}-{revision}_{arch}.deb"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        root.mkdir(mode=0o755)
        tree(root, arch, version, wheel_version, wheels, lock, uv.read_bytes() if uv else fetch_uv(arch))
        # The wheels are zip files already: gzip costs seconds where xz costs minutes and saves nothing more.
        command = ["dpkg-deb", "--root-owner-group", "-Zgzip", "-z1", "--build", str(root), str(deb)]
        subprocess.run(command, check=True, stdout=sys.stderr)  # stdout is for the path of what was built
    return deb


def stanza(deb: Path) -> str:
    """A .deb's entry in Packages: its control file, then where it is and what it must hash to."""
    fields = subprocess.run(["dpkg-deb", "-f", str(deb)], check=True, capture_output=True, text=True).stdout
    return f"{fields.rstrip()}\nFilename: ./{deb.name}\nSize: {deb.stat().st_size}\nSHA256: {sha256(deb)}\n"


def merged(new: list[str], kept: str) -> str:
    """Packages: the new stanzas, then every published one (`kept`, an earlier Packages) whose file isn't replaced
    by a new one. The published .deb files themselves stay where they are."""

    def filename(text: str) -> str:
        return next((line for line in text.splitlines() if line.startswith("Filename:")), "")

    replaced = {filename(s) for s in new}
    old = [s.strip() + "\n" for s in re.split(r"\n\s*\n", kept) if s.strip() and filename(s) not in replaced]
    return "\n".join([*new, *old])


def release(files: dict[str, bytes], architectures: list[str], now: dt.datetime) -> str:
    """The Release file for a flat repository: what is signed, and the hash of each index beside it."""
    lines = [
        "Origin: Liberational",
        "Label: Liberational",
        "Suite: stable",
        "Codename: stable",
        f"Architectures: {' '.join(architectures)}",
        f"Date: {now.astimezone(dt.UTC).strftime('%a, %d %b %Y %H:%M:%S UTC')}",
        "SHA256:",
        *(f" {hashlib.sha256(data).hexdigest()} {len(data)} {name}" for name, data in sorted(files.items())),
    ]
    return "\n".join(lines) + "\n"


def index(directory: Path, key: str, kept: str = "") -> None:
    """Write Packages, Packages.gz, Release, Release.gpg, InRelease, and the public key beside the .deb files."""
    debs = sorted(directory.glob("*.deb"))
    if not debs:
        raise SystemExit(f"{directory}: no .deb files")
    packages = merged([stanza(deb) for deb in debs], kept).encode()
    files = {"Packages": packages, "Packages.gz": gzip.compress(packages, mtime=0)}
    for name, data in files.items():
        (directory / name).write_bytes(data)
    architectures = sorted(set(re.findall(r"^Architecture: (\S+)$", packages.decode(), re.MULTILINE)))
    (directory / "Release").write_text(release(files, architectures, dt.datetime.now(dt.UTC)))
    gpg = ["gpg", "--batch", "--yes", "--local-user", key]
    for arguments in (
        ["--clearsign", "--output", "InRelease", "Release"],
        ["--armor", "--detach-sign", "--output", "Release.gpg", "Release"],
        ["--export", "--output", KEYRING, key],
    ):
        subprocess.run([*gpg, *arguments], check=True, cwd=directory)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="one architecture's .deb from a release's files")
    b.add_argument("dist", type=Path)
    b.add_argument("--arch", choices=sorted(ARCHITECTURES), required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--revision", type=int, default=1, help="the package's own revision of this release (default 1)")
    b.add_argument("--uv", type=Path, help="a uv binary to carry, instead of downloading the pinned one")
    i = sub.add_parser("index", help="the signed index of a directory of .deb files")
    i.add_argument("directory", type=Path)
    i.add_argument("--key", required=True, help="the signing key's fingerprint, in gpg's keyring")
    i.add_argument("--keep", type=Path, help="a published Packages whose other versions stay listed")
    a = parser.parse_args(argv)
    if a.command == "build":
        print(build(a.dist, a.arch, a.out, a.revision, a.uv))
    else:
        index(a.directory, a.key, a.keep.read_text() if a.keep and a.keep.is_file() else "")
        print(f"{a.directory}: indexed and signed", file=sys.stderr)


if __name__ == "__main__":
    main()
