"""The .deb builder turns a release's files into a package whose dependencies, version, and index apt reads right."""

import datetime as dt
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "tools" / "apt_package.py"
    spec = importlib.util.spec_from_file_location("apt_package_tool", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["apt_package_tool"] = module
    spec.loader.exec_module(module)
    return module


apt = _load()

SYSTEM_PACKAGES = """
[espeak-ng]
library = "espeak-ng"
apt = "espeak-ng"

[ffmpeg]
command = "ffmpeg"
mise = true
apt = "ffmpeg"

[ffprobe]
command = "ffprobe"
apt = "ffmpeg"

[sox]
command = "sox"
apt = "sox"
when = "nvidia"

[git]
command = "git"
apt = "git"
build = true

[bluez]
command = "bluetoothctl"
apt = "bluez"
optional = true

[ollama]
command = "ollama"
mise = true
brew = "ollama"

[blueutil]
command = "blueutil"
brew = "blueutil"
optional = true
"""


@pytest.mark.parametrize(
    ("wheel", "revision", "expected"),
    [
        ("0.1.0", 1, "0.1.0-1"),
        ("0.2.0", 3, "0.2.0-3"),
        ("0.2.0rc1", 1, "0.2.0~rc1-1"),
        ("1.0.0a2", 1, "1.0.0~a2-1"),
        ("1.0.0b10", 2, "1.0.0~b10-2"),
    ],
)
def test_a_wheel_version_becomes_a_debian_version(wheel, revision, expected):
    assert apt.debian_version(wheel, revision) == expected


@pytest.mark.parametrize("wheel", ["", "v0.1.0", "0.1.0.dev1", "0.1.0+local", "0.1.0-1"])
def test_a_version_that_is_not_a_release_is_refused(wheel):
    with pytest.raises(ValueError, match="not a release version"):
        apt.debian_version(wheel)


@pytest.mark.skipif(shutil.which("dpkg") is None, reason="needs dpkg to compare versions")
def test_a_prerelease_sorts_before_its_release_for_dpkg():
    older, newer = apt.debian_version("0.2.0rc1"), apt.debian_version("0.2.0")
    assert subprocess.run(["dpkg", "--compare-versions", older, "lt", newer]).returncode == 0


def test_relations_sort_system_packages_by_who_needs_them():
    assert apt.relations(SYSTEM_PACKAGES) == {
        "Depends": ["ca-certificates", "espeak-ng", "ffmpeg"],  # ffmpeg once, though two entries name it
        "Recommends": ["sox"],  # an NVIDIA machine's; Ollama has no apt package and is recommended by neither
        "Suggests": ["bluez"],  # optional; git (a build's) and blueutil (Homebrew's) are in none
    }


def test_the_repository_file_is_read_and_names_only_packages_a_running_system_needs():
    relations = apt.relations((apt.ROOT / "config" / "system-packages.toml").read_text())
    depends = set(relations["Depends"])
    assert {"ffmpeg", "libpipewire-0.3-0", "libportaudio2"} <= depends
    assert not depends & {"git", "build-essential", "pkg-config", "libavcodec-dev"}


def test_the_package_recommends_nothing_apt_cannot_resolve():
    relations = apt.relations((apt.ROOT / "config" / "system-packages.toml").read_text())
    assert "ollama" not in set(relations["Recommends"]) | set(relations["Suggests"])


def test_control_has_the_fields_apt_needs_and_leaves_out_an_empty_relation():
    text = apt.control("0.1.0-1", "arm64", '[curl]\ncommand = "curl"\napt = "curl"\n', 1234)
    fields = dict(line.split(": ", 1) for line in text.splitlines() if not line.startswith(" "))
    assert fields["Package"] == "liberational"
    assert fields["Version"] == "0.1.0-1"
    assert fields["Architecture"] == "arm64"
    assert fields["Installed-Size"] == "1234"
    assert fields["Depends"] == "ca-certificates, curl"
    assert "Suggests" not in fields
    assert text.endswith("\n")
    assert "\n\n" not in text  # a blank line would end the stanza early


def _wheel(path: Path, entry_points: str) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("liberational_pack-0.1.0.dist-info/entry_points.txt", entry_points)
    return path


def test_commands_are_the_wheels_pack_console_scripts(tmp_path):
    wheel = _wheel(
        tmp_path / "liberational_pack-0.1.0-py3-none-any.whl",
        "[console_scripts]\npack-setup = a:main\npack-app = b:main\nother = c:main\n\n[gui_scripts]\npack-gui = d:main\n",
    )
    assert apt.commands(wheel) == ["pack-app", "pack-setup"]


def _release(dist: Path, machine: str = "x86_64", platform: str = "linux-x86_64") -> None:
    dist.mkdir()
    _wheel(dist / "liberational_pack-0.3.0-py3-none-any.whl", "[console_scripts]\npack-app = b:main\n")
    (dist / f"continuum_py-0.1.0-cp314-cp314-manylinux_2_36_{machine}.whl").write_bytes(b"sdk")
    (dist / f"continuum_daemons-0.1.0-py3-none-manylinux_2_36_{machine}.whl").write_bytes(b"daemons")
    (dist / f"liberational-pack-0.3.0-lock-{platform}.txt").write_text("numpy==2.4.1\n")


def test_release_files_picks_this_architectures_wheels_and_lock(tmp_path):
    dist = tmp_path / "dist"
    _release(dist)
    (dist / "continuum_py-0.1.0-cp314-cp314-manylinux_2_36_aarch64.whl").write_bytes(b"other")
    (dist / "liberational-pack-0.3.0-lock-linux-aarch64.txt").write_text("")
    version, wheels, lock = apt.release_files(dist, "amd64")
    assert version == "0.3.0"
    assert [w.name for w in wheels] == [
        "liberational_pack-0.3.0-py3-none-any.whl",
        "continuum_py-0.1.0-cp314-cp314-manylinux_2_36_x86_64.whl",
        "continuum_daemons-0.1.0-py3-none-manylinux_2_36_x86_64.whl",
    ]
    assert lock.name == "liberational-pack-0.3.0-lock-linux-x86_64.txt"


def test_release_files_refuses_a_directory_with_two_releases(tmp_path):
    dist = tmp_path / "dist"
    _release(dist)
    _wheel(dist / "liberational_pack-0.4.0-py3-none-any.whl", "")
    with pytest.raises(SystemExit, match="expected one"):
        apt.release_files(dist, "amd64")


def test_release_files_refuses_a_directory_without_this_architecture(tmp_path):
    dist = tmp_path / "dist"
    _release(dist)
    with pytest.raises(SystemExit, match="found 0"):
        apt.release_files(dist, "arm64")


def test_the_release_stamp_changes_when_a_file_does(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text("numpy==2.4.1\n")
    first = apt.release_stamp("0.3.0-1", "0.3.0", [lock])
    assert first.splitlines()[:2] == ["package: 0.3.0-1", "wheel: 0.3.0"]
    assert first.splitlines()[2] == f"{hashlib.sha256(b'numpy==2.4.1\n').hexdigest()}  lock.txt"
    lock.write_text("numpy==2.4.2\n")
    assert apt.release_stamp("0.3.0-1", "0.3.0", [lock]) != first


@pytest.mark.skipif(shutil.which("dpkg-deb") is None, reason="needs dpkg-deb to build a package")
def test_build_lays_out_a_package_dpkg_reads(tmp_path):
    dist = tmp_path / "dist"
    _release(dist)
    uv = tmp_path / "uv"
    uv.write_bytes(b"#!/bin/sh\n")
    deb = apt.build(dist, "amd64", tmp_path / "out", revision=2, uv=uv)
    assert deb.name == "liberational_0.3.0-2_amd64.deb"

    def dpkg(option: str, *names: str) -> str:
        command = ["dpkg-deb", option, str(deb), *names]
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout

    assert dpkg("-f", "Version").strip() == "0.3.0-2"
    listing = dpkg("-c")
    assert "./usr/bin/pack-app -> ../lib/liberational/launch" in listing
    assert "./usr/lib/liberational/wheels/lock.txt" in listing
    assert "./usr/share/applications/liberational.desktop" in listing
    assert "systemd" not in listing
    assert "/etc/" not in listing
    assert all(" root/root " in line for line in listing.splitlines())
    assert dpkg("-I", "prerm").startswith("#!/bin/sh\n")  # dpkg runs it, so it has to be there and be a script
    stanza = apt.stanza(deb)
    assert f"Filename: ./{deb.name}\n" in stanza
    assert f"Size: {deb.stat().st_size}\n" in stanza
    assert f"SHA256: {hashlib.sha256(deb.read_bytes()).hexdigest()}\n" in stanza


@pytest.mark.parametrize("name", ["prerm", "postinst", "postrm"])
def test_a_maintainer_script_starts_with_the_shared_functions(name):
    text = apt.maintainer_script(name)
    assert text.startswith("#!/bin/sh\n")
    assert text.index("quit_applications()") < text.index((apt.FILES / name).read_text())


NEW = "Package: liberational\nVersion: 0.2.0-1\nArchitecture: amd64\nFilename: ./liberational_0.2.0-1_amd64.deb\n"
PUBLISHED = (
    "Package: liberational\nVersion: 0.1.0-1\nArchitecture: amd64\nFilename: ./liberational_0.1.0-1_amd64.deb\n"
    "\n"
    "Package: liberational\nVersion: 0.2.0-1\nArchitecture: amd64\nFilename: ./liberational_0.2.0-1_amd64.deb\n"
    "SHA256: stale\n"
)


def test_merging_keeps_published_versions_and_replaces_a_rebuilt_file():
    stanzas = [s for s in apt.merged([NEW], PUBLISHED).split("\n\n") if s.strip()]
    assert [s.split("\n")[1] for s in stanzas] == ["Version: 0.2.0-1", "Version: 0.1.0-1"]
    assert "stale" not in apt.merged([NEW], PUBLISHED)


def test_merging_with_nothing_published_is_the_new_stanzas():
    assert apt.merged([NEW], "") == NEW


def test_release_lists_each_index_with_its_hash_and_size():
    now = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC)
    text = apt.release({"Packages": b"abc", "Packages.gz": b"zz"}, ["amd64", "arm64"], now)
    lines = text.splitlines()
    assert "Architectures: amd64 arm64" in lines
    assert "Date: Fri, 02 Jan 2026 03:04:05 UTC" in lines
    assert lines[lines.index("SHA256:") + 1 :] == [
        f" {hashlib.sha256(b'abc').hexdigest()} 3 Packages",
        f" {hashlib.sha256(b'zz').hexdigest()} 2 Packages.gz",
    ]
