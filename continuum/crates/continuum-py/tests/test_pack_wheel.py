"""The daemons wheel puts each binary on the PATH as an executable, its RECORD matches its contents, and it never
promises an older glibc than a binary needs."""

import base64
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packaging"))
import pack_wheel

OBJDUMP = """
0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.2.5) free
0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.34)  pthread_create
0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.9)   pipe2
0000000000000000      DF *UND*  0000000000000000 (GCC_3.3)     _Unwind_Backtrace
"""


def record_of(archive: zipfile.ZipFile, dist_info: str) -> dict[str, str]:
    lines = archive.read(f"{dist_info}/RECORD").decode().splitlines()
    return {line.split(",")[0]: line.split(",", 1)[1] for line in lines}


def digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode() + f",{len(data)}"


def test_the_newest_glibc_symbol_is_compared_as_numbers_not_text() -> None:
    assert pack_wheel.glibc_needed(OBJDUMP) == (2, 34)


def test_a_binary_with_no_glibc_symbols_needs_none() -> None:
    assert pack_wheel.glibc_needed("(GCC_3.3) _Unwind_Backtrace") == (0, 0)


@pytest.mark.parametrize(
    ("platform", "promised"),
    [
        ("manylinux_2_36_x86_64", (2, 36)),
        ("manylinux_2_9_aarch64", (2, 9)),
        ("macosx_14_0_arm64", None),
    ],
)
def test_a_manylinux_tag_promises_its_glibc(platform: str, promised: tuple[int, int] | None) -> None:
    assert pack_wheel.glibc_promised(platform) == promised


@pytest.fixture
def binaries(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "release"
    bin_dir.mkdir()
    (bin_dir / "continuum-broker").write_bytes(b"\x7fELF broker")
    (bin_dir / "continuum-gateway").write_bytes(b"\x7fELF gateway")
    return bin_dir


@pytest.fixture
def objdump(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every binary's symbols are OBJDUMP: glibc 2.34."""

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert command[:2] == ["objdump", "-T"]
        return subprocess.CompletedProcess(command, 0, stdout=OBJDUMP, stderr="")

    monkeypatch.setattr(pack_wheel.subprocess, "run", run)


def test_the_wheel_installs_each_binary_as_an_executable_script(binaries: Path, objdump: None, tmp_path: Path) -> None:
    licenses = tmp_path / "licenses"
    licenses.mkdir()
    (licenses / "FFmpeg-COPYING.LGPLv2.1.txt").write_text("LGPL")

    wheel = pack_wheel.pack(
        binaries,
        tmp_path / "dist",
        "0.1.0",
        "manylinux_2_36_x86_64",
        licenses,
        ["continuum-broker", "continuum-gateway"],
    )

    assert wheel.name == "continuum_daemons-0.1.0-py3-none-manylinux_2_36_x86_64.whl"
    with zipfile.ZipFile(wheel) as archive:
        script = archive.getinfo("continuum_daemons-0.1.0.data/scripts/continuum-broker")
        assert archive.read(script) == b"\x7fELF broker"
        assert (script.external_attr >> 16) & 0o111 == 0o111
        record = record_of(archive, "continuum_daemons-0.1.0.dist-info")
        assert sorted(record) == sorted(archive.namelist())
        assert record["continuum_daemons-0.1.0.data/scripts/continuum-gateway"] == digest(b"\x7fELF gateway")
        assert record["continuum_daemons-0.1.0.dist-info/licenses/FFmpeg-COPYING.LGPLv2.1.txt"] == digest(b"LGPL")
        assert record["continuum_daemons-0.1.0.dist-info/RECORD"] == ","
        wheel_file = archive.read("continuum_daemons-0.1.0.dist-info/WHEEL").decode()
        assert "Tag: py3-none-manylinux_2_36_x86_64\n" in wheel_file
        assert "Root-Is-Purelib: false\n" in wheel_file
        metadata = archive.read("continuum_daemons-0.1.0.dist-info/METADATA").decode()
        assert "Name: continuum-daemons\n" in metadata
        assert "Version: 0.1.0\n" in metadata
        assert "License-File: FFmpeg-COPYING.LGPLv2.1.txt\n" in metadata


def test_a_binary_that_needs_a_newer_glibc_than_the_tag_is_refused(
    binaries: Path, objdump: None, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match=r"continuum-broker needs glibc 2\.34"):
        pack_wheel.pack(binaries, tmp_path / "dist", "0.1.0", "manylinux_2_28_x86_64", None, ["continuum-broker"])
    assert not list((tmp_path / "dist").glob("*.whl"))


def test_a_macos_wheel_is_packed_without_reading_glibc_symbols(binaries: Path, tmp_path: Path) -> None:
    # No objdump fixture: a call to it would run the real one on a file that isn't a binary, and fail.
    wheel = pack_wheel.pack(binaries, tmp_path / "dist", "0.1.0", "macosx_14_0_arm64", None, ["continuum-broker"])
    assert wheel.name == "continuum_daemons-0.1.0-py3-none-macosx_14_0_arm64.whl"


def test_adding_licenses_keeps_the_wheels_files_and_modes_and_records_the_new_ones(tmp_path: Path) -> None:
    wheel = tmp_path / "continuum_py-0.1.0-cp314-cp314-manylinux_2_36_x86_64.whl"
    native = "continuum_py/_native.abi3.so"
    pack_wheel.write_wheel(
        wheel,
        [
            (native, b"\x7fELF native", pack_wheel.EXECUTABLE),
            ("continuum_py-0.1.0.dist-info/METADATA", b"Name: continuum-py\n", pack_wheel.REGULAR),
        ],
        "continuum_py-0.1.0.dist-info",
    )
    licenses = tmp_path / "licenses"
    licenses.mkdir()
    (licenses / "OpenH264-LICENSE.txt").write_text("BSD")

    pack_wheel.add_licenses(wheel, licenses)
    pack_wheel.add_licenses(wheel, licenses)  # a second run replaces, never duplicates

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) == 4
        assert archive.read(native) == b"\x7fELF native"
        assert archive.getinfo(native).external_attr == pack_wheel.EXECUTABLE
        record = record_of(archive, "continuum_py-0.1.0.dist-info")
        assert sorted(record) == sorted(names)
        assert record["continuum_py-0.1.0.dist-info/licenses/OpenH264-LICENSE.txt"] == digest(b"BSD")
        assert record[native] == digest(b"\x7fELF native")
