"""continuum-py's PEP 517 backend: maturin, with the optional `ffmpeg` feature (H.264 decoding for network-camera
vision) turned on whenever this machine has FFmpeg's development libraries.

Without this, every `uv sync` or `uv pip install` built the default features and silently dropped the decoder.
The probe is pkg-config, as `mise run continuum-check` does. A machine without the headers still builds, without the
decoder; if the ffmpeg build itself fails (headers but no libclang, say), it says so and builds again without it.
CONTINUUM_PY_FFMPEG=0 turns the feature off, and =1 makes a failed ffmpeg build an error.

The build also needs protoc: continuum-search pulls in LanceDB, which runs it. mise installs it (mise.toml), but a
`uv sync` outside `mise run` (a launchd or systemd job's PATH, say) doesn't see mise's tools, so when PROTOC isn't set
and protoc isn't on the PATH, the backend asks `mise which protoc` and sets PROTOC to the answer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import maturin
from maturin import (
    build_sdist,
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)

__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
]

# What ffmpeg-next's default features link: all of them must be there or the build fails.
FFMPEG_LIBS = ("libavcodec", "libavformat", "libavutil", "libswscale", "libswresample", "libavfilter", "libavdevice")
FEATURE = "--features=ffmpeg"


def ffmpeg_wanted(env: Mapping[str, str], have_libs: Callable[[], bool]) -> bool:
    """Whether to build with ffmpeg: CONTINUUM_PY_FFMPEG=0 or =1 decides, otherwise whether the libraries are here."""
    choice = env.get("CONTINUUM_PY_FFMPEG", "").strip()
    if choice in {"0", "1"}:
        return choice == "1"
    return have_libs()


def ffmpeg_libs_present() -> bool:
    """Whether pkg-config finds every FFmpeg development library the feature links."""
    if shutil.which("pkg-config") is None:
        return False
    try:
        return subprocess.run(["pkg-config", "--exists", *FFMPEG_LIBS], check=False).returncode == 0
    except OSError:
        return False


# Where mise's installers put it, for a job whose PATH has none of them.
MISE_PATHS = ("~/.local/bin/mise", "/opt/homebrew/bin/mise", "/usr/local/bin/mise")


def protoc_from_mise(
    env: Mapping[str, str],
    which: Callable[[str], str | None],
    ask: Callable[[str], str | None],
    places: Sequence[str] = MISE_PATHS,
) -> str | None:
    """The protoc to set PROTOC to: None when PROTOC is set or protoc is on the PATH (the build finds it itself),
    else mise's, from `ask(mise)` (`mise which protoc`'s answer), when mise is on the PATH or in one of `places`."""
    if env.get("PROTOC", "").strip() or which("protoc"):
        return None
    mise = which("mise") or next((str(path) for path in (Path(p).expanduser() for p in places) if path.is_file()), None)
    if mise is None:
        return None
    found = ask(mise)
    return found if found and Path(found).is_file() else None


def mise_which_protoc(mise: str) -> str | None:
    """`mise which protoc`, run from the checkout so its mise.toml counts, or None if mise can't say (protoc not
    installed, or the config not trusted)."""
    checkout = Path(__file__).resolve().parents[2]
    try:
        done = subprocess.run(
            [mise, "which", "protoc"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            cwd=checkout if (checkout / "mise.toml").is_file() else None,
        )
    except OSError, subprocess.TimeoutExpired:
        return None
    return (done.stdout.strip() or None) if done.returncode == 0 else None


def _find_protoc() -> None:
    found = protoc_from_mise(os.environ, shutil.which, mise_which_protoc)
    if found is not None:
        sys.stderr.write(f"continuum-py: protoc isn't on the PATH; using mise's, {found}\n")
        os.environ["PROTOC"] = found


def _with_feature(build: Callable[[], str]) -> str:
    """Run a maturin build with the ffmpeg feature added to MATURIN_PEP517_ARGS, restoring the variable after."""
    before = os.environ.get("MATURIN_PEP517_ARGS")
    os.environ["MATURIN_PEP517_ARGS"] = f"{before or ''} {FEATURE}".strip()
    try:
        return build()
    finally:
        if before is None:
            del os.environ["MATURIN_PEP517_ARGS"]
        else:
            os.environ["MATURIN_PEP517_ARGS"] = before


def _build(build: Callable[[], str]) -> str:
    _find_protoc()
    if not ffmpeg_wanted(os.environ, ffmpeg_libs_present):
        sys.stderr.write(
            "continuum-py: building without the ffmpeg feature (no FFmpeg development libraries found by pkg-config, "
            "or CONTINUUM_PY_FFMPEG=0); network-camera vision will not decode H.264\n"
        )
        return build()
    try:
        return _with_feature(build)
    except SystemExit:
        # maturin exits 1 on a failed build. A forced ffmpeg build stays an error.
        if os.environ.get("CONTINUUM_PY_FFMPEG", "").strip() == "1":
            raise
        sys.stderr.write(
            "continuum-py: the ffmpeg build failed (see above); building again without it, so network-camera vision "
            "will not decode H.264. Fix the FFmpeg build (config/system-packages.toml), or set CONTINUUM_PY_FFMPEG=0.\n"
        )
        return build()


def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _build(lambda: maturin.build_wheel(wheel_directory, config_settings, metadata_directory))


def build_editable(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _build(lambda: maturin.build_editable(wheel_directory, config_settings, metadata_directory))
