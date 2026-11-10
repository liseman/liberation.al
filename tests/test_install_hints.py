"""Every `uv sync` the docs and error messages tell someone to run keeps the extras they already have.

uv sync is exact: it removes every package the named extras don't need. `mise run setup` installs the `faces` extra
(and `voice-clone` on NVIDIA machines), so a plain `uv sync`, or one that names only another extra, uninstalls face
recognition. A hint must pass --inexact, or name both extras setup installs."""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ("MIGRATING.md", "README.md", "CONTRIBUTING.md")
# A step to run: a code line, or a `uv sync` with options in a sentence or an error message. A bare `uv sync` inside a
# sentence ("installed by `uv sync`") names the command and isn't a step.
COMMAND = re.compile(r"^\s*(uv sync\b[^`)\n#\"]*)|(uv sync --[^`)\n#\"]*)", re.M)


def commands() -> list[tuple[str, str]]:
    found = []
    for name in DOCS:
        found += [(name, m.group(m.lastindex or 0).strip()) for m in COMMAND.finditer((ROOT / name).read_text())]
    for p in sorted((ROOT / "src").rglob("*.py")):
        found += [
            (str(p.relative_to(ROOT)), m.group(m.lastindex or 0).strip()) for m in COMMAND.finditer(p.read_text())
        ]
    return found


HINTS = commands()


def test_the_commands_are_found():
    # Guards the pattern: if it matched nothing, the parametrized test below would have no cases and pass silently.
    assert (
        "MIGRATING.md",
        "uv sync --inexact --extra faces --extra segmentation --extra deep-asr --extra voice-clone",
    ) in HINTS


@pytest.mark.parametrize("where, command", HINTS, ids=[f"{where}: {command}" for where, command in HINTS])
def test_no_hint_uninstalls_the_extras_already_installed(where, command):
    assert "--inexact" in command or ("--extra faces" in command and "--extra voice-clone" in command)
