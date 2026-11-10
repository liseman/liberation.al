"""Run the tests that cover what changed, and only those (AGENTS.md, "Run only the tests you need").

    mise run test-changed                 # against origin/main: committed, staged, and unstaged changes
    mise run test-changed -- --base main  # against another ref
    mise run test-changed -- --dry-run    # say what would run, run nothing
    tools/test_changed.py --only python   # one side only (the hosted checks run each in its own job)

Python: a changed test file runs as it is; a changed module under src/liberational/, or a script under tools/, runs
the test files named for it (tests/test_<module>*.py) and those that import or load it. Frontend: the changed files under frontend/src/ go to
`vitest related`, which finds the tests that import them. Nothing else is run, never the full suite: a change no test
covers runs nothing, and says so. Run it through mise, which runs it at the lowest priority (tools/niced.sh).
"""

import argparse
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = "src/liberational/"
FRONT = "frontend/src/"
FRONT_CODE = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".css")
MAX_IMPORTERS = 20  # a module more tests than this import (dashboard/context.py) runs only the tests named for it


def changed(base: str, root: pathlib.Path = ROOT) -> list[str]:
    """Paths (relative to the repo) that differ from where this branch left base, plus uncommitted and untracked
    ones, deleted files left out."""

    def git(*args: str) -> list[str]:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout
        return [line for line in out.splitlines() if line]

    since = git("merge-base", base, "HEAD")[0]
    names = {
        *git("diff", "--name-only", "--diff-filter=d", since),
        *git("ls-files", "--others", "--exclude-standard"),
    }
    return sorted(n for n in names if (root / n).exists())


def module_of(path: str) -> str | None:
    """src/liberational/audio/chunks.py is liberational.audio.chunks; a package's __init__.py is the package; a
    script tools/measure.py is tools.measure (its tests load it by file name)."""
    if path.startswith("tools/") and path.endswith(".py") and path.count("/") == 1:
        return "tools." + path[len("tools/") : -len(".py")]
    if not (path.startswith(SRC) and path.endswith(".py")):
        return None
    parts = path[len("src/") : -len(".py")].split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports(text: str, module: str) -> bool:
    """Whether test code imports module: by its dotted name, or by name from its package
    (`from liberational.audio import ambient, chunks`, parenthesized lists included). A tools script also counts
    when its file is named ("measure.py", loaded with importlib)."""
    if re.search(rf"\b{re.escape(module)}\b", text):
        return True
    if module.startswith("tools."):
        return f'"{module.removeprefix("tools.")}.py"' in text
    package, _, name = module.rpartition(".")
    if not package:
        return False
    for m in re.finditer(rf"from {re.escape(package)} import (\([^)]*\)|[^\n]*)", text):
        if re.search(rf"\b{re.escape(name)}\b", m.group(1)):
            return True
    return False


def python_tests(
    paths: Iterable[str], tests: Sequence[pathlib.Path], root: pathlib.Path = ROOT, say: Callable[[str], None] = print
) -> list[str]:
    """The test files that cover the changed paths: changed tests themselves, tests named for a changed module, and
    tests that import one. A module more than MAX_IMPORTERS tests import is a hub (the dashboard's shared context): only the
    tests named for it run, and say() names how many were left out. Sorted, each once."""
    out: set[str] = set()
    modules = []
    for p in paths:
        if p.startswith("tests/") and pathlib.PurePath(p).name.startswith("test_") and p.endswith(".py"):
            out.add(p)
        elif (m := module_of(p)) is not None:
            modules.append(m)
    if not modules:
        return sorted(out)
    texts = {t: t.read_text(errors="replace") for t in tests}
    for m in modules:
        stem = m.rsplit(".", 1)[-1]
        named = {t for t in texts if t.stem == f"test_{stem}" or t.stem.startswith(f"test_{stem}_")}
        importers = {t for t, text in texts.items() if imports(text, m)} - named
        if len(importers) > MAX_IMPORTERS:
            say(
                f"test-changed: {len(importers)} tests import {m}; running only the {len(named)} named for it. "
                "Add the ones your change touches with pytest -k."
            )
            importers = set()
        out |= {str(t.relative_to(root)) for t in named | importers}
    return sorted(out)


def frontend_files(paths: Iterable[str]) -> list[str]:
    """Changed frontend source files, as vitest related wants them (relative to frontend/)."""
    return [p.removeprefix("frontend/") for p in paths if p.startswith(FRONT) and p.endswith(FRONT_CODE)]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="what to compare with (default origin/main)")
    ap.add_argument("--dry-run", action="store_true", help="say what would run, run nothing")
    ap.add_argument("--only", choices=("python", "frontend"), help="run only the Python or only the frontend tests")
    args = ap.parse_args(argv)
    paths = changed(args.base)
    py = python_tests(paths, sorted((ROOT / "tests").glob("test_*.py"))) if args.only != "frontend" else []
    front = frontend_files(paths) if args.only != "python" else []
    runs: list[tuple[list[str], pathlib.Path]] = []
    if py:
        runs.append((["uv", "run", "pytest", "-q", *py], ROOT))
    if front:
        runs.append((["npx", "--no-install", "vitest", "related", "--run", *front], ROOT / "frontend"))
    if not runs:
        print(f"test-changed: no tests cover the {len(paths)} changed file(s) against {args.base}; nothing to run")
        return 0
    status = 0
    for cmd, cwd in runs:
        print("test-changed:", " ".join(cmd), flush=True)
        if not args.dry_run:
            status = subprocess.run(cmd, cwd=cwd).returncode or status
    return status


if __name__ == "__main__":
    sys.exit(main())
