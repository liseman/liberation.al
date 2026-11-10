"""tools/test_changed.py: which tests cover a change, and that it never falls back to the full suite."""

import importlib.util
import pathlib
import subprocess

import pytest

_spec = importlib.util.spec_from_file_location(
    "test_changed", pathlib.Path(__file__).resolve().parents[1] / "tools" / "test_changed.py"
)
assert _spec and _spec.loader
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)


def make_tests(tmp_path, files):
    d = tmp_path / "tests"
    d.mkdir()
    for name, text in files.items():
        (d / name).write_text(text)
    return sorted(d.glob("test_*.py"))


def test_a_changed_module_runs_the_tests_named_for_it_and_those_importing_it(tmp_path):
    tests = make_tests(
        tmp_path,
        {
            "test_chunks.py": "",
            "test_chunks_index.py": "",
            "test_range.py": "from liberational.audio import (\n    ambient,\n    chunks,\n)\n",
            "test_server.py": "from liberational.dashboard import server\n",
            "test_unrelated.py": "from liberational.audio import chunky\n",  # not chunks
        },
    )
    got = tc.python_tests(["src/liberational/audio/chunks.py", "README.md"], tests, root=tmp_path)
    assert got == ["tests/test_chunks.py", "tests/test_chunks_index.py", "tests/test_range.py"]


def test_a_changed_test_and_a_tools_script_run_their_tests(tmp_path):
    tests = make_tests(
        tmp_path,
        {"test_measure.py": "", "test_marks.py": 'spec("measure", ROOT / "tools" / "measure.py")\n', "test_x.py": ""},
    )
    got = tc.python_tests(["tools/measure.py", "tests/test_x.py"], tests, root=tmp_path)
    assert got == ["tests/test_marks.py", "tests/test_measure.py", "tests/test_x.py"]


def test_a_hub_module_runs_only_the_tests_named_for_it(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "MAX_IMPORTERS", 2)
    tests = make_tests(
        tmp_path, {f"test_{i}.py": "import liberational.dashboard.server\n" for i in range(3)} | {"test_server.py": ""}
    )
    said = []
    got = tc.python_tests(["src/liberational/dashboard/server.py"], tests, root=tmp_path, say=said.append)
    assert got == ["tests/test_server.py"]
    assert said and "3 tests import liberational.dashboard.server" in said[0]


def test_a_package_init_is_the_package_and_other_files_are_nothing(tmp_path):
    assert tc.module_of("src/liberational/audio/__init__.py") == "liberational.audio"
    assert [tc.module_of(p) for p in ("docs/x.md", "tools/sub/x.py", "frontend/src/a.ts")] == [None] * 3
    assert tc.frontend_files(["frontend/src/a.tsx", "frontend/src/a.module.css", "frontend/package.json"]) == [
        "src/a.tsx",
        "src/a.module.css",
    ]


def test_a_change_no_test_covers_runs_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tc, "changed", lambda base: ["docs/README.md"])
    runs = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: runs.append(a))
    assert tc.main(["--base", "origin/main"]) == 0
    assert runs == [] and "nothing to run" in capsys.readouterr().out


@pytest.mark.parametrize(("only", "ran"), [("python", "pytest"), ("frontend", "vitest")])
def test_only_runs_one_side(monkeypatch, only, ran):
    monkeypatch.setattr(tc, "changed", lambda base: ["tests/test_x.py", "frontend/src/a.ts"])
    runs = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: runs.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert tc.main(["--only", only]) == 0
    assert len(runs) == 1 and ran in runs[0]


def test_it_lists_what_changed_since_the_branch_left_base(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    (tmp_path / "a.py").write_text("1")
    (tmp_path / "gone.py").write_text("1")
    git("add", ".")
    git("commit", "-qm", "base")
    git("checkout", "-qb", "work")
    (tmp_path / "a.py").write_text("2")  # changed, uncommitted
    (tmp_path / "new.py").write_text("1")  # untracked
    (tmp_path / "gone.py").unlink()  # deleted: nothing to test
    assert tc.changed("main", root=pathlib.Path(tmp_path)) == ["a.py", "new.py"]
