"""Runtime data never lives in git.

Git treats ignored files as disposable: when a commit tracks a file that another checkout has as ignored runtime data,
checking out or pulling that commit overwrites the local file without a word. In an earlier incident, one install's
data/state/transcript.jsonl and speakers.json were force-added and committed, and the next pull on another install
replaced its transcript and voices with them. These tests fail if anything gitignored is tracked again, and check the
pre-commit hook in .githooks/ that refuses to commit such a file.

They also check the hooks that refuse leftover merge conflict markers: pre-commit for ordinary commits and
for finishing a conflicted merge, pre-merge-commit for a merge that had no conflicts.

Every git command here runs without the user's global or system git config (#29), and each temporary repo has its own
identity and this repository's hooks. The hooks also run tools/privacy_check.py from the top of the repo they're in:
each temporary checkout gets a stand-in that passes (ignored, so it is never committed), since
tests/test_privacy_check.py covers the real one.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "pre-commit"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)


PRIVACY_STUB = "import sys\n\nsys.exit(0)\n"


@pytest.fixture(autouse=True)
def isolated_git_config(monkeypatch):
    """No global or system git config (a hooks path, signing, or a default branch there) reaches these tests."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _privacy_stub(checkout: pathlib.Path) -> None:
    """The stand-in for tools/privacy_check.py that the hooks run in `checkout`."""
    (checkout / "tools").mkdir(exist_ok=True)
    (checkout / "tools" / "privacy_check.py").write_text(PRIVACY_STUB)


def _git(*args: str, cwd: pathlib.Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def test_no_ignored_file_is_tracked():
    tracked_but_ignored = _git("ls-files", "--cached", "--ignored", "--exclude-standard").split()
    assert tracked_but_ignored == [], (
        f"these files are gitignored but tracked; untrack them with `git rm --cached` : {tracked_but_ignored}"
    )


def test_data_holds_only_the_placeholder():
    assert _git("ls-files", "data").split() == ["data/state/.gitkeep"]


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.name", "Dana", cwd=repo)
    _git("config", "user.email", "dana@example.com", cwd=repo)
    _git("config", "core.hooksPath", str(HOOK.parent), cwd=repo)
    (repo / ".gitignore").write_text("data/*\ntools/\n")
    _privacy_stub(repo)
    (repo / "data").mkdir()
    (repo / "data" / "transcript.jsonl").write_text("{}\n")
    (repo / "notes.txt").write_text("hello\n")
    return repo


def test_hook_refuses_a_force_added_ignored_file(tmp_path):
    repo = _repo(tmp_path)
    _git("add", "-f", "data/transcript.jsonl", "notes.txt", cwd=repo)
    done = subprocess.run(["git", "commit", "-q", "-m", "x"], cwd=repo, capture_output=True, text=True)
    assert done.returncode != 0
    assert "data/transcript.jsonl" in done.stderr


def test_hook_allows_ordinary_commits(tmp_path):
    repo = _repo(tmp_path)
    _git("add", ".gitignore", "notes.txt", cwd=repo)
    _git("commit", "-q", "-m", "x", cwd=repo)
    (repo / "notes.txt").unlink()
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "remove", cwd=repo)


CONFLICTED = "top\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> other\n"


def _commit(repo: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "commit", "-q", *args], cwd=repo, capture_output=True, text=True)


def test_hook_refuses_conflict_markers_and_names_the_lines(tmp_path):
    repo = _repo(tmp_path)
    _git("add", ".gitignore", "notes.txt", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    (repo / "Cargo.toml").write_text(CONFLICTED)
    _git("add", "Cargo.toml", cwd=repo)
    done = _commit(repo, "-m", "x")
    assert done.returncode != 0
    assert [l.strip() for l in done.stderr.splitlines() if "leftover" in l] == [
        "Cargo.toml:2: leftover conflict marker",
        "Cargo.toml:4: leftover conflict marker",
        "Cargo.toml:6: leftover conflict marker",
    ]


def test_hook_lets_other_whitespace_problems_through(tmp_path):
    repo = _repo(tmp_path)
    (repo / "notes.txt").write_text("trailing spaces   \n")
    _git("add", "notes.txt", cwd=repo)
    assert _commit(repo, "-m", "x").returncode == 0


def test_a_conflicted_merge_finished_with_the_markers_staged_is_refused(tmp_path):
    repo = _repo(tmp_path)
    _git("add", ".gitignore", "notes.txt", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("checkout", "-q", "-b", "other", cwd=repo)
    (repo / "notes.txt").write_text("theirs\n")
    _git("commit", "-q", "-am", "theirs", cwd=repo)
    _git("checkout", "-q", "-", cwd=repo)
    (repo / "notes.txt").write_text("ours\n")
    _git("commit", "-q", "-am", "ours", cwd=repo)
    head = _git("rev-parse", "HEAD", cwd=repo)
    assert subprocess.run(["git", "merge", "-q", "other"], cwd=repo, capture_output=True).returncode != 0
    _git("add", "notes.txt", cwd=repo)  # what the merge script did: stage the conflicted file as it was
    done = _commit(repo, "--no-edit")
    assert done.returncode != 0
    assert "notes.txt:" in done.stderr
    assert _git("rev-parse", "HEAD", cwd=repo) == head


@pytest.mark.parametrize("edit", [pytest.param(["--no-edit"], id="no-edit"), pytest.param([], id="editor")])
def test_a_clean_merge_that_would_bring_in_markers_is_refused(tmp_path, edit):
    repo = _repo(tmp_path)
    _git("add", ".gitignore", "notes.txt", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("checkout", "-q", "-b", "other", cwd=repo)
    (repo / "Cargo.toml").write_text(CONFLICTED)
    _git("add", "Cargo.toml", cwd=repo)
    _git("commit", "-q", "--no-verify", "-m", "markers slipped past the hook", cwd=repo)
    _git("checkout", "-q", "-", cwd=repo)
    (repo / "other.txt").write_text("unrelated\n")
    _git("add", "other.txt", cwd=repo)
    _git("commit", "-q", "-m", "ours", cwd=repo)  # so the merge isn't a fast-forward and makes a commit
    head = _git("rev-parse", "HEAD", cwd=repo)
    done = subprocess.run(
        ["git", "merge", *edit, "other"],
        cwd=repo,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_EDITOR": "true"},
    )
    assert done.returncode != 0
    assert "pre-merge-commit" in done.stderr and "Cargo.toml:2" in done.stderr
    assert _git("rev-parse", "HEAD", cwd=repo) == head


def test_the_first_commit_is_checked_too(tmp_path):
    repo = _repo(tmp_path)
    (repo / "Cargo.toml").write_text(CONFLICTED)
    _git("add", "Cargo.toml", cwd=repo)
    assert _commit(repo, "-m", "x").returncode != 0


@pytest.mark.parametrize(
    "attributes, refused",
    [pytest.param("", True, id="no-opt-out"), pytest.param("doc.md conflict-marker-size=32\n", False, id="opted-out")],
)
def test_a_markdown_heading_underlined_with_seven_equals_needs_an_opt_out(tmp_path, attributes, refused):
    repo = _repo(tmp_path)
    (repo / "doc.md").write_text("Title\n=======\n")
    (repo / ".gitattributes").write_text(attributes)
    _git("add", "doc.md", ".gitattributes", cwd=repo)
    assert (_commit(repo, "-m", "x").returncode != 0) == refused


def test_markers_staged_then_cleaned_only_in_the_working_tree_are_still_refused(tmp_path):
    repo = _repo(tmp_path)
    (repo / "Cargo.toml").write_text(CONFLICTED)
    _git("add", "Cargo.toml", cwd=repo)
    (repo / "Cargo.toml").write_text("resolved, but never staged\n")
    assert _commit(repo, "-m", "x").returncode != 0


def _conflict(repo: pathlib.Path, path: str) -> None:
    """Two branches change path differently, and merging them stops with a conflict in it."""
    (repo / path).write_text("base\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("checkout", "-q", "-b", "other", cwd=repo)
    (repo / path).write_text("theirs\n")
    _git("commit", "-q", "-am", "theirs", cwd=repo)
    _git("checkout", "-q", "-", cwd=repo)
    (repo / path).write_text("ours\n")
    _git("commit", "-q", "-am", "ours", cwd=repo)
    assert subprocess.run(["git", "merge", "-q", "other"], cwd=repo, capture_output=True).returncode != 0


def test_the_opt_out_still_catches_a_real_conflict_in_that_file(tmp_path):
    repo = _repo(tmp_path)
    (repo / ".gitattributes").write_text("doc.md conflict-marker-size=32\n")
    _conflict(repo, "doc.md")
    assert (repo / "doc.md").read_text().startswith("<" * 32)  # git wrote markers of the opted-out size
    _git("add", "doc.md", cwd=repo)
    assert _commit(repo, "--no-edit").returncode != 0


@pytest.fixture
def pushing(tmp_path):
    """A repo with a bare origin whose main has one clean commit."""
    origin = tmp_path / "origin.git"
    _git("init", "-q", "--bare", str(origin), cwd=tmp_path)
    repo = _repo(tmp_path)
    _git("add", ".gitignore", "notes.txt", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("remote", "add", "origin", str(origin), cwd=repo)
    _git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=repo)
    _git("fetch", "-q", "origin", cwd=repo)
    return repo, origin


def _push(repo: pathlib.Path, ref: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "push", "-q", "origin", f"HEAD:{ref}"], cwd=repo, capture_output=True, text=True)


@pytest.mark.parametrize(
    "ref", [pytest.param("refs/heads/main", id="existing-branch"), pytest.param("refs/heads/new", id="new-branch")]
)
def test_a_push_of_commits_that_skipped_pre_commit_is_refused(pushing, ref):
    repo, origin = pushing
    before = _git("for-each-ref", cwd=origin)
    # rebase --continue and git am commit without running pre-commit; --no-verify stands in for them.
    (repo / "Cargo.toml").write_text(CONFLICTED)
    _git("add", "Cargo.toml", cwd=repo)
    _git("commit", "-q", "--no-verify", "-m", "rebased with markers", cwd=repo)
    done = _push(repo, ref)
    assert done.returncode != 0
    assert "pre-push" in done.stderr and "Cargo.toml:2" in done.stderr
    assert _git("for-each-ref", cwd=origin) == before


def test_a_clean_push_goes_through(pushing):
    repo, origin = pushing
    (repo / "notes.txt").write_text("more\n")
    _git("commit", "-q", "-am", "more", cwd=repo)
    assert _push(repo, "refs/heads/main").returncode == 0
    assert _git("rev-parse", "main", cwd=origin) == _git("rev-parse", "HEAD", cwd=repo)


@pytest.fixture
def frontend_change(tmp_path):
    """A repo whose branch `other` changes frontend/, and a PATH whose `mise` only says it was run."""
    repo = _repo(tmp_path)
    (repo / "frontend").mkdir()
    (repo / "frontend" / "app.ts").write_text("one\n")
    _git("add", ".gitignore", "notes.txt", "frontend", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "-M", "main", cwd=repo)
    _git("checkout", "-q", "-b", "other", cwd=repo)
    (repo / "frontend" / "app.ts").write_text("two\n")
    _git("commit", "-q", "-am", "frontend", cwd=repo)
    _git("checkout", "-q", "main", cwd=repo)
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "mise").write_text('#!/bin/sh\necho "mise ran: $*" >&2\n')
    (stub / "mise").chmod(0o755)
    env = {**os.environ, "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}"}
    return repo, env


def _merge(checkout: pathlib.Path, env: dict[str, str]) -> str:
    done = subprocess.run(
        ["git", "merge", "-q", "--no-edit", "other"], cwd=checkout, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return done.stderr


def test_a_merge_that_changes_the_frontend_rebuilds_it_in_the_main_checkout(frontend_change):
    repo, env = frontend_change
    assert "mise ran: run frontend" in _merge(repo, env)


def test_a_worktree_never_rebuilds_the_frontend(frontend_change, tmp_path):
    repo, env = frontend_change
    worktree = tmp_path / "wt"
    _git("worktree", "add", "-q", "-b", "work", str(worktree), "main", cwd=repo)
    _privacy_stub(worktree)
    assert "mise ran" not in _merge(worktree, env)
