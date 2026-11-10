# Working in this repo (for agents)

Read this before changing anything here. It applies to every agent: OpenClaw agents, coding agents, and anything they spawn. People should read [CONTRIBUTING.md](CONTRIBUTING.md); this file is the stricter, agent-facing version of the same rules.

**This is typed Python.** Every function in `src/` and `tools/` has type hints, and `mise run lint` runs the ty type checker after Ruff. Untyped code fails lint, so don't ship it (see "Type everything" below).

The goal is a codebase that a new contributor (person or agent) can understand from [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) without reading session logs. Every change should leave it at least as organized as you found it.

## Before you start

1. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) to find where your change belongs. Check the [project board](https://github.com/liseman/liberation.al/projects) and the open issues to see whether it's already tracked or someone is on it, and [docs/ROADMAP.md](docs/ROADMAP.md) for the bigger plan (see "Tracking work" below).
2. Read [docs/CONVENTIONS.md](docs/CONVENTIONS.md): what may and may not appear in this public repository.
3. Don't run the full test suite to get a baseline. Run the tests for the area you're about to change (see "Run only the tests you need" in Rules) and note what already fails there, so you can tell your breakage from existing breakage.

## How we build

This is research. Building the right thing, fast and directly, beats preserving what already exists. Be pragmatic: pick the most direct path that's correct, and don't spend work protecting the past.

- **Existing data is expendable.** Machine-made data (recordings, detections, poses, tracks, the catalog, and worker state) can be dropped, reset, or left unread whenever that's the most direct way to the right design. Don't write migrations, compatibility shims, dual paths, or readers that keep old rows working. The one exception is what the owner taught the system (named people, voice prints, labeled pets and objects, drawn zones, and settings) plus the owner inputs under "A state reset keeps owner inputs": carry those across a reset with an explicit export and import.
- **Gaps are fine.** Don't hold a removal until its replacement lands, keep an old path alive in the meantime, or build a bridge service to avoid a gap. Delete what's wrong now. While a feature is being rebuilt, its readers say so plainly, never that nothing happened.
- **Fast first, tune later.** A working end-to-end path with stock models beats a careful slow one. Take the simplest option at each step; accuracy studies and tuning come after it runs.
- **Use the solved approach.** When NVRs, video pipelines, or time-series stores already solve a problem, do it their way: a fixed history window that deletes old video, one hardware decode per stream shared by every consumer, time-series data in per-hour columnar files, and models rather than hand-written rules.
- **Question whether we need it.** Before moving, porting, or keeping a feature, ask whether it earns its place. Often the right move is to delete it. Not everything belongs on a model either: when a model already does a job (see-actions finds people, poses, tracks, and actions at full frame rate on the GPU), move the closely related jobs onto it and delete the duplicate or rule-based code, but keep simple code that's simple for a reason (a geometry check, a counter, or a config lookup). Use judgment, not a rule.
- **Measure before you guess.** Profile ([docs/PROFILING.md](docs/PROFILING.md)) to find where the CPU goes before optimizing. If video work is eating the CPU, something is wrong: decoding belongs on the GPU.
- **Guard what correctness needs, not hypothetical data loss.** Whoever deploys runs destructive steps in one window; your PR says in one line what it drops.

## Where things go

| Path | What belongs there |
|---|---|
| `src/liberational/` | The application package. One subpackage per area: `speech/` (ASR, TTS, playback), `audio/` (device integration and routing), `listen/` (transcription and speaker identification), `see/` (vision analysis workers), `vision/` (vision models and readers), `people/` (identity, profiles), `memory/` (history, summaries, object memory), `assistant/` (voice loop, auth, agent escalation), `phone/` (Twilio bridge and call tools), `dashboard/` (the web server; `static/app/` is the frontend's build, gitignored), `demo/` (demo mode's made-up household and `pack-demo`, #17), `spatial/` (point clouds of places and the floorplans derived from them, #1), `domain/` (the domain model types from [docs/DOMAIN-MODEL.md](docs/DOMAIN-MODEL.md): no I/O, and contexts share only `domain/kernel.py`). `config.py` is the settings module, `knowledge_eventlog.py` is the domain event log's knowledge-store adapter, and `continuum.py` converts the domain kernel's types to Continuum's payload fields (it sits outside `domain/` because it knows every context's record types). |
| `continuum/` | Continuum Broker, the v2 data layer ([docs/CONTINUUM.md](docs/CONTINUUM.md)): a Rust Cargo workspace with one crate per area under `crates/`, and the FlatBuffers schema in `schemas/`. Build, test, and check it with the `continuum-*` mise tasks. Continuum is the active data layer used by the Python workers under `src/`. |
| `frontend/` | The dashboard and object memory pages: TypeScript, React, and [Mantine](https://mantine.dev/), built with Vite. `mise run frontend` builds it into `src/liberational/dashboard/static/app/` (gitignored, so never edit or commit it). Tests are Vitest files next to the code (`*.test.ts`, `*.test.tsx`). `frontend/src/dashboard/` has the shell and one folder per tab; [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (Dashboard frontend) says what goes where. |
| `tests/` | Tests (`test_<module>.py`) and their fixtures in `tests/data/`. |
| `tools/` | Benchmarks, probes, and maintenance scripts you run by hand. Each has a docstring that says how. |
| `config/` | `defaults.toml` (every setting, documented), `pack.example.toml`, and `system-packages.toml` (what's needed from the operating system). |
| `services/` | `services.toml`: every daemon, declared once and rendered to launchd or systemd by `pack-services`. |
| `src/liberational/resources/` | Symlinks to the files in `config/` and `services/` that the code reads at run time, so they ship in the wheel. A new one gets a symlink here and is read through `config.RESOURCES`; the owner's files go through `config.owner_file()`, `config.devices_file()`, and `config.path()`, never a path built from the checkout. |
| `packaging/` | What a platform's package is made of, beyond the wheels: `deb/` has the `.deb`'s launcher, maintainer scripts, desktop entry, icon, and container test ([docs/RELEASES.md](docs/RELEASES.md), "The apt repository"). `tools/apt_package.py` builds it. |
| `docs/` | Architecture, roadmap, phase specs, hardware, history, and `notes/` (old per-feature session notes; don't add new ones). |
| `.claude/skills/` | Skills that coding agents load for this repo: `python-tests` (how to write tests here) and `property-based-testing` (vendored unmodified from Trail of Bits, see its `NOTICE.md`). |

Daemons and CLIs are commands declared under `[project.scripts]` in `pyproject.toml` (`uv run pack-ambient`, `uv run pack-config`, and so on). A new daemon gets a `main()`, a `pack-<name>` command, and an entry in `services/services.toml`. Never hand-write plists or unit files. Import other modules with absolute imports (`from liberational.speech import asr`); never edit `sys.path`.

## Tracking work

Work is tracked as GitHub issues in `liseman/liberation.al`, shown on the project board linked from the repository's
[Projects tab](https://github.com/liseman/liberation.al/projects). Both are public. [docs/CONVENTIONS.md](docs/CONVENTIONS.md)
says what an issue, a pull request, a commit message, and a card may contain; read it before you write any of them.

| Column | Meaning | Who moves cards here |
|---|---|---|
| Ideas | Proposed; nobody has looked at it yet | Anyone, including agents |
| Vetting | Being discussed: worth doing, and how? | Anyone |
| Ready | Agreed and clear enough to pick up | **Maintainers only** |
| In progress | Being worked on | Whoever picks it up |
| Done | Merged or closed | Closing the issue does it; if the card stays put, move it |

- **Before starting**, look for an existing issue (`gh issue list --repo liseman/liberation.al --search "<words>"`). Use it if there is one; don't open a duplicate.
- **Got an idea, or found a bug you're not fixing now?** Open an issue that says what and why, and put it in Ideas. Don't move your own proposal to Ready, and don't start building it until a maintainer has.
- **Picking up an issue**: assign it to the GitHub account you work under and move it to In progress. That's the claim. Don't comment with an agent, session, or machine name; if parallel work needs telling apart, name the branch. Only pick up Ready issues, or ones a maintainer asked you to do.
- **Pull requests** say `Closes #N` in the body, so merging closes the issue. Use `Part of #N` if the PR doesn't finish it.
- **Merging a PR**: the `privacy` check must pass. The hosted `checks` workflow is a backstop, not a gate: don't wait for its runs before merging, and if one turns red after a merge, fix forward. Merge after a cursory pass, not an exhaustive one: build, run the tests of each package or crate you changed (Cargo validation only when Rust source, Cargo manifests or locks, or Rust build inputs changed against `origin/main`; for Rust, `tools/build-slot.sh cargo test --no-fail-fast -p <crate>` for every crate you touched, and `tools/build-slot.sh cargo check --workspace --all-targets` when you change a public type others use), and run `mise run fmt` and lint (with the type check) on them; run `mise run frontend` (the production build) only if you changed frontend code. Read the diff yourself before merging. For very complex code (intricate concurrency, unsafe code or atomics, a tricky algorithm), ask for a read-only review first. Note in a PR comment what you ran, with counts. If main moved, merge it in and push, and rerun your targeted tests only if the merge touched the files you changed. Merge with a merge commit: `gh pr merge N --repo liseman/liberation.al --merge`. Never add `--delete-branch` from a checkout that running services use: it switches that checkout to main and pulls it under them. Delete the branch yourself afterward. When the change touches MLX code, also run `mise run lint` on an Apple Silicon Mac, where `ty check` sees the MLX dependencies.
- **A new key in `config/defaults.toml`** means every config-reading service needs a restart after the pull, even if the feature ships off: a running process that doesn't know the key fails as soon as `pack.toml` sets it. Say so in the PR.
- **After it lands in main**, clean up after yourself: remove your worktree (`git worktree remove <path>`), delete your local branch (`git branch -d <branch>`), and delete the branch on GitHub (`git push origin --delete <branch>`).
- **Goal issues** (label `goal`) are umbrellas for a big outcome. The concrete work items are sub-issues under them. Work on the sub-issues, not the goal.
- Don't close or reassign someone else's issue, and don't change the board's columns, fields, views, or workflows.

To set or change a card's column (`item-add` is safe to repeat: it returns the card that's already there), read the board's number and IDs once with `gh project list --owner liseman` and `gh project field-list <number> --owner liseman --format json`, then:

```sh
item=$(gh project item-add <number> --owner liseman --url https://github.com/liseman/liberation.al/issues/N --format json --jq .id)
gh project item-edit --id "$item" --project-id <project id> --field-id <Status field id> --single-select-option-id <option id>
```

The `gh project` commands need the `project` scope. If `gh` says it's missing, ask a person to run `gh auth refresh -h github.com -s project`; issue commands work without it. If you can't reach GitHub at all, tell the owner what you would have filed instead of dropping it.

## This repository is public

The system records real homes and real people, and none of that may reach this repository: not in code, tests,
docs, commit messages, branch names, issues, pull requests, comments, or the board. [docs/CONVENTIONS.md](docs/CONVENTIONS.md)
has the full list. The short version:

- No names of real people or pets, no transcript lines or other live content, no addresses or room layouts, no
  hostnames or network addresses, no secrets, and no screenshots of real rooms or real data.
- No agent, worker, session, or orchestrator names, and no session or task IDs. Don't sign your work, and don't say
  which machine you ran on. Describe the work, not who did it.
- Nothing copied from the project's earlier private tracker. Write it fresh.
- The Git hooks (`mise run hooks`) run `tools/privacy_check.py` before every commit, commit message, and push, and
  the `privacy` check runs it again on GitHub. Never skip a hook with `--no-verify`. A false alarm gets an entry in
  `.privacy-allow.toml` with a reason. The hooks can't read an issue or a comment: check those yourself before you
  post them.
- Coordination between machines, and anything that names a person, a machine, or a session, happens outside this
  repository.

## Rules

**Work in your own worktree.** A worker edits only its own worktree under `.worktrees/<name>` at the repository root (gitignored, so Ruff, ty, and the hooks skip it), on a branch off origin/main (`git worktree add .worktrees/<name> -b <area>/<description> origin/main`), and starts each command with an explicit `cd` into that worktree. The live checkout is read only. No command that discards changes (`git reset --hard`, `git checkout -- .`, `git clean`) runs in any checkout.

**Examples illustrate goals.** A request's concrete examples (squats, walk-bys, "when did I last see Bob") show what a goal means; they are never the spec. Build and file work for the general goal, such as answering any question about what happened on video, or analyzing events in general, and use the example as one test case.

**A state reset keeps owner inputs.** When the owner asks to wipe or reset state ("clean slate", "trash the state data"), that means machine-made data: the catalog and the rest of `data/state`, `data/logs`, and the machine's face files (unknown groups, learned rows). `pack-reset-catalog --keep-taught` is that reset. What the owner taught stays: `knowledge.sqlite3` and the taught files beside it (`faces/db.npz` and `faces/nots.npz`, voiceprints, clones, person references), `paths.voices` (the voice clone sample), `paths.measurement` (hand-made labels), and `data/models`, unless they name them. Before any reset that removes taught state (`--fresh`, a new install), run `pack-knowledge export` to a path outside the deletion and check it with `verify`; `import` brings it back (docs/ARCHITECTURE.md lists what it carries). List what you'll remove and ask if anything else is in doubt. A reset once trashed `data/voices` along with the state.

**Put code where it belongs.**
- Extend an existing module before adding a new one. Search first (`grep -rn`); a lot already exists.
- No one-off scripts in the code tree. Probes, experiments, and benchmarks you won't keep go in `scratch/` (gitignored) or your own workspace. Benchmarks worth keeping go with the other tools, with a docstring saying how to run them.
- Don't add files at the repo root. Don't add per-feature notes files (`NOTES-<feature>.md`) next to code.

**Write things down in the right place.**
- New daemon, port, state file (`paths.state`, `data/state/` by default), or external service: update the component table and the `state/` contract in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) in the same commit.
- Design decisions, benchmarks, and gotchas a future contributor needs: the module docstring, or `docs/` if it spans modules. Not a dated log.
- Your own session history and memory: your tool's own memory, outside the repository, never the code tree.
- Individual pieces of planned or unfinished work: an issue on the board (see "Tracking work"). The bigger plan and its reasoning: [docs/ROADMAP.md](docs/ROADMAP.md).

**No hardcoded personal or machine values.**
- Names, phone numbers, hostnames, ports, model names, paths, and device IDs go through `liberational.config`, with a generic default and a comment in `config/defaults.toml`. The owner's real values belong in their gitignored `pack.toml`, never in code, prompts, tests, or docs.
- Prompts refer to people through `config.owner_or()` and `config.assistant_name()`.
- Tests use made-up people (Alice, Dana), not real contacts.

**Keep it portable.**
- It must keep working on Apple Silicon (MLX) and on Linux (sherpa-onnx, Ollama, CUDA). If you add a platform-specific path, add the fallback, or make the feature fail with a clear message elsewhere.
- No absolute paths into one person's home directory, and no Homebrew paths; use `shutil.which` and config.

**Type everything.**
- Every function in `src/` and `tools/` has type hints on its parameters and return (Ruff's `ANN` rules enforce it), and `ty check` passes over `src/`, `tests/`, and `tools/`. Tests are checked but don't need annotations.
- Use builtin generics and `X | None` (`list[str]`, `dict[str, float]`), and the most specific type that's true. `Any` is for values whose shape the code really doesn't fix, like JSON payloads.
- A `# ty: ignore[rule]` needs a comment saying why. Don't silence the checker to get past it.
- Don't exclude files from ty or Ruff's `ANN` rules in `pyproject.toml` to get past the checker. Nothing is excluded now; keep it that way.

**Log with levels.**
- A service logs through `liberational.logs` (`LOG = logs.get("<tag>")`, then `LOG.info(...)`, `LOG.warning(...)`, and so on), never bare `print` and never a `log()` wrapper of its own: each line is `<stamp> [tag] LEVEL message` (the stamp ISO local date and time with offset, `2026-11-03T02:20:15-08:00`), and the Debug tab filters by level. CLI output for a person to read stays `print`.
- ERROR: something failed and needs attention (a crash, a model that won't load, a write that failed). WARN: degraded, retrying, or falling back (a service it needs is down, a retry, a camera reconnecting). INFO: normal progress (started, a pass done, counts). DEBUG: per-item detail, off unless `logging.level` or `logging.per_service` turns it on.
- A client that hangs up, or a retry that will likely succeed, is not an error. A traceback goes with its line (`LOG.exception`), not on its own.
- A Rust daemon calls `continuum_log::init("<service>")` first thing and logs with `tracing`'s macros, never `eprintln!`; its level is `RUST_LOG`, which `pack-services install` sets from the same `logging` settings.
- A counter reported every few seconds or once a minute (frame rates, mic levels, health) goes through `logs.Periodic` (Rust: `continuum_log::periodic!`): INFO every 10 minutes, DEBUG between. An idle service writes fewer than 100 INFO lines an hour.
- A native library's own log (FFmpeg, ONNX Runtime, OpenCV, transformers) is set to errors where it initializes or in the service environment (services.py), since its lines carry no stamp or level.
- A library that `print`s (ONNX Runtime's fallback notice, a model's warmup progress) is called inside `logs.printed(LOG)`, which makes each printed line a log line. A program we don't own (Ollama) runs with `log_relay = true` in services.toml, so `log_relay.py` writes its lines in the one format.

**Run only the tests you need.** Never run the full test suite (`mise run test`, a bare `pytest`, or a bare `vitest run`) unless a person asks for one. Many sessions running the whole suite again and again swamped the machine.
- Run the tests that cover what you changed: the matching Python test files (`pytest tests/test_<area>.py`, or `-k` for a few names) and, for frontend changes, `vitest related <changed files>` or the specific test files. `mise run test-changed` does this for you against origin/main: the test files named for or importing each changed module under `src/liberational/` or script under `tools/`, plus `vitest related` on changed frontend files (`mise run test-changed -- --dry-run` shows them first). For a module most tests import, such as the dashboard's shared `context.py`, it runs only the tests named for it, and you add the ones your change touches with `-k`. The dashboard server is no longer one: each area's tests import that area's module and take the running dashboard from the `serve_dashboard` fixture (or the `dashboard_server` fixture when they need the request handler itself), so a change to one area selects that area's tests.
- Lint and type-check what you changed, and run `mise run frontend` (the production build) only when you changed frontend code.
- After merging main in, rerun only those targeted tests, plus the tests for any code main changed next to yours.
- Name what you ran in the PR comment, so a reviewer can see the coverage.

**Share the machine.** Several sessions, the live services, Ollama, and the person using the computer share it, and stacked test runs have pushed it past a load average of 50.
- **Rust exception:** run Cargo, rustc, and sccache startup at standard priority, one Cargo command at a time machine-wide across all repository worktrees and agents, through `tools/build-slot.sh`. Its default is `CARGO_BUILD_JOBS=8`, overridable by the caller. Do not wrap Rust commands, the build wrapper, or maturin builds in `nice`, `ionice`, `taskpolicy`, or `tools/niced.sh`. Leave an already-running sccache server alone while another worker is compiling. `mise run continuum-release` already takes that lock (`mise.toml` runs `tools/build-slot.sh`). Run it bare. Wrapping it again deadlocks every Cargo user on the machine: the lock is not reentrant, so the inner call waits on the lock the outer call still holds.
- Run Cargo validation only when Rust source, Cargo manifests/locks, or Rust build inputs changed against `origin/main`. Python, frontend, docs, and shell-only changes do not warrant Cargo checks. Explicit deployment or requested builds still run even on a clean main checkout; this is a validation policy, not a build-command skip guard.
- Every non-Rust test, lint, type check, and build runs at the lowest CPU and disk priority. The mise tasks (`test-changed`, `lint`, `typecheck`, `fmt`, `frontend`, and `test`) already do, through `tools/niced.sh`. A test run or build you start yourself (pytest on a file, `vitest run`, `vite build`, `tsc`) goes through `tools/test-slot.sh`, which runs it at the lowest priority and lets at most two such runs go at once machine-wide across all worktrees and agents (`TEST_SLOTS`), the rest waiting their turn: `tools/test-slot.sh .venv/bin/python -m pytest -q tests/test_x.py`. Quick lint and format commands (`ruff`, `biome`, `ty`) just go under `nice -n 19 ionice -c3` (macOS has no `ionice`: `nice -n 19` alone).
- Cap non-Rust parallelism at 4: pytest at most `-n 4` if the suite ever uses xdist (it doesn't now), and Vitest at most 4 workers (`maxWorkers` in `frontend/vite.config.ts` sets it for every run).
- No benchmarks, stress runs, or mutation loops on a shared machine without asking first.
- **Deploy chains.** When one shell runs several deploy steps, start it with `set -o pipefail`. A `| tail` otherwise hides a failed build (the pipeline status is tail's), and the next step, such as a cascade, runs anyway.
- Measure or try out large models against a separate Ollama instance, or with their callers stopped, never against the live one while the services run. A model that doesn't fit beside the services' always-loaded ones evicts them, the services load theirs straight back, and the two evict each other for as long as the run goes (63 reloads in 20 minutes on the Linux machine).

**No browser automation or remote devices.**
- Never use the claude-in-chrome tools, agent-browser, computer-use, or anything else that drives a browser or reaches another machine. The Chrome extension drives a person's own laptop, not this machine.
- Screenshots and visual checks come only from a headless browser (Playwright) on the machine itself, from a worktree, against the worktree Vite server described under Tests. Close it when you're done.
  Playwright is a frontend dev dependency. Fetch its headless Chromium once per machine, and again after a Playwright upgrade (each release pins its own browser build), with `mise run playwright` (opt-in, about 280 MB on disk), then take each page in light and dark from `frontend/`, for example `npx --no-install playwright screenshot --color-scheme=dark --viewport-size=1280,800 --wait-for-timeout=1500 http://localhost:<vite port>/<page> /tmp/<page>-dark.png`, and the same with `--color-scheme=light`. Look at the PNG files; never drive a visible browser, or a browser on anyone's own computer.

**Declare dependencies; don't script installs.**
- Python packages: `uv add <pkg>` (newest version; pin older only when something breaks, and say why in the commit). Continuum's SDK, `continuum_py`, is a path dependency on `continuum/` that maturin builds from source on `uv sync`, so every machine needs the Rust toolchain (rustup); uv rebuilds it only when the Rust or the package changes (`continuum/pyproject.toml`'s cache-keys). v1 modules still import it only for types; the modules ported onto Continuum are listed in `pyproject.toml`'s ruff per-file-ignores.
- In the live checkout, a bare `uv sync` removes the optional extras (torch, transformers, and onnx), and the vision and listen services fail on their next start. Sync with `tools/build-slot.sh uv sync --inexact` and the extras `mise run setup` installs: `faces`, `segmentation`, `deep-asr`, and `voice-clone` where `nvidia-smi` is on PATH. [MIGRATING.md](MIGRATING.md) has the command.
- Command-line tools: `mise.toml`.
- Programs and libraries from the operating system (apt and Homebrew packages, such as sox or espeak-ng): `config/system-packages.toml`, with why and the services that use it. `mise run setup` installs them after asking, `pack-services install` warns when one is missing, and the dashboard's Machine panel lists them.
- Models: `config/defaults.toml` under `[models]`, fetched by `mise run models`.
- No new `install-*.sh` scripts and no `pip install` instructions.

**Tests.**
- Add or update tests for behavior you change, next to the existing ones.
- Write and review tests by [.claude/skills/python-tests/SKILL.md](.claude/skills/python-tests/SKILL.md) (Claude Code loads it automatically; other agents read it before writing tests). Property-based tests use Hypothesis, with [.claude/skills/property-based-testing/](.claude/skills/property-based-testing/SKILL.md) as the guide.
- Tests must never read or write the real `data/` directory (state, logs, faces, voices, calls, or models). Point modules at a temp dir and set identity with `config.override(...)`.
- Tests must never touch the real services: nothing in a test may run `pack-services`, `systemctl`, or `launchctl`, or re-exec the process. `tests/conftest.py` refuses those calls and fails the test; patch the function that would make them. `pack-services install` and `uninstall` also refuse to run from a git worktree or a temporary directory. A temporary worktree is for running the suite only, never `pack-services`, `mise run dev`, or a dashboard. One exception, for looking at frontend work before it merges: from a worktree you may run Vite alone (`frontend/node_modules/.bin/vite`, never `pack-dashboard`, `pack-services`, or `mise run dev`) on a free port, with a proxy that forwards GET requests only to the live dashboard. Use it to view pages (in headless Playwright on the same machine, never a browser on another one), never to save, delete, or start anything, and stop it when you're done.
- `mise run fmt` and `mise run lint` are clean, and the tests that cover your change pass (apart from failures that were already there), before you commit. Run them as "Run only the tests you need" and "Share the machine" say.

**Commits.**
- Commit your own work when it's done, as small focused commits with messages that say what changed and why. Nothing commits your work for you.
- Never commit runtime data (anything under `data/`, or `.env*`), and never `git add -f` a gitignored file. A tracked runtime file is overwritten on every other install the next time it pulls, so committing your transcript or `speakers.json` replaces other people's. Relabels, merges, and other fixes to runtime data happen in place on the machine that owns it, never as a commit. The pre-commit hook (`mise run hooks`) refuses these files and `tests/test_repo_hygiene.py` fails if one is tracked; don't work around either.
- Never commit unresolved conflict markers. When a merge, rebase, or script step fails, stop there and resolve it; don't let the next step commit what's staged. The pre-commit and pre-merge-commit hooks refuse staged `<<<<<<<`, `=======`, and `>>>>>>>` lines, and pre-push refuses pushed commits that add them, which covers rebase and `git am`, since they skip pre-commit. Don't skip any of them with `--no-verify`.
- If a change breaks an existing install (renamed setting, moved file, new service), add the steps to [MIGRATING.md](MIGRATING.md) in the same commit.

## Before you finish

- `mise run fmt` and `mise run lint` pass, and so do the tests that cover your change (never the full suite: "Run only the tests you need" in Rules), with non-Rust checks niced and Rust commands at standard priority through `tools/build-slot.sh`. Lint includes `ty check`, so every function you added or changed has type hints and the checker is clean. It also fails on anything `mise run fmt` would change (`ruff format --check` and `biome format`), so run fmt first.
- `git status` shows only the changes you meant to make, and no stray files.
- Docs touched by your change are updated.
- Your PR says `Closes #N` (or `Part of #N`) for the issue it works on, and anything you found but didn't fix is filed as an issue in Ideas.
- Your workspace notes say what you did and where, so the next session can pick it up without rereading the code.
