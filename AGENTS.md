# Working in this repo (for agents)

Read this before changing anything. It applies to every coding agent and anything it spawns. People should read
[CONTRIBUTING.md](CONTRIBUTING.md); this file adds the rules agents need.

Start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) to find where a change belongs, and
[docs/CONVENTIONS.md](docs/CONVENTIONS.md) for what may appear in this public repository.

## Rules that prevent damage

1. **This repository is public.** No names of real people or pets, no transcript lines or other live content, no
   addresses or room layouts, no hostnames or network addresses, no secrets, and no agent, session, or machine
   names. That covers code, tests, docs, commit messages, branch names, issues, pull requests, and comments.
   [docs/CONVENTIONS.md](docs/CONVENTIONS.md) is the full list. Tests use made-up people: Alice, Dana, Sam, and Rex.
2. **Never skip a hook with `--no-verify`.** The hooks (`mise run hooks`) run `tools/privacy_check.py` before every
   commit and push. Fix a false alarm by changing the text, or add an entry to `.privacy-allow.toml` with a reason.
   The hooks cannot read an issue or a comment: check those yourself before posting.
3. **The live checkout is read only.** Work in your own worktree on a branch off `origin/main`
   (`git worktree add .worktrees/<name> -b <area>/<description> origin/main`) and start each command with a `cd`
   into it. Never run `git reset --hard`, `git checkout -- .`, `git clean`, or `git stash` in any checkout.
4. **Never touch the running services from a worktree.** No `pack-services`, `pack-dashboard`, `mise run dev`, or
   `uv sync` there, and nothing in a test may run `pack-services`, `systemctl`, or `launchctl`. In the live checkout,
   never run a bare `uv sync`: it removes the optional extras and the services fail on their next start
   ([MIGRATING.md](MIGRATING.md) has the command).
5. **Never run the full test suite** (`mise run test`, a bare `pytest`, or a bare `vitest run`) unless a person asks.
   Run the tests for what you changed: `mise run test-changed`, or
   `tools/test-slot.sh .venv/bin/python -m pytest -q tests/test_<area>.py`. Guard a computed test list against being
   empty, since an empty list runs everything.
6. **Share the machine.** Tests and builds you start go through `tools/test-slot.sh` (lowest priority, two at a
   time). Every Cargo command goes through `tools/build-slot.sh`, one at a time machine-wide, at standard priority.
   Run `mise run continuum-release` bare: wrapping it deadlocks. Run Cargo only when Rust source or Cargo files
   changed. No benchmarks, stress runs, or loops without asking, and no large models against the live Ollama.
7. **Never commit runtime data.** Nothing under `data/`, no `.env*`, and never `git add -f` a gitignored file. Tests
   never read or write the real `data/`: point modules at a temp directory and set identity with
   `config.override(...)`.
8. **A state reset keeps what the owner taught.** "Wipe the state" means machine-made data:
   `pack-reset-catalog --keep-taught`. The knowledge store, the face gallery, voice references, `paths.voices`,
   `paths.measurement`, and models stay unless the owner names them. Before any reset that removes taught state, run
   `pack-knowledge export` to a path outside the deletion and `verify` it.
9. **No browser automation or remote devices.** Never drive a browser on anyone's computer. Screenshots come only
   from headless Playwright on the machine itself (`mise run playwright` fetches its browser).
10. **Never merge with `--delete-branch` from a checkout that running services use.** It switches that checkout to
    main and pulls it under them.

## How we build

This is research. Building the right thing directly beats preserving what exists.

- **Machine-made data is expendable.** Recordings, detections, tracks, the catalog, and worker state can be dropped
  or reset. Do not write migrations, compatibility shims, or dual paths for them. What the owner taught is the
  exception: carry it across with an explicit export and import.
- **Gaps are fine.** Delete what is wrong now. While a feature is being rebuilt, its readers say so plainly.
- **Fast first, tune later.** A working end-to-end path with stock models beats a careful slow one.
- **Use the solved approach.** A fixed history window, one hardware decode per stream, per-hour columnar files, and
  models instead of hand-written rules.
- **Question whether we need it.** Often the right move is to delete a feature. Keep simple code that is simple for
  a reason.
- **Measure before you guess.** Profile ([docs/PROFILING.md](docs/PROFILING.md)) before optimizing.
- **Examples illustrate goals.** A request's examples show what a goal means; build for the general goal.

## Where things go

| Path | What belongs there |
|---|---|
| `src/liberational/` | The application package, one subpackage per area ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)) |
| `continuum/` | The Rust data layer ([docs/CONTINUUM.md](docs/CONTINUUM.md)) |
| `frontend/` | The dashboard; its build output `src/liberational/dashboard/static/app/` is gitignored ([docs/DASHBOARD.md](docs/DASHBOARD.md)) |
| `tests/` | `test_<module>.py` and fixtures in `tests/data/` |
| `tools/` | Benchmarks, probes, and maintenance scripts worth keeping, each with a docstring saying how to run it |
| `config/` | `defaults.toml` (every setting, documented), examples, and `system-packages.toml` |
| `services/services.toml` | Every daemon, declared once |
| `src/liberational/resources/` | Symlinks to the `config/` and `services/` files the code reads at run time |
| `packaging/` | What a platform's package holds beyond the wheels: `deb/` is the `.deb`'s launcher, maintainer scripts, desktop entry, icon, and container test ([docs/RELEASES.md](docs/RELEASES.md)) |
| `docs/` | Documents ([docs/README.md](docs/README.md) is the map), written by the documentation maintainer |
| `.claude/skills/` | Skills to load before a kind of work: `python-tests` and `property-based-testing` before tests, `diataxis` before documentation, `mantine` before dashboard UI, `modern-web-guidance` before other frontend work |
| `scratch/` | Gitignored: experiments you won't keep |

- Extend an existing module before adding one. Search first.
- No files at the repo root, no one-off scripts in the code tree, and no per-feature notes files.
- A new daemon gets a `main()`, a `pack-<name>` command in `pyproject.toml`, and an entry in
  `services/services.toml`. Never hand-write unit files.
- Use absolute imports (`from liberational.speech import asr`). Never edit `sys.path`.

## Code rules

- **No hardcoded personal or machine values.** Names, numbers, hostnames, ports, model names, and paths go through
  `liberational.config`, with a generic default and a comment in `config/defaults.toml`. Prompts name people through
  `config.owner_or()` and `config.assistant_name()`.
- **Keep it portable.** It runs on Apple Silicon (MLX) and Linux (CUDA, ONNX Runtime). Add the fallback, or fail with
  a clear message. No absolute home paths and no Homebrew paths.
- **Type everything.** Every function in `src/` and `tools/` has type hints, and `mise run lint` (Ruff, then ty)
  passes. A `# ty: ignore[rule]` needs a comment saying why. Do not exclude files from the checkers.
- **Log with levels.** A service logs through `liberational.logs` (`LOG = logs.get("<tag>")`), never `print`. ERROR
  needs attention; WARN is degraded or retrying; INFO is normal progress; DEBUG is per-item detail. Periodic counters
  go through `logs.Periodic`. A Rust daemon calls `continuum_log::init` and uses `tracing`.
- **Declare dependencies.** Python: `uv add <pkg>`, newest version, pinned only on breakage. Tools: `mise.toml`.
  System packages: `config/system-packages.toml`. Models: `[models]` in `config/defaults.toml`. No install scripts.
- **Tests.** Add or update tests for behavior you change. Follow
  [.claude/skills/python-tests/SKILL.md](.claude/skills/python-tests/SKILL.md).
- **Dashboard UI.** Before writing or reviewing it, load
  [.claude/skills/mantine/SKILL.md](.claude/skills/mantine/SKILL.md) (our conventions, then Mantine's skills and docs).
  For HTML, CSS, browser APIs, performance, or accessibility, also load `.claude/skills/modern-web-guidance`
  (`SKILL.md`, then `PROJECT.md`); ours and Mantine's conventions win where they differ.

## Write things down in the right place

- **Do not edit `docs/` or `README.md`.** One documentation maintainer writes them, following the Diataxis skill
  ([.claude/skills/diataxis/SKILL.md](.claude/skills/diataxis/SKILL.md), then `PROJECT.md` beside it). Load that
  skill before touching documentation at all.
- A change that affects what a user or contributor needs to know: say so in a "Docs:" section of the PR body (what
  changed, in a few lines). That covers a new daemon, port, state file, or external service, and a design decision
  that spans modules. The documentation maintainer turns it into documentation.
- Two things ship in your own PR, because an install breaks without them: the steps in [MIGRATING.md](MIGRATING.md)
  for a change that breaks an existing install, and the release notes in [docs/RELEASES.md](docs/RELEASES.md). Keep
  them short; the documentation maintainer edits them afterwards for form.
- A design decision or gotcha inside one module: its docstring or a comment. Those stay yours. Not a dated log.
- A new key in `config/defaults.toml`: document it there, and say in the PR that every config-reading service needs
  a restart.
- Planned work: an issue on the board. Your session history: your own tool's memory, never the repository.

## Tracking and merging

Work is tracked as GitHub issues on the [project board](https://github.com/liseman/liberation.al/projects).

| Column | Meaning | Who moves cards here |
|---|---|---|
| Ideas | Proposed | Anyone |
| Vetting | Being discussed | Anyone |
| Ready | Agreed and clear enough to pick up | Maintainers only |
| In progress | Being worked on | Whoever picks it up |
| Done | Merged or closed | Closing the issue does it |

- Look for an existing issue before starting. File what you find but won't fix in Ideas.
- Pick up only Ready issues or ones a maintainer asked for. Assign the issue to your GitHub account and move its
  card to In progress. That is the claim.
- A pull request says `Closes #N` (or `Part of #N`), what changed, which checks ran with counts, and "Docs:" when
  the change affects what a user or contributor needs to know.
- Before merging: `mise run fmt`, lint and type-check what you changed, run the tests that cover it, and read the
  diff. The `privacy` check must pass. The hosted `checks` workflow is informational: do not wait for it, and fix
  forward if it turns red.
- Merge `main` in first, then `gh pr merge N --repo liseman/liberation.al --merge`.
- After the merge, remove your worktree and delete the branch locally and on GitHub.

To move a card (`item-add` is safe to repeat):

```sh
item=$(gh project item-add <number> --owner liseman --url https://github.com/liseman/liberation.al/issues/N --format json --jq .id)
gh project item-edit --id "$item" --project-id <project id> --field-id <Status field id> --single-select-option-id <option id>
```

`gh project list --owner liseman` and `gh project field-list <number> --owner liseman --format json` give the ids.
