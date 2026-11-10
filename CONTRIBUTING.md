# Contributing

Start with [MANIFESTO.md](MANIFESTO.md). It's short, and it's the reason this project exists.

Then read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces fit together, and [docs/ROADMAP.md](docs/ROADMAP.md) for where things are headed and what's up for grabs.

## Setup

Follow [Install](README.md#install) and [Configure](README.md#configure) in the README: mise, a Rust toolchain, the clone, `mise run setup`, and `mise run continuum-release`. The pack runs on macOS with Apple Silicon (the original target) and on Linux. Tools and tasks are declared in `mise.toml`, Python dependencies in `pyproject.toml`, the programs and libraries needed from the operating system in `config/system-packages.toml`, and models in `config/defaults.toml`.

What the README leaves out, for someone changing the code:

- **System packages.** Some of what Liberational needs comes from the operating system, not from uv or mise: on Linux, PortAudio, espeak-ng, OpenCV's graphics libraries, and (with an NVIDIA GPU, for voice cloning) sox, plus a C++ compiler and git everywhere. Each is listed with why in `config/system-packages.toml`. You don't install them by hand: `mise run setup` lists the missing ones and asks before running `sudo apt-get install` or `brew install`. Optional ones (an XVF3800 microphone's libusb, PipeWire's `pw-dump`, the Mac audio routing helpers) install with `uv run pack-system install --optional`. Liberational knows apt-get (Debian and Ubuntu) and Homebrew; on another Linux, `uv run pack-system check` lists the packages to find with your own package manager.
- **whisper-server.** `mise run setup` also runs `mise run whisper`, which builds whisper.cpp's `whisper-server` (for the optional whisper service) into `data/whisper.cpp/`: with CUDA when `nvcc` is installed, Metal on Apple Silicon, and CPU otherwise. It needs `git` and a C++ compiler, which the system packages step installs (on macOS the compiler comes with the Xcode Command Line Tools: `xcode-select --install`), plus the CUDA toolkit for GPU builds; `cmake` comes from `mise.toml`. The first build takes a few minutes, mostly for CUDA.
- **Extras.** Setup installs the `faces`, `segmentation`, and `deep-asr` extras, plus `voice-clone` when `nvidia-smi` is present. Other extras, including `oakd` and `mic-array`, are declared in `pyproject.toml`. Preserve the extras your installation uses when syncing; a plain `uv sync` removes optional packages. On a shared development machine, run a sync that can rebuild the native SDK through `tools/build-slot.sh`, at standard priority.
- **Git hooks.** `mise run setup` turns on the hooks in `.githooks/` and writes your machine's private deny list for the privacy check, a list of its actual secrets (`mise run hooks` does both on its own). See "Ground rules" below.
- **Devices and services.** Configure the capture devices in `config/devices.toml` and review the selected services before installing them. [Architecture](docs/ARCHITECTURE.md) describes their dependencies; [the Continuum guide](docs/CONTINUUM.md) covers capture configuration. Check service status and fresh observations after startup. For code changes, run the targeted checks described below.

## Running things

- `mise run dashboard` starts the dashboard at http://localhost:8765/. It works with an empty `data/` directory, so it's a good first thing to run. Without a `pack.toml` it opens on a setup screen that asks for your settings, saves `pack.toml`, and restarts with them; the same settings are editable there later.
- `mise run dev` runs it with auto-reload for frontend work: Vite serves `frontend/` with hot reload, and the dashboard restarts when Python or config changes. Do not edit the generated `dashboard/static/app/` build.
- Each daemon and tool is a command: `uv run pack-services list`, `uv run pack-listen`, `uv run pack-config`, and so on. They're listed under `[project.scripts]` in `pyproject.toml`; the code is in `src/liberational/`. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) lists what each one does.
- To run the daemons as background services, `uv run pack-services install` sets them up with launchd (macOS) or systemd (Linux); `pack-services list` shows which ones and `pack-services status` checks them. `pack-services stop <name>` stops one until the computer restarts (it stays installed), and `start <name>` starts it again; `uninstall <name>` is what keeps one off. They're declared in `services/services.toml`.
- `mise tasks` lists the available tasks.

### Platform status

Speech uses Parakeet on MLX for Apple Silicon and sherpa-onnx elsewhere. Kokoro
provides speech synthesis on both platforms. Qwen3-TTS voice cloning uses MLX on
Apple Silicon or the `voice-clone` extra on supported Linux NVIDIA installations.
Vision backends and model artifacts vary by platform; inspect `config/defaults.toml`
and the [architecture](docs/ARCHITECTURE.md) rather than assuming every optional
model or device is available. A successful dependency install is not a hardware
validation.

## Ground rules

- **This repository is public.** Read [docs/CONVENTIONS.md](docs/CONVENTIONS.md) before you write code, a commit message, an issue, a pull request, or a comment. None of them may carry names of real people or pets, transcript lines or other live content, addresses or floor plans of a real home, hostnames or network addresses, secrets, or agent and session names. Tests and examples use made-up people and made-up data.
- **Never commit private data.** `data/` (state, calls, faces, voices, logs, and models), `.env*`, and camera frames hold real recordings of real people. They're gitignored; keep it that way, and check `git status` before you commit. The hooks that `mise run setup` turns on (or `mise run hooks` on its own) refuse gitignored files, so `git add -f` can't sneak one in, and they run `tools/privacy_check.py` on every commit, commit message, and push: a secret scan (gitleaks, which `mise install` provides), patterns for personal details, and your machine's private deny list of actual secrets (the credentials and tokens in your config, your phone numbers, your tailnet's addresses, and your street address once you add it), which `tools/privacy_check.py denylist` builds from your own install and which never leaves your machine. The hooks don't know names, so those are yours to keep out. Never skip the hooks with `--no-verify`. A false alarm is fixed by changing the text, or by an entry in `.privacy-allow.toml` with a reason.
- **Tests must not touch the real `data/`.** Point modules at a temp directory (see `tests/test_voice_auth.py` for the pattern). `git status` should be clean after `mise run test`.
- **Keep it local-first.** New features should work with local models. Cloud services are fine as optional add-ons that degrade gracefully when they're absent.
- **Prefer the newest versions** of Python and packages. Pin an older version only when something breaks, and say why in the commit.
- **Keep Apple Silicon working** when you add support for another platform, and the other way around.

## Making a change

1. Find or open the issue for it on the [project board](https://github.com/liseman/liberation.al/projects): new ideas go in Ideas, and a maintainer moves them to Ready once they're agreed. Then branch from `main`.
2. Make the change, with tests where it's practical.
3. Run `mise run fmt` and `mise run lint` (Ruff, then the ty type checker), then the tests that cover what you changed (`tools/test-slot.sh .venv/bin/python -m pytest tests/test_<area>.py`, and Vitest through the same slot wrapper for frontend changes). Don't run the full suite unless someone asks: it's slow and the machine is shared. The code is typed: every function in `src/` and `tools/` has type hints, and `ty check` passes.
4. Open a pull request that explains what changed and why, and how you checked it on real hardware if it touches audio, cameras, or models. Put `Closes #N` in the description so merging closes the issue.

Issue numbers in comments and docs written before this repository was published refer to the project's earlier tracker, not to issues here.

## Checks before merging

Two workflows run on GitHub-hosted runners (`ubuntu-latest`).

`privacy` runs `tools/privacy_check.py`, the same check the hooks run, on every
push and pull request (the whole tree, the pushed commits and their messages,
and a pull request's title and description). It must pass before a merge. It
can only report: by the time it runs, the push is already public, so when it
fails, follow "If something private gets out" in
[docs/CONVENTIONS.md](docs/CONVENTIONS.md) before anything else.

`checks` runs on every pull request and every push to `main`, and looks only at
what changed against the base, and its jobs must pass before a merge too:

| Job | Runs when | What it does |
|---|---|---|
| `python` | a `.py` file, `pyproject.toml`, or `uv.lock` changed | `uv sync` (no extras), Ruff and `ruff format --check` on the changed Python, `ty check` on the changed files under `src/`, `tests/`, and `tools/` (the whole tree for both when `pyproject.toml` or `uv.lock` changed), then `tools/test_changed.py --only python`: the same tests `mise run test-changed` picks, on the CPU |
| `frontend` | anything under `frontend/` changed | `npm ci`, Biome and tsc (`npm run lint`), `biome format`, the production build (`npm run build`), and `vitest related` on the changed files |
| `rust` | anything under `continuum/`, or a `Cargo.toml` or `Cargo.lock`, changed | `cargo fmt --check`, `cargo check --workspace --all-targets`, and `cargo test --workspace`, with the build cached between runs |

The hosted runner has no GPU, no models, no Ollama, and none of the optional
extras (`faces`, `segmentation`, `deep-asr`, `voice-clone`, and the rest). Tests
that need one skip themselves there (`pytest.importorskip`, or a `skipif` on the
model files or the device), so a skip in the hosted log is expected, and those
tests still run locally. What stays local: anything that needs a GPU, a model,
or Ollama, the FFmpeg build of Continuum (`continuum-check`'s clippy with the
`ffmpeg` feature), clippy itself, `ty check` with the MLX dependencies on an
Apple Silicon Mac, and checks on real hardware.

There is no waiting for a GitHub-side review: every
change is reviewed and checked locally before it merges. Review the diff
locally (a review agent, a local `codex review`, or a careful read), validate what it finds with a
failing test, and fix the real problems. Then merge `main` in and run
`mise run fmt`, `mise run lint` (Ruff, ty, Biome, and tsc, plus the format
checks), the tests that cover the files you changed (never the full suite
unless someone asks), and `mise run frontend` (the production build) when you
changed frontend code. A change that touches MLX code also gets
`mise run lint` on an Apple Silicon Mac, where `ty check` sees the MLX
dependencies. Record what you ran and the results in a comment on the pull
request.
