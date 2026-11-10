# Contributing

Start with [MANIFESTO.md](MANIFESTO.md): it is short, and it is why this project exists. Then read
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the parts fit together and [docs/ROADMAP.md](docs/ROADMAP.md)
for where it is going.

## Setup

Follow [Install](README.md#install) and [Configure](README.md#configure) in the README. The pack runs on Linux with
an NVIDIA GPU and on macOS with Apple Silicon.

| Declared in | What |
|---|---|
| `mise.toml` | Tools and tasks (`mise tasks` lists them) |
| `pyproject.toml` | Python dependencies, extras, and the `pack-*` commands |
| `config/system-packages.toml` | Programs and libraries from the operating system, each with why |
| `config/defaults.toml` | Every setting and model |
| `services/services.toml` | Every daemon and scheduled job |

- **System packages.** `mise run setup` lists the missing ones and asks before installing them with apt or Homebrew.
  On another Linux, `uv run pack-system check` lists what to install yourself.
- **Extras.** Setup installs the `faces`, `segmentation`, and `deep-asr` extras, plus `voice-clone` when
  `nvidia-smi` is present. A plain sync removes optional packages, so keep your extras when you sync
  ([MIGRATING.md](MIGRATING.md) has the command).
- **Git hooks.** `mise run setup` turns on the hooks in `.githooks/` (`mise run hooks` does it alone).
- **Devices.** List your cameras and microphones in `config/devices.toml`
  ([docs/CONTINUUM-CAPTURE.md](docs/CONTINUUM-CAPTURE.md)).

## Running things

- `mise run dashboard` starts the dashboard at <http://localhost:8765/>. It works with an empty `data/` directory.
  Without a `pack.toml` it opens on a setup screen.
- `mise run dev` runs it with hot reload for frontend work.
- `mise run demo` opens the dashboard on a made-up household, with no real data.
- `uv run pack-app` opens the application, which runs every service. `uv run pack-services status` checks them
  ([docs/SERVICES.md](docs/SERVICES.md)).
- Each daemon and tool is a command under `[project.scripts]` in `pyproject.toml`, such as `uv run pack-config`.

## Ground rules

- **This repository is public.** Read [docs/CONVENTIONS.md](docs/CONVENTIONS.md) before you write code, a commit
  message, an issue, or a pull request. None may carry names of real people or pets, transcript lines, addresses,
  hostnames, or secrets. Tests and examples use made-up people and data.
- **Never commit private data.** `data/` and `.env*` hold real recordings of real people. They are gitignored; keep
  it that way. The hooks refuse gitignored files and run `tools/privacy_check.py` on every commit and push. They do
  not know names, so those are yours to keep out. Never skip the hooks with `--no-verify`. A false alarm is fixed by
  changing the text, or by an entry in `.privacy-allow.toml` with a reason.
- **Tests must not touch the real `data/`.** Point modules at a temp directory.
- **Keep it local-first.** New features work with local models. Hosted services are optional and degrade
  gracefully.
- **Keep both platforms working:** Apple Silicon and Linux.
- **Prefer the newest versions** of Python and packages. Pin an older one only when something breaks, and say why.
- **The code is typed.** Every function in `src/` and `tools/` has type hints.

## Making a change

1. Find or open an issue on the [project board](https://github.com/liseman/liberation.al/projects). New ideas go in
   Ideas; a maintainer moves them to Ready.
2. Branch from `main` and make the change, with tests.
3. Run `mise run fmt` and `mise run lint` (Ruff, ty, Biome, and tsc), then the tests that cover what you changed:
   `mise run test-changed`, or `tools/test-slot.sh .venv/bin/python -m pytest tests/test_<area>.py`. Do not run the
   full suite: it is slow and the machine may be shared. Run `mise run frontend` if you changed frontend code.
4. Open a pull request that says `Closes #N`, what changed and why, and what you ran. Say how you checked it on real
   hardware if it touches audio, cameras, or models.

## Checks on GitHub

- **`privacy`** runs `tools/privacy_check.py` on every push and pull request. It must pass before a merge. If it
  fails, the push is already public: follow "If something private gets out" in
  [docs/CONVENTIONS.md](docs/CONVENTIONS.md) first.
- **`checks`** lints and tests only what a change touches (`python`, `frontend`, and `rust` jobs). It is
  informational, not required: nobody waits for it, and a red run after a merge is fixed forward. The hosted runner
  has no GPU, models, or Ollama, so tests that need one skip there and run locally.

Read the diff yourself before merging, and merge `main` in first.
