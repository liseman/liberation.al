# Services

How the pack's processes are declared, started, restarted, and logged, and where an install keeps its files.
[COMPONENTS.md](COMPONENTS.md) lists the services themselves.

## Declaring a service

`services/services.toml` declares every daemon and scheduled job once; its header documents each key. A new daemon
gets a `main()`, a `pack-<name>` command in `pyproject.toml`, and an entry there. Which services run on a machine is
`services.enable` and `services.disable` in `pack.toml`, on top of each service's `default`.

## One application

`pack-app` (`app_window.py`) is the application the owner opens: the supervisor plus a small Tk window that shows
whether the pack is recording, opens the dashboard, and quits. Quit and closing the window both drain and stop every
service, so nothing records without the window. `pack-supervisor run` is the same without a window.

The supervisor (`supervisor.py`):

- starts each selected service as its own child process, in cascade order;
- restarts a child that exits, waiting its `throttle`, doubled per failure in a row up to 5 minutes;
- runs scheduled jobs on its own clock, and runs a missed daily job once shortly after starting;
- on quit stops readers, capture, the tier worker (once Tier 0 is flushed), the broker, then the dashboard and
  gateway;
- builds each child's environment from the pack's settings and `services.toml`, passing through only the session
  variables audio and display need, never the launching shell's loader paths;
- records state in `state/supervisor/` and answers `pack-services` over `control.sock`.

Nothing starts at login unless `app.start_at_login` is on. On Linux that writes an XDG autostart entry for
`pack-app` (`login_item.py`).

`pack-services install` can instead render each service to a systemd user unit or a launchd agent. The two modes
never run the same services, and `pack-services migrate` moves a unit-based install to the supervisor.

## Startup and readiness

- The broker opens the catalog, rebuilds or restores its availability index, and creates a new Tier 0 arena before
  it serves clients. It never exposes a partial index.
- A service with `wait_for_broker = true` starts through `continuum_ready.py`, which waits up to 60 seconds for an
  SDK handshake and a catalog request. Permanent errors fail at once.
- A service with `wait_for_gpu = true` (`ollama`, `see-actions`) starts through `gpu_ready.py`, which waits up to 15
  minutes for `nvidia-smi -L` to list a GPU, then starts anyway. A program that picks its device once would
  otherwise stay on the CPU.
- The broker, dashboard, and gateway are ungated, so diagnostics and recorded playback work while the broker is
  down.
- Device readiness belongs to each consumer. One absent camera never blocks unrelated services.

## The cascade

A restart or install that includes the broker or capture runs `service_cascade.py`: stop readers, stop capture and
the broker, confirm the old processes exited, then start the broker, capture, and readers in that order.
`cascade_health.py` checks that each declared device delivers fresh frames or audio. A missing configured device
fails the cascade by name. After a failure the cascade restarts whatever it stopped and reports what would not
start. Services the owner stopped stay stopped unless the request names them. `state/continuum/service-cascade.lock`
serializes cascades.

`pack-services restart --cascade` keeps the dashboard and gateway up; restart those by name when a change touches
them.

The tier worker starts through `python -m liberational.services exec continuum-tier-worker`, which reads the current
settings at each start, so a settings change needs only a restart.

## Logs

Services log through `liberational.logs`: one line per record, `<ISO stamp with offset> [tag] LEVEL message`, to
`paths.logs/<service>.log`. `logging.level` and `logging.per_service` set the level. The hourly `log-trim` service
copies a log past 10 MB to `<service>.log.1` and empties it. Rust daemons log through `continuum_log` in the same
format. The dashboard's Debug tab tails and filters the logs. [AGENTS.md](../AGENTS.md) has the level rules.

## Install layout

`config.py` keeps the code's resources apart from the owner's files; its docstring is the full rule.

| | In a checkout, or under `PACK_ROOT` | Installed from a wheel |
|---|---|---|
| Resources (`defaults.toml`, `labels.toml`, `system-packages.toml`, `services.toml`) | `src/liberational/resources/`, symlinks into `config/` and `services/` | The package directory |
| `pack.toml` | The top of the checkout (`PACK_CONFIG` names another file) | `~/.config/liberational/pack.toml` |
| The device manifest | `config/devices.toml` | `~/.config/liberational/devices.toml` |
| State, logs, and models | `data/` | `~/.local/share/liberational/` |
| Programs | The environment's bin directory (`.venv/bin`) | The same |

The installed directories follow `XDG_CONFIG_HOME` and `XDG_DATA_HOME`. On macOS both are
`~/Library/Application Support/liberational`.

- `mise run setup` sets up a checkout. `pack-setup` sets up an installed package: it builds nothing, lists missing
  system packages, installs the machine's Python extras, fetches models, and says how to open the application.
- `mise run wheel` builds the frontend and the wheel into `dist/`, then runs `tools/check_wheel.py`, which fails on
  a wheel missing a resource, the frontend, or a command.
- A release is that wheel plus Continuum's SDK and daemons per platform. [RELEASES.md](RELEASES.md) covers
  installing and making one.
- On Ubuntu and Debian a release also installs with apt. The `.deb` (`tools/apt_package.py`, `packaging/deb/`)
  carries the wheels, the lock, a pinned uv, a desktop entry, and the `pack-*` commands as links to one launcher,
  and registers nothing with systemd. `pack-setup` builds each user's environment in `~/.local/lib/liberational`.
  An upgrade has a running application drain and quit, rebuilds that environment, and says to open it again.
  [RELEASES.md](RELEASES.md) has the repository, the signing key, and the container test.
