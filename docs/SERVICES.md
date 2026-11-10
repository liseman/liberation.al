# Services

How the pack's processes are declared, supervised, started, restarted, and logged, and where an install keeps its
files. [RUNNING.md](RUNNING.md) has the day-to-day commands; [COMPONENTS.md](COMPONENTS.md) lists the services
themselves.

## Declaring a service

`services/services.toml` declares every daemon and scheduled job once; its header documents each key. A new daemon
gets a `main()`, a `pack-<name>` command in `pyproject.toml`, and an entry there.

`services.selected()` (`services.py`) chooses which services run on a machine: it starts from each service's `default`,
adds services turned on by their `enabled_by` setting (`bluetooth.enabled`, `network_devices.enabled`) or required by
another setting (`whisper` when `asr.engine = "whisper"`), adds `services.enable` and removes `services.disable` from
`pack.toml`, filters to `platforms`, and drops services marked `source_build = true` outside a checkout.

## One application

`pack-app` (`app_window.py`) is the application the owner opens: the supervisor plus a small Tk window that stays on
screen for as long as any service runs. The window shows whether the pack is recording, paused for privacy, or missing
a declared camera or microphone (`continuum-capture` keeps running when one device fails to open, so the window is
what surfaces a silent device), opens the dashboard, and quits. **Quit and stop recording** and closing the window
(which asks first) both drain Tier 0 and stop every service, so nothing records without the window. `pack-supervisor
run` runs the same supervisor without a window.

Tk (`tkinter`) ships with `uv`'s Python on Linux and macOS, so the window needs no system GTK bindings, C build step,
or extra GUI wheels, and appears on Wayland desktops where tray icons require extra extensions. The supervisor runs in
a thread of the `pack-app` process so child services inherit the application's camera and microphone permissions on
macOS while Tk keeps the main thread.

The supervisor (`supervisor.py`):

- starts each selected service as its own child process in group order: `dashboard` and `continuum-gateway`,
  `continuum-broker`, `continuum-tier-worker`, `continuum-capture`, then the remaining readers;
- restarts a child that exits unexpectedly after its `throttle` delay, doubled on each consecutive failure up to 5
  minutes, and resets the failure count once a run stays up for 60 seconds;
- runs scheduled jobs on its own clock, and runs a missed daily job once 2 minutes after starting if it has run before;
- keeps the service list it read at startup until `pack-supervisor reload` clears the cached settings and re-reads
  `services.selected()`, stopping removed services and starting newly added ones;
- on quit stops readers, stops capture, lets `continuum-tier-worker` flush Tier 0 and stops it, stops the broker, and
  stops the dashboard and gateway last;
- builds each child's environment from the pack's settings and `services.toml`, passing through only the session
  variables audio and display need, never the launching shell's loader or packaging variables;
- writes live state to `state/supervisor/status.json` and transition history to `events.jsonl`, and serves
  `control.sock` (mode `0600`). During a requested restart, `status.json` sets `restarting`, so the window says
  `Restarting (N of M up)` rather than reporting capture as down.

Nothing starts at login unless `app.start_at_login` is on. On Linux that writes an XDG autostart entry
(`~/.config/autostart/liberational.desktop`) for `pack-app` (`login_item.py`).

While a supervisor runs, `pack-services` acts as its client over `control.sock` and `status.json`. Without a
supervisor, `pack-services install` can instead render and load per-service `systemd` user units or `launchd` agents.
At startup the supervisor refuses to run if any selected service is already running outside it, so the two modes never
manage the same service; `pack-services migrate` stops and removes per-service units.

## Startup and readiness

- The broker opens the catalog, restores or rebuilds its availability index, and creates a new Tier 0 arena before it
  serves clients ([CONTINUUM-BROKER.md](CONTINUUM-BROKER.md)). It never exposes a partial index.
- A service with `wait_for_broker = true` starts through `continuum_ready.py`, which waits up to 60 seconds for an SDK
  handshake and a catalog request. Permanent errors fail at once.
- A service with `wait_for_gpu = true` (`ollama`, `see-actions`) starts through `gpu_ready.py`, which on Linux with
  NVIDIA hardware waits up to 15 minutes for `nvidia-smi -L` to list a GPU, then starts anyway. A program that picks
  its device once would otherwise stay on the CPU after a slow driver start at boot.
- `continuum-broker`, `dashboard`, and `continuum-gateway` are ungated, so diagnostics and recorded playback work
  while the broker is down.
- Outside a cascade, device readiness belongs to each consumer. One absent camera never blocks unrelated services.

## The cascade

A restart or unit install that includes `continuum-broker` or `continuum-capture` (or `pack-services restart --cascade`,
or `pack-supervisor restart` with no names) runs `service_cascade.py`.

1. **Stop in reverse dependency order.** Stop readers other than `continuum-tier-worker`, stop `continuum-capture`,
   wait up to 20 seconds for `continuum-tier-worker` to flush remaining Tier 0 payloads to disk, stop
   `continuum-tier-worker`, and stop `continuum-broker`, confirming after each stage that the old processes and their
   children exited.
2. **Start and prove the core.** Start `continuum-broker` and wait for a new Tier 0 arena generation
   (`continuum_ready.py`). Start `continuum-tier-worker` before capture so it subscribes before capture writes frames
   that could be reclaimed under pressure. Start `continuum-capture`, confirm through `continuum_ready.py` that every
   enabled camera and microphone in `devices.toml` delivers fresh frames or audio on the new arena, then start the
   remaining readers.
3. **Refresh diagnostics and verify stability.** `dashboard` and `continuum-gateway` stay up through all core stages.
   When a request includes them (including `pack-services restart --cascade` or `pack-supervisor restart` with no
   names), the cascade stops and restarts them once capture and the readers are back; restarting `continuum-broker` or
   `continuum-capture` by name leaves both running untouched and leaves stopped services stopped unless named.
   `cascade_health.py` then checks that process identities, capture worker cohorts (excluding replaceable RTSP
   workers), restart counts, and schedules hold steady, and runs `continuum_status` on the vision port when any `see-*`
   service is active, before `continuum_ready.py` re-checks the broker's arena identity.

A missing configured device or an unexpected mid-cascade restart fails the cascade and names the stage. Recovery
restarts whatever the cascade stopped and reports anything that would not come back. Unit-mode cascades serialize on
`state/continuum/service-cascade.lock`, and every cascade appends its stages to `paths.logs/cascade.log`.

`continuum-tier-worker` declares `fresh_config`, which wraps its unit command in
`python -m liberational.services exec continuum-tier-worker` so unit-based restarts expand the current settings at each
start without re-rendering unit files.

## Logs

Python services log through `liberational.logs`: one line per record, `<ISO stamp with offset> [tag] LEVEL message`, to
`paths.logs/<service>.log`. Rust daemons log through `continuum_log` in the same format, and third-party programs with
`log_relay = true` (`ollama`, `ask-ollama`) run under `liberational.log_relay`, which tags their lines in the same
format and drops lines below the configured level. `logging.level` and `logging.per_service` set the threshold. The
hourly `log-trim` job copies any log past 10 MB to `<service>.log.1` and empties the active file.
[AGENTS.md](../AGENTS.md) has the level rules, and [RUNNING.md](RUNNING.md#read-and-rotate-service-logs) covers reading
and rotating logs.

## Install layout

`config.py` keeps the code's resources apart from the owner's files; its docstring is the full rule.

| | In a checkout, or under `PACK_ROOT` | Installed from a wheel or `apt` |
|---|---|---|
| Resources (`defaults.toml`, `labels.toml`, `system-packages.toml`, `services.toml`) | `src/liberational/resources/`, symlinks into `config/` and `services/` | The package directory |
| `pack.toml` | The top of the checkout (`PACK_CONFIG` names another file) | `~/.config/liberational/pack.toml` |
| The device manifest | `config/devices.toml` | `~/.config/liberational/devices.toml` |
| State, logs, and models | `data/` | `~/.local/share/liberational/` |
| Programs | `.venv/bin` | The tool environment's `bin/` (`~/.local/share/uv/tools/liberational-pack/bin` with `uv`, or `~/.local/lib/liberational/liberational-pack/bin` per user under `/usr/bin/pack-*` with `apt`) |

The installed directories follow `XDG_CONFIG_HOME` and `XDG_DATA_HOME`. On macOS both are
`~/Library/Application Support/liberational`.

- `mise run setup` sets up a checkout. `pack-setup` sets up an installed package: it builds nothing, checks system
  packages, installs this machine's Python extras against `resources/constraints.txt`, fetches models, and says how to
  open the application.
- `mise run wheel` builds the frontend and the wheel into `dist/`, then runs `tools/check_wheel.py`, which fails on a
  wheel missing a resource, the frontend, or a command.
- [RELEASES.md](RELEASES.md) covers release wheels, platform locks, and the `apt` package.
