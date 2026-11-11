# Running the pack and its services

Open and quit the application, check what is running, stop or restart individual services, restart everything after an
update, turn services off or on across runs, start at login, and read logs.

In a wheel or `apt` install, the `pack-*` commands are on `PATH`. In a checkout, run them from the main checkout with
`.venv/bin/<command>` or `uv run --no-sync <command>`. [SERVICES.md](SERVICES.md) explains how the supervisor and
restart cascade work; [COMPONENTS.md](COMPONENTS.md) and [`services/services.toml`](../services/services.toml) list
every service.

## Open and quit the application

In a desktop session, open the application from your desktop's applications menu or a terminal:

```sh
pack-app
```

`pack-app` starts every selected service and keeps a small status window on screen. Click **Open the dashboard** to
open the web interface in your browser. Click **Quit and stop recording** to flush Tier 0 and stop every service.
Closing the window from its title bar does the same after asking "Stop recording and quit?". Once everything has
stopped, **Close** ends the window and **Start again** runs the services once more.

On a machine without a desktop session, run the supervisor in the foreground instead:

```sh
pack-supervisor run
```

Stop a running supervisor (with or without a window) from another terminal:

```sh
pack-supervisor quit
```

## See what is running

Check each selected service's state:

```sh
pack-services status
```

Show each service's state, restart count, and last transition timestamp from the running supervisor:

```sh
pack-supervisor status
```

List every declared service (`*` marks the ones selected on this machine), or print only what the open application
runs (nothing while it isn't open):

```sh
pack-services list
pack-services installed
```

Check end-to-end health (services, broker socket, capture streams, Tier 0 headroom, tier-worker flush lag, and
gateway):

```sh
pack-continuum-status
```

## Stop, start, or restart one service

Stop one or more services for the rest of the current run:

```sh
pack-services stop journal
```

A stopped service stays off until you start it again or reopen the application:

```sh
pack-services start journal
```

Restart leaf services without cycling capture or the broker:

```sh
pack-services restart see-captions journal
```

Restart `continuum-broker` or `continuum-capture` through the staged cascade, which stops readers first and flushes
Tier 0 before cycling the core:

```sh
pack-services restart continuum-broker
```

Naming `continuum-broker` or `continuum-capture` leaves `dashboard` and `continuum-gateway` running without restarting
them, and keeps stopped services off unless you name them too. `pack-supervisor start`, `pack-supervisor stop`, and
`pack-supervisor restart` take the same service names directly.

## Restart everything after an update

Run a full staged restart of every selected service:

```sh
pack-services restart --cascade
```

`pack-services restart --cascade` (or `pack-supervisor restart` with no names) keeps `dashboard` and
`continuum-gateway` up while the core restarts, refreshes both once capture and the readers are back, and starts any
selected service that was stopped. While a requested restart runs, the `pack-app` window shows
`Restarting (N of M up)`.

When an update changes the supervisor itself or replaces the environment, quit the application (`pack-supervisor quit`
or **Quit and stop recording**) and run `pack-app` again. Checkout upgrades and one-time migration steps are in
[MIGRATING.md](../MIGRATING.md).

## Turn a service off or on for good

Update `[services]` in `pack.toml`:

```sh
pack-services disable see-captions
pack-services enable bluetooth
pack-services reset see-captions
```

`disable` adds names to `services.disable`, `enable` adds names to `services.enable`, and `reset` removes them from
both lists so they follow their defaults in [`services/services.toml`](../services/services.toml). None of the three
starts or stops a running process on its own.

Apply the updated `[services]` selection to a running supervisor without restarting unchanged services:

```sh
pack-supervisor reload
```

`reload` stops services removed from the selection and starts newly added ones.

## Start at login

On Linux, tick **Start at login** in the `pack-app` window, or turn on **Start at login** in the dashboard's Settings
(`start_at_login = true` under `[app]` in `pack.toml`). That writes `$XDG_CONFIG_HOME/autostart/liberational.desktop`
(`~/.config/autostart/liberational.desktop`), so your desktop opens `pack-app` when you sign in. Untick the box or set
`start_at_login = false` to remove the file.

On macOS, the checkbox is disabled because login items require an application bundle.

## Read and rotate service logs

Open the dashboard's **Debug** tab to tail and filter logs by service and level, or read the files directly under
`paths.logs` (`data/logs/` in a checkout, `~/.local/share/liberational/logs/` on Linux, and
`~/Library/Application Support/liberational/logs/` on macOS; see [SERVICES.md](SERVICES.md#install-layout)):

```sh
tail -f ~/.local/share/liberational/logs/continuum-capture.log
```

Each service writes `<service>.log`, with one rotated `<service>.log.1` file beside it; staged restarts also write
`cascade.log`. To change verbosity, set `level` or `per_service` (for example, `per_service = ["see-actions=debug"]`)
under `[logging]` in `pack.toml` and restart that service.

The `log-trim` job rotates logs past 10 MB every hour. Run the same rotation immediately with:

```sh
pack-services trim-logs
```
