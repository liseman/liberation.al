# Dashboard

The web app the owner uses: `pack-dashboard` (`src/liberational/dashboard/`, port 8765) and the frontend in
`frontend/`. Python paths are relative to `src/liberational/`.

## Server

- **Routes.** Every route is registered in one table (`dashboard/routes.py`) with decorators such as
  `@routes.get("/pets")` and `@routes.post("/zones")`. Exact paths match first, then patterns in registration order,
  then catch-alls. `tests/data/dashboard_routes.txt` lists every route, so moving one cannot add or drop it.
- **One module per area.** An area's routes live in its own module (`activities.py`, `pets.py`, `zones.py`,
  `places.py`, `journal.py`, `presence.py`, `ingest.py`, `reports.py`, `research.py`, `settings.py`, and the People
  modules). `server.py` loads them and keeps the HTTP plumbing: authorization, compression, caching, the live feed,
  and static files.
- **State.** A module reads the state directory as `dashboard/context.py`'s `STATE`. Tests move it with the
  `move_dashboard_state` fixture.
- **Authorization.** A request that is not direct loopback needs the dashboard password. Writes that change the
  machine (services, settings, devices, and teaching) are accepted only from the machine itself, same origin.
- **Slow reads run elsewhere.** Face groups, voice groups, the People list, Reports' days, and Review's day history
  are computed in long-lived child processes at low priority. A request returns `pending` and the page polls. Crops
  and thumbnails are cut by a background worker; a GET for one that is not ready returns 202.
- **The live feed.** `/events` streams the transcript, scene, speakers, device status, and shell status. Pages fall
  back to polling when the stream stalls.

## Frontend

TypeScript, React, and [Mantine](https://mantine.dev/). Vite builds `frontend/` into `dashboard/static/app/`, which is
gitignored.

- `mise run frontend` builds it. `mise run dev` runs Vite's dev server beside the dashboard with hot reload.
- With no build, a page says how to make one. Hashed assets are cached for a year; pages are fetched fresh.
- `mise run lint` runs Biome and `tsc`. Tests are Vitest files beside the code (Testing Library, jsdom, and MSW).

| Path under `frontend/src/` | Contents |
|---|---|
| `dashboard/App.tsx` | The shell: top bar, command palette, status pill, and gear menu |
| `dashboard/live/`, `review/`, `journal/`, `known/`, `calls/`, `inbox/`, `reports/`, `ingest/` | One folder per tab |
| `dashboard/people/`, `activities/`, `places/`, `zones/`, `whosthis/` | Known's tabs, the zone editor, and the naming inbox |
| `dashboard/settings/` | The Settings drawer and its sections |
| `dashboard/camera/`, `dashboard/player/` | The camera mux, overlays, and the WebCodecs player |
| `dashboard/ask/` | The Ask panel |
| `dashboard/lib/` | Code several tabs share: the clip player, formatting, and preferences |
| `shared/` | Code the whole app shares: HTTP, telemetry, theme, and fonts |

- **State.** Server data goes through TanStack Query. The live feed writes stream events into the query cache. Small
  UI state is in Zustand stores.
- **Loading.** Live and the shell load first. Every other tab, Settings, and Ask load on first use.
- **Video.** A Web Worker decodes the gateway's frames with WebCodecs and draws them with the overlay of the same
  capture time. Each frame carries its own flip flag, so pictures and overlays are drawn upright.
- **Review** keeps time and media outside React in `review/engine.ts` and keeps its place in the URL hash, so other
  tabs link to a moment with `reviewURL()`.
- **Camera ids** are exact manifest names. Frontend dictionaries keyed by camera use Maps, so an id such as
  `constructor` is safe.
- **Telemetry.** `shared/telemetry.ts` posts page errors and slow loads to `POST /frontend/log`, which appends to
  `logs/frontend.log`. Paths only, local only.

## Tabs

| Tab | Shows |
|---|---|
| Live | Camera feeds with face, pose, and object overlays; the recent conversation; Ask |
| Review | A day's timeline, transcript, summary, and recorded video, with filters by person, thing, event, and camera |
| Journal | Month, week, day, and hour entries with their evidence |
| Known | People, Pets, Objects, Activities, Devices, and Places: everything the owner names and corrects |
| Calls | The call form, recordings, and transcripts |
| Inbox | Research reports |
| Reports | D3 charts over per-day aggregates cached in `state/reports.sqlite3` |
| Ingest | Imported media and its sources |
| Debug | Service logs, filtered by level |

"Who's this?" in People lists unnamed faces and voices with guesses. Choosing a name claims the group and returns an
undo token. Naming never happens by typing alone.

## Settings

The Settings drawer reads `GET /config` (`config.schema()`: every setting in `config/defaults.toml` with its
effective value and its comment as the description; secrets are never sent) and saves with `POST /config`.

- Saving writes only values that differ from the defaults into `pack.toml`, keeps the file's comments, restarts the
  services that read the changed settings, and re-executes the dashboard.
- With no `pack.toml`, the dashboard opens on the same form as first-run setup.
- **Cameras and microphones** edits the device manifest. Changing which devices are used restarts capture; changing
  one camera's settings restarts only that camera's worker. Sensitivity applies live.
- **Network cameras** discovers RTSP cameras through UniFi Protect or ONVIF ([NETWORK.md](NETWORK.md)).
- **Services** shows each service's state, last error, and history, and switches services on and off
  ([SERVICES.md](SERVICES.md)).
- **This machine** shows processor, memory, loaded models, and missing system packages.

A setting whose comment in `defaults.toml` has a `# New feature: <name>` line appears in a New features panel.

## Remote access

A dashboard opened over HTTPS cannot open the gateway's plain WebSocket on port 8766. The page uses `wss` on port
8767, so put a TLS proxy in front of the gateway. With Tailscale, for the tailnet only:

```sh
tailscale serve --bg --https=8767 http://127.0.0.1:8766
```

Then name the host the gateway should accept in `pack.toml` and restart the gateway:

```toml
[dashboard]
gateway_allow_host = "<machine>.<tailnet>.ts.net"
```

Keep that proxy on the tailnet. The gateway asks for no password, so a public Funnel would publish the cameras.
