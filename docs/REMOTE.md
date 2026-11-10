# How to use the dashboard from another device

Open the pack's dashboard in a browser on a phone or laptop, sign in with the dashboard password, and (when you enable it)
name people and pets from there. Over HTTPS on a tailnet, put the camera gateway behind TLS so live video works.
[DASHBOARD.md](DASHBOARD.md#server) explains authorization; [CONTINUUM-GATEWAY.md](CONTINUUM-GATEWAY.md#who-may-connect)
explains who may open a gateway session.

## Set a dashboard password

Every request that is not a direct loopback call to the dashboard must pass HTTP Basic auth. With no password set, the
dashboard asks for none, including from other machines.

On the Linux machine the pack runs on, either put the password in `<state>/dash_password` (one line, no newline needed)
or set `dashboard.password` in `pack.toml` (see [config/defaults.toml](../config/defaults.toml)). Restart the dashboard
service after you change it (`pack-services restart dashboard`, or toggle it in Services on the machine itself).

## Open the dashboard remotely

The dashboard listens on `ports.dashboard` (8765 by default). On the same LAN or tailnet, open
`http://<host>:8765` in a browser, where `<host>` is the machine's name or address. The browser prompts for the
dashboard password (the realm is `pack dashboard`).

To serve the dashboard over HTTPS on the tailnet only, run on the Linux machine:

```sh
tailscale serve --bg --https=8765 http://127.0.0.1:8765
```

Then open `https://<machine>.example-tailnet.ts.net:8765`. Teaching posts must come from that same origin as the page
(the dashboard accepts `https://` in the `Origin` header when Tailscale serves it).

## What a remote viewer can do

Without a password, do not expose the dashboard: loopback requests are trusted, but every other client is treated as a
remote viewer once a password exists.

| Area | On the machine the pack runs on | Remote viewer (password accepted) |
|---|---|---|
| Live, Journal, Known (read), Review, Calls, Inbox, Reports, Debug | Full use | Full use, except where a control flag disables writes |
| Review camera history (`GET /history/at`) | Always | Only if `dashboard.remote_camera_history` is on and a password is set |
| Naming, pets, zones, transcript fixes, activities, device links, Inbox skips | Full | Only if `dashboard.remote_teaching` is on, same origin, and a password is set |
| Settings, Services, devices, ingest, owner identity (`/people/me`), Learn surroundings | Full | Read only; the UI says only the machine Liberational runs on can change these |
| Camera exposure controls | Full | Read only (`Camera controls are read only.`) |

Owner identity changes and claiming the owner's voice stay on the machine itself even when remote teaching is on.

## Turn on remote teaching

Remote teaching is off by default. On the machine itself, open Settings, find `dashboard.remote_teaching`, save, and
let the dashboard restart. Or add to `pack.toml` on that machine:

```toml
[dashboard]
remote_teaching = true
```

Then restart the dashboard. Remote teaching never works without a dashboard password.

To let password-checked remote browsers open Review's recorded video and camera history, also set
`dashboard.remote_camera_history = true` the same way. It too requires a password.

## Put the camera gateway behind TLS for HTTPS

A page loaded over `https://` cannot open the gateway's plain WebSocket on `ports.gateway` (8766 by default). The player
uses `wss` on port 8767 when the page is HTTPS ([frontend/src/dashboard/player/protocol.ts](../frontend/src/dashboard/player/protocol.ts)).

On the Linux machine, for the tailnet only:

```sh
tailscale serve --bg --https=8767 http://127.0.0.1:8766
```

Add the tailnet name the browser will use to reach the gateway:

```toml
[dashboard]
gateway_allow_host = "machine.example-tailnet.ts.net"
```

Restart the `continuum-gateway` service. Keep the proxy on the tailnet. The gateway asks for no password, so a public
Funnel would publish the cameras.

## Related

- [DASHBOARD.md](DASHBOARD.md): the dashboard server and frontend.
- [NAMING.md](NAMING.md): naming people and pets (including from another device when remote teaching is on).
- [config/defaults.toml](../config/defaults.toml): `dashboard.password`, `remote_teaching`, `remote_camera_history`,
  and `gateway_allow_host`.
- [services/services.toml](../services/services.toml): `dashboard` and `continuum-gateway`.
