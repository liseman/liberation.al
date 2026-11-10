# How to use the dashboard from another device

This guide shows you how to open the dashboard from a phone or another computer, protect it with a password, allow
naming from there, and get live video over HTTPS on a tailnet. [DASHBOARD.md](DASHBOARD.md#server) explains how the
server decides what a request may do.

## Set a dashboard password

With no password set, the dashboard asks for none, from any machine. Set one before another device can reach it.

1. On the machine the pack runs on, set `dashboard.password` in `pack.toml`, or leave it empty and put the password
   in `dash_password` in the state directory (`data/state/` in a checkout).

   ```toml
   [dashboard]
   password = "<a password>"
   ```

2. Restart the dashboard:

   ```sh
   uv run pack-services restart dashboard
   ```

A browser on the machine itself is never asked. Every other browser is asked for the password; any user name
will do.

## Open the dashboard from another device

Open `http://<host>:8765/` in a browser, where `<host>` is the machine's name or address on your network or tailnet
and 8765 is `ports.dashboard`.

## Know what another device may do

| | On the machine itself | From another device, with the password |
|---|---|---|
| Look at Live, Review, the Journal, Known, and Reports | Yes | Yes |
| Review's "what the cameras saw" (captions and face names) | Yes | Only with `dashboard.remote_camera_history` on |
| Teach: names, faces, voices, pets, zones, transcript fixes, and activity answers | Yes | Only with `dashboard.remote_teaching` on |
| Settings, services, devices, ingest, and the Learn surroundings switch | Yes | No: the page shows them read only |
| Say who the owner is | Yes | No |

## Allow teaching from another device

1. On the machine itself, turn on `dashboard.remote_teaching` in Settings, or in `pack.toml`:

   ```toml
   [dashboard]
   remote_teaching = true
   ```

2. Teach from the dashboard's own page on the other device. A request from any other page is refused.

It never works without a dashboard password. `dashboard.remote_camera_history` is turned on the same way.

## Get video over HTTPS on a tailnet

A dashboard opened over HTTPS cannot open the camera gateway's plain WebSocket on `ports.gateway` (8766). The page
then uses `wss` on port 8767, so put a TLS proxy in front of the gateway there.

1. On the machine the pack runs on, serve the gateway to the tailnet only:

   ```sh
   tailscale serve --bg --https=8767 http://127.0.0.1:8766
   ```

2. Name the host the gateway should accept, in `pack.toml`:

   ```toml
   [dashboard]
   gateway_allow_host = "machine.example-tailnet.ts.net"
   ```

3. Restart the gateway:

   ```sh
   uv run pack-services restart continuum-gateway
   ```

Keep that proxy on the tailnet. The gateway asks for no password, so a public Funnel would publish the cameras.

## Related

- [DASHBOARD.md](DASHBOARD.md#server): how the server authorizes a request.
- [CONTINUUM-GATEWAY.md](CONTINUUM-GATEWAY.md#who-may-connect): which hosts and origins the gateway accepts.
- [NAMING.md](NAMING.md): naming people and pets.
- [config/defaults.toml](../config/defaults.toml), `[dashboard]`: the settings used here.
