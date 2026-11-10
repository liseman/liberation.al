# How to set up network inventory, device links, and cameras

Every local-network feature is off by default and read-only toward the network. [BLUETOOTH.md](BLUETOOTH.md) covers the
Bluetooth advertisement sensors.

## What network observations prove and do not prove

A router client entry (`connected = true`, from a local UniFi controller) shows only that a network interface was
associated with the network at poll time; it never proves who held the device or which room a person was in. When no
UniFi controller is configured (or when a controller poll fails with no devices), `pack-network-devices` falls back to
this machine's passive OS ARP cache and any configured local DHCP lease files (`source = "host_network"`,
`connected = null`). Cached ARP and DHCP rows can linger after a device leaves and never establish presence or absence.

On each pass, the same worker also reads Wi-Fi access points the OS has already cached (`nmcli --rescan no` on Linux,
CoreWLAN cached results on macOS) as a signal of which place this machine is in. It never connects, probes, or requests
an active Wi-Fi scan.

Absence (`"not_seen"`) is reported only when every source a person or thing is linked through has fresh, complete,
authoritative coverage within `devices.gap_s` showing none of its linked devices connected. A failed or partial router
poll, a host ARP/DHCP pass, or any Bluetooth link (`absence_is_not_inferred = true`) reports non-sighting as
`"unknown"`, never `"not_seen"`.

## Set up network inventory

1. To poll a local UniFi Network controller, configure `[unifi]` in `pack.toml` with a literal private, loopback, or
   link-local HTTPS IP address (`mode = "legacy"` with read-only `username` and `password`, or `mode = "integration"`
   with `api_key`, plus `ca_file` when using a private CA). Hostnames, cloud URLs, and HTTP redirects are refused. Leave
   `unifi.url` unset to read only this machine's passive ARP cache and any local `network_devices.dhcp_lease_paths`.
2. Set `network_devices.enabled = true` in `pack.toml`.
3. Test one read without writing to the segment store:

   ```sh
   uv run pack-network-devices --check
   ```

4. Install or restart the `network-devices` service (`uv run pack-services install network-devices` when running OS
   service units, or restart the supervisor).

## Device links

`src/liberational/proximity/devices.py` (`pack-devices`) tallies Bluetooth and network sightings and joins them to
people or things only through links the owner creates. Nothing links a device automatically.

1. Open **Known**, then **Devices** (or run `uv run pack-devices`, [BLUETOOTH.md](BLUETOOTH.md#linking-devices)), and
   link a device (`net_mac` or a Bluetooth selector) to a person or thing.
2. Optionally name your router's access points in **Known**, **Devices** (or under `devices.access_points` in
   `pack.toml`) so Ask can report which access point a device used, and optionally record which place or room a
   stationary device stays in (`device.location`) for "where is my <thing>" questions.

Sightings within `devices.gap_s` merge into stays on demand and are never stored. Device maker names come from the
offline IEEE and Bluetooth SIG registries fetched by `mise run models`.

## Network cameras

Release packages include RTSP H.264 capture support. In a source checkout, build it first with
`mise run continuum-py-ffmpeg` and `mise run continuum-release` ([CONTINUUM-SDK.md](CONTINUUM-SDK.md)).

1. Put any controller or camera credentials under `[unifi_protect]`, `[unifi]`, or `[network_cameras]` in `pack.toml`.
2. Open **Settings**, **Devices**, and go to the **Network cameras** card.
3. Press **Scan** to run explicit discovery (`src/liberational/dashboard/network_cameras.py`). Discovery queries UniFi
   Protect (`[unifi_protect]`, falling back to `[unifi]`) and sends ONVIF Media1 WS-Discovery multicast probes plus
   unicast UDP probes to `network_cameras.onvif_hosts` and, when `[unifi]` is set, up to 256 private client IPs from
   the UniFi controller. Discovery never opens media or enables a stream, and ONVIF probing is partial: missing a camera
   does not prove none is on the network.
4. Press **Add** on a discovered camera (or enter an RTSP URL directly). A discovered RTSP URL stays in server memory
   behind a 5-minute proposal ID so credentials never reach the page; **Add** writes a `kind = "rtsp"` device to
   `config/devices.toml` (mode `0600`) and restarts `continuum-capture`.

Each configured camera has a **Use** switch (`[[device]].enabled`). Turning it off stops that camera's capture worker
while keeping its recorded history readable. Because `continuum-broker` sizes capture rings at startup, turning a
disabled camera back on requires a broker restart.

## Reference

### Streams and state files

| Item | Store | Holds |
|---|---|---|
| `network.device_observation` | Segment store (`unifi_network` or `host_network`) | One row per observed device: native MAC, IP, hostname, vendor, `connected`, and access point ID and name |
| `network.inventory` | Segment store (`unifi_network` or `host_network`) | One row per poll: `complete`, `authoritative`, `device_count`, and warnings (written after device rows so an interrupted poll never looks complete) |
| `wifi.access_point` | Segment store (`wifi`) | Cached OS Wi-Fi entries (`bssid`, `ssid`, `rssi_dbm`, `channel`) |
| `device.link`, `device.location` | `state/knowledge.sqlite3` | Owner-taught links from selectors to people or things, and places or rooms where devices live |
| `state/cache/device_tallies.json` | State cache | Rebuildable device sighting tallies |
| `config/devices.toml` | Config | Configured capture devices, including `kind = "rtsp"` cameras |

### Settings and services

All keys are documented under `[network_devices]`, `[unifi]`, `[unifi_protect]`, `[network_cameras]`, and `[devices]` in
[config/defaults.toml](../config/defaults.toml). The `network-devices` daemon (`enabled_by = "network_devices.enabled"`)
is declared in [services/services.toml](../services/services.toml).
