# Network devices and cameras

Optional local-network features. All are off by default and read-only toward the network. Paths are relative to
`src/liberational/`. [BLUETOOTH.md](BLUETOOTH.md) covers the Bluetooth sensors.

## Network inventory

`pack-network-devices` (`proximity/network_worker.py`, `unifi.py`) is on when `network_devices.enabled` is true. It
polls a local UniFi controller once a minute for its client list. The controller address and read-only credentials
go in `pack.toml` under `[unifi]`.

- The session allows only fixed read-only routes over local HTTPS, with no redirects, DNS, or cloud. TLS verifies by
  default; `unifi.ca_file` supplies a private CA.
- It writes `network.device_observation` (one row per device: native MAC, hostname, IP, vendor, and access point)
  and `network.inventory` (one row per complete poll).
- A failed or partial poll never claims complete coverage. Passive ARP and DHCP rows have `connected = null` and
  never establish presence.
- `pack-network-devices --check` reads one inventory and prints counts without publishing.

The same worker records the Wi-Fi access points the machine hears (`proximity/wifi.py`, `wifi.access_point`), as a
signal of which place it is in. It reads only what the OS already scanned (`nmcli --rescan no`, or CoreWLAN) and
never connects, probes, or asks for a scan.

## Device links

`proximity/devices.py` (`pack-devices`) tallies Bluetooth and network sightings and joins them to people through
links the owner makes in Known, Devices (`device.link` in the knowledge store). Nothing links a device to a person
automatically.

- Stays are computed when asked and never stored.
- A device's absence is reported only with fresh, complete coverage from every source it is linked through.
  Otherwise presence is unknown.
- Each device's maker comes from the IEEE and Bluetooth SIG registries, fetched by `mise run models` and read
  offline.
- `state/cache/device_tallies.json` is a rebuildable cache.

## Network cameras

Settings, Network cameras finds RTSP cameras by explicit request, through UniFi Protect (`[unifi_protect]`, else
`[unifi]`) or ONVIF WS-Discovery. Discovery never opens media or enables a stream. ONVIF is partial: it does not
prove every camera was found.

- A found camera's RTSP URL stays in server memory behind a short-lived id; the page sees only scheme and host.
- Add writes a `kind = "rtsp"` device to `config/devices.toml` and restarts capture.
- Each configured camera has a Use switch that sets `[[device]].enabled`. A disabled camera gets no capture worker,
  and its history stays readable. The broker sizes capture rings at start, so a re-enabled camera needs a broker
  restart.
- Credentials belong in `[unifi]`, `[unifi_protect]`, or `[network_cameras]` in `pack.toml`.
