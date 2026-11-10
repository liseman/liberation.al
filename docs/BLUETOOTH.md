# How to set up Bluetooth advertisement sensors

The optional `pack-bluetooth` service (`src/liberational/proximity/worker.py`) records nearby BLE advertisements. It is
off by default. Its preferred source is one or more USB ESP32 receivers
([firmware/esp32-ble-scanner/README.md](../firmware/esp32-ble-scanner/README.md)).

## What a sighting is and what it does not prove

With USB ESP32 sensors, scanning is strictly passive: no observed device is connected, paired, probed, or sent a scan
request. A sighting proves only that a radio advertisement was heard. An advertised name or a strong RSSI never proves
which room a device was in, how far away it was, whether someone was home, or who carried it.

Raw identifiers are never merged, so a rotating BLE address appears as a new identifier each time it rotates. Each
ESP32 frame carries the board's chip ID, boot ID, and uptime; the host stamps wall time on receipt and tracks, per
sensor and boot, the smallest `receipt - uptime` offset. USB buffering makes observation times approximate rather than
synchronized radio time. A gap in a board's sequence counter means reports dropped at the board's output cap or across
a reset, not devices that left. Because BLE advertisements can be missed even when a radio is nearby,
`bluetooth.coverage` marks `absence_is_not_inferred = true`: Bluetooth sightings alone never establish absence, only
`"recent"` or `"unknown"`.

## Set up USB ESP32 sensors

1. Build and flash one or more ESP32 boards following
   [firmware/esp32-ble-scanner/README.md](../firmware/esp32-ble-scanner/README.md), then plug them into USB.
2. Optionally save a human-readable label (1 to 32 UTF-8 bytes) in a board's storage while `pack-bluetooth` is stopped:

   ```sh
   uv run pack-bluetooth --set-name Desk --port <serial-port>
   ```

   A sensor's chip ID is its identity and its name is a label; `bluetooth.sensor_names = ["AABBCCDDEEFF=Desk"]`
   overrides labels from configuration.
3. Enable Bluetooth sensing in `pack.toml`, and list `serial_ports` explicitly if unrelated serial devices share USB
   UART bridges on the machine:

   ```toml
   [bluetooth]
   enabled = true
   serial_ports = ["/dev/ttyACM0"]
   ```

   With `serial_ports` empty, the worker auto-discovers Espressif native USB ports and common UART bridges, up to
   `bluetooth.max_sensors` boards at once, retrying unplugged ports.
4. Start the `bluetooth` service: under the application, `uv run pack-supervisor reload` follows the changed
   selection; on an install with a unit per service, `uv run pack-services install bluetooth`.

## Computer Bluetooth instead of an ESP32

The machine's own adapter can act as the source (`bluez` on Linux, `corebluetooth` on macOS). Because OS discovery may
send active scan requests, it requires explicit opt-in and never acts as an automatic fallback when ESP32 sensors are
unplugged. Strict passive evidence comes only from the ESP32 source.

1. Install the optional dependency:

   ```sh
   uv sync --inexact --extra host-bluetooth
   ```

   On Linux, ensure the BlueZ daemon is running and the adapter is powered. On macOS, grant Bluetooth privacy
   permission.
2. Enable host scanning in `pack.toml` and start or restart the `bluetooth` service:

   ```toml
   [bluetooth]
   enabled = true
   source = "bluez"          # "corebluetooth" on macOS
   allow_active_scan = true
   ```

Neither backend connects, pairs, reads GATT attributes, or changes adapter power. Sightings are marked
`scan_mode = "os_managed_active"`: their counts reflect OS callbacks (which can be cached or merged) rather than raw
packets, and CoreBluetooth reports app-scoped UUIDs rather than MAC addresses.

## Linking devices

Nothing links a device to a person or thing automatically.

1. Open **Known**, then **Devices**, to inspect seen devices, link one to a person or thing, or record which place or
   room a device stays in.
2. Or manage links from the command line:

   ```sh
   uv run pack-devices list --hours 24
   uv run pack-devices link <selector> <value> --person Dana    # or --thing Keys
   uv run pack-devices links
   uv run pack-devices unlink <link-id>
   ```

A link writes an owner record (`device.link`) in the knowledge store without altering recorded sightings. Ask answers
questions such as "is Dana home" and "when did Sam get here" from a linked person's sightings and states that device
evidence is unconfirmed. [NETWORK.md](NETWORK.md#device-links) explains how Bluetooth and network sightings combine for
presence, absence, and device locations.

## Reference

### Streams

Both streams live in the segment store under slot `"ble"`. Reports past the configured per-window device and rate
caps are dropped and counted in `bluetooth.coverage`.

| Stream | Holds |
|---|---|
| `bluetooth.sighting` | One aggregate per window, sensor, and native identifier: report count, RSSI range, advertised name, manufacturer and service data, raw payload, scan mode, and sensor clock evidence |
| `bluetooth.coverage` | Per-window collector accounting: queue, sequence-counter, and format loss, backlog, and `absence_is_not_inferred = true` |

### Link selectors

| Selector | Matches |
|---|---|
| `ble_identifier` | A static BLE address or OS-scoped identifier |
| `ble_fingerprint` | An exact advertisement payload fingerprint, for devices that rotate addresses |
| `ble_name` | An exact advertised BLE local name |
| `net_mac` | A network MAC address ([NETWORK.md](NETWORK.md#device-links)) |

### Replay

Setting `bluetooth.source = "replay"` reads `bluetooth.replay_path` (JSON Lines of `received_at` and `line`, one wire
frame per line) without opening a radio, and marks emitted records as replay. Use synthetic captures in tests; keep
real captures out of Git.

### Settings and services

All bounds, ports, and adapter keys are documented under `[bluetooth]` and `[devices]` in
[config/defaults.toml](../config/defaults.toml). The `bluetooth` daemon (`enabled_by = "bluetooth.enabled"`) is
declared in [services/services.toml](../services/services.toml).
