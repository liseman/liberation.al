# Bluetooth advertisement sensors

The optional `pack-bluetooth` service records nearby BLE advertisements. It is off by default. The preferred source
is one or more USB ESP32 receivers that scan passively: no observed device is connected, paired, probed, or sent a
scan request. The [firmware README](../firmware/esp32-ble-scanner/README.md) covers the boards, flashing, and the
wire format.

A sighting is evidence that a radio was heard. A name or a strong signal never proves a room, a distance, presence
at home, or who carried the device. No data means unknown, not absence.

## Configuration

Settings are under `[bluetooth]` in `config/defaults.toml`.

```toml
[bluetooth]
enabled = true
```

- **Ports.** With `serial_ports` empty, the service discovers Espressif native USB and common UART bridges. Not
  every device on those bridges is an ESP32, so list ports explicitly to avoid opening unrelated serial devices.
- **Sensors.** Up to `max_sensors` boards run at once, and an unplugged port is retried. A sensor's chip ID is its
  identity; its name is a label.
- **Naming a board.** `pack-bluetooth --set-name Desk --port <serial-port>` saves a name in the board's own
  storage. Run it while the collector is stopped. `sensor_names = ["AABBCCDDEEFF=Desk"]` overrides a name from
  config.
- **Bounds.** Reports are aggregated every `aggregate_seconds` (5) for at most `max_devices` (1024) identifiers and
  `max_records_per_second` (50). Past those bounds, new reports are dropped and counted.

## Streams

| Stream | Holds |
|---|---|
| `bluetooth.sighting` | One aggregate per sensor and native identifier: report count, RSSI range, advertised name, manufacturer and service data, raw payload, scan mode, and the sensor's clock evidence |
| `bluetooth.coverage` | What the collector lost: queue, counter, and format loss, and backlog. It never infers absence |

Both are segment streams. Raw identifiers are never merged: a rotating address is a new identifier.

## Clocks

Each frame carries the board's chip ID, boot ID, and uptime. The host stamps wall time on receipt and keeps, per
sensor and boot, the smallest `receipt time - uptime` offset. An observation's time uses that offset. USB buffering
makes it an approximate clock, not synchronized radio time. A gap in a board's counter means dropped reports, not
devices that left.

## Computer Bluetooth instead of an ESP32

The machine's own radio can be the source: BlueZ on Linux, CoreBluetooth on macOS. Both may send scan requests, so
they need explicit authorization, and neither is a fallback when ESP32 sensors disappear.

```toml
[bluetooth]
enabled = true
source = "bluez"          # "corebluetooth" on macOS
allow_active_scan = true
```

- Install the `host-bluetooth` Python extra. Linux also needs the BlueZ service and a powered adapter; macOS needs
  Bluetooth privacy permission.
- Neither backend connects, pairs, reads GATT, or changes adapter power.
- Sightings are marked `scan_mode = os_managed_active`. Their counts are OS reports, which can be cached or merged,
  not radio packets. CoreBluetooth gives app-scoped UUIDs, never MAC addresses.
- Strict passive evidence comes only from the ESP32 source.

## Replay

`source = "replay"` reads `replay_path`, a JSONL file of `received_at` and `line` (one wire frame), and never opens
a radio. Replayed records are marked as replay. Use made-up data in tests; real captures stay out of Git.

## Linking devices

Nothing links a device to a person automatically. The owner links one in Known, Devices, or with:

```sh
uv run pack-devices list
uv run pack-devices link <selector> <value> --person NAME    # or --thing NAME
uv run pack-devices links                                     # each link and its id
uv run pack-devices unlink <link id>
```

Selectors are `ble_identifier` (a static address), `ble_fingerprint` (an exact payload, for a rotating address),
`ble_name`, and `net_mac`. A link is an owner record (`device.link`); sightings are never changed.

Ask answers "is Dana home" and "when did Sam get here" from a linked person's sightings and says the evidence is
unconfirmed. [NETWORK.md](NETWORK.md#device-links) has the rules for presence and absence.
