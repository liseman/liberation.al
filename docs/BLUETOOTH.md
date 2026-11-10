# Bluetooth advertisement sensors

The optional `pack-bluetooth` service prefers nearby BLE advertisements from one or more USB ESP32 receivers.
ESP32 acquisition is offline, and no observed device is connected, paired, probed, or sent a scan request. The firmware
uses strictly passive NimBLE scanning, accepts duplicates, and sets scan window equal to interval. It has no Wi-Fi
or BLE transmit workflow. The primary targets are ESP32-C3 and ESP32-S3. Each has native-USB and UART images;
see the [firmware README](../firmware/esp32-ble-scanner/README.md) for selection, flashing, naming, and BLE1 framing.
Optional computer discovery uses BlueZ on Linux or CoreBluetooth on macOS. Both can send scan requests and need
separate explicit authorization; neither connects, pairs, reads GATT, or changes adapter power.

## Configuration

All settings are documented in `config/defaults.toml` under `[bluetooth]`. The service is off by default. Set
`enabled = true` to select it during normal service installation. An explicit `[services] disable = ["bluetooth"]`
still wins. `source = "esp32"` is the default. Empty `serial_ports` discovers Espressif native USB and common
ESP32 UART bridges (Espressif, Silicon Labs, WCH, and FTDI vendor IDs). Not every device using these bridges is an
ESP32; specify a list of ports to avoid opening unrelated serial devices. The host never intentionally toggles
DTR/RTS to reset a sensor, but some USB drivers can glitch those lines when opening a port.

Up to `max_sensors` boards run concurrently, and unplugged ports are retried. Only a valid checksummed hello
admits a sensor. Unknown and unnamed advertisers are retained just like named ones. A sensor's chip ID is its
stable identifier; its saved board name or a `sensor_names = ["AABBCCDDEEFF=Desk"]` override is only a friendly
label. Renaming never merges sensors or advertisers. The explicit command
`pack-bluetooth --set-name Desk --port <serial-port>` saves a name in local board NVS and waits for a checksummed
hello confirming it. Run naming before acquisition, or stop the collector owning that serial port first. Acquisition
never sends this command automatically.

## Clock and coverage contract

BLE1 hello and advertisement frames carry chip ID, boot ID, and monotonic microsecond uptime. Advertisements
also carry the board's counter, original address/type, RSSI, raw advertising bytes, and channel when known. The
host stamps wall time on serial receipt. Each connection keeps a per-sensor, per-boot minimum
`receipt_time - device_uptime` offset. The mapped observation time uses that offset, while receipt time and offset
remain explicit evidence. USB buffering makes this an approximate host-aligned clock, not synchronized radio time.
Reconnect starts a new estimate. Boot IDs distinguish counter resets; unsigned counter wrap is handled. Repeated
or backward reports are counted and excluded. Missing counter positions indicate dropped reports, not devices
that left. A name or strong RSSI cannot prove a room, distance, home presence, or personal identity.

The firmware listens to legacy BLE advertisements on the primary advertising channels; extended/coded-PHY
advertising is not claimed. The firmware has a bounded queue and output rate. The host has a bounded serial line, read size, pending queue,
sensor count, identifier map, payload size, and fingerprint/member/evidence caches. Default aggregation is five
seconds, at most 1,024 pending identifiers, and 50 total output records per second across all three streams.
An aggregate can span longer while waiting its turn. Saturation drops new reports/identifiers explicitly; this is
best-effort observation with coverage loss, not a guarantee that every radio packet or device was received. A
shutdown gets at most five seconds of normal-rate draining, and logs remaining backlog. The sink has at most one in-flight acknowledged commit, with a five-second deadline. Candidates are emitted
only after their raw sightings are committed; failed or timed-out writes are logged as unacknowledged, not
assumed durable. No data means unknown, not absence. `bluetooth.coverage` carries source errors/loss counters, pending devices, and candidate cache bounds.

## Continuum streams, version 1

The service registers inline Text streams; no new broker wire schema or host state file is added.

| Stream | Layer | Body |
|---|---|---|
| `bluetooth.sighting` | Layer1Raw | Original UUIDv7 record ID and observation span; backend, sensor chip ID/name, identifier kind/value and address type; report count; RSSI sample count/min/max/mean, with missing values null; latest advertised name, manufacturer/service data hex, service UUIDs, TX power, and raw payload; scan mode; boot/counter/uptime/channel and receipt/offset ranges; explicit payload changes, truncation, and malformed flags |
| `bluetooth.device_candidate` | Layer3Interpretation | A separate UUIDv7 ID, `status=unconfirmed`, exact payload fingerprint and equality evidence, native members and spans, raw sighting IDs as parents, and a caveat that shared payloads are not identity |
| `bluetooth.coverage` | Layer2Decision | Source queue/counter/format loss, aggregation backlog/capacity loss, candidate cache counts, and an explicit refusal to infer absence |

Raw addresses stay distinct. BlueZ keeps `identifier_kind=bluetooth_address` plus its supplied public/random
address type; missing types stay unknown. CoreBluetooth uses `identifier_kind=corebluetooth_uuid`, an app-scoped
platform identifier, never a fabricated MAC. The shared decoder normalizes names, manufacturer data, service
data/UUIDs, and TX power. OS reports expose already-decoded fields and leave `raw_payload` unknown; they do not
claim to reconstruct an on-air packet. Malformed/truncated fields are explicit and prevent rotation
candidate eligibility. A matching name or service UUID alone is never enough. Candidates require repeated reports
with exact complete manufacturer/service bytes, at least eight payload bytes with four distinct byte values, and
distinct native identifiers on the same sensor/backend. Member counts and spans cover only the two retained
sighting aggregates named as causal parents, rather than lifetime totals; every contributing sighting is therefore
represented in the purge ancestry. These thresholds are conservative heuristics, not proof
of stable identity; factory payloads can still be shared. Acquisition creates no People/Things link or alert.

## Optional computer discovery

Provision the `host-bluetooth` Python extra alongside the machine's existing extras during normal dependency
maintenance. It declares `dbus-fast` on Linux and PyObjC CoreBluetooth/libdispatch on macOS; ESP32 needs none of
them. Linux also needs the local BlueZ system-bus service (`bluez` in the optional OS package manifest), a powered
LE adapter, and permission to acquire a discovery session. No root-only HCI socket, pairing, or adapter power change
is used. macOS needs Bluetooth privacy permission for the process that launches the worker; packaged launchers
must provide Apple's Bluetooth usage-description metadata. Missing dependencies, denied permission, powered-off
adapters, and lost discovery sessions fail explicitly. Native hardware/permission behavior still needs validation
on the target machine; fake tests do not establish radio behavior.

Opt in locally, choosing the backend for that machine:

```toml
[bluetooth]
enabled = true
source = "bluez" # "corebluetooth" on macOS; default/preferred remains "esp32"
allow_active_scan = true # authorizes OS discovery that may send scan requests
host_adapter = "" # BlueZ: auto-select a powered adapter, or hciN / its D-Bus path
host_queue_size = 512
```

On macOS `host_adapter` is only a stable observation label (empty means `system-default`), because CoreBluetooth
does not expose controller selection. Neither host backend is an automatic fallback when ESP32 sensors disappear.
Selecting a host backend without `allow_active_scan=true` fails before touching native Bluetooth APIs.

BlueZ acquires its own LE discovery session with duplicate reports and no service/pattern restriction, so unknown
advertisers are retained. Initial daemon-cached devices seed bounded identity metadata but are not fresh sightings.
Device removal/property invalidation clears stale fields; cache eviction and malformed reports remain visible.
Stopping releases only this process's discovery lease. CoreBluetooth scans all services with duplicate callbacks
on a dedicated serial dispatch queue, retains no peripheral objects, and ignores callbacks from stopped generations.
Both record `scan_mode=os_managed_active`, not strict passive scanning. Counts are OS reports, which can be cached
or coalesced, rather than radio-packet counts.

BlueZ merges discovery across applications. Although this client requests LE, Device1 notifications do not expose
the bearer of each report; another application's classic discovery can contribute properties. Coverage therefore
records `requested_transport=le` and `reported_transport=unknown_shared_discovery`. BlueZ sightings do not certify
LE-only packets or fresh reception of every cached field. Strict link-layer passive BLE evidence still comes from
the preferred ESP32 source. Both host backends timestamp callback receipt on the host; no native radio capture
clock or sensor clock offset is fabricated.

Host callback queues hold at most `host_queue_size` reports (1 through 4096); the existing `max_devices` bounds
BlueZ's device cache and the shared aggregation. Each normalized report has at most 64 entries per field group
and a shared 64 KiB payload budget, with truncation preserved through aggregation. Coverage includes backend,
adapter, and scan mode in `acquisition`, plus queue loss, backend cache/lifecycle counters, and aggregation backlog.
No host state file, port, device-identity merge, person/thing linking change, or alert is introduced.

## Replay and validation

`source = "replay"` requires `replay_path`, a local JSONL file. Each row has `received_at` (epoch seconds) and
`line` (one complete BLE1 ASCII frame). Replay never opens a radio or serial port. Lines, sensor count, and payloads
have the same bounds as acquisition. Replayed records are explicitly marked with backend and scan mode `replay`,
so they do not masquerade as live sensor observations. Use synthetic data for repository tests; real captures belong outside Git.

Host tests cover wire framing/checksums, C/Python golden vectors, AD decoding, malformed/oversized input, multiple
sensor clocks, boot/counter wrap and gaps, unplug/reconnect, bounded aggregation/rates, native identifiers,
unconfirmed candidate provenance, service gating, and Continuum header/inline encoding. Firmware protocol tests
run with sanitizers, and all release variants are cross-compiled. Devbox has no Bluetooth adapter, and no ESP32
was attached for these checks. A plugged-in board test remains required before claiming hardware validation.
Fake D-Bus and CoreBluetooth/libdispatch tests cover discovery lifecycle, native identifiers, unavailable/invalid
fields, duplicate delivery, bounded queues/caches, cancellation and cleanup. No host radio was scanned for #1342;
first Linux/macOS hardware and permission checks remain operator work. Wireless sensor transport is separate work.

## Linking devices to people and things (#1329)

`pack-devices list` shows every device seen (linked or not, with counts), and `pack-devices link <selector> <value>
--person NAME` or `--thing NAME` links one. A link is an owner assertion on `device.link`, revised in place and
retired with `pack-devices unlink`; sightings are never changed. Selectors: `ble_identifier` (a static address),
`ble_fingerprint` (the exact payload fingerprint the candidates use, for a device whose address rotates), `ble_name`
(the advertised name), and `net_mac` (a MAC from the router's client list, #1328).

The optional `pack-device-presence` service (on with `bluetooth.enabled`) turns each linked person's sightings into
stays: sightings no more than `devices.gap_s` apart. Each closed stay becomes an unconfirmed `presence.device_span`
candidate whose parents are its first and last sightings and the link. Ask answers "is Joe home", "when did Sam get
here", and "where did I last have my AirPods" from the sightings, joined with face visits for people. When no sensor
or complete router poll was running, it says it can't tell. A device seen is not proof of who carried it, and every
answer says so. Nothing alerts.

Settings → Devices → Phones, watches, and tags provides name/address search and Network/Bluetooth filtering before
the initial row limit. Choose a device, click Link, select A person, enter the configured person's name, and click
Link. This is an explicit owner assertion; hostnames never create person links automatically. Network devices use
their network MAC selector, while Bluetooth devices retain native identifiers or an explicitly chosen payload/name
selector. Assign both to the same person to combine their positive sightings into stays.

Live's Linked device evidence panel reads `GET /device-presence`, using the same `devices.home_status` policy as
Ask. It is separate from visually established Here now and physical arrival/departure events. A recent connected
network observation or linked Bluetooth sighting supports an unconfirmed positive. A negative needs fresh complete
authoritative coverage applicable to every linked source; unrelated, partial, inactive, unavailable, and passive
`absence_is_not_inferred` evidence stays unknown. A missing store is unavailable, never a successful empty roster.
"Am I home?" resolves the configured owner and still requires an explicit link. Archived reads preserve each
record's timestamps and half-open window rather than copying the archive chunk's span onto every observation.

The Live read window covers the freshness gap up to one hour. If a larger configured gap clips that window,
missing observations cannot certify a negative. Link changes invalidate both Settings and Live queries. Live
polls every fifteen seconds and marks retained data unknown after failed or stale updates. Read-only link queries
do not register streams; registration happens only on explicit writes.

For a network-only installation, enable finished-stay persistence explicitly from the main checkout after pulling:

```sh
.venv/bin/pack-services enable device-presence
.venv/bin/pack-services install device-presence
```

These are operator deployment commands, not test or worktree commands. Do not enable Bluetooth merely to select
this service. Ask and Live read current device evidence directly; the worker persists finished stays, so an open
stay is not yet a `presence.device_span`. A phone link and running network collector are still prerequisites.
