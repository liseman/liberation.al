# Receive-only BLE sensor

This ESP-IDF project turns an ESP32-C3, ESP32-S3, or ESP32 into a USB serial BLE advertisement receiver. NimBLE is compiled with the observer role only. Scanning is passive, duplicate filtering is disabled, and its 50 ms scan window equals its interval. The application never advertises, connects, pairs, reads GATT, or transmits scan requests. Wi-Fi is not initialized.

One source project supports all three chips. The binaries are target-specific: ESP32 and ESP32-S3 use different Xtensa targets, and ESP32-C3 uses RISC-V. Flash the image for the chip printed on the board, not whichever board has a similar connector.

## Toolchain and build

The declared toolchain is **ESP-IDF v6.1**, the current stable release used for this firmware. Its NimBLE, FreeRTOS, and device drivers come from that exact SDK release; no extra components are downloaded by the project. Install it with [Espressif's official tools](https://docs.espressif.com/projects/esp-idf/en/v6.1/esp32/get-started/index.html), then activate its `export.sh`. Keep the SDK and its compiler cache outside the repository. The build script checks the SDK's exact Git tag and takes the repository's shared heavy-build lock (`tools/build-slot.sh`) separately for each variant. It runs at nice 19 and idle disk priority, where supported, with at most four build jobs. Firmware clears `RUSTC_WRAPPER` before taking the slot so it cannot start a Rust cache server at background priority. Do not run direct `idf.py build` or `cmake --build` commands alongside another shared-machine build; use this wrapper.

From the repository checkout:

```bash
firmware/esp32-ble-scanner/build.sh /tmp/ble-scanner-v1
```

The first hardware target is ESP32-C3, followed by ESP32-S3; those are the boards available for operator verification. The default builds four explicit variants: `esp32c3-native`, `esp32c3-uart`, `esp32s3-native`, and `esp32s3-uart`. Pass any of these after the output directory to build only selected variants. Plain ESP32 support is optional through `esp32-uart`. The output directory contains:

- `ble-scanner-v1-<variant>.bin`: bootloader, partition table, and application merged for flashing at **0x0**.
- `ble-scanner-v1-<variant>.sdkconfig`: the resolved configuration for the image.
- `SHA256SUMS`: SHA-256 checksums for the merged images.
- Separate `build-<variant>/` directories for inspecting logs and component images.

Images and generated build files do not belong in Git. Reproducible-build mode omits changing compilation timestamps. Keep the SDK version, target, and resolved configuration with a release.

## Ports and flashing

Choose the image by both chip and USB connection:

| Image variant | Application serial transport |
|---|---|
| `esp32c3-native` | Native USB Serial/JTAG, typically Espressif VID:PID `303a:1001`, `/dev/ttyACM*` on Linux |
| `esp32s3-native` | Native USB Serial/JTAG through the board's native USB connector, typically `303a:1001` / `/dev/ttyACM*` |
| `esp32c3-uart` | USB-to-UART bridge (often CP210x or CH340), typically `/dev/ttyUSB*`, 115200 baud, 8N1 |
| `esp32s3-uart` | The board's USB-to-UART bridge connector (often CP210x or CH340), typically `/dev/ttyUSB*`, 115200 baud, 8N1 |
| `esp32-uart` (optional) | ESP32-DevKitC UART0 through its USB-to-UART bridge, 115200 baud, 8N1 |

A board may expose either connector or both. Port names differ on macOS (`/dev/cu.*`); identify the chip and USB device, rather than choosing by filename alone. Native USB and bridge images run the same scanner and protocol. Flashing the wrong console variant is harmless to the board, but it may produce no application output on the chosen connector; reflash the matching variant. The default build produces both choices without waiting for board details.

No ESP-IDF toolchain is needed to flash a release image. With uv available, verify the release's checksum and run this one command, substituting the board's actual serial port:

```bash
uv tool run --from esptool esptool --chip auto --port <serial-port> write-flash 0x0 ble-scanner-v1-esp32c3-native.bin
```

The command detects the chip automatically. Choose the matching `esp32c3` or `esp32s3` image, and substitute `-uart.bin` for a bridge connection. An existing esptool installation can run the same command without the `uv tool run --from esptool` prefix. If automatic bootloader entry fails, hold BOOT while pressing RESET, start the flash command, then release BOOT. The application serial port may re-enumerate after flashing. Close any serial monitor before starting the Pack scanner. Native USB ignores the host's baud rate; the UART image needs 115200.

Application and bootloader logging are disabled, and the application writes only checksummed protocol lines. The chip's ROM may still print a startup banner, and a reset or unplug may cut a line. The host must discard lines with an unknown prefix, invalid fields, or a bad checksum and recover at the next newline. Flashing and real-board verification are operator steps, not part of automated tests.

## BLE1 protocol

ASCII records end in a single newline. All numeric fields except hexadecimal fields are decimal. CRC-32 is the IEEE polynomial used by Python's `zlib.crc32`, initialized and finalized as that function does. The checksum is eight uppercase hexadecimal digits, calculated over every ASCII byte before the final `|<checksum>` separator. The newline and that final separator are not covered.

```text
BLE1|HELLO|<chip_id>|<boot_id>|<uptime_us>|<name_utf8_hex>|<crc32>
BLE1|ADV|<chip_id>|<boot_id>|<counter>|<uptime_us>|<address>|<address_type>|<rssi>|<channel>|<payload_hex>|<crc32>
```

- `chip_id`: 12 uppercase hexadecimal digits from the chip's factory base MAC, with no colons. This identifies the sensor, not a nearby advertiser.
- `boot_id`: eight uppercase hexadecimal digits from a fresh random 32-bit value on every boot. It separates clock and counter resets; it is not an authentication token or a globally unique device identity.
- `uptime_us`: monotonic microseconds since sensor boot. For an advertisement, it is captured in the receive callback, before queueing. For HELLO, it is captured just before formatting the line.
- `name_utf8_hex`: one to 32 UTF-8 bytes, encoded as lowercase hex. An unnamed sensor reports its chip ID. HELLO is emitted at startup, every five seconds, and after a successfully persisted name change.
- `counter`: unsigned 32-bit counter incremented for every received legacy report, **before** queue and serialization drops. It wraps modulo 2^32. A gap means reports were not delivered to the host; it does not count undetected radio packets.
- `address`: uppercase colon-separated advertiser address. Address types are `0` public, `1` random, `2` public identity, and `3` random identity. The scanner does not resolve private addresses.
- `rssi`: signed dBm from this report, without averaging. HCI value `127` means unavailable, so the host must retain the sighting with unknown RSSI, not treat it as 127 dBm.
- `channel`: `0` means unknown. The protocol reserves `37`, `38`, and `39`, but NimBLE's legacy report API does not expose the channel, so this implementation always emits `0`.
- `payload_hex`: the exact legacy advertisement bytes, zero to 31 bytes, in lowercase hex. The host decodes AD structures. Extended advertisements are not enabled in this firmware.

The host stamps receipt time independently. It can estimate a per-sensor wall-clock offset from HELLO and uptime, but USB buffering and scheduling add uncertainty; this is not hardware clock synchronization. Keep both receive uptime and host receipt time when comparing multiple sensors.

## Naming a sensor

Names persist in the sensor's NVS and survive normal resets and an application-only reflash. The initial merged image includes erased padding over the NVS region, so reflashing that merged image can erase a saved name. Erasing flash or changing the NVS format can also remove it. Reapply the name after flashing a merged image; do not assume a name survived. The only accepted input command is a checksummed line:

```text
NAME1|<name_utf8_hex>|<crc32>
```

Use the same checksum convention as BLE1 output. Names must contain one to 32 valid UTF-8 bytes, with no control characters or embedded NUL. Invalid, oversized, partial, or bad-checksum commands are ignored. A successfully committed name appears in an immediate HELLO, which is the acknowledgment. The host must not automatically rename an attached board.

The host utility names a chosen sensor explicitly and waits for its HELLO acknowledgment:

```bash
pack-bluetooth --set-name door --port <serial-port>
```

Do not run two readers on the same port. This does not connect to any nearby Bluetooth device: it writes the USB serial command to the sensor. For protocol debugging, the following example prints the exact command for the generic name `door`; a serial terminal can send that line:

```bash
python - <<'PY'
import zlib
prefix = 'NAME1|' + 'door'.encode().hex()
print(f'{prefix}|{zlib.crc32(prefix.encode()):08X}')
PY
```

## Bounded operation and gaps

The receive callback copies at most 31 bytes into a statically allocated queue of 64 reports and never waits for serial output. A separate task emits advertisement lines at a fixed maximum rate, plus HELLO frames: 128 per second on native USB and 32 per second on a UART bridge. It wakes once per FreeRTOS tick (10 ms at the default 100 Hz) and sends every line due by then, so the tick does not lower the rate; idle time banks at most 20 ms of lines, so a quiet spell cannot release a burst. It never builds a device cache. The UART rate fits a 115200-baud link, including frame overhead; native USB ignores the baud rate and carries the higher rate easily. The USB write deadline is 20 ms, so an unplugged host cannot block scanning indefinitely. Full queues, malformed controller reports, resets, unplugging, and partial serial writes can all create counter gaps. A continuously open scan window does not guarantee every radio advertisement is received, and no absence or physical proximity claim follows from a gap.

`menuconfig` can reduce queue depth or change the output rate, sensor default name, and serial transport. Rates are capped at 64 per second on UART and 256 per second on native USB. Increasing the rate on UART may exceed the link's capacity; treat dropped or delayed reports as unavailable evidence.

A sensor that hears more reports than its output rate leaves counter gaps for the excess. The host reports these as over the sensor's output cap, separately from host-side loss. Because the host merges each device's reports into 5-second windows, the excess mostly thins out repeats and signal samples; a device that advertises rarely can still miss a window.

## Protocol checks

The formatting and command parser are plain C, independent of ESP-IDF. Run their synthetic host tests from the repository checkout:

```bash
nice -n 19 cc -std=c11 -Wall -Wextra -Werror -pedantic \
  -Ifirmware/esp32-ble-scanner/main \
  firmware/esp32-ble-scanner/main/protocol.c \
  firmware/esp32-ble-scanner/tests/test_protocol.c -o /tmp/ble1-protocol-test
nice -n 19 /tmp/ble1-protocol-test
```

Use `ionice -c3` with these commands on Linux. The tests cover the standard CRC vector, address order, integer boundaries, checksum framing, bounded output, bad commands, and UTF-8 validation. They do not prove hardware reception, RF scan behavior, board wiring, or persistence across power loss.

API references: [ESP-IDF BLE discovery](https://docs.espressif.com/projects/esp-idf/en/v6.1/esp32c3/api-guides/ble/get-started/ble-device-discovery.html), [NimBLE observer scan parameters](https://github.com/espressif/esp-idf/blob/v6.1/examples/bluetooth/nimble/blecent/main/main.c), and [merged image tooling](https://docs.espressif.com/projects/esp-idf/en/v6.1/esp32/api-guides/tools/idf-py.html).
