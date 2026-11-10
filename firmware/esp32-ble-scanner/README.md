# Receive-only BLE sensor

This ESP-IDF project turns an ESP32-C3, ESP32-S3, or ESP32 into a USB serial BLE advertisement receiver for
`pack-bluetooth` ([docs/BLUETOOTH.md](../../docs/BLUETOOTH.md)). It scans passively: it never advertises, connects,
pairs, reads GATT, or sends scan requests, and Wi-Fi is not initialized.

One source project supports all three chips, but the images are chip-specific. Flash the image for the chip printed
on the board.

## Build

The toolchain is **ESP-IDF v6.1**. Install it with
[Espressif's tools](https://docs.espressif.com/projects/esp-idf/en/v6.1/esp32/get-started/index.html), activate its
`export.sh`, and keep the SDK outside the repository. From the checkout:

```bash
firmware/esp32-ble-scanner/build.sh /tmp/ble-scanner-v1
```

The script checks the SDK's tag and builds each variant through the repository's shared build slot
(`tools/build-slot.sh`), so do not run `idf.py build` directly on a shared machine. It builds `esp32c3-native`,
`esp32c3-uart`, `esp32s3-native`, and `esp32s3-uart`; name variants after the output directory to build only those.
`esp32-uart` is optional.

The output directory holds `ble-scanner-v1-<variant>.bin` (a merged image for flashing at `0x0`), the resolved
`.sdkconfig` for each, and `SHA256SUMS`. Images do not belong in Git.

## Flash

Choose the image by chip and by which USB connector you use:

| Variant | Serial transport |
|---|---|
| `esp32c3-native`, `esp32s3-native` | Native USB Serial/JTAG, usually `303a:1001`, `/dev/ttyACM*` on Linux |
| `esp32c3-uart`, `esp32s3-uart`, `esp32-uart` | A USB-to-UART bridge (CP210x or CH340), `/dev/ttyUSB*`, 115200 baud, 8N1 |

Flashing needs no ESP-IDF. Check the image's checksum, then:

```bash
uv tool run --from esptool esptool --chip auto --port <serial-port> write-flash 0x0 ble-scanner-v1-esp32c3-native.bin
```

If the board does not enter its bootloader, hold BOOT while pressing RESET, start the command, then release BOOT.
The wrong console variant is harmless but prints nothing on the connector you chose: flash the matching one.

## BLE1 protocol

The firmware writes only checksummed ASCII lines, each ending in one newline. The host discards any line with an
unknown prefix, a bad field, or a bad checksum.

```text
BLE1|HELLO|<chip_id>|<boot_id>|<uptime_us>|<name_utf8_hex>|<crc32>
BLE1|ADV|<chip_id>|<boot_id>|<counter>|<uptime_us>|<address>|<address_type>|<rssi>|<channel>|<payload_hex>|<crc32>
```

| Field | Meaning |
|---|---|
| `crc32` | Eight uppercase hex digits: Python's `zlib.crc32` over every byte before the final `\|<crc32>` |
| `chip_id` | 12 uppercase hex digits from the factory MAC. It identifies the sensor, not an advertiser |
| `boot_id` | Eight uppercase hex digits, random on every boot. It separates clock and counter resets |
| `uptime_us` | Monotonic microseconds since boot, captured in the receive callback |
| `name_utf8_hex` | 1 to 32 UTF-8 bytes as lowercase hex. An unnamed sensor reports its chip ID |
| `counter` | Unsigned 32-bit, incremented for every received report before any drop. A gap means reports were not delivered |
| `address` | Uppercase colon-separated advertiser address |
| `address_type` | `0` public, `1` random, `2` public identity, `3` random identity |
| `rssi` | Signed dBm. `127` means unavailable |
| `channel` | Always `0` (unknown): NimBLE's legacy report API does not expose it |
| `payload_hex` | The exact legacy advertisement bytes, 0 to 31, as lowercase hex |

HELLO is sent at startup, every five seconds, and after a name change. Numeric fields are decimal unless noted.
Extended advertisements are not enabled.

## Naming a sensor

```bash
pack-bluetooth --set-name door --port <serial-port>
```

That sends the one input command the firmware accepts and waits for the HELLO that confirms it:

```text
NAME1|<name_utf8_hex>|<crc32>
```

A name is 1 to 32 valid UTF-8 bytes with no control characters. It is saved in the board's NVS and survives a
reset. Flashing a merged image can erase it, so name the board again afterward. Do not run two readers on one
port.

## Bounds

- The receive callback copies each report into a fixed queue of 64 and never waits on serial output.
- Output is rate-limited: 128 lines a second on native USB and 32 on a UART bridge (`menuconfig` can change both).
- A full queue, a reset, or an unplugged cable leaves a counter gap. A gap is lost evidence, never proof that a
  device left.

## Protocol tests

The formatter and command parser are plain C, independent of ESP-IDF:

```bash
nice -n 19 cc -std=c11 -Wall -Wextra -Werror -pedantic \
  -Ifirmware/esp32-ble-scanner/main \
  firmware/esp32-ble-scanner/main/protocol.c \
  firmware/esp32-ble-scanner/tests/test_protocol.c -o /tmp/ble1-protocol-test
nice -n 19 /tmp/ble1-protocol-test
```

They cover framing, checksums, bounds, and UTF-8 validation. They do not prove radio reception; that needs a board.
