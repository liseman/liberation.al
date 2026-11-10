/* BLE1 is deliberately independent of ESP-IDF so framing can be tested on the host. */
#include "protocol.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

uint32_t ble1_crc32(const void *data, size_t length)
{
    const uint8_t *bytes = data;
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < length; ++i) {
        crc ^= bytes[i];
        for (unsigned bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (UINT32_C(0xedb88320) & (0u - (crc & 1u)));
        }
    }
    return crc ^ UINT32_MAX;
}

static size_t finish(char *out, size_t capacity, int prefix_length)
{
    if (prefix_length < 0 || (size_t)prefix_length + 11 > capacity) {
        if (capacity) {
            out[0] = '\0';
        }
        return 0;
    }
    uint32_t crc = ble1_crc32(out, (size_t)prefix_length);
    int suffix = snprintf(out + prefix_length, capacity - (size_t)prefix_length,
                          "|%08" PRIX32 "\n", crc);
    return suffix == 10 ? (size_t)prefix_length + 10 : 0;
}

static void hex_bytes(char *out, const uint8_t *data, size_t length)
{
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < length; ++i) {
        out[2 * i] = hex[data[i] >> 4];
        out[2 * i + 1] = hex[data[i] & 15];
    }
    out[2 * length] = '\0';
}

size_t ble1_hello(char *out, size_t capacity, const char *chip_id,
                  uint32_t boot_id, uint64_t uptime_us, const char *name)
{
    size_t length = strlen(name);
    if (length == 0 || length > BLE1_NAME_MAX || strlen(chip_id) != 12) {
        return 0;
    }
    char hex_name[2 * BLE1_NAME_MAX + 1];
    hex_bytes(hex_name, (const uint8_t *)name, length);
    int prefix = snprintf(out, capacity, "BLE1|HELLO|%s|%08" PRIX32 "|%" PRIu64 "|%s",
                          chip_id, boot_id, uptime_us, hex_name);
    return finish(out, capacity, prefix);
}

size_t ble1_advertisement(char *out, size_t capacity, const char *chip_id,
                          uint32_t boot_id, const struct ble1_report *report)
{
    if (strlen(chip_id) != 12 || report->payload_len > BLE1_PAYLOAD_MAX || report->address_type > 3 ||
        (report->channel != 0 && report->channel != 37 && report->channel != 38 && report->channel != 39)) {
        return 0;
    }
    char payload[2 * BLE1_PAYLOAD_MAX + 1];
    hex_bytes(payload, report->payload, report->payload_len);
    const uint8_t *a = report->address;
    int prefix = snprintf(out, capacity,
                          "BLE1|ADV|%s|%08" PRIX32 "|%" PRIu32 "|%" PRIu64
                          "|%02X:%02X:%02X:%02X:%02X:%02X|%u|%d|%u|%s",
                          chip_id, boot_id, report->counter, report->uptime_us,
                          a[5], a[4], a[3], a[2], a[1], a[0], report->address_type,
                          report->rssi, report->channel, payload);
    return finish(out, capacity, prefix);
}

static int hex_digit(char value)
{
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    return -1;
}

static bool valid_name(const uint8_t *name, size_t length)
{
    /* Reject controls, overlong encodings, surrogate code points, and values above Unicode's limit. */
    for (size_t i = 0; i < length;) {
        uint8_t first = name[i++];
        if (first < 0x20 || first == 0x7f) {
            return false;
        }
        if (first < 0x80) {
            continue;
        }
        unsigned extra;
        uint32_t value;
        uint32_t minimum;
        if (first >= 0xc2 && first <= 0xdf) {
            extra = 1; value = first & 0x1f; minimum = 0x80;
        } else if (first >= 0xe0 && first <= 0xef) {
            extra = 2; value = first & 0x0f; minimum = 0x800;
        } else if (first >= 0xf0 && first <= 0xf4) {
            extra = 3; value = first & 0x07; minimum = 0x10000;
        } else {
            return false;
        }
        if (i + extra > length) {
            return false;
        }
        for (unsigned j = 0; j < extra; ++j) {
            uint8_t next = name[i++];
            if ((next & 0xc0) != 0x80) {
                return false;
            }
            value = (value << 6) | (next & 0x3f);
        }
        if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff) ||
            (value >= 0x80 && value <= 0x9f)) {
            return false;
        }
    }
    return true;
}

size_t ble1_name_command(const char *line, size_t length, char name[BLE1_NAME_MAX + 1])
{
    if (length && line[length - 1] == '\n') {
        --length;
    }
    if (length && line[length - 1] == '\r') {
        --length;
    }
    const size_t prefix = 6; /* NAME1| */
    if (length < prefix + 2 + 9 || length > prefix + 2 * BLE1_NAME_MAX + 9 ||
        memcmp(line, "NAME1|", prefix) != 0 || line[length - 9] != '|') {
        return 0;
    }
    size_t hex_length = length - prefix - 9;
    if (hex_length % 2) {
        return 0;
    }
    uint32_t expected = 0;
    for (size_t i = length - 8; i < length; ++i) {
        int value = hex_digit(line[i]);
        if (value < 0) {
            return 0;
        }
        expected = (expected << 4) | (uint32_t)value;
    }
    if (expected != ble1_crc32(line, length - 9)) {
        return 0;
    }
    char decoded[BLE1_NAME_MAX + 1];
    for (size_t i = 0; i < hex_length / 2; ++i) {
        int high = hex_digit(line[prefix + 2 * i]);
        int low = hex_digit(line[prefix + 2 * i + 1]);
        if (high < 0 || low < 0) {
            return 0;
        }
        decoded[i] = (char)((high << 4) | low);
    }
    size_t decoded_length = hex_length / 2;
    if (!valid_name((const uint8_t *)decoded, decoded_length)) {
        return 0;
    }
    decoded[decoded_length] = '\0';
    memcpy(name, decoded, decoded_length + 1);
    return decoded_length;
}

struct ble1_pace ble1_pace_init(uint32_t lines_per_second, int64_t burst_us)
{
    struct ble1_pace pace = {
        .next_us = 0,
        .interval_us = lines_per_second ? (1000000 + (int64_t)lines_per_second - 1) / (int64_t)lines_per_second
                                    : 1000000, /* Rounded up: never above the configured rate. */
        .burst_us = burst_us > 0 ? burst_us : 0,
    };
    return pace;
}

int ble1_pace_ready(struct ble1_pace *pace, int64_t now_us)
{
    if (now_us - pace->next_us > pace->burst_us) {
        pace->next_us = now_us - pace->burst_us;
    }
    return pace->next_us <= now_us;
}

void ble1_pace_sent(struct ble1_pace *pace)
{
    pace->next_us += pace->interval_us;
}
