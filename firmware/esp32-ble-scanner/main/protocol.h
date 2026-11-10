#ifndef BLE_SCANNER_PROTOCOL_H
#define BLE_SCANNER_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#define BLE1_NAME_MAX 32
#define BLE1_PAYLOAD_MAX 31
#define BLE1_LINE_MAX 256

struct ble1_report {
    uint32_t counter;
    uint64_t uptime_us;
    uint8_t address[6]; /* NimBLE's least-significant-byte-first address. */
    uint8_t address_type;
    int8_t rssi;
    uint8_t channel;
    uint8_t payload_len;
    uint8_t payload[BLE1_PAYLOAD_MAX];
};

uint32_t ble1_crc32(const void *data, size_t length);
size_t ble1_hello(char *out, size_t capacity, const char *chip_id,
                  uint32_t boot_id, uint64_t uptime_us, const char *name);
size_t ble1_advertisement(char *out, size_t capacity, const char *chip_id,
                          uint32_t boot_id, const struct ble1_report *report);
/* Returns the decoded UTF-8 byte count, or zero for an invalid NAME1 frame. */
size_t ble1_name_command(const char *line, size_t length, char name[BLE1_NAME_MAX + 1]);

/* Paces advertisement lines to a steady rate however coarse the caller's polling is. Idle time banks at most
 * burst_us of credit, so a long quiet spell cannot release a burst on the link. */
struct ble1_pace {
    int64_t next_us;
    int64_t interval_us;
    int64_t burst_us;
};

struct ble1_pace ble1_pace_init(uint32_t lines_per_second, int64_t burst_us);
/* Whether one more line may go out at now_us. Call ble1_pace_sent after each line that does. */
int ble1_pace_ready(struct ble1_pace *pace, int64_t now_us);
void ble1_pace_sent(struct ble1_pace *pace);

#endif
