#include "protocol.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void checksum_matches(const char *line)
{
    const char *separator = strrchr(line, '|');
    assert(separator != NULL);
    char expected[11];
    snprintf(expected, sizeof(expected), "|%08X\n", ble1_crc32(line, (size_t)(separator - line)));
    assert(strcmp(separator, expected) == 0);
}

/* Lines sent over seconds of 10 ms wakes (a 100 Hz FreeRTOS tick) with the queue never empty. */
static long sent_over(uint32_t rate, int seconds)
{
    struct ble1_pace pace = ble1_pace_init(rate, 20000);
    long sent = 0;
    for (int64_t now = 0; now < (int64_t)seconds * 1000000; now += 10000) {
        while (ble1_pace_ready(&pace, now)) {
            ble1_pace_sent(&pace);
            ++sent;
        }
    }
    return sent;
}

static void pacing_holds_rate_at_tick_polling(void)
{
    /* One line per due wake held 32/s to 25/s at a 10 ms tick: each 31.25 ms interval waited for the 40 ms wake. */
    const uint32_t rates[] = {1, 25, 32, 64, 100, 128, 256};
    for (size_t i = 0; i < sizeof(rates) / sizeof(rates[0]); ++i) {
        long expected = (long)rates[i] * 10;
        long sent = sent_over(rates[i], 10);
        /* The last wake is at 9.99 s, so the run can end up to one tick's worth of lines short, never over. */
        assert(sent <= expected + 1 && sent >= expected - (long)rates[i] / 100 - 1);
    }
}

static void pacing_banks_at_most_one_burst(void)
{
    struct ble1_pace pace = ble1_pace_init(128, 20000);
    assert(ble1_pace_ready(&pace, 0));
    ble1_pace_sent(&pace);
    /* A second of empty queue, then a full one: only the 20 ms of banked credit goes out at once. */
    int burst = 0;
    while (ble1_pace_ready(&pace, 1000000)) {
        ble1_pace_sent(&pace);
        ++burst;
    }
    assert(burst == 3); /* 20 ms at 7813 us per line, plus the line due now. */
    assert(!ble1_pace_ready(&pace, 1000000));
}

int main(void)
{
    assert(ble1_crc32("123456789", 9) == 0xcbf43926u);
    assert(ble1_crc32("", 0) == 0);
    char out[BLE1_LINE_MAX];
    assert(ble1_hello(out, sizeof(out), "AABBCCDDEEFF", 0x1234abcd, 123456, "door") > 0);
    assert(strcmp(out, "BLE1|HELLO|AABBCCDDEEFF|1234ABCD|123456|646f6f72|60BF9CC2\n") == 0);
    checksum_matches(out);
    struct ble1_report r = {
        .counter = 4294967295u, .uptime_us = 18446744073709551615ull,
        .address = {0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa},
        .address_type = 1, .rssi = -60, .channel = 0,
        .payload_len = 3, .payload = {2, 1, 6},
    };
    assert(ble1_advertisement(out, sizeof(out), "AABBCCDDEEFF", 0x1234abcd, &r) > 0);
    assert(strstr(out, "|4294967295|18446744073709551615|AA:BB:CC:DD:EE:FF|1|-60|0|020106|") != NULL);
    checksum_matches(out);
    char small[8];
    assert(ble1_advertisement(small, sizeof(small), "AABBCCDDEEFF", 1, &r) == 0);
    assert(small[0] == '\0');
    r.payload_len = BLE1_PAYLOAD_MAX + 1;
    assert(ble1_advertisement(out, sizeof(out), "AABBCCDDEEFF", 1, &r) == 0);
    r.payload_len = 0;
    r.channel = 40;
    assert(ble1_advertisement(out, sizeof(out), "AABBCCDDEEFF", 1, &r) == 0);
    char name[BLE1_NAME_MAX + 1] = "unchanged";
    strcpy(out, "NAME1|646f6f72|B9E21551\n");
    assert(ble1_name_command(out, strlen(out), name) == 4);
    assert(strcmp(name, "door") == 0);
    out[6] = '0';
    assert(ble1_name_command(out, strlen(out), name) == 0);
    assert(strcmp(name, "door") == 0);
    const char invalid_utf8[] = "NAME1|c080";
    snprintf(out, sizeof(out), "%s|%08X\n", invalid_utf8, ble1_crc32(invalid_utf8, strlen(invalid_utf8)));
    assert(ble1_name_command(out, strlen(out), name) == 0);
    const char unicode_name[] = "NAME1|636166c3a9";
    snprintf(out, sizeof(out), "%s|%08X\r\n", unicode_name, ble1_crc32(unicode_name, strlen(unicode_name)));
    assert(ble1_name_command(out, strlen(out), name) == 5);
    assert(strcmp(name, "caf\xc3\xa9") == 0);
    pacing_holds_rate_at_tick_polling();
    pacing_banks_at_most_one_burst();
    puts("BLE1 protocol tests passed");
    return 0;
}
