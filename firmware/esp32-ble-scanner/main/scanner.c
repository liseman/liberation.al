/* Receive-only observer: no central/peripheral role, advertisements, connections, or GATT operations. */
#include "protocol.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "sdkconfig.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"

#if CONFIG_BLE_SCANNER_USB_SERIAL_JTAG
#include "driver/usb_serial_jtag.h"
#else
#include "driver/uart.h"
#include "soc/uart_pins.h"
#endif

_Static_assert(sizeof(CONFIG_BLE_SCANNER_NAME) <= BLE1_NAME_MAX + 1, "Initial sensor name exceeds 32 UTF-8 bytes");

static StaticQueue_t report_queue_control;
static uint8_t report_queue_storage[CONFIG_BLE_SCANNER_QUEUE_SIZE * sizeof(struct ble1_report)];
static QueueHandle_t reports;
static uint32_t report_counter;
static uint32_t boot_id;
static char chip_id[13];
static char sensor_name[BLE1_NAME_MAX + 1];
static nvs_handle_t settings;

static bool serial_init(void)
{
#if CONFIG_BLE_SCANNER_USB_SERIAL_JTAG
    usb_serial_jtag_driver_config_t config = {.tx_buffer_size = 4096, .rx_buffer_size = 512};
    return usb_serial_jtag_driver_install(&config) == ESP_OK;
#else
    const uart_config_t config = {
        .baud_rate = CONFIG_BLE_SCANNER_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    return uart_param_config(UART_NUM_0, &config) == ESP_OK &&
           uart_set_pin(UART_NUM_0, U0TXD_GPIO_NUM, U0RXD_GPIO_NUM, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE) == ESP_OK &&
           uart_driver_install(UART_NUM_0, 512, 4096, 0, NULL, 0) == ESP_OK;
#endif
}

static void serial_write(const char *line, size_t length)
{
    if (length == 0) {
        return;
    }
#if CONFIG_BLE_SCANNER_USB_SERIAL_JTAG
    /* A missing USB reader must not stall the receive callback or grow an unbounded queue. */
    (void)usb_serial_jtag_write_bytes(line, length, pdMS_TO_TICKS(20));
#else
    (void)uart_write_bytes(UART_NUM_0, line, length);
#endif
}

static int serial_read(char *out, size_t capacity)
{
#if CONFIG_BLE_SCANNER_USB_SERIAL_JTAG
    return usb_serial_jtag_read_bytes(out, capacity, 0);
#else
    return uart_read_bytes(UART_NUM_0, out, capacity, 0);
#endif
}

static void hello(void)
{
    char line[BLE1_LINE_MAX];
    size_t length = ble1_hello(line, sizeof(line), chip_id, boot_id,
                               (uint64_t)esp_timer_get_time(), sensor_name);
    serial_write(line, length);
}

static void command_bytes(const char *bytes, size_t length)
{
    static char line[96];
    static size_t used;
    static bool discard;
    for (size_t i = 0; i < length; ++i) {
        if (bytes[i] == '\n') {
            if (!discard) {
                char updated[BLE1_NAME_MAX + 1];
                size_t count = ble1_name_command(line, used, updated);
                if (count && nvs_set_str(settings, "name", updated) == ESP_OK && nvs_commit(settings) == ESP_OK) {
                    memcpy(sensor_name, updated, count + 1);
                    hello(); /* The persisted name is acknowledged by the next checksummed HELLO. */
                }
            }
            used = 0;
            discard = false;
        } else if (!discard && used < sizeof(line)) {
            line[used++] = bytes[i];
        } else {
            discard = true;
        }
    }
}

static void serial_task(void *unused)
{
    (void)unused;
    char line[BLE1_LINE_MAX];
    char input[64];
    struct ble1_report report;
    int64_t next_hello = 0;
    /* The loop wakes once per FreeRTOS tick (10 ms at the default 100 Hz), so it sends every line that is due on
     * each wake rather than one, or the tick, not the configured rate, would set the output rate. */
    struct ble1_pace pace = ble1_pace_init(CONFIG_BLE_SCANNER_REPORTS_PER_SECOND, INT64_C(20000));
    while (true) {
        int64_t now = esp_timer_get_time();
        if (now >= next_hello) {
            hello();
            next_hello = now + INT64_C(5000000);
        }
        int count = serial_read(input, sizeof(input));
        if (count > 0) {
            command_bytes(input, (size_t)count);
        }
        while (ble1_pace_ready(&pace, now) && xQueueReceive(reports, &report, 0) == pdTRUE) {
            size_t length = ble1_advertisement(line, sizeof(line), chip_id, boot_id, &report);
            serial_write(line, length);
            ble1_pace_sent(&pace);
        }
        vTaskDelay(pdMS_TO_TICKS(5) ? pdMS_TO_TICKS(5) : 1);
    }
}

static void start_scan(void);

static int advertisement_event(struct ble_gap_event *event, void *unused)
{
    (void)unused;
    if (event->type == BLE_GAP_EVENT_DISC) {
        const struct ble_gap_disc_desc *found = &event->disc;
        struct ble1_report report = {
            .counter = ++report_counter,
            .uptime_us = (uint64_t)esp_timer_get_time(),
            .address_type = found->addr.type,
            .rssi = found->rssi,
            .channel = 0, /* NimBLE's legacy report does not expose the radio channel. */
            .payload_len = found->length_data,
        };
        if (found->length_data > BLE1_PAYLOAD_MAX || found->addr.type > 3) {
            return 0; /* Counter gap makes an unsupported report visible to the host. */
        }
        memcpy(report.address, found->addr.val, sizeof(report.address));
        memcpy(report.payload, found->data, found->length_data);
        /* No per-advertisement allocation, blocking UART write, device filter, or connection. */
        (void)xQueueSend(reports, &report, 0);
    } else if (event->type == BLE_GAP_EVENT_DISC_COMPLETE) {
        start_scan();
    }
    return 0;
}

static void start_scan(void)
{
    uint8_t own_address_type;
    if (ble_hs_util_ensure_addr(0) != 0 || ble_hs_id_infer_auto(0, &own_address_type) != 0) {
        return;
    }
    struct ble_gap_disc_params scan = {
        .itvl = 0x50, /* 50 ms, in 0.625 ms units. */
        .window = 0x50,
        .filter_policy = 0,
        .limited = 0,
        .passive = 1,
        .filter_duplicates = 0,
    };
    (void)ble_gap_disc(own_address_type, BLE_HS_FOREVER, &scan, advertisement_event, NULL);
}

static void host_task(void *unused)
{
    (void)unused;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

void app_main(void)
{
    /* Never put app logs, discovery dumps, or terminal escapes on the framed stream. */
    esp_log_level_set("*", ESP_LOG_NONE);
    esp_err_t status = nvs_flash_init();
    if (status == ESP_ERR_NVS_NO_FREE_PAGES || status == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        if (nvs_flash_erase() != ESP_OK) {
            return;
        }
        status = nvs_flash_init();
    }
    if (status != ESP_OK || nvs_open("ble-sensor", NVS_READWRITE, &settings) != ESP_OK) {
        return;
    }
    uint8_t mac[6];
    if (esp_efuse_mac_get_default(mac) != ESP_OK) {
        return;
    }
    snprintf(chip_id, sizeof(chip_id), "%02X%02X%02X%02X%02X%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    size_t name_size = sizeof(sensor_name);
    if (nvs_get_str(settings, "name", sensor_name, &name_size) != ESP_OK || name_size < 2) {
        snprintf(sensor_name, sizeof(sensor_name), "%s",
                 CONFIG_BLE_SCANNER_NAME[0] ? CONFIG_BLE_SCANNER_NAME : chip_id);
    }
    boot_id = esp_random();
    reports = xQueueCreateStatic(CONFIG_BLE_SCANNER_QUEUE_SIZE, sizeof(struct ble1_report),
                                 report_queue_storage, &report_queue_control);
    if (reports == NULL || !serial_init()) {
        return;
    }
    hello();
    if (xTaskCreate(serial_task, "ble-serial", 4096, NULL, 4, NULL) != pdPASS || nimble_port_init() != ESP_OK) {
        return;
    }
    ble_hs_cfg.sync_cb = start_scan;
    nimble_port_freertos_init(host_task);
}
