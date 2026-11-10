set(scanner_drivers esp_driver_uart)
if(IDF_TARGET STREQUAL "esp32c3" OR IDF_TARGET STREQUAL "esp32s3")
    list(APPEND scanner_drivers esp_driver_usb_serial_jtag)
endif()
idf_component_register(SRCS "scanner.c" "protocol.c"
                       INCLUDE_DIRS "."
                       REQUIRES bt nvs_flash esp_timer esp_hw_support ${scanner_drivers})
