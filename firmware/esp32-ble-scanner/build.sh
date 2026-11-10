#!/usr/bin/env bash
# Build flash-at-0x0 release images with an already installed ESP-IDF v6.1 toolchain.
# Usage: ./build.sh OUTPUT_DIRECTORY [esp32c3-native esp32c3-uart esp32s3-native esp32s3-uart]
set -euo pipefail
if [[ ${BLE_BUILD_NICED:-0} != 1 ]]; then
    export BLE_BUILD_NICED=1
    if command -v ionice >/dev/null; then
        exec nice -n 19 ionice -c3 bash "$0" "$@"
    fi
    exec nice -n 19 bash "$0" "$@"
fi
: "${IDF_PATH:?Activate ESP-IDF v6.1 with its export.sh before building}"
version=$(git -C "$IDF_PATH" describe --tags --exact-match)
[[ "$version" == v6.1 ]] || { echo "Expected ESP-IDF v6.1, found $version" >&2; exit 2; }
output=${1:?Usage: build.sh OUTPUT_DIRECTORY [esp32c3-native esp32c3-uart esp32s3-native esp32s3-uart]}
shift
mkdir -p "$output"
output=$(cd "$output" && pwd)
project=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$project/../.." && pwd)
if [[ $# == 0 ]]; then
    set -- esp32c3-native esp32c3-uart esp32s3-native esp32s3-uart
fi
if [[ ${BLE_FIRMWARE_SLOT_HELD:-0} == 1 && $# != 1 ]]; then
    echo "A firmware build slot runs exactly one variant" >&2
    exit 2
fi
for requested in "$@"; do
    case "$requested" in
        esp32c3|esp32s3) variant="$requested-native" ;;
        esp32) variant="esp32-uart" ;;
        esp32c3-native|esp32c3-uart|esp32s3-native|esp32s3-uart|esp32-uart) variant="$requested" ;;
        *) echo "Unsupported variant: $requested" >&2; exit 2 ;;
    esac
    if [[ ${BLE_FIRMWARE_SLOT_HELD:-0} != 1 ]]; then
        # One variant per shared heavy-build slot. Clear Rust's optional wrapper before taking the slot:
        # a niced firmware build must not start a Rust sccache server at background priority.
        (
            cd "$repo"
            env RUSTC_WRAPPER= BLE_FIRMWARE_SLOT_HELD=1 \
                "$repo/tools/build-slot.sh" bash "$project/build.sh" "$output" "$variant"
        )
        continue
    fi
    target=${variant%-*}
    transport=${variant##*-}
    work="$output/build-$variant"
    defaults="$project/sdkconfig.defaults"
    # A release rerun must use current checked-in defaults, not a previous menuconfig cache.
    cp "$project/sdkconfig.defaults" "$output/sdkconfig-$variant"
    if [[ "$transport" == uart ]]; then
        defaults="$defaults;$project/sdkconfig.defaults.uart"
        cat "$project/sdkconfig.defaults.uart" >> "$output/sdkconfig-$variant"
    fi
    idf.py -C "$project" -B "$work" -D "IDF_TARGET=$target" \
        -D "SDKCONFIG=$output/sdkconfig-$variant" \
        -D "SDKCONFIG_DEFAULTS=$defaults" reconfigure
    cmake --build "$work" --parallel 4
    idf.py -C "$project" -B "$work" merge-bin -o "$output/ble-scanner-v1-$variant.bin" -f raw
    cp "$output/sdkconfig-$variant" "$output/ble-scanner-v1-$variant.sdkconfig"
done
(
    cd "$output"
    shopt -s nullglob
    images=(ble-scanner-v1-*-native.bin ble-scanner-v1-*-uart.bin)
    ((${#images[@]})) || { echo "No explicit variant images were built" >&2; exit 2; }
    if command -v sha256sum >/dev/null; then
        sha256sum "${images[@]}" > SHA256SUMS
    else
        shasum -a 256 "${images[@]}" > SHA256SUMS
    fi
)
printf 'Merged images and SHA256SUMS: %s\n' "$output"
