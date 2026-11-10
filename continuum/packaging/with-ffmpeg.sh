#!/usr/bin/env bash
# Runs a command with the wheels' private FFmpeg and build directory set, after packaging/build.sh has built them:
#
#   packaging/wheels.sh packaging/with-ffmpeg.sh cargo test -p continuum-tier --features ffmpeg-static
set -euo pipefail
cd "$(dirname "$0")/.."
work="${CONTINUUM_WHEELS_WORK:-$PWD/target/wheels}"
prefix="$work/ffmpeg-$(uname -m)"
[ -f "$prefix/.recipe" ] || { echo "no private FFmpeg in $prefix: run packaging/build.sh first" >&2; exit 1; }
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export CARGO_TARGET_DIR="$work/target"
exec "$@"
