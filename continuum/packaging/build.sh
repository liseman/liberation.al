#!/usr/bin/env bash
# Builds Continuum's two wheels for the platform it runs on (docs/CONTINUUM-SDK.md):
#
#   continuum_py-<version>-cp314-cp314-<platform>.whl      the Python SDK, with the H.264 decoder
#   continuum_daemons-<version>-py3-none-<platform>.whl    the daemons, installed onto the environment's PATH
#
# On Linux it runs inside packaging/Dockerfile's image (packaging/wheels.sh starts it), whose glibc is the oldest the
# wheels support. On macOS it runs as is, and needs rustup, uv, pkg-config, protoc, cmake, meson, and ninja.
#
# It needs python3.14 (uv's), maturin, cargo, and what packaging/ffmpeg.sh needs (a C and C++ compiler, make, cmake,
# meson, ninja, nasm on x86-64, curl). Wheels go to continuum/dist/; everything else it makes stays under
# continuum/target/wheels/.
set -euo pipefail
cd "$(dirname "$0")/.."

out="${CONTINUUM_WHEELS_OUT:-$PWD/dist}"
work="${CONTINUUM_WHEELS_WORK:-$PWD/target/wheels}"
arch="$(uname -m)"
mkdir -p "$out" "$work"

# The daemons a packaged install runs or calls. continuum-capture starts continuum-capture-worker from beside itself.
DAEMONS=(continuum-broker continuum-capture continuum-capture-worker continuum-tier-worker continuum-vector-worker
  continuum-gateway continuum-export)

case "$(uname -s)" in
  Linux)
    # The oldest glibc the wheels run on: Debian 12's, which is also the build image's.
    glibc="${CONTINUUM_WHEELS_GLIBC:-2_36}"
    platform="manylinux_${glibc}_${arch}"
    compatibility=(--compatibility "manylinux_${glibc}")
    ;;
  Darwin)
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    platform="macosx_${MACOSX_DEPLOYMENT_TARGET//./_}_${arch}"
    # macOS ships bash 3.2, where expanding an empty array is an error under set -u: see its use below.
    compatibility=()
    ;;
  *)
    echo "no wheels for $(uname -s)" >&2
    exit 1
    ;;
esac

# The private FFmpeg, rebuilt only when its recipe changes. Cargo can't see that its libraries changed (the path
# pkg-config gives it is the same), so the crate that links them is cleaned, and everything that uses it relinks.
prefix="$work/ffmpeg-$arch"
recipe=$(cksum < packaging/ffmpeg.sh)
if [ "$(cat "$prefix/.recipe" 2>/dev/null)" != "$recipe" ]; then
  rm -rf "$prefix"
  packaging/ffmpeg.sh "$prefix" "$work/ffmpeg-src"
  cargo clean --release --target-dir "$work/target" -p ffmpeg-sys-next
  printf '%s\n' "$recipe" > "$prefix/.recipe"
fi

# Only the private build: a system FFmpeg on the build machine must never be what gets linked.
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
[ "$(pkg-config --variable=prefix libavcodec)" = "$prefix" ] || { echo "pkg-config found another FFmpeg" >&2; exit 1; }
export CARGO_TARGET_DIR="$work/target"
# A wheel carries no debugging symbols; the checkout's own release build (mise run continuum-release) keeps its own.
export CARGO_PROFILE_RELEASE_STRIP=symbols

python="$(uv python find 3.14)"
version=$("$python" -c 'import tomllib; print(tomllib.load(open("Cargo.toml", "rb"))["workspace"]["package"]["version"])')

rm -f "$out"/continuum_py-*-"$platform".whl "$out"/continuum_daemons-*-"$platform".whl
maturin build --release --features ffmpeg-static --interpreter "$python" ${compatibility[@]+"${compatibility[@]}"} --out "$out"
sdk=("$out"/continuum_py-*.whl)
"$python" packaging/pack_wheel.py licenses "${sdk[0]}" "$prefix/licenses"

cargo build --release --features continuum-tier/ffmpeg-static,continuum-capture/ffmpeg "${DAEMONS[@]/#/--bin=}"
"$python" packaging/pack_wheel.py pack --bin-dir "$CARGO_TARGET_DIR/release" --out "$out" --version "$version" \
  --platform "$platform" --licenses "$prefix/licenses" "${DAEMONS[@]}"
ls -l "$out"
