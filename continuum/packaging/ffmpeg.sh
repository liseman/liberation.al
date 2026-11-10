#!/usr/bin/env bash
# Builds the private FFmpeg the wheels link statically (docs/CONTINUUM.md, "Wheels"): an LGPL build with only what
# the Rust code calls, so it carries no GPL part and nothing a distribution's FFmpeg would pull in.
#
#   packaging/ffmpeg.sh <prefix> <work directory>
#
# What's in it, and why:
#   - the H.264 decoder and parser: network-camera vision, the tier worker, and export;
#   - OpenH264 (BSD) as the software H.264 encoder. libx264 is GPL, so an LGPL build can't have it;
#   - NVENC and NVDEC through nv-codec-headers (MIT) on Linux, which load the NVIDIA driver's libraries at run time;
#     VideoToolbox on macOS;
#   - SVT-AV1 (BSD) as the AV1 encoder and dav1d (BSD) as the AV1 decoder: the opt-in archive pass, and everything
#     that reads what it wrote;
#   - the RTSP demuxer over TCP and UDP, and the MP4 muxer;
#   - libswscale.
# Left out on purpose: TLS (so no rtsps:// cameras), and libavdevice, libavfilter, and libswresample, which nothing
# calls.
#
# Every source is pinned by version and SHA-256. Bump a version here, then build and run the tier and capture
# tests against it (packaging/wheels.sh says how).
set -euo pipefail

prefix="$1"
work="$2"
jobs="${CARGO_BUILD_JOBS:-8}"

FFMPEG=8.0.3
FFMPEG_SHA=6136812ea6d4e68bdba27e33c2a94382711cdf4f8602ffef056ff792bd6f9818
OPENH264=2.6.0
OPENH264_SHA=558544ad358283a7ab2930d69a9ceddf913f4a51ee9bf1bfb9e377322af81a69
# The oldest headers FFmpeg 8 accepts, so the oldest NVIDIA driver (530 and later) still encodes.
SVTAV1=4.3.0
SVTAV1_SHA=95ce8fc2381ba707e65ac8b9b2e165915bfbe9cd4d2f7b0f58d64ce31bd98f6d
DAV1D=1.5.4
DAV1D_SHA=686616b7c69eb88d44459391ab25cac13b6647a3b288835c5784e71c1514a5c5
NVCODEC=12.1.14.0
NVCODEC_SHA=62b30ab37e4e9be0d0c5b37b8fee4b094e38e570984d56e1135a6b6c2c164c9f

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# fetch <url> <sha256> <directory name>: downloads, checks, and unpacks into $work/<directory name>.
fetch() {
  local url="$1" sha="$2" name="$3" file="$work/${1##*/}"
  [ -f "$file" ] || curl -fsSL --retry 3 -o "$file" "$url"
  local got
  got=$(sha256 "$file")
  [ "$got" = "$sha" ] || { echo "$file: SHA-256 is $got, not $sha" >&2; exit 1; }
  rm -rf "${work:?}/$name"
  mkdir -p "$work/$name"
  tar -xf "$file" -C "$work/$name" --strip-components=1
}

mkdir -p "$prefix/licenses" "$work"
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

fetch "https://github.com/cisco/openh264/archive/refs/tags/v$OPENH264.tar.gz" "$OPENH264_SHA" openh264
make -C "$work/openh264" -j"$jobs" PREFIX="$prefix" install-static >/dev/null
cp "$work/openh264/LICENSE" "$prefix/licenses/OpenH264-LICENSE.txt"

# SVT-AV1: the encoder library only, with no link-time optimization, whose objects only the same compiler can link.
fetch "https://gitlab.com/AOMediaCodec/SVT-AV1/-/archive/v$SVTAV1/SVT-AV1-v$SVTAV1.tar.gz" "$SVTAV1_SHA" svt-av1
cmake -S "$work/svt-av1" -B "$work/svt-av1/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=OFF -DBUILD_APPS=OFF -DBUILD_TESTING=OFF -DSVT_AV1_LTO=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON >/dev/null
cmake --build "$work/svt-av1/build" -j"$jobs" >/dev/null
cmake --install "$work/svt-av1/build" >/dev/null
cp "$work/svt-av1/LICENSE.md" "$prefix/licenses/SVT-AV1-LICENSE.md"
cp "$work/svt-av1/PATENTS.md" "$prefix/licenses/SVT-AV1-PATENTS.md"

fetch "https://downloads.videolan.org/pub/videolan/dav1d/$DAV1D/dav1d-$DAV1D.tar.xz" "$DAV1D_SHA" dav1d
meson setup "$work/dav1d/build" "$work/dav1d" --prefix="$prefix" --libdir=lib --buildtype=release \
  --default-library=static -Db_lto=false -Denable_tools=false -Denable_tests=false >/dev/null
meson compile -C "$work/dav1d/build" -j "$jobs" >/dev/null
meson install -C "$work/dav1d/build" >/dev/null
cp "$work/dav1d/COPYING" "$prefix/licenses/dav1d-COPYING.txt"

hardware=()
case "$(uname -s)" in
  Linux)
    fetch "https://github.com/FFmpeg/nv-codec-headers/releases/download/n$NVCODEC/nv-codec-headers-$NVCODEC.tar.gz" \
      "$NVCODEC_SHA" nv-codec-headers
    make -C "$work/nv-codec-headers" PREFIX="$prefix" install >/dev/null
    # The headers carry their MIT license in each file; keep one as the notice.
    sed -n '1,/\*\//p' "$work/nv-codec-headers/include/ffnvcodec/nvEncodeAPI.h" > "$prefix/licenses/nv-codec-headers-LICENSE.txt"
    hardware=(--enable-ffnvcodec --enable-cuda --enable-cuvid --enable-nvdec --enable-nvenc
      --enable-encoder=h264_nvenc --enable-decoder=h264_cuvid --enable-hwaccel=h264_nvdec)
    ;;
  Darwin)
    hardware=(--enable-videotoolbox --enable-encoder=h264_videotoolbox)
    ;;
esac

fetch "https://ffmpeg.org/releases/ffmpeg-$FFMPEG.tar.xz" "$FFMPEG_SHA" ffmpeg
cd "$work/ffmpeg"
./configure --prefix="$prefix" --pkg-config-flags=--static \
  --enable-static --disable-shared --enable-pic --disable-debug --disable-doc --disable-programs \
  --disable-avdevice --disable-avfilter --disable-swresample \
  --disable-autodetect --disable-everything \
  --enable-libopenh264 --enable-encoder=libopenh264 \
  --enable-libsvtav1 --enable-encoder=libsvtav1 --enable-libdav1d --enable-decoder=libdav1d \
  --enable-decoder=h264 --enable-parser=h264 \
  --enable-bsf=h264_mp4toannexb,extract_extradata \
  --enable-demuxer=rtsp --enable-muxer=mp4 \
  --enable-protocol=file,tcp,udp,rtp \
  "${hardware[@]}" >/dev/null
grep -q '^#define CONFIG_GPL 0' config.h || { echo "the FFmpeg build is not LGPL" >&2; exit 1; }
make -j"$jobs" >/dev/null
make install >/dev/null
cp COPYING.LGPLv2.1 "$prefix/licenses/FFmpeg-COPYING.LGPLv2.1.txt"
sed -n 's/^#define FFMPEG_CONFIGURATION //p' config.h > "$prefix/licenses/FFmpeg-configuration.txt"
echo "FFmpeg $FFMPEG, OpenH264 $OPENH264, SVT-AV1 $SVTAV1, dav1d $DAV1D in $prefix"
