#!/usr/bin/env bash
# Builds the private FFmpeg the wheels link statically (docs/CONTINUUM-SDK.md): an LGPL build with only what
# the Rust code calls, so it carries no GPL part and nothing a distribution's FFmpeg would pull in.
#
#   packaging/ffmpeg.sh <prefix> <work directory>
#
# What's in it, and why:
#   - the H.264 decoder and parser: network-camera vision, the tier worker, and export;
#   - OpenH264 (BSD) as the software H.264 encoder. libx264 is GPL, so an LGPL build can't have it;
#   - NVENC and NVDEC through nv-codec-headers (MIT) on Linux, which load the NVIDIA driver's libraries at run time;
#     VideoToolbox on macOS;
#   - the RTSP demuxer over TCP and UDP, and the MP4 muxer;
#   - libswscale.
# Left out on purpose: AV1 (the opt-in archive pass needs SVT-AV1 and dav1d), TLS (so no rtsps:// cameras), and
# libavdevice, libavfilter, and libswresample, which nothing calls.
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
    # OpenH264's pkg-config file names GNU's C++ library; macOS has only libc++.
    sed -i '' 's/-lstdc++/-lc++/g' "$prefix/lib/pkgconfig/openh264.pc"
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
echo "FFmpeg $FFMPEG, OpenH264 $OPENH264 in $prefix"
