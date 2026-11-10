#!/usr/bin/env bash
# Builds Continuum's wheels for this machine's platform into continuum/dist/ (`mise run continuum-wheels`).
#
#   packaging/wheels.sh                   build both wheels
#   packaging/wheels.sh <command...>      run a command in the build environment instead, for example the tier
#                                         tests against the private FFmpeg:
#                                           packaging/wheels.sh packaging/with-ffmpeg.sh \
#                                             cargo test -p continuum-tier --features ffmpeg-static
#
# On Linux the build runs in packaging/Dockerfile's image (Docker is the one requirement), as the calling user, with
# the repository mounted at /work. On macOS it runs here. It compiles the whole workspace in release mode, so run it
# through tools/build-slot.sh on a shared machine, as the mise task does.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
[ "$#" -gt 0 ] || set -- packaging/build.sh

if [ "$(uname -s)" != Linux ]; then
  cd "$repo/continuum"
  exec "$@"
fi

image="continuum-wheels:debian12"
docker build -q -t "$image" "$here" >/dev/null
mkdir -p "$repo/continuum/target/wheels/user"
# CONTINUUM_WHEELS_DOCKER adds arguments. The tests that start a broker need its Tier 0 arena in /dev/shm, which a
# container gets 64 MB of: `--shm-size=12g` fits the default profile. A test binary built here also runs on the host,
# which is how to try NVENC and NVDEC against the private FFmpeg (a container needs GPU passthrough set up for that).
# shellcheck disable=SC2086
exec docker run --rm --user "$(id -u):$(id -g)" ${CONTINUUM_WHEELS_DOCKER:-} \
  -e HOME=/work/continuum/target/wheels/user \
  -e CARGO_HOME=/work/continuum/target/wheels/user/cargo \
  -e CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}" \
  -v "$repo:/work" -w /work/continuum "$image" "$@"
