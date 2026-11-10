#!/usr/bin/env bash
# Installs the wheels in continuum/dist/ into a clean container of one distribution release and checks that the SDK
# imports and every daemon starts, with no Rust toolchain or FFmpeg there:
#
#   packaging/smoke.sh ubuntu:24.04
#
# The container gets what a packaged install gets from the system: PipeWire's client library, and uv for Python 3.14.
set -euo pipefail
image="$1"
dist="$(cd "$(dirname "$0")/../dist" && pwd)"

docker run --rm -i -v "$dist:/dist:ro" "$image" sh -eu <<'EOF'
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ca-certificates curl libpipewire-0.3-0 >/dev/null
for tool in rustc cargo ffmpeg; do
  if command -v "$tool" >/dev/null; then echo "$tool is installed: not a clean machine" >&2; exit 1; fi
done
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="/venv/bin:$HOME/.local/bin:$PATH"
uv venv -q --python 3.14 /venv
uv pip install -q --python /venv/bin/python /dist/*_"$(uname -m)".whl
. /etc/os-release
echo "$PRETTY_NAME, glibc $(ldd --version | sed -n '1s/.* //p')"
python -c 'import continuum_py; assert continuum_py.h264_decoder_available(); print("continuum_py imports, with its H.264 decoder")'
continuum-broker --version
continuum-gateway --version
continuum-export --version
continuum-vector-worker --version
continuum-capture --capabilities | tee /tmp/capture
grep -q '"rtsps":true' /tmp/capture || { echo "the capture supervisor has no TLS for rtsps:// cameras" >&2; exit 1; }
continuum-tier-worker --capabilities | tee /tmp/tier
grep -q '"av1_archive":true' /tmp/tier || { echo "the tier worker has no AV1 encoder" >&2; exit 1; }
# The capture worker has no query: without its arguments it says how it's called and exits 2 or more.
continuum-capture-worker 2>&1 | grep -q '^usage: continuum-capture-worker' && echo "continuum-capture-worker starts"
EOF
