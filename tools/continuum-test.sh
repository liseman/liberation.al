#!/usr/bin/env bash
# The Continuum workspace's tests (`mise run continuum-test`, which runs this under tools/gated.sh): the Rust tests
# (with the ffmpeg feature where FFmpeg's headers are), then the Python bindings' tests against a Rust frame producer
# and a real broker, then the C ABI's C and C++ examples against a real broker, then the event log, continuum
# status, registry, and forgotten-name tests. Rust steps keep standard priority; Python checks stay niced. The gate's CARGO_BUILD_JOBS default is set below.
set -euo pipefail
# The caller holds the single shared build lock (tools/build-slot.sh) for this entire gate. Rust commands run
# at standard priority with 8 build jobs by default; set CARGO_BUILD_JOBS to override.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# Every gate builds into one target directory shared by all worktrees, so dependencies compile once instead of
# once per worktree. The gate lock (tools/gated.sh) keeps two gates from building into it at once. Set CARGO_TARGET_DIR
# to override.
cd "$(dirname "$0")/../continuum"
common=$(git rev-parse --git-common-dir)
common=$(cd "$common" && pwd)
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$common/continuum-target}"
# Cargo names the workspace's own crates by their path relative to the workspace, and judges freshness by mtime, so
# after another worktree's gate it could take that worktree's newer build of a crate as current and test the wrong
# code. So when a different worktree built last, clean the workspace's own crates first; dependencies stay cached.
here=$(git rev-parse --show-toplevel)
stamp="$CARGO_TARGET_DIR/.last-worktree"
if [ "$(cat "$stamp" 2>/dev/null)" != "$here" ]; then
  mapfile -t crates < <(cargo metadata --no-deps --format-version 1 \
    | python3 -c 'import json, sys; print("\n".join(p["name"] for p in json.load(sys.stdin)["packages"]))')
  # An empty list would make cargo clean wipe the whole shared directory, so stop instead.
  [ "${#crates[@]}" -gt 0 ] || { echo "cargo metadata listed no workspace crates" >&2; exit 1; }
  cargo clean -q "${crates[@]/#/--package=}"
  mkdir -p "$CARGO_TARGET_DIR"
  printf '%s\n' "$here" > "$stamp"
fi

if pkg-config --exists libavcodec; then
  cargo test -p continuum-tier --features ffmpeg
fi
cargo test --workspace
cargo build -q -p continuum-shm --example frame_producer --example pcm_producer \
  -p continuum-broker --bin continuum-broker -p continuum-tier --bin continuum-tier-worker
# The Python tests find the broker and the producers in the shared target directory.
export CONTINUUM_BROKER="${CONTINUUM_BROKER:-$CARGO_TARGET_DIR/debug/continuum-broker}"
export CONTINUUM_FRAME_PRODUCER="${CONTINUUM_FRAME_PRODUCER:-$CARGO_TARGET_DIR/debug/examples/frame_producer}"
export CONTINUUM_PCM_PRODUCER="${CONTINUUM_PCM_PRODUCER:-$CARGO_TARGET_DIR/debug/examples/pcm_producer}"
VIRTUAL_ENV=.venv .venv/bin/maturin develop -q --uv
../tools/niced.sh .venv/bin/python -m pytest -q -p no:cacheprovider crates/continuum-py/tests
cargo build -q -p continuum-ffi
../tools/niced.sh crates/continuum-ffi/examples/cpp-worker/run.sh
# Event log, continuum status, registry, and forgotten-name store. The suite fixtures stay on:
# test_registry.py needs them, and the old --noconftest was for a v1 conftest that is gone.
PYTHONPATH=../src ../tools/niced.sh .venv/bin/python -m pytest -q -p no:cacheprovider \
  ../tests/test_eventlog.py ../tests/test_continuum_status.py \
  ../tests/test_registry.py ../tests/test_forgotten_store.py
