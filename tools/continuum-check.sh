#!/usr/bin/env bash
# The Continuum workspace's checks (`mise run continuum-check`, which runs this under tools/gated.sh): rustfmt,
# clippy with warnings as errors (and with the ffmpeg feature where FFmpeg's headers are), ruff and ty on the Python
# bindings, and that the generated schema code and the C header are current. Rust steps keep standard priority; Python checks stay niced. The
# gate's CARGO_BUILD_JOBS default is set below.
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

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
if pkg-config --exists libavcodec; then
  cargo clippy -p continuum-tier --all-targets --features ffmpeg -- -D warnings
else
  echo "FFmpeg headers not found (pkg-config libavcodec): skipping the ffmpeg feature check"
fi
../tools/niced.sh .venv/bin/ruff check python packaging crates/continuum-py/tests
../tools/niced.sh .venv/bin/ruff format --check python packaging crates/continuum-py/tests
../tools/niced.sh .venv/bin/ty check --python .venv python packaging crates/continuum-py/tests

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if ! (cd crates/continuum-ffi && cbindgen --quiet --config cbindgen.toml --crate continuum-ffi --output "$tmp/continuum.h" .); then
  echo "cbindgen failed (above)" >&2
  exit 1
fi
if ! cmp -s "$tmp/continuum.h" crates/continuum-ffi/include/continuum.h; then
  echo "the C header is stale: run mise run continuum-ffi-header" >&2
  exit 1
fi
flatc --rust --gen-all -o "$tmp" schemas/continuum.fbs
flatc --rust --gen-all -o "$tmp" schemas/overlay.fbs
flatc --ts -o "$tmp/ts" schemas/overlay.fbs
if ! cmp -s "$tmp/continuum_generated.rs" crates/continuum-core/src/generated/continuum_generated.rs \
  || ! cmp -s "$tmp/overlay_generated.rs" crates/continuum-gateway/src/generated/overlay_generated.rs \
  || ! diff -rq "$tmp/ts" ../frontend/src/dashboard/player/generated >/dev/null; then
  echo "generated schema code is stale: run mise run continuum-codegen" >&2
  exit 1
fi
