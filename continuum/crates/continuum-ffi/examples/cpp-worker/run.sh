#!/usr/bin/env bash
# Builds the example worker and runs it, and c_check, against a real broker (for `mise run continuum-ffi-example`).
# Run from continuum/ after `cargo build -p continuum-ffi -p continuum-broker`.
set -euo pipefail
target="${CARGO_TARGET_DIR:-target}/debug"
case "$target" in /*) ;; *) target="$PWD/$target" ;; esac
build="$(mktemp -d)"
sockdir="$(mktemp -d)"
# Only the broker this script started: an unset pid must not become `kill 0` (the whole process group).
trap 'if [ -n "${broker:-}" ]; then kill "$broker" 2>/dev/null || true; fi; rm -rf "$build" "$sockdir"' EXIT
cmake -S crates/continuum-ffi/examples/cpp-worker -B "$build" -DCONTINUUM_LIB_DIR="$target" >/dev/null
cmake --build "$build" --parallel 2 >/dev/null
"$build/c_check"
"$target/continuum-broker" --socket "$sockdir/b.sock" 2>/dev/null &
broker=$!
for _ in $(seq 100); do [ -S "$sockdir/b.sock" ] && break; sleep 0.05; done
"$build/cpp_worker" "$sockdir/b.sock"
