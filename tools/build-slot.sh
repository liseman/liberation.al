#!/usr/bin/env bash
# Serialize Rust builds, tests, and checks across every worktree of this repository. Cargo, rustc, and
# future sccache startup keep the invoking shell's standard priority; do not wrap this in tools/niced.sh.
# CARGO_BUILD_JOBS defaults to 8 and remains overridable. Full gates hold this lock too, inside tools/gated.sh.
#
#   tools/build-slot.sh cargo test --no-fail-fast -p continuum-broker
#
# The command inherits the lock, so descendants keep it held until they exit. macOS uses Python's fcntl when
# flock(1) is absent. Both paths use the same lock in the common git directory shared by all worktrees.
set -euo pipefail
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# An existing server is left alone: never interrupt another worker's compilations to change its priority.
case "${RUSTC_WRAPPER:-}" in
  *sccache) "$RUSTC_WRAPPER" --start-server >/dev/null 2>&1 || true ;;
esac
common="$(git rev-parse --git-common-dir)"
lock="$(cd "$common" && pwd)/continuum-build.lock"
if command -v flock >/dev/null && [ -z "${GATE_NO_FLOCK:-}${BUILD_SLOT_NO_FLOCK:-}" ]; then
  flock -n "$lock" true || echo "waiting for another Rust build ($lock)" >&2
  exec flock "$lock" "$@"
fi
# -c preserves the wrapped command's stdin (unlike feeding this Python program through a heredoc).
exec "${BUILD_SLOT_PYTHON:-${GATE_PYTHON:-python3}}" -c '
import fcntl, os, sys
lock, cmd = sys.argv[1], sys.argv[2:]
fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print(f"waiting for another Rust build ({lock})", file=sys.stderr, flush=True)
    fcntl.flock(fd, fcntl.LOCK_EX)
os.set_inheritable(fd, True)
os.execvp(cmd[0], cmd)
' "$lock" "$@"
