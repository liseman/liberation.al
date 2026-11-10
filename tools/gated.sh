#!/usr/bin/env bash
# Runs a command holding the Continuum gate lock, so only one full `continuum-check` or `continuum-test`
# runs at a time on a machine: every worktree shares the repository's common git directory, and the lock is a file
# there. While another worktree holds it, this says so and waits.
#
#   tools/gated.sh COMMAND [ARG...]
#
# The command itself holds the lock (it's an open file the command inherits), so the gate stays shut for as long as
# the command, or anything it started, is running, and opens when they've all exited, however they end. One
# consequence: a process a gate leaves running in the background (a test that never reaps a child, say) keeps the
# gate shut until it exits; `fuser -v` on the lock file shows which. flock(1) does this where it exists (Linux);
# elsewhere (macOS ships no flock) Python takes the same kind of lock and execs the command with it. GATE_PYTHON names
# that Python (default python3).
set -euo pipefail
common="$(git rev-parse --git-common-dir)"
lock="$(cd "$common" && pwd)/continuum-gate.lock"
if command -v flock >/dev/null && [ -z "${GATE_NO_FLOCK:-}" ]; then
  flock -n "$lock" true || echo "waiting for another worktree's gate ($lock)" >&2
  exec flock "$lock" "$@"
fi
exec "${GATE_PYTHON:-python3}" - "$lock" "$@" <<'PY'
import fcntl, os, sys
lock, cmd = sys.argv[1], sys.argv[2:]
fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print(f"waiting for another worktree's gate ({lock})", file=sys.stderr, flush=True)
    fcntl.flock(fd, fcntl.LOCK_EX)
# Hand the locked file to the command, as flock(1) does: it holds the lock from here until it and its children exit.
os.set_inheritable(fd, True)
os.execvp(cmd[0], cmd)
PY
