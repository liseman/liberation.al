#!/usr/bin/env bash
# Run a non-Rust test, check, or build in one of TEST_SLOTS (default 2) slots shared by every worktree and agent on
# this machine, at the lowest CPU and disk priority (AGENTS.md, "Share the machine"). Stacked worker test runs pushed
# devbox's load average past 50; this caps them like tools/build-slot.sh caps Cargo (which stays at standard priority
# and doesn't use this).
#
#   tools/test-slot.sh .venv/bin/python -m pytest -q tests/test_objects.py
#   tools/test-slot.sh npx vitest run src/objects
#
# A run takes the first free slot, else waits for one; the command and its children hold it until they exit. The
# slots are lock files in the git directory all worktrees share. The command's arguments, stdin, and exit status pass
# through unchanged.
set -euo pipefail
common="$(git rev-parse --git-common-dir)"
dir="$(cd "$common" && pwd)"
# -c preserves the wrapped command's stdin (unlike feeding this program through a heredoc).
exec "${TEST_SLOT_PYTHON:-python3}" -c '
import fcntl, os, shutil, sys, time
base, slots, cmd = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
fds = [os.open(f"{base}/test-slot-{i}.lock", os.O_CREAT | os.O_RDWR, 0o644) for i in range(max(1, slots))]
held, said = None, False
while held is None:
    for fd in fds:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = fd
            break
        except BlockingIOError:
            pass
    if held is None:
        if not said:
            print(f"waiting for a test slot ({len(fds)} in use)", file=sys.stderr, flush=True)
            said = True
        time.sleep(0.5)
for fd in fds:
    if fd != held:
        os.close(fd)
os.set_inheritable(held, True)
os.setpriority(os.PRIO_PROCESS, 0, 19)
if shutil.which("ionice"):
    cmd = ["ionice", "-c3", *cmd]
elif shutil.which("taskpolicy"):
    cmd = ["taskpolicy", "-b", *cmd]
os.execvp(cmd[0], cmd)
' "$dir" "${TEST_SLOTS:-2}" "$@"
