#!/bin/sh
# Run a command at the lowest CPU and disk priority, so checks never crowd out the live services or the person at the
# computer (AGENTS.md, "Share the machine"; #270). The mise tasks run through this. Linux: nice -n 19 and ionice -c3
# (idle disk class). macOS has no ionice: nice -n 19 with taskpolicy -b (background, which also throttles disk), else
# nice alone.
if command -v ionice >/dev/null 2>&1; then
  exec nice -n 19 ionice -c3 "$@"
elif command -v taskpolicy >/dev/null 2>&1; then
  exec nice -n 19 taskpolicy -b "$@"
fi
exec nice -n 19 "$@"
