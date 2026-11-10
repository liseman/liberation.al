#!/usr/bin/env bash
# Link the Continuum daemons this checkout built (continuum/target/release/continuum-*) into .venv/bin, where the
# services find every program they run (config.program()). `mise run continuum-release` runs it after the build;
# `mise run continuum-link` runs it alone, which is needed once after the environment is made again.
#
#   tools/link-daemons.sh
#
# A link, not a copy: the next build is picked up with no second step. A checkout with no .venv (a worktree that
# only runs tests) is left alone.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
bin="$root/.venv/bin"
if [ ! -d "$bin" ]; then
    echo "link-daemons: no $bin; nothing linked (run it again after \`mise run setup\`)"
    exit 0
fi
linked=0
for built in "$root"/continuum/target/release/continuum-*; do
    if [ -f "$built" ] && [ -x "$built" ]; then
        ln -sfn "$built" "$bin/$(basename "$built")"
        linked=$((linked + 1))
    fi
done
if [ "$linked" -eq 0 ]; then
    echo "link-daemons: no daemons in $root/continuum/target/release; build them with \`mise run continuum-release\`" >&2
    exit 1
fi
echo "link-daemons: $linked daemons linked into $bin"
