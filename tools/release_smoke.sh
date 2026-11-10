#!/usr/bin/env bash
# The release's smoke test: install a release's files on this machine as a new person would, set it up, and start
# it. The release workflow runs it in a clean container with no GPU, and no release is published when it fails.
#
#   tools/release_smoke.sh <directory with the release's wheels and locks>
#
# It changes the machine it runs on (it installs system packages and a uv tool, and writes the user's pack.toml), so
# run it in a container or a throwaway runner, never on a machine with an install:
#
#   docker run --rm -v "$PWD/dist:/dist:ro" -v "$PWD/tools:/tools:ro" ubuntu:24.04 /tools/release_smoke.sh /dist
#
# What it checks, in order:
#   1. `uv tool install` of the package with this platform's lock, from the files alone.
#   2. `pack-system install --yes`, then `pack-setup` with no models: it must end with nothing left to do, which
#      means this machine's extras installed at the locked versions and Continuum's SDK and daemons are there.
#   3. Demo mode (`pack-demo`): the dashboard serves its page and the demo household's people.
#   4. The application itself (`pack-supervisor run`) with only the broker and the dashboard selected, since the
#      machine has no cameras, microphones, or models: both run, the dashboard answers, and `quit` stops everything.
set -euo pipefail

dist="$(cd "$1" && pwd)"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) platform=linux-x86_64 ;;
  Linux-aarch64) platform=linux-aarch64 ;;
  Darwin-arm64) platform=macos-arm64 ;;
  *) echo "no release for $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac
lock=("$dist"/*-lock-"$platform".txt)
[ -f "${lock[0]}" ] || { echo "no lock for $platform in $dist" >&2; exit 1; }

step() { printf '\n== %s\n' "$*"; }

# wait_for <url> <file> [<accept>]: what <url> answers, saved to <file>, within two minutes. The dashboard serves its
# page to a request that accepts text/html, and its JSON to any other.
wait_for() {
  for _ in $(seq 1 120); do
    if curl -fsS -H "Accept: ${3:-application/json}" -o "$2" "$1" 2>/dev/null; then return 0; fi
    sleep 1
  done
  echo "nothing answered at $1" >&2
  return 1
}

if [ "$(uname -s)" = Linux ] && [ "$(id -u)" = 0 ]; then
  step "A bare image: curl, certificates, and sudo (pack-system installs through it)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q >/dev/null
  apt-get install -yq --no-install-recommends ca-certificates curl sudo >/dev/null
fi
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
for tool in rustc cargo; do
  if [ "${SMOKE_ALLOW_TOOLCHAIN:-}" != 1 ] && command -v "$tool" >/dev/null; then
    echo "$tool is installed: not a clean machine (SMOKE_ALLOW_TOOLCHAIN=1 on a hosted runner that has it)" >&2
    exit 1
  fi
done

step "Install from the release's files, pinned to ${lock[0]##*/}"
uv tool install --python 3.14 --find-links "$dist" --constraints "${lock[0]}" liberational-pack --with continuum-daemons
pack-config ports.dashboard >/dev/null

step "System packages"
pack-system install --yes

step "pack-setup, without the models"
pack-setup --no-models --no-ollama

work="$(mktemp -d)"
step "Demo mode"
pack-demo --dir "$work/demo" --port 8766 > "$work/demo.log" 2>&1 &
demo=$!
trap 'kill "$demo" 2>/dev/null || true; cat "$work"/*.log 2>/dev/null | tail -60' EXIT
wait_for http://127.0.0.1:8766/ "$work/page.html" text/html
grep -q '<title>Liberational</title>' "$work/page.html" || { echo "the demo's page isn't the dashboard" >&2; exit 1; }
wait_for http://127.0.0.1:8766/people "$work/people.json"
grep -q 'Dana' "$work/people.json" || { echo "the demo's people don't include the demo household" >&2; exit 1; }
kill "$demo"
wait "$demo" || true
echo "the demo dashboard served its page and the demo household"

step "The application: the supervisor with the broker and the dashboard"
python="$(uv tool dir)/liberational-pack/bin/python"
"$python" - <<'EOF'
from liberational import config, services

keep = {"continuum-broker", "dashboard"}
off = sorted(set(services.selected()) - keep)
path = config.local_config()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("[services]\ndisable = [" + ", ".join(f'"{name}"' for name in off) + "]\n")
print(f"{path}: {len(off)} services off, {sorted(keep)} on")
EOF
port="$(pack-config ports.dashboard)"
pack-supervisor run > "$work/supervisor.log" 2>&1 &
supervisor=$!
trap 'kill "$supervisor" 2>/dev/null || true; cat "$work"/*.log 2>/dev/null | tail -60' EXIT
wait_for "http://127.0.0.1:$port/" "$work/app.html" text/html
grep -q '<title>Liberational</title>' "$work/app.html" || { echo "the application's page isn't the dashboard" >&2; exit 1; }
pack-supervisor status | tee "$work/status.txt"
for name in continuum-broker dashboard; do
  grep -Eq "^$name +running" "$work/status.txt" || { echo "$name isn't running" >&2; exit 1; }
done
pack-supervisor quit
wait "$supervisor"
if curl -fsS -o /dev/null "http://127.0.0.1:$port/" 2>/dev/null; then
  echo "the dashboard still answers after quit" >&2
  exit 1
fi
trap - EXIT
echo
echo "The release installs, sets up, and starts on $platform."
