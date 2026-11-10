#!/usr/bin/env bash
# Runs inside a clean container, as root, for tools/apt_smoke.sh: /repo1 and /repo2 are two signed repositories
# with revisions 1 and 2 of the package. A person, alice, installs, sets up, and runs it; root upgrades and removes.
#
# What it checks, in order:
#   1. `apt-get install liberational` from the signed repository: its Depends resolve on this release, nothing is
#      registered with systemd, and no git, mise, rustc, node, or compiler came with it.
#   2. The application refuses to open before setup, and pack-setup refuses to run as root.
#   3. alice's `pack-setup` (without the models) builds her environment and ends with nothing left to do.
#   4. The application runs: the supervisor with the broker and the dashboard, as tools/release_smoke.sh starts it.
#   5. An upgrade to revision 2 while it runs: it drains and quits, her environment is rebuilt for the new release,
#      apt says to open it again, her settings, devices, and data are byte for byte the same, and it runs again.
#   6. `apt-get remove` while it runs: it quits, the environment and the commands go, her files stay.
#   7. `apt-get purge` with no terminal lists her files and keeps them; with a terminal and a "y" it deletes them.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

step() { printf '\n== %s (%ss)\n' "$*" "$SECONDS"; }
fail() { echo "FAILED: $*" >&2; exit 1; }
alice() { runuser -u alice -- env HOME=/home/alice USER=alice UV_CACHE_DIR="${UV_CACHE_DIR:-}" PATH=/usr/bin:/bin "$@"; }
use_repo() {
  echo "deb [signed-by=/$1/liberational-archive-keyring.gpg] file:/$1 ./" > /etc/apt/sources.list.d/liberational.list
  apt-get update -q > /dev/null
}
wait_for() {
  for _ in $(seq 1 120); do
    if curl -fsS -H "Accept: ${3:-application/json}" -o "$2" "$1" 2> /dev/null; then return 0; fi
    sleep 1
  done
  fail "nothing answered at $1"
}
open_application() {
  alice pack-supervisor run > "/tmp/supervisor-$1.log" 2>&1 &
  wait_for "http://127.0.0.1:$port/" /tmp/app.html text/html
  grep -q '<title>Liberational</title>' /tmp/app.html || fail "the application's page isn't the dashboard"
  alice pack-supervisor status > /tmp/status.txt
  for name in continuum-broker dashboard; do
    grep -Eq "^$name +running" /tmp/status.txt || fail "$name isn't running"
  done
}
running() { pgrep -u alice -f 'pack-supervisor run' > /dev/null; }
owner_files() { (cd /home/alice && find .config/liberational .local/share/liberational/taught -type f -exec sha256sum {} + | sort -k2); }

env=/home/alice/.local/lib/liberational
useradd -m alice
if [ -d /uv-cache ]; then
  chown -R alice /uv-cache # alice's number differs between images
  export UV_CACHE_DIR=/uv-cache
fi

step "1. Install from the signed repository"
use_repo repo1
apt-get install -yq liberational > /tmp/install.log 2>&1 || { tail -40 /tmp/install.log; fail "apt-get install"; }
grep -q "pack-setup" /tmp/install.log || fail "the install didn't say to run pack-setup"
dpkg -s liberational | grep -E '^(Version|Depends|Recommends|Installed-Size):'
[ -z "$(dpkg -L liberational | grep -E 'systemd|/etc/' || true)" ] || fail "the package has systemd or /etc files"
for tool in git mise rustc cargo node cc gcc; do
  ! command -v "$tool" > /dev/null || fail "$tool was installed"
done
[ -x /usr/bin/pack-app ] && [ -f /usr/share/applications/liberational.desktop ] || fail "no command or desktop entry"

step "2. Nothing opens before setup, and nothing is set up as root"
! alice pack-app 2> /tmp/unset.log || fail "pack-app opened with no environment"
grep -q "Run pack-setup" /tmp/unset.log || fail "pack-app didn't say to run pack-setup"
! pack-setup 2> /tmp/root.log || fail "pack-setup ran as root"

step "3. pack-setup, without the models"
alice pack-setup --no-models --no-ollama > /tmp/setup.log 2>&1 || { tail -40 /tmp/setup.log; fail "pack-setup"; }
tail -3 /tmp/setup.log
cmp -s /usr/lib/liberational/release "$env/liberational-pack/.release" || fail "the environment isn't stamped"
du -sh "$env" | sed 's/^/the environment: /'

step "4. The application runs"
alice "$env/liberational-pack/bin/python" - << 'EOF'
from liberational import config, services

keep = {"continuum-broker", "dashboard"}
off = sorted(set(services.selected()) - keep)
path = config.local_config()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("[services]\ndisable = [" + ", ".join(f'"{name}"' for name in off) + "]\n")
config.devices_file().write_text("# the owner's devices\n")
taught = config.data_dir() / "taught"
taught.mkdir(parents=True, exist_ok=True)
(taught / "knowledge.sqlite3").write_bytes(b"what the owner taught")
EOF
port="$(alice pack-config ports.dashboard)"
owner_files > /tmp/before.txt
[ "$(wc -l < /tmp/before.txt)" = 3 ] || fail "expected pack.toml, devices.toml, and the taught file"
open_application 1

step "5. Upgrade while it runs"
use_repo repo2
apt-get install -yq liberational > /tmp/upgrade.log 2>&1 || { tail -40 /tmp/upgrade.log; fail "the upgrade"; }
grep -E "Liberational (is running|quit after|was upgraded)" /tmp/upgrade.log
grep -q "Liberational quit after" /tmp/upgrade.log || fail "the running application wasn't asked to quit"
grep -q "Liberational was upgraded. Open it again." /tmp/upgrade.log || fail "the upgrade didn't say to open it again"
! running || fail "the application is still running after the upgrade"
grep -q "quit: stop" /tmp/supervisor-1.log && grep -q "stopped" /tmp/supervisor-1.log || fail "it didn't drain and stop"
dpkg -s liberational | grep -q '^Version: .*-2$' || fail "revision 2 isn't installed"
grep -q '^package: .*-2$' "$env/liberational-pack/.release" || fail "the environment wasn't rebuilt for revision 2"
owner_files | cmp -s - /tmp/before.txt || fail "the upgrade changed the owner's files"
open_application 2

step "6. Remove while it runs"
apt-get remove -yq liberational > /tmp/remove.log 2>&1 || { tail -40 /tmp/remove.log; fail "apt-get remove"; }
grep -E "Liberational quit after|^Removed|^Kept" /tmp/remove.log
! running || fail "the application is still running after the removal"
[ ! -e "$env" ] && [ ! -e /usr/bin/pack-app ] || fail "the environment or the commands were left"
owner_files | cmp -s - /tmp/before.txt || fail "the removal changed the owner's files"

step "7. Purge asks before it deletes"
apt-get purge -yq liberational > /tmp/purge.log 2>&1 || { tail -40 /tmp/purge.log; fail "apt-get purge"; }
grep -q "^Kept. To delete them yourself" /tmp/purge.log || fail "a purge with no terminal didn't keep the files"
owner_files | cmp -s - /tmp/before.txt || fail "a purge with no terminal changed the owner's files"
apt-get install -yq liberational > /dev/null 2>&1
# A terminal (script's), and a "y" typed once the question is on it.
(for _ in $(seq 1 100); do
  if grep -q "Delete all of the above" /tmp/purge-tty.log 2> /dev/null; then break; fi
  sleep 1
done; printf 'y\n'; sleep 3) |
  DEBIAN_FRONTEND=readline timeout 120 script -qec "apt-get purge -y liberational" /dev/null > /tmp/purge-tty.log 2>&1 ||
  fail "a purge at a terminal didn't ask, or didn't finish"
grep -q "^Deleted" /tmp/purge-tty.log || fail "a yes at the terminal didn't delete"
[ ! -e /home/alice/.config/liberational ] && [ ! -e /home/alice/.local/share/liberational ] || fail "a yes left files"

step "Done"
echo "$(. /etc/os-release && echo "$PRETTY_NAME"): install, setup, run, upgrade, removal, and purge all passed."
