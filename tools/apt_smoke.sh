#!/usr/bin/env bash
# The .deb's test: install, set up, run, upgrade while running, remove, and purge, in clean containers.
#
#   tools/apt_smoke.sh <directory with a release's wheels and locks> [image ...]
#
# It builds this machine's architecture's .deb twice from the same files (revisions 1 and 2, so there is something
# to upgrade to), puts each in a repository signed with a key made for this run and thrown away after it, and runs
# packaging/deb/smoke.sh in each image (default: ubuntu:24.04 and debian:12) with the two repositories mounted read
# only. Everything happens in the containers: nothing is installed on this machine, which needs docker, gpg,
# dpkg-deb, and uv. The .deb files it tested are left in <work>/repo1 and repo2 (APT_SMOKE_WORK, default a new
# temporary directory), so the release workflow publishes the one that passed.
#
# APT_SMOKE_CACHE names a directory to keep uv's downloads in between containers and runs (several gigabytes that
# are otherwise fetched once per container).
set -euo pipefail

dist="$(cd "$1" && pwd)"
shift
images=("$@")
[ "${#images[@]}" -gt 0 ] || images=(ubuntu:24.04 debian:12)
here="$(cd "$(dirname "$0")/.." && pwd)"
case "$(uname -m)" in
  x86_64) arch=amd64 ;;
  aarch64) arch=arm64 ;;
  *) echo "no package for $(uname -m)" >&2; exit 1 ;;
esac
work="${APT_SMOKE_WORK:-$(mktemp -d)}"
mkdir -p "$work"

export GNUPGHOME="$work/gnupg"
rm -rf "$GNUPGHOME" && mkdir -m 700 "$GNUPGHOME"
trap 'gpgconf --kill all 2>/dev/null || true; rm -rf "$GNUPGHOME"' EXIT
gpg --batch --quiet --passphrase '' --quick-generate-key "Liberational test repository (thrown away)" ed25519 sign 1d
key="$(gpg --batch --list-keys --with-colons | awk -F: '$1 == "fpr" { print $10; exit }')"

uv_binary=()
for revision in 1 2; do
  repo="$work/repo$revision"
  rm -rf "$repo"
  deb="$(uv run "$here/tools/apt_package.py" build "$dist" --arch "$arch" --out "$repo" --revision "$revision" "${uv_binary[@]}")"
  if [ "$revision" = 1 ]; then # the second build carries the same uv instead of downloading it again
    dpkg-deb --fsys-tarfile "$deb" | tar -xO ./usr/lib/liberational/bin/uv > "$work/uv"
    uv_binary=(--uv "$work/uv")
  fi
  uv run "$here/tools/apt_package.py" index "$repo" --key "$key"
done
desktop-file-validate "$here/packaging/deb/liberational.desktop" 2>/dev/null || [ "$?" = 127 ]

cache=()
if [ -n "${APT_SMOKE_CACHE:-}" ]; then
  mkdir -p "$APT_SMOKE_CACHE"
  cache=(-v "$APT_SMOKE_CACHE:/uv-cache")
fi
for image in "${images[@]}"; do
  printf '\n######## %s\n' "$image"
  docker run --rm -v "$work/repo1:/repo1:ro" -v "$work/repo2:/repo2:ro" "${cache[@]}" \
    -v "$here/packaging/deb/smoke.sh:/smoke.sh:ro" "$image" /smoke.sh
done
printf '\nThe package installs, sets up, runs, upgrades, and is removed on: %s\n' "${images[*]}"
