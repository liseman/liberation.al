# Releases

A release is a set of files on a GitHub Release that install the system on a machine with no checkout, no compiler,
and no Rust toolchain. A checkout with mise stays the development loop and is not affected by any of this.

## What a release holds

| File | What it is |
|---|---|
| `liberational_pack-<version>-py3-none-any.whl` | The application: the Python package, its settings and service declarations, the dashboard's frontend build, and the lock below without hashes (`resources/constraints.txt`) |
| `continuum_py-<c>-cp314-cp314-<platform>.whl` | Continuum's Python SDK, one per platform ([CONTINUUM.md](CONTINUUM.md), "Wheels") |
| `continuum_daemons-<c>-py3-none-<platform>.whl` | Continuum's daemons, one per platform; they land in the environment's `bin/` |
| `liberational-pack-<version>-lock-<platform>.txt` | Every third-party dependency for that platform, each extra's included, pinned with hashes |
| `SHA256SUMS` | The SHA-256 of every file above |

The platforms are `linux-x86_64` and `linux-aarch64` (glibc 2.36 and later: Ubuntu 24.04 and 26.04, Debian 12 and
13) and `macos-arm64` (macOS 14 and later). Continuum's wheels carry their own version (`<c>`).

Neither the GPU libraries nor the models are in a release: `pack-setup` fetches this machine's extras, pinned to the
lock, and the models.

## Installing one

With [uv](https://docs.astral.sh/uv/) installed, download a release's files into one directory (the wheel, the two
Continuum wheels and the lock for your platform; `gh release download v0.2.0 --repo liseman/liberation.al -D release`
gets them all), then:

```sh
uv tool install --python 3.14 --find-links release \
  --constraints release/liberational-pack-0.2.0-lock-linux-x86_64.txt \
  liberational-pack --with continuum-daemons
pack-system install    # the system packages it needs, after asking
pack-setup             # this machine's extras and the models
pack-app               # open the application
```

`--constraints` pins every dependency to the lock and checks each download against its hash. The three wheels
themselves are checked against `SHA256SUMS` (`sha256sum -c --ignore-missing SHA256SUMS` in that directory).
`--with continuum-daemons` is what puts the daemons in the tool's environment, where the application looks for them.

## The apt repository

On Ubuntu (24.04 and 26.04) and Debian (12 and 13), on x86-64 and arm64, the same release installs with apt, with
no uv, git, or compiler to install first:

```sh
sudo install -d /etc/apt/keyrings
curl -fsSL https://github.com/liseman/liberation.al/releases/download/apt/liberational-archive-keyring.gpg \
  | sudo tee /etc/apt/keyrings/liberational.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/liberational.gpg] https://github.com/liseman/liberation.al/releases/download/apt ./" \
  | sudo tee /etc/apt/sources.list.d/liberational.list
sudo apt update && sudo apt install liberational
pack-setup             # as yourself, not with sudo: this machine's environment and the models
```

Then open Liberational from the applications menu, or with `pack-app`.

**What the package is.** A bootstrap of about 140 MB (`tools/apt_package.py`, `packaging/deb/`): the release's three
wheels and this platform's lock, a pinned `uv` (no distribution ships Python 3.14, so `uv` fetches it), a desktop
entry and an icon, and the `pack-*` commands in `/usr/bin`, each a link to one launcher. Its `Depends` are the apt
packages in `config/system-packages.toml`, so `pack-system install` has nothing left to do. It registers nothing
with systemd and builds nothing as root. Ollama has no apt package: the package recommends it and `pack-setup` says
how to get it.

**The environment is each user's own.** `pack-setup` builds it in `~/.local/lib/liberational` with the same
`uv tool install` as above, then goes on as it does for any installed package (the extras, then the models). Every
other command refuses to run until the environment is the installed release's, and says to run `pack-setup`. Nothing
of the system is shared between users of one machine.

**Upgrading.** `sudo apt update && sudo apt upgrade` has a running application drain and quit (the supervisor stops
the readers, then capture, lets the tier worker store what capture left, and stops the broker last), rebuilds the
environment of every user who has one, as that user, with the arguments they last gave `pack-setup`, and says to
open the application again. If a rebuild can't finish (no network), the upgrade still completes and says who has to
run `pack-setup`; the application won't open on a half-built environment. `pack.toml`, `devices.toml`, and
everything under `~/.local/share/liberational` are not touched. What a release resets is in its notes, below.

**Removing.** `sudo apt remove liberational` quits a running application and deletes the package and each user's
environment (code, several gigabytes, that `pack-setup` makes again). Settings, devices, recordings, what the owner
taught, and models stay. `sudo apt purge liberational` lists each user's settings and data directories with their
sizes and deletes them only after a `y` typed at the terminal; run without a terminal it deletes nothing.

**Where it is hosted.** The repository is the assets of one GitHub Release, tagged `apt`: the `.deb` files, a
`Packages` index, a signed `Release` (`InRelease`), and the public key. It is a flat repository because a release's
assets have no directories, and it is a release rather than GitHub Pages because a `.deb` is over Pages' 100 MB a
file. Earlier versions stay listed, so `apt install liberational=0.1.0-1` still works.

**How it is built.** The release workflow calls `.github/workflows/apt.yml` after the smoke tests. On each
architecture it builds the `.deb` from the files the run has just built and runs `tools/apt_smoke.sh`: in clean
`ubuntu:24.04` and `debian:12` containers, from a repository signed with a key made for that run, it installs the
package, runs `pack-setup` as an ordinary user, starts the application, upgrades to a second revision while it runs
(it must drain, quit, and come back on the new one with the owner's files byte for byte the same), removes it, and
purges it both without a terminal and with a `y`. A tag's run then signs the index with the project's key and
uploads everything to the `apt` release; a rehearsal publishes nothing. To package a release that is already
published, or to ship a fix to the package alone, run it by hand:
`gh workflow run apt.yml -f tag=v0.2.0 -f revision=2`.

To try the test on files built or downloaded locally (it needs docker, gpg, dpkg-deb, and uv, and installs nothing
on the machine it runs on):

```sh
tools/apt_smoke.sh dist                   # both images
tools/apt_smoke.sh dist debian:13         # or the ones you name
```

### The signing key

The repository is signed with a key that only the project's owner and the `APT_SIGNING_KEY` Actions secret hold. No
private key is ever committed: the tests make their own and delete it. Until the secret is set, a tag's run still
publishes the GitHub Release and then fails its `apt` job, saying so.

To make the key, once, on a trusted machine:

```sh
export GNUPGHOME="$(mktemp -d)"
gpg --batch --passphrase '' --quick-generate-key "Liberational apt repository" ed25519 sign 5y
gpg --list-secret-keys --with-fingerprint                          # note the fingerprint
gpg --armor --export-secret-keys | gh secret set APT_SIGNING_KEY --repo liseman/liberation.al
gpg --armor --export-secret-keys > liberational-apt-key.asc        # keep this copy offline, then delete it here
rm -rf "$GNUPGHOME"
```

The key has no passphrase because the workflow has to use it unattended; the secret and the offline copy are what
protect it. Then publish the releases that already exist: `gh workflow run apt.yml -f tag=v0.1.0`. The workflow
exports the public half as `liberational-archive-keyring.gpg` beside the index, which is the file the install
command above fetches.

The key expires after five years. Before then, extend it from the offline copy (`gpg --quick-set-expire <fingerprint>
5y`), set the secret again, and publish any release: machines pick up the new expiry by fetching the keyring again.
If the key is ever exposed, make a new one the same way, set the secret, publish, and tell people to fetch the
keyring again; apt refuses the repository until they do, which is the point.

## Making one

1. Set `version` in `pyproject.toml` (and run `uv lock`, which records it), and add a `## <version>` section at the
   end of this file's notes, below. The section has to say what an upgrade resets, even when that is nothing: this
   project resets machine-made data now and then, and a person upgrading has to know before they do.
2. Merge that, then tag the merge commit `v<version>` and push the tag.

The tag starts `.github/workflows/release.yml` on free hosted runners. It refuses a tag that doesn't match
`pyproject.toml` or has no notes here, builds the frontend and the wheels, writes the locks from `uv.lock`
(`tools/release.py`), and then runs the smoke test on each platform (`tools/release_smoke.sh`: in a clean
`ubuntu:24.04` container on Linux, on the runner itself on macOS, with no GPU anywhere). The smoke test installs from
the built files as above, runs `pack-system install` and `pack-setup` without the models, starts demo mode and asks
the dashboard for its page and the demo household, and then runs the supervisor with the broker and the dashboard
and quits it. Only when that passes on all three platforms is the GitHub Release published, with this file's section
as its notes. A failed smoke test publishes nothing; fix it, move the tag, and push again.

Started by hand (`gh workflow run release.yml`, or the Actions tab), the same workflow is a rehearsal: it builds and
smoke-tests everything, keeps the files as the run's artifacts, and publishes nothing. It never runs for a pull
request. A version like `0.2.0rc1` (tag `v0.2.0rc1`) is published as a pre-release.

To try the smoke test on files built locally (`mise run wheel`, `mise run continuum-wheels`, and the locks from
`tools/release.py`), run it in a container, never on a machine with an install: it installs system packages and
writes the user's `pack.toml`.

```sh
docker run --rm -v "$PWD/dist:/dist:ro" -v "$PWD/tools:/tools:ro" ubuntu:24.04 /tools/release_smoke.sh /dist
```

# Notes

One section per version, newest last. The release workflow publishes a version's section as that release's notes.

## 0.1.0

The first release: the application as wheels for Linux x86-64, Linux arm64, and macOS on Apple Silicon, installed
with `uv tool install` and set up with `pack-setup`.

The wheels' private FFmpeg has no AV1 encoder and no TLS, so an installed system can't run the opt-in AV1 archive
pass or read `rtsps://` cameras; a checkout can.

An upgrade resets nothing: there is no earlier release to upgrade from.
