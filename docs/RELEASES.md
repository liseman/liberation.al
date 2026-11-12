# Releases

A release is a set of files on a GitHub Release that install the system on a machine with no checkout, no compiler, and
no Rust toolchain. A checkout with `mise` remains the development workflow ([CONTRIBUTING.md](../CONTRIBUTING.md)).

## What a release holds

| File | What it is |
|---|---|
| `liberational_pack-<version>-py3-none-any.whl` | The application: the Python package, its settings and service declarations, the dashboard's frontend build, and the marker-aware lock without hashes (`resources/constraints.txt`) |
| `continuum_py-<c>-cp314-cp314-<platform>.whl` | Continuum's Python SDK, one per platform ([CONTINUUM-SDK.md](CONTINUUM-SDK.md#wheels)) |
| `continuum_daemons-<c>-py3-none-<platform>.whl` | Continuum's daemons, one per platform; they install into the environment's `bin/` |
| `liberational-pack-<version>-lock-<platform>.txt` | Every third-party dependency for that platform, all extras included, pinned with SHA-256 hashes |
| `SHA256SUMS` | The SHA-256 checksum of every file above |

The platforms are `linux-x86_64` and `linux-aarch64` (glibc 2.36 and later: Ubuntu 24.04 and 26.04, Debian 12 and 13)
and `macos-arm64` (macOS 14 and later). Continuum's wheels carry their own version (`<c>`) and statically link a
private LGPL (version 3 or later) FFmpeg build with H.264, AV1 (for the archive), and TLS (`rtsps://` cameras); see
[CONTINUUM-SDK.md](CONTINUUM-SDK.md#wheels).

Neither GPU libraries nor models are in the release files: `pack-setup` installs this machine's Python extras against
the lock and fetches the models.

## Install with apt

On Ubuntu (24.04 and 26.04) and Debian (12 and 13), on x86-64 and arm64, install from the project's `apt` repository
with no `uv`, `git`, or compiler first, but with `ca-certificates` and `curl`, which `apt` needs before it can reach
the repository. The commands need an account with `sudo` (or root):

```sh
sudo apt-get update && sudo apt-get install -y ca-certificates curl
sudo install -d /etc/apt/keyrings
curl -fsSL https://github.com/liseman/liberation.al/releases/download/apt/liberational-archive-keyring.gpg \
  | sudo tee /etc/apt/keyrings/liberational.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/liberational.gpg] https://github.com/liseman/liberation.al/releases/download/apt ./" \
  | sudo tee /etc/apt/sources.list.d/liberational.list
sudo apt update && sudo apt install liberational
pack-setup             # run as yourself, never with sudo: fetches the models, and pulls Ollama's once it is installed
```

Ollama has no apt package, so the `.deb` names it only under Recommended packages. Install it from
<https://ollama.com/download>, then run `pack-setup` again to pull its models.

Open Liberational from the applications menu, or run `pack-app`, or run `pack-supervisor run` on a machine without a
desktop session ([RUNNING.md](RUNNING.md)). Pass `--check` to `pack-setup` to list what it would install without
installing or downloading (its exit status is not a completeness check: it is 0 while optional extras and models
are still absent), `--no-models` to skip model downloads, or `--no-ollama` to skip Ollama pulls.

### Upgrade, remove, or purge an apt install

Upgrade with `apt`:

```sh
sudo apt update && sudo apt upgrade
```

The upgrade drains and quits a running application ([SERVICES.md](SERVICES.md#one-application)), rebuilds the
environment of every user who has one (as that user, with the arguments they last gave `pack-setup`), and leaves
`pack.toml`, `devices.toml`, and `~/.local/share/liberational` untouched. Open `pack-app` again afterward. If a rebuild
cannot finish (for example, without network access), `apt upgrade` still completes and prints which user needs to run
`pack-setup`. Check [Notes](#notes) before upgrading to see whether a version resets machine-made data. To pin an older
package version, run `sudo apt install liberational=0.1.0-1`.

Remove the package, each user's `~/.local/lib/liberational` environment, and `~/.config/autostart/liberational.desktop`,
while keeping settings, devices, recordings, taught state, and models:

```sh
sudo apt remove liberational
```

Delete settings and data as well (interactive terminals only; non-interactive runs delete nothing):

```sh
sudo apt purge liberational
```

`purge` lists each user's configuration and data directories with their sizes and deletes them only after `y` is typed
at the terminal.

## Install with uv

On any supported Linux or macOS machine with [uv](https://docs.astral.sh/uv/) and the [GitHub
CLI](https://cli.github.com) installed, download a release's files for your platform into one directory (`gh release
download v0.1.0 --repo liseman/liberation.al -D release` fetches all assets), check them, and install the tool:

```sh
(cd release && sha256sum -c --ignore-missing SHA256SUMS)
uv tool install --python 3.14 --find-links release \
  --constraints release/liberational-pack-0.1.0-lock-linux-x86_64.txt \
  liberational-pack --with continuum-daemons
pack-system install    # installs missing system packages, after asking
pack-setup             # installs this machine's Python extras, and pulls Ollama's once it is installed
pack-app               # opens the application, or `pack-supervisor run` without a desktop session
```

On macOS, use `shasum -a 256 -c SHA256SUMS` for the first line: `sha256sum` is GNU coreutils. Replace
`linux-x86_64` with `linux-aarch64` or `macos-arm64` to match your machine. `--constraints` pins every third-party
dependency to the lock and verifies its SHA-256 hash; `--with continuum-daemons` installs the Rust daemons into the
tool's `bin/` directory. Ollama has no apt package, so install it from <https://ollama.com/download>, then run
`pack-setup` again to pull its models.

## Make a release

1. Set `version` in `pyproject.toml`, run `uv lock` so `uv.lock` records it, and add a `## <version>` section at the
   end of [Notes](#notes) below. Every section must state what an upgrade resets, even when that is nothing: the
   project resets machine-made data from time to time, and `tools/release.py notes` refuses a section without that
   statement.
2. Merge that change to `main`, tag the merge commit `v<version>` (a tag like `v0.2.0rc1` publishes as a pre-release),
   and push the tag.

If a smoke test fails during the workflow run, nothing is published; fix the failure on `main`, move the tag, and push
again.

- **Rehearse without publishing:** run `gh workflow run release.yml` (or start it from the Actions tab) to build,
  smoke-test, and retain the workflow artifacts without creating a GitHub Release or updating `apt`.
- **Publish or revise the `.deb` alone:** run `gh workflow run apt.yml -f tag=v0.2.0 -f revision=2` to package an
  already published release or ship a packaging fix under a new revision number.
- **Run the smoke tests locally:** put a release's three wheels and lock into `dist/` (downloaded from a release or
  rehearsal artifact, or built locally with `src/liberational/resources/constraints.txt` from
  `tools/release.py constraints`, `mise run wheel`, `mise run continuum-wheels` copied from `continuum/dist/`, and
  `tools/release.py lock`), then run the container tests. Never run `tools/release_smoke.sh` directly on a host with an
  install, because it installs system packages and overwrites `pack.toml`:

```sh
docker run --rm -v "$PWD/dist:/dist:ro" -v "$PWD/tools:/tools:ro" ubuntu:24.04 /tools/release_smoke.sh /dist
tools/apt_smoke.sh dist                   # ubuntu:24.04 and debian:12 containers (needs docker, gpg, dpkg-deb, uv)
tools/apt_smoke.sh dist debian:13         # or specific container images
```

### The signing key

The `apt` repository is signed with an `ed25519` key held only offline by the project owner and in the
`APT_SIGNING_KEY` GitHub Actions secret. No private key is committed; container smoke tests generate a throwaway key
per run and delete it. Until `APT_SIGNING_KEY` is set, a tag push still publishes the GitHub Release and then fails its
`apt` job with a message saying why.

Generate the key once on a trusted machine:

```sh
export GNUPGHOME="$(mktemp -d)"
gpg --batch --passphrase '' --quick-generate-key "Liberational apt repository" ed25519 sign 5y
gpg --list-secret-keys --with-fingerprint                          # record the fingerprint
gpg --armor --export-secret-keys | gh secret set APT_SIGNING_KEY --repo liseman/liberation.al
gpg --armor --export-secret-keys > liberational-apt-key.asc        # store offline, then delete this local file
rm -rf "$GNUPGHOME"
```

The key has no passphrase so the workflow can sign unattended; the repository secret and offline copy protect it. After
setting the secret, publish any existing release into the repository with `gh workflow run apt.yml -f tag=v0.1.0`. The
workflow exports the public key as `liberational-archive-keyring.gpg` beside the index.

The key expires after five years. Before expiry, extend it from the offline copy
(`gpg --quick-set-expire <fingerprint> 5y`), update `APT_SIGNING_KEY`, and publish a release; machines pick up the new
expiry by fetching `liberational-archive-keyring.gpg` again. If the key is ever exposed, generate a replacement key the
same way, update `APT_SIGNING_KEY`, publish, and instruct users to re-fetch the keyring (`apt` refuses the repository
until they do).

## The apt repository

### What the package is

The `.deb` (`tools/apt_package.py`, `packaging/deb/`) is a bootstrap package of over 100 MB: it carries the release's
three wheels and platform lock in `/usr/lib/liberational/wheels/`, a pinned `uv` binary in
`/usr/lib/liberational/bin/uv` (no target distribution ships Python 3.14, so `uv` fetches it), a desktop entry and
icon, and `/usr/bin/pack-*` symlinks to `/usr/lib/liberational/launch`.

Package relations come from `config/system-packages.toml`: required unconditional `apt` packages (plus
`ca-certificates`) go in `Depends`, hardware-conditional packages (`sox`, on a machine with an NVIDIA GPU) and
`ollama` go in `Recommends`, optional packages (`arp-scan`) go in `Suggests`, and source-build packages
(`build = true`) are omitted.
The package registers nothing with `systemd` and compiles nothing as root.

### Per-user environments

Each user owns an isolated tool environment in `~/.local/lib/liberational/liberational-pack`, built by `pack-setup`
with the same `uv tool install` command used for wheel installs. `pack-setup` copies `/usr/lib/liberational/release` to
`~/.local/lib/liberational/liberational-pack/.release` when the build finishes and saves the user's flags in
`~/.local/lib/liberational/setup-args`. Every other `pack-*` command compares `.release` against
`/usr/lib/liberational/release` and refuses to run on a missing or mismatched environment, telling the user to run
`pack-setup`. Users on the same machine share only the immutable bootstrap files under `/usr`.

### Where it is hosted

The repository lives on a single GitHub Release tagged `apt`: the `.deb` files, `Packages` and `Packages.gz`, `Release`,
`Release.gpg`, and `InRelease`, and `liberational-archive-keyring.gpg`. It uses Debian's flat repository layout (`./`)
because GitHub Release assets have no subdirectories, and uses GitHub Releases rather than GitHub Pages because each
`.deb` exceeds Pages' 100 MB per-file limit. Older `.deb` versions stay attached to the `apt` release so version pins
keep working.

### How releases and packages are built

Pushing a `v<version>` tag triggers `.github/workflows/release.yml` (pull requests never run it). The workflow checks
that the tag matches `pyproject.toml` and has a section under [Notes](#notes), builds the frontend, exports `uv.lock`
into `resources/constraints.txt` and per-platform hash locks (`tools/release.py`), builds and verifies the application
wheel (`tools/check_wheel.py --lock`), and builds Continuum's wheels on Linux x86-64, Linux arm64, and macOS arm64.

Before publishing anything, `tools/release_smoke.sh` runs on all three platforms (in a clean `ubuntu:24.04` container
on Linux and on the runner on macOS, without a GPU): it installs from the built wheels and lock, runs
`pack-system install --yes` and `pack-setup --no-models --no-ollama`, verifies `pack-demo` serves the dashboard and
demo household, and starts `pack-supervisor run` with `continuum-broker` and `dashboard` before stopping it with
`pack-supervisor quit`. Only after all three platforms pass does the workflow publish the GitHub Release.

`.github/workflows/apt.yml` then runs on x86-64 and arm64. On each architecture, `tools/apt_smoke.sh` builds revisions
1 and 2 of the `.deb` in temporary signed repositories and runs `packaging/deb/smoke.sh` inside clean `ubuntu:24.04`
and `debian:12` containers: installing as root, running `pack-setup` as an unprivileged user, starting the supervisor,
upgrading from revision 1 to revision 2 while it runs (verifying it drains, stops, and restarts on revision 2 with the
user's files unchanged byte for byte), removing the package, and purging it both non-interactively and with `y`. On a
tag run, the `publish` job signs the merged index with `APT_SIGNING_KEY` and uploads the `.deb` files and index to the
`apt` release.

# Notes

One section per version, newest last. The release workflow publishes a version's section as that release's notes.

## 0.1.0

The first release runs as one application: `pack-app` opens its window, and `pack-supervisor run` runs without one.
The release workflow builds application and Continuum wheels for Linux x86-64, Linux arm64, and Apple Silicon
macOS, plus dependency locks and checksums. Linux also has a `.deb` and an apt repository. There is no macOS
installer package; Apple Silicon uses the wheels with `uv tool install`. Run `pack-setup` after installing.

A fresh install resets nothing.

A checkout from before the first public commit (2026-11-09) does not upgrade in place. Export what was taught
(`pack-knowledge export`), set up fresh, and import the bundle. Recorded history is not carried.

A checkout from 2026-11-09 or later follows [MIGRATING.md](../MIGRATING.md), starting with its ordered path for
older checkouts before pulling:

- It runs as one application. `python -m liberational.services migrate` removes the earlier per-service units. A
  dry run prints the plan and changes nothing. `pack-services` no longer renders, installs, or uninstalls a unit per
  service: it is the open application's client, and with none open it changes nothing and says to open one.
- It resets recorded history. The catalog is format 8, and the broker refuses an older catalog.
  `pack-reset-catalog --keep-taught` prints the plan; `--yes` resets. The knowledge store, the face gallery, voice
  references, and models stay. The reset also works with the application closed; open it afterwards.
- A place's plan is a Pascal Editor scene. An older plan converts at commit `add5028c` with
  `.venv/bin/python -m liberational.spatial.plan_convert` (a dry run, then `--yes`) before this version is checked
  out. The converter was removed after that commit. The old line format, its text editor, and the vendored
  floorplan library are gone. An unconverted plan is kept but no longer shown; conversion keeps the old versions.
- Recorded walk imports and nightly camera re-registration are gone. The machine-made 3D map and splat generation
  are removed. Old map and splat files stay on disk unread unless you reset places or the catalog.
  `pack-reset-places` (a dry run, then `--yes`) deletes those files, camera registrations, point clouds, and stored
  positions, while keeping corrected plans and place names in the knowledge store.
- Delete these removed settings from `pack.toml`: `spatial.runtime`, `spatial.gpu`, `spatial.splat_runtime`,
  `[models.hf.mast3r]`, and `[models.hf.depth_anything]`. Earlier checkouts also remove `spatial.cameras_enabled`,
  `spatial.splats_enabled`, `spatial.geocalib_runtime`, `spatial.cut3r_runtime`, `[models.files.geocalib]`,
  `[models.files.cut3r]`, and `spatial-cameras` under `[services]`. The upgrade guide covers camera settings moved
  to the device manifest. Restart every service that reads configuration after changing these settings.
- Stored DINOv3 pet embeddings are not read. `tools/restore_pet_references.py` carries taught pet examples onto
  DINOv2 before a restart. MIGRATING.md has the arguments.
- The position layer's records gained an `approximate` column, which a part written before this version does not
  have: run `pack-positions --reset` after pulling, then restart the dashboard and the journal. It places the last
  30 days again from the raw samples.

The new 3D view is optional and off unless configured. Set `spatial.pointmap_runtime` to a MASt3R-SLAM setup you
install yourself, set `spatial.pointmap_weights` to its weights, and turn on Learn surroundings. The pack downloads
none of that setup or its weights. MASt3R-SLAM and the weights have a non-commercial licence (CC BY-NC-SA 4.0).
This needs CUDA. For a fixed camera with a visible floor, `mount.height_m` in the device manifest sizes its cloud;
otherwise the size is approximate.

The view shows one camera's cloud by itself, or every camera's cloud placed where its camera stands on your plan.
The clouds are separate layers, never blended into one model or a splat, and cameras are placed on the plan from
their spots and its walls. People and pets are placed on the plan from each camera's spot.

Voice naming no longer lets a person's own unnamed lines block their name. Positions on an owner plan no longer
put people or animals behind a wall their camera cannot see through. The plan editor also gains version history,
camera spots, and controls for the plan's scale.

An unnamed voice now opens only when a line recurs, so a one-off line (a show playing near a microphone) stays
unattributed. A pet's card says how many of its taught sightings keep no saved picture, and a picture saved after the
first naming is kept by naming the sighting again. A camera's tilt can now be confirmed by the straight edges in its
picture, and an animal whose floor point a solid wall hides is placed at the furniture, marked approximate.

## 0.1.1

A release of fixes and additions on top of 0.1.0. No setting has to be removed, and nothing taught is touched.

An upgrade resets one thing, and it is machine-made: stored positions. The position layer's records gained an
`approximate` column that a part written before this version does not have, so run `pack-positions --reset` after
upgrading and restart the application. It places the last 30 days again from the raw samples. A place's 3D map is
also machine-made and builds up again by itself. The knowledge store, the face gallery, voice references, pets, plans,
and recordings stay as they are.

After upgrading, run `pack-setup` (or `pack-models`) once: the person and animal detector is now RF-DETR Medium, a new
model file, and vision does not start without it.

The 3D view:

- A place has one map that builds up. Each camera's shot is lined up with the plan and with what the map already
  holds, then added up to a set density, so the map fills in as cameras see more and does not grow without limit. A
  shot that cannot be brought into agreement is left out and its row says so. What a newer shot looks straight through
  leaves the map, so furniture that moves is updated. The view says how much of the plan's floor the map covers.
- A camera that is turned or carried is followed without a button. While it moves it is tracked from its live
  frames, its keyframes join the map as they are made, and its row counts the points the move has added. Your plan and
  the spots on it are never written.
- The view opens on All cameras, updates in place without reloading, and fills the window with the page's usual
  margin under it, as do the drawn plan and the plan editor.

Live:

- Enhance steadies the picture of the large camera and highlights small changes, in three looks: Subtle, Contrast,
  and Colour. It is off until you turn it on, and it keeps a 1920x1080 stream at 60 frames a second in Firefox and
  Chromium. The Colour look uses Eulerian video magnification, a technique whose patents (held by MIT) permit
  non-commercial use.
- A heart rate can be read from the colour of skin in the picture of a close camera that delivers about 30 frames a
  second in colour, and is shown beside a person's name only while the reading can be trusted. It is approximate and
  not a medical device. It needs that camera recorded at a higher quality (`encoding = {enabled = true, cq = 20}` in
  its device entry), and `vision.heart_rate = false` turns it off.
- The conversation keeps its place when a name is saved, and a camera option shows its saved state as a small mark
  beside the control, so nothing shifts.

Recognition and memory:

- Find objects looks for an object by the name you type when SAM 3 is installed. SAM 3 is an optional, gated download
  with its own licence that is not open source, and it is never bundled; without it Find objects falls back to SAM 2.1,
  and then to the model it used before. Settings says which is in use and why.
- A remembered object can be renamed and removed from the Objects tab, and the plan editor adds, moves, reshapes, and
  removes objects and unexplored areas.
- An unnamed voice opens only when a line recurs, so playback near a microphone no longer fills the unnamed list.
  Naming a pet always keeps its recovery picture. An animal up on furniture is placed at the furniture, marked
  approximate, where it used to get no place.

Installing and running:

- `pack-setup --check` fails and names what is missing when extras or models are absent. A Linux machine with no
  NVIDIA GPU no longer downloads the GPU extras it cannot use, and setup says what will not work there. The package no
  longer recommends `ollama`, which has no apt package: install it from its own site.
- `pack-services` is only the open application's client. It no longer renders, installs, or uninstalls a unit per
  service; `python -m liberational.services migrate` still removes units an older install left behind.
- A whole-codebase audit removed dead code and fixed a number of small defects and slow paths.
