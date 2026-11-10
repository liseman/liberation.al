# Upgrading an existing install

This guide covers the current Continuum installation. Fresh setup is in
[CONTRIBUTING.md](CONTRIBUTING.md); components and state paths are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The old genericize and v1 import
procedures are retired. Do not run an old catch-up script against current data.

## The 3D view has the SuperSplat viewer's controls

The Places 3D view is now PlayCanvas's SuperSplat viewer, embedded: orbit, fly, and walk modes, with its own help
panel (H) and fullscreen. It is a frontend change only: after the pull, `mise run frontend` and restart the
dashboard. Nothing is dropped.

## The 3D view draws with PlayCanvas, and splats are served as SOG

The Places 3D view moved from three.js and Spark to PlayCanvas, which reads a splat as SOG, not SPZ. After the pull:

```sh
mise run frontend                 # the dashboard's new build (PlayCanvas replaces three.js and Spark)
.venv/bin/pack-spatial-splat      # gives each kept splat its SOG copy from its PLY, and removes its SPZ
```

Then restart the dashboard. Until `pack-spatial-splat` has run (the daily place job runs it too), a place whose splat
was made before the pull shows its points, not its splat. No splat is retrained, and nothing else is dropped.

## Fixed-camera splats train against a depth model

A fixed-camera place's splat now trains against a depth image per camera from Depth Anything 3's metric model, which
runs in gsplat's runtime. On a machine that makes splats (CUDA), after the pull:

```sh
mise run gsplat   # rebuilds data/gsplat: the same gsplat, plus Depth Anything 3's pinned code
mise run models   # fetches data/models/depth-anything-3-metric (1.3 GB)
```

`[models.hf.depth_anything]` is a new key in `config/defaults.toml`, so restart every config-reading service after
the pull. Until both commands have run, splats train as before, without the depth, and the log says which is
missing. Nothing is dropped.

## One application instead of a service per daemon

The system now runs as one application: `pack-app` (the window) or `pack-supervisor run` (no window) starts every
selected service as its own child process, and quitting it drains and stops them. Earlier installs ran each service
as a systemd user unit or a launchd agent. The application refuses to start while those run, so remove them once:

```sh
.venv/bin/python -m liberational.services migrate --dry-run   # prints every command and file; changes nothing
.venv/bin/python -m liberational.services migrate             # stop the units in order, and remove them
.venv/bin/pack-app                                            # open the application
```

What the migration does: stops the readers, then capture, waits up to 20 s for the tier worker to flush what capture
left, stops it, the broker, and last the dashboard and the gateway; disables the units, deletes their files (and any
`pack-<name>.service.d` override directories), and reloads the service manager. It knows the units by file name, so
it works on any later checkout, and running it again does nothing.

What it leaves alone: `pack.toml`, `config/devices.toml`, and everything under `data/`. The application runs from
the same checkout and reads the same files; nothing has to move. `PACK_ROOT` still points a command at a checkout
from anywhere.

What changes for you:

- **It runs while the application is open.** Nothing starts at boot or at login unless you turn on "Start at
  login" in Settings (off by default). Recording, the scheduled jobs, and the dashboard stop when you quit or log
  out. A machine that should record all day keeps the application open in a session that stays logged in.
- **Linger.** The dry run says whether linger is on for your user (`loginctl enable-linger` made the units start at
  boot). The application doesn't need it. `migrate --disable-linger` turns it off; leave it if other user services
  of yours want it.
- **`pack-services` talks to the application.** While it runs, `pack-services status`, `start`, `stop`, and
  `restart` (and `restart --cascade`, the staged restart of everything) are requests to it, and so are Settings'
  Services buttons. With the application closed they have nothing to act on.

To go back: check out the commit you were on before the pull and run `pack-services install` there, which writes the
units again.

## `pack-setup` sets up an installed package; a checkout is unchanged

`pack-setup` is new: the setup command for a package installed with no checkout (docs/SERVICES.md, "Install layout").
`mise run setup` is still how a checkout is set up, and nothing in a checkout behaves
differently: its services keep mise's shims on their `PATH`, `pack-system` lists the same packages, and the
`whisper` service is declared as before (its server is now named by its whole path, the same file). No setting was
added, nothing stored changes, and no service needs a restart.

`config/system-packages.toml` has a new field, `build = true`, for a package only a build from source needs, and
`services/services.toml` has `source_build = true`, for a service whose program a checkout builds. An installed
package leaves both kinds out.

## The model fetcher is a command: `pack-models`

`tools/fetch_models.py` moved into the package as `pack-models`, so an installed package has it with no checkout.
`mise run models` and `mise run setup` run it as before. Anything that ran the script by its path runs the module,
which works straight after the pull:

```sh
.venv/bin/python -m liberational.fetch_models --list
```

The `pack-models` command itself appears in `.venv/bin` the next time the environment is synced (`mise run models`
does it; so does the sync under "Build and restart matching components"). Nothing stored changes, and no service
needs a restart.

## Continuum's daemons are found in the environment, not the checkout

Nothing builds a path from the checkout to a program any more. `services/services.toml` names the daemons by
command (`continuum-broker`, not `continuum/target/release/continuum-broker`), and they are found where the `pack-*`
commands are: the bin directory of the environment the services run in. In a checkout that is `.venv/bin`, and
`mise run continuum-release` now links the daemons there after it builds them.

After the pull, and before anything starts or restarts a Continuum service, link the daemons already built (no
build, a second or two):

```sh
mise run continuum-link
```

Run it again whenever `.venv` is made from scratch. A daemon's `--state-dir` and `--manifest` arguments are now
whole paths (they were relative to the checkout); they name the same files, and nothing stored changes or is
dropped.

## The manifest import and its `[vision]` keys are gone

`continuum-capture --from-v1` is removed, and so are the settings it read: `vision.cams`, `vision.width`,
`vision.height`, `vision.fps`, and `vision.hold_fps`. Cameras come only from `config/devices.toml`
(`config/devices.example.toml` shows the format, and Settings' Devices panel writes it): with no manifest the
dashboard lists no cameras, where it used to fall back to `vision.cams`. Per-camera size, frame rate, and `hold_fps`
are the manifest's `[video]` keys.

If `pack.toml` still sets one of those five keys, delete the line. This lists them:

```sh
.venv/bin/pack-config --check
```

The Python SDK's in-place registration calls are renamed (`register_v1_index` to `register_audio_index`,
`register_v1_video_index` to `register_video_index`, `register_v1_flac` to `register_flac`, and `v1_gop_units` to
`gop_units`), so rebuild the SDK and the Rust binaries (`mise run continuum-release`, then the sync under "Build and
restart matching components") and restart every service after the pull. Stream names (`<stream>.audio_v1`,
`<stream>.video_v1`) and stored data are unchanged; nothing is dropped.

## Pet embeddings are DINOv2 (2026-11-09)

The pet embedding model is now Meta's DINOv2 ViT-S/14 (`facebook/dinov2-small`, Apache-2.0, ungated) in place of
DINOv3 ViT-S/16 and its DINOv3 License. `[models.hf.animal_embedding]` pins its revision and the checksum of its
weights, and names a new embedding space (`space = "facebook/dinov2-small"`). Recognition's thresholds are set for
it (`MATCH_SIM` 0.55, `GROUP_SIM` 0.65, `MIN_GALLERY_SIM` 0.60 in `people/pet_recognition.py`).

What it drops: every stored animal embedding and taught pet example (`pet.reference`) of the DINOv3 space stops being
read. They stay on disk, and nothing is deleted. Pet profiles and the owner's labels are untouched. Until a species
has examples in the new space, its pets aren't told apart by appearance.

The examples taught so far are vectors with no photo kept, so they can't be embedded again. Instead, carry the pets
over: the tool below takes the unlabeled sightings that still have a saved photo (`state/pet_shots`), names each by
the old examples (embedding the photo with DINOv3, which must still be in the Hugging Face cache), and gives the ones
it is sure of (0.75 or more alike, 0.10 clear of any other pet, where recognition itself asked 0.65 and 0.05) to that
pet as new DINOv2 examples. In order:

```sh
# 1. After the pull, before restarting anything: fetch DINOv2 (88 MB; its checksum is verified).
.venv/bin/python -m liberational.fetch_models --no-ollama

# 2. See what the carry-over would do (writes nothing), then do it. It takes a few minutes on the CPU.
.venv/bin/python tools/restore_pet_references.py --carry-from facebook/dinov3-vits16-pretrain-lvd1689m:tight \
    --old-model facebook/dinov3-vits16-pretrain-lvd1689m --dry-run
.venv/bin/python tools/restore_pet_references.py --carry-from facebook/dinov3-vits16-pretrain-lvd1689m:tight \
    --old-model facebook/dinov3-vits16-pretrain-lvd1689m

# 3. Restart every config-reading service (the model entry changed), the dashboard and gateway included.
```

`--old-model` is the DINOv3 repository in the cache: `facebook/dinov3-vits16-pretrain-lvd1689m`, or
`computer-vision-ai-lab/dinov3-vits16-pretrain-lvd1689m` on an install that fetched the ungated copy. The tool prints
how many examples each pet got and whether each species' held-out check passes. A pet it lists with fewer than 6
needs teaching again on People's Pets page. The carried labels show there like any label (each records
`carried_from`), and a wrong one can be cleared.

From now on a label keeps its sighting's saved photo in the knowledge store (`pet.photo`), which a catalog reset and
`pack-knowledge export` keep. A later model change is then
`.venv/bin/python tools/restore_pet_references.py` with no arguments, which embeds every labeled sighting's kept
photo again.

DINOv3 is no longer in the model list, and nothing reads it after the carry-over: delete its download when you like
(`models--facebook--dinov3-vits16-pretrain-lvd1689m` or `models--computer-vision-ai-lab--dinov3-vits16-pretrain-lvd1689m`
under `~/.cache/huggingface/hub`). A `pack.toml` that sets `models.hf.animal_embedding.repo` to another model must
also set `space` to that model's name and drop or replace `sha256`.

## The gateway's stream listing comes from the catalog

The gateway no longer walks the chunk directory to list streams while the broker is up: the broker's listing carries
each stream's recorded extent, and the listing's `chunks` and `frames` counts are gone. The broker, the gateway, and
the dashboard's frontend have to be upgraded together, or the dashboard shows no camera as live. After the pull:
`mise run continuum-release`, `mise run frontend`, then `pack-services restart --cascade` (which restarts the broker
and its readers) and `pack-services restart continuum-gateway dashboard`, which the cascade leaves running. Nothing
is deleted and the catalog's format is unchanged.

## Video chunks are a minute long (catalog format 8)

A camera now writes one chunk file and one catalog row a minute, and the catalog stores each chunk's frame index
compactly (docs/CONTINUUM-STORAGE.md). The catalog file's format changed with it, and the broker refuses a
format 7 file, so this upgrade needs a catalog reset. It deletes the catalog, the recorded video and audio it indexes,
the worker state derived from it, and the segment store; what you taught the system (people, voices, faces, pets,
zones, and settings) stays. After the pull, from the live checkout:

```sh
mise run continuum-release
.venv/bin/pack-reset-catalog --keep-taught          # prints the plan
.venv/bin/pack-reset-catalog --keep-taught --yes    # stops the services, resets, and starts them again
```

Build first: the reset starts the broker and the tier worker again, and they have to be the new ones.

## Chunk locators are relative to the state directory (2026-11-09)

The catalog used to store each Tier 1 and Tier 2 chunk's file as an absolute path, so an install whose checkout (and
`data/`) moved could no longer read any older audio or video. New chunks are stored relative to the state directory
(`paths.state`, the one that holds `continuum/`), and every reader joins them to its own configured state directory.
A catalog reset (such as the one for format 8, above) starts with relative locators and needs nothing more. Otherwise
a one-off migration rewrites the locators already in the catalog. It takes the broker's lock and refuses to run while
the broker holds it, is safe to run again (a second run rewrites nothing), and commits in short batches. From the live
checkout, after the pull:

```sh
mise run continuum-release                     # builds continuum-relative-locators with the daemons
mise run continuum-py-ffmpeg                   # the Python SDK, whose recent-window reader takes the root now
uv run --no-sync pack-services stop continuum-broker
continuum/target/release/continuum-relative-locators --catalog data/state/continuum/catalog.redb
uv run --no-sync pack-services restart --cascade
uv run --no-sync pack-services restart continuum-gateway dashboard
```

Use your own `paths.state` in the `--catalog` path if it isn't `data/state`. The cascade starts the broker again and
restarts every service that reads chunks with the new code (a reader still on the old code would read a relative
locator against its working directory); it leaves the gateway and the dashboard running, hence the last line. The
migration prints how many locator rows it looked at and how many it rewrote. A locator that names a file outside any
state directory (a v1 recording registered in place) keeps its absolute path. A link from an old checkout's `data/`
to the new one is no longer needed once this has run. A chunk the catalog names but whose file is gone now reads as
missing (one error per file in the reader's log, a 404 from `/clip`), not as a stale locator.

## Questions are read by the local model

Ask and the voice loop no longer read a question's time, people, and kind with patterns. The local model reads each
question once into a structured query (`memory/question.py`, #60), so every history question needs the model that
`recall.model` names (empty means `ask.model`, else `models.ollama.llm`), also with `recall.enabled` off. When that
model doesn't answer, Ask says it couldn't read the question, and a spoken question is answered as a general one.
There is nothing to migrate and no data to reset. After pulling, restart the dashboard, the voice loop, and research
(`pack-services restart dashboard voice research`, or the names your install uses), since each holds the old parser
in memory.

`pack-camera-history` and the voice loop's camera answer now say only what the cameras show now. A camera question
about the past ("what did camera 2 see in the last hour") is answered by Ask, from the same captions, so ask it in
Ask or `pack-ask`.

## Segment parts close as one batch

A segment part used to keep one record batch for every append for good, so a stream appended a row at a time (the
coverage streams) cost about 770 bytes a row on disk and seconds to read a day of. Writers now rewrite a part as one
batch when it closes, and its name ends `-v1`. This needs the Rust binaries and the Python SDK rebuilt
(`mise run continuum-release` and `mise run continuum-py`, the latter with the FFmpeg decoder where the install has
it) and every service restarted, since each one reads or writes the segment store.

Parts closed before the upgrade keep the old layout and still read correctly, only slower. To rewrite them once, after
the restart, from the live checkout (safe while the services run; it touches closed parts only, deletes nothing, and
prints how many it rewrote):

```sh
.venv/bin/python -c "import continuum_py; from liberational import segment_log; print(continuum_py.compact_segments(segment_log.root()))"
```

## Positions are placed when written

The Places page reads where people and pets were from the position layer (`position.placed`), which the nightly
`pack-spatial-cameras` run fills from the raw footprints. After this upgrade the layer is empty, so the page shows no
paths until that run, whose first pass places the last 30 days (about 20 s for each day of a busy place). To fill it
at once, from the live checkout:

```sh
.venv/bin/python tools/place_positions.py
```

Restart the dashboard after the pull. Nothing is deleted: the raw footprints stay where they were.

## Remote dashboard cameras

A dashboard opened over HTTPS cannot open the gateway's plain WebSocket on port 8766. The page uses `wss` on port
8767, a TLS proxy in front of the gateway. The gateway itself listens on `0.0.0.0:8766` and does not start
WebTransport. On a machine that serves the dashboard through Tailscale, for the tailnet only:

```sh
tailscale serve --bg --https=8767 http://127.0.0.1:8766
```

That proxy stays on the tailnet. The gateway asks for no password, so a public Funnel would publish the cameras.
In `pack.toml`, name the MagicDNS host the gateway should accept (comma-separated; empty means this machine's own
names only):

```toml
[dashboard]
gateway_allow_host = "<machine>.<tailnet>.ts.net"
```

Then `uv run --no-sync pack-services install continuum-gateway`, so the unit's `--allow-host` picks up the setting.
The process does not read `pack.toml` itself. Cameras then show for a dashboard opened on the tailnet. From outside
the tailnet they stay unavailable.

## Cameras are views: mounts and rigs in devices.toml

A camera is an angle on whatever holds it, not a room (#5). Its fixed-mount settings moved from `pack.toml` onto its
own entry in `config/devices.toml`, as `mount`, and a camera on a moving rig now says where it sits on the rig
instead (spatial/rig.py; `config/devices.example.toml` shows both). The old settings are removed, so `pack-config
--check` names them and a service that reads them fails until they're moved:

- `[spatial] camera_heights_m = { cam0 = 1.5 }` becomes `mount = { height_m = 1.5 }` on `cam0`'s `[[device]]`.
- `[spatial] outdoor_cameras = ["patio"]` becomes `mount = { outdoor = true }` on `patio`'s `[[device]]` (one table
  with both keys when a camera has both).
- `[vision] places = ["cam0: front camera"]` is renamed `[vision] camera_names`, same format. Name the view ("front
  camera"), not the room it looks at.
- A camera on a wearable or moving rig: declare the rig once (`[[rig]]` with a `name`) and give each of its cameras
  `mount = { rig = "<name>", at = [x, y, z], turn = [yaw, pitch, roll], hfov_deg = <field of view> }`. A rig camera
  is never one of the fixed-camera places (`pack-spatial-cameras`), with or without a `fisheye` calibration.

Delete the old keys from `pack.toml` after moving them. Capture refuses a `mount` key it doesn't know and a
`mount.rig` with no `[[rig]]`, so restart `continuum-capture` once the manifest is edited, and then every
config-reading service, dashboard included (the renamed `vision.camera_names` key). No machine-made data changes.

## Pet embeddings are cut tight

Animal embeddings now crop the detector's box itself instead of a square with context, a new embedding space
(`<repo>:tight` in each row's and `pet.reference`'s `model`). References of the old space are ignored, so pet
recognition is off until the owner's labels get new ones. After pulling, restart `see-actions`, `see-captions`, and the
dashboard, then run once:

```sh
uv run python tools/restore_pet_references.py --dry-run   # the labels with no current-space example
uv run python tools/restore_pet_references.py
```

It finds each labeled track's animal in its saved photo (`state/pet_shots`, kept 30 days) and keeps a tight-crop
reference; a photo with no animal found leaves its label without one. Square-crop embeddings of unlabeled sightings
age out with `pets.label_days`.

## The shared Ollama is a pack service

The Ollama at `ollama.url` now runs as the `ollama` pack service, which waits for the NVIDIA driver before it starts.
On Linux, stop and disable the system one so the two don't fight over port 11434, keep its model store, and install:

```sh
sudo systemctl disable --now ollama
```

In `pack.toml`, under `[ollama]`, set `models = "/usr/share/ollama/.ollama/models"` (the system Ollama's store) and,
to pin it, `gpu = "<UUID from nvidia-smi -L>"`, then `uv run --no-sync pack-services install`. The store's files are
owned by the `ollama` user and readable by everyone, which is enough to serve them; pulling new models into it needs
write access (`sudo chown -R $USER /usr/share/ollama/.ollama/models`). Every config-reading service needs a restart
after the pull (new `ollama.gpu` and `ollama.models` keys). On macOS, the pack service replaces the Ollama app or
Homebrew's Ollama too; see the next part.

The service's server tuning is now settings under `[ollama]` (2026-11-09): `keep_alive` (`OLLAMA_KEEP_ALIVE`,
default Ollama's own), `flash_attention` (on by default), `kv_cache_type` (default `f16`), and `max_loaded_models`
(default 0, Ollama's own). `config/defaults.toml` says why each default is what it is. These are new keys, so every
config-reading service needs a restart after the pull: quit the application and open it again.

### A Mac that ran Homebrew's Ollama

A Mac whose Ollama ran as Homebrew's launchd service (`brew services start ollama`) moves it to the pack's `ollama`
service, keeping its tuning and its model store:

1. Note the tuning and the store the old server used. Homebrew's service sets its variables in its plist
   (`cat "$(brew --prefix)/opt/ollama/homebrew.mxcl.ollama.plist"`, under `EnvironmentVariables`), and some were set
   with `launchctl setenv` instead (`launchctl getenv OLLAMA_KEEP_ALIVE`, and the same for `OLLAMA_FLASH_ATTENTION`,
   `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_MAX_LOADED_MODELS`, and `OLLAMA_MODELS`).
2. Stop it and keep it from starting at login: `brew services stop ollama`. Quit the Ollama app too if it is running,
   and turn off its own "launch at login". Leave the formula installed: the pack service runs that `ollama` binary.
   Check nothing still listens with `lsof -iTCP:11434 -sTCP:LISTEN` (no output).
3. In `pack.toml`, under `[ollama]`, put the tuning the defaults don't already give. For a server that ran with
   `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_FLASH_ATTENTION=1`, and `OLLAMA_KV_CACHE_TYPE=q8_0`:

   ```toml
   [ollama]
   keep_alive = "-1"      # a large model stays loaded until Ollama stops
   kv_cache_type = "q8_0"
   # flash_attention = true is already the default
   ```

   Keep the model store: if step 1 showed an `OLLAMA_MODELS`, set `models` to that path; with none, the old server
   used `~/.ollama/models`, which is the service's default as well, so leave `models` unset. If the old server set
   `OLLAMA_MAX_LOADED_MODELS`, set `max_loaded_models` to the same number.
4. Turn the pack's service on: it is on by default, so remove `"ollama"` from `[services] disable` in `pack.toml` if
   it is there. Then quit the application and open it again (`pack-app`), which also restarts every service for the
   new keys.
5. Check it: `.venv/bin/pack-services status` lists `ollama` as running, and `curl -s http://127.0.0.1:11434/api/tags`
   lists the models the old server had. The first answers after the switch load their models, so they are slow once.

To go back: quit the application, set `[services] disable = ["ollama"]`, and `brew services start ollama`.

## Prepare the upgrade

Use the machine's main checkout for deployment, never a validation worktree.
Review local changes and the intended commit before pulling. Keep the machine's
ignored `pack.toml`, `config/devices.toml`, and credentials. Device names identify
streams, so do not replace an existing manifest with an example or rename devices
as part of an unrelated update.

Inspect the selected and installed services before changing dependencies:

```sh
git status --short
uv run --no-sync pack-services list
uv run --no-sync pack-services status
```

Use `--no-sync` so inventory does not change the environment. Stop affected readers and writers in the coordinated
deployment window before updating their dependencies or changing model spaces.
Preserve intentionally disabled services.

After updating the code, check local settings against the current declarations:

```sh
uv run --no-sync pack-config --check
```

Remove obsolete keys and service names it reports. Compare model overrides with
`config/defaults.toml` separately: the checker permits custom entries under
`models`. Update a model entry's URL, path, revision, and checksum together.

## Build and restart matching components

Keep the extras the installation uses when synchronizing Python dependencies.
A bare `uv sync` removes the optional extras (torch, transformers, and onnx),
and the vision and listen services fail on their next start. `uv sync --inexact`
retains other installed packages. Name the extras `mise run setup` installs:
`faces`, `segmentation`, `deep-asr`, and `voice-clone` when `nvidia-smi` is on
PATH. Python sync can compile the path-dependent Continuum SDK, so run it
through the shared Rust slot:

```sh
# No NVIDIA GPU:
tools/build-slot.sh uv sync --inexact --extra faces --extra segmentation --extra deep-asr
# nvidia-smi on PATH, as mise run setup does:
tools/build-slot.sh uv sync --inexact --extra faces --extra segmentation --extra deep-asr --extra voice-clone
```

Install the Rust toolchain with rustup first. `mise install` supplies `protoc`
and the other declared tools; [CONTRIBUTING.md](CONTRIBUTING.md) covers setup. Build changed native
services with `mise run continuum-release`. For video decoding, use
`mise run continuum-py-ffmpeg` when the installed SDK needs rebuilding with FFmpeg
support. Fetch changed model artifacts with `mise run models`. Build changed
frontend code with `mise run frontend` before serving it.

Refresh changed service definitions with `pack-services install <service>` and
restart affected services with `pack-services restart <service>`. A broker or
capture restart uses the managed cascade. For a coordinated full refresh,
`pack-services install --cascade` renders and starts the reviewed selected set,
including selected services that were stopped. Run these commands through
`uv run --no-sync` once the environment is ready.

Restart every installed config-reading service after adding or removing defaults,
including delayed consumers and the dashboard. A running process with the old
schema can reject newly saved settings even when that feature is disabled.
Update native readers and their SDK together when a protocol or decoding change
requires them; a Python source update alone does not replace loaded native code.

## Model-space changes

The model manifest is the source of truth for current artifacts. These cutovers
cannot reuse their predecessors' embeddings:

| Area | Current space | Incompatible predecessor |
|---|---|---|
| Faces | AdaFace IR50 WebFace4M RGB | ArcFace/buffalo_l recognition vectors |
| Voices | ReDimNet2 B3 VoxCeleb2 large-margin, 192 values | CAM++ vectors |
| Pets | DINOv2 ViT-S/14 (`facebook/dinov2-small`) on tight box crops, 384 values | DINOv3 vectors, and any model's on square crops, despite the same dimensions |
| Journal retrieval | Qwen3-Embedding 0.6B, 1024 values | Qwen3-Embedding 4B index and embeddings |

Preserve owner names, profiles, photos, reference media, zones, and Objects while
retiring incompatible generated vectors and their old identity bindings. Rebuild
references from retained media or teach fresh observations. Keep voice and unknown
face allocation high-water marks so a new ID cannot inherit retained clone media,
photos, or an old label. Clear obsolete bindings across their revision history;
removing only the newest record can expose an older claim.

A model cutover needs a scoped deletion plan. A catalog reset alone does not
remove vectors and references held outside its deletion list. Conversely, deleting
only the vector index does not reset broker subscription cursors: already
acknowledged entries can be skipped. Regenerate journal entries, embeddings, and
indexes in the new space with matching subscription state. Do not restore an old
vector-bearing teaching bundle wholesale afterward.

AdaFace requires a local ONNX export after its declared source and checkpoint are
fetched; follow `uv run tools/export_adaface.py --help` and use the configured model paths.
Review the checkpoint's dataset-license conditions; the operator reads and accepts a
model's terms, and an agent does not accept them for them.

Deep transcription now uses Qwen3-ASR in the scheduled process and writes
`transcript_revisions_qwen.sqlite3`; the old `transcript_revisions.sqlite3` is not
read. Source utterances and owner corrections remain. The optional Whisper service
is still relevant when `asr.engine = "whisper"` selects it.

## Preserve owner inputs and inspect reset scope

A routine upgrade does not require a reset. When a reset is part of the change,
preserve owner inputs explicitly: taught knowledge, names, photos, clone media,
original person and voice references, settings, and hand-made measurements.
`paths.voices`, `paths.measurement`, and model weights are not disposable history.

`pack-knowledge export --source sqlite --output <bundle.zip>` and
`pack-knowledge verify --input <bundle.zip>` create and check a portable knowledge
bundle. Use an output outside any planned deletion paths. This is an allowlisted
export, not a backup of every file under `data/`: preserve measurements, credentials,
and any other required files separately. Import with
`pack-knowledge import --input <bundle.zip>` only when compatible with the target
model spaces. Import restores knowledge and managed attachments, then puts the
taught files (face gallery, voiceprints, clones, person references, the owner's
voice sample) back where they live when nothing is there yet. It never overwrites
a file, names any it left alone, and leaves configuration TOML in the bundle.
docs/PEOPLE.md lists what the bundle carries and what it leaves out.

Preview the actual reset plan without changing anything:

```sh
uv run --no-sync pack-reset-catalog --keep-taught --camera "<configured-camera>"
```

`--keep-taught` deletes the catalog, its recorded audio/video, segment observations,
and listed derived state. It leaves the knowledge store and taught files alone;
it neither converts embeddings nor clears every generated cache. `--fresh` also
deletes taught stores, galleries, voice clones, owner voice files, and matching
teaching bundles, and truncates logs. It is not the normal history-reset mode.
Use the printed deletion list and `src/liberational/reset_catalog.py` as the exact
scope; adding `--yes` executes it.

The reset needs an installed broker and a valid camera manifest. It enables H.264
for the selected camera if that camera captures MJPEG. It restarts the broker and
services that were running, so arrange the stopped set before executing it. A
failure during deletion leaves services stopped; a restart failure can leave only
part of the set running. Follow its recovery report before starting more readers.

## Verify the result

Check service status, model loads, and fresh capture and transcript timestamps.
Confirm newly taught faces and voices with new observations, and check journal
search against fresh evidence. Existing roster photos do not prove recognition
works. A scheduled worker needs an armed timer, not a permanent process.

Compare retained owner inputs before and after any reset. Record the deployed
commit, model selections, checks, and remaining gaps in the deployment issue.
While history rebuilds, report incomplete or unavailable coverage rather than
zero activity.
