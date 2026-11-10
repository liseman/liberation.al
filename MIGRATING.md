# Upgrading an existing install

This guide covers the current Continuum installation. Fresh setup is in
[CONTRIBUTING.md](CONTRIBUTING.md); components and state paths are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The old genericize and v1 import
procedures are retired. Do not run an old catch-up script against current data.

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

## Cameras are views: mounts and rigs in devices.toml (2026-11-09)

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

## Pet embeddings are cut tight (2026-11-08)

Animal embeddings now crop the detector's box itself instead of a square with context (#3024), a new embedding space
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

## The shared Ollama is a pack service (2026-11-08)

The Ollama at `ollama.url` now runs as the `ollama` pack service, which waits for the NVIDIA driver before it starts.
On Linux, stop and disable the system one so the two don't fight over port 11434, keep its model store, and install:

```sh
sudo systemctl disable --now ollama
```

In `pack.toml`, under `[ollama]`, set `models = "/usr/share/ollama/.ollama/models"` (the system Ollama's store) and,
to pin it, `gpu = "<UUID from nvidia-smi -L>"`, then `uv run --no-sync pack-services install`. The store's files are
owned by the `ollama` user and readable by everyone, which is enough to serve them; pulling new models into it needs
write access (`sudo chown -R $USER /usr/share/ollama/.ollama/models`). Every config-reading service needs a restart
after the pull (new `ollama.gpu` and `ollama.models` keys). On macOS, quit the Ollama app instead.

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
| Pets | DINOv3 ViT-S/16 on tight box crops, 384 values | DINOv2 vectors, and DINOv3 on square crops, despite the same dimensions |
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
Review the checkpoint's dataset-license conditions. DINOv3 requires the operator's
acceptance of its model terms and authenticated access before fetching; an agent
does not accept those terms for them.

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
docs/ARCHITECTURE.md lists what the bundle carries and what it leaves out.

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
