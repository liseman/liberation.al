# Upgrading an existing install

This guide covers a checkout installed from this repository's first public commit (2026-11-09) or later. For
anything older, export what you taught (`pack-knowledge export`), set up fresh from [README.md](README.md), and
import it.

Fresh setup is in the [README](README.md). Components and state paths are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Every upgrade

Deploy from the machine's main checkout, never a worktree. Keep your `pack.toml`, `config/devices.toml`, and
credentials. Device names identify streams, so do not rename devices as part of an unrelated update.

1. **Look before pulling.**

   ```sh
   git status --short
   uv run --no-sync pack-services status
   ```

2. **Pull, then check your settings.** Remove any key or service name this reports:

   ```sh
   uv run --no-sync pack-config --check
   ```

3. **Sync Python, keeping your extras.** A bare `uv sync` removes the optional extras (torch, transformers, and
   onnx), and the vision and listen services fail on their next start.

   ```sh
   # No NVIDIA GPU:
   tools/build-slot.sh uv sync --inexact --extra faces --extra segmentation --extra deep-asr
   # nvidia-smi on PATH, as mise run setup does:
   tools/build-slot.sh uv sync --inexact --extra faces --extra segmentation --extra deep-asr --extra voice-clone
   ```

4. **Rebuild what changed.**

   | Changed | Run |
   |---|---|
   | Anything under `continuum/` | `mise run continuum-release` |
   | The SDK, on a machine with network cameras | `mise run continuum-py-ffmpeg` |
   | Models in `config/defaults.toml` | `mise run models` |
   | Anything under `frontend/` | `mise run frontend` |

5. **Restart.** Quit the application and open it again (`pack-app`), or `pack-services restart --cascade`. The
   cascade leaves the dashboard and gateway running: restart those by name when they changed
   (`pack-services restart continuum-gateway dashboard`). A new or removed key in `config/defaults.toml` needs every
   config-reading service restarted, the dashboard included.

6. **Verify.** `mise run continuum-status`, then check that capture and transcript timestamps are fresh. A scheduled
   job needs an armed timer, not a running process.

## Changes that need a step

Newest first. Do each one your install has not had yet.

### Positions are placed every hour by `pack-positions`

`tools/place_positions.py` is gone: its work is the `pack-positions` command, which the new `positions` service runs
every hour at 20 past. It needs the command installed and the service added.

1. **Install the new command** in the checkout's environment, keeping the extras your install uses (a bare
   `uv sync` removes them; "Sync Python, keeping your extras" above has the command).
2. **Add the service.** `uv run --no-sync pack-services install positions`, or under the application, quit it and
   open it again.
3. **Bring the positions up to date once**, so the first hourly run is short: `pack-positions --days 7` records up to
   a week of footprints for cameras never recorded and places them. `pack-positions --dry-run` first prints what
   that would place per camera and changes nothing.

### Spatial memory starts again from what the cameras see

Phone walks (`pack-walkthrough`, the Places page's "Import a walk") and the nightly fixed-camera pass
(`pack-spatial-cameras`, the `spatial-cameras` job at 01:00) are gone, with the tools built on them
(`tools/rebuild_place.py`, `restore_owner_plan.py`, `fit_intrinsics.py`, `camera_poses.py`, `splat_walk_check.py`, and
`walkthrough_closure.py`). A place's map is made by a new job that learns it from what the cameras see. Until that
job runs, nothing trains splats on a schedule (`pack-spatial-splat` still runs by hand); positions are placed by
`pack-positions` (above).

1. **Remove the scheduled job.** An install that still renders a unit per service removes its timer, before the
   pull:

   ```sh
   uv run --no-sync pack-services uninstall spatial-cameras
   ```

   Under the application, the job goes once the pull is in: quit the application and open it again, or have it
   read the service list again without a restart, and check the job is no longer listed:

   ```sh
   uv run --no-sync pack-supervisor reload
   uv run --no-sync pack-services status
   ```

2. **Pull, and remove what `pack-config --check` reports** from `pack.toml`: `spatial-cameras` under `[services]`,
   and the removed keys `spatial.cameras_enabled`, `spatial.splats_enabled`, `spatial.geocalib_runtime`,
   `spatial.cut3r_runtime`, `[models.files.geocalib]`, and `[models.files.cut3r]`.

3. **Sync, build the dashboard, and restart everything**, as in "Every upgrade". Keys left `config/defaults.toml`,
   so every config-reading service restarts, the dashboard included.

4. **Reset the places.** The maps, splats, walks, camera registrations, and stored positions made by the removed
   jobs go; your corrected plans, place names, and held camera intrinsics stay (they are in the knowledge store).
   The first command only prints what would go and what would stay:

   ```sh
   uv run --no-sync pack-reset-places
   uv run --no-sync pack-reset-places --yes
   uv run --no-sync pack-services restart dashboard journal
   ```

   A place whose plan you corrected stays on the Places page with that plan, without a 3D view, and nobody is
   placed on it until it is mapped again. Other places are listed again once they are mapped.

5. **Optionally free the disk.** Nothing reads these any more: `data/cut3r/`, `data/geocalib/`,
   `data/models/cut3r/` (3.2 GB), and `data/models/geocalib/`.

### The Places 3D view draws with PlayCanvas

After the pull, on a machine that makes splats:

```sh
mise run frontend
.venv/bin/pack-spatial-splat      # gives each kept splat its SOG copy
```

Then restart the dashboard. Until the second command runs, a place shows its points, not its splat.

### Splats train against a depth model

On a machine that makes splats (CUDA):

```sh
mise run gsplat   # rebuilds data/gsplat with Depth Anything 3's code
mise run models   # fetches data/models/depth-anything-3-metric (1.3 GB)
```

Restart every config-reading service (`[models.hf.depth_anything]` is a new key).

### One application instead of a service per daemon

The system runs as one application: `pack-app` (with a window) or `pack-supervisor run` (without). It refuses to
start while the old systemd units or launchd agents run, so remove them once:

```sh
.venv/bin/python -m liberational.services migrate --dry-run   # prints every command and file
.venv/bin/python -m liberational.services migrate             # stops the units in order and removes them
.venv/bin/pack-app
```

The migration leaves `pack.toml`, `config/devices.toml`, and everything under `data/` alone.

- The system now runs only while the application is open. Nothing starts at login unless you turn on "Start at
  login" in Settings.
- `pack-services status`, `start`, `stop`, and `restart` are requests to the running application.
- `migrate --disable-linger` turns off systemd linger if nothing else of yours needs it.

### Continuum's daemons are found in the environment

`services/services.toml` names the daemons by command, found in `.venv/bin`. Before starting any Continuum service,
link the daemons already built:

```sh
mise run continuum-link
```

Run it again whenever `.venv` is made from scratch.

### Cameras come only from the device manifest

`vision.cams`, `vision.width`, `vision.height`, `vision.fps`, and `vision.hold_fps` are removed. Cameras come from
`config/devices.toml`, and per-camera size and rate are its `[video]` keys. Delete the old keys from `pack.toml`
(`pack-config --check` lists them), rebuild Continuum and the SDK, and restart every service.

### Pet embeddings are DINOv2

Stored animal embeddings and taught pet examples from the earlier model stop being read. Pet profiles and labels
are untouched. Carry the examples over before restarting anything:

```sh
.venv/bin/python -m liberational.fetch_models --no-ollama
.venv/bin/python tools/restore_pet_references.py --carry-from facebook/dinov3-vits16-pretrain-lvd1689m:tight \
    --old-model facebook/dinov3-vits16-pretrain-lvd1689m --dry-run
.venv/bin/python tools/restore_pet_references.py --carry-from facebook/dinov3-vits16-pretrain-lvd1689m:tight \
    --old-model facebook/dinov3-vits16-pretrain-lvd1689m
```

Then restart every config-reading service. The tool prints how many examples each pet got; a pet with fewer than 6
needs teaching again in Known, Pets. Labels now keep their photo, so a later model change needs only
`tools/restore_pet_references.py` with no arguments.

### The gateway lists streams from the catalog

Upgrade the broker, the gateway, and the frontend together, or the dashboard shows no camera as live:
`mise run continuum-release`, `mise run frontend`, `pack-services restart --cascade`, then
`pack-services restart continuum-gateway dashboard`.

### Catalog format 8: a reset

A camera writes one chunk file a minute, and the broker refuses an older catalog. This upgrade resets recorded
history. What you taught (people, voices, faces, pets, zones, and settings) stays.

```sh
mise run continuum-release
.venv/bin/pack-reset-catalog --keep-taught          # prints the plan
.venv/bin/pack-reset-catalog --keep-taught --yes    # stops the services, resets, and starts them again
```

Build first: the reset starts the new broker and tier worker.

### Camera mounts moved to the device manifest

A camera is a view, not a room. Move these from `pack.toml` to the camera's `[[device]]` in `config/devices.toml`
(`config/devices.example.toml` shows the format), then delete the old keys:

| Old, in `pack.toml` | New |
|---|---|
| `[spatial] camera_heights_m = { cam0 = 1.5 }` | `mount = { height_m = 1.5 }` on `cam0` |
| `[spatial] outdoor_cameras = ["patio"]` | `mount = { outdoor = true }` on `patio` |
| `[vision] places` | `[vision] camera_names`, same format |

Restart `continuum-capture`, then every config-reading service.

### The shared Ollama is a pack service

The Ollama at `ollama.url` runs as the `ollama` service. Stop any other Ollama on port 11434 first.

- **Linux:** `sudo systemctl disable --now ollama`. To keep its model store, set
  `models = "/usr/share/ollama/.ollama/models"` under `[ollama]` in `pack.toml`.
- **macOS:** `brew services stop ollama`, and quit the Ollama app and turn off its launch at login. Leave the
  formula installed: the service runs that binary.

Server tuning is settings under `[ollama]`: `keep_alive`, `flash_attention`, `kv_cache_type`, `max_loaded_models`,
and `gpu`. Carry over any `OLLAMA_*` variables your old server set. Then restart the application and check
`curl -s http://127.0.0.1:11434/api/tags` lists your models.

## Model changes

Embeddings from different models cannot be compared, even at the same size. After one of these changes, the old
vectors are ignored, and references are rebuilt from kept media or taught again.

| Area | Current model | Replaced |
|---|---|---|
| Faces | AdaFace IR50 WebFace4M RGB | ArcFace |
| Voices | ReDimNet2 B3, 192 values | CAM++ |
| Pets | DINOv2 ViT-S/14 on tight crops, 384 values | DINOv3, and square crops |
| Journal search | Qwen3-Embedding 0.6B, 1024 values | Qwen3-Embedding 4B |

AdaFace needs a local ONNX export after its files are fetched: see `uv run tools/export_adaface.py --help`. Read the
checkpoint's dataset license first; accepting a model's terms is the operator's decision.

## Resets and what they keep

A routine upgrade needs no reset. When one does:

1. **Export what you taught**, to a path outside anything the reset deletes, and check it:

   ```sh
   uv run --no-sync pack-knowledge export --source sqlite --output <bundle.zip>
   uv run --no-sync pack-knowledge verify --input <bundle.zip>
   ```

   The bundle is an allowlist, not a backup of `data/`: keep measurements (`paths.measurement`) and credentials
   yourself. [docs/PEOPLE.md](docs/PEOPLE.md#moving-taught-state) lists what it carries.

2. **Preview the reset.** Without `--yes` it only prints the plan:

   ```sh
   uv run --no-sync pack-reset-catalog --keep-taught
   ```

| Mode | Deletes | Keeps |
|---|---|---|
| `--keep-taught` | The catalog, recordings, the segment store, the vector index, and derived state | The knowledge store, the face gallery, voice references, and models |
| `--fresh` | All of the above and everything taught | Models |

`pack-reset-places` resets spatial memory alone: it deletes every place's maps, splats, camera registrations, and
stored positions, and keeps the knowledge store, so your corrected plans and place names stay. It stops no service.

`pack-knowledge import --input <bundle.zip>` restores a bundle. It never overwrites a file and names any it left
alone. `src/liberational/reset_catalog.py` lists every step of a reset.

While history rebuilds after a reset, readers report unavailable coverage, not zero activity.
