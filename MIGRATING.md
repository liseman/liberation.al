# Upgrading an existing install

This guide shows how to update a checkout of this repository that already runs: the steps every upgrade takes, the
extra steps some changes need, and how to reset state. It covers a checkout installed from the first public commit
(2026-11-09) or later. For anything older, export what you taught (`pack-knowledge export`), set up fresh from the
[README](README.md), and import it.

Components and state paths are in [docs/COMPONENTS.md](docs/COMPONENTS.md).

## Every upgrade

Deploy from the machine's main checkout, never a worktree. Keep your `pack.toml`, `config/devices.toml`, and
credentials. Device names identify streams, so do not rename devices as part of an unrelated update.

1. **Look before pulling.**

   ```sh
   git status --short
   uv run --no-sync pack-services status
   ```

2. **Pull, then check your settings.** Remove any key or service name the check reports.

   ```sh
   git pull
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

5. **Restart.** Quit the application and open it again (`pack-app`), or restart the services in stages:

   ```sh
   uv run --no-sync pack-services restart --cascade
   ```

   With no service named, the cascade restarts every selected service: the dashboard and the gateway stay up while
   the broker, capture, and the readers restart, and are restarted last. That also covers a key added to or removed
   from `config/defaults.toml`. [docs/RUNNING.md](docs/RUNNING.md) has the other ways to restart.

6. **Verify.**

   ```sh
   mise run continuum-status
   uv run --no-sync pack-services status
   ```

   Check that capture and transcript timestamps are fresh. A scheduled job reads `scheduled` under the application
   (an active timer on an install with a unit per service), not `running`.

## Changes that need a step

Newest first. Do each one your install has not had yet.

### A place's plan is a scene, and the line DSL is no longer read

A place's plan is stored as a Pascal Editor scene and edited in the plan editor on the Places page (Edit plan). The
vendored floorplan library, the text plan editor, `tools/import_owner_plan.py`, and every reader and writer of the
old line DSL are removed, with `pack-plan-convert`.

An install with an owner plan saved before this change converts it first, at the last commit that has the
converter. Skip this if the Places page already shows a Versions button in the plan editor and every version opens.

1. **Convert at the commit before the removal.** The first command changes nothing: it prints what each plan holds
   before and after, and names a page that draws both to look over. The second converts the plan that holds and
   every earlier version; the old versions stay in the store untouched.

   ```sh
   git checkout add5028c
   .venv/bin/python -m liberational.spatial.plan_convert
   .venv/bin/python -m liberational.spatial.plan_convert --yes
   git checkout main
   ```

2. **Upgrade as usual** ("Every upgrade"), then rebuild the dashboard (`mise run frontend`) and restart it.

A plan left in the old format isn't lost, but it is no longer shown: the Places page says so, and Edit plan starts
a new one. A machine-made plan in the old format is not converted: `pack-reset-places` clears it, or drawing the
plan in the editor replaces it.

### Positions are placed every hour by `pack-positions`

The `positions` service runs `pack-positions` every hour at 20 past. It replaces `tools/place_positions.py`.

1. **Install the command.** Sync the checkout's environment, keeping your extras ("Every upgrade", step 3).

2. **Add the service.** Under the application, quitting it and opening it again does the same.

   ```sh
   uv run --no-sync pack-services install positions
   ```

3. **Bring the positions up to date once**, so the first hourly run is short. The first command prints what a run
   would place for each camera and changes nothing. The second records up to a week of footprints for cameras never
   recorded, and places them.

   ```sh
   uv run --no-sync pack-positions --dry-run
   uv run --no-sync pack-positions --days 7
   ```

### Spatial memory starts again from what the cameras see

A place's map is learned from what the cameras see, by the `surroundings` service, while the Live tab's "Learn
surroundings" switch is on ([docs/SPATIAL.md](docs/SPATIAL.md#learning-surroundings)). `pack-positions` places
positions (above), and `pack-spatial-splat` trains a place's splat by hand.

Removed: phone walks (`pack-walkthrough`, and the Places page's "Import a walk"), the nightly fixed-camera pass
(`pack-spatial-cameras`, the `spatial-cameras` job), and the tools built on them (`tools/rebuild_place.py`,
`restore_owner_plan.py`, `fit_intrinsics.py`, `camera_poses.py`, `splat_walk_check.py`, and
`walkthrough_closure.py`).

1. **Remove the scheduled job.**

   If the install has a unit per service, remove the job's timer before the pull:

   ```sh
   uv run --no-sync pack-services uninstall spatial-cameras
   ```

   If it runs under the application, pull first (step 2). Then quit the application and open it again, or have it
   read the service list again without a restart, and check that the job is not listed:

   ```sh
   uv run --no-sync pack-supervisor reload
   uv run --no-sync pack-services status
   ```

2. **Pull, and remove what `pack-config --check` reports** from `pack.toml`: `spatial-cameras` under `[services]`,
   and the removed keys `spatial.cameras_enabled`, `spatial.splats_enabled`, `spatial.geocalib_runtime`,
   `spatial.cut3r_runtime`, `[models.files.geocalib]`, and `[models.files.cut3r]`.

3. **Sync, build the dashboard, and restart everything**, as in "Every upgrade". Keys were removed from
   `config/defaults.toml`, so every config-reading service restarts, the dashboard included.

4. **Reset the places.** The first command only prints what would go and what would stay.

   ```sh
   uv run --no-sync pack-reset-places
   uv run --no-sync pack-reset-places --yes
   uv run --no-sync pack-services restart dashboard journal
   ```

   The maps, splats, walks, camera registrations, and stored positions that the removed jobs made go. Your
   corrected plans, place names, and held camera intrinsics stay: they are in the knowledge store. A place whose
   plan you corrected stays on the Places page with that plan, without a 3D view, and nobody is placed on it until
   it is mapped again. Other places are listed again once they are mapped.

5. **Optionally, free the disk.** Nothing reads `data/cut3r/`, `data/geocalib/`, `data/models/cut3r/` (3.2 GB), or
   `data/models/geocalib/`.

### The Places 3D view draws with PlayCanvas

On a machine that makes splats, after the pull:

1. **Build the dashboard.**

   ```sh
   mise run frontend
   ```

2. **Give each kept splat its SOG copy.** Until this runs, a place shows its points, not its splat.

   ```sh
   .venv/bin/pack-spatial-splat
   ```

3. **Restart the dashboard.**

   ```sh
   uv run --no-sync pack-services restart dashboard
   ```

### Splats train against a depth model

On a machine that makes splats (CUDA):

1. **Rebuild `data/gsplat`** with Depth Anything 3's code.

   ```sh
   mise run gsplat
   ```

2. **Fetch the depth model** into `data/models/depth-anything-3-metric` (1.3 GB).

   ```sh
   mise run models
   ```

3. **Restart every config-reading service**: `[models.hf.depth_anything]` is a new key.

### One application instead of a service per daemon

The system runs as one application: `pack-app` (with a window) or `pack-supervisor run` (without). It refuses to
start while the systemd units or launchd agents of an earlier install run, so remove them once.

1. **Preview.** This prints every command and file, and changes nothing.

   ```sh
   .venv/bin/python -m liberational.services migrate --dry-run
   ```

2. **Remove the units.** This stops them in order and removes them. It leaves `pack.toml`, `config/devices.toml`,
   and everything under `data/` alone.

   ```sh
   .venv/bin/python -m liberational.services migrate
   ```

   If nothing else of yours needs systemd linger, add `--disable-linger` to turn it off.

3. **Open the application.**

   ```sh
   .venv/bin/pack-app
   ```

After this:

- The system runs only while the application is open. Nothing starts at login unless you tick "Start at login" in
  the application's window (the `app.start_at_login` setting).
- `pack-services status`, `start`, `stop`, and `restart` are requests to the running application.

[docs/SERVICES.md](docs/SERVICES.md#one-application) explains the application.

### Continuum's daemons are found in the environment

`services/services.toml` names the daemons by command, found in `.venv/bin`. Before starting any Continuum service,
link the daemons already built:

```sh
mise run continuum-link
```

Run it again whenever `.venv` is made from scratch.

### Cameras come only from the device manifest

`vision.cams`, `vision.width`, `vision.height`, `vision.fps`, and `vision.hold_fps` are removed. Cameras come from
`config/devices.toml`, and per-camera size and rate are its `[video]` keys.

1. **Delete the removed keys** from `pack.toml`. `uv run --no-sync pack-config --check` lists them.
2. **Rebuild Continuum and the SDK** ("Every upgrade", step 4).
3. **Restart every service.**

### Pet embeddings are DINOv2

Stored animal embeddings and taught pet examples from the DINOv3 model are not read. Pet profiles and labels are
untouched.

1. **Carry the examples over, before restarting anything.** The first run of the tool is a dry run.

   ```sh
   .venv/bin/python -m liberational.fetch_models --no-ollama
   .venv/bin/python tools/restore_pet_references.py --carry-from facebook/dinov3-vits16-pretrain-lvd1689m:tight \
       --old-model facebook/dinov3-vits16-pretrain-lvd1689m --dry-run
   .venv/bin/python tools/restore_pet_references.py --carry-from facebook/dinov3-vits16-pretrain-lvd1689m:tight \
       --old-model facebook/dinov3-vits16-pretrain-lvd1689m
   ```

2. **Restart every config-reading service.**

3. **Teach again where needed.** The tool prints how many examples each pet got. A pet with fewer than 6 needs
   teaching again in Known, Pets.

Labels keep their photo, so a later model change needs only `tools/restore_pet_references.py` with no arguments.

### The gateway lists streams from the catalog

Upgrade the broker, the gateway, and the frontend together, or the dashboard shows no camera as live:

```sh
mise run continuum-release
mise run frontend
uv run --no-sync pack-services restart --cascade
```

### Catalog format 8: a reset

A camera writes one chunk file a minute, and the broker refuses an older catalog. This upgrade resets
recorded history. What you taught (people, voices, faces, pets, zones, and settings) stays.

1. **Build first**: the reset starts the broker and the tier worker.

   ```sh
   mise run continuum-release
   ```

2. **Preview the reset.** This prints the plan.

   ```sh
   .venv/bin/pack-reset-catalog --keep-taught
   ```

3. **Reset.** This stops the services, resets, and starts them again.

   ```sh
   .venv/bin/pack-reset-catalog --keep-taught --yes
   ```

### Camera mounts moved to the device manifest

A camera is a view, not a room.

1. **Move these** from `pack.toml` to the camera's `[[device]]` in `config/devices.toml`
   (`config/devices.example.toml` shows the format), and delete the old keys:

   | Old, in `pack.toml` | New |
   |---|---|
   | `[spatial] camera_heights_m = { cam0 = 1.5 }` | `mount = { height_m = 1.5 }` on `cam0` |
   | `[spatial] outdoor_cameras = ["patio"]` | `mount = { outdoor = true }` on `patio` |
   | `[vision] places` | `[vision] camera_names`, same format |

2. **Restart `continuum-capture`**, then every config-reading service.

### The shared Ollama is a pack service

The Ollama at `ollama.url` runs as the `ollama` service.

1. **Stop any other Ollama on port 11434.**

   - **Linux:** `sudo systemctl disable --now ollama`. To keep its model store, set
     `models = "/usr/share/ollama/.ollama/models"` under `[ollama]` in `pack.toml`.
   - **macOS:** `brew services stop ollama`, and quit the Ollama app and turn off its launch at login. Leave the
     formula installed: the service runs that binary.

2. **Carry over your server tuning.** The settings are under `[ollama]` in
   [config/defaults.toml](config/defaults.toml). Carry over any `OLLAMA_*` variables the other server set.

3. **Restart the application**, and check that this lists your models:

   ```sh
   curl -s http://127.0.0.1:11434/api/tags
   ```

## Model changes

Embeddings from different models cannot be compared, even at the same size. After one of these changes, the vectors
of the replaced model are ignored, and references are rebuilt from kept media or taught again.

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

3. **Reset**, with the same mode and `--yes`:

   ```sh
   uv run --no-sync pack-reset-catalog --keep-taught --yes
   ```

| Mode | Deletes | Keeps |
|---|---|---|
| `--keep-taught` | The catalog, recordings, the segment store, the vector index, and derived state | The knowledge store, the face gallery, voice references, and models |
| `--fresh` | All of the above and everything taught | Models |

`src/liberational/reset_catalog.py` lists every step of a reset. While history rebuilds after one, readers report
unavailable coverage, not zero activity.

To restore a bundle, run `uv run --no-sync pack-knowledge import --input <bundle.zip>`. It never overwrites a file,
and names any it left alone.

To reset spatial memory alone, run `uv run --no-sync pack-reset-places --yes` (without `--yes` it only prints the
plan). It deletes every place's maps, splats, camera registrations, and stored positions, and keeps the knowledge
store, so your corrected plans and place names stay. It stops no service: restart the dashboard and the journal
afterwards, as in "Spatial memory starts again from what the cameras see", step 4.
