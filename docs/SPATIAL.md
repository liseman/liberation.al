# Spatial memory

How the pack builds a map and floorplan of each indoor place and puts people on it. [WALKTHROUGH.md](WALKTHROUGH.md)
has the steps to record a walk. Paths are relative to `src/liberational/`. Everything here is optional, and the
reconstruction jobs need CUDA.

## What a place is

A place is a chain of revisions under `state/places/<place>/` (`memory/places.py`). Each revision is written once:

| File | Contents |
|---|---|
| `<revision>.json` | Producer, source visits, identifying signals, the floor plane, the map-to-plan transform, and each registered fixed camera's pose |
| `<revision>.map` | The point cloud (binary PLY) |
| `<revision>.plan` | The floorplan DSL |
| `<revision>.gs.ply`, `<revision>.sog` | An optional Gaussian splat and its copy for the 3D view |

The newest two maps and splats are kept; plans and revision records stay. All of it is machine-made and a reset
deletes it. The owner's place names and plan corrections (`place.named`, `place.plan`) live in the knowledge store
and supersede the machine's view without changing it.

## Jobs

| Job | When | Does |
|---|---|---|
| `pack-spatial-cameras` (`spatial/cameras.py`) | Nightly at 01:00, or by hand | Takes each fixed camera's best lit midday frame, matches every pair with MASt3R, groups cameras that see the same room, and registers each group as a place revision with its cloud, floor, plan, and camera poses |
| `pack-walkthrough VIDEO\|STRAY_FOLDER` (`spatial/walkthrough.py`) | By hand, once per walk | Turns a phone walk into a revision. A Stray Scanner export uses ARKit's poses and LiDAR depth; a plain video uses CUT3R, with MASt3R closing revisits |
| `spatial/fuse.py` | After each job above | Levels the new revision, scales it by measured camera heights (`mount.height_m`), registers it against the place's fused map, and merges it. A LiDAR walk outranks camera snapshots |
| `pack-spatial-splat` (`spatial/splats.py`) | After the nightly pass when `spatial.splats_enabled` | Trains one Gaussian splat per revision with gsplat |
| `spatial/positions.py` placer | End of each nightly pass | Places people's and pets' foot points on the plan |

The dashboard's Known, Places tab uploads a walk (`dashboard/walk_import.py`), which runs `pack-walkthrough --match`
as one job at a time under `state/walk_imports/`.

`state/places.lock` keeps the nightly job and a walk import from writing places at once.

## Positions

`spatial/footprints.py` records raw foot points as cameras saw them (`position.sample`). The placer maps them to
plan coordinates and writes `position.placed` and `position.path`: who or what, place, room, coordinates, time,
confidence, and source. When a registration moves a camera's placement by more than 0.25 m, the placer places that
place's samples again into a new generation, so a path is always in one frame.

`memory/places.py:places_for_span` tells the journal which places and rooms a span touched.

## Hand tools

Each defaults to a dry run that prints what would change. Their docstrings have the options.

| Tool | Does |
|---|---|
| `tools/import_owner_plan.py` | Registers an owner-supplied floorplan into a place's map and saves it as the owner plan |
| `tools/fit_intrinsics.py` | Fits each fixed camera's focal and lens once and holds them (`camera.intrinsics`) |
| `tools/rebuild_place.py` | Rebuilds a place's map from its walks and fresh camera snapshots |
| `tools/place_positions.py` | Runs the placer by hand; `--reset` places every place again |
| `tools/calibrate_lens.py`, `tools/calibrate_fisheye.py` | Calibrate a camera's lens |

## Model runtimes

Each runs in its own environment, set up by a mise task, with weights under `data/models/`.

| Model | Task | License |
|---|---|---|
| MASt3R (pair matching and alignment) | `mise run mast3r` | CC BY-NC-SA 4.0 |
| CUT3R (poses from video) | `mise run cut3r` | CC BY-NC-SA 4.0 |
| GeoCalib (gravity and focal) | `mise run geocalib` | See its repository |
| gsplat with Depth Anything 3 (splats) | `mise run gsplat` | Apache-2.0 |
| MegaLoc (revisit retrieval, CPU) | Fetched with the models | MIT |

MASt3R and CUT3R are non-commercial. That is accepted for now.
