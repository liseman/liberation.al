# Spatial memory

How the pack keeps a map and floorplan of each indoor place and puts people on it. Paths are relative to
`src/liberational/`. Everything here is optional, and the model runtimes need CUDA.

Spatial memory is being built again from what the cameras see. Phone walks, the nightly fixed-camera registration,
and the fusion of their maps are gone, and no job writes a place's map at present. What stays is what the new job
builds on: the place store, the owner's plans, the splat trainer, the model runtimes, and the position layer.

## What a place is

A place is a chain of revisions under `state/places/<place>/` (`memory/places.py`). Each revision is written once:

| File | Contents |
|---|---|
| `<revision>.json` | Producer, source visits, identifying signals, the floor plane, the map-to-plan transform, and each registered fixed camera's pose |
| `<revision>.map` | The point cloud (binary PLY) |
| `<revision>.plan` | The floorplan DSL |
| `<revision>.gs.ply`, `<revision>.sog` | An optional Gaussian splat and its copy for the 3D view |

The newest two maps and splats are kept; plans and revision records stay. All of it is machine-made and a reset
deletes it: `pack-reset-places` (`spatial/reset.py`) deletes every place's maps, splats, camera registrations, and
stored positions, and keeps a place whose owner plan holds, with that plan and no map. Without `--yes` it lists what
would go and what would stay. The owner's place names and plan corrections (`place.named`, `place.plan`) live in the knowledge store
and supersede the machine's view without changing it.

## Jobs

Nothing runs on a schedule. These run by hand, and the job that maps a place calls them when it does:

| Job | Does |
|---|---|
| `pack-spatial-splat [--place ID]` (`spatial/splats.py`) | Trains one Gaussian splat per revision that registers fixed cameras, from a day of their recorded frames, with gsplat and a Depth Anything 3 depth prior. `train_views` trains one from many posed views of a moving camera |
| `tools/place_positions.py` | Records the fixed cameras' raw footprints (`spatial/footprints.py`) and runs the placer (`spatial/positions.py`), which places people's and pets' foot points on the plan |

`spatial/cameras.py` is what jobs that work from recorded frames share: which cameras are fixed, reading a camera's
recorded frame, whether a frame is lit (`daylight`) and sharp (`sharpness`), the people in it (`detector`,
`person_mask`), running MASt3R (`run_mast3r`), and a cloud's floor and floorplan (`floor_along`, `layout`).
`spatial/retrieval.py` scores how alike two frames' places look (MegaLoc). `spatial/poses.py` stores each camera's
pose over time for the placer to read; nothing writes it at present.

`state/places.lock` keeps two jobs from writing places at once.

## Positions

`spatial/footprints.py` records raw foot points as cameras saw them (`position.sample`). A foot point is placed
through its camera's pose in a place's map, so while no revision registers a camera the placer places nothing and
says so, and the Places page says nobody can be placed there. The placer maps them to
plan coordinates and writes `position.placed` and `position.path`: who or what, place, room, coordinates, time,
confidence, and source. When a registration moves a camera's placement by more than 0.25 m, the placer places that
place's samples again into a new generation, so a path is always in one frame.

`memory/places.py:places_for_span` tells the journal which places and rooms a span touched.

## Hand tools

Each defaults to a dry run that prints what would change. Their docstrings have the options.

| Tool | Does |
|---|---|
| `tools/import_owner_plan.py` | Registers an owner-supplied floorplan into a place's map and saves it as the owner plan |
| `tools/place_positions.py` | Records footprints and runs the placer; `--reset` places every place again |
| `tools/calibrate_lens.py`, `tools/calibrate_fisheye.py` | Calibrate a camera's lens |

## Model runtimes

Each runs in its own environment, set up by a mise task, with weights under `data/models/`.

| Model | Task | License |
|---|---|---|
| MASt3R (pair matching and alignment) | `mise run mast3r` | CC BY-NC-SA 4.0 |
| gsplat with Depth Anything 3 (splats) | `mise run gsplat` | Apache-2.0 |
| MegaLoc (place recognition between frames, CPU) | Fetched with the models | MIT |

MASt3R is non-commercial. That is accepted for now.
