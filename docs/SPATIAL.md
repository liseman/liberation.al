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
| `splat.note` | Why the newest splat pass made no splat for the revision it tried (too few lit daytime frames, no cameras to take frames from, no trainer, or a failed run), for the 3D view to say |

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

## Learning surroundings

The design for a place's map that the cameras refine as they see it (#254). It replaces the walk import and the
nightly pass (#252), and each part below says which issue builds it. Until a part is built, its readers say so.

**The switch** (#255). `spatial.learn_surroundings` (off by default) and `spatial.learn_cameras` (camera names; empty
means the first camera in the device manifest). Both are live settings: the sampler reads `pack.toml` again when it
changes, so the Live tab's button and camera picker take effect in about a second with no restart.

**The sampler** (#256). `pack-surroundings` holds no decoder connection while the switch is off. While it's on it
takes one BGR picture every 5 s from each chosen camera, from see-decode's shared pictures. It skips, cheapest check
first: a dark or night-mode frame, a frame whose 32 x 18 thumbnail matches the last one looked at (a still camera
costs a few milliseconds a sample), a frame much less sharp than the camera's usual, and a frame the person detector
finds a person or an animal in. Such a frame is skipped, not masked: the view comes by again without them. What it's
doing is in `state/surroundings/status.json` (`spatial/surroundings_worker.py`).

**The gate** (#257). "Likely to improve the map" is measured twice, cheaply first.

1. *Before a pose is known*, by the frame's place-recognition descriptor (MegaLoc, on the CPU, about 0.26 s): its
   novelty is 1 minus its highest cosine similarity to the kept frames. Measured on six live cameras, read only:
   a still camera's frames 5 s apart are 0.95 to 0.99 alike (54 pairs), the same view one to three hours apart under
   changing daylight 0.61 to 0.83 (9 pairs), and two different cameras at most 0.18 (15 pairs). So a frame under 0.85
   alike to every kept frame is kept: that takes a new viewpoint, an uncovered area, and a scene or light that
   changed, and drops the copies. A frame at least 0.85 alike to a kept one replaces it when it's 1.5 times as
   sharp (better detail), and is dropped otherwise (`spatial/surroundings_frames.py`).
2. *Once it has a pose* (in the map step), by the map itself: the current splat is drawn from the frame's pose.
   The share of the frame the splat doesn't cover is the uncovered area, and the error of what it does draw is the
   change or the missing detail. A frame the splat already draws well is dropped. The map step's pull request sets
   those two thresholds from measurements.

Kept frames are JPEGs with their descriptors under `state/surroundings/`, at most 300 in all and 4 per view (frames
at least 0.5 alike are one view, and a fifth look at it pushes out its oldest). At the bound, the frame most like
another kept frame goes first. They are machine-made: a reset deletes them with the places.

**The map** (#258). One step at a time, niced, on the GPU with the most free memory, started by the sampler when
new frames are waiting, never on a schedule.

- *Poses come from the frames.* A camera may be on a moving rig or on a fixed mount, so neither is assumed. A
  place's first frame founds its map: Depth Anything 3's metric depth lifted to points, the frame's own pose the
  origin. A later frame is paired with the kept frames it's most like through MASt3R, whose pair pose, scaled by the
  kept frame's depth, gives the frame's pose in the map. A frame no pair places is dropped. The splat trainer's own
  pose refinement takes up what's left.
- *The splat is continued, not rebuilt.* The trainer starts from the place's current splat, adds points where a new
  frame shows what the map lacked, and trains a short while on the old and new views. Each step writes one revision
  of the place (`memory/places.py`); the newest two maps and splats are kept, as before.
- *Places.* Frames that register with a place's kept frames join that place. Frames that register with each other
  and with no place found a new one. Telling places apart by the rig's signals is not part of this.

**The 3D view** (#259) shows the newest splat, reloads when a newer one is written, and says whether learning is on,
how many views the map has, and when it last improved.

**Positions** (#260). Foot points keep being recorded as each camera saw them. A learned map gets a floor from its own
surfaces. A camera whose kept views all sit at one pose is registered in the revision as a fixed camera, and its
foot points are placed on the map's floor in metres, through the same geometry as before. A camera on a moving rig
needs a pose for each moment (#7), so its foot points stay unplaced until then. While a place has no map, or a camera
has no pose, readers say that and show no position.

**The owner's plan** (#261) is kept as taught state. Lining a learned map up with it comes after the map exists: until
then the 3D view and positions are in the map's own frame, and the plan is drawn on its own.

## Model runtimes

Each runs in its own environment, set up by a mise task, with weights under `data/models/`.

| Model | Task | License |
|---|---|---|
| MASt3R (pair matching and alignment) | `mise run mast3r` | CC BY-NC-SA 4.0 |
| gsplat with Depth Anything 3 (splats) | `mise run gsplat` | Apache-2.0 |
| MegaLoc (place recognition between frames, CPU) | Fetched with the models | MIT |

MASt3R is non-commercial. That is accepted for now.
