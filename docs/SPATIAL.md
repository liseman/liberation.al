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
| `splat.note` | Why the newest splat pass made no splat for the revision it tried (too few lit daytime frames, no cameras to take frames from, no trainer, or a failed run), for the 3D view to say |

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
| `tools/rebuild_place.py` | Rebuilds a place's map from its walks (their export folders with `--walk`, or their revisions' maps) and fresh camera snapshots, in the owner plan's frame |
| `tools/restore_owner_plan.py` | Repairs a place whose fused map drifted off the frame the owner plan was drawn in |
| `tools/place_positions.py` | Runs the placer by hand; `--reset` places every place again |
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
| CUT3R (poses from video) | `mise run cut3r` | CC BY-NC-SA 4.0 |
| GeoCalib (gravity and focal) | `mise run geocalib` | See its repository |
| gsplat with Depth Anything 3 (splats) | `mise run gsplat` | Apache-2.0 |
| MegaLoc (revisit retrieval, CPU) | Fetched with the models | MIT |

MASt3R and CUT3R are non-commercial. That is accepted for now.
