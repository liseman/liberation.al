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
| `tools/place_positions.py` | Works out the tilt of each camera with a spot on its place's plan (`spatial/spots.py`), records the fixed cameras' raw footprints (`spatial/footprints.py`), and runs the placer (`spatial/positions.py`), which places people's and pets' foot points on the plan. `--dry-run` prints what it would place per camera and changes nothing |

`spatial/cameras.py` is what jobs that work from recorded frames share: which cameras are fixed, reading a camera's
recorded frame, whether a frame is lit (`daylight`) and sharp (`sharpness`), the people in it (`detector`,
`person_mask`), running MASt3R (`run_mast3r`), and a cloud's floor and floorplan (`floor_along`, `layout`).
`spatial/retrieval.py` scores how alike two frames' places look (MegaLoc). `spatial/poses.py` stores each camera's
pose over time for the placer to read; nothing writes it at present.

`state/places.lock` keeps two jobs from writing places at once.

## Positions

`spatial/footprints.py` records raw foot points as cameras saw them (`position.sample`). A foot point is placed
through its camera's pose, which comes from one of two things:

- **The owner's spot on the plan** (`spatial/spots.py`). A plan's `x-camera <camera> at x,y height <m> yaw <deg>` line
  says where the camera is, how high, and which way it faces, and that is enough: the camera has a pose in the plan's
  own frame, with its held focal and lens, and its foot points land on the plan's floor with no map. The one thing a
  spot doesn't say is how far the camera is tilted. That is worked out once per camera from the people it sees
  walking: someone walking is upright and as tall at the far end of the room as at the near end, and only one tilt
  makes them so. The fit is checked against the spot's height (people must come out a person's height) and kept in
  `state/places/aims.json`. A spot outranks any pose a map gives the camera: a map may refine it only within 0.3 m
  and 5 degrees, and whoever applies that logs it.
- **A map's registration** (a revision's `cameras`), for a camera with no spot.

A camera with neither places nobody. A spot that lacks something says what, per camera: no height, no facing (yaw),
no known lens, too few people seen walking to work out the tilt yet, or people who come out the wrong height. The
Places page lists those under the plan, and the rest of the place's cameras place people as usual. The placer maps
foot points to plan coordinates and writes `position.placed` and `position.path`: who or what, place, room,
coordinates, time, confidence, and source. When a spot or a registration moves a camera's placement by more than
0.25 m, the placer places that place's samples again into a new generation, so a path is always in one frame.

`memory/places.py:places_for_span` tells the journal which places and rooms a span touched.

## Hand tools

Each defaults to a dry run that prints what would change. Their docstrings have the options.

| Tool | Does |
|---|---|
| `tools/import_owner_plan.py` | Registers an owner-supplied floorplan into a place's map and saves it as the owner plan |
| `tools/place_positions.py` | Fits spot cameras' tilts, records footprints, and runs the placer; `--dry-run` changes nothing, `--days N` reaches further back the first time, and `--reset` places every place again |
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
2. *Once it has a pose* (in the map step), by the map itself. The frame's pose is first fitted to the current splat,
   then the splat is drawn from it: the share of the frame it doesn't cover is the uncovered area, and the error of
   what it does draw is the change or the missing detail. A frame the splat covers 0.97 of and draws at 21 dB or
   better adds nothing, and is dropped (`surroundings_map.well_drawn` has the measurements: copies of kept frames
   drew at 21.6 to 24.7 dB, the same view in changed light at 11.7 to 20.1 dB, and a new viewpoint at 11.4 dB with
   0.67 of it covered).

Kept frames are JPEGs with their descriptors under `state/surroundings/`, at most 300 in all and 4 per view (frames
at least 0.5 alike are one view, and a fifth look at it pushes out its oldest). At the bound, the frame most like
another kept frame goes first. They are machine-made: a reset deletes them with the places.

**The map** (#258, `spatial/surroundings_map.py`). One step at a time, niced, on the GPU with the most free memory,
started by the sampler when kept frames are waiting (at most one a minute), never on a schedule; `pack-surroundings-map`
runs one by hand. A step takes up to 12 waiting frames.

- *Poses come from the frames.* A camera may be on a moving rig or on a fixed mount, so neither is assumed. A
  place's first frame founds its map, its own pose the origin (so the map isn't levelled until its floor is found,
  #260). A later frame is paired with the three kept frames it's most like through MASt3R, whose pair pose, scaled
  to metres by Depth Anything 3's metric depth of both frames, gives the frame's pose in the map. Frames are placed
  outward from the ones that have a pose, each through its pair with the most agreeing matches. A frame no pair
  places is tried in three steps, then dropped. The splat trainer's pose refinement takes up what's left.
- *A camera's optics.* Its held intrinsics (`lens.Held`, taught state) give its focal and the lens to undo; a lens in
  the device manifest is undone the same way. A camera with neither gets its focal from MASt3R, once, kept from then
  on. Two fixed cameras with held intrinsics came out 2.9 m apart; with MASt3R's focals (7% and 15% off) 4.8 m.
- *The splat is continued, not rebuilt.* The trainer starts from the place's current splat, adds each new frame's
  depth points as new Gaussians, and trains 1,500 steps on the old and new views (3,000 for a place's first), each
  against its depth. Each step writes one revision of the place (`memory/places.py`) in the place's frame, which
  never moves; the newest two maps and splats are kept, as before.
- *Places.* Frames that register with a place's kept frames join that place. Three or more that register with each
  other and with no place found a new one. A first learned map takes the one listed place that has no map (an
  owner's plan kept across a reset). Telling places apart by the rig's signals is not part of this.

Measured on the Linux machine, shared with the live services (one place, two fixed cameras, frames 640 px wide): a
step founding a place from two frames took 53 s in all, 27 s of it training, to 50,000 Gaussians at 23.2 dB on its
own views; a step adding two frames, one from the second camera, 69 s, 29 s training, to 93,000 Gaussians at 24.0 dB;
a step taking eight near-copies of kept frames 75 s, 19 s training, dropping five of them as already drawn; and a step
with nothing waiting does nothing. The trainer used 0.1 to 0.3 GiB of GPU memory. With the newest of seven views
held out, 1,500 steps resumed from the place's earlier splat took 23 s and scored 12.1 dB on it, against 38 s and
9.9 dB for 3,000 steps from the map's points.

`surroundings_map.summary` and `surroundings_map.views` are what a page and the position layer read: whether
learning is on, how many frames are kept and waiting, whether a step is running, and each place's newest revision,
views, and when it last improved; and each view's camera, time, pose, and focal.

**The 3D view** (#259) shows the newest splat, reloads when a newer one is written, and says whether learning is on,
how many views the map has, and when it last improved. `GET /places/scene` carries that as `learning`, from the map
step's `summary` (`dashboard/places.py`), also for a place with no map, where the page says what would start one.
While the view is open it asks for the scene again every 10 s with learning on (every 60 s with it off). A newer
map and splat are drawn once all their bytes are here, the ones before staying drawn until then, from where the
person was looking. A splat that learning is still adding views to is in the state `improving`.

**What learning is doing** (#283). While the switch is on, the Places page marks each place "Learning" and says what
is going on, from `GET /surroundings/status` (the map step's `summary`): the cameras sampled, the frames kept and
waiting, what a running step is doing, and when the last one ended. A running step writes its phase (`pairing`,
`depth`, `training`, `writing`) to `state/surroundings/progress.json`, and the trainer its step count to
`training.json` beside it, so training shows a measured bar and the other phases an unmeasured one. Both are read
only while a step holds its lock. The sampler's status carries each camera's last 12 looks by outcome, and the page
says why a camera's frames aren't being kept (a person or an animal in view, nothing new, dark, or blurred) when
none of them was. The page asks every 3 s while a step runs, every 10 s while learning is on, and every 60 s
while it's off.

**Positions** (#260). Foot points keep being recorded as each camera saw them. A learned map gets a floor from its own
surfaces. A camera whose kept views all sit at one pose is registered in the revision as a fixed camera, and its
foot points are placed on the map's floor in metres, through the same geometry as before. A camera on a moving rig
needs a pose for each moment (#7), so its foot points stay unplaced until then. While a place has no map, or a camera
has no pose, readers say that and show no position.

**The owner's plan** (#261) is kept as taught state, and a learned map is lined up with it
(`spatial/plan_lineup.py`): a turn, a shift, and a scale on the floor, written as the `plan_transform` of a revision
with no map of its own, with the owner's plan saved again, unchanged, as drawn in that revision's frame. Learned
revisions carry the transform over, so the 3D view, the plan, and the positions share one frame from then on.

- *Evidence.* Two or more fixed cameras the plan has a measured spot for (`x-camera`) give the line-up outright: each
  camera registered in the map must land on its spot. Otherwise the map's wall band is fitted to the plan's walls,
  and a single measured spot rules out the fits that put its camera elsewhere.
- *Found once.* A wall fit is kept only when the map lies on at least 30% of the plan's wall length, scores 1.15
  times better than the best different fit (one square room fits its plan four ways), and then sits on the walls
  (a median of 10 cm from a wall's face). Until then the place is unaligned and its status says why (no map, no
  floor, too little of the plan, more than one fit, or a poor fit): the 3D view and positions stay in the map's own
  frame, and the plan is drawn on its own. A fit is tried again only after the map's walls grew by a fifth.
- *Checked as it grows.* Each later map is measured on the plan through the transform it has, with no search. A map
  that left its plan is fitted once more; when that finds nothing the line-up stays and is flagged as drifted.

`plan_lineup.advance(state)` does all of it, is called at the end of every map step, and costs one small read when
nothing changed; `plan_lineup.status(state, place)` is what a reader says. The made-up maps in its tests (noise of a
few centimetres, a scale 8% to 17% off) line up within a centimetre or two at the rooms' corners, in under 4 s for a
whole home on the Linux machine.

## Model runtimes

Each runs in its own environment, set up by a mise task, with weights under `data/models/`.

| Model | Task | License |
|---|---|---|
| MASt3R (pair matching and alignment) | `mise run mast3r` | CC BY-NC-SA 4.0 |
| gsplat with Depth Anything 3 (splats) | `mise run gsplat` | Apache-2.0 |
| MegaLoc (place recognition between frames, CPU) | Fetched with the models | MIT |

MASt3R is non-commercial. That is accepted for now.
