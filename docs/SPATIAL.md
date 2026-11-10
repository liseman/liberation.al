# Spatial memory

Why the pack keeps a map and a plan of each indoor place, how the cameras learn the map, and how a person a camera
sees ends up as a point on the plan. Paths are relative to `src/liberational/`. Everything here is optional, and the
model runtimes need CUDA. To do it on your own install, follow [Places](PLACES.md).

A camera sees a person as pixels. Saying which room they are in takes two things: where the floor is under that
pixel, and where that floor is on a plan. So each place has a map the cameras refine as they see it, a plan, and a
pose for every camera that places people. The owner can supply the plan and each camera's spot on it, and whatever
the owner supplies outranks what the machine works out.

## What a place is

A place is an indoor place, such as a home. The machine's view of it is a chain of revisions under
`state/places/<place>/` (`memory/places.py`). A revision is written once and never edited:

| File | Contents |
|---|---|
| `<revision>.json` | Producer, source visits, identifying signals, the floor plane, the map-to-plan transform, and each registered fixed camera's pose |
| `<revision>.map` | The point cloud (binary PLY) |
| `<revision>.plan` | A machine-made plan: a scene, as the plan editor loads and saves it (`spatial/plan_scene.py`) |
| `<revision>.gs.ply`, `<revision>.sog` | A Gaussian splat of the map and its copy for the 3D view |

A revision that changes only the floor, the cameras, or the transform points at the earlier map. The newest two maps
and splats are kept; plans and revision records stay.

All of that is an interpretation, so it can be deleted and made again. What the owner taught is in the knowledge store
and supersedes it without changing it: a place's name (`place.named`), its plan (`place.plan`, with every earlier
version: a scene saved from the plan editor), and each camera's
held focal and lens. While a place has an owner's plan, that is its only plan: the Places page, the 3D view, and the
position layer use it and no machine plan stands in for it.

## Learning surroundings

A map is learned from pictures the cameras already take, with no walk-through and no schedule. Two live settings
switch it, `spatial.learn_surroundings` and `spatial.learn_cameras` (`spatial/surroundings.py`): no restart.

**The sampler** (`pack-surroundings`, `spatial/surroundings_worker.py`) looks at a picture from each chosen camera
every few seconds and drops most of them, cheapest check first: a dark or night-mode frame, a frame that looks like
the camera's last judged one, a frame much less sharp than the camera's usual, and a frame more than a quarter
covered by people or animals. A frame with someone in a smaller part of it goes on with those pixels marked, and the
map leaves them out. Dropping every frame with a person in it would never learn a room someone sits in.

**The first gate** (`spatial/surroundings_frames.py`) asks whether a frame is likely to improve the map before it has
a pose, by its place-recognition descriptor (MegaLoc). A frame under 0.85 alike to every kept frame is kept: that
takes a new viewpoint and a known view that looks different, and drops the copies. A copy replaces the kept frame it
matches only when it is 1.5 times as sharp. Kept frames live under `state/surroundings/`, at most 300 in all and 4 a
view (a fifth look at a view pushes out its oldest), so a view's frames follow how it looks today.

**The map step** (`pack-surroundings-map`, `spatial/surroundings_map.py`) is started by the sampler when kept frames
are waiting, one at a time. It takes up to 12 of them and writes one revision:

- *Poses come from the frames.* A camera may be fixed or on a moving rig, so neither is assumed. A waiting frame is
  paired with the kept frames it is most alike, MASt3R measures each pair, and Depth Anything 3's metric depth scales
  the result to metres. Frames are placed outward from the ones that have a pose. A place's first frame founds its
  map with its own camera as the origin. A camera's held lens, or the one in the device manifest, is undone first; a
  camera with neither gets its focal from MASt3R once and keeps it.
- *The splat is continued, not rebuilt.* Training starts from the place's current splat and adds the new frames'
  depth points, which is quicker than training again from the map's points.
- *The second gate.* Once a new frame has a pose, the current splat is drawn from it. A frame the splat already
  covers (0.97 of it) and draws well (21 dB) adds nothing and is dropped.
- *Places.* A frame that pairs with a place's frames joins that place. Three or more that pair only with each other
  found a new one. A first learned map takes the one listed place that has no map (an owner plan kept by a reset).

**A camera with a spot needs no pair.** Fixed cameras in different rooms overlap nowhere, so pairing alone gives a
home several unconnected maps, or none. A camera whose spot on the owner's plan gives a pose (below) has its frames
placed at that pose, in the plan's own frame. Its depth is checked against the plan: the floor it shows must come
out at the plan's floor, and the depth is scaled to put it there, since the height is tape-measured and the depth
model's metres are not. A scale outside 0.75 to 1.35 refuses the view. A place with such a view stands in its
plan's frame from then on, and a map learned earlier in a frame of its own is started again there.

**A spot is checked against the plan, once** (`spatial/spot_check.py`). A view's depth is too rough to measure a spot
by how near its points fall to the plan's walls, so the check uses what the picture and the plan agree on whatever
the depth's scale: the directions straight edges vanish in, a floor that is flat and at the plan's level, rays that
must not pass through a solid wall, the plan's wall-floor lines and door jambs drawn into the picture, and walls
that face into the room. A fixed camera's views all stand at one pose, so the search runs once for a spot and its
result is kept (`state/places/refined.json`) until the spot, the plan, or the camera's lens or focal changes.

**A frame nothing places is parked.** It stays kept, with the reason: its view overlaps nothing in a map and its
camera has no spot, the spot lacks something, or its floor doesn't fit the plan's. A step for it would find what the
last one found, so none starts until a map gains a view or the camera's spot changes. After a day it is dropped and
the camera's next frame takes its place.

## Positions

`spatial/footprints.py` records raw foot points as each fixed camera saw them (`position.sample`). The placer
(`spatial/positions.py`) puts each on the floor through its camera's pose and writes `position.placed` and
`position.path`: who or what, place, room, coordinates, time, confidence, and source. A camera's pose comes from one
of two things:

- **The owner's spot on the plan** (`spatial/spots.py`). A camera's spot, set in the plan editor and kept in the
  plan's scene, says where the camera is, how high, and which way it faces. With the camera's held focal and lens (or
  the spot's `fov`) that is a pose in the plan's own frame, and its foot points land on the plan with no map at all. A
  held focal belongs to the lens it was measured through: a camera with a lens in the device manifest and no focal
  measured through it is read with the focal that lens's calibration gives. The one thing a spot doesn't say is how
  far the camera is tilted. That is fitted once per camera from the people it sees walking: someone walking is upright
  and as tall at the far end of the room as at the near end, and only one tilt makes them so. Nobody's height is
  assumed, and people sitting don't count. The fit is checked against the spot's height (people must come out a
  person's height) and kept in `state/places/aims.json`.
- **A map's registration** (a revision's `cameras`), for a camera with no spot.

The owner measured the spot, so it outranks what the machine works out. The check above may refine the pose a spot
gives only within 0.3 m and 5 degrees, and it keeps that refinement beside the spot, which is never rewritten. When
the view fits the plan clearly better beyond that bound, nothing is applied: the spot holds, and the Places page
suggests checking it, with how far off the view wants to sit. A lens in the device manifest is part of the pose, so
a changed lens calibration has the tilt fitted again. A camera with neither places nobody, and no position is
guessed: the placer and the Places page say what each camera lacks.

**A learned map's floor and fixed cameras** (`spatial/surroundings_positions.py`). A map founded on a camera frame
isn't level and registers no camera. Up is found from the map's own flat surfaces, and the floor is the lowest layer
of the surfaces that face up; the revision's floor plane is the levelling, and the map file isn't rewritten. A camera
that is on no rig and whose kept views all sit at one pose (within 0.5 m and 8 degrees) is registered as fixed at
that pose. A camera on a rig, or seen from several poses, needs a pose for each moment: it isn't registered, and its
foot points stay unplaced.

A foot point the camera's spot could not see by the plan, because a solid wall stands between them, gets no
position: the subject was up on furniture, not on the floor where the ray lands. The placer logs how many it left
out for each camera, people and animals apart.

When a spot or a registration moves where a camera puts people by more than 0.25 m, the placer places that place's
samples again into a new generation, so a path is always in one frame. `memory/places.py:places_for_span` tells the
journal which places and rooms a span touched.

## The owner's plan and the map

A learned map with no spot view stands in a frame of its own, and the owner's plan in a frame no map made.
`spatial/plan_lineup.py` finds the move between them on the floor (a turn, a shift, and a scale) and writes it as
the `plan_transform` of a revision, so the 3D view, the plan, and the positions share one frame.

- *Evidence.* Two or more fixed cameras with a measured spot give the line-up outright: each camera registered in
  the map must land on its spot. Otherwise the map's walls are fitted to the plan's, and a single spot rules out the
  fits that put its camera elsewhere.
- *Enough to tell.* A wall fit is kept only when the map lies on at least 30% of the plan's wall length, scores 1.15
  times better than the best different fit (a square room fits its plan four ways), and then sits on the walls.
  Until then the place is unaligned and its status says why: the 3D view draws the map with no plan on it, and the
  plan view shows the plan as drawn.
- *Found once, then checked.* Each later map is measured on the plan through the transform it has, with no search.
  A map that left its plan is fitted once more; when that finds nothing the line-up stays and is flagged as drifted.
  A place that stands in its plan's frame (a spot view put it there) needs no search at all.

## Jobs

[Components](COMPONENTS.md) lists each job with what it reads and writes. Two are services
([services/services.toml](../services/services.toml)): `surroundings` runs the sampler, which starts the map steps,
and `positions` runs `pack-positions` every hour. `pack-surroundings-map`, `pack-positions`, `pack-spatial-splat`,
and `pack-reset-places` run by hand; each module's docstring has its options. `spatial/cameras.py` holds what the
jobs share, and `spatial/poses.py` reads a camera's pose over time (no job records observations there, so a camera
is read at the pose its revision holds).

## Hand tools

`tools/calibrate_lens.py` and `tools/calibrate_fisheye.py` calibrate a camera's lens. Each prints what it found and
writes only with `--write`; their docstrings have the options, and [Cameras](CAMERAS.md) has the steps.

## Model runtimes

MASt3R and gsplat each run in their own environment. `mise run models` fetches the weights into `data/models/`.

| Model | Task | License |
|---|---|---|
| MASt3R (pair matching and alignment) | `mise run mast3r` | CC BY-NC-SA 4.0 |
| gsplat with Depth Anything 3 (splats and metric depth) | `mise run gsplat` | Apache-2.0 |
| MegaLoc (place recognition between frames, CPU) | `mise run models` | MIT |

MASt3R's code and weights are non-commercial. The project accepts that.
