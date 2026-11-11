# Spatial memory

Why the pack keeps a plan of each indoor place, how it collects pictures of the place, and how a person a camera sees
ends up as a point on the plan. Paths are relative to `src/liberational/`. Everything here is optional. To do it on your
own install, follow [Places](PLACES.md).

A camera sees a person as pixels. Saying which room they are in takes two things: where the floor is under that pixel,
and where that floor is on a plan. So each place has a plan, and every camera that places people has a pose. The owner
supplies the plan and each camera's spot on it, and whatever the owner supplies outranks what the machine holds.

## What a place is

A place is an indoor place, such as a home. The machine's view of it is a chain of revisions under
`state/places/<place>/` (`memory/places.py`). A revision is written once and never edited:

| File | Contents |
|---|---|
| `<revision>.json` | Producer, source visits, identifying signals, the floor plane, the map-to-plan transform, and each registered fixed camera's pose |
| `<revision>.map` | A point cloud (binary PLY) |
| `<revision>.plan` | A plan: a scene, as the plan editor loads and saves it (`spatial/plan_scene.py`) |
| `<revision>.gs.ply`, `<revision>.sog` | A Gaussian splat of the map and its copy for the 3D view |

A revision that changes only the floor, the cameras, or the transform points at the earlier map. The newest two maps
and splats are kept; plans and revision records stay.

The 3D view shows a revision only when its producer is listed in `SHOWN_PRODUCERS` (`memory/places.py`). None is, so
the view has no machine-made map to show, and says so: "No 3D map of this place yet: the 3D map is being rebuilt with
a new method."

What the owner taught is in the knowledge store and supersedes what the machine holds without changing it: a place's
name (`place.named`), its plan (`place.plan`, with every earlier version: a scene saved from the plan editor), and each
camera's held focal and lens. While a place has an owner's plan, that is its only plan: the Places page, the 3D view,
and the position layer use it and no machine plan stands in for it.

## Learning surroundings

While Learn surroundings is on, the pack collects pictures from the cameras it is told to, with no walk-through and no
schedule. Two live settings switch it, `spatial.learn_surroundings` and `spatial.learn_cameras`
(`spatial/surroundings.py`): no restart.

**The sampler** (`pack-surroundings`, `spatial/surroundings_worker.py`) looks at a picture from each chosen camera
every few seconds and drops most of them, cheapest check first: a dark or night-mode frame, a frame that looks like
the camera's last judged one, a frame much less sharp than the camera's usual, and a frame more than a quarter
covered by people or animals. A frame with someone in a smaller part of it goes on with those pixels marked, kept
beside the picture so that whatever reads it can leave them out. Dropping every frame with a person in it would never
collect a room someone sits in.

**The gate** (`spatial/surroundings_frames.py`) asks whether a frame shows something the kept frames don't, by its
place-recognition descriptor (MegaLoc, run on the CPU; `mise run models` fetches it). A frame under 0.85 alike to every
kept frame is kept: that takes a new viewpoint and a known view that looks different, and drops the copies. A copy
replaces the kept frame it matches only when it is 1.5 times as sharp. Kept frames live under `state/surroundings/`,
at most 300 in all and 4 a view (a fifth look at a view pushes out its oldest), so a view's frames follow how it looks
today.

The sampler writes what it is doing to `state/surroundings/status.json`: whether learning is on, per camera what its
recent looks came to and when a frame was last kept, and how many frames are kept. `GET /surroundings/status` serves
it to the Places page, which shows it as the panel [Check what the panel shows](PLACES.md#check-what-the-panel-shows)
describes.

## Positions

`spatial/footprints.py` records raw foot points as each fixed camera saw them (`position.sample`). The placer
(`spatial/positions.py`) puts each on the floor through its camera's pose and writes `position.placed` and
`position.path`: who or what, place, room, coordinates, time, confidence, and source. A camera's pose comes from one
of two things:

- **The owner's spot on the plan** (`spatial/spots.py`). A camera's spot, set in the plan editor and kept in the
  plan's scene, says where the camera is, how high, and which way it faces. With the camera's focal and lens that is a
  pose in the plan's own frame, and its foot points land on the plan with no map at all. The focal comes from the
  camera's held focal and lens when they were measured through the lens its frames are read through, else from the
  spot's `fov`, else from the calibration of the lens in the device manifest. The one thing a spot doesn't say is how
  far the camera is tilted. That is fitted once per camera from the people it sees walking: someone walking is upright
  and as tall at the far end of the room as at the near end, and only one tilt makes them so. Nobody's height is
  assumed, and people sitting don't count. The fit is checked against the spot's height (people must come out a
  person's height) and kept in `state/places/aims.json`. A changed spot height, field of view, or lens has the tilt
  fitted again.
- **A registration in a revision** (a revision's `cameras`), for a camera with no spot.

A camera with neither places nobody, and no position is guessed: the placer and the Places page say what each camera
lacks.

The owner measured the spot, so it outranks a registration. A refinement of a spot within 0.3 m and 5 degrees that is
stored in `state/places/refined.json` is still read (an owner's change to the spot drops it); nothing writes new ones.

A foot point the camera's spot could not see by the plan, because a solid wall stands between them, gets no
position: the subject was up on furniture, not on the floor where the ray lands. The placer logs how many it left
out for each camera, people and animals apart.

When a spot or a registration moves where a camera puts people by more than 0.25 m, the placer places that place's
samples again into a new generation, so a path is always in one frame. `memory/places.py:places_for_span` tells the
journal which places and rooms a span touched.

## Jobs

[Components](COMPONENTS.md) lists each job with what it reads and writes. Two are services
([services/services.toml](../services/services.toml)): `surroundings` runs the sampler, and `positions` runs
`pack-positions` every hour. `pack-positions` and `pack-reset-places` also run by hand; each module's docstring has its
options. `spatial/cameras.py` holds what the jobs share, and `spatial/poses.py` reads a camera's pose over time (no
job records observations there, so a camera is read at its spot, or at the pose its revision holds).

## Hand tools

`tools/calibrate_lens.py` and `tools/calibrate_fisheye.py` calibrate a camera's lens. Each prints what it found and
writes only with `--write`; their docstrings have the options, and [Cameras](CAMERAS.md) has the steps.
