# Spatial memory

Why the pack keeps a plan of each indoor place, how it collects pictures of the place, what a camera's own point cloud
is, and how a person a camera sees ends up as a point on the plan. Paths are relative to `src/liberational/`.
Everything here is optional. To do it on your own install, follow [Places](PLACES.md).

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
no revision's map is shown. What the 3D view shows is one camera's own point cloud at a time
([A camera's own point cloud](#a-cameras-own-point-cloud)).

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

## A camera's own point cloud

A camera that stays put can't measure a room: nothing it sees shifts against anything else. What it can give is its
own view with depth. So each camera chosen for Learn surroundings gets one point cloud, made from one picture of that
camera (`spatial/camera_clouds.py`). Clouds of different cameras are never blended into one, and a cloud registers
no camera: positions don't read it. The 3D view shows one camera's cloud by itself, or every camera's cloud placed
where its camera stands ([The scene of all cameras](#the-scene-of-all-cameras)).

- **Which picture** (`spatial/surroundings_clouds.py`). One with nobody in it: five sampled pictures in a row with no
  person or animal, all alike, give one clean picture, their per-pixel median. That is about 20 seconds of an empty
  view.
- **When it is made again.** A camera gets a cloud when it has none, and a new one when its view has changed (the
  room was rearranged, or the camera moved or turned), judged by the picture's thumbnail against the one its cloud was
  made from. The new cloud replaces the old one: nothing is carried forward.
- **What makes it.** MASt3R-SLAM's single-frame step gives every pixel a point and a confidence. Points the model is
  unsure of are left out (bright windows, glass, and sky usually), and so are the pixels the camera burns its clock
  or name into (the device manifest's `masks`). A picture the model is unsure of over most of it, or one with no
  depth to speak of, gives no cloud, and the camera's earlier cloud stays.
- **What is cleaned out.** Points that aren't on a surface the camera really saw: a pixel at a jump in depth, where
  the model strings points between a near thing and the far one behind it; a surface seen so nearly edge-on that its
  depth is a guess; and a small patch left standing alone. These are the sheets and streaks a cloud would show from
  anywhere but its own camera.
- **Its size.** The cloud stays in the model's own metres, and the 3D view says its size is approximate. A camera's
  mounting height does not size it. The height reads the floor a metre or two from the camera, the walls are several
  metres away, and the model's near and far depths don't share one scale, so sizing by the height puts the walls
  farther off than the model's own metres do.
- **Its floor.** The floor is the lowest big surface under the camera. A mounting height in the device manifest
  (`mount.height_m`) checks the find: a plane that would put the camera far from that height (a sofa seat, with the
  floor out of sight) isn't its floor, and is passed over.
- **Its frame.** The cloud is level: z is up, the floor is z = 0, and the camera stands above the origin, looking
  along +y. With no floor found the camera is taken to be upright and the lowest points stand in for the floor.
- **Where it is kept.** `state/camera_clouds/<camera>/` holds the camera's one cloud, the picture it was made from,
  and its record, and `notes.json` beside the folders says why a camera's last try made none. `pack-reset-places`
  deletes them.

A pass runs on a GPU and is skipped, with the reason, when none has 4 GB free. Nothing is made unless
`spatial.pointmap_runtime` names where MASt3R-SLAM is set up ([Model runtimes](#model-runtimes)).

## The scene of all cameras

The 3D view's **All cameras** puts every camera's cloud of a place into one scene (`spatial/cloud_scene.py`). The
clouds are placed, never blended: the scene is a list of clouds with a transform each, and every point stays its own
camera's. Its frame is the owner's plan, in metres. Three things decide where a cloud goes, in this order.

1. **The owner's spot.** A camera's spot on the plan says where it is and which way it faces. The cloud is level
   already, so placing it is a turn about the vertical and a shift: the camera lands on its spot, facing its facing,
   with the cloud's floor on the plan's. A camera with no spot, or a spot with no facing, is not in the scene.
2. **A pair of cameras that see the same space.** When a camera's cloud is made, its picture is compared with the
   pictures of up to five other cameras of the place, the nearest by their spots first, by MASt3R's two-picture
   matching. A confident match says where one camera stands as seen from the other, and pulls the two towards what
   the pictures say. Each stays near its own spot: the spots are the reference, a match weighs ten times a spot, and
   the two cameras share the correction, so the scene stays on the plan.
3. **A camera that was moved.** The picture a camera was first placed with at its spot is its anchor. Each new
   picture is compared with it. One that registers within 0.3 m and 4 degrees hasn't moved, and becomes the anchor.
   One that registers farther has moved: its cloud follows by what the pictures measured, and the new spot is offered
   to the owner. One that doesn't register is looking at other space: it is tried against the other cameras'
   pictures and placed from the first that registers. If none does, the camera leaves the scene and says it was
   moved and isn't placed, until its spot is set again or a picture registers. A moved camera's cloud is never left
   at its old spot.

**When a match is confident.** It is measured, not judged. Each direction must be sure of at least 3 percent of the
pixels in the other's frame, the matcher must find at least 15 percent of them one way or the other, and the two
directions must agree within 8 degrees, within 0.25 m or a tenth of the distance, and within a tenth in scale.
Cameras that look at different space share no pixel the model is sure of, so they fall well short.

**The bound.** A match may move a camera at most 0.75 m and 20 degrees from its spot. One that would move it farther
says that one of the two spots is off: the view says so, and nothing is moved.

**Daylight.** A pair that matches in daylight often doesn't at night. While neither camera has moved and neither
spot has changed, the last confident measurement stays in use, and a later look that isn't confident doesn't
replace it.

**What is written.** Nothing here changes a spot or the plan. The pairs' measurements are kept in
`state/camera_clouds/pairs.json`, and each camera's folder holds its anchor (`anchor.json`, `anchor.png`) and, once
its picture stopped matching the anchor, `moved.json`. A measurement made from another picture or another spot is not
used. The spot offered for a moved camera is saved, as a new version of the plan, only when the owner accepts it; the
anchor it replaced is set aside so that Undo puts the plan's earlier version and the anchor back. The scene's points
are drawn on request into `state/camera_clouds/scenes/` and kept until the scene changes.

**What a scene is good for.** Each camera's view standing in the right place, not a whole room. Clouds that overlap
meet with a seam of tens of centimetres, and cameras that share no space are only as right as their spots. A point
more than 0.3 m outside every room of the plan is marked, and the view leaves it out unless asked: a cloud's depth is
the model's estimate, and can run through a wall.

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

## Model runtimes

| Model | Used for | Set up | License |
|---|---|---|---|
| MASt3R-SLAM, with MASt3R's weights | Each camera's point cloud | By hand, in a folder of its own that `spatial.pointmap_runtime` names, with the weights at `spatial.pointmap_weights`. [MIGRATING.md](../MIGRATING.md#each-cameras-own-point-cloud) has the steps | CC BY-NC-SA 4.0 |
| MegaLoc | Telling which pictures are worth keeping (CPU) | `mise run models` | MIT |

MASt3R-SLAM's code and MASt3R's weights are non-commercial. The pack downloads and installs neither.
