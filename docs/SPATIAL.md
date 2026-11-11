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
no revision's map is shown, and no splat is drawn. What the 3D view shows is cameras' own point clouds: one camera's
by itself, or every camera's together ([A camera's own point cloud](#a-cameras-own-point-cloud),
[The scene of all cameras](#the-scene-of-all-cameras)).

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
- **Its size.** The cloud is made in the model's own metres, which are approximate. A mounting height alone does not
  size it: the height reads the floor a metre or two from the camera, the walls are several metres away, and the
  model's near and far depths don't share one scale. A cloud gets its size from the plan's walls, when it is fitted to
  its room ([The scene of all cameras](#the-scene-of-all-cameras)).
- **Its floor.** The floor is the lowest big surface under the camera. A mounting height in the device manifest
  (`mount.height_m`) checks the find: a plane that would put the camera far from that height (a sofa seat, with the
  floor out of sight) isn't its floor, and is passed over.
- **Its frame.** The cloud is level: z is up, the floor is z = 0, and the camera stands above the origin, looking
  along +y. It is levelled by its own floor. With no floor in view its walls say which way is up (a wall is
  vertical), and the lowest points stand in for the floor. With neither, the camera is taken to be upright.
- **Which cloud is kept.** A view looks different by night. For a camera that has a spot and hasn't moved, a new
  cloud that is worse than the one it has (no floor against a floor, or far fewer points) doesn't replace it.
- **Where it is kept.** `state/camera_clouds/<camera>/` holds the camera's one cloud, the picture it was made from,
  and its record, and `notes.json` beside the folders says why a camera's last try made none. `pack-reset-places`
  deletes them.

A pass runs on a GPU and is skipped, with the reason, when none has 4 GB free. Nothing is made unless
`spatial.pointmap_runtime` names where MASt3R-SLAM is set up ([Model runtimes](#model-runtimes)).

## The scene of all cameras

The 3D view's **All cameras** puts every camera's cloud of a place into one scene (`spatial/cloud_scene.py`). The
clouds are placed, never blended: the scene is a list of clouds with a transform each, and every point stays its own
camera's. Its frame is the owner's plan, in metres, and the plan is the reference throughout: a cloud is one
picture's estimate of depth, and the plan's rooms have walls of known length.

A cloud is brought onto the plan in four moves, each using better evidence than the one before
(`spatial/cloud_align.py`). None of them writes a spot or the plan.

1. **Placed by the owner's spot.** A camera's spot on the plan says where it is and which way it faces. The cloud is
   level already, so placing it is a turn about the vertical and a shift: the camera lands on its spot, facing its
   facing, with the cloud's floor on the plan's. A camera with no spot, or a spot with no facing, is not in the scene.
   A camera that was turned or carried is placed by its own cloud's walls instead: a working pose kept beside the
   cloud, never a change to the spot ([A camera that was turned or carried](#a-camera-that-was-turned-or-carried)).
2. **Squared to the plan.** A cloud's own walls say how its camera really faces, better than a facing set by hand.
   The cloud is turned about its camera so that its walls run along the plan's walls near it, by at most 15 degrees
   from the pose it is placed by. A camera farther round than that is placed again by the walls it sees, facing the
   way they fit, which is how a spot set facing the wrong way still lands its cloud right. Where a cloud wall has one
   clear counterpart on the plan, the cloud is also slid onto it, by at most 0.5 m. Nearer walls count more, since a
   cloud's far depth is its least sure.
3. **Fitted to the room.** Each wall of the squared cloud is matched to the plan wall its camera sees there: the
   first plan wall on the sight lines from the camera's spot to the cloud wall, so never one behind another, when it
   runs the same way (within 10 degrees) and the cloud wall is within 0.8 to 1.25 of its distance. Of several flat
   surfaces in the cloud that face one plan wall, the farthest is the wall, and the nearer ones are the fronts of
   what stands against it. The cloud is then resized about its camera and slid so that its matched walls land on the
   plan's. One wall gives the size alone; walls on both sides give the slide too, by at most 0.5 m after the resize.
   Where the cloud has a floor and its camera a measured height (`mount.height_m`), the cloud is sized upwards by
   the height and across by the walls, which keeps a floor flat and a wall upright.
4. **Adjusted against another camera.** A pair of cameras that see the same space can be measured against each
   other from their pictures ([below](#cameras-that-see-the-same-space)). The measurement moves only a camera that
   could not be squared. A squared camera holds its place, and between two squared cameras the measurement is a
   check that says how far off they are.

A cloud that can't be fitted stays as it was after squaring, in the model's own metres, and its row in the view says
why: it has no wall in view; the wall it sees isn't where a wall of the plan is (an island, a half wall, or furniture
the plan doesn't draw); or the fit would resize or slide it past the bounds, which usually means its spot is off.

What comes of squaring and fitting is kept beside the cloud (`align.json`) and holds only for the cloud, the spot,
and the plan it was made from. When any of those changes it is made again.

### Flat walls and floor

In a fitted cloud, the points of the floor are drawn on the floor, and the points of each matched wall on that wall
of the plan, where they were along it and up it. A wall drawn this way is the plan's wall with what the camera saw
painted on it. Where the camera saw nothing there is a hole: nothing is added. Everything else, the fronts of
furniture included, stays the points it is, and no point is drawn twice. This is a way of drawing, an option of the
view that is on by default: the cloud itself is not changed, and with the option cleared every point is where its
cloud has it.

### Cameras that see the same space

When a camera's cloud is made, its picture is compared with the pictures of up to five other cameras of the place,
the nearest by their spots first, by MASt3R's two-picture matching. A confident match says where one camera stands as
seen from the other.

- **Confident** is measured, not judged. Each direction must be sure of at least 3 percent of the pixels in the
  other's frame, the matcher must find at least 15 percent of them one way or the other, and the two directions must
  agree within 8 degrees, within 0.25 m or a tenth of the distance, and within a tenth in scale. Cameras that look at
  different space share no pixel the model is sure of, so they fall well short.
- **The spots stay the reference.** A match weighs ten times a spot, and two cameras that are both free to move share
  the correction, so the scene stays on the plan.
- **The bound.** A match may move a camera at most 0.75 m and 20 degrees from its spot. One that would move it farther
  says that one of the two spots is off: the view says so, and nothing is moved.
- **Only a camera on its own spot.** A camera that was turned or carried is placed by the plan's walls instead, and a
  match adjusts only cameras standing on their spots.
- **Daylight.** A pair that matches in daylight often doesn't at night. While neither camera has moved and neither
  spot has changed, the last confident measurement stays in use, and a later look that isn't confident doesn't
  replace it.

Registering overlapping clouds to each other from their points and colours was tried and left out. Two cameras see a
room from two sides, and each cloud is a relief with its own bend, so few surfaces are shared: the registration slid
one relief along the other, tipped it off level, and ended farther from what the two pictures measured.

### A camera that was turned or carried

The picture a camera was last placed with is its anchor. Each new picture is compared with it, and one that registers
within 0.3 m and 4 degrees says the camera still stands there, and becomes the anchor. Beyond that the camera is
placed again, without asking anyone, from the first of these that answers (`spatial/cloud_scene.py`):

- **Its pictures.** The new picture registers with the anchor farther than that (it still overlaps its old view), or
  with another camera's picture, and the two-view measurement says where it stands now.
- **Its own cloud.** The cloud's walls against the plan's (`cloud_align.locate`), with no limit on how far it turned
  or went: where it stood facing another way, or anywhere on the plan its walls fit.

The answer is the camera's working pose, kept beside its cloud (`follow.json`) and used in place of the owner's spot;
the cloud is then squared and fitted from it as any cloud is. The owner's spot and plan are never written. A camera
whose pictures didn't change but whose walls line up with the plan only far from its spot's facing is placed this way
too, since a spot set facing the wrong way is not a camera that moved.

An answer counts when the plan's walls account for at least half of the cloud's walls, measured by their points, less
the share of its points that fall outside every room. An answer found away from where the camera stood must also hold
walls in two directions, since one wall says how far the camera is from it and nothing about where along it. The
camera is taken to have stayed where it stood whenever an answer there is within 0.15 of the best anywhere: a camera
that still fits its place was turned, not carried. And the answer must stand alone, because a corner fits other
corners and a wall fits other walls: when a second answer more than 0.75 m or 20 degrees away scores within 0.15 of
the best, the view fits two places and nothing is placed.

A view that can't be placed that way (no wall in view, your plan has no wall to place it by, the walls it sees fit
nowhere, or they fit in two places) is not drawn. The camera's row says why, it keeps the working pose it had for
when its old view comes back, and every new cloud is tried again. The owner has one way back: **Use the plan's spot**
puts the camera on its spot again, which holds until its view next changes, and **Follow it again** takes that back.

### A camera that is moving

Between two still views a moving camera has no place in the scene, and a cloud needs a view that held still. So
`spatial/camera_track.py` gives it a pose a few times a second from its live frames, with MASt3R-SLAM's own frame
tracker (in its runtime, as a process that stays up for the length of a track): each frame is matched to the current
keyframe and solved against that keyframe's points, and a frame that shares too little with it becomes the next one.

A track starts near where the camera stood, when two samples in a row are less than 0.9 alike with nobody in view
(someone walking past a still camera changes its picture too, so a start that sees no motion ends within ten seconds),
the camera has a place in the scene, the runtime is set up, and a GPU has the memory. One camera is tracked at a
time. Every keyframe gives a cloud of its own. The first is placed by its walls near where the camera stood, which
fixes how the track's frame lies on the plan. Each later one is put where the track says it is, then held to the
plan's walls from there, which sets the track's frame on the plan again and takes out what it had drifted: the
camera's height and tilt come from the keyframe's floor and its scale from the plan's walls. A keyframe that no wall
holds stays where the track put it, and the scene says how long ago a wall last held it.

Under the camera's own points the scene draws the last six keyframes' clouds (thinned to 30,000 points each) and its
path (two minutes of poses), kept beside the cloud (`track.json`, `track/`). A track ends when the camera has come to
rest (within 5 cm and 2 degrees of where it was ten seconds before), after ten minutes, or after five restarts, and a
camera whose track ended is left alone for a minute. When the view has held still long enough for a cloud, that cloud
is placed from where the track left the camera, the track's files go, and it is an ordinary camera again. A frame
the tracker can't match is skipped; after six in a row the track is begun again from where the camera was last placed,
and after five of those it ends.

It costs about 10 seconds to load the model at the start of a track, then 0.2 s of one GPU a frame at two frames a
second (about two fifths of that GPU while a camera moves) and 2.9 GB of its memory, all given back when the track
ends.

### What is kept, and when it is made

Everything is kept beside the clouds, in `state/camera_clouds/`: each camera's squaring and fit (`align.json`), its
anchor (`anchor.json`, `anchor.png`), what became of it once its picture stopped matching the anchor (`moved.json`),
its working pose when it left its spot (`follow.json`), a moving camera's track and its last keyframes' clouds
(`track.json`, `track/`), the pairs' measurements (`pairs.json`), and the scene's drawn points (`scenes/`), made on
request and kept until the scene changes. A measurement made from another picture or another spot is not used.

The 3D view brings a place's stored clouds up to date each time it asks for the scene: an older cloud with no floor
is stood up by its walls, and each camera's squaring and fit are made where they are missing or stale. That needs no
picture and no model, and costs little when nothing is stale, so clouds that are already made stand level and fitted
with Learn surroundings off.

### What a scene is good for

Each camera's view standing in the right place, with the room's own walls and floor where the plan has them. It is
not a whole room: a camera shows only what it saw, clouds that overlap meet with a seam, and cameras that share no
space are only as right as their spots. A cloud's walls aren't quite square to each other or to its floor, and
squaring lays the walls with the most points along the plan and leaves the rest out by that much; the view says when
a cloud's walls bend. A point more than 0.3 m outside every room of the plan is marked, and the view leaves it out
unless asked.

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
