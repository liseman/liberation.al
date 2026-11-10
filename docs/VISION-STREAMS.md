# Vision streams

The records the vision workers (`src/liberational/see/`) write, for anything that reads them directly: the gateway's
overlays, Review, Ask, and the dashboard's pages. Every vision stream lives in the segment store. Owner teaching about what the cameras saw (zones, pet labels, activity names and
ratings) is in the knowledge store, not here: see [Owner records](#owner-records).

## Where the frames come from

`see-decode` (`see/decoded_worker.py`, `see/shared_frames.py`, `see/decoded_cache.py`) decodes each camera once
(NVDEC for H.264) and shares full-rate frames with the vision readers. A frame's id (`frame_id`) is the shared
decode's UUID for it. Frames are upright: capture's `flip` is applied before any worker looks.

## Common to every segment stream

- **Layout.** One writer per stream and camera, under `paths.state/segments/<stream>/<camera>/`. The camera is the
  directory, not a column; a record read back carries it as `cam`.
- **Columns.** `segment_log.py` declares each stream's typed columns once, with the codec that turns a record into a
  row and back (`segment_log.STREAMS`). see-actions' `pose.track` is declared in `see/actions.py` (`POSE_SCHEMA`).
- **Time.** Every row has `ts_ns` (UTC nanoseconds) and a writer-assigned `row_seq`. Span streams also have
  `end_ns`. A record read back has `ts_start` and `ts_end` in seconds. A point (one frame) has `end_ns == ts_ns`.
- **Ids.** Span-shaped rows have their own `id`; naming and ratings name a span by it.
  `action.window` and `action.coverage` have none: their record id is built from camera, track, and start.
- **No parents.** A row names no broker parents or span dependencies. Forgetting a person reaches the segment
  store through its own deletions.
- **Boxes** are normalized to the frame, `[x1, y1, x2, y2]`, top left to bottom right, as shares 0 to 1 of the
  frame's width and height, unless a stream says otherwise.
- **Track ids** are see-actions' own, camera- and run-local (they start from the run's start time, so they stay
  unique across restarts), never identities. Every stream that names a person track uses the same id.
- **Coverage.** Each analysis stage writes a `<stage>.coverage` stream beside its outputs: the spans it actually
  looked at. A count over a window without coverage is unknown, never zero.
- **Reading.** `segment_log.at(state).range(stream, start, end)` returns every row overlapping the window.

## see-actions: people

`see-actions` (`see/actions_worker.py`, `see/actions.py`, `see/gpu_pose.py`) runs on every configured camera
at its analysis rate (`vision.actions_fps`, overridden by `vision.actions_camera_fps`): it finds people and their halpe26 skeletons, tracks them (ByteTrack-style), and scores each track's
recent joints with ST-GCN++ (`see/stgcn.py`, `see/action_classes.py`). When the backend loads MoViNet-A2,
`see/action_clips.py` also scores person crops for reading, watching TV, phone use, and sleep.

RF-DETR Small supplies person and COCO animal boxes. Its decoder ranks the top 300 query/class pairs
without NMS; overlapping boxes alone do not discard adjacent people. The person score floor is 0.60,
and animals retain their separate verification path. The tracker and pose-confidence gate decide track births.

CUDA placement defaults to the highest-rate camera on the last visible GPU and the other cameras on the first.
Set `vision.actions_camera_gpus` in `pack.toml` to override individual manifest camera names, for example
`actions_camera_gpus = { front = 0 }` under `[vision]`. Indexes refer to devices visible to the process, starting at
zero; an invalid index fails startup before loading models. The empty default keeps automatic placement, and
non-CUDA backends ignore the mapping. Restart see-actions after changing it. When first pulling the new defaults
key, restart every config-reading service before setting the override so their cached schemas recognize it.

On uncalibrated frames, pose crops keep RTMPose's stock 1.25 padding and 3:4 aspect ratio, but black out other
person boxes scored at least 0.5 outside the target box. The target and a one-source-pixel interpolation margin
stay intact, including all overlaps. This reduces a padded crop's tendency to follow the neighboring person:
a bounded 50-still diagnostic recovered two additional starts, with no lost control starts or births on the
18 definite-empty views. These repeated static inputs do not measure temporal tracking or action accuracy.
People overlapping inside the target box remain unresolved; this mask cannot separate them. Calibrated fisheye
views keep their existing projection and crops.

| Stream | Fields |
|---|---|
| `pose.track` | One row per tracked person per frame: `ts_ns`, `frame_id`, `track_id` (int), `box` (normalized x1, y1, x2, y2, and the detector's score), `joints` (26 joints of x, y, and confidence, each times 10,000 as int16; a joint not placed is -1, -1, 0; see [halpe26](#the-halpe26-joint-order)), `model`. |
| `action.window` | One scored window of one track (ST-GCN++ every 0.5 s; MoViNet crop windows every 2 s): `ts_ns` and `end_ns` (the window's first and last frame), `track_id`, `classes` (the top 5 class indices, most likely first; NTU-120 for ST-GCN++, Kinetics-600 for MoViNet, as identified by `model`), `scores` (their probabilities), `model`. |
| `action.span` | A model action: consecutive qualifying skeleton windows, or a crop action retained by the clip model's hold/minimum-observation rules (`see/action_clips.py`). `id`, `track_id`, `subject_type`, `species`, `animals_visible`, `kind` (`sit_down`, `stand_up`, `pick_up`, `put_down`, `fall`, `drink`, `eat`, `phone_call`, `phone_use`, `read`, `watch_tv`, or `sleep`), `direction`, `supported` (intervals), `observations`, `interpretation`. Read back with `candidate` true and `camera_moved` false. |
| `action.coverage` | The frames see-actions analyzed: `ts_ns`, `end_ns`, `frames`, `model`. A gap in the frames ends every track and the coverage run. Presence answers read it. |

Person tracks born inside detected TV/laptop boxes or owner-drawn screen zones are withheld from these outputs
until they leave the screen test. Real tracks that later overlap a screen remain real. The original recording is
unchanged; see [Architecture](ARCHITECTURE.md#people-shown-on-screens) for the filter boundary.

see-actions keeps no checkpoint: a restart starts new tracks from the live frames, as a gap would.

## see-actions: movement and walk-bys

`see/action_movement.py` samples see-actions' tracks about every 0.2 s for movement fragments and crossings.

| Stream | Fields |
|---|---|
| `movement.track` | `id`, `track_id`, `version`, `points` (`[time, x, y]` foot anchors), `supported` visible intervals, `gaps`, `entry` and `exit` boundaries, `direction`, `camera_motion`, `finished`, `closure`. A boundary has `status`, `reason`, and, when known, the image `edge` and uncertainty `time: [lo, hi]`; an unknown boundary has a reason and no invented time. Overlapping fragments are unioned for dwell and deduplicated by track and boundary, never added as separate arrivals. |
| `movement.coverage` | `id`, `version`, `frames`, `supported`, `gaps`, `tracking` (`complete` or `partial`), `camera_motion`, `finished`, `closure`. Missing tracked or stable-camera evidence leaves coverage unknown. |
| `walkby.span` | One walk-by, kind `crossing`: `id`, `track_id`, `version`, `direction`, the crossing `line` (vertical, horizontal, or a segment), `crossings` and `tracks` counts, `followed`, `camera_moved`, and whether camera motion was unknown. A walk-by while the camera moved, or with motion unknown, isn't counted. |
| `walkby.coverage` | A stretch where walk-bys could have been seen: `id` and `frames`. Failed or missing inputs split it. |

## see-actions: faces

`see/head_faces.py` crops each confirmed track's head from pose joints 0 to 4 about once a second and runs
SCRFD detection and AdaFace IR50 RGB recognition on the crop. The track's name is what the gallery names the mean of its looks so far,
so the pose track is the identity over time. A look with insufficient nose confidence or
feature norm writes only `face.signal`: it does not name the track, update a face group, or save a face photo.
The saved held-out check can disable automatic naming; see [Architecture](ARCHITECTURE.md#see-actions-the-vision-pipeline).

| Stream | Fields |
|---|---|
| `face.signal` | One looked-at head: `id`, `match_id` (a fresh unattached ID when the look is set aside), `head` (five joint confidences), `frontal` (landmark frontalness), and `norm` (raw AdaFace feature norm). |
| `face.match` | One look at a tracked head: `id` (also its `detection`, which visit shots name), `track_id` (the `pose.track` id), `frame_id`, `bbox` (the face, normalized), `det`, `name` (None while unknown), `sim`, `cluster` (`Face N` when unnamed), `quality`. |
| `face.visit` | A named person's looks on one camera, joined while each follows the last within 10 s (`see.faces.VisitBuilder`): `name`, `best_sim`, `samples`, `shots` (its three best looks, best first: `{detection, ts, bbox, quality, sim}`). It spans the first look to the last. |
| `face.embedding` | One per `face.match`, followed by the vector worker for nearest-face search: `match_id`, `track_id`, `name` (display only), `person` (the named person's id), `group` (an unknown face's group subject), `vector` (512 floats, unit AdaFace IR50 RGB), `model`. |

## see-actions: animals

`see/action_animals.py` samples the COCO animal classes from the same detector pass. A box in a person's head is
dropped, each species has its own tracker, and a track is published only once confirmed (persistence, a confident
peak, and the image check).

| Stream | Fields |
|---|---|
| `animal.detection` | One confirmed box: `ts_ns`, `frame_id`, `track_id` (`<run>:<species>:<n>`), `species`, `box` (normalized x1, y1, x2, y2, and score), `verified` (the track passed the image check), `model`. |
| `animal.embedding` | One appearance embedding of a confirmed track: `ts_ns`, `track_id`, `species`, `box`, `vector` (384 floats, unit length, DINOv2 ViT-S/14), `model`. Pet recognition reads it. |
| `animal.coverage` | The spans of samples looked at: `id` and `frames`. A covered span without detections saw no animal; a gap or a stop splits it. |

## see-zones and presence

`see-zones` (`see/zones_worker.py`, `see/zones.py`) follows see-actions' `pose.track` and `action.coverage`
(`see/tracked_frames.py`) against the owner's drawn zones. It checkpoints its input cursor and each writer's next
`row_seq` in `paths.state/see-zones.json`.

| Stream | Fields |
|---|---|
| `zone.visit` | A tracked subject's visit to a zone: `id`, `zone` (its id), `zone_name`, `definition` (the zone set's version id), `subject_kind` (`person`), `track_id`, `label`, `dwell_s` (first to last frame with the box's bottom centre inside), `frames`, `entry_seen` (seen outside first), `exit_seen`, `ended` (`left`: seen outside for 1.5 s; `lost`: not detected for 120 s; `censored`: no coverage for 10 s; `zones_changed`: the owner saved new zones), `version`. Read back with `candidate` true and its `interpretation`. Written when the visit ends; it spans entry to the last frame inside. |
| `zone.coverage` | Where zone visits could have been seen: `id`, `subject_kind`, `definition`, `zones` (the zone ids in force), `frames`. A failed frame, or one with an untracked box inside a zone, is not coverage. Coverage is published only up to the start of the camera's earliest open visit, so it never runs ahead of a visit not yet written. |
| `presence.span` | A time someone was in view (`memory/presence_episodes.py`): `id`, `frames`, `max_people`, `tracks` (distinct track ids, never identities), `left_censored` (began right after unwatched time), `right_censored` (unwatched time ended it). It spans the first to the last frame with anyone tracked, and is written once nobody was seen for `presence.absence_s` of analyzed frames. |

## see-captions and see-objects

`see-captions` (`see/caption_worker.py`, `see/captions.py`) describes views with the VLM
and grounds the objects in view; `see-objects` (`see/object_worker.py`) finds the owner's remembered objects.

| Stream | Fields |
|---|---|
| `camera.caption` | One looked-at frame: `id`, `frame_id`, `frame_seq`, `yaw_deg` (0), `caption`, `objects` (the grounded boxes, as `camera.objects` has them; None when grounding is off), `source`, `synthetic`, `gated` (the change gate skipped the VLM), `observation` (`status`: `fresh`, `not_observed`, or `error`; `observed_at`, `backend`, `prompt_version`), `change` (`mad`, `frac`: how far the view moved since the last caption), `brightness` (`mean` 0 to 255, `dark_fraction`), `ms` (`vlm`). A repeat has `unchanged` true and `since` (the repeated caption's time). `deferred` true marks a repeat whose view changed but whose caption interval (`vision.network_caption_every_s`, `[vision.caption_every_s]`) held the VLM back. `faces` is always None. |
| `camera.objects` | The boxes in a camera's view at a frame, which the gateway draws over Live: `id`, `frame_id`, `frame_seq`, `objects` (each `label`, `bbox` in frame pixels, `bbox_norm`, and `ts`, the grounding time), `source` (the VLM backend), `observation` (`status`: `fresh`, `cached`, `failed`, or `unevaluated`; `observed_at`, `attempted_at`, optional `reason`). A fresh empty list is grounder output; an unknown status is not evidence of absence. |
| `object.match` | A visual match of a remembered object, a moment: `id`, `object_id`, `name`, `bbox` (source pixels), `score`, `frame_id`, `frame_seq`, `evidence` (the frame as a small JPEG, base64). Read back with `kind` `visual_match`. |

## Owner records

What the owner taught about the video lives in `knowledge.sqlite3` (`knowledge.py`), outside the catalog reset:

- `zone.defined`: a camera's whole set of drawn zones (`see/zones.py`): `cam`, `zones` (each `{"id", "name",
  "polygon"}`: a stable `zone-<hex>` id, a name unique on that camera ignoring case, and 3 to 32 normalized `[x, y]`
  corners of a simple polygon covering at least 0.01% of the frame; at most 16 zones; optional `kind` is `area` or
  `screen`), and `source` (`owner`). Screen zones mask screen-content tracks and are excluded from ordinary
  zone-visit counting. The
  newest per camera holds. `POST /zones` takes the id of the version edited, so a save over a newer version is
  refused.
- `activity.named` and `activity.feedback`: the owner's names and Right/Wrong ratings on walk-bys and model actions,
  keyed by the span's id over its time, with `activity`, `name` or `verdict`, `source` (`owner`), and the span's
  `kind`, `cam`, and `stream`. The latest per span wins. A Wrong span is left out of Ask's counts.
- `pet.profile` (`people/pets.py`): a named pet, with `pet_id`, `name`, `species`, `active`, and `source`.
- `pet.sighting` (`people/pet_recognition.py`): the owner's label for one animal track: `cam`, `track_id`,
  `species`, `label` (a named pet's id, `other`, or empty when cleared), `active`, `source`, and `sighting_start`.
  The newest per track wins.
- `object.sighting`: the owner's own crops of remembered objects.

## The halpe26 joint order

`pose.track` joint *i* is:

0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear, 5 left_shoulder, 6 right_shoulder, 7 left_elbow,
8 right_elbow, 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip, 13 left_knee, 14 right_knee, 15 left_ankle,
16 right_ankle, 17 head, 18 neck, 19 hip, 20 left_big_toe, 21 right_big_toe, 22 left_small_toe, 23 right_small_toe,
24 left_heel, 25 right_heel.

The bones (`frontend/src/dashboard/pose/skeleton.ts`, `BONES`): 15-13, 13-11, 11-19, 16-14, 14-12, 12-19, 17-18,
18-19, 18-5, 5-7, 7-9, 18-6, 6-8, 8-10, 1-2, 0-1, 0-2, 1-3, 2-4, 15-20, 15-22, 15-24, 16-21, 16-23, 16-25.

"Left" and "right" are the person's own.
