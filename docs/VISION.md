# Vision pipeline

How camera pictures become rows, and why each stage is built the way it is.
[VISION-STREAMS.md](VISION-STREAMS.md) is the field reference for the streams named here. Paths are relative to
`src/liberational/`.

Two ideas run through every stage. Each camera is decoded once and every model reads that one decode, because
decoding is the cost the workers would otherwise each pay. And every stage writes coverage beside its output (the
spans it looked at), so a reader can tell "nothing happened" from "nobody looked".

```text
continuum-capture -> <camera>.video -> see-decode -> see-actions  -> pose, action, face, movement, walkby, animal
                                                  -> see-captions -> camera.caption, camera.objects, animal.grounding
                                                  -> see-objects  -> object.match
                                          see-actions' tracks -> see-zones -> zone.visit, presence.span
```

## Encoding at capture

Every local MJPEG camera is stored as H.264 unless its `[device.video]` has `encoding = { enabled = false }`.
Capture decodes each JPEG to NV12 and encodes on the GPU (`h264_nvenc`, or `h264_videotoolbox` on macOS), falling
back to `libx264` with a warning. A keyframe starts every `keyframe_interval_ms` (2000) and after any break, so
chunks close on decodable boundaries. RTSP cameras are stored as received.

## Shared decoding

`pack-see-decode` (`see/decoded_worker.py`) decodes each manifest camera once and shares the pictures through
`see/shared_frames.py:decoded_frames(camera, format=, every_ns=)`.

- H.264 decodes on NVDEC on `vision.shared_decode_gpu`. MJPEG and UYVY decode once on the CPU. A failed GPU decode is
  reported before a software decoder takes over; there is no silent fallback.
- Each reader asks for its own rate. A picture no reader takes is never downloaded or converted.
- NV12 stays in sensor orientation. BGR is upright, converted once for all readers, at most one picture per half
  second.
- Each `DecodedFrame` carries the source header (frame id and capture time), the decoder epoch, and `gap_before` for
  any loss. A slow reader sees a gap; it never stalls decode.
- The handoff is host memory (mmap slots under `paths.decoded`), not CUDA IPC.
- A camera decodes only while a reader is attached. Recording and the gateway read the encoded stream and do not
  depend on this service.

## see-actions

`pack-see-actions` (`see/actions_worker.py`) runs `see/actions.py`'s `ActionPipeline` on the machine's vision backend
(`see/vision_backend.py`): CUDA when available, Core ML and MPS on Apple Silicon, and the CPU only by name.

Per frame:

1. **Detect.** RF-DETR Small (`see/gpu_pose.py`) finds people and COCO animals. There is no NMS, so adjacent people
   are both kept.
2. **Track.** A ByteTrack-style tracker (`vision/crossings.py`) follows them.
3. **Pose.** RTMPose gives each tracked person 26 halpe26 joints. On a camera with a `fisheye` calibration, each
   person is first resampled into a pinhole view (`see/fisheye.py`).

From the tracks:

- **Skeletons.** Each tracked person's joints are a `pose.track` row. `action.coverage` spans the frames analyzed.
- **Actions.** ST-GCN++ with NTU RGB+D 120 weights (`see/stgcn.py`) scores each track's recent skeletons about twice
  a second. The top five classes are an `action.window` row; a kind held over consecutive windows becomes an
  `action.span`. MoViNet-A2 (`see/action_clips.py`), when the backend has it, scores person crops for reading,
  watching TV, phone use, and sleep.
- **Faces.** `see/head_faces.py` looks at each confirmed track's head about once a second, embeds the face (SCRFD,
  then AdaFace), and names the mean of the track's looks against the gallery. It writes `face.match`, `face.visit`,
  and `face.embedding`. [PEOPLE.md](PEOPLE.md#faces) has the naming rules.
- **Movement and walk-bys.** `see/action_movement.py` samples tracks into `movement.track` and, through the
  counting-line check and `see/walk_bys.py`, into `walkby.span`. A walk-by while the camera moved is flagged and not
  counted.
- **Animals.** `see/action_animals.py` tracks each species, confirms a track with SigLIP 2, and embeds confirmed
  boxes with DINOv2 (`animal.detection`, `animal.embedding`, `animal.coverage`).

Track ids are unique across restarts, and every reader joins on them as strings. A restart or a decode gap starts new
tracks and breaks coverage; nothing bridges frames that were never seen.

Embeddings from different models are never compared, even at equal dimensions. Each row names its model space.

## People on screens

`see/screens.py` keeps people shown on a TV or laptop out of the record. A new track is screen content when at least
80% of its box is inside a detected screen and the box is at most half the screen's area, or when it is inside an
owner-drawn zone of kind `screen`. A screen track is excluded from faces, actions, poses, movement, and overlays. Once
its box leaves the screen it is released for good. Recordings keep the original pixels.

The same module writes `state/screens_on.json` (when each drawn screen showed a moving picture).
`listen/media_tag.py` combines it with an audio tagger to mark broadcast speech as media, which the owner can
override.

## Captions and objects

`pack-see-captions` (`see/caption_worker.py`, `see/captions.py`) captions a view when its change gate says the view
changed, and now and then while it is idle. The gate compares the view with the one at the last caption, not with the
last frame, so slow drift adds up. Each caption describes the frame on its own. A skipped tick says `not_observed`; it
never repeats an earlier caption as fresh. Grounding (`camera.objects`) runs on a view's first look, after a caption
that movement asked for, and otherwise rarely, because it is the costliest call. `observation.status` tells a fresh
result from a cached, failed, or skipped one.

The captions are also an animal detector (`see/caption_animals.py`): a grounding that boxes an animal, or a fresh
caption that names one, gives `animal.grounding` rows and their embeddings, in the same shapes as see-actions'
animal streams. They are separate streams because a stream has one writer per camera.

`pack-see-objects` (`see/object_worker.py`, `memory/objects.py`) finds the objects the owner outlined: SIFT features
inside the saved mask, matched with RANSAC in each camera's newest frame and confirmed in two frames. It writes
`object.match`. With no object taught it decodes nothing.

Outlining is on demand. `vision/segmentation.py` runs EdgeTAM in one child process when the owner clicks an object,
draws a box, or asks to find objects in a frozen frame, so a model fault cannot take the dashboard down. The child
stays warm between requests and is stopped when idle, so an idle dashboard holds no GPU memory for it.
`objects.segmentation_timeout_s` bounds each request. A missing model is an error, never a box presented as a mask.

## Zones and came-into-view

`pack-see-zones` (`see/zones_worker.py`) follows see-actions' analyzed frames, people's and animals'. No model is
involved.

- For each owner-drawn zone (`zone.defined`) it writes `zone.visit` (a track's foot point inside the polygon: entry,
  last frame inside, dwell, and how it ended) and `zone.coverage`. An animal's visit carries its species; a pet's name
  is matched to its track by the reader.
- It writes `presence.span` episodes: one starts at the first analyzed frame with anyone tracked and ends once
  nobody was seen for `presence.absence_s` (10). A gap over `presence.gap_s` (30) censors it. An episode means
  someone was in view. It is not an entry, a headcount, or an identity.
- Its checkpoint is `state/see-zones.json`, so a restart never repeats an output.

`memory/zone_counts.py` counts visits of at least 10 seconds, joining split tracks within 30 seconds.
`memory/presence_counts.py` counts came-into-view episodes.

## Pets

`people/pets.py` keeps owner-named pets (`pet.profile`) and the owner's labels of animal sightings (`pet.sighting`).
A sighting is one animal track: the mean of its embeddings, from see-actions' `animal.embedding` and see-captions'
`animal.grounding_embedding`.

`people/pet_recognition.py` names a new sighting by appearance only for a species whose labels pass a held-out
check: each labeled sighting is recognized from the others, and too many wrong or undecided answers keep recognition
off. Until then a best match is shown as a suggestion and nothing answers for a pet by appearance. A species with
exactly one named pet has no second class to get wrong, so its labeled sightings are a gallery with a threshold of
their own. The module's docstring has the thresholds.

A species listed in `pets.classifier_species` (cats, as shipped) that has several pets and at least
`pets.classifier_min_labels` labels is also tried with a classifier over every labeled example: a calibrated logistic
regression (`pets.classifier_regularization`). The labels are split in time: the oldest half trains it, the next
quarter calibrates it, and the newest quarter tests it, with a gap between visits so that one visit is never on both
sides. Its operating point allows no error seen in calibration, and it replaces the nearest-example rule only when it
decides at least as many of the test sightings with no more of them wrong. Otherwise, and for every other species,
the nearest-example rule stands.

A sighting recognition names stays on the pet's card and is not offered for naming. One that waits carries its
reason. Each pet's card reports how well it is learned: over the labeled sightings of the last `pets.learning_days`
days that recognition was not taught from, the share named correctly, the dates those sightings span, how many
examples are usable, and how many labels are stranded with no example in the current embedding space. It is an
estimate on sightings the owner checked, not a record of what was named automatically.

Unlabeled sightings are offered for labeling for `pets.label_days`. A label keeps its embedding and its photo
(`pet.reference`, `pet.photo`), so it survives a reset, and `tools/restore_pet_references.py` can embed references
again after a model change, for every label whose photo is still kept. It cannot bring back a photo that is gone.

A pet's training images are the sightings labeled as it, listed on its card with the least like the rest first. An
example with no kept photo is listed without one. Removing some clears their labels, marks them not that pet, and
erases their `pet.reference` and `pet.photo` records in one write. Every reader keys its kept answers by the
knowledge store's generation, so recognition follows on the next read with no restart. [NAMING.md](NAMING.md) has
the steps, and [PEOPLE.md](PEOPLE.md#what-the-naming-answers-store) what Ignore and Block store for a sighting.
