# Vision pipeline

How camera pictures become rows. [VISION-STREAMS.md](VISION-STREAMS.md) is the field reference for the streams named
here. Paths are relative to `src/liberational/`.

```text
continuum-capture -> <camera>.video -> see-decode -> see-actions  -> pose, action, face, movement, walkby, animal
                                                  -> see-captions -> camera.caption, camera.objects
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

`pack-see-captions` (`see/caption_worker.py`) captions a view when its change gate says the view changed, and at
least every 30 seconds while idle. Each caption describes the frame on its own. A skipped tick says "not
re-observed"; it never repeats an earlier caption as fresh. Grounding (`camera.objects`) runs on a view's first look,
after a caption that movement asked for, and otherwise every 5 minutes, because it is the costliest call.
`observation.status` tells a fresh result from a cached, failed, or skipped one.

`pack-see-objects` (`see/object_worker.py`, `memory/objects.py`) finds the objects the owner outlined: SIFT features
inside the saved mask, matched with RANSAC in each camera's newest frame every 5 seconds and confirmed in two frames.
It writes `object.match`.

Outlining is on demand. `vision/segmentation.py` runs EdgeTAM in one child process when the owner clicks an object,
draws a box, or asks to find objects in a frozen frame. The child stays warm for 120 seconds, and
`objects.segmentation_timeout_s` bounds each request. A missing model is an error, never a box presented as a mask.

## Zones and came-into-view

`pack-see-zones` (`see/zones_worker.py`) follows see-actions' analyzed frames. No model is involved.

- For each owner-drawn zone (`zone.defined`) it writes `zone.visit` (a track's foot point inside the polygon: entry,
  last frame inside, dwell, and how it ended) and `zone.coverage`.
- It writes `presence.span` episodes: one starts at the first analyzed frame with anyone tracked and ends once
  nobody was seen for `presence.absence_s` (10). A gap over `presence.gap_s` (30) censors it. An episode means
  someone was in view. It is not an entry, a headcount, or an identity.
- Its checkpoint is `state/see-zones.json`, so a restart never repeats an output.

`memory/zone_counts.py` counts visits of at least 10 seconds, joining split tracks within 30 seconds.
`memory/presence_counts.py` counts came-into-view episodes.

## Pets

`people/pets.py` keeps owner-named pets (`pet.profile`) and the owner's labels of animal sightings (`pet.sighting`).
`people/pet_recognition.py` names a new sighting by appearance only for a species whose labels pass a held-out
check. Unlabeled sightings are offered for labeling for `pets.label_days` (14). A label keeps its photo, so
`tools/restore_pet_references.py` can embed references again after a model change.
