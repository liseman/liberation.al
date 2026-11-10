# Video segmentation for Liberational: research and recommendation

> Historical research from October 2026. Recommendations, model availability,
> licensing notes, and implementation paths below reflect that date and have not
> been revalidated. See [Architecture](../ARCHITECTURE.md) for the current model
> pipeline; this is not a deployment guide.

Date: 2026-10-23. Research only; nothing installed, nothing in the repo changed.

## TL;DR

Use **SAM 3** (Meta, November 2025; SAM 3.1 update March 2026) as the open-vocabulary "find and track every X" engine, but run it **offline in `vision/backfill.py` and on demand**, not on every live frame: it is an 848M-parameter model that Meta itself only claims runs near real time for about five objects on an H100/H200, and nobody has published a fast consumer-GPU or Mac number. It takes text prompts directly, so the labels Qwen3-VL already produces become its prompts. For the live path, if we want it at all, use **EdgeTAM** (Apache 2.0, SAM 2 derived, in Hugging Face transformers with streaming) seeded with the boxes `vision/objects.py` already gets from Qwen3-VL, at about 5 fps per camera. The biggest real wins are (1) longer, more reliable "when did I last see X" answers by tracking bodies when faces turn away and by adding object sightings to the backfill, and (2) a cheap, durable fix for the live overlay's stale boxes once the pack is actually worn. Squat form analysis is mostly **not** a segmentation problem, and the pack's outward-facing cameras mostly can't see the wearer's own legs; see "What is not worth it".

## Context that shaped the advice

- The original comparison considered Apple Silicon and NVIDIA CUDA machines with outward-facing fisheye cameras. Its measurements do not establish performance on other hardware.
- `mlx-vlm` is already a macOS dependency in pyproject.toml, and it ships a SAM 3 port. That makes SAM 3 on the Mac nearly free to try.
- `vision/objects.py` grounds "notable objects, not people" with Qwen3-VL every 30 s per view at most (about 2.6 s per call) and carries boxes forward on the assumption that "the cameras are mostly still". That assumption breaks the moment the pack is worn and walking.
- `vision/live.py` runs InsightFace on every frame (about 11 ms at 720p on a 3090) with an IoU plus embedding tracker; face tracks die after 1 s unseen.
- `vision/backfill.py` samples recordings at `backfill.sample_fps = 2.0` under `max_gpu_share = 0.25` and pauses when the GPU is over 70 % busy.

## 1. The landscape (October 2026)

### Meta's SAM family

| Model | Released | Prompts | Video | Notes |
|---|---|---|---|---|
| SAM 3 | 2025-11-19 | text (noun phrase), image exemplar, points, boxes, masks | yes: detector plus SAM 2 style memory tracker sharing one Perception Encoder backbone | Finds *all* instances of a concept, with identities across frames. 848M params. |
| SAM 3.1 "Object Multiplex" | 2026-03-27 | same | yes, objects processed jointly in fixed-size buckets | Drop-in checkpoint; faster at many objects. Official repo only, no transformers integration yet. |
| SAM 3 Tracker (PVS) | 2025-11 | points, boxes, masks (one object per prompt) | yes | "Updated SAM 2 Video with the same API", SAM 3 backbone. In transformers as `Sam3TrackerVideoModel`. |
| SAM 3D Objects / SAM 3D Body | 2025-11-19 | image plus optional mask/keypoints | no (single image) | 3D reconstruction and 3D human mesh; relevant to form analysis, not tracking. |
| SAM 2 / 2.1 | 2024-07 / 2024-09 | points, boxes, masks | yes, memory bank, streaming | Apache 2.0. Tiny 38.9M to Large 224.4M. Repo quiet since Dec 2024. |

No SAM 4 or later SAM release turned up as of this writing; the latest Meta release in the line is SAM 3.1 (March 2026).

### Efficient trackers (SAM 2 lineage)

- **EdgeTAM** (Meta, CVPR 2025): SAM 2 with a 2D Spatial Perceiver compressing the memory bank. 16 fps on an iPhone 15 Pro Max, 150.9 fps on an A100 with torch.compile. J&F: DAVIS17 87.7, MOSE 70.0, SA-V val 72.3, SA-V test 71.7. Apache 2.0. In transformers (`EdgeTamVideoModel`, streaming supported). CoreML export script exists but only for image encoder, prompt encoder, and mask decoder (no memory modules).
- **EfficientTAM** (Meta, Dec 2024): plain ViT encoder plus efficient memory. About 2x faster than SAM 2 B+ on A100, about 10 fps on iPhone 15 Pro Max, "comparable" to SAM 2 B+. Apache 2.0. MPS support added Jan 2025; no update since.
- **SAM 2.1 Tiny/Small**: the reference small trackers. Tiny: 91.2 fps on A100 (official), but a user measured only 25 to 26 fps on an RTX 3090 for one object (sam2 issue #448). SA-V test J&F about 76.5 for Tiny, noticeably better than EdgeTAM's 71.7.
- **SAM2Long, SAM 2++, PanoSAM2**: research add-ons (training-free memory trees for long video; a 360 degree distortion-aware variant). Interesting, but not things to depend on.

### Efficient text-promptable (SAM 3 lineage)

- **EfficientSAM3** (University of Bristol): SAM 3 distilled into RepViT, TinyViT, and EfficientViT students (about 89 to 95M params) with MobileCLIP text encoders; also "SAM3-LiteText" (full SAM 3 vision encoder, small text encoder, 550 to 631M). Code Apache 2.0, but the weights derive from SAM 3, so treat them as SAM License. Stage 3 checkpoints landed 2026-06-11; **no published accuracy or latency numbers** and the video memory weights are still on the to-do list. Watch, don't build on it yet.
- MobileSAM, EdgeSAM, EfficientSAM (v1): image only, point/box prompts, superseded. Not useful for video.

### Non-SAM alternatives worth knowing

- **RF-DETR-Seg** (Roboflow, Jan 2026, ICLR 2026): real-time instance segmentation, Apache 2.0, 3.4 to 21.8 ms on a T4 with TensorRT FP16, COCO classes only (closed vocabulary). Same repo has an Apache 2.0 keypoint (pose) model in preview. Fast and clean licensing, but it can't take "barbell" or "red mug" from our VLM without fine-tuning.
- **Ultralytics** (YOLO11/YOLO26 seg and pose, YOLOE, their SAM wrappers): AGPL-3.0. Avoid for a product we might sell unless we buy their enterprise license.

### Which take prompts straight from our VLM

- **Text labels** ("guitar", "tv", "barbell"): SAM 3 / 3.1 (and EfficientSAM3 later). This matches objects.py's output exactly: its `label` fields become SAM 3 concept prompts.
- **Boxes** (Qwen3-VL `bbox_2d`, converted from the 0 to 1000 scale): SAM 3 Tracker, SAM 2.1, EdgeTAM, EfficientTAM. One box prompt per object, then propagate.
- SAM 3 also accepts a box as a positive exemplar ("more things like this one"), which is handy when the VLM's label is vague.

## 2. Serious candidates in detail

### SAM 3 / SAM 3.1

- **License**: Meta's custom "SAM License". Royalty-free, worldwide, permits use, modification, and distribution; you must ship a copy of the agreement with redistributed weights or derivatives; no use prohibited by trade controls (military, warfare); a clause against reverse engineering; Meta may amend it; termination means deleting the materials. It does not forbid commercial use, and secondary sources describe it as allowing commercial applications. Weights are gated on Hugging Face (accept terms, share contact info). Before selling, get a lawyer to read it, but it is not a non-commercial license like Sapiens v1's CC BY-NC.
- **Size**: 848M params; about 3.4 GB fp32 download; MLX 8-bit 1.04 GB, 5-bit 0.72 GB.
- **Published speed**: about 30 ms per image with 100+ objects on an H200; video "near real time for about five concurrent objects"; 16 fps (SAM 3) to 32 fps (SAM 3.1) on one H100 for medium object counts; 7x at 128 objects with 3.1. Native input is 1008 px (configurable down to 560 px, with an accuracy warning).
- **Measured by others, less rosy**: 5 to 6 fps on a 1080p video on an H200 (sam3 issue #425, open, no maintainer reply), a 4090 with ONNX Runtime plus TensorRT EP at about 1.1 s per image (issue #424), a user on an RTX 3060 reporting the SAM 3.1 video example took over an hour versus 5 minutes for SAM 3 (issue #542), and a "memory not released in video predictions" issue (#305).
- **RTX 3090 estimate (unmeasured)**: the 3090 has roughly a tenth of an H100's dense bf16 throughput and no FlashAttention 3 (Hopper only), so expect a few frames per second for detect plus track at 1008 px, better at 560 px or with detection every Nth frame. Secondary sources put VRAM at about 4 GB fp16 and 8 GB fp32; plan for 6 to 10 GB with a video session. This is exactly what the first benchmark has to pin down.
- **Apple Silicon**: `mlx-vlm` has a full port (`mlx_vlm.models.sam3`: text and box prompts, `Sam3VideoPredictor`, a `track_video` CLI that detects every 2nd frame by default, and a realtime webcam mode). Its README reports about 11 fps at 224 px and about 8 fps at 336 px, plus about 30 ms per extra prompt (no chip named). A community PyTorch MPS run on an M4 Max measured about 0.8 s per frame at fp16 for video tracking at full resolution, peak 3.7 to 4.7 GB. Community CoreML (SAM 3.1 FP16) and ONNX exports exist but are undocumented or license-encumbered (Embedl's TensorRT build forbids hosted redistribution, and warns newer TensorRT versions produce wrong masks).
- **How video works**: per frame the detector finds all instances of each prompt; a SAM 2 style tracker propagates each masklet from memory (conditioning frames plus recent confident frames); detections and tracks are matched by IoU; unmatched detections start new tracks; tracks are periodically re-conditioned on high-confidence detections (every 16 frames by default) to stop drift. Pre-loaded video mode uses a 15-frame "hotstart" buffer to drop spurious and duplicate tracks; **streaming mode disables hotstart** and gives more false positives (transformers docs). So SAM 3 is at its best on recorded segments, which is exactly backfill.
- **Maintenance**: active (3.1 in March 2026); transformers integration for SAM 3 (not 3.1) is maintained by Hugging Face; open performance issues with little maintainer response.

### EdgeTAM

- **License**: Apache 2.0 (code and checkpoints). Cleanest option for a product.
- **Size and speed**: small Hiera-tiny class encoder plus a perceiver-compressed memory; 150.9 fps A100 compiled, 16 fps iPhone 15 Pro Max uncompressed. 22x faster than SAM 2 per the authors. RTX 3090 estimate (unmeasured): tens of fps per stream, comfortably enough for 3 cameras at 5 to 10 fps each.
- **Quality**: a few J&F points below SAM 2.1 Tiny and B+ (SA-V test 71.7 versus about 76.5 for 2.1 Tiny). For "keep a box on the TV while I walk past" that is fine.
- **Video**: SAM 2 memory bank, point/box/mask prompts, multiple objects, streaming in transformers (`EdgeTamVideoModel` with `Sam2VideoProcessor`, frame-at-a-time `model(inference_session, frame=...)`).
- **Mac**: PyTorch MPS through transformers (same code as CUDA); the CoreML export lacks the memory modules, so a CoreML video tracker is a porting job.
- **Maintenance**: original repo is small and quiet (21 commits), but the transformers port is the maintained path.

### SAM 2.1 Tiny / Small

- Apache 2.0, 38.9M / 46M params, the best-understood tracker, `vos_optimized=True` full-model compile since Dec 2024. Apple publishes CoreML SAM 2.1 (image encoder, prompt encoder, mask decoder only, so again no video memory). Measured 25 to 26 fps on a 3090 for one object uncompiled. Upstream repo has been quiet since Dec 2024; transformers carries `Sam2VideoModel`. Choose it over EdgeTAM if EdgeTAM's masks prove too sloppy on fisheye frames.

### SAM 3 Tracker (box/point prompts on the SAM 3 backbone)

- Same SAM License and roughly the same cost per frame as SAM 3's tracker, better VOS scores than SAM 2.1 L (MOSEv2 60.3 versus 47.9). The right choice when quality matters more than speed and we already have boxes, for example re-tracking a person body in backfill.

### EfficientSAM3 (watch list)

- The thing we'd actually want on the Mac live path (text prompts at a tenth the size), but no published numbers and no released video memory weights as of 2026-06-11. Re-check in a few months.

## 3. Uses for us, ranked by value

1. **Body tracks that outlive the face (goal 2, also goal 1)**. Today a sighting exists only while InsightFace sees a face, so someone who turns away, bends over, or walks off sideways drops out, and visits fragment. In backfill, seed a box tracker (SAM 3 Tracker, or EdgeTAM for speed) from each confident named face with a body box (or SAM 3's "person" prompt and pick the instance containing the face), propagate forward and backward at the sample rate until the tracker loses it, and attach the face's identity to the body track. That gives longer, more accurate visits in `sightings.sqlite3`, and "when did I last see Bob" answers that don't stop at the last frontal frame. It also gives `people/identity.py` more co-occurrence time to link voice and face. Highest value, offline only, uses the GPU budget we already have.
2. **Object sightings in backfill ("when did I last see my keys", "was the TV on")**. Run SAM 3 on sampled frames with a per-window vocabulary pulled from caption history (`camera_history.sqlite3` notable objects) plus a fixed list the owner cares about, store tracks per label, and let `memory/history_qa.py` answer object questions from records instead of captions. This generalizes goal 2 beyond people at modest cost. It is also the honest answer for "what weight" when plates are visible to a camera.
3. **Stable live object boxes between grounding calls**. Once the pack moves, carried-forward boxes are wrong within a second. EdgeTAM seeded from each grounding result keeps boxes (and masks) on the objects between 30 s calls, lets `scene.json` boxes stay correct, and gives a confidence signal ("the TV left view") that can trigger an earlier grounding call instead of waiting for MIN_S. Medium value now (cameras on a desk), high once the hardware is worn.
4. **Privacy redaction at export time**. If this is sold, bystanders matter. InsightFace face boxes already give a cheap face blur on every frame; full-body masks (SAM 3 "person") are worth it only when exporting or sharing a clip, run offline on that clip. On the Mac, Apple Vision's person segmentation and person instance mask requests are free, on-device alternatives. Medium value, low urgency, small code.
5. **Masks for a nicer overlay**. Mask outlines look better than boxes on the dashboard. Cosmetic; comes for free with item 3, not worth doing alone.
6. **Person masks to help pose estimation for squats (goal 3)**. Low value for segmentation specifically; see below.

### What is not worth it (or not yet)

- **Segmentation does not solve squat counting or form.** Pose models (top-down RTMPose/ViTPose, bottom-up RTMO, RF-DETR Keypoint) find people and joints on their own; a mask adds little to 2D keypoints. Where masks help is optional prompting of SAM 3D Body for 3D joint angles, an offline nicety.
- **The pack can't see its wearer.** The three cameras face outward from the pack, so "how many squats did I do" and "how was my form" will usually not have the wearer's legs in frame. Options that work: (a) rep counting from motion the pack does see: the Teensy IMU (not started) or camera ego-motion (the whole scene bobs vertically in all three cameras once per rep, which plain optical flow catches without any model), (b) form analysis only when a second, fixed camera sees the lifter (a phone or a gym camera added as a source), or a mirror, and (c) the case where the wearer is coaching someone else, where normal pose estimation on that person works. Weight comes most reliably from speech ("putting on 135"), as ROADMAP.md already says; SAM 3 on "weight plate" plus a VLM read of the plate crop is a fallback when a camera sees the bar.
- **Running SAM 3 on every live frame of three cameras.** Not on a 3090, not on an M5 Pro, not alongside InsightFace, the VLM, and TTS sharing the GPU.
- **SAM 3.1 specifically, for now.** Its gains are at many objects (2x at medium counts, 7x at 128); we track a handful. It needs the official repo (no transformers, no MLX), and one user reports it far slower than SAM 3 on a consumer GPU. Benchmark it on CUDA backfill later; default to SAM 3 through transformers so Linux and Mac share code shape.
- **Our own CoreML port of a video tracker.** None of the CoreML exports include the memory modules; MPS through transformers, or MLX through mlx-vlm, gets us running on the Mac without a porting project.

### Pose models that pair well (for when goal 3 has a camera that sees the lifter)

- **CUDA and Mac via ONNX**: RTMPose / RTMW (whole body) through `rtmlib`, Apache 2.0, needs only onnxruntime (CUDA EP here, CoreML EP on the Mac). Upstream activity has slowed (last push Feb 2025), but the models are stable. RF-DETR Keypoint (Apache 2.0, 71.8 COCO AP, 9.7 ms T4) is a newer option, still marked preview.
- **Mac native**: Apple Vision `VNDetectHumanBodyPoseRequest` (2D) and `VNDetectHumanBodyPose3DRequest` (17 joints in meters, most prominent person), free and on the Neural Engine. Good enough for depth and back angle checks from a side view.
- **Offline 3D**: SAM 3D Body (SAM License, 631M to 840M params, single image, accepts masks and 2D keypoints as prompts). Use it on a few frames per rep (bottom position) for knee and hip angles, not on every frame.
- **Avoid**: YOLO-pose (AGPL), Sapiens v1 (CC BY-NC; Sapiens2 has its own license whose terms I couldn't find).

### Fisheye caveat

All of these models are trained mostly on rectilinear images. I found no evaluation on 160 degree fisheye; PanoSAM2 reports plain SAM 2 masks degrade under strong panoramic distortion. Test on raw frames versus per-region undistorted crops (`cv2.fisheye`) in the prototype; objects near the image edge are where it will show.

## 4. Recommendation

**Models**
- Backfill and on-demand: **SAM 3** (transformers `Sam3VideoModel` / `Sam3TrackerVideoModel` on CUDA; `mlx_vlm.models.sam3` on the Mac), bf16, 1008 px first and 560 px as the speed fallback. Revisit SAM 3.1 on CUDA after the baseline exists.
- Live (optional, second): **EdgeTAM** (transformers `EdgeTamVideoModel`, streaming; MPS on the Mac), box prompts from Qwen3-VL grounding. SAM 2.1 Tiny as the fallback if EdgeTAM's quality on fisheye disappoints.
- Pose (only once a camera sees the lifter): rtmlib RTMPose on both platforms, Apple Vision on the Mac.

**Where and at what rate**
- Live: EdgeTAM at about 5 fps per camera (every 6th frame at 30 fps), on its own thread so face detection cadence is untouched, on GPU 1 here. Reseed from every grounding result; drop a track when the tracker's object score stays low for about 1 s; when most tracks die (the view changed), ask for a grounding call early.
- Backfill, budget arithmetic: at `max_gpu_share = 0.25`, a day gives about 6 GPU-hours (21,600 s). If SAM 3 costs about 0.2 s per frame on a 3090, that's about 108,000 frames a day, about 0.4 fps across three cameras recording 24 h. So:
  - object sightings at 0.2 to 0.5 fps per camera, only on windows where captions changed or people were present;
  - body tracks at the existing 2 fps, only around face sightings (seeded, not detected everywhere);
  - dense 5 to 10 fps only for event windows (an "exercise" span flagged by captions or speech), the input goal 3 needs anyway.
- The Mac is the tighter budget (one GPU shared with the MLX VLM, ASR, and TTS); measure there before turning anything on by default.

**Integration plan in terms of our files**
1. `vision/segment.py` (new): one small interface over both backends: `Segmenter.seed_boxes(frame, boxes, labels)`, `Segmenter.seed_text(frame, prompts)`, `Segmenter.step(frame) -> [{track, label, box, score, mask_rle}]`, plus `run_clip(path, start, end, prompts, fps)` for backfill. CUDA: transformers; macOS arm64: mlx-vlm for SAM 3, transformers on MPS for EdgeTAM. Lazy imports, like `vision/faces.py`.
2. `vision/objects.py`: grounding results become seeds. Replace "carry forward until TTL" with "track until lost"; keep `parse`, `to_pixels`, and `draw` as they are. A later tracker would return tracked boxes when it is running, and the last carried boxes otherwise. `draw` gains optional mask contours.
3. `vision/live.py`: `FastPath` takes an optional segmenter and runs it from a separate thread on every Nth frame (same pattern as `stream_loop`), so a slow segmentation step never delays face detection; `annotate` draws tracked objects. Stats (`det_ms`) get a `seg_ms` sibling so the dashboard shows the cost.
4. `vision/backfill.py`: two new passes after the face pass, each with their own progress column per segment, both under the existing `Throttle`: (a) body tracks seeded from that segment's named face sightings; (b) object sightings with SAM 3 text prompts from that window's caption vocabulary. Long segments go through in 30 to 60 s windows so video session memory stays bounded (see sam3 issue #305).
5. `vision/sightings.py`: `bodies` and `objects` tables (ts, cam, segment, track, label or person, normalized box, score, optional mask RLE), and visits rebuilt from faces plus bodies.
6. `memory/history_qa.py`: object questions ("when did I last see my keys") query the objects table first, then captions.
7. `config/defaults.toml`: a `[segment]` section (enabled, live model, backfill model, live fps, backfill fps, device, resolution), defaulting to off. A `segment` optional dependency in `pyproject.toml` for Linux (torch, transformers); the Mac already has mlx-vlm.

**What to prototype first, and how to measure it**

Prototype 0, a benchmark script outside the services (a `tools/` script or the scratchpad), on 3 to 5 real recorded segments from `state/video/`:
- Models: SAM 3 (transformers, bf16, 1008 and 560 px, detect every 1 and every 4 frames), SAM 3 Tracker with box seeds, EdgeTAM, SAM 2.1 Tiny; SAM 3.1 from the official repo if it installs on Python 3.14 (the repo asks for 3.12+ and torch 2.7+; a separate venv is fine).
- Object counts 1, 4, and 8 (seeded from real Qwen3-VL grounding output on those frames).
- Report p50 and p95 ms per frame, peak VRAM (`torch.cuda.max_memory_allocated`), and the effect on the live path when run concurrently on the same GPU (InsightFace `det_ms` and caption rate from the existing stats) and on GPU 1.
- Same script on the Mac mini: mlx-vlm SAM 3 at 8-bit and 5-bit, 336, 560, and 1008 px; EdgeTAM on MPS; and the captioner's rate while it runs.

Quality, without a labeling project:
- Tracking: use the next grounding call as free ground truth. For each object seeded at t, compare the tracked box at t + 30 s with the fresh Qwen3-VL box of the same label (IoU over 0.5 counts as kept). Report kept rate, drift, and ID switches, on desk footage and on walking footage.
- Body tracks: on 20 hand-checked spans where a named face disappears, check whether the extended track stays on the right person; report the median visit gain and the wrong-person rate. Ship only if wrong-person is near zero.
- Object sightings: hand-check 100 random hits in the /review player for precision; ask 10 real "when did I last see X" questions and compare with captions-only answers.
- Fisheye: rerun the tracking numbers on undistorted crops for objects near the image edge.

Decision gates: SAM 3 backfill goes in if it fits the 0.25 share at a useful rate and object-sighting precision is high (over 90 % on the sample). EdgeTAM live goes in if it costs the face path under about 2 ms per frame and keeps boxes on target noticeably better than carry-forward on walking footage. Otherwise stop at backfill.

## Sources

- SAM 3 repo, news, requirements, 848M params: https://github.com/facebookresearch/sam3
- SAM License text: https://github.com/facebookresearch/sam3/blob/main/LICENSE
- SAM 3.1 release notes (Object Multiplex, benchmarks): https://raw.githubusercontent.com/facebookresearch/sam3/main/RELEASE_SAM3p1.md
- Meta blog (SAM 3 and 3.1 speed claims): https://ai.meta.com/blog/segment-anything-model-3/
- SAM 3 paper: https://arxiv.org/abs/2511.16719
- SAM 3.1 on Hugging Face (gated, no transformers integration): https://huggingface.co/facebook/sam3.1
- transformers SAM 3 Video (streaming, hotstart caveat, config): https://huggingface.co/docs/transformers/en/model_doc/sam3_video
- transformers SAM 3 Tracker Video: https://huggingface.co/docs/transformers/en/model_doc/sam3_tracker_video
- transformers EdgeTAM Video: https://huggingface.co/docs/transformers/en/model_doc/edgetam_video
- sam3 issue #425 (5 to 6 fps on H200): https://github.com/facebookresearch/sam3/issues/425
- sam3 issue #424 (about 1.1 s per image on 4090 with TensorRT EP): https://github.com/facebookresearch/sam3/issues/424
- sam3 issue #542 (SAM 3.1 slow on RTX 3060): https://github.com/facebookresearch/sam3/issues/542
- sam3 issue #305 (video memory not released): https://github.com/facebookresearch/sam3/issues/305
- MLX SAM 3 model card: https://huggingface.co/mlx-community/sam3-8bit
- mlx-vlm SAM 3 module (fps at 224 and 336 px, video CLI): https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/sam3
- M4 Max SAM 3 MPS measurement (community): https://github.com/VolksRat71/sam-ui/pull/57 and https://github.com/VolksRat71/sam-ui/pull/60
- SAM 3.1 CoreML FP16 (community): https://huggingface.co/dustbeast/SAM3.1-CoreML-FP16
- Embedl SAM 3 TensorRT build (license, TensorRT caveat): https://huggingface.co/embedl/sam3
- SAM 2 repo (2.1 sizes and A100 fps, Apache 2.0): https://github.com/facebookresearch/sam2
- sam2 issue #448 (2.1 Tiny at 25 to 26 fps on RTX 3090): https://github.com/facebookresearch/sam2/issues/448
- Apple CoreML SAM 2.1: https://huggingface.co/apple/coreml-sam2.1-tiny
- EdgeTAM repo: https://github.com/facebookresearch/EdgeTAM
- EfficientTAM repo and paper: https://github.com/yformer/EfficientTAM and https://arxiv.org/abs/2411.18933
- EfficientSAM3 repo and paper: https://github.com/SimonZeng7108/efficientsam3 and https://arxiv.org/abs/2511.15833
- RF-DETR (segmentation and keypoint tables, licenses): https://cdn.jsdelivr.net/gh/roboflow/rf-detr@develop/README.md
- SAM 3D Body: https://github.com/facebookresearch/sam-3d-body
- rtmlib (RTMPose without mmcv): https://github.com/Tau-J/rtmlib
- Apple Vision 3D body pose and person segmentation (WWDC23 notes): https://wwdcnotes.com/notes/wwdc23/111241
- Sapiens license (CC BY-NC 4.0): https://huggingface.co/facebook/sapiens
- PanoSAM2 (distortion-aware SAM 2 for 360 video): https://arxiv.org/abs/2604.07901
- Secondary: SAM 3.1 coverage https://www.labellerr.com/blog/sam-3-vs-sam-3-1-performance-comparison/ ; Roboflow on SAM 3 https://blog.roboflow.com/what-is-sam3/
