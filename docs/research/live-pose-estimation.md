# Live pose estimation for Liberational: research and recommendation

> Historical research from October 2026. Recommendations, model availability,
> licensing notes, and implementation paths below reflect that date and have not
> been revalidated. See [Architecture](../ARCHITECTURE.md) for the current model
> pipeline; this is not a deployment guide.

Research only; nothing installed, nothing in the repo changed. Companion to [video-segmentation.md](video-segmentation.md), which concluded that goal 3 (squats, weight, form) is a pose problem, not a mask problem.

## TL;DR

Use **RTMPose through `rtmlib`** (Apache 2.0, ONNX Runtime only, no mmcv) as the pose engine on both platforms: **RTMO-m** (one stage, boxes plus 17 keypoints in one call, about 12 ms per frame on a V100) on the live path of whichever camera can see the lifter, and **RTMPose-l or RTMW-l at 384x288** (26 keypoints with feet, or 133 whole body) in the offline backfill at 5 to 10 fps on exercise windows only. **RF-DETR Keypoint** (Apache 2.0, 71.8 COCO AP, June 2026, still a preview) is the strongest one-stage alternative and should be in the benchmark. For 3D, run **SAM 3D Body** (SAM License, 631M to 840M params, about 1 s per frame on an RTX 6000 Ada, or 110 to 150 ms with the MIT-licensed Fast SAM 3D Body) on a few frames per rep in backfill, never live. Avoid Ultralytics YOLO-pose (AGPL), OpenPose (non-commercial), Sapiens v1 (CC BY-NC), and Sapiens2 (its license forbids surveillance and biometric processing, which is what a face-identifying wearable does).

The honest part: the pack's three outward-facing fisheye cameras do not see the wearer, so form analysis needs a second camera that does (a phone on a tripod streaming to the pack is the practical answer), or the wearer is coaching someone else. A single side view gives reliable rep counts, back angle, and a usable depth call with about 5 to 10 degrees of uncertainty at the threshold; knee tracking and left-right symmetry need a frontal view; bar speed needs 60 fps and a known plate diameter and is good for mean velocity, not peak. Weight comes from speech first; a VLM read of the plates is a cross-check, not a source. As the wearer-only fallback, the whole scene bobbing once per rep in all three cameras is countable from the optical flow `vision/live.py` already computes, with no model.

## Context that shaped the advice

- The original comparison considered Apple Silicon and NVIDIA CUDA machines with outward-facing fisheye cameras. Its measurements do not establish performance on other hardware.
- `vision/live.py` runs InsightFace (buffalo_l, 640 detection) on every 720p30 frame at about 11 ms on a 3090, leaving 13 to 15 ms per frame; its `Tracker` is an IoU plus embedding matcher, and its `Motion` class already fits a per-frame similarity transform (shift, rotation, scale) from sparse Lucas-Kanade flow on a 320 px grayscale copy, about 1 ms on the CPU.
- `vision/loop.py` has a `Source` abstraction (a camera index or a directory of images), `fisheye_rectify` (equidistant model, 100 degree rectilinear tile through `cv2.fisheye`), and a per-view `tick` that captions and grounds with the VLM.
- `vision/backfill.py` samples recordings at `backfill.sample_fps = 2.0` under `max_gpu_share = 0.25`, pauses when the GPU is over 70 % busy, and writes to `state/sightings.sqlite3` (`sightings`, `segments`, `visits` tables). `memory/history_qa.py` answers "when did I last see X" from `visits`.
- The product may be sold, so Apache, MIT, and BSD are fine, and AGPL, non-commercial, and "no biometrics" licenses are not.

## 1. The landscape (October 2026)

### 2D top-down (detect people, then pose each crop)

| Model | License | COCO AP (256x192 unless noted) | Params | Published latency | Runs through |
|---|---|---|---|---|---|
| RTMPose-t / s / m / l | Apache 2.0 | 68.5 / 72.2 / 75.8 / 76.5 | 3.3M / 5.5M / 13.6M / 27.7M | 1.06 / 1.39 / 2.29 / 3.46 ms per crop (GTX 1660 Ti, TensorRT FP16); 3.2 / 4.5 / 11.1 / 18.9 ms on an i7-11700 CPU (ONNX Runtime, 1 thread); community RTX 3090 TensorRT about 1.0 to 1.4 ms | rtmlib (ONNX) |
| RTMPose-x 384x288 (Body8) | Apache 2.0 | 78.8 (COCO 17), 80.0 (Body8 26 keypoints) | 49M | not published | rtmlib |
| RTMW-l / x 384x288 (133 whole body) | Apache 2.0 | 70.1 / 70.2 COCO-WholeBody | 17.7 / 29.3 GFLOPs | not published | rtmlib |
| DWPose-l 384x288 (133 whole body, distilled) | Apache 2.0 | 66.5 COCO-WholeBody | 10.1 GFLOPs | 7.7 ms GPU (1660 Ti), 44.6 ms CPU | rtmlib |
| ViTPose-B / L / H | Apache 2.0 | 75.8 / 78.3 / 79.1 (classic decoder) | 86M / 307M / 632M | 17.1 GFLOPs for B (about 9x RTMPose-m); no consumer GPU numbers published | transformers `VitPoseForPoseEstimation`, rtmlib ONNX (ViTPose++ s/b/l 75.8 / 77.0 / 78.6), an MLX port of base |
| ViTPose++ whole body s/b/l | Apache 2.0 | 54.4 / 57.4 / 60.6 COCO-WholeBody | | | rtmlib |
| HRNet-W32/W48 | Apache 2.0 (MMPose) | 74.4 / 75.1 | 28.5M / 63.6M | slower than RTMPose-m at lower AP | MMPose only; legacy |

Top-down needs a person detector. rtmlib ships YOLOX-nano to x (Apache 2.0, person AP 38.9 to 61.3) and, as of 0.0.16, an RF-DETR person detector. For a single lifter in frame the detector can run every Nth frame and the crop can be carried by the keypoints themselves (rtmlib's `PoseTracker` does exactly this).

### 2D bottom-up and one-stage (one pass, all people)

| Model | License | COCO AP | Latency | Notes |
|---|---|---|---|---|
| RTMO-s / m / l (640x640) | Apache 2.0 | 68.6 / 72.6 / 74.8 (Body7 training) | 8.9 / 12.4 / 19.1 ms per frame, V100, ONNX Runtime | Latency does not grow with the number of people; faster than RTMPose plus detector at more than about four people. CrowdPose 83.8 AP for RTMO-l. In rtmlib. |
| RF-DETR Keypoint Preview (576x576) | Apache 2.0 | 71.8 (74.2 at 888) | 9.7 ms, T4, TensorRT FP16 | Launched 2026-06-22 as a preview; API and weights may change. DINOv2 backbone, 40.7M params per the README (the launch post says 126.4M; one of them is counting differently). Outputs per-keypoint confidence and a covariance ellipse (uncertainty for free). `rfdetr` 1.11.2 (2026-10-04), Python 3.10 or later, torch based; only in the `rfdetr` package, export for keypoints undocumented. |
| YOLO26-pose, YOLO11-pose | AGPL-3.0 or paid Enterprise | 71.0 (YOLO26x) | 10.6 ms (x, T4) | Avoid unless we buy the license. |
| OpenPose | CMU non-commercial research license | about 65 (legacy) | | Avoid; also unmaintained. |
| MoveNet Lightning / Thunder | Apache 2.0 | single person, no COCO AP published | real time on phones | 2021, TF Hub and Kaggle, no updates found; needs a TensorFlow or LiteRT runtime. Legacy. |

### Lightweight on-device

- **MediaPipe Pose Landmarker (BlazePose GHUM 3D)**: Apache 2.0. Lite 3 MB, Full 6 MB, Heavy 26 MB. 33 landmarks including heels and foot index, with image coordinates plus "world" coordinates in meters whose z is up to scale, not metric (model card). Google's own PDJ (PCK@0.2) is 87.0 / 91.8 / 94.2 for lite / full / heavy. Pixel 3 CPU: 44 / 18 / 4 fps. One person by default (`num_poses` raises it), out of scope beyond about 4 m from the camera, and the model card lists fitness and repetition counting as intended uses. `mediapipe` 1.1.1 (2026-10-14) ships wheels for macOS arm64 and Linux x86_64 with classifiers up to Python 3.14. No successor announced; the pose page was last updated 2026-10-09 and the Holistic page says an upgrade is "coming soon". It runs its own graph (TFLite, CPU on desktop), so it does not share the ONNX Runtime session or the CUDA EP we already have.
- **Apple Vision**: `VNDetectHumanBodyPoseRequest` (2D, 19 points, several people, macOS 11 and later) and `VNDetectHumanBodyPose3DRequest` (macOS 14 and later, 17 joints in meters relative to the root joint, `cameraOriginMatrix`, 2D projection API, one skeleton for the most prominent person, `bodyHeight` from depth when available and a 1.8 m reference otherwise). Free, on the Neural Engine, no published accuracy, and no Python API: it needs `pyobjc-framework-Vision` or a small Swift helper process. No changes found at WWDC 2025 or 2026.

### 3D from a single camera

| Model | License | What it gives | Accuracy | Speed | Status |
|---|---|---|---|---|---|
| SAM 3D Body | SAM License (see the segmentation report; commercial use allowed, ship the license, gated weights) | Full body mesh and joints on Meta's MHR rig from one image; takes 2D keypoints and masks as prompts; hand decoder | 3DPW MPJPE 54.8, EMDB 61.7 (ViT-H 631M and DINOv3-H+ 840M nearly equal) | 0.8 to 2.0 fps on an RTX 6000 Ada (500 ms per crop with ground-truth boxes, 1.0 to 1.25 s per frame with detection), per the Fast SAM 3D Body paper | Released 2025-11-19, CVPR 2026, native in ComfyUI since 2026-08. Not in transformers. |
| Fast SAM 3D Body | MIT (code); weights are SAM 3D Body's | Training-free acceleration of the above, plus a feedforward MHR to SMPL mapping | 3DPW MPJPE 58.9 vs 57.6, EMDB 65.3 vs 67.9 (automatic protocol); PA-MPJPE within 1 mm either way | 6.6 to 9.0 fps on an RTX 6000 Ada (111 to 152 ms); about 65 ms on an RTX 5090 | arXiv 2026-03. Its demo uses YOLO11m-pose (AGPL) as the detector; swap in RTMO boxes and keypoints, which SAM 3D Body accepts as prompts. |
| 4D-Humans / HMR 2.0 | MIT (code) | Per-frame SMPL mesh plus PHALP tracking | 3DPW MPJPE about 70 (2023) | a few fps | Quiet since 2023. SMPL itself is non-commercial unless licensed from Meshcapade, which applies to every SMPL-output model below. |
| WHAM | MIT (code), SMPL | World-grounded motion from video; takes ViTPose 2D keypoints | 3DPW MPJPE about 60 (CVPR 2024) | near real time on a desktop GPU | Research code. |
| TRAM | MIT (code), SMPL | Camera trajectory plus human motion (DROID-SLAM plus VIMO) | better than WHAM on EMDB | slow (SLAM) | Research code. |
| MotionBERT | Apache 2.0 | 2D keypoint sequences (243 frames) lifted to 3D, Human3.6M skeleton | state of the art on Human3.6M in 2023 | fast (keypoints only) | Clean license, lab-centric training data; the kind of lifter PriorFormer (2025, 380 us per frame on a GPU, 36 mm) continues. |
| RTMW3D-x | Apache 2.0 | 133 whole-body keypoints with pseudo-depth from one image | 68.0 COCO-WholeBody (2D) | RTMW cost | In rtmlib as `Wholebody3d`. Depth is relative, like MediaPipe's z. |
| NLF (Neural Localizer Fields) | unconfirmed | Any body point in 3D from one image | NeurIPS 2024 | GPU friendly | Repo stale over a year; skip. |
| Sapiens2 (0.1B to 5B) | Sapiens2 License (custom) | 2D pose (+4 mAP over Sapiens), segmentation, normals, pointmaps | state of the art, ICLR 2026 | heavy (1024 px ViTs) | The license is not non-commercial, but its acceptable-use policy forbids surveillance and biometric processing. Not for this product. Sapiens v1 is CC BY-NC. |
| OnlineHMR | unknown | Fully online world-grounded HMR from video | CVPR 2026 | | Watch. |

No 2026 release replaces WHAM or TRAM for commercial use; the one real 2026 development is that SAM 3D Body got fast enough (Fast SAM 3D Body) to run on every frame of a short clip offline.

### Whole body (hands, feet, face)

For lifting form the useful extras are the **feet** (heel and toe give ankle angle, heel rise, and foot pressure shift) and the **hands** (grip width, wrist position under the bar). RTMPose's 26-keypoint Body8 models add neck, hip center, head, and six foot points at no extra cost, which is the sweet spot; RTMW (133) and DWPose add 42 hand points and 68 face points for about 2 to 4x the compute. Face points are irrelevant here and InsightFace already covers identity.

### Multi-person tracking for a gym

ByteTrack (MIT, the paper's reference code) is the standard association; Roboflow's `trackers` (Apache 2.0: SORT, ByteTrack, OC-SORT) and `supervision` 0.31.1 (MIT) carry maintained Python versions. For pose, track the person boxes (RTMO and RF-DETR emit them; RTMPose's are the detector's) and carry keypoints on the track; an OKS-based match is a one-line improvement over IoU. `vision/live.py`'s `Tracker` already does IoU plus embedding for faces and would take a pose track with a different distance function. In a gym the association question is "which track is the lifter we are analyzing": pick the track whose box overlaps the barbell or that the owner pointed at ("watch Bob's squats"), and keep the others for counting people, not form.

## 2. Serious candidates in detail

### RTMPose family through rtmlib (RTMPose, RTMO, RTMW, DWPose, ViTPose ONNX)

- **License**: Apache 2.0 for rtmlib, MMPose, and the checkpoints; DWPose (IDEA) is Apache 2.0 too.
- **Dependencies**: `rtmlib` 0.0.16 (2026-08-04, Python 3.10 or later) needs numpy, onnxruntime, opencv-python, opencv-contrib-python, and tqdm. Two of those collide: `opencv-python` and `opencv-contrib-python` both install the `cv2` package and overwrite each other, and we already lock `opencv-python` 5.0.0.93. Install rtmlib with `--no-deps` (its own code is a few hundred lines over ONNX Runtime), or vendor the two files we need into `vision/pose.py`. Python 3.14 is not a problem: everything it imports is already in `uv.lock`.
- **Backends**: ONNX Runtime with `CPUExecutionProvider`, `CUDAExecutionProvider`, or (for `device="mps"`) `CoreMLExecutionProvider` when available, which is the same provider list `vision/faces.py` already picks for InsightFace, so one session type on both platforms. OpenVINO and TensorRT are optional.
- **Accuracy and speed**: see the table above. Model-only numbers; add the detector (YOLOX-s 640: a few ms on a 3090) or use RTMO. Note MMPose says its model zoo entries use flip test, so expect a point less in a pipeline.
- **Maintenance**: MMPose's last release is v1.3.2 (2024-07-12); rtmlib still gets releases (0.0.16 added RF-DETR detectors and ViTPose ONNX). The ONNX files are frozen artifacts, so "unmaintained upstream" costs us little: nothing in the runtime path depends on MMPose.
- **Mac**: unmeasured. The CoreML EP compiles supported ops to CoreML and falls back per node; RTMPose is plain convolutions (CSPNeXt backbone, SimCC head), so expect it to compile cleanly. An M1 CoreML measurement of an unrelated 256x256 model ran at 7 to 8 ms, with a warmup penalty. Even CPU-only, RTMPose-m at 11 ms per crop on a 2021 i7 suggests under 10 ms on an M5 Pro core. Measure before relying on it.

### RF-DETR Keypoint Preview

- Apache 2.0, 71.8 COCO AP at 576 (above YOLO26x-pose's 71.0), 9.7 ms on a T4 with TensorRT FP16, released 2026-06-22. One model gives boxes, 17 keypoints, per-keypoint confidence, and covariance ellipses. Fine-tuning on a custom skeleton (for example adding the barbell ends as keypoints) is its primary intended use, and that is the most interesting thing about it for us.
- Costs: the `rfdetr` package pulls torch and friends (fine on Linux, where the `voice-clone` extra already brings torch 2.11; on the Mac it would run on MPS, unmeasured), the keypoint export path is undocumented, and "preview" means the weights may change under us. Benchmark it; do not build on it until it leaves preview.

### ViTPose / ViTPose++ through transformers

- Apache 2.0, in transformers 5.x as `VitPoseForPoseEstimation` with the `usyd-community` checkpoints (base, large, huge, and ViTPose++ with six dataset experts including COCO-WholeBody). Top-down: feed person boxes from any detector. The transformers path runs on CUDA and MPS with the same code, and torchao int4 is documented for the huge model.
- ViTPose-B at 17.1 GFLOPs per crop is about 9x RTMPose-m for 0 AP gain at 256x192 (75.8 both); ViTPose-H buys 3.3 AP for 40x the compute. That is a backfill trade, not a live one. Its real advantage is robustness on unusual views (strong pretraining, multi-dataset experts), which matters on fisheye and odd camera angles, so keep ViTPose++-L in the backfill benchmark.

### SAM 3D Body and Fast SAM 3D Body

- The only serious single-image 3D option with a license that allows a product (SAM License; MHR rig rather than SMPL). 3DPW MPJPE 54.8 is the best published number in this table. Takes our 2D keypoints and masks as prompts, so RTMPose output improves it rather than competing with it.
- At 0.5 to 1.25 s per frame (original) or 110 to 150 ms (Fast, on an RTX 6000 Ada; a 3090 will be slower), it is a backfill tool for a few frames per rep, exactly as the segmentation report said. Requires a separate environment (the repo asks for its own conda setup; Fast SAM 3D Body has TensorRT scripts). Not on the Mac: no MLX or CoreML port exists.
- What 3D buys over 2D for squats: joint angles that do not depend on the camera being exactly side-on, and knee position relative to the foot without a second camera. What it does not buy: metric accuracy near the depth threshold (55 to 65 mm MPJPE is 2 to 3 cm per joint, comparable to the 2D side-view error once scaled).

### MediaPipe Pose Landmarker

- Apache 2.0, trivial to install, single person, 33 landmarks with feet, CPU only on desktop, and Google's own model card says it was built for fitness and rep counting. It is the right fallback on a machine with no GPU and the fastest way to a demo, but it is a separate runtime (TFLite graph) and its "3D" z is relative. Use it as the control in the accuracy benchmark, not as the engine.

### Apple Vision 3D body pose

- Free on the Mac, 17 joints in meters, one person. No accuracy numbers from Apple, no Python API, and no Linux equivalent, so it cannot be the shared engine. Worth a weekend as a Mac-only second opinion on depth and back angle through pyobjc once the rtmlib path works, mostly because it costs no GPU time at all.

## 3. What a squat analyzer actually needs

Per rep, from a camera that sees the lifter, with the COCO 17 (or Body8 26) skeleton:

| Measure | How | Reliable from one 2D side view? | Needs |
|---|---|---|---|
| Rep count | Hysteresis state machine on the knee angle (hip, knee, ankle) or on the hip keypoint's vertical position: "up" above about 150 degrees, "down" below about 110, count on the up transition; require 0.5 to 6 s per cycle | Yes. This is the robust part: a 20 degree hysteresis swallows all the jitter. | Any 2D pose, 15 fps or more |
| Bar path | Track the plate (a 450 mm disc in side view: Hough circle or template, then Lucas-Kanade on it, which `live.py`'s `Motion` already does for the background), or use the wrist keypoints as a proxy | Yes for shape (forward drift, "good morning" reps). The plate disc also gives the pixel-to-meter scale for free. | Side view, plate visible |
| Depth (hip crease below knee) | Compare the hip keypoint's y with the knee's at the bottom frame, or threshold the knee angle | Partly. The COCO "hip" keypoint is near the greater trochanter, not the hip crease, so it sits below the crease by a few cm and reads deeper than a judge would; and 2D pose underestimates peak knee flexion (the running study's worst case was peak flexion; the 2026 MediaPipe squat study found knee flexion off by -2.3 +/- 6.4 degrees at a true lateral view and -13.9 +/- 15.1 degrees once the camera is 45 degrees off axis, worse in loose shorts). Ota et al. (Gait and Posture 2020, 20 adults, side view vs Vicon) found ICCs "almost perfect or substantial" for trunk, knee, and ankle, only "fair" for the hip, with fixed biases on knee and ankle. | Report "clearly deep", "clearly high", or "borderline (within 10 degrees)"; calibrate the hip offset once per lifter from a known-deep rep |
| Back angle | Trunk (shoulder to hip) vs vertical, and the trunk to femur angle, at the bottom and at the sticking point | Yes, from a true side view; degrades off axis like everything else | Side view |
| Knee tracking (valgus) | Frontal plane projection angle (hip, knee, ankle) or the knee-to-ankle separation ratio | No, not from the side: the far leg is occluded and valgus is a frontal-plane motion. From a frontal view, 2D correlates r 0.64 to 0.78 with 3D for single-leg squats, the knee-to-ankle ratio reached ICC 0.96 in a drop-jump study, and a 2026 study found OpenPose and BlazePose "reliable and largely consistent" for frontal-plane valgus. | Frontal (or 45 degree) view, or 3D |
| Left-right symmetry | Compare left and right knee and hip angles and the bar tilt | No from the side (one leg hides the other); yes from the front | Frontal view, or two views |
| Bar speed (mean concentric velocity as a load proxy) | Plate center y over the concentric phase, scaled by plate diameter, smoothed; velocity drop across a set predicts proximity to failure | Mean velocity yes with care: a 240 fps phone plus Kinovea agreed with a transducer within 0.03 m/s, but the Metric VBT app (phone camera) was found not reliable against Vicon in a 2024 PeerJ study. At 30 fps a 0.6 s concentric is 18 samples, fine for mean velocity, too coarse for peak. | 60 fps or more, plate visible, camera perpendicular to the bar |

**Smoothing.** Apply a One Euro filter (Casiez et al. 2012) per keypoint, tuned for low lag (min cutoff about 1 Hz, beta about 0.01 at 30 fps), then compute angles; or a constant-velocity Kalman filter on the angle series. Gate every joint on its confidence (RF-DETR's covariance ellipses are the best version of this), and interpolate across one or two dropped frames rather than freezing. Rep counting should run on the smoothed angle with hysteresis, never on raw peaks. Keep the raw series in the event record so the smoothing can be changed later.

**Camera placement.** Side view perpendicular to the bar, lens at about hip height, 3 to 4 m away, whole body in frame with margin (MediaPipe's own limit is 4 m), 1080p at 60 fps if bar speed matters, plain background and tight clothing help every model. A second, frontal camera (or a 45 degree compromise) is what knee tracking and symmetry need; a mirror in a side-view frame gives a free second view if the pose model is run on the mirrored half separately.

**Weight.** Speech first, as ROADMAP.md says ("putting on 135"), with the source stored. Vision as a cross-check: when a plate stack is visible, crop the bar end (the plate tracker already knows where) and ask the VLM to count plates and read their colors (competition colors: red 25, blue 20, yellow 15, green 10, white 5 kg; gym pounds plates vary). No published evaluation of VLM plate counting exists; counting overlapping discs of similar color is exactly what VLMs get wrong, so treat a vision read that disagrees with speech as a prompt to ask, not as truth. The one measurement vision can make that speech cannot is the bar velocity drop across the set, which says how heavy it felt.

## 4. Fisheye and the wearable

**No pose model has been evaluated on 160 degree fisheye.** The closest study (Käs et al., ICRA 2025, with the FISHnCHIPS fisheye pose dataset) compared projections for 3D pose on fisheye images and found pinhole undistortion "inadequate" for close-up subjects, the double sphere model best, and the right choice dependent on how much of the field of view the person covers. An earlier paper notes that Human3.6M's distortion is "almost negligible" compared with fisheye, so training data gives no help at the rim.

**Undistortion costs almost nothing, but choose where.** `loop.py`'s `fisheye_rectify` is a precomputed `cv2.remap` (about 1 to 3 ms per 720p frame on the CPU) that keeps 100 degrees and drops the rim. For pose the cheaper and better option is to run the model on the raw frame (people near the center are only mildly distorted) and undistort only the keypoints with `cv2.fisheye.undistortPoints` before computing angles, which costs microseconds and keeps the rim. The prototype should run both and compare on people near the edge.

**The pack cannot see its wearer's legs; it can see the room move.** A squat moves the pack down and up by 0.3 to 0.6 m once per rep at 0.3 to 1 Hz, with a pitch change at the bottom, and all three cameras see the scene translate vertically together. `live.py`'s `Motion` already produces the per-frame similarity transform; integrating its vertical shift, band-passing 0.2 to 1.5 Hz, and counting peaks with hysteresis across all three views is a rep counter with no model and no new compute. The precedents are Jiang and Grauman (2017), who inferred a wearer's full-body pose from how the scene moves in a chest camera, Runia et al. (2018), who count repetitions from flow fields under camera motion, and RepNet. Limits, stated plainly: it counts cycles, says nothing about form, needs the context that this is a squat set (speech, a caption that mentions a barbell, or a gym location) because walking is periodic too, and the Teensy IMU (not started) is the better sensor for the same signal when it exists. Egocentric pose from a downward fisheye (Mo2Cap2, xR-EgoPose) is a real research line, but it assumes a head-mounted camera looking down at the body, which is a hardware change, not a software one.

## 5. Recommendation

**Models**

- Live, Linux and Mac: **RTMO-m** through rtmlib (one ONNX session, boxes plus 17 keypoints, Apache 2.0) on the camera that sees the lifter; RTMO-s if the Mac measurement is tight. Switch to RTMPose-m Body8 (26 keypoints) top-down with rtmlib's `PoseTracker` when feet matter and only one lifter is in frame. Same code on CUDA and CoreML EPs. RF-DETR Keypoint goes in the benchmark and replaces RTMO if it wins on fisheye frames and leaves preview.
- Backfill: **RTMPose-l Body8 at 384x288** (or RTMW-l for hands) at 5 to 10 fps on exercise windows; ViTPose++-L through transformers as the quality comparison. **Fast SAM 3D Body** (its own venv, SAM License weights, RTMO boxes instead of its YOLO detector) on the bottom frame of each rep for 3D hip and knee angles, optional, Linux only.
- Fallbacks and controls: MediaPipe Pose Landmarker (CPU, single person) on machines without a GPU and as the accuracy control; Apple Vision 3D as a Mac-only check later.
- Do not use: Ultralytics YOLO-pose (AGPL), OpenPose (non-commercial), Sapiens v1 (CC BY-NC), Sapiens2 (no biometrics clause), anything that outputs SMPL in a product (SMPL is licensed separately from the model code; MHR avoids this).

**Where and at what rate**

- Live pose runs only on a camera flagged as a lifter view, on its own thread in `FastPath` at every 2nd or 3rd frame (10 to 15 pose fps is plenty for counting; 30 for bar speed), so the face path's cadence is untouched. Here that means GPU 1 while GPU 0 carries faces and the VLM; on the Mac everything shares one GPU, so measure the captioner's rate with pose on.
- Backfill: a pose pass only on event windows (a transcript mention of a set, a caption that says barbell or squat rack, the ego-motion counter firing, or a lifter-view camera being active), under the existing `Throttle`. At the 0.25 share and about 5 ms per frame, a day's budget covers hours of exercise footage; the constraint is finding the windows, not processing them.

**Second camera as a source, in terms of our files**

1. `config/defaults.toml`, `[vision]`: cameras get an optional `url` (RTSP or HTTP MJPEG from a phone app or a gym camera) next to the index, plus `rectilinear = true` (skip `fisheye_rectify`), `role = "lifter"`, and `record = true` so the stream lands in `state/video/` like the others. A `[pose]` section: `enabled`, `live_model`, `backfill_model`, `live_every` (frames), `backfill_fps`, `device`, `min_conf`, defaulting to off. A `pose` optional dependency in `pyproject.toml` (rtmlib with `--no-deps` semantics, or vendored).
2. `vision/loop.py`: `Source` learns a URL (`cv2.VideoCapture(url)` with the same watchdog and retry as `CamThread`); `build_views` passes the role through; `tick` skips captioning on a lifter view unless asked, and the view is still recorded.
3. `vision/pose.py` (new): `Pose` (lazy ONNX session, `detect(frame) -> [(box, keypoints[K,2], scores[K])]`, with `undistort_points` for fisheye views), `OneEuro`, `angles(kps)` (knee, hip, trunk, shank, left and right), `RepCounter` (hysteresis state machine returning rep spans and bottom frames), `BarTracker` (plate disc, path, and velocity with the 450 mm scale), and `EgoRepCounter` fed from `live.Motion`. Pure functions on numpy arrays so the tests in `tests/test_pose.py` run on recorded keypoint JSON with no model.
4. `vision/live.py`: `FastPath` takes an optional `pose` callable and runs it on every Nth frame from a side thread (the `stream_loop` pattern), keeps the last skeleton per track, and `annotate` draws it; stats gain `pose_ms` beside `det_ms` so the dashboard shows the cost. The `/boxes` SSE stream carries keypoints when present.
5. `vision/backfill.py`: a pose pass after the face pass, per segment, only inside event windows, writing rep rows; the ego-motion counter runs on the pack's own segments at the sample rate as a third pass.
6. Events store: a new `state/events.sqlite3` (or tables in `sightings.sqlite3`): `events(id, kind, start, end, cam, person, source, attrs_json)` for "squat set" with weight and its source, and `reps(event_id, n, t_top, t_bottom, knee_min_deg, hip_min_deg, trunk_deg, depth_call, bar_mcv, bar_path_json, conf)`. This is the ROADMAP's "event abstraction" (goal 3, step 1), and the same table takes non-exercise events later.
7. `memory/history_qa.py`: `_events(state, query)` answers "how many squats did I do", "what weight", and "how was my form" from `events` and `reps` first, with the transcript line that gave the weight as evidence; `collect` adds it beside `_camera`, `_transcripts`, and `_who`. The `/review` page gets rep markers on the timeline.

**What to prototype first, and how to measure it**

Prototype 0, a benchmark script in `tools/` (nothing in the services), on 3 to 5 recorded segments plus two phone clips of real squats:
- Models: RTMO-s/m, RTMPose-m and -l (Body8), RTMW-l, ViTPose-B and ViTPose++-L (transformers), RF-DETR Keypoint Preview, MediaPipe heavy as the control. Report p50 and p95 ms per frame, peak VRAM, and InsightFace `det_ms` while pose runs on the same GPU and on GPU 1.
- Same script on the Mac: rtmlib on the CoreML EP and on CPU, ViTPose on MPS, with the captioner's rate while pose runs.
- Fisheye: each model on raw frames vs `fisheye_rectify` tiles vs raw plus keypoint undistortion, scored by agreement with the model's own output on the rectified tile for people near the edge, and by eye in the review player.

Prototype 1, the analyzer, offline, on hand-counted footage: 20 sets (two lifters, 5 to 10 reps each, side view and 45 degree, a few with loose clothing) from a phone. Metrics: rep count exact-match rate and mean absolute error (gate: zero miscounts on side view); depth call agreement with two human raters on each rep (Cohen's kappa; gate: no "deep" call on a rep both raters called high); trunk angle plausibility; bar mean velocity against a manual 240 fps timing on ten reps (expect within 0.05 m/s). Run MediaPipe through the same code as the control.

Prototype 2, live: the phone as a `url` source, the overlay on the dashboard, events written, "how many squats did I do today" answered from events. Gate: pose adds under 2 ms to `det_ms` on a shared GPU, or it moves to the other GPU.

Prototype 3, the wearer-only fallback: `EgoRepCounter` on pack recordings of 10 hand-counted sets. Gate: exact count on at least 8 of 10 with no counts during a walking segment of the same length.

Decision gates in one line each: ship live pose only on lifter-view cameras; ship depth calls only with the borderline band; ship weight from vision only as a cross-check; ship the ego-motion counter only with a context gate; revisit RF-DETR Keypoint when it leaves preview and SAM 3D Body on the Mac when an MLX or CoreML port exists.

## Sources

- rtmlib (README, model zoo, backends): https://github.com/Tau-J/rtmlib and https://pypi.org/project/rtmlib/
- rtmlib provider selection (`device="mps"` maps to CoreML): https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/base.py
- RTMPose model zoo and latency tables (1660 Ti, i7-11700, Snapdragon 865): https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md
- RTMPose community benchmark (RTX 3090 TensorRT numbers): https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose/benchmark
- RTMPose paper: https://arxiv.org/abs/2303.07399
- RTMO model zoo (V100 ONNX Runtime latency, CrowdPose): https://github.com/open-mmlab/mmpose/blob/main/projects/rtmo/README.md
- RTMW paper (2D and 3D whole body): https://arxiv.org/abs/2407.08634
- MMPose releases (v1.3.2, 2024-07-12): https://github.com/open-mmlab/mmpose/releases
- DWPose license (Apache 2.0): https://github.com/IDEA-Research/DWPose/blob/onnx/LICENSE
- ViTPose paper (COCO val AP table, GFLOPs): https://arxiv.org/abs/2204.12484
- ViTPose license (Apache 2.0): https://github.com/ViTAE-Transformer/ViTPose/blob/main/LICENSE
- transformers ViTPose docs (`VitPoseForPoseEstimation`, ViTPose++ experts, torchao): https://huggingface.co/docs/transformers/en/model_doc/vitpose
- ViTPose MLX port: https://huggingface.co/nazarkozak/vitpose-base-simple-mlx
- RF-DETR README (keypoint preview table, licenses): https://github.com/roboflow/rf-detr
- RF-DETR Keypoint docs: https://rfdetr.roboflow.com/latest/learn/run/keypoints/
- RF-DETR Keypoint launch (2026-06-22, 74.2 AP at 888): https://blog.roboflow.com/launch-rf-detr-keypoint-in-roboflow/
- rfdetr on PyPI (1.11.2, Python 3.10 or later): https://pypi.org/project/rfdetr/
- Ultralytics licensing (AGPL-3.0 or Enterprise): https://www.ultralytics.com/license
- OpenPose license (non-commercial): https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE
- MoveNet (Apache 2.0, 2021): https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html
- MediaPipe Pose Landmarker guide: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- BlazePose GHUM 3D model card (sizes, PDJ, Pixel 3 fps, limits, Apache 2.0): https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf
- mediapipe on PyPI (1.1.1, wheels, classifiers): https://pypi.org/project/mediapipe/
- Apple Vision 3D body pose and person segmentation (WWDC23 session 111241): https://developer.apple.com/videos/play/wwdc2023/111241/
- Apple `DetectHumanBodyPose3DRequest`: https://developer.apple.com/documentation/vision/detecthumanbodypose3drequest
- SAM 3D Body repo (SAM License, sizes, benchmarks): https://github.com/facebookresearch/sam-3d-body
- SAM 3D Body weights (gated): https://huggingface.co/facebook/sam-3d-body-dinov3
- Fast SAM 3D Body paper (fps and MPJPE tables, RTX 6000 Ada): https://arxiv.org/abs/2603.15603
- Fast SAM 3D Body repo (MIT, 65 ms on an RTX 5090, YOLO11 detector): https://github.com/yangtiming/Fast-SAM-3D-Body
- SAM 3D Body in ComfyUI (2026-08-23): https://comfyui-wiki.com/en/news/2026-08-23-sam-3d-body-native-comfyui
- 4D-Humans license (MIT): https://github.com/shubham-goel/4D-Humans/blob/main/LICENSE.md
- WHAM license (MIT): https://github.com/yohanshin/WHAM/blob/main/LICENSE
- TRAM license (MIT) and paper: https://github.com/yufu-wang/tram and https://arxiv.org/abs/2403.17346
- MotionBERT license (Apache 2.0): https://github.com/Walter0807/MotionBERT/blob/main/LICENSE
- PriorFormer (2025 lifter): https://arxiv.org/abs/2508.18238
- NLF (NeurIPS 2024): https://arxiv.org/abs/2407.07532
- OnlineHMR (CVPR 2026): https://arxiv.org/abs/2603.17355
- Sapiens2 paper and license: https://arxiv.org/abs/2604.21681 and https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md
- Sapiens v1 license (CC BY-NC 4.0): https://huggingface.co/facebook/sapiens
- ByteTrack paper: https://arxiv.org/abs/2110.06864
- Roboflow trackers (Apache 2.0): https://trackers.roboflow.com/latest
- supervision on PyPI (0.31.1, MIT): https://pypi.org/project/supervision/
- One Euro filter: https://gery.casiez.net/1euro/
- Ota et al., OpenPose vs Vicon for squats (Gait and Posture 2020): https://repository.kulib.kyoto-u.ac.jp/dspace/handle/2433/252801
- MediaPipe knee angle vs inertial reference across camera angles (2026): https://tesi.univpm.it/handle/20.500.12075/23610
- Camera viewing angle and OpenPose validity (Sensors 2025): https://www.mdpi.com/1424-8220/25/4/1076
- Knee valgus from 2D pose, OpenPose and BlazePose (BMC Musculoskeletal Disorders 2026): https://www.medicalschool-hamburg.de/files/inhalte/News/2026/2026-janisch_et_al-bmc_musculoskeletal_disorders.pdf
- 2D knee valgus measures vs 3D (IJSPT 2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC9159712
- Smartphone 240 fps plus Kinovea vs linear transducer for bar velocity (JHSE 2021): https://www.jhse.ua.es/article/view/2021-v16-n4-smartphone-high-speed-camera-kinovea-velocity-based-
- Metric VBT validity vs Vicon (PeerJ 2024): https://doaj.org/article/426d2074cb27453f96495e10851f987b
- Fisheye projection methods for 3D pose (Käs et al., ICRA 2025, FISHnCHIPS): https://arxiv.org/abs/2506.19747
- Distortion-aware 3D pose (fisheye as "heavy" distortion): https://arxiv.org/abs/2111.15056
- Seeing Invisible Poses (wearer pose from chest-camera scene motion): https://arxiv.org/abs/1603.07763
- Repetition estimation from flow under camera motion (Runia et al.): https://arxiv.org/abs/1806.06984
- RepNet: https://research.google/blog/repnet-counting-repetitions-in-videos/
- Egocentric fisheye pose survey (CVPRW 2024): https://openaccess.thecvf.com/content/CVPR2024W/Rhobin/papers/Azam_A_Survey_on_3D_Egocentric_Human_Pose_Estimation_CVPRW_2024_paper.pdf
- Barbell and plate detection datasets (Roboflow Universe, CC BY 4.0): https://universe.roboflow.com/hika/weightplate
