# Storage status and projections

Where the system's data lives, how much disk it takes, and measured archive codec candidates. All units are
decimal: MB = 1,000,000 bytes, GB = 1,000 MB, and TB = 1,000 GB.

## Where data lives

The catalog, recordings, segment store, vector index, and knowledge store live under `paths.state`.
Face references use `paths.faces`, and imported originals and converted media use `paths.ingest`.

| Store | Path | What it holds | Kept |
| --- | --- | --- | --- |
| Recordings | `continuum/tier`, `continuum/tier2` | Camera video and microphone audio, written by the tier worker. A microphone's chunk becomes Opus as it closes; `tier2` holds only older Opus chunks. | Video 30 days; audio indefinitely, less silent audio past 30 days (see Retention) |
| Broker catalog | `continuum/catalog.redb` | The recording index and remaining broker payloads, including journal entries, person facts, summaries, and Ask history (`memory_log.py`). | Until a catalog reset |
| Segment store | `segments/<stream>/<camera>/` | Machine output: poses, actions, movement, walk-bys, faces, animals, zones, presence, captions, objects, speech, voice embeddings, network, Bluetooth, and positions. One stream, one camera, and one UTC hour per file: Arrow IPC parts with zstd compression, one record batch per append, each row carrying `ts_ns` and a per-stream, per-camera `row_seq`. The filesystem is the index: there's no shared database. Columns are declared in `src/liberational/segment_log.py`; [VISION-STREAMS.md](VISION-STREAMS.md) lists the vision streams' fields. | Until a catalog reset; per-frame rows are condensed or dropped past 30 days (see Retention) |
| Vector index | `continuum/vector` | Face, voice, and journal search, rebuilt from segment embeddings and broker memory-log streams. | Until a catalog reset |
| Knowledge store | `knowledge.sqlite3` | What the owner taught: people and identity records, pet profiles and sighting labels, drawn zones, activity names and ratings, device links, object crops, and confirmed voice references (`src/liberational/knowledge.py`). | Preserved by `--keep-taught`; removed by `--fresh` |
| Face references | `faces/db.npz` | The face matcher's reference embeddings. | Preserved by `--keep-taught`; removed by `--fresh` |
| Places | `places/<place>/` | Spatial memory's machine view of each indoor place (`memory/places.py`): revision records, `.map` point clouds, and `.plan` floorplans. The fixed cameras' places (`pack-spatial-cameras`, daily at midday) write a binary PLY cloud of about 2 MB per place per run, with each camera's pose, the floor plane, and the plan's frame in the revision record. A phone walkthrough (`pack-walkthrough`, by hand) writes one revision per video, its cloud at 3 cm like a fused map: a two-minute walk through a whole home came to 1.6 million points, about 24 MB. Tier 1 follows each with a fused revision: the place's map so far, one point per 3 cm cube with how often it was seen and missed. A whole home's surfaces at 3 cm come to roughly 1 to 2 million points, about 15 to 30 MB a fused map; things that come and go fade out, so it stops growing once the home is covered. | Revision records and plans until a reset; the newest 2 maps per place, so a few MB per place however long it runs |

Outside `paths.state`, under the repo's `data/`: model files (`data/models`, `mise run models`), including MASt3R's
2.75 GB checkpoint and CUT3R's 3.2 GB one on CUDA machines, and MASt3R's and CUT3R's own runtimes (`data/mast3r` and
`data/cut3r`, about 7 GB each: pinned code and a venv with its own torch, `mise run mast3r` and `mise run cut3r`). All
are rebuilt by those tasks and untouched by a reset.

Machine-made data is expendable. The catalog reset (`pack-reset-catalog --keep-taught`) deletes the catalog,
the recordings it indexes, the vector index, the shared decode's frame ring, the segment store, and the derived
workers' checkpoints, then restarts. This mode preserves `knowledge.sqlite3` and `faces/db.npz`.
The separate `--fresh` mode removes taught state too, including face and voice references; see
[MIGRATING.md](../MIGRATING.md) for the full reset boundary.

## Retention

Camera video is deleted 30 days after the broker committed it (`pack-video-window`, service `video-window`,
nightly; `video_window.days`), so an import of old media gets the whole window from its import. The deletion is an
age-out: the chunks and their frame rows go, the spans are recorded as aged out, and everything derived from the
video stays. Review and Ask say the footage expired for those spans. If free space on the recordings volume drops
below 15% (`video_window.free_floor_percent`), the oldest video goes early, an hour at a time, never the newest day.

Before it ages anything out, each run keeps the moments the journal's day pages cite (`memory/journal_evidence.py`):
a 6 s clip (480 px wide, 8 fps, H.264 at CRF 32), a still (640 px JPEG), and the caption, in
`state/journal_evidence/<YYYY-MM-DD>/`, kept for good. At most 3 camera moments per cited hour are kept. A day page
citing about 10 hours is about 30 moments; at an estimated 50 to 150 KB a clip and 15 to 60 KB a still, that is
**about 2 to 6 MB a day**, under 2.2 GB a year. The run's log line counts what it promoted (`promoted_clip`,
`promoted_still`, `promoted_gone`) and `promoted_bytes`; the per-clip figure is an estimate until a month of those
lines gives a measured one.

Past the same window, the meaning is kept and the measurement dropped (replacing the earlier keep-everything rule for
per-frame rows and silent audio). The nightly `video-window` run then takes each
ended local day older than the window whose day page (`journal.page`) is written, oldest first, at most 7 a night
(`src/liberational/retention.py`):

| Data | Past the window |
| --- | --- |
| Dense coverage (`action.coverage`, `walkby.coverage`, `animal.coverage`, `zone.coverage`) | Condensed to contiguous intervals (rows at most 2 s apart, within one UTC hour, frames summed) in a `<stream>.condensed` companion that every segment read merges in; the originals are dropped |
| Device sightings (`network.device_observation`, `bluetooth.sighting`) | Condensed to visits (one device's identical polls at most 60 s apart, within one UTC hour) the same way |
| Positions (`position.sample`: about one row a second per tracked person or pet, some 30,000 rows a day for a busy camera) | Thinned to paths the same way: each track's (or entity's) first sample in every 10 s, and its last |
| Per-frame rows (`pose.track`, `face.match`, `face.signal`) and per-frame face embeddings (`face.embedding`) | Dropped: the kept spans already condense them |
| Spans and events (`action.span`, `walkby.span`, `zone.visit`, `presence.span`, `face.visit`), captions, transcripts, voice embeddings, the journal | Kept |
| Audio | Chunks that no speech touches (an utterance, or a second the voice gate heard speech, with 5 s either side) are aged out like old video; speech stays |
| Taught state, face references | Kept, never decays |

A day without a day page is kept whole until it has one. The drops are segment-store forgets of kind `reset`, so what
was learned from the rows (the vector index, the journal) stands. `state/retention.json` records each finished day
with what went; the log names each stream's rows and bytes per day. `movement.*` and `action.window` are not touched
yet. Explicit deletion and resets remain separate operations.

## Six-camera archive baseline

A bounded inventory of the original camera and five native H.264 network cameras. Each row uses 64 closed chunks
and reads container metadata, not decoded pictures. The two rates divide file bytes by the capture hull and by the
summed stored spans, respectively. They bracket short-sample cadence effects; they are not a confidence interval or
a measured 24-hour workload. Different cameras cover different windows.

| Source | Stored resolution and approximate fps | Sample hull, seconds | Projected GB/day, hull to stored spans |
| --- | --- | ---: | ---: |
| Original camera, Tier 2 H.264 | 960 × 540, 30 | 127.0 | 13.675 to 13.905 |
| Network landscape A, native H.264 | 1280 × 720, 30 | 204.0 | 7.918 to 8.000 |
| Network portrait, native H.264 | 720 × 1280, 30 | 214.0 | 7.126 to 7.196 |
| Network landscape B, native H.264 | 1280 × 720, 30 | 204.0 | 7.566 to 7.645 |
| Network low-rate A, native H.264 | 896 × 512, about 10 | 255.9 | 0.704 to 0.717 |
| Network low-rate B, native H.264 | 896 × 512, about 10 | 255.9 | 0.752 to 0.771 |
| **All six video streams** | | | **37.741 to 38.235** |

Audio, the catalog, the segment store, temporary predecessor overlap, and other disk use are additional. Native
H.264 Tier 2 regrouping copies compressed pictures; it does not lower their bitrate. At this rate the 30-day video
window holds about **1.15 TB** of video.

## Recommendation and measured candidates

Keep the current encoding. Thirty days of video at the sampled six-camera rate is about **1.15 TB**, which fits the
measured **1.7 TB free** (decimal) with room to spare; the window's 15% free-space floor is a safety net, not the
working limit. Audio, the catalog, the segment store, and temporary copies are additional.

If more space is needed after that, **GPU H.264 P5/CQ 36** is the simpler compatibility candidate to review:
about **15.1 GB/day** and **8.4 CPU core-hours/day**, or roughly **0.35 average CPU cores**. GPU HEVC P5/CQ 36
is about **15.6 GB/day** at similar CPU cost and generally better spatial scores in these samples. HEVC playback
can require the dashboard's existing bounded H.264 compatibility conversion, adding CPU during viewing. These
are prototype FFmpeg measurements, not an integrated archive worker or a demonstrated sustained production rate.
Choose a quality level only after inspecting the private comparison stills; this experiment did not test
representative daytime motion. H.264 and HEVC at CQ 28 **increase** the sampled fleet's storage.

Software encoding is a poor fit for the current load: the x264 levels consume **75 to 111 CPU core-hours/day**,
about **3 to 5 cores continuously**. SVT-AV1 preset 8/CRF 52 wins some original-camera and portrait comparisons,
but its fleet projection is about 16 GB/day at 82 CPU core-hours/day. Preset 6 costs about twice as much CPU
for similar bytes. GPU AV1 is unavailable on the measured RTX 3090.

All camera columns below are approximate **GB/day**, using two short clips per camera and the same source
inventory as the baseline. CPU is **additional transcode work**, including GPU-driver initialization,
decode/filter/encode/mux, with scoring and validation excluded. Zero for keeping the encoding does not mean
the existing capture or analysis pipeline uses no CPU. CRF and CQ numbers are not equal-quality labels.

| Candidate | Original | Landscape A | Portrait | Landscape B | Low-rate A | Low-rate B | Total GB/day | CPU core-h/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Keep current encoding | 13.67 | 7.92 | 7.13 | 7.57 | 0.70 | 0.75 | 37.7 | 0 additional |
| CPU x264 slow CRF 24 | 10.33 | 8.86 | 6.92 | 7.54 | 1.21 | 1.30 | 36.2 | 111.3 |
| CPU x264 slow CRF 30 | 3.86 | 4.57 | 3.23 | 3.00 | 0.70 | 0.76 | 16.1 | 94.8 |
| CPU x264 slow CRF 36 | 1.33 | 1.54 | 1.10 | 0.90 | 0.38 | 0.42 | 5.7 | 74.7 |
| CPU AV1 P8 CRF 52 | 3.16 | 5.72 | 3.19 | 2.85 | 0.50 | 0.55 | 16.0 | 81.7 |
| GPU H.264 P5 CQ 28 | 13.34 | 9.27 | 8.21 | 9.35 | 1.32 | 1.39 | 42.9 | 8.2 |
| GPU H.264 P5 CQ 36 | 3.54 | 3.81 | 3.05 | 3.06 | 0.78 | 0.86 | 15.1 | 8.4 |
| GPU HEVC P5 CQ 28 | 12.42 | 11.67 | 10.78 | 10.46 | 1.56 | 1.60 | 48.5 | 8.3 |
| GPU HEVC P5 CQ 36 | 3.51 | 4.16 | 3.65 | 2.84 | 0.69 | 0.74 | 15.6 | 8.3 |

The [aggregate measurement data](storage-codec-benchmark.json) includes each camera's source-byte denominator,
encoded bytes, duration, and CPU time so every table cell can be recomputed. It contains no media, runtime IDs,
or source paths. The byte ratio uses compressed packet sizes in decoded source-frame order for the same number of pictures
as each output, rather than mixing a twelve-second encode with its full 25 to 28-second source-copy rate.
The two sample cohorts are combined by bytes within each camera. Fixed CNK/index overhead is not separately
remodeled. These are short-sample projections, not a full-day audit or a total-storage cap. All 48 GPU outputs
contain exactly the selected source-frame count; no frame-rate reduction is part of this candidate table.

## AV1 archive

The tier worker can re-encode camera video to AV1 (SVT-AV1) once it is a day old. It is off by default
(`video_archive.av1 = false` in `config/defaults.toml`); turn it on in `pack.toml` and restart the tier worker
(`pack-services restart continuum-tier-worker`; a Settings change restarts it on its own).

```toml
[video_archive]
av1 = true
```

**Tradeoff.** AV1 saves disk at a sustained CPU cost. The measurements above found that 30 days of
H.264 fit the measured installation; they do not establish current free space or another machine's capacity.
Measure the actual disk and CPU budget before enabling the archive.

**What it does.**

- A camera chunk becomes AV1 `after_hours` (24) after it was recorded. Until then Review, face and pet crops, and
  the vision workers read the original H.264, and the recent window and live view never see AV1.
- Existing H.264 is converted too, newest first, except the last two days before the video window ages it out.
- Each chunk is re-encoded in place: same frames, times, and ids, so nothing derived from the video changes.
- A chunk a purge cut, one with B-frames, or one that won't decode stays H.264.
- Turning it off stops the pass. Video already converted stays AV1, and everything reads both.

**Cost.** One chunk is encoded at a time, at the lowest CPU priority (nice 19 and `SCHED_IDLE`), with SVT-AV1's
parallelism capped at `threads` (4), so it only takes CPU nothing else wants; if that isn't enough, it falls behind
rather than slowing anything, and the oldest backlog is the part left as H.264.

- Disk: on 12 live chunks from four cameras at night, preset 8 and CRF 52 left 44% of the H.264's bytes.
  Those chunks were about 2 to 4 seconds long and each its own GOP, which cost about 20% against the 10-second
  GOPs measured above (a 14-second trial: 328 KB in four chunk-sized encodes, 273 KB in one). A camera's chunk is
  now a minute (docs/CONTINUUM.md, "Video chunks"), and an archived one has a keyframe every 10 seconds.
- CPU: on the same chunks, about 0.042 core-seconds a frame, H.264 decode included. Across six cameras (about 140
  frames a second in all) that is about 140 core-hours a day, or 5.8 cores all day: more than the table's 82, which
  used SVT-AV1 4.2 on 12-second clips; the tier worker links the system's SVT-AV1 (2.3 on the Linux machine) and opens
  one encoder per chunk. Preset 10 took 0.017 core-seconds a frame (about 2.4 cores) and left 35% of the bytes, but a
  faster preset at the same CRF is a softer picture: compare stills before choosing it.
- Logs: a summary line every 10 minutes (`tier worker: AV1 archive: ...`): chunks converted, bytes before and after,
  chunks kept as H.264 by reason, and whether the backlog is done.

**Playback.** Review sends AV1 to the browser as it is (WebCodecs). Chrome, Edge, and Firefox decode it, and so does
Safari on recent Apple devices (M3 or A17 and newer). Elsewhere Review says the video "was archived as AV1, which this
browser can't play". Exports re-encode AV1 to H.264, so clips play everywhere.

**Settings** (`[video_archive]`): `av1` (off), `after_hours` (24, at least 1), `preset` (8), `crf` (52), `threads` (4).
The tier worker reads them from its environment at start (`CONTINUUM_AV1_*`, filled from config by the service
launcher), so a binary from before this change ignores them.

## Archive codec comparison method

The offline comparison uses twelve immutable video copies: two non-overlapping windows from each of the six
sources. Low/high labels select container bitrate, not visually verified quiet/motion scenes. Each source copy
covers about 25 to 28 seconds; each measured encode uses its first twelve seconds. These short overnight samples
cannot establish daytime motion quality or a steady-state annual storage rate.

The full-rate comparison uses x264 `slow` CRF 24/30/36 and SVT-AV1 presets 6/8, normal random-access prediction,
lookahead, and closed ten-second GOPs. AV1 CRF values span a different range because CRF numbers do not represent
equal quality across codecs. The installed measurement stack is FFmpeg 9.0.2 and SVT-AV1 4.2.0 on the Linux machine (Core i9-7900X). A separate
low-delay, one-second-GOP CBR experiment is not evidence for the offline archive policy.

SSIM and luma XPSNR compare decoded output against decoded stored H.264, measuring **additional generation loss**,
not fidelity to the original scene. Presentation-order frame indices align the comparisons, and displayed-frame
counts and decoded timestamp prefixes distinguish verified endpoint rounding from unmatched output.
Three software sample prefixes had ambiguous source timestamps; their quality scores remain provisional. Compare measured quality overlap; any bracketed rate interpolation
is an estimate, not a measured equal-quality encode. Reduced-FPS scores use the same sampled reference frames;
they do not score the omitted motion. FFmpeg documents the filters in its [filter reference](https://ffmpeg.org/ffmpeg-filters.html).

Each encode runs serially, niced, with idle disk priority and a two-logical-CPU affinity budget. CPU seconds divided
by video seconds equals core-hours per video-hour. This includes decode/filter/encode/mux work, but excludes the
separate quality and validation passes. Loaded-host wall time includes scheduling delays and is not a dedicated
machine throughput claim. Daily size projections scale the measured source inventory by encoded payload-rate
ratios; fixed chunk/index overhead and a changing scene workload remain additional uncertainty.

The GPU extension uses the same twelve copies, CUDA decoding, and NVENC preset P5/HQ with VBR constant-quality
values CQ 28/36, lookahead 20, spatial/temporal AQ, three B-frames, and at most 300 frames per GOP. Each run
selects the first 360 frames for nominal 30 fps or 120 for nominal 10 fps, assigning a frame-index clock so
capture-timestamp jitter cannot change the selected endpoint. Reference and encoded displayed-frame counts
must match. CQ and CRF values are different encoder controls; the comparison images matter more than their numbers.
[NVIDIA's FFmpeg guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/index.html)
describes the device-surface path and VBR-CQ controls.

The measured GPU is an RTX 3090. Runtime probes succeeded for `h264_nvenc` and `hevc_nvenc`; `av1_nvenc` returned
"Codec not supported" and "Provided device doesn't support required NVENC features". A compiled encoder name
is not hardware support. No alternate GPU or software codec was silently substituted.

## Measured media and disk occupancy

A single-camera, single-microphone inventory read 8.605 MB of chunk headers and
CRC-checked footers, not video frames or audio bodies. Files were deduplicated by inode. Two unfinished files
contribute to occupancy but not the closed-file rate samples. These are files present on disk, including residual
or grace-period copies.

| Tier and kind | Files present | File bytes, GB | Allocated disk, GB | Closed-file rate, GB/hour |
| --- | ---: | ---: | ---: | --- |
| Tier 1 video | 980 | 7.374 | 7.376 | 14.42, from 64 files covering about 128 s |
| Tier 1 audio | 741 | 0.184 | 0.186 | 0.348, from a 64-file cohort covering 160.21 source-sample seconds |
| Tier 2 video | 19,764 | 7.032 | 7.072 | 0.682 mean across five full hours |
| Tier 2 audio | 15,997 | 0.383 | 0.424 | About 0.0315 for the recent COP2 hours |

The Tier 1 audio cohort contained 64 closed PCM files, 161 batches, and 15,479,656 total file bytes. Reading only
the 20-byte format headers identified CPB2 FLAC24, mono 48 kHz throughout; no audio samples were read or decoded.
Sample counts gave 160.21 seconds, agreeing with the 160.2398-second capture hull, and no duplicate record IDs were
found. The resulting rates were 347.84 MB/source-sample-hour and 347.77 MB/capture-hull-hour, rounded here to
**0.348 GB/hour**. This is a short sample, not a full-day invariant.

For Tier 2 video, the full UTC hours beginning 10:00 through 14:00 contained 678.2 to 684.8 MB/hour, averaging
682.0 MB/hour. Tier 2 audio was 31.3 to 31.5 MB/hour for COP2 hours and about 40.5 to 40.8 MB/hour for COP1 hours.
These are serialized chunk-file rates, including their format overhead, not an encoding benchmark. They do not
measure short-term overlap while Tier 1 predecessors await retirement.

The filesystem reported 3.936 TB capacity and 1.870 TB available at inventory time. This report makes no disk-full
date prediction from a short sample. Models, logs, vector files, imported media, additional devices, and any retained
copies outside this inventory need separate accounting.

## Footprint under the 30-day window

Video holds steady at the window: about 38 GB/day times 30 days, **about 1.15 TB**, plus whatever imports add in
their own 30 days. Audio, the segment store, the catalog, and the knowledge store grow without a cap:

| Component | Daily | Year | Evidence |
| --- | ---: | ---: | --- |
| Video (six cameras) | 38 GB/day, held at 30 days | about 1.15 TB, steady | Six-camera baseline above |
| Audio (Opus) | about 0.76 GB/day per microphone | about 0.28 TB per microphone | COP2 Tier 2 rate above |
| Journal evidence (promoted clips and stills) | about 2 to 6 MB/day | under 2.2 GB | Estimate; `promoted_bytes` in each video-window run |
| Segment store, catalog, and knowledge store | not yet measured | not yet measured | No full-day measurement since the catalog reset |

The open cost is the segment store. A full day of its growth, by stream, is the next useful measurement.
