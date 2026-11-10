# Storage and retention

What is kept, for how long, and roughly how much disk it takes. [ARCHITECTURE.md](ARCHITECTURE.md#where-data-lives)
lists the stores and [CONTINUUM-STORAGE.md](CONTINUUM-STORAGE.md) how recordings are written.

## The rule

Past 30 days, the meaning is kept and the measurement is dropped. Video goes; what was derived from it, speech
audio, and everything the owner taught stay.

## Retention

`pack-video-window` (the `video-window` service) runs nightly. For each ended local day older than
`video_window.days` (30) whose journal day page is written, oldest first, at most 7 days a night
(`src/liberational/retention.py`):

| Data | Past the window |
|---|---|
| Camera video | Aged out. Review and Ask say the footage expired |
| Moments a day page cites | Kept for good as a 6 s clip and a still under `state/journal_evidence/`, at most 3 per cited hour |
| Audio | Chunks no speech touches are aged out; speech, with 5 s either side, stays |
| Dense coverage (`action.coverage`, `walkby.coverage`, `animal.coverage`, `zone.coverage`) | Condensed to intervals in a `<stream>.condensed` companion that reads merge in |
| Device sightings (`network.device_observation`, `bluetooth.sighting`) | Condensed to visits the same way |
| Positions (`position.sample`) | Thinned to one sample per track every 10 s |
| Per-frame rows (`pose.track`, `face.match`, `face.signal`, `face.embedding`) | Dropped; the kept spans already summarize them |
| Spans and events, captions, transcripts, voice embeddings, and the journal | Kept |
| Taught state and face references | Kept; never decays |

- Video ages from when the broker committed it, so imported old media gets a full window.
- A day without a day page is kept whole until it has one.
- If free space on the recordings volume drops below `video_window.free_floor_percent` (15), the oldest video goes
  early, an hour at a time, never the newest day.
- `state/retention.json` records each finished day and what went.

Forgetting a person and resetting the catalog are separate operations. `pack-reset-catalog --keep-taught` deletes
all machine-made data at once; `--keep-places` also keeps `state/places`. [MIGRATING.md](../MIGRATING.md) has the
steps.

## How much disk

Measured on the Linux machine with six cameras and one microphone. These are short samples, not guarantees.

| Component | Rate | Steady state or yearly |
|---|---|---|
| Video, six cameras (one 960x540 and three 720p at 30 fps, two 896x512 at 10 fps) | About 38 GB a day | About 1.15 TB, held at 30 days |
| Audio (Opus), per microphone | About 0.76 GB a day | About 0.28 TB a year |
| Journal evidence | About 2 to 6 MB a day (an estimate) | Under 2.2 GB a year |
| Segment store, catalog, and knowledge store | Not yet measured | |
| A place's maps | A few MB to 30 MB each; the newest 2 are kept | |
| Models and model runtimes under `data/` | Several GB each; rebuilt by their mise tasks | |

Decision: keep the cameras' H.264 as recorded. Thirty days of it fits the development machine, and re-encoding
costs sustained CPU. GPU H.264 or HEVC at a higher quantizer would cut video to about 15 GB a day if space runs
short; software encoders cost 3 to 5 cores continuously.

## AV1 archive

The tier worker can re-encode camera video to AV1 (SVT-AV1) once it is a day old. It is off by default. It saves
disk at a sustained CPU cost, so measure both before turning it on:

```toml
[video_archive]
av1 = true
```

Then restart the tier worker (`pack-services restart continuum-tier-worker`).

- A chunk becomes AV1 `after_hours` (24) after recording. Until then every reader sees the original H.264.
- Existing H.264 is converted too, newest first.
- Each chunk is re-encoded in place with the same frames, times, and ids, so nothing derived changes.
- A chunk a purge cut, one with B-frames, or one that will not decode stays H.264.
- Turning it off stops the pass. Converted video stays AV1, and everything reads both.
- One chunk is encoded at a time at the lowest priority, with `threads` (4) encoder threads. If that is not enough
  it falls behind; it never slows anything else.
- On a short sample, preset 8 and CRF 52 left about 44% of the H.264's bytes at about 0.04 core-seconds a frame.
- Review plays AV1 where the browser decodes it (Chrome, Edge, Firefox, and Safari on recent Apple devices), and
  says so where it cannot. Exports re-encode to H.264.
- A wheel install has no AV1 encoder ([CONTINUUM-SDK.md](CONTINUUM-SDK.md#wheels)).

Settings under `[video_archive]`: `av1`, `after_hours`, `preset` (8), `crf` (52), and `threads` (4).
