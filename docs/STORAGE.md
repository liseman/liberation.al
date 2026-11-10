# Storage and retention

What is kept, for how long, and roughly how much disk it takes. [ARCHITECTURE.md](ARCHITECTURE.md#where-data-lives)
lists the stores, and [CONTINUUM-STORAGE.md](CONTINUUM-STORAGE.md) describes how recordings are written.

## The rule

Past 30 days, the meaning is kept and the raw measurement is dropped. Camera video goes; what was derived from it,
speech audio, cited journal moments, and everything the owner taught stay.

## Retention

`pack-video-window` (the scheduled `video-window` service; pass `--dry-run` to count without deleting) runs nightly.
For each ended local day older than `video_window.days` (30) whose journal day page is written, oldest first, at most 7
days per run (`src/liberational/video_window.py`, `src/liberational/retention.py`):

| Data | Past the window |
|---|---|
| Camera video | Aged out. Review and Ask report the footage as expired |
| Moments a day page cites | Promoted ahead of aging out as a 6 s H.264 clip (480 px, 8 fps) and a 640 px JPEG still under `state/journal_evidence/<YYYY-MM-DD>/`, at most 3 per cited hour |
| Audio | Chunks no speech touches are aged out; speech, with 5 s on either side, stays |
| Dense coverage (`action.coverage`, `walkby.coverage`, `animal.coverage`, `zone.coverage`) | Condensed to intervals in `<stream>.condensed`, which segment reads merge in automatically |
| Device sightings (`network.device_observation`, `bluetooth.sighting`) | Condensed to visits in `<stream>.condensed` the same way |
| Positions (`position.sample`, `position.placed`, `position.path`) | Thinned in `<stream>.condensed` to one sample per path or track every 10 s (plus the final sample; zone rows stay whole) |
| Per-frame rows (`pose.track`, `face.match`, `face.signal`, `face.embedding`) | Dropped; kept spans already summarize them |
| Spans and events, movement tracks, captions, transcripts, voice and animal embeddings, objects, and the journal | Kept |
| Taught state, device links, and face and voice references | Kept; never decays |

- Video ages from when the broker committed it (tracked per stream in `state/video_window.json`), so imported old media
  gets a full window.
- A day without a journal day page stays uncondensed until its day page is written.
- If free space on the recordings volume drops below `video_window.free_floor_percent` (15), the oldest video is aged
  out early an hour at a time (at most 48 hours per run), never touching the newest 24 hours.
- `state/retention.json` records each finished day and the row counts condensed or dropped.

Forgetting a person and resetting the catalog are separate operations. `uv run pack-reset-catalog --keep-taught` prints
what a machine-data reset removes; add `--yes` to carry it out, and add `--keep-places` to keep `state/places/`.
[MIGRATING.md](../MIGRATING.md) has the full procedure.

## How much disk

Measured on the Linux machine with six cameras and one microphone. Rates vary with scene motion, resolution, and speech
activity.

| Component | Rate | Steady state or yearly |
|---|---|---|
| Video, six cameras (one 960x540 and three 720p at 30 fps, two 896x512 at 10 fps), native H.264 | About 38 GB a day | About 1.15 TB, held at 30 days (roughly half with AV1 archive on) |
| Audio (Opus), per microphone | About 0.76 GB a day | About 0.28 TB a year before silent-chunk aging |
| Journal evidence (`state/journal_evidence/`) | About 2 to 6 MB a day (estimated) | Under 2.2 GB a year |
| Catalog (`catalog.redb`), vectors (`vector/`), segments (`state/segments/`), and `knowledge.sqlite3` | Scales with stream count and speech/event rate | Live `catalog.redb` and `vector/` sizes appear in Reports, System health; [tools/catalog_sizing.py](../tools/catalog_sizing.py) estimates per-stream catalog growth |
| A place's maps (`state/places/`) | A few MB to 30 MB each | Only the newest 2 maps per place are kept |
| Models and model runtimes under `data/` | Several GB each | Rebuilt by their tasks in [mise.toml](../mise.toml) |

## AV1 archive

`continuum-tier-worker` can re-encode closed H.264 camera chunks to AV1 (`SVT-AV1`, preset 8 and CRF 52 by default)
once they reach `video_archive.after_hours` (24). It is off by default because it trades roughly half the archived
video bytes for sustained background CPU.

Release installs (`continuum_daemons` and `continuum_py` wheels) bundle a private FFmpeg built with `SVT-AV1` and
`dav1d`, so the archive pass needs no source checkout or system FFmpeg. In a source checkout, `continuum-tier-worker`
links the system's shared FFmpeg libraries, which must include `libsvtav1` ([CONTINUUM-SDK.md](CONTINUUM-SDK.md)).

To turn it on, set `av1 = true` in `pack.toml` and restart the tier worker:

```toml
[video_archive]
av1 = true
```

```sh
uv run pack-services restart continuum-tier-worker
```

- Each eligible chunk is re-encoded in place (with a keyframe every 10 s and the same frames, timestamps, sequence
  numbers, and payload IDs) and swapped in by rename, so derived records never change. Until then, readers see the
  original H.264.
- Existing H.264 backlog is converted newest first. When the video window is on, conversion stops 2 days before a chunk
  expires so footage about to age out is not re-encoded; when the video window is off, the backlog goes back 30 days.
- MJPEG chunks, chunks cut by a purge, chunks with out-of-order frames (B-frames), multi-unit records, chunks whose
  first frame is not a keyframe, and chunks with any undecodable frame stay as recorded.
- One chunk is encoded at a time at lowest process priority. If the encoder falls behind, live capture and queries are
  unaffected.
- Turning `video_archive.av1` off stops new conversions; chunks already converted stay AV1, and readers handle both.
- Review plays AV1 where the browser supports it (Chrome, Edge, Firefox, and Safari on recent Apple hardware) and says
  so where it cannot. MP4 exports (`continuum-export`) re-encode AV1 chunks to H.264.

All retention and archive settings are documented under `[video_window]` and `[video_archive]` in
[config/defaults.toml](../config/defaults.toml), and the `video-window` and `continuum-tier-worker` services are
declared in [services/services.toml](../services/services.toml).
