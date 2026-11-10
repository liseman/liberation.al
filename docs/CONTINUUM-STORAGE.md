# Continuum storage

Reference for Tier 0, Tier 1, and Tier 2 recordings, the chunk container, the segment store, and the vector worker.
[CONTINUUM.md](CONTINUUM.md) is the overview, and [STORAGE.md](STORAGE.md) covers retention policy and disk use.

## Tiers

| | Tier 0 hot | Tier 1 warm | Tier 2 cold |
|---|---|---|---|
| Where | The shared memory arena | Chunk files under `state/continuum/tier/` | `state/continuum/tier2/` |
| Kept | 10 seconds per compressed camera or audio stream; 2 seconds per raw camera | Video for `video_window.days` (30); audio with speech and the rest until purged | Until purged |
| Video | H.264 access units (network cameras, and Linux MJPEG cameras encoded at capture), JPEG frames, or raw frames | `H264Au` access units (or `Av1Au` temporal units after the archive pass), one per record; `Passthrough` JPEG frames; `Lz4` raw frames when FFmpeg is absent | None |
| Audio | PCM at the device's rate, every channel | `Opus`, encoded as each chunk closes (`PcmBatch` FLAC or LZ4 fallback) | `Opus` |

Tier 2 is read-only cold storage: nothing writes to it, and chunks there are cataloged, read, and purged like Tier 1
chunks. Committed chunks are never recompressed except by the opt-in AV1 archive.

## The tier worker

`continuum-tier-worker` follows each capture stream by durable subscription and appends each payload to the stream's
open chunk file as it lands. It never holds the catalog: it hands each chunk to the broker with `CommitChunk`.

- A slot is marked flushed, and the subscription's position saved, only once its chunk part is committed. A crash loses
  nothing: recovery finalizes open files and commits what the catalog lacks.
- It never waits for a chunk's worth of data, so a capture ring only has to cover the flush lag.
- A commit the broker refuses for good sets that chunk aside as `.rejected` and does not hold the stream back.
- A camera layout change (size, stride, format, or flip) or a new capture epoch closes the chunk and starts another.
- It copies large derived payloads to single-record sidecar files and attaches them with `CommitPayloadCopy`.
- It carries out purges on its files before acknowledging them ([CONTINUUM-BROKER.md](CONTINUUM-BROKER.md#forgetting)).

## Chunk files

A chunk file is an append-only little-endian container (`continuum-tier`'s `chunk` module, version 2):

| Section | Layout |
|---|---|
| Header (24 B) | `"CNK1"`, `version: u16` (2; version 1 without `id` is still read), `flags: u16` (bit 0: native camera flip), `stream_id: u32`, `format: u32`, `created_ns: u64` |
| Frame record | `tag: u8` (`"R"` live, `"T"` tombstone), `len: u32`, `start_ns: u64`, `end_ns: u64`, `seq: u32`, `id: [u8; 16]` (Tier 0 `Uuid`), `crc32: u32`, payload bytes |
| Provenance record | `tag: u8` (`"P"`), written before a frame for crash recovery only (`device_seq: u64`, `epoch: u32`, `flags: u8`); omitted from the footer index and dropped on purge rewrite |
| Footer | Per-frame entries (`start_ns`, `end_ns`, `seq`, `offset`, `len`, `cut`, `id`), `count: u32`, `crc32: u32`, `"CNKX"` |

`ChunkFormat` tags are `Passthrough` (1), `Lz4` (2), `Flac` (3), `ArrowIpc` (4), `Opus` (6), `PcmBatch` (7), `Mp4` (8,
registered in place and read through `inplace_video` rather than `ChunkReader`), `H264Au` (10), and `Av1Au` (11).

- Crash safe: a crash loses at most the record being written, and a corrupt record is skipped by resyncing on the next
  valid CRC.
- A purge turns cut records into tombstones in place so frame indexes never shift.
- The gateway hands `H264Au` and `Av1Au` records to the browser as they are, with no demuxing.
- A chunk file is not an MP4. Export a range for a media player with `continuum-export`:

```sh
continuum-export --out clip.mp4 [--from-ns N] [--to-ns N] [--codec auto|copy|h264] [--force] CHUNK...
```

Files live under `streams/<stream>/<YYYYMMDDHH>/`, the UTC hour they were created in. When `--state-dir` is set, a
chunk's locator is relative to the state directory, so a moved install still reads its chunks.

### Video: one file a minute, committed in parts

A camera's chunk is one file and one catalog row per minute, closed on the first keyframe after the minute is up.
Because Tier 0 holds at most 10 seconds, the tier worker commits the open chunk's new records every 2 seconds, on a
keyframe. Each part is a `CommitChunk` with `first_frame` set, and the chunk gets a new commit sequence, so a reader
following by sequence sees it grow.

- A chunk's id is a UUIDv7 derived from its file's name, so recovery after a crash names the same chunk.
- An open file can be longer than its catalog row says. Read it by scanning its records.
- A part the catalog refuses for good (a purge hid the chunk, or timestamps stepped back) closes the file; the
  uncommitted records move to a new chunk.
- Purges, the archive, and age-outs act on closed files only.

### Audio

A microphone's chunk closes every 15 seconds. As it closes, and before it is committed, the tier worker encodes it to
`Opus` with the same payload ids, times, and sequence numbers, and keeps the original block stamps losslessly. A chunk
Opus cannot represent (an unknown rate or an unsupported channel layout) is committed as lossless `PcmBatch` (FLAC, or
LZ4 if FLAC rejects the parameters).

### AV1 archive

Off by default (`video_archive.av1`, passed to the tier worker as `CONTINUUM_AV1_ARCHIVE`). When on, the tier worker
re-encodes closed `H264Au` camera chunks older than `video_archive.after_hours` (24, minimum 1) to `Av1Au` with
SVT-AV1, skipping chunks within two days of `video_window.days` expiry. It swaps the finished file in by rename and
updates catalog offsets with `RewriteChunk`.

SVT-AV1 and dav1d are built into the wheels' private FFmpeg, so the archive pass and reading `Av1Au` chunks work in a
wheel install as well as a checkout (`continuum-tier-worker --capabilities` reports `"av1_archive": true`).
[STORAGE.md](STORAGE.md) covers the policy and settings.

## Reading recordings

A query returns a chunk as one payload whose locator names the file. A consumer on the same machine reads the records
itself; in Python, `continuum_py.read.records` turns any query item into its payloads. If the locator went stale
between the query and the read, it looks again once. The file is open only during the read, so a purge's rewrite is
never held up.

`continuum_tier::recent::H264Window` gives any model the last few minutes of a camera's native H.264, from the last
keyframe, and then follows the live edge. A gap in device sequence or a new capture epoch is reported, never bridged.

Decoded frames for the vision workers come from `pack-see-decode` ([VISION.md](VISION.md#shared-decoding)). The broker
never decodes.

## The video window

`pack-video-window` (`src/liberational/video_window.py`) runs nightly:

1. It promotes moments cited by the journal's day pages (`src/liberational/memory/journal_evidence.py`) so their clips,
   stills, and captions stay after the source video ages out.
2. It ages out `*.video` chunks committed more than `video_window.days` (30) ago, and ages out earlier hours from the
   oldest kept video when free space falls below `video_window.free_floor_percent` (never into the newest day).
3. It runs the retention pass (`src/liberational/retention.py`) for ended days past the window whose day page is
   written, condensing dense segment streams and aging out audio chunks no speech touches.

- Ages count from when a payload was committed, not its capture time, so imported media keeps the full window.
- An age-out removes only the named chunks and their absorbed frame rows; derived tracks, captions, and the journal
  stay.
- Removed video spans are recorded in the catalog, and the gateway reports them as `expired`.
  [STORAGE.md](STORAGE.md) covers retention policy.

## The segment store

`continuum-segments` holds machine-derived time series under `state/segments/<stream>/<slot>/`: zstd-compressed Arrow
IPC stream files, one per stream, slot (a camera, a microphone, or a writer), and UTC hour.

| Path in `<stream>/<slot>/` | Role |
|---|---|
| `<YYYYMMDDHH>-<part>-s<first>.arrows` | Open part, one record batch per append |
| `<YYYYMMDDHH>-<part>-s<first>-<end>[-v<n>].arrows` | Closed part (`row_seq` in `first..end`), rewritten as one batch (`-v1`, incremented on rewrite) |
| `.writer.lock` | Lock held by the single writer for that stream and slot |
| `.seq` | Next `row_seq`, kept across retention so reopened writers never repeat a row |
| `.deletions` | Append-only forget log honored immediately on read before affected parts are rewritten |

- The filesystem is the index; processes read and write with no shared database.
- Every row carries `ts_ns` (UTC nanoseconds, which selects the hour) and `row_seq` (strictly increasing per stream and
  slot). Interval streams (tracks, visits, crossings, and `<stage>.coverage`) also carry `end_ns`.
- Python uses `src/liberational/segment_log.py`, which declares each stream's typed columns (`Log.range` for records,
  `Log.table` and `Log.covered` for Arrow columns).

## The vector worker

`continuum-vector-worker` keeps LanceDB tables in `state/continuum/vector/`, one per embedding space, plus a full-text
table.

- **In.** It follows `voice.embedding` and `face.embedding` in the segment store, the journal's
  `journal.embedding.<digest>` streams on the broker, and each `journal.entry`'s text.
- **Out.** A consumer commits a `vector.query` or `text.query` payload and waits for the `vector.search` or
  `text.search` result whose parent is its query. In Python: `continuum_py.search.Searcher`. A search with no answer
  raises `TimeoutError`; it never hangs. `k` is 1 to 40.
- **Purges.** It deletes the named rows, compacts every table, rebuilds its text index, and prunes old versions before
  it acknowledges, because a LanceDB delete alone leaves the bytes on disk.
- Its tables are rebuildable from their sources.
