# Continuum storage

How recordings reach disk and leave it, and the two stores beside the broker. [CONTINUUM.md](CONTINUUM.md) is the
overview, and [STORAGE.md](STORAGE.md) covers retention policy and disk use.

## Tiers

| | Tier 0 hot | Tier 1 warm | Tier 2 cold |
|---|---|---|---|
| Where | The shared memory arena | Chunk files under `state/continuum/tier/` | `state/continuum/tier2/` |
| Kept | About 10 seconds per capture stream | Video 30 days; audio and the rest until purged | Until purged |
| Video | The camera's native payload: H.264 access units, or JPEG or raw frames | H.264 access units, one per record; MJPEG frames as they are | None |
| Audio | PCM at the device's rate, every channel | Opus, encoded as each chunk closes | Opus chunks an earlier version wrote |

Nothing writes Tier 2 now. Its chunks are still cataloged, read, and purged like any other. Nothing is recompressed
after it is committed, except by the opt-in AV1 archive.

## The tier worker

`continuum-tier-worker` follows each capture stream by durable subscription and appends each payload to the stream's
open chunk file as it lands. It never holds the catalog: it hands each chunk to the broker with `CommitChunk`.

- A slot is marked flushed, and the subscription's position saved, only once its chunk part is committed. A crash
  loses nothing: recovery finalizes open files and commits what the catalog lacks.
- It never waits for a chunk's worth of data, so a capture ring only has to cover the flush lag.
- A commit the broker refuses for good sets that chunk aside as `.rejected` and does not hold the stream back.
- A camera layout change (size, stride, format, or flip) or a new capture epoch closes the chunk and starts another.
- It also copies large derived payloads to single-record sidecar files and attaches them with `CommitPayloadCopy`.
- It carries out purges on its files before acknowledging them ([CONTINUUM-BROKER.md](CONTINUUM-BROKER.md#forgetting)).

## Chunk files

A chunk file is the tier worker's own append-only container (`continuum-tier`'s `chunk` module): a header,
self-describing records (length, times, sequence, and a CRC each), and the frame index as a footer.

- It is crash safe: a crash loses at most the record being written.
- A purge cuts records as tombstones.
- The gateway hands H.264 records to the browser as they are, with no demuxing.
- It is not a file a video player opens. `continuum-export --out clip.mp4 [--from-ns N] [--to-ns N] CHUNK...` writes
  a range as an ordinary MP4.

Files live under `streams/<stream>/<YYYYMMDDHH>/`, the UTC hour they were created in. A chunk's locator is relative
to the state directory, so a moved install still reads its chunks.

### Video: one file a minute, committed in parts

A camera's chunk is one file and one catalog row per minute, closed on the first keyframe after the minute is up.
Because Tier 0 holds only 10 seconds, the tier worker commits the open chunk's new records every 2 seconds, on a
keyframe. Each part is a `CommitChunk` with `first_frame` set, and the chunk gets a new commit sequence, so a reader
following by sequence sees it grow.

- A chunk's id is a UUIDv7 derived from its file's name, so recovery after a crash names the same chunk.
- An open file can be longer than its catalog row says. Read it by scanning its records.
- A part the catalog refuses for good (a purge hid the chunk, or timestamps stepped back) closes the file; the
  uncommitted records move to a new chunk.
- Purges, the archive, and age-outs act on closed files only.

### Audio

A microphone's chunk closes every 15 seconds. As it closes, and before it is committed, the tier worker encodes it to
Opus with the same payload ids, times, and sequence numbers, and keeps the original block stamps losslessly. A chunk
Opus cannot represent (an unknown rate or an unsupported channel layout) is committed as lossless PCM.

### AV1 archive

Off by default (`video_archive.av1`). When on, the tier worker re-encodes each camera chunk to AV1 once it is
`after_hours` old (24), in the same one-unit records, and swaps the file in by rename. Readers go by the chunk's
format. [STORAGE.md](STORAGE.md) has the policy.

## Reading recordings

A query returns a chunk as one payload whose locator names the file. A consumer on the same machine reads the records
itself; in Python, `continuum_py.read.records` turns any query item into its payloads. If the locator went stale
between the query and the read, it looks again once. The file is open only during the read, so a purge's rewrite is
never held up.

`continuum_tier::recent::H264Window` gives any model the last few minutes of a camera's native H.264, from the last
keyframe, and then follows the live edge. A gap in device sequence or a new capture epoch is reported, never bridged.

Decoded frames for the vision workers come from `pack-see-decode` ([VISION.md](VISION.md#shared-decoding)). The
broker never decodes.

## The video window

`pack-video-window` runs nightly and ages out `*.video` chunks committed more than `video_window.days` (30) ago.

- Ages count from when a payload was committed, not its own time, so imported old media is not aged out at once.
- An age-out removes whole chunks and nothing derived from them. Tracks, captions, and the journal stay.
- The removed spans are recorded, and the gateway reports them as `expired`.

## The segment store

`continuum-segments` holds everything workers derive, under `state/segments/<stream>/<slot>/`: zstd-compressed Arrow
IPC parts, one per stream, slot, and UTC hour. A slot is a camera, a microphone, or a writer.

- The filesystem is the index. Any number of processes read and write with no shared database.
- One writer per stream and slot, held by `.writer.lock`. `.seq` keeps the next row number, so a writer reopened at a
  checkpoint never repeats a row.
- An open part holds one record batch per append and can be followed as it grows. When it closes, its writer rewrites
  it as one batch.
- A forget appends to `.deletions`, which every read honors at once, then rewrites the parts that held the rows.
- Python uses `src/liberational/segment_log.py`, which declares each stream's typed columns. Readers that show rows
  take records (`Log.range`); readers that count or join take columns (`Log.table`, `Log.covered`).

## The vector worker

`continuum-vector-worker` keeps LanceDB tables in `state/continuum/vector/`, one per embedding space, plus a
full-text table.

- **In.** It follows `voice.embedding` and `face.embedding` in the segment store, the journal's
  `journal.embedding.<digest>` streams on the broker, and each `journal.entry`'s text.
- **Out.** A consumer commits a `vector.query` or `text.query` payload and waits for the `vector.search` or
  `text.search` result whose parent is its query. In Python: `continuum_py.search.Searcher`. A search with no answer
  raises `TimeoutError`; it never hangs. `k` is 1 to 40.
- **Purges.** It deletes the named rows, compacts every table, rebuilds its text index, and prunes old versions
  before it acknowledges, because a LanceDB delete alone leaves the bytes on disk.
- Its tables are rebuildable from their sources.
