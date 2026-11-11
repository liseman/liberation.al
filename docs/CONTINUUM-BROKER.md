# Continuum broker

The broker's shared memory, socket protocol, scheduler, catalog, and purges. [CONTINUUM.md](CONTINUUM.md) is the
overview. The FlatBuffers schemas in `continuum/schemas/` are the reference for every message and field.

## Nothing stalls the broker

A voice-turn preemption must go out within 5 ms whatever else is happening, so no client and no slow job may hold up
anything else.

- **The scheduler's task does memory-only work** on its own thread. It owns sessions, the availability index, and the
  scheduler, and never touches the catalog.
- **One writer thread writes the catalog** in short transactions, with an urgent queue (registration and ingest) and
  a background queue (purges and other long jobs). All of a tick's ingest notices commit in one transaction. Only one
  commit a second fsyncs, so a crash can lose about the last second of commits; the tier worker absorbs those frames
  again.
- **The index follows the catalog.** The writer commits, then tells the scheduler. The index never shows what is not
  committed.
- **Reads run in parallel** on a reader pool, each on its own snapshot.
- **The voice turn and control messages go first**, ahead of ordinary requests.
- **Sessions are isolated.** Every queue is bounded. A session that stops reading is closed; nobody waits for it.
- **A failure stays contained or stops the broker.** A panicking read loses itself. A panicking writer or scheduler
  aborts the broker so its supervisor restarts it.

A clean stop saves the availability index in the catalog, and the next start restores it instead of rebuilding from
every payload. Any mismatch falls back to the full rebuild.

## Tier 0: the shared memory arena

`continuum-shm` is the allocator. Two rules shape it: every offset is a multiple of 16,384 bytes, and a slow reader
never stalls a writer.

- **Alignment.** `PAGE_ALIGN` (16 KB) satisfies Apple Silicon's 16 KB pages and Linux's 4 KB pages. It lets a Metal
  worker wrap a whole slot without a copy and a CUDA worker register the whole mapping once.
- **Layout.** One arena per broker, created at start: a superblock, a slot directory, slab size classes (`64 KB`,
  `2 MB`, and `16 MB` in the baseline profiles), and a ring region (up to 256 carved rings).
- **Sizing.** The daemon combines the machine profile's classes with reservations for the manifest's enabled cameras
  (`ring_seconds`: 10 seconds by default for compressed and network cameras, 2 seconds for raw cameras and macOS USB
  cameras). On Linux, unless `--profile` is passed explicitly, the daemon shrinks the profile's headroom to fit half of
  `/dev/shm` and refuses to start when even the smallest camera plan does not fit, saying what to change. A broker
  started through the library with default Tier 0 settings (`Config::tier0 = None`) fits its profile's slot classes and
  ring region to half of `/dev/shm` by the same rule (halving slot counts up to 10 times, never below one slot).
  `continuum-broker --print-tier0-plan` prints the daemon's plan.
- **Naming.** `/cnt.<hash>.<generation>`, with the generation kept beside the socket in `<socket>.arena`. The arena is
  never resized; a new size is a new generation. A killed broker leaves its arena behind, and the next broker on that
  socket unlinks it.

### Slots

A producer claims a slot with a compare-and-swap, writes into the slab, marks it ready, and sends an `IngestNotice`
with the payload header. Claiming needs no message. Each slot has a generation, so a reader holding a stale
reference detects reuse.

The startup plan reserves each enabled camera's `ring_seconds` in the smallest 16 KB-aligned slab classes that fit its
frames (and geometric classes up to `rtsp.ring_bytes` for network cameras). Capture claims slots in that class with
`claim_overwriting` and never waits: when none is free, it overwrites the oldest unpinned ready slot, flushed or not.
An unflushed overwrite that leaves a span with no other copy is logged and counted as an archive gap.

### Rings

A fixed-stride stream (PCM audio) gets a ring carved from the ring region (`ring_seconds`, 10 seconds by default): one
producer, any number of readers, each with its own position.

- Records carry sequence numbers (`head` and `started` in the ring control block), so a reader copies under a seqlock,
  sees any gap when lapped, and never reads torn data.
- The catalog indexes a ring in batches (1 second by default), not per record.
- A live reader follows the ring's sequence word and wakes on a futex (or the macOS equivalent), so it sees a block
  within about one block period. Nothing that answers in real time waits for a batch to close.
- Each PCM record starts with a 16-byte stamp: capture time on the monotonic clock and the first sample's index.

### Reading

A query answered from Tier 0 pins the slot for 5 seconds (`MAX_LEASE` is 60 seconds) and returns its locator. The
reader maps the arena once and reads in place. Dropping the pin is a decrement in shared memory. A disconnected
client's pins are dropped for it.

When a size class runs out, the broker reclaims the oldest ready slots, flushed ones first. It never reclaims an
unflushed slot of a stream the tier worker follows; it warns of pressure instead. A slot pinned past its lease is
reclaimed once it is flushed, and the reader detects the generation change and can query again from Tier 1.

## Payload headers

Every payload has a `PayloadHeader` ([`continuum/schemas/continuum.fbs`](../continuum/schemas/continuum.fbs)), stored
in the catalog and sent in messages. The bytes sit in a slab, inline in the catalog, or in a chunk.

| Field | Meaning |
|---|---|
| `id` | 16-byte UUIDv7 |
| `stream_id` | Stream id (`u32`) |
| `seq` | Stream sequence number (`u32`) |
| `start_ns`, `end_ns` | Span in UTC nanoseconds (`u64`); equal for an instant |
| `layer` | `Layer1Raw`, `Layer2Projected`, or `Layer3Interpretive` |
| `format` | `RawBytes`, `EncodedVideo`, `EncodedAudio`, `ArrowRecordBatch`, `DlpackTensor`, or `TextUtf8` |
| `shape_name`, `shape_version` | Named shape and its version |
| `producer` | `producer_id`, `model_name`, `model_version`, `weights_hash`, and `code_hash` |
| `parents` | Exact parent payload ids |
| `evidence` | Supporting records (`id`, `start_ns`, `end_ns`) for layer 2 and 3 claims |
| `subjects` | Entity ids the payload is about, indexed for entity purges |
| `confidence` | Optional detector or model score (`f32`) |
| `source_tier` | Coldest tier any input came from; compressed inputs score lower |
| `compute` | `wall_ns`, `gpu_ns`, `bytes_in`, and `bytes_out` |
| `capture` | Sensor stamps (`mono_ns`, `device_ts`, `arrival_ns`, `device_seq`), layout (`stride`, `width`, `height`, `flip`, `sample_rate_hz`), `duration_ns`, and `capture_epoch` |

A header is immutable. Where the bytes are is a separate catalog table, since bytes move between tiers. A payload of
at most 64 KiB can be stored inline in the catalog. [`src/liberational/continuum.py`](../src/liberational/continuum.py)
converts Python UUIDv7 strings and float epoch-second timestamps to and from Continuum's ids and `u64` nanoseconds.

Two frames are continuous only within one nonzero `capture_epoch` with consecutive `device_seq` values. A frame's
`end_ns` is extended only when its source attests `duration_ns`; it is never guessed from the nominal rate.

## Protocol

Control messages are FlatBuffers tables in a `ControlEnvelope { request_id, payload }`, one `u32`-length-prefixed
envelope at a time. A reply carries its request's id. Messages the broker sends on its own carry 0.

| Group | Messages | Purpose |
|---|---|---|
| Connecting | `Hello`, `Welcome`, `Heartbeat`, `Goodbye`, `Error` | Handshake passes the arena file descriptor (`SCM_RIGHTS`) and current voice turn; mismatches in major schema version are refused |
| Registration | `RegisterStream`, `StreamRegistered`, `RegisterWorker`, `WorkerRegistered` | Declare a stream or a worker's capabilities; grants belong to the connection |
| Ingest | `IngestNotice`, `Ingested`, `IngestRejected` | Publish a written slot or inline payload; identical repeats are idempotent, and differing headers for one id conflict |
| Query | `Query`, `QueryResult`, `ListStreams`, `StreamList` | Read a stream by time range or in commit order after a sequence, or list streams |
| Promises | `PromiseQuote`, `CommitPromise`, `PromiseAccepted`, `CancelPromise`, `FulfillPromise`, `PromiseFailed` | Quote the missing part of a query, then dispatch jobs on commit |
| Jobs and leases | `DispatchJob`, `JobRejected`, `QuoteRequest`, `LeaseGrant`, `RenewLease`, `LeaseReleased`, `CommitLease`, `AbortLease`, `LeasePreempted` | Allocate a worker's output slots and compute share |
| Subscriptions | `Subscribe`, `Subscribed`, `StreamEvent`, `CaughtUp`, `AckEvents`, `Unsubscribe`, `Unsubscribed` | Follow streams live, from a time, or from a durable saved position |
| Coverage | `CoverageGapQuery`, `CoverageGapResponse` | Ranges a stage has not processed |
| System state | `SetSystemState`, `SystemStateChanged` | Read and broadcast the voice turn |
| Purge and forgotten names | `Purge`, `PurgeNotice`, `PurgeAck`, `PurgeResult`, `ListPurges`, `PurgeList`, `GetPurge`, `PurgeRecord`, `ForgetKeeper`, `KeeperForgotten`, `RememberForgotten`, `ForgottenRemembered`, `IsForgotten`, `ForgottenStatus`, `ListForgotten`, `ForgottenList` | Cascading purges, keeper acknowledgements, and keyed hashes of forgotten names |
| Chunks and copies | `CommitChunk`, `ChunkCommitted`, `RewriteChunk`, `ChunkRewritten`, `CommitPayloadCopy`, `PayloadCopyCommitted`, `GetChunk`, `ChunkInfo`, `ResolveFrames`, `FramesResolved` | Tier-worker chunk commits, rewrites, sidecar copies, and absorbed-frame lookups |
| Worker state | `SaveWorkerState`, `WorkerStateSaved`, `LoadWorkerState`, `WorkerStateLoaded`, `ForgetWorkerState`, `WorkerStateForgotten` | One durable checkpoint row per worker |
| Aging and retention | `AgeOutFrames`, `FramesAgedOut`, `ListAgedOut`, `AgedOutList`, `CommitSeqAt`, `CommitSeq`, `GetStreamGeneration`, `StreamGeneration`, `PrepareRetentionIndex`, `RetentionPrepared`, `ListRetention`, `RetentionList` | Video-window age-outs, commit-clock cutoffs, and retention indexing |
| Recording applicability | `ListRecordingInputs`, `RecordingInputs` | Coherent pages of recording inputs for audio retention |
| Stats | `GetCatalogStats`, `CatalogStats` | Per-table row and byte counts in the catalog |

Rules a client relies on:

- **Paging.** A query result holds at most `max_items` (1024 by default) and 8 MiB of inline bytes, inside one 16 MiB
  socket frame (`MAX_FRAME`). Check `truncated`, not `complete`: a sparse window has gaps and is still fully listed. For
  a sequence query, keep following `next_after` until it is 0.
- **Positions.** A subscription position is a commit sequence, the catalog's order of everything committed. A
  subscription over several streams sends their payloads merged in that order.
- **The live edge.** A query on a ring stream that reaches the end of the stream also returns the unsealed tail as a
  ring position range.
- **Durable subscribers** resume from their last acknowledged position.
- **Declared, not built.** Nine messages the schema declares have no handler: `CoverageGapQuery`, `AgeOutFrames`,
  `FramesAgedOut`, `GetStreamGeneration`, `StreamGeneration`, `PrepareRetentionIndex`, `RetentionPrepared`,
  `ListRetention`, and `RetentionList`. A client that sends one gets an `Error` of code `Unavailable`, naming the
  message ("<name> is not implemented yet"). `CoverageGapResponse` is their other half and is never sent.

## Scheduler

| Band | What runs there |
|---|---|
| P0 Critical | Capture, the live voice turn |
| P1 Realtime | Live VAD and ASR, live overlays, the Tier 0 to Tier 1 flush |
| P2 Nearline | Identification, pose, tracking |
| P3 Background | Journal rollups, backfill |

- Every job runs under a lease that holds memory and a share of a compute domain. Budgets come from the machine
  profile (`mac_mini_64gb` or `linux_dual_3090`), so running the Mac profile on Linux enforces the Mac's limits.
- When the voice turn leaves `Idle`, P2 and P3 leases on the GPU and Neural Engine are preempted at once. The broker
  controls the lease, not a running kernel: a worker stops at its next batch boundary.
- Python background work that calls a model server outside a lease waits for an idle voice turn through
  `liberational.voice_turn.wait_quiet()`, for at most `voice.background_max_wait_s`. It does not cancel a request
  already submitted.
- The broker learns each capability's cost per item from committed jobs and uses it to quote latency.

## Catalog

`continuum-catalog` is redb, embedded in the broker and opened by it alone. Keys are fixed-width and big-endian, so
byte order is key order and range scans need no parsing. An offline tool opens it read-only while the broker is
stopped. Never copy a live catalog: the copy is torn.

| Tables | Answer |
|---|---|
| `META` | Catalog format, next stream id and commit sequence, keepers, and recording era |
| `STREAMS`, `STREAM_NAMES` | Registered streams and their revisions |
| `TIME_INDEX`, `STREAM_LONGEST`, `STREAM_SEQ`, `PAYLOAD_SEQ` | What a stream holds by time and by commit order |
| `PAYLOADS`, `LOCATIONS` | A payload's header and where its bytes are |
| `CHUNKS`, `CHUNK_FRAMES`, `FRAMES`, `CHUNK_PROVENANCE` | Each chunk's frame index, absorbed frames, and capture provenance |
| `PROV_PARENTS`, `PROV_CHILDREN`, `SUBJECTS` | Provenance and everything about an entity |
| `DEPS`, `DEPS_OF`, `DEPS_LONGEST` | Range dependencies on stretches of streams |
| `WORKER_STATE`, `WORKER_STATE_INPUTS` | Worker checkpoints and the input streams each names |
| `COVERAGE` | Which ranges a stage has processed |
| `CURSORS` | Durable subscription positions |
| `PURGES`, `PURGE_ORDER`, `PURGE_JOBS`, `DOOMED`, `PURGE_WORK`, `PURGE_FRONTIER`, `PURGE_CUTS`, `PURGE_CUT_ORDER`, `PURGE_LATE`, `PURGE_LATE_CUTS`, `PURGE_IDS`, `PURGE_COUNTS`, `PURGE_REMOVED`, `PURGE_REMOVED_FRAMES` | Purge records and the step-by-step state of running purges |
| `FORGOTTEN`, `AGED_OUT` | Keyed hashes of forgotten names, and aged-out spans |
| `RECORDING_META`, `RECORDING_ROWS`, `RECORDING_IDS`, `RECORDING_SUBJECTS`, `RECORDING_PURGE` | Recording applicability ledger for audio retention |
| `CLOCK_MAP`, `COMMIT_CLOCK` | Monotonic-to-UTC samples per boot, and commit sequence by UTC time |
| `HOT_IDS`, `RESTART_PROJECTION` | Availability index saved on a clean stop |

- **Frames are absorbed into chunks.** A raw frame's header exists while the frame is in Tier 0. When the tier worker
  commits the chunk, the chunk becomes the payload the catalog keeps. A frame keeps a small `FRAMES` row for an hour
  after its chunk ends, or longer while something names it as a parent or subject.
- **Dependencies.** A derived payload can depend on a stretch of a stream instead of naming each frame. A purge that
  removes anything in that stretch takes the dependent too.
- **Revisions.** Each stream has an invalidation revision that advances when anything on it is hidden or deleted, and
  a forget revision that ignores age-outs. Cached readers compare the first; worker state compares the second.
- **Worker state.** A save names each input stream's revision as the worker read it and is refused if something was
  forgotten since. A purge deletes the rows that name its streams.
- **Format.** The catalog file is format 8. The broker refuses any other format and says to run
  `pack-reset-catalog`. There are no upgrades.
- **Scrubbing.** redb never returns freed pages to the disk. A scrubber thread copies live rows into a fresh file in
  small steps and swaps it in, which shrinks the file and leaves no purged bytes behind. It runs when reclaimable
  space is large, or a day after a purge. `--no-catalog-scrub` turns it off.
- **Stats.** `pack-catalog-stats` reports where the file's bytes go.

## Forgetting

A `Purge` names a time range, an entity, or a list of payload ids, with a reason. It covers the broker's data; the
segment store forgets its own rows (`continuum_py`'s `forget_segments`).

1. Find the roots, then walk provenance forward to everything derived from them.
2. Hide each payload from every read as it is found. A payload committed during the purge under a hidden parent is
   hidden as it lands.
3. Remove the rows, remove the purged spans from coverage, mark cut frames in their chunks, and write the report.
4. Zero and free the Tier 0 slots.
5. Send `PurgeNotice` to every worker that keeps copies (the tier worker and the vector worker). The tier worker
   rewrites or deletes chunk files. The vector worker deletes rows, compacts, and prunes old table versions. Each
   answers `PurgeAck`.
6. The purge is complete when every keeper known to the catalog has acknowledged and the keeper grace period has
   passed. A keeper that is offline is told when it registers; `ForgetKeeper` retires a keeper that will never return.

Rules:

- A purge runs as short steps on the writer, between ingest commits, and resumes after a restart.
- A purge record never names the person: an entity purge keeps no entity id, and the caller's detail must not
  either.
- `FORGOTTEN` holds keyed hashes of forgotten names, outside the provenance graph, so no purge removes the record
  that someone was forgotten.
- Files registered in place (imported `.flac` and `.mp4`) are cut from the catalog but not rewritten or deleted.
- **An age-out is not a forget.** It removes only the payloads it names; everything derived from them stays, and the
  removed spans are recorded in `AGED_OUT` so readers say "footage expired", not "never recorded".
- Any client on the socket may purge; the socket's 0600 mode is the gate.
