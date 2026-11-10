//! The tier worker's loop against the in-memory catalog, as the broker would drive it: a producer writes into a
//! small Tier 0 ring that may reuse a slot only once the loop has marked it flushed, the loop ticks every frame,
//! and every closed chunk is committed. Short and bounded (#552's acceptance): 16 slots, 4 seconds at 30 fps, so
//! the ring wraps seven times; no hour-long run.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use continuum_catalog::{Catalog, MemoryCatalog, NewPayload, StreamInfo, StreamKind};
use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StorageLocator, StreamId, Uuid};
use continuum_tier::{
    Appended, ChunkFormat, Closed, CommitFailure, FlushSettings, Outputs, Payload, StreamShape, WorkerLoop, hydrate,
};

const MS: u64 = 1_000_000;
const S: u64 = 1_000_000_000;
const T0: u64 = 1_700_000_000 * S;
const FRAME_NS: u64 = 33_333_333;
const SLOTS: usize = 16;

fn uuid(n: u32) -> Uuid {
    let mut b = [0u8; 16];
    b[0] = 0x01;
    b[6] = 0x70;
    b[8] = 0x80;
    b[12..16].copy_from_slice(&n.to_be_bytes());
    Uuid(b)
}

fn frame_bytes(i: u32) -> Vec<u8> {
    let mut v = vec![0xffu8, 0xd8];
    v.extend((0..600).map(|k| u8::try_from((k + i as usize) % 251).expect("small")));
    v
}

fn shape(format: ChunkFormat) -> StreamShape {
    StreamShape {
        chunk_format: format,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::EncodedVideo,
        shape_name: "MJPEG".to_owned(),
        shape_version: 1,
        pcm: None,
        raw: None,
    }
}

fn register_cam(catalog: &MemoryCatalog) -> StreamId {
    catalog
        .register_stream(StreamInfo {
            name: "cam0".to_owned(),
            kind: StreamKind::Sensor,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::EncodedVideo,
            shape_name: "MJPEG".to_owned(),
            shape_version: 1,
            rate_hz: Some(30.0),
            anchor: None,
        })
        .expect("register")
}

/// The broker's ingest of a Tier 0 frame: its header, indexed before the worker sees it.
fn ingest(catalog: &MemoryCatalog, id: Uuid, stream: StreamId, start_ns: u64, seq: u32) {
    let mut spec = HeaderSpec::raw(id, stream, start_ns, start_ns);
    spec.seq = seq;
    spec.format = PayloadFormat::EncodedVideo;
    spec.shape_name = "MJPEG".to_owned();
    spec.shape_version = 1;
    let header = HeaderBuf::build(&spec).expect("header");
    catalog.insert(vec![NewPayload { header, location: None, coverage: None }]).expect("ingest");
}

/// What the broker would do with the loop's results: mark slots flushed, commit chunks (failing on demand).
#[derive(Clone, Default)]
struct Broker {
    catalog: Rc<MemoryCatalog>,
    flushed: Rc<RefCell<BTreeSet<Uuid>>>,
    fail_commits: Rc<RefCell<u32>>,
    committed: Rc<RefCell<Vec<std::path::PathBuf>>>,
    rejected: Rc<RefCell<Vec<std::path::PathBuf>>>,
}

impl Outputs for Broker {
    fn flushed(&mut self, ids: &[Uuid]) -> Result<(), String> {
        self.flushed.borrow_mut().extend(ids.iter().copied());
        Ok(())
    }

    fn commit(&mut self, closed: &Closed) -> Result<(), CommitFailure> {
        {
            let mut fail = self.fail_commits.borrow_mut();
            if *fail > 0 {
                *fail -= 1;
                return Err(CommitFailure::Retry("the broker is busy".into()));
            }
        }
        // The catalog refusing the chunk itself is for good, as the broker reports it.
        self.catalog
            .commit_chunk_with_provenance(closed.chunk.clone(), &closed.absorbed, closed.provenance.as_ref())
            .map_err(|e| CommitFailure::Reject(e.to_string()))?;
        self.committed.borrow_mut().push(closed.path.clone());
        Ok(())
    }

    fn rejected(&mut self, _closed: &Closed, aside: &std::path::Path) {
        self.rejected.borrow_mut().push(aside.to_path_buf());
    }
}

#[test]
fn nothing_is_lost_when_a_ring_that_reuses_only_flushed_slots_wraps_many_times() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = Rc::new(MemoryCatalog::new());
    let cam = register_cam(&catalog);
    let broker = Broker { catalog: Rc::clone(&catalog), ..Broker::default() };
    let settings = FlushSettings { chunk_ns: 700 * MS, audio_chunk_ns: 700 * MS, sync_ns: 200 * MS };
    let mut w = WorkerLoop::new(dir.path(), settings, broker.clone()).expect("loop");
    w.declare(cam, shape(ChunkFormat::Passthrough));

    // Each slot holds the id written into it last.
    let mut ring: Vec<Option<Uuid>> = vec![None; SLOTS];
    let mut truth: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    for i in 0..120u32 {
        let t = T0 + u64::from(i) * FRAME_NS;
        let slot = i as usize % SLOTS;
        // The producer may overwrite a slot only once the worker flushed what was in it: with a 200 ms sync and
        // a 533 ms ring, it never has to wait.
        if let Some(old) = ring[slot] {
            assert!(broker.flushed.borrow().contains(&old), "frame {i} would overwrite an unflushed slot");
        }
        let id = uuid(i);
        ring[slot] = Some(id);
        ingest(&catalog, id, cam, t, i);
        let bytes = frame_bytes(i);
        truth.insert(t, bytes.clone());
        w.payload(
            &Payload { pcm_rate_hz: 0, id, stream: cam, start_ns: t, end_ns: t, seq: i, bytes, provenance: None },
            t,
        )
        .expect("payload");
        w.tick(t).expect("tick");
    }
    w.shutdown().expect("shutdown");
    assert_eq!(w.uncommitted(), 0);
    let counts = w.counts();
    assert_eq!((counts.appended, counts.flushed, counts.refused), (120, 120, 0));
    assert!(counts.committed >= 5, "chunks closed on time: {counts:?}");

    // Any 50 ms slice, found through the catalog alone, hydrates to what the producer wrote.
    for start in [T0 + 100 * MS, T0 + 1_500 * MS, T0 + 3_700 * MS] {
        let end = start + 50 * MS;
        let expected: Vec<&Vec<u8>> = truth.range(start..end).map(|(_, b)| b).collect();
        let page = catalog.range(cam, Interval::new(start, end).expect("window"), None, 10).expect("range");
        let mut got = Vec::new();
        for item in &page.items {
            assert!(item.frames.is_some(), "only chunks remain: the frames were compacted into them");
            let Some(StorageLocator::Chunk { uri, .. }) = item.locations.first() else { panic!("a Tier 1 chunk") };
            got.extend(hydrate(std::path::Path::new(uri), start, end).expect("hydrate").into_iter().map(|h| h.bytes));
        }
        assert_eq!(got.iter().collect::<Vec<_>>(), expected, "slice at {start}");
    }
}

#[test]
fn a_failed_commit_is_tried_again_in_order_and_nothing_is_committed_twice() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = Rc::new(MemoryCatalog::new());
    let cam = register_cam(&catalog);
    let broker = Broker { catalog: Rc::clone(&catalog), ..Broker::default() };
    *broker.fail_commits.borrow_mut() = 3;
    let mut w = WorkerLoop::new(
        dir.path(),
        FlushSettings { chunk_ns: 300 * MS, audio_chunk_ns: 300 * MS, sync_ns: 100 * MS },
        broker.clone(),
    )
    .expect("loop");
    w.declare(cam, shape(ChunkFormat::Passthrough));
    for i in 0..60u32 {
        let t = T0 + u64::from(i) * FRAME_NS;
        ingest(&catalog, uuid(i), cam, t, i);
        w.payload(
            &Payload {
                pcm_rate_hz: 0,
                id: uuid(i),
                stream: cam,
                start_ns: t,
                end_ns: t,
                seq: i,
                bytes: frame_bytes(i),
                provenance: None,
            },
            t,
        )
        .expect("payload");
        w.tick(t).expect("tick");
    }
    w.shutdown().expect("shutdown");
    assert_eq!(w.uncommitted(), 0, "every chunk committed once the broker came back");
    assert_eq!(w.counts().commit_retries, 3);
    let committed = broker.committed.borrow();
    let mut sorted = committed.clone();
    sorted.sort();
    assert_eq!(*committed, sorted, "in order");
    sorted.dedup();
    assert_eq!(sorted.len(), committed.len(), "none twice");
    // Every frame was durable and reported, even while its chunk waited to be committed.
    assert_eq!(broker.flushed.borrow().len(), 60);
}

#[test]
fn raw_frames_are_lz4_in_the_chunk_and_an_undeclared_stream_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = Rc::new(MemoryCatalog::new());
    let cam = register_cam(&catalog);
    let broker = Broker { catalog: Rc::clone(&catalog), ..Broker::default() };
    let mut w = WorkerLoop::new(dir.path(), FlushSettings::default(), broker.clone()).expect("loop");
    w.declare(cam, shape(ChunkFormat::Lz4));
    let raw = vec![7u8; 4_096];
    ingest(&catalog, uuid(1), cam, T0, 1);
    w.payload(
        &Payload {
            pcm_rate_hz: 0,
            id: uuid(1),
            stream: cam,
            start_ns: T0,
            end_ns: T0,
            seq: 1,
            bytes: raw.clone(),
            provenance: None,
        },
        T0,
    )
    .expect("payload");
    let stray = Payload {
        pcm_rate_hz: 0,
        id: uuid(2),
        stream: StreamId(99),
        start_ns: T0,
        end_ns: T0,
        seq: 2,
        bytes: vec![1],
        provenance: None,
    };
    assert!(w.payload(&stray, T0).is_err());
    assert_eq!(w.counts().refused, 1);
    w.shutdown().expect("shutdown");
    let path = broker.committed.borrow()[0].clone();
    // The file holds it compressed; hydrating gives back the producer's bytes.
    let records = continuum_tier::ChunkReader::open(&path).expect("open").records().to_vec();
    assert!((records[0].len as usize) < raw.len(), "compressed in the chunk: {} bytes", records[0].len);
    assert_eq!(hydrate(&path, T0, T0 + 1).expect("hydrate")[0].bytes, raw);
}

#[test]
fn a_chunk_the_broker_refuses_for_good_is_set_aside_and_the_rest_go_on() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = Rc::new(MemoryCatalog::new());
    let cam = register_cam(&catalog);
    let broker = Broker { catalog: Rc::clone(&catalog), ..Broker::default() };
    let mut w = WorkerLoop::new(
        dir.path(),
        FlushSettings { chunk_ns: 300 * MS, audio_chunk_ns: 300 * MS, sync_ns: 100 * MS },
        broker.clone(),
    )
    .expect("loop");
    w.declare(cam, shape(ChunkFormat::Passthrough));
    for i in 0..30u32 {
        let t = T0 + u64::from(i) * FRAME_NS;
        // Frames 0 to 4 were never ingested: the catalog refuses the chunk that absorbs them.
        if i >= 5 {
            ingest(&catalog, uuid(i), cam, t, i);
        }
        w.payload(
            &Payload {
                pcm_rate_hz: 0,
                id: uuid(i),
                stream: cam,
                start_ns: t,
                end_ns: t,
                seq: i,
                bytes: frame_bytes(i),
                provenance: None,
            },
            t,
        )
        .expect("payload");
        w.tick(t).expect("tick");
    }
    w.shutdown().expect("shutdown");
    assert_eq!(w.uncommitted(), 0, "nothing wedged behind the refused chunk");
    assert_eq!(w.counts().rejected, 1);
    let aside = broker.rejected.borrow()[0].clone();
    assert!(aside.exists() && aside.extension().is_some_and(|e| e == "rejected"), "{}", aside.display());
    assert!(w.counts().committed >= 2, "{:?}", w.counts());
}

#[test]
fn a_chunk_a_crash_left_open_commits_with_the_provenance_capture_attested() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = Rc::new(MemoryCatalog::new());
    let cam = register_cam(&catalog);
    let attested = |i: u32| continuum_catalog::FrameProvenance {
        device_seq: u64::from(i) + 900,
        epoch: 41,
        stamped_at_arrival: false,
    };
    {
        let broker = Broker { catalog: Rc::clone(&catalog), ..Broker::default() };
        let mut w = WorkerLoop::new(
            dir.path(),
            FlushSettings { chunk_ns: 10 * S, audio_chunk_ns: 10 * S, sync_ns: 100 * MS },
            broker,
        )
        .expect("loop");
        w.declare(cam, shape(ChunkFormat::Passthrough));
        for i in 0..3u32 {
            let t = T0 + u64::from(i) * FRAME_NS;
            ingest(&catalog, uuid(i), cam, t, i);
            let payload = Payload {
                pcm_rate_hz: 0,
                id: uuid(i),
                stream: cam,
                start_ns: t,
                end_ns: t,
                seq: i,
                bytes: frame_bytes(i),
                // Frame 1's producer said nothing about it.
                provenance: (i != 1).then(|| attested(i)),
            };
            w.payload(&payload, t).expect("payload");
        }
        // Killed before closing the chunk.
        std::mem::forget(w);
    }
    let broker = Broker { catalog: Rc::clone(&catalog), ..Broker::default() };
    let mut w = WorkerLoop::new(dir.path(), FlushSettings::default(), broker.clone()).expect("loop");
    w.recovered(continuum_tier::recover(dir.path()).expect("recover"));
    w.declare(cam, shape(ChunkFormat::Passthrough));
    assert_eq!(broker.committed.borrow().len(), 1);
    let Some(continuum_catalog::Resolved::AbsorbedFrame { chunk, .. }) = catalog.get(uuid(0)).expect("get") else {
        panic!("absorbed into the recovered chunk")
    };
    let provenance = catalog.chunk_provenance(chunk.header.id()).expect("read").expect("kept through the crash");
    assert_eq!(provenance.frames, [attested(0), continuum_catalog::FrameProvenance::default(), attested(2)]);
}

#[test]
fn a_chunk_a_crash_left_open_is_committed_when_its_stream_returns_and_not_appended_twice() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = Rc::new(MemoryCatalog::new());
    let cam = register_cam(&catalog);
    // The first run appends 20 frames and dies before closing its chunk.
    {
        let broker = Broker { catalog: Rc::clone(&catalog), ..Broker::default() };
        let mut w = WorkerLoop::new(
            dir.path(),
            FlushSettings { chunk_ns: 10 * S, audio_chunk_ns: 10 * S, sync_ns: 100 * MS },
            broker,
        )
        .expect("loop");
        w.declare(cam, shape(ChunkFormat::Passthrough));
        for i in 0..20u32 {
            let t = T0 + u64::from(i) * FRAME_NS;
            ingest(&catalog, uuid(i), cam, t, i);
            w.payload(
                &Payload {
                    pcm_rate_hz: 0,
                    id: uuid(i),
                    stream: cam,
                    start_ns: t,
                    end_ns: t,
                    seq: i,
                    bytes: frame_bytes(i),
                    provenance: None,
                },
                t,
            )
            .expect("payload");
        }
        std::mem::forget(w);
    }
    // The next run recovers it; the broker sends the 20 frames again (they weren't acked), then 10 new ones.
    let broker = Broker { catalog: Rc::clone(&catalog), ..Broker::default() };
    let mut w = WorkerLoop::new(dir.path(), FlushSettings::default(), broker.clone()).expect("loop");
    w.recovered(continuum_tier::recover(dir.path()).expect("recover"));
    w.declare(cam, shape(ChunkFormat::Passthrough));
    assert_eq!(w.counts().recovered, 1);
    assert_eq!(broker.committed.borrow().len(), 1, "committed as soon as its stream was declared");
    for i in 0..30u32 {
        let t = T0 + u64::from(i) * FRAME_NS;
        if i >= 20 {
            ingest(&catalog, uuid(i), cam, t, i);
        }
        let got = w
            .payload(
                &Payload {
                    pcm_rate_hz: 0,
                    id: uuid(i),
                    stream: cam,
                    start_ns: t,
                    end_ns: t,
                    seq: i,
                    bytes: frame_bytes(i),
                    provenance: None,
                },
                t,
            )
            .expect("payload");
        assert_eq!(got, if i < 20 { Appended::Recovered } else { Appended::Appended }, "frame {i}");
    }
    w.shutdown().expect("shutdown");
    assert_eq!(w.uncommitted(), 0);
    // Every frame is in exactly one committed chunk.
    let page = catalog.range(cam, Interval::new(T0, T0 + S).expect("window"), None, 10).expect("range");
    let frames: usize = page.items.iter().map(|i| i.frames.as_ref().map_or(0, Vec::len)).sum();
    assert_eq!(frames, 30);
}

/// A microphone batch as capture writes it: `records` ring records, each a 16-byte stamp then 480 mono f32
/// samples, every sample `sample(index)`.
fn mic_batch(records: u64, sample: impl Fn(u64) -> f32) -> Vec<u8> {
    let mut out = Vec::new();
    for r in 0..records {
        out.extend_from_slice(&(r * 10 * MS).to_le_bytes());
        out.extend_from_slice(&(r * 480).to_le_bytes());
        for i in 0..480 {
            out.extend_from_slice(&sample(r * 480 + i).to_le_bytes());
        }
    }
    out
}

/// A worker following one microphone (mono f32, 480 samples a record), and the broker it commits to.
fn mic_worker(dir: &std::path::Path, settings: FlushSettings) -> (WorkerLoop<Broker>, Broker, StreamId) {
    let catalog = Rc::new(MemoryCatalog::new());
    let mic = register_cam(&catalog);
    let broker = Broker { catalog: Rc::clone(&catalog), ..Broker::default() };
    let mut w = WorkerLoop::new(dir, settings, broker.clone()).expect("loop");
    let layout = continuum_tier::pcm::PcmLayout::of("PCM_F32LEx1", 16 + 480 * 4).expect("layout");
    w.declare(mic, StreamShape { pcm: Some(layout), ..shape(ChunkFormat::PcmBatch) });
    (w, broker, mic)
}

/// Payload `n` of the microphone: `bytes` at `T0 + n` seconds, at `rate`.
fn mic_payload(broker: &Broker, mic: StreamId, n: u32, rate: u32, bytes: &[u8]) -> Payload {
    let start = T0 + u64::from(n) * S;
    ingest(&broker.catalog, uuid(n), mic, start, n);
    Payload {
        pcm_rate_hz: rate,
        id: uuid(n),
        stream: mic,
        start_ns: start,
        end_ns: start + 500 * MS,
        seq: n,
        bytes: bytes.to_vec(),
        provenance: None,
    }
}

/// Every file in the chunk directory, by name.
fn files(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> =
        std::fs::read_dir(dir).expect("dir").map(|e| e.expect("entry").file_name().to_string_lossy().into()).collect();
    names.sort();
    names
}

/// #1942: a microphone's chunk becomes Opus as it closes, on time (`tick`) or at shutdown, before it's committed.
#[test]
fn a_microphone_chunk_is_committed_as_opus_with_its_payloads_ids_and_times() {
    #[expect(clippy::cast_precision_loss, reason = "test signal")]
    let tone = |i: u64| (i as f32 * 0.05).sin() * 0.5;
    let batch = mic_batch(50, tone);
    for on_time in [true, false] {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = FlushSettings { chunk_ns: 2 * S, audio_chunk_ns: 2 * S, sync_ns: 200 * MS };
        let (mut w, broker, mic) = mic_worker(dir.path(), settings);
        let payloads: Vec<Payload> = (1..=2).map(|n| mic_payload(&broker, mic, n, 48_000, &batch)).collect();
        for p in &payloads {
            assert_eq!(w.payload(p, p.start_ns).expect("payload"), Appended::Appended);
        }
        let pcm_file = files(dir.path());
        assert_eq!(pcm_file.len(), 1, "one open PCM chunk");
        if on_time {
            w.tick(T0 + 4 * S).expect("tick");
        } else {
            w.shutdown().expect("shutdown");
        }
        assert_eq!(w.counts().committed, 1);
        assert_eq!(w.counts().opus_fallbacks, 0);
        assert_eq!(files(dir.path()), pcm_file, "the Opus chunk took the PCM file's place; nothing else is left");
        let path = broker.committed.borrow()[0].clone();
        let reader = continuum_tier::ChunkReader::open(&path).expect("open");
        assert_eq!(reader.format(), ChunkFormat::Opus);
        let records: Vec<_> = reader.records().iter().map(|r| (r.id, r.start_ns, r.end_ns, r.seq)).collect();
        let sent: Vec<_> = payloads.iter().map(|p| (p.id, p.start_ns, p.end_ns, p.seq)).collect();
        assert_eq!(records, sent);
        // The catalog's frames point at the Opus records, and its locator has the Opus file's length.
        let page = broker.catalog.range(mic, Interval::new(T0, T0 + 4 * S).expect("window"), None, 10).expect("range");
        let item = &page.items[0];
        let frames = item.frames.as_ref().expect("a chunk");
        assert_eq!(
            frames.iter().map(|f| (f.start_ns, f.offset, f.length)).collect::<Vec<_>>(),
            reader.records().iter().map(|r| (r.start_ns, r.offset, u64::from(r.len))).collect::<Vec<_>>()
        );
        let Some(StorageLocator::Chunk { length, .. }) = item.locations.first() else { panic!("a chunk locator") };
        assert_eq!(*length, std::fs::metadata(&path).expect("metadata").len());
        let hydrated = hydrate(&path, T0 + 2 * S, T0 + 2 * S + 1).expect("hydrate");
        assert_eq!(hydrated.len(), 1);
        assert_eq!(hydrated[0].record.id, uuid(2));
        assert_eq!(hydrated[0].bytes.len(), batch.len(), "the same ring records, decoded");
        assert_eq!(&hydrated[0].bytes[..16], &batch[..16], "stamps are exact");
    }
}

/// A batch without a known rate (capture didn't say one) can't become Opus: its chunk is committed as PCM and
/// counted, the first logged.
#[test]
fn a_microphone_chunk_without_a_known_rate_stays_pcm_and_is_counted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (mut w, broker, mic) =
        mic_worker(dir.path(), FlushSettings { audio_chunk_ns: 2 * S, ..FlushSettings::default() });
    let batch = mic_batch(50, |_| 0.25);
    for n in 1..=2 {
        let p = mic_payload(&broker, mic, n, 0, &batch);
        w.payload(&p, p.start_ns).expect("payload");
        // Each payload in its own chunk.
        w.tick(p.start_ns + 3 * S).expect("tick");
    }
    assert_eq!(w.counts().committed, 2);
    assert_eq!(w.counts().opus_fallbacks, 2);
    for path in broker.committed.borrow().iter() {
        assert_eq!(continuum_tier::ChunkReader::open(path).expect("open").format(), ChunkFormat::PcmBatch);
        let hydrated = hydrate(path, T0, T0 + 4 * S).expect("hydrate");
        assert_eq!(hydrated[0].bytes, batch, "lossless, as before");
    }
}

/// A rate Opus can't keep (44.1 kHz) leaves the chunk PCM, whose batches are FLAC when exact and LZ4 otherwise.
#[test]
fn microphone_batches_are_flac_when_exact_and_lz4_otherwise_and_both_hydrate_byte_for_byte() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = Rc::new(MemoryCatalog::new());
    let mic = register_cam(&catalog);
    let broker = Broker { catalog: Rc::clone(&catalog), ..Broker::default() };
    let mut w = WorkerLoop::new(dir.path(), FlushSettings::default(), broker.clone()).expect("loop");
    let layout = continuum_tier::pcm::PcmLayout::of("PCM_F32LEx1", 16 + 480 * 4).expect("layout");
    w.declare(mic, StreamShape { pcm: Some(layout), ..shape(ChunkFormat::PcmBatch) });
    // A 24 bit tone, as PipeWire converts S24 to f32 (exact), then the same tone after a software gain of 0.7.
    #[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation, reason = "test signal")]
    let tone = |i: u64| ((i as f32 * 0.01).sin() * 4_000_000.0) as i32 as f32 / 8_388_608.0;
    let exact = mic_batch(100, tone);
    let gained = mic_batch(100, |i| tone(i) * 0.7);
    for (n, bytes) in [(1, &exact), (2, &gained)] {
        let start = T0 + u64::from(n) * S;
        ingest(&catalog, uuid(n), mic, start, n);
        let p = Payload {
            pcm_rate_hz: 44_100,
            id: uuid(n),
            stream: mic,
            start_ns: start,
            end_ns: start,
            seq: n,
            bytes: bytes.clone(),
            provenance: None,
        };
        assert_eq!(w.payload(&p, start).expect("payload"), Appended::Appended);
    }
    assert_eq!(w.counts().pcm_fallbacks, 1, "only the gained batch falls back");
    w.shutdown().expect("shutdown");
    assert_eq!(w.counts().opus_fallbacks, 1, "44.1 kHz isn't an Opus rate");
    let path = broker.committed.borrow()[0].clone();
    let mut reader = continuum_tier::ChunkReader::open(&path).expect("open");
    assert_eq!(reader.format(), ChunkFormat::PcmBatch);
    let codecs: Vec<_> =
        (0..2).map(|i| continuum_tier::pcm::codec_of(&reader.read(i).expect("read")).expect("codec")).collect();
    assert_eq!(codecs, [continuum_tier::pcm::PcmCodec::Flac24, continuum_tier::pcm::PcmCodec::Lz4]);
    let hydrated = hydrate(&path, T0, T0 + 3 * S).expect("hydrate");
    assert!(hydrated.iter().all(|h| h.pcm == Some(continuum_tier::pcm::PcmFormat { sample_rate_hz: 44_100, layout })));
    let back: Vec<Vec<u8>> = hydrated.into_iter().map(|h| h.bytes).collect();
    assert_eq!(back, [exact, gained]);
}

#[test]
fn a_camera_resize_closes_its_jpeg_chunk_before_the_next_frame() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Rc::new(MemoryCatalog::new());
    let cam = register_cam(&catalog);
    let broker = Broker { catalog: Rc::clone(&catalog), ..Broker::default() };
    let mut worker = WorkerLoop::new(dir.path(), FlushSettings::default(), broker.clone()).unwrap();
    worker.declare(cam, shape(ChunkFormat::Passthrough));
    let sizes = [(64u16, 48u16), (32, 24)];
    for (i, (width, height)) in sizes.into_iter().enumerate() {
        let seq = u32::try_from(i).unwrap();
        let t = T0 + u64::from(seq) * FRAME_NS;
        ingest(&catalog, uuid(seq), cam, t, seq);
        let mut bytes = Vec::new();
        jpeg_encoder::Encoder::new(&mut bytes, 90)
            .encode(
                &vec![96; usize::from(width) * usize::from(height) * 3],
                width,
                height,
                jpeg_encoder::ColorType::Rgb,
            )
            .unwrap();
        worker
            .payload(
                &Payload {
                    pcm_rate_hz: 0,
                    id: uuid(seq),
                    stream: cam,
                    start_ns: t,
                    end_ns: t,
                    seq,
                    bytes,
                    provenance: None,
                },
                t,
            )
            .unwrap();
    }
    worker.shutdown().unwrap();
    let chunks = broker.committed.borrow();
    assert_eq!(chunks.len(), 2, "a resolution change is a boundary even under two seconds");
    for (i, path) in chunks.iter().enumerate() {
        let mut reader = continuum_tier::ChunkReader::open(path).unwrap();
        assert_eq!(reader.records().len(), 1);
        assert_eq!(reader.records()[0].start_ns, T0 + u64::try_from(i).unwrap() * FRAME_NS);
        let picture = continuum_tier::hydrate::decode_jpeg(&reader.read(0).unwrap()).unwrap();
        assert_eq!((picture.width, picture.height), if i == 0 { (64, 48) } else { (32, 24) });
    }
}

fn rtsp_shape() -> StreamShape {
    StreamShape {
        chunk_format: ChunkFormat::H264Au,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::EncodedVideo,
        shape_name: "H264AnnexB".to_owned(),
        shape_version: 1,
        pcm: None,
        raw: None,
    }
}

fn rtsp_catalog(catalog: &MemoryCatalog) -> StreamId {
    catalog
        .register_stream(StreamInfo {
            name: "network-camera.video".to_owned(),
            kind: StreamKind::Sensor,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::EncodedVideo,
            shape_name: "H264AnnexB".to_owned(),
            shape_version: 1,
            rate_hz: Some(30.0),
            anchor: None,
        })
        .unwrap()
}

fn rtsp_payload(catalog: &MemoryCatalog, stream: StreamId, sequence: u32, presentation: u32) -> Payload {
    let start = T0 + u64::from(presentation) * FRAME_NS;
    let mut header = HeaderSpec::raw(uuid(sequence), stream, start, start);
    header.seq = sequence;
    header.format = PayloadFormat::EncodedVideo;
    header.shape_name = "H264AnnexB".to_owned();
    catalog
        .insert(vec![NewPayload { header: HeaderBuf::build(&header).unwrap(), location: None, coverage: None }])
        .unwrap();
    Payload {
        id: uuid(sequence),
        stream,
        start_ns: start,
        end_ns: start,
        seq: sequence,
        bytes: vec![0, 0, 0, 1, if sequence == 0 { 0x65 } else { 0x41 }, sequence as u8],
        pcm_rate_hz: 0,
        provenance: None,
    }
}

#[test]
fn rtsp_access_units_keep_original_bytes_and_decode_order_while_catalog_indexes_presentation_order() {
    let dir = tempfile::tempdir().unwrap();
    let broker = Broker::default();
    let stream = rtsp_catalog(&broker.catalog);
    let mut worker = WorkerLoop::new(
        dir.path(),
        FlushSettings { chunk_ns: 60 * S, audio_chunk_ns: 60 * S, sync_ns: S },
        broker.clone(),
    )
    .unwrap();
    worker.declare(stream, rtsp_shape());
    let mut expected = Vec::new();
    for (sequence, presentation) in [0, 3, 1, 2].into_iter().enumerate() {
        let payload = rtsp_payload(&broker.catalog, stream, sequence as u32, presentation);
        expected.push(payload.bytes.clone());
        worker.payload(&payload, T0 + sequence as u64 * FRAME_NS).unwrap();
    }
    worker.shutdown().unwrap();
    assert!(broker.rejected.borrow().is_empty());
    let [path] = broker.committed.borrow().clone().try_into().expect("one independently indexed chunk");
    let mut reader = continuum_tier::ChunkReader::open(&path).unwrap();
    let units: Vec<_> =
        (0..4).map(|index| continuum_tier::gop::decode(&reader.read(index).unwrap()).unwrap().remove(0)).collect();
    assert_eq!(units.iter().map(|u| &u.bytes).collect::<Vec<_>>(), expected.iter().collect::<Vec<_>>());
    assert_eq!(
        units.iter().map(|u| u.start_ns).collect::<Vec<_>>(),
        [T0, T0 + 3 * FRAME_NS, T0 + FRAME_NS, T0 + 2 * FRAME_NS]
    );
    assert_eq!(units.iter().map(|u| u.key).collect::<Vec<_>>(), [true, false, false, false]);
    for (sequence, index) in [0, 3, 1, 2].into_iter().enumerate() {
        let Some(continuum_catalog::Resolved::AbsorbedFrame { frame_index, .. }) =
            broker.catalog.get(uuid(sequence as u32)).unwrap()
        else {
            panic!("original payload is still addressable")
        };
        assert_eq!(frame_index, index);
    }
    for (logical, physical) in [0, 2, 3, 1].into_iter().enumerate() {
        let hydrated = continuum_tier::hydrate::hydrate_frame(&path, logical).unwrap();
        assert_eq!(hydrated.index, logical);
        assert_eq!(hydrated.record.seq, physical as u32);
        assert_eq!(continuum_tier::gop::decode(&hydrated.bytes).unwrap()[0].bytes, expected[physical]);
    }
    // A presentation-time purge cuts the B frame's physical record, not its time-sorted catalog position.
    continuum_tier::service::purge_file(
        &path,
        &[Interval::new(T0 + FRAME_NS, T0 + 2 * FRAME_NS).unwrap()],
        &BTreeSet::new(),
        false,
    )
    .unwrap();
    let reader = continuum_tier::ChunkReader::open(&path).unwrap();
    assert_eq!(reader.records().iter().map(|r| r.cut).collect::<Vec<_>>(), [false, false, true, false]);
}

#[test]
fn recovered_rtsp_chunks_commit_original_ids_with_sorted_presentation_indexes() {
    let dir = tempfile::tempdir().unwrap();
    let broker = Broker::default();
    let stream = rtsp_catalog(&broker.catalog);
    let hour = continuum_tier::layout::hour_dir(dir.path(), stream, T0);
    continuum_tier::layout::make_hour_dir(dir.path(), &hour).unwrap();
    let path = hour.join("unfinished.cnk");
    let mut writer = continuum_tier::ChunkWriter::create(&path, stream, ChunkFormat::H264Au, T0).unwrap();
    for (sequence, presentation) in [0, 3, 1, 2].into_iter().enumerate() {
        let payload = rtsp_payload(&broker.catalog, stream, sequence as u32, presentation);
        let encoded = continuum_tier::gop::encode(&[continuum_tier::gop::Unit {
            start_ns: payload.start_ns,
            end_ns: payload.end_ns,
            key: sequence == 0,
            bytes: payload.bytes,
        }])
        .unwrap();
        writer.append_payload(payload.id, payload.start_ns, payload.end_ns, payload.seq, &encoded).unwrap();
    }
    writer.sync().unwrap();
    drop(writer);
    let recovered = continuum_tier::recover(dir.path()).unwrap();
    let mut worker = WorkerLoop::new(dir.path(), FlushSettings::default(), broker.clone()).unwrap();
    worker.recovered(recovered);
    worker.declare(stream, rtsp_shape());
    assert!(broker.rejected.borrow().is_empty());
    assert_eq!(broker.committed.borrow().as_slice(), &[path]);
    for (sequence, index) in [0, 3, 1, 2].into_iter().enumerate() {
        let Some(continuum_catalog::Resolved::AbsorbedFrame { frame_index, .. }) =
            broker.catalog.get(uuid(sequence as u32)).unwrap()
        else {
            panic!("recovered original payload")
        };
        assert_eq!(frame_index, index);
    }
}

fn rtsp_frame(catalog: &MemoryCatalog, stream: StreamId, sequence: u32, key: bool) -> Payload {
    let mut payload = rtsp_payload(catalog, stream, sequence, sequence);
    payload.bytes = vec![0, 0, 0, 1, if key { 0x65 } else { 0x41 }, sequence as u8];
    payload
}

#[test]
fn after_a_close_away_from_a_keyframe_a_native_stream_drops_frames_until_its_next_idr() {
    let dir = tempfile::tempdir().unwrap();
    let broker = Broker::default();
    let stream = rtsp_catalog(&broker.catalog);
    let mut worker =
        WorkerLoop::new(dir.path(), FlushSettings { chunk_ns: S, audio_chunk_ns: S, sync_ns: S }, broker.clone())
            .unwrap();
    worker.declare(stream, rtsp_shape());
    for (sequence, key) in [(0, true), (1, false), (2, false)] {
        let frame = rtsp_frame(&broker.catalog, stream, sequence, key);
        assert_eq!(worker.payload(&frame, T0 + u64::from(sequence) * FRAME_NS).unwrap(), Appended::Appended);
    }
    // The camera stalls: its chunk is closed on time, between keyframes.
    worker.tick(T0 + 3 * S).unwrap();
    assert!(worker.open_streams().is_empty(), "a quiet stream's chunk closes");
    let mut got = Vec::new();
    for (sequence, key) in [(3, false), (4, false), (5, true), (6, false)] {
        let frame = rtsp_frame(&broker.catalog, stream, sequence, key);
        got.push(worker.payload(&frame, T0 + 3 * S + u64::from(sequence) * FRAME_NS).unwrap());
    }
    assert_eq!(got, [Appended::Undecodable, Appended::Undecodable, Appended::Appended, Appended::Appended]);
    assert_eq!(worker.counts().undecodable, 2);
    worker.shutdown().unwrap();
    let committed = broker.committed.borrow().clone();
    assert_eq!(committed.len(), 2);
    let mut reader = continuum_tier::ChunkReader::open(&committed[1]).unwrap();
    let first = continuum_tier::gop::decode(&reader.read(0).unwrap()).unwrap().remove(0);
    assert!(first.key, "the stream's next chunk starts on its keyframe");
    assert_eq!(reader.records().len(), 2);
}

#[test]
fn a_microphone_chunk_stays_open_for_fifteen_seconds_where_video_closes_after_two() {
    let batch = mic_batch(50, |_| 0.25);
    let dir = tempfile::tempdir().expect("tempdir");
    let (mut w, broker, mic) = mic_worker(dir.path(), FlushSettings::default());
    for n in 1..=4 {
        let p = mic_payload(&broker, mic, n, 48_000, &batch);
        w.payload(&p, p.start_ns).expect("payload");
    }
    w.tick(T0 + 5 * S).expect("tick");
    assert_eq!(w.counts().committed, 0, "past video's 2 s, an audio chunk is still open");
    w.tick(T0 + 17 * S).expect("tick");
    assert_eq!(w.counts().committed, 1, "it closes once 15 s old");
    let settings = FlushSettings::default();
    assert_eq!(settings.open_ns(ChunkFormat::H264Au), 2 * S);
    assert_eq!(settings.open_ns(ChunkFormat::PcmBatch), 15 * S);
}
