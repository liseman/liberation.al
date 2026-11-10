//! The tier library against the in-memory catalog: flushing into chunks, committing them, hydrating the same slice
//! hot and warm, surviving a Tier 0 wrap, recovering an open chunk, and decoding a JPEG on request.
//!
//! The wrap test is deliberately short and bounded: a synthetic producer, small frames, a 16-slot ring, and 4
//! seconds of frames at 30 fps, so the ring wraps seven times within one test. A long soak run is a separate thing
//! that is run on demand.

use std::collections::{BTreeMap, BTreeSet};

use continuum_catalog::{Catalog, FrameProvenance, MemoryCatalog, NewPayload, Resolved, StreamInfo, StreamKind};
use continuum_core::{
    DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StorageLocator, StorageTier, StreamId, Uuid,
};
use continuum_tier::{
    ChunkFormat, ChunkReader, FlushSettings, Flusher, Incoming, StreamShape, codecs, decode_jpeg, hydrate, recover,
};

const MS: u64 = 1_000_000;
const S: u64 = 1_000_000_000;
const T0: u64 = 1_700_000_000 * S;
const FRAME_NS: u64 = 33_333_333;

fn uuid(n: u32) -> Uuid {
    let mut b = [0u8; 16];
    b[0] = 0x01;
    b[6] = 0x70;
    b[8] = 0x80;
    b[12..16].copy_from_slice(&n.to_be_bytes());
    Uuid(b)
}

fn frame_bytes(i: u32) -> Vec<u8> {
    // A fake JPEG-sized payload with a recognizable pattern per frame.
    let mut v = vec![0xffu8, 0xd8];
    v.extend((0..600).map(|k| u8::try_from((k + i as usize) % 251).expect("small")));
    v
}

fn shape() -> StreamShape {
    StreamShape {
        chunk_format: ChunkFormat::Passthrough,
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

fn frame_payload(id: Uuid, stream: StreamId, start_ns: u64, seq: u32) -> NewPayload {
    let mut spec = HeaderSpec::raw(id, stream, start_ns, start_ns);
    spec.seq = seq;
    spec.format = PayloadFormat::EncodedVideo;
    spec.shape_name = "MJPEG".to_owned();
    spec.shape_version = 1;
    NewPayload { header: HeaderBuf::build(&spec).expect("header"), location: None, coverage: None }
}

#[test]
fn frames_flush_into_timed_chunks_that_the_catalog_compacts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = MemoryCatalog::new();
    let cam = register_cam(&catalog);
    let mut flusher = Flusher::new(
        dir.path(),
        FlushSettings { chunk_ns: S, part_ns: S, audio_chunk_ns: S, ..FlushSettings::default() },
    )
    .expect("flusher");
    flusher.declare(cam, shape());

    let mut closed = Vec::new();
    let mut hot: BTreeMap<u64, (Uuid, Vec<u8>)> = BTreeMap::new();
    for i in 0..90u32 {
        let t = T0 + u64::from(i) * FRAME_NS;
        let id = uuid(i);
        let bytes = frame_bytes(i);
        catalog.insert(vec![frame_payload(id, cam, t, i)]).expect("insert");
        let (record, done) = flusher
            .append(Incoming { id, stream: cam, start_ns: t, end_ns: t, seq: i, bytes: &bytes, key: true }, t)
            .expect("append");
        assert_eq!(record.start_ns, t);
        hot.insert(t, (id, bytes));
        closed.extend(done);
    }
    closed.extend(flusher.close_all().expect("close"));
    // Three seconds of frames at a one second chunk length.
    assert_eq!(closed.len(), 3, "{:?}", closed.iter().map(|c| c.records.len()).collect::<Vec<_>>());
    assert_eq!(closed.iter().map(|c| c.records.len()).sum::<usize>(), 90);

    for c in &closed {
        assert_eq!(c.absorbed.len(), c.chunk.frames.len());
        let span = c.chunk.payload.header.span().expect("span");
        assert_eq!(span.start_ns(), c.records[0].start_ns);
        assert!(matches!(c.chunk.payload.location, Some(StorageLocator::Chunk { tier: StorageTier::Tier1Warm, .. })));
        catalog.commit_chunk(c.chunk.clone(), &c.absorbed).expect("commit_chunk");
    }

    // A frame's own header is compacted away; its FRAMES row maps it to the chunk, which carries the frame index in
    // its place (every absorbed frame keeps the row, so a late child still finds its parent).
    let (first_id, _) = &hot[&T0];
    assert!(matches!(catalog.get(*first_id).expect("get"), Some(Resolved::AbsorbedFrame { frame_index: 0, .. })));
    let chunk_id = closed[0].chunk.payload.header.id();
    let Resolved::Payload(indexed) = catalog.get(chunk_id).expect("get").expect("known") else { panic!("chunk") };
    // 30 frames at 33.33 ms land inside the first second (frame 30 at 0.99999999 s), so the chunk holds 31.
    assert_eq!(indexed.frames.as_ref().map(Vec::len), Some(closed[0].records.len()));
    assert_eq!(closed[0].records.len(), 31);
    assert_eq!(indexed.locations.len(), 1);

    // The same 50 ms slice reads the same hot and hydrated: frame 31 (1.033 s) is the one inside [1.000, 1.050).
    let (lo, hi) = (T0 + S, T0 + S + 50 * MS);
    let expected: Vec<&Vec<u8>> = hot.range(lo..hi).map(|(_, (_, b))| b).collect();
    assert_eq!(expected.len(), 1);
    let chunk_with_slice =
        closed.iter().find(|c| c.records.iter().any(|r| r.start_ns >= lo && r.start_ns < hi)).expect("chunk");
    let got = hydrate(&chunk_with_slice.path, lo, hi).expect("hydrate");
    assert_eq!(got.iter().map(|h| &h.bytes).collect::<Vec<_>>(), expected);
    assert_eq!(got[0].index, 0, "the slice starts at the chunk's first frame");
}

#[test]
fn attested_provenance_lands_in_frame_order_with_unknown_frames_left_unknown() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = MemoryCatalog::new();
    let cam = register_cam(&catalog);
    let mut flusher = Flusher::new(
        dir.path(),
        FlushSettings { chunk_ns: S, part_ns: S, audio_chunk_ns: S, ..FlushSettings::default() },
    )
    .expect("flusher");
    flusher.declare(cam, shape());
    let attested = |i: u32| FrameProvenance { device_seq: u64::from(i) + 500, epoch: 77, stamped_at_arrival: i == 2 };
    let mut ids = Vec::new();
    for i in 0..3u32 {
        let t = T0 + u64::from(i) * FRAME_NS;
        let id = uuid(i);
        catalog.insert(vec![frame_payload(id, cam, t, i)]).expect("insert");
        // Frame 1's producer said nothing about it.
        if i != 1 {
            flusher.attest(cam, id, attested(i));
        }
        let bytes = frame_bytes(i);
        flusher
            .append(Incoming { id, stream: cam, start_ns: t, end_ns: t, seq: i, bytes: &bytes, key: true }, t)
            .expect("append");
        ids.push(id);
    }
    let closed = flusher.close_all().expect("close");
    assert_eq!(closed.len(), 1);
    let c = &closed[0];
    assert_eq!(c.absorbed, ids);
    let rows = c.provenance.as_ref().expect("two frames were attested").frames.clone();
    assert_eq!(rows, vec![attested(0), FrameProvenance::default(), attested(2)], "epoch 0 is unknown, not continuous");
    catalog.commit_chunk_with_provenance(c.chunk.clone(), &c.absorbed, c.provenance.as_ref()).expect("commit");
    assert_eq!(catalog.chunk_provenance(c.chunk.payload.header.id()).expect("read"), c.provenance);

    // The next chunk's frames were never attested: no provenance at all, rather than a row of unknowns.
    let t = T0 + 3 * FRAME_NS;
    let bytes = frame_bytes(3);
    flusher
        .append(Incoming { id: uuid(3), stream: cam, start_ns: t, end_ns: t, seq: 3, bytes: &bytes, key: true }, t)
        .expect("append");
    assert_eq!(flusher.close_all().expect("close")[0].provenance, None);
}

#[test]
fn nothing_is_lost_when_a_small_ring_wraps_many_times() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut flusher = Flusher::new(
        dir.path(),
        FlushSettings { chunk_ns: 700 * MS, part_ns: 700 * MS, audio_chunk_ns: 700 * MS, ..FlushSettings::default() },
    )
    .expect("flusher");
    let cam = StreamId(1);
    flusher.declare(cam, shape());

    // A 16-slot ring; 120 frames at 30 fps wrap it seven times.
    const SLOTS: usize = 16;
    let mut ring: Vec<Option<(u32, Vec<u8>)>> = vec![None; SLOTS];
    let mut truth: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    let mut closed = Vec::new();
    for i in 0..120u32 {
        let t = T0 + u64::from(i) * FRAME_NS;
        let bytes = frame_bytes(i);
        ring[i as usize % SLOTS] = Some((i, bytes.clone()));
        // The tier worker follows the stream and appends each payload as it lands, before the slot is reused.
        let (_, done) = flusher
            .append(Incoming { id: uuid(i), stream: cam, start_ns: t, end_ns: t, seq: i, bytes: &bytes, key: true }, t)
            .expect("append");
        closed.extend(done);
        truth.insert(t, bytes);
    }
    closed.extend(flusher.close_all().expect("close"));
    // Every frame is in exactly one chunk, in order.
    let mut all: Vec<(u64, u32)> = closed.iter().flat_map(|c| c.records.iter().map(|r| (r.start_ns, r.seq))).collect();
    assert_eq!(all.len(), 120);
    all.sort_unstable();
    assert_eq!(all.iter().map(|(_, s)| *s).collect::<Vec<_>>(), (0..120).collect::<Vec<_>>());
    assert!(closed.len() >= 5, "chunks closed on time: {}", closed.len());

    // Any 50 ms slice hydrates to what the producer wrote, including frames the ring has long overwritten.
    for start in [T0 + 100 * MS, T0 + 1_500 * MS, T0 + 3_700 * MS] {
        let end = start + 50 * MS;
        let expected: Vec<&Vec<u8>> = truth.range(start..end).map(|(_, b)| b).collect();
        let mut got = Vec::new();
        for c in &closed {
            if c.records.iter().any(|r| r.start_ns >= start && r.start_ns < end) {
                got.extend(hydrate(&c.path, start, end).expect("hydrate").into_iter().map(|h| h.bytes));
            }
        }
        assert_eq!(got.iter().collect::<Vec<_>>(), expected, "slice at {start}");
    }
    // The ring itself only holds the last 16.
    assert!(ring.iter().all(|s| s.as_ref().is_some_and(|(i, _)| *i >= 104)));
}

#[test]
fn a_chunk_left_open_by_a_crash_is_recovered_and_committable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = MemoryCatalog::new();
    let cam = register_cam(&catalog);
    let mut flusher = Flusher::new(
        dir.path(),
        FlushSettings { chunk_ns: 10 * S, part_ns: 10 * S, audio_chunk_ns: 10 * S, ..FlushSettings::default() },
    )
    .expect("flusher");
    flusher.declare(cam, shape());
    let mut ids = Vec::new();
    for i in 0..12u32 {
        let t = T0 + u64::from(i) * FRAME_NS;
        catalog.insert(vec![frame_payload(uuid(i), cam, t, i)]).expect("insert");
        flusher
            .append(
                Incoming {
                    id: uuid(i),
                    stream: cam,
                    start_ns: t,
                    end_ns: t,
                    seq: i,
                    bytes: &frame_bytes(i),
                    key: true,
                },
                t,
            )
            .expect("append");
        ids.push(uuid(i));
    }
    let durable = flusher.sync_all().expect("sync");
    assert_eq!(durable, ids, "every record synced, every slot flushable");
    assert_eq!(flusher.open_streams(), vec![cam]);
    // The worker dies here: no close, no footer.
    drop(flusher);

    let recovered = recover(dir.path()).expect("recover");
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].records.len(), 12);
    assert_eq!(recovered[0].stream, cam);
    let reader = ChunkReader::open(&recovered[0].path).expect("open");
    assert!(reader.finalized());
    // A second start finds it finished: it comes back only to be checked against the catalog, which may lack its
    // last part.
    let again = recover(dir.path()).expect("recover");
    assert_eq!(again.iter().map(|r| (&r.path, r.records.len())).collect::<Vec<_>>(), [(&recovered[0].path, 12)]);

    // The recovered chunk commits like any other, with the ids the worker kept beside it.
    let chunk = continuum_tier::flush::new_chunk(
        &recovered[0].path,
        cam,
        &shape(),
        &recovered[0].records,
        recovered[0].created_ns,
    )
    .expect("chunk");
    let chunk_id = chunk.payload.header.id();
    catalog.commit_chunk(chunk, &ids).expect("commit");
    assert!(
        matches!(catalog.get(ids[11]).expect("get"), Some(Resolved::AbsorbedFrame { frame_index: 11, .. })),
        "compacted into the chunk, its row mapping it there"
    );
    let Some(Resolved::Payload(indexed)) = catalog.get(chunk_id).expect("get") else { panic!("chunk") };
    assert_eq!(indexed.frames.as_ref().map(Vec::len), Some(12));
}

#[test]
fn raw_frames_round_trip_through_lz4_chunks() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut flusher = Flusher::new(dir.path(), FlushSettings::default()).expect("flusher");
    let depth = StreamId(2);
    flusher.declare(
        depth,
        StreamShape {
            chunk_format: ChunkFormat::Lz4,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: "Z16".to_owned(),
            shape_version: 1,
            pcm: None,
            raw: None,
        },
    );
    let raw: Vec<u8> = (0..64 * 1024).map(|i| u8::try_from((i / 512) % 200).expect("small")).collect();
    let packed = codecs::lz4_compress(&raw);
    flusher
        .append(
            Incoming { id: uuid(1), stream: depth, start_ns: T0, end_ns: T0, seq: 0, bytes: &packed, key: true },
            T0,
        )
        .expect("append");
    let closed = flusher.close_all().expect("close");
    let got = hydrate(&closed[0].path, T0, T0 + 1).expect("hydrate");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].bytes, raw, "hydrate undoes the LZ4");
    assert!(closed[0].records[0].len < 4096, "stored compressed: {}", closed[0].records[0].len);
}

#[test]
fn a_jpeg_decodes_to_rgb_on_request() {
    let (w, h) = (16u16, 8u16);
    let mut rgb = Vec::with_capacity(usize::from(w) * usize::from(h) * 3);
    for y in 0..h {
        for x in 0..w {
            rgb.extend_from_slice(&[if x < 8 { 220 } else { 30 }, u8::try_from(y * 20).expect("small"), 90]);
        }
    }
    let mut jpeg = Vec::new();
    jpeg_encoder::Encoder::new(&mut jpeg, 95).encode(&rgb, w, h, jpeg_encoder::ColorType::Rgb).expect("encode");
    let picture = decode_jpeg(&jpeg).expect("decode");
    assert_eq!((picture.width, picture.height), (16, 8));
    assert_eq!(picture.rgb.len(), rgb.len());
    let diff: f64 =
        picture.rgb.iter().zip(&rgb).map(|(a, b)| f64::from(a.abs_diff(*b))).sum::<f64>() / rgb.len() as f64;
    assert!(diff < 12.0, "mean difference {diff}");
    let nv12 = continuum_tier::rgb_to_nv12(&picture);
    assert_eq!(nv12.len(), 16 * 8 + 2 * 8 * 4);
    assert!(nv12[0] > nv12[15], "the left half is brighter");
}

#[test]
fn a_flac_stream_stays_flac_in_its_chunk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut flusher = Flusher::new(dir.path(), FlushSettings::default()).expect("flusher");
    let mic = StreamId(3);
    flusher.declare(
        mic,
        StreamShape {
            chunk_format: ChunkFormat::Flac,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: "FLAC".to_owned(),
            shape_version: 1,
            pcm: None,
            raw: None,
        },
    );
    let samples: Vec<i32> = (0..9600i32).map(|i| ((i % 200) - 100) * 50).collect();
    let flac = codecs::flac_encode(&samples, 2, 16, 48_000).expect("flac");
    let block = Interval::new(T0, T0 + 100 * MS).expect("interval");
    flusher
        .append(
            Incoming {
                id: uuid(9),
                stream: mic,
                start_ns: block.start_ns(),
                end_ns: block.end_ns(),
                seq: 0,
                bytes: &flac,
                key: true,
            },
            T0,
        )
        .expect("append");
    let closed = flusher.close_all().expect("close");
    let got = hydrate(&closed[0].path, T0 + 10 * MS, T0 + 20 * MS).expect("hydrate");
    assert_eq!(got.len(), 1, "a slice inside the block finds the block");
    assert_eq!(&got[0].bytes[..4], b"fLaC");
    assert_eq!(got[0].record.end_ns, block.end_ns());
}

#[test]
fn an_out_of_order_payload_starts_a_new_chunk_and_both_commit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = MemoryCatalog::new();
    let cam = register_cam(&catalog);
    let mut flusher = Flusher::new(
        dir.path(),
        FlushSettings { chunk_ns: 10 * S, part_ns: 10 * S, audio_chunk_ns: 10 * S, ..FlushSettings::default() },
    )
    .expect("flusher");
    flusher.declare(cam, shape());
    let mut closed = Vec::new();
    // Frames at 0, 1, 2, then a retransmit of an earlier time, then 4.
    for (i, t) in [(0u32, T0), (1, T0 + FRAME_NS), (2, T0 + 2 * FRAME_NS), (3, T0 - FRAME_NS), (4, T0 + 4 * FRAME_NS)] {
        catalog.insert(vec![frame_payload(uuid(i), cam, t, i)]).expect("insert");
        let (_, done) = flusher
            .append(
                Incoming {
                    id: uuid(i),
                    stream: cam,
                    start_ns: t,
                    end_ns: t,
                    seq: i,
                    bytes: &frame_bytes(i),
                    key: true,
                },
                T0 + 5 * FRAME_NS,
            )
            .expect("append");
        closed.extend(done);
    }
    closed.extend(flusher.close_all().expect("close"));
    assert_eq!(
        closed.iter().map(|c| c.records.len()).collect::<Vec<_>>(),
        vec![3, 2],
        "the late frame opened a new chunk"
    );
    for c in &closed {
        catalog.commit_chunk(c.chunk.clone(), &c.absorbed).expect("both chunks commit in frame order");
    }
}

#[test]
fn a_chunk_closes_at_the_catalog_frame_limit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut flusher = Flusher::new(
        dir.path(),
        FlushSettings { chunk_ns: 60 * S, part_ns: 60 * S, audio_chunk_ns: 60 * S, ..FlushSettings::default() },
    )
    .expect("flusher");
    let imu = StreamId(5);
    flusher.declare(
        imu,
        StreamShape {
            chunk_format: ChunkFormat::Passthrough,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: "IMU".to_owned(),
            shape_version: 1,
            pcm: None,
            raw: None,
        },
    );
    let n = continuum_catalog::MAX_CHUNK_FRAMES as u32 + 10;
    let mut closed = Vec::new();
    for i in 0..n {
        let t = T0 + u64::from(i) * 1_000_000;
        let (_, done) = flusher
            .append(
                Incoming { id: uuid(i), stream: imu, start_ns: t, end_ns: t, seq: i, bytes: &[1, 2, 3], key: true },
                t,
            )
            .expect("append");
        closed.extend(done);
    }
    closed.extend(flusher.close_all().expect("close"));
    assert_eq!(closed[0].records.len(), continuum_catalog::MAX_CHUNK_FRAMES);
    assert_eq!(closed[1].records.len(), 10);
}

#[test]
fn hydrate_by_frame_index_survives_a_purge_rewrite() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut flusher = Flusher::new(dir.path(), FlushSettings::default()).expect("flusher");
    let cam = StreamId(1);
    flusher.declare(cam, shape());
    for i in 0..4u32 {
        let t = T0 + u64::from(i) * FRAME_NS;
        flusher
            .append(
                Incoming {
                    id: uuid(i),
                    stream: cam,
                    start_ns: t,
                    end_ns: t,
                    seq: i,
                    bytes: &frame_bytes(i),
                    key: true,
                },
                T0,
            )
            .expect("append");
    }
    let closed = flusher.close_all().expect("close");
    let path = &closed[0].path;
    let rewritten = continuum_tier::chunk::rewrite_without(path, &[1]).expect("rewrite");
    assert_eq!(rewritten.records.len(), 4);
    // The catalog's FRAMES rows still point at indexes 0, 2, 3; index 1 is a tombstone.
    assert_eq!(continuum_tier::hydrate_frame(path, 3).expect("frame 3").bytes, frame_bytes(3));
    assert_eq!(continuum_tier::hydrate_frame(path, 2).expect("frame 2").bytes, frame_bytes(2));
    assert!(matches!(continuum_tier::hydrate_frame(path, 1), Err(continuum_tier::TierError::Purged(1))));
    let slice = hydrate(path, T0, T0 + 4 * FRAME_NS).expect("hydrate");
    assert_eq!(slice.iter().map(|h| h.index).collect::<Vec<_>>(), vec![0, 2, 3]);
}

#[test]
fn a_truncated_jpeg_is_an_error_not_a_grey_picture() {
    let (w, h) = (32u16, 32u16);
    let rgb = vec![200u8; usize::from(w) * usize::from(h) * 3];
    let mut jpeg = Vec::new();
    jpeg_encoder::Encoder::new(&mut jpeg, 90).encode(&rgb, w, h, jpeg_encoder::ColorType::Rgb).expect("encode");
    assert!(decode_jpeg(&jpeg).is_ok());
    let cut = &jpeg[..jpeg.len() * 2 / 3];
    assert!(decode_jpeg(cut).is_err(), "a frame cut short by a camera glitch must not decode silently");
}

#[test]
fn recovery_cleans_up_rewrite_leftovers_and_empty_chunks() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut flusher = Flusher::new(dir.path(), FlushSettings::default()).expect("flusher");
    let cam = StreamId(1);
    flusher.declare(cam, shape());
    flusher
        .append(
            Incoming { id: uuid(1), stream: cam, start_ns: T0, end_ns: T0, seq: 0, bytes: &frame_bytes(1), key: true },
            T0,
        )
        .expect("append");
    let closed = flusher.close_all().expect("close");
    // An interrupted rewrite left its temp file; a crash left a chunk with only a header.
    std::fs::write(closed[0].path.with_extension("rewrite"), b"CNK1 garbage").expect("write");
    // The newest chunk of its stream, as a crash leaves it, in the same hour directory.
    let empty = closed[0].path.with_file_name(format!("1-{}-1.cnk", T0 + 1));
    let w = continuum_tier::ChunkWriter::create(&empty, cam, ChunkFormat::Passthrough, T0 + 1).expect("create");
    std::mem::forget(w);
    // Only the stream's newest finished chunk comes back, to be checked against the catalog.
    let recovered = recover(dir.path()).expect("recover");
    assert_eq!(recovered.iter().map(|r| &r.path).collect::<Vec<_>>(), [&closed[0].path], "{recovered:?}");
    assert!(!closed[0].path.with_extension("rewrite").exists());
    assert!(!empty.exists());
    assert!(closed[0].path.exists());
}

#[test]
fn a_slot_is_flushed_only_once_its_record_is_on_disk() {
    // The worker appends frames and marks a slot flushed only when sync_due reports its id. A power cut right
    // after a kill loses everything after the last sync (simulated by truncating the chunk to its synced length);
    // nothing reported flushed may be among the lost.
    let dir = tempfile::tempdir().expect("tempdir");
    let settings = FlushSettings { chunk_ns: 60 * S, part_ns: 60 * S, audio_chunk_ns: 60 * S, sync_ns: 200 * MS };
    let mut flusher = Flusher::new(dir.path(), settings).expect("flusher");
    let cam = StreamId(1);
    flusher.declare(cam, shape());
    let mut flushed: BTreeSet<Uuid> = BTreeSet::new();
    let mut synced_len = 0u64;
    let mut ticks = 0;
    for i in 0..40u32 {
        let t = T0 + u64::from(i) * FRAME_NS;
        flusher
            .append(
                Incoming {
                    id: uuid(i),
                    stream: cam,
                    start_ns: t,
                    end_ns: t,
                    seq: i,
                    bytes: &frame_bytes(i),
                    key: true,
                },
                t,
            )
            .expect("append");
        let durable = flusher.sync_due(t).expect("sync");
        if !durable.is_empty() {
            ticks += 1;
            synced_len = std::fs::metadata(flusher.open_path(cam).expect("open")).expect("meta").len();
        }
        flushed.extend(durable);
    }
    let path = flusher.open_path(cam).expect("open").to_path_buf();
    assert!((4..=7).contains(&ticks), "about one sync per 200 ms over 1.3 s, got {ticks}");
    assert!(flusher.pending() > 0, "the last frames are appended but not yet durable");
    assert_eq!(flushed.len() + flusher.pending(), 40);
    // Kill, then power cut: the unsynced tail is gone.
    std::mem::forget(flusher);
    std::fs::OpenOptions::new().write(true).open(&path).expect("open").set_len(synced_len).expect("truncate");
    let recovered = recover(dir.path()).expect("recover");
    let on_disk: BTreeSet<Uuid> = recovered[0].records.iter().map(|r| uuid(r.seq)).collect();
    assert!(flushed.is_subset(&on_disk), "a flushed slot's record is always on disk");
    assert_eq!(flushed, on_disk, "and nothing durable went unreported");
    assert!(on_disk.len() < 40);
}

#[test]
fn closing_a_chunk_reports_the_ids_it_made_durable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut flusher = Flusher::new(dir.path(), FlushSettings::default()).expect("flusher");
    let cam = StreamId(1);
    flusher.declare(cam, shape());
    for i in 0..6u32 {
        let t = T0 + u64::from(i) * FRAME_NS;
        flusher
            .append(
                Incoming {
                    id: uuid(i),
                    stream: cam,
                    start_ns: t,
                    end_ns: t,
                    seq: i,
                    bytes: &frame_bytes(i),
                    key: true,
                },
                t,
            )
            .expect("append");
        if i == 2 {
            assert_eq!(flusher.sync_all().expect("sync"), vec![uuid(0), uuid(1), uuid(2)]);
            assert_eq!(flusher.sync_all().expect("nothing new"), Vec::<Uuid>::new());
        }
    }
    let closed = flusher.close_all().expect("close");
    assert_eq!(closed[0].absorbed.len(), 6);
    assert_eq!(closed[0].durable, vec![uuid(3), uuid(4), uuid(5)]);
}

#[test]
#[cfg(unix)]
fn a_chunk_closed_before_a_failed_create_is_parked_and_handed_back() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = MemoryCatalog::new();
    let cam = register_cam(&catalog);
    let mut flusher = Flusher::new(
        dir.path(),
        FlushSettings { chunk_ns: S, part_ns: S, audio_chunk_ns: S, ..FlushSettings::default() },
    )
    .expect("flusher");
    flusher.declare(cam, shape());
    let bytes = frame_bytes(0);
    let first = Incoming { id: uuid(1), stream: cam, start_ns: 0, end_ns: 0, seq: 0, bytes: &bytes, key: true };
    flusher.append(first, 0).expect("append");
    // The directory the next chunk goes in (its stream's hour) can't take a new file now; the next payload is
    // past the chunk's time, so the old chunk closes and the new one can't be created.
    let hour = continuum_tier::layout::hour_dir(dir.path(), cam, S);
    std::fs::set_permissions(&hour, std::fs::Permissions::from_mode(0o500)).expect("chmod");
    let late = Incoming { id: uuid(2), stream: cam, start_ns: S, end_ns: S, seq: 1, bytes: &bytes, key: true };
    let err = flusher.append(late, S).expect_err("the create fails");
    assert!(matches!(err, continuum_tier::TierError::Io(_) | continuum_tier::TierError::Chunk(_)), "{err}");
    let parked = flusher.parked();
    assert_eq!(parked.len(), 1, "the finished chunk is handed back, not dropped");
    assert_eq!(parked[0].absorbed, [uuid(1)]);
    assert!(parked[0].records.len() == 1 && parked[0].to_commit().frames.len() == 1);
    assert!(flusher.parked().is_empty(), "handed back once");
    assert!(flusher.close_all().expect("close").is_empty(), "nothing is open after the failed create");
    std::fs::set_permissions(&hour, std::fs::Permissions::from_mode(0o700)).expect("chmod back");
}

/// A new chunk goes in its stream's UTC-hour directory, not the chunk directory itself.
#[test]
fn a_new_chunk_goes_in_its_streams_hour_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = MemoryCatalog::new();
    let cam = register_cam(&catalog);
    let mut flusher = Flusher::new(
        dir.path(),
        FlushSettings { chunk_ns: S, part_ns: S, audio_chunk_ns: S, ..FlushSettings::default() },
    )
    .expect("flusher");
    flusher.declare(cam, shape());
    let bytes = frame_bytes(0);
    let at = 1_794_151_800_000_000_000; // 2026-11-08T15:30Z
    let frame = Incoming { id: uuid(1), stream: cam, start_ns: at, end_ns: at, seq: 0, bytes: &bytes, key: true };
    flusher.append(frame, at).expect("append");
    let closed = flusher.close_all().expect("close");
    let expected = dir.path().join("streams").join(cam.0.to_string()).join("2026110815");
    assert_eq!(closed[0].path.parent(), Some(expected.as_path()));
    assert!(continuum_tier::layout::chunk_paths(dir.path(), None).expect("list").contains(&closed[0].path));
}

/// An unfinished chunk in an hour directory (a crash mid-write) is found and finalized by recovery.
#[test]
fn recovery_finds_an_unfinished_chunk_in_an_hour_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let hour = continuum_tier::layout::hour_dir(dir.path(), StreamId(3), 5);
    continuum_tier::layout::make_hour_dir(dir.path(), &hour).expect("hour");
    let path = hour.join("3-5-1.cnk");
    let mut w = continuum_tier::ChunkWriter::create(&path, StreamId(3), continuum_tier::ChunkFormat::Passthrough, 5)
        .expect("create");
    w.append(5, 5, 0, b"bytes").expect("append");
    w.sync().expect("sync");
    drop(w); // no footer: as a crash leaves it
    let found = continuum_tier::recover(dir.path()).expect("recover");
    assert_eq!(found.len(), 1);
    assert_eq!((found[0].path.clone(), found[0].stream, found[0].records.len()), (path, StreamId(3), 1));
}

/// Recovery opens a stream's chunks newest first and stops at its first finished one, which it hands back too
/// (the catalog may lack its last part). A run finishes a stream's chunk before it opens the next, so nothing older
/// can be unfinished; opening every chunk in the archive took minutes at each start, while the encoded cameras'
/// Tier 0 rings filled with unflushed frames.
#[test]
fn recovery_stops_at_each_streams_newest_finished_chunk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let chunk = |stream: u32, created: u64, finish: bool| {
        let hour = continuum_tier::layout::hour_dir(dir.path(), StreamId(stream), created);
        continuum_tier::layout::make_hour_dir(dir.path(), &hour).expect("hour");
        let path = hour.join(format!("{stream}-{created}-0.cnk"));
        let mut w = continuum_tier::ChunkWriter::create(
            &path,
            StreamId(stream),
            continuum_tier::ChunkFormat::Passthrough,
            created,
        )
        .expect("create");
        w.append(created, created, 0, b"bytes").expect("append");
        w.sync().expect("sync");
        if finish {
            w.close().expect("close");
        }
        path
    };
    let behind = chunk(3, T0, false); // older than a finished chunk: never opened
    let finished = chunk(3, T0 + S, true);
    let newest = chunk(3, T0 + 2 * S, false);
    let other = chunk(4, T0, false); // another stream with no finished chunk at all
    let found = continuum_tier::recover(dir.path()).expect("recover");
    let paths: Vec<_> = found.iter().map(|r| r.path.clone()).collect();
    assert_eq!(paths, vec![finished, newest, other]);
    assert!(!ChunkReader::open(&behind).expect("open").finalized(), "left as it was, unopened by recovery");
}

/// docs/CONTINUUM.md, "Video chunks": an open H.264 chunk hands over its new records every `part_ns`, only ever
/// before a keyframe (so a part decodes from its start and never presents before the last one), and the file closes
/// on the first keyframe after `chunk_ns`.
#[test]
fn an_h264_chunk_hands_over_parts_before_keyframes_and_closes_on_one() {
    const TENTH: u64 = 100 * MS;
    let dir = tempfile::tempdir().expect("tempdir");
    let settings = FlushSettings { chunk_ns: 4 * S, part_ns: S, audio_chunk_ns: 4 * S, sync_ns: 200 * MS };
    let mut flusher = Flusher::new(dir.path(), settings).expect("flusher");
    let cam = StreamId(7);
    flusher.declare(
        cam,
        StreamShape { chunk_format: ChunkFormat::H264Au, shape_name: "H264AnnexB".to_owned(), ..shape() },
    );
    // Ten frames a second with a keyframe every 1.5 s, for 7 seconds.
    let mut handed = Vec::new();
    for i in 0..70u32 {
        let t = T0 + u64::from(i) * TENTH;
        let p =
            Incoming { id: uuid(i), stream: cam, start_ns: t, end_ns: t, seq: i, bytes: &[0, 0, 1], key: i % 15 == 0 };
        handed.extend(flusher.append(p, t).expect("append").1);
        assert!(flusher.rotate(t).expect("rotate").is_empty(), "a stream still sending is cut on its keyframes");
    }
    let seen: Vec<(u32, bool, u32, usize)> =
        handed.iter().map(|c| (c.first_frame, c.more, c.records[0].seq, c.records.len())).collect();
    // Parts before the keyframes at 1.5 s and 3 s; the file's last part at 4.5 s, the first keyframe past 4 s; then
    // the next file's first part at 6 s.
    assert_eq!(seen, [(0, true, 0, 15), (15, true, 15, 15), (30, false, 30, 15), (0, true, 45, 15)]);
    let id = |c: &continuum_tier::Closed| c.chunk.payload.header.id();
    assert!(id(&handed[0]) == id(&handed[1]) && id(&handed[1]) == id(&handed[2]), "one chunk, three commits");
    assert_ne!(id(&handed[2]), id(&handed[3]));
    assert_eq!(handed[0].path, handed[2].path);
    // Each commit carries the chunk as it is by then: the whole span so far, and the file's length so far.
    let ends: Vec<u64> = handed[..3].iter().map(|c| c.chunk.payload.header.end_ns()).collect();
    assert_eq!(ends, [T0 + 14 * TENTH + 1, T0 + 29 * TENTH + 1, T0 + 44 * TENTH + 1]);
    assert!(handed[..3].iter().all(|c| c.chunk.payload.header.start_ns() == T0));
    let lengths: Vec<u64> = handed[..3]
        .iter()
        .map(|c| match &c.chunk.payload.location {
            Some(StorageLocator::Chunk { length, .. }) => *length,
            other => panic!("a chunk file: {other:?}"),
        })
        .collect();
    assert!(lengths[0] < lengths[1] && lengths[1] < lengths[2], "{lengths:?}");
    assert_eq!(lengths[2], std::fs::metadata(&handed[2].path).expect("file").len(), "footer included");
    assert!(handed[0].durable.len() == 15, "a part is synced before it is handed over: {:?}", handed[0].durable.len());
    // The camera stops. Once it has been quiet for a second, its waiting records are handed over on time.
    let quiet = T0 + 69 * TENTH + 1_100 * MS;
    assert_eq!(flusher.due(quiet), vec![cam]);
    let last = flusher.rotate(quiet).expect("rotate");
    assert_eq!(last.iter().map(|c| (c.first_frame, c.more, c.records.len())).collect::<Vec<_>>(), [(15, true, 10)]);
    assert!(flusher.due(quiet + TENTH).is_empty(), "nothing waiting, and the file isn't due yet");
    // At shutdown the file gets its footer; every record was handed over already.
    assert!(flusher.close_all().expect("close").is_empty());
    assert!(ChunkReader::open(&handed[3].path).expect("open").finalized());
}

/// A purge that deletes an hour's last chunk removes its directory; a new chunk in that hour makes it again.
#[test]
fn a_chunk_after_its_hour_directory_was_pruned_is_still_created() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = MemoryCatalog::new();
    let cam = register_cam(&catalog);
    let mut flusher = Flusher::new(
        dir.path(),
        FlushSettings { chunk_ns: S, part_ns: S, audio_chunk_ns: S, ..FlushSettings::default() },
    )
    .expect("flusher");
    flusher.declare(cam, shape());
    let bytes = frame_bytes(0);
    let at = 1_794_151_800_000_000_000; // 2026-11-08T15:30Z
    let first = Incoming { id: uuid(1), stream: cam, start_ns: at, end_ns: at, seq: 0, bytes: &bytes, key: true };
    flusher.append(first, at).expect("append");
    let closed = flusher.close_all().expect("close");
    continuum_tier::chunk::remove(&closed[0].path).expect("purged");
    let hour = closed[0].path.parent().expect("hour").to_path_buf();
    assert!(!hour.exists(), "its last chunk went, and the directory with it");
    let next =
        Incoming { id: uuid(2), stream: cam, start_ns: at + S, end_ns: at + S, seq: 1, bytes: &bytes, key: true };
    flusher.append(next, at + S).expect("made again");
    let closed = flusher.close_all().expect("close");
    assert_eq!(closed[0].path.parent(), Some(hour.as_path()));
}
