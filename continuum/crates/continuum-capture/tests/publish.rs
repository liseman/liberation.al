//! Publishing into a standalone shared memory arena: slots, the overwrite rule, the handles ring, and batched PCM.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use continuum_capture::clock::Anchor;
use continuum_capture::publish::{
    AudioStream, Capture, HANDLE_RECORD, PCM_RECORD_HEADER, PublishError, Sink, SinkError, StreamMeta, VideoStream,
};
use continuum_core::{BootId, HeaderBuf, PayloadFormat, SlotHandle, StorageLocator, StreamId};
use continuum_shm::{Arena, ArenaConfig, ReadFrom, RingSpec, SlabClass};

/// A uniquely named arena, unlinked when the test ends.
struct Named(Arena);

impl Drop for Named {
    fn drop(&mut self) {
        Arena::unlink(self.0.name()).ok();
    }
}

const SLOTS: u32 = 4;
const PCM_STRIDE: usize = 960; // 10 ms of 48 kHz mono s16

fn arena() -> Named {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let name = format!("/cnt.cap.{}.{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
    let config = ArenaConfig {
        classes: vec![SlabClass { slot_size: 64 * 1024, slot_count: SLOTS }],
        rings: vec![
            RingSpec { stride: HANDLE_RECORD as u32, capacity: 16 },
            RingSpec { stride: (PCM_STRIDE + PCM_RECORD_HEADER) as u32, capacity: 1_000 },
        ],
        ..ArenaConfig::default()
    };
    Named(Arena::create(&name, &config).unwrap())
}

#[derive(Default)]
struct Notices(Vec<(HeaderBuf, Option<StorageLocator>)>);

impl Sink for Notices {
    fn ingest(&mut self, header: HeaderBuf, locator: Option<StorageLocator>) -> Result<(), SinkError> {
        self.0.push((header, locator));
        Ok(())
    }
}

fn meta(format: PayloadFormat, shape: &str, residual_offset_ns: i64) -> StreamMeta {
    StreamMeta {
        stream: StreamId(7),
        format,
        shape_name: shape.into(),
        anchor: Anchor {
            boot: BootId([1; 16]),
            mono_ns: 1_000_000_000,
            utc_ns: 1_800_000_000_000_000_000,
            bracket_ns: 0,
        },
        residual_offset_ns,
    }
}

fn frame(n: u64) -> Capture {
    Capture { mono_ns: 2_000_000_000 + n * 33_333_333, device_seq: n, width: 1920, height: 1080, ..Capture::default() }
}

#[test]
fn each_frame_lands_in_a_slot_and_its_notice_says_where_and_when() {
    let a = arena();
    let mut v = VideoStream::new(meta(PayloadFormat::EncodedVideo, "MJPEG", 33_000_000), a.0.clone(), 0, 42, None);
    let mut sink = Notices::default();
    let p = v
        .publish(frame(0), &mut sink, |buf| {
            buf[..4].copy_from_slice(b"\xff\xd8\xff\xe0");
            4
        })
        .unwrap();
    let (header, locator) = &sink.0[0];
    let cap = header.header().capture().unwrap();
    assert_eq!((cap.width(), cap.height(), cap.device_seq()), (1920, 1080, 0));
    // A second after the anchor, less the 33 ms residual offset.
    assert_eq!(header.start_ns(), 1_800_000_000_000_000_000 + 1_000_000_000 - 33_000_000);
    let Some(StorageLocator::Shm { slot, length, slot_generation, arena, .. }) = locator else { panic!("{locator:?}") };
    assert_eq!((*slot, *length, *slot_generation), (p.slot.index, 4, p.slot.generation));
    assert_eq!(arena, a.0.name());
    let lease = a.0.lease(p.slot, Duration::from_secs(1)).unwrap();
    // SAFETY: the lease pins the slot for the next second, so its bytes don't change while they're read.
    let bytes = unsafe { std::slice::from_raw_parts(lease.as_ptr(), lease.len()) };
    assert_eq!(&bytes[..4], b"\xff\xd8\xff\xe0", "the camera's own JPEG bytes, not decoded");
}

#[test]
fn encoded_network_sequences_preserve_decode_order_and_pressure_never_overwrites() {
    let a = arena();
    let mut stream = VideoStream::new(meta(PayloadFormat::EncodedVideo, "H264AnnexB", 0), a.0.clone(), 0, 42, None);
    let mut sink = Notices::default();
    for (index, seq) in [7, 9, u64::from(u32::MAX), u64::from(u32::MAX) + 1].into_iter().enumerate() {
        let mut capture = frame(0);
        capture.device_seq = seq;
        capture.mono_ns -= index as u64;
        stream.publish_in_class(0, capture, &mut sink, |_| 10).unwrap();
        assert_eq!(sink.0[index].0.header().seq(), seq as u32);
        assert_eq!(sink.0[index].0.header().capture().unwrap().device_seq(), seq);
    }
    assert!(stream.publish_in_class(0, frame(11), &mut sink, |_| 10).is_err());
    assert_eq!(sink.0.len(), SLOTS as usize);
}

#[test]
fn a_full_ring_overwrites_the_oldest_unpinned_slot_and_never_waits() {
    let a = arena();
    let mut v = VideoStream::new(meta(PayloadFormat::EncodedVideo, "MJPEG", 0), a.0.clone(), 0, 42, None);
    let mut sink = Notices::default();
    let first: Vec<SlotHandle> =
        (0..u64::from(SLOTS)).map(|n| v.publish(frame(n), &mut sink, |_| 10).unwrap().slot).collect();
    // A reader holds the oldest frame: the next write takes the second oldest instead.
    let held = a.0.lease(first[0], Duration::from_secs(5)).unwrap();
    let p = v.publish(frame(4), &mut sink, |_| 10).unwrap();
    assert_eq!(p.slot.index, first[1].index, "the oldest unpinned");
    assert_eq!(p.unflushed_overwrites, 1, "it wasn't flushed to Tier 1: a gap the broker logs");
    assert!(held.is_valid(), "the pinned frame is untouched");
    assert_eq!(v.published(), 5);
}

#[test]
fn a_live_reader_waits_on_the_handles_ring_and_gets_each_slot() {
    let a = arena();
    let handles = a.0.ring_writer(0).unwrap();
    let mut reader = a.0.ring_reader(0, ReadFrom::Latest).unwrap();
    let mut v = VideoStream::new(meta(PayloadFormat::EncodedVideo, "MJPEG", 0), a.0.clone(), 0, 42, Some(handles));
    let mut sink = Notices::default();
    let waiter = reader.waiter();
    let p = v.publish(frame(0), &mut sink, |_| 10).unwrap();
    assert!(waiter.wait(Duration::from_millis(500)), "the publish woke it");
    let mut rec = [0u8; HANDLE_RECORD];
    reader.read(&mut rec).unwrap().unwrap();
    let got = SlotHandle::from_le_bytes(rec[..SlotHandle::ENCODED_LEN].try_into().unwrap());
    assert_eq!(got, p.slot);
}

#[test]
fn audio_is_noticed_once_per_batch_and_a_gap_ends_a_batch() {
    let a = arena();
    let ring = a.0.ring_writer(1).unwrap();
    let mut s = AudioStream::new(meta(PayloadFormat::RawBytes, "PCM_S16LE", 0), ring, 48_000, 480, 1.0).unwrap();
    let mut sink = Notices::default();
    let block = [0u8; PCM_STRIDE];
    let mut t = 2_000_000_000u64;
    for _ in 0..150 {
        s.block(&block, t, t + 1_000_000, &mut sink).unwrap();
        t += 10_000_000;
    }
    assert_eq!(sink.0.len(), 1, "a second of audio is one notice");
    let (h, loc) = &sink.0[0];
    assert_eq!(h.end_ns() - h.start_ns(), 1_000_000_000);
    assert_eq!(h.header().capture().unwrap().sample_rate_hz(), 48_000);
    assert!(loc.is_none(), "the records stay in the ring");
    // An overrun: three blocks lost. The half-second batch in progress is sent at once.
    t += 30_000_000;
    let after = s.block(&block, t, t, &mut sink).unwrap();
    assert_eq!(after.stamp.gap_samples, 1_440);
    assert_eq!(sink.0.len(), 2);
    assert_eq!(sink.0[1].0.end_ns() - sink.0[1].0.start_ns(), 500_000_000);
    assert_eq!(s.counts(), (2, 1));
    // The next batch starts where the gap ends, so the catalog's coverage shows the gap.
    for _ in 0..99 {
        t += 10_000_000;
        s.block(&block, t, t, &mut sink).unwrap();
    }
    let third = &sink.0[2].0;
    let gap_ns = third.start_ns() - sink.0[1].0.end_ns();
    assert!(gap_ns.abs_diff(30_000_000) < 1_000_000, "the gap is {gap_ns} ns");
}

#[test]
fn a_block_of_the_wrong_size_is_refused() {
    let a = arena();
    let mut s =
        AudioStream::new(meta(PayloadFormat::RawBytes, "PCM_S16LE", 0), a.0.ring_writer(1).unwrap(), 48_000, 480, 1.0)
            .unwrap();
    assert!(matches!(s.block(&[0u8; 10], 0, 0, &mut Notices::default()), Err(PublishError::Format(_))));
    // A 0 Hz format is refused when the stream is made, not by a panic later. (Another arena: a ring has one writer.)
    let b = arena();
    let zero =
        AudioStream::new(meta(PayloadFormat::RawBytes, "PCM_S16LE", 0), b.0.ring_writer(1).unwrap(), 0, 480, 1.0);
    assert!(matches!(zero, Err(PublishError::Format(_))));
}

#[test]
fn batches_from_a_drifting_device_touch_exactly_and_each_record_carries_its_stamp() {
    let a = arena();
    let mut reader = a.0.ring_reader(1, ReadFrom::Oldest).unwrap();
    let mut s =
        AudioStream::new(meta(PayloadFormat::RawBytes, "PCM_S16LE", 0), a.0.ring_writer(1).unwrap(), 48_000, 480, 1.0)
            .unwrap();
    let mut sink = Notices::default();
    let block = [7u8; PCM_STRIDE];
    // 50 ppm slow: each 10 ms block takes 10.0005 ms of host time.
    for i in 0..400u64 {
        let t = 2_000_000_000 + i * 10_000_500;
        s.block(&block, t, t, &mut sink).unwrap();
    }
    assert_eq!(sink.0.len(), 4);
    for pair in sink.0.windows(2) {
        assert_eq!(pair[0].0.end_ns(), pair[1].0.start_ns(), "no overlap or hole between batches");
    }
    let mut stamps = Vec::new();
    for _ in 0..3 {
        let mut bytes = vec![0u8; PCM_STRIDE + PCM_RECORD_HEADER];
        reader.read(&mut bytes).unwrap().unwrap();
        let host = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let sample = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(&bytes[PCM_RECORD_HEADER..], &block[..]);
        stamps.push((host, sample));
    }
    assert_eq!(stamps.iter().map(|s| s.1).collect::<Vec<_>>(), vec![0, 480, 960]);
    assert_eq!(stamps[0].0, 2_000_000_000);
}
