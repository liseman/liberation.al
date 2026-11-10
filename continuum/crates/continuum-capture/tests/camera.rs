//! The camera loop on a scripted frame source and a scripted clock, publishing into a real standalone arena: frames
//! are stamped and published, driver drops and oversize frames are counted, and a stall, a wake, an unplug, and a
//! stop each end the loop the right way. No device is opened.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use continuum_capture::clock::{Anchor, Clocks};
use continuum_capture::publish::{Sink, SinkError, StreamMeta, VideoStream};
use continuum_capture::stamp::{V4L2_TIMESTAMP_MONOTONIC, V4L2_TSTAMP_SRC_SOE};
use continuum_capture::video::{CameraConfig, CameraCounts, CameraExit, FrameSource, Mode, RawFrame, run_camera};
use continuum_core::{BootId, HeaderBuf, PayloadFormat, StorageLocator, StreamId};
use continuum_shm::{Arena, ArenaConfig, SlabClass};

const SLOT: usize = 16 * 1024;
const FRAME_NS: u64 = 33_333_333;
const SOE: u32 = V4L2_TIMESTAMP_MONOTONIC | V4L2_TSTAMP_SRC_SOE;

struct Named(Arena);

impl Drop for Named {
    fn drop(&mut self) {
        Arena::unlink(self.0.name()).ok();
    }
}

fn arena() -> Named {
    arena_of(SLOT, 8)
}

fn arena_of(slot_size: usize, slot_count: u32) -> Named {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let name = format!("/cnt.cam.{}.{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
    let config = ArenaConfig { classes: vec![SlabClass { slot_size, slot_count }], ..ArenaConfig::default() };
    Named(Arena::create(&name, &config).unwrap())
}

/// A clock the script moves: monotonic, and how long the host has been asleep.
#[derive(Default)]
struct Clock {
    mono: AtomicU64,
    asleep: AtomicU64,
}

impl Clocks for Clock {
    fn monotonic_ns(&self) -> u64 {
        self.mono.load(Ordering::SeqCst)
    }
    fn utc_ns(&self) -> u64 {
        1_800_000_000_000_000_000 + self.monotonic_ns() + self.asleep.load(Ordering::SeqCst)
    }
    fn continuous_ns(&self) -> u64 {
        self.monotonic_ns() + self.asleep.load(Ordering::SeqCst)
    }
    fn boot_id(&self) -> BootId {
        BootId([3; 16])
    }
}

enum Step {
    /// A frame of this many bytes, exposed at this monotonic time, with this driver sequence; it arrives 5 ms later.
    Frame { len: usize, at_ns: u64, seq: u32, flags: u32 },
    /// A start-of-exposure frame whose source says how long it was presented (0: it doesn't say), as
    /// AVFoundation's sample buffers do.
    Presented { at_ns: u64, seq: u32, duration_ns: u64 },
    /// Nothing within the poll: the clock moves by the timeout.
    Timeout,
    /// The host sleeps this long, then the poll times out.
    Sleep(u64),
    /// The device goes away.
    Unplug,
    /// The supervisor's SIGTERM: it sets the stop flag and interrupts the wait.
    Interrupt,
}

struct Script<'c> {
    clock: &'c Clock,
    steps: VecDeque<Step>,
    stop_when_done: &'c AtomicBool,
}

impl FrameSource for Script<'_> {
    fn next_frame(&mut self, timeout_ms: u32, on_frame: &mut dyn FnMut(&RawFrame<'_>)) -> std::io::Result<bool> {
        let Some(step) = self.steps.pop_front() else {
            self.stop_when_done.store(true, Ordering::SeqCst);
            return Ok(false);
        };
        match step {
            Step::Frame { len, at_ns, seq, flags } => {
                self.clock.mono.store(at_ns + 5_000_000, Ordering::SeqCst);
                let bytes = vec![0xab; len];
                let raw = RawFrame {
                    bytes: &bytes,
                    tv_sec: (at_ns / 1_000_000_000) as i64,
                    tv_usec: (at_ns % 1_000_000_000 / 1_000) as i64,
                    flags,
                    sequence: seq,
                    duration_ns: 0,
                };
                on_frame(&raw);
                Ok(true)
            }
            Step::Presented { at_ns, seq, duration_ns } => {
                self.clock.mono.store(at_ns + 5_000_000, Ordering::SeqCst);
                let bytes = vec![0xab; 100];
                let raw = RawFrame {
                    bytes: &bytes,
                    tv_sec: (at_ns / 1_000_000_000) as i64,
                    tv_usec: (at_ns % 1_000_000_000 / 1_000) as i64,
                    flags: SOE,
                    sequence: seq,
                    duration_ns,
                };
                on_frame(&raw);
                Ok(true)
            }
            Step::Timeout => {
                self.clock.mono.fetch_add(u64::from(timeout_ms) * 1_000_000, Ordering::SeqCst);
                Ok(false)
            }
            Step::Sleep(ns) => {
                self.clock.asleep.fetch_add(ns, Ordering::SeqCst);
                Ok(false)
            }
            Step::Unplug => Err(std::io::Error::from_raw_os_error(libc::ENODEV)),
            Step::Interrupt => {
                self.stop_when_done.store(true, Ordering::SeqCst);
                Err(std::io::Error::from(std::io::ErrorKind::Interrupted))
            }
        }
    }
}

#[derive(Default)]
struct Notices(Vec<(HeaderBuf, Option<StorageLocator>)>);

impl Sink for Notices {
    fn ingest(&mut self, header: HeaderBuf, locator: Option<StorageLocator>) -> Result<(), SinkError> {
        self.0.push((header, locator));
        Ok(())
    }
}

const ANCHOR: Anchor = Anchor { boot: BootId([3; 16]), mono_ns: 0, utc_ns: 1_800_000_000_000_000_000, bracket_ns: 0 };

fn setup() -> (Named, VideoStream, CameraConfig) {
    let a = arena();
    let meta = StreamMeta {
        stream: StreamId(9),
        format: PayloadFormat::EncodedVideo,
        shape_name: "MJPEG".into(),
        anchor: ANCHOR,
        residual_offset_ns: 0,
    };
    let stream = VideoStream::new(meta, a.0.clone(), 0, 1, None);
    let config = CameraConfig {
        mode: Mode { fourcc: *b"MJPG", width: 1280, height: 720 },
        stride: 0,
        frame_interval_ns: FRAME_NS,
        flip: true,
        stall_ns: 1_000_000_000,
        slot_bytes: SLOT,
    };
    (a, stream, config)
}

fn run(steps: Vec<Step>, config: CameraConfig, stream: &mut VideoStream) -> (CameraExit, CameraCounts, Notices) {
    let clock = Clock::default();
    clock.mono.store(1_000_000_000, Ordering::SeqCst);
    let stop = AtomicBool::new(false);
    let mut source = Script { clock: &clock, steps: steps.into(), stop_when_done: &stop };
    let mut sink = Notices::default();
    let mut counts = CameraCounts::default();
    let exit =
        run_camera(&mut source, stream, &mut sink, &clock, &config, &stop, &mut counts, &mut |_| {}, &mut |_, _| {});
    (exit, counts, sink)
}

#[test]
fn frames_are_stamped_published_and_counted_until_the_camera_is_unplugged() {
    let (_a, mut stream, config) = setup();
    // On whole microseconds, as a V4L2 timestamp is.
    let t = |n: u64| 2_000_000_000 + n * 33_333_000;
    let steps = vec![
        Step::Frame { len: 9_000, at_ns: t(0), seq: 10, flags: SOE },
        Step::Frame { len: 9_000, at_ns: t(1), seq: 11, flags: SOE },
        // The driver dropped two frames (12 and 13).
        Step::Frame { len: 9_000, at_ns: t(4), seq: 14, flags: SOE },
        // Bigger than a slot: dropped and counted.
        Step::Frame { len: SLOT + 1, at_ns: t(5), seq: 15, flags: SOE },
        // An end-of-frame stamp, and one on a clock that isn't monotonic.
        Step::Frame { len: 9_000, at_ns: t(6), seq: 16, flags: V4L2_TIMESTAMP_MONOTONIC },
        Step::Frame { len: 9_000, at_ns: t(7), seq: 17, flags: 0x4000 },
        // Marked damaged by the driver: skipped.
        Step::Frame { len: 9_000, at_ns: t(8), seq: 18, flags: SOE | 0x40 },
        Step::Unplug,
    ];
    let (exit, counts, sink) = run(steps, config, &mut stream);
    assert!(matches!(exit, CameraExit::Source(ref e) if e.raw_os_error() == Some(libc::ENODEV)), "{exit:?}");
    assert_eq!(
        counts,
        CameraCounts {
            frames: 5,
            driver_dropped: 2,
            oversize: 1,
            arena_full_dropped: 0,
            unflushed_overwrites: 0,
            arrival_stamped: 1,
            damaged: 1
        }
    );
    let starts: Vec<u64> = sink.0.iter().map(|(h, _)| h.start_ns() - ANCHOR.utc_ns).collect();
    assert_eq!(starts, vec![t(0), t(1), t(4), t(6) - FRAME_NS, t(7) + 5_000_000], "SOE, EOF moved back, arrival");
    let first = sink.0[0].0.header().capture().unwrap();
    assert_eq!((first.device_seq(), first.width(), first.height(), first.flip()), (10, 1280, 720, true));
    assert!(matches!(sink.0[0].1, Some(StorageLocator::Shm { length: 9_000, .. })));
}

#[test]
fn a_source_duration_ends_the_frame_and_a_missing_one_leaves_an_instant() {
    let (_a, mut stream, config) = setup();
    let steps = vec![
        Step::Presented { at_ns: 2_000_000_000, seq: 0, duration_ns: 33_333_333 },
        // The source said nothing for this one; the camera's nominal 30 fps must not fill it in.
        Step::Presented { at_ns: 2_033_333_000, seq: 1, duration_ns: 0 },
        // A frame after a gap keeps its own duration rather than stretching to the next frame.
        Step::Presented { at_ns: 2_500_000_000, seq: 9, duration_ns: 16_666_667 },
        Step::Unplug,
    ];
    let (_, counts, sink) = run(steps, config, &mut stream);
    assert_eq!(counts.frames, 3);
    let spans: Vec<(u64, u64)> =
        sink.0.iter().map(|(h, _)| (h.start_ns() - ANCHOR.utc_ns, h.end_ns() - ANCHOR.utc_ns)).collect();
    assert_eq!(
        spans,
        vec![(2_000_000_000, 2_033_333_333), (2_033_333_000, 2_033_333_000), (2_500_000_000, 2_516_666_667)]
    );
}

#[test]
fn no_frame_for_the_stall_time_is_a_stall() {
    let (_a, mut stream, config) = setup();
    let mut steps = vec![Step::Frame { len: 100, at_ns: 2_000_000_000, seq: 0, flags: SOE }];
    steps.extend((0..5).map(|_| Step::Timeout)); // 5 x 200 ms polls
    steps.push(Step::Frame { len: 100, at_ns: 9_000_000_000, seq: 1, flags: SOE });
    let (exit, counts, _) = run(steps, config, &mut stream);
    assert!(matches!(exit, CameraExit::Stalled), "{exit:?}");
    assert_eq!(counts.frames, 1, "it stalled before the late frame");
}

#[test]
fn a_wake_hands_back_a_new_anchor_without_closing_the_device() {
    let (_a, mut stream, config) = setup();
    let steps =
        vec![Step::Frame { len: 100, at_ns: 2_000_000_000, seq: 0, flags: SOE }, Step::Sleep(3_600_000_000_000)];
    let (exit, _, _) = run(steps, config, &mut stream);
    let CameraExit::Woke(anchor) = exit else { panic!("{exit:?}") };
    assert_eq!(anchor.utc_ns - anchor.mono_ns, ANCHOR.utc_ns + 3_600_000_000_000, "the new mapping includes the sleep");
}

#[test]
fn frames_of_one_session_share_an_epoch_and_the_session_after_a_wake_gets_another() {
    let (_a, mut stream, config) = setup();
    let before = vec![
        Step::Frame { len: 100, at_ns: 2_000_000_000, seq: 0, flags: SOE },
        Step::Frame { len: 100, at_ns: 2_033_333_000, seq: 1, flags: SOE },
        Step::Sleep(3_600_000_000_000),
    ];
    let (_, _, a) = run(before, config, &mut stream);
    // The device stays open across the wake and its sequence carries on, but continuity across a sleep isn't
    // claimed: the next session is a new epoch.
    let (_, _, b) = run(
        vec![Step::Frame { len: 100, at_ns: 3_000_000_000, seq: 2, flags: SOE }, Step::Unplug],
        config,
        &mut stream,
    );
    let epoch = |n: &Notices, i: usize| n.0[i].0.header().capture().unwrap().capture_epoch();
    assert_ne!(epoch(&a, 0), 0, "zero means unknown");
    assert_eq!(epoch(&a, 0), epoch(&a, 1));
    assert_ne!(epoch(&b, 0), epoch(&a, 0));
}

#[test]
fn a_stop_that_interrupts_a_wait_is_a_clean_stop() {
    let (_a, mut stream, config) = setup();
    let steps = vec![Step::Frame { len: 100, at_ns: 2_000_000_000, seq: 0, flags: SOE }, Step::Interrupt];
    let (exit, _, _) = run(steps, config, &mut stream);
    assert!(matches!(exit, CameraExit::Stopped), "{exit:?}");
}

#[test]
fn health_is_reported_every_ten_seconds() {
    let (_a, mut stream, config) = setup();
    // Frames 4 s apart from 2 s to 26 s: the loop starts at 1 s, so reports fall due at 11 s and 21 s.
    let steps = (0..7)
        .map(|i| Step::Frame { len: 100, at_ns: 2_000_000_000 + i * 4_000_000_000, seq: i as u32, flags: SOE })
        .collect();
    let config = CameraConfig { stall_ns: 60_000_000_000, ..config };
    let clock = Clock::default();
    clock.mono.store(1_000_000_000, Ordering::SeqCst);
    let stop = AtomicBool::new(false);
    let mut source = Script { clock: &clock, steps, stop_when_done: &stop };
    let mut reports = Vec::new();
    let mut counts = CameraCounts::default();
    let exit = run_camera(
        &mut source,
        &mut stream,
        &mut Notices::default(),
        &clock,
        &config,
        &stop,
        &mut counts,
        &mut |c| reports.push(c.frames),
        &mut |_, _| {},
    );
    assert!(matches!(exit, CameraExit::Stopped), "{exit:?}");
    assert_eq!(reports.len(), 2, "{reports:?}");
    assert!(reports[0] < reports[1], "{reports:?}");
}

#[test]
fn the_stop_flag_ends_the_loop() {
    let (_a, mut stream, config) = setup();
    let (exit, counts, _) =
        run(vec![Step::Frame { len: 100, at_ns: 2_000_000_000, seq: 0, flags: SOE }], config, &mut stream);
    assert!(matches!(exit, CameraExit::Stopped), "{exit:?}");
    assert_eq!(counts.frames, 1);
}

#[test]
fn raw_1080p_slot_exhaustion_drops_frames_then_resumes_with_real_timestamps_and_sequence_gaps() {
    use continuum_capture::publish::Capture;
    use continuum_shm::{Lease, WriteSlot};
    let bytes = 1920 * 1080 * 2;
    let arena = arena_of(bytes, 2);
    let meta = StreamMeta {
        stream: StreamId(9),
        format: PayloadFormat::RawBytes,
        shape_name: "UYVY".into(),
        anchor: ANCHOR,
        residual_offset_ns: 0,
    };
    let mut stream = VideoStream::new(meta, arena.0.clone(), 0, 1, None);
    let mut notices = Notices::default();
    let first_at = 2_000_000_000;
    let first = stream
        .publish(
            Capture { mono_ns: first_at, device_seq: 9, width: 1920, height: 1080, stride: 3840, ..Capture::default() },
            &mut notices,
            |slot| {
                slot[..bytes].fill(128);
                bytes
            },
        )
        .unwrap();
    let pin = arena.0.lease(first.slot, std::time::Duration::from_secs(30)).unwrap();
    let writing = arena.0.claim(0, 2).unwrap();
    struct Pressure<'c> {
        script: Script<'c>,
        held: Option<(Lease, WriteSlot)>,
        polls: usize,
    }
    impl FrameSource for Pressure<'_> {
        fn next_frame(&mut self, timeout_ms: u32, frame: &mut dyn FnMut(&RawFrame<'_>)) -> std::io::Result<bool> {
            if self.polls == 2 {
                self.held.take();
            }
            self.polls += 1;
            self.script.next_frame(timeout_ms, frame)
        }
    }
    let clock = Clock::default();
    clock.mono.store(first_at, Ordering::SeqCst);
    let stop = AtomicBool::new(false);
    let steps = (10..=12)
        .map(|seq| Step::Frame { len: bytes, at_ns: first_at + u64::from(seq - 9) * 33_333_000, seq, flags: SOE })
        .collect();
    let mut source = Pressure {
        script: Script { clock: &clock, steps, stop_when_done: &stop },
        held: Some((pin, writing)),
        polls: 0,
    };
    let config = CameraConfig {
        mode: Mode { fourcc: *b"UYVY", width: 1920, height: 1080 },
        stride: 3840,
        frame_interval_ns: FRAME_NS,
        flip: false,
        stall_ns: FRAME_NS,
        slot_bytes: bytes,
    };
    let mut counts = CameraCounts::default();
    let mut offers = 0;
    let exit = run_camera(
        &mut source,
        &mut stream,
        &mut notices,
        &clock,
        &config,
        &stop,
        &mut counts,
        &mut |_| {},
        &mut |_, _| offers += 1,
    );
    assert!(matches!(exit, CameraExit::Stopped), "arena pressure must not terminate capture: {exit:?}");
    assert_eq!((counts.frames, counts.arena_full_dropped, counts.driver_dropped, offers), (1, 2, 0, 1));
    let got: Vec<_> = notices
        .0
        .iter()
        .map(|(header, _)| (header.header().capture().unwrap().device_seq(), header.start_ns() - ANCHOR.utc_ns))
        .collect();
    assert_eq!(got, [(9, first_at), (12, first_at + 3 * 33_333_000)], "drops must not renumber or retime later frames");
}

#[test]
fn non_pressure_shared_memory_and_notice_errors_still_end_capture() {
    use continuum_capture::publish::PublishError;
    let (arena, _, config) = setup();
    let meta = StreamMeta {
        stream: StreamId(9),
        format: PayloadFormat::EncodedVideo,
        shape_name: "MJPEG".into(),
        anchor: ANCHOR,
        residual_offset_ns: 0,
    };
    let mut bad_class = VideoStream::new(meta, arena.0.clone(), 999, 1, None);
    let steps = vec![Step::Frame { len: 100, at_ns: 2_000_000_000, seq: 10, flags: SOE }];
    let (exit, counts, _) = run(steps, config, &mut bad_class);
    assert!(matches!(exit, CameraExit::Publish(PublishError::Shm(continuum_shm::Error::NoSuchSlot))));
    assert_eq!(counts.arena_full_dropped, 0);

    struct Refused;
    impl Sink for Refused {
        fn ingest(&mut self, _: HeaderBuf, _: Option<StorageLocator>) -> Result<(), SinkError> {
            Err(SinkError("synthetic broker disconnect".into()))
        }
    }
    let (_arena, mut stream, config) = setup();
    let clock = Clock::default();
    let stop = AtomicBool::new(false);
    let mut source = Script {
        clock: &clock,
        steps: vec![Step::Frame { len: 100, at_ns: 2_000_000_000, seq: 10, flags: SOE }].into(),
        stop_when_done: &stop,
    };
    let mut counts = CameraCounts::default();
    let exit = run_camera(
        &mut source,
        &mut stream,
        &mut Refused,
        &clock,
        &config,
        &stop,
        &mut counts,
        &mut |_| {},
        &mut |_, _| panic!("failed publication must not offer steering"),
    );
    assert!(matches!(exit, CameraExit::Publish(PublishError::Sink(_))));
    assert_eq!((counts.frames, counts.arena_full_dropped), (0, 0));
}
