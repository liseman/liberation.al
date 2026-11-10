//! The microphone loop on scripted PCM and a scripted clock, publishing into a real standalone ring: PipeWire-sized
//! quanta become 10 ms blocks and one notice a second, a lost quantum is a gap, and a stall, a wake, a failure, and
//! a stop each end the loop the right way. No device is opened.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use continuum_capture::audio::{Chunk, ChunkSource, MicExit, chunk_channel, run_microphone};
use continuum_capture::clock::{Anchor, Clocks};
use continuum_capture::publish::{AudioStream, Sink, SinkError, StreamMeta};
use continuum_capture::reblock::Reblocker;
use continuum_core::{BootId, HeaderBuf, PayloadFormat, StorageLocator, StreamId};
use continuum_shm::{Arena, ArenaConfig, RingSpec};

const RATE: u32 = 48_000;
const QUANTUM: u64 = 1_024;
const BLOCK: u64 = 480;

struct Named(Arena);

impl Drop for Named {
    fn drop(&mut self) {
        Arena::unlink(self.0.name()).ok();
    }
}

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
        1_800_000_000_000_000_000 + self.continuous_ns()
    }
    fn continuous_ns(&self) -> u64 {
        self.monotonic_ns() + self.asleep.load(Ordering::SeqCst)
    }
    fn boot_id(&self) -> BootId {
        BootId([4; 16])
    }
}

enum Step {
    /// Quantum `n` (1024 frames of 48 kHz mono S16), captured at its nominal time, arriving 2 ms after it ends.
    Quantum(u64),
    Timeout,
    Sleep(u64),
    Fail,
}

fn quantum_ns(n: u64) -> u64 {
    1_000_000_000 + n * QUANTUM * 1_000_000_000 / u64::from(RATE)
}

struct Script<'c> {
    clock: &'c Clock,
    steps: VecDeque<Step>,
    stop: &'c AtomicBool,
}

impl ChunkSource for Script<'_> {
    fn next_chunk(&mut self, timeout_ms: u32) -> std::io::Result<Option<Chunk>> {
        let Some(step) = self.steps.pop_front() else {
            self.stop.store(true, Ordering::SeqCst);
            return Ok(None);
        };
        match step {
            Step::Quantum(n) => {
                let at = quantum_ns(n);
                self.clock.mono.store(quantum_ns(n + 1) + 2_000_000, Ordering::SeqCst);
                Ok(Some(Chunk { bytes: vec![0; QUANTUM as usize * 2], captured_ns: at }))
            }
            Step::Timeout => {
                self.clock.mono.fetch_add(u64::from(timeout_ms) * 1_000_000, Ordering::SeqCst);
                Ok(None)
            }
            Step::Sleep(ns) => {
                self.clock.asleep.fetch_add(ns, Ordering::SeqCst);
                Ok(None)
            }
            Step::Fail => Err(std::io::Error::other("node removed")),
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

const ANCHOR: Anchor = Anchor { boot: BootId([4; 16]), mono_ns: 0, utc_ns: 1_800_000_000_000_000_000, bracket_ns: 0 };

struct Run {
    exit: MicExit,
    notices: Notices,
    stream: AudioStream,
    _arena: Named,
}

fn run(steps: Vec<Step>) -> Run {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let name = format!("/cnt.mic.{}.{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
    let arena = Named(
        Arena::create(
            &name,
            &ArenaConfig { rings: vec![RingSpec { stride: 960 + 16, capacity: 2_000 }], ..ArenaConfig::default() },
        )
        .unwrap(),
    );
    let meta = StreamMeta {
        stream: StreamId(5),
        format: PayloadFormat::RawBytes,
        shape_name: "PCM_S16LEx1".into(),
        anchor: ANCHOR,
        residual_offset_ns: 0,
    };
    let mut stream = AudioStream::new(meta, arena.0.ring_writer(0).unwrap(), RATE, BLOCK, 1.0).unwrap();
    let mut reblocker = Reblocker::new(2, BLOCK as usize, RATE);
    let clock = Clock::default();
    clock.mono.store(1_000_000_000, Ordering::SeqCst);
    let stop = AtomicBool::new(false);
    let mut source = Script { clock: &clock, steps: steps.into(), stop: &stop };
    let mut notices = Notices::default();
    let exit =
        run_microphone(&mut source, &mut stream, &mut reblocker, &mut notices, &clock, 500_000_000, &stop, &mut |_| {});
    Run { exit, notices, stream, _arena: arena }
}

fn span(h: &HeaderBuf) -> (u64, u64) {
    (h.start_ns() - ANCHOR.utc_ns, h.end_ns() - ANCHOR.utc_ns)
}

#[test]
fn quanta_become_one_notice_a_second_and_the_rest_is_sent_on_stop() {
    // 100 quanta: 102,400 frames, 213 whole blocks (2.13 s).
    let r = run((0..100).map(Step::Quantum).collect());
    assert!(matches!(r.exit, MicExit::Stopped), "{:?}", r.exit);
    assert_eq!(r.stream.counts(), (3, 0), "two full seconds, then the rest at the stop");
    let spans: Vec<_> = r.notices.0.iter().map(|(h, _)| span(h)).collect();
    assert_eq!(spans[0], (1_000_000_000, 2_000_000_000));
    assert_eq!(spans[1], (2_000_000_000, 3_000_000_000));
    assert_eq!(spans[2], (3_000_000_000, 3_000_000_000 + 13 * 10_000_000));
}

#[test]
fn a_lost_quantum_is_a_gap_that_ends_the_batch() {
    let mut steps: Vec<Step> = (0..30).map(Step::Quantum).collect();
    steps.extend((31..60).map(Step::Quantum)); // quantum 30 never arrives
    let r = run(steps);
    let (_, gaps) = r.stream.counts();
    assert_eq!(gaps, 1);
    let spans: Vec<_> = r.notices.0.iter().map(|(h, _)| span(h)).collect();
    // The first batch ends where quantum 30 would have begun (its last whole block), the next starts at quantum 31.
    assert!(spans[0].1 <= quantum_ns(30), "{spans:?}");
    assert_eq!(spans[1].0, quantum_ns(31), "{spans:?}");
}

#[test]
fn silence_for_the_stall_time_is_a_stall_and_what_came_before_is_sent() {
    let mut steps: Vec<Step> = (0..10).map(Step::Quantum).collect();
    steps.extend((0..10).map(|_| Step::Timeout));
    let r = run(steps);
    assert!(matches!(r.exit, MicExit::Stalled), "{:?}", r.exit);
    assert_eq!(r.notices.0.len(), 1, "the partial batch is sent on the way out");
}

#[test]
fn a_wake_reanchors_and_a_failure_ends_the_loop() {
    let r = run(vec![Step::Quantum(0), Step::Quantum(1), Step::Sleep(60_000_000_000)]);
    let MicExit::Woke(anchor) = r.exit else { panic!("{:?}", r.exit) };
    assert_eq!(anchor.utc_ns - anchor.mono_ns, ANCHOR.utc_ns + 60_000_000_000);
    assert_eq!(r.notices.0.len(), 1, "the batch before the sleep is sent with the old anchor");
    let r = run(vec![Step::Quantum(0), Step::Fail]);
    assert!(matches!(r.exit, MicExit::Source(ref e) if e.to_string() == "node removed"), "{:?}", r.exit);
}

#[test]
fn the_callback_channel_never_blocks_and_counts_what_it_drops() {
    let (tx, mut rx) = chunk_channel(2);
    for n in 0..5 {
        tx.send(Chunk { bytes: vec![0; 4], captured_ns: n });
    }
    assert_eq!(rx.dropped(), 3);
    assert_eq!(rx.next_chunk(10).unwrap().map(|c| c.captured_ns), Some(0));
    assert_eq!(rx.next_chunk(10).unwrap().map(|c| c.captured_ns), Some(1));
    assert_eq!(rx.next_chunk(10).unwrap(), None, "a timeout");
    tx.fail("xrun recovery failed".into());
    assert_eq!(rx.next_chunk(10).unwrap_err().to_string(), "xrun recovery failed");
    drop(tx);
    assert!(rx.next_chunk(10).is_err(), "the stream ended");
}
