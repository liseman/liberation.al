//! A worker end to end against a real broker, in-process, with fake devices and a scripted clock: it registers its
//! stream, publishes, and after the host sleeps and wakes registers again with a new anchor, without the device
//! being closed. No device is opened.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_capture::audio::{Chunk, ChunkSource, Microphone};
use continuum_capture::clock::Clocks;
use continuum_capture::manifest::Manifest;
use continuum_capture::register::PcmFormat;
use continuum_capture::stamp::{V4L2_TIMESTAMP_MONOTONIC, V4L2_TSTAMP_SRC_SOE};
use continuum_capture::video::{Camera, FrameSource, Mode, Opened, RawFrame};
use continuum_capture::worker::{WorkerEnd, camera, microphone};
use continuum_catalog::{Catalog, RedbCatalog};
use continuum_client::{Client, ClientOptions};
use continuum_core::{BootId, Interval};

const SLEEP_NS: u64 = 3_600_000_000_000;

/// A clock the fake devices move: monotonic, and the time spent asleep.
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
        BootId([9; 16])
    }
}

enum Step {
    Frame,
    Sleep,
}

/// A camera that delivers its script (a frame every 33 ms, or a sleep), then stops the worker.
struct FakeCamera<'a> {
    clock: &'a Clock,
    steps: VecDeque<Step>,
    stop: &'a AtomicBool,
    sequence: u32,
}

impl FrameSource for FakeCamera<'_> {
    fn next_frame(&mut self, _timeout_ms: u32, on_frame: &mut dyn FnMut(&RawFrame<'_>)) -> std::io::Result<bool> {
        match self.steps.pop_front() {
            None => {
                self.stop.store(true, Ordering::SeqCst);
                Ok(false)
            }
            Some(Step::Sleep) => {
                self.clock.asleep.fetch_add(SLEEP_NS, Ordering::SeqCst);
                Ok(false)
            }
            Some(Step::Frame) => {
                let at = self.clock.mono.fetch_add(33_000_000, Ordering::SeqCst) + 33_000_000;
                let bytes = vec![0x80u8; 2_000];
                on_frame(&RawFrame {
                    bytes: &bytes,
                    tv_sec: (at / 1_000_000_000) as i64,
                    tv_usec: (at % 1_000_000_000 / 1_000) as i64,
                    flags: V4L2_TIMESTAMP_MONOTONIC | V4L2_TSTAMP_SRC_SOE,
                    sequence: self.sequence,
                    duration_ns: 0,
                });
                self.sequence += 1;
                Ok(true)
            }
        }
    }
}

impl Camera for FakeCamera<'_> {
    fn start(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A microphone that delivers 10 ms chunks of 48 kHz mono f32 (or a sleep), then stops the worker.
struct FakeMic<'a> {
    clock: &'a Clock,
    steps: VecDeque<Step>,
    stop: &'a AtomicBool,
}

impl ChunkSource for FakeMic<'_> {
    fn next_chunk(&mut self, _timeout_ms: u32) -> std::io::Result<Option<Chunk>> {
        match self.steps.pop_front() {
            None => {
                self.stop.store(true, Ordering::SeqCst);
                Ok(None)
            }
            Some(Step::Sleep) => {
                self.clock.asleep.fetch_add(SLEEP_NS, Ordering::SeqCst);
                Ok(None)
            }
            Some(Step::Frame) => {
                let at = self.clock.mono.fetch_add(10_000_000, Ordering::SeqCst);
                Ok(Some(Chunk { bytes: vec![0; 480 * 4], captured_ns: at }))
            }
        }
    }
}

impl Microphone for FakeMic<'_> {
    fn dropped(&self) -> u64 {
        0
    }
}

struct Broker {
    dir: tempfile::TempDir,
    rt: tokio::runtime::Runtime,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl Broker {
    fn start() -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("continuum").join("broker.sock");
        let catalog = Box::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
        let server = rt
            .block_on(Server::bind_with_catalog(small_tier0(Config::new(socket, Duration::from_secs(30))), catalog))
            .unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let task = rt.spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        Self { dir, rt, stop: Some(stop), task: Some(task) }
    }

    fn client(&self, name: &str) -> Client {
        let socket = self.dir.path().join("continuum").join("broker.sock");
        self.rt.block_on(Client::connect(&socket, ClientOptions::new(name))).unwrap()
    }

    /// Stops the broker and opens its catalog once its writer has let go of it.
    fn stop(mut self, client: Client) -> (tempfile::TempDir, RedbCatalog) {
        self.rt.block_on(async {
            client.heartbeat(None).await.unwrap();
            client.close("done").await.unwrap();
        });
        let _ = self.stop.take().unwrap().send(());
        self.rt.block_on(self.task.take().unwrap()).unwrap();
        let path = self.dir.path().join("catalog.redb");
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            match RedbCatalog::open(&path) {
                Ok(c) => return (self.dir, c),
                Err(e) if std::time::Instant::now() < deadline => {
                    let _ = e;
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => panic!("the broker never closed its catalog: {e}"),
            }
        }
    }
}

fn manifest(kind: &str, name: &str) -> Manifest {
    Manifest::parse(&format!("[[device]]\nname = \"{name}\"\nkind = \"{kind}\"\nmatch = {{ name = \"x\" }}\n")).unwrap()
}

#[test]
fn a_camera_worker_registers_publishes_and_registers_again_after_a_wake() {
    let name = format!("cam{}", std::process::id());
    // Opted out of H.264, so the scripted frames publish as the MJPEG they are.
    let m = Manifest::parse(&format!(
        "[[device]]\nname = \"{name}\"\nkind = \"camera\"\nmatch = {{ name = \"x\" }}\nvideo = {{ encoding = {{ enabled = false }} }}\n"
    ))
    .unwrap();
    let device = &m.devices[0];
    let broker = Broker::start();
    let mut client = broker.client("capture test");
    let clock = Clock { mono: AtomicU64::new(5_000_000_000), asleep: AtomicU64::new(0) };
    let stop = AtomicBool::new(false);
    let steps = [Step::Frame, Step::Frame, Step::Frame, Step::Sleep, Step::Frame, Step::Frame].into_iter().collect();
    let mut cam = FakeCamera { clock: &clock, steps, stop: &stop, sequence: 0 };
    let opened = Opened {
        mode: Mode { fourcc: *b"MJPG", width: 1280, height: 720 },
        stride: 0,
        max_frame_bytes: 4_000,
        frame_interval_ns: 33_333_333,
        hold: None,
        shape: "MJPEG".into(),
    };
    // A scripted camera: auto exposure finds no device at this node and leaves it alone.
    let socket = broker.dir.path().join("continuum").join("broker.sock");
    let manifest = broker.dir.path().join("absent-devices.toml");
    let node = ("/nonexistent/video9", socket.as_path(), manifest.as_path());
    let end = camera(broker.rt.handle(), &mut client, device, node, &mut cam, &opened, &clock, &stop);
    assert_eq!(end, WorkerEnd::Stopped);
    let (_dir, catalog) = broker.stop(client);
    let (id, info) =
        catalog.streams(None, 10).unwrap().into_iter().find(|(_, s)| s.name == format!("{name}.video")).unwrap();
    let anchor = info.anchor.unwrap();
    assert_eq!(anchor.utc_ns - anchor.mono_ns, 1_800_000_000_000_000_000 + SLEEP_NS, "the anchor after the wake");
    assert_eq!(info.shape_name, "MJPEG");
    let page = catalog.range(id, Interval::new(0, u64::MAX).unwrap(), None, 10).unwrap();
    assert_eq!(page.items.len(), 5, "every frame, before and after the wake");
}

#[test]
fn a_microphone_worker_publishes_batches_and_registers_again_after_a_wake() {
    let name = format!("mic{}", std::process::id());
    let m = manifest("microphone", &name);
    let device = &m.devices[0];
    let broker = Broker::start();
    let mut client = broker.client("capture test");
    let clock = Clock { mono: AtomicU64::new(5_000_000_000), asleep: AtomicU64::new(0) };
    let stop = AtomicBool::new(false);
    let mut steps: VecDeque<Step> = (0..150).map(|_| Step::Frame).collect();
    steps.push_back(Step::Sleep);
    steps.extend((0..50).map(|_| Step::Frame));
    let mut mic = FakeMic { clock: &clock, steps, stop: &stop };
    let pcm = PcmFormat { rate_hz: 48_000, channels: 1, bytes_per_sample: 4, float: true };
    let end = microphone(broker.rt.handle(), &mut client, device, &mut mic, pcm, &clock, &stop);
    assert_eq!(end, WorkerEnd::Stopped);
    let (_dir, catalog) = broker.stop(client);
    let (id, info) =
        catalog.streams(None, 10).unwrap().into_iter().find(|(_, s)| s.name == format!("{name}.audio")).unwrap();
    let anchor = info.anchor.unwrap();
    assert_eq!(anchor.utc_ns - anchor.mono_ns, 1_800_000_000_000_000_000 + SLEEP_NS);
    assert_eq!(info.shape_name, "PCM_F32LEx1");
    let page = catalog.range(id, Interval::new(0, u64::MAX).unwrap(), None, 10).unwrap();
    // One full second, the half second before the sleep, and the half second after it (sent at the stop).
    let spans: Vec<u64> = page.items.iter().map(|i| i.header.end_ns() - i.header.start_ns()).collect();
    assert_eq!(spans.len(), 3, "{spans:?}");
    assert!(page.items[2].header.start_ns() - page.items[1].header.end_ns() >= SLEEP_NS, "the sleep is a hole");
    // Each batch names its records in the ring the broker carved: 100 blocks for the full second.
    let rings: Vec<(u64, u32)> = page
        .items
        .iter()
        .map(|i| match i.locations.first() {
            Some(continuum_core::StorageLocator::Ring { first_seq, count, .. }) => (*first_seq, *count),
            other => panic!("a ring batch, not {other:?}"),
        })
        .collect();
    assert_eq!(rings[0].1, 100, "{rings:?}");
    assert_eq!(rings[1].0, rings[0].0 + 100, "batches follow each other in the ring: {rings:?}");
}

/// A broker config with a small Tier 0 (the profile's arena is gigabytes): 32 slots of 64 KiB.
fn small_tier0(mut config: Config) -> Config {
    // A camera claims the class its planned frame size fits: 2 MiB for the scripted 1280x720 MJPEG camera.
    config.tier0 = Some(vec![
        continuum_sched::SlotClass { size_bytes: 64 * 1024, count: 32 },
        continuum_sched::SlotClass { size_bytes: 2 << 20, count: 8 },
    ]);
    config
}
