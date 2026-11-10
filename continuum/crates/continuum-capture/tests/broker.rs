//! Capture against a real broker, in-process: register a stream with its clock anchor, publish audio into a ring,
//! and find the batches indexed in the broker's catalog.

use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_capture::clock::{SystemClocks, take_anchor};
use continuum_capture::manifest::Manifest;
use continuum_capture::publish::{AudioStream, StreamMeta};
use continuum_capture::register::{PcmFormat, audio_spec};
use continuum_catalog::{Catalog, RedbCatalog};
use continuum_client::{Client, ClientOptions};
use continuum_core::{Interval, PayloadFormat, StreamId};
use continuum_shm::{Arena, ArenaConfig, RingSpec};

const MANIFEST: &str = r#"
    [[device]]
    name = "wave3"
    kind = "microphone"
    match = { name = "Wave 3" }
"#;

struct Named(Arena);

impl Drop for Named {
    fn drop(&mut self) {
        Arena::unlink(self.0.name()).ok();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_microphone_registers_with_its_anchor_and_its_batches_are_indexed() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("continuum").join("broker.sock");
    let catalog_path = dir.path().join("catalog.redb");
    // A small Tier 0 arena: the profile's is gigabytes.
    let mut config = small_tier0(Config::new(socket.clone(), Duration::from_secs(30)));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16_384, count: 4 }]);
    let server = Server::bind_with_catalog(config, Box::new(RedbCatalog::open(&catalog_path).unwrap())).await.unwrap();
    let core = server.core();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));

    let manifest = Manifest::parse(MANIFEST).unwrap();
    let device = &manifest.devices[0];
    let audio = device.audio().unwrap();
    let pcm = PcmFormat { rate_hz: 48_000, channels: 1, bytes_per_sample: 2, float: false };
    let clocks = SystemClocks::new().unwrap();
    let anchor = take_anchor(&clocks);
    let mut client = Client::connect(&socket, ClientOptions::new("capture-test")).await.unwrap();
    let registered = client.register_stream(&audio_spec(device, &audio, pcm, &anchor)).await.unwrap();

    // The ring the broker's lease would give: a standalone arena until slots come with the shm-in-core step.
    let name = format!("/cnt.capb.{}", std::process::id());
    let ring = Named(
        Arena::create(
            &name,
            &ArenaConfig {
                rings: vec![RingSpec { stride: pcm.record_bytes(audio.block_ms), capacity: 1_000 }],
                ..ArenaConfig::default()
            },
        )
        .unwrap(),
    );
    let meta = StreamMeta {
        stream: StreamId(registered.stream_id),
        format: PayloadFormat::RawBytes,
        shape_name: pcm.shape_name(),
        anchor,
        residual_offset_ns: 0,
    };
    let mut stream = AudioStream::new(
        meta,
        ring.0.ring_writer(0).unwrap(),
        pcm.rate_hz,
        pcm.frames_per_block(audio.block_ms),
        audio.batch_seconds,
    )
    .unwrap();
    let block = vec![0u8; pcm.block_bytes(audio.block_ms) as usize];
    let start = anchor.mono_ns + 1_000_000_000;
    for i in 0..200u64 {
        let t = start + i * 10_000_000;
        stream.block(&block, t, t, &mut client).unwrap();
    }
    assert_eq!(stream.counts(), (2, 0));

    // Everything sent has reached the writer once a heartbeat comes back; then the writer commits.
    client.heartbeat(None).await.unwrap();
    core.sync_writer().await.unwrap();
    client.close("done").await.unwrap();
    stop.send(()).unwrap();
    task.await.unwrap();

    drop(core);
    // The broker's writer thread closes the catalog as it exits, just after the server task ends.
    let catalog = {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            match RedbCatalog::open(&catalog_path) {
                Ok(c) => break c,
                Err(e) if std::time::Instant::now() < deadline => {
                    let _ = e;
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Err(e) => panic!("the broker never closed its catalog: {e}"),
            }
        }
    };
    let id = StreamId(registered.stream_id);
    let info = catalog.stream(id).unwrap().unwrap();
    assert_eq!(info.name, "wave3.audio");
    assert_eq!(info.anchor.map(|a| a.mono_ns), Some(anchor.mono_ns), "the anchor it registered with");
    let page = catalog.range(id, Interval::new(0, u64::MAX).unwrap(), None, 10).unwrap();
    assert_eq!(page.items.len(), 2, "two one-second batches");
    let first = &page.items[0].header;
    assert_eq!(first.start_ns(), anchor.utc_of(start));
    assert_eq!(first.end_ns() - first.start_ns(), 1_000_000_000);
}

#[test]
fn the_example_manifest_parses() {
    let text =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../config/devices.example.toml")).unwrap();
    let m = Manifest::parse(&text).unwrap();
    assert_eq!(m.devices.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(), vec!["facecam", "wave3"]);
}

/// A broker config with a small Tier 0 (the profile's arena is gigabytes): 32 slots of 64 KiB.
fn small_tier0(mut config: Config) -> Config {
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 64 * 1024, count: 32 }]);
    config
}

/// Real registration and camera publication over more than the default raw retention window. Only two 4 MiB
/// slots and one reusable frame buffer are allocated; this tests overwrite flow, not the planner's memory budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_1080p_at_30fps_registers_and_runs_beyond_its_default_two_second_ring() {
    use continuum_capture::clock::Clocks;
    use continuum_capture::publish::{Sink, SinkError, VideoStream};
    use continuum_capture::register::video_spec;
    use continuum_capture::stamp::{V4L2_TIMESTAMP_MONOTONIC, V4L2_TSTAMP_SRC_SOE};
    use continuum_capture::video::{CameraConfig, CameraCounts, CameraExit, FrameSource, Mode, RawFrame, run_camera};
    use continuum_core::{BootId, HeaderBuf, StorageLocator};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    const FRAME_BYTES: usize = 1920 * 1080 * 2;
    const STEP_NS: u64 = 33_333_000; // whole microseconds, as the device timestamp API reports
    const FRAMES: u32 = 62;
    const UTC_BASE: u64 = 1_800_000_000_000_000_000;
    struct Clock(AtomicU64);
    impl Clocks for Clock {
        fn monotonic_ns(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
        fn utc_ns(&self) -> u64 {
            UTC_BASE + self.monotonic_ns()
        }
        fn continuous_ns(&self) -> u64 {
            self.monotonic_ns()
        }
        fn boot_id(&self) -> BootId {
            BootId([7; 16])
        }
    }
    struct Frames<'a> {
        clock: &'a Clock,
        stop: &'a AtomicBool,
        bytes: Vec<u8>,
        next: u32,
    }
    impl FrameSource for Frames<'_> {
        fn next_frame(&mut self, _: u32, frame: &mut dyn FnMut(&RawFrame<'_>)) -> std::io::Result<bool> {
            if self.next == FRAMES {
                self.stop.store(true, Ordering::SeqCst);
                return Ok(false);
            }
            let at = 2_000_000_000 + u64::from(self.next) * STEP_NS;
            self.clock.0.store(at + 5_000_000, Ordering::SeqCst);
            frame(&RawFrame {
                bytes: &self.bytes,
                tv_sec: (at / 1_000_000_000) as i64,
                tv_usec: (at % 1_000_000_000 / 1000) as i64,
                flags: V4L2_TIMESTAMP_MONOTONIC | V4L2_TSTAMP_SRC_SOE,
                sequence: self.next,
                duration_ns: 0,
            });
            self.next += 1;
            Ok(true)
        }
    }
    struct Committed {
        client: Arc<Client>,
        runtime: tokio::runtime::Handle,
    }
    impl Sink for Committed {
        fn ingest(&mut self, header: HeaderBuf, locator: Option<StorageLocator>) -> Result<(), SinkError> {
            // Commit each notice before reusing its tiny slot pool, so this test checks capture flow rather than
            // racing the asynchronous broker's stale-locator rejection with an unpaced synthetic producer.
            self.runtime
                .block_on(self.client.ingest_acked(&header, locator.as_ref()))
                .map(|_| ())
                .map_err(|error| SinkError(error.to_string()))
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let catalog_path = dir.path().join("catalog.redb");
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 4 * 1024 * 1024, count: 2 }]);
    config.tier0_ring_bytes = 0;
    let server = Server::bind_with_catalog(config, Box::new(RedbCatalog::open(&catalog_path).unwrap())).await.unwrap();
    let core = server.core();
    let (stop_server, stopped) = tokio::sync::oneshot::channel();
    let server_task = tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let manifest = Manifest::parse("[[device]]\nname='cam0'\nkind='camera'\nmatch={name='Synthetic camera'}\nvideo={format='UYVY',width=1920,height=1080,fps=30}\n").unwrap();
    let device = &manifest.devices[0];
    let video = device.video().unwrap();
    assert_eq!(video.ring_seconds, 2.0);
    let clock = Clock(AtomicU64::new(1_000_000_000));
    let anchor = take_anchor(&clock);
    let client = Arc::new(Client::connect(&socket, ClientOptions::new("raw-capture-test")).await.unwrap());
    let spec = video_spec(device, &video, &anchor, 0);
    assert_eq!((spec.shape_name.as_str(), spec.format, spec.nominal_rate_hz), ("UYVY", PayloadFormat::RawBytes, 30.0));
    let registered = client.register_stream(&spec).await.unwrap();
    let stream_id = StreamId(registered.stream_id);
    let arena = client.arenas()[0].arena.clone();
    let runtime = tokio::runtime::Handle::current();
    let producer = Arc::clone(&client);
    let (counts, published, offers) = tokio::task::spawn_blocking(move || {
        let mut stream = VideoStream::new(
            StreamMeta {
                stream: stream_id,
                format: PayloadFormat::RawBytes,
                shape_name: "UYVY".into(),
                anchor,
                residual_offset_ns: 0,
            },
            arena,
            0,
            1,
            None,
        );
        let stop = AtomicBool::new(false);
        let mut source = Frames { clock: &clock, stop: &stop, bytes: vec![128; FRAME_BYTES], next: 0 };
        let config = CameraConfig {
            mode: Mode { fourcc: *b"UYVY", width: 1920, height: 1080 },
            stride: 3840,
            frame_interval_ns: 33_333_333,
            flip: false,
            stall_ns: 1_000_000_000,
            slot_bytes: 4 * 1024 * 1024,
        };
        let mut counts = CameraCounts::default();
        let mut offers = 0;
        let exit = run_camera(
            &mut source,
            &mut stream,
            &mut Committed { client: producer, runtime },
            &clock,
            &config,
            &stop,
            &mut counts,
            &mut |_| {},
            &mut |_, _| offers += 1,
        );
        assert!(matches!(exit, CameraExit::Stopped), "raw publication must keep running: {exit:?}");
        (counts, stream.published(), offers)
    })
    .await
    .unwrap();
    assert_eq!(
        (counts.frames, counts.arena_full_dropped, counts.oversize, published, offers),
        (u64::from(FRAMES), 0, 0, u64::from(FRAMES), FRAMES)
    );
    assert!(u64::from(FRAMES - 1) * STEP_NS > 2_000_000_000);
    client.heartbeat(None).await.unwrap();
    core.sync_writer().await.unwrap();
    client.close("done").await.unwrap();
    drop(client);
    stop_server.send(()).unwrap();
    server_task.await.unwrap();
    drop(core);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let catalog = loop {
        match RedbCatalog::open(&catalog_path) {
            Ok(catalog) => break catalog,
            Err(_) if std::time::Instant::now() < deadline => tokio::time::sleep(Duration::from_millis(20)).await,
            Err(error) => panic!("broker did not release the temporary catalog: {error}"),
        }
    };
    let info = catalog.stream(stream_id).unwrap().unwrap();
    assert_eq!((info.name.as_str(), info.shape_name.as_str()), ("cam0.video", "UYVY"));
    let page = catalog.range(stream_id, Interval::new(0, u64::MAX).unwrap(), None, 100).unwrap();
    assert_eq!(page.items.len(), FRAMES as usize);
    for (seq, item) in page.items.iter().enumerate() {
        let capture = item.header.header().capture().unwrap();
        assert_eq!(
            (capture.device_seq(), capture.width(), capture.height(), capture.stride()),
            (seq as u64, 1920, 1080, 3840)
        );
        assert_eq!(item.header.start_ns(), UTC_BASE + 2_000_000_000 + seq as u64 * STEP_NS);
    }
}
