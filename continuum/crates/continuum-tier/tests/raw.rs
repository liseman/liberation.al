//! A raw camera's Tier 1: UYVY frames through the worker loop come out as H.264, one access unit per record
//! in `H264Au` chunks that each start on a keyframe, and decode back to the pictures that went in. A worker killed
//! mid-GOP leaves a chunk recovery reads back whole. Short: 3 seconds of 320x240 at 30 fps.
#![cfg(feature = "ffmpeg")]

use std::path::{Path, PathBuf};

use continuum_core::{DomainLayer, PayloadFormat, StreamId, Uuid};
use continuum_tier::raw::{RawLayout, to_nv12};
use continuum_tier::video::decode_gop;
use continuum_tier::{
    ChunkFormat, ChunkReader, Closed, CommitFailure, FlushSettings, Outputs, Payload, StreamShape, WorkerLoop, gop,
};

const W: u32 = 320;
const H: u32 = 240;
const FRAME_NS: u64 = 33_333_333;
const T0: u64 = 1_700_000_000_000_000_000;
const CAM: StreamId = StreamId(4);

/// Keep independent GPU tests from exhausting native encoder sessions. The concurrent-open regression below
/// deliberately creates its own bounded set of threads while holding this same test-level lock.
fn gpu() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Both close orders must leave the test process usable when several cameras overlap.
#[test]
fn nvenc_sessions_can_close_in_creation_or_reverse_order() {
    let _gpu = gpu();
    for order in ["creation", "reverse", "replacement"] {
        for close in ["finish", "drop"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["nvenc_close_order_child", "--exact", "--ignored", "--nocapture"])
                .env("CONTINUUM_TEST_NVENC_CLOSE_ORDER", order)
                .env("CONTINUUM_TEST_NVENC_CLOSE", close)
                .output()
                .unwrap();
            assert!(output.status.success(), "{order}/{close}: {output:?}");
        }
    }
}

/// Run each close order in a fresh process so earlier tests cannot change the session order.
#[test]
#[ignore = "subprocess fixture for nvenc_sessions_can_close_in_creation_or_reverse_order"]
fn nvenc_close_order_child() {
    use continuum_tier::raw::RawEncoder;

    let order = std::env::var("CONTINUUM_TEST_NVENC_CLOSE_ORDER").expect("run through the parent test");
    let close = std::env::var("CONTINUUM_TEST_NVENC_CLOSE").expect("run through the parent test");
    let Ok(first) = RawEncoder::open(layout(), 30, &["h264_nvenc"]) else {
        eprintln!("NVENC is unavailable on this machine");
        return;
    };
    let second = RawEncoder::open(layout(), 30, &["h264_nvenc"]).unwrap();
    let third = RawEncoder::open(layout(), 30, &["h264_nvenc"]).unwrap();
    let mut sessions = vec![first, second, third];
    if order == "reverse" {
        sessions.reverse();
    }
    let mut remaining = std::collections::VecDeque::from(sessions);
    let mut replaced = false;
    while let Some(mut session) = remaining.pop_front() {
        let units = session.encode(&uyvy(0), T0, T0).unwrap();
        if close == "finish" {
            let mut units = units;
            units.extend(session.finish().unwrap());
            assert_eq!(units.len(), 1);
        } else {
            drop(session);
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
        if order == "replacement" && !replaced {
            remaining.push_back(RawEncoder::open(layout(), 30, &["h264_nvenc"]).unwrap());
            replaced = true;
        }
    }
    // A later, non-overlapping session starts a new lifecycle and still returns every frame.
    let mut next = RawEncoder::open(layout(), 30, &["h264_nvenc"]).unwrap();
    let mut units = next.encode(&uyvy(1), T0 + FRAME_NS, T0 + FRAME_NS).unwrap();
    units.extend(next.finish().unwrap());
    assert_eq!(units.len(), 1);
}

#[test]
fn concurrent_codec_initialization_keeps_frames_and_times() {
    use continuum_tier::h264::H264_ENCODERS;
    use continuum_tier::raw::RawEncoder;
    use std::sync::Barrier;

    let _gpu = gpu();
    let ready = Barrier::new(4);
    std::thread::scope(|scope| {
        for n in 0..4u32 {
            let ready = &ready;
            scope.spawn(move || {
                ready.wait();
                let mut encoder = RawEncoder::open(layout(), 30, H264_ENCODERS).unwrap();
                let at = T0 + u64::from(n) * FRAME_NS;
                let mut units = encoder.encode(&uyvy(n), at, at).unwrap();
                units.extend(encoder.finish().unwrap());
                let units: Vec<gop::Unit> = units
                    .into_iter()
                    .map(|u| gop::Unit { start_ns: u.start_ns, end_ns: u.end_ns, key: u.key, bytes: u.bytes })
                    .collect();
                let decoded = decode_gop(&units).unwrap();
                assert_eq!(decoded.len(), 1);
                assert_eq!((decoded[0].width, decoded[0].height, decoded[0].start_ns), (W, H, at));
                let wanted = to_nv12(&layout(), &uyvy(n)).unwrap();
                let luma = (W * H) as usize;
                assert!(psnr(&wanted[..luma], &decoded[0].nv12[..luma]) > 30.0);
            });
        }
    });
}

fn uuid(n: u32) -> Uuid {
    let mut b = [0u8; 16];
    b[0] = 0x01;
    b[6] = 0x70;
    b[8] = 0x80;
    b[12..16].copy_from_slice(&n.to_be_bytes());
    Uuid(b)
}

/// Frame `i`: a gradient panning two pixels a frame, with a bright square drifting across, limited range.
fn uyvy(i: u32) -> Vec<u8> {
    let mut out = vec![0u8; (W * 2 * H) as usize];
    let sx = (i * 3) % (W - 40);
    for y in 0..H {
        for x in (0..W).step_by(2) {
            let luma = |x: u32| -> u8 {
                if (sx..sx + 40).contains(&x) && (100..140).contains(&y) {
                    220
                } else {
                    u8::try_from(32 + ((x + 2 * i) % 160) / 2 + y / 4).unwrap()
                }
            };
            let at = ((y * W * 2) + x * 2) as usize;
            out[at..at + 4].copy_from_slice(&[128, luma(x), 128, luma(x + 1)]);
        }
    }
    out
}

fn layout() -> RawLayout {
    RawLayout::of("UYVY", W, H, 0).unwrap()
}

fn shape() -> StreamShape {
    StreamShape {
        chunk_format: ChunkFormat::Lz4, // as the service's shape_of gives a raw stream; declare upgrades it
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::RawBytes,
        shape_name: "UYVY".to_owned(),
        shape_version: 1,
        pcm: None,
        raw: Some(layout()),
    }
}

/// Collects closed chunks; every commit succeeds.
#[derive(Default)]
struct Kept {
    chunks: Vec<PathBuf>,
}

impl Outputs for Kept {
    fn flushed(&mut self, _ids: &[Uuid]) -> Result<(), String> {
        Ok(())
    }

    fn commit(&mut self, closed: &Closed) -> Result<(), CommitFailure> {
        self.chunks.push(closed.path.clone());
        Ok(())
    }
}

/// Feeds frames `0..n` with the clock at each frame's time, ticking after each.
fn feed(w: &mut WorkerLoop<Kept>, n: u32) {
    for i in 0..n {
        let t = T0 + u64::from(i) * FRAME_NS;
        let p = Payload {
            pcm_rate_hz: 0,
            id: uuid(i),
            stream: CAM,
            start_ns: t,
            end_ns: t,
            seq: i,
            bytes: uyvy(i),
            provenance: None,
        };
        w.payload(&p, t).expect("append");
        w.tick(t).expect("tick");
    }
}

/// A chunk's records as gop units, in order.
fn units(path: &Path) -> (ChunkFormat, Vec<(Uuid, gop::Unit)>) {
    let mut r = ChunkReader::open(path).unwrap();
    let ids: Vec<Uuid> = r.records().iter().map(|rec| rec.id).collect();
    let out = (0..ids.len())
        .map(|i| {
            let mut u = gop::decode(&r.read(i).unwrap()).unwrap();
            assert_eq!(u.len(), 1, "one access unit per record");
            (ids[i], u.remove(0))
        })
        .collect();
    (r.format(), out)
}

/// The mean squared error of two luma planes, as PSNR in dB.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0f64.powi(2) / mse.max(1e-9)).log10()
}

#[test]
fn raw_frames_become_h264_chunks_that_start_on_keyframes_and_decode_back() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let settings = FlushSettings {
        chunk_ns: 1_000_000_000,
        part_ns: 1_000_000_000,
        audio_chunk_ns: 1_000_000_000,
        sync_ns: 100_000_000,
    };
    let mut w = WorkerLoop::new(dir.path(), settings, Kept::default()).unwrap();
    w.declare(CAM, shape());
    feed(&mut w, 90);
    w.shutdown().unwrap();
    assert_eq!(w.counts().appended, 90, "every frame appended, none held back");

    let chunks = w.outputs().chunks.clone();
    assert!(chunks.len() >= 2, "three seconds rotate at least once: {}", chunks.len());
    let mut all = vec![];
    for path in &chunks {
        let (format, us) = units(path);
        assert_eq!(format, ChunkFormat::H264Au);
        assert!(us[0].1.key, "{} starts on a keyframe", path.display());
        // Each chunk decodes on its own.
        let frames = decode_gop(&us.iter().map(|(_, u)| u.clone()).collect::<Vec<_>>()).unwrap();
        assert_eq!(frames.len(), us.len());
        all.extend(us.into_iter().map(|(id, u)| (id, u.start_ns)).zip(frames));
    }
    assert_eq!(all.len(), 90);
    for (i, ((id, start), frame)) in all.iter().enumerate() {
        let i = u32::try_from(i).unwrap();
        assert_eq!((*id, *start, frame.start_ns), (uuid(i), T0 + u64::from(i) * FRAME_NS, *start), "frame {i}");
        let want = to_nv12(&layout(), &uyvy(i)).unwrap();
        let luma = (W * H) as usize;
        let db = psnr(&want[..luma], &frame.nv12[..luma]);
        assert!(db > 30.0, "frame {i} decodes to {db:.1} dB of the input");
    }
}

#[test]
fn a_worker_killed_mid_gop_leaves_a_chunk_recovery_reads_whole() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let settings = FlushSettings {
        chunk_ns: 10_000_000_000,
        part_ns: 10_000_000_000,
        audio_chunk_ns: 10_000_000_000,
        sync_ns: 100_000_000,
    };
    let mut w = WorkerLoop::new(dir.path(), settings, Kept::default()).unwrap();
    w.declare(CAM, shape());
    feed(&mut w, 45); // a GOP and a half
    let appended = w.counts().appended;
    drop(w); // killed: no shutdown, the chunk is still open

    let found = continuum_tier::recover(dir.path()).unwrap();
    assert_eq!(found.len(), 1, "one open chunk");
    let rec = found.into_iter().next().unwrap();
    assert_eq!(rec.format, ChunkFormat::H264Au);
    assert_eq!(rec.records.len() as u64, appended, "every appended frame is in it, with its own id");
    assert_eq!(
        rec.records.iter().map(|r| r.id).collect::<Vec<_>>(),
        (0..45).map(uuid).take(rec.records.len()).collect::<Vec<_>>()
    );
    let (_, us) = units(&rec.path);
    assert!(us[0].1.key);
    let frames = decode_gop(&us.into_iter().map(|(_, u)| u).collect::<Vec<_>>()).unwrap();
    assert_eq!(frames.len() as u64, appended);
}

/// Each encoder this machine has (NVENC and a software encoder on the Linux machine; VideoToolbox on the Mac): a keyframe first
/// and then every second exactly, every frame back in order with its own times, decoding close to the input.
#[test]
fn each_h264_encoder_here_keys_every_second_and_decodes_back() {
    let _gpu = gpu();
    use continuum_tier::raw::RawEncoder;
    let mut ran = 0;
    for name in continuum_tier::h264::H264_ENCODERS {
        let Ok(mut enc) = RawEncoder::open(layout(), 30, &[*name]) else {
            eprintln!("{name}: not here");
            continue;
        };
        ran += 1;
        let mut aus = vec![];
        for i in 0..60u32 {
            let t = T0 + u64::from(i) * FRAME_NS;
            aus.extend(enc.encode(&uyvy(i), t, t).unwrap());
        }
        aus.extend(enc.finish().unwrap());
        assert_eq!(aus.len(), 60, "{name}");
        let keys: Vec<usize> = aus.iter().enumerate().filter(|(_, a)| a.key).map(|(i, _)| i).collect();
        assert_eq!(keys, vec![0, 30], "{name}: a keyframe each second");
        assert!(aus.iter().enumerate().all(|(i, a)| a.start_ns == T0 + i as u64 * FRAME_NS), "{name}: in order");
        let units: Vec<gop::Unit> = aus
            .into_iter()
            .map(|a| gop::Unit { start_ns: a.start_ns, end_ns: a.end_ns, key: a.key, bytes: a.bytes })
            .collect();
        let frames = decode_gop(&units).unwrap();
        let luma = (W * H) as usize;
        let db = psnr(&to_nv12(&layout(), &uyvy(59)).unwrap()[..luma], &frames[59].nv12[..luma]);
        assert!(db > 30.0, "{name}: the last frame decodes to {db:.1} dB");
    }
    assert!(ran > 0, "at least a software encoder opens");
}

#[test]
fn resized_raw_frames_reopen_the_encoder_without_losing_time_or_pixels() {
    let _gpu = gpu();
    use continuum_tier::CameraFrameInfo;
    let dir = tempfile::tempdir().unwrap();
    let mut worker = WorkerLoop::new(dir.path(), FlushSettings::default(), Kept::default()).unwrap();
    worker.declare(CAM, shape());
    let frames = [(320, 240, 10), (640, 480, 11), (160, 120, 12), (160, 120, 0)];
    for (i, (width, height, device_seq)) in frames.into_iter().enumerate() {
        let layout = RawLayout::of("UYVY", width, height, 0).unwrap();
        let mut source = shape();
        source.raw = Some(layout);
        let t = T0 + u64::try_from(i).unwrap() * FRAME_NS;
        let luma = 64 + u8::try_from(i).unwrap() * 32;
        let bytes = [128, luma, 128, luma].repeat(layout.frame_bytes() / 4);
        let p = Payload {
            pcm_rate_hz: 0,
            id: uuid(u32::try_from(i).unwrap()),
            stream: CAM,
            start_ns: t,
            end_ns: t,
            seq: u32::try_from(i).unwrap(),
            bytes,
            provenance: None,
        };
        let info = CameraFrameInfo { width, height, stride: layout.stride, flip: false, device_seq };
        worker.video_payload(&p, source, info, t).unwrap();
    }
    worker.shutdown().unwrap();
    let chunks = &worker.outputs_mut().chunks;
    assert_eq!(chunks.len(), 4, "growth, shrink, and a same-size rate restart each start a chunk");
    for (i, path) in chunks.iter().enumerate() {
        let (format, records) = units(path);
        assert_eq!(format, ChunkFormat::H264Au);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, uuid(u32::try_from(i).unwrap()));
        assert!(records[0].1.key);
        let decoded = decode_gop(&[records[0].1.clone()]).unwrap();
        assert_eq!(decoded.len(), 1);
        let (width, height, _) = frames[i];
        assert_eq!((decoded[0].width, decoded[0].height), (width, height));
        assert_eq!(decoded[0].start_ns, T0 + u64::try_from(i).unwrap() * FRAME_NS);
        let expected = 64 + u8::try_from(i).unwrap() * 32;
        assert!(decoded[0].nv12[0].abs_diff(expected) <= 3);
    }
}

#[test]
fn a_raw_pixel_format_change_is_an_encoder_boundary() {
    let _gpu = gpu();
    use continuum_tier::CameraFrameInfo;
    let dir = tempfile::tempdir().unwrap();
    let mut worker = WorkerLoop::new(dir.path(), FlushSettings::default(), Kept::default()).unwrap();
    worker.declare(CAM, shape());
    for (i, name) in ["UYVY", "NV12"].into_iter().enumerate() {
        let raw = RawLayout::of(name, W, H, 0).unwrap();
        let mut source = shape();
        source.shape_name = name.to_owned();
        source.raw = Some(raw);
        let bytes = if i == 0 {
            [128, 96, 128, 96].repeat(raw.frame_bytes() / 4)
        } else {
            let mut b = vec![160; (W * H) as usize];
            b.extend(vec![128; (W * H / 2) as usize]);
            b
        };
        let at = T0 + u64::try_from(i).unwrap() * FRAME_NS;
        let p = Payload {
            id: uuid(u32::try_from(i).unwrap()),
            stream: CAM,
            start_ns: at,
            end_ns: at,
            seq: u32::try_from(i).unwrap(),
            bytes,
            pcm_rate_hz: 0,
            provenance: None,
        };
        worker
            .video_payload(
                &p,
                source,
                CameraFrameInfo {
                    width: W,
                    height: H,
                    stride: raw.stride,
                    flip: false,
                    device_seq: u64::try_from(i).unwrap(),
                },
                at,
            )
            .unwrap();
    }
    worker.shutdown().unwrap();
    let chunks = &worker.outputs_mut().chunks;
    assert_eq!(chunks.len(), 2);
    for (i, path) in chunks.iter().enumerate() {
        let (_, records) = units(path);
        let decoded = decode_gop(&[records[0].1.clone()]).unwrap();
        assert_eq!((decoded[0].width, decoded[0].height), (W, H));
        assert!(decoded[0].nv12[0].abs_diff(if i == 0 { 96 } else { 160 }) <= 3);
    }
}
