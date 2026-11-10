//! Reading a raw camera's Tier 1 (`H264Au`, one access unit per record) as video: a range that starts mid-GOP
//! decodes from its keyframe, a purge's tombstone skips the rest of its GOP, and export copies from the keyframe. The chunk is
//! made the way the tier worker makes it: 3 seconds of 320x240 UYVY at 30 fps through the worker loop. The opt-in
//! AV1 archive pass turns the same chunk into AV1 and it reads back the same way.
#![cfg(feature = "ffmpeg")]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Mutex, MutexGuard, PoisonError};

use continuum_core::{DomainLayer, PayloadFormat, StreamId, Uuid};
use continuum_tier::av1::{Av1Settings, Encoded, encode_chunk};
use continuum_tier::export::{ExportCodec, export};
use continuum_tier::raw::{RawLayout, to_nv12};
use continuum_tier::service::purge_file;
use continuum_tier::tier2::read_frames;
use continuum_tier::{
    ChunkFormat, ChunkReader, Closed, CommitFailure, FlushSettings, Outputs, Payload, StreamShape, WorkerLoop, gop,
};

const W: u32 = 320;
const H: u32 = 240;
const FRAME_NS: u64 = 33_333_333;
const T0: u64 = 1_700_000_000_000_000_000;
const CAM: StreamId = StreamId(4);
const FRAMES: u32 = 90;

/// One encoder session at a time: concurrent NVENC sessions in one process can crash it.
fn gpu() -> MutexGuard<'static, ()> {
    static GPU: Mutex<()> = Mutex::new(());
    GPU.lock().unwrap_or_else(PoisonError::into_inner)
}

fn uuid(n: u32) -> Uuid {
    let mut b = [0u8; 16];
    b[0] = 0x01;
    b[6] = 0x70;
    b[8] = 0x80;
    b[12..16].copy_from_slice(&n.to_be_bytes());
    Uuid(b)
}

fn at(i: u32) -> u64 {
    T0 + u64::from(i) * FRAME_NS
}

/// Frame `i`: a gradient panning two pixels a frame with a bright square drifting across, so frames differ.
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
            let o = ((y * W * 2) + x * 2) as usize;
            out[o..o + 4].copy_from_slice(&[128, luma(x), 128, luma(x + 1)]);
        }
    }
    out
}

fn layout() -> RawLayout {
    RawLayout::of("UYVY", W, H, 0).unwrap()
}

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

/// One `H264Au` chunk of frames `0..FRAMES` (chunks long enough not to rotate), and the indexes of its keyframes.
fn chunk(dir: &Path) -> (PathBuf, Vec<u32>) {
    let settings = FlushSettings { chunk_ns: 60_000_000_000, audio_chunk_ns: 60_000_000_000, sync_ns: 100_000_000 };
    let mut w = WorkerLoop::new(dir, settings, Kept::default()).unwrap();
    let shape = StreamShape {
        chunk_format: ChunkFormat::Lz4,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::RawBytes,
        shape_name: "UYVY".to_owned(),
        shape_version: 1,
        pcm: None,
        raw: Some(layout()),
    };
    w.declare(CAM, shape);
    for i in 0..FRAMES {
        let p = Payload {
            pcm_rate_hz: 0,
            id: uuid(i),
            stream: CAM,
            start_ns: at(i),
            end_ns: at(i),
            seq: i,
            bytes: uyvy(i),
            provenance: None,
        };
        w.payload(&p, at(i)).unwrap();
        w.tick(at(i)).unwrap();
    }
    w.shutdown().unwrap();
    let [path] = w.outputs().chunks.clone().try_into().expect("one chunk");
    let mut r = ChunkReader::open(&path).unwrap();
    assert_eq!(r.format(), ChunkFormat::H264Au);
    let keys = (0..FRAMES).filter(|&i| gop::decode(&r.read(i as usize).unwrap()).unwrap()[0].key).collect::<Vec<_>>();
    (path, keys)
}

/// The PSNR of a decoded frame's luma against frame `i`'s.
fn psnr(i: u32, nv12: &[u8]) -> f64 {
    let want = to_nv12(&layout(), &uyvy(i)).unwrap();
    let luma = (W * H) as usize;
    let mse = want[..luma].iter().zip(&nv12[..luma]).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum::<f64>()
        / luma as f64;
    10.0 * (255.0f64.powi(2) / mse.max(1e-9)).log10()
}

fn index_of(start_ns: u64) -> u32 {
    u32::try_from((start_ns - T0) / FRAME_NS).unwrap()
}

#[test]
fn a_range_starting_mid_gop_decodes_from_its_keyframe() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let (path, keys) = chunk(dir.path());
    assert_eq!(keys, [0, 30, 60], "a keyframe every second");

    let frames = read_frames(&path, at(40), at(50)).unwrap();
    let got: Vec<u32> = frames.iter().map(|f| index_of(f.start_ns)).collect();
    assert_eq!(got, (40..50).collect::<Vec<_>>(), "exactly the range's frames");
    for f in &frames {
        let i = index_of(f.start_ns);
        assert!(psnr(i, &f.nv12) > 30.0, "frame {i} decodes to the picture that went in");
    }
}

#[test]
fn a_tombstone_skips_the_rest_of_its_gop() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = chunk(dir.path());
    purge_file(&path, &[], &BTreeSet::from([uuid(35)]), false).unwrap();

    let got: Vec<u32> = read_frames(&path, at(0), at(FRAMES)).unwrap().iter().map(|f| index_of(f.start_ns)).collect();
    let want: Vec<u32> = (0..35).chain(60..FRAMES).collect();
    assert_eq!(got, want, "36 to 59 can't decode without 35; the next keyframe resumes");
}

#[test]
fn export_copies_from_the_keyframe_before_the_range_to_its_end() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = chunk(dir.path());
    let out = dir.path().join("out.mp4");

    let copied = export(&[path.as_path()], &out, at(40), at(65), ExportCodec::Copy).unwrap();
    assert_eq!((copied.frames, copied.size), (35, (W, H)), "from keyframe 30 to frame 64");
    let exact = export(&[path.as_path()], &out, at(40), at(65), ExportCodec::H264).unwrap();
    assert_eq!(exact.frames, 25, "a re-encode is exactly the range");
}

// The opt-in AV1 archive pass: small, fast encoder settings, since the tests check the container, not quality.
const FAST: Av1Settings = Av1Settings { preset: 12, crf: 40, parallelism: 1 };

fn ready(path: &Path) -> continuum_tier::av1::Ready {
    match encode_chunk(path, FAST, &AtomicBool::new(false)).unwrap() {
        Encoded::Ready(ready) => ready,
        Encoded::Kept(why) => panic!("kept: {why}"),
    }
}

fn kept(path: &Path, stop: bool) -> String {
    match encode_chunk(path, FAST, &AtomicBool::new(stop)).unwrap() {
        Encoded::Ready(_) => panic!("converted"),
        Encoded::Kept(why) => why,
    }
}

#[test]
fn the_archive_pass_keeps_every_record_and_reads_back_as_the_same_frames() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = chunk(dir.path());
    let identity = |r: &continuum_tier::Record| (r.id, r.start_ns, r.end_ns, r.seq, r.cut);
    let before: Vec<_> = ChunkReader::open(&path).unwrap().records().iter().map(identity).collect();
    let before_len = std::fs::metadata(&path).unwrap().len();

    let rewritten = ready(&path).swap(&path).unwrap().expect("the file didn't change");

    let mut r = ChunkReader::open(&path).unwrap();
    assert_eq!(r.format(), ChunkFormat::Av1Au);
    assert_eq!(r.records().iter().map(identity).collect::<Vec<_>>(), before, "ids, times, and sequence kept");
    assert_eq!(rewritten.length, std::fs::metadata(&path).unwrap().len());
    assert!(rewritten.length < before_len, "{} bytes from {before_len}", rewritten.length);
    let keys = (0..FRAMES).filter(|&i| gop::decode(&r.read(i as usize).unwrap()).unwrap()[0].key).collect::<Vec<_>>();
    assert_eq!(keys, [0], "the chunk is one GOP, opening on its keyframe");
    let frames = read_frames(&path, at(0), at(FRAMES)).unwrap();
    assert_eq!(frames.iter().map(|f| index_of(f.start_ns)).collect::<Vec<_>>(), (0..FRAMES).collect::<Vec<_>>());
    for f in &frames {
        let i = index_of(f.start_ns);
        assert!(psnr(i, &f.nv12) > 28.0, "frame {i} decodes to about the picture that went in");
    }
    let mid = read_frames(&path, at(40), at(50)).unwrap();
    assert_eq!(mid.iter().map(|f| index_of(f.start_ns)).collect::<Vec<_>>(), (40..50).collect::<Vec<_>>());
    let out = dir.path().join("out.mp4");
    let auto = export(&[path.as_path()], &out, at(40), at(65), ExportCodec::Auto).unwrap();
    assert_eq!(auto.frames, 25, "AV1 exports as H.264 for exactly the range");
    assert!(export(&[path.as_path()], &out, at(40), at(65), ExportCodec::Copy).is_err(), "copy is H.264 only");
    assert!(kept(&path, false).contains("Av1Au"), "an AV1 chunk isn't converted again");
}

#[test]
fn the_archive_pass_leaves_a_purged_chunk_as_h264() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = chunk(dir.path());
    purge_file(&path, &[], &BTreeSet::from([uuid(35)]), false).unwrap();
    assert_eq!(kept(&path, false), "purged frames");
    assert_eq!(ChunkReader::open(&path).unwrap().format(), ChunkFormat::H264Au);
}

#[test]
fn the_archive_pass_never_swaps_over_a_file_that_changed_after_it_was_read() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = chunk(dir.path());
    let pending = ready(&path);
    purge_file(&path, &[], &BTreeSet::from([uuid(80)]), false).unwrap(); // a purge rewrote it meanwhile

    assert!(pending.swap(&path).unwrap().is_none());
    let r = ChunkReader::open(&path).unwrap();
    assert_eq!((r.format(), r.records()[80].cut), (ChunkFormat::H264Au, true), "the purge's file stays");
}

#[test]
fn a_stopped_archive_pass_leaves_the_chunk_alone() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = chunk(dir.path());
    assert_eq!(kept(&path, true), "stopped");
    assert_eq!(ChunkReader::open(&path).unwrap().format(), ChunkFormat::H264Au);
}
