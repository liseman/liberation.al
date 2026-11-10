//! Exporting chunks as an ordinary MP4, then reading the file back with FFmpeg's own demuxer and decoder, as a
//! player or ffprobe would. Bounded: two seconds of small synthetic frames.
#![cfg(feature = "ffmpeg")]

use std::path::{Path, PathBuf};

use continuum_core::StreamId;
use continuum_tier::Picture;
use continuum_tier::chunk::{ChunkFormat, ChunkWriter};
use continuum_tier::export::{ExportCodec, export};
use continuum_tier::gop::{self, Unit};
use continuum_tier::h264::{H264Encoder, H264Settings};
use ffmpeg_next as ff;

const W: u32 = 320;
const H: u32 = 240;
const T0: u64 = 1_793_227_877_000_000_000;
const STEP: u64 = 66_666_667; // 15 fps
const STREAM: StreamId = StreamId(7);

fn picture(n: u64) -> Picture {
    let (w, h) = (W as usize, H as usize);
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let v = ((x + y) / 2 + n as usize * 4) % 256;
            rgb.extend_from_slice(&[v as u8, (v / 2) as u8, 255 - v as u8]);
        }
    }
    Picture { width: W, height: H, rgb }
}

/// One MJPEG Tier 1 chunk of 30 frames.
fn tier1(dir: &Path) -> PathBuf {
    let path = dir.join("t1.cnk");
    let mut w = ChunkWriter::create(&path, STREAM, ChunkFormat::Passthrough, T0).unwrap();
    for n in 0..30 {
        let mut jpeg = Vec::new();
        let p = picture(n);
        jpeg_encoder::Encoder::new(&mut jpeg, 90)
            .encode(&p.rgb, W as u16, H as u16, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        w.append(T0 + n * STEP, T0 + (n + 1) * STEP, n as u32, &jpeg).unwrap();
    }
    w.close().unwrap();
    path
}

/// One encoding test at a time: a GeForce card takes only a few NVENC sessions at once, and NVIDIA's library has
/// crashed (in its own thread) when the tests opened more than that together.
fn gpu() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `frames` (start, end, picture) as an H.264 Tier 1 chunk at `path`, one access unit per record as a raw camera
/// writes it ([`ChunkFormat::H264Au`]), a keyframe every `gop` frames.
fn au_chunk(path: &Path, stream: StreamId, frames: &[(u64, u64, Picture)], gop: u32) -> PathBuf {
    let (width, height) = (frames[0].2.width, frames[0].2.height);
    let settings = H264Settings { gop_frames: gop, frame_rate: 15, bit_rate: 400_000, ..H264Settings::default() };
    let mut encoder = H264Encoder::open(width, height, settings).unwrap();
    let mut writer = ChunkWriter::create(path, stream, ChunkFormat::H264Au, frames[0].0).unwrap();
    let mut seq = 0;
    let mut write = |units: Vec<continuum_tier::h264::AccessUnit>| {
        for au in units {
            let unit = Unit { start_ns: au.start_ns, end_ns: au.end_ns, key: au.key, bytes: au.bytes };
            writer.append(unit.start_ns, unit.end_ns, seq, &gop::encode(&[unit]).unwrap()).unwrap();
            seq += 1;
        }
    };
    for (start, end, picture) in frames {
        write(encoder.encode(&continuum_tier::rgb_to_nv12(picture), *start, *end).unwrap());
    }
    write(encoder.finish().unwrap());
    writer.close().unwrap();
    path.to_path_buf()
}

/// The same 30 frames as [`tier1`] as an H.264 chunk, two GOPs of 15, starting `offset` after `T0`.
fn h264_at(dir: &Path, name: &str, stream: StreamId, offset: u64) -> PathBuf {
    let frames: Vec<_> = (0..30).map(|n| (T0 + offset + n * STEP, T0 + offset + (n + 1) * STEP, picture(n))).collect();
    au_chunk(&dir.join(name), stream, &frames, 15)
}

/// The same frames as [`tier1`] as an H.264 chunk, two GOPs of 15.
fn h264(dir: &Path) -> PathBuf {
    h264_at(dir, "h264.cnk", STREAM, 0)
}

/// What a player sees in the file: the codec, the size, and how many frames decode.
fn probe(path: &Path) -> (ff::codec::Id, (u32, u32), usize) {
    // The export initialized FFmpeg; do not rewrite ffmpeg-next's global error table here.
    let mut input = ff::format::input(path).unwrap();
    let stream = input.streams().best(ff::media::Type::Video).unwrap();
    let index = stream.index();
    let params = stream.parameters();
    let id = params.id();
    let mut decoder = ff::codec::context::Context::from_parameters(params).unwrap().decoder().video().unwrap();
    let size = (decoder.width(), decoder.height());
    let mut frames = 0;
    let mut picture = ff::frame::Video::empty();
    for (s, packet) in input.packets() {
        if s.index() == index {
            decoder.send_packet(&packet).unwrap();
            while decoder.receive_frame(&mut picture).is_ok() {
                frames += 1;
            }
        }
    }
    decoder.send_eof().unwrap();
    while decoder.receive_frame(&mut picture).is_ok() {
        frames += 1;
    }
    (id, size, frames)
}

#[test]
fn an_h264_chunk_copies_into_a_playable_mp4() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let chunk = h264(dir.path());
    let out = dir.path().join("all.mp4");
    let done = export(&[chunk.as_path()], &out, T0, T0 + 30 * STEP, ExportCodec::Copy).unwrap();
    assert_eq!((done.frames, done.size), (30, (W, H)));
    assert_eq!(probe(&out), (ff::codec::Id::H264, (W, H), 30));
}

#[test]
fn a_copy_starts_the_range_at_its_keyframe() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let chunk = h264(dir.path());
    let out = dir.path().join("slice.mp4");
    // Frames 3 to 7 are in the first GOP, so the copy starts at its keyframe, frame 0, and ends with frame 7.
    let done = export(&[chunk.as_path()], &out, T0 + 3 * STEP, T0 + 8 * STEP, ExportCodec::Copy).unwrap();
    assert_eq!(done.frames, 8);
    assert_eq!(probe(&out).2, 8);
}

#[test]
fn h264_export_is_exact_and_works_from_mjpeg_and_from_h264() {
    let _gpu = gpu();
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    for (chunk, dir) in [(tier1(a.path()), a.path()), (h264(b.path()), b.path())] {
        let out = dir.join("slice.mp4");
        let done = export(&[chunk.as_path()], &out, T0 + 3 * STEP, T0 + 8 * STEP, ExportCodec::H264).unwrap();
        assert_eq!(done.frames, 5, "{}", chunk.display());
        assert_eq!(probe(&out), (ff::codec::Id::H264, (W, H), 5), "{}", chunk.display());
    }
}

#[test]
fn the_wrong_requests_are_refused() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let mjpeg = tier1(dir.path());
    let out = dir.path().join("x.mp4");
    // A copy needs H.264 chunks; a range must be non-empty and hold something.
    assert!(export(&[mjpeg.as_path()], &out, T0, T0 + STEP, ExportCodec::Copy).is_err());
    assert!(export(&[mjpeg.as_path()], &out, T0 + STEP, T0 + STEP, ExportCodec::H264).is_err());
    assert!(export(&[mjpeg.as_path()], &out, T0 + 100 * STEP, T0 + 200 * STEP, ExportCodec::H264).is_err());
}

#[test]
fn the_command_writes_a_clip_from_h264_by_default_as_a_copy() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let chunk = h264(dir.path());
    let out = dir.path().join("cli.mp4");
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_continuum-export"))
        .arg("--out")
        .arg(&out)
        .arg(&chunk)
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert!(String::from_utf8_lossy(&run.stdout).contains("30 frames of 320x240 h264"));
    assert_eq!(probe(&out), (ff::codec::Id::H264, (W, H), 30));
}

#[test]
fn a_purged_gop_is_left_out_of_a_copy() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let chunk = h264(dir.path());
    // Cutting the first keyframe leaves the rest of its GOP undecodable, so the copy skips to the next keyframe.
    continuum_tier::chunk::rewrite_without(&chunk, &[0]).unwrap();
    let out = dir.path().join("rest.mp4");
    let done = export(&[chunk.as_path()], &out, T0, T0 + 30 * STEP, ExportCodec::Copy).unwrap();
    assert_eq!(done.frames, 15);
    assert_eq!(probe(&out).2, 15);
}

#[test]
fn chunks_out_of_order_or_from_two_streams_are_refused_and_leave_no_file() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    // Two H.264 chunks of one stream, the second later in time.
    let early = h264(dir.path());
    let late = h264_at(dir.path(), "late.cnk", STREAM, 100 * STEP);
    let out = dir.path().join("x.mp4");
    let all = (T0, T0 + 200 * STEP);
    assert!(export(&[early.as_path(), late.as_path()], &out, all.0, all.1, ExportCodec::Copy).is_ok());
    std::fs::remove_file(&out).unwrap();
    assert!(export(&[late.as_path(), early.as_path()], &out, all.0, all.1, ExportCodec::Copy).is_err());
    assert!(!out.exists() && !dir.path().join("x.mp4.partial").exists());
    let other = h264_at(dir.path(), "other.cnk", StreamId(8), 0);
    assert!(export(&[early.as_path(), other.as_path()], &out, all.0, all.1, ExportCodec::Copy).is_err());
    assert!(!out.exists());
}

#[test]
fn the_command_refuses_a_bad_request_and_an_existing_file() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let chunk = h264(dir.path());
    let out = dir.path().join("cli.mp4");
    let cli = |extra: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_continuum-export"))
            .arg("--out")
            .arg(&out)
            .args(extra)
            .arg(&chunk)
            .output()
            .unwrap()
    };
    assert!(!cli(&["--from-ns", "5", "--to-ns", "5"]).status.success());
    assert!(cli(&[]).status.success());
    let again = cli(&[]);
    assert!(!again.status.success() && String::from_utf8_lossy(&again.stderr).contains("--force"));
    assert!(cli(&["--force"]).status.success());
}

#[test]
fn a_one_nanosecond_single_frame_gop_exports_a_decodable_picture() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let chunk = au_chunk(&dir.path().join("single.cnk"), STREAM, &[(T0, T0 + 1, picture(0))], 15);
    let out = dir.path().join("single.mp4");
    let result = export(&[chunk.as_path()], &out, T0, T0 + 2, ExportCodec::Copy).unwrap();
    assert_eq!(result.frames, 1);
    assert_eq!(probe(&out), (ff::codec::Id::H264, (W, H), 1));
}

/// Two chronological chunks of white frames with independently chosen source dimensions.
fn sized_chunks(dir: &Path, sizes: [(u16, u16); 2]) -> [PathBuf; 2] {
    std::array::from_fn(|chunk| {
        let path = dir.join(format!("size-{chunk}.cnk"));
        let mut writer = ChunkWriter::create(&path, STREAM, ChunkFormat::Passthrough, T0).unwrap();
        let (width, height) = sizes[chunk];
        let rgb = vec![240; usize::from(width) * usize::from(height) * 3];
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, 95).encode(&rgb, width, height, jpeg_encoder::ColorType::Rgb).unwrap();
        for frame in 0..2 {
            let n = (chunk * 2 + frame) as u64;
            writer.append(T0 + n * STEP, T0 + (n + 1) * STEP, n as u32, &jpeg).unwrap();
        }
        writer.close().unwrap();
        path
    })
}

/// Decoded frame times in microseconds, plus luma at the left edge and center of each picture.
fn exported_luma(path: &Path) -> Vec<(i64, u8, u8, u8)> {
    use ff::Rescale;

    let mut input = ff::format::input(path).unwrap();
    let stream = input.streams().best(ff::media::Type::Video).unwrap();
    let index = stream.index();
    let time_base = stream.time_base();
    let mut decoder =
        ff::codec::context::Context::from_parameters(stream.parameters()).unwrap().decoder().video().unwrap();
    let mut frames = Vec::new();
    let mut decoded = ff::frame::Video::empty();
    let mut receive = |decoder: &mut ff::decoder::Video| {
        while decoder.receive_frame(&mut decoded).is_ok() {
            let row = decoded.height() as usize / 2 * decoded.stride(0);
            frames.push((
                decoded.pts().unwrap().rescale(time_base, ff::Rational(1, 1_000_000)),
                decoded.data(0)[row + 2],
                decoded.data(0)[row + decoded.width() as usize / 2],
                decoded.data(0)[2 * decoded.stride(0) + decoded.width() as usize / 2],
            ));
        }
    };
    for (stream, packet) in input.packets() {
        if stream.index() == index {
            decoder.send_packet(&packet).unwrap();
            receive(&mut decoder);
        }
    }
    decoder.send_eof().unwrap();
    receive(&mut decoder);
    frames
}

#[test]
fn an_export_across_sizes_fits_smaller_frames_with_black_bars_in_either_size_order() {
    let _gpu = gpu();
    for sizes in [[(64, 48), (32, 32)], [(32, 32), (64, 48)], [(64, 48), (64, 32)], [(64, 32), (64, 48)]] {
        let dir = tempfile::tempdir().unwrap();
        let chunks = sized_chunks(dir.path(), sizes);
        let out = dir.path().join("one-span.mp4");
        let done =
            export(&[chunks[0].as_path(), chunks[1].as_path()], &out, T0, T0 + 4 * STEP, ExportCodec::H264).unwrap();
        assert_eq!((done.frames, done.size), (4, (64, 48)));
        assert_eq!(probe(&out), (ff::codec::Id::H264, (64, 48), 4));
        let frames = exported_luma(&out);
        assert_eq!(frames.len(), 4);
        for (n, &(at, edge, center, top)) in frames.iter().enumerate() {
            assert!((at - (n as u64 * STEP / 1000) as i64).abs() <= 2, "frame {n}: {at}");
            assert!(center > 180, "frame {n}: source center was lost ({center})");
            if sizes[n / 2] == (32, 32) {
                assert!(edge < 24, "frame {n}: pillarbox isn't black ({edge})");
            } else {
                assert!(edge > 180, "frame {n}: native-size frame was cropped ({edge})");
            }
            if sizes[n / 2] == (64, 32) {
                assert!(top < 24, "frame {n}: letterbox isn't black ({top})");
            } else {
                assert!(top > 180, "frame {n}: picture was cropped ({top})");
            }
        }
    }
}

#[test]
fn only_frames_in_the_export_range_choose_the_canvas() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let chunks = sized_chunks(dir.path(), [(64, 48), (32, 32)]);
    let out = dir.path().join("selected.mp4");
    let done =
        export(&[chunks[0].as_path(), chunks[1].as_path()], &out, T0 + 2 * STEP, T0 + 4 * STEP, ExportCodec::Auto)
            .unwrap();
    assert_eq!((done.frames, done.size), (2, (32, 32)));
    assert_eq!(probe(&out), (ff::codec::Id::H264, (32, 32), 2));
}

#[test]
fn auto_reencodes_mixed_size_gops_but_explicit_copy_explains_the_boundary() {
    let _gpu = gpu();
    let dir = tempfile::tempdir().unwrap();
    let chunks: [PathBuf; 2] = std::array::from_fn(|i| {
        let (width, height) = [(32, 32), (64, 48)][i];
        let white = Picture { width, height, rgb: vec![240; (width * height * 3) as usize] };
        let frames: Vec<_> = (0..2)
            .map(|frame| {
                let n = (i * 2 + frame) as u64;
                (T0 + n * STEP, T0 + (n + 1) * STEP, white.clone())
            })
            .collect();
        au_chunk(&dir.path().join(format!("gop-{i}.cnk")), STREAM, &frames, 2)
    });
    let paths = [chunks[0].as_path(), chunks[1].as_path()];
    let out = dir.path().join("auto.mp4");
    let done = export(&paths, &out, T0, T0 + 4 * STEP, ExportCodec::Auto).unwrap();
    assert_eq!((done.frames, done.size), (4, (64, 48)));
    assert_eq!(probe(&out), (ff::codec::Id::H264, (64, 48), 4));
    let cli = dir.path().join("cli-auto.mp4");
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_continuum-export"))
        .arg("--out")
        .arg(&cli)
        .args(paths)
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(probe(&cli), (ff::codec::Id::H264, (64, 48), 4));
    let copy = dir.path().join("copy.mp4");
    let error = export(&paths, &copy, T0, T0 + 4 * STEP, ExportCodec::Copy).unwrap_err();
    assert!(error.to_string().contains("--codec h264"), "{error}");
    assert!(!copy.exists());
}
