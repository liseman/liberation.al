//! The H.264 browser stream: a few seconds of synthetic frames through each encoder that opens here, then decoded
//! back. Bounded: 20 small frames per encoder.
#![cfg(feature = "ffmpeg")]

use continuum_tier::h264::{AccessUnit, H264Encoder, H264Settings};
use ffmpeg_next as ff;

const W: u32 = 160;
const H: u32 = 96;
const FRAMES: u64 = 20;
const GOP: u32 = 10;
const T0: u64 = 1_793_227_877_000_000_000;
const STEP: u64 = 66_666_667; // 15 fps

/// A moving gradient, so frames differ and the encoder has something to predict.
fn frame(n: u64) -> Vec<u8> {
    let (w, h) = (W as usize, H as usize);
    let mut out = vec![128u8; w * h * 3 / 2];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = ((x + y + n as usize * 4) % 256) as u8;
        }
    }
    out
}

fn settings() -> H264Settings {
    H264Settings { gop_frames: GOP, frame_rate: 15, bit_rate: 500_000, ..H264Settings::default() }
}

fn encode(names: &[&str]) -> Option<(String, Vec<AccessUnit>)> {
    let mut enc = H264Encoder::open_with(names, W, H, settings()).ok()?;
    let name = enc.encoder_name().to_owned();
    let mut out = Vec::new();
    for n in 0..FRAMES {
        out.extend(enc.encode(&frame(n), T0 + n * STEP, T0 + (n + 1) * STEP).unwrap());
    }
    out.extend(enc.finish().unwrap());
    Some((name, out))
}

/// The NAL unit types in an Annex B access unit.
fn nal_types(au: &[u8]) -> Vec<u8> {
    let mut types = Vec::new();
    let mut i = 0;
    while i + 3 <= au.len() {
        if au[i..i + 3] == [0, 0, 1] {
            if let Some(&b) = au.get(i + 3) {
                types.push(b & 0x1f);
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    types
}

/// What the decoder makes of a frame: its size, the luma of its top-left pixel (the gradient's phase, which rises
/// 4 per frame, so it shows the order), and the colour signalling it read from the stream.
#[derive(Debug, PartialEq)]
struct Decoded {
    size: (u32, u32),
    corner: u8,
    space: ff::util::color::Space,
    range: ff::util::color::Range,
}

fn take(picture: &ff::frame::Video) -> Decoded {
    Decoded {
        size: (picture.width(), picture.height()),
        corner: picture.data(0)[0],
        space: picture.color_space(),
        range: picture.color_range(),
    }
}

fn decode(aus: &[AccessUnit]) -> Vec<Decoded> {
    // Encoding initialized FFmpeg; do not rewrite ffmpeg-next's global error table here.
    let codec = ff::decoder::find(ff::codec::Id::H264).unwrap();
    let mut dec = ff::codec::context::Context::new_with_codec(codec).decoder().video().unwrap();
    let mut out = Vec::new();
    let mut picture = ff::frame::Video::empty();
    for au in aus {
        dec.send_packet(&ff::Packet::copy(&au.bytes)).unwrap();
        while dec.receive_frame(&mut picture).is_ok() {
            out.push(take(&picture));
        }
    }
    dec.send_eof().unwrap();
    while dec.receive_frame(&mut picture).is_ok() {
        out.push(take(&picture));
    }
    out
}

fn sizes(decoded: &[Decoded]) -> Vec<(u32, u32)> {
    decoded.iter().map(|d| d.size).collect()
}

fn check(name: &str, aus: &[AccessUnit]) {
    assert_eq!(aus.len(), FRAMES as usize, "{name}: one access unit per frame");
    for (n, au) in aus.iter().enumerate() {
        let n = n as u64;
        assert_eq!((au.start_ns, au.end_ns), (T0 + n * STEP, T0 + (n + 1) * STEP), "{name}: frame {n}'s times");
        assert!(
            au.bytes.starts_with(&[0, 0, 1]) || au.bytes.starts_with(&[0, 0, 0, 1]),
            "{name}: frame {n} start code"
        );
        let types = nal_types(&au.bytes);
        let keyframe = n.is_multiple_of(u64::from(GOP));
        assert_eq!(au.key, keyframe, "{name}: frame {n} keyframe flag");
        if keyframe {
            for t in [7, 8, 5] {
                assert!(types.contains(&t), "{name}: keyframe {n} lacks NAL type {t}: {types:?}");
            }
        } else {
            assert!(!types.contains(&7) && !types.contains(&5), "{name}: frame {n} is not a delta frame: {types:?}");
        }
    }
    // Main profile (profile_idc 77) in the first SPS.
    let first = &aus[0].bytes;
    let sps = (0..first.len() - 4).find(|&i| first[i..i + 3] == [0, 0, 1] && first[i + 3] & 0x1f == 7).unwrap();
    assert_eq!(first[sps + 4], 77, "{name}: profile");
    let decoded = decode(aus);
    assert_eq!(sizes(&decoded), vec![(W, H); FRAMES as usize], "{name}: decodes back");
    // In order: each frame's top-left luma is near its own phase, 4n (lossy, so within a few levels).
    for (n, d) in decoded.iter().enumerate() {
        assert!(d.corner.abs_diff((n * 4) as u8) <= 6, "{name}: frame {n} decoded as phase {}", d.corner);
    }
    // The stream says what the pixels are: BT.601 (SMPTE 170M) and full range, as rgb_to_nv12 makes them.
    assert_eq!((decoded[0].space, decoded[0].range), (ff::util::color::Space::SMPTE170M, ff::util::color::Range::JPEG));
    // A browser that joins at the second keyframe decodes from there.
    let joined = decode(&aus[GOP as usize..]);
    assert_eq!(sizes(&joined), vec![(W, H); (FRAMES - u64::from(GOP)) as usize], "{name}: joins mid-stream");
}

#[test]
fn libx264_makes_a_browser_stream() {
    let (name, aus) = encode(&["libx264"]).expect("libx264 is in this FFmpeg");
    check(&name, &aus);
}

#[test]
fn the_hardware_encoder_makes_a_browser_stream_where_it_opens() {
    // NVENC on Linux with an NVIDIA GPU, VideoToolbox on the Mac; skipped where neither opens.
    match encode(&["h264_nvenc", "h264_videotoolbox"]) {
        Some((name, aus)) => {
            eprintln!("hardware H.264 encoder: {name}");
            check(&name, &aus);
        }
        None => eprintln!("no hardware H.264 encoder opened here; skipped"),
    }
}

#[test]
fn bad_input_is_refused() {
    assert!(H264Encoder::open(W + 1, H, settings()).is_err());
    assert!(H264Encoder::open(0, H, settings()).is_err());
    assert!(H264Encoder::open(W, H, H264Settings { gop_frames: 0, ..settings() }).is_err());
    let none = H264Encoder::open_with(&["no_such_encoder"], W, H, settings());
    assert!(matches!(none, Err(e) if e.to_string().contains("no_such_encoder")));
    let mut enc = H264Encoder::open_with(&["libx264"], W, H, settings()).unwrap();
    assert!(enc.encode(&[0u8; 10], T0, T0 + 1).is_err());
    assert!(enc.encode(&frame(0), T0 + 1, T0).is_err());
}

#[test]
fn explicit_capture_cut_starts_an_independently_decodable_keyframe() {
    let settings = H264Settings { gop_frames: 100, low_delay: true, threads: 1, ..settings() };
    let mut encoder = H264Encoder::open_with(&["libx264"], W, H, settings).unwrap();
    let mut units = Vec::new();
    for n in 0..8 {
        units.extend(encoder.encode_keyframe(&frame(n), T0 + n * STEP, T0 + (n + 1) * STEP, n == 3).unwrap());
    }
    units.extend(encoder.finish().unwrap());
    assert_eq!(units.len(), 8);
    assert!(units[3].key);
    assert_eq!(units[3].start_ns, T0 + 3 * STEP);
    for kind in [7, 8, 5] {
        assert!(nal_types(&units[3].bytes).contains(&kind));
    }
    assert_eq!(decode(&units[3..]).len(), 5);
}

#[test]
fn invalid_nvenc_quality_is_rejected_before_opening_any_hardware() {
    let error = H264Encoder::open_nvenc(W, H, settings(), 0, 52).err().unwrap();
    assert!(error.to_string().contains("CQ"));
}
