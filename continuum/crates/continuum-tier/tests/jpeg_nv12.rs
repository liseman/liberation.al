//! A camera's JPEG decoded straight to NV12 ([`JpegNv12`]) against the older route through RGB
//! ([`decode_jpeg`] then [`rgb_to_nv12`]): the same picture within rounding, the same strictness, and far less work.
//!
//! The timing comparison is ignored by default; run it optimized:
//!
//! ```text
//! tools/build-slot.sh cargo test --release -p continuum-tier --test jpeg_nv12 -- --ignored --nocapture
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use continuum_tier::{JpegNv12, decode_jpeg, rgb_to_nv12};
use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

/// A made-up picture with smooth shading and a few hard edges, as RGB.
fn scene(w: u16, h: u16) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(usize::from(w) * usize::from(h) * 3);
    for y in 0..h {
        for x in 0..w {
            let block = u8::from((x / 8 + y / 8) % 2 == 0) * 40;
            let r = (u32::from(x) * 200 / u32::from(w.max(1))) as u8 + block;
            let g = (u32::from(y) * 200 / u32::from(h.max(1))) as u8 + block / 2;
            let b = 60 + ((u32::from(x) + u32::from(y)) * 120 / u32::from(w + h)) as u8;
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    rgb
}

fn jpeg(w: u16, h: u16, sampling: SamplingFactor) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = Encoder::new(&mut bytes, 90);
    encoder.set_sampling_factor(sampling);
    encoder.encode(&scene(w, h), w, h, ColorType::Rgb).expect("encode");
    bytes
}

/// (mean, largest) absolute difference between two planes.
fn difference(a: &[u8], b: &[u8]) -> (f64, u8) {
    assert_eq!(a.len(), b.len());
    let total: u64 = a.iter().zip(b).map(|(x, y)| u64::from(x.abs_diff(*y))).sum();
    (total as f64 / a.len() as f64, a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0))
}

/// The direct picture matches the one made through RGB: luma within 1 level on average and 3 at worst, chroma
/// within 1 on average and 4 at worst (the RGB route rounds twice and clips colours RGB can't hold).
fn assert_matches_the_rgb_route(w: u16, h: u16, sampling: SamplingFactor) {
    let bytes = jpeg(w, h, sampling);
    let through_rgb = rgb_to_nv12(&decode_jpeg(&bytes).expect("decode"));
    let mut direct = Vec::new();
    let size = JpegNv12::default().decode(&bytes, &mut direct).expect("decode");
    assert_eq!(size, (u32::from(w), u32::from(h)));
    assert_eq!(direct.len(), through_rgb.len(), "the same layout, odd edges included");
    let luma = usize::from(w) * usize::from(h);
    let (mean, worst) = difference(&direct[..luma], &through_rgb[..luma]);
    assert!(mean < 1.0 && worst <= 3, "luma differs by {mean:.2} on average, {worst} at worst");
    let (mean, worst) = difference(&direct[luma..], &through_rgb[luma..]);
    assert!(mean < 1.0 && worst <= 4, "chroma differs by {mean:.2} on average, {worst} at worst");
}

#[test]
fn a_420_jpeg_decodes_to_the_picture_the_rgb_route_makes() {
    assert_matches_the_rgb_route(64, 48, SamplingFactor::F_2_2);
}

#[test]
fn a_422_jpeg_decodes_to_the_picture_the_rgb_route_makes() {
    assert_matches_the_rgb_route(64, 48, SamplingFactor::F_2_1);
}

#[test]
fn a_444_jpeg_decodes_to_the_picture_the_rgb_route_makes() {
    assert_matches_the_rgb_route(64, 48, SamplingFactor::F_1_1);
}

#[test]
fn an_odd_sized_jpeg_repeats_its_last_column_and_row_as_the_rgb_route_does() {
    assert_matches_the_rgb_route(33, 17, SamplingFactor::F_2_2);
    assert_matches_the_rgb_route(33, 17, SamplingFactor::F_2_1);
}

#[test]
fn a_truncated_jpeg_is_an_error_not_a_picture() {
    let bytes = jpeg(64, 48, SamplingFactor::F_2_1);
    let mut out = Vec::new();
    assert!(JpegNv12::default().decode(&bytes, &mut out).is_ok());
    for keep in [bytes.len() / 2, bytes.len() - 20] {
        let cut = &bytes[..keep];
        assert!(decode_jpeg(cut).is_err(), "the RGB route refuses {keep} of {} bytes", bytes.len());
        assert!(JpegNv12::default().decode(cut, &mut out).is_err(), "and so does the direct one");
    }
    assert!(JpegNv12::default().decode(b"not a JPEG", &mut out).is_err());
}

#[test]
fn a_grayscale_jpeg_takes_the_rgb_route() {
    let (w, h) = (32u16, 16u16);
    let gray: Vec<u8> = (0..u32::from(w) * u32::from(h)).map(|i| (i % 251) as u8).collect();
    let mut bytes = Vec::new();
    Encoder::new(&mut bytes, 90).encode(&gray, w, h, ColorType::Luma).expect("encode");
    let mut direct = Vec::new();
    assert_eq!(JpegNv12::default().decode(&bytes, &mut direct).expect("decode"), (32, 16));
    assert_eq!(direct, rgb_to_nv12(&decode_jpeg(&bytes).expect("decode")));
}

#[test]
fn one_decoder_serves_frames_of_any_size_in_turn() {
    let mut decoder = JpegNv12::default();
    let mut out = Vec::new();
    for (w, h) in [(64u16, 48u16), (16, 8), (64, 48)] {
        let bytes = jpeg(w, h, SamplingFactor::F_2_1);
        decoder.decode(&bytes, &mut out).expect("decode");
        assert_eq!(out, {
            let mut fresh = Vec::new();
            JpegNv12::default().decode(&bytes, &mut fresh).expect("decode");
            fresh
        });
    }
}

/// CPU time a frame for a 1280 by 720 4:2:2 JPEG, the shape a USB camera sends, by both routes.
#[test]
#[ignore = "a timing comparison: run it with --release --ignored --nocapture"]
fn timing_direct_against_the_rgb_route() {
    const FRAMES: u32 = 60;
    let bytes = jpeg(1280, 720, SamplingFactor::F_2_1);
    let time = |work: &mut dyn FnMut()| {
        work();
        let began = std::time::Instant::now();
        for _ in 0..FRAMES {
            work();
        }
        began.elapsed().as_secs_f64() * 1000.0 / f64::from(FRAMES)
    };
    let decode_rgb = time(&mut || drop(std::hint::black_box(decode_jpeg(&bytes).expect("decode"))));
    let picture = decode_jpeg(&bytes).expect("decode");
    let convert = time(&mut || drop(std::hint::black_box(rgb_to_nv12(&picture))));
    let (mut decoder, mut out) = (JpegNv12::default(), Vec::new());
    let direct = time(&mut || {
        decoder.decode(&bytes, &mut out).expect("decode");
        std::hint::black_box(&out);
    });
    println!("1280x720 4:2:2 JPEG, {} bytes, {FRAMES} frames each, single thread", bytes.len());
    println!(
        "  through RGB: decode {decode_rgb:.2} ms + convert {convert:.2} ms = {:.2} ms a frame",
        decode_rgb + convert
    );
    println!("  direct:      {direct:.2} ms a frame ({:.1} times less)", (decode_rgb + convert) / direct);
}
