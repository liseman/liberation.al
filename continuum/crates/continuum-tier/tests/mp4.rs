//! The MP4 sample table reader against a file FFmpeg wrote with B-frames, as Ingest's conversion does: `data/bframes.mp4`
//! is 2 seconds of a test pattern at 10 fps, 64x48, H.264 High with 2 B-frames and a keyframe every 10 frames
//! (`ffmpeg -f lavfi -i testsrc=size=64x48:rate=10:duration=2 -c:v libx264 -profile:v high -bf 2 -g 10
//! -keyint_min 10 -sc_threshold 0`). The expected values are ffprobe's packet table for it.

use std::io::Cursor;

use continuum_tier::mp4::{read_video, to_annex_b};

const FIXTURE: &[u8] = include_bytes!("data/bframes.mp4");

/// ffprobe's (pts, size, pos, key) per packet, in decode order; the time base is 1/10240.
const PACKETS: [(i64, u32, u64, bool); 20] = [
    (0, 1369, 48, true),
    (3072, 175, 1417, false),
    (1024, 12, 1592, false),
    (2048, 12, 1604, false),
    (6144, 186, 1616, false),
    (4096, 14, 1802, false),
    (5120, 12, 1816, false),
    (9216, 174, 1828, false),
    (7168, 14, 2002, false),
    (8192, 12, 2016, false),
    (10240, 730, 2028, true),
    (13312, 163, 2758, false),
    (11264, 12, 2921, false),
    (12288, 12, 2933, false),
    (16384, 169, 2945, false),
    (14336, 14, 3114, false),
    (15360, 12, 3128, false),
    (19456, 140, 3140, false),
    (17408, 14, 3280, false),
    (18432, 12, 3294, false),
];

#[test]
fn the_sample_table_matches_ffprobe_with_the_edit_list_applied() {
    let v = read_video(&mut Cursor::new(FIXTURE)).expect("read");
    assert_eq!((&v.codec, v.width, v.height, v.timescale), (b"avc1", 64, 48, 10_240));
    let got: Vec<(i64, u32, u64, bool)> = v.samples.iter().map(|s| (s.pts, s.size, s.offset, s.key)).collect();
    assert_eq!(got, PACKETS);
    assert!(v.samples.windows(2).all(|w| w[1].dts == w[0].dts + u64::from(w[0].duration)), "decode times run on");
    assert_eq!(v.duration(), 20_480, "2 seconds");
}

#[test]
fn gops_cover_their_samples_bytes_and_presentation_span() {
    let v = read_video(&mut Cursor::new(FIXTURE)).expect("read");
    let gops = v.gops();
    assert_eq!(gops.len(), 2);
    assert_eq!((gops[0].first, gops[0].count, gops[0].pts, gops[0].pts_end), (0, 10, 0, 10_240));
    assert_eq!((gops[1].first, gops[1].count, gops[1].pts, gops[1].pts_end), (10, 10, 10_240, 20_480));
    assert_eq!((gops[0].offset, gops[0].length), (48, 2028 - 48));
    assert_eq!((gops[1].offset, gops[1].length), (2028, 3294 + 12 - 2028));
}

#[test]
fn a_keyframe_becomes_annex_b_with_its_parameter_sets() {
    let v = read_video(&mut Cursor::new(FIXTURE)).expect("read");
    let s = v.samples[0];
    let raw = &FIXTURE[s.offset as usize..s.offset as usize + s.size as usize];
    let key = to_annex_b(&v, raw, true).expect("annex b");
    // SPS (type 7) and PPS (type 8) first, then the IDR slice (type 5) somewhere after.
    let types: Vec<u8> = key.windows(5).filter(|w| w[..4] == [0, 0, 0, 1]).map(|w| w[4] & 0x1f).collect();
    assert_eq!(&types[..2], &[7, 8], "{types:?}");
    assert!(types.contains(&5), "{types:?}");
    let delta = v.samples[1];
    let raw = &FIXTURE[delta.offset as usize..delta.offset as usize + delta.size as usize];
    let types: Vec<u8> = to_annex_b(&v, raw, false)
        .expect("annex b")
        .windows(5)
        .filter(|w| w[..4] == [0, 0, 0, 1])
        .map(|w| w[4] & 0x1f)
        .collect();
    assert!(!types.contains(&7) && !types.contains(&5), "{types:?}");
}

#[test]
fn an_unfinished_or_damaged_file_is_an_error_not_a_panic() {
    // A file still being written: no moov yet.
    let moov_at = FIXTURE.windows(4).position(|w| w == b"moov").expect("moov") - 4;
    assert!(read_video(&mut Cursor::new(&FIXTURE[..moov_at])).is_err());
    // Every truncation and a sample of single-byte damage inside the moov.
    for cut in (moov_at..FIXTURE.len()).step_by(7) {
        let _ = read_video(&mut Cursor::new(&FIXTURE[..cut]));
    }
    for at in (moov_at + 8..FIXTURE.len()).step_by(3) {
        for b in [0x00, 0xff, 0x7f] {
            let mut d = FIXTURE.to_vec();
            d[at] = b;
            let _ = read_video(&mut Cursor::new(d));
        }
    }
}

/// The fixture with `bytes` written at the place `at` bytes after the first `kind` box's type.
fn patched(kind: &[u8; 4], at: usize, bytes: &[u8]) -> Vec<u8> {
    let mut d = FIXTURE.to_vec();
    let p = d.windows(4).position(|w| w == kind).expect("box") + 4 + at;
    d[p..p + bytes.len()].copy_from_slice(bytes);
    d
}

#[test]
fn a_box_claiming_a_64_bit_size_past_its_parent_is_an_error() {
    // mvhd's size becomes 1 (a 64-bit size follows its type), and that size u64::MAX.
    let mut d = patched(b"mvhd", 0, &u64::MAX.to_be_bytes());
    let p = d.windows(4).position(|w| w == b"mvhd").expect("mvhd");
    d[p - 4..p].copy_from_slice(&1u32.to_be_bytes());
    assert!(read_video(&mut Cursor::new(d)).is_err());
}

#[test]
fn a_uniform_size_table_claiming_more_samples_than_the_file_holds_is_refused_before_allocating() {
    // stsz: version and flags, then a uniform size of 1 and ten million samples, from a file of 4 KB.
    let mut d = patched(b"stsz", 4, &1u32.to_be_bytes());
    let p = d.windows(4).position(|w| w == b"stsz").expect("stsz") + 8;
    d[p..p + 4].copy_from_slice(&10_000_000u32.to_be_bytes());
    let err = read_video(&mut Cursor::new(d)).expect_err("refused");
    assert!(err.to_string().contains("don't fit"), "{err}");
}
