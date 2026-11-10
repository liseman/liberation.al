//! Just enough of an H.264 sequence parameter set to tell a harmless in-band change (level, constraint flags,
//! reference count, VUI timing) from one that changes the capture layout (profile, chroma format, bit depth,
//! cropped size, or frame/field coding), which needs a new registration. Nothing here touches the bytes
//! that are recorded.

use std::fmt;

/// The fields of one SPS this module reads. `timing` is None when the VUI has none or ends before it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sps {
    /// profile_idc (66 Baseline, 77 Main, 100 High, ...).
    pub profile: u8,
    /// The constraint_set flags byte.
    pub constraints: u8,
    /// level_idc (31 is level 3.1).
    pub level: u8,
    /// seq_parameter_set_id.
    pub id: u32,
    /// chroma_format_idc (1 is 4:2:0).
    pub chroma_format: u32,
    /// Luma and chroma bit depths.
    pub bit_depth: (u32, u32),
    /// Cropped picture width in pixels.
    pub width: u32,
    /// Cropped picture height in pixels.
    pub height: u32,
    /// Frames only, no field coding.
    pub frame_mbs_only: bool,
    /// max_num_ref_frames.
    pub max_refs: u32,
    /// Whether VUI parameters are present.
    pub vui: bool,
    /// The VUI's num_units_in_tick, time_scale, and fixed_frame_rate flag.
    pub timing: Option<(u32, u32, bool)>,
}

/// What a decoder and the capture layout depend on. Two sets with the same layout decode into the same pictures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// profile_idc.
    pub profile: u8,
    /// chroma_format_idc.
    pub chroma_format: u32,
    /// Luma and chroma bit depths.
    pub bit_depth: (u32, u32),
    /// Cropped width in pixels.
    pub width: u32,
    /// Cropped height in pixels.
    pub height: u32,
    /// Frames only, no field coding.
    pub frame_mbs_only: bool,
}

impl Sps {
    /// The fields that decide the capture layout.
    #[must_use]
    pub fn layout(&self) -> Layout {
        Layout {
            profile: self.profile,
            chroma_format: self.chroma_format,
            bit_depth: self.bit_depth,
            width: self.width,
            height: self.height,
            frame_mbs_only: self.frame_mbs_only,
        }
    }
}

impl fmt::Display for Sps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "profile={} constraints={:#04x} level={} id={} chroma={} depth={}/{} size={}x{} frame_mbs_only={} refs={} vui={}",
            self.profile,
            self.constraints,
            self.level,
            self.id,
            self.chroma_format,
            self.bit_depth.0,
            self.bit_depth.1,
            self.width,
            self.height,
            u8::from(self.frame_mbs_only),
            self.max_refs,
            u8::from(self.vui),
        )?;
        match self.timing {
            Some((units, scale, fixed)) => write!(f, " timing={units}/{scale}{}", if fixed { " fixed" } else { "" }),
            None => write!(f, " timing=none"),
        }
    }
}

/// A short stable fingerprint of a parameter set's bytes for logs (FNV-1a, 64 bits).
#[must_use]
pub fn fingerprint(nal: &[u8]) -> String {
    let hash = nal
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3));
    format!("{hash:016x}")
}

/// A parameter set for a log line: its fingerprint, its fields when it is a readable SPS, and its bytes (parameter
/// sets carry no secrets, and they are small).
#[must_use]
pub fn describe(nal: &[u8]) -> String {
    let hex: String = nal.iter().take(96).map(|byte| format!("{byte:02x}")).collect();
    match nal.first().map(|byte| byte & 31) {
        Some(7) => match parse(nal) {
            Some(sps) => format!("SPS {} [{sps}] {hex}", fingerprint(nal)),
            None => format!("SPS {} [unreadable] {hex}", fingerprint(nal)),
        },
        _ => format!("PPS {} {hex}", fingerprint(nal)),
    }
}

/// Reads an SPS NAL unit (header byte included). None unless the layout fields read; a VUI that ends early only
/// leaves `timing` unknown, since some recorders send one cut short.
#[must_use]
pub fn parse(nal: &[u8]) -> Option<Sps> {
    if nal.first()? & 31 != 7 {
        return None;
    }
    let rbsp = unescape(&nal[1..]);
    let mut bits = Bits { bytes: &rbsp, at: 0 };
    let profile = bits.u8()?;
    let constraints = bits.u8()?;
    let level = bits.u8()?;
    let id = bits.ue()?;
    let (mut chroma_format, mut bit_depth, mut separate_planes) = (1, (8, 8), false);
    if matches!(profile, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135) {
        chroma_format = bits.ue()?;
        if chroma_format > 3 {
            return None;
        }
        if chroma_format == 3 {
            separate_planes = bits.flag()?;
        }
        bit_depth = (bits.ue()?.checked_add(8)?, bits.ue()?.checked_add(8)?);
        bits.flag()?; // qpprime_y_zero_transform_bypass
        if bits.flag()? {
            for list in 0..if chroma_format == 3 { 12 } else { 8 } {
                if bits.flag()? {
                    bits.scaling_list(if list < 6 { 16 } else { 64 })?;
                }
            }
        }
    }
    bits.ue()?; // log2_max_frame_num_minus4
    match bits.ue()? {
        0 => {
            bits.ue()?;
        }
        1 => {
            bits.flag()?;
            bits.se()?;
            bits.se()?;
            for _ in 0..bits.ue()?.min(255) {
                bits.se()?;
            }
        }
        2 => {}
        _ => return None,
    }
    let max_refs = bits.ue()?;
    bits.flag()?; // gaps_in_frame_num_value_allowed
    let width_mbs = bits.ue()?.checked_add(1)?;
    let height_units = bits.ue()?.checked_add(1)?;
    let frame_mbs_only = bits.flag()?;
    if !frame_mbs_only {
        bits.flag()?; // mb_adaptive_frame_field
    }
    bits.flag()?; // direct_8x8_inference
    let crop = if bits.flag()? { [bits.ue()?, bits.ue()?, bits.ue()?, bits.ue()?] } else { [0; 4] };
    let field = 2 - u32::from(frame_mbs_only);
    let (unit_x, unit_y): (u32, u32) = match (separate_planes, chroma_format) {
        (true, _) | (_, 0) => (1, field),
        (_, 1) => (2, 2 * field),
        (_, 2) => (2, field),
        _ => (1, field),
    };
    // Checked throughout: a corrupt set's sizes overflow u32, and an unreadable set beats a wrapped size.
    let width = width_mbs.checked_mul(16)?.checked_sub(unit_x.checked_mul(crop[0].checked_add(crop[1])?)?)?;
    let height =
        (field * 16).checked_mul(height_units)?.checked_sub(unit_y.checked_mul(crop[2].checked_add(crop[3])?)?)?;
    let vui = bits.flag()?;
    let timing = if vui { bits.vui_timing() } else { None };
    Some(Sps {
        profile,
        constraints,
        level,
        id,
        chroma_format,
        bit_depth,
        width,
        height,
        frame_mbs_only,
        max_refs,
        vui,
        timing,
    })
}

/// The payload without emulation-prevention bytes (the 3 in 00 00 03).
fn unescape(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut zeros = 0;
    for &byte in bytes {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    out
}

struct Bits<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Bits<'_> {
    fn bit(&mut self) -> Option<u32> {
        let byte = *self.bytes.get(self.at / 8)?;
        let bit = (byte >> (7 - self.at % 8)) & 1;
        self.at += 1;
        Some(u32::from(bit))
    }

    fn bits(&mut self, count: u32) -> Option<u32> {
        (0..count).try_fold(0u32, |value, _| Some((value << 1) | self.bit()?))
    }

    fn flag(&mut self) -> Option<bool> {
        Some(self.bit()? == 1)
    }

    fn u8(&mut self) -> Option<u8> {
        u8::try_from(self.bits(8)?).ok()
    }

    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        Some((1u32 << zeros) - 1 + self.bits(zeros)?)
    }

    fn se(&mut self) -> Option<i64> {
        let code = i64::from(self.ue()?);
        Some(if code % 2 == 1 { (code + 1) / 2 } else { -(code / 2) })
    }

    fn scaling_list(&mut self, size: usize) -> Option<()> {
        let (mut last, mut next) = (8i64, 8i64);
        for _ in 0..size {
            if next != 0 {
                next = (last + self.se()? + 256).rem_euclid(256);
            }
            if next != 0 {
                last = next;
            }
        }
        Some(())
    }

    /// The VUI's timing info, skipping the fields before it; None when absent or when the VUI ends first.
    fn vui_timing(&mut self) -> Option<(u32, u32, bool)> {
        if self.flag()? && self.bits(8)? == 255 {
            self.bits(32)?; // sample aspect ratio
        }
        if self.flag()? {
            self.flag()?; // overscan_appropriate
        }
        if self.flag()? {
            self.bits(4)?; // video_format, video_full_range
            if self.flag()? {
                self.bits(24)?; // colour primaries, transfer, matrix
            }
        }
        if self.flag()? {
            self.ue()?;
            self.ue()?;
        }
        if !self.flag()? {
            return None;
        }
        Some((self.bits(32)?, self.bits(32)?, self.flag()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kitchen's SPS as recorded (Reolink E1 Pro through UniFi Protect): its VUI announces timing it doesn't carry.
    pub const KITCHEN: &[u8] = &[
        0x67, 0x64, 0x00, 0x33, 0xac, 0x15, 0x14, 0xa0, 0xe0, 0x10, 0x68, 0x40, 0x00, 0x01, 0x3d, 0x80, 0x00, 0x18,
        0xce,
    ];
    /// Patio's SPS as recorded: Main profile, 1280x720, 29.97 fps VUI timing.
    pub const PATIO: &[u8] = &[
        0x67, 0x4d, 0x40, 0x1f, 0xa6, 0x80, 0x50, 0x05, 0xba, 0x6e, 0x02, 0x02, 0x02, 0x80, 0x00, 0x01, 0xf4, 0x80,
        0x00, 0x75, 0x4e, 0x4e, 0xd0, 0x44, 0x23, 0x50,
    ];

    #[test]
    fn kitchen_reads_as_high_profile_896x512_with_its_cut_short_timing_unknown() {
        let sps = parse(KITCHEN).unwrap();
        assert_eq!(
            sps.layout(),
            Layout { profile: 100, chroma_format: 1, bit_depth: (8, 8), width: 896, height: 512, frame_mbs_only: true }
        );
        assert_eq!((sps.level, sps.id, sps.max_refs, sps.vui, sps.timing), (51, 0, 1, true, None));
    }

    #[test]
    fn patio_reads_as_main_profile_with_its_vui_timing() {
        let sps = parse(PATIO).unwrap();
        assert_eq!((sps.profile, sps.level, sps.width, sps.height), (77, 31, 1280, 720));
        assert_eq!(sps.timing, Some((1001, 60060, true)));
    }

    /// Fields of a High profile SPS to write, everything else fixed (4:2:0, POC type 2, no VUI).
    struct Fields {
        depth_minus8: u32,
        width_mbs_minus1: u32,
        height_units_minus1: u32,
        frame_mbs_only: bool,
        crop: Option<[u32; 4]>,
    }

    /// Bits an encoder writes, most significant first.
    #[derive(Default)]
    struct Writer(Vec<bool>);

    impl Writer {
        fn u(&mut self, count: u32, value: u64) {
            self.0.extend((0..count).rev().map(|i| (value >> i) & 1 == 1));
        }

        fn ue(&mut self, value: u32) {
            let code = u64::from(value) + 1;
            let len = 64 - code.leading_zeros();
            self.u(len - 1, 0);
            self.u(len, code);
        }

        /// The NAL unit: SPS header byte, these bits, the stop bit, and emulation prevention.
        fn nal(mut self) -> Vec<u8> {
            self.u(1, 1);
            while !self.0.len().is_multiple_of(8) {
                self.0.push(false);
            }
            let mut nal = vec![0x67];
            let mut zeros = 0;
            for byte in self.0.chunks(8).map(|b| b.iter().fold(0u8, |v, &bit| (v << 1) | u8::from(bit))) {
                if zeros >= 2 && byte <= 3 {
                    nal.push(3);
                    zeros = 0;
                }
                zeros = if byte == 0 { zeros + 1 } else { 0 };
                nal.push(byte);
            }
            nal
        }
    }

    /// The SPS NAL unit for `f`, as an encoder writes it.
    fn high_sps(f: &Fields) -> Vec<u8> {
        let mut w = Writer::default();
        w.u(8, 100); // profile
        w.u(8, 0); // constraints
        w.u(8, 40); // level
        w.ue(0); // id
        w.ue(1); // chroma_format 4:2:0
        w.ue(f.depth_minus8);
        w.ue(f.depth_minus8);
        w.u(2, 0); // qpprime bypass, no scaling matrix
        w.ue(0); // log2_max_frame_num_minus4
        w.ue(2); // pic_order_cnt_type
        w.ue(1); // max_num_ref_frames
        w.u(1, 0); // gaps
        w.ue(f.width_mbs_minus1);
        w.ue(f.height_units_minus1);
        w.u(1, u64::from(f.frame_mbs_only));
        if !f.frame_mbs_only {
            w.u(1, 0); // mb_adaptive_frame_field
        }
        w.u(1, 1); // direct_8x8_inference
        w.u(1, u64::from(f.crop.is_some()));
        for side in f.crop.iter().flatten() {
            w.ue(*side);
        }
        w.u(1, 0); // no VUI
        w.nal()
    }

    const HD: Fields =
        Fields { depth_minus8: 0, width_mbs_minus1: 119, height_units_minus1: 67, frame_mbs_only: true, crop: None };

    #[test]
    fn a_written_1080p_set_reads_back_with_its_crop() {
        let sps = parse(&high_sps(&Fields { crop: Some([0, 0, 0, 4]), ..HD })).expect("readable");
        assert_eq!((sps.profile, sps.bit_depth, sps.width, sps.height), (100, (8, 8), 1920, 1080));
        let interlaced = Fields { height_units_minus1: 33, frame_mbs_only: false, ..HD };
        assert_eq!(parse(&high_sps(&interlaced)).expect("readable").height, 1088, "34 field pairs of 16 rows");
    }

    /// Sizes and depths past u32 wrapped to a wrong layout in release builds (and panicked in debug). A corrupt
    /// set like these is unreadable, as one cut short is.
    #[test]
    fn a_set_whose_size_or_depth_overflows_does_not_read() {
        let huge = u32::MAX - 1; // the largest ue(v) the parser reads
        let cases = [
            ("bit depth", Fields { depth_minus8: huge, ..HD }),
            ("width", Fields { width_mbs_minus1: 1 << 28, ..HD }),
            ("frame height", Fields { height_units_minus1: 1 << 28, ..HD }),
            ("field height", Fields { height_units_minus1: 1 << 27, frame_mbs_only: false, ..HD }),
            ("crop", Fields { crop: Some([1 << 31, 0, 0, 0]), ..HD }),
        ];
        for (what, fields) in cases {
            assert_eq!(parse(&high_sps(&fields)), None, "{what}");
        }
    }

    #[test]
    fn a_set_cut_off_before_its_size_does_not_read() {
        assert_eq!(parse(&KITCHEN[..8]), None);
        assert_eq!(parse(&[0x68, 0xee, 0x3c, 0xb0]), None, "a PPS is not an SPS");
    }

    #[test]
    fn a_log_line_names_the_fingerprint_fields_and_bytes() {
        let line = describe(KITCHEN);
        assert!(line.starts_with(&format!("SPS {} [profile=100 ", fingerprint(KITCHEN))), "{line}");
        assert!(line.contains("size=896x512") && line.contains("timing=none") && line.ends_with("18ce"), "{line}");
        assert_ne!(fingerprint(KITCHEN), fingerprint(PATIO));
    }
}
