//! Read-only device-list JSON. Camera frame intervals are rational seconds, not rounded manifest frame rates.
//! Size/interval ranges stay symbolic; enumeration order is not a default or preference. The Linux adapters only
//! issue ENUM_FMT, ENUM_FRAMESIZES, and ENUM_FRAMEINTERVALS, never format/control/stream-setting requests.

use continuum_capture::manifest::DeviceKind;
use continuum_capture::supervisor::DeviceInfo;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Options {
    pub json: bool,
    pub format: Format,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Format {
    name: String,
    fourcc: [u8; 4],
}

impl Format {
    fn parse(value: &str) -> Result<Self, String> {
        let upper = value.to_ascii_uppercase();
        if upper == "MJPEG" || upper == "MJPG" {
            return Ok(Self { name: "MJPEG".into(), fourcc: *b"MJPG" });
        }
        if value.len() != 4 || !value.bytes().all(|b| b.is_ascii_graphic() || b == b' ') || value.trim().is_empty() {
            return Err("--format needs MJPEG or a four-byte ASCII FourCC, such as UYVY".into());
        }
        Ok(Self { fourcc: value.as_bytes().try_into().expect("four ASCII bytes"), name: value.into() })
    }
}

impl Options {
    pub(super) fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Self { json: false, format: Format::parse("MJPEG")? };
        let mut list = false;
        let mut format = false;
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--list" if !list => list = true,
                "--json" if !options.json => options.json = true,
                "--format" if !format => {
                    options.format = Format::parse(iter.next().ok_or("--format needs a value")?)?;
                    format = true;
                }
                _ => return Err(format!("unknown or repeated device-list argument {arg:?}")),
            }
        }
        if !list {
            return Err("--json and --format require --list".into());
        }
        if format && !options.json {
            return Err("--format requires --list --json".into());
        }
        Ok(options)
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct Fraction {
    numerator: u32,
    denominator: u32,
}

#[cfg(target_os = "linux")]
impl Fraction {
    fn valid(self) -> bool {
        self.numerator != 0 && self.denominator != 0
    }
    fn inverse(self) -> Self {
        Self { numerator: self.denominator, denominator: self.numerator }
    }
    fn at_most(self, other: Self) -> bool {
        u64::from(self.numerator) * u64::from(other.denominator)
            <= u64::from(other.numerator) * u64::from(self.denominator)
    }
}

#[cfg(target_os = "linux")]
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Interval {
    Discrete { interval: Fraction, fps: Fraction },
    Stepwise { min: Fraction, max: Fraction, step: Fraction },
    Continuous { min: Fraction, max: Fraction, step: Fraction },
}

#[cfg(target_os = "linux")]
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Size {
    Discrete { width: u32, height: u32, intervals: Vec<Interval>, diagnostics: Vec<String> },
    Stepwise { min_width: u32, max_width: u32, step_width: u32, min_height: u32, max_height: u32, step_height: u32 },
    Continuous { min_width: u32, max_width: u32, step_width: u32, min_height: u32, max_height: u32, step_height: u32 },
}

#[cfg(not(target_os = "linux"))]
type Size = serde_json::Value; // No mode descriptors are constructed on platforms without a capability backend.

#[derive(Debug, Serialize)]
struct CameraModes {
    status: &'static str,
    fourcc: String,
    sizes: Vec<Size>,
    diagnostics: Vec<String>,
}

impl CameraModes {
    fn unavailable(format: &Format, why: String) -> Self {
        Self {
            status: "unavailable",
            fourcc: String::from_utf8_lossy(&format.fourcc).into(),
            sizes: Vec::new(),
            diagnostics: vec![why],
        }
    }
}

#[derive(Serialize)]
struct ListedDevice<'a> {
    kind: &'static str,
    key: &'a str,
    node: &'a str,
    name: &'a str,
    usb: Option<String>,
    paths: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    camera_modes: Option<CameraModes>,
}

#[derive(Serialize)]
struct Listing<'a> {
    schema_version: u32,
    format: &'a str,
    fourcc: String,
    devices: Vec<ListedDevice<'a>>,
    diagnostics: Vec<String>,
}

fn encode(
    devices: &[DeviceInfo],
    diagnostics: Vec<String>,
    format: &Format,
    mut probe: impl FnMut(&DeviceInfo, &Format) -> CameraModes,
) -> Result<String, serde_json::Error> {
    let mut ordered: Vec<_> = devices.iter().collect();
    ordered.sort_by(|a, b| a.key.cmp(&b.key));
    let devices = ordered
        .into_iter()
        .map(|device| ListedDevice {
            kind: match device.kind {
                DeviceKind::Camera => "camera",
                DeviceKind::Microphone => "microphone",
                DeviceKind::Xvf3800 => "xvf3800",
                DeviceKind::Teensy => "teensy",
                DeviceKind::Rtsp => "rtsp",
            },
            key: &device.key,
            node: &device.node,
            name: &device.name,
            usb: device.usb.map(|(vendor, product)| format!("{vendor:04x}:{product:04x}")),
            paths: &device.paths,
            camera_modes: (device.kind == DeviceKind::Camera).then(|| probe(device, format)),
        })
        .collect();
    serde_json::to_string(&Listing {
        schema_version: 1,
        format: &format.name,
        fourcc: String::from_utf8_lossy(&format.fourcc).into(),
        devices,
        diagnostics,
    })
}

pub(super) fn json(
    devices: &[DeviceInfo],
    diagnostics: Vec<String>,
    options: &Options,
) -> Result<String, serde_json::Error> {
    encode(devices, diagnostics, &options.format, camera_modes)
}

/// The existing human-readable --list line; no mode query is needed for text output.
pub(super) fn text(device: &DeviceInfo) -> String {
    let usb = device.usb.map(|(vendor, product)| format!("{vendor:04x}:{product:04x}")).unwrap_or_else(|| "-".into());
    format!("{:?}\tname {:?}\tusb {usb}\tnode {}\tkey {}", device.kind, device.name, device.node, device.key)
}

#[cfg(target_os = "linux")]
fn camera_modes(device: &DeviceInfo, format: &Format) -> CameraModes {
    linux::probe(std::path::Path::new(&device.node), format)
}

#[cfg(not(target_os = "linux"))]
fn camera_modes(_device: &DeviceInfo, format: &Format) -> CameraModes {
    CameraModes::unavailable(format, "camera mode enumeration is available only through Linux V4L2".into())
}

#[cfg(target_os = "linux")]
mod linux {
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::Path;

    use v4l::frameinterval::{FrameInterval, FrameIntervalEnum};
    use v4l::framesize::{FrameSize, FrameSizeEnum};
    use v4l::v4l_sys::{v4l2_fmtdesc, v4l2_frmivalenum, v4l2_frmsizeenum};

    use super::{CameraModes, Format, Fraction, Interval, Size};

    const MAX_ITEMS: u32 = 128;
    const MAX_INTERVALS: u32 = 2048;
    const DISCRETE: u32 = 1;
    const CONTINUOUS: u32 = 2;
    const STEPWISE: u32 = 3;

    // v4l 0.14's high-level enumerators suppress errors after index zero and continue past range entries. Keep the
    // same binding/conversion API, but stop at a range, distinguish EINVAL from real failures, and cap every loop.
    trait Enumerate {
        fn format(&self, index: u32) -> io::Result<[u8; 4]>;
        fn size(&self, fourcc: [u8; 4], index: u32) -> io::Result<FrameSize>;
        fn interval(&self, fourcc: [u8; 4], width: u32, height: u32, index: u32) -> io::Result<FrameInterval>;
    }

    struct Queries(File);

    impl Queries {
        fn query<T>(&self, request: v4l::v4l2::vidioc::_IOC_TYPE, value: &mut T) -> io::Result<()> {
            // SAFETY: every caller pairs the request with its matching V4L2 ABI structure. The initialized structure
            // is exclusively borrowed, writable, and alive for the synchronous ioctl; the kernel retains no pointer.
            unsafe { v4l::v4l2::ioctl(self.0.as_raw_fd(), request, std::ptr::from_mut(value).cast()) }
        }
    }

    impl Enumerate for Queries {
        fn format(&self, index: u32) -> io::Result<[u8; 4]> {
            // SAFETY: this bindgen ABI type contains only integer fields and byte arrays; all-zero is valid, including
            // its reserved fields. Set the required input fields before calling the read-only enumeration ioctl.
            let mut value: v4l2_fmtdesc = unsafe { std::mem::zeroed() };
            value.index = index;
            value.type_ = v4l::buffer::Type::VideoCapture as u32;
            self.query(v4l::v4l2::vidioc::VIDIOC_ENUM_FMT, &mut value)?;
            Ok(value.pixelformat.to_le_bytes())
        }
        fn size(&self, fourcc: [u8; 4], index: u32) -> io::Result<FrameSize> {
            // SAFETY: this bindgen ABI type's integer fields and integer-only union admit all-zero. Only the
            // driver's reported union tag is decoded, by v4l's checked FrameSize conversion, after ioctl succeeds.
            let mut value: v4l2_frmsizeenum = unsafe { std::mem::zeroed() };
            value.index = index;
            value.pixel_format = u32::from_le_bytes(fourcc);
            self.query(v4l::v4l2::vidioc::VIDIOC_ENUM_FRAMESIZES, &mut value)?;
            FrameSize::try_from(value).map_err(|why| io::Error::new(io::ErrorKind::InvalidData, why))
        }
        fn interval(&self, fourcc: [u8; 4], width: u32, height: u32, index: u32) -> io::Result<FrameInterval> {
            // SAFETY: same integer-only ABI/union invariant as ENUM_FRAMESIZES. The matching v4l conversion selects
            // only the union member authorized by the returned type tag, after this read-only ioctl succeeds.
            let mut value: v4l2_frmivalenum = unsafe { std::mem::zeroed() };
            value.index = index;
            value.pixel_format = u32::from_le_bytes(fourcc);
            value.width = width;
            value.height = height;
            self.query(v4l::v4l2::vidioc::VIDIOC_ENUM_FRAMEINTERVALS, &mut value)?;
            FrameInterval::try_from(value).map_err(|why| io::Error::new(io::ErrorKind::InvalidData, why))
        }
    }

    fn scan<T>(
        limit: u32,
        mut query: impl FnMut(u32) -> io::Result<T>,
        range: impl Fn(&T) -> bool,
    ) -> (Vec<T>, Option<String>) {
        let mut values = Vec::new();
        for index in 0..limit {
            match query(index) {
                Ok(value) => {
                    let last = range(&value);
                    values.push(value);
                    if last {
                        return (values, None);
                    }
                }
                Err(error) if error.raw_os_error() == Some(libc::EINVAL) => return (values, None),
                Err(error) => return (values, Some(format!("index {index}: {error}"))),
            }
        }
        (values, Some(format!("enumeration stopped at the {limit}-entry safety limit")))
    }

    fn fraction(value: v4l::fraction::Fraction) -> Fraction {
        Fraction { numerator: value.numerator, denominator: value.denominator }
    }

    fn interval(value: FrameInterval) -> Result<Interval, String> {
        match value.interval {
            FrameIntervalEnum::Discrete(raw) if value.typ == DISCRETE => {
                let interval = fraction(raw);
                if !interval.valid() {
                    return Err("zero numerator or denominator in a discrete frame interval".into());
                }
                Ok(Interval::Discrete { fps: interval.inverse(), interval })
            }
            FrameIntervalEnum::Stepwise(raw) if value.typ == STEPWISE || value.typ == CONTINUOUS => {
                let (min, max, step) = (fraction(raw.min), fraction(raw.max), fraction(raw.step));
                if !min.valid() || !max.valid() || !step.valid() || !min.at_most(max) {
                    return Err("invalid rational bounds/step in a frame interval range".into());
                }
                if value.typ == CONTINUOUS {
                    Ok(Interval::Continuous { min, max, step })
                } else {
                    Ok(Interval::Stepwise { min, max, step })
                }
            }
            _ => Err(format!("unsupported frame interval type {}", value.typ)),
        }
    }

    fn enumerate(device: &impl Enumerate, format: &Format) -> CameraModes {
        let mut report = CameraModes {
            status: "ok",
            fourcc: String::from_utf8_lossy(&format.fourcc).into(),
            sizes: Vec::new(),
            diagnostics: Vec::new(),
        };
        let (formats, error) = scan(MAX_ITEMS, |index| device.format(index), |_| false);
        if let Some(error) = error {
            report.diagnostics.push(format!("ENUM_FMT: {error}"));
        }
        if !formats.contains(&format.fourcc) {
            report.status = if report.diagnostics.is_empty() { "unsupported" } else { "unavailable" };
            report.diagnostics.push(format!("{} was not reported by ENUM_FMT", format.name));
            return report;
        }
        // Complete the size enumeration before asking any size-specific intervals: interleaved ioctls can invalidate
        // the driver's enumeration snapshot (kernel V4L2 userspace API's consistency contract).
        let (sizes, error) = scan(MAX_ITEMS, |index| device.size(format.fourcc, index), |size| size.typ != DISCRETE);
        if let Some(error) = error {
            report.diagnostics.push(format!("ENUM_FRAMESIZES: {error}"));
        }
        if sizes.is_empty() {
            report.diagnostics.push("ENUM_FRAMESIZES reported no sizes for this format".into());
        }
        let mut remaining = MAX_INTERVALS;
        for size in sizes {
            match size.size {
                FrameSizeEnum::Discrete(raw) if size.typ == DISCRETE && raw.width != 0 && raw.height != 0 => {
                    let (values, error) = scan(
                        MAX_ITEMS.min(remaining),
                        |index| device.interval(format.fourcc, raw.width, raw.height, index),
                        |interval| interval.typ != DISCRETE,
                    );
                    remaining = remaining.saturating_sub(u32::try_from(values.len()).unwrap_or(remaining));
                    let mut diagnostics = Vec::new();
                    if let Some(error) = error {
                        diagnostics.push(format!("ENUM_FRAMEINTERVALS: {error}"));
                    }
                    if values.is_empty() {
                        diagnostics.push("no frame intervals reported for this size".into());
                    }
                    let intervals = values
                        .into_iter()
                        .filter_map(|value| match interval(value) {
                            Ok(value) => Some(value),
                            Err(why) => {
                                diagnostics.push(why);
                                None
                            }
                        })
                        .collect();
                    if !diagnostics.is_empty() {
                        report.status = "partial";
                    }
                    report.sizes.push(Size::Discrete { width: raw.width, height: raw.height, intervals, diagnostics });
                }
                FrameSizeEnum::Stepwise(raw) if size.typ == STEPWISE || size.typ == CONTINUOUS => {
                    if raw.min_width == 0
                        || raw.min_height == 0
                        || raw.max_width < raw.min_width
                        || raw.max_height < raw.min_height
                        || raw.step_width == 0
                        || raw.step_height == 0
                    {
                        report.diagnostics.push("invalid size-range bounds or zero step".into());
                        continue;
                    }
                    let range = if size.typ == CONTINUOUS {
                        Size::Continuous {
                            min_width: raw.min_width,
                            max_width: raw.max_width,
                            step_width: raw.step_width,
                            min_height: raw.min_height,
                            max_height: raw.max_height,
                            step_height: raw.step_height,
                        }
                    } else {
                        Size::Stepwise {
                            min_width: raw.min_width,
                            max_width: raw.max_width,
                            step_width: raw.step_width,
                            min_height: raw.min_height,
                            max_height: raw.max_height,
                            step_height: raw.step_height,
                        }
                    };
                    report.sizes.push(range);
                    report
                        .diagnostics
                        .push("frame intervals are size-specific and were not queried for symbolic size ranges".into());
                }
                _ => report.diagnostics.push(format!("invalid or unsupported frame size type {}", size.typ)),
            }
        }
        if !report.diagnostics.is_empty() {
            report.status = if report.sizes.is_empty() { "unavailable" } else { "partial" };
        }
        report
    }

    pub(super) fn probe(path: &Path, format: &Format) -> CameraModes {
        // A second read-only, nonblocking descriptor is sufficient for capability ioctls. Never request buffers,
        // STREAMON, S_FMT, S_PARM, or controls, so a live worker's format and stream are left alone.
        match OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK).open(path) {
            Ok(file) => enumerate(&Queries(file), format),
            Err(error) => CameraModes::unavailable(format, format!("open {} read-only: {error}", path.display())),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;
        use std::cell::RefCell;
        use v4l::{FourCC, fraction::Fraction as RawFraction};

        struct Fake {
            calls: RefCell<Vec<String>>,
            range_sizes: bool,
            supported: bool,
            format_error: Option<(u32, i32)>,
            size_error: Option<(u32, i32)>,
            interval_error: Option<(u32, i32)>,
        }
        impl Default for Fake {
            fn default() -> Self {
                Self {
                    calls: RefCell::new(Vec::new()),
                    range_sizes: false,
                    supported: true,
                    format_error: None,
                    size_error: None,
                    interval_error: None,
                }
            }
        }
        fn end() -> io::Error {
            io::Error::from_raw_os_error(libc::EINVAL)
        }
        fn fail(at: Option<(u32, i32)>, index: u32) -> io::Result<()> {
            if let Some((target, code)) = at
                && index == target
            {
                return Err(io::Error::from_raw_os_error(code));
            }
            Ok(())
        }
        fn raw_interval(typ: u32, interval: FrameIntervalEnum) -> FrameInterval {
            FrameInterval { index: 0, fourcc: FourCC::new(b"MJPG"), width: 1920, height: 1080, typ, interval }
        }
        impl Enumerate for Fake {
            fn format(&self, index: u32) -> io::Result<[u8; 4]> {
                self.calls.borrow_mut().push(format!("format:{index}"));
                fail(self.format_error, index)?;
                if index == 0 { Ok(if self.supported { *b"MJPG" } else { *b"UYVY" }) } else { Err(end()) }
            }
            fn size(&self, _fourcc: [u8; 4], index: u32) -> io::Result<FrameSize> {
                self.calls.borrow_mut().push(format!("size:{index}"));
                fail(self.size_error, index)?;
                if index != 0 {
                    return Err(end());
                }
                let size = if self.range_sizes {
                    FrameSizeEnum::Stepwise(v4l::framesize::Stepwise {
                        min_width: 1,
                        max_width: u32::MAX,
                        step_width: 1,
                        min_height: 1,
                        max_height: u32::MAX,
                        step_height: 1,
                    })
                } else {
                    FrameSizeEnum::Discrete(v4l::framesize::Discrete { width: 1920, height: 1080 })
                };
                Ok(FrameSize {
                    index,
                    fourcc: FourCC::new(b"MJPG"),
                    typ: if self.range_sizes { CONTINUOUS } else { DISCRETE },
                    size,
                })
            }
            fn interval(&self, _fourcc: [u8; 4], width: u32, height: u32, index: u32) -> io::Result<FrameInterval> {
                self.calls.borrow_mut().push(format!("interval:{width}:{height}:{index}"));
                fail(self.interval_error, index)?;
                if index != 0 {
                    return Err(end());
                }
                Ok(raw_interval(
                    DISCRETE,
                    FrameIntervalEnum::Discrete(RawFraction { numerator: 1001, denominator: 30000 }),
                ))
            }
        }

        #[test]
        fn discrete_projection_preserves_true_ntsc_rate_and_finishes_sizes_before_intervals() {
            let fake = Fake::default();
            let report = enumerate(&fake, &Format::parse("MJPEG").unwrap());
            let value = serde_json::to_value(report).unwrap();
            assert_eq!(value["status"], "ok");
            assert_eq!(
                value["sizes"][0],
                json!({"kind":"discrete","width":1920,"height":1080,"intervals":[{"kind":"discrete","interval":{"numerator":1001,"denominator":30000},"fps":{"numerator":30000,"denominator":1001}}],"diagnostics":[]})
            );
            assert_eq!(
                *fake.calls.borrow(),
                ["format:0", "format:1", "size:0", "size:1", "interval:1920:1080:0", "interval:1920:1080:1"]
            );
        }

        #[test]
        fn huge_continuous_size_ranges_remain_one_symbolic_record_without_interval_guesses() {
            let fake = Fake { range_sizes: true, ..Fake::default() };
            let value = serde_json::to_value(enumerate(&fake, &Format::parse("MJPEG").unwrap())).unwrap();
            assert_eq!(value["status"], "partial");
            assert_eq!(value["sizes"].as_array().unwrap().len(), 1);
            assert_eq!(value["sizes"][0]["kind"], "continuous");
            assert_eq!(value["sizes"][0]["max_width"], u32::MAX);
            assert_eq!(*fake.calls.borrow(), ["format:0", "format:1", "size:0"]);
            assert!(value["diagnostics"][0].as_str().unwrap().contains("not queried"));
        }

        #[test]
        fn interval_ranges_are_symbolic_and_invalid_rationals_are_diagnostics() {
            for typ in [STEPWISE, CONTINUOUS] {
                let value = interval(raw_interval(
                    typ,
                    FrameIntervalEnum::Stepwise(v4l::frameinterval::Stepwise {
                        min: RawFraction { numerator: 1, denominator: 60 },
                        max: RawFraction { numerator: 1, denominator: 5 },
                        step: RawFraction { numerator: 1, denominator: 120 },
                    }),
                ))
                .unwrap();
                let value = serde_json::to_value(value).unwrap();
                assert_eq!(value["kind"], if typ == STEPWISE { "stepwise" } else { "continuous" });
                assert_eq!(value["min"], json!({"numerator":1,"denominator":60}));
                assert!(value.get("fps").is_none());
            }
            for (numerator, denominator) in [(0, 30), (1, 0)] {
                assert!(
                    interval(raw_interval(
                        DISCRETE,
                        FrameIntervalEnum::Discrete(RawFraction { numerator, denominator })
                    ))
                    .unwrap_err()
                    .contains("zero")
                );
            }
            assert!(
                interval(raw_interval(
                    STEPWISE,
                    FrameIntervalEnum::Stepwise(v4l::frameinterval::Stepwise {
                        min: RawFraction { numerator: 1, denominator: 5 },
                        max: RawFraction { numerator: 1, denominator: 60 },
                        step: RawFraction { numerator: 1, denominator: 120 },
                    })
                ))
                .is_err()
            );
        }

        #[test]
        fn unsupported_formats_and_real_errors_do_not_become_successful_empty_modes() {
            let unsupported = Fake { supported: false, ..Fake::default() };
            assert_eq!(enumerate(&unsupported, &Format::parse("MJPEG").unwrap()).status, "unsupported");
            assert_eq!(unsupported.calls.borrow().len(), 2);
            let broken = Fake { format_error: Some((0, libc::EACCES)), ..Fake::default() };
            let report = enumerate(&broken, &Format::parse("MJPEG").unwrap());
            assert_eq!(report.status, "unavailable");
            assert!(report.diagnostics[0].contains("ENUM_FMT"));
            let broken = Fake { interval_error: Some((1, libc::EIO)), ..Fake::default() };
            let value = serde_json::to_value(enumerate(&broken, &Format::parse("MJPEG").unwrap())).unwrap();
            assert_eq!(value["status"], "partial");
            assert_eq!(value["sizes"][0]["intervals"].as_array().unwrap().len(), 1);
            assert!(value["sizes"][0]["diagnostics"][0].as_str().unwrap().contains("index 1"));
            let broken = Fake { size_error: Some((0, libc::ENOTTY)), ..Fake::default() };
            assert_eq!(enumerate(&broken, &Format::parse("MJPEG").unwrap()).status, "unavailable");
        }

        #[test]
        fn every_scan_is_bounded_and_einval_is_the_only_clean_end() {
            let mut calls = 0;
            let (values, diagnostic) = scan(
                3,
                |index| {
                    calls += 1;
                    Ok(index)
                },
                |_| false,
            );
            assert_eq!(calls, 3);
            assert_eq!(values, [0, 1, 2]);
            assert!(diagnostic.unwrap().contains("safety limit"));
            let (values, diagnostic) = scan::<u32>(3, |_| Err(end()), |_| false);
            assert!(values.is_empty());
            assert!(diagnostic.is_none());
            let (_, diagnostic) = scan::<u32>(3, |_| Err(io::Error::from_raw_os_error(libc::EIO)), |_| false);
            assert!(diagnostic.unwrap().contains("index 0"));
        }

        #[test]
        fn total_interval_budget_bounds_a_driver_that_never_ends_its_lists() {
            struct Endless(std::cell::Cell<u32>);
            impl Enumerate for Endless {
                fn format(&self, index: u32) -> io::Result<[u8; 4]> {
                    if index == 0 { Ok(*b"MJPG") } else { Err(end()) }
                }
                fn size(&self, _fourcc: [u8; 4], index: u32) -> io::Result<FrameSize> {
                    Ok(FrameSize {
                        index,
                        fourcc: FourCC::new(b"MJPG"),
                        typ: DISCRETE,
                        size: FrameSizeEnum::Discrete(v4l::framesize::Discrete { width: index + 1, height: 1080 }),
                    })
                }
                fn interval(&self, _fourcc: [u8; 4], width: u32, height: u32, index: u32) -> io::Result<FrameInterval> {
                    self.0.set(self.0.get() + 1);
                    let mut value = raw_interval(
                        DISCRETE,
                        FrameIntervalEnum::Discrete(RawFraction { numerator: 1, denominator: 30 }),
                    );
                    value.width = width;
                    value.height = height;
                    value.index = index;
                    Ok(value)
                }
            }
            let fake = Endless(std::cell::Cell::new(0));
            let report = enumerate(&fake, &Format::parse("MJPEG").unwrap());
            assert_eq!(fake.0.get(), 2048);
            assert_eq!(report.status, "partial");
            assert!(report.diagnostics.iter().any(|why| why.contains("safety limit")));
            let count: usize = report
                .sizes
                .iter()
                .map(|size| match size {
                    Size::Discrete { intervals, .. } => intervals.len(),
                    _ => 0,
                })
                .sum();
            assert_eq!(count, 2048);
        }

        #[test]
        fn failed_opens_and_non_camera_files_are_reported_without_touching_a_camera() {
            let root = tempfile::tempdir().unwrap();
            let format = Format::parse("MJPEG").unwrap();
            assert_eq!(probe(&root.path().join("absent"), &format).status, "unavailable");
            let path = root.path().join("ordinary-file");
            std::fs::write(&path, b"not a camera").unwrap();
            let report = probe(&path, &format);
            assert_eq!(report.status, "unavailable");
            assert!(report.diagnostics[0].contains("ENUM_FMT"));
            assert_eq!(std::fs::read(&path).unwrap(), b"not a camera");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).into()).collect()
    }

    #[test]
    fn text_list_stays_the_default_and_json_format_flags_are_order_independent() {
        assert!(!Options::parse(&args(&["--list"])).unwrap().json);
        let options = Options::parse(&args(&["--format", "mjpeg", "--json", "--list"])).unwrap();
        assert!(options.json);
        assert_eq!(options.format.name, "MJPEG");
        assert_eq!(options.format.fourcc, *b"MJPG");
        assert_eq!(Options::parse(&args(&["--list", "--json", "--format", "UYVY"])).unwrap().format.fourcc, *b"UYVY");
        assert_eq!(Options::parse(&args(&["--list", "--json", "--format", "Y16 "])).unwrap().format.fourcc, *b"Y16 ");
    }

    #[test]
    fn literal_fourcc_preserves_case() {
        for value in ["pBAA", "pgAA", "uyvy"] {
            let format = Format::parse(value).unwrap();
            assert_eq!(format.fourcc.as_slice(), value.as_bytes());
            assert_eq!(format.name, value);
        }
    }

    #[test]
    fn bad_flags_and_formats_are_rejected_before_discovery() {
        for values in [
            vec!["--list", "--wat"],
            vec!["--json"],
            vec!["--list", "--format", "UYVY"],
            vec!["--list", "--json", "--format"],
            vec!["--list", "--list"],
            vec!["--list", "--json", "--json"],
            vec!["--list", "--json", "--format", "longer"],
            vec!["--list", "--json", "--format", "éab"],
            vec!["--list", "--json", "--format", "    "],
        ] {
            assert!(Options::parse(&args(&values)).is_err(), "{values:?}");
        }
    }

    #[test]
    fn json_preserves_identity_and_other_devices_when_a_camera_cannot_be_opened() {
        let camera = DeviceInfo {
            key: "camera-key".into(),
            kind: DeviceKind::Camera,
            name: "Example camera".into(),
            usb: Some((0x123, 0xab)),
            node: "/dev/video99".into(),
            paths: vec!["/dev/v4l/by-id/example".into()],
        };
        assert_eq!(text(&camera), "Camera\tname \"Example camera\"\tusb 0123:00ab\tnode /dev/video99\tkey camera-key");
        let microphone = DeviceInfo {
            key: "mic-key".into(),
            kind: DeviceKind::Microphone,
            name: "Example mic".into(),
            usb: None,
            node: "mic-node".into(),
            paths: Vec::new(),
        };
        let format = Format::parse("MJPEG").unwrap();
        let mut queries = 0;
        let text = encode(&[microphone, camera], vec!["optional registry unavailable".into()], &format, |_, format| {
            queries += 1;
            CameraModes::unavailable(format, "permission denied".into())
        })
        .unwrap();
        assert_eq!(queries, 1, "microphones are not opened to query camera modes");
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["format"], "MJPEG");
        assert_eq!(value["fourcc"], "MJPG");
        assert_eq!(value["devices"][0]["key"], "camera-key");
        assert_eq!(value["devices"][0]["usb"], "0123:00ab");
        assert_eq!(value["devices"][0]["paths"][0], "/dev/v4l/by-id/example");
        assert_eq!(value["devices"][0]["camera_modes"]["status"], "unavailable");
        assert_eq!(value["devices"][1]["kind"], "microphone");
        assert!(value["devices"][1].get("camera_modes").is_none());
        assert!(value["devices"][1]["usb"].is_null());
        assert_eq!(value["diagnostics"][0], "optional registry unavailable");
    }
}
