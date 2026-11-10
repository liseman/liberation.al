//! The V4L2 frame source (Linux): opens a camera at the first mode on its fallback ladder, holds its frame rate if
//! asked, and streams memory-mapped buffers. Only the parts that need a device live here; the choices it makes are
//! in `video` and `stamp`, tested without one. No test in this crate opens a device.

use std::io;
use std::path::Path;
use std::time::Duration;

use v4l::buffer::Type;
use v4l::control::{Control, Value};
use v4l::framesize::FrameSizeEnum;
use v4l::io::traits::CaptureStream;
use v4l::prelude::{Device, MmapStream};
use v4l::video::Capture;
use v4l::video::capture::Parameters;
use v4l::{Format, FourCC};

use crate::manifest::VideoSettings;
pub use crate::video::Opened;
use crate::video::{Camera, Controls, FrameSource, Mode, RawFrame, fourcc, hold_frame_rate, ladder};

/// Driver buffers: enough that a slow publish never starves the driver, few enough to keep latency low.
const BUFFERS: u32 = 4;

/// The first frame after `VIDIOC_STREAMON` can take a while (the camera settles its exposure); longer than this is
/// a failed open.
pub const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(10);

/// A V4L2 camera, streaming.
pub struct V4l2Source {
    // Dropped before the device: the stream unmaps its buffers and turns streaming off.
    stream: MmapStream<'static>,
    device: Device,
    started: bool,
    /// The first frame, taken by [`V4l2Source::start`], handed out by the first `next_frame`.
    first: Option<OwnedFrame>,
}

/// A frame copied out of the driver's buffer.
struct OwnedFrame {
    bytes: Vec<u8>,
    tv_sec: i64,
    tv_usec: i64,
    flags: u32,
    sequence: u32,
}

impl OwnedFrame {
    fn frame(&self) -> RawFrame<'_> {
        RawFrame {
            bytes: &self.bytes,
            tv_sec: self.tv_sec,
            tv_usec: self.tv_usec,
            flags: self.flags,
            sequence: self.sequence,
            // V4L2 buffers carry no presentation duration.
            duration_ns: 0,
        }
    }
}

/// Whether a stepwise size range includes `want`.
#[must_use]
pub fn stepwise_includes(min: u32, max: u32, step: u32, want: u32) -> bool {
    want >= min && want <= max && (step == 0 || (want - min).is_multiple_of(step))
}

impl V4l2Source {
    /// Opens the camera at `path` in the manifest's format at the first size on its ladder the camera both offers
    /// and accepts, asks for the manifest's frame rate, and holds it in dim light if asked.
    ///
    /// # Errors
    ///
    /// The device's error, or `Unsupported` when it offers none of the ladder's sizes in the format.
    pub fn open(path: &Path, video: &VideoSettings) -> io::Result<(Self, Opened)> {
        let device = Device::with_path(path)?;
        let want = fourcc(&video.format);
        let offered = offered_modes(&device, want, &ladder(video))?;
        let mut chosen = None;
        for size in ladder(video) {
            if !offered.iter().any(|m| (m.width, m.height) == size) {
                continue;
            }
            let format = device.set_format(&Format::new(size.0, size.1, FourCC::new(&want)))?;
            if (format.width, format.height, format.fourcc.repr) == (size.0, size.1, want) {
                chosen = Some(format);
                break;
            }
        }
        let format = chosen.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                format!("{} offers none of {:?} in {}", path.display(), ladder(video), video.format),
            )
        })?;
        let params = device.set_params(&Parameters::with_fps(video.fps.max(1)))?;
        let frame_interval_ns = if params.interval.denominator == 0 {
            1_000_000_000 / u64::from(video.fps.max(1))
        } else {
            u64::from(params.interval.numerator) * 1_000_000_000 / u64::from(params.interval.denominator)
        };
        let hold = if video.hold_fps { hold_frame_rate(&mut DeviceControls(&device)) } else { None };
        let mut stream = MmapStream::with_buffers(&device, Type::VideoCapture, BUFFERS)?;
        stream.set_timeout(FIRST_FRAME_TIMEOUT);
        let opened = Opened {
            shape: video.format.clone(),
            mode: Mode { fourcc: want, width: format.width, height: format.height },
            stride: format.stride,
            max_frame_bytes: format.size,
            frame_interval_ns,
            hold,
        };
        Ok((Self { stream, device, started: false, first: None }, opened))
    }

    /// Starts streaming and waits for the first frame (up to [`FIRST_FRAME_TIMEOUT`]), keeping it for the first
    /// `next_frame`. Only after this has the camera claimed its USB bandwidth, so the worker says it's open after
    /// this, not after `open`: the next camera's open waits until then.
    ///
    /// # Errors
    ///
    /// The device's error, or `TimedOut` if no first frame came.
    pub fn start(&mut self) -> io::Result<()> {
        if self.started {
            return Ok(());
        }
        let mut first = None;
        self.next_frame(0, &mut |raw| {
            first = Some(OwnedFrame {
                bytes: raw.bytes.to_vec(),
                tv_sec: raw.tv_sec,
                tv_usec: raw.tv_usec,
                flags: raw.flags,
                sequence: raw.sequence,
            });
        })?;
        self.first = first;
        Ok(())
    }
}

/// The modes the camera offers in `want`, at the ladder's sizes (a stepwise range is checked, not expanded).
fn offered_modes(device: &Device, want: [u8; 4], ladder: &[(u32, u32)]) -> io::Result<Vec<Mode>> {
    let mut out = Vec::new();
    if !device.enum_formats()?.iter().any(|d| d.fourcc.repr == want) {
        return Ok(out);
    }
    for size in device.enum_framesizes(FourCC::new(&want))? {
        match size.size {
            FrameSizeEnum::Discrete(d) => out.push(Mode { fourcc: want, width: d.width, height: d.height }),
            FrameSizeEnum::Stepwise(s) => {
                for &(width, height) in ladder {
                    if stepwise_includes(s.min_width, s.max_width, s.step_width, width)
                        && stepwise_includes(s.min_height, s.max_height, s.step_height, height)
                    {
                        out.push(Mode { fourcc: want, width, height });
                    }
                }
            }
        }
    }
    Ok(out)
}

struct DeviceControls<'d>(&'d Device);

impl Controls for DeviceControls<'_> {
    fn set(&mut self, id: u32, value: i64) -> bool {
        self.0.set_control(Control { id, value: Value::Integer(value) }).is_ok()
    }
}

impl Camera for V4l2Source {
    fn start(&mut self) -> io::Result<()> {
        V4l2Source::start(self)
    }
}

impl FrameSource for V4l2Source {
    fn next_frame(&mut self, timeout_ms: u32, on_frame: &mut dyn FnMut(&RawFrame<'_>)) -> io::Result<bool> {
        // After the first frame, wait here rather than in the crate's dequeue: a dequeue that times out leaves the
        // crate ready to queue the same buffer again, which the driver refuses. With one buffer held and the rest
        // queued, a readable device means a frame is waiting, and `next` doesn't block.
        if let Some(first) = self.first.take() {
            on_frame(&first.frame());
            return Ok(true);
        }
        if self.started {
            let ready = match self.device.handle().poll(libc::POLLIN, i32::try_from(timeout_ms).unwrap_or(i32::MAX)) {
                Ok(n) => n,
                // A signal (the supervisor's stop) interrupted the wait: no frame, and the loop checks `stop`.
                Err(e) if e.kind() == io::ErrorKind::Interrupted => return Ok(false),
                Err(e) => return Err(e),
            };
            if ready == 0 {
                return Ok(false);
            }
        }
        let (bytes, meta) = match self.stream.next() {
            Ok(frame) => frame,
            // Only the first frame can time out here (FIRST_FRAME_TIMEOUT): the camera didn't start.
            Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "the camera sent no first frame"));
            }
            Err(e) => return Err(e),
        };
        self.started = true;
        let used = (meta.bytesused as usize).min(bytes.len());
        on_frame(&RawFrame {
            bytes: &bytes[..used],
            tv_sec: meta.timestamp.sec,
            tv_usec: meta.timestamp.usec,
            flags: meta.flags.bits(),
            sequence: meta.sequence,
            // V4L2 buffers carry no presentation duration.
            duration_ns: 0,
        });
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stepwise_range_includes_sizes_on_its_steps() {
        assert!(stepwise_includes(320, 1920, 8, 1280));
        assert!(!stepwise_includes(320, 1920, 8, 1284));
        assert!(!stepwise_includes(320, 1920, 8, 2560));
        assert!(stepwise_includes(640, 640, 0, 640), "a zero step is a single size");
    }
}
