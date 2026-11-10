//! A camera's raw video in Tier 1: short-GOP H.264 instead of LZ4.
//!
//! A camera that can't send MJPEG (the Mac's, through AVFoundation, give UYVY) would otherwise fill Tier 1 with raw
//! frames: 1280x720 UYVY at 30 fps is about 200 GB an hour, and LZ4 barely shrinks camera noise (about 120 GB). So the
//! tier worker encodes each raw frame as it lands ([`RawEncoder`]): H.264 with a keyframe every second, low delay (no
//! lookahead or B-frames holding frames back), hardware first (NVENC on the Linux machine's GPU, VideoToolbox on the Mac, then
//! libx264), about a gigabyte an hour. Each access unit is one record of a [`crate::ChunkFormat::H264Au`] chunk (see
//! [`crate::worker`]). H.264 rather than HEVC so a browser plays it.
//!
//! A frame's layout comes from its header's capture info (width, height, and stride; the stream's shape names the
//! pixel format): rows are indexed by stride, which AVFoundation pads. 4:2:2 chroma is averaged over row pairs into
//! 4:2:0. The camera's limited range and BT.601 colors are kept as they came and signalled in the stream. MJPEG streams
//! are untouched: they're compressed already. A worker built without the `ffmpeg` feature, or where no H.264 encoder
//! opens, stores raw frames as LZ4 and says so.

use crate::TierError;

/// A raw pixel format a camera sends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RawFormat {
    /// Packed 4:2:2, `U Y0 V Y1` (AVFoundation's `2vuy`).
    Uyvy,
    /// Packed 4:2:2, `Y0 U Y1 V` (V4L2's `YUYV`).
    Yuyv,
    /// 4:2:0, a luma plane then interleaved chroma.
    Nv12,
}

impl RawFormat {
    /// The format a stream's shape names, if it's one of these.
    #[must_use]
    pub fn of_shape(shape: &str) -> Option<Self> {
        match shape {
            "UYVY" | "2vuy" => Some(Self::Uyvy),
            "YUYV" | "YUY2" => Some(Self::Yuyv),
            "NV12" => Some(Self::Nv12),
            _ => None,
        }
    }
}

/// A raw frame's layout: its format, size, and bytes per row (of the luma plane, for NV12).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RawLayout {
    /// The pixel format.
    pub format: RawFormat,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Bytes per row; at least the row's own bytes.
    pub stride: u32,
}

impl RawLayout {
    /// The layout from a stream's shape and a frame's capture info, or `None` for a shape that isn't raw video, an
    /// odd or zero size (4:2:0 needs even ones), or a stride too short for a row.
    #[must_use]
    pub fn of(shape: &str, width: u32, height: u32, stride: u32) -> Option<Self> {
        let format = RawFormat::of_shape(shape)?;
        if width == 0 || height == 0 || width % 2 == 1 || height % 2 == 1 {
            return None;
        }
        let row = match format {
            RawFormat::Uyvy | RawFormat::Yuyv => width.checked_mul(2)?,
            RawFormat::Nv12 => width,
        };
        let stride = if stride == 0 { row } else { stride };
        (stride >= row).then_some(Self { format, width, height, stride })
    }

    /// The bytes a frame of this layout takes.
    #[must_use]
    pub fn frame_bytes(&self) -> usize {
        let (stride, h) = (self.stride as usize, self.height as usize);
        match self.format {
            RawFormat::Uyvy | RawFormat::Yuyv => stride * h,
            // The chroma plane follows the luma plane at the same stride, half as many rows.
            RawFormat::Nv12 => stride * h + stride * (h / 2),
        }
    }
}

/// A raw frame as tightly packed NV12 (`width * height * 3 / 2` bytes), what the encoder takes. 4:2:2 chroma is
/// averaged over each pair of rows; nothing else changes (the range stays as it came).
///
/// # Errors
///
/// [`TierError::Codec`] for a frame shorter than its layout says.
pub fn to_nv12(layout: &RawLayout, frame: &[u8]) -> Result<Vec<u8>, TierError> {
    if frame.len() < layout.frame_bytes() {
        return Err(TierError::Codec(format!(
            "a {:?} {}x{} frame is {} bytes, not {}",
            layout.format,
            layout.width,
            layout.height,
            layout.frame_bytes(),
            frame.len()
        )));
    }
    let (w, h, stride) = (layout.width as usize, layout.height as usize, layout.stride as usize);
    let mut out = vec![0u8; w * h * 3 / 2];
    let (luma, chroma) = out.split_at_mut(w * h);
    match layout.format {
        RawFormat::Nv12 => {
            for row in 0..h {
                luma[row * w..(row + 1) * w].copy_from_slice(&frame[row * stride..row * stride + w]);
            }
            let plane = &frame[stride * h..];
            for row in 0..h / 2 {
                chroma[row * w..(row + 1) * w].copy_from_slice(&plane[row * stride..row * stride + w]);
            }
        }
        RawFormat::Uyvy | RawFormat::Yuyv => {
            // Byte offsets in each 4-byte macropixel: Y0, U, Y1, V.
            let (y0, u, y1, v) = if layout.format == RawFormat::Uyvy { (1, 0, 3, 2) } else { (0, 1, 2, 3) };
            for row in 0..h {
                let src = &frame[row * stride..row * stride + w * 2];
                let dst = &mut luma[row * w..(row + 1) * w];
                for (px, m) in src.chunks_exact(4).enumerate() {
                    dst[2 * px] = m[y0];
                    dst[2 * px + 1] = m[y1];
                }
            }
            for row in 0..h / 2 {
                let a = &frame[(2 * row) * stride..(2 * row) * stride + w * 2];
                let b = &frame[(2 * row + 1) * stride..(2 * row + 1) * stride + w * 2];
                let dst = &mut chroma[row * w..(row + 1) * w];
                for (px, (ma, mb)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
                    dst[2 * px] = (u16::from(ma[u]) + u16::from(mb[u])).div_ceil(2) as u8;
                    dst[2 * px + 1] = (u16::from(ma[v]) + u16::from(mb[v])).div_ceil(2) as u8;
                }
            }
        }
    }
    Ok(out)
}

/// One stream's all-intra encoder (the `ffmpeg` feature).
/// A raw camera's short-GOP H.264 encoder: frames of one layout in, access units out, low delay.
#[cfg(feature = "ffmpeg")]
pub struct RawEncoder {
    layout: RawLayout,
    encoder: crate::h264::H264Encoder,
}

#[cfg(feature = "ffmpeg")]
impl RawEncoder {
    /// Opens an encoder for frames of `layout` at about `frame_rate` a second, a keyframe every second, low delay,
    /// limited range (the camera's), at [`bit_rate`] for its size: hardware first ([`crate::h264::H264_ENCODERS`]).
    ///
    /// # Errors
    ///
    /// [`TierError::Codec`] when no H.264 encoder opens.
    pub fn open(layout: RawLayout, frame_rate: u32, names: &[&str]) -> Result<Self, TierError> {
        let rate = frame_rate.max(1);
        let settings = crate::h264::H264Settings {
            gop_frames: rate,
            frame_rate: rate,
            bit_rate: bit_rate(layout.width, layout.height, rate),
            full_range: false,
            low_delay: true,
            threads: 0,
        };
        let encoder = crate::h264::H264Encoder::open_with(names, layout.width, layout.height, settings)?;
        Ok(Self { layout, encoder })
    }

    /// The layout it encodes.
    #[must_use]
    pub fn layout(&self) -> RawLayout {
        self.layout
    }

    /// The encoder in use, like `h264_nvenc`.
    #[must_use]
    pub fn encoder_name(&self) -> &str {
        self.encoder.encoder_name()
    }

    /// Encodes one raw frame; returns the access units ready so far, in order, each with its frame's times (an
    /// encoder may hold the first few back).
    ///
    /// # Errors
    ///
    /// A frame that doesn't fit the layout, or an encoder failure.
    pub fn encode(
        &mut self,
        frame: &[u8],
        start_ns: u64,
        end_ns: u64,
    ) -> Result<Vec<crate::encode::AccessUnit>, TierError> {
        let nv12 = to_nv12(&self.layout, frame)?;
        self.encoder.encode(&nv12, start_ns, end_ns)
    }

    /// Flushes the encoder: the access units of frames it still held.
    ///
    /// # Errors
    ///
    /// An encoder failure.
    pub fn finish(self) -> Result<Vec<crate::encode::AccessUnit>, TierError> {
        self.encoder.finish()
    }
}

/// The target bit rate for short-GOP frames of `width` x `height` at `frame_rate`, in bits a second: `BITS_PER_PIXEL`
/// thousandths of a bit a pixel.
#[must_use]
pub fn bit_rate(width: u32, height: u32, frame_rate: u32) -> usize {
    let pixels = u64::from(width) * u64::from(height) * u64::from(frame_rate.max(1));
    usize::try_from(pixels * BITS_PER_PIXEL / 1000).unwrap_or(usize::MAX)
}

/// Thousandths of a bit a pixel for a raw camera's short-GOP H.264 (a P-frame needs far fewer than an intra one).
pub const BITS_PER_PIXEL: u64 = 50;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uyvy_and_yuyv_become_nv12_with_chroma_averaged_over_row_pairs() {
        // 4x2: each macropixel's bytes say which row and pixel pair they are.
        let layout = RawLayout::of("UYVY", 4, 2, 0).expect("layout");
        let frame = [10, 1, 20, 2, 30, 3, 40, 4, 12, 5, 22, 6, 32, 7, 42, 8];
        let nv12 = to_nv12(&layout, &frame).expect("nv12");
        assert_eq!(&nv12[..8], &[1, 2, 3, 4, 5, 6, 7, 8], "luma");
        assert_eq!(&nv12[8..], &[11, 21, 31, 41], "U and V averaged over the two rows");
        let yuyv = RawLayout::of("YUYV", 4, 2, 0).expect("layout");
        let frame = [1, 10, 2, 20, 3, 30, 4, 40, 5, 12, 6, 22, 7, 32, 8, 42];
        assert_eq!(to_nv12(&yuyv, &frame).expect("nv12"), nv12);
    }

    #[test]
    fn a_padded_stride_is_skipped_and_a_short_frame_refused() {
        let layout = RawLayout::of("UYVY", 2, 2, 8).expect("layout");
        let frame = [9, 1, 9, 2, 0xee, 0xee, 0xee, 0xee, 9, 3, 9, 4, 0xee, 0xee, 0xee, 0xee];
        assert_eq!(to_nv12(&layout, &frame).expect("nv12"), [1, 2, 3, 4, 9, 9]);
        assert!(to_nv12(&layout, &frame[..15]).is_err());
        let nv12 = RawLayout::of("NV12", 2, 2, 4).expect("layout");
        assert_eq!(to_nv12(&nv12, &[1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0]).expect("nv12"), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn only_raw_shapes_with_even_sizes_and_room_for_a_row_have_a_layout() {
        assert_eq!(RawLayout::of("MJPEG", 1280, 720, 0), None);
        assert_eq!(RawLayout::of("UYVY", 1279, 720, 0), None);
        assert_eq!(RawLayout::of("UYVY", 1280, 720, 1000), None, "a stride shorter than a row");
        assert_eq!(RawLayout::of("UYVY", 1280, 720, 0).map(|l| l.stride), Some(2560));
    }
}
