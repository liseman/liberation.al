//! The codec probe the worker runs at start, so a missing codec shows up once in the log instead of at runtime.
//!
//! Without the `ffmpeg` feature the probe reports the built-in codecs only (LZ4, FLAC, JPEG decode) and says that
//! hardware codecs need the feature. With it, it asks FFmpeg which of the encoders and decoders the tiers use are
//! present (docs/CONTINUUM-CAPTURE.md).

/// What the worker can encode and decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Probe {
    /// LZ4 for raw frames (always, pure Rust).
    pub lz4: bool,
    /// FLAC for PCM (always, pure Rust).
    pub flac: bool,
    /// JPEG decode on the CPU (always, pure Rust).
    pub jpeg_decode: bool,
    /// FFmpeg, when built with the `ffmpeg` feature: the version and the codecs of interest that are present.
    pub ffmpeg: Option<Ffmpeg>,
}

/// What FFmpeg offers here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ffmpeg {
    /// libavcodec's version string.
    pub avcodec_version: String,
    /// Encoders present, by FFmpeg name (`h264_nvenc`, `h264_videotoolbox`, `libx264`, ...).
    pub encoders: Vec<String>,
    /// Decoders present, by FFmpeg name (`h264_cuvid`, `h264`, `mjpeg`, ...).
    pub decoders: Vec<String>,
}

/// The encoders the tiers use, in the order the worker prefers them.
pub const ENCODERS_OF_INTEREST: &[&str] =
    &["h264_nvenc", "h264_videotoolbox", "libx264", "libopenh264", "libsvtav1", "flac", "libopus"];

/// The decoders the tiers use.
pub const DECODERS_OF_INTEREST: &[&str] = &["h264_cuvid", "h264", "libdav1d", "mjpeg", "flac", "libopus"];

/// Runs the probe. Cheap and read-only: it opens no device and encodes nothing.
#[must_use]
pub fn probe() -> Probe {
    Probe { lz4: true, flac: true, jpeg_decode: true, ffmpeg: probe_ffmpeg() }
}

#[cfg(feature = "ffmpeg")]
fn probe_ffmpeg() -> Option<Ffmpeg> {
    drop(crate::encode::codec_open().ok()?);
    let present =
        |names: &[&str], find: fn(&str) -> bool| names.iter().filter(|n| find(n)).map(|n| (*n).to_owned()).collect();
    Some(Ffmpeg {
        avcodec_version: ffmpeg_next::codec::version().to_string(),
        encoders: present(ENCODERS_OF_INTEREST, |n| ffmpeg_next::encoder::find_by_name(n).is_some()),
        decoders: present(DECODERS_OF_INTEREST, |n| ffmpeg_next::decoder::find_by_name(n).is_some()),
    })
}

#[cfg(not(feature = "ffmpeg"))]
fn probe_ffmpeg() -> Option<Ffmpeg> {
    None
}

/// Whether this FFmpeg can encode the AV1 archive (SVT-AV1). The wheels' private FFmpeg has it
/// (docs/CONTINUUM-SDK.md), and so does a distribution's.
#[must_use]
pub fn av1_encoder_available() -> bool {
    #[cfg(feature = "ffmpeg")]
    {
        crate::encode::codec_open().is_ok() && ffmpeg_next::encoder::find_by_name("libsvtav1").is_some()
    }
    #[cfg(not(feature = "ffmpeg"))]
    {
        false
    }
}

impl Probe {
    /// One line for the log.
    #[must_use]
    pub fn summary(&self) -> String {
        match &self.ffmpeg {
            Some(f) => format!(
                "codecs: lz4, flac, jpeg (cpu); ffmpeg avcodec {}: encoders [{}], decoders [{}]",
                f.avcodec_version,
                f.encoders.join(", "),
                f.decoders.join(", ")
            ),
            None => "codecs: lz4, flac, jpeg (cpu); no ffmpeg (built without the ffmpeg feature): H.264 and hardware codecs unavailable".to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_builtin_codecs_are_always_there() {
        let p = probe();
        assert!(p.lz4 && p.flac && p.jpeg_decode);
        assert!(p.summary().starts_with("codecs: lz4, flac, jpeg (cpu);"));
    }
}
