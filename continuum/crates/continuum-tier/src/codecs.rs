//! Payload transforms that need no FFmpeg: LZ4 for raw frames, FLAC for PCM, and the passthrough for payloads that
//! are already compressed (MJPEG frames, Arrow IPC that the producer compressed).
//!
//! H.264 and the hardware codecs are behind the `ffmpeg` feature in [`crate::probe`] and the modules it gates.

use flacenc::component::BitRepr;
use flacenc::error::Verify;

use crate::TierError;

/// LZ4's worst-case expansion: a block never decompresses to more than this many times its size (the format's
/// literal run encoding bounds it at 255 to 1, plus a small constant).
const LZ4_MAX_RATIO: usize = 255;
const LZ4_SLACK: usize = 64;

/// Compresses a raw payload with LZ4, prefixing its length so [`lz4_decompress`] needs nothing else.
#[must_use]
pub fn lz4_compress(raw: &[u8]) -> Vec<u8> {
    lz4_flex::compress_prepend_size(raw)
}

/// The inverse of [`lz4_compress`]. The length prefix is bounded by what LZ4 can expand to, so a corrupt or
/// hostile prefix can't make this allocate gigabytes.
///
/// # Errors
///
/// [`TierError::Codec`] for bytes that aren't an LZ4 block with a plausible length prefix.
pub fn lz4_decompress(bytes: &[u8]) -> Result<Vec<u8>, TierError> {
    if bytes.len() < 4 {
        return Err(TierError::Codec("lz4: no length prefix".to_owned()));
    }
    let declared = u32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes")) as usize;
    let bound = bytes.len().saturating_mul(LZ4_MAX_RATIO).saturating_add(LZ4_SLACK);
    if declared > bound {
        return Err(TierError::Codec(format!(
            "lz4: declared size {declared} exceeds what {} bytes can hold",
            bytes.len()
        )));
    }
    lz4_flex::decompress(&bytes[4..], declared).map_err(|e| TierError::Codec(format!("lz4: {e}")))
}

/// Interleaved PCM as FLAC. Samples are `i32` holding `bits_per_sample` significant bits (16 for S16, 24 for
/// S24), `channels` interleaved, at `sample_rate` Hz.
///
/// # Errors
///
/// [`TierError::Codec`] when the encoder refuses the parameters (an unsupported bit depth, a sample count that
/// isn't a multiple of the channel count).
pub fn flac_encode(
    samples: &[i32],
    channels: usize,
    bits_per_sample: usize,
    sample_rate: usize,
) -> Result<Vec<u8>, TierError> {
    if channels == 0 || !samples.len().is_multiple_of(channels) {
        return Err(TierError::Codec(format!("flac: {} samples do not split into {channels} channels", samples.len())));
    }
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|e| TierError::Codec(format!("flac config: {e:?}")))?;
    let source = flacenc::source::MemSource::from_samples(samples, channels, bits_per_sample, sample_rate);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| TierError::Codec(format!("flac: {e:?}")))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream.write(&mut sink).map_err(|e| TierError::Codec(format!("flac write: {e:?}")))?;
    Ok(sink.as_slice().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz4_round_trips_and_shrinks_repetitive_frames() {
        let frame: Vec<u8> = (0..64 * 1024).map(|i| u8::try_from((i / 256) % 7).expect("small")).collect();
        let packed = lz4_compress(&frame);
        assert!(packed.len() < frame.len() / 10, "{} vs {}", packed.len(), frame.len());
        assert_eq!(lz4_decompress(&packed).expect("decompress"), frame);
        assert!(lz4_decompress(b"not lz4").is_err());
        assert_eq!(lz4_decompress(&lz4_compress(&[])).expect("empty"), Vec::<u8>::new());
    }

    #[test]
    fn an_lz4_bomb_is_refused_without_allocating() {
        let bomb = [0xff, 0xff, 0xff, 0xff, 0x00];
        let err = lz4_decompress(&bomb).expect_err("refused");
        assert!(err.to_string().contains("exceeds"), "{err}");
    }

    #[test]
    fn flac_encodes_stereo_s16_and_refuses_a_ragged_frame() {
        let samples: Vec<i32> = (0..4800i32).map(|i| ((i % 100) - 50) * 100).collect();
        let flac = flac_encode(&samples, 2, 16, 48_000).expect("encode");
        assert_eq!(&flac[..4], b"fLaC");
        assert!(flac.len() < samples.len() * 2);
        assert!(flac_encode(&samples[..4799], 2, 16, 48_000).is_err());
        assert!(flac_encode(&samples, 0, 16, 48_000).is_err());
    }
}
