//! The 16 byte header at the start of every binary message, the same over WebSockets and WebTransport.
//!
//! | Bytes | Field | Meaning |
//! |---|---|---|
//! | 0 to 3 | `stream_id: u32` | Which stream |
//! | 4 | `payload_type: u8` | `0x01` H.264 or HEVC access unit (Annex B), `0x02` FlatBuffer overlay, `0x03` Arrow table, `0x04` JPEG frame |
//! | 5 | `flags: u8` | Bit 0: keyframe, bit 1: live, bit 2: rotate camera pixels 180 degrees, bit 3: a media context id follows, bit 4: from the `watch` set, bit 5: a request id follows (#2380) |
//! | 6 to 7 | reserved | Zero, so `capture_us` stays 8 byte aligned |
//! | 8 to 15 | `capture_us: u64` | Capture time in microseconds UTC (the payload's `start_ns` divided by 1000) |
//!
//! After the 16 bytes, a message can carry up to two u32 extensions, in this order: the media context id (bit 3), then
//! the request id (bit 5), the `req` of the request that sent it.
//!
//! Integers are little-endian. The browser reads `capture_us` with `DataView.getBigUint64(8, true)`; as a UTC
//! microsecond count it fits a JavaScript number exactly until the year 2255. The TypeScript side of this lives in
//! `frontend/src/dashboard/player/header.ts`, and both have the same test vectors.

use std::fmt;

/// The header's size in bytes.
pub const HEADER_LEN: usize = 16;
/// Bit 0 of `flags`: the access unit starts with a keyframe (every JPEG frame is one).
pub const FLAG_KEYFRAME: u8 = 0x01;
/// Bit 1 of `flags`: the frame comes from a live subscription, not a `range` (the two can interleave on a connection).
pub const FLAG_LIVE: u8 = 0x02;
/// Bit 2 of `flags`: rotate picture pixels 180 degrees before drawing upright vision overlays.
pub const FLAG_FLIP: u8 = 0x04;
/// Bit 3: a little-endian media context ID follows the ordinary header.
pub const FLAG_MEDIA_CONTEXT: u8 = 0x08;
/// Bit 4: the frame comes from the connection's `watch` set, not its `live` feed (both set the live bit, #2380).
pub const FLAG_WATCH: u8 = 0x10;
/// Bit 5: a little-endian request ID (the request's `req`) follows the header, after any media context ID (#2380).
pub const FLAG_REQUEST: u8 = 0x20;

/// What a binary message carries after its header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PayloadType {
    /// One H.264 or HEVC access unit in Annex B form (start codes, in-band parameter sets on keyframes).
    Video = 0x01,
    /// A FlatBuffer `Overlay` (`schemas/overlay.fbs`): boxes, skeletons, polylines, and captions for one frame.
    Overlay = 0x02,
    /// An Arrow IPC table (a batch of IMU samples, SLAM poses, or any tabular derived data).
    Arrow = 0x03,
    /// One JPEG frame from an MJPEG camera.
    Jpeg = 0x04,
}

impl PayloadType {
    /// The type for a tag byte, if it is one.
    #[must_use]
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0x01 => Some(Self::Video),
            0x02 => Some(Self::Overlay),
            0x03 => Some(Self::Arrow),
            0x04 => Some(Self::Jpeg),
            _ => None,
        }
    }

    /// Every type, for tests that cover each.
    pub const ALL: [Self; 4] = [Self::Video, Self::Overlay, Self::Arrow, Self::Jpeg];
}

/// Why bytes aren't a header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HeaderError {
    /// Fewer than 16 bytes.
    #[error("a frame needs a 16 byte header, got {0} bytes")]
    Short(usize),
    /// An unknown payload type tag.
    #[error("unknown payload type {0:#04x}")]
    UnknownType(u8),
    /// The reserved bytes weren't zero (a message from a newer gateway, or not a frame at all).
    #[error("reserved header bytes are not zero")]
    Reserved,
}

/// The decoded header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameHeader {
    /// Which stream the payload belongs to.
    pub stream_id: u32,
    /// What follows the header.
    pub payload_type: PayloadType,
    /// Whether the payload is a keyframe (meaningful for video; always set for JPEG).
    pub keyframe: bool,
    /// From a live subscription, not a `range`.
    pub live: bool,
    /// Rotate picture pixels 180 degrees (not a mirror). Vision overlays are already upright.
    pub flip: bool,
    /// The capture time, microseconds UTC.
    pub capture_us: u64,
}

impl FrameHeader {
    /// A header for a payload captured at `start_ns` (nanoseconds UTC, the catalog's unit).
    #[must_use]
    pub fn at_ns(stream_id: u32, payload_type: PayloadType, keyframe: bool, start_ns: u64) -> Self {
        Self { stream_id, payload_type, keyframe, live: false, flip: false, capture_us: start_ns / 1_000 }
    }

    /// The 16 bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..4].copy_from_slice(&self.stream_id.to_le_bytes());
        b[4] = self.payload_type as u8;
        b[5] = if self.keyframe { FLAG_KEYFRAME } else { 0 }
            | if self.live { FLAG_LIVE } else { 0 }
            | if self.flip { FLAG_FLIP } else { 0 };
        b[8..16].copy_from_slice(&self.capture_us.to_le_bytes());
        b
    }

    /// Reads a header from the start of a message.
    ///
    /// # Errors
    ///
    /// [`HeaderError`] for a short message, an unknown type, or reserved bytes that aren't zero.
    pub fn decode(bytes: &[u8]) -> Result<Self, HeaderError> {
        if bytes.len() < HEADER_LEN {
            return Err(HeaderError::Short(bytes.len()));
        }
        if bytes[6] != 0 || bytes[7] != 0 {
            return Err(HeaderError::Reserved);
        }
        let payload_type = PayloadType::from_u8(bytes[4]).ok_or(HeaderError::UnknownType(bytes[4]))?;
        Ok(Self {
            stream_id: u32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes")),
            payload_type,
            keyframe: bytes[5] & FLAG_KEYFRAME != 0,
            live: bytes[5] & FLAG_LIVE != 0,
            flip: bytes[5] & FLAG_FLIP != 0,
            capture_us: u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")),
        })
    }

    /// A whole message: the header followed by `payload`.
    #[must_use]
    pub fn frame(&self, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
        out.extend_from_slice(&self.encode());
        out.extend_from_slice(payload);
        out
    }
}

impl fmt::Display for FrameHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stream {} {:?}{} at {} us",
            self.stream_id,
            self.payload_type,
            if self.keyframe { " key" } else { "" },
            self.capture_us
        )
    }
}

/// Splits a message into its header and payload.
///
/// # Errors
///
/// [`HeaderError`] as [`FrameHeader::decode`].
pub fn split(message: &[u8]) -> Result<(FrameHeader, &[u8]), HeaderError> {
    let header = FrameHeader::decode(message)?;
    let offset = extensions_end(message);
    if message.len() < offset {
        return Err(HeaderError::Short(message.len()));
    }
    Ok((header, &message[offset..]))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// The same vectors as frontend/src/dashboard/player/header.test.ts, so both ends agree byte for byte.
    const VECTORS: [(FrameHeader, [u8; 16]); 4] = [
        (
            FrameHeader {
                stream_id: 7,
                payload_type: PayloadType::Video,
                keyframe: true,
                live: false,
                flip: false,
                capture_us: 1_700_000_000_123_456,
            },
            [7, 0, 0, 0, 0x01, 0x01, 0, 0, 0x40, 0x22, 0x20, 0x18, 0x24, 0x0a, 0x06, 0x00],
        ),
        (
            FrameHeader {
                stream_id: 0x0102_0304,
                payload_type: PayloadType::Overlay,
                keyframe: false,
                live: false,
                flip: false,
                capture_us: 1,
            },
            [4, 3, 2, 1, 0x02, 0x00, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            FrameHeader {
                stream_id: 0,
                payload_type: PayloadType::Arrow,
                keyframe: false,
                live: false,
                flip: false,
                capture_us: 0,
            },
            [0, 0, 0, 0, 0x03, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            FrameHeader {
                stream_id: u32::MAX,
                payload_type: PayloadType::Jpeg,
                keyframe: true,
                live: false,
                flip: false,
                capture_us: u64::MAX,
            },
            [0xff, 0xff, 0xff, 0xff, 0x04, 0x01, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        ),
    ];

    #[test]
    fn the_live_flag_is_bit_1_and_round_trips() {
        let h = FrameHeader {
            stream_id: 9,
            payload_type: PayloadType::Jpeg,
            keyframe: true,
            live: true,
            flip: false,
            capture_us: 2,
        };
        assert_eq!(h.encode(), [9, 0, 0, 0, 0x04, 0x03, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(FrameHeader::decode(&h.encode()).expect("decode"), h);
    }

    #[test]
    fn flipped_live_picture_uses_bit_2_without_changing_the_legacy_bits_or_alignment() {
        let header = FrameHeader {
            stream_id: 9,
            payload_type: PayloadType::Jpeg,
            keyframe: true,
            live: true,
            flip: true,
            capture_us: 2,
        };
        assert_eq!(header.encode(), [9, 0, 0, 0, 0x04, 0x07, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(FrameHeader::decode(&header.encode()).expect("decode"), header);
        // Old readers only inspect the first two bits and still see the same keyframe and live flags.
        assert_eq!(header.encode()[5] & 0x03, 0x03);
    }

    #[test]
    fn future_flags_are_ignored_but_old_frames_are_not_flipped() {
        let mut bytes = VECTORS[0].1;
        bytes[5] |= 0x80;
        assert_eq!(FrameHeader::decode(&bytes).expect("unknown flag ignored"), VECTORS[0].0);
        assert!(!FrameHeader::at_ns(3, PayloadType::Video, true, 1_000).flip);
    }

    #[test]
    fn every_payload_type_encodes_to_its_vector_and_back() {
        for (header, bytes) in VECTORS {
            assert_eq!(header.encode(), bytes, "{header}");
            assert_eq!(FrameHeader::decode(&bytes).expect("decode"), header);
        }
        assert_eq!(VECTORS.iter().map(|(h, _)| h.payload_type).collect::<Vec<_>>(), PayloadType::ALL.to_vec());
    }

    #[test]
    fn bad_headers_are_refused() {
        assert_eq!(FrameHeader::decode(&[0; 15]), Err(HeaderError::Short(15)));
        let mut b = VECTORS[0].1;
        b[4] = 0x09;
        assert_eq!(FrameHeader::decode(&b), Err(HeaderError::UnknownType(9)));
        let mut b = VECTORS[0].1;
        b[6] = 1;
        assert_eq!(FrameHeader::decode(&b), Err(HeaderError::Reserved));
    }

    #[test]
    fn a_frame_is_the_header_then_the_payload_and_at_ns_divides() {
        let h = FrameHeader::at_ns(3, PayloadType::Jpeg, true, 1_234_567_890);
        assert_eq!(h.capture_us, 1_234_567);
        let f = h.frame(b"jpeg");
        let (back, payload) = split(&f).expect("split");
        assert_eq!((back, payload), (h, &b"jpeg"[..]));
    }

    proptest! {
        #[test]
        fn any_header_round_trips(stream_id: u32, tag in 1u8..=4, keyframe: bool, live: bool, flip: bool, capture_us: u64) {
            let payload_type = PayloadType::from_u8(tag).expect("known");
            let h = FrameHeader { stream_id, payload_type, keyframe, live, flip, capture_us };
            prop_assert_eq!(FrameHeader::decode(&h.encode()).expect("decode"), h);
        }
    }
}

/// Attach an opt-in context without changing the legacy header layout.
#[must_use]
pub fn with_media_context(header: FrameHeader, context_id: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = header.encode().to_vec();
    out[5] |= FLAG_MEDIA_CONTEXT;
    out.extend_from_slice(&context_id.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Where the payload starts: after the header and the extensions its flags announce.
fn extensions_end(message: &[u8]) -> usize {
    let flags = message[5];
    HEADER_LEN + if flags & FLAG_MEDIA_CONTEXT != 0 { 4 } else { 0 } + if flags & FLAG_REQUEST != 0 { 4 } else { 0 }
}

/// A whole message tagged with the request that sent it (#2380): the request flag set and `req` placed after the
/// header and any media context ID. A message already tagged, or too short to be one, comes back unchanged.
#[must_use]
pub fn with_request(message: &[u8], req: u32) -> Vec<u8> {
    if message.len() < HEADER_LEN || message[5] & FLAG_REQUEST != 0 {
        return message.to_vec();
    }
    let at = if message[5] & FLAG_MEDIA_CONTEXT != 0 { HEADER_LEN + 4 } else { HEADER_LEN };
    if message.len() < at {
        return message.to_vec();
    }
    let mut out = Vec::with_capacity(message.len() + 4);
    out.extend_from_slice(&message[..at]);
    out[5] |= FLAG_REQUEST;
    out.extend_from_slice(&req.to_le_bytes());
    out.extend_from_slice(&message[at..]);
    out
}

/// A whole message marked as one of the `watch` set's (#2380): the live and watch flags, including on the stream's
/// first frame, which comes from the recording.
#[must_use]
pub fn watched(mut message: Vec<u8>) -> Vec<u8> {
    if message.len() >= HEADER_LEN {
        message[5] |= FLAG_LIVE | FLAG_WATCH;
    }
    message
}

/// The request ID a message carries, if any.
/// # Errors
/// Returns the same validation errors as [`split`].
pub fn request_id(message: &[u8]) -> Result<Option<u32>, HeaderError> {
    split(message)?;
    if message[5] & FLAG_REQUEST == 0 {
        return Ok(None);
    }
    let at = if message[5] & FLAG_MEDIA_CONTEXT != 0 { HEADER_LEN + 4 } else { HEADER_LEN };
    Ok(Some(u32::from_le_bytes(message[at..at + 4].try_into().expect("validated extension"))))
}

/// Read an optional context ID; reject a truncated extension.
/// # Errors
/// Returns the same validation errors as [`split`].
pub fn media_context(message: &[u8]) -> Result<Option<u32>, HeaderError> {
    split(message)?;
    Ok((message[5] & FLAG_MEDIA_CONTEXT != 0)
        .then(|| u32::from_le_bytes(message[16..20].try_into().expect("validated extension"))))
}

#[cfg(test)]
mod context_tests {
    use super::*;
    #[test]
    fn context_ids_do_not_depend_on_equal_capture_times_and_legacy_bytes_stay_identical() {
        let header = FrameHeader::at_ns(4, PayloadType::Jpeg, true, 5000);
        assert_eq!(header.frame(b"jpeg").len(), 20);
        assert_eq!(media_context(&header.frame(b"jpeg")).unwrap(), None);
        for id in [1, u32::MAX] {
            let packet = with_media_context(header, id, b"jpeg");
            assert_eq!(&packet[16..20], &id.to_le_bytes());
            assert_eq!(split(&packet).unwrap(), (header, b"jpeg".as_slice()));
            assert_eq!(media_context(&packet).unwrap(), Some(id));
            for n in 16..20 {
                assert!(split(&packet[..n]).is_err());
            }
        }
    }
}

#[cfg(test)]
mod request_tests {
    use super::*;

    #[test]
    fn a_request_id_follows_the_header_and_any_media_context_and_the_payload_is_unchanged() {
        let header = FrameHeader::at_ns(4, PayloadType::Jpeg, true, 5000);
        let plain = with_request(&header.frame(b"jpeg"), 7);
        assert_eq!((request_id(&plain).unwrap(), media_context(&plain).unwrap()), (Some(7), None));
        assert_eq!(&plain[16..20], &7u32.to_le_bytes());
        assert_eq!(split(&plain).unwrap(), (header, b"jpeg".as_slice()));
        let both = with_request(&with_media_context(header, 3, b"jpeg"), u32::MAX);
        assert_eq!((&both[16..20], &both[20..24]), (&3u32.to_le_bytes()[..], &u32::MAX.to_le_bytes()[..]));
        assert_eq!((media_context(&both).unwrap(), request_id(&both).unwrap()), (Some(3), Some(u32::MAX)));
        assert_eq!(split(&both).unwrap(), (header, b"jpeg".as_slice()));
        assert_eq!(with_request(&both, 9), both, "tagged once");
        for n in 16..24 {
            assert!(split(&both[..n]).is_err(), "a truncated extension at {n}");
        }
        assert_eq!(request_id(&header.frame(b"jpeg")).unwrap(), None);
    }

    #[test]
    fn a_watched_frame_keeps_its_other_flags() {
        let header = FrameHeader { flip: true, ..FrameHeader::at_ns(4, PayloadType::Video, true, 5000) };
        let message = watched(header.frame(b"au"));
        assert_eq!(message[5], FLAG_KEYFRAME | FLAG_FLIP | FLAG_LIVE | FLAG_WATCH);
        assert_eq!(split(&message).unwrap(), (FrameHeader { live: true, ..header }, b"au".as_slice()));
    }
}
