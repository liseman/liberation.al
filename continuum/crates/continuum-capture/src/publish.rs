//! Writing captured payloads into Tier 0 and telling the broker (docs/CONTINUUM.md, "Writing", "Rings", and "How
//! frames land in shared memory").
//!
//! - [`VideoStream`]: each frame into a slot of the stream's own slot ring (the device's native payload, MJPEG bytes
//!   for every camera by default), overwriting the oldest slot nobody has pinned, flushed or not, so capture never
//!   waits. Each published slot's handle is pushed into a small ring too, whose sequence word a live reader waits on
//!   (`SeqWait`), and an `IngestNotice` goes to the broker for every frame.
//! - [`AudioStream`]: each block into the stream's PCM ring, published with a release store and a `SeqWait` wake,
//!   and one `IngestNotice` per batch (a second by default), whose header covers the batch's time. A batch never spans
//!   a gap: an overrun ends it, so the catalog's coverage shows the gap.

use continuum_core::schema::continuum as fb;
use continuum_core::{DomainLayer, HeaderBuf, PayloadFormat, SlotHandle, StorageLocator, StorageTier, StreamId, Uuid};
use continuum_shm::{Arena, RingWriter};

use crate::clock::Anchor;
use crate::fit::{AudioClock, BlockStamp};

/// Where notices go: the broker's client, or a test's fake.
pub trait Sink {
    /// Sends one `IngestNotice`: the payload's header, and where its bytes are (none for a ring batch, whose
    /// records stay in the ring; see the crate's docs).
    ///
    /// # Errors
    /// When the notice can't be sent (the broker went).
    fn ingest(&mut self, header: HeaderBuf, locator: Option<StorageLocator>) -> Result<(), SinkError>;
}

/// The broker's client is a sink: `ingest` queues the notice for the socket and returns at once, so capture never
/// waits on the broker. A notice the broker refuses comes back as an `IngestRejected` event, tied to the payload id.
impl Sink for continuum_client::Client {
    fn ingest(&mut self, header: HeaderBuf, locator: Option<StorageLocator>) -> Result<(), SinkError> {
        continuum_client::Client::ingest(self, &header, locator.as_ref()).map_err(|e| SinkError(e.to_string()))
    }
}

/// A notice couldn't be sent.
#[derive(Debug, thiserror::Error)]
#[error("the notice couldn't be sent: {0}")]
pub struct SinkError(pub String);

/// Why publishing failed.
#[derive(Debug, thiserror::Error)]
pub enum PublishError {
    /// The shared memory refused (every slot pinned or being written, a frame larger than a slot).
    #[error(transparent)]
    Shm(#[from] continuum_shm::Error),
    /// The header didn't build.
    #[error("the header didn't build: {0}")]
    Header(#[from] continuum_core::DecodeError),
    /// The broker couldn't be told.
    #[error(transparent)]
    Sink(#[from] SinkError),
    /// The stream's format can't be published (a rate of 0, a block of no samples, a ring of the wrong stride).
    #[error("bad audio format: {0}")]
    Format(String),
}

/// Bytes in front of each PCM record in a ring: the block's capture time (host monotonic nanoseconds, from the
/// fit) and the index of its first sample, both u64 little-endian, the shape the ring tests' producer uses. A live
/// reader gets each block's time from the ring itself, without waiting for its batch's notice.
pub const PCM_RECORD_HEADER: usize = 16;

/// What capture knew about a sample, kept in the header so its time can be re-fitted later.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capture {
    /// The best estimate of when it was captured, host monotonic nanoseconds.
    pub mono_ns: u64,
    /// The device's own timestamp, raw.
    pub device_ts: u64,
    /// When the worker received it, host monotonic nanoseconds.
    pub arrival_ns: u64,
    /// The device's sequence number (a V4L2 buffer's, an audio block's first sample).
    pub device_seq: u64,
    /// Video: bytes per row (0 for a compressed frame), width, height, and whether readers flip it.
    pub stride: u32,
    /// Video width.
    pub width: u32,
    /// Video height.
    pub height: u32,
    /// Mounted upside down: readers flip it.
    pub flip: bool,
    /// Audio source sample rate, or zero for nonaudio/unknown legacy producers.
    pub sample_rate_hz: u32,
    /// Video: how long the source says the frame was presented, nanoseconds. Zero when the source attests no
    /// duration, and the frame stays an instant: it is never estimated from a nominal or encoder frame rate.
    pub duration_ns: u64,
    /// Video: the capture session's opaque id, the same for every frame of one device open or RTSP connection and
    /// new after a reopen, reconnect, or wake, so `device_seq` compares only within it. Zero is unknown.
    pub capture_epoch: u32,
}

/// What a stream's headers say about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamMeta {
    /// Its id from `StreamRegistered`.
    pub stream: StreamId,
    /// How its bytes are encoded.
    pub format: PayloadFormat,
    /// `MJPEG`, `UYVY`, `PCM_S16LE`, and so on.
    pub shape_name: String,
    /// The clock anchor it registered with.
    pub anchor: Anchor,
    /// Its fixed latency, measured by a clap-and-flash check, nanoseconds subtracted from every stamp.
    pub residual_offset_ns: i64,
}

impl StreamMeta {
    /// UTC nanoseconds for a monotonic stamp, after the residual offset.
    #[must_use]
    pub fn utc_of(&self, mono_ns: u64) -> u64 {
        let utc = self.anchor.utc_of(mono_ns);
        if self.residual_offset_ns >= 0 {
            utc.saturating_sub(self.residual_offset_ns.unsigned_abs())
        } else {
            utc.saturating_add(self.residual_offset_ns.unsigned_abs())
        }
    }
}

/// Where an audio batch ends: the fit's time for its next sample, unless the fit puts that at or before the batch's
/// start (a fit that changed mode or re-anchored mid-batch), and then its nominal end, start plus samples over rate.
/// A batch with no duration gives the catalog no bound on the audio it holds, which marks the microphone's whole
/// recording history unknown and blocks aging on every camera that reads it (#1669: a 60 ms wave3 batch).
fn batch_end(start_mono_ns: u64, fitted: Option<u64>, nominal_end: u64) -> u64 {
    match fitted {
        Some(end) if end > start_mono_ns => end,
        _ => nominal_end.max(start_mono_ns + 1),
    }
}

/// Builds a sensor payload's header, with its capture fields.
///
/// # Errors
/// [`continuum_core::DecodeError`] when the span ends before it starts.
pub fn sensor_header(
    meta: &StreamMeta,
    id: Uuid,
    start_ns: u64,
    end_ns: u64,
    seq: u32,
    capture: &Capture,
) -> Result<HeaderBuf, continuum_core::DecodeError> {
    let mut b = flatbuffers::FlatBufferBuilder::new();
    let shape = b.create_string(&meta.shape_name);
    let cap = fb::CaptureInfo::create(
        &mut b,
        &fb::CaptureInfoArgs {
            capture_mono_ns: capture.mono_ns,
            device_ts: capture.device_ts,
            arrival_ns: capture.arrival_ns,
            stride: capture.stride,
            width: capture.width,
            height: capture.height,
            flip: capture.flip,
            device_seq: capture.device_seq,
            sample_rate_hz: capture.sample_rate_hz,
            capture_epoch: capture.capture_epoch,
        },
    );
    let fid = fb::Uuid::from(id);
    let header = fb::PayloadHeader::create(
        &mut b,
        &fb::PayloadHeaderArgs {
            id: Some(&fid),
            stream_id: meta.stream.0,
            start_ns,
            end_ns,
            seq,
            layer: DomainLayer::Layer1Raw,
            format: meta.format,
            shape_name: Some(shape),
            source_tier: StorageTier::Tier0Hot,
            capture: Some(cap),
            ..Default::default()
        },
    );
    fb::finish_payload_header_buffer(&mut b, header);
    HeaderBuf::new(b.finished_data().to_vec())
}

fn new_id() -> Uuid {
    Uuid(*uuid::Uuid::now_v7().as_bytes())
}

/// A fresh capture epoch for a new capture session (a device open, an RTSP connection, or the loop after a wake):
/// random and nonzero, so only equality means anything, and zero stays "unknown".
#[must_use]
pub fn new_capture_epoch() -> u32 {
    // The last bytes of a v7 UUID are random.
    let b = uuid::Uuid::now_v7().into_bytes();
    u32::from_le_bytes([b[12], b[13], b[14], b[15]]).max(1)
}

/// A camera stream's slot ring in Tier 0.
pub struct VideoStream {
    meta: StreamMeta,
    arena: Arena,
    class: u16,
    owner: u64,
    handles: Option<RingWriter>,
    published: u64,
    overwritten_unflushed: u64,
}

/// What publishing a frame did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Published {
    /// The slot it went into.
    pub slot: SlotHandle,
    /// Slots it overwrote that the tier worker hadn't flushed yet: gaps in Tier 1 (the broker logs them).
    pub unflushed_overwrites: usize,
}

/// A slot handle's record in the handles ring: class, index, generation, and 4 bytes of padding, so the stride is a
/// multiple of 8.
pub const HANDLE_RECORD: usize = 16;

/// How a publish claims its arena slot.
#[derive(Clone, Copy)]
enum Claim {
    /// A raw camera's own ring: take the oldest unpinned slot, flushed or not, so the camera never waits.
    Overwrite,
    /// An encoded camera (#1931): take the oldest unpinned slot already flushed to Tier 1, never an unflushed one, so
    /// the ring recycles without waiting for the broker's reclaim sweep and the archive keeps every access unit.
    Recycle,
    /// Only a free slot.
    Free,
}
impl VideoStream {
    /// A stream writing into size class `class` of `arena` as `owner`, optionally announcing each slot in a ring of
    /// slot handles (stride [`HANDLE_RECORD`]) for live readers.
    #[must_use]
    pub fn new(meta: StreamMeta, arena: Arena, class: u16, owner: u64, handles: Option<RingWriter>) -> Self {
        Self { meta, arena, class, owner, handles, published: 0, overwritten_unflushed: 0 }
    }

    /// Writes one frame: `write` fills the slot and returns how many bytes it used, then the slot is published,
    /// announced in the handles ring, and noticed to the broker. It never waits: with no free slot it overwrites the
    /// oldest one nobody has pinned.
    ///
    /// # Errors
    /// [`PublishError`] when every slot is pinned or being written, the frame overflows the slot, or the notice
    /// can't be sent.
    pub fn publish(
        &mut self,
        capture: Capture,
        sink: &mut dyn Sink,
        write: impl FnOnce(&mut [u8]) -> usize,
    ) -> Result<Published, PublishError> {
        self.publish_with_policy(capture, sink, write, Claim::Overwrite, 0)
    }

    /// Publishes into the smallest fitting encoded-data class without overwriting another producer's slots.
    /// The caller enforces its byte budget before claiming; pressure drops the new frame rather than old data.
    /// # Errors
    /// The same publishing errors as [`Self::publish`].
    pub fn publish_in_class(
        &mut self,
        class: u16,
        capture: Capture,
        sink: &mut dyn Sink,
        write: impl FnOnce(&mut [u8]) -> usize,
    ) -> Result<Published, PublishError> {
        self.class = class;
        self.publish_with_policy(capture, sink, write, Claim::Free, capture.device_seq as u32)
    }

    /// Publishes an encoded camera access unit without overwriting unflushed source data.
    /// # Errors
    /// Arena pressure, invalid header or broker failure.
    pub fn publish_encoded(
        &mut self,
        capture: Capture,
        bytes: &[u8],
        sink: &mut dyn Sink,
    ) -> Result<Published, PublishError> {
        let write = |slot: &mut [u8]| {
            slot[..bytes.len()].copy_from_slice(bytes);
            bytes.len()
        };
        self.publish_with_policy(capture, sink, write, Claim::Recycle, capture.device_seq as u32)
    }

    fn publish_with_policy(
        &mut self,
        capture: Capture,
        sink: &mut dyn Sink,
        write: impl FnOnce(&mut [u8]) -> usize,
        claim: Claim,
        sequence: u32,
    ) -> Result<Published, PublishError> {
        let (mut slot, gaps) = match claim {
            Claim::Overwrite => self.arena.claim_overwriting(self.class, self.owner)?,
            Claim::Recycle => (self.arena.claim_recycling(self.class, self.owner)?, Vec::new()),
            Claim::Free => (self.arena.claim(self.class, self.owner)?, Vec::new()),
        };
        let len = write(slot.as_mut_slice());
        let offset = slot.offset() as u64;
        let handle = slot.publish(len)?;
        if let Some(ring) = self.handles.as_mut() {
            let mut rec = [0u8; HANDLE_RECORD];
            rec[..SlotHandle::ENCODED_LEN].copy_from_slice(&handle.to_le_bytes());
            ring.push(&rec)?;
        }
        let at = self.meta.utc_of(capture.mono_ns);
        let header =
            sensor_header(&self.meta, new_id(), at, at.saturating_add(capture.duration_ns), sequence, &capture)?;
        let locator = StorageLocator::Shm {
            arena: self.arena.name().to_owned(),
            arena_generation: self.arena.generation(),
            slab_class: self.class,
            slot: handle.index,
            slot_generation: handle.generation,
            offset,
            length: len as u64,
        };
        sink.ingest(header, Some(locator))?;
        self.published += 1;
        self.overwritten_unflushed += gaps.len() as u64;
        Ok(Published { slot: handle, unflushed_overwrites: gaps.len() })
    }

    /// Places frames on a new clock anchor from now on (after a wake). The worker registers the stream again with it.
    pub fn set_anchor(&mut self, anchor: Anchor) {
        self.meta.anchor = anchor;
    }

    /// The registered stream, for capture diagnostics without exposing the rest of the publisher's state.
    pub(crate) fn stream_id(&self) -> StreamId {
        self.meta.stream
    }

    /// Frames published so far.
    #[must_use]
    pub fn published(&self) -> u64 {
        self.published
    }
}

/// A microphone stream's PCM ring in Tier 0.
pub struct AudioStream {
    meta: StreamMeta,
    ring: RingWriter,
    clock: AudioClock,
    rate_hz: u32,
    samples_per_block: u64,
    batch_samples: u64,
    batch: Option<Batch>,
    last_end: Option<BatchEnd>,
    record: Vec<u8>,
    /// The broker's name for the ring (arena, its generation, the ring's index), so each batch's notice can name
    /// its records; `None` for a ring the broker doesn't know (a test's, or a standalone one).
    ring_id: Option<(String, u32, u16)>,
    batches: u64,
    gaps: u64,
}

/// Where the last batch ended (its next sample and that sample's time), so the next batch starts exactly there.
#[derive(Clone, Copy, Debug)]
struct BatchEnd {
    sample: u64,
    mono_ns: u64,
}

#[derive(Clone, Copy, Debug)]
struct Batch {
    first_seq: u64,
    first_sample: u64,
    start_mono_ns: u64,
    samples: u64,
    first_arrival_ns: u64,
}

/// What publishing an audio block did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockPublished {
    /// The block's record sequence in the ring.
    pub seq: u64,
    /// Where the clock placed it.
    pub stamp: BlockStamp,
    /// Whether this block finished a batch, so a notice went out.
    pub noticed: bool,
}

impl AudioStream {
    /// A stream writing `samples_per_block` frames per record into `ring` at `rate_hz`, sending one notice per
    /// `batch_seconds`. Each record is [`PCM_RECORD_HEADER`] bytes of stamp, then the block.
    ///
    /// # Errors
    /// [`PublishError::Format`] for a rate or block of 0.
    pub fn new(
        meta: StreamMeta,
        ring: RingWriter,
        rate_hz: u32,
        samples_per_block: u64,
        batch_seconds: f32,
    ) -> Result<Self, PublishError> {
        if rate_hz == 0 || samples_per_block == 0 {
            return Err(PublishError::Format(format!("{rate_hz} Hz, {samples_per_block} samples a block")));
        }
        if ring.stride() <= PCM_RECORD_HEADER {
            return Err(PublishError::Format(format!("a ring stride of {} leaves no room for PCM", ring.stride())));
        }
        let batch_samples = ((f64::from(batch_seconds) * f64::from(rate_hz)) as u64).max(samples_per_block);
        Ok(Self {
            meta,
            clock: AudioClock::new(f64::from(rate_hz)),
            rate_hz,
            samples_per_block,
            batch_samples,
            batch: None,
            last_end: None,
            record: vec![0; ring.stride()],
            ring_id: None,
            ring,
            batches: 0,
            gaps: 0,
        })
    }

    /// Names the ring in each batch's notice: a `RingLocation` in `arena` (at `generation`), ring `index`, so the
    /// broker (and the tier worker, through it) can find the batch's records.
    #[must_use]
    pub fn with_ring_locator(mut self, arena: String, generation: u32, index: u16) -> Self {
        self.ring_id = Some((arena, generation, index));
        self
    }

    /// Counts gaps in `unit` (samples by default).
    #[must_use]
    pub fn with_gap_unit(mut self, unit: crate::fit::GapUnit) -> Self {
        self.clock = self.clock.with_gap_unit(unit);
        self
    }

    /// Writes one block (exactly the ring's stride less [`PCM_RECORD_HEADER`] bytes) whose first sample the driver captured at
    /// `captured_ns`, received at `arrival_ns`. A gap before it (an overrun) ends the batch in progress first.
    ///
    /// # Errors
    /// [`PublishError`] when the block isn't the ring's stride, or a notice can't be sent.
    pub fn block(
        &mut self,
        pcm: &[u8],
        captured_ns: u64,
        arrival_ns: u64,
        sink: &mut dyn Sink,
    ) -> Result<BlockPublished, PublishError> {
        if pcm.len() + PCM_RECORD_HEADER != self.record.len() {
            return Err(PublishError::Format(format!(
                "a block of {} bytes in records of {}",
                pcm.len(),
                self.record.len()
            )));
        }
        let stamp = self.clock.block(self.samples_per_block, captured_ns);
        if stamp.gap_samples > 0 {
            self.gaps += 1;
            self.flush(sink)?;
        }
        self.record[..8].copy_from_slice(&stamp.host_ns.to_le_bytes());
        self.record[8..16].copy_from_slice(&stamp.first_sample.to_le_bytes());
        self.record[PCM_RECORD_HEADER..].copy_from_slice(pcm);
        let seq = self.ring.push(&self.record)?;
        // A batch that follows on from the last (no gap, no wake) starts exactly where it ended.
        let start_mono_ns = match self.last_end {
            Some(end) if end.sample == stamp.first_sample => end.mono_ns,
            _ => stamp.host_ns,
        };
        let batch = self.batch.get_or_insert(Batch {
            first_seq: seq,
            first_sample: stamp.first_sample,
            start_mono_ns,
            samples: 0,
            first_arrival_ns: arrival_ns,
        });
        batch.samples += self.samples_per_block;
        let noticed = batch.samples >= self.batch_samples;
        if noticed {
            self.flush(sink)?;
        }
        Ok(BlockPublished { seq, stamp, noticed })
    }

    /// Sends the batch in progress, if any (at a gap, a wake, or a stop).
    ///
    /// # Errors
    /// When the notice can't be sent.
    pub fn flush(&mut self, sink: &mut dyn Sink) -> Result<(), PublishError> {
        let Some(b) = self.batch.take() else { return Ok(()) };
        let start = self.meta.utc_of(b.start_mono_ns);
        // The end is the fit's time for the next sample, so the next batch (which starts there) touches this one.
        let next_sample = b.first_sample + b.samples;
        let nominal_end = b.start_mono_ns + b.samples * 1_000_000_000 / u64::from(self.rate_hz);
        let end_mono = batch_end(b.start_mono_ns, self.clock.host_ns_of(next_sample), nominal_end);
        self.last_end = Some(BatchEnd { sample: next_sample, mono_ns: end_mono });
        let end = self.meta.utc_of(end_mono);
        let capture = Capture {
            mono_ns: b.start_mono_ns,
            device_ts: b.first_sample,
            arrival_ns: b.first_arrival_ns,
            device_seq: b.first_seq,
            sample_rate_hz: self.rate_hz,
            ..Capture::default()
        };
        let header = sensor_header(&self.meta, new_id(), start, end, 0, &capture)?;
        let records = u32::try_from(b.samples / self.samples_per_block).unwrap_or(u32::MAX);
        let locator = self.ring_id.as_ref().map(|(arena, generation, ring)| StorageLocator::Ring {
            arena: arena.clone(),
            arena_generation: *generation,
            ring: *ring,
            first_seq: b.first_seq,
            count: records,
        });
        sink.ingest(header, locator)?;
        self.batches += 1;
        Ok(())
    }

    /// After a wake: the clock anchors afresh, and the stream takes the new anchor its re-registration got.
    pub fn rewake(&mut self, anchor: Anchor, sink: &mut dyn Sink) -> Result<(), PublishError> {
        self.flush(sink)?;
        self.last_end = None;
        self.clock.reset();
        self.meta.anchor = anchor;
        Ok(())
    }

    /// Notices sent, and gaps seen, so far.
    #[must_use]
    pub fn counts(&self) -> (u64, u64) {
        (self.batches, self.gaps)
    }

    /// The device's drift against the host, once fitted, in parts per million.
    #[must_use]
    pub fn drift_ppm(&self) -> Option<f64> {
        self.clock.drift_ppm()
    }
}

#[cfg(test)]
mod batch_end_tests {
    use super::batch_end;

    #[test]
    fn a_fitted_end_after_the_start_is_kept() {
        assert_eq!(batch_end(1_000, Some(61_000), 60_000), 61_000);
    }

    #[test]
    fn a_fitted_end_at_or_before_the_start_takes_the_nominal_end() {
        // wave3's instant original: the fit put its end exactly at its start.
        assert_eq!(batch_end(1_000, Some(1_000), 61_000), 61_000);
        assert_eq!(batch_end(1_000, Some(400), 61_000), 61_000);
        assert_eq!(batch_end(1_000, None, 61_000), 61_000);
    }

    #[test]
    fn the_end_is_always_after_the_start() {
        assert_eq!(batch_end(1_000, Some(1_000), 1_000), 1_001);
    }
}
