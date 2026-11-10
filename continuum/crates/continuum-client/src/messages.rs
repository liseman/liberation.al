//! Building the client's `ControlEnvelope`s, and the owned forms of what the broker sends back.

use continuum_core::schema::continuum as fb;
use continuum_core::{SCHEMA_MAJOR, SCHEMA_MINOR, StorageLocator};
use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, WIPOffset};

/// The SDK's version, sent in `Hello`.
pub const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) fn finish(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    payload_type: fb::ControlPayload,
    payload: WIPOffset<UnionWIPOffset>,
) -> Vec<u8> {
    let env =
        fb::ControlEnvelope::create(b, &fb::ControlEnvelopeArgs { request_id, payload_type, payload: Some(payload) });
    b.finish(env, None);
    b.finished_data().to_vec()
}

/// `Hello`: who's connecting, and the schema it speaks.
pub(crate) fn hello(request_id: u64, client_name: &str, pid: u32) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let name = b.create_string(client_name);
    let sdk = b.create_string(SDK_VERSION);
    let schema = fb::SchemaVersion::create(&mut b, &fb::SchemaVersionArgs { major: SCHEMA_MAJOR, minor: SCHEMA_MINOR });
    let h = fb::Hello::create(
        &mut b,
        &fb::HelloArgs { client_name: Some(name), pid, sdk_version: Some(sdk), schema: Some(schema) },
    );
    finish(&mut b, request_id, fb::ControlPayload::Hello, h.as_union_value())
}

/// What a worker reports with a heartbeat.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Telemetry {
    /// Jobs waiting in the worker.
    pub queue_depth: u32,
    /// Models loaded.
    pub models_resident: Vec<String>,
    /// RAM in use, in bytes.
    pub ram_bytes: u64,
    /// VRAM in use, in bytes.
    pub vram_bytes: u64,
}

/// `Heartbeat`, with the worker's telemetry if it has some.
pub(crate) fn heartbeat(request_id: u64, now_ns: u64, telemetry: Option<&Telemetry>) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let telemetry = telemetry.map(|t| {
        let models: Vec<_> = t.models_resident.iter().map(|m| b.create_string(m)).collect();
        let models = b.create_vector(&models);
        fb::Telemetry::create(
            &mut b,
            &fb::TelemetryArgs {
                queue_depth: t.queue_depth,
                models_resident: Some(models),
                ram_bytes: t.ram_bytes,
                vram_bytes: t.vram_bytes,
            },
        )
    });
    let h = fb::Heartbeat::create(&mut b, &fb::HeartbeatArgs { now_ns, telemetry });
    finish(&mut b, request_id, fb::ControlPayload::Heartbeat, h.as_union_value())
}

/// `Goodbye`.
pub(crate) fn goodbye(request_id: u64, reason: &str) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let reason = b.create_string(reason);
    let g = fb::Goodbye::create(&mut b, &fb::GoodbyeArgs { reason: Some(reason) });
    finish(&mut b, request_id, fb::ControlPayload::Goodbye, g.as_union_value())
}

/// Where a stream's payloads go: a size class's slots, or a fixed-stride ring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StreamStorage {
    /// Payloads in slots of this size class.
    Slab {
        /// The size class.
        class: u16,
    },
    /// Fixed-size records (PCM blocks, IMU samples) in a ring.
    Ring {
        /// Bytes per record. A multiple of 8 copies by words, anything else by bytes.
        stride_bytes: u32,
        /// How many seconds the ring holds; 0 takes the broker's default (10 s).
        seconds: f32,
    },
}

/// A stream to register (`RegisterStream`).
#[derive(Clone, Debug, PartialEq)]
pub struct StreamSpec {
    /// The stream's name, like `cam0.video` or `pose.body`.
    pub name: String,
    /// A sensor (captured) or derived (computed) stream.
    pub kind: fb::StreamKind,
    /// How its payloads are encoded.
    pub format: fb::PayloadFormat,
    /// A raw format like `NV12`, or a named shape for derived data.
    pub shape_name: String,
    /// The shape's version.
    pub shape_version: u32,
    /// Payloads per second, nominally (0 if irregular).
    pub nominal_rate_hz: f32,
    /// Slots or a ring.
    pub storage: StreamStorage,
    /// Sensors only: the clock anchor taken at capture start.
    pub anchor: Option<fb::ClockAnchor>,
    /// The domain layer of its payloads (a sensor's are `Layer1Raw`).
    pub layer: fb::DomainLayer,
}

pub(crate) fn register_stream(request_id: u64, spec: &StreamSpec) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let name = b.create_string(&spec.name);
    let shape = b.create_string(&spec.shape_name);
    let (slab_class, ring_stride_bytes, ring_seconds) = match spec.storage {
        StreamStorage::Slab { class } => (class, 0, 0.0),
        StreamStorage::Ring { stride_bytes, seconds } => (0, stride_bytes, seconds),
    };
    let r = fb::RegisterStream::create(
        &mut b,
        &fb::RegisterStreamArgs {
            name: Some(name),
            kind: spec.kind,
            format: spec.format,
            shape_name: Some(shape),
            shape_version: spec.shape_version,
            nominal_rate_hz: spec.nominal_rate_hz,
            slab_class,
            ring_stride_bytes,
            ring_seconds,
            anchor: spec.anchor.as_ref(),
            layer: spec.layer,
        },
    );
    finish(&mut b, request_id, fb::ControlPayload::RegisterStream, r.as_union_value())
}

/// A stream's standing lease from `StreamRegistered`: the slots or the ring the producer writes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamLease {
    /// The lease's id.
    pub lease_id: u64,
    /// The arena it's in.
    pub arena: String,
    /// That arena's generation.
    pub arena_generation: u32,
    /// The first slot of the lease's slots.
    pub first_slot: u32,
    /// How many slots.
    pub slot_count: u32,
    /// Bytes per slot.
    pub slot_bytes: u64,
    /// For a fixed-stride ring, its offset in the arena (0 for slots).
    pub ring_offset: u64,
    /// For a fixed-stride ring, its record count (0 for slots).
    pub ring_records: u32,
    /// For a ring carved at registration: its index, what a `RingLocation` names.
    pub ring_index: u16,
    /// For a carved ring: where its records start in the arena (`ring_offset` is its control block).
    pub ring_data_offset: u64,
    /// For a carved ring: bytes per record.
    pub ring_stride: u32,
    /// For a carved ring: its generation, which readers and the writer check.
    pub ring_generation: u32,
}

impl StreamLease {
    pub(crate) fn from_fb(l: &fb::StreamLease<'_>) -> Self {
        Self {
            lease_id: l.lease_id(),
            arena: l.arena().unwrap_or_default().to_owned(),
            arena_generation: l.arena_generation(),
            first_slot: l.first_slot(),
            slot_count: l.slot_count(),
            slot_bytes: l.slot_bytes(),
            ring_offset: l.ring_offset(),
            ring_records: l.ring_records(),
            ring_index: l.ring_index(),
            ring_data_offset: l.ring_data_offset(),
            ring_stride: l.ring_stride(),
            ring_generation: l.ring_generation(),
        }
    }

    /// The carved ring this lease describes, to open with `Arena::carved_ring_writer` (or a reader), when the stream
    /// is a ring stream; `None` for a slot stream.
    #[must_use]
    pub fn carved_ring(&self) -> Option<continuum_shm::CarvedRing> {
        (self.ring_records > 0 && self.ring_stride > 0).then_some(continuum_shm::CarvedRing {
            ctrl_offset: self.ring_offset,
            data_offset: self.ring_data_offset,
            stride: self.ring_stride,
            capacity: self.ring_records,
            generation: self.ring_generation,
        })
    }
}

/// The broker's answer to `RegisterStream`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamRegistered {
    /// The stream's id.
    pub stream_id: u32,
    /// Its standing lease, for a capture stream.
    pub lease: Option<StreamLease>,
}

impl StreamRegistered {
    pub(crate) fn from_fb(r: fb::StreamRegistered<'_>) -> Self {
        let lease = r.lease().map(|l| StreamLease::from_fb(&l));
        Self { stream_id: r.stream_id(), lease }
    }
}

/// One size class of an arena, from `Welcome`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeClass {
    /// Bytes per slab.
    pub slab_bytes: u64,
    /// How many slabs.
    pub count: u32,
}

/// An arena as `Welcome` describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaInfo {
    /// Its name.
    pub name: String,
    /// Its generation.
    pub generation: u32,
    /// Its size in bytes.
    pub size_bytes: u64,
    /// Its size classes.
    pub classes: Vec<SizeClass>,
}

impl ArenaInfo {
    pub(crate) fn from_fb(a: fb::ArenaInfo<'_>) -> Self {
        let classes = a
            .classes()
            .map(|cs| cs.iter().map(|c| SizeClass { slab_bytes: c.slab_bytes(), count: c.count() }).collect())
            .unwrap_or_default();
        Self {
            name: a.name().unwrap_or_default().to_owned(),
            generation: a.generation(),
            size_bytes: a.size_bytes(),
            classes,
        }
    }
}

/// A `StorageLocator` table inside `b` (core's `StorageLocator::encode` makes a buffer of its own; a message needs
/// the table in the message's buffer).
pub(crate) fn locator<'a>(
    b: &mut FlatBufferBuilder<'a>,
    l: &continuum_core::StorageLocator,
) -> WIPOffset<fb::StorageLocator<'a>> {
    use continuum_core::StorageLocator as L;
    let (location_type, location) = match l {
        L::Shm { arena, arena_generation, slab_class, slot, slot_generation, offset, length } => {
            let arena = b.create_string(arena);
            let shm = fb::ShmLocation::create(
                b,
                &fb::ShmLocationArgs {
                    arena: Some(arena),
                    arena_generation: *arena_generation,
                    slab_class: *slab_class,
                    slot: *slot,
                    slot_generation: *slot_generation,
                    offset: *offset,
                    length: *length,
                },
            );
            (fb::Location::ShmLocation, shm.as_union_value())
        }
        L::Ring { arena, arena_generation, ring, first_seq, count } => {
            let arena = b.create_string(arena);
            let r = fb::RingLocation::create(
                b,
                &fb::RingLocationArgs {
                    arena: Some(arena),
                    arena_generation: *arena_generation,
                    ring: *ring,
                    first_seq: *first_seq,
                    count: *count,
                },
            );
            (fb::Location::RingLocation, r.as_union_value())
        }
        L::Inline { bytes } => {
            let bytes = b.create_vector(bytes);
            let inline = fb::InlineLocation::create(b, &fb::InlineLocationArgs { bytes: Some(bytes) });
            (fb::Location::InlineLocation, inline.as_union_value())
        }
        L::Chunk { uri, offset, length, .. } => {
            let uri = b.create_string(uri);
            let chunk = fb::ChunkLocation::create(
                b,
                &fb::ChunkLocationArgs { uri: Some(uri), offset: *offset, length: *length },
            );
            (fb::Location::ChunkLocation, chunk.as_union_value())
        }
    };
    fb::StorageLocator::create(b, &fb::StorageLocatorArgs { tier: l.tier(), location_type, location: Some(location) })
}

/// `IngestNotice` for one payload, with request id 0: the broker answers only a rejection, and that comes back with
/// the same id 0, so it arrives as an event.
pub(crate) fn ingest(header: &continuum_core::HeaderBuf, at: Option<&continuum_core::StorageLocator>) -> Vec<u8> {
    ingest_numbered(0, header, at)
}

/// An `IngestNotice` with a request id: 0 for the silent path, anything else for an `Ingested` answer.
pub(crate) fn ingest_numbered(
    request_id: u64,
    header: &continuum_core::HeaderBuf,
    at: Option<&continuum_core::StorageLocator>,
) -> Vec<u8> {
    ingest_depending(request_id, header, at, &[])
}

/// An `IngestNotice` with its dependencies (minor 31); none is the plain notice.
pub(crate) fn ingest_depending(
    request_id: u64,
    header: &continuum_core::HeaderBuf,
    at: Option<&continuum_core::StorageLocator>,
    dependencies: &[crate::Dependency],
) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let header_bytes = b.create_vector(header.as_bytes());
    let locator = at.map(|l| locator(&mut b, l));
    let dependencies = (!dependencies.is_empty()).then(|| {
        let deps: Vec<_> = dependencies
            .iter()
            .map(|d| {
                let range = fb::TimeRange::new(d.start_ns, d.end_ns);
                fb::Dependency::create(
                    &mut b,
                    &fb::DependencyArgs { stream_id: d.stream_id, range: Some(&range), revision: d.revision },
                )
            })
            .collect();
        b.create_vector(&deps)
    });
    let n = fb::IngestNotice::create(
        &mut b,
        &fb::IngestNoticeArgs {
            stream_id: header.stream().0,
            lease_id: 0,
            locator,
            ring_first_seq: 0,
            ring_last_seq: 0,
            header_bytes: Some(header_bytes),
            dependencies,
        },
    );
    finish(&mut b, request_id, fb::ControlPayload::IngestNotice, n.as_union_value())
}

/// `SetSystemState`, one way (request id 0).
pub(crate) fn set_system_state(voice_turn: fb::VoiceTurnState) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let s = fb::SetSystemState::create(&mut b, &fb::SetSystemStateArgs { voice_turn });
    finish(&mut b, 0, fb::ControlPayload::SetSystemState, s.as_union_value())
}

/// `JobRejected`, one way (request id 0).
pub(crate) fn job_rejected(job_id: continuum_core::Uuid, reason: &str) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let reason = b.create_string(reason);
    let id = fb::Uuid::new(&job_id.0);
    let j = fb::JobRejected::create(&mut b, &fb::JobRejectedArgs { job_id: Some(&id), reason: Some(reason) });
    finish(&mut b, 0, fb::ControlPayload::JobRejected, j.as_union_value())
}

/// A whole `PurgeNotice` (request id 0, no more pages), made from gathered pages.
pub(crate) fn purge_notice(
    purge_id: continuum_core::Uuid,
    payload_ids: &[continuum_core::Uuid],
    spans: &[crate::worker::PurgedSpan],
) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let ids: Vec<_> = payload_ids.iter().map(|u| fb::Uuid::new(&u.0)).collect();
    let ids = b.create_vector(&ids);
    let spans: Vec<_> = spans
        .iter()
        .map(|s| {
            let range = fb::TimeRange::new(s.range.0, s.range.1);
            fb::PurgedSpan::create(&mut b, &fb::PurgedSpanArgs { stream_id: s.stream_id, range: Some(&range) })
        })
        .collect();
    let spans = b.create_vector(&spans);
    let id = fb::Uuid::new(&purge_id.0);
    let p = fb::PurgeNotice::create(
        &mut b,
        &fb::PurgeNoticeArgs { purge_id: Some(&id), payload_ids: Some(ids), spans: Some(spans), more: false },
    );
    finish(&mut b, 0, fb::ControlPayload::PurgeNotice, p.as_union_value())
}

/// `PurgeAck`, one way (request id 0).
pub(crate) fn purge_ack(purge_id: continuum_core::Uuid) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let id = fb::Uuid::new(&purge_id.0);
    let a = fb::PurgeAck::create(&mut b, &fb::PurgeAckArgs { purge_id: Some(&id) });
    finish(&mut b, 0, fb::ControlPayload::PurgeAck, a.as_union_value())
}

/// `Purge` of a time range, on some streams (all when `streams` is empty).
pub(crate) fn purge_range(request_id: u64, range: (u64, u64), streams: &[String], reason: &str) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let names: Vec<_> = streams.iter().map(|n| b.create_string(n)).collect();
    let streams = b.create_vector(&names);
    let reason = b.create_string(reason);
    let range = fb::TimeRange::new(range.0, range.1);
    let p = fb::Purge::create(
        &mut b,
        &fb::PurgeArgs {
            range: Some(&range),
            streams: Some(streams),
            entity_id: None,
            reason: Some(reason),
            ids: None,
            detail: None,
            age_out: false,
            committed_before_ns: 0,
            aged_gap_ns: 0,
        },
    );
    finish(&mut b, request_id, fb::ControlPayload::Purge, p.as_union_value())
}

pub(crate) fn chunk_frames<'a>(
    b: &mut FlatBufferBuilder<'a>,
    frames: &[crate::worker::ChunkFrame],
) -> WIPOffset<flatbuffers::Vector<'a, fb::ChunkFrame>> {
    let frames: Vec<fb::ChunkFrame> =
        frames.iter().map(|f| fb::ChunkFrame::new(f.start_ns, f.offset, f.length, f.cut)).collect();
    b.create_vector(&frames)
}

/// `CommitPayloadCopy`: a durable copy retaining the original payload identity and provenance.
pub(crate) fn commit_payload_copy(
    request_id: u64,
    payload: continuum_core::Uuid,
    location: &StorageLocator,
) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let id = fb::Uuid(payload.0);
    let location = locator(&mut b, location);
    let body = fb::CommitPayloadCopy::create(
        &mut b,
        &fb::CommitPayloadCopyArgs { payload_id: Some(&id), location: Some(location) },
    );
    finish(&mut b, request_id, fb::ControlPayload::CommitPayloadCopy, body.as_union_value())
}

/// `CommitChunk`: a chunk, or the part of one from frame `first_frame` on, and the Tier 0 payloads it absorbed, in
/// frame order.
pub(crate) fn commit_chunk(
    request_id: u64,
    chunk: &crate::worker::NewChunk,
    first_frame: u32,
    absorbed: &[continuum_core::Uuid],
    provenance: Option<&[crate::worker::FrameProvenance]>,
) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let header_bytes = b.create_vector(chunk.header.as_bytes());
    let location = locator(&mut b, &chunk.location);
    let frames = chunk_frames(&mut b, &chunk.frames);
    let ids: Vec<fb::Uuid> = absorbed.iter().map(|u| fb::Uuid(u.0)).collect();
    let absorbed = b.create_vector(&ids);
    let frame_provenance = provenance.map(|rows| {
        let rows: Vec<fb::FrameProvenance> = rows
            .iter()
            .map(|f| fb::FrameProvenance::new(f.device_seq, f.epoch, u8::from(f.stamped_at_arrival)))
            .collect();
        b.create_vector(&rows)
    });
    let c = fb::CommitChunk::create(
        &mut b,
        &fb::CommitChunkArgs {
            header_bytes: Some(header_bytes),
            location: Some(location),
            frames: Some(frames),
            absorbed: Some(absorbed),
            frame_provenance,
            provenance_version: if provenance.is_some() { crate::worker::PROVENANCE_VERSION } else { 0 },
            first_frame,
        },
    );
    finish(&mut b, request_id, fb::ControlPayload::CommitChunk, c.as_union_value())
}

/// `RewriteChunk`: a chunk's new frame index and location after the tier worker rewrote it.
pub(crate) fn rewrite_chunk(
    request_id: u64,
    chunk_id: continuum_core::Uuid,
    location: &continuum_core::StorageLocator,
    frames: &[crate::worker::ChunkFrame],
) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let id = fb::Uuid(chunk_id.0);
    let location = locator(&mut b, location);
    let frames = chunk_frames(&mut b, frames);
    let r = fb::RewriteChunk::create(
        &mut b,
        &fb::RewriteChunkArgs { chunk_id: Some(&id), location: Some(location), frames: Some(frames) },
    );
    finish(&mut b, request_id, fb::ControlPayload::RewriteChunk, r.as_union_value())
}

/// A `Purge` of one kind: a range (with its streams), an entity, or ids, with the caller's JSON detail.
pub(crate) fn purge(request_id: u64, target: &crate::PurgeTarget, reason: &str, detail: &str) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let reason = b.create_string(reason);
    let detail = b.create_string(detail);
    let mut args = fb::PurgeArgs { reason: Some(reason), detail: Some(detail), ..Default::default() };
    let range;
    let entity;
    match target {
        crate::PurgeTarget::Range { range: (a, z), streams } => {
            let names: Vec<_> = streams.iter().map(|n| b.create_string(n)).collect();
            args.streams = Some(b.create_vector(&names));
            range = fb::TimeRange::new(*a, *z);
            args.range = Some(&range);
        }
        crate::PurgeTarget::Entity(id) => {
            entity = fb::Uuid::from(*id);
            args.entity_id = Some(&entity);
        }
        crate::PurgeTarget::Ids(ids) => {
            let ids: Vec<fb::Uuid> = ids.iter().map(|&u| fb::Uuid::from(u)).collect();
            args.ids = Some(b.create_vector(&ids));
        }
        crate::PurgeTarget::AgeOut { ids, committed_before_ns, aged_gap_ns } => {
            let ids: Vec<fb::Uuid> = ids.iter().map(|&u| fb::Uuid::from(u)).collect();
            args.ids = Some(b.create_vector(&ids));
            args.age_out = true;
            args.committed_before_ns = *committed_before_ns;
            args.aged_gap_ns = *aged_gap_ns;
        }
    }
    let p = fb::Purge::create(&mut b, &args);
    finish(&mut b, request_id, fb::ControlPayload::Purge, p.as_union_value())
}

pub(crate) fn list_purges(request_id: u64, after: Option<continuum_core::Uuid>, limit: u32) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let after = after.map(fb::Uuid::from);
    let l = fb::ListPurges::create(&mut b, &fb::ListPurgesArgs { after: after.as_ref(), limit });
    finish(&mut b, request_id, fb::ControlPayload::ListPurges, l.as_union_value())
}

pub(crate) fn remember_forgotten(request_id: u64, hash: [u8; 32]) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let hash = fb::ForgottenHash(hash);
    let r = fb::RememberForgotten::create(&mut b, &fb::RememberForgottenArgs { hash: Some(&hash) });
    finish(&mut b, request_id, fb::ControlPayload::RememberForgotten, r.as_union_value())
}

pub(crate) fn list_forgotten(request_id: u64, after: Option<[u8; 32]>, limit: u32) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let after = after.map(fb::ForgottenHash);
    let r = fb::ListForgotten::create(&mut b, &fb::ListForgottenArgs { after: after.as_ref(), limit });
    finish(&mut b, request_id, fb::ControlPayload::ListForgotten, r.as_union_value())
}

pub(crate) fn is_forgotten(request_id: u64, hash: [u8; 32]) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let hash = fb::ForgottenHash(hash);
    let r = fb::IsForgotten::create(&mut b, &fb::IsForgottenArgs { hash: Some(&hash) });
    finish(&mut b, request_id, fb::ControlPayload::IsForgotten, r.as_union_value())
}

pub(crate) fn forget_keeper(request_id: u64, name: &str) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let name = b.create_string(name);
    let f = fb::ForgetKeeper::create(&mut b, &fb::ForgetKeeperArgs { name: Some(name) });
    finish(&mut b, request_id, fb::ControlPayload::ForgetKeeper, f.as_union_value())
}

pub(crate) fn get_purge(request_id: u64, purge_id: continuum_core::Uuid) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let id = fb::Uuid::from(purge_id);
    let g = fb::GetPurge::create(&mut b, &fb::GetPurgeArgs { purge_id: Some(&id) });
    finish(&mut b, request_id, fb::ControlPayload::GetPurge, g.as_union_value())
}

/// A `ListStreams`.
pub(crate) fn list_streams(request_id: u64) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let l = fb::ListStreams::create(&mut b, &fb::ListStreamsArgs {});
    finish(&mut b, request_id, fb::ControlPayload::ListStreams, l.as_union_value())
}

/// A local diagnostic sent before an overflowing mixed-use connection closes.
pub(crate) fn subscription_overflow(id: u64) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let reason = b.create_string(&format!("SDK subscription {id} exceeded its bounded queue; connection closed. Resume from the last acknowledged position, using a dedicated Backpressure reader for long catch-up."));
    let value = fb::Error::create(&mut b, &fb::ErrorArgs { code: fb::ErrorCode::Unavailable, message: Some(reason) });
    finish(&mut b, 0, fb::ControlPayload::Error, value.as_union_value())
}

#[cfg(test)]
mod tests {
    use continuum_core::{HeaderBuf, HeaderSpec, StorageLocator, StorageTier, StreamId, Uuid};

    use super::*;

    fn header() -> HeaderBuf {
        let mut id = [0u8; 16];
        id[6] = 0x70;
        id[8] = 0x80;
        HeaderBuf::build(&HeaderSpec::raw(Uuid(id), StreamId(9), 10, 20)).unwrap()
    }

    /// What the broker reads back from a notice: the locator and the header bytes.
    fn read_back(bytes: &[u8]) -> (u64, u32, Option<StorageLocator>, Vec<u8>) {
        let env = flatbuffers::root::<fb::ControlEnvelope>(bytes).unwrap();
        let n = env.payload_as_ingest_notice().unwrap();
        let locator = n.locator().map(|l| StorageLocator::from_table(&l).unwrap());
        (env.request_id(), n.stream_id(), locator, n.header_bytes().unwrap().bytes().to_vec())
    }

    #[test]
    fn every_kind_of_locator_reads_back_as_itself() {
        let h = header();
        for at in [
            StorageLocator::Shm {
                arena: "hot".into(),
                arena_generation: 3,
                slab_class: 2,
                slot: 7,
                slot_generation: 11,
                offset: 16384 * 5,
                length: 12_441_600,
            },
            StorageLocator::Inline { bytes: b"small".to_vec() },
            StorageLocator::Chunk { tier: StorageTier::Tier2Cold, uri: "file:///a/b".into(), offset: 1, length: 2 },
        ] {
            let (request_id, stream, locator, header_bytes) = read_back(&ingest(&h, Some(&at)));
            assert_eq!(request_id, 0, "a notice is never numbered");
            assert_eq!(stream, 9, "the stream comes from the header");
            assert_eq!(locator.as_ref(), Some(&at));
            assert_eq!(header_bytes, h.as_bytes());
        }
        assert_eq!(read_back(&ingest(&h, None)).2, None);
    }
}
