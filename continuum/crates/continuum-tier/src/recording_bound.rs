//! The end an instant audio original's own samples bound, for `continuum-recording-bound`: its chunk
//! frame's batch (CPB2, or the Opus batch Tier 2 re-encoded it to) says how many sample frames it holds and at what
//! true rate.

use std::path::Path;

use continuum_catalog::{Catalog, InstantOriginal, Resolved};
use continuum_core::{Interval, StorageLocator, StorageTier, Uuid};

use crate::ChunkReader;
use crate::pcm::sample_frames;

/// Capture's clock-fit drift bound, as a fraction: the end is widened by this much, and by one block.
pub const DRIFT: f64 = 1e-3;

/// The end the original's own samples bound, or why it can't be bounded: its chunk frame must be found, that frame's
/// record must have absorbed exactly this original at its own start, and the record must be a CPB2 or Opus batch
/// whose blocks per second agree with the stream's declared rate (when there is one).
///
/// The frame is found through the original's frame row, or, once that row is pruned (an hour after its chunk ends),
/// among the stream's chunks with an uncut frame at the original's start.
///
/// `root` is the data root (the state directory that holds `continuum/`), which relative chunk locators name files
/// under.
///
/// # Errors
/// Why the original is left as it is.
pub fn bounded_end(
    c: &dyn Catalog,
    root: &Path,
    original: &InstantOriginal,
    stream_rate: Option<f64>,
) -> Result<u64, String> {
    let candidates = match original.frame {
        Some(frame) => vec![frame],
        None => frames_at(c, original)?,
    };
    if candidates.is_empty() {
        return Err("its samples aren't in a chunk yet".into());
    }
    let mut why = String::new();
    for (chunk, index) in candidates {
        match own_record(c, root, original, chunk, index) {
            Ok(Some(bytes)) => return end_of(original, &bytes, stream_rate),
            Ok(None) => why = "no chunk frame at its start is this original's own record".into(),
            Err(e) => return Err(e),
        }
    }
    Err(why)
}

/// The stream's chunk frames, uncut, that start exactly at the original's start.
fn frames_at(c: &dyn Catalog, original: &InstantOriginal) -> Result<Vec<(Uuid, u32)>, String> {
    let at = Interval::covering(original.start_ns, original.start_ns).ok_or("its start has no extent")?;
    let mut out = Vec::new();
    let mut after = None;
    loop {
        let page = c.range(original.stream, at, after, 256).map_err(|e| e.to_string())?;
        for item in &page.items {
            for (index, frame) in item.frames.iter().flatten().enumerate() {
                if frame.start_ns == original.start_ns && !frame.cut {
                    out.push((item.header.id(), u32::try_from(index).map_err(|e| e.to_string())?));
                }
            }
        }
        match page.next {
            Some(next) => after = Some(next),
            None => return Ok(out),
        }
    }
}

/// The record bytes of frame `index` of `chunk` when that record absorbed this original at its start; `None` when it
/// is some other record.
fn own_record(
    c: &dyn Catalog,
    root: &Path,
    original: &InstantOriginal,
    chunk: Uuid,
    index: u32,
) -> Result<Option<Vec<u8>>, String> {
    let Some(Resolved::Payload(indexed)) = c.get(chunk).map_err(|e| e.to_string())? else {
        return Err(format!("chunk {chunk} isn't cataloged"));
    };
    // Tier 1 first: it holds the PCM batches as written. A Tier 2 copy may be Opus, whose header keeps the count.
    let chunk_uri = |tier: Option<StorageTier>| {
        indexed.locations.iter().find_map(|location| match location {
            StorageLocator::Chunk { uri, tier: at, .. } if tier.is_none_or(|tier| *at == tier) => Some(uri.clone()),
            _ => None,
        })
    };
    let uri = chunk_uri(Some(StorageTier::Tier1Warm)).or_else(|| chunk_uri(None)).ok_or("its chunk has no file")?;
    let file = continuum_core::chunk_path::resolve(root, &uri);
    let mut reader = ChunkReader::open(&file).map_err(|e| e.to_string())?;
    let record = *reader.records().get(index as usize).ok_or("its frame index is past the chunk's records")?;
    if record.id != original.original || record.start_ns != original.start_ns || record.cut {
        return Ok(None);
    }
    reader.read(index as usize).map(Some).map_err(|e| e.to_string())
}

/// The end a batch record's sample count bounds.
fn end_of(original: &InstantOriginal, bytes: &[u8], stream_rate: Option<f64>) -> Result<u64, String> {
    let (frames, format) = if crate::opus::is_batch(bytes) {
        crate::opus::sample_frames(bytes).map_err(|e| e.to_string())?
    } else {
        sample_frames(bytes).map_err(|e| e.to_string())?.ok_or("a CPB1 batch carries no rate")?
    };
    let rate = f64::from(format.sample_rate_hz);
    let block = (u64::from(format.layout.stride) - 16) / (4 * u64::from(format.layout.channels));
    let span_ns = (frames as f64 * 1e9 / rate * (1.0 + DRIFT) + block as f64 * 1e9 / rate).ceil() as u64;
    if frames == 0 || span_ns == 0 {
        return Err("its batch holds no samples".into());
    }
    let end_ns = original.start_ns + span_ns;
    // A microphone stream declares its ring blocks per second (capture registers 1000 / block_ms), not its sample
    // rate, so the batch's blocks per second is what has to agree.
    let blocks_per_second = rate / block as f64;
    if stream_rate.is_some_and(|declared| (declared - blocks_per_second).abs() > 0.5) {
        return Err(format!(
            "its batch says {blocks_per_second} blocks/s ({rate} Hz, {block} frames a block) but the stream declares \
             {stream_rate:?} (its samples would bound it to {end_ns} ns)"
        ));
    }
    Ok(end_ns)
}
