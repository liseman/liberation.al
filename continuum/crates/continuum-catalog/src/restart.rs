//! Opaque planned-restart checkpoints. The catalog validates freshness, not the broker's codec.
use continuum_core::Uuid;

/// Hard bound on opaque projection bytes, independent of the caller's read budget.
/// About 24 bytes per segment row, so this holds the codec's 64M rows; it stays well inside redb's value limit.
pub const MAX_RESTART_PROJECTION_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// Maximum exact hot payload identities returned with a checkpoint.
pub const MAX_RESTART_HOT_IDS: usize = 1_000_000;

/// Why [`Catalog::load_restart_projection`](crate::Catalog::load_restart_projection) found nothing usable. Each one
/// means the next start rebuilds the index from the catalog; none is an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestartMiss {
    /// This catalog implementation keeps no projection.
    Unsupported,
    /// No projection is stored: the last stop didn't publish one, or a write since removed it.
    NotSaved,
    /// The stored projection is larger than the read budget.
    TooLarge,
    /// The stored projection belongs to an older catalog revision or another catalog.
    Stale,
    /// The stored projection's checksum doesn't match its bytes.
    Corrupt,
    /// More Tier 0 payloads than a projection restores; the full load clears them instead.
    TooManyHotIds,
}

impl std::fmt::Display for RestartMiss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Unsupported => "this catalog keeps no restart projection",
            Self::NotSaved => "no restart projection is stored",
            Self::TooLarge => "the stored projection is over the read budget",
            Self::Stale => "the stored projection is from an older catalog revision",
            Self::Corrupt => "the stored projection fails its checksum",
            Self::TooManyHotIds => "too many Tier 0 payloads to restore from a projection",
        })
    }
}

/// A catalog incarnation and checked, transactionally advanced mutation counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestartStamp {
    /// Random identity created once when restart tracking is initialized.
    pub incarnation: Uuid,
    /// Every normal committed catalog write advances this value.
    pub revision: u64,
}

/// An opaque projection valid in the same snapshot as its exact hot-copy identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestartProjection {
    /// Freshness token of the projected catalog state.
    pub stamp: RestartStamp,
    /// Broker-owned, versioned codec bytes.
    pub bytes: Vec<u8>,
    /// Every payload with a catalog Tier 0 locator, including hidden payloads.
    pub hot_ids: Vec<Uuid>,
}

use crate::codec;
use crate::store::{Reader, W, index_of};
use crate::tables::{HOT_IDS, LOCATIONS, META, RESTART_PROJECTION};
use crate::{CatalogError, Result};

fn invalid(message: &str) -> CatalogError {
    CatalogError::Storage(message.to_owned().into())
}

pub(crate) fn stamp(r: &impl Reader) -> Result<Option<RestartStamp>> {
    let identity = r.get(META, b"restart_incarnation")?;
    let revision = r.get(META, b"restart_revision")?;
    match (identity, revision) {
        (None, None) => Ok(None),
        (Some(id), Some(rev)) if id.len() == 16 && rev.len() == 8 => {
            let mut bytes = [0; 16];
            bytes.copy_from_slice(&id);
            let revision = codec::u64_at(&rev, 0)?;
            if bytes == [0; 16] || revision == 0 {
                return Err(invalid("invalid restart tracking stamp"));
            }
            Ok(Some(RestartStamp { incarnation: Uuid(bytes), revision }))
        }
        _ => Err(invalid("incomplete restart tracking stamp")),
    }
}

pub(crate) fn ready(r: &impl Reader) -> Result<bool> {
    match r.get(META, b"restart_hot_ready")?.as_deref() {
        None => Ok(false),
        Some([1]) => Ok(true),
        _ => Err(invalid("invalid restart hot index readiness")),
    }
}

/// Starts restart tracking in a new, empty file: a fresh incarnation at revision 1, and an empty hot index that is
/// ready (there are no locations yet to index).
pub(crate) fn create(w: &mut W) -> Result<()> {
    w.put(META, b"restart_incarnation", uuid::Uuid::now_v7().as_bytes())?;
    w.put(META, b"restart_revision", &1u64.to_be_bytes())?;
    w.put(META, b"restart_hot_ready", &[1])
}

/// Called before normal writes commit. All index and revision changes share their transaction.
pub(crate) fn changed(w: &mut W) -> Result<()> {
    let old = stamp(w)?.ok_or_else(|| invalid("the catalog is missing its restart tracking"))?;
    if !ready(w)? {
        return Err(invalid("restart hot index is not ready"));
    }
    let locations = index_of(LOCATIONS);
    let keys: Vec<_> = w
        .touched
        .iter()
        .filter(|(table, key)| *table == locations && key.len() == 17 && key[16] == 0)
        .map(|(_, key)| key.clone())
        .collect();
    for key in keys {
        if w.has(LOCATIONS, &key)? {
            w.put(HOT_IDS, &key[..16], &[])?;
        } else {
            w.del(HOT_IDS, &key[..16])?;
        }
    }
    let next = old.revision.checked_add(1).ok_or_else(|| invalid("restart revision exhausted"))?;
    // An invalidated projection may contain identities and times a privacy purge must retire.
    // Delete it in this same transaction so scrub carries the deletion rather than stale bytes.
    w.del(RESTART_PROJECTION, b"current")?;
    w.put(META, b"restart_revision", &next.to_be_bytes())
}

// An accidental-corruption check for opaque bytes, not an authentication primitive.
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3))
}

pub(crate) fn publish(w: &mut W, expected: RestartStamp, bytes: &[u8]) -> Result<bool> {
    if bytes.len() > MAX_RESTART_PROJECTION_BYTES {
        return Err(crate::CatalogError::Invalid("restart projection exceeds byte limit".into()));
    }
    if stamp(w)? != Some(expected) || !ready(w)? {
        return Ok(false);
    }
    let mut encoded = Vec::with_capacity(32 + bytes.len());
    encoded.extend_from_slice(&expected.incarnation.0);
    encoded.extend_from_slice(&expected.revision.to_be_bytes());
    encoded.extend_from_slice(&checksum(bytes).to_be_bytes());
    encoded.extend_from_slice(bytes);
    w.put(RESTART_PROJECTION, b"current", &encoded)?;
    Ok(true)
}

pub(crate) fn load(r: &impl Reader, budget: usize) -> Result<std::result::Result<RestartProjection, RestartMiss>> {
    let current = stamp(r)?.ok_or_else(|| invalid("the catalog is missing its restart tracking"))?;
    if !ready(r)? {
        return Err(invalid("restart hot index is not ready"));
    }
    // Check the stored value's size while it is borrowed, before allocating it.
    let mut raw = Err(RestartMiss::NotSaved);
    r.scan(RESTART_PROJECTION, b"current", None, &mut |key, value| {
        if key == b"current" {
            raw = if value.len() < 32 {
                Err(RestartMiss::Corrupt)
            } else if value.len() - 32 > budget.min(MAX_RESTART_PROJECTION_BYTES) {
                Err(RestartMiss::TooLarge)
            } else {
                Ok(value.to_vec())
            };
        }
        Ok(false)
    })?;
    let mut raw = match raw {
        Ok(raw) => raw,
        Err(miss) => return Ok(Err(miss)),
    };
    if raw[..16] != current.incarnation.0 || codec::u64_at(&raw, 16)? != current.revision {
        return Ok(Err(RestartMiss::Stale));
    }
    if codec::u64_at(&raw, 24)? != checksum(&raw[32..]) {
        return Ok(Err(RestartMiss::Corrupt));
    }
    let mut hot_ids = Vec::new();
    let mut oversized = false;
    r.scan(HOT_IDS, &[], None, &mut |key, value| {
        if key.len() != 16 || !value.is_empty() {
            return Err(invalid("invalid restart hot index row"));
        }
        if hot_ids.len() == MAX_RESTART_HOT_IDS {
            oversized = true;
            return Ok(false);
        }
        hot_ids.push(codec::id_at(key, 0)?);
        Ok(true)
    })?;
    if oversized {
        return Ok(Err(RestartMiss::TooManyHotIds));
    }
    drop(raw.drain(..32));
    Ok(Ok(RestartProjection { stamp: current, bytes: raw, hot_ids }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Catalog, CoverageMark, NewPayload, RedbCatalog, StreamInfo, StreamKind};
    use continuum_core::{
        DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StageId, StorageLocator, StorageTier,
    };
    use redb::ReadableDatabase;
    use std::time::Duration;

    fn info() -> StreamInfo {
        StreamInfo {
            name: "synthetic".into(),
            kind: StreamKind::Sensor,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: "synthetic".into(),
            shape_version: 1,
            rate_hz: None,
            anchor: None,
        }
    }
    fn hot() -> StorageLocator {
        StorageLocator::Shm {
            arena: "synthetic".into(),
            arena_generation: 1,
            slab_class: 0,
            slot: 0,
            slot_generation: 1,
            offset: 0,
            length: 10,
        }
    }

    #[test]
    fn checkpoint_tracks_mutations_not_only_payload_head_and_keeps_exact_hot_ids() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("catalog.redb");
        let c = RedbCatalog::open(&path).unwrap();
        let stream = c.register_stream(info()).unwrap();
        let id = Uuid([1; 16]);
        c.insert(vec![NewPayload {
            header: HeaderBuf::build(&HeaderSpec::raw(id, stream, 1, 2)).unwrap(),
            location: Some(hot()),
            coverage: None,
        }])
        .unwrap();
        let first = c.restart_stamp().unwrap().unwrap();
        assert!(c.save_restart_projection(first, b"opaque").unwrap());
        assert_eq!(c.restart_stamp().unwrap(), Some(first));
        let got = c.load_restart_projection(6).unwrap().unwrap();
        assert_eq!(got.hot_ids, vec![id]);
        assert_eq!(got.bytes, b"opaque");
        assert_eq!(c.load_restart_projection(5).unwrap(), Err(RestartMiss::TooLarge));
        let head = c.head_seq().unwrap();
        c.add_coverage(CoverageMark { source: stream, stage: StageId(1), span: Interval::new(1, 2).unwrap() }).unwrap();
        assert_eq!(c.head_seq().unwrap(), head);
        assert_eq!(c.load_restart_projection(100).unwrap(), Err(RestartMiss::NotSaved));
        assert!(!c.save_restart_projection(first, b"stale").unwrap());
        let covered = c.restart_stamp().unwrap().unwrap();
        assert!(covered.revision > first.revision);
        c.clear_location(id, StorageTier::Tier0Hot).unwrap();
        let cleared = c.restart_stamp().unwrap().unwrap();
        assert!(cleared.revision > covered.revision);
        assert!(c.save_restart_projection(cleared, b"new").unwrap());
        assert!(c.load_restart_projection(100).unwrap().unwrap().hot_ids.is_empty());
        c.close(Duration::from_secs(1)).unwrap();
        let reopened = RedbCatalog::open(&path).unwrap();
        assert_eq!(reopened.restart_stamp().unwrap(), Some(cleared));
        assert_eq!(reopened.load_restart_projection(100).unwrap().unwrap().bytes, b"new");
    }

    #[test]
    fn hot_index_tracks_new_hot_copies_and_failed_writes_do_not_advance_stamp() {
        let temp = tempfile::tempdir().unwrap();
        let c = RedbCatalog::open(temp.path().join("catalog.redb")).unwrap();
        let stream = c.register_stream(info()).unwrap();
        c.restart_stamp().unwrap();
        let id = Uuid([2; 16]);
        c.insert(vec![NewPayload {
            header: HeaderBuf::build(&HeaderSpec::raw(id, stream, 1, 2)).unwrap(),
            location: None,
            coverage: None,
        }])
        .unwrap();
        c.set_location(id, hot()).unwrap();
        let stamp = c.restart_stamp().unwrap().unwrap();
        assert!(c.set_location(Uuid([3; 16]), hot()).is_err());
        assert_eq!(c.restart_stamp().unwrap(), Some(stamp));
        assert!(c.save_restart_projection(stamp, b"hot").unwrap());
        assert_eq!(c.load_restart_projection(10).unwrap().unwrap().hot_ids, vec![id]);
        c.clear_locations(&[id], StorageTier::Tier0Hot).unwrap();
        let stamp = c.restart_stamp().unwrap().unwrap();
        assert!(c.save_restart_projection(stamp, b"cold").unwrap());
        assert!(c.load_restart_projection(10).unwrap().unwrap().hot_ids.is_empty());
    }
    #[test]
    fn known_native_stream_noop_repair_and_scrub_preserve_checkpoint() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("native.redb");
        let c = RedbCatalog::open(&path).unwrap();
        let mut native = info();
        native.name = "synthetic.audio".into();
        native.shape_name = "PCM_F32LEx2".into();
        c.register_stream(native).unwrap();
        let stamp = c.restart_stamp().unwrap().unwrap();
        assert!(c.save_restart_projection(stamp, b"native").unwrap());
        c.close(Duration::from_secs(1)).unwrap();
        let c = RedbCatalog::open(&path).unwrap();
        assert!(c.recording_restored().is_empty());
        assert_eq!(c.load_restart_projection(100).unwrap().unwrap().stamp, stamp);
        c.request_scrub().unwrap();
        let stamp = c.restart_stamp().unwrap().unwrap();
        assert!(c.save_restart_projection(stamp, b"scrub").unwrap());
        let mut swapped = false;
        for _ in 0..10_000 {
            if matches!(c.scrub_step(100).unwrap(), crate::ScrubProgress::Swapped) {
                swapped = true;
                break;
            }
        }
        assert!(swapped);
        assert_eq!(c.load_restart_projection(100).unwrap().unwrap().stamp, stamp);
        c.close(Duration::from_secs(1)).unwrap();
        let c = RedbCatalog::open(&path).unwrap();
        assert_eq!(c.load_restart_projection(100).unwrap().unwrap().bytes, b"scrub");
    }

    #[test]
    fn a_new_file_has_tracking_and_a_corrupt_cache_falls_back() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("catalog.redb");
        let c = RedbCatalog::open(&path).unwrap();
        c.register_stream(info()).unwrap();
        let stamp = c.restart_stamp().unwrap().unwrap();
        c.close(Duration::from_secs(1)).unwrap();
        {
            let db = redb::Database::create(&path).unwrap();
            let txn = db.begin_read().unwrap();
            let table = txn.open_table(META).unwrap();
            assert_eq!(
                codec::u64_at(table.get(b"format".as_slice()).unwrap().unwrap().value(), 0).unwrap(),
                crate::redb_catalog::FORMAT
            );
            assert_eq!(super::stamp(&crate::store::R::new(db.begin_read().unwrap())).unwrap(), Some(stamp));
            assert_eq!(table.get(b"restart_hot_ready".as_slice()).unwrap().unwrap().value(), &[1]);
        }
        let c = RedbCatalog::open(&path).unwrap();
        assert!(c.save_restart_projection(stamp, b"opaque").unwrap());
        c.close(Duration::from_secs(1)).unwrap();
        {
            let db = redb::Database::create(&path).unwrap();
            let txn = db.begin_write().unwrap();
            txn.open_table(RESTART_PROJECTION).unwrap().insert(b"current".as_slice(), b"bad".as_slice()).unwrap();
            txn.commit().unwrap();
        }
        let c = RedbCatalog::open(&path).unwrap();
        assert_eq!(c.load_restart_projection(100).unwrap(), Err(RestartMiss::Corrupt));
        c.close(Duration::from_secs(1)).unwrap();
        {
            let db = redb::Database::create(&path).unwrap();
            let txn = db.begin_write().unwrap();
            txn.open_table(META).unwrap().remove(b"restart_revision".as_slice()).unwrap();
            txn.commit().unwrap();
        }
        assert!(RedbCatalog::open(&path).is_err(), "tracking corruption is not a cache miss");
    }

    #[test]
    fn purge_steps_invalidate_without_new_head_and_remove_exact_hot_ids() {
        let temp = tempfile::tempdir().unwrap();
        let c = RedbCatalog::open(temp.path().join("catalog.redb")).unwrap();
        let stream = c.register_stream(info()).unwrap();
        let id = Uuid([7; 16]);
        c.insert(vec![NewPayload {
            header: HeaderBuf::build(&HeaderSpec::raw(id, stream, 1, 2)).unwrap(),
            location: Some(hot()),
            coverage: None,
        }])
        .unwrap();
        let stamp = c.restart_stamp().unwrap().unwrap();
        assert!(c.save_restart_projection(stamp, b"before").unwrap());
        let head = c.head_seq().unwrap();
        c.purge(crate::PurgeSelector::Ids(vec![id]), "test", "synthetic", Uuid([8; 16]), 10).unwrap();
        assert_eq!(c.head_seq().unwrap(), head);
        assert_eq!(c.load_restart_projection(100).unwrap(), Err(RestartMiss::NotSaved));
        let after = c.restart_stamp().unwrap().unwrap();
        assert!(after.revision > stamp.revision);
        assert!(c.save_restart_projection(after, b"after").unwrap());
        assert!(c.load_restart_projection(100).unwrap().unwrap().hot_ids.is_empty());
    }

    #[test]
    fn exhausted_revision_refuses_mutation_atomically() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("catalog.redb");
        let c = RedbCatalog::open(&path).unwrap();
        let stamp = c.restart_stamp().unwrap().unwrap();
        assert!(c.save_restart_projection(stamp, b"empty").unwrap());
        c.close(Duration::from_secs(1)).unwrap();
        {
            let db = redb::Database::create(&path).unwrap();
            let txn = db.begin_write().unwrap();
            txn.open_table(META)
                .unwrap()
                .insert(b"restart_revision".as_slice(), u64::MAX.to_be_bytes().as_slice())
                .unwrap();
            txn.commit().unwrap();
        }
        let c = RedbCatalog::open(&path).unwrap();
        assert!(c.register_stream(info()).is_err());
        assert!(c.streams(None, 10).unwrap().is_empty());
        assert_eq!(c.restart_stamp().unwrap().unwrap().revision, u64::MAX);
    }
    #[test]
    fn privacy_purge_deletes_stale_projection_before_scrub_can_copy_it() {
        const MARKER: &[u8] = b"synthetic-restart-projection-private-marker-7f43c9d2-erase-completely";
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("privacy.redb");
        let c = RedbCatalog::open(&path).unwrap();
        let stream = c.register_stream(info()).unwrap();
        let id = Uuid([10; 16]);
        c.insert(vec![NewPayload {
            header: HeaderBuf::build(&HeaderSpec::raw(id, stream, 1, 2)).unwrap(),
            location: None,
            coverage: None,
        }])
        .unwrap();
        let stamp = c.restart_stamp().unwrap().unwrap();
        assert!(c.save_restart_projection(stamp, MARKER).unwrap());
        c.purge(crate::PurgeSelector::Ids(vec![id]), "test", "synthetic", Uuid([11; 16]), 10).unwrap();
        assert_eq!(c.load_restart_projection(1024).unwrap(), Err(RestartMiss::NotSaved));
        c.close(Duration::from_secs(1)).unwrap();
        {
            let db = redb::Database::create(&path).unwrap();
            let txn = db.begin_read().unwrap();
            let projection = txn.open_table(RESTART_PROJECTION).unwrap();
            assert!(
                projection.get(b"current".as_slice()).unwrap().is_none(),
                "invalidated projection bytes must be deleted, not merely hidden by a changed revision"
            );
        }
        let c = RedbCatalog::open(&path).unwrap();
        c.request_scrub().unwrap();
        let mut swapped = false;
        for _ in 0..10_000 {
            if matches!(c.scrub_step(100).unwrap(), crate::ScrubProgress::Swapped) {
                swapped = true;
                break;
            }
        }
        assert!(swapped);
        c.close(Duration::from_secs(1)).unwrap();
        {
            let db = redb::Database::create(&path).unwrap();
            let txn = db.begin_read().unwrap();
            assert!(txn.open_table(RESTART_PROJECTION).unwrap().get(b"current".as_slice()).unwrap().is_none());
        }
        assert!(
            !std::fs::read(&path).unwrap().windows(MARKER.len()).any(|bytes| bytes == MARKER),
            "the replacement scrub file must not retain the erased opaque marker"
        );
        let c = RedbCatalog::open(&path).unwrap();
        assert_eq!(c.load_restart_projection(1024).unwrap(), Err(RestartMiss::NotSaved));
    }
}
