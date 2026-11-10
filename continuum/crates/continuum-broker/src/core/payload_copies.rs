//! Nonabsorbing durable copies of derived payloads: original identity and provenance stay indexed.

use super::*;

impl Sched {
    pub(super) fn commit_payload_copy(
        &mut self,
        session: SessionId,
        request_id: u64,
        payload: Uuid,
        location: StorageLocator,
    ) {
        let Some(keeper) = self.keeper_name(session).filter(|_| self.workers.contains_key(&session)) else {
            return self.reply_error(
                session,
                request_id,
                ErrorCode::NotAllowed,
                "only a registered copy keeper commits payload copies",
            );
        };
        if let Err(error) =
            self.urgent.try_send(Urgent::CommitPayloadCopy { session, request_id, payload, location, keeper })
        {
            self.refuse_busy(session, request_id, &error);
        }
    }

    pub(super) fn payload_copy_committed(
        &mut self,
        session: SessionId,
        request_id: u64,
        payload: Uuid,
        result: Result<Option<(StreamId, Interval)>, (ErrorCode, String)>,
    ) {
        match result {
            Ok(added) => {
                if let Some((stream, span)) = added {
                    let _ = self.index.add_copy(stream, span, StorageTier::Tier1Warm);
                }
                let bytes = protocol::payload_copy_committed(&mut self.builder, request_id, payload);
                self.push(session, bytes);
            }
            Err((code, message)) => self.reply_error(session, request_id, code, &message),
        }
    }
}

pub(super) fn commit_copy_job(
    catalog: &dyn Catalog,
    session: SessionId,
    request_id: u64,
    payload: Uuid,
    location: StorageLocator,
    keeper: &str,
    events: &mpsc::UnboundedSender<Event>,
) {
    // Registration writes can be waiting on a full urgent queue. Persist responsibility before acknowledging bytes.
    let result = catalog
        .add_keeper(keeper)
        .map_err(|e| (error_code(&e), e.to_string()))
        .and_then(|()| store_copy(catalog, payload, location));
    let _ = events.send(Event::PayloadCopyCommitted { session, request_id, payload, result });
}

fn store_copy(
    catalog: &dyn Catalog,
    payload: Uuid,
    location: StorageLocator,
) -> Result<Option<(StreamId, Interval)>, (ErrorCode, String)> {
    let map = |error: CatalogError| (error_code(&error), error.to_string());
    match &location {
        StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri, offset, length }
            if !uri.is_empty() && *length > 0 && offset.checked_add(*length).is_some() => {}
        _ => return Err((ErrorCode::BadRequest, "payload copy must be a nonempty Tier1Warm chunk range".into())),
    }
    let known = match catalog.get(payload).map_err(map)? {
        Some(Resolved::Payload(known)) if known.frames.is_none() => known,
        Some(_) => {
            return Err((ErrorCode::BadRequest, "payload copies cannot target absorbed frames or chunks".into()));
        }
        None => return Err((ErrorCode::NotFound, format!("no visible payload {payload}"))),
    };
    if !catalog.stream(known.header.stream()).map_err(map)?.is_some_and(|info| info.kind == CatalogKind::Derived) {
        return Err((ErrorCode::BadRequest, "payload copies require a Derived stream".into()));
    }
    if known.locations.iter().any(|l| matches!(l, StorageLocator::Inline { .. })) {
        return Err((ErrorCode::BadRequest, "inline bytes already have durable storage".into()));
    }
    if let Some(existing) = known.locations.iter().find(|l| l.tier() == StorageTier::Tier1Warm) {
        return if existing == &location {
            Ok(None)
        } else {
            Err((ErrorCode::Conflict, "payload already has a different warm copy".into()))
        };
    }
    catalog.set_location(payload, location).map_err(map)?;
    Ok(known.header.span().map(|span| (known.header.stream(), span)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_catalog::{MemoryCatalog, StreamInfo};
    use continuum_core::{DomainLayer, HeaderSpec, PayloadFormat};

    #[test]
    fn a_copy_cannot_attach_to_a_payload_hidden_by_an_in_progress_purge() {
        use continuum_catalog::{PurgeProgress, PurgeSelector};
        let catalog = MemoryCatalog::new();
        let stream = catalog
            .register_stream(StreamInfo {
                name: "embedding".into(),
                kind: CatalogKind::Derived,
                layer: DomainLayer::Layer2Decision,
                format: PayloadFormat::RawBytes,
                shape_name: String::new(),
                shape_version: 0,
                rate_hz: None,
                anchor: None,
            })
            .unwrap();
        let payload = Uuid([1; 16]);
        for n in 1..=3u8 {
            let mut spec = HeaderSpec::raw(Uuid([n; 16]), stream, u64::from(n) * 10, u64::from(n) * 10 + 5);
            if n != 1 {
                spec.parents = vec![payload];
            }
            catalog
                .insert(vec![NewPayload { header: HeaderBuf::build(&spec).unwrap(), location: None, coverage: None }])
                .unwrap();
        }
        let purge = Uuid([4; 16]);
        catalog.start_purge(PurgeSelector::Ids(vec![payload]), "test", "", purge, 1).unwrap();
        assert!(matches!(catalog.purge_step(purge, 1).unwrap(), PurgeProgress::Running { .. }));
        assert!(catalog.get(payload).unwrap().is_none());
        let location =
            StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri: "copy.cnk".into(), offset: 0, length: 65536 };
        assert_eq!(store_copy(&catalog, payload, location).unwrap_err().0, ErrorCode::NotFound);
        assert_eq!(catalog.head_seq().unwrap(), 3);
    }

    #[test]
    fn derived_copy_keeps_original_header_sequence_and_provenance_and_retries_once() {
        let catalog = MemoryCatalog::new();
        let stream = |name: &str, kind| {
            catalog
                .register_stream(StreamInfo {
                    name: name.into(),
                    kind,
                    layer: DomainLayer::Layer2Decision,
                    format: PayloadFormat::RawBytes,
                    shape_name: String::new(),
                    shape_version: 0,
                    rate_hz: None,
                    anchor: None,
                })
                .unwrap()
        };
        let raw = stream("camera", CatalogKind::Sensor);
        let derived = stream("embedding", CatalogKind::Derived);
        let parent = Uuid([1; 16]);
        let payload = Uuid([2; 16]);
        catalog
            .insert(vec![NewPayload {
                header: HeaderBuf::build(&HeaderSpec::raw(parent, raw, 10, 20)).unwrap(),
                location: None,
                coverage: None,
            }])
            .unwrap();
        let mut spec = HeaderSpec::raw(payload, derived, 10, 20);
        spec.parents = vec![parent];
        let header = HeaderBuf::build(&spec).unwrap();
        let seq =
            catalog.insert(vec![NewPayload { header: header.clone(), location: None, coverage: None }]).unwrap()[0].seq;
        let location =
            StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri: "derived.cnk".into(), offset: 0, length: 65536 };
        for (uri, offset, length) in [("", 0, 1), ("bad", 0, 0), ("bad", u64::MAX, 2)] {
            let invalid = StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri: uri.into(), offset, length };
            assert_eq!(store_copy(&catalog, payload, invalid).unwrap_err().0, ErrorCode::BadRequest);
        }
        assert_eq!(store_copy(&catalog, parent, location.clone()).unwrap_err().0, ErrorCode::BadRequest);
        assert_eq!(
            store_copy(&catalog, payload, location.clone()).unwrap(),
            Some((derived, Interval::new(10, 20).unwrap()))
        );
        assert_eq!(store_copy(&catalog, payload, location.clone()).unwrap(), None);
        let Some(Resolved::Payload(known)) = catalog.get(payload).unwrap() else {
            panic!("original payload must remain indexed")
        };
        assert_eq!(known.header, header);
        assert_eq!(known.commit_seq, seq);
        assert_eq!(known.locations, vec![location.clone()]);
        assert!(known.frames.is_none());
        assert_eq!(catalog.parents(payload).unwrap(), vec![parent]);
        assert_eq!(catalog.children(parent).unwrap(), vec![payload]);
        let other =
            StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri: "other.cnk".into(), offset: 0, length: 65536 };
        assert_eq!(store_copy(&catalog, payload, other).unwrap_err().0, ErrorCode::Conflict);
        let report =
            catalog.purge(continuum_catalog::PurgeSelector::Ids(vec![parent]), "test", "", Uuid([3; 16]), 1).unwrap();
        assert!(report.removed.iter().any(|removed| removed.id == payload && removed.locations == [location.clone()]));
        assert_eq!(store_copy(&catalog, payload, location).unwrap_err().0, ErrorCode::NotFound);
    }
}
