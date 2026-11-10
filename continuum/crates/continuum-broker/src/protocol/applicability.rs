//! Bounded schema28 recording applicability. Original inputs and unsafe history remain independent of
//! surviving playback locations; no reply claims analysis completion merely because source bytes were aged.

use continuum_catalog::{RecordingCursor, RecordingPage};
use continuum_core::schema::continuum as fb;
use continuum_core::{Interval, StreamId, Uuid};
use flatbuffers::FlatBufferBuilder;

use super::finish;

/// Owned request values, validated before queueing a reader snapshot.
#[derive(Clone, Debug)]
pub struct ListRecordingAsk {
    /// Immutable registered source name.
    pub stream: String,
    /// Expected immutable source ID; name lookup must still match it.
    pub stream_id: StreamId,
    /// Padded analysis window.
    pub window: Interval,
    /// Exclusive last examined row.
    pub after: Option<RecordingCursor>,
    /// Snapshot invalidation revision, when supplied.
    pub expected_revision: Option<u64>,
    /// Maximum positive/unknown ledger rows returned.
    pub max_items: usize,
    /// Maximum index rows examined.
    pub max_scan: usize,
    /// Maximum actual wire bytes, including the envelope and length prefix.
    pub max_bytes: usize,
}

fn limit(n: u32, default: usize, maximum: usize) -> Result<usize, String> {
    if n as usize > maximum {
        return Err("recording page limit exceeds maximum".into());
    }
    Ok(if n == 0 { default } else { n as usize })
}

/// Parse and bound an applicability request before dispatch.
/// # Errors
/// Missing source/window/identity, malformed cursor, or excessive limits.
pub fn list_ask(value: &fb::ListRecordingInputs<'_>) -> Result<ListRecordingAsk, String> {
    let stream = value.stream().ok_or("recording page without source")?;
    let range = value.range().ok_or("recording page without range")?;
    let window = Interval::new(range.start_ns(), range.end_ns()).ok_or("invalid recording window")?;
    if stream.is_empty() || stream.len() > 4096 || value.stream_id() == 0 {
        return Err("invalid recording source identity".into());
    }
    let after = value
        .after()
        .map(|cursor| {
            let id = Uuid(cursor.id().ok_or("recording cursor without ID")?.0);
            if id == Uuid([0; 16]) {
                return Err("nil recording cursor ID");
            }
            Ok(RecordingCursor { start_ns: cursor.start_ns(), id })
        })
        .transpose()?;
    Ok(ListRecordingAsk {
        stream: stream.into(),
        stream_id: StreamId(value.stream_id()),
        window,
        after,
        expected_revision: value.has_revision().then_some(value.expected_revision()),
        max_items: limit(value.max_items(), 64, 128)?,
        max_scan: limit(value.max_scan(), 256, 4096)?,
        max_bytes: limit(value.max_bytes(), 256 * 1024, 1024 * 1024)?,
    })
}

/// Serialize one coherent bounded snapshot. An oversized answer is an explicit error, never silent truncation.
/// # Errors
/// The actual FlatBuffer plus transport prefix exceeds the requested byte budget.
pub fn page(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    stream: StreamId,
    page: &RecordingPage,
    max_bytes: usize,
) -> Result<Vec<u8>, String> {
    b.reset();
    let mut inputs = Vec::with_capacity(page.inputs.len());
    for input in &page.inputs {
        let id = fb::Uuid(input.chunk_id.0);
        let spans: Vec<_> = input.spans.iter().map(|span| fb::TimeRange::new(span.start_ns(), span.end_ns())).collect();
        let spans = b.create_vector(&spans);
        let shape = b.create_string(&input.shape_name);
        inputs.push(fb::RecordingInput::create(
            b,
            &fb::RecordingInputArgs {
                start_ns: input.start_ns,
                first_seq: input.first_seq,
                chunk_id: Some(&id),
                spans: Some(spans),
                source_generation: input.source_generation,
                channels: input.channels.unwrap_or(0),
                shape_name: Some(shape),
                shape_version: input.shape_version,
            },
        ));
    }
    let inputs = b.create_vector(&inputs);
    let unknown: Vec<_> = page.unknown.iter().map(|span| fb::TimeRange::new(span.start_ns(), span.end_ns())).collect();
    let unknown = b.create_vector(&unknown);
    let id = page.next.map(|cursor| fb::Uuid(cursor.id.0));
    let next = page.next.map(|cursor| {
        fb::RecordingCursor::create(b, &fb::RecordingCursorArgs { start_ns: cursor.start_ns, id: id.as_ref() })
    });
    let value = fb::RecordingInputs::create(
        b,
        &fb::RecordingInputsArgs {
            era_start_seq: page.era_start_seq,
            legacy_overlap: page.legacy_overlap,
            stream_id: stream.0,
            inputs: Some(inputs),
            unknown: Some(unknown),
            next,
            history_known: page.history_known,
            revision: page.revision,
            through_seq: page.through_seq,
            source_generation: page.source_generation,
            scanned: u32::try_from(page.scanned).map_err(|_| "recording scan count overflow")?,
        },
    );
    let bytes = finish(b, request_id, fb::ControlPayload::RecordingInputs, value.as_union_value());
    if bytes.len().saturating_add(4) > max_bytes {
        return Err("encoded recording page exceeds byte budget".into());
    }
    Ok(bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_requires_identity_and_bounded_limits() {
        let mut b = FlatBufferBuilder::new();
        let name = b.create_string("test.audio");
        let range = fb::TimeRange::new(1, 2);
        let ask = fb::ListRecordingInputs::create(
            &mut b,
            &fb::ListRecordingInputsArgs { stream: Some(name), range: Some(&range), ..Default::default() },
        );
        b.finish(ask, None);
        assert!(list_ask(&flatbuffers::root::<fb::ListRecordingInputs<'_>>(b.finished_data()).unwrap()).is_err());
        assert!(limit(129, 64, 128).is_err());
        assert_eq!(limit(0, 64, 128).unwrap(), 64);
    }

    #[test]
    fn wire_budget_counts_cursor_shapes_and_transport_envelope() {
        let input = continuum_catalog::RecordingInput {
            start_ns: 1,
            first_seq: 1,
            chunk_id: Uuid([2; 16]),
            spans: vec![Interval::new(1, 2).unwrap()],
            source_generation: 0,
            channels: Some(1),
            shape_name: "a".repeat(1024),
            shape_version: 1,
        };
        let value = RecordingPage {
            era_start_seq: 1,
            legacy_overlap: false,
            inputs: vec![input],
            unknown: vec![Interval::new(1, 2).unwrap()],
            next: Some(RecordingCursor { start_ns: 1, id: Uuid([2; 16]) }),
            history_known: true,
            revision: 4,
            through_seq: 8,
            source_generation: 2,
            scanned: 1,
        };
        let mut b = FlatBufferBuilder::new();
        let bytes = page(&mut b, 7, StreamId(1), &value, 4096).unwrap();
        assert!(page(&mut b, 7, StreamId(1), &value, bytes.len() + 3).is_err());
        let decoded = super::super::read_envelope(&bytes).unwrap().payload_as_recording_inputs().unwrap();
        assert_eq!(decoded.stream_id(), 1);
        assert_eq!(decoded.inputs().unwrap().len(), 1);
        assert_eq!(decoded.unknown().unwrap().len(), 1);
    }
}
