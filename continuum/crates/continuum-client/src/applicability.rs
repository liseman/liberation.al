//! Durable original recording applicability. A complete empty page is meaningful only with certified history;
//! erased or unresolved inputs remain unknown, and preserved originals still require analysis completion receipts.

use continuum_core::schema::continuum as fb;
use continuum_core::{StreamId, Uuid};
use flatbuffers::FlatBufferBuilder;

use crate::messages::finish;
use crate::{Client, Error, unexpected};

/// Broker minor required for interval-level original recording history.
pub const RECORDING_SCHEMA_MINOR: u16 = 28;
const MAX_RUNS: usize = 4096;
const MAX_BYTES: u32 = 1024 * 1024;

/// Original recording identity and truthful input runs, unaffected by playback aging.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordingInput {
    /// Immutable original temporal key, including instant video eligibility rows.
    pub start_ns: u64,
    /// Earliest original raw ingress sequence; chunk packaging cannot move the era boundary.
    pub first_seq: u64,
    /// Original chunk named by source-bound analysis receipts.
    pub chunk_id: Uuid,
    /// Normalized positive-duration original input runs, never an encoded playback hull.
    pub spans: Vec<(u64, u64)>,
    /// Historical source epoch, which may legitimately precede the current source epoch.
    pub source_generation: u64,
    /// Original registered PCM channel count; unknown remains unavailable evidence.
    pub channels: Option<u16>,
    /// Original registered shape.
    pub shape_name: String,
    /// Original registered shape version.
    pub shape_version: u32,
}

/// Exclusive last examined ledger row, including unknown rows omitted from positive inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecordingCursor {
    /// Original row start, nanoseconds UTC.
    pub start_ns: u64,
    /// Stable ledger row identity.
    pub id: Uuid,
}

/// Bounded page from one catalog snapshot. Keep the first watermark while paging, then guard publication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordingPage {
    /// First original ingress sequence eligible for this certified era.
    pub era_start_seq: u64,
    /// Legacy unknown history overlaps this query; diagnostic only.
    pub legacy_overlap: bool,
    /// Original inputs whose runs intersect the requested window.
    pub inputs: Vec<RecordingInput>,
    /// Intersecting erased, unresolved, or otherwise unknown input intervals.
    pub unknown: Vec<(u64, u64)>,
    /// Continue even an empty page when this is present.
    pub next: Option<RecordingCursor>,
    /// Only complete certified-era history can prove no applicable input in this era.
    pub history_known: bool,
    /// Invalidation revision from this same snapshot.
    pub revision: u64,
    /// Catalog commit watermark from this same snapshot.
    pub through_seq: u64,
    /// Current source generation, distinct from each original input's historical generation.
    pub source_generation: u64,
    /// Index rows examined, including rows not returned.
    pub scanned: usize,
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Protocol(format!("recording applicability: {message}"))
}

fn cursor(value: fb::RecordingCursor<'_>) -> Result<RecordingCursor, Error> {
    let id = Uuid(value.id().ok_or_else(|| invalid("cursor without identity"))?.0);
    if id == Uuid([0; 16]) {
        return Err(invalid("nil cursor identity"));
    }
    Ok(RecordingCursor { start_ns: value.start_ns(), id })
}

fn runs(values: Option<flatbuffers::Vector<'_, fb::TimeRange>>, label: &str) -> Result<Vec<(u64, u64)>, Error> {
    let values = values.ok_or_else(|| invalid(format!("missing {label}")))?;
    if values.len() > MAX_RUNS {
        return Err(invalid(format!("excessive {label}")));
    }
    let mut result = Vec::with_capacity(values.len());
    for range in values {
        let span = (range.start_ns(), range.end_ns());
        if span.0 >= span.1 || result.last().is_some_and(|last: &(u64, u64)| last.1 >= span.0) {
            return Err(invalid(format!("noncanonical {label}")));
        }
        result.push(span);
    }
    Ok(result)
}

impl Client {
    /// Read original inputs and unsafe history independently of remaining playback bytes.
    /// # Errors
    /// Older brokers, malformed/boundedness-violating replies, stale revisions, and ordinary broker refusals.
    #[allow(clippy::too_many_arguments)]
    pub async fn recording_page(
        &self,
        stream_id: StreamId,
        stream: &str,
        window: (u64, u64),
        after: Option<RecordingCursor>,
        expected_revision: Option<u64>,
        max_items: u32,
        max_scan: u32,
        max_bytes: u32,
    ) -> Result<RecordingPage, Error> {
        if self.broker_schema().1 < RECORDING_SCHEMA_MINOR {
            return Err(invalid("broker predates schema28 recording history"));
        }
        if stream_id.0 == 0
            || stream.is_empty()
            || window.0 >= window.1
            || max_items > 128
            || max_scan > 4096
            || max_bytes > MAX_BYTES
            || after.is_some_and(|c| c.id == Uuid([0; 16]))
        {
            return Err(invalid("invalid source, window, cursor, or limits"));
        }
        let item_limit = if max_items == 0 { 64 } else { max_items as usize };
        let scan_limit = if max_scan == 0 { 256 } else { max_scan as usize };
        let byte_limit = if max_bytes == 0 { 256 * 1024 } else { max_bytes as usize };
        let reply = self
            .shared
            .request(|request_id| {
                let mut b = FlatBufferBuilder::new();
                let stream = b.create_string(stream);
                let range = fb::TimeRange::new(window.0, window.1);
                let id = after.map(|c| fb::Uuid(c.id.0));
                let after = after.map(|c| {
                    fb::RecordingCursor::create(
                        &mut b,
                        &fb::RecordingCursorArgs { start_ns: c.start_ns, id: id.as_ref() },
                    )
                });
                let ask = fb::ListRecordingInputs::create(
                    &mut b,
                    &fb::ListRecordingInputsArgs {
                        stream: Some(stream),
                        stream_id: stream_id.0,
                        range: Some(&range),
                        after,
                        has_revision: expected_revision.is_some(),
                        expected_revision: expected_revision.unwrap_or(0),
                        max_items,
                        max_scan,
                        max_bytes,
                    },
                );
                finish(&mut b, request_id, fb::ControlPayload::ListRecordingInputs, ask.as_union_value())
            })
            .await?;
        if reply.as_bytes().len().saturating_add(4) > byte_limit {
            return Err(invalid("reply exceeds byte budget"));
        }
        let value =
            reply.envelope().payload_as_recording_inputs().ok_or_else(|| unexpected("RecordingInputs", &reply))?;
        if value.stream_id() != stream_id.0 {
            return Err(invalid("reply source identity differs from request"));
        }
        decode_page(value, window, after, expected_revision, item_limit, scan_limit)
    }
}

fn decode_page(
    value: fb::RecordingInputs<'_>,
    window: (u64, u64),
    after: Option<RecordingCursor>,
    expected_revision: Option<u64>,
    max_items: usize,
    max_scan: usize,
) -> Result<RecordingPage, Error> {
    if expected_revision.is_some_and(|v| v != value.revision())
        || value.revision() == u64::MAX
        || value.source_generation() == u64::MAX
        || (value.history_known() && value.era_start_seq() == 0)
    {
        return Err(invalid("snapshot revision or source generation unavailable"));
    }
    let inputs = value.inputs().ok_or_else(|| invalid("missing inputs"))?;
    if inputs.len() > max_items || inputs.len() > value.scanned() as usize || value.scanned() as usize > max_scan {
        return Err(invalid("page exceeds requested limits"));
    }
    let mut result = Vec::with_capacity(inputs.len());
    let mut previous = after;
    for input in inputs {
        let chunk_id = Uuid(input.chunk_id().ok_or_else(|| invalid("input without identity"))?.0);
        let spans = runs(input.spans(), "input runs")?;
        let shape_name = input.shape_name().ok_or_else(|| invalid("input without shape"))?;
        if chunk_id == Uuid([0; 16])
            || (!spans.is_empty() && !spans.iter().any(|&(a, b)| a < window.1 && b > window.0))
            || shape_name.is_empty()
            || shape_name.len() > 4096
            || input.source_generation() == u64::MAX
            || input.first_seq() < value.era_start_seq()
            || input.first_seq() == 0
            || input.first_seq() > value.through_seq()
        {
            return Err(invalid("invalid original input"));
        }
        let key = RecordingCursor { start_ns: input.start_ns(), id: chunk_id };
        if previous.is_some_and(|p| key <= p) {
            return Err(invalid("inputs not ordered after cursor"));
        }
        previous = Some(key);
        result.push(RecordingInput {
            start_ns: input.start_ns(),
            first_seq: input.first_seq(),
            chunk_id,
            spans,
            source_generation: input.source_generation(),
            channels: (input.channels() != 0).then_some(input.channels()),
            shape_name: shape_name.into(),
            shape_version: input.shape_version(),
        });
    }
    let unknown = runs(value.unknown(), "unknown intervals")?;
    if result.len().saturating_add(unknown.len()) > max_items {
        return Err(invalid("positive and unknown rows exceed the requested item budget"));
    }
    if unknown.iter().any(|&(a, b)| a < window.0 || b > window.1) {
        return Err(invalid("unknown interval outside window"));
    }
    let next = value.next().map(cursor).transpose()?;
    if after.zip(next).is_some_and(|(a, b)| b <= a)
        || previous.zip(next).is_some_and(|(a, b)| b < a)
        || (next.is_some() && value.scanned() == 0)
    {
        return Err(invalid("page cursor did not advance"));
    }
    Ok(RecordingPage {
        era_start_seq: value.era_start_seq(),
        legacy_overlap: value.legacy_overlap(),
        inputs: result,
        unknown,
        next,
        history_known: value.history_known(),
        revision: value.revision(),
        through_seq: value.through_seq(),
        source_generation: value.source_generation(),
        scanned: value.scanned() as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_unknown(
        known: bool,
        revision: u64,
        scanned: u32,
        next_start: Option<u64>,
    ) -> Result<RecordingPage, Error> {
        let mut b = FlatBufferBuilder::new();
        let inputs = b.create_vector::<flatbuffers::WIPOffset<fb::RecordingInput<'_>>>(&[]);
        let unknown = b.create_vector(&[fb::TimeRange::new(12, 14)]);
        let id = fb::Uuid([1; 16]);
        let next = next_start.map(|start| {
            fb::RecordingCursor::create(&mut b, &fb::RecordingCursorArgs { start_ns: start, id: Some(&id) })
        });
        let value = fb::RecordingInputs::create(
            &mut b,
            &fb::RecordingInputsArgs {
                inputs: Some(inputs),
                unknown: Some(unknown),
                next,
                history_known: known,
                era_start_seq: 1,
                through_seq: 10,
                revision,
                scanned,
                ..Default::default()
            },
        );
        b.finish(value, None);
        decode_page(
            flatbuffers::root::<fb::RecordingInputs<'_>>(b.finished_data()).unwrap(),
            (10, 20),
            Some(RecordingCursor { start_ns: 5, id: Uuid([1; 16]) }),
            Some(3),
            1,
            10,
        )
    }

    #[test]
    fn legacy_and_erased_history_remain_distinct_from_empty() {
        let page = decode_unknown(false, 3, 1, None).unwrap();
        assert!(!page.history_known);
        assert_eq!(page.unknown, [(12, 14)]);
        assert!(page.inputs.is_empty());
    }

    #[test]
    fn stale_revisions_nonadvancing_cursors_and_scan_overflow_are_rejected() {
        assert!(decode_unknown(true, 4, 1, None).is_err());
        assert!(decode_unknown(true, 3, 0, Some(6)).is_err());
        assert!(decode_unknown(true, 3, 1, Some(5)).is_err());
        assert!(decode_unknown(true, 3, 11, None).is_err());
        assert_eq!(decode_unknown(true, 3, 1, Some(6)).unwrap().next.unwrap().start_ns, 6);
    }

    fn decode_instant_video(first_seq: u64) -> Result<RecordingPage, Error> {
        let mut b = FlatBufferBuilder::new();
        let id = fb::Uuid([9; 16]);
        let spans = b.create_vector::<fb::TimeRange>(&[]);
        let shape = b.create_string("H264");
        let input = fb::RecordingInput::create(
            &mut b,
            &fb::RecordingInputArgs {
                start_ns: 12,
                first_seq,
                chunk_id: Some(&id),
                spans: Some(spans),
                shape_name: Some(shape),
                shape_version: 1,
                ..Default::default()
            },
        );
        let inputs = b.create_vector(&[input]);
        let unknown = b.create_vector::<fb::TimeRange>(&[]);
        let value = fb::RecordingInputs::create(
            &mut b,
            &fb::RecordingInputsArgs {
                era_start_seq: 5,
                through_seq: 10,
                legacy_overlap: true,
                history_known: true,
                inputs: Some(inputs),
                unknown: Some(unknown),
                scanned: 1,
                ..Default::default()
            },
        );
        b.finish(value, None);
        decode_page(
            flatbuffers::root::<fb::RecordingInputs<'_>>(b.finished_data()).unwrap(),
            (10, 20),
            None,
            None,
            1,
            10,
        )
    }

    #[test]
    fn instant_video_eligibility_has_no_invented_duration_and_legacy_is_diagnostic() {
        let page = decode_instant_video(5).unwrap();
        assert!(page.legacy_overlap);
        assert!(page.unknown.is_empty());
        assert_eq!(page.inputs[0].start_ns, 12);
        assert!(page.inputs[0].spans.is_empty());
        assert!(decode_instant_video(4).is_err());
        assert!(decode_instant_video(11).is_err());
    }

    #[test]
    fn unknown_runs_share_the_requested_item_budget() {
        let mut b = FlatBufferBuilder::new();
        let inputs = b.create_vector::<flatbuffers::WIPOffset<fb::RecordingInput<'_>>>(&[]);
        let unknown = b.create_vector(&[fb::TimeRange::new(12, 14), fb::TimeRange::new(16, 18)]);
        let value = fb::RecordingInputs::create(
            &mut b,
            &fb::RecordingInputsArgs {
                era_start_seq: 1,
                through_seq: 10,
                history_known: true,
                inputs: Some(inputs),
                unknown: Some(unknown),
                scanned: 1,
                ..Default::default()
            },
        );
        b.finish(value, None);
        let value = flatbuffers::root::<fb::RecordingInputs<'_>>(b.finished_data()).unwrap();
        assert!(decode_page(value, (10, 20), None, None, 1, 10).is_err());
        assert_eq!(decode_page(value, (10, 20), None, None, 2, 10).unwrap().unknown.len(), 2);
    }

    #[test]
    fn missing_unknown_vector_is_not_evaluated_absence() {
        let mut b = FlatBufferBuilder::new();
        let inputs = b.create_vector::<flatbuffers::WIPOffset<fb::RecordingInput<'_>>>(&[]);
        let value = fb::RecordingInputs::create(
            &mut b,
            &fb::RecordingInputsArgs { inputs: Some(inputs), history_known: true, ..Default::default() },
        );
        b.finish(value, None);
        assert!(
            decode_page(
                flatbuffers::root::<fb::RecordingInputs<'_>>(b.finished_data()).unwrap(),
                (1, 2),
                None,
                None,
                1,
                10
            )
            .is_err()
        );
    }
}
