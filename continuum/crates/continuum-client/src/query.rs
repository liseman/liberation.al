//! Time queries (`Query`, answered by `QueryResult`): what a stream holds over a time range, as headers and
//! locators, whether the range is complete, the missing intervals, and for a ring stream reaching the live edge, the
//! unsealed tail to read from the ring.
//!
//! `HitOnly` returns what exists. `QuoteIfMissing` and `CommitWithinBudget` also get a `PromiseQuote` for the missing
//! part; [`Client::query_quoted`] returns it, and `crate::promise` has the promise calls.

use continuum_core::schema::continuum as fb;
use flatbuffers::FlatBufferBuilder;

use crate::messages::finish;
use crate::worker::{Item, item_of};
use crate::{Client, Error};

/// A time query.
#[derive(Clone, Debug)]
pub struct QuerySpec {
    /// The stream, by name.
    pub stream: String,
    /// The time range, UTC nanoseconds, half open.
    pub range: (u64, u64),
    /// What to do about what's missing.
    pub mode: fb::QueryMode,
    /// The band to quote or run missing work at.
    pub band: fb::PriorityBand,
    /// A latency budget in milliseconds; 0 for none.
    pub latency_budget_ms: u32,
    /// At most this many items; 0 for the broker's own limit.
    pub max_items: u32,
    /// A sequence query: the payloads in commit order after `after_seq`, the range (when not empty) only as a
    /// filter, `HitOnly` only. The result's `next_after` says where to continue. [`Client::query_all`] uses it.
    pub by_seq: bool,
    /// For a sequence query, where to start (0 for the beginning).
    pub after_seq: u64,
    /// Only payloads about any of these (empty for no filter).
    pub subjects: Vec<continuum_core::Uuid>,
    /// Only payloads starting before this, UTC nanoseconds; 0 for no bound.
    pub start_before: u64,
    /// Only these payloads (empty for no filter).
    pub ids: Vec<continuum_core::Uuid>,
    /// For a sequence query, the most catalog rows to scan before answering; 0 for the broker's default.
    pub max_scan: u32,
}

impl QuerySpec {
    /// What exists on `stream` over `range` (UTC ns, half open), with no budget, at P2, unlimited.
    pub fn hit_only(stream: impl Into<String>, range: (u64, u64)) -> Self {
        Self {
            stream: stream.into(),
            range,
            mode: fb::QueryMode::HitOnly,
            band: fb::PriorityBand::P2Nearline,
            latency_budget_ms: 0,
            max_items: 0,
            by_seq: false,
            after_seq: 0,
            subjects: Vec::new(),
            start_before: 0,
            ids: Vec::new(),
            max_scan: 0,
        }
    }
}

/// The part of a ring stream past its last sealed batch: read it from the ring directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingTail {
    /// The arena the ring is in.
    pub arena: String,
    /// The ring's offset in it.
    pub ring_offset: u64,
    /// The first unsealed record's sequence.
    pub first_seq: u64,
    /// The newest record's sequence.
    pub last_seq: u64,
}

/// What a query found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryResult {
    /// The payloads, in time order: their headers and where their bytes are.
    pub items: Vec<Item>,
    /// Whether the items cover the whole range.
    pub complete: bool,
    /// The intervals with nothing (UTC ns, half open).
    pub missing: Vec<(u64, u64)>,
    /// For a ring stream reaching the live edge, the unsealed tail.
    pub tail: Option<RingTail>,
    /// For a sequence query that stopped at its limits, the `after_seq` to continue from; 0 when nothing remains.
    pub next_after: u64,
    /// The page was cut short of the range (at `max_items`, or at the broker's inline or serialized-frame byte
    /// cap): more may follow the last item. `complete` is also false for a range with gaps, so this is the flag to page on.
    pub truncated: bool,
    /// Each item's commit sequence, in the same order as `items`: its position in commit order, which subscriptions
    /// and the event log use (0 where the broker doesn't give one).
    pub seqs: Vec<u64>,
    /// A `PromiseQuote` for the missing part follows ([`Client::query_quoted`] returns it).
    pub quoted: bool,
}

pub(crate) fn query(request_id: u64, spec: &QuerySpec) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let stream = b.create_string(&spec.stream);
    let range = fb::TimeRange::new(spec.range.0, spec.range.1);
    let uuids = |b: &mut FlatBufferBuilder<'static>, v: &[continuum_core::Uuid]| {
        (!v.is_empty()).then(|| {
            let ids: Vec<_> = v.iter().map(|u| fb::Uuid::new(&u.0)).collect();
            b.create_vector(&ids)
        })
    };
    let subjects = uuids(&mut b, &spec.subjects);
    let ids = uuids(&mut b, &spec.ids);
    let q = fb::Query::create(
        &mut b,
        &fb::QueryArgs {
            stream: Some(stream),
            // A sequence query over everything sends no range.
            range: (!spec.by_seq || spec.range.0 < spec.range.1).then_some(&range),
            latency_budget_ms: spec.latency_budget_ms,
            band: spec.band,
            mode: spec.mode,
            max_items: spec.max_items,
            by_seq: spec.by_seq,
            after_seq: spec.after_seq,
            subjects,
            start_before: spec.start_before,
            ids,
            max_scan: spec.max_scan,
        },
    );
    finish(&mut b, request_id, fb::ControlPayload::Query, q.as_union_value())
}

pub(crate) fn result_of(r: &fb::QueryResult<'_>) -> Result<QueryResult, Error> {
    let mut items = Vec::new();
    let mut seqs = Vec::new();
    for (i, it) in r.items().into_iter().flatten().enumerate() {
        items.push(item_of(&it).map_err(|e| Error::Protocol(format!("QueryResult item {i}'s {e}")))?);
        seqs.push(it.commit_seq());
    }
    let missing = r.missing().map(|v| v.iter().map(|t| (t.start_ns(), t.end_ns())).collect()).unwrap_or_default();
    let tail = r.tail().map(|t| RingTail {
        arena: t.arena().unwrap_or_default().to_owned(),
        ring_offset: t.ring_offset(),
        first_seq: t.first_seq(),
        last_seq: t.last_seq(),
    });
    Ok(QueryResult {
        items,
        complete: r.complete(),
        missing,
        tail,
        next_after: r.next_after(),
        truncated: r.truncated(),
        seqs,
        quoted: r.quoted(),
    })
}

impl Client {
    /// Asks what `spec.stream` holds over `spec.range`.
    ///
    /// # Errors
    ///
    /// `Error::Broker` if the broker refuses (an unknown stream, say); [`Error::Protocol`] for a result it can't
    /// read; and as any request.
    pub async fn query(&self, spec: &QuerySpec) -> Result<QueryResult, Error> {
        let reply = self.shared.request(|id| query(id, spec)).await?;
        let r = reply.envelope().payload_as_query_result().ok_or_else(|| crate::unexpected("QueryResult", &reply))?;
        result_of(&r)
    }

    /// Every item over the range (and the filters), as a sequence query paged by the broker's cursor
    /// (`next_after`) until nothing remains, then sorted by start time (and seq). Commit order isn't time order: an
    /// import commits old payloads late. `missing` and `tail` are left empty; ask a time query for those.
    ///
    /// # Errors
    ///
    /// As [`Client::query`]; also [`Error::Protocol`] if a truncated page has no continuation cursor.
    pub async fn query_all(&self, spec: &QuerySpec) -> Result<QueryResult, Error> {
        let mut page = spec.clone();
        page.mode = fb::QueryMode::HitOnly;
        page.by_seq = true;
        let mut all: Vec<(Item, u64)> = Vec::new();
        loop {
            let r = self.query(&page).await?;
            all.extend(r.items.into_iter().zip(r.seqs));
            match continuation(page.after_seq, r.next_after, r.truncated)? {
                Some(next) => page.after_seq = next,
                None => break,
            }
        }
        all.sort_by_key(|(it, _)| {
            let h = it.header.header();
            (h.start_ns(), h.seq())
        });
        let (items, seqs) = all.into_iter().unzip();
        Ok(QueryResult {
            items,
            complete: true,
            missing: Vec::new(),
            tail: None,
            next_after: 0,
            truncated: false,
            seqs,
            quoted: false,
        })
    }
}

/// Sequence drains must either advance or end; gaps (`complete=false`) alone don't imply another page.
fn continuation(after: u64, next: u64, truncated: bool) -> Result<Option<u64>, Error> {
    if next == 0 {
        if truncated {
            return Err(Error::Protocol("the broker truncated a sequence query without a continuation cursor".into()));
        }
        return Ok(None);
    }
    if next <= after {
        return Err(Error::Protocol(format!("the broker's query cursor didn't advance ({next} after {after})")));
    }
    Ok(Some(next))
}

#[cfg(test)]
mod tests {
    use super::continuation;
    use crate::Error;

    #[test]
    fn truncated_sequence_result_without_cursor_cannot_complete_a_drain() {
        assert!(matches!(continuation(5, 0, true), Err(Error::Protocol(_))));
    }

    #[test]
    fn sequence_continuations_end_or_advance_without_looping() {
        assert_eq!(continuation(5, 0, false).unwrap(), None);
        assert_eq!(continuation(5, 6, true).unwrap(), Some(6));
        // Scan-budget pages may advance without being marked truncated.
        assert_eq!(continuation(5, 6, false).unwrap(), Some(6));
        assert!(matches!(continuation(5, 5, true), Err(Error::Protocol(_))));
        assert!(matches!(continuation(5, 4, true), Err(Error::Protocol(_))));
    }
}
