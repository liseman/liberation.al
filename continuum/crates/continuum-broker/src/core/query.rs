//! The consumer flow (docs/CONTINUUM-BROKER.md): a `Query` answered from the
//! scheduler's view with a `QueryResult` (what exists) and, when something is missing and the mode asks, a
//! `PromiseQuote`; `CommitPromise` and `CancelPromise`; the `DispatchJob`s the scheduler's plans turn into, sent
//! to the workers that hold the capabilities; `FulfillPromise` as outputs land; `PromiseFailed` when a step fails.
//! The scheduler's decisions are the engine's (`continuum-sched`); this module turns them into messages.
//!
//! Two rules shape it. The no-stall rule: nothing here reads the catalog on the scheduler thread. A `QueryResult`'s
//! headers, a job's inputs, and a fulfilment's outputs are fetched on the reader pool ([`ReadJob`]), which answers
//! with an [`Event`] the way the writer does, and the message is built when the event comes back. The reply rule: a
//! `Query` is answered once, by its `QueryResult`, with the request id; when the engine also quoted, the
//! `PromiseQuote` follows carrying the same request id on the push path (a client that waits for one reply per
//! request hands it to its event handler). `CommitPromise` and `CancelPromise` are answered with `PromiseAccepted`.
//!
//! Tier 0 items are pinned for the reader through [`Sched::pin_for_reading`], one call site for every locator
//! handed out, so the slot arena can take over pinning without this module changing: an item whose slot can't be
//! pinned falls back to its next copy, or is left out.

use super::*;

/// Registered streams with their deletion and forget revisions, as read off the scheduler thread.
pub(super) type StreamListingRow =
    (StreamId, continuum_catalog::StreamInfo, Option<u64>, Option<u64>, Option<Interval>);
pub(super) type StreamListingResult = Result<Vec<StreamListingRow>, (ErrorCode, String)>;

/// A payload as the reader pool found it: its header and every copy, hottest first.
pub(super) struct ReadItem {
    pub header: HeaderBuf,
    pub locations: Vec<StorageLocator>,
    pub seq: u64,
}

/// Parts are issued in engine action order, but independent reader threads can finish them in any order.
/// Bound unresolved parts by the reader queue's capacity so a stuck first read cannot grow the reorder buffer.
#[derive(Default)]
pub(super) struct FulfillmentOrder {
    issued: usize,
    sent: usize,
    final_issued: bool,
    ready: std::collections::BTreeMap<usize, (Result<Vec<ReadItem>, String>, bool)>,
}

/// What the reader pool found for a query: the items, and whether the cut at `max_items` left some out.
pub(super) struct QueryItems {
    pub items: Vec<ReadItem>,
    pub truncated: bool,
}

/// What a `Query` is answered with once its items are read.
pub(super) struct QueryReply {
    pub session: SessionId,
    pub request_id: u64,
    pub complete: bool,
    pub missing: Vec<Interval>,
    pub quote: Option<Quote>,
    /// Where a sequence query continues (0 when it reached the end, or for a time query).
    pub next_after: u64,
    /// Sequence queries can resume when the wire budget cuts the returned prefix.
    pub by_seq: bool,
    /// A ring stream's unsealed records, for a time query that reaches past its newest indexed batch.
    pub tail: Option<RingTailSeqs>,
}

/// A ring's records past its newest indexed batch: `first_seq` to `last_seq`, inclusive.
#[derive(Clone, Copy, Debug)]
pub(super) struct RingTailSeqs {
    pub index: u16,
    pub ring: continuum_shm::CarvedRing,
    pub first_seq: u64,
    pub last_seq: u64,
}

/// Rows a sequence query scans when it gave no `max_scan`.
const DEFAULT_MAX_SCAN: usize = 4096;

/// Streams read per catalog page when listing them.
const STREAM_LIST_PAGE: usize = 1000;

/// The most items one `QueryResult` carries when the query gave no `max_items`.
pub(super) const DEFAULT_MAX_ITEMS: usize = 1024;

/// Preserve the established inline page boundary independently of header/locator overhead.
pub(super) const MAX_RESULT_INLINE_BYTES: usize = 8 * 1024 * 1024;
/// A second reader-pool cap accounts for entire items, leaving room for result metadata.
const MAX_RESULT_ITEM_BYTES: usize = continuum_wire::MAX_FRAME - 1024;

fn inline_len(locations: &[StorageLocator]) -> usize {
    locations.iter().fold(0usize, |sum, locator| {
        sum.saturating_add(match locator {
            StorageLocator::Inline { bytes } => bytes.len(),
            _ => 0,
        })
    })
}

/// An upper bound for the serialized item, including its largest possible fallback locator.
/// 512 covers the fixed tables, vtables, vector offsets and alignment for any locator variant.
fn item_budget(header: &HeaderBuf, locations: &[StorageLocator]) -> usize {
    header.as_bytes().len().saturating_add(locations.iter().map(locator_bytes).max().unwrap_or(0)).saturating_add(512)
}

fn locator_bytes(locator: &StorageLocator) -> usize {
    match locator {
        StorageLocator::Inline { bytes } => bytes.len(),
        StorageLocator::Shm { arena, .. } | StorageLocator::Ring { arena, .. } => arena.len(),
        StorageLocator::Chunk { uri, .. } => uri.len(),
    }
}

/// A capability's name for the worker, from the stream it produces (a hydrate has none).
pub(super) fn capability_name(index: &AvailabilityIndex, cap: &Capability) -> String {
    if cap.role == CapabilityRole::Hydrate { String::new() } else { index.name_of(cap.produces).unwrap_or_default() }
}

impl Sched {
    /// The consumer id a session queries as, minted on its first query.
    fn consumer_of(&mut self, session: SessionId) -> ConsumerId {
        if let Some(&c) = self.consumers.get(&session) {
            return c;
        }
        self.next_consumer += 1;
        let c = ConsumerId(self.next_consumer);
        self.consumers.insert(session, c);
        self.consumer_sessions.insert(c, session);
        c
    }

    /// Pins a Tier 0 locator for `session` to read, under the 5 s read lease, held until the session ends or the pin
    /// expires (see [`Sched::pin_for_reader`]). Every Tier 0 locator a message hands out goes through here; an `Err`
    /// (the slot is stale or reused, or there's no session to hold the pin) means the item falls back to another
    /// copy or is left out.
    pub(super) fn pin_for_reading(
        &mut self,
        session: Option<SessionId>,
        locator: &StorageLocator,
    ) -> Result<(), String> {
        let session = session.ok_or("nobody to hold the pin")?;
        self.pin_for_reader(session, locator)
    }

    /// The copy of a payload a reader gets: the hottest one that can be pinned (Tier 0 through
    /// [`Sched::pin_for_reading`]), else the hottest other copy, else none.
    pub(super) fn readable(
        &mut self,
        session: Option<SessionId>,
        locations: &[StorageLocator],
    ) -> Option<StorageLocator> {
        for l in locations {
            if l.tier() != StorageTier::Tier0Hot || self.pin_for_reading(session, l).is_ok() {
                return Some(l.clone());
            }
        }
        None
    }

    /// The items as sent, each with the copy `session` gets (its Tier 0 copies pinned for it).
    fn wire_items(&mut self, session: Option<SessionId>, items: Vec<ReadItem>) -> Vec<protocol::WireItem> {
        items
            .into_iter()
            .filter_map(|it| {
                let locator = self.readable(session, &it.locations);
                (locator.is_some() || it.locations.is_empty()).then_some(protocol::WireItem {
                    header: it.header,
                    locator,
                    seq: it.seq,
                })
            })
            .collect()
    }

    /// The stream and worker names a quote's plan steps mention, looked up now so the builder can be borrowed after.
    fn plan_names(&self, quote: &Quote) -> (HashMap<StreamId, String>, HashMap<WorkerId, String>) {
        let steps = quote.plan.steps.iter().chain(quote.alternatives.iter().flat_map(|a| a.plan.steps.iter()));
        let mut streams = HashMap::new();
        let mut workers = HashMap::new();
        for s in steps {
            streams.entry(s.stream).or_insert_with(|| self.index.name_of(s.stream).unwrap_or_default());
            if let Some(w) = s.worker {
                workers.entry(w).or_insert_with(|| self.worker_names.get(&w).cloned().unwrap_or_default());
            }
        }
        (streams, workers)
    }

    /// A `ListStreams`: every registered stream from the catalog, on the reader pool, in pages of
    /// [`STREAM_LIST_PAGE`].
    pub(super) fn list_streams(&mut self, session: SessionId, request_id: u64) {
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            let read = || -> StreamListingResult {
                let mut all = Vec::new();
                let mut after = None;
                loop {
                    let page = catalog.streams(after, STREAM_LIST_PAGE).map_err(|e| (error_code(&e), e.to_string()))?;
                    let n = page.len();
                    after = page.last().map(|(id, _)| *id);
                    for (id, info) in page {
                        let revision =
                            catalog.invalidation_revision(id).map_err(|e| (error_code(&e), e.to_string()))?;
                        let forget = catalog.forget_revision(id).map_err(|e| (error_code(&e), e.to_string()))?;
                        let extent = catalog.stream_extent(id).map_err(|e| (error_code(&e), e.to_string()))?;
                        all.push((id, info, revision, forget, extent));
                    }
                    if n < STREAM_LIST_PAGE {
                        return Ok(all);
                    }
                }
            };
            let _ = events.send(Event::StreamsRead { session, request_id, result: read() });
        });
        if let Err(why) = self.read(job) {
            self.reply_error(session, request_id, ErrorCode::Unavailable, why);
        }
    }

    /// The stream listing came back: each stream is live when a producer session holds it now.
    pub(super) fn streams_read(&mut self, session: SessionId, request_id: u64, result: StreamListingResult) {
        match result {
            Ok(streams) => {
                let rows: Vec<_> = streams
                    .into_iter()
                    .map(|(id, info, revision, forget, extent)| {
                        let live = self.producers.contains_key(&id);
                        (id, info, live, revision, forget, extent)
                    })
                    .collect();
                let bytes = protocol::stream_list(&mut self.builder, request_id, &rows);
                self.push(session, bytes);
            }
            Err((code, msg)) => self.reply_error(session, request_id, code, &msg),
        }
    }

    /// Sends a read to the pool, or says why it couldn't.
    pub(super) fn read(&mut self, job: ReadJob) -> Result<(), &'static str> {
        self.reads.try_send(job).map_err(|e| match e {
            TrySendError::Full(_) => "the broker's readers are busy; send it again shortly",
            TrySendError::Disconnected(_) => "the broker's readers have stopped",
        })
    }

    /// A sequence query: the stream's payloads in commit order after a position, as the catalog filters them, read
    /// on the reader pool. Nothing is quoted or made, so the engine isn't asked.
    fn seq_query(
        &mut self,
        session: SessionId,
        request_id: u64,
        stream: StreamId,
        ask: &protocol::QueryAsk,
        seq: protocol::SeqQuery,
    ) {
        // A caller's scan cap holds when it set no item cap: the default item cap shrinks to it, instead of raising
        // the scan to the default 1024. A scan cap below an item cap the caller did set rises to it (the contract:
        // never below max_items).
        let limit = match (ask.max_items, seq.max_scan) {
            (0, 0) => DEFAULT_MAX_ITEMS,
            (0, scan) => DEFAULT_MAX_ITEMS.min(scan as usize),
            (items, _) => items as usize,
        };
        let max_scan = if seq.max_scan == 0 { DEFAULT_MAX_SCAN } else { seq.max_scan as usize }.max(limit);
        let range = ask.range;
        // Half of the range test goes to the catalog (a payload starting at or after the range's end can't overlap
        // it), so a page's limit counts candidates that can pass; the other half is checked on what comes back.
        let mut filter = seq.filter;
        if let Some(r) = range {
            filter.window = Some(r); // the catalog reads the window, not the whole history
            filter.start_before = Some(filter.start_before.map_or(r.end_ns(), |s| s.min(r.end_ns())));
        }
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            let read = catalog.query(stream, seq.after_seq, &filter, limit, max_scan).map_err(|e| e.to_string());
            let (items, next_after) = match read {
                Ok(page) => {
                    let mut items: Vec<ReadItem> = Vec::new();
                    let mut bytes: usize = 0;
                    let mut inline_bytes: usize = 0;
                    let mut next_after = page.next_after;
                    for p in page.items {
                        let keep = range.is_none_or(|r| {
                            p.header.span().is_some_and(|s| s.start_ns() < r.end_ns() && r.start_ns() < s.end_ns())
                        });
                        if !keep {
                            continue;
                        }
                        let len = item_budget(&p.header, &p.locations);
                        let inline = inline_len(&p.locations);
                        if let Some(last) = items.last().filter(|_| {
                            bytes.saturating_add(len) > MAX_RESULT_ITEM_BYTES
                                || inline_bytes.saturating_add(inline) > MAX_RESULT_INLINE_BYTES
                        }) {
                            // The page is full of bytes: the next one starts right after the last item kept.
                            next_after = Some(last.seq);
                            break;
                        }
                        bytes = bytes.saturating_add(len);
                        inline_bytes = inline_bytes.saturating_add(inline);
                        items.push(ReadItem { header: p.header, locations: p.locations, seq: p.commit_seq });
                    }
                    (Ok(QueryItems { items, truncated: false }), next_after)
                }
                Err(e) => (Err(e), None),
            };
            let reply = QueryReply {
                session,
                request_id,
                complete: next_after.is_none(),
                missing: Vec::new(),
                quote: None,
                next_after: next_after.unwrap_or(0),
                by_seq: true,
                tail: None,
            };
            let _ = events.send(Event::QueryRead { reply, items });
        });
        if let Err(why) = self.read(job) {
            self.reply_error(session, request_id, ErrorCode::Unavailable, why);
        }
    }

    /// A `Query`: ask the engine, then fetch what exists on the reader pool and answer from the event.
    pub(super) fn query(&mut self, session: SessionId, request_id: u64, ask: protocol::QueryAsk) {
        let Some(stream) = self.index.stream_named(&ask.stream) else {
            return self.reply_error(
                session,
                request_id,
                ErrorCode::NotFound,
                &format!("no stream named {}", ask.stream),
            );
        };
        if let Some(seq) = ask.by_seq.clone() {
            return self.seq_query(session, request_id, stream, &ask, seq);
        }
        let Some(range) = ask.range else {
            return self.reply_error(session, request_id, ErrorCode::BadRequest, "Query needs a range");
        };
        let consumer = self.consumer_of(session);
        let req = QueryReq {
            consumer,
            stream,
            range,
            budget_us: ask.budget_us,
            strict: false,
            band: ask.band,
            mode: ask.mode,
        };
        let live = self.live();
        let out = self.engine.query(&IndexView::new(&self.index, &live), req, now_ns());
        let actions = out.actions;
        let (covered, missing, quote) = match out.decision {
            QueryDecision::Hit { covered, missing } => (covered, missing, None),
            QueryDecision::Quoted { covered, quote } | QueryDecision::Committed { covered, quote } => {
                self.promise_owners.insert(quote.promise, session);
                let missing = quote.missing.clone();
                (covered, missing, Some(quote))
            }
            QueryDecision::Refused(r) => {
                let code = match r {
                    Refusal::UnknownStream(_) | Refusal::NotCaptured(_) => ErrorCode::NotFound,
                    Refusal::StreamNotLive(_) | Refusal::NoProducer(_) | Refusal::NoHydrator => ErrorCode::Unavailable,
                    Refusal::Catalog(_) => ErrorCode::Internal,
                };
                return self.reply_error(session, request_id, code, &r.to_string());
            }
        };
        let promise = quote.as_ref().map(|q| q.promise);
        let tail = self.ring_tail(stream, range);
        let complete = missing.is_empty() && tail.is_none();
        let reply = QueryReply { session, request_id, complete, missing, quote, next_after: 0, by_seq: false, tail };
        let limit = if ask.max_items == 0 { DEFAULT_MAX_ITEMS } else { ask.max_items as usize };
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            let items = read_covered(catalog, stream, &covered, limit, MAX_RESULT_INLINE_BYTES);
            let _ = events.send(Event::QueryRead { reply, items });
        });
        // The query's read is queued before anything the query set off, so under CommitWithinBudget the answer is
        // at least queued ahead of the results (parts may still land first: the doc says so).
        match self.read(job) {
            Ok(()) => self.run_actions(actions),
            Err(why) => {
                // A quote nobody will see is cancelled, so nothing runs for it.
                if let Some(p) = promise {
                    if let Ok(actions) = self.engine.cancel_promise(p, now_ns()) {
                        self.run_actions(actions);
                    }
                    self.promise_ended(p);
                }
                self.reply_error(session, request_id, ErrorCode::Unavailable, why);
            }
        }
    }

    /// Time passed in the engine: quotes nobody committed have expired there without an action, so the maps that
    /// follow promises are trimmed to the ones still alive.
    pub(super) fn tick(&mut self, now_ns: u64) {
        let actions = self.engine.tick(now_ns);
        self.run_actions(actions);
        let alive = |p: &PromiseId, engine: &Engine| {
            matches!(
                engine.promise_state(*p),
                Some(PromiseState::Quoted | PromiseState::Armed | PromiseState::Dispatching)
            )
        };
        // The engine can finish before its final catalog read. Delivery still owns that promise until
        // all earlier parts have gone out, including across scheduler ticks and cancellation requests.
        self.promise_owners.retain(|p, _| self.fulfillments.contains_key(p) || alive(p, &self.engine));
        self.delivered.retain(|p, _| self.fulfillments.contains_key(p) || alive(p, &self.engine));
    }

    /// The reader pool answered a query: pin what's in Tier 0 and send the result, then the quote if there is one.
    pub(super) fn query_read(&mut self, reply: QueryReply, items: Result<QueryItems, String>) {
        let found = match items {
            Ok(found) => found,
            Err(e) => {
                warn!(error = %e, "reading a query's items failed");
                return self.query_size_error(&reply, ErrorCode::Internal, &e);
            }
        };
        let summary = self.tier0.summary().clone();
        let tail = reply.tail.map(|t| protocol::Tail {
            arena: &summary,
            index: t.index,
            ring: t.ring,
            first_seq: t.first_seq,
            last_seq: t.last_seq,
        });
        let paging = protocol::ResultPaging {
            complete: reply.complete && !found.truncated,
            next_after: reply.next_after,
            truncated: found.truncated || reply.next_after != 0,
            quoted: reply.quote.is_some(),
        };
        // Bound before pinning: every possible fallback locator is covered. Metadata is charged as well,
        // including the missing-interval vector and ring arena name. Never allocate an oversized frame.
        let metadata_raw =
            reply.missing.len().saturating_mul(16).saturating_add(tail.as_ref().map_or(0, |t| t.arena.name.len()));
        let metadata = metadata_raw.saturating_add(1024);
        let metadata_fits = metadata <= continuum_wire::MAX_FRAME
            || (metadata_raw <= continuum_wire::MAX_FRAME
                && protocol::query_result(
                    &mut self.builder,
                    reply.request_id,
                    &[],
                    paging,
                    &reply.missing,
                    tail.as_ref(),
                )
                .len()
                    <= continuum_wire::MAX_FRAME);
        let mut used = metadata;
        let mut count = 0;
        for item in &found.items {
            let next = used.saturating_add(item_budget(&item.header, &item.locations));
            if next > continuum_wire::MAX_FRAME {
                break;
            }
            used = next;
            count += 1;
        }
        // Conservative overhead may exclude a valid single item near the limit. Try its actual encodings,
        // before taking pins, with nondefault paging fields so their presence is accounted for.
        if count == 0 && !found.items.is_empty() && metadata_fits {
            let first = &found.items[0];
            let mut fits = true;
            let candidates: Vec<_> =
                if first.locations.is_empty() { vec![None] } else { first.locations.iter().map(Some).collect() };
            for locator in candidates {
                if first
                    .header
                    .as_bytes()
                    .len()
                    .saturating_add(locator.map_or(0, locator_bytes))
                    .saturating_add(metadata_raw)
                    > continuum_wire::MAX_FRAME
                {
                    fits = false;
                    break;
                }
                let one =
                    [protocol::WireItem { header: first.header.clone(), locator: locator.cloned(), seq: first.seq }];
                let worst =
                    protocol::ResultPaging { complete: true, next_after: u64::MAX, truncated: true, quoted: true };
                if protocol::query_result(
                    &mut self.builder,
                    reply.request_id,
                    &one,
                    worst,
                    &reply.missing,
                    tail.as_ref(),
                )
                .len()
                    > continuum_wire::MAX_FRAME
                {
                    fits = false;
                    break;
                }
            }
            if fits {
                count = 1;
            }
        }
        if !metadata_fits || (count == 0 && !found.items.is_empty()) {
            return self.query_size_error(
                &reply,
                ErrorCode::BadRequest,
                "query result metadata or first item exceeds the wire frame size limit",
            );
        }
        let cut = count < found.items.len();
        let next_after = if cut && reply.by_seq { found.items[count - 1].seq } else { reply.next_after };
        let paging = protocol::ResultPaging {
            complete: paging.complete && !cut,
            next_after,
            truncated: paging.truncated || cut,
            ..paging
        };
        let pins_before = self.reader_pins.get(&reply.session).map_or(0, Vec::len);
        let items = self.wire_items(Some(reply.session), found.items.into_iter().take(count).collect());
        // Unreadable Tier 0 rows may have been omitted. Resume after the last row actually handed out;
        // rescanning an unreadable gap is safe, advancing past an excluded matching row is not.
        let paging = if cut && reply.by_seq {
            protocol::ResultPaging { next_after: items.last().map_or(next_after, |item| item.seq), ..paging }
        } else {
            paging
        };
        let bytes =
            protocol::query_result(&mut self.builder, reply.request_id, &items, paging, &reply.missing, tail.as_ref());
        // The conservative preflight should always suffice; keep this exact guard at the send boundary.
        if bytes.len() > continuum_wire::MAX_FRAME {
            if let Some(pins) = self.reader_pins.get_mut(&reply.session) {
                pins.truncate(pins_before);
            }
            return self.query_size_error(
                &reply,
                ErrorCode::BadRequest,
                "serialized query result exceeds the wire frame size limit",
            );
        }
        self.push(reply.session, bytes);
        if let Some(quote) = reply.quote {
            let (streams, workers) = self.plan_names(&quote);
            let stream = |s: StreamId| streams.get(&s).cloned().unwrap_or_default();
            let worker = |w: WorkerId| workers.get(&w).cloned().unwrap_or_default();
            let names = protocol::PlanNames { stream: &stream, worker: &worker };
            let bytes = protocol::promise_quote(&mut self.builder, reply.request_id, &quote, &names);
            self.push(reply.session, bytes);
        }
    }

    /// An unseen quote must not outlive a query that failed before delivering its result.
    fn query_size_error(&mut self, reply: &QueryReply, code: ErrorCode, message: &str) {
        if let Some(quote) = &reply.quote {
            if let Ok(actions) = self.engine.cancel_promise(quote.promise, now_ns()) {
                self.run_actions(actions);
            }
            self.promise_ended(quote.promise);
        }
        self.reply_error(reply.session, reply.request_id, code, message);
    }

    /// A `CommitPromise`: run the plan (or the chosen alternative) and say where the promise stands.
    pub(super) fn commit_promise(
        &mut self,
        session: SessionId,
        request_id: u64,
        promise: PromiseId,
        option: u32,
        band: PriorityBand,
    ) {
        match self.promise_owners.get(&promise) {
            None => return self.reply_error(session, request_id, ErrorCode::NotFound, "no such promise"),
            Some(&owner) if owner != session => {
                return self.reply_error(session, request_id, ErrorCode::NotAllowed, "not this session's promise");
            }
            Some(_) => {}
        }
        let live = self.live();
        let alternative = option.checked_sub(1);
        match self.engine.commit_promise(&IndexView::new(&self.index, &live), promise, alternative, band, now_ns()) {
            Ok(actions) => {
                // The answer goes first; what the commit set off (a dispatch, or a failure when the only producer
                // is gone) follows it, so the consumer never hears of a promise's end before its acceptance.
                let life = match self.engine.promise_state(promise) {
                    Some(PromiseState::Armed) => PromiseLife::Armed,
                    Some(PromiseState::Dispatching) => PromiseLife::Dispatching,
                    _ => PromiseLife::Failed,
                };
                let bytes = protocol::promise_accepted(&mut self.builder, request_id, promise, life);
                self.push(session, bytes);
                self.run_actions(actions);
            }
            Err(e) => {
                // Reject this request without ending an earlier accepted drain whose catalog reads remain pending.
                if !self.promise_alive(promise) && !self.fulfillments.contains_key(&promise) {
                    self.promise_ended(promise);
                }
                self.reply_error(session, request_id, commit_error_code(&e), &e.to_string());
            }
        }
    }

    /// A `CancelPromise`.
    pub(super) fn cancel_promise(&mut self, session: SessionId, request_id: u64, promise: PromiseId) {
        match self.promise_owners.get(&promise) {
            None => return self.reply_error(session, request_id, ErrorCode::NotFound, "no such promise"),
            Some(&owner) if owner != session => {
                return self.reply_error(session, request_id, ErrorCode::NotAllowed, "not this session's promise");
            }
            Some(_) => {}
        }
        // Computation may be complete while ordered parts are still being read. Those parts remain cancellable.
        if self.fulfillments.contains_key(&promise) && !self.promise_alive(promise) {
            self.promise_ended(promise);
            let bytes = protocol::promise_accepted(&mut self.builder, request_id, promise, PromiseLife::Cancelled);
            self.push(session, bytes);
            return;
        }
        match self.engine.cancel_promise(promise, now_ns()) {
            Ok(actions) => {
                self.run_actions(actions);
                self.promise_ended(promise);
                let bytes = protocol::promise_accepted(&mut self.builder, request_id, promise, PromiseLife::Cancelled);
                self.push(session, bytes);
            }
            Err(e) => {
                if !self.promise_alive(promise) {
                    self.promise_ended(promise);
                }
                self.reply_error(session, request_id, ErrorCode::NotFound, &e.to_string());
            }
        }
    }

    fn promise_alive(&self, promise: PromiseId) -> bool {
        matches!(
            self.engine.promise_state(promise),
            Some(PromiseState::Quoted | PromiseState::Armed | PromiseState::Dispatching)
        )
    }

    /// A promise is over (fulfilled, failed, cancelled, or its consumer left): nothing more is tracked for it.
    fn promise_ended(&mut self, promise: PromiseId) {
        self.promise_owners.remove(&promise);
        self.delivered.remove(&promise);
        self.fulfillments.remove(&promise);
    }

    /// A worker's `JobRejected`: the engine replans or fails the promises it served.
    pub(super) fn job_rejected(&mut self, session: SessionId, job: JobId, reason: &str) {
        // Only the worker the job was dispatched to may reject it.
        let Some(&worker) = self.workers.get(&session) else {
            debug!(%session, "JobRejected from a session that registered no worker");
            return;
        };
        if self.jobs.get(&job).map(|&(w, _)| w) != Some(worker) {
            debug!(%session, job = job.0, "JobRejected for a job not dispatched to this worker");
            return;
        }
        let live = self.live();
        match self.engine.job_rejected(&IndexView::new(&self.index, &live), job, reason, now_ns()) {
            Ok(actions) => self.run_actions(actions),
            Err(e) => debug!(error = %e, "JobRejected for a job the engine doesn't know"),
        }
    }

    /// A consumer's session ended: its promises end too.
    pub(super) fn consumer_gone(&mut self, session: SessionId) {
        if let Some(consumer) = self.consumers.remove(&session) {
            self.consumer_sessions.remove(&consumer);
            let owned: Vec<_> = self.promise_owners.iter().filter_map(|(p, s)| (*s == session).then_some(*p)).collect();
            for promise in owned {
                self.promise_ended(promise);
            }
            let actions = self.engine.consumer_gone(consumer, now_ns());
            self.run_actions(actions);
        }
    }

    /// An `Action::Dispatch`: resolve the job's inputs on the reader pool, then send the `DispatchJob`.
    pub(super) fn dispatch(&mut self, d: Dispatch) {
        if !self.worker_sessions.contains_key(&d.worker) {
            debug!(worker = d.worker.0, "a dispatch to a worker that is gone");
            return self.staging_failed(d.output.lease, "the worker is gone");
        }
        // The worker may commit on this lease: it's its own from here.
        let lease = d.output.lease;
        self.leases.insert(lease, (d.worker, d.output.stream));
        self.staging.insert(lease);
        let events = self.events.clone();
        let inputs = d.inputs.clone();
        let job: ReadJob = Box::new(move |catalog| {
            let mut items = Ok(Vec::new());
            for input in &inputs {
                match read_covered(
                    catalog,
                    input.stream,
                    &[Covered { interval: input.range, tier: StorageTier::Tier0Hot }],
                    usize::MAX,
                    // A job gets every input or fails; it's never cut short (a query result is, and says so).
                    usize::MAX,
                ) {
                    Ok(found) => {
                        if let Ok(all) = &mut items {
                            all.extend(found.items);
                        }
                    }
                    Err(e) => {
                        items = Err(e);
                        break;
                    }
                }
            }
            let _ = events.send(Event::Staged { dispatch: d, inputs: items });
        });
        if let Err(why) = self.read(job) {
            self.staging_failed(lease, why);
        }
    }

    /// The inputs of a dispatch were read: wire their locators to the worker's session and send the job. The session's
    /// reader pins ([`crate::tier0::READ_PIN`]) cover the slots for a few seconds; nothing is held for the lease.
    pub(super) fn staged(&mut self, d: Dispatch, inputs: Result<Vec<ReadItem>, String>) {
        self.staging.remove(&d.output.lease);
        // The lease may have ended while the inputs were read (the promise was cancelled, the lease preempted or
        // expired): the engine already said so with LeaseEnded, and the worker has nothing to run.
        if !self.leases.contains_key(&d.output.lease) {
            debug!(lease = d.output.lease.0, "a job's inputs were read after its lease ended; not dispatched");
            return;
        }
        let Some(&session) = self.worker_sessions.get(&d.worker) else {
            return self.staging_failed(d.output.lease, "the worker is gone");
        };
        let inputs = match inputs {
            Ok(items) => self.wire_items(Some(session), items),
            Err(e) => return self.staging_failed(d.output.lease, &e),
        };
        self.jobs.insert(d.job, (d.worker, d.output.lease));
        let name = self.cap_names.get(&d.capability).cloned().unwrap_or_default();
        let bytes = protocol::dispatch_job(&mut self.builder, &d, &inputs, &name);
        self.push(session, bytes);
    }

    /// The broker couldn't stage a job's inputs: the engine corrects its accounting and replans or fails.
    fn staging_failed(&mut self, lease: LeaseId, why: &str) {
        self.staging.remove(&lease);
        warn!(lease = lease.0, why, "couldn't stage a job's inputs");
        let live = self.live();
        match self.engine.claim_failed(&IndexView::new(&self.index, &live), lease, now_ns()) {
            Ok(actions) => self.run_actions(actions),
            Err(e) => debug!(error = %e, "claim_failed for a lease the engine doesn't know"),
        }
    }

    /// An `Action::Fulfill`: fetch outputs in parallel, then deliver parts in engine action order.
    pub(super) fn fulfill(
        &mut self,
        promise: PromiseId,
        consumer: ConsumerId,
        outputs: Vec<OutputRef>,
        final_part: bool,
    ) {
        // Cancellation/failure may have ended a promise while later engine actions were already queued.
        if !self.promise_owners.contains_key(&promise) {
            return;
        }
        let order = self.fulfillments.entry(promise).or_default();
        if order.final_issued {
            return;
        }
        if order.issued - order.sent >= READS {
            return self.promise_failed(promise, consumer, "too many fulfillment parts await ordered delivery");
        }
        let ordinal = order.issued;
        let Some(next) = order.issued.checked_add(1) else {
            return self.promise_failed(promise, consumer, "fulfillment part counter exhausted");
        };
        order.issued = next;
        order.final_issued = final_part;
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            // A panicking catalog read must also end the promise, rather than leave its earlier ordinal missing.
            let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut items = Vec::with_capacity(outputs.len());
                for o in &outputs {
                    match catalog.get(o.id) {
                        Ok(Some(Resolved::Payload(p))) => {
                            items.push(ReadItem { header: p.header, locations: p.locations, seq: p.commit_seq });
                        }
                        Ok(_) => {}
                        Err(e) => return Err(e.to_string()),
                    }
                }
                Ok(items)
            }));
            let items = read.unwrap_or_else(|_| {
                tracing::error!("reading a promise fulfillment panicked");
                Err("reading a promise fulfillment panicked".into())
            });
            let _ = events.send(Event::Fulfilled { consumer, promise, items, final_part, ordinal });
        });
        if let Err(why) = self.read(job) {
            self.promise_failed(promise, consumer, why);
        }
    }

    /// A completed read can go out only after every earlier part, including the terminal part or read error.
    pub(super) fn fulfilled(
        &mut self,
        consumer: ConsumerId,
        promise: PromiseId,
        items: Result<Vec<ReadItem>, String>,
        final_part: bool,
        ordinal: usize,
    ) {
        let Some(order) = self.fulfillments.get_mut(&promise) else {
            return; // a cancelled, failed, disconnected, or already delivered promise cannot revive
        };
        order.ready.insert(ordinal, (items, final_part));
        loop {
            let next = self.fulfillments.get_mut(&promise).and_then(|order| {
                let ready = order.ready.remove(&order.sent)?;
                order.sent += 1;
                Some(ready)
            });
            let Some((items, final_part)) = next else { break };
            let session = self.consumer_sessions.get(&consumer).copied();
            let items = match items {
                Ok(items) => self.wire_items(session, items),
                Err(e) => return self.promise_failed(promise, consumer, &e),
            };
            let n = u32::try_from(items.len()).unwrap_or(u32::MAX);
            *self.delivered.entry(promise).or_insert(0) += n;
            if final_part {
                self.promise_ended(promise);
            }
            if let Some(session) = session {
                let bytes = protocol::fulfill_promise(&mut self.builder, promise, &items, final_part);
                self.push(session, bytes);
            }
            if final_part {
                break;
            }
        }
    }

    /// An `Action::FailPromise`, or a fulfilment the broker couldn't read.
    pub(super) fn promise_failed(&mut self, promise: PromiseId, consumer: ConsumerId, reason: &str) {
        if !self.promise_owners.contains_key(&promise) {
            return;
        }
        let delivered = self.delivered.get(&promise).copied().unwrap_or(0);
        self.promise_ended(promise);
        // A delivery error can precede the engine's own completion. Release its armed capture and job leases;
        // ownership is already gone, so cancellation actions cannot send a second terminal response.
        if self.promise_alive(promise)
            && let Ok(actions) = self.engine.cancel_promise(promise, now_ns())
        {
            self.run_actions(actions);
        }
        if let Some(&session) = self.consumer_sessions.get(&consumer) {
            let bytes = protocol::promise_failed(&mut self.builder, promise, reason, delivered);
            self.push(session, bytes);
        }
    }
}

fn commit_error_code(e: &continuum_sched::Error) -> ErrorCode {
    match e {
        continuum_sched::Error::QuoteExpired(_) => ErrorCode::Conflict,
        _ => ErrorCode::NotFound,
    }
}

/// The payloads of `stream` in the covered intervals, at most `limit` of them and `max_bytes` of inline bytes (at
/// least one item either way), each with every copy. Runs on the reader pool.
fn read_covered(
    catalog: &dyn Catalog,
    stream: StreamId,
    covered: &[Covered],
    limit: usize,
    max_bytes: usize,
) -> Result<QueryItems, String> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let mut bytes: usize = 0;
    let mut inline_bytes: usize = 0;
    for c in covered {
        let mut after = None;
        loop {
            if items.len() >= limit {
                return Ok(QueryItems { items, truncated: true });
            }
            let page = catalog.range(stream, c.interval, after, limit - items.len()).map_err(|e| e.to_string())?;
            for p in page.items {
                if !seen.insert(p.header.id()) {
                    continue;
                }
                let len = item_budget(&p.header, &p.locations);
                let inline = inline_len(&p.locations);
                if !items.is_empty()
                    && (bytes.saturating_add(len) > MAX_RESULT_ITEM_BYTES
                        || inline_bytes.saturating_add(inline) > max_bytes)
                {
                    return Ok(QueryItems { items, truncated: true });
                }
                bytes = bytes.saturating_add(len);
                inline_bytes = inline_bytes.saturating_add(inline);
                items.push(ReadItem { header: p.header, locations: p.locations, seq: p.commit_seq });
            }
            match page.next {
                Some(next) => after = Some(next),
                None => break,
            }
        }
    }
    Ok(QueryItems { items, truncated: false })
}
