//! Subscriptions: `Subscribe`, `StreamEvent`, `CaughtUp`, `AckEvents`, and `Unsubscribe` (docs/CONTINUUM-BROKER.md).
//!
//! A subscription follows one or more streams by commit sequence, the catalog's global order of everything ever
//! committed (`Indexed::commit_seq`), which is what `StreamEvent.position`, `AckEvents.position`, and a durable
//! subscription's saved cursor all mean. Positions are strictly increasing within a subscription, whatever streams
//! it spans.
//!
//! Three starts. `Live` begins after the newest sequence this task has seen: `Subscribed` says `live` and the first
//! `StreamEvent` is the next commit on a followed stream. `Time` and `Saved` catch up first: the reader pool pages the
//! catalog by commit sequence (one round per [`CATCH_UP_LIMIT`] items a stream, [`CATCH_UP_SCAN`] rows scanned), the
//! task sends what the round found in sequence order, and live events that land meanwhile wait in a buffer. A round
//! that exhausts every followed stream ends the catch-up: the task drains the buffer past the round's head, sends
//! `CaughtUp`, and the subscription is live. Each round reads `head_seq` before its queries, so everything up to that
//! head is complete across streams, a stream still being paged is complete only through the last row its page
//! scanned (a filter can leave a page with no items), and what a round found above that is left for the buffer or
//! the next round (a stream exhausted early could be missing a later commit below another stream's last item).
//!
//! The buffer holds at most [`BUFFER_CAP`] events; past that it is dropped and the catch-up takes one more round,
//! which reads the dropped events back from the catalog. No stall: reads run on the reader pool and cursors are
//! written through the writer's urgent queue. A subscriber that stops reading fills its outbound queue and is
//! closed by [`Sched::push`], and a durable one resumes from its last acknowledged position.
//!
//! Acknowledgements for one durable name collapse to its newest position, and only one write for that name sits in
//! the writer's queue. A full queue keeps the position here. The next tick sends it. A queue slot that cannot be
//! taken does not forget the acknowledgement.
//!
//! A catch-up goes at the subscriber's pace: a round is issued only for as many items as the session's outbound
//! queue has room for, less what its other rounds in flight were promised (`Session::reserved`), and otherwise waits
//! for a later tick. Without that, rounds came back faster than a client read them, and a long backlog (a durable
//! follower restarting behind a busy stream) filled the queue and closed the session every time it reconnected.

use std::collections::{HashMap, VecDeque};

use continuum_catalog::QueryFilter;

use super::*;

/// Items a catch-up round asks the catalog for, per stream, at most.
const CATCH_UP_LIMIT: usize = 256;
/// Room in a session's outbound queue a catch-up round leaves for everything else (replies, other events).
const CATCH_UP_SLACK: usize = 64;
/// The fewest items per stream worth a round; with less room than that, the round waits for the client to read.
const CATCH_UP_MIN: usize = 16;
/// Rows a catch-up round may scan per stream, so a selective filter stays bounded.
const CATCH_UP_SCAN: usize = 4096;
/// Live events a catching-up subscription keeps before it gives up on the buffer and takes another round.
const BUFFER_CAP: usize = 4096;

/// Durable acknowledgement positions waiting on, or already handed to, the writer's queue.
///
/// The queue used to take one message per acknowledgement. A catch-up that acks every event fills that queue, and
/// `try_send` then drops the position. The catalog write still happens once per name per tick. This map is what
/// keeps the newest position when the queue cannot take another message.
#[derive(Default)]
pub(super) struct CursorAcks {
    /// Newest position not yet confirmed in the catalog, per durable name.
    pending: HashMap<String, u64>,
    /// Position currently sitting in the writer's queue, per durable name. One name, one queued write.
    sent: HashMap<String, u64>,
    /// The full-queue warning has already been logged for this stall.
    warned: bool,
}

impl CursorAcks {
    pub(super) fn restart_checkpoint_clean(&self) -> bool {
        self.pending.is_empty() && self.sent.is_empty()
    }

    /// Remember `position` if it is newer than anything still unconfirmed for `name`.
    pub(super) fn note(&mut self, name: &str, position: u64) {
        let slot = self.pending.entry(name.to_owned()).or_insert(0);
        *slot = (*slot).max(position);
    }

    /// The newest position acknowledged under `name` that the catalog may not hold yet: queued or still waiting.
    pub(super) fn newest(&self, name: &str) -> Option<u64> {
        self.pending.get(name).copied().max(self.sent.get(name).copied())
    }

    /// Names with no write queued. The caller marks each one sent only after the queue accepts it.
    pub(super) fn ready(&self) -> Vec<(String, u64)> {
        let mut names: Vec<_> = self
            .pending
            .iter()
            .filter(|(name, _)| !self.sent.contains_key(name.as_str()))
            .map(|(name, position)| (name.clone(), *position))
            .collect();
        names.sort();
        names
    }

    pub(super) fn mark_sent(&mut self, name: String, position: u64) {
        self.sent.insert(name, position);
        self.warned = false;
    }

    /// True the first time a stall should be logged. Later failures stay quiet until a send succeeds.
    pub(super) fn warn_blocked(&mut self) -> bool {
        if self.warned {
            false
        } else {
            self.warned = true;
            true
        }
    }

    /// Positions newer than the write queued for their name, or with none queued: what a sync sends before it is
    /// answered, so everything acknowledged ahead of it is written ahead of it.
    pub(super) fn unqueued(&self) -> Vec<(String, u64)> {
        let mut names: Vec<_> = self
            .pending
            .iter()
            .filter(|(name, position)| self.sent.get(name.as_str()).is_none_or(|sent| sent < *position))
            .map(|(name, position)| (name.clone(), *position))
            .collect();
        names.sort();
        names
    }

    /// The catalog has `position`. Drop it, and drop the queued mark so a newer pending position can be sent,
    /// unless a sync queued a newer write behind this one: its mark stays until that one lands.
    pub(super) fn wrote(&mut self, name: &str, position: u64) {
        if self.sent.get(name).is_some_and(|sent| *sent <= position) {
            self.sent.remove(name);
        }
        if self.pending.get(name).is_some_and(|pending| *pending <= position) {
            self.pending.remove(name);
        }
    }

    /// The write did not land. The position stays pending so a later tick can send it again.
    pub(super) fn failed(&mut self, name: &str) {
        self.sent.remove(name);
    }
}

/// Queues a sync behind every acknowledged position the writer doesn't have yet, even one waiting behind a write
/// already queued for its name, so the sync is answered only once they are in the catalog. One write per
/// name stays the rule for ordinary acks (a catch-up acks every event); a sync is rare. When the queue can't take
/// them all, the sync isn't queued: its reply drops, which the caller sees as busy, and the rest stay pending.
/// `try_send` is the writer's queue (`Sender::try_send`), true when it took the job.
pub(super) fn queue_sync(
    cursors: &mut CursorAcks,
    mut try_send: impl FnMut(Urgent) -> bool,
    reply: oneshot::Sender<()>,
) {
    for (name, position) in cursors.unqueued() {
        if !try_send(Urgent::Cursor { name: name.clone(), position }) {
            return;
        }
        cursors.mark_sent(name, position);
    }
    try_send(Urgent::Sync(reply));
}

/// A live subscription.
pub(super) struct Subscription {
    /// Unique for the broker's life: a session can reuse a subscription id after unsubscribing, and a catch-up page
    /// read for the earlier one must not reach the later.
    token: u64,
    streams: Vec<StreamId>,
    durable: Option<String>,
    pinned: bool,
    subjects: Option<Vec<Uuid>>,
    shape_name: Option<String>,
    /// Payloads from this time on, for a `Time` start (kept through the catch-up).
    from_ns: Option<u64>,
    /// The newest position sent (or started after).
    position: u64,
    /// Outbound room promised to this subscription's catch-up round in flight (released when it comes back).
    reserved: usize,
    state: State,
}

enum State {
    /// Reading history; `round` says whether a read is in flight (else the next tick issues one).
    CatchingUp {
        round: bool,
        buffered: VecDeque<Buffered>,
        lost: bool,
    },
    Live,
}

struct Buffered {
    seq: u64,
    header: HeaderBuf,
    locator: Option<StorageLocator>,
}

/// Answers a catch-up round with an error if its read is dropped before answering (it panicked).
struct Answered {
    events: Option<mpsc::UnboundedSender<Event>>,
    session: SessionId,
    id: u64,
    token: u64,
}

impl Drop for Answered {
    fn drop(&mut self) {
        if let Some(events) = self.events.take() {
            let pages = Err("the read didn't finish".to_owned());
            let _ = events.send(Event::CaughtUpPage { session: self.session, id: self.id, token: self.token, pages });
        }
    }
}

/// What a catch-up round found on one stream.
pub(super) struct CatchUpPage {
    /// Items in commit order: sequence, header, and copies.
    pub items: Vec<(u64, HeaderBuf, Vec<StorageLocator>)>,
    /// The last commit sequence the page scanned, when it stopped short (`SeqPage::next_after`); `None` when the
    /// stream has nothing more past what was looked at.
    pub next_after: Option<u64>,
    /// The catalog's head sequence before the round's queries (the same for every page of a round).
    pub head: u64,
}

impl Subscription {
    fn keeps(&self, header: &HeaderBuf) -> bool {
        if let Some(t) = self.from_ns
            && !header.span().map_or(header.start_ns() >= t, |s| s.end_ns() > t)
        {
            return false;
        }
        if let Some(shape) = &self.shape_name
            && header.shape_name() != shape
        {
            return false;
        }
        if let Some(subjects) = &self.subjects {
            let mine = header.subjects();
            if !subjects.iter().any(|s| mine.contains(s)) {
                return false;
            }
        }
        true
    }
}

impl Sched {
    /// A `Subscribe`: resolve its streams, then start it (after reading its saved cursor, for a `Saved` start).
    pub(super) fn subscribe(&mut self, session: SessionId, request_id: u64, ask: protocol::SubscribeAsk) {
        if self.subs.contains_key(&(session, ask.id)) {
            let msg = format!("subscription {} is open already on this session", ask.id);
            return self.reply_error(session, request_id, ErrorCode::Conflict, &msg);
        }
        for name in &ask.streams {
            if self.index.stream_named(name).is_none() {
                let msg = format!("no stream named {name}");
                return self.reply_error(session, request_id, ErrorCode::NotFound, &msg);
            }
        }
        if ask.start != protocol::SubscribeStart::Saved {
            return self.start_subscription(session, request_id, ask, None);
        }
        let name = ask.durable_name.clone().unwrap_or_default();
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            let position = catalog.cursor(&name).map_err(|e| e.to_string()).and_then(|saved| match saved {
                None => Ok(None),
                Some(bytes) => {
                    let bytes: [u8; 8] = bytes.try_into().map_err(|_| format!("cursor {name} isn't a position"))?;
                    Ok(Some(u64::from_le_bytes(bytes)))
                }
            });
            let _ = events.send(Event::CursorRead { session, request_id, ask, position });
        });
        if let Err(why) = self.read(job) {
            self.reply_error(session, request_id, ErrorCode::Unavailable, why);
        }
    }

    /// The saved cursor came back: start the subscription after it.
    pub(super) fn cursor_read(
        &mut self,
        session: SessionId,
        request_id: u64,
        ask: protocol::SubscribeAsk,
        position: Result<Option<u64>, String>,
    ) {
        match position {
            Ok(saved) => {
                // An acknowledgement still on its way to the catalog counts: the subscriber already handled it.
                let unsaved = ask.durable_name.as_deref().and_then(|name| self.cursors.newest(name));
                let saved = saved.unwrap_or(0).max(unsaved.unwrap_or(0));
                self.start_subscription(session, request_id, ask, Some(saved));
            }
            Err(e) => {
                warn!(error = %e, "reading a durable subscription's position failed");
                self.reply_error(session, request_id, ErrorCode::Internal, &e);
            }
        }
    }

    /// Records the subscription, answers `Subscribed`, and issues the first catch-up round unless it starts live.
    fn start_subscription(
        &mut self,
        session: SessionId,
        request_id: u64,
        ask: protocol::SubscribeAsk,
        saved: Option<u64>,
    ) {
        if !self.sessions.contains_key(&session) {
            return; // gone while the cursor was read
        }
        if self.subs.contains_key(&(session, ask.id)) {
            let msg = format!("subscription {} was opened while its saved position was read", ask.id);
            return self.reply_error(session, request_id, ErrorCode::Conflict, &msg);
        }
        let streams: Vec<StreamId> = ask.streams.iter().filter_map(|n| self.index.stream_named(n)).collect();
        if streams.len() != ask.streams.len() {
            return self.reply_error(session, request_id, ErrorCode::NotFound, "a stream went away");
        }
        let (position, from_ns, live) = match ask.start {
            protocol::SubscribeStart::Live => (self.head_seq, None, true),
            protocol::SubscribeStart::Time(t) => (0, Some(t), false),
            protocol::SubscribeStart::Saved => (saved.unwrap_or(0), None, false),
        };
        let state =
            if live { State::Live } else { State::CatchingUp { round: false, buffered: VecDeque::new(), lost: false } };
        self.sub_tokens += 1;
        let sub = Subscription {
            token: self.sub_tokens,
            streams: streams.clone(),
            durable: ask.durable_name,
            pinned: ask.pinned,
            subjects: ask.subjects,
            shape_name: ask.shape_name,
            from_ns,
            position,
            reserved: 0,
            state,
        };
        let key = (session, ask.id);
        for s in &streams {
            self.subs_by_stream.entry(*s).or_default().push(key);
        }
        if sub.durable.is_some() && self.is_keeper(session) {
            self.owe_archive(streams.iter().copied());
        }
        self.subs.insert(key, sub);
        let rings: Vec<_> = streams
            .iter()
            .filter_map(|s| {
                let (index, ring) = self.tier0.ring_of(*s)?;
                Some((s.0, protocol::RingGrant { arena: self.tier0.summary(), index, ring }))
            })
            .collect();
        let bytes = protocol::subscribed(&mut self.builder, request_id, ask.id, position, live, &rings);
        self.push(session, bytes);
        if !live {
            self.catch_up_round(key);
        }
    }

    /// Sends one catch-up round to the reader pool, sized to the room in the session's outbound queue. When the
    /// pool is busy, or the client hasn't read enough of what it was sent, the next tick tries again.
    fn catch_up_round(&mut self, key: (SessionId, u64)) {
        let Some(sub) = self.subs.get(&key) else { return };
        let State::CatchingUp { round: false, .. } = &sub.state else { return };
        let Some(s) = self.sessions.get(&key.0) else { return };
        let room = s.outbound.capacity().saturating_sub(s.reserved + CATCH_UP_SLACK);
        // Every item of the round may be sent, plus its `CaughtUp`.
        let per = sub.streams.len().max(1);
        let limit = (room.saturating_sub(1) / per).min(CATCH_UP_LIMIT);
        // A subscription to very many streams can never have CATCH_UP_MIN each: it takes what an empty queue holds.
        let floor = CATCH_UP_MIN.min((s.outbound.max_capacity().saturating_sub(CATCH_UP_SLACK + 1) / per).max(1));
        if limit < floor {
            return; // the client is behind: a later tick, once it has read some
        }
        let reserve = limit * sub.streams.len() + 1;
        let streams = sub.streams.clone();
        let after = sub.position;
        let filter = QueryFilter { subjects: sub.subjects.clone(), start_before: None, window: None, ids: None };
        let shape = sub.shape_name.clone();
        let from_ns = sub.from_ns;
        let token = sub.token;
        let (session, id) = key;
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            // A read that panics still answers, or the subscription would wait for its round forever.
            let mut answered = Answered { events: Some(events), session, id, token };
            let read = || -> Result<Vec<CatchUpPage>, String> {
                let head = catalog.head_seq().map_err(|e| e.to_string())?;
                let mut pages = Vec::with_capacity(streams.len());
                for stream in &streams {
                    let page =
                        catalog.query(*stream, after, &filter, limit, CATCH_UP_SCAN).map_err(|e| e.to_string())?;
                    let items = page
                        .items
                        .into_iter()
                        .filter(|it| shape.as_ref().is_none_or(|s| it.header.shape_name() == s))
                        .filter(|it| {
                            from_ns
                                .is_none_or(|t| it.header.span().map_or(it.header.start_ns() >= t, |s| s.end_ns() > t))
                        })
                        .map(|it| (it.commit_seq, it.header, it.locations))
                        .collect();
                    pages.push(CatchUpPage { items, next_after: page.next_after, head });
                }
                Ok(pages)
            };
            let pages = read();
            if let Some(events) = answered.events.take() {
                let _ = events.send(Event::CaughtUpPage { session, id, token, pages });
            }
        });
        match self.read(job) {
            Ok(()) => {
                if let Some(sub) = self.subs.get_mut(&key)
                    && let State::CatchingUp { round, .. } = &mut sub.state
                {
                    *round = true;
                    sub.reserved = reserve;
                    if let Some(s) = self.sessions.get_mut(&key.0) {
                        s.reserved += reserve;
                    }
                }
            }
            Err(why) => debug!(why, "a catch-up round waits for the reader pool"),
        }
    }

    /// A catch-up round came back: send what is complete, and either take another round or go live.
    pub(super) fn caught_up_page(
        &mut self,
        session: SessionId,
        id: u64,
        token: u64,
        pages: Result<Vec<CatchUpPage>, String>,
    ) {
        let key = (session, id);
        let Some(sub) = self.subs.get_mut(&key) else { return };
        // A page read for an earlier subscription under this id (unsubscribed, then reopened while its read was out).
        // No test drives this: it needs a catalog read held in flight across an unsubscribe, which the session tests
        // have no way to do.
        if sub.token != token {
            return;
        }
        self.release(key);
        let Some(sub) = self.subs.get_mut(&key) else { return };
        let pages = match pages {
            Ok(pages) => pages,
            Err(e) => {
                warn!(error = %e, "a subscription's catch-up read failed; dropping the subscription");
                self.drop_subscription(key);
                let msg = format!("subscription {id}: catching up failed: {e}");
                return self.reply_error(session, 0, ErrorCode::Internal, &msg);
            }
        };
        let State::CatchingUp { round, .. } = &mut sub.state else { return };
        *round = false;
        // Complete through the round's head on every stream; a stream still being paged is complete only through
        // where its page stopped.
        let head = pages.first().map_or(0, |p| p.head);
        let all_exhausted = pages.iter().all(|p| p.next_after.is_none());
        // A stream still being paged is complete only through the last row its page scanned (not its last kept
        // item: a filter can leave a page with no items at all).
        let safe = pages.iter().filter_map(|p| p.next_after).fold(head, u64::min);
        let mut items: Vec<(u64, HeaderBuf, Vec<StorageLocator>)> =
            pages.into_iter().flat_map(|p| p.items).filter(|(seq, _, _)| *seq <= safe).collect();
        items.sort_by_key(|(seq, _, _)| *seq);
        let pinned = sub.pinned;
        let mut out = Vec::with_capacity(items.len());
        for (seq, header, locations) in items {
            let locator = if pinned { self.readable(Some(session), &locations) } else { None };
            if pinned && locator.is_none() && !locations.is_empty() {
                continue; // a copy exists but none can be pinned now
            }
            out.push((seq, protocol::WireItem { header, locator, seq }));
        }
        let Some(sub) = self.subs.get_mut(&key) else { return };
        let mut position = sub.position.max(safe);
        for (seq, item) in &out {
            position = position.max(*seq);
            let bytes = protocol::stream_event(&mut self.builder, id, *seq, item);
            self.push(session, bytes);
        }
        let Some(sub) = self.subs.get_mut(&key) else { return }; // the push may have closed the session
        sub.position = position;
        let State::CatchingUp { buffered, lost, .. } = &mut sub.state else { return };
        if !all_exhausted || *lost {
            *lost = false;
            return self.catch_up_round(key);
        }
        let waiting: Vec<Buffered> = buffered.drain(..).filter(|b| b.seq > position).collect();
        sub.state = State::Live;
        for b in waiting {
            self.send_live(key, b.seq, &b.header, b.locator.as_ref());
        }
        let Some(sub) = self.subs.get(&key) else { return };
        let bytes = protocol::caught_up(&mut self.builder, id, sub.position);
        self.push(session, bytes);
    }

    /// A payload landed: every subscription on its stream that keeps it gets it now, or after its catch-up.
    pub(super) fn publish(&mut self, header: &HeaderBuf, locator: Option<&StorageLocator>, seq: u64) {
        self.head_seq = self.head_seq.max(seq);
        let Some(keys) = self.subs_by_stream.get(&header.stream()) else { return };
        for key in keys.clone() {
            let Some(sub) = self.subs.get_mut(&key) else { continue };
            if !sub.keeps(header) {
                continue;
            }
            match &mut sub.state {
                State::CatchingUp { buffered, lost, .. } => {
                    if buffered.len() >= BUFFER_CAP {
                        buffered.clear();
                        *lost = true;
                    } else {
                        buffered.push_back(Buffered { seq, header: header.clone(), locator: locator.cloned() });
                    }
                }
                State::Live => self.send_live(key, seq, header, locator),
            }
        }
    }

    /// One live event, unless the subscription has it already (a catch-up round that included it).
    fn send_live(&mut self, key: (SessionId, u64), seq: u64, header: &HeaderBuf, locator: Option<&StorageLocator>) {
        let Some(sub) = self.subs.get(&key) else { return };
        if seq <= sub.position {
            return;
        }
        let locator = match (sub.pinned, locator) {
            (false, _) | (true, None) => None,
            // A copy exists but can't be pinned now: the event is left out rather than sent without its bytes.
            (true, Some(l)) => match self.readable(Some(key.0), std::slice::from_ref(l)) {
                Some(l) => Some(l),
                None => return,
            },
        };
        let item = protocol::WireItem { header: header.clone(), locator, seq };
        let bytes = protocol::stream_event(&mut self.builder, key.1, seq, &item);
        if let Some(sub) = self.subs.get_mut(&key) {
            sub.position = seq;
        }
        self.push(key.0, bytes);
    }

    /// An `AckEvents`: a durable subscription's position goes to the catalog through the writer.
    pub(super) fn ack_events(&mut self, session: SessionId, id: u64, position: u64) {
        let Some(sub) = self.subs.get(&(session, id)) else {
            return debug!(id, "an AckEvents for a subscription that isn't open");
        };
        let Some(name) = sub.durable.clone() else { return };
        self.cursors.note(&name, position);
        self.flush_cursors();
    }

    /// Queue one catalog write per durable name that is not already queued. A full queue keeps the position.
    pub(super) fn flush_cursors(&mut self) {
        for (name, position) in self.cursors.ready() {
            match self.urgent.try_send(Urgent::Cursor { name: name.clone(), position }) {
                Ok(()) => self.cursors.mark_sent(name, position),
                Err(e) => {
                    if self.cursors.warn_blocked() {
                        warn!(
                            error = %e,
                            "couldn't queue a durable subscription's position; keeping the newest for the next tick"
                        );
                    }
                    break;
                }
            }
        }
    }

    /// The streams `session`'s durable subscriptions follow, for a keeper registering after it subscribed.
    pub(super) fn durable_streams_of(&self, session: SessionId) -> Vec<StreamId> {
        self.subs
            .iter()
            .filter(|((s, _), sub)| *s == session && sub.durable.is_some())
            .flat_map(|(_, sub)| sub.streams.iter().copied())
            .collect()
    }

    /// The writer finished a tick's cursor writes. A newer acknowledgement that arrived meanwhile is sent now.
    /// A failed write waits for the scheduler tick, so a catalog error cannot spin the two threads.
    pub(super) fn cursors_written(&mut self, saved: Vec<(String, u64)>, failed: Vec<String>) {
        for name in failed {
            self.cursors.failed(&name);
        }
        for (name, position) in saved {
            self.cursors.wrote(&name, position);
        }
        self.flush_cursors();
    }

    /// An `Unsubscribe`.
    pub(super) fn unsubscribe(&mut self, session: SessionId, request_id: u64, id: u64) {
        if !self.drop_subscription((session, id)) {
            let msg = format!("subscription {id} isn't open on this session");
            return self.reply_error(session, request_id, ErrorCode::NotFound, &msg);
        }
        let bytes = protocol::unsubscribed(&mut self.builder, request_id, id);
        self.push(session, bytes);
    }

    /// The session closed: its subscriptions go (a durable one keeps its saved position).
    pub(super) fn unsubscribe_all(&mut self, session: SessionId) {
        let keys: Vec<_> = self.subs.keys().filter(|(s, _)| *s == session).copied().collect();
        for key in keys {
            self.drop_subscription(key);
        }
    }

    /// Gives back the outbound room promised to a subscription's catch-up round.
    fn release(&mut self, key: (SessionId, u64)) {
        let Some(sub) = self.subs.get_mut(&key) else { return };
        let reserved = std::mem::take(&mut sub.reserved);
        if let Some(s) = self.sessions.get_mut(&key.0) {
            s.reserved = s.reserved.saturating_sub(reserved);
        }
    }

    fn drop_subscription(&mut self, key: (SessionId, u64)) -> bool {
        self.release(key);
        let Some(sub) = self.subs.remove(&key) else { return false };
        for s in sub.streams {
            if let Some(keys) = self.subs_by_stream.get_mut(&s) {
                keys.retain(|k| *k != key);
                if keys.is_empty() {
                    self.subs_by_stream.remove(&s);
                }
            }
        }
        true
    }

    /// Issues the catch-up rounds a busy reader pool refused.
    pub(super) fn subscribe_tick(&mut self) {
        let waiting: Vec<_> = self
            .subs
            .iter()
            .filter(|(_, s)| matches!(s.state, State::CatchingUp { round: false, .. }))
            .map(|(k, _)| *k)
            .collect();
        for key in waiting {
            self.catch_up_round(key);
        }
        // A full writer queue leaves the newest acknowledgement here. Retry it even when no new ack arrived,
        // so a catch-up that has already gone quiet still reaches the catalog.
        self.flush_cursors();
    }
}

#[cfg(test)]
mod tests {
    use super::{CursorAcks, Urgent, queue_sync};
    use tokio::sync::oneshot;

    #[test]
    fn a_burst_keeps_the_newest_position_and_queues_one_write() {
        let mut acks = CursorAcks::default();
        for position in 1..=5_000 {
            acks.note("see-faces", position);
        }
        acks.note("see-faces", 12);
        let ready = acks.ready();
        assert_eq!(ready, vec![("see-faces".to_owned(), 5_000)]);
        acks.mark_sent(ready[0].0.clone(), ready[0].1);
        acks.note("see-faces", 5_010);
        assert!(acks.ready().is_empty(), "the name already has a write queued");
        acks.wrote("see-faces", 5_000);
        assert_eq!(acks.ready(), vec![("see-faces".to_owned(), 5_010)]);
    }

    #[test]
    fn a_sync_sends_a_position_waiting_behind_a_queued_write_and_it_is_written_once() {
        let mut acks = CursorAcks::default();
        acks.note("tier", 10);
        acks.mark_sent("tier".to_owned(), 10);
        acks.note("tier", 20);
        assert!(acks.ready().is_empty(), "an ordinary flush waits for the queued write");
        assert_eq!(acks.unqueued(), vec![("tier".to_owned(), 20)]);
        acks.mark_sent("tier".to_owned(), 20);
        acks.wrote("tier", 10);
        assert!(acks.ready().is_empty(), "20 is still queued: the older write landing doesn't send it again");
        assert!(acks.unqueued().is_empty());
        acks.wrote("tier", 20);
        assert_eq!(acks.newest("tier"), None);
    }

    #[test]
    fn a_sync_is_refused_rather_than_queued_ahead_of_positions_the_queue_could_not_take() {
        // The writer's queue refuses one job, then frees up: the sync must not slip in behind the refused position.
        let label = |job: &Urgent| match job {
            Urgent::Cursor { name, position } => format!("{name} {position}"),
            Urgent::Sync(_) => "sync".to_owned(),
            _ => "other job".to_owned(),
        };
        let mut queued = Vec::new();
        let mut refuse = 1;
        let mut queue = |job: Urgent| {
            if queued.len() == 1 && refuse > 0 {
                refuse -= 1;
                return false;
            }
            queued.push(label(&job));
            true
        };
        let mut acks = CursorAcks::default();
        acks.note("faces", 5);
        acks.note("tier", 6);
        let (reply, mut answer) = oneshot::channel();
        queue_sync(&mut acks, &mut queue, reply);
        assert!(answer.try_recv().is_err(), "the reply dropped: the caller hears busy, not synced");
        // With room again, the position left over goes first and the sync follows it.
        let (reply, _answer) = oneshot::channel();
        queue_sync(&mut acks, &mut queue, reply);
        assert_eq!(queued, ["faces 5", "tier 6", "sync"]);
    }

    #[test]
    fn a_full_queue_does_not_forget_the_position() {
        let mut acks = CursorAcks::default();
        acks.note("see-faces", 40);
        acks.note("see-bodies", 7);
        let ready = acks.ready();
        assert_eq!(ready.len(), 2);
        assert!(acks.warn_blocked());
        assert!(!acks.warn_blocked(), "one stall logs once");
        assert_eq!(acks.ready(), ready, "nothing was marked sent, so both positions stay due");
        acks.mark_sent(ready[0].0.clone(), ready[0].1);
        assert!(acks.warn_blocked(), "a successful queue ends the stall, so the next one logs again");
        acks.failed(&ready[0].0);
        assert_eq!(acks.ready().len(), 2, "a failed catalog write is queued again");
    }

    #[test]
    fn a_resubscribe_sees_an_acknowledgement_the_catalog_does_not_have_yet() {
        let mut acks = CursorAcks::default();
        assert_eq!(acks.newest("see-faces"), None);
        acks.note("see-faces", 40);
        acks.mark_sent("see-faces".to_owned(), 40);
        assert_eq!(acks.newest("see-faces"), Some(40), "queued, not yet written");
        acks.note("see-faces", 55);
        assert_eq!(acks.newest("see-faces"), Some(55), "waiting behind the queued write");
        acks.wrote("see-faces", 55);
        assert_eq!(acks.newest("see-faces"), None, "the catalog has it now");
    }
}
