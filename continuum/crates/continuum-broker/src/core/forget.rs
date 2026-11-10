//! `PurgeNotice` and `PurgeAck`: reaching the copies a purge can't remove itself (docs/CONTINUUM.md, "Forgetting",
//! step 4).
//!
//! The catalog's purge job removes rows and hides payloads; a worker that keeps its own copies (the tier worker's chunk
//! files, the vector worker's rows) says so with `RegisterWorker.keeps_copies`, and a purge is not complete until each
//! such worker has been told what went and has answered. When a purge's last step has run (`Event::PurgeDone`) the
//! task records it as awaiting acknowledgements, sends a `PurgeNotice` (the removed ids in pages of
//! [`ForgetConfig::notice_ids`], and on the first page the spans the steps removed per stream) to every registered
//! keeper, and completes the purge (the catalog's `complete_purge`, through the writer's urgent queue, never on this
//! task) once every keeper it waits for has acknowledged and a grace period has passed. The grace
//! ([`ForgetConfig::keeper_grace`]) is for a keeper that is between runs when the purge finishes, which is the normal
//! case for a purge a previous broker run left pending: a keeper that registers during it is sent the notice and must
//! acknowledge too.
//!
//! The keepers a purge waits for are every name the catalog remembers (`Catalog::keepers`: each name that ever
//! registered with `keeps_copies`, recorded through the writer as it registers), less the names its record says
//! acknowledged (`Catalog::purge_acked`, recorded as each ack arrives, ahead of the completion on the same queue).
//! So neither a broker restart nor a keeper offline past the grace completes a purge with that keeper's copies
//! left: the purge waits, connected keeper or not, and tells the keeper when it registers. A keeper that was retired
//! or renamed would otherwise stall every purge; `ForgetKeeper` drops its name and releases what waited on it.
//!
//! Acknowledgements are owed by worker name, not session, so one connected keeper per name: a second registration
//! under a connected keeper's name is refused. A keeper that disconnects before acknowledging is sent the notice
//! again when it re-registers, and a purge whose keeper never comes back stays pending in the catalog (listed by
//! `pending_purges` at the next start, and in the purge listing as finished but not complete), warned about once
//! past the grace while a keeper it waits for isn't connected (naming it), once more past
//! [`ForgetConfig::ack_timeout`], and sent again to connected keepers at that cadence.

use std::collections::BTreeMap;

use continuum_catalog::{ForgottenHash, PurgeKind, PurgeRecord, PurgeSummary};

use super::*;

/// How purges wait for the workers that keep their own copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForgetConfig {
    /// How long a finished purge waits for keepers to register before it can complete, so one that is restarting
    /// when the purge finishes still hears of it.
    pub keeper_grace: Duration,
    /// How long a notified keeper may take before the wait is logged and the notice sent again.
    pub ack_timeout: Duration,
    /// Payload ids per `PurgeNotice` page: a day's purge names millions of frames, and a frame is 16 MiB at most.
    pub notice_ids: usize,
}

impl Default for ForgetConfig {
    fn default() -> Self {
        Self { keeper_grace: Duration::from_secs(30), ack_timeout: Duration::from_secs(60), notice_ids: 4096 }
    }
}

/// Removed spans per stream, by start, kept disjoint.
type Spans = BTreeMap<StreamId, BTreeMap<u64, u64>>;

/// The most purge notices one keeper has outstanding (sent, unacknowledged, within the ack timeout). Well under the
/// session outbound queue (server.rs `OUTBOUND`, 1024 messages), so a keeper owed a backlog is never closed for it.
pub(super) const MAX_IN_FLIGHT: usize = 256;

/// A purge whose rows are gone, waiting for the keepers' acknowledgements.
struct Awaiting {
    ids: Vec<Uuid>,
    spans: Vec<(StreamId, Interval)>,
    /// When its last step ran.
    finished_ns: u64,
    /// Keepers (by worker name) not yet heard from: told, or known to the catalog and not connected.
    waiting: HashSet<String>,
    /// Keepers that acknowledged, so a re-registration isn't told again.
    acked: HashSet<String>,
    /// When each waiting keeper was last sent the notice.
    sent_ns: HashMap<String, u64>,
    warned: bool,
    /// Warned once that a keeper it waits for isn't connected, past the grace.
    warned_offline: bool,
}

/// A catalog write about keepers, kept until the writer takes it.
#[derive(Clone, Debug)]
enum KeeperWrite {
    Add(String),
    Forget(String),
    Acked(Uuid, String),
}

impl KeeperWrite {
    fn urgent(&self) -> Urgent {
        match self {
            Self::Add(name) => Urgent::AddKeeper(name.clone()),
            Self::Forget(name) => Urgent::ForgetKeeper(name.clone()),
            Self::Acked(id, name) => Urgent::PurgeAcked { id: *id, name: name.clone() },
        }
    }
}

/// What the scheduler task keeps about purges and keepers.
pub(super) struct Forget {
    config: ForgetConfig,
    /// Sessions whose worker keeps copies, by the worker's name.
    keepers: HashMap<SessionId, String>,
    /// Every keeper's name the catalog remembers, connected or not: what a purge waits for.
    known: HashSet<String>,
    /// Catalog writes about keepers the writer's queue was too full to take, retried in order at each tick (a new
    /// keeper's name must reach the catalog before a restart, or a purge finishing while it is offline would
    /// complete without it).
    unsaved: Vec<KeeperWrite>,
    /// Spans the steps of a running purge removed, per stream, until the purge finishes.
    spans: HashMap<Uuid, Spans>,
    awaiting: HashMap<Uuid, Awaiting>,
    completing: HashSet<Uuid>,
}

impl Forget {
    pub(super) fn restart_checkpoint_clean(&self) -> bool {
        self.unsaved.is_empty() && self.spans.is_empty() && self.awaiting.is_empty() && self.completing.is_empty()
    }

    pub(super) fn new(config: ForgetConfig, known_keepers: Vec<String>) -> Self {
        Self {
            config,
            keepers: HashMap::new(),
            known: known_keepers.into_iter().collect(),
            unsaved: Vec::new(),
            spans: HashMap::new(),
            awaiting: HashMap::new(),
            completing: HashSet::new(),
        }
    }
}

/// Merges `span` into `spans` for `stream`, joining it with any it touches (overlapping or adjacent), in time
/// logarithmic in the spans kept, since a purge step names up to its whole budget of rows.
fn add_span(spans: &mut Spans, stream: StreamId, span: Interval) {
    let (mut start, mut end) = (span.start_ns(), span.end_ns());
    let by_start = spans.entry(stream).or_default();
    // The one starting before this span can reach it; everything starting inside it is swallowed.
    if let Some((&s, &e)) = by_start.range(..start).next_back()
        && e >= start
    {
        start = s;
        end = end.max(e);
        by_start.remove(&s);
    }
    let inside: Vec<u64> = by_start.range(start..=end).map(|(&s, _)| s).collect();
    for s in inside {
        if let Some(e) = by_start.remove(&s) {
            end = end.max(e);
        }
    }
    by_start.insert(start, end);
}

/// A purge's spans as the notice lists them.
fn listed(spans: &Spans) -> Vec<(StreamId, Interval)> {
    spans
        .iter()
        .flat_map(|(&stream, by_start)| {
            by_start.iter().filter_map(move |(&s, &e)| Interval::new(s, e).map(|i| (stream, i)))
        })
        .collect()
}

impl Sched {
    /// A purge step ran: remember what it removed, for the notice.
    pub(super) fn purge_stepped(&mut self, id: Uuid, delta: &PurgeDelta) {
        let spans = self.forget.spans.entry(id).or_default();
        for d in &delta.doomed {
            add_span(spans, d.stream, d.span);
        }
        for cut in &delta.cut_frames {
            add_span(spans, cut.stream, cut.slot);
        }
    }

    pub(super) fn purge_failed(&mut self, id: Uuid) {
        self.forget.spans.remove(&id);
    }

    /// A purge's rows are gone: tell the keepers, and wait for every keeper the catalog knows (less `acked`, the ones
    /// its record says answered in an earlier run), connected or not.
    pub(super) fn purge_finished(
        &mut self,
        report: &PurgeReport,
        kind: Option<&PurgeKind>,
        acked: Vec<String>,
        now_ns: u64,
    ) {
        let id = report.purge_id;
        let mut spans = self.forget.spans.remove(&id).unwrap_or_default();
        // A purge re-stepped after a restart has no step deltas to gather spans from; a range purge's record says
        // what it covered: its range on the streams it named, or on every stream the broker knows when it named
        // none. The report's removed rows can't stand in (a purge that only cut frames inside chunks removed no
        // payload), and a span too wide costs a keeper a rewrite, never a copy left behind.
        if spans.is_empty()
            && let Some(PurgeKind::Range { span, streams }) = kind
        {
            let on: Vec<StreamId> = match streams {
                Some(named) => named.clone(),
                None => self.index.streams(),
            };
            for stream in on {
                add_span(&mut spans, stream, *span);
            }
        }
        let spans = listed(&spans);
        let ids: Vec<Uuid> = report.removed.iter().map(|r| r.id).chain(report.removed_frames.iter().copied()).collect();
        let acked: HashSet<String> = acked.into_iter().collect();
        let mut purge = Awaiting {
            ids,
            spans,
            finished_ns: now_ns,
            waiting: self.forget.known.difference(&acked).cloned().collect(),
            acked,
            sent_ns: HashMap::new(),
            warned: false,
            warned_offline: false,
        };
        let keepers: Vec<(SessionId, String)> = self.forget.keepers.iter().map(|(s, n)| (*s, n.clone())).collect();
        for (session, name) in keepers {
            if !purge.acked.contains(&name) {
                self.notify(session, &name, id, &mut purge, now_ns);
            }
        }
        self.forget.awaiting.insert(id, purge);
        self.complete_if_done(id, now_ns);
    }

    /// Sends the purge's notice to one keeper, in pages of `notice_ids` ids (the spans on the first; one page even
    /// when nothing has an id); the keeper acknowledges once, after the last page.
    fn notify(&mut self, session: SessionId, name: &str, id: Uuid, purge: &mut Awaiting, now_ns: u64) {
        purge.waiting.insert(name.to_owned());
        purge.sent_ns.insert(name.to_owned(), now_ns);
        let per_page = self.forget.config.notice_ids.max(1);
        let pages: Vec<&[Uuid]> = if purge.ids.is_empty() { vec![&[]] } else { purge.ids.chunks(per_page).collect() };
        let last = pages.len() - 1;
        for (n, ids) in pages.into_iter().enumerate() {
            let spans = if n == 0 { &purge.spans[..] } else { &[][..] };
            let bytes = protocol::purge_notice(&mut self.builder, id, ids, spans, n < last);
            self.push(session, bytes);
        }
    }

    /// Whether a connected keeper has this worker name.
    pub(super) fn keeper_named(&self, name: &str) -> bool {
        self.forget.keepers.values().any(|n| n == name)
    }

    pub(super) fn keeper_name(&self, session: SessionId) -> Option<String> {
        self.forget.keepers.get(&session).cloned()
    }

    /// A `PurgeAck`.
    pub(super) fn purge_acked(&mut self, session: SessionId, purge: Uuid, now_ns: u64) {
        let Some(name) = self.forget.keepers.get(&session).cloned() else {
            return debug!(purge = %purge, "a PurgeAck from a session whose worker keeps no copies");
        };
        let Some(p) = self.forget.awaiting.get_mut(&purge) else {
            return debug!(purge = %purge, name, "a PurgeAck for a purge that isn't waiting");
        };
        p.waiting.remove(&name);
        p.acked.insert(name.clone());
        // Recorded ahead of the completion on the same queue, so a restart in between tells the rest only.
        self.keeper_write(KeeperWrite::Acked(purge, name));
        self.complete_if_done(purge, now_ns);
    }

    /// Hands a keeper write to the writer, or keeps it for the next tick when the queue is full. Earlier unsaved
    /// writes go first, so the catalog sees them in order.
    fn keeper_write(&mut self, write: KeeperWrite) {
        self.forget.unsaved.push(write);
        self.flush_keeper_writes();
    }

    fn flush_keeper_writes(&mut self) {
        while let Some(write) = self.forget.unsaved.first() {
            match self.urgent.try_send(write.urgent()) {
                Ok(()) => {
                    self.forget.unsaved.remove(0);
                }
                Err(e) => {
                    warn!(error = %e, ?write, "the writer is full; a keeper write waits for the next tick");
                    return;
                }
            }
        }
    }

    /// A `ForgetKeeper`: the name is dropped from what purges wait for, now and in the catalog, and the purges that
    /// waited on it alone complete.
    pub(super) fn forget_keeper(&mut self, session: SessionId, request_id: u64, name: String, now_ns: u64) {
        // A connected keeper under that name would be told of the next purge and re-add the name on its next
        // registration, so the name can only be dropped once the worker is gone.
        if self.keeper_named(&name) {
            let msg = format!("a worker named {name} that keeps copies is connected: stop it before forgetting it");
            return self.reply_error(session, request_id, ErrorCode::Conflict, &msg);
        }
        let known = self.forget.known.remove(&name);
        let mut released = 0u32;
        let ids: Vec<Uuid> = self.forget.awaiting.keys().copied().collect();
        for id in &ids {
            if let Some(p) = self.forget.awaiting.get_mut(id)
                && p.waiting.remove(&name)
            {
                p.sent_ns.remove(&name);
                released += 1;
            }
        }
        for id in ids {
            self.complete_if_done(id, now_ns);
        }
        if known {
            self.keeper_write(KeeperWrite::Forget(name.clone()));
        }
        warn!(name, known, released, "a keeper's name was dropped: purges no longer wait for it");
        let bytes = protocol::keeper_forgotten(&mut self.builder, request_id, &name, released);
        self.push(session, bytes);
    }

    /// Whether `session` registered a worker that keeps copies.
    pub(super) fn is_keeper(&self, session: SessionId) -> bool {
        self.forget.keepers.contains_key(&session)
    }

    /// A worker that keeps copies registered: every purge still waiting is sent to it (unless it acknowledged
    /// already, under the same name, before it restarted). The streams it already follows durably are owed to the
    /// archive from now on.
    pub(super) fn keeper_registered(&mut self, session: SessionId, name: String, now_ns: u64) {
        self.forget.keepers.insert(session, name.clone());
        let followed = self.durable_streams_of(session);
        self.owe_archive(followed);
        if self.forget.known.insert(name.clone()) {
            self.keeper_write(KeeperWrite::Add(name.clone()));
        }
        // At most MAX_IN_FLIGHT notices now: a keeper back from a long outage can be owed thousands, more than its
        // session's outbound queue holds, and sending them all at once got it closed for not reading before it
        // could acknowledge any, so it never caught up (a Mac, 2026-11-02: 1541 owed, the tier worker
        // reconnecting every 12 s). The rest are marked owed and go out at the tick as acknowledgements come in.
        let mut budget = MAX_IN_FLIGHT;
        let ids: Vec<Uuid> = self.forget.awaiting.keys().copied().collect();
        for id in ids {
            let Some(mut purge) = self.forget.awaiting.remove(&id) else { continue };
            if !purge.acked.contains(&name) {
                if budget > 0 {
                    budget -= 1;
                    self.notify(session, &name, id, &mut purge, now_ns);
                } else {
                    purge.waiting.insert(name.clone());
                    purge.sent_ns.remove(&name);
                }
            }
            self.forget.awaiting.insert(id, purge);
        }
    }

    pub(super) fn keeper_gone(&mut self, session: SessionId) {
        self.forget.keepers.remove(&session);
    }

    /// Completes purges whose wait is over; warns about, and re-sends, the ones a keeper is slow to acknowledge.
    pub(super) fn forget_tick(&mut self, now_ns: u64) {
        self.flush_keeper_writes();
        let timeout = ns(self.forget.config.ack_timeout);
        // Room per connected keeper: MAX_IN_FLIGHT less the notices sent it within the ack timeout, still unacked.
        let mut room: HashMap<String, usize> =
            self.forget.keepers.values().map(|n| (n.clone(), MAX_IN_FLIGHT)).collect();
        for p in self.forget.awaiting.values() {
            for (n, sent) in &p.sent_ns {
                if p.waiting.contains(n)
                    && now_ns.saturating_sub(*sent) < timeout
                    && let Some(r) = room.get_mut(n)
                {
                    *r = r.saturating_sub(1);
                }
            }
        }
        let ids: Vec<Uuid> = self.forget.awaiting.keys().copied().collect();
        for id in ids {
            if self.complete_if_done(id, now_ns) {
                continue;
            }
            let Some(mut purge) = self.forget.awaiting.remove(&id) else { continue };
            if purge.waiting.is_empty() {
                self.forget.awaiting.insert(id, purge);
                continue;
            }
            if !purge.warned_offline && now_ns.saturating_sub(purge.finished_ns) >= ns(self.forget.config.keeper_grace)
            {
                let offline: Vec<&String> =
                    purge.waiting.iter().filter(|n| !self.forget.keepers.values().any(|c| c == *n)).collect();
                if !offline.is_empty() {
                    purge.warned_offline = true;
                    warn!(
                        purge = %id,
                        ?offline,
                        "a purge waits for a worker that keeps copies and isn't connected; it is told when it \
                         registers, or ForgetKeeper drops the name"
                    );
                }
            }
            if !purge.warned && now_ns.saturating_sub(purge.finished_ns) >= timeout {
                purge.warned = true;
                let waiting: Vec<&String> = purge.waiting.iter().collect();
                warn!(purge = %id, ?waiting, "a purge is still waiting for workers to remove their copies");
            }
            let due: Vec<(SessionId, String)> = self
                .forget
                .keepers
                .iter()
                .filter(|(_, n)| purge.waiting.contains(*n))
                .filter(|(_, n)| now_ns.saturating_sub(purge.sent_ns.get(*n).copied().unwrap_or(0)) >= timeout)
                .map(|(s, n)| (*s, n.clone()))
                .collect();
            for (session, name) in due {
                let Some(r) = room.get_mut(&name).filter(|r| **r > 0) else { continue };
                *r -= 1;
                self.notify(session, &name, id, &mut purge, now_ns);
            }
            self.forget.awaiting.insert(id, purge);
        }
    }

    /// Completes the purge when its grace has passed and every keeper it waits for has acknowledged. True when it did.
    fn complete_if_done(&mut self, id: Uuid, now_ns: u64) -> bool {
        let grace = ns(self.forget.config.keeper_grace);
        let done = self
            .forget
            .awaiting
            .get(&id)
            .is_some_and(|p| p.waiting.is_empty() && now_ns.saturating_sub(p.finished_ns) >= grace);
        if !done {
            return false;
        }
        if self.forget.completing.contains(&id) {
            return true;
        }
        if let Err(e) = self.urgent.try_send(Urgent::CompletePurge { id, now_ns }) {
            // The writer is full; the next tick tries again.
            warn!(error = %e, purge = %id, "couldn't queue a purge's completion");
            return false;
        }
        self.forget.completing.insert(id);
        true
    }

    /// A queued completion is not complete until the catalog accepts it. Shared physical obligations can outlive
    /// an early keeper ACK; retain the pending purge and retry after cleanup instead of losing it until restart.
    pub(super) fn purge_completed(&mut self, id: Uuid, result: Result<(), CatalogError>) {
        self.forget.completing.remove(&id);
        match result {
            Ok(()) => {
                self.forget.awaiting.remove(&id);
            }
            Err(CatalogError::Conflict(message)) => debug!(purge = %id, %message, "purge awaits physical cleanup"),
            Err(error) => warn!(purge = %id, %error, "couldn't mark a purge complete; retrying"),
        }
    }
}

/// Purge records in one `PurgeList` when the request gave no limit, and the most it may ask for.
const DEFAULT_PURGE_LIST: usize = 100;
const MAX_PURGE_LIST: usize = 1000;

impl Sched {
    /// A `ListPurges`: read on the reader pool, answered straight from the read.
    pub(super) fn list_purges(&mut self, session: SessionId, request_id: u64, after: Option<Uuid>, limit: u32) {
        let limit = if limit == 0 { DEFAULT_PURGE_LIST } else { (limit as usize).min(MAX_PURGE_LIST) };
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            let result = catalog.purges(after, limit).map_err(|e| (error_code(&e), e.to_string()));
            let _ = events.send(Event::PurgesRead { session, request_id, result });
        });
        if let Err(why) = self.read(job) {
            self.reply_error(session, request_id, ErrorCode::Unavailable, why);
        }
    }

    /// A `GetPurge`: read on the reader pool.
    pub(super) fn get_purge(&mut self, session: SessionId, request_id: u64, purge: Uuid) {
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            let result = catalog.purge_record(purge).map_err(|e| (error_code(&e), e.to_string()));
            let _ = events.send(Event::PurgeRecordRead { session, request_id, purge, result });
        });
        if let Err(why) = self.read(job) {
            self.reply_error(session, request_id, ErrorCode::Unavailable, why);
        }
    }

    pub(super) fn purges_read(
        &mut self,
        session: SessionId,
        request_id: u64,
        result: Result<Vec<PurgeSummary>, (ErrorCode, String)>,
    ) {
        match result {
            Ok(purges) => {
                let bytes = protocol::purge_list(&mut self.builder, request_id, &purges);
                self.push(session, bytes);
            }
            Err((code, msg)) => self.reply_error(session, request_id, code, &msg),
        }
    }

    pub(super) fn purge_record_read(
        &mut self,
        session: SessionId,
        request_id: u64,
        purge: Uuid,
        result: Result<Option<PurgeRecord>, (ErrorCode, String)>,
    ) {
        match result {
            Ok(Some(record)) => {
                let bytes = protocol::purge_record(&mut self.builder, request_id, purge, &record);
                self.push(session, bytes);
            }
            Ok(None) => self.reply_error(session, request_id, ErrorCode::NotFound, &format!("no purge {purge}")),
            Err((code, msg)) => self.reply_error(session, request_id, code, &msg),
        }
    }
}

fn ns(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

impl Sched {
    /// A `RememberForgotten`: the hash goes to `FORGOTTEN` through the writer, which answers.
    pub(super) fn remember_forgotten(&mut self, session: SessionId, request_id: u64, hash: ForgottenHash) {
        if let Err(e) = self.urgent.try_send(Urgent::RememberForgotten { session, request_id, hash }) {
            self.refuse_busy(session, request_id, &e);
        }
    }

    pub(super) fn forgotten_remembered(
        &mut self,
        session: SessionId,
        request_id: u64,
        result: Result<(), (ErrorCode, String)>,
    ) {
        match result {
            Ok(()) => {
                let bytes = protocol::forgotten_remembered(&mut self.builder, request_id);
                self.push(session, bytes);
            }
            Err((code, msg)) => self.reply_error(session, request_id, code, &msg),
        }
    }

    pub(super) fn list_forgotten(
        &mut self,
        session: SessionId,
        request_id: u64,
        after: Option<ForgottenHash>,
        limit: u32,
    ) {
        if !(1..=1024).contains(&limit) {
            self.reply_error(session, request_id, ErrorCode::BadRequest, "forgotten page limit must be 1..=1024");
            return;
        }
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            let result = catalog.list_forgotten(after, limit as usize).map_err(|e| (error_code(&e), e.to_string()));
            let _ = events.send(Event::ForgottenListed { session, request_id, result });
        });
        if let Err(why) = self.read(job) {
            self.reply_error(session, request_id, ErrorCode::Unavailable, why);
        }
    }

    pub(super) fn forgotten_listed(
        &mut self,
        session: SessionId,
        request_id: u64,
        result: Result<continuum_catalog::ForgottenPage, (ErrorCode, String)>,
    ) {
        match result {
            Ok(page) => {
                let bytes = protocol::forgotten_list(&mut self.builder, request_id, &page);
                self.push(session, bytes);
            }
            Err((code, msg)) => self.reply_error(session, request_id, code, &msg),
        }
    }

    /// An `IsForgotten`: read on the reader pool.
    pub(super) fn is_forgotten(&mut self, session: SessionId, request_id: u64, hash: ForgottenHash) {
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            let result = catalog.is_forgotten(hash).map_err(|e| (error_code(&e), e.to_string()));
            let _ = events.send(Event::ForgottenRead { session, request_id, result });
        });
        if let Err(why) = self.read(job) {
            self.reply_error(session, request_id, ErrorCode::Unavailable, why);
        }
    }

    pub(super) fn forgotten_read(
        &mut self,
        session: SessionId,
        request_id: u64,
        result: Result<bool, (ErrorCode, String)>,
    ) {
        match result {
            Ok(forgotten) => {
                let bytes = protocol::forgotten_status(&mut self.builder, request_id, forgotten);
                self.push(session, bytes);
            }
            Err((code, msg)) => self.reply_error(session, request_id, code, &msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans_of(adds: &[(u32, u64, u64)]) -> Vec<(u32, u64, u64)> {
        let mut spans = Spans::new();
        for &(s, a, b) in adds {
            add_span(&mut spans, StreamId(s), Interval::new(a, b).unwrap());
        }
        listed(&spans).into_iter().map(|(s, i)| (s.0, i.start_ns(), i.end_ns())).collect()
    }

    #[test]
    fn spans_merge_overlap_and_adjacency_per_stream_in_any_order() {
        assert_eq!(spans_of(&[(1, 10, 20), (1, 20, 30)]), [(1, 10, 30)], "adjacent");
        assert_eq!(spans_of(&[(1, 10, 20), (1, 15, 40)]), [(1, 10, 40)], "overlapping");
        assert_eq!(spans_of(&[(1, 30, 40), (1, 10, 20)]), [(1, 10, 20), (1, 30, 40)], "disjoint, sorted");
        assert_eq!(spans_of(&[(1, 10, 20), (1, 30, 40), (1, 50, 60), (1, 15, 55)]), [(1, 10, 60)], "swallows");
        assert_eq!(spans_of(&[(1, 10, 20), (1, 5, 50)]), [(1, 5, 50)], "a wider one covers");
        assert_eq!(spans_of(&[(1, 10, 20), (2, 10, 20)]), [(1, 10, 20), (2, 10, 20)], "streams apart");
        assert_eq!(spans_of(&[(1, 10, 20), (1, 21, 30)]), [(1, 10, 20), (1, 21, 30)], "a gap stays");
    }
}
