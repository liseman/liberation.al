//! Subscriptions: follow streams from now, from a time, or from a durable name's saved position, get each payload as
//! a `StreamEvent` (its header, and a locator when pinned), learn when catch-up is over (`CaughtUp`), and acknowledge
//! what's handled so a durable subscription resumes from there.
//!
//! The broker sends `StreamEvent` and `CaughtUp` with request id 0; the client routes them by subscription id to the
//! `Subscription` they belong to. Queues are bounded. The default policy closes a connection on overflow so a
//! stalled consumer cannot block request replies. Dedicated subscription clients can opt into Backpressure;
//! they must use another client for RPCs. Neither policy acknowledges undelivered events.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

use continuum_core::Uuid;
use continuum_core::schema::continuum as fb;
use flatbuffers::FlatBufferBuilder;
use tokio::sync::mpsc;

use crate::messages::finish;
use crate::worker::{Item, item_of};
use crate::{Client, Error, Message, Shared};

/// How many events wait for a slow subscriber before the client stops reading the socket.
pub const QUEUE: usize = 4096;

/// How a full subscription queue affects its connection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SubscriptionOverflow {
    /// Close promptly instead of blocking numbered replies. Durable consumers reconnect from their last ACK.
    #[default]
    Close,
    /// Preserve bounded backpressure on a dedicated subscription connection. Never issue RPCs on this client:
    /// use a separate client for reads, writes, registration, and checkpoints. Subscription ACKs are one way.
    Backpressure,
}

/// Where a subscription starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// The next commit on a followed stream, nothing earlier.
    Live,
    /// Every payload whose span reaches this time (UTC ns) or starts after it, then live.
    Time(u64),
    /// After the position saved under the durable name (everything, when none is saved yet). Needs `durable_name`.
    Saved,
}

/// What to follow.
#[derive(Clone, Debug)]
pub struct SubscribeSpec {
    /// The streams, by name.
    pub streams: Vec<String>,
    /// Where to start.
    pub start: Start,
    /// For a durable subscription: the name its position is saved under.
    pub durable_name: Option<String>,
    /// Whether events carry a locator (a pinned copy) as well as the header.
    pub pinned: bool,
    /// Only payloads about one of these subjects, when not empty.
    pub subjects: Vec<Uuid>,
    /// Only payloads of this shape, when set.
    pub shape_name: Option<String>,
}

impl SubscribeSpec {
    /// Follows `streams` from `start`, with no filters, not durable, headers only.
    pub fn new(streams: impl IntoIterator<Item = impl Into<String>>, start: Start) -> Self {
        Self {
            streams: streams.into_iter().map(Into::into).collect(),
            start,
            durable_name: None,
            pinned: false,
            subjects: Vec::new(),
            shape_name: None,
        }
    }
}

/// Something a subscription delivered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubEvent {
    /// A payload at a position (positions strictly increase).
    Event {
        /// Its commit sequence.
        position: u64,
        /// Its header, and where its bytes are when the subscription is pinned.
        item: Item,
    },
    /// Catch-up is over: everything up to `position` was sent, and what follows is live.
    CaughtUp {
        /// The last position of the catch-up.
        position: u64,
    },
    /// An event that couldn't be read (its position, when known).
    Undecodable {
        /// Its position.
        position: u64,
        /// What was wrong.
        reason: String,
    },
}

/// Routes an event the broker pushed to its subscription. Returns the message back if it isn't a subscription's.
pub(crate) async fn route(shared: &Shared, message: Message) -> Result<Option<Message>, u64> {
    let env = message.envelope();
    let (id, event) = if let Some(e) = env.payload_as_stream_event() {
        let event = match e.item().ok_or_else(|| "item is missing".to_owned()).and_then(|it| item_of(&it)) {
            Ok(item) => SubEvent::Event { position: e.position(), item },
            Err(reason) => SubEvent::Undecodable { position: e.position(), reason },
        };
        (e.subscription_id(), event)
    } else if let Some(c) = env.payload_as_caught_up() {
        (c.subscription_id(), SubEvent::CaughtUp { position: c.position() })
    } else {
        return Ok(Some(message));
    };
    let route = shared.subscriptions.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(&id).cloned();
    if let Some(Route { tx, ended }) = route {
        let event = match tx.try_send(event) {
            Ok(()) | Err(mpsc::error::TrySendError::Closed(_)) => return Ok(None),
            Err(mpsc::error::TrySendError::Full(event)) => event,
        };
        if shared.subscription_overflow == SubscriptionOverflow::Close {
            // Serialize the overflow decision with local unsubscribe. A cloned stale route must not close an
            // otherwise healthy connection after its subscription was already removed.
            if overflow_is_current(shared, id, &tx) {
                return Err(id);
            }
            return Ok(None);
        }
        let mut ended = ended.subscribe();
        tokio::select! {
            _ = tx.send(event) => {}
            _ = ended.wait_for(|e| *e) => {}
        }
    }
    // An event for a subscription that ended is dropped: it was unsubscribed and the broker hadn't heard yet.
    Ok(None)
}

/// Linearizes a connection-wide overflow decision with local unsubscribe/replacement.
fn overflow_is_current(shared: &Shared, id: u64, tx: &mpsc::Sender<SubEvent>) -> bool {
    let routes = shared.subscriptions.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    routes.get(&id).is_some_and(|current| current.tx.same_channel(tx))
}

/// A live subscription. `next` gives what it delivers; `ack` (durable ones) saves the position; `unsubscribe` ends
/// it. Dropping it ends it too (without waiting for the broker's answer).
#[derive(Debug)]
pub struct Subscription {
    handle: SubscriptionHandle,
    position: u64,
    live: bool,
    rings: Vec<(continuum_core::StreamId, crate::StreamLease)>,
    events: mpsc::Receiver<SubEvent>,
}

/// Acknowledges a subscription's events and ends it, apart from the `Subscription` (so one task can read while
/// another acks or unsubscribes).
#[derive(Clone, Debug)]
pub struct SubscriptionHandle {
    shared: Weak<Shared>,
    id: u64,
}

impl SubscriptionHandle {
    /// Says everything up to `position` is handled: a durable subscription resumes after it. One way.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] if the connection is closed.
    pub fn ack(&self, position: u64) -> Result<(), Error> {
        let shared = self.shared.upgrade().ok_or(Error::Closed)?;
        if shared.closed.load(Ordering::SeqCst) {
            return Err(Error::Closed);
        }
        let mut b = FlatBufferBuilder::new();
        let a = fb::AckEvents::create(&mut b, &fb::AckEventsArgs { subscription_id: self.id, position });
        let bytes = finish(&mut b, 0, fb::ControlPayload::AckEvents, a.as_union_value());
        shared.writes.send((bytes, None)).map_err(|_| Error::Closed)
    }

    /// Ends it without waiting: unroutes it and sends `Unsubscribe` under a request id nobody waits for (the
    /// broker's answer is discarded). Only if it was still routed: an unsubscribe already told the broker.
    fn end_quietly(&self) {
        if let Some(shared) = self.shared.upgrade()
            && forget(&shared, self.id)
        {
            let rid = shared.next_id.fetch_add(1, Ordering::Relaxed);
            let _ = shared.writes.send((unsubscribe(rid, self.id), None));
        }
    }

    /// Ends the subscription and waits for the broker's `Unsubscribed`: a `next` waiting on it returns `None`. A
    /// durable subscription keeps its saved position. Ending one that already ended does nothing.
    ///
    /// # Errors
    ///
    /// As any request.
    pub async fn unsubscribe(&self) -> Result<(), Error> {
        let shared = self.shared.upgrade().ok_or(Error::Closed)?;
        if !forget(&shared, self.id) {
            return Ok(());
        }
        let id = self.id;
        let reply = shared.request(|rid| unsubscribe(rid, id)).await?;
        match reply.envelope().payload_as_unsubscribed() {
            Some(u) if u.subscription_id() == id => Ok(()),
            _ => Err(crate::unexpected("Unsubscribed", &reply)),
        }
    }
}

impl Subscription {
    /// The client's number for it.
    pub fn id(&self) -> u64 {
        self.handle.id
    }

    /// The position it started after (0 for everything).
    pub fn start_position(&self) -> u64 {
        self.position
    }

    /// Whether it was live from the start (nothing to catch up on, so no `CaughtUp` comes).
    pub fn live_at_start(&self) -> bool {
        self.live
    }

    /// The ring of each subscribed stream that has one, as the stream's writer got it: open a reader on it with
    /// [`crate::StreamLease::carved_ring`] to read the records a `RingLocation` names.
    #[must_use]
    pub fn rings(&self) -> &[(continuum_core::StreamId, crate::StreamLease)] {
        &self.rings
    }

    /// A reader on a subscribed stream's ring (`stream`, or the first stream with one), from the next record written
    /// (or the oldest still there), in an arena `client` mapped. `None` when there's no such ring or its arena isn't
    /// mapped.
    #[must_use]
    pub fn ring_reader(
        &self,
        client: &Client,
        stream: Option<continuum_core::StreamId>,
        from: continuum_shm::ReadFrom,
    ) -> Option<continuum_shm::RingReader> {
        let (_, lease) = self.rings.iter().find(|(id, _)| stream.is_none_or(|s| s == *id))?;
        let ring = lease.carved_ring()?;
        client.arena(&lease.arena)?.carved_ring_reader(&ring, from).ok()
    }

    /// The next event, or `None` once it's unsubscribed or the connection ends (or closes). If the broker drops the
    /// subscription itself (a catalog read failing during catch-up), it says so with an `Error` at request id 0,
    /// which arrives from `Client::next_event` naming the subscription in its message; this keeps waiting then.
    pub async fn next(&mut self) -> Option<SubEvent> {
        self.events.recv().await
    }

    /// The next event if one has already arrived, without waiting: `None` when none is waiting (or it ended; `next`
    /// tells the two apart). For draining everything ready after one wait.
    pub fn try_next(&mut self) -> Option<SubEvent> {
        self.events.try_recv().ok()
    }

    /// As `SubscriptionHandle::ack`.
    ///
    /// # Errors
    ///
    /// As `SubscriptionHandle::ack`.
    pub fn ack(&self, position: u64) -> Result<(), Error> {
        self.handle.ack(position)
    }

    /// A handle that acknowledges this subscription's events, or ends it, from elsewhere.
    pub fn handle(&self) -> SubscriptionHandle {
        self.handle.clone()
    }

    /// Ends it and waits for the broker's `Unsubscribed`. A durable subscription keeps its saved position.
    ///
    /// # Errors
    ///
    /// As any request.
    pub async fn unsubscribe(self) -> Result<(), Error> {
        self.handle.unsubscribe().await
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.handle.end_quietly();
    }
}

/// Where a subscription's events go, and the signal that it ended.
#[derive(Clone, Debug)]
pub(crate) struct Route {
    tx: mpsc::Sender<SubEvent>,
    ended: Arc<tokio::sync::watch::Sender<bool>>,
}

/// A subscription between its Subscribe and its Subscribed: ends it on drop unless disarmed.
struct Pending(Option<SubscriptionHandle>);

impl Pending {
    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            h.end_quietly();
        }
    }
}

/// Stops routing a subscription's events (its `next` then returns `None`, and a send waiting on its full queue gives
/// up). Whether it was still routed.
/// Ends every subscription (the connection is going): each `next` returns `None`, and a send parked on a full queue
/// gives way, so the read loop can see the socket close.
pub(crate) fn end_all(shared: &Shared) {
    let routes = std::mem::take(&mut *shared.subscriptions.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
    for r in routes.into_values() {
        r.ended.send_replace(true);
    }
}

fn forget(shared: &Shared, id: u64) -> bool {
    let route = shared.subscriptions.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(&id);
    route.is_some_and(|r| {
        r.ended.send_replace(true);
        true
    })
}

fn unsubscribe(request_id: u64, subscription_id: u64) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let u = fb::Unsubscribe::create(&mut b, &fb::UnsubscribeArgs { subscription_id });
    finish(&mut b, request_id, fb::ControlPayload::Unsubscribe, u.as_union_value())
}

fn subscribe(request_id: u64, subscription_id: u64, spec: &SubscribeSpec) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let streams: Vec<_> = spec.streams.iter().map(|s| b.create_string(s)).collect();
    let streams = b.create_vector(&streams);
    let durable_name = spec.durable_name.as_deref().map(|n| b.create_string(n));
    let subjects = (!spec.subjects.is_empty()).then(|| {
        let ids: Vec<_> = spec.subjects.iter().map(|u| fb::Uuid::new(&u.0)).collect();
        b.create_vector(&ids)
    });
    let shape_name = spec.shape_name.as_deref().map(|n| b.create_string(n));
    let (start, start_ns) = match spec.start {
        Start::Live => (fb::StartAt::Live, 0),
        Start::Time(ns) => (fb::StartAt::Time, ns),
        Start::Saved => (fb::StartAt::Saved, 0),
    };
    let s = fb::Subscribe::create(
        &mut b,
        &fb::SubscribeArgs {
            subscription_id,
            streams: Some(streams),
            start,
            start_ns,
            durable_name,
            pinned: spec.pinned,
            subjects,
            shape_name,
        },
    );
    finish(&mut b, request_id, fb::ControlPayload::Subscribe, s.as_union_value())
}

impl Client {
    /// Subscribes: sends `Subscribe` and waits for `Subscribed`. Events start arriving on the returned
    /// `Subscription` right away (the client routes them by its id, so they never reach `next_event`).
    ///
    /// # Errors
    ///
    /// [`Error::Protocol`] for `Start::Saved` without a durable name (before sending); `Error::Broker` if the broker
    /// refuses (an unknown stream, say); and as any request.
    pub async fn subscribe(&self, spec: &SubscribeSpec) -> Result<Subscription, Error> {
        if spec.start == Start::Saved && spec.durable_name.is_none() {
            return Err(Error::Protocol("a Saved start needs a durable_name".into()));
        }
        let shared = Arc::clone(&self.shared);
        let id = shared.next_subscription.fetch_add(1, Ordering::Relaxed);
        let (tx, events) = mpsc::channel(QUEUE);
        // Registered before sending, so no event can arrive before there's somewhere to put it.
        let route = Route { tx, ended: Arc::new(tokio::sync::watch::channel(false).0) };
        shared.subscriptions.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(id, route);
        // Until it's handed over, the subscription belongs to this call: if the call fails or is dropped (a timeout,
        // a cancelled task), the guard unroutes it and tells the broker, which may have it live already.
        let handle = SubscriptionHandle { shared: Arc::downgrade(&shared), id };
        let guard = Pending(Some(handle.clone()));
        let reply = shared.request(|rid| subscribe(rid, id, spec)).await?;
        let Some(s) = reply.envelope().payload_as_subscribed().filter(|s| s.subscription_id() == id) else {
            return Err(crate::unexpected("Subscribed", &reply));
        };
        let (position, live) = (s.position(), s.live());
        let rings = s
            .rings()
            .map(|v| {
                v.iter()
                    .filter_map(|r| {
                        Some((continuum_core::StreamId(r.stream_id()), crate::StreamLease::from_fb(&r.lease()?)))
                    })
                    .collect()
            })
            .unwrap_or_default();
        guard.disarm();
        Ok(Subscription { handle, position, live, rings, events })
    }
}

#[cfg(test)]
mod tests;

/// The bytes of a length-prefixed ring record (a little-endian u32 length, then that many bytes, zero-padded to the
/// stride), or `None` for an empty or torn one. Ring streams of variable-size records (`face.boxes`) use this layout.
#[must_use]
pub fn length_prefixed(record: &[u8]) -> Option<&[u8]> {
    let n = u32::from_le_bytes(record.get(..4)?.try_into().ok()?) as usize;
    (n > 0).then_some(())?;
    record.get(4..4usize.checked_add(n)?)
}

#[cfg(test)]
mod length_prefixed_tests {
    use super::length_prefixed;

    #[test]
    fn a_record_is_its_length_then_its_bytes_and_anything_else_is_none() {
        assert_eq!(length_prefixed(&[3, 0, 0, 0, b'a', b'b', b'c', 0, 0]), Some(&b"abc"[..]));
        assert_eq!(length_prefixed(&[0; 16]), None, "an empty slot");
        assert_eq!(length_prefixed(&[9, 0, 0, 0, b'a']), None, "a length past the record");
        assert_eq!(length_prefixed(&[1, 0]), None, "shorter than its length");
    }
}
