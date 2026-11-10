//! The consumer's promises (docs/CONTINUUM.md, "Consumer flow: query, quote, and commit"): a query in a quoting
//! mode (`QuoteIfMissing`) is answered by its `QueryResult` and, when something missing can be made, a
//! `PromiseQuote` for it; [`Client::query_quoted`] returns both. [`Client::commit_promise`] runs the plan (or a
//! chosen alternative), and [`Client::cancel_promise`] calls it off; both are answered with where the promise stands
//! ([`PromiseAccepted`]), and a commit hands back a [`Promise`]: the results come later, pushed, each part of the
//! outputs (`FulfillPromise`, the last with `final_part`) or a `PromiseFailed`, routed to the promise by its id, so
//! [`Promise::fulfilled`] gathers them and they never mix into `Client::next_event`. A promise nobody holds (one a
//! `CommitWithinBudget` query committed, until [`Client::promise`] is asked for it) reaches `next_event` as a
//! [`crate::BrokerEvent::Fulfilled`] or [`crate::BrokerEvent::PromiseFailed`].

use std::sync::{Arc, PoisonError};

use continuum_core::Uuid;
use continuum_core::schema::continuum as fb;
use flatbuffers::FlatBufferBuilder;
use tokio::sync::{mpsc, oneshot};

use crate::messages::finish;
use crate::query::{QueryResult, QuerySpec};
use crate::worker::{Item, item_of};
use crate::{Client, Error, Message, Shared, unexpected};

/// One step of a quoted plan: a tier read (`worker_id` empty), or a worker running a capability.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanStep {
    /// The tier read, for a read step.
    pub read_tier: fb::StorageTier,
    /// The worker that runs it, or empty for a read.
    pub worker_id: String,
    /// The stream it produces.
    pub produces: String,
    /// How many items it covers.
    pub items: u32,
    /// Its estimated time, milliseconds.
    pub est_ms: f32,
}

/// One way to make what's missing.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanOption {
    /// Its steps, in order.
    pub steps: Vec<PlanStep>,
    /// Its estimated latency, milliseconds.
    pub est_latency_ms: f32,
    /// Its estimated cost.
    pub cost_units: f32,
    /// It runs past the query's latency budget.
    pub exceeds_budget: bool,
    /// How sure the estimate is, when the broker says.
    pub expected_confidence: Option<f32>,
}

/// A quote for what a query found missing: commit it ([`Client::commit_promise`]) before `expires_at_ns`, or let it
/// lapse.
#[derive(Clone, Debug, PartialEq)]
pub struct PromiseQuote {
    /// The promise to commit or cancel.
    pub promise_id: Uuid,
    /// What it waits on: compute, capture, or both.
    pub status: fb::PromiseStatus,
    /// The broker's pick first (option 0), then the alternatives.
    pub options: Vec<PlanOption>,
    /// Work already running covers some of it, so committing joins it.
    pub shares_in_flight: bool,
    /// When the quote lapses, UTC nanoseconds.
    pub expires_at_ns: u64,
}

/// Where a promise stands after a commit or a cancel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromiseAccepted {
    /// The promise.
    pub promise_id: Uuid,
    /// Armed (waiting for its range to be captured), Dispatching, Cancelled, or Failed (a `PromiseFailed` follows).
    pub life: fb::PromiseLife,
}

fn uuid_of(id: Option<&fb::Uuid>) -> Uuid {
    Uuid(id.map_or([0; 16], |u| u.0))
}

impl PromiseQuote {
    pub(crate) fn from_fb(q: &fb::PromiseQuote<'_>) -> Self {
        let options = q
            .options()
            .into_iter()
            .flatten()
            .map(|o| PlanOption {
                steps: o
                    .steps()
                    .into_iter()
                    .flatten()
                    .map(|s| PlanStep {
                        read_tier: s.read_tier(),
                        worker_id: s.worker_id().unwrap_or_default().to_owned(),
                        produces: s.produces().unwrap_or_default().to_owned(),
                        items: s.items(),
                        est_ms: s.est_ms(),
                    })
                    .collect(),
                est_latency_ms: o.est_latency_ms(),
                cost_units: o.cost_units(),
                exceeds_budget: o.exceeds_budget(),
                expected_confidence: o.expected_confidence(),
            })
            .collect();
        Self {
            promise_id: uuid_of(q.promise_id()),
            status: q.status(),
            options,
            shares_in_flight: q.shares_in_flight(),
            expires_at_ns: q.expires_at_ns(),
        }
    }
}

fn commit_promise(request_id: u64, promise: Uuid, option: u32, band: fb::PriorityBand) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let id = fb::Uuid(promise.0);
    let c = fb::CommitPromise::create(&mut b, &fb::CommitPromiseArgs { promise_id: Some(&id), option, band });
    finish(&mut b, request_id, fb::ControlPayload::CommitPromise, c.as_union_value())
}

fn cancel_promise(request_id: u64, promise: Uuid) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let id = fb::Uuid(promise.0);
    let c = fb::CancelPromise::create(&mut b, &fb::CancelPromiseArgs { promise_id: Some(&id) });
    finish(&mut b, request_id, fb::ControlPayload::CancelPromise, c.as_union_value())
}

fn accepted(reply: &Message) -> Result<PromiseAccepted, Error> {
    let env = reply.envelope();
    let a = env.payload_as_promise_accepted().ok_or_else(|| unexpected("PromiseAccepted", reply))?;
    Ok(PromiseAccepted { promise_id: uuid_of(a.promise_id()), life: a.life() })
}

/// One part of a promise's outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fulfilment {
    /// The outputs in this part.
    pub items: Vec<Item>,
    /// No more parts follow.
    pub final_part: bool,
}

/// A committed promise: its outputs as they come. Dropping it stops routing them here (later ones go to
/// `Client::next_event`); it doesn't cancel the promise ([`Client::cancel_promise`] does).
pub struct Promise {
    id: Uuid,
    token: u64,
    rx: mpsc::UnboundedReceiver<Message>,
    shared: Arc<Shared>,
    done: bool,
}

impl std::fmt::Debug for Promise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Promise").field("id", &self.id).field("done", &self.done).finish_non_exhaustive()
    }
}

impl Promise {
    /// Routes promise `id`'s outputs to a new handle. One handle per promise: a second would take the first's
    /// outputs, so it's refused.
    fn new(shared: &Arc<Shared>, id: Uuid) -> Result<Self, Error> {
        let (tx, rx) = mpsc::unbounded_channel();
        let token = shared.next_promise_token.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut promises = shared.promises.lock().unwrap_or_else(PoisonError::into_inner);
        if promises.contains_key(&id) {
            return Err(Error::Protocol(format!("promise {id} is held already: one Promise per promise")));
        }
        promises.insert(id, (token, tx));
        drop(promises);
        Ok(Self { id, token, rx, shared: Arc::clone(shared), done: false })
    }

    /// The promise's id.
    #[must_use]
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// The next part of the outputs, or `None` once the final part has come or the promise was cancelled.
    ///
    /// # Errors
    ///
    /// [`Error::PromiseFailed`] when the promise failed, [`Error::Closed`] when the connection ends, and
    /// [`Error::Protocol`] for a part this SDK can't read.
    pub async fn next_part(&mut self) -> Result<Option<Fulfilment>, Error> {
        if self.done {
            return Ok(None);
        }
        let Some(message) = self.rx.recv().await else {
            // The route was ended: by a cancel (nothing more comes), or by the connection closing.
            self.done = true;
            return if self.shared.closed.load(std::sync::atomic::Ordering::SeqCst) {
                Err(Error::Closed)
            } else {
                Ok(None)
            };
        };
        let env = message.envelope();
        if let Some(f) = env.payload_as_promise_failed() {
            self.done = true;
            return Err(Error::PromiseFailed {
                reason: f.reason().unwrap_or_default().to_owned(),
                delivered: f.delivered(),
            });
        }
        let f = env.payload_as_fulfill_promise().ok_or_else(|| unexpected("FulfillPromise", &message))?;
        let items = f
            .items()
            .into_iter()
            .flatten()
            .map(|it| item_of(&it))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| Error::Protocol(format!("a FulfillPromise item's {e}")))?;
        self.done = f.final_part();
        Ok(Some(Fulfilment { items, final_part: f.final_part() }))
    }

    /// Every output, gathered from each part through the final one.
    ///
    /// # Errors
    ///
    /// As [`Promise::next_part`].
    pub async fn fulfilled(mut self) -> Result<Vec<Item>, Error> {
        let mut items = Vec::new();
        while let Some(part) = self.next_part().await? {
            items.extend(part.items);
        }
        Ok(items)
    }
}

impl Drop for Promise {
    fn drop(&mut self) {
        let mut promises = self.shared.promises.lock().unwrap_or_else(PoisonError::into_inner);
        if promises.get(&self.id).is_some_and(|(token, _)| *token == self.token) {
            promises.remove(&self.id);
        }
    }
}

/// Hands a `FulfillPromise` or `PromiseFailed` to the [`Promise`] holding its id; anything else, or one nobody
/// holds, comes back for `Client::next_event`. A final part or a failure ends the route.
pub(crate) fn route(shared: &Shared, message: Message) -> Option<Message> {
    let env = message.envelope();
    let (id, last) = if let Some(f) = env.payload_as_fulfill_promise() {
        (f.promise_id().map(|u| Uuid(u.0)), f.final_part())
    } else if let Some(f) = env.payload_as_promise_failed() {
        (f.promise_id().map(|u| Uuid(u.0)), true)
    } else {
        return Some(message);
    };
    let Some(id) = id else { return Some(message) };
    let mut promises = shared.promises.lock().unwrap_or_else(PoisonError::into_inner);
    let Some((_, tx)) = (if last { promises.remove(&id) } else { promises.get(&id).cloned() }) else {
        return Some(message);
    };
    drop(promises);
    // A Promise dropped since: the message goes nowhere, as its holder asked.
    let _ = tx.send(message);
    None
}

/// Forgets a query's quote waiter when the query is done or abandoned.
struct ForgetQuote<'a> {
    client: &'a Client,
    id: Option<u64>,
}

impl Drop for ForgetQuote<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            self.client.shared.quotes.lock().unwrap_or_else(PoisonError::into_inner).remove(&id);
        }
    }
}

impl Client {
    /// Asks what `spec.stream` holds over `spec.range`, as [`Client::query`], and the quote for what's missing when
    /// the mode asks for one (`QuoteIfMissing`, `CommitWithinBudget`) and the broker gave one. No quote when nothing
    /// is missing, nothing missing can be made, or the mode is `HitOnly`. A `CommitWithinBudget` quote is committed
    /// already.
    ///
    /// # Errors
    ///
    /// As [`Client::query`], and [`Error::Timeout`] when the result says a quote follows and none comes.
    pub async fn query_quoted(&self, spec: &QuerySpec) -> Result<(QueryResult, Option<PromiseQuote>), Error> {
        let (tx, rx) = oneshot::channel::<Message>();
        let mut tx = Some(tx);
        let mut forget = ForgetQuote { client: self, id: None };
        // The waiter goes in under the query's id before it's sent, so a quote can't arrive before anyone waits.
        let reply = self
            .shared
            .request(|id| {
                if let Some(tx) = tx.take() {
                    self.shared.quotes.lock().unwrap_or_else(PoisonError::into_inner).insert(id, tx);
                }
                forget.id = Some(id);
                crate::query::query(id, spec)
            })
            .await?;
        let r = reply.envelope().payload_as_query_result().ok_or_else(|| unexpected("QueryResult", &reply))?;
        let result = crate::query::result_of(&r)?;
        if !result.quoted {
            return Ok((result, None));
        }
        let quote = match tokio::time::timeout(self.shared.timeout, rx).await {
            Ok(Ok(m)) => m,
            Ok(Err(_)) => return Err(Error::Closed),
            Err(_) => return Err(Error::Timeout(self.shared.timeout)),
        };
        let q = quote.envelope().payload_as_promise_quote().ok_or_else(|| unexpected("PromiseQuote", &quote))?;
        Ok((result, Some(PromiseQuote::from_fb(&q))))
    }

    /// Commits a quoted promise (`CommitPromise`): run option `option` of its quote (0 for the broker's pick) at
    /// `band`. Returns where it stands and the [`Promise`] its outputs come to (routed there before the commit is
    /// sent, so none is missed).
    ///
    /// # Errors
    ///
    /// As any request: NotFound for a promise this session wasn't quoted (or whose quote lapsed), BadRequest for an
    /// option the quote doesn't have.
    pub async fn commit_promise(
        &self,
        promise: Uuid,
        option: u32,
        band: fb::PriorityBand,
    ) -> Result<(PromiseAccepted, Promise), Error> {
        let handle = Promise::new(&self.shared, promise)?;
        let reply = self.shared.request(|id| commit_promise(id, promise, option, band)).await?;
        Ok((accepted(&reply)?, handle))
    }

    /// A [`Promise`] for one committed already (by a `CommitWithinBudget` query): its outputs from now on come to it.
    /// A part that arrived before this call went to `Client::next_event`.
    ///
    /// # Errors
    ///
    /// [`Error::Protocol`] when a `Promise` for it is held already.
    pub fn promise(&self, promise: Uuid) -> Result<Promise, Error> {
        Promise::new(&self.shared, promise)
    }

    /// Calls a promise off (`CancelPromise`): its jobs are revoked, and nothing more is delivered. A held
    /// [`Promise`] for it then ends: its `next_part` gives `None` (and `fulfilled` what came before the cancel).
    ///
    /// # Errors
    ///
    /// As any request: NotFound for a promise that's gone (fulfilled, failed, or cancelled already).
    pub async fn cancel_promise(&self, promise: Uuid) -> Result<PromiseAccepted, Error> {
        let reply = self.shared.request(|id| cancel_promise(id, promise)).await?;
        let answer = accepted(&reply)?;
        // Nothing more comes for it, so a held Promise's wait ends rather than hanging.
        self.shared.promises.lock().unwrap_or_else(PoisonError::into_inner).remove(&promise);
        Ok(answer)
    }
}
