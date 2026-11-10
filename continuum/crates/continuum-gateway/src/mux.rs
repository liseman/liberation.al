//! What one connection runs beside its control messages, the same over WebSockets and WebTransport: the
//! `live` feed, the `watch` set, and the requests sent with a `req`. A transport hands each control message to
//! [`Mux::control`] and sends what it answers, and sends whatever [`Mux::next`] yields in between.
//!
//! - **Requests with a `req`** (`range`, `frame`, `spans`) run beside each other instead of in turn. Their replies and
//!   frames are tagged with the `req` ([`crate::header::with_request`]) and sent a message from each in turn, so a
//!   seek isn't stuck behind the range before it; `cancel` drops what one hasn't sent. A request without a `req` is
//!   answered before the next control message is read, untagged, as it always was.
//! - **The watch set**: many streams on one connection. A stream at a rate gets its newest frame that decodes alone
//!   (a JPEG frame, or a video keyframe) at most that many times a second, newest only, so a slow page only ever
//!   waits for the latest. A stream at 0 gets every frame, whole GOPs, through the `live` feed's queue, with its
//!   overlays. Each stream's newest frame (within the `frame` lookback, so nothing from before a privacy pause) goes
//!   out at once, so a tile paints before its first live frame.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio::time::Instant;

use crate::control::{Reply, Request, Watched, WatchedStream, parse};
use crate::header::{self, PayloadType};
use crate::session::{LiveQueue, LiveRun, Outgoing, Session};

/// A watched stream's rate when the request names none, frames a second.
pub const DEFAULT_WATCH_FPS: f64 = 1.0;
/// The highest rate a watched stream gets short of every frame (0).
pub const MAX_WATCH_FPS: f64 = 10.0;
/// The most streams one watch set follows.
pub const MAX_WATCHED: usize = 32;
/// The most requests with a `req` one connection runs at once.
pub const MAX_REQUESTS: usize = 16;

fn error(message: impl Into<String>) -> Outgoing {
    Outgoing::Text(Reply::Error { message: message.into() }.to_json())
}

/// The `req` a request carries, if it's one that can.
fn req_of(request: &Request) -> Option<u32> {
    match request {
        Request::Range { req, .. }
        | Request::Frame { req, .. }
        | Request::Spans { req, .. }
        | Request::Streams { req } => *req,
        _ => None,
    }
}

/// A message tagged with the request that sent it: a reply gains `"req"`, a frame the header's request id.
#[must_use]
pub fn tag(out: Outgoing, req: u32) -> Outgoing {
    match out {
        Outgoing::Binary(bytes) => Outgoing::Binary(header::with_request(&bytes, req)),
        Outgoing::Text(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(serde_json::Value::Object(mut map)) => {
                map.insert("req".to_owned(), req.into());
                Outgoing::Text(serde_json::Value::Object(map).to_string())
            }
            _ => Outgoing::Text(text),
        },
    }
}

/// One connection's running work: its `live` feed, its watch set, and its requests with a `req`.
#[derive(Default)]
pub struct Mux {
    live: LiveRun,
    watch: WatchRun,
    requests: Requests,
}

impl Mux {
    /// Answers one control message: what to send now. A request with a `req` answers nothing here; its replies come
    /// from [`Mux::next`].
    ///
    /// # Errors
    ///
    /// The blocking task of a request without a `req` failed (the transport ends the connection, as before).
    pub async fn control(
        &mut self,
        session: &Arc<Session>,
        text: &str,
    ) -> Result<Vec<Outgoing>, tokio::task::JoinError> {
        let request = match parse(text) {
            Ok(request) => request,
            Err(message) => return Ok(vec![error(message)]),
        };
        Ok(match request {
            Request::Live { stream, overlays } => self.live.start(session, stream, &overlays).await,
            Request::Stop => {
                let out = self.live.stop(None).await;
                if out.is_empty() { session.handle_request(Request::Stop) } else { out }
            }
            Request::Watch { streams, max_fps, media_context } => {
                self.watch.start(session, &streams, max_fps, media_context).await
            }
            Request::Unwatch => {
                let out = self.watch.stop(None).await;
                if out.is_empty() { vec![error("unwatch: nothing is watched")] } else { out }
            }
            Request::Cancel { req } => self.requests.cancel(req),
            request => match req_of(&request) {
                Some(req) => self.requests.start(Arc::clone(session), request, req),
                None => {
                    // A range reads chunk files or waits on the broker: off the runtime's threads.
                    let s = Arc::clone(session);
                    tokio::task::spawn_blocking(move || s.handle_request(request)).await?
                }
            },
        })
    }

    /// The next message the running work has for the page; pending while it has none.
    pub async fn next(&mut self) -> Option<Outgoing> {
        tokio::select! {
            out = self.live.next() => out,
            out = self.watch.next() => out,
            out = self.requests.next() => out,
        }
    }

    /// Ends everything running (the connection is closing).
    pub async fn close(&mut self) {
        self.live.stop(None).await;
        self.watch.stop(None).await;
        self.requests.abort_all();
    }
}

/// The requests sent with a `req` that haven't finished sending.
#[derive(Default)]
struct Requests {
    tasks: Vec<(u32, JoinHandle<()>)>,
    shared: Arc<RequestQueue>,
    /// Blocking reads still running, cancelled ones included: `abort` stops a request's task but not the read it
    /// started on the blocking pool, so the limit counts these too, or a page that starts and cancels in a loop
    /// would queue reads without bound.
    running: Arc<AtomicUsize>,
}

/// Holds one place in [`Requests::running`] until the blocking read it rides with ends, however that ends.
struct Running(Arc<AtomicUsize>);

impl Drop for Running {
    fn drop(&mut self) {
        self.0.fetch_sub(1, SeqCst);
    }
}

/// Each request's replies, waiting for the transport, a message from each in turn.
#[derive(Default)]
struct RequestQueue {
    state: Mutex<RequestState>,
    ready: Notify,
}

/// Under one lock, so a reply is published only while its request still holds its place: a cancel or a replacement
/// takes the place first, and the old task's replies, however late, are dropped.
#[derive(Default)]
struct RequestState {
    /// Replies not sent yet, by request and the generation that made them.
    queues: VecDeque<(u32, u64, VecDeque<Outgoing>)>,
    /// Each request holding a place, from its start until its last reply is sent: its generation. The limit counts
    /// these, so replies a slow page hasn't drained still count (each range can hold tens of MiB).
    places: std::collections::HashMap<u32, u64>,
    next_generation: u64,
}

impl RequestState {
    /// Frees a request's place and drops its unsent replies; whether it held one.
    fn release(&mut self, req: u32) -> bool {
        self.queues.retain(|(r, _, _)| *r != req);
        self.places.remove(&req).is_some()
    }
}

impl RequestQueue {
    /// A finished request's replies, queued only while it still holds its place (a cancel or a replacement may have
    /// taken it while it ran: its replies are dropped then, however late they come).
    fn publish(&self, req: u32, generation: u64, tagged: VecDeque<Outgoing>) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.places.get(&req) != Some(&generation) {
            return;
        }
        if tagged.is_empty() {
            state.places.remove(&req);
            return;
        }
        state.queues.push_back((req, generation, tagged));
        drop(state);
        self.ready.notify_one();
    }
}

impl Requests {
    fn start(&mut self, session: Arc<Session>, request: Request, req: u32) -> Vec<Outgoing> {
        self.tasks.retain(|(_, t)| !t.is_finished());
        let shared = Arc::clone(&self.shared);
        let generation = {
            let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
            // The same id again replaces the one still running under it.
            if state.release(req) {
                self.abort(req);
            }
            if state.places.len() >= MAX_REQUESTS {
                return vec![tag(
                    error(format!("at most {MAX_REQUESTS} requests run or wait to be sent at once: cancel one")),
                    req,
                )];
            }
            if self.running.load(SeqCst) >= MAX_REQUESTS {
                return vec![tag(
                    error(format!("at most {MAX_REQUESTS} reads run at once, cancelled ones until they end: retry")),
                    req,
                )];
            }
            state.next_generation += 1;
            let generation = state.next_generation;
            state.places.insert(req, generation);
            generation
        };
        self.running.fetch_add(1, SeqCst);
        let running = Running(Arc::clone(&self.running));
        let task = tokio::spawn(async move {
            let read = move || {
                let _running = running;
                session.handle_request(request)
            };
            let out = match tokio::task::spawn_blocking(read).await {
                Ok(out) => out,
                Err(e) => vec![error(format!("the request failed: {e}"))],
            };
            shared.publish(req, generation, out.into_iter().map(|o| tag(o, req)).collect());
        });
        self.tasks.push((req, task));
        Vec::new()
    }

    fn abort(&mut self, req: u32) {
        self.tasks.retain(|(r, t)| {
            if *r == req {
                t.abort();
            }
            *r != req
        });
    }

    fn cancel(&mut self, req: u32) -> Vec<Outgoing> {
        let held = self.shared.state.lock().unwrap_or_else(PoisonError::into_inner).release(req);
        self.abort(req);
        if held { vec![Outgoing::Text(Reply::Cancelled { req }.to_json())] } else { Vec::new() }
    }

    fn abort_all(&mut self) {
        for (_, task) in self.tasks.drain(..) {
            task.abort();
        }
        let mut state = self.shared.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.queues.clear();
        state.places.clear();
    }

    /// The next reply of the running requests, one request after another; pending while none has any. A request
    /// whose last reply this is gives up its place.
    async fn next(&mut self) -> Option<Outgoing> {
        loop {
            {
                let mut state = self.shared.state.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some((req, generation, mut queue)) = state.queues.pop_front() {
                    let out = queue.pop_front();
                    if queue.is_empty() {
                        if state.places.get(&req) == Some(&generation) {
                            state.places.remove(&req);
                        }
                    } else {
                        state.queues.push_back((req, generation, queue));
                    }
                    if out.is_some() {
                        return out;
                    }
                    continue;
                }
            }
            self.shared.ready.notified().await;
        }
    }
}

/// A connection's watch set: one live feed per stream, read into each stream's slot.
#[derive(Default)]
pub struct WatchRun {
    tasks: Vec<JoinHandle<()>>,
    shared: Option<Arc<WatchShared>>,
}

impl Drop for WatchRun {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

struct WatchShared {
    slots: Vec<Slot>,
    /// Sent ahead of every slot: each stream's first frame (with its `media` reply), and the set's end.
    outbox: Mutex<VecDeque<Outgoing>>,
    ready: Notify,
    turn: std::sync::atomic::AtomicUsize,
    sent: AtomicU64,
    dropped: AtomicU64,
    skipped: AtomicU64,
    /// Feeds still running; the last to end says the set ended.
    running: std::sync::atomic::AtomicUsize,
    ended: AtomicBool,
}

/// One watched stream's frames waiting for the transport.
struct Slot {
    /// Every frame (a stream at 0), through the live feed's queue: whole GOPs, a slow page loses the oldest.
    every: Option<LiveQueue>,
    /// The newest frame that decodes alone (a stream at a rate), and when this stream last sent one.
    rated: Mutex<Rated>,
}

struct Rated {
    gap: Duration,
    newest: Option<Outgoing>,
    last: Option<Instant>,
}

enum Pop {
    Out(Outgoing),
    Wait(Option<Instant>),
}

impl WatchShared {
    fn pop(&self) -> Pop {
        if let Some(out) = self.outbox.lock().unwrap_or_else(PoisonError::into_inner).pop_front() {
            if matches!(out, Outgoing::Binary(_)) {
                self.sent.fetch_add(1, Relaxed);
            }
            return Pop::Out(out);
        }
        let now = Instant::now();
        let mut due: Option<Instant> = None;
        let n = self.slots.len();
        let start = self.turn.load(Relaxed);
        for k in 0..n {
            let i = (start + k) % n;
            let slot = &self.slots[i];
            let out = if let Some(queue) = &slot.every {
                queue.pop()
            } else {
                let mut rated = slot.rated.lock().unwrap_or_else(PoisonError::into_inner);
                match rated.last.map(|t| t + rated.gap) {
                    Some(at) if rated.newest.is_some() && at > now => {
                        due = Some(due.map_or(at, |d| d.min(at)));
                        None
                    }
                    _ => rated.newest.take().inspect(|_| rated.last = Some(now)),
                }
            };
            if let Some(out) = out {
                self.turn.store((i + 1) % n, Relaxed);
                if matches!(out, Outgoing::Binary(_)) {
                    self.sent.fetch_add(1, Relaxed);
                }
                return Pop::Out(out);
            }
        }
        Pop::Wait(due)
    }

    /// A frame from a stream's feed: queued whole for a stream at 0, else kept when it decodes alone (the newest).
    fn push(&self, i: usize, out: Outgoing) {
        let slot = &self.slots[i];
        let out = match out {
            Outgoing::Binary(bytes) => Outgoing::Binary(header::watched(bytes)),
            text @ Outgoing::Text(_) => text,
        };
        if let Some(queue) = &slot.every {
            queue.push(out);
        } else {
            let alone = match &out {
                Outgoing::Binary(bytes) => header::split(bytes).ok().is_some_and(|(h, _)| {
                    h.payload_type == PayloadType::Jpeg || (h.payload_type == PayloadType::Video && h.keyframe)
                }),
                Outgoing::Text(_) => false,
            };
            if !alone {
                self.skipped.fetch_add(1, Relaxed);
                return;
            }
            let mut rated = slot.rated.lock().unwrap_or_else(PoisonError::into_inner);
            if rated.newest.replace(out).is_some() {
                self.dropped.fetch_add(1, Relaxed);
            }
        }
        self.ready.notify_one();
    }

    fn unwatched(&self, reason: Option<String>) -> Reply {
        let dropped = self.dropped.load(Relaxed)
            + self.slots.iter().filter_map(|s| s.every.as_ref()).map(|q| q.dropped.load(Relaxed)).sum::<u64>();
        Reply::Unwatched { frames: self.sent.load(Relaxed), dropped, skipped: self.skipped.load(Relaxed), reason }
    }
}

/// A stream's rate as asked: frames a second, 0 for every frame.
fn rate(asked: Option<f64>) -> Result<f64, String> {
    match asked {
        None => Ok(DEFAULT_WATCH_FPS),
        Some(fps) if fps.is_finite() && fps >= 0.0 => Ok(fps.min(MAX_WATCH_FPS)),
        Some(fps) => Err(format!("watch: max_fps must be 0 (every frame) or more, not {fps}")),
    }
}

impl WatchRun {
    /// Starts following the set and answers `watching`, or an `error` (a set already running ends first, but only
    /// once every new feed has opened, so a failed request leaves the old set in place).
    pub async fn start(
        &mut self,
        session: &Arc<Session>,
        streams: &[Watched],
        max_fps: Option<f64>,
        media_context: bool,
    ) -> Vec<Outgoing> {
        let mut wanted: Vec<(u32, f64, Vec<String>)> = Vec::new();
        for w in streams {
            let (id, fps, overlays) = match w {
                Watched::Id(id) => (*id, max_fps, Vec::new()),
                Watched::Stream { id, max_fps: own, overlays } => (*id, own.or(max_fps), overlays.clone()),
            };
            let fps = match rate(fps) {
                Ok(fps) => fps,
                Err(message) => return vec![error(message)],
            };
            if !wanted.iter().any(|(w, _, _)| *w == id) {
                wanted.push((id, fps, overlays));
            }
        }
        if wanted.is_empty() || wanted.len() > MAX_WATCHED {
            return vec![error(format!("watch: name 1 to {MAX_WATCHED} streams"))];
        }
        let mut feeds = Vec::with_capacity(wanted.len());
        for (id, _, overlays) in &wanted {
            match session.start_live(*id, overlays).await {
                Ok(feed) => feeds.push(feed),
                Err(message) => return vec![error(format!("watch: stream {id}: {message}"))],
            }
        }
        let mut out = self.stop(Some("replaced by a new watch request".to_owned())).await;
        let mut listed = Vec::with_capacity(wanted.len());
        let mut slots = Vec::with_capacity(wanted.len());
        for ((id, fps, _), feed) in wanted.iter().zip(&feeds) {
            let overlays = match feed.opened() {
                Reply::Live { overlays, .. } => overlays,
                _ => Vec::new(),
            };
            listed.push(WatchedStream { id: *id, max_fps: *fps, overlays });
            slots.push(Slot {
                every: (*fps == 0.0).then(LiveQueue::default),
                rated: Mutex::new(Rated {
                    gap: if *fps > 0.0 { Duration::from_secs_f64(1.0 / fps) } else { Duration::ZERO },
                    newest: None,
                    last: None,
                }),
            });
        }
        out.push(Outgoing::Text(Reply::Watching { streams: listed }.to_json()));
        let shared = Arc::new(WatchShared {
            slots,
            outbox: Mutex::new(VecDeque::new()),
            ready: Notify::new(),
            turn: std::sync::atomic::AtomicUsize::new(0),
            sent: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            running: std::sync::atomic::AtomicUsize::new(feeds.len()),
            ended: AtomicBool::new(false),
        });
        for (i, mut feed) in feeds.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            let session = Arc::clone(session);
            let id = wanted[i].0;
            self.tasks.push(tokio::spawn(async move {
                // The stream's newest frame first, so its tile paints now; the rate counts from it.
                let first = tokio::task::spawn_blocking(move || session.newest(id, media_context)).await;
                {
                    let mut outbox = shared.outbox.lock().unwrap_or_else(PoisonError::into_inner);
                    for out in first.unwrap_or_default().into_iter().chain(feed.standing.take()) {
                        if matches!(out, Outgoing::Binary(_)) {
                            shared.slots[i].rated.lock().unwrap_or_else(PoisonError::into_inner).last =
                                Some(Instant::now());
                        }
                        outbox.push_back(match out {
                            Outgoing::Binary(bytes) => Outgoing::Binary(header::watched(bytes)),
                            text => text,
                        });
                    }
                }
                shared.ready.notify_one();
                let mut skipped = 0;
                while let Some(out) = feed.next().await {
                    shared.skipped.fetch_add(feed.skipped - skipped, Relaxed);
                    skipped = feed.skipped;
                    shared.push(i, out);
                }
                // The broker connection ended: the last feed to end says the set did.
                if shared.running.fetch_sub(1, Relaxed) == 1 {
                    let reply = shared.unwatched(Some("the broker connection ended".to_owned()));
                    // Queued before it's marked ended, so `next` sends it before it gives the set up.
                    shared
                        .outbox
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push_back(Outgoing::Text(reply.to_json()));
                    shared.ended.store(true, Relaxed);
                    shared.ready.notify_one();
                }
            }));
        }
        self.shared = Some(shared);
        out
    }

    /// Ends the set, if one runs, and answers `unwatched` (nothing when none ran, or it ended and said so).
    pub async fn stop(&mut self, reason: Option<String>) -> Vec<Outgoing> {
        for task in self.tasks.drain(..) {
            task.abort();
            let _ = task.await;
        }
        let Some(shared) = self.shared.take() else { return Vec::new() };
        if shared.ended.load(Relaxed) {
            return Vec::new();
        }
        vec![Outgoing::Text(shared.unwatched(reason).to_json())]
    }

    /// The next frame of the set; pending while none is due.
    pub async fn next(&mut self) -> Option<Outgoing> {
        loop {
            let Some(shared) = self.shared.clone() else { return std::future::pending().await };
            match shared.pop() {
                Pop::Out(out) => return Some(out),
                Pop::Wait(_) if shared.ended.load(Relaxed) => {
                    // Its `unwatched` went out already: nothing more comes.
                    self.shared = None;
                    self.tasks.clear();
                    return None;
                }
                Pop::Wait(Some(at)) => {
                    tokio::select! {
                        () = shared.ready.notified() => {}
                        () = tokio::time::sleep_until(at) => {}
                    }
                }
                Pop::Wait(None) => shared.ready.notified().await,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use continuum_core::StreamId;
    use continuum_tier::{ChunkFormat, ChunkWriter};

    use super::*;
    use crate::header::{FLAG_WATCH, FrameHeader, request_id};
    use crate::origin::OriginPolicy;
    use crate::source::ChunkSource;

    /// A session over one chunk file of stream 5: four JPEG frames, 1 ms then every 33 ms.
    fn session(dir: &Path) -> Arc<Session> {
        let mut w =
            ChunkWriter::create(&crate::source::tests::cnk(dir, "5-1-1.cnk"), StreamId(5), ChunkFormat::Passthrough, 1)
                .expect("create");
        for i in 0..4u32 {
            let t = 1_000_000 + u64::from(i) * 33_000_000;
            w.append(t, t, i, &[0xff, 0xd8, i as u8]).expect("append");
        }
        w.close().expect("close");
        Arc::new(Session::new(Arc::new(ChunkSource::new(dir)), None, OriginPolicy::default()))
    }

    fn json(out: &Outgoing) -> serde_json::Value {
        match out {
            Outgoing::Text(t) => serde_json::from_str(t).expect("json"),
            Outgoing::Binary(_) => serde_json::Value::Null,
        }
    }

    /// What the mux sends until a reply of `kind` (`done`, `frame`) comes, at most 5 s.
    async fn until(mux: &mut Mux, kind: &str) -> Vec<Outgoing> {
        let mut got = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let out = mux.next().await.expect("a message");
                let last = json(&out)["type"] == kind;
                got.push(out);
                if last {
                    return;
                }
            }
        })
        .await
        .expect("the reply came");
        got
    }

    #[tokio::test]
    async fn a_request_with_a_req_answers_through_next_with_every_reply_and_frame_tagged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (s, mut mux) = (session(dir.path()), Mux::default());
        let now = mux.control(&s, r#"{"type":"range","stream":5,"start_us":0,"end_us":200000,"req":7}"#).await;
        assert!(now.expect("answered").is_empty(), "nothing inline: it runs beside the others");
        let got = until(&mut mux, "done").await;
        let frames: Vec<_> = got
            .iter()
            .filter_map(|o| match o {
                Outgoing::Binary(b) => Some(b),
                _ => None,
            })
            .collect();
        assert_eq!(frames.len(), 4);
        assert!(frames.iter().all(|b| request_id(b).expect("header") == Some(7)));
        assert_eq!(json(got.last().expect("done"))["req"], 7);
    }

    /// A session like `session`'s, whose stream listing takes `delay`.
    fn slow_session(dir: &Path, delay: Duration) -> Arc<Session> {
        session(dir); // writes the chunk
        Arc::new(Session::new(Arc::new(ChunkSource::slow_listing(dir, delay)), None, OriginPolicy::default()))
    }

    /// Live's camera poll asked `streams` on the page's one connection. Untagged, the connection waited for the
    /// listing (about 1.8 s when it re-read live chunks) and sent nothing else, so every watched stream's queue
    /// overflowed and waited for a keyframe. Tagged, the listing runs beside the rest and frames keep flowing.
    #[tokio::test]
    async fn a_slow_tagged_streams_holds_back_none_of_the_connections_frames() {
        let dir = tempfile::tempdir().expect("tempdir");
        let delay = Duration::from_millis(1500);
        let (s, mut mux) = (slow_session(dir.path(), delay), Mux::default());

        let held = std::time::Instant::now();
        mux.control(&s, r#"{"type":"streams"}"#).await.expect("answered");
        assert!(held.elapsed() >= delay, "untagged, the connection waits for the listing, as before");

        // A fresh session: the listing above is cached for a few seconds, and this one has to be slow too.
        let fresh = tempfile::tempdir().expect("tempdir");
        let (s, mut mux) = (slow_session(fresh.path(), delay), Mux::default());
        let asked = std::time::Instant::now();
        let now = mux.control(&s, r#"{"type":"streams","req":3}"#).await.expect("ok");
        assert!(now.is_empty() && asked.elapsed() < delay / 3, "tagged, it's answered later: nothing waits on it");
        mux.control(&s, r#"{"type":"range","stream":5,"start_us":0,"end_us":200000,"req":7}"#).await.expect("ok");
        let got = until(&mut mux, "done").await;
        assert!(asked.elapsed() < delay, "the range's frames came while the listing was still running");
        assert_eq!(got.iter().filter(|o| matches!(o, Outgoing::Binary(_))).count(), 4);
        assert!(got.iter().all(|o| json(o)["type"] != "streams"), "the listing hadn't answered yet");
        let listed = until(&mut mux, "streams").await;
        assert_eq!(json(listed.last().expect("streams"))["req"], 3);
    }

    #[tokio::test]
    async fn a_request_without_a_req_is_answered_inline_and_untagged_as_before() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (s, mut mux) = (session(dir.path()), Mux::default());
        let out = mux.control(&s, r#"{"type":"frame","stream":5,"at_us":50000}"#).await.expect("answered");
        let Outgoing::Binary(frame) = &out[0] else { panic!("a frame first: {out:?}") };
        assert_eq!(request_id(frame).expect("header"), None);
        assert!(json(&out[1]).get("req").is_none());
    }

    /// A finished request whose replies the page hasn't drained still counts toward the limit.
    #[tokio::test]
    async fn requests_waiting_to_be_sent_count_toward_the_limit_until_they_drain() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (s, mut mux) = (session(dir.path()), Mux::default());
        for req in 0..MAX_REQUESTS as u32 {
            let range = format!(r#"{{"type":"range","stream":5,"start_us":0,"end_us":200000,"req":{req}}}"#);
            assert!(mux.control(&s, &range).await.expect("ok").is_empty());
        }
        // Wait until every one has finished and only waits to be sent.
        tokio::time::timeout(Duration::from_secs(5), async {
            while mux.requests.shared.state.lock().expect("state").queues.len() < MAX_REQUESTS {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("all finished");
        let refused = mux.control(&s, r#"{"type":"frame","stream":5,"at_us":50000,"req":99}"#).await.expect("ok");
        assert_eq!(json(&refused[0])["type"], "error");
        assert_eq!(json(&refused[0])["req"], 99);
        // Draining one request frees its place.
        let mut drained = 0;
        while drained < MAX_REQUESTS * 5 {
            let out = mux.next().await.expect("a reply");
            drained += 1;
            if json(&out)["type"] == "done" {
                break;
            }
        }
        assert!(mux.control(&s, r#"{"type":"frame","stream":5,"at_us":50000,"req":99}"#).await.expect("ok").is_empty());
    }

    #[tokio::test]
    async fn a_cancelled_request_sends_nothing_more_and_a_later_one_is_answered() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (s, mut mux) = (session(dir.path()), Mux::default());
        mux.control(&s, r#"{"type":"range","stream":5,"start_us":0,"end_us":200000,"req":1}"#).await.expect("ok");
        // Wait for its replies to be queued, unsent.
        tokio::time::timeout(Duration::from_secs(5), async {
            while mux.requests.shared.state.lock().expect("state").queues.is_empty() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("queued");
        let cancelled = mux.control(&s, r#"{"type":"cancel","req":1}"#).await.expect("ok");
        assert_eq!(json(&cancelled[0]), serde_json::json!({"type": "cancelled", "req": 1}));
        mux.control(&s, r#"{"type":"frame","stream":5,"at_us":50000,"req":2}"#).await.expect("ok");
        let got = until(&mut mux, "frame").await;
        assert!(
            got.iter().all(|o| match o {
                Outgoing::Binary(b) => request_id(b).expect("header") == Some(2),
                Outgoing::Text(_) => json(o)["req"] == 2,
            }),
            "nothing of the cancelled one: {got:?}"
        );
        assert!(
            mux.control(&s, r#"{"type":"cancel","req":1}"#).await.expect("ok").is_empty(),
            "nothing left to cancel"
        );
    }

    /// A cancel frees the place at once, but the request's blocking read runs to its end. Started and
    /// cancelled in a loop, those reads piled up without bound; they count toward the limit until they finish.
    #[tokio::test]
    async fn cancelled_requests_still_reading_count_toward_the_limit_until_their_reads_end() {
        let dir = tempfile::tempdir().expect("tempdir");
        let delay = Duration::from_millis(1500);
        let (s, mut mux) = (slow_session(dir.path(), delay), Mux::default());
        for req in 0..MAX_REQUESTS as u32 {
            assert!(mux.control(&s, &format!(r#"{{"type":"streams","req":{req}}}"#)).await.expect("ok").is_empty());
            let cancelled = mux.control(&s, &format!(r#"{{"type":"cancel","req":{req}}}"#)).await.expect("ok");
            assert_eq!(json(&cancelled[0])["type"], "cancelled");
        }
        let refused = mux.control(&s, r#"{"type":"streams","req":99}"#).await.expect("ok");
        assert_eq!(json(&refused[0])["type"], "error", "every cancelled listing is still reading: {refused:?}");
        assert_eq!(json(&refused[0])["req"], 99);
        // Once the reads end (one listing, then the rest answer from its cache), there's room again.
        tokio::time::timeout(Duration::from_secs(10), async {
            while mux.requests.running.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the cancelled reads ended");
        assert!(mux.control(&s, r#"{"type":"streams","req":99}"#).await.expect("ok").is_empty());
        let listed = until(&mut mux, "streams").await;
        assert_eq!(json(listed.last().expect("streams"))["req"], 99);
    }

    #[tokio::test]
    async fn watch_needs_the_broker_and_bad_rates_are_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (s, mut mux) = (session(dir.path()), Mux::default());
        let out = mux.control(&s, r#"{"type":"watch","streams":[5]}"#).await.expect("ok");
        assert!(json(&out[0])["message"].as_str().is_some_and(|m| m.contains("stream 5") && m.contains("broker")));
        let bad = mux.control(&s, r#"{"type":"watch","streams":[5],"max_fps":-1}"#).await.expect("ok");
        assert!(json(&bad[0])["message"].as_str().is_some_and(|m| m.contains("max_fps")));
        let none = mux.control(&s, r#"{"type":"watch","streams":[]}"#).await.expect("ok");
        assert_eq!(json(&none[0])["type"], "error");
        let idle = mux.control(&s, r#"{"type":"unwatch"}"#).await.expect("ok");
        assert_eq!(json(&idle[0])["type"], "error");
    }

    #[test]
    fn a_rate_is_capped_and_zero_is_every_frame() {
        assert_eq!(rate(None), Ok(DEFAULT_WATCH_FPS));
        assert_eq!(rate(Some(0.0)), Ok(0.0));
        assert_eq!(rate(Some(2.5)), Ok(2.5));
        assert_eq!(rate(Some(60.0)), Ok(MAX_WATCH_FPS));
        assert!(rate(Some(-1.0)).is_err());
        assert!(rate(Some(f64::NAN)).is_err());
    }

    #[test]
    fn a_reply_gains_its_req_and_a_frame_its_request_id() {
        let text = tag(Outgoing::Text(r#"{"type":"done","stream":5}"#.to_owned()), 9);
        assert_eq!(json(&text), serde_json::json!({"type": "done", "stream": 5, "req": 9}));
        let frame = tag(Outgoing::Binary(FrameHeader::at_ns(5, PayloadType::Jpeg, true, 1).frame(b"j")), 9);
        let Outgoing::Binary(bytes) = frame else { panic!("a frame") };
        assert_eq!(request_id(&bytes).expect("header"), Some(9));
    }

    fn watched_set(every: bool, gap_ms: u64) -> WatchShared {
        WatchShared {
            slots: vec![Slot {
                every: every.then(LiveQueue::default),
                rated: Mutex::new(Rated { gap: Duration::from_millis(gap_ms), newest: None, last: None }),
            }],
            outbox: Mutex::new(VecDeque::new()),
            ready: Notify::new(),
            turn: std::sync::atomic::AtomicUsize::new(0),
            sent: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            running: std::sync::atomic::AtomicUsize::new(1),
            ended: AtomicBool::new(false),
        }
    }

    fn video(at_us: u64, key: bool) -> Outgoing {
        let h = FrameHeader { live: true, ..FrameHeader::at_ns(5, PayloadType::Video, key, at_us * 1_000) };
        Outgoing::Binary(h.frame(&[0, 0, 1, if key { 0x65 } else { 0x41 }]))
    }

    fn capture(out: &Pop) -> u64 {
        match out {
            Pop::Out(Outgoing::Binary(b)) => {
                assert_ne!(b[5] & FLAG_WATCH, 0, "marked as the watch set's");
                header::split(b).expect("header").0.capture_us
            }
            _ => panic!("a frame"),
        }
    }

    #[tokio::test]
    async fn a_stream_at_a_rate_sends_its_newest_frame_that_decodes_alone_at_most_that_often() {
        let set = watched_set(false, 60);
        set.push(0, video(1, false));
        assert!(matches!(set.pop(), Pop::Wait(None)), "a lone P-frame can't be drawn");
        set.push(0, video(2, true));
        assert_eq!(capture(&set.pop()), 2, "the first goes at once");
        set.push(0, video(3, true));
        set.push(0, video(4, true));
        assert!(matches!(set.pop(), Pop::Wait(Some(_))), "not before 60 ms");
        tokio::time::sleep(Duration::from_millis(70)).await;
        assert_eq!(capture(&set.pop()), 4, "only the newest");
        assert!(matches!(set.pop(), Pop::Wait(None)));
        assert_eq!((set.sent.load(Relaxed), set.dropped.load(Relaxed), set.skipped.load(Relaxed)), (2, 1, 1));
    }

    #[test]
    fn a_stream_at_zero_sends_every_frame_whole_gops() {
        let set = watched_set(true, 0);
        set.push(0, video(1, false)); // before its first keyframe: no reference to decode against
        for (at, key) in [(2, true), (3, false), (4, false)] {
            set.push(0, video(at, key));
        }
        let sent: Vec<u64> = (0..3).map(|_| capture(&set.pop())).collect();
        assert_eq!(sent, [2, 3, 4]);
        assert!(matches!(set.pop(), Pop::Wait(None)));
    }
}

#[cfg(test)]
mod request_tests {
    use super::*;

    /// A task past its last await isn't stopped by `abort`, so its replies must be refused instead.
    #[test]
    fn a_late_reply_of_a_cancelled_or_replaced_request_is_dropped() {
        let queue = RequestQueue::default();
        queue.state.lock().expect("state").places.insert(1, 2); // req 1 replaced: generation 2 holds the place
        queue.publish(1, 1, VecDeque::from([Outgoing::Text("{}".to_owned())]));
        assert!(queue.state.lock().expect("state").queues.is_empty(), "generation 1's replies never go out");
        queue.publish(1, 2, VecDeque::from([Outgoing::Text("{}".to_owned())]));
        assert_eq!(queue.state.lock().expect("state").queues.len(), 1);
        queue.state.lock().expect("state").release(1);
        queue.publish(1, 2, VecDeque::from([Outgoing::Text("{}".to_owned())]));
        assert!(queue.state.lock().expect("state").queues.is_empty(), "nor once cancelled");
    }
}
