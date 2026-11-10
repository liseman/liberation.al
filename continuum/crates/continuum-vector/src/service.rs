//! The vector worker as a service (`continuum-vector-worker`): a writer connection that answers dispatched searches
//! and purge notices, answers live queries, files embeddings from each embedding stream it follows and interpretation
//! text from each text stream, and compacts the store now and then (docs/CONTINUUM.md, "Vector worker").
//!
//! Subscription readers are separate from RPCs: their explicit backpressure cannot block an ingest or query reply.
//! A closed reader reconnects with a bounded delay; its durable subscription replays anything not acknowledged.
//! It runs until the writer connection ends or `stop` is set, and returns; the supervisor restarts it. Each
//! stream is followed by a durable subscription of its own, so a stream that isn't registered yet (its producer hasn't
//! started) is waited for without holding up the others. Embedding streams named by a prefix (the journal's
//! `journal.embedding.<digest>`, one per embedding model) are found as they're registered.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use continuum_client::{Client, ClientOptions, Start, SubscribeSpec, Subscription, SubscriptionOverflow};

use crate::VectorStore;
use crate::jobs::{TEXT_QUERY, VECTOR_QUERY};
use crate::worker::{INDEX_SUBSCRIPTION, SearchWorker, WorkerError, register};

/// The broker embedding streams followed when none are named: none now. Voice and face embeddings are in the segment
/// store (`Settings::segment_embeddings`: voice, #1817; faces from see-actions' head faces, #1738).
pub const DEFAULT_EMBEDDINGS: [&str; 0] = [];
/// The embedding streams followed by prefix when none are named: the journal's, one per embedding model (#550).
pub const DEFAULT_EMBEDDING_PREFIXES: [&str; 1] = ["journal.embedding."];
/// The text streams filed for full-text search when none are named: the journal's entries (#550).
pub const DEFAULT_TEXTS: [&str; 1] = ["journal.entry"];

/// What the service needs.
#[derive(Clone, Debug)]
pub struct Settings {
    /// The broker's socket.
    pub socket: PathBuf,
    /// The store's directory.
    pub store: PathBuf,
    /// The embedding streams to follow.
    pub embeddings: Vec<String>,
    /// Every stream whose name starts with one of these is followed as an embedding stream, as it's registered.
    pub embedding_prefixes: Vec<String>,
    /// The interpretation text streams to file for full-text search.
    pub texts: Vec<String>,
    /// The segment store's directory (`paths.state/segments`), or None to follow no segment stream.
    pub segments: Option<PathBuf>,
    /// The embedding streams to follow in the segment store.
    pub segment_embeddings: Vec<crate::segments::SegmentStream>,
    /// Reported to the broker with the registration.
    pub version_hash: String,
    /// How often to look for an embedding stream that isn't registered yet.
    pub wait_for_streams: Duration,
    /// How often to compact the store (merge small files, rebuild the text index, drop old versions).
    pub compact_every: Duration,
}

impl Settings {
    /// The defaults for a broker socket and a store directory.
    pub fn new(socket: PathBuf, store: PathBuf) -> Self {
        Self {
            socket,
            store,
            embeddings: DEFAULT_EMBEDDINGS.iter().map(|s| (*s).to_owned()).collect(),
            embedding_prefixes: DEFAULT_EMBEDDING_PREFIXES.iter().map(|s| (*s).to_owned()).collect(),
            texts: DEFAULT_TEXTS.iter().map(|s| (*s).to_owned()).collect(),
            segments: None,
            segment_embeddings: crate::segments::default_streams(),
            version_hash: env!("CARGO_PKG_VERSION").to_owned(),
            wait_for_streams: Duration::from_secs(10),
            compact_every: Duration::from_secs(3600),
        }
    }
}

/// The durable subscription name for one embedding stream.
#[must_use]
pub fn index_subscription(stream: &str) -> String {
    format!("{INDEX_SUBSCRIPTION}:{stream}")
}

/// The durable subscription name for a stream followed by prefix (`journal.embedding.<digest>`). It differs from
/// [`index_subscription`] so these streams start over once: until the space names were fixed, every embedding on
/// them was skipped (an Ollama model name, with its colon, was refused) and acknowledged, so a cursor kept from then
/// would never file them. Nothing on them was filed, so starting over files each one once.
#[must_use]
pub fn prefixed_subscription(stream: &str) -> String {
    format!("{INDEX_SUBSCRIPTION}:2:{stream}")
}

/// Why the service stopped.
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    /// Connecting, registering, or subscribing failed.
    #[error("broker: {0}")]
    Broker(String),
    /// The store couldn't be opened.
    #[error("store: {0}")]
    Store(#[from] crate::Error),
    /// Filing embeddings failed in the store; the event wasn't acknowledged, so a restart files it again.
    #[error("{0}")]
    Worker(#[from] WorkerError),
}

/// Registers, retrying while a previous session's registration lingers (the broker keeps a keeper's name until it
/// sees that session go).
async fn register_retrying(client: &Client, version_hash: &str) -> Result<crate::worker::SearchStreams, ServiceError> {
    let mut last = String::new();
    for _ in 0..60 {
        match register(client, version_hash).await {
            Ok(streams) => return Ok(streams),
            Err(WorkerError::Client(e)) if e.contains("Conflict") => last = e,
            Err(e) => return Err(ServiceError::Broker(e.to_string())),
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Err(ServiceError::Broker(format!("still registered by an earlier session after 30 s: {last}")))
}

/// What a followed stream holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Embeddings,
    Texts,
}

/// Own the control pump as well as the socket; cancelling intake must not leave an Arc keeping either alive.
struct Reader {
    client: Arc<Client>,
    controls: tokio::task::JoinHandle<()>,
}

impl std::ops::Deref for Reader {
    type Target = Client;
    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.controls.abort();
    }
}

impl Reader {
    async fn stop(&mut self, reason: &str) {
        let _ = self.client.close(reason).await;
        self.controls.abort();
        let _ = (&mut self.controls).await;
    }
}

/// Each intake has its own reader, so another stream's full queue cannot block subscribing or acknowledging it.
async fn subscribe_reader(socket: &Path, spec: &SubscribeSpec) -> Result<(Reader, Subscription), ServiceError> {
    let mut options = ClientOptions::new("continuum-vector-reader");
    options.subscription_overflow = SubscriptionOverflow::Backpressure;
    let client = Arc::new(Client::connect(socket, options).await.map_err(|e| ServiceError::Broker(e.to_string()))?);
    match client.subscribe(spec).await {
        Ok(sub) => {
            let control_client = Arc::clone(&client);
            let controls = tokio::spawn(async move {
                while let Some(message) = control_client.next_event().await {
                    if let Some(error) = message.envelope().payload_as_error() {
                        tracing::warn!("reader error {:?}: {}", error.code(), error.message().unwrap_or(""));
                        let _ = control_client.close("reader error").await;
                        break;
                    }
                    // Readers are not keepers: voice/control updates are drained here; purge work stays on writer.
                }
            });
            Ok((Reader { client, controls }, sub))
        }
        Err(error) => {
            let _ = client.close("subscription failed").await;
            Err(ServiceError::Broker(error.to_string()))
        }
    }
}

/// Follows one stream once registered, reconnecting a closed reader without reopening the keeper or store.
async fn follow(
    worker: Arc<SearchWorker>,
    socket: PathBuf,
    stream: String,
    durable: String,
    wait: Duration,
    kind: Kind,
) -> Result<(), ServiceError> {
    loop {
        let known: HashSet<String> = worker
            .client()
            .list_streams()
            .await
            .map_err(|e| ServiceError::Broker(e.to_string()))?
            .into_iter()
            .map(|s| s.name)
            .collect();
        if known.contains(&stream) {
            break;
        }
        tokio::time::sleep(wait).await;
    }
    let mut spec = SubscribeSpec::new([stream.as_str()], Start::Saved);
    spec.durable_name = Some(durable);
    spec.pinned = true;
    let mut completed = 0;
    loop {
        match subscribe_reader(&socket, &spec).await {
            Ok((mut reader, mut sub)) => {
                let skipped = |position: u64, why: String| {
                    tracing::warn!("{stream} at {position}: skipped: {why}");
                };
                completed = completed.max(sub.start_position());
                // Filing precedes ack. Reconnect can lose that last ack, but must not append the row twice.
                let result = if completed > 0 && sub.ack(completed).is_err() {
                    Err(WorkerError::Client("the reader closed before its checkpoint could be acknowledged".into()))
                } else {
                    match kind {
                        Kind::Embeddings => worker.index_from(&mut sub, completed, |p| completed = p, skipped).await,
                        Kind::Texts => worker.index_texts_from(&mut sub, completed, |p| completed = p, skipped).await,
                    }
                };
                reader.stop("reconnecting subscription").await;
                if let Err(error) = result {
                    if !matches!(error, WorkerError::Client(_)) {
                        return Err(error.into());
                    }
                    tracing::warn!("{stream}: {error}; reconnecting reader");
                }
            }
            Err(error) => tracing::warn!("{stream}: {error}; reconnecting reader"),
        }
        // A lost reader must not restart the keeper registration or reopen the store in a tight loop.
        tokio::time::sleep(wait.max(Duration::from_millis(500))).await;
    }
}

/// Live queries have timed out while disconnected; a replacement reader starts live again.
async fn serve_live(worker: Arc<SearchWorker>, socket: PathBuf, wait: Duration) -> Result<(), ServiceError> {
    let mut spec = SubscribeSpec::new([VECTOR_QUERY, TEXT_QUERY], Start::Live);
    spec.pinned = true;
    loop {
        match subscribe_reader(&socket, &spec).await {
            Ok((mut reader, mut sub)) => {
                let result = worker
                    .serve(&mut sub, |position, why| {
                        tracing::warn!("query at {position}: {why}");
                    })
                    .await;
                reader.stop("reconnecting live queries").await;
                result?;
            }
            Err(error) => tracing::warn!("live queries: {error}; reconnecting reader"),
        }
        tokio::time::sleep(wait.max(Duration::from_millis(500))).await;
    }
}

/// Follows every stream whose name starts with one of `prefixes` as an embedding stream, finding new ones every
/// `wait`, until one of those subscriptions ends or fails.
async fn follow_prefixed(
    worker: Arc<SearchWorker>,
    prefixes: Vec<String>,
    socket: PathBuf,
    named: Vec<String>,
    wait: Duration,
) -> Result<(), ServiceError> {
    let mut followed: HashSet<String> = named.into_iter().collect(); // followed by name already
    let mut tasks = tokio::task::JoinSet::new();
    let mut tick = tokio::time::interval(wait);
    loop {
        tokio::select! {
            _ = tick.tick() => {
                let streams = worker.client().list_streams().await.map_err(|e| ServiceError::Broker(e.to_string()))?;
                for name in streams.into_iter().map(|s| s.name) {
                    if prefixes.iter().any(|p| name.starts_with(p.as_str())) && followed.insert(name.clone()) {
                        let worker = Arc::clone(&worker);
                        let socket = socket.clone();
                        tasks.spawn(async move {
                            follow(worker, socket, name.clone(), prefixed_subscription(&name), wait, Kind::Embeddings).await?;
                            Err::<(), _>(ServiceError::Broker(format!("the subscription to {name} ended")))
                        });
                    }
                }
            }
            Some(done) = tasks.join_next() => {
                return match done {
                    Ok(r) => r,
                    Err(e) => Err(ServiceError::Broker(format!("a task failed: {e}"))),
                };
            }
        }
    }
}

/// Runs the service until the broker connection ends, a stream's filing fails in the store, or `stop` is set.
///
/// # Errors
///
/// [`ServiceError`] for what stopped it; `Ok` only after `stop`.
pub async fn run(settings: Settings, stop: Arc<AtomicBool>) -> Result<(), ServiceError> {
    let client = Arc::new(
        Client::connect(&settings.socket, ClientOptions::new("continuum-vector-worker"))
            .await
            .map_err(|e| ServiceError::Broker(format!("{}: {e}", settings.socket.display())))?,
    );
    let streams = register_retrying(&client, &settings.version_hash).await?;
    let store = VectorStore::open(&settings.store).await?;
    let worker = Arc::new(SearchWorker::new(Arc::clone(&client), store, streams));
    // The session id, so the broker's log of this session (why it closed one, say) can be found.
    tracing::info!(
        "session {}: serving searches, store {}, following {:?}, {:?}*, text from {:?}, and \
         {:?} in the segment store {:?}",
        client.session_id(),
        settings.store.display(),
        settings.embeddings,
        settings.embedding_prefixes,
        settings.texts,
        settings.segment_embeddings.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
        settings.segments
    );

    let mut tasks = tokio::task::JoinSet::new();
    {
        let worker = Arc::clone(&worker);
        tasks.spawn(async move {
            worker.run(|e| tracing::warn!("{e}")).await;
            Err(ServiceError::Broker(
                "the broker closed the connection (it closes a client whose outbound queue fills, or one that breaks \
                 the protocol; its log says which, under this session's id)"
                    .into(),
            ))
        });
    }
    {
        let worker = Arc::clone(&worker);
        let socket = settings.socket.clone();
        let wait = settings.wait_for_streams;
        tasks.spawn(serve_live(worker, socket, wait));
    }
    let named = [(Kind::Embeddings, &settings.embeddings), (Kind::Texts, &settings.texts)];
    for (kind, streams) in named {
        for stream in streams.clone() {
            let worker = Arc::clone(&worker);
            let wait = settings.wait_for_streams;
            let socket = settings.socket.clone();
            tasks.spawn(async move {
                follow(worker, socket, stream.clone(), index_subscription(&stream), wait, kind).await?;
                Err(ServiceError::Broker(format!("the subscription to {stream} ended")))
            });
        }
    }
    if !settings.embedding_prefixes.is_empty() {
        let (worker, prefixes) = (Arc::clone(&worker), settings.embedding_prefixes.clone());
        let (named, wait) = (settings.embeddings.clone(), settings.wait_for_streams);
        tasks.spawn(async move {
            follow_prefixed(worker, prefixes, settings.socket.clone(), named, wait).await?;
            Err(ServiceError::Broker("following the prefixed embedding streams ended".into()))
        });
    }
    if let Some(root) = settings.segments.clone() {
        for stream in settings.segment_embeddings.clone() {
            let (worker, root, dir, wait) =
                (Arc::clone(&worker), root.clone(), settings.store.clone(), settings.wait_for_streams);
            tasks.spawn(async move {
                // A segment intake's failure (a bad file, the store briefly unreadable) retries; it doesn't stop the
                // searches the service answers.
                loop {
                    let followed = crate::segments::follow(
                        worker.store(),
                        dir.clone(),
                        root.clone(),
                        stream.clone(),
                        Duration::from_secs(1),
                    );
                    if let Err(e) = followed.await {
                        tracing::warn!("segments {}: {e}; again in {wait:?}", stream.name);
                    }
                    tokio::time::sleep(wait).await;
                }
            });
        }
    }
    {
        let worker = Arc::clone(&worker);
        let every = settings.compact_every;
        tasks.spawn(async move {
            let mut tick = tokio::time::interval(every);
            tick.tick().await; // the first tick is now
            loop {
                tick.tick().await;
                if let Err(e) = worker.store().compact().await {
                    tracing::warn!("compacting: {e}");
                }
            }
        });
    }
    let mut stopping = tokio::time::interval(Duration::from_millis(200));
    let result = loop {
        tokio::select! {
            done = tasks.join_next() => {
                break match done {
                    Some(Ok(r)) => r,
                    Some(Err(e)) => Err(ServiceError::Broker(format!("a task failed: {e}"))),
                    None => Ok(()),
                };
            }
            _ = stopping.tick() => {
                if stop.load(Ordering::SeqCst) {
                    break Ok(());
                }
            }
        }
    };
    tasks.shutdown().await;
    let _ = client.close("stopping").await;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_broker::server::{Config, Server};
    use continuum_client::{StreamSpec, StreamStorage, SubEvent};
    use continuum_core::schema::continuum as fb;
    use continuum_core::{HeaderBuf, HeaderSpec, StorageLocator, StreamId, Uuid};

    /// The reader policy matters: a default Close intake fails its ack once a paused backlog overflows, and a
    /// Backpressure reader on the same durable cursor then resumes at 0 and files every event.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_default_close_intake_fails_its_ack_and_a_backpressure_reader_replays_everything() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("b.sock");
        let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        let writer = Client::connect(&socket, ClientOptions::new("vector-policy-writer")).await.unwrap();
        let stream = writer
            .register_stream(&StreamSpec {
                name: "vector-policy".into(),
                kind: fb::StreamKind::Derived,
                format: fb::PayloadFormat::RawBytes,
                shape_name: "fixture".into(),
                shape_version: 1,
                nominal_rate_hz: 0.0,
                storage: StreamStorage::Slab { class: 0 },
                anchor: None,
                layer: fb::DomainLayer::Layer2Decision,
            })
            .await
            .unwrap();
        let mut spec = SubscribeSpec::new(["vector-policy"], Start::Saved);
        spec.pinned = true;
        spec.durable_name = Some("vector-policy-proof".into());
        let close_reader = Client::connect(&socket, ClientOptions::new("vector-close-reader")).await.unwrap();
        let mut sub = close_reader.subscribe(&spec).await.unwrap();
        let count = continuum_client::subscribe::QUEUE + 32;
        for n in 0..count {
            let h = HeaderSpec::raw(
                Uuid(*uuid::Uuid::now_v7().as_bytes()),
                StreamId(stream.stream_id),
                n as u64 + 1,
                n as u64 + 1,
            );
            writer
                .ingest_acked(&HeaderBuf::build(&h).unwrap(), Some(&StorageLocator::Inline { bytes: vec![1] }))
                .await
                .unwrap();
        }
        // The paused Close reader overflowed: it delivers its bounded prefix, then ends, and its ack fails.
        let mut seen = 0;
        let mut last = 0;
        while let Some(event) = tokio::time::timeout(Duration::from_secs(10), sub.next()).await.unwrap() {
            if let SubEvent::Event { position, .. } = event {
                seen += 1;
                last = position;
            }
        }
        assert!(seen < count, "a Close reader cannot hold a backlog past its queue");
        assert!(matches!(sub.ack(last), Err(continuum_client::Error::Closed)));
        writer.list_streams().await.expect("the writer connection is unaffected");
        drop(sub);
        drop(close_reader);

        let (reader, mut sub) = subscribe_reader(&socket, &spec).await.unwrap();
        assert_eq!(sub.start_position(), 0, "the failed ack saved nothing");
        let mut replayed = 0;
        while replayed < count {
            match tokio::time::timeout(Duration::from_secs(10), sub.next()).await.unwrap().unwrap() {
                SubEvent::Event { position, .. } => {
                    sub.ack(position).expect("a Backpressure reader acknowledges");
                    replayed += 1;
                }
                SubEvent::CaughtUp { .. } => {}
                SubEvent::Undecodable { reason, .. } => panic!("{reason}"),
            }
        }
        sub.unsubscribe().await.unwrap();
        reader.close("policy proof").await.unwrap();
        writer.close("done").await.unwrap();
        let _ = stop.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_paused_intake_does_not_block_writer_rpcs_and_reconnects_after_acked_position() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("b.sock");
        let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        let writer = Client::connect(&socket, ClientOptions::new("vector-writer-proof")).await.unwrap();
        let stream = writer
            .register_stream(&StreamSpec {
                name: "vector-proof".into(),
                kind: fb::StreamKind::Derived,
                format: fb::PayloadFormat::RawBytes,
                shape_name: "fixture".into(),
                shape_version: 1,
                nominal_rate_hz: 0.0,
                storage: StreamStorage::Slab { class: 0 },
                anchor: None,
                layer: fb::DomainLayer::Layer2Decision,
            })
            .await
            .unwrap();
        let mut spec = SubscribeSpec::new(["vector-proof"], Start::Saved);
        spec.pinned = true;
        spec.durable_name = Some("vector-reader-proof".into());
        let (reader, mut sub) = subscribe_reader(&socket, &spec).await.unwrap();
        let count = continuum_client::subscribe::QUEUE + 32;
        for n in 0..count {
            let h = HeaderSpec::raw(
                Uuid(*uuid::Uuid::now_v7().as_bytes()),
                StreamId(stream.stream_id),
                n as u64 + 1,
                n as u64 + 1,
            );
            writer
                .ingest_acked(&HeaderBuf::build(&h).unwrap(), Some(&StorageLocator::Inline { bytes: vec![1] }))
                .await
                .unwrap();
        }
        tokio::time::timeout(Duration::from_secs(5), writer.list_streams()).await.unwrap().unwrap();
        let last = tokio::time::timeout(Duration::from_secs(10), async {
            let mut seen = 0;
            let mut last = 0;
            while seen < count {
                match sub.next().await.expect("reader keeps its bounded backlog") {
                    SubEvent::Event { position, item } => {
                        assert_eq!(item.header.start_ns(), seen as u64 + 1);
                        assert!(position > last);
                        sub.ack(position).expect("backpressure reader can still acknowledge");
                        last = position;
                        seen += 1;
                    }
                    SubEvent::CaughtUp { .. } => {}
                    SubEvent::Undecodable { reason, .. } => panic!("{reason}"),
                }
            }
            last
        })
        .await
        .expect("reader drains instead of deadlocking or closing");
        sub.unsubscribe().await.unwrap();
        reader.close("reconnect proof").await.unwrap();
        // Cursor writes are asynchronous; wait for persistence, not an arbitrary delay.
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let (again, resumed) = subscribe_reader(&socket, &spec).await.unwrap();
                let saved = resumed.start_position();
                resumed.unsubscribe().await.unwrap();
                again.close("cursor checked").await.unwrap();
                if saved == last {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("a replacement reader resumes at its acknowledged position");
        writer.list_streams().await.unwrap();
        writer.close("done").await.unwrap();
        let _ = stop.send(());
    }

    async fn interrupted_intake_does_not_append_replayed_rows(kind: Kind) {
        use futures::FutureExt;
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("b.sock");
        let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        let writer = Arc::new(Client::connect(&socket, ClientOptions::new("vector-replay-writer")).await.unwrap());
        let streams = register(&writer, "fixture").await.unwrap();
        let worker = SearchWorker::new(
            Arc::clone(&writer),
            VectorStore::open(&dir.path().join("store")).await.unwrap(),
            streams,
        );
        let stream = writer
            .register_stream(&StreamSpec {
                name: "intake-proof".into(),
                kind: fb::StreamKind::Derived,
                format: fb::PayloadFormat::RawBytes,
                shape_name: "fixture".into(),
                shape_version: 1,
                nominal_rate_hz: 0.0,
                storage: StreamStorage::Slab { class: 0 },
                anchor: None,
                layer: fb::DomainLayer::Layer2Decision,
            })
            .await
            .unwrap();
        let mut spec = SubscribeSpec::new(["intake-proof"], Start::Saved);
        spec.pinned = true;
        spec.durable_name = Some("vector-replay-proof".into());
        let mut completed = 0;
        for n in 1..=2u64 {
            let mut h = HeaderSpec::raw(Uuid(*uuid::Uuid::now_v7().as_bytes()), StreamId(stream.stream_id), n, n + 1);
            h.shape_name = "embedding/fixture/v1".into();
            let bytes = match kind {
                Kind::Embeddings => [1.0f32, 0.0].into_iter().flat_map(f32::to_le_bytes).collect(),
                Kind::Texts => br#"{"text":"fixture document"}"#.to_vec(),
            };
            writer.ingest_acked(&HeaderBuf::build(&h).unwrap(), Some(&StorageLocator::Inline { bytes })).await.unwrap();
            let (reader, mut sub) = subscribe_reader(&socket, &spec).await.unwrap();
            let after = completed;
            let handled = |position| {
                completed = position;
                // Poll close through its synchronous closed flag, after the store write but before ack.
                let _ = reader.close("lost acknowledgement").now_or_never();
            };
            let skipped = |position, why| panic!("unexpected skipped event at {position}: {why}");
            let result = match kind {
                Kind::Embeddings => worker.index_from(&mut sub, after, handled, skipped).await,
                Kind::Texts => worker.index_texts_from(&mut sub, after, handled, skipped).await,
            };
            assert!(matches!(result, Err(WorkerError::Client(_))), "{result:?}");
            assert!(completed > after, "a closed reader does not lose the successfully filed position");
            writer.list_streams().await.expect("reader closure leaves RPCs usable");
        }
        let hits = match kind {
            Kind::Embeddings => worker
                .store()
                .search(
                    &crate::EmbeddingSpace { model: "fixture".into(), version: "v1".into(), dim: 2 },
                    &[1.0, 0.0],
                    10,
                    &crate::Filter::default(),
                )
                .await
                .unwrap(),
            Kind::Texts => worker.store().text_search("fixture", 10, &crate::Filter::default()).await.unwrap(),
        };
        assert_eq!(hits.len(), 2, "the unacknowledged first row must not be appended again on replay");
        assert_ne!(hits[0].key.payload_id, hits[1].key.payload_id);
        writer.close("done").await.unwrap();
        let _ = stop.send(());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn embedding_reader_reconnect_does_not_append_an_already_filed_row() {
        interrupted_intake_does_not_append_replayed_rows(Kind::Embeddings).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn text_reader_reconnect_does_not_append_an_already_filed_row() {
        interrupted_intake_does_not_append_replayed_rows(Kind::Texts).await;
    }
}
