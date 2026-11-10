//! One client's session, the same over either transport: control messages in, frames and replies out.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use crate::media::{MediaContext, MetadataCache};
use continuum_client::{Client, QuerySpec, Start, SubEvent, SubscribeSpec, Subscription};
use continuum_core::{StorageLocator, StreamId, Uuid};
use continuum_shm::SlotHandle;
use continuum_tier::{ChunkError, ChunkFormat};

use crate::control::{LiveOverlay, Reply, Request, WebTransportInfo, parse};
use crate::header::{FrameHeader, PayloadType};
use crate::origin::OriginPolicy;
use crate::overlay;
use crate::segment_overlays;
use crate::source::{
    AuEntry, ChunkSource, Frame, RangeRead, SourceError, StreamSummary, classify, limit_gops, merge_record_groups,
    record_frames, select_h264_gops,
};
use crate::vision::{self, Group};

/// Aged-out runs asked of the broker at a time (its own default page is 1,000).
const AGED_PAGE: u32 = 1_000;
/// Pages of aged-out runs one spans reply reads: a window with more runs than this lists only the first ones.
const MAX_AGED_PAGES: usize = 64;
/// Stretches of frames closer than this are one span, unless the request says otherwise: 2 s.
pub const DEFAULT_SPAN_GAP_US: u64 = 2_000_000;
/// The most a `spans` request may join across: 60 s.
pub const MAX_SPAN_GAP_US: u64 = 60_000_000;

/// Sorts `spans` and joins those that overlap or are at most `gap` apart.
#[must_use]
pub fn merge_spans(spans: impl IntoIterator<Item = (u64, u64)>, gap: u64) -> Vec<(u64, u64)> {
    let mut all: Vec<(u64, u64)> = spans.into_iter().map(|(a, b)| (a, b.max(a.saturating_add(1)))).collect();
    all.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (a, b) in all {
        match out.last_mut() {
            Some(last) if a <= last.1.saturating_add(gap) => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// The most frames one `range` request sends; a bigger range is cut and `done` says where to continue.
pub const MAX_RANGE_FRAMES: usize = 600;
/// About the most payload bytes one `range` request sends (64 MiB), the other cut: a 1080p MJPEG range is cut here
/// long before 600 frames.
pub const MAX_RANGE_BYTES: usize = 64 * 1024 * 1024;

// A native snapshot is an independent access unit, not a shortened predictive range. Chunk candidates retain
// the existing archive reader because their hull can contain a newer IDR than an individually indexed raw unit.
enum NativeSnapshot {
    Fallback,
    Selected(Option<Frame>),
}

/// The certificate hash the WebTransport server currently presents, lower-case hex; it changes when a self-signed
/// identity rotates, so sessions read it when they say hello rather than keeping a copy.
pub type SharedHash = Arc<RwLock<String>>;

/// Where the WebTransport endpoint is, when it's on.
#[derive(Clone, Debug)]
pub struct WebTransportEndpoint {
    /// The `https://host:port` URL.
    pub url: String,
    /// The current certificate hash (empty for a certificate browsers trust on their own).
    pub hash: SharedHash,
}

impl WebTransportEndpoint {
    /// What a page needs right now.
    #[must_use]
    pub fn info(&self) -> WebTransportInfo {
        WebTransportInfo { url: self.url.clone(), cert_hash: self.hash.read().map(|h| h.clone()).unwrap_or_default() }
    }
}

/// A message to the client.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outgoing {
    /// A control reply, JSON.
    Text(String),
    /// A frame: the 16 byte header then the payload.
    Binary(Vec<u8>),
}

/// The broker, when the gateway runs beside one: streams and ranges come from it, and Tier 0 bytes from its arenas.
/// The connection is reconnected when it dies (every [`RECONNECT_EVERY`] at most), and while it is down the chunk
/// directory serves alone.
pub struct BrokerLink {
    socket: PathBuf,
    client: RwLock<Option<Arc<Client>>>,
    /// The runtime the client lives on, for the blocking request handlers.
    rt: tokio::runtime::Handle,
    /// Stream names and registered shape names by id, from the last listing.
    names: Mutex<HashMap<u32, String>>,
    shapes: Mutex<HashMap<u32, String>>,
    last_attempt: Mutex<Option<std::time::Instant>>,
    metadata: Mutex<MetadataCache>,
}

/// The least time between two attempts to reconnect to the broker.
const RECONNECT_EVERY: std::time::Duration = std::time::Duration::from_secs(5);

impl BrokerLink {
    fn client(&self) -> Option<Arc<Client>> {
        self.client.read().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    /// The client, reconnecting (rate limited) when the last one died; `None` while the broker is unreachable.
    async fn client_or_reconnect(&self) -> Option<Arc<Client>> {
        if let Some(c) = self.client() {
            return Some(c);
        }
        {
            let mut last = self.last_attempt.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if last.is_some_and(|t| t.elapsed() < RECONNECT_EVERY) {
                return None;
            }
            *last = Some(std::time::Instant::now());
        }
        match Client::connect(&self.socket, continuum_client::ClientOptions::new("continuum-gateway")).await {
            Ok(c) => {
                tracing::info!(socket = %self.socket.display(), "reconnected to the broker");
                let c = Arc::new(c);
                *self.client.write().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(&c));
                Some(c)
            }
            Err(e) => {
                tracing::debug!(error = %e, "the broker is still unreachable; serving the chunk directory alone");
                None
            }
        }
    }

    /// From a blocking thread.
    fn client_or_reconnect_blocking(&self) -> Option<Arc<Client>> {
        self.rt.block_on(self.client_or_reconnect())
    }

    /// The connection died: forget it (a later request reconnects), unless a newer one took its place.
    fn lost(&self, dead: &Arc<Client>, e: &continuum_client::Error) {
        if matches!(e, continuum_client::Error::Closed) {
            let mut c = self.client.write().unwrap_or_else(std::sync::PoisonError::into_inner);
            if c.as_ref().is_some_and(|c| Arc::ptr_eq(c, dead)) {
                tracing::warn!("the broker connection ended; serving the chunk directory until it is back");
                *c = None;
            }
        }
    }

    fn remember(&self, listed: &[continuum_client::StreamDesc]) {
        let mut names = self.names.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut shapes = self.shapes.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for s in listed {
            names.insert(s.id.0, s.name.clone());
            shapes.insert(s.id.0, s.shape_name.clone());
        }
    }

    fn name_of(&self, stream: u32) -> Option<String> {
        self.names.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(&stream).cloned()
    }

    fn id_of(&self, name: &str) -> Option<u32> {
        self.names
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .find(|(_, n)| *n == name)
            .map(|(id, _)| *id)
    }

    fn shape_of(&self, stream: u32) -> Option<String> {
        self.shapes.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(&stream).cloned()
    }
}

/// The most items one broker page for a range may ask for, when payloads from before the window keep filling it.
const MAX_QUERY_ITEMS: usize = 4096;

/// What every session shares.
pub struct Session {
    source: Arc<ChunkSource>,
    webtransport: Option<WebTransportEndpoint>,
    origins: OriginPolicy,
    broker: Option<BrokerLink>,
    /// The segment store (`<state>/segments`), for the overlay streams it holds (see [`segment_overlays`]).
    segments: Option<PathBuf>,
}

impl Session {
    /// Sessions over `source`, advertising `webtransport` when it's on, for browsers the `origins` policy allows.
    #[must_use]
    pub fn new(source: Arc<ChunkSource>, webtransport: Option<WebTransportEndpoint>, origins: OriginPolicy) -> Self {
        Self { source, webtransport, origins, broker: None, segments: None }
    }

    /// Reads the overlay streams the segment store holds ([`segment_overlays::STREAMS`]) from `root`.
    #[must_use]
    pub fn with_segments(mut self, root: PathBuf) -> Self {
        self.segments = Some(root);
        self
    }

    /// Puts the broker in the loop: `streams` lists its registrations, `range` asks it with a `Query` (reading Tier 0
    /// bytes from its arenas and chunk frames from the files it names), and `live` follows its subscriptions. `client`
    /// is the connection made at startup, if one was; the link reconnects to `socket` when it is missing or dies, and
    /// serves the chunk directory alone meanwhile. Called on the runtime the client lives on (the request handlers
    /// block on it from their own threads).
    #[must_use]
    pub fn with_broker(mut self, socket: PathBuf, client: Option<Arc<Client>>) -> Self {
        self.broker = Some(BrokerLink {
            socket,
            client: RwLock::new(client),
            rt: tokio::runtime::Handle::current(),
            names: Mutex::new(HashMap::new()),
            shapes: Mutex::new(HashMap::new()),
            last_attempt: Mutex::new(None),
            metadata: Mutex::new(MetadataCache::default()),
        });
        self
    }

    /// Whether the broker is in the loop right now (connected).
    #[must_use]
    pub fn has_broker(&self) -> bool {
        self.broker.as_ref().is_some_and(|l| l.client().is_some())
    }

    /// The WebTransport endpoint as a page needs it now, when it's on.
    #[must_use]
    pub fn webtransport(&self) -> Option<WebTransportInfo> {
        self.webtransport.as_ref().map(WebTransportEndpoint::info)
    }

    /// Which browser origins may open a session.
    #[must_use]
    pub fn origins(&self) -> &OriginPolicy {
        &self.origins
    }

    /// The source.
    #[must_use]
    pub fn source(&self) -> &ChunkSource {
        &self.source
    }

    /// The `hello` that opens a session over `transport`.
    #[must_use]
    pub fn hello(&self, transport: &str) -> Outgoing {
        Outgoing::Text(
            Reply::Hello {
                // streams_req: a `streams` with `req` runs as a request of its own
                features: vec!["watch".to_owned(), "req".to_owned(), "streams_req".to_owned()],
                transport: transport.to_owned(),
                webtransport: self.webtransport(),
                broker: self.has_broker(),
                media_context: true,
                overlay_groups: if self.has_broker() {
                    Group::ALL.iter().map(|g| g.name().to_owned()).collect()
                } else {
                    Vec::new()
                },
            }
            .to_json(),
        )
    }

    /// Handles one control message: everything to send back, in order. `live` and `stop` are the transport's to
    /// run (see [`Session::start_live`]); here they answer with an error.
    #[must_use]
    pub fn handle(&self, text: &str) -> Vec<Outgoing> {
        match parse(text) {
            Ok(r) => self.handle_request(r),
            Err(message) => vec![Outgoing::Text(Reply::Error { message }.to_json())],
        }
    }

    /// Handles a parsed request (the blocking ones; see [`Session::handle`]).
    #[must_use]
    pub fn handle_request(&self, request: Request) -> Vec<Outgoing> {
        match request {
            Request::Ping => vec![Outgoing::Text(Reply::Pong.to_json())],
            Request::Stop => {
                vec![Outgoing::Text(Reply::Error { message: "stop: nothing is live".to_owned() }.to_json())]
            }
            Request::Streams { .. } => vec![Outgoing::Text(
                match self.streams() {
                    Ok(streams) => Reply::Streams { streams },
                    Err(e) => Reply::Error { message: format!("listing streams: {e}") },
                }
                .to_json(),
            )],
            Request::Live { stream, .. } => vec![Outgoing::Text(
                Reply::Error { message: format!("live stream {stream}: this transport doesn't run live sessions") }
                    .to_json(),
            )],
            // The connection's own (crate::mux): a session alone has none to run.
            Request::Watch { .. } | Request::Unwatch | Request::Cancel { .. } => vec![Outgoing::Text(
                Reply::Error { message: "watch and cancel need a connection".to_owned() }.to_json(),
            )],
            Request::Range { stream, start_us, end_us, after_us, max_frames, overlays, media_context, .. } => {
                self.range_context(stream, start_us, end_us, max_frames, overlays.as_deref(), media_context, after_us)
            }
            Request::Frame { stream, at_us, media_context, .. } => self.frame(stream, at_us, media_context),
            Request::Spans { stream, start_us, end_us, gap_us, media_context, .. } => {
                let gap = gap_us.unwrap_or(DEFAULT_SPAN_GAP_US).min(MAX_SPAN_GAP_US);
                match self.spans(stream, start_us.saturating_mul(1_000), end_us.saturating_mul(1_000), media_context) {
                    Ok((raw, source_partial)) => {
                        let spans = merge_spans(raw.into_iter().map(|(a, b)| (a / 1_000, b.div_ceil(1_000))), gap)
                            .into_iter()
                            .map(|(a, b)| (a.max(start_us), b.min(end_us)))
                            .filter(|(a, b)| a < b)
                            .collect();
                        vec![Outgoing::Text(
                            Reply::Spans {
                                stream,
                                start_us,
                                end_us,
                                spans,
                                expired: self.expired(stream, start_us, end_us),
                                partial: source_partial,
                            }
                            .to_json(),
                        )]
                    }
                    Err(e) => vec![Outgoing::Text(Reply::Error { message: format!("spans: {e}") }.to_json())],
                }
            }
        }
    }

    /// The stretches of `[start_us, end_us)` whose video aged out past the history window (read a page of
    /// `AGED_PAGE` runs at a time, at most `MAX_AGED_PAGES` pages), in microseconds, merged
    /// and clipped to the window. Empty without a broker, or when the broker can't say: never an error, since the
    /// spans reply stands without it.
    fn expired(&self, stream: u32, start_us: u64, end_us: u64) -> Vec<(u64, u64)> {
        let Some(link) = &self.broker else { return Vec::new() };
        let Some(client) = link.client_or_reconnect_blocking() else { return Vec::new() };
        let Ok(Some(name)) = self.stream_name(link, &client, stream) else { return Vec::new() };
        let (start_ns, end_ns) = (start_us.saturating_mul(1_000), end_us.saturating_mul(1_000));
        // The broker answers at most a page of runs, in order, so read on from the last one's end.
        let mut aged = Vec::new();
        let mut from = start_ns;
        for _ in 0..MAX_AGED_PAGES {
            let Ok(page) = link.rt.block_on(client.aged_out(&name, (from, end_ns), AGED_PAGE)) else {
                return Vec::new();
            };
            let full = page.len() >= AGED_PAGE as usize;
            if let Some(&(_, end)) = page.last() {
                from = end;
            }
            aged.extend(page);
            if !full || from >= end_ns {
                break;
            }
        }
        merge_spans(aged.into_iter().map(|(a, b)| (a / 1_000, b.div_ceil(1_000))), 0)
            .into_iter()
            .map(|(a, b)| (a.max(start_us), b.min(end_us)))
            .filter(|(a, b)| a < b)
            .collect()
    }

    /// The frame showing at `at_us`, alone, for scrubbing: the newest frame at or before it, or for GOP video the
    /// keyframe at or before it (which decodes alone), then a `frame` reply. Read with `range` over a short
    /// look-back, widened for a keyframe further back.
    /// A stream's newest frame that decodes alone, within the `frame` lookback (15 s), with its `media` reply when
    /// asked: what a `watch` paints a stream's tile with at once. Nothing for a stream with no recent frame
    /// (a camera paused for privacy sends nothing new).
    #[must_use]
    pub fn newest(&self, stream: u32, media_context: bool) -> Vec<Outgoing> {
        let now_us = u64::try_from(
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_micros()),
        )
        .unwrap_or(u64::MAX);
        self.frame(stream, now_us, media_context)
            .into_iter()
            .filter(|o| match o {
                Outgoing::Binary(_) => true,
                Outgoing::Text(t) => {
                    serde_json::from_str::<serde_json::Value>(t).ok().is_some_and(|v| v["type"] == "media")
                }
            })
            .collect()
    }

    fn frame(&self, stream: u32, at_us: u64, media_context: bool) -> Vec<Outgoing> {
        match self.native_snapshot(stream, at_us) {
            Ok(NativeSnapshot::Selected(frame)) => {
                let capture_us = frame.as_ref().map(|f| f.header.capture_us);
                let mut out = Vec::new();
                if let Some(frame) = frame {
                    if media_context {
                        let context = frame.media.expect("native snapshot carries its exact geometry");
                        out.push(Outgoing::Text(context.value.to_string()));
                        out.push(Outgoing::Binary(crate::header::with_media_context(
                            frame.header,
                            context.context_id,
                            &frame.bytes,
                        )));
                    } else {
                        out.push(Outgoing::Binary(frame.header.frame(&frame.bytes)));
                    }
                }
                out.push(Outgoing::Text(Reply::Frame { stream, at_us, capture_us }.to_json()));
                return out;
            }
            Ok(NativeSnapshot::Fallback) => {}
            Err(message) => {
                return vec![Outgoing::Text(Reply::Error { message: format!("frame: {message}") }.to_json())];
            }
        }
        for back in FRAME_LOOK_BACK_US {
            let out = self.range_context(
                stream,
                at_us.saturating_sub(back),
                at_us.saturating_add(1),
                None,
                None,
                media_context,
                None,
            );
            let mut best: Option<(u64, &Outgoing)> = None;
            for o in &out {
                match o {
                    Outgoing::Binary(b) => {
                        let Ok((h, _)) = crate::header::split(b) else { continue };
                        let shows = match h.payload_type {
                            PayloadType::Jpeg => true,
                            PayloadType::Video => h.keyframe,
                            _ => false,
                        };
                        if shows && h.capture_us <= at_us && best.is_none_or(|(t, _)| h.capture_us >= t) {
                            best = Some((h.capture_us, o));
                        }
                    }
                    // An error (no such stream, the broker down) is the answer.
                    Outgoing::Text(t) if t.contains(r#""type":"error""#) => return out,
                    Outgoing::Text(_) => {}
                }
            }
            if let Some((capture_us, o)) = best {
                let reply = Reply::Frame { stream, at_us, capture_us: Some(capture_us) };
                let mut selected = Vec::new();
                if let Outgoing::Binary(bytes) = o
                    && let Ok(Some(context)) = crate::header::media_context(bytes)
                {
                    selected.extend(out.iter().filter(|message| matches!(message, Outgoing::Text(text) if serde_json::from_str::<serde_json::Value>(text).ok().is_some_and(|v| v["type"] == "media" && v["context_id"].as_u64() == Some(u64::from(context))))).cloned());
                }
                selected.push(o.clone());
                selected.push(Outgoing::Text(reply.to_json()));
                return selected;
            }
        }
        vec![Outgoing::Text(Reply::Frame { stream, at_us, capture_us: None }.to_json())]
    }

    /// An IDR is independently decodable: its current GOP need not have a future closing keyframe. This path
    /// deliberately does not change the strict complete-chain policy of recorded ranges or archived GOP reads.
    fn native_snapshot(&self, stream: u32, at_us: u64) -> Result<NativeSnapshot, String> {
        let Some(link) = &self.broker else { return Ok(NativeSnapshot::Fallback) };
        let Some(client) = link.client_or_reconnect_blocking() else { return Ok(NativeSnapshot::Fallback) };
        let Some(name) = self.stream_name(link, &client, stream)? else { return Ok(NativeSnapshot::Fallback) };
        if link.shape_of(stream).as_deref() != Some("H264AnnexB") {
            return Ok(NativeSnapshot::Fallback);
        }
        let start_ns = at_us.saturating_sub(FRAME_LOOK_BACK_US[2]).saturating_mul(1000);
        let end_ns = at_us.saturating_add(1).saturating_mul(1000);
        let mut spec = QuerySpec::hit_only(name, (start_ns, end_ns));
        spec.max_items = MAX_QUERY_ITEMS as u32;
        let result = link
            .rt
            .block_on(client.query(&spec))
            .map_err(|_| "native snapshot unavailable: broker query failed".to_owned())?;
        if result.truncated {
            return Err("native snapshot unavailable: bounded catalog context was exhausted".into());
        }
        // An overlapping archive hull can hide a newer keyframe. Keep its existing exact revision/tombstone path.
        let chunk_end = result
            .items
            .iter()
            .filter(|item| matches!(item.locator, Some(StorageLocator::Chunk { .. })))
            .map(|item| item.header.end_ns())
            .max();
        let mut items = result.items;
        items.sort_by_key(|item| std::cmp::Reverse((item.header.start_ns(), item.header.seq())));
        let mut total = 0usize;
        for item in items {
            let at = item.header.start_ns();
            if matches!(item.locator, Some(StorageLocator::Chunk { .. })) || at < start_ns || at >= end_ns {
                continue;
            }
            let locator =
                item.locator.as_ref().ok_or("native snapshot unavailable: an access unit has no readable locator")?;
            let bytes = slot_or_inline_bytes(&client, locator)
                .ok_or("native snapshot unavailable: access-unit bytes became unreadable")?;
            total = total.saturating_add(bytes.len());
            if total > MAX_RANGE_BYTES {
                return Err("native snapshot unavailable: bounded payload context was exhausted".into());
            }
            let Some((PayloadType::Video, key, "avc")) = classify(ChunkFormat::Passthrough, &bytes, Some("avc")) else {
                return Err("native snapshot unavailable: malformed H264 access unit".into());
            };
            if !key {
                continue;
            }
            if chunk_end.is_some_and(|end| end > at) {
                return Ok(NativeSnapshot::Fallback);
            }
            // A fresh browser decoder requires in-band parameter sets on the selected actual IDR. The capture
            // producer guarantees these; a missing parameter set remains an actionable error, never a picture.
            let pps =
                bytes.windows(5).any(|b| b[..3] == [0, 0, 1] && b[3] & 0x80 == 0 && b[3] & 0x60 != 0 && b[3] & 31 == 8);
            if classify(ChunkFormat::Passthrough, &bytes, None) != Some((PayloadType::Video, true, "avc")) || !pps {
                return Err("native snapshot unavailable: IDR has no in-band parameter sets".into());
            }
            let capture =
                item.header.header().capture().ok_or("native snapshot unavailable: capture geometry is missing")?;
            let media = MediaContext::native(stream, capture.width(), capture.height())?;
            return Ok(NativeSnapshot::Selected(Some(Frame {
                header: FrameHeader {
                    flip: capture.flip(),
                    ..FrameHeader::at_ns(stream, PayloadType::Video, true, at)
                },
                bytes,
                codec: "avc",
                media: Some(media),
            })));
        }
        Ok(if chunk_end.is_some() { NativeSnapshot::Fallback } else { NativeSnapshot::Selected(None) })
    }

    #[allow(clippy::too_many_arguments)]
    fn range_context(
        &self,
        stream: u32,
        start_us: u64,
        end_us: u64,
        max_frames: Option<usize>,
        overlays: Option<&[String]>,
        media_context: bool,
        after_us: Option<u64>,
    ) -> Vec<Outgoing> {
        if end_us <= start_us {
            return vec![Outgoing::Text(
                Reply::Error { message: "range: end_us must be after start_us".to_owned() }.to_json(),
            )];
        }
        let groups: Vec<Group> = match overlays.unwrap_or_default().iter().map(|n| Group::named(n).ok_or(n)).collect() {
            Ok(groups) => groups,
            Err(n) => {
                return vec![Outgoing::Text(
                    Reply::Error { message: format!("range: unknown overlay group {n:?}") }.to_json(),
                )];
            }
        };
        let max = max_frames.map_or(MAX_RANGE_FRAMES, |m| m.clamp(1, MAX_RANGE_FRAMES));
        let (start_ns, end_ns) = (start_us.saturating_mul(1_000), end_us.saturating_mul(1_000));
        let read = match &self.broker {
            Some(link) => self.broker_frames(link, stream, start_ns, end_ns, max, MAX_RANGE_BYTES),
            None => self.source.frames(stream, start_ns, end_ns, max, MAX_RANGE_BYTES).map_err(|e| e.to_string()),
        };
        let read = match read {
            Ok(r) => r,
            Err(e) => return vec![Outgoing::Text(Reply::Error { message: format!("range: {e}") }.to_json())],
        };
        let frames = match after_us {
            Some(after) => unreceived_frames(read.frames, after),
            None => read.frames,
        };
        let marks = if groups.is_empty() {
            None
        } else {
            match self.range_overlays(stream, &groups, start_ns, end_ns) {
                Ok(m) => Some(m),
                Err(e) => return vec![Outgoing::Text(Reply::Error { message: format!("range: {e}") }.to_json())],
            }
        };
        let mut out = Vec::with_capacity(frames.len() * 2 + 1);
        let mut codecs = BTreeSet::new();
        let count = frames.len();
        // Each frame's bytes are moved into its message, so the range is held once, not twice.
        let mut announced = HashSet::new();
        for f in frames {
            codecs.insert(f.codec.to_owned());
            let message = if media_context && matches!(f.header.payload_type, PayloadType::Video | PayloadType::Jpeg) {
                let context =
                    f.media.clone().map(Ok).unwrap_or_else(|| MediaContext::new(stream, None, None, "unsupported"));
                let context = match context {
                    Ok(context) => context,
                    Err(message) => return vec![Outgoing::Text(Reply::Error { message }.to_json())],
                };
                if announced.insert(context.context_id) {
                    out.push(Outgoing::Text(context.value.to_string()));
                }
                crate::header::with_media_context(f.header, context.context_id, &f.bytes)
            } else {
                f.header.frame(&f.bytes)
            };
            out.push(Outgoing::Binary(message));
            if let Some(m) = &marks
                && matches!(f.header.payload_type, PayloadType::Video | PayloadType::Jpeg)
                && m.has(f.header.capture_us)
            {
                let header = FrameHeader {
                    stream_id: stream,
                    payload_type: PayloadType::Overlay,
                    keyframe: false,
                    flip: false,
                    live: false,
                    capture_us: f.header.capture_us,
                };
                out.push(Outgoing::Binary(header.frame(&overlay::encode(&m.overlay(f.header.capture_us)))));
            }
        }
        out.push(Outgoing::Text(
            Reply::Done {
                stream,
                start_us,
                end_us,
                frames: count,
                skipped: read.skipped,
                next_us: read.next_us,
                codecs: codecs.into_iter().collect(),
            }
            .to_json(),
        ));
        out
    }
}

/// A follow-on range can overlap the previous GOP through its frame durations. Keep any GOP that has a newer
/// picture intact, including older reordered pictures and its IDR prefix. Independent records are single groups.
/// This runs after the bounded read: the read's pagination cursor stays valid even if a page was all already sent.
fn unreceived_frames(frames: Vec<Frame>, after_us: u64) -> Vec<Frame> {
    let mut frames = frames.into_iter().peekable();
    let mut out = Vec::new();
    while let Some(first) = frames.next() {
        let video = first.header.payload_type == PayloadType::Video;
        let codec = first.codec;
        let mut group = vec![first];
        while video
            && frames
                .peek()
                .is_some_and(|f| f.header.payload_type == PayloadType::Video && f.codec == codec && !f.header.keyframe)
        {
            group.push(frames.next().expect("peeked access unit"));
        }
        if group.iter().any(|f| f.header.capture_us > after_us) {
            out.extend(group);
        }
    }
    out
}

impl Session {
    /// The streams. With the broker up, they are its registrations, each with its name, whether a producer is
    /// connected, and the recorded extent the catalog holds for it (two reads of its time index, whatever the
    /// video window holds), so no chunk file is opened and no chunk directory listed. Without a broker, or while
    /// it is unreachable, they come from the chunk directory alone.
    ///
    /// # Errors
    ///
    /// The broker refusing the listing, or the chunk directory failing to list when it is the fallback.
    pub fn streams(&self) -> Result<Vec<StreamSummary>, String> {
        /// A listing slower than this is logged.
        const SLOW_LISTING: std::time::Duration = std::time::Duration::from_millis(250);
        let from_chunks = || self.source.streams().map_err(|e| e.to_string());
        let Some(link) = &self.broker else { return from_chunks() };
        let Some(client) = link.client_or_reconnect_blocking() else { return from_chunks() };
        let asked = std::time::Instant::now();
        let listed = link.rt.block_on(client.list_streams());
        // DEBUG: a loaded machine makes most listings slow, and at INFO they were most of the gateway's log (#59).
        if asked.elapsed() > SLOW_LISTING {
            tracing::debug!(broker = ?asked.elapsed(), "a slow stream listing");
        }
        let listed = match listed {
            Ok(l) => l,
            Err(e @ continuum_client::Error::Closed) => {
                link.lost(&client, &e);
                return from_chunks();
            }
            Err(e) => return Err(format!("the broker: {e}")),
        };
        link.remember(&listed);
        let mut out: Vec<StreamSummary> = listed
            .into_iter()
            .map(|s| {
                let (start_ns, end_ns) = s.extent.unwrap_or((0, 0));
                StreamSummary {
                    id: s.id.0,
                    format: s.shape_name.to_lowercase(),
                    name: s.name,
                    live: s.live,
                    start_us: start_ns / 1_000,
                    end_us: end_ns.div_ceil(1_000),
                }
            })
            .collect();
        out.sort_by_key(|s| s.id);
        Ok(out)
    }

    /// A stream's name for the broker, from the last listing (refreshed once on a miss); `None` when the broker
    /// doesn't know the id.
    fn stream_name(&self, link: &BrokerLink, client: &Arc<Client>, stream: u32) -> Result<Option<String>, String> {
        if let Some(name) = link.name_of(stream) {
            return Ok(Some(name));
        }
        match link.rt.block_on(client.list_streams()) {
            Ok(listed) => link.remember(&listed),
            Err(e) => {
                link.lost(client, &e);
                return Err(format!("the broker: {e}"));
            }
        }
        Ok(link.name_of(stream))
    }

    /// A range through the broker: a `Query` for what exists, each item's bytes from where it says (a Tier 0 slot in
    /// the broker's arena, inline bytes, or the records of a chunk file), in time order, cut as the chunk path cuts.
    /// Without a reachable broker, or for a stream only the chunk directory knows, the chunk directory answers.
    fn broker_frames(
        &self,
        link: &BrokerLink,
        stream: u32,
        start_ns: u64,
        end_ns: u64,
        max: usize,
        max_bytes: usize,
    ) -> Result<RangeRead, String> {
        let chunks_only = || self.source.frames(stream, start_ns, end_ns, max, max_bytes).map_err(|e| e.to_string());
        let Some(client) = link.client_or_reconnect_blocking() else { return chunks_only() };
        let Some(name) = self.stream_name(link, &client, stream)? else {
            return Err(format!("no stream {stream}"));
        };
        if link.shape_of(stream).as_deref() == Some("H264AnnexB") {
            return self.native_h264_frames(link, &client, name, stream, start_ns, end_ns, max, max_bytes);
        }
        let mut spec = QuerySpec::hit_only(name, (start_ns, end_ns));
        // One item past the cap, so a cut at `max` frames always has a key to continue from (a Tier 0 range of
        // exactly `max` items would otherwise end with no `next_us` and the rest unreachable). Payloads that began
        // before the window but reach into it come first and can fill a small page by themselves (ring batches, a
        // long chunk); such a page says nothing about the window, so it is asked again with a larger one.
        let mut want = max.saturating_add(1);
        let result = loop {
            spec.max_items = u32::try_from(want).unwrap_or(u32::MAX);
            let result = match link.rt.block_on(client.query(&spec)) {
                Ok(r) => r,
                Err(e @ continuum_client::Error::Closed) => {
                    link.lost(&client, &e);
                    return chunks_only();
                }
                Err(e) => return Err(format!("the broker: {e}")),
            };
            let all_before = result.items.iter().all(|i| i.header.start_ns() < start_ns);
            if result.items.len() >= want && all_before && want < MAX_QUERY_ITEMS {
                want = (want * 2).min(MAX_QUERY_ITEMS);
                continue;
            }
            break result;
        };
        // The broker's page may end before the window does: it holds more than `max` (one past the cap was asked
        // for), or the broker cut it at a cap of its own (`truncated`). Either way the range continues after it.
        let page_cut = result.items.len() > max || result.truncated;
        let mut last_item_us: Option<u64> = None;
        let mut keyed: Vec<(u64, Frame)> = Vec::new();
        let mut skipped = 0;
        let mut codec: Option<&'static str> = None;
        let mut chunks_seen = HashSet::new();
        // Where the GOPs of a file registered in place were cut off by the budget, so the range continues there.
        let mut continue_at: Option<u64> = None;
        // Items come in time order, so once one more frame than the cap is in hand (with a later key to cut at)
        // the rest can wait for the next request: nothing past the cut is read.
        for item in result.items {
            if keyed.len() > max && keyed.last().is_some_and(|(k, _)| keyed.first().is_none_or(|(f, _)| f != k)) {
                break;
            }
            let start = item.header.start_ns();
            if start >= start_ns {
                last_item_us = Some(start / 1_000);
            }
            match item.locator {
                Some(locator @ (StorageLocator::Shm { .. } | StorageLocator::Inline { .. })) => {
                    // A chunk record is served by its start; a Tier 0 payload the same way (a spanning payload that
                    // began before the range belongs to the page before).
                    if start < start_ns || start >= end_ns {
                        continue;
                    }
                    match slot_or_inline_bytes(&client, &locator) {
                        Some(bytes) => {
                            match record_frames(
                                ChunkFormat::Passthrough,
                                stream,
                                start,
                                bytes,
                                &mut codec,
                                item.header.header().capture().is_some_and(|c| c.flip()),
                            ) {
                                Some(mut frames) => {
                                    if let Some(capture) = item.header.header().capture() {
                                        let context = MediaContext::native(stream, capture.width(), capture.height())?;
                                        for f in &mut frames {
                                            f.media = Some(context.clone());
                                        }
                                    }
                                    keyed.extend(frames.into_iter().map(|f| (start, f)));
                                }
                                None => skipped += 1,
                            }
                        }
                        None => skipped += 1,
                    }
                }
                Some(locator @ StorageLocator::Chunk { .. }) => {
                    let StorageLocator::Chunk { uri, .. } = &locator else { unreachable!() };
                    // Grouped Arrow records use logical group IDs, not chunk IDs. Only video media
                    // needs a playback snapshot; preserve the existing bounded table-group reader.
                    let context = if item.header.header().format()
                        == continuum_core::schema::continuum::PayloadFormat::EncodedVideo
                    {
                        self.chunk_media(
                            link,
                            &client,
                            stream,
                            item.header.id(),
                            &locator,
                            item.header.header().capture().map(|c| (c.width(), c.height())),
                        )?
                    } else {
                        MediaContext::new(stream, None, None, "unsupported")?
                    };
                    if !chunks_seen.insert((uri.clone(), item.header.id())) {
                        continue;
                    }
                    let path = Path::new(uri.strip_prefix("file://").unwrap_or(uri));
                    let budget = max.saturating_add(1).saturating_sub(keyed.len());
                    // A file registered in place: an MP4 at its own path, read by GOP.
                    if item.header.shape_name() == continuum_tier::inplace_video::SHAPE {
                        let (mut frames, more_skipped, more) =
                            mp4_frames(path, stream, start, start_ns, end_ns, budget, &mut codec)?;
                        for (_, frame) in &mut frames {
                            frame.header.flip = item.header.header().capture().is_some_and(|c| c.flip());
                        }
                        for (_, frame) in &mut frames {
                            frame.media = Some(context.clone());
                        }
                        keyed.extend(frames);
                        skipped += more_skipped;
                        continue_at = match (continue_at, more) {
                            (Some(a), Some(b)) => Some(a.min(b)),
                            (a, b) => a.or(b),
                        };
                        continue;
                    }
                    match ChunkSource::chunk_frames_snapshot(
                        path,
                        stream,
                        start_ns,
                        end_ns,
                        budget,
                        context.frames.as_deref(),
                    ) {
                        Ok((mut frames, more_skipped)) => {
                            for (_, frame) in &mut frames {
                                frame.media = Some(context.clone());
                            }
                            keyed.extend(frames);
                            skipped += more_skipped;
                        }
                        // A chunk purged between the query and the read, or one this process can't reach: nothing
                        // to serve from it, counted so the page knows.
                        Err(SourceError::Chunk(ChunkError::Io(e))) if e.kind() == std::io::ErrorKind::NotFound => {
                            skipped += 1;
                        }
                        Err(SourceError::Chunk(ChunkError::NotAChunk(_))) => skipped += 1,
                        Err(e) => return Err(e.to_string()),
                    }
                }
                // Ring batches (audio, IMU) have no player path yet; a header-only payload has no bytes.
                Some(StorageLocator::Ring { .. }) | None => skipped += 1,
            }
        }
        keyed.sort_by_key(|(k, _)| *k);
        let mut out = RangeRead::default();
        let mut sent_bytes = 0usize;
        let mut last_key: Option<u64> = None;
        for (key, frame) in keyed {
            let full = out.frames.len() >= max || sent_bytes >= max_bytes;
            // A cut never splits the frames that share a record key (a GOP's units, records in one microsecond).
            if full && last_key.is_some_and(|k| k / 1_000 != key / 1_000) {
                out.next_us = Some(key / 1_000);
                break;
            }
            sent_bytes += frame.bytes.len();
            out.frames.push(frame);
            last_key = Some(key);
        }
        // The cut above fires only when a frame past the cap is in hand. A page the broker cut, with skipped items or
        // a cap of its own making it short of `max` frames, would otherwise end with no `next_us`, and the player
        // would take the range as finished: continue after the last frame sent (or the last item that started in
        // the window, when none did). Both lie inside the window, so the cursor always moves forward.
        if out.next_us.is_none() && page_cut {
            let after = last_key.map(|k| k / 1_000).or(last_item_us).map(|us| us + 1);
            out.next_us = after.filter(|&us| us < end_ns / 1_000);
        }
        out.skipped = skipped;
        if out.next_us.is_none()
            && let Some(at) = continue_at
        {
            out.next_us = Some(at / 1_000);
        }
        Ok(out)
    }
    /// Tier0 catalog items are PTS-sorted. Recover commit/decode order before grouping; sequence counters may
    /// reset on reconnect, so they detect gaps but never order distinct capture epochs.
    #[allow(clippy::too_many_arguments)]
    fn native_h264_frames(
        &self,
        link: &BrokerLink,
        client: &Client,
        name: String,
        stream: u32,
        start_ns: u64,
        end_ns: u64,
        max: usize,
        max_bytes: usize,
    ) -> Result<RangeRead, String> {
        const CONTEXT_NS: u64 = 60 * 1_000_000_000;
        let mut spec =
            QuerySpec::hit_only(name, (start_ns.saturating_sub(CONTEXT_NS), end_ns.saturating_add(CONTEXT_NS)));
        spec.max_items = MAX_QUERY_ITEMS as u32;
        let result = link
            .rt
            .block_on(client.query(&spec))
            .map_err(|_| "H264 range unavailable: broker context query failed".to_owned())?;
        if result.truncated {
            return Err("H264 range unavailable: bounded catalog context was exhausted".to_owned());
        }
        let mut ordered: Vec<_> = result.items.into_iter().zip(result.seqs).collect();
        ordered.sort_by_key(|(item, commit)| (*commit, item.header.seq()));
        let mut raw = Vec::new();
        let mut raw_bytes = 0usize;
        let mut keyed = Vec::new();
        let mut skipped = 0;
        let mut seen = HashSet::new();
        let mut codec = Some("avc");
        for (item, _) in ordered {
            let start = item.header.start_ns();
            match item.locator {
                Some(locator @ (StorageLocator::Inline { .. } | StorageLocator::Shm { .. })) => {
                    let seq = item
                        .header
                        .header()
                        .capture()
                        .map(|c| c.device_seq())
                        .ok_or_else(|| "H264 range unavailable: missing native decode sequence".to_owned())?;
                    let mut frame = slot_or_inline_bytes(client, &locator).and_then(|bytes| {
                        record_frames(
                            ChunkFormat::Passthrough,
                            stream,
                            start,
                            bytes,
                            &mut codec,
                            item.header.header().capture().is_some_and(|c| c.flip()),
                        )
                        .and_then(|mut frames| frames.pop())
                    });
                    if let Some(frame) = &mut frame
                        && let Some(capture) = item.header.header().capture()
                    {
                        frame.media = Some(MediaContext::native(stream, capture.width(), capture.height())?);
                    }
                    raw_bytes = raw_bytes.saturating_add(frame.as_ref().map_or(0, |f| f.bytes.len()));
                    if raw_bytes + keyed.iter().map(|(_, f): &(u64, Frame)| f.bytes.len()).sum::<usize>()
                        > 128 * 1024 * 1024
                    {
                        return Err("H264 range unavailable: bounded payload context was exhausted".to_owned());
                    }
                    raw.push(Ok(AuEntry { seq, start_ns: start, end_ns: item.header.end_ns(), frame }));
                }
                Some(locator @ StorageLocator::Chunk { .. }) => {
                    let StorageLocator::Chunk { uri, .. } = &locator else { unreachable!() };
                    let context = self.chunk_media(
                        link,
                        client,
                        stream,
                        item.header.id(),
                        &locator,
                        item.header.header().capture().map(|c| (c.width(), c.height())),
                    )?;
                    if !seen.insert((uri.clone(), item.header.id())) {
                        continue;
                    }
                    let path = Path::new(uri.strip_prefix("file://").unwrap_or(uri));
                    let (mut frames, lost) = ChunkSource::chunk_frames_snapshot(
                        path,
                        stream,
                        start_ns,
                        end_ns,
                        max.saturating_add(1),
                        context.frames.as_deref(),
                    )
                    .map_err(|e| e.to_string())?;
                    if raw.len().saturating_add(keyed.len()).saturating_add(frames.len()) > 8192
                        || raw_bytes + keyed.iter().chain(frames.iter()).map(|(_, f)| f.bytes.len()).sum::<usize>()
                            > 128 * 1024 * 1024
                    {
                        return Err("H264 range unavailable: bounded payload context was exhausted".to_owned());
                    }
                    for (_, frame) in &mut frames {
                        frame.media = Some(context.clone());
                    }
                    keyed.extend(frames);
                    skipped += lost;
                }
                _ => return Err("H264 range unavailable: an access unit has no readable locator".to_owned()),
            }
        }
        let (frames, lost) =
            select_h264_gops(raw, start_ns, end_ns, max, max_bytes, false).map_err(|e| e.to_string())?;
        if lost > 0 {
            return Err("H264 range unavailable: incomplete reference prefix or sequence gap".to_owned());
        }
        keyed = merge_record_groups(frames, keyed);
        Ok(limit_gops(keyed, skipped, max, max_bytes, start_ns))
    }
}

impl Session {
    /// The time each payload or chunk of `stream` overlapping `[start_ns, end_ns)` covers, nanoseconds: from the
    /// broker's index when it's in the loop, else from the chunk files.
    fn chunk_media(
        &self,
        link: &BrokerLink,
        client: &Client,
        stream: u32,
        chunk: Uuid,
        locator: &StorageLocator,
        native_geometry: Option<(u32, u32)>,
    ) -> Result<Arc<MediaContext>, String> {
        if client.broker_schema().1 < 25 {
            return MediaContext::new(stream, None, None, "unsupported");
        }
        let listed = match link.rt.block_on(client.list_streams()) {
            Ok(v) => v,
            Err(error) => return Err(format!("authoritative chunk revision unavailable: {error}")),
        };
        let revision = listed.iter().find(|s| s.id.0 == stream).and_then(|s| s.invalidation_revision);
        let Some(revision) = revision.filter(|r| *r != u64::MAX) else {
            return Err("authoritative chunk revision unavailable".into());
        };
        let key = format!("{}:{revision}:{chunk}:{locator:?}:{native_geometry:?}", client.session_id());
        if let Some(cached) = link.metadata.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(&key) {
            return Ok(cached);
        }
        let info = authoritative_chunk(link.rt.block_on(client.get_chunk(chunk)))?;
        // A chunk still being written grows between the query and this read (a part every 2 s): the same file, only
        // longer, and this snapshot's frame index is the newer of the two.
        let same_file = |known: &StorageLocator| match (known, locator) {
            (
                StorageLocator::Chunk { tier: a, uri: x, offset: o, .. },
                StorageLocator::Chunk { tier: b, uri: y, offset: p, .. },
            ) => a == b && x == y && o == p,
            _ => known == locator,
        };
        if info.chunk_id != chunk || info.stream.0 != stream || !info.locations.iter().any(same_file) {
            return Err("chunk locator changed while opening playback".into());
        }
        let after = link
            .rt
            .block_on(client.list_streams())
            .ok()
            .and_then(|v| v.into_iter().find(|s| s.id.0 == stream))
            .and_then(|s| s.invalidation_revision);
        if after != Some(revision) {
            return Err("chunk metadata changed while opening playback".into());
        }
        let mut context = MediaContext::new(stream, Some(&info), Some(revision), "known")?;
        if let Some((width, height)) = native_geometry
            && width > 0
            && height > 0
        {
            Arc::make_mut(&mut context).value["native_geometry"] = serde_json::json!({"width":width,"height":height});
        }
        link.metadata.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(key, context.clone());
        Ok(context)
    }

    fn spans(
        &self,
        stream: u32,
        start_ns: u64,
        end_ns: u64,
        allow_partial: bool,
    ) -> Result<(Vec<(u64, u64)>, bool), String> {
        let from_chunks = || {
            let mut spans = self.source.chunk_spans(stream, start_ns, end_ns).map_err(|e| e.to_string())?;
            let truncated = spans.len() > MAX_QUERY_ITEMS;
            if truncated && !allow_partial {
                return Err("bounded source spans exhausted; request a smaller window".into());
            }
            spans.truncate(MAX_QUERY_ITEMS);
            Ok((spans, truncated))
        };
        let Some(link) = &self.broker else { return from_chunks() };
        let Some(client) = link.client_or_reconnect_blocking() else { return from_chunks() };
        let Some(name) = self.stream_name(link, &client, stream)? else { return from_chunks() };
        let mut spec = QuerySpec::hit_only(name, (start_ns, end_ns));
        spec.max_items = MAX_QUERY_ITEMS as u32;
        match link.rt.block_on(client.query(&spec)) {
            Ok(r) if r.truncated && !allow_partial => {
                Err("bounded source span query exhausted; request a smaller window".into())
            }
            Ok(r) => Ok((r.items.iter().map(|i| (i.header.start_ns(), i.header.end_ns())).collect(), r.truncated)),
            Err(e) => {
                link.lost(&client, &e);
                Err(format!("the broker: {e}"))
            }
        }
    }

    /// The vision overlays of `groups` for the camera of `stream` over a range, read from the broker.
    fn range_overlays(
        &self,
        stream: u32,
        groups: &[Group],
        start_ns: u64,
        end_ns: u64,
    ) -> Result<vision::Overlays, String> {
        let Some(link) = &self.broker else { return Err("overlay groups need the broker".to_owned()) };
        let Some(client) = link.client_or_reconnect_blocking() else {
            return Err("overlay groups need the broker, which is unreachable right now".to_owned());
        };
        let name = self.stream_name(link, &client, stream)?.ok_or_else(|| format!("no stream {stream}"))?;
        let cam = vision::camera_of(&name).ok_or_else(|| format!("{name} isn't a camera's video stream"))?;
        let mut marks = vision::Overlays::new(cam, groups.iter().copied());
        let known: HashSet<String> = match link.rt.block_on(client.list_streams()) {
            Ok(listed) => listed.into_iter().map(|s| s.name).collect(),
            Err(e) => {
                link.lost(&client, &e);
                return Err(format!("the broker: {e}"));
            }
        };
        // Every record of the window, the whole window (a frame's records may come from any of the streams).
        marks.keep_all();
        // Matches first, so every face box in the window carries its track's name (a detection can come before its
        // track's first match).
        let mut streams = marks.streams();
        streams.sort_by_key(|vs| *vs != "face.match");
        for vs in streams {
            if segment_overlays::holds(vs) {
                if let Some(root) = &self.segments {
                    for (start_ns, body) in segment_overlays::window(root, vs, cam, start_ns, end_ns) {
                        marks.apply(vs, start_ns, &body);
                    }
                }
                continue;
            }
            // A vision stream that no worker has registered yet has nothing to draw, and the face.boxes ring holds
            // only the last minute, live: recorded video's faces come from face.detection and face.match.
            if !known.contains(vs) || vs == vision::FACE_BOXES {
                continue;
            }
            let found = match link.rt.block_on(client.query_all(&QuerySpec::hit_only(vs, (start_ns, end_ns)))) {
                Ok(r) => r,
                Err(e) => {
                    link.lost(&client, &e);
                    return Err(format!("the broker: {e}"));
                }
            };
            for item in found.items {
                if let Some(bytes) = item.locator.as_ref().and_then(|l| slot_or_inline_bytes(&client, l)) {
                    marks.apply(vs, item.header.start_ns(), &bytes);
                }
            }
        }
        Ok(marks)
    }
}

/// What [`mp4_frames`] read: the frames by GOP start, how many units it couldn't classify, and where to continue.
type Mp4Read = (Vec<(u64, Frame)>, usize, Option<u64>);

/// The frames of an MP4 registered in place (`continuum_tier::inplace_video`), which starts at `file_start_ns`:
/// every GOP whose frames overlap `[start_ns, end_ns)`, whole from its keyframe, its access units in decode order
/// with their presentation times, each keyed by its GOP's start so a cut never splits a GOP; at most about `budget`
/// frames (always at least one GOP). Also returns the start of the first GOP left for a later request, when the
/// budget stopped it, so the range can say where to continue.
///
/// Known, won't fix: GOPs a purge cut in the catalog are still served from the file (the file is never rewritten,
/// and the cut flags don't reach the gateway).
fn mp4_frames(
    path: &Path,
    stream: u32,
    file_start_ns: u64,
    start_ns: u64,
    end_ns: u64,
    budget: usize,
    codec: &mut Option<&'static str>,
) -> Result<Mp4Read, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let video = continuum_tier::mp4::read_video(&mut file).map_err(|e| e.to_string())?;
    let frames = continuum_tier::inplace_video::gop_frames(&video, file_start_ns);
    let gops = video.gops();
    let mut out = Vec::new();
    let mut skipped = 0;
    for (i, f) in frames.iter().enumerate() {
        let gop_end = file_start_ns.saturating_add(
            u64::try_from(gops[i].pts_end.max(0)).unwrap_or(0) * 1_000_000_000 / u64::from(video.timescale),
        );
        if f.start_ns >= end_ns || gop_end <= start_ns {
            continue;
        }
        if !out.is_empty() && out.len() >= budget {
            return Ok((out, skipped, Some(f.start_ns)));
        }
        let units = continuum_tier::inplace_video::gop_units(path, file_start_ns, i).map_err(|e| e.to_string())?;
        for u in units {
            match classify(ChunkFormat::Passthrough, &u.bytes, *codec) {
                Some((payload_type, _, name)) => {
                    *codec = Some(name);
                    let header = FrameHeader::at_ns(stream, payload_type, u.key, u.pts_ns);
                    out.push((f.start_ns, Frame { header, bytes: u.bytes, codec: name, media: None }));
                }
                None => skipped += 1,
            }
        }
    }
    Ok((out, skipped, None))
}

/// A payload's bytes from a Tier 0 slot (in one of the client's mapped arenas, by name and generation) or from an
/// inline locator; `None` for a slot that is gone or an arena this client doesn't have, and for other locators.
fn slot_or_inline_bytes(client: &Client, locator: &StorageLocator) -> Option<Vec<u8>> {
    match locator {
        StorageLocator::Inline { bytes } => Some(bytes.clone()),
        StorageLocator::Shm { arena, arena_generation, slab_class, slot, slot_generation, .. } => {
            let mapped =
                client.arenas().iter().find(|a| a.info.name == *arena && a.info.generation == *arena_generation)?;
            let handle = SlotHandle { class: *slab_class, index: *slot, generation: *slot_generation };
            read_slot(&mapped.arena, handle)
        }
        _ => None,
    }
}

/// A gap in decode order invalidates dependent video until a real random-access picture.
struct VideoContinuity {
    previous: Option<u64>,
    waiting: bool,
}

impl Default for VideoContinuity {
    fn default() -> Self {
        Self { previous: None, waiting: true }
    }
}

impl VideoContinuity {
    fn broken(&mut self) {
        self.waiting = true;
    }

    fn accept(&mut self, seq: Option<u64>, key: bool) -> bool {
        if let (Some(last), Some(next)) = (self.previous, seq)
            && next != last.wrapping_add(1)
        {
            self.broken();
        }
        self.previous = seq;
        if key {
            self.waiting = false;
        }
        !self.waiting
    }
}

/// A live view: one pinned subscription over the followed stream and its overlay streams, read into frames.
pub struct LiveFeed {
    client: Arc<Client>,
    sub: Subscription,
    camera: u32,
    overlays: Vec<LiveOverlay>,
    /// Each overlay stream's registered shape name, for payloads whose header carries none.
    overlay_shapes: HashMap<u32, String>,
    /// The vision streams followed for overlay groups, by id, and the camera's overlays built from them.
    vision_streams: HashMap<u32, &'static str>,
    vision: Option<vision::Overlays>,
    /// The objects standing in view when the feed opened, sent right after `live` (see [`standing_objects`]).
    pub(crate) standing: Option<Outgoing>,
    codec: Option<&'static str>,
    continuity: VideoContinuity,
    /// The camera stream's raw shape (UYVY, YUYV, NV12), whose Tier 0 frames go out as JPEG (see [`raw_jpeg`]).
    raw_shape: Option<String>,
    /// Events left out (no bytes, a shape with no converter, an unreadable payload).
    pub skipped: u64,
    /// Overlay records that don't come as subscription events: the `face.boxes` and `pose.boxes` rings (Tier 0, no
    /// events) and the segment store's overlay streams, each followed by a thread of its own into one queue.
    side: Option<Side>,
    /// Bodies or poses read from `pose.track`'s segments because see-actions hadn't registered the `pose.boxes` ring
    /// when the feed opened: the ring is looked for again every [`POSE_RING_RECHECK`], and taken when it appears.
    pose_wait: Option<PoseWait>,
}

/// How often a live feed on `pose.track`'s segments looks for the `pose.boxes` ring.
const POSE_RING_RECHECK: std::time::Duration = std::time::Duration::from_secs(5);

/// What a feed needs to move its bodies and poses from the segments to the ring.
struct PoseWait {
    next: tokio::time::Instant,
    /// The segment store, when the gateway has one.
    root: Option<PathBuf>,
    cam: String,
    /// The segment streams to keep following once poses come from the ring.
    others: Vec<&'static str>,
    tx: tokio::sync::mpsc::Sender<segment_overlays::Live>,
}

/// The side sources' queue, and the threads feeding it (dropping them stops them).
struct Side {
    rx: tokio::sync::mpsc::Receiver<segment_overlays::Live>,
    _rings: Vec<Ring>,
    _segments: Option<segment_overlays::Follow>,
    /// A subscription of its own to `pose.boxes`, when the ring was found after the feed opened.
    _pose_sub: Option<Subscription>,
}

/// A thread following a ring of overlay records (`face.boxes`, `pose.boxes`) into the side queue, each record as one
/// of `stream`'s. A page slower than the ring loses the oldest (live boxes are disposable); dropping it stops the
/// thread within a second.
struct Ring {
    stop: Arc<std::sync::atomic::AtomicBool>,
}

impl Ring {
    fn follow(
        mut reader: continuum_shm::RingReader,
        stream: &'static str,
        tx: tokio::sync::mpsc::Sender<segment_overlays::Live>,
    ) -> Self {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ended = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut buf = vec![0u8; reader.stride()];
            while !ended.load(std::sync::atomic::Ordering::Relaxed) && !tx.is_closed() {
                reader.wait(std::time::Duration::from_secs(1));
                while let Ok(Some(_)) = reader.read(&mut buf) {
                    let Some(json) = continuum_client::length_prefixed(&buf) else { continue };
                    let start_ns = serde_json::from_slice::<serde_json::Value>(json)
                        .ok()
                        .and_then(|v| v.get("start_ns").and_then(serde_json::Value::as_u64));
                    if let Some(start_ns) = start_ns {
                        let _ = tx.try_send((stream, start_ns, json.to_vec()));
                    }
                }
            }
        });
        Self { stop }
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl LiveFeed {
    /// Looks for see-actions' `pose.boxes` ring once more. When it's there, bodies and poses come from it from now
    /// on, and the segment follower keeps the other streams; when not, the next look is [`POSE_RING_RECHECK`] on.
    async fn recheck_pose_ring(&mut self) {
        let Some(wait) = self.pose_wait.as_mut() else { return };
        wait.next = tokio::time::Instant::now() + POSE_RING_RECHECK;
        let Ok(listed) = self.client.list_streams().await else { return };
        if !listed.iter().any(|s| s.name == vision::POSE_BOXES) {
            return;
        }
        let mut spec = SubscribeSpec::new(vec![vision::POSE_BOXES.to_owned()], Start::Live);
        spec.pinned = true;
        let Ok(sub) = self.client.subscribe(&spec).await else { return };
        let Some(reader) = sub.ring_reader(&self.client, None, continuum_shm::ReadFrom::Latest) else { return };
        let Some(wait) = self.pose_wait.take() else { return };
        let Some(side) = self.side.as_mut() else { return };
        side._rings.push(Ring::follow(reader, "pose.track", wait.tx.clone()));
        side._segments = match wait.root {
            Some(root) if !wait.others.is_empty() => {
                Some(segment_overlays::Follow::start(root, wait.cam.clone(), wait.others, wait.tx))
            }
            _ => None,
        };
        side._pose_sub = Some(sub);
        tracing::info!(camera = %wait.cam, "live poses now from the pose.boxes ring (it appeared after the feed opened)");
    }

    /// The `live` reply that opens the feed.
    #[must_use]
    pub fn opened(&self) -> Reply {
        Reply::Live { stream: self.camera, overlays: self.overlays.clone() }
    }

    /// The next frame or overlay to send, or `None` once the subscription has ended (the broker connection ended).
    /// Events without something to send are skipped here.
    pub async fn next(&mut self) -> Option<Outgoing> {
        loop {
            let recheck = self.pose_wait.as_ref().map(|w| w.next);
            let event = match self.side.as_mut() {
                Some(side) => tokio::select! {
                    () = async { if let Some(at) = recheck { tokio::time::sleep_until(at).await } else { std::future::pending().await } } => {
                        self.recheck_pose_ring().await;
                        continue;
                    }
                    event = self.sub.next() => event?,
                    Some((vs, start_ns, json)) = side.rx.recv() => {
                        if let Some(o) = self.vision.as_mut().and_then(|m| m.apply(vs, start_ns, &json)) {
                            let h = FrameHeader {
                                stream_id: self.camera,
                                payload_type: PayloadType::Overlay,
                                keyframe: false,
                                flip: false,
                                live: true,
                                capture_us: o.capture_us,
                            };
                            return Some(Outgoing::Binary(h.frame(&overlay::encode(&o))));
                        }
                        continue;
                    }
                },
                None => self.sub.next().await?,
            };
            let (header, locator) = match event {
                SubEvent::Event { item, .. } => (item.header, item.locator),
                SubEvent::CaughtUp { .. } => continue,
                SubEvent::Undecodable { .. } => {
                    self.continuity.broken();
                    self.skipped += 1;
                    continue;
                }
            };
            let stream = header.stream().0;
            let Some(bytes) = locator.as_ref().and_then(|l| slot_or_inline_bytes(&self.client, l)) else {
                if stream == self.camera {
                    self.continuity.broken();
                }
                self.skipped += 1;
                continue;
            };
            let capture_us = header.start_ns() / 1_000;
            if stream == self.camera
                && let Some(shape) = &self.raw_shape
            {
                let capture = header.header().capture();
                let flip = capture.is_some_and(|c| c.flip());
                let jpeg = capture.and_then(|c| raw_jpeg(shape, c.width(), c.height(), c.stride(), &bytes));
                let Some(jpeg) = jpeg else {
                    self.skipped += 1;
                    continue;
                };
                let h = FrameHeader {
                    stream_id: self.camera,
                    payload_type: PayloadType::Jpeg,
                    keyframe: true,
                    live: true,
                    flip,
                    capture_us,
                };
                return Some(Outgoing::Binary(h.frame(&jpeg)));
            }
            if stream == self.camera {
                let Some(mut frames) = record_frames(
                    ChunkFormat::Passthrough,
                    stream,
                    header.start_ns(),
                    bytes,
                    &mut self.codec,
                    header.header().capture().is_some_and(|c| c.flip()),
                ) else {
                    self.skipped += 1;
                    continue;
                };
                // A Tier 0 payload is one record, so one frame (a GOP record would be several; Tier 0 holds none).
                let Some(mut frame) = frames.pop() else { continue };
                if frame.header.payload_type == PayloadType::Video
                    && !self.continuity.accept(header.header().capture().map(|c| c.device_seq()), frame.header.keyframe)
                {
                    self.skipped += 1;
                    continue;
                }
                frame.header.live = true;
                let mut message = frame.header.encode().to_vec();
                message.extend_from_slice(&frame.bytes);
                return Some(Outgoing::Binary(message));
            }
            // A vision record: merged into its frame's overlay, which goes out whole.
            if let (Some(vs), Some(marks)) = (self.vision_streams.get(&stream), self.vision.as_mut()) {
                match marks.apply(vs, header.start_ns(), &bytes) {
                    Some(o) => {
                        let h = FrameHeader {
                            stream_id: self.camera,
                            payload_type: PayloadType::Overlay,
                            keyframe: false,
                            flip: false,
                            live: true,
                            capture_us: o.capture_us,
                        };
                        return Some(Outgoing::Binary(h.frame(&overlay::encode(&o))));
                    }
                    // Another camera's record, or one that changed nothing here.
                    None => continue,
                }
            }
            // An overlay stream: the `Overlay` FlatBuffer passes through once it decodes (the shape from the header,
            // or the stream's registered one when the header carries none); other shapes wait for their converter
            // (none yet: the vision workers' payload shapes are being settled).
            let shape = header.shape_name();
            let is_overlay = shape == OVERLAY_SHAPE
                || (shape.is_empty() && self.overlay_shapes.get(&stream).is_some_and(|s| s == OVERLAY_SHAPE));
            if is_overlay && overlay::decode(&bytes).is_ok() {
                let h = FrameHeader {
                    stream_id: stream,
                    payload_type: PayloadType::Overlay,
                    keyframe: false,
                    flip: false,
                    live: true,
                    capture_us,
                };
                return Some(Outgoing::Binary(h.frame(&bytes)));
            }
            self.skipped += 1;
        }
    }
}

/// The shape name of a stream whose payloads are already the player's `Overlay` FlatBuffer.
pub const OVERLAY_SHAPE: &str = "overlay";

impl Session {
    /// Starts following `stream` live with `overlays` (stream names), as one pinned subscription on the broker.
    ///
    /// # Errors
    ///
    /// A message for the client: no broker (none configured, or unreachable), an unknown stream or overlay name, or
    /// the broker refusing.
    pub async fn start_live(&self, stream: u32, overlays: &[String]) -> Result<LiveFeed, String> {
        let Some(link) = &self.broker else {
            return Err("live streams need the broker, and the gateway runs without one".to_owned());
        };
        let Some(client) = link.client_or_reconnect().await else {
            return Err("live streams need the broker, which is unreachable right now".to_owned());
        };
        // The names come from the last listing; a miss (a camera the page knows by id, an overlay by name) refreshes
        // it once, here on the runtime (never blocking on it).
        let known =
            |link: &BrokerLink| link.name_of(stream).is_some() && overlays.iter().all(|n| link.id_of(n).is_some());
        if !known(link) {
            match client.list_streams().await {
                Ok(listed) => link.remember(&listed),
                Err(e) => {
                    link.lost(&client, &e);
                    return Err(format!("the broker: {e}"));
                }
            }
        }
        let camera = link.name_of(stream).ok_or_else(|| format!("the broker knows no stream {stream}"))?;
        // Overlay groups are the vision streams' records, translated; any other name is a stream already in the
        // overlay shape.
        let groups: Vec<Group> = overlays.iter().filter_map(|n| Group::named(n)).collect();
        // The listing is refreshed when a page names a stream it lacks, and groups name none: so bodies or poses
        // look for the pose.boxes ring in a fresh listing whenever the last one didn't have it.
        if groups.iter().any(|g| matches!(g, Group::Bodies | Group::Poses))
            && link.id_of(vision::POSE_BOXES).is_none()
            && let Ok(listed) = client.list_streams().await
        {
            link.remember(&listed);
        }
        let overlays: Vec<String> = overlays.iter().filter(|n| Group::named(n).is_none()).cloned().collect();
        let overlays = &overlays;
        let mut vision_streams = HashMap::new();
        let mut standing = None;
        let vision = if groups.is_empty() {
            None
        } else {
            let cam = vision::camera_of(&camera).ok_or_else(|| format!("{camera} isn't a camera's video stream"))?;
            if groups.contains(&Group::Objects)
                && let Some(root) = &self.segments
            {
                standing = standing_objects(root, cam).map(|o| {
                    let h = FrameHeader {
                        stream_id: stream,
                        payload_type: PayloadType::Overlay,
                        keyframe: false,
                        flip: false,
                        live: true,
                        capture_us: o.capture_us,
                    };
                    Outgoing::Binary(h.frame(&overlay::encode(&o)))
                });
            }
            let marks = vision::Overlays::new(cam, groups);
            // Live bodies and poses come from see-actions' pose.boxes ring as each frame is analyzed, when it has
            // registered one; else from pose.track's segments, which reach the store in 1 s batches.
            if marks.streams().contains(&"pose.track")
                && let Some(id) = link.id_of(vision::POSE_BOXES)
            {
                vision_streams.insert(id, vision::POSE_BOXES);
            }
            for vs in marks.streams() {
                // The segment store's streams are followed beside the subscription; a broker stream no worker has
                // registered yet is left out, with nothing to draw.
                if segment_overlays::holds(vs) {
                    continue;
                }
                if let Some(id) = link.id_of(vs) {
                    vision_streams.insert(id, vs);
                }
            }
            Some(marks)
        };
        let mut listed = Vec::with_capacity(overlays.len());
        let mut overlay_shapes = HashMap::new();
        for name in overlays {
            let id = link.id_of(name).ok_or_else(|| format!("the broker knows no stream named {name}"))?;
            if let Some(shape) = link.shape_of(id) {
                overlay_shapes.insert(id, shape);
            }
            listed.push(LiveOverlay { name: name.clone(), id });
        }
        let mut streams = vec![camera];
        streams.extend(overlays.iter().cloned());
        for (&id, &vs) in &vision_streams {
            streams.push(vs.to_owned());
            listed.push(LiveOverlay { name: vs.to_owned(), id });
        }
        let mut spec = SubscribeSpec::new(streams, Start::Live);
        spec.pinned = true;
        let sub = match client.subscribe(&spec).await {
            Ok(s) => s,
            Err(e) => {
                link.lost(&client, &e);
                return Err(format!("the broker: {e}"));
            }
        };
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        // Each ring's records go to the overlay as the stream whose shape they have.
        let mut rings = Vec::new();
        for (ring, as_stream) in [(vision::FACE_BOXES, vision::FACE_BOXES), (vision::POSE_BOXES, "pose.track")] {
            let Some((&id, _)) = vision_streams.iter().find(|(_, vs)| **vs == ring) else { continue };
            if let Some(reader) = sub.ring_reader(&client, Some(StreamId(id)), continuum_shm::ReadFrom::Latest) {
                rings.push((ring, Ring::follow(reader, as_stream, tx.clone())));
            }
        }
        let poses_live = rings.iter().any(|(ring, _)| *ring == vision::POSE_BOXES);
        let from_segments: Vec<&'static str> = vision
            .as_ref()
            .map(|m| {
                m.streams()
                    .into_iter()
                    .filter(|vs| segment_overlays::holds(vs) && !(poses_live && *vs == "pose.track"))
                    .collect()
            })
            .unwrap_or_default();
        let rings: Vec<Ring> = rings.into_iter().map(|(_, r)| r).collect();
        let wants_poses = vision.as_ref().is_some_and(|m| m.streams().contains(&"pose.track"));
        let pose_wait = match vision.as_ref() {
            Some(m) if wants_poses && !poses_live => Some(PoseWait {
                next: tokio::time::Instant::now() + POSE_RING_RECHECK,
                root: self.segments.clone(),
                cam: m.camera().to_owned(),
                others: from_segments.iter().copied().filter(|vs| *vs != "pose.track").collect(),
                tx: tx.clone(),
            }),
            _ => None,
        };
        if let Some(m) = vision.as_ref()
            && m.streams().contains(&"pose.track")
        {
            let path = if poses_live { "the pose.boxes ring" } else { "pose.track segments, until the ring appears" };
            // Per camera per connection, so DEBUG (#59).
            tracing::debug!(camera = %m.camera(), "live poses from {path}");
        }
        let segments = match (&self.segments, vision.as_ref(), from_segments.is_empty()) {
            (Some(root), Some(m), false) => {
                Some(segment_overlays::Follow::start(root.clone(), m.camera().to_owned(), from_segments, tx))
            }
            _ => None,
        };
        let side = (!rings.is_empty() || segments.is_some() || pose_wait.is_some()).then_some(Side {
            rx,
            _rings: rings,
            _segments: segments,
            _pose_sub: None,
        });
        Ok(LiveFeed {
            side,
            client,
            sub,
            camera: stream,
            overlays: listed,
            overlay_shapes,
            vision_streams,
            vision,
            standing,
            codec: (link.shape_of(stream).as_deref() == Some("H264AnnexB")).then_some("avc"),
            continuity: VideoContinuity::default(),
            raw_shape: link.shape_of(stream).filter(|s| continuum_tier::raw::RawFormat::of_shape(s).is_some()),
            skipped: 0,
            pose_wait,
        })
    }
}

/// A raw camera's Tier 0 frame (UYVY, YUYV, NV12; its layout from the capture info) as a JPEG at half size, for the
/// live view: the page can't draw raw frames, and a raw camera's short-GOP H.264 is Tier 1, written after the fact
/// Without this a raw camera (the Arducams on the Mac, which offer no MJPEG there) showed no live picture at all.
/// None when the frame can't be laid out or decoded.
fn raw_jpeg(shape: &str, width: u32, height: u32, stride: u32, bytes: &[u8]) -> Option<Vec<u8>> {
    let layout = continuum_tier::raw::RawLayout::of(shape, width, height, stride)?;
    let nv12 = continuum_tier::raw::to_nv12(&layout, bytes).ok()?;
    let rgb = continuum_tier::hydrate::nv12_to_rgb(&nv12, layout.width, layout.height, true)?;
    let (w, h) = (usize::try_from(rgb.width).ok()?, usize::try_from(rgb.height).ok()?);
    let (hw, hh) = (w / 2, h / 2);
    if hw == 0 || hh == 0 {
        return None;
    }
    let mut half = Vec::with_capacity(hw * hh * 3);
    for y in 0..hh {
        let row = &rgb.rgb[(y * 2) * w * 3..];
        for x in 0..hw {
            half.extend_from_slice(&row[x * 6..x * 6 + 3]);
        }
    }
    let mut out = Vec::new();
    jpeg_encoder::Encoder::new(&mut out, 75)
        .encode(&half, u16::try_from(hw).ok()?, u16::try_from(hh).ok()?, jpeg_encoder::ColorType::Rgb)
        .ok()?;
    Some(out)
}

/// How far back a live session looks for the objects standing in a camera's view: the caption worker writes each
/// camera's grounded objects again with every caption, at least every 10 s, so a minute finds the newest.
const STANDING_OBJECTS_NS: u64 = 60 * 1_000_000_000;

/// The camera's newest camera.objects record from the last [`STANDING_OBJECTS_NS`] as an overlay, so a page that
/// starts following shows the objects in view now instead of waiting for the next grounding. None without one.
fn standing_objects(root: &std::path::Path, cam: &str) -> Option<overlay::Overlay> {
    let now =
        u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos()).ok()?;
    let found = segment_overlays::window(
        root,
        "camera.objects",
        cam,
        now.saturating_sub(STANDING_OBJECTS_NS),
        now.saturating_add(1),
    );
    let (start_ns, body) = found.last()?;
    let mut marks = vision::Overlays::new(cam, [Group::Objects]);
    marks.apply("camera.objects", *start_ns, body)
}

/// How far back a `frame` request looks for its frame, widening: a camera's frame is a few tens of milliseconds
/// back, a GOP's keyframe up to its length (a few seconds).
const FRAME_LOOK_BACK_US: [u64; 3] = [500_000, 3_000_000, 15_000_000];

/// Copies a Tier 0 slot's bytes under a short lease of this reader's own (the broker pinned it for the query, and
/// the lease catches a slot reused since). `None` when the slot is gone.
fn read_slot(arena: &continuum_shm::Arena, handle: SlotHandle) -> Option<Vec<u8>> {
    for _ in 0..2 {
        let lease = arena.lease(handle, std::time::Duration::from_secs(1)).ok()?;
        let mut bytes = vec![0; lease.len()];
        match lease.copy_to(&mut bytes) {
            Ok(()) => return Some(bytes),
            Err(continuum_shm::Error::Expired) => {}
            Err(_) => return None,
        }
    }
    None
}

/// A connection's live following: at most one feed at a time, read by a task into a bounded queue the transport
/// drains. When the page is slower than the stream the oldest queued frame is dropped (counted), so the page sees the
/// newest decodable frames and a slow page never stalls the subscription. Video loss suppresses dependent
/// pictures until the next actual random-access picture; overlays have a separate bounded queue.
#[derive(Default)]
pub struct LiveRun {
    task: Option<tokio::task::JoinHandle<()>>,
    queue: Option<Arc<LiveQueue>>,
}

impl Drop for LiveRun {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

/// The frames waiting for the transport, with the counts the `stopped` reply reports.
#[derive(Default)]
pub(crate) struct LiveQueue {
    frames: Mutex<QueuedFrames>,
    ready: tokio::sync::Notify,
    /// Frames handed to the transport.
    pub(crate) sent: std::sync::atomic::AtomicU64,
    /// Frames dropped because the page was slower than the stream.
    pub(crate) dropped: std::sync::atomic::AtomicU64,
    /// Events the feed left out.
    skipped: std::sync::atomic::AtomicU64,
    /// The feed ended (its `stopped` is the last thing queued).
    ended: std::sync::atomic::AtomicBool,
}

struct QueuedFrames {
    video: std::collections::VecDeque<Outgoing>,
    other: std::collections::VecDeque<Outgoing>,
    waiting: bool,
    overlay_turn: bool,
}
impl Default for QueuedFrames {
    fn default() -> Self {
        Self {
            video: std::collections::VecDeque::new(),
            other: std::collections::VecDeque::new(),
            waiting: true,
            overlay_turn: false,
        }
    }
}

/// How many frames the live queue holds before dropping the oldest; a few frames of a 30 fps stream.
const LIVE_QUEUE: usize = 8;

impl LiveQueue {
    /// Video loss invalidates the queued reference chain. Overlays have their own independent cap.
    pub(crate) fn push(&self, out: Outgoing) {
        {
            let mut q = self.frames.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let header = match &out {
                Outgoing::Binary(b) => crate::header::split(b).ok().map(|(h, _)| h),
                _ => None,
            };
            let video = header.is_some_and(|h| matches!(h.payload_type, PayloadType::Video | PayloadType::Jpeg));
            if video {
                let h = header.expect("video header");
                let dependent = h.payload_type == PayloadType::Video;
                if q.video.len() >= LIVE_QUEUE {
                    if dependent {
                        let lost = q.video.len();
                        q.video.clear();
                        q.waiting = true;
                        self.dropped.fetch_add(lost as u64, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        q.video.pop_front();
                        self.dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                if dependent && h.keyframe {
                    q.waiting = false;
                }
                if dependent && q.waiting {
                    self.dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                } else {
                    q.video.push_back(out);
                }
            } else {
                q.other.push_back(out);
                if q.other.len() > LIVE_QUEUE {
                    q.other.pop_front();
                    self.dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        self.ready.notify_one();
    }

    pub(crate) fn pop(&self) -> Option<Outgoing> {
        let mut q = self.frames.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // Overlay traffic cannot evict video references; video is drained before its final stopped control reply.
        if q.overlay_turn && matches!(q.other.front(), Some(Outgoing::Binary(_))) {
            q.overlay_turn = false;
            return q.other.pop_front();
        }
        if let Some(video) = q.video.pop_front() {
            q.overlay_turn = true;
            return Some(video);
        }
        q.other.pop_front()
    }

    fn counts(&self) -> (u64, u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.sent.load(Relaxed), self.dropped.load(Relaxed), self.skipped.load(Relaxed))
    }

    fn stopped(&self, reason: Option<String>) -> Reply {
        let (frames, dropped, skipped) = self.counts();
        Reply::Stopped { frames, dropped, skipped, reason }
    }
}

impl LiveRun {
    /// Starts following and answers `live`, or an `error` (a feed already running ends first, but only once the new
    /// one has opened, so a failed request leaves the old feed in place).
    pub async fn start(&mut self, session: &Session, stream: u32, overlays: &[String]) -> Vec<Outgoing> {
        let mut feed = match session.start_live(stream, overlays).await {
            Ok(feed) => feed,
            Err(message) => return vec![Outgoing::Text(Reply::Error { message }.to_json())],
        };
        let mut out = self.stop(Some("replaced by a new live request".to_owned())).await;
        out.push(Outgoing::Text(feed.opened().to_json()));
        out.extend(feed.standing.take());
        let queue = Arc::new(LiveQueue::default());
        let shared = Arc::clone(&queue);
        self.task = Some(tokio::spawn(async move {
            use std::sync::atomic::Ordering::Relaxed;
            while let Some(frame) = feed.next().await {
                shared.skipped.store(feed.skipped, Relaxed);
                shared.push(frame);
            }
            // The subscription ended under us (the broker connection ended): the page hears it last.
            shared.skipped.store(feed.skipped, Relaxed);
            let reply = shared.stopped(Some("the broker connection ended".to_owned()));
            shared.ended.store(true, Relaxed);
            shared.push(Outgoing::Text(reply.to_json()));
        }));
        self.queue = Some(queue);
        out
    }

    /// Ends the feed, if one runs, and answers `stopped` (nothing when none ran).
    pub async fn stop(&mut self, reason: Option<String>) -> Vec<Outgoing> {
        let Some(task) = self.task.take() else { return Vec::new() };
        task.abort();
        let _ = task.await;
        let Some(queue) = self.queue.take() else { return Vec::new() };
        if queue.ended.load(std::sync::atomic::Ordering::Relaxed) {
            // The feed had ended on its own and said so already.
            return Vec::new();
        }
        vec![Outgoing::Text(queue.stopped(reason).to_json())]
    }

    /// The next queued frame, or pending forever when nothing is live (for a `select!` beside the control stream).
    pub async fn next(&mut self) -> Option<Outgoing> {
        loop {
            let Some(queue) = &self.queue else { return std::future::pending().await };
            if let Some(out) = queue.pop() {
                if matches!(out, Outgoing::Binary(_)) {
                    queue.sent.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return Some(out);
            }
            if queue.ended.load(std::sync::atomic::Ordering::Relaxed) {
                // The feed's `stopped` went out already: nothing more comes.
                self.queue = None;
                self.task = None;
                return None;
            }
            queue.ready.notified().await;
        }
    }
}

#[cfg(test)]
mod live_queue_tests {
    use super::*;

    fn video(at: u64, key: bool) -> Outgoing {
        Outgoing::Binary(FrameHeader::at_ns(8, PayloadType::Video, key, at).frame(&[
            0,
            0,
            1,
            if key { 0x65 } else { 0x41 },
            0x88,
        ]))
    }

    #[test]
    fn video_queue_overflow_discards_deltas_until_next_idr() {
        let q = LiveQueue::default();
        q.push(video(0, true));
        for at in 1..=LIVE_QUEUE {
            q.push(video(at as u64, false));
        }
        assert!(q.pop().is_none(), "no broken reference chain is sent");
        q.push(video(10, false));
        assert!(q.pop().is_none());
        q.push(video(11, true));
        q.push(video(9, false)); // PTS regression is valid decode order.
        assert_eq!(q.pop(), Some(video(11, true)));
        assert_eq!(q.pop(), Some(video(9, false)));
        assert_eq!(q.counts().1, LIVE_QUEUE as u64 + 2);
    }

    #[test]
    fn overlay_flood_is_bounded_and_does_not_evict_video_references() {
        let q = LiveQueue::default();
        q.push(video(0, true));
        q.push(video(1, false));
        for at in 0..100 {
            q.push(Outgoing::Binary(FrameHeader::at_ns(8, PayloadType::Overlay, false, at).frame(b"overlay")));
        }
        assert_eq!(q.pop(), Some(video(0, true)));
        assert!(
            matches!(q.pop(),Some(Outgoing::Binary(ref bytes)) if crate::header::split(bytes).expect("header").0.payload_type==PayloadType::Overlay)
        );
        assert_eq!(q.pop(), Some(video(1, false)));
        assert_eq!(q.frames.lock().expect("queue").other.len(), LIVE_QUEUE - 1);
    }

    #[test]
    fn native_sequence_gap_requires_idr_and_allows_reconnect_reset_at_idr() {
        let mut c = VideoContinuity::default();
        assert!(!c.accept(Some(1), false));
        assert!(c.accept(Some(2), true));
        assert!(c.accept(Some(3), false));
        assert!(!c.accept(Some(5), false));
        assert!(!c.accept(Some(6), false));
        assert!(c.accept(Some(1), true));
        assert!(c.accept(Some(2), false));
        c.broken();
        assert!(!c.accept(Some(3), false));
    }

    #[test]
    fn spans_join_when_they_overlap_or_the_gap_is_small_enough() {
        let got = merge_spans([(10, 20), (0, 5), (21, 30), (40, 40), (50, 60)], 2);
        assert_eq!(got, [(0, 5), (10, 30), (40, 41), (50, 60)], "an instant covers one microsecond");
        assert_eq!(merge_spans([(0, 5), (8, 9)], 3), [(0, 9)]);
        assert!(merge_spans([], 1).is_empty());
    }

    #[test]
    fn a_full_live_queue_drops_the_oldest_and_counts_it() {
        let q = LiveQueue::default();
        for i in 0..(LIVE_QUEUE as u8 + 3) {
            q.push(Outgoing::Binary(vec![i]));
        }
        assert_eq!(q.counts(), (0, 3, 0), "three dropped, none sent yet");
        let first = q.pop().expect("a frame");
        assert!(matches!(first, Outgoing::Binary(b) if b == vec![3]), "the oldest three are gone");
        let mut n = 1;
        while q.pop().is_some() {
            n += 1;
        }
        assert_eq!(n, LIVE_QUEUE, "the newest frames stayed");
    }
}

#[cfg(test)]
mod tests {
    use continuum_core::StreamId;
    use continuum_tier::{ChunkFormat, ChunkWriter};

    use super::*;
    use crate::header::split;

    /// Where a test's chunk `name` of `stream` created at `created_ns` goes (`continuum_tier::layout`), its hour made.
    fn in_hour(dir: &std::path::Path, stream: u32, created_ns: u64, name: &str) -> std::path::PathBuf {
        let hour = continuum_tier::layout::hour_dir(dir, StreamId(stream), created_ns);
        continuum_tier::layout::make_hour_dir(dir, &hour).expect("hour");
        hour.join(name)
    }

    fn session(dir: &std::path::Path) -> Session {
        let mut w = ChunkWriter::create(&in_hour(dir, 5, 1, "5-1-1.cnk"), StreamId(5), ChunkFormat::Passthrough, 1)
            .expect("create");
        for i in 0..4u32 {
            let t = 1_000_000 + u64::from(i) * 33_000_000;
            w.append(t, t, i, &[0xff, 0xd8, i as u8]).expect("append");
        }
        w.close().expect("close");
        Session::new(
            Arc::new(ChunkSource::new(dir)),
            Some(WebTransportEndpoint { url: "https://h:1".to_owned(), hash: Arc::new(RwLock::new("ab".to_owned())) }),
            OriginPolicy::default(),
        )
    }

    #[test]
    fn without_a_broker_spans_come_from_the_chunk_files_clipped_to_the_window() {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = session(dir.path());
        let spans = |json: &str| match s.handle(json).pop() {
            Some(Outgoing::Text(t)) => match serde_json::from_str::<Reply>(&t) {
                Ok(Reply::Spans { spans, .. }) => spans,
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        };
        // Four frames from 1 ms to 100 ms: one stretch, ending just past the last instant.
        assert_eq!(spans(r#"{"type":"spans","stream":5,"start_us":0,"end_us":1000000}"#), [(1_000, 100_001)]);
        assert_eq!(spans(r#"{"type":"spans","stream":5,"start_us":50000,"end_us":60000}"#), [(50_000, 60_000)]);
        assert!(spans(r#"{"type":"spans","stream":5,"start_us":200000,"end_us":300000}"#).is_empty());
        assert!(spans(r#"{"type":"spans","stream":6,"start_us":0,"end_us":1000000}"#).is_empty());
    }

    #[test]
    fn a_frame_request_sends_the_one_frame_showing_at_a_time() {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = session(dir.path());
        // Frames at 1, 34, 67 ms...: at 50 ms the frame showing is the one at 34 ms, alone.
        let out = s.handle(r#"{"type":"frame","stream":5,"at_us":50000}"#);
        assert_eq!(out.len(), 2, "{out:?}");
        let Outgoing::Binary(f) = &out[0] else { panic!("a frame") };
        let (h, _) = split(f).expect("frame");
        assert_eq!((h.payload_type, h.capture_us), (PayloadType::Jpeg, 34_000));
        let Outgoing::Text(reply) = &out[1] else { panic!("the reply") };
        assert!(reply.contains(r#""type":"frame","stream":5,"at_us":50000,"capture_us":34000"#), "{reply}");
        // Before the first frame there's none to show.
        let none = s.handle(r#"{"type":"frame","stream":5,"at_us":500}"#);
        assert!(matches!(&none[..], [Outgoing::Text(t)] if t.contains(r#""capture_us":null"#)), "{none:?}");
        // A stream with no frames there has none to show either.
        let other = s.handle(r#"{"type":"frame","stream":99,"at_us":50000}"#);
        assert!(matches!(&other[..], [Outgoing::Text(t)] if t.contains(r#""capture_us":null"#)), "{other:?}");
    }

    #[test]
    fn a_raw_uyvy_frame_goes_out_as_a_half_size_jpeg() {
        let (w, h) = (64u32, 48u32);
        let uyvy: Vec<u8> = (0..w * h / 2).flat_map(|_| [128u8, 120, 128, 120]).collect();
        let jpeg = raw_jpeg("UYVY", w, h, w * 2, &uyvy).expect("a JPEG");
        let picture = continuum_tier::hydrate::decode_jpeg(&jpeg).unwrap();
        assert_eq!((picture.width, picture.height), (32, 24));
        assert!(raw_jpeg("UYVY", w, h, w * 2, &uyvy[..10]).is_none(), "a short frame is left out");
    }

    #[test]
    fn exact_frame_requests_keep_native_sizes_across_consecutive_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let sizes = [(64u16, 48u16), (32, 24)];
        for (i, (width, height)) in sizes.into_iter().enumerate() {
            let at = 1_000_000 + u64::try_from(i).unwrap() * 33_333_333;
            let path = in_hour(dir.path(), 5, at, &format!("5-{at}-{i}.cnk"));
            let mut chunk = ChunkWriter::create(&path, StreamId(5), ChunkFormat::Passthrough, at).unwrap();
            let mut bytes = Vec::new();
            jpeg_encoder::Encoder::new(&mut bytes, 90)
                .encode(
                    &vec![96; usize::from(width) * usize::from(height) * 3],
                    width,
                    height,
                    jpeg_encoder::ColorType::Rgb,
                )
                .unwrap();
            chunk.append(at, at, u32::try_from(i).unwrap(), &bytes).unwrap();
            chunk.close().unwrap();
        }
        let session = Session::new(Arc::new(ChunkSource::new(dir.path())), None, OriginPolicy::default());
        for (i, (width, height)) in sizes.into_iter().enumerate() {
            let at = (1_000_000 + u64::try_from(i).unwrap() * 33_333_333) / 1000;
            let out = session.handle(&format!(r#"{{"type":"frame","stream":5,"at_us":{at}}}"#));
            let Outgoing::Binary(bytes) = &out[0] else { panic!("frame reply: {out:?}") };
            let (header, jpeg) = split(bytes).unwrap();
            assert_eq!(header.payload_type, PayloadType::Jpeg);
            assert_eq!(header.capture_us, at);
            let picture = continuum_tier::hydrate::decode_jpeg(jpeg).unwrap();
            assert_eq!((picture.width, picture.height), (u32::from(width), u32::from(height)));
        }
    }

    #[test]
    fn follow_on_keeps_new_gops_whole_even_with_older_reordered_pictures() {
        let frame = |us, keyframe, payload_type| Frame {
            header: FrameHeader { stream_id: 5, payload_type, keyframe, flip: false, live: false, capture_us: us },
            bytes: vec![1],
            codec: "avc",
            media: None,
        };
        let frames = vec![
            frame(100, true, PayloadType::Video),
            frame(140, false, PayloadType::Video),
            frame(120, false, PayloadType::Video),
            frame(130, true, PayloadType::Video),
            frame(180, false, PayloadType::Video),
            frame(160, false, PayloadType::Video),
            frame(140, true, PayloadType::Jpeg),
            frame(190, true, PayloadType::Jpeg),
        ];
        let kept = unreceived_frames(frames, 140);
        assert_eq!(kept.iter().map(|f| f.header.capture_us).collect::<Vec<_>>(), [130, 180, 160, 190]);
        assert!(kept[0].header.keyframe, "a newer GOP keeps its older IDR prefix");
    }

    #[test]
    fn follow_on_excludes_duration_overlap_from_a_completed_archived_gop() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            ChunkWriter::create(&in_hour(dir.path(), 5, 0, "5-0-1.cnk"), StreamId(5), ChunkFormat::H264Au, 0).unwrap();
        for (seq, (pts, key)) in [
            (100_000_000, true),
            (300_000_000, false),
            (200_000_000, false),
            (400_000_000, true),
            (600_000_000, false),
            (500_000_000, false),
            (700_000_000, true),
        ]
        .into_iter()
        .enumerate()
        {
            let bytes = if key { vec![0, 0, 1, 0x67, 0x4d, 0, 0, 1, 0x65, 0x88] } else { vec![0, 0, 1, 0x41, 0x88] };
            let unit = continuum_tier::gop::Unit { start_ns: pts, end_ns: pts + 40_000_000, key, bytes };
            writer.append(pts, pts + 40_000_000, seq as u32, &continuum_tier::gop::encode(&[unit]).unwrap()).unwrap();
        }
        writer.close().unwrap();
        let session = Session::new(Arc::new(ChunkSource::new(dir.path())), None, OriginPolicy::default());
        let times = |out: Vec<Outgoing>| {
            out.into_iter()
                .filter_map(|m| match m {
                    Outgoing::Binary(b) => Some(split(&b).unwrap().0.capture_us),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let follow =
            times(session.handle(r#"{"type":"range","stream":5,"start_us":300001,"end_us":650000,"after_us":300000}"#));
        assert_eq!(follow, [400_000, 600_000, 500_000]);
        let seek = times(session.handle(r#"{"type":"range","stream":5,"start_us":300001,"end_us":650000}"#));
        assert_eq!(seek, [100_000, 300_000, 200_000, 400_000, 600_000, 500_000]);
    }

    #[test]
    fn follow_on_independent_records_keep_the_bounded_reads_next_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let s = session(dir.path());
        let out =
            s.handle(r#"{"type":"range","stream":5,"start_us":0,"end_us":200000,"max_frames":3,"after_us":67000}"#);
        assert_eq!(out.len(), 1);
        assert!(matches!(&out[0], Outgoing::Text(t) if t.contains("\"frames\":0") && t.contains("\"next_us\":100000")));
        let out = s.handle(r#"{"type":"range","stream":5,"start_us":100000,"end_us":200000,"after_us":67000}"#);
        assert!(matches!(&out[0], Outgoing::Binary(b) if split(b).unwrap().0.capture_us == 100000));
    }

    #[test]
    fn a_range_sends_frames_then_done() {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = session(dir.path());
        assert!(
            matches!(s.hello("websocket"), Outgoing::Text(t) if t.contains("\"webtransport\":{\"url\":\"https://h:1\",\"cert_hash\":\"ab\""))
        );
        let out = s.handle(r#"{"type":"range","stream":5,"start_us":0,"end_us":200000,"max_frames":3}"#);
        assert_eq!(out.len(), 4, "3 frames, done");
        let mut times = Vec::new();
        for f in &out[..3] {
            let Outgoing::Binary(f) = f else { panic!("binary") };
            let (fh, _) = split(f).expect("frame");
            assert_eq!(fh.payload_type, PayloadType::Jpeg);
            times.push(fh.capture_us);
        }
        assert_eq!(times, vec![1_000, 34_000, 67_000]);
        let Outgoing::Text(done) = &out[3] else { panic!("done") };
        assert!(done.contains(r#""frames":3,"skipped":0,"next_us":100000,"codecs":["jpeg"]"#), "{done}");

        let out = s.handle(r#"{"type":"range","stream":5,"start_us":5,"end_us":5}"#);
        assert!(matches!(&out[0], Outgoing::Text(t) if t.contains("end_us must be after")));
        let out = s.handle(r#"{"type":"range","stream":5,"start_us":0,"end_us":9,"overlays":["real"]}"#);
        assert!(matches!(&out[0], Outgoing::Text(t) if t.contains("unknown overlay group")));
        assert!(
            matches!(&s.handle(r#"{"type":"live","stream":5}"#)[0], Outgoing::Text(t) if t.contains("live sessions"))
        );
        assert!(matches!(&s.handle("{}")[0], Outgoing::Text(t) if t.contains("bad request")));
        assert_eq!(s.handle(r#"{"type":"ping"}"#), vec![Outgoing::Text(r#"{"type":"pong"}"#.to_owned())]);
        assert!(matches!(&s.handle(r#"{"type":"streams"}"#)[0], Outgoing::Text(t) if t.contains(r#""id":5"#)));
    }
}

// A failed atomic lookup cannot authorize reading the earlier query's physical locator. In particular, a full
// purge between Query and GetChunk must stop playback rather than bypassing catalog cuts with unknown metadata.
fn authoritative_chunk(
    result: Result<continuum_client::ChunkInfo, continuum_client::Error>,
) -> Result<continuum_client::ChunkInfo, String> {
    result.map_err(|error| format!("authoritative chunk snapshot unavailable: {error}"))
}

#[cfg(test)]
mod snapshot_failure_tests {
    use super::*;
    #[test]
    fn purge_not_found_and_connection_failure_never_authorize_a_physical_read() {
        for error in [
            continuum_client::Error::Broker {
                code: continuum_core::schema::continuum::ErrorCode::NotFound,
                message: "purged after query".into(),
            },
            continuum_client::Error::Closed,
        ] {
            assert!(authoritative_chunk(Err(error)).is_err());
        }
    }
}
