//! The client's control messages and the gateway's replies: small JSON objects with a `type`, sent as text frames
//! over WebSockets and as length-prefixed messages on the control stream over WebTransport.
//!
//! | From | `type` | Fields | Purpose |
//! |---|---|---|---|
//! | gateway | `hello` | `features` (`watch`, `req`: what this gateway answers, so a client can fall back on an older one), `transport`, `webtransport` (`{url, cert_hash}` when it's on), `broker` (the broker is connected), `overlay_groups` (the vision overlays a page can ask for: `faces`, `bodies`, `poses`, `objects`) | First message of a session |
//! | client | `streams` | | List the streams: the broker's registrations merged with the recorded ones |
//! | gateway | `streams` | `streams: [{id, name, live, format, start_us, end_us, chunks, frames}]` | The list |
//! | client | `range` | `stream`, `start_us`, `end_us`, `max_frames` (optional), `after_us` (optional completed-playback frontier), `overlays` (overlay groups, like `["faces", "poses"]`, each frame's sent after it) | Send the frames whose start lies in the range, in time order, then `done` |
//! | gateway | `done` | `stream`, `start_us`, `end_us`, `frames`, `skipped`, `next_us` (set when the range was cut at `max_frames`), `codecs` | The range is sent |
//! | client | `frame` | `stream`, `at_us` | The one frame showing at a time, for scrubbing: the newest frame at or before it (a JPEG), or the keyframe at or before it (H.264, HEVC, AV1), so it decodes alone |
//! | gateway | `frame` | `stream`, `at_us`, `capture_us` (the frame's time, or null when there's none within 15 s before) | The frame was sent just before (when there is one) |
//! | client | `spans` | `stream`, `start_us`, `end_us`, `gap_us` (optional: 2 s by default, at most 60 s) | Which stretches of the window have frames, so a page can draw a timeline and play only what's there |
//! | gateway | `spans` | `stream`, `start_us`, `end_us`, `spans: [[start_us, end_us], ...]` | The covered stretches, in order, merged where the gap between them is at most `gap_us` |
//! | client | `live` | `stream`, `overlays` (overlay groups, or names of streams already in the overlay shape; optional) | Follow a stream live with its overlays, until `stop`, another `live`, or the connection ends (needs the broker) |
//! | gateway | `live` | `stream`, `overlays: [{name, id}]` | Following; frames and overlays follow |
//! | client | `stop` | | Stop following live |
//! | gateway | `stopped` | `frames`, `dropped`, `skipped`, `reason` (when the gateway ended it) | Live following ended |
//! | client | `watch` | `streams` (each an id, or `{id, max_fps, overlays}`), `max_fps` (the default for the bare ids: 1, at most 10; 0 is every frame), `media_context` | Follow many streams on one connection, beside any `live`. A stream at a rate gets its newest frame that decodes alone (a JPEG frame, or a video keyframe) at most that many times a second, newest only; one at 0 gets every frame, whole GOPs, as `live` sends them, with its overlays. Each stream's newest frame (within 15 s) goes out at once. A new `watch` replaces the set |
//! | gateway | `watching` | `streams: [{id, max_fps, overlays: [{name, id}]}]` | Following; frames carry the live and watch flags |
//! | client | `unwatch` | | Stop the watch set |
//! | gateway | `unwatched` | `frames`, `dropped`, `skipped`, `reason` (when the gateway ended it) | The watch set ended |
//! | client | `cancel` | `req` | Drop what a request with that `req` hasn't sent yet |
//! | gateway | `cancelled` | `req` | It was dropped (nothing comes when it had finished) |
//! | client | `ping` | | Liveness |
//! | gateway | `pong` | | |
//! | gateway | `error` | `message` | A request was bad or failed |
//!
//! **Request ids**: `range`, `frame`, and `spans` take an optional `req` (a u32 the client picks). A request
//! with one runs beside the connection's others instead of after them; every reply it causes (`done`, `frame`,
//! `spans`, `media`, `error`) carries the same `req`, and its binary frames the header's request flag with the id after
//! the header ([`crate::header::with_request`]), so overlapping seeks share one connection and `cancel` ends the one
//! given up. A request without `req` is answered in order, untagged, as before.
//!
//! The browser paces playback itself from each frame's `capture_us`, so a range is sent as fast as the transport
//! takes it, capped at `max_frames` (the gateway's limit at most) and at about 64 MiB, never cut between frames
//! that share a start microsecond, so `next_us` always moves forward: a player asks for the next window as it plays
//! and for the window around a scrub position.

use serde::{Deserialize, Serialize};

use crate::source::StreamSummary;

/// A request from the client.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// List the streams with recorded frames. With `req` it runs beside the connection's other work, as a range does:
    /// listing reads chunk files and can take seconds, and an untagged request holds the connection's frames until
    /// it's answered, long enough for a watched stream's queue to overflow and wait for a keyframe.
    Streams {
        /// Tags the reply, and runs the listing as a request of its own.
        #[serde(default)]
        req: Option<u32>,
    },
    /// Send the frames of a time range.
    Range {
        /// The stream.
        stream: u32,
        /// From, microseconds UTC.
        start_us: u64,
        /// To, exclusive.
        end_us: u64,
        /// Highest capture timestamp already received in complete GOPs. Omit for seeks.
        /// Whole predictive GOPs are suppressed only when all their frames are at or before it.
        #[serde(default)]
        after_us: Option<u64>,
        /// At most this many frames (the gateway caps it).
        #[serde(default)]
        max_frames: Option<usize>,
        /// Overlay groups (`faces`, `bodies`, `poses`, `objects`).
        #[serde(default)]
        overlays: Option<Vec<String>>,
        /// Include explicitly identified playback metadata.
        #[serde(default)]
        media_context: bool,
        /// The client's id for this request: answered beside the others, its replies and frames tagged.
        #[serde(default)]
        req: Option<u32>,
    },
    /// The one frame showing at a time, alone (scrubbing).
    Frame {
        /// The stream.
        stream: u32,
        /// The time, microseconds UTC.
        at_us: u64,
        /// Include explicitly identified playback metadata.
        #[serde(default)]
        media_context: bool,
        /// The client's id for this request.
        #[serde(default)]
        req: Option<u32>,
    },
    /// The stretches of a window that have frames.
    Spans {
        /// The stream.
        stream: u32,
        /// From, microseconds UTC.
        start_us: u64,
        /// To, exclusive.
        end_us: u64,
        /// Stretches this close are one (2 s by default, at most 60 s).
        #[serde(default)]
        gap_us: Option<u64>,
        /// Allow a partial list, marked `partial`, instead of an error when the window holds more items than one
        /// reply lists.
        #[serde(default)]
        media_context: bool,
        /// The client's id for this request.
        #[serde(default)]
        req: Option<u32>,
    },
    /// Follow a stream live, with overlay streams drawn on its frames, until `stop`, another `live`, or the end of
    /// the connection. Frames the connection can't take fast enough are dropped, newest kept.
    Live {
        /// The stream.
        stream: u32,
        /// Overlay streams to follow with it, by name (payloads in the `Overlay` FlatBuffer pass through; other
        /// shapes need a converter the gateway knows).
        #[serde(default)]
        overlays: Vec<String>,
    },
    /// Stop following live.
    Stop,
    /// Follow many streams on one connection, beside any `live`. Replaces the watch set already running.
    Watch {
        /// The streams: a bare id at the request's `max_fps`, or one with its own rate and overlays.
        streams: Vec<Watched>,
        /// The rate for the bare ids, frames a second (1 by default; 0 is every frame; the gateway caps it).
        #[serde(default)]
        max_fps: Option<f64>,
        /// Include explicitly identified playback metadata, as `range` and `frame` do.
        #[serde(default)]
        media_context: bool,
    },
    /// Stop the watch set.
    Unwatch,
    /// Drop what the request with this id hasn't sent yet.
    Cancel {
        /// The request's id.
        req: u32,
    },
    /// Liveness.
    Ping,
}

/// One stream of a `watch`: a bare id, or one with its own rate and overlays.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Watched {
    /// At the request's rate, with no overlays.
    Id(u32),
    /// At its own rate (0: every frame, whole GOPs), with overlays (as `live` takes them).
    Stream {
        /// The stream.
        id: u32,
        /// Frames a second, at most (the request's when left out).
        #[serde(default)]
        max_fps: Option<f64>,
        /// Overlay groups or overlay stream names, as `live` takes them.
        #[serde(default)]
        overlays: Vec<String>,
    },
}

/// One stream a `watching` follows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WatchedStream {
    /// The stream.
    pub id: u32,
    /// Frames a second, at most; 0 is every frame.
    pub max_fps: f64,
    /// The overlay streams followed with it.
    pub overlays: Vec<LiveOverlay>,
}

/// An overlay stream a `live` follows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveOverlay {
    /// Its name.
    pub name: String,
    /// Its stream id, which its frames carry.
    pub id: u32,
}

/// What a WebTransport client needs to connect.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebTransportInfo {
    /// The `https://host:port` URL.
    pub url: String,
    /// The certificate's SHA-256, lower-case hex, for `serverCertificateHashes` (self-signed only; empty for a
    /// certificate the browser trusts on its own).
    pub cert_hash: String,
}

/// A reply from the gateway.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Reply {
    /// The first message of a session.
    Hello {
        /// What this gateway answers beyond the basics (`watch`, `req`), so a client can fall back on an older one.
        #[serde(default)]
        features: Vec<String>,
        /// `websocket` or `webtransport`.
        transport: String,
        /// How to reach the WebTransport endpoint, when it's on.
        webtransport: Option<WebTransportInfo>,
        /// The broker is in the loop: streams and ranges come from it (else from the chunk directory alone).
        #[serde(default)]
        broker: bool,
        /// The overlay groups a page can ask for (with the broker).
        #[serde(default)]
        overlay_groups: Vec<String>,
        /// Supports per-request media contexts and partial span lists.
        #[serde(default)]
        media_context: bool,
    },
    /// The streams with recorded frames.
    Streams {
        /// By id.
        streams: Vec<StreamSummary>,
    },
    /// A range was sent.
    Done {
        /// The stream.
        stream: u32,
        /// The range asked for.
        start_us: u64,
        /// Its end.
        end_us: u64,
        /// Frames sent.
        frames: usize,
        /// Records the browser can't decode, left out.
        skipped: usize,
        /// Where to continue when the range was cut at `max_frames`.
        next_us: Option<u64>,
        /// The codecs seen (`jpeg`, `avc`, `hevc`, `av1`, `arrow`), so the page can configure its decoder.
        codecs: Vec<String>,
    },
    /// The frame a `frame` request asked for was sent (or there was none).
    Frame {
        /// The stream.
        stream: u32,
        /// The time asked for.
        at_us: u64,
        /// The frame's capture time, or None when no frame was found.
        capture_us: Option<u64>,
    },
    /// The stretches of a window that have frames.
    Spans {
        /// The stream.
        stream: u32,
        /// The window asked about.
        start_us: u64,
        /// Its end.
        end_us: u64,
        /// The covered stretches, `[start_us, end_us)` each, in order.
        spans: Vec<(u64, u64)>,
        /// The stretches whose video expired (aged out past the history window), `[start_us, end_us)` each,
        /// in order: a reader says the footage expired there instead of that nothing was recorded.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        expired: Vec<(u64, u64)>,
        /// The window held more items than one reply lists, so `spans` is incomplete: ask again over smaller
        /// windows. Only set when the request allowed it (`media_context`).
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        partial: bool,
    },
    /// Following live: frames follow until `stopped`.
    Live {
        /// The stream followed.
        stream: u32,
        /// The overlay streams followed with it.
        overlays: Vec<LiveOverlay>,
    },
    /// Live following ended (a `stop`, a new `live`, or the broker ending the subscription).
    Stopped {
        /// Frames and overlays sent.
        frames: u64,
        /// Frames and overlays dropped because the connection was slower than the stream (the oldest go first).
        dropped: u64,
        /// Events left out: no bytes, an overlay shape with no converter, an unreadable payload.
        skipped: u64,
        /// Why, when the gateway ended it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Following a watch set: frames follow until `unwatched`.
    Watching {
        /// The streams followed.
        streams: Vec<WatchedStream>,
    },
    /// The watch set ended (an `unwatch`, a new `watch`, or the broker connection ending).
    Unwatched {
        /// Frames and overlays sent.
        frames: u64,
        /// Frames replaced by a newer one, or dropped because the connection was slower than the stream.
        dropped: u64,
        /// Frames left out: ones that don't decode alone (at a rate), and events with no bytes.
        skipped: u64,
        /// Why, when the gateway ended it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// A request's unsent replies were dropped.
    Cancelled {
        /// The request's id.
        req: u32,
    },
    /// Liveness.
    Pong,
    /// A request was bad or failed.
    Error {
        /// Why.
        message: String,
    },
}

impl Reply {
    /// The reply as JSON text.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("a reply serializes")
    }
}

/// Parses a request.
///
/// # Errors
///
/// The JSON error's text, for an `error` reply.
pub fn parse(text: &str) -> Result<Request, String> {
    serde_json::from_str(text).map_err(|e| format!("bad request: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_parse_and_replies_serialize() {
        assert_eq!(parse(r#"{"type":"streams"}"#), Ok(Request::Streams { req: None }));
        assert_eq!(parse(r#"{"type":"streams","req":9}"#), Ok(Request::Streams { req: Some(9) }));
        assert_eq!(
            parse(r#"{"type":"range","stream":3,"start_us":10,"end_us":20}"#),
            Ok(Request::Range {
                stream: 3,
                start_us: 10,
                end_us: 20,
                after_us: None,
                max_frames: None,
                overlays: None,
                media_context: false,
                req: None,
            })
        );
        assert!(matches!(
            parse(r#"{"type":"range","stream":3,"start_us":10,"end_us":20,"overlays":["faces","poses"]}"#),
            Ok(Request::Range { overlays: Some(g), .. }) if g == ["faces", "poses"]
        ));
        assert!(matches!(
            parse(r#"{"type":"range","stream":3,"start_us":10,"end_us":20,"after_us":9}"#),
            Ok(Request::Range { after_us: Some(9), .. })
        ));
        assert_eq!(parse(r#"{"type":"live","stream":1}"#), Ok(Request::Live { stream: 1, overlays: vec![] }));
        assert_eq!(
            parse(r#"{"type":"frame","stream":2,"at_us":7}"#),
            Ok(Request::Frame { stream: 2, at_us: 7, media_context: false, req: None })
        );
        assert!(matches!(
            parse(r#"{"type":"frame","stream":2,"at_us":7,"req":42}"#),
            Ok(Request::Frame { req: Some(42), .. })
        ));
        assert!(matches!(
            parse(r#"{"type":"spans","stream":2,"start_us":1,"end_us":9,"req":3}"#),
            Ok(Request::Spans { req: Some(3), .. })
        ));
        assert_eq!(parse(r#"{"type":"cancel","req":42}"#), Ok(Request::Cancel { req: 42 }));
        assert_eq!(parse(r#"{"type":"unwatch"}"#), Ok(Request::Unwatch));
        assert_eq!(
            parse(r#"{"type":"watch","streams":[1,{"id":2,"max_fps":0,"overlays":["faces"]}],"max_fps":2}"#),
            Ok(Request::Watch {
                streams: vec![
                    Watched::Id(1),
                    Watched::Stream { id: 2, max_fps: Some(0.0), overlays: vec!["faces".to_owned()] },
                ],
                max_fps: Some(2.0),
                media_context: false,
            })
        );
        assert_eq!(parse(r#"{"type":"stop"}"#), Ok(Request::Stop));
        assert!(parse(r#"{"type":"dance"}"#).expect_err("unknown").starts_with("bad request"));
        assert!(parse("nope").is_err());
        let json = Reply::Done {
            stream: 1,
            start_us: 0,
            end_us: 5,
            frames: 2,
            skipped: 0,
            next_us: Some(4),
            codecs: vec!["jpeg".to_owned()],
        }
        .to_json();
        assert_eq!(
            json,
            r#"{"type":"done","stream":1,"start_us":0,"end_us":5,"frames":2,"skipped":0,"next_us":4,"codecs":["jpeg"]}"#
        );
        assert_eq!(Reply::Pong.to_json(), r#"{"type":"pong"}"#);
        let back: Reply = serde_json::from_str(&json).expect("reply");
        assert!(matches!(back, Reply::Done { frames: 2, .. }));
    }
}
