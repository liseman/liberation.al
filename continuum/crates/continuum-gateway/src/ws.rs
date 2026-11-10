//! The HTTP side: the WebSocket endpoint (the default transport), a health check, the stream list as JSON, and the
//! WebTransport details a page needs to switch transports.
//!
//! | Route | Purpose |
//! |---|---|
//! | `GET /healthz` | `ok` |
//! | `GET /streams` | The streams with recorded frames, JSON |
//! | `GET /wt` | `{url, cert_hash}` when WebTransport is on, else `null` (200, so a browser asking logs no error) |
//! | `GET /ws` | The WebSocket: text frames are control messages ([`crate::control`]), binary frames start with the 16 byte header |
//!
//! The dashboard page comes from another origin (the dashboard's port), so the GET routes answer CORS requests, and
//! the upgrade checks the `Origin` header, both under [`crate::origin::OriginPolicy`]: the same host as the request,
//! or a listed origin. A browser sends no preflight for a WebSocket upgrade, so without that check any site a viewer
//! visits could open a session from their browser and read the cameras.

use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{debug, info, warn};

use crate::control::Reply;
use crate::mux::Mux;
use crate::session::{Outgoing, Session};

/// The router over `session`.
pub fn router(session: Arc<Session>) -> Router {
    let policy = session.origins().clone();
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin, parts| {
            let host = parts.headers.get(header::HOST).and_then(|h| h.to_str().ok());
            origin.to_str().is_ok_and(|o| policy.allows(Some(o), host))
        }))
        .allow_methods([axum::http::Method::GET]);
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/streams", get(streams))
        .route("/wt", get(webtransport))
        .route("/ws", get(upgrade))
        .layer(cors)
        .with_state(session)
}

/// Serves `router` on `listener` until the task is dropped or the process ends.
///
/// # Errors
///
/// An I/O error from the listener.
pub async fn serve(listener: TcpListener, router: Router) -> std::io::Result<()> {
    info!(addr = %listener.local_addr()?, "gateway listening (WebSockets)");
    axum::serve(listener, router).await
}

async fn streams(State(session): State<Arc<Session>>) -> Response {
    // The listing asks the broker (or, without one, lists the chunk directory): off the async threads.
    let listed = tokio::task::spawn_blocking(move || session.streams()).await;
    match listed.unwrap_or_else(|e| Err(e.to_string())) {
        Ok(streams) => Json(streams).into_response(),
        Err(message) => (StatusCode::INTERNAL_SERVER_ERROR, Json(Reply::Error { message })).into_response(),
    }
}

async fn webtransport(State(session): State<Arc<Session>>) -> Response {
    match session.webtransport() {
        Some(info) => Json(info).into_response(),
        None => Json(serde_json::Value::Null).into_response(), // off: start the gateway with --webtransport
    }
}

/// The request's `Origin` as the policy sees it: `Ok(None)` for none, `Err` for one that isn't text (never from a
/// browser; refused rather than treated as absent).
fn origin_of(headers: &HeaderMap) -> Result<Option<&str>, ()> {
    match headers.get(header::ORIGIN) {
        None => Ok(None),
        Some(v) => v.to_str().map(Some).map_err(|_| ()),
    }
}

async fn upgrade(ws: WebSocketUpgrade, headers: HeaderMap, State(session): State<Arc<Session>>) -> Response {
    let origin = origin_of(&headers);
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok());
    if !origin.is_ok_and(|o| session.origins().allows(o, host)) {
        warn!(
            origin = origin.ok().flatten().unwrap_or(""),
            "refused a websocket from an origin the policy doesn't allow"
        );
        return (StatusCode::FORBIDDEN, "this origin may not open a gateway session").into_response();
    }
    ws.max_message_size(64 * 1024).on_upgrade(move |socket| run(socket, session))
}

async fn send_all(sink: &mut futures_util::stream::SplitSink<WebSocket, Message>, out: Vec<Outgoing>) -> bool {
    for o in out {
        let msg = match o {
            Outgoing::Text(t) => Message::Text(t.into()),
            Outgoing::Binary(b) => Message::Binary(b.into()),
        };
        if let Err(e) = sink.send(msg).await {
            debug!(error = %e, "websocket send failed; closing");
            return false;
        }
    }
    true
}

async fn run(socket: WebSocket, session: Arc<Session>) {
    let (mut sink, mut stream) = socket.split();
    if !send_all(&mut sink, vec![session.hello("websocket")]).await {
        return;
    }
    let mut mux = Mux::default();
    loop {
        tokio::select! {
            next = stream.next() => {
                let Some(next) = next else { break };
                let msg = match next {
                    Ok(m) => m,
                    Err(e) => {
                        debug!(error = %e, "websocket closed");
                        break;
                    }
                };
                match msg {
                    Message::Text(text) => {
                        let out = match mux.control(&session, &text).await {
                            Ok(out) => out,
                            Err(e) => {
                                warn!(error = %e, "a request's task failed");
                                break;
                            }
                        };
                        if !send_all(&mut sink, out).await {
                            break;
                        }
                    }
                    Message::Binary(_) => {
                        warn!("binary message from a client; ignored");
                    }
                    Message::Close(_) => break,
                    Message::Ping(_) | Message::Pong(_) => {}
                }
            }
            out = mux.next() => {
                let Some(out) = out else { continue };
                if !send_all(&mut sink, vec![out]).await {
                    break;
                }
            }
        }
    }
    mux.close().await;
}
