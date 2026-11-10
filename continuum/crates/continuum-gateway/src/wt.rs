//! The WebTransport transport (opt-in, [`crate::cert`]). The client opens one bidirectional stream for control:
//! each message, either way, is a `u32` little-endian length then the JSON. Every binary message from the gateway
//! is its own unidirectional stream (the 16 byte header then the payload, the stream finished after it), which is
//! how WebTransport frames messages and lets the browser read each as one `ArrayBuffer`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tracing::{debug, info, warn};
use wtransport::endpoint::IncomingSession;
use wtransport::{Connection, Endpoint, Identity, ServerConfig};

use crate::cert::{CertSource, ROTATE_AFTER, identity, leaf_hash_hex};
use crate::mux::Mux;
use crate::session::{Outgoing, Session};

/// The largest control message accepted, bytes.
pub const MAX_CONTROL: u32 = 64 * 1024;

pub use crate::session::SharedHash;

/// Runs the WebTransport server on `bind` with `source`'s identity, updating `hash` when a self-signed identity
/// rotates, until the task is dropped.
///
/// # Errors
///
/// An identity error at start, or an I/O error binding.
pub async fn run(
    bind: SocketAddr,
    source: CertSource,
    hash: SharedHash,
    session: Arc<Session>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let id = identity(&source).await?;
    if let Ok(mut h) = hash.write() {
        *h = leaf_hash_hex(&id);
    }
    let endpoint = Endpoint::server(config(bind, id))?;
    info!(addr = %endpoint.local_addr()?, "gateway listening (WebTransport)");
    let rotate = matches!(source, CertSource::SelfSigned { .. });
    let mut next_rotation = tokio::time::Instant::now() + ROTATE_AFTER;
    loop {
        tokio::select! {
            incoming = endpoint.accept() => {
                let session = Arc::clone(&session);
                tokio::spawn(async move {
                    if let Err(e) = handle(incoming, session).await {
                        debug!(error = %e, "webtransport session ended");
                    }
                });
            }
            () = tokio::time::sleep_until(next_rotation), if rotate => {
                next_rotation = tokio::time::Instant::now() + ROTATE_AFTER;
                match identity(&source).await {
                    Ok(id) => {
                        if let Ok(mut h) = hash.write() {
                            *h = leaf_hash_hex(&id);
                        }
                        if let Err(e) = endpoint.reload_config(config(bind, id), false) {
                            warn!(error = %e, "couldn't rotate the self-signed certificate");
                        } else {
                            info!("rotated the self-signed certificate");
                        }
                    }
                    Err(e) => warn!(error = %e, "couldn't make a new self-signed certificate"),
                }
            }
        }
    }
}

fn config(bind: SocketAddr, id: Identity) -> ServerConfig {
    ServerConfig::builder()
        .with_bind_address(bind)
        .with_identity(id)
        .keep_alive_interval(Some(Duration::from_secs(5)))
        .max_idle_timeout(Some(Duration::from_secs(30)))
        .expect("a 30 s idle timeout fits a VarInt")
        .build()
}

async fn handle(
    incoming: IncomingSession,
    session: Arc<Session>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let request = incoming.await?;
    debug!(path = request.path(), from = %request.remote_address(), "webtransport session");
    if !session.origins().allows(request.origin(), Some(request.authority())) {
        warn!(
            origin = request.origin().unwrap_or(""),
            "refused a webtransport session from an origin the policy doesn't allow"
        );
        request.forbidden().await;
        return Ok(());
    }
    let conn = request.accept().await?;
    let (mut send, mut recv) = conn.accept_bi().await?;
    write_control(&mut send, &session.hello("webtransport")).await?;
    // Control messages are read by their own task: `read_exact` isn't cancellation safe, so it can't sit in the
    // `select!` beside the live frames (a length prefix split across reads would be lost).
    let (control_tx, mut control) = tokio::sync::mpsc::channel::<Result<String, String>>(4);
    let reader = tokio::spawn(async move {
        loop {
            let mut len = [0u8; 4];
            if recv.read_exact(&mut len).await.is_err() {
                return;
            }
            let len = u32::from_le_bytes(len);
            if len > MAX_CONTROL {
                let _ = control_tx
                    .send(Err(format!("control message of {len} bytes is over the {MAX_CONTROL} byte limit")))
                    .await;
                return;
            }
            let mut body = vec![0u8; len as usize];
            if recv.read_exact(&mut body).await.is_err() {
                return;
            }
            let text = match String::from_utf8(body) {
                Ok(t) => t,
                Err(e) => {
                    let _ = control_tx.send(Err(e.to_string())).await;
                    return;
                }
            };
            if control_tx.send(Ok(text)).await.is_err() {
                return;
            }
        }
    });
    let mut mux = Mux::default();
    let result: Result<(), Box<dyn std::error::Error + Send + Sync>> = loop {
        let outs = tokio::select! {
            next = control.recv() => match next {
                None => break Ok(()),
                Some(Err(e)) => break Err(e.into()),
                Some(Ok(text)) => match mux.control(&session, &text).await {
                    Ok(out) => out,
                    Err(e) => break Err(e.into()),
                },
            },
            out = mux.next() => out.into_iter().collect(),
        };
        let mut failed = None;
        for out in outs {
            let sent = match out {
                Outgoing::Text(_) => write_control(&mut send, &out).await,
                Outgoing::Binary(bytes) => send_uni(&conn, &bytes).await,
            };
            if let Err(e) = sent {
                failed = Some(e);
                break;
            }
        }
        if let Some(e) = failed {
            break Err(e);
        }
    };
    mux.close().await;
    reader.abort();
    result
}

async fn write_control(
    send: &mut wtransport::SendStream,
    out: &Outgoing,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let Outgoing::Text(text) = out else { return Err("a binary message on the control stream".into()) };
    let len = u32::try_from(text.len())?;
    send.write_all(&len.to_le_bytes()).await?;
    send.write_all(text.as_bytes()).await?;
    Ok(())
}

async fn send_uni(conn: &Connection, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut stream = conn.open_uni().await?.await?;
    stream.write_all(bytes).await?;
    stream.finish().await?;
    Ok(())
}
