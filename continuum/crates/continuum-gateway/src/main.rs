//! `continuum-gateway`: the browser gateway (docs/CONTINUUM.md, "Browser gateway").
//!
//! ```text
//! continuum-gateway --state-dir data/state                       # WebSockets on 127.0.0.1:8766
//! continuum-gateway --listen 0.0.0.0:8766 --webtransport 0.0.0.0:8767 --self-signed devbox,devbox.local,100.64.0.9
//! continuum-gateway --webtransport 0.0.0.0:8767 --cert tls/cert.pem --key tls/key.pem
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use continuum_gateway::cert::{CertSource, pem_source};
use continuum_gateway::session::WebTransportEndpoint;
use continuum_gateway::{ChunkSource, OriginPolicy, Session, ws, wt};
use std::sync::RwLock;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(about = "Continuum's browser gateway: WebSockets and WebTransport, frames and overlays for WebCodecs")]
struct Args {
    /// The pack's state directory; chunks are read from <state-dir>/continuum/tier unless --chunks-dir says otherwise.
    #[arg(long, default_value = "data/state")]
    state_dir: PathBuf,
    /// The tier worker's chunk directory.
    #[arg(long)]
    chunks_dir: Option<PathBuf>,
    /// The broker's socket; <state-dir>/continuum/broker.sock unless set. The gateway runs from the chunk
    /// directory alone when no broker answers at startup.
    #[arg(long)]
    broker_socket: Option<PathBuf>,
    /// Where to serve HTTP and WebSockets.
    #[arg(long, default_value = "127.0.0.1:8766")]
    listen: SocketAddr,
    /// Also serve WebTransport (HTTP/3 over QUIC) on this address. Needs --cert and --key, or --self-signed.
    #[arg(long)]
    webtransport: Option<SocketAddr>,
    /// The host name pages use to reach the WebTransport endpoint (the URL they are given). Defaults to the listen
    /// address's host.
    #[arg(long)]
    public_host: Option<String>,
    /// A PEM certificate chain the browser trusts (Tailscale's, for instance).
    #[arg(long, requires = "key")]
    cert: Option<PathBuf>,
    /// Its PEM private key.
    #[arg(long, requires = "cert")]
    key: Option<PathBuf>,
    /// Make a self-signed ECDSA P-256 certificate (14 days, regenerated weekly) for these comma-separated names and
    /// addresses, and give pages its hash for serverCertificateHashes.
    #[arg(long, value_delimiter = ',', conflicts_with = "cert")]
    self_signed: Option<Vec<String>>,
    /// Browser origins allowed to open a session besides pages served from the gateway's own host (the dashboard
    /// on the same machine always is). Repeatable or comma-separated: `https://dash.example:8765`.
    #[arg(long, value_delimiter = ',')]
    allow_origin: Vec<String>,
    /// Host names this gateway is reached by, besides the machine's host name (long and short), its .local name,
    /// localhost, any IP address, --public-host, and the --self-signed names (a Tailscale MagicDNS name, say). A
    /// browser request whose Host is none of these is refused, which is what defeats DNS rebinding. Repeatable or
    /// comma-separated; empty entries are ignored, so a service can pass an unset variable.
    #[arg(long, value_delimiter = ',')]
    allow_host: Vec<String>,
}

/// The names this gateway answers to: the built-in ones, the machine's, and the command line's.
fn served_names(args: &Args) -> Vec<String> {
    let mut names = vec!["localhost".to_owned()];
    if let Ok(h) = hostname::get()
        && let Some(h) = h.to_str()
    {
        // A machine with a domain reports `devbox.example.com`; its short name is what the LAN and mDNS use.
        let short = h.split('.').next().unwrap_or(h);
        names.push(h.to_owned());
        names.push(short.to_owned());
        names.push(format!("{short}.local"));
    }
    names.extend(args.allow_host.iter().cloned());
    names.extend(args.public_host.iter().cloned());
    names.extend(args.self_signed.iter().flatten().cloned());
    names
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Args::parse();
    let chunks = args.chunks_dir.clone().unwrap_or_else(|| args.state_dir.join("continuum").join("tier"));
    let source = Arc::new(ChunkSource::new(&chunks));
    {
        // The first full listing of the chunk directory, off the request path (see ChunkSource::streams).
        let warm = Arc::clone(&source);
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            match warm.streams() {
                Ok(s) => tracing::info!(streams = s.len(), took = ?started.elapsed(), "listed the recorded chunks"),
                Err(e) => tracing::warn!(error = %e, "couldn't list the recorded chunks"),
            }
        });
    }
    let broker_socket =
        args.broker_socket.clone().unwrap_or_else(|| args.state_dir.join("continuum").join("broker.sock"));
    let broker = match continuum_client::Client::connect(
        &broker_socket,
        continuum_client::ClientOptions::new("continuum-gateway"),
    )
    .await
    {
        Ok(c) => {
            info!(socket = %broker_socket.display(), "the broker is in the loop");
            Some(Arc::new(c))
        }
        Err(e) => {
            warn!(socket = %broker_socket.display(), error = %e, "no broker: serving the chunk directory alone");
            None
        }
    };
    // The link reconnects when the broker comes (back); until then the chunk directory serves alone.
    // Overlay streams in the segment store (camera.objects, #1791) are read from <state-dir>/segments.
    let segments = args.state_dir.join("segments");
    let data_root = std::path::absolute(&args.state_dir).unwrap_or_else(|_| args.state_dir.clone());
    let with_broker = |s: Session| {
        s.with_broker(broker_socket.clone(), broker.clone())
            .with_segments(segments.clone())
            .with_data_root(data_root.clone())
    };
    let origins = OriginPolicy::new(args.allow_origin.clone(), served_names(&args));

    let mut wt_task = None;
    let hash = Arc::new(RwLock::new(String::new()));
    let webtransport = match args.webtransport {
        None => None,
        Some(bind) => {
            let cert = match (pem_source(args.cert.as_deref(), args.key.as_deref()), args.self_signed.clone()) {
                (Some(pem), _) => pem,
                (None, Some(names)) => CertSource::SelfSigned { names },
                (None, None) => {
                    error!("--webtransport needs --cert and --key, or --self-signed");
                    return std::process::ExitCode::from(2);
                }
            };
            let host = args.public_host.clone().unwrap_or_else(|| bind.ip().to_string());
            let url = format!("https://{host}:{}", bind.port());
            let source2 = Arc::clone(&source);
            let hash2 = Arc::clone(&hash);
            // The WebTransport sessions see the same chunk source; their hello carries the live hash too.
            let endpoint = WebTransportEndpoint { url, hash: Arc::clone(&hash) };
            let wt_sessions =
                Arc::new(with_broker(Session::new(Arc::clone(&source2), Some(endpoint.clone()), origins.clone())));
            wt_task = Some(tokio::spawn(async move { wt::run(bind, cert, hash2, wt_sessions).await }));
            // Wait for the identity so the hash is known before the HTTP side answers.
            for _ in 0..100 {
                if hash.read().is_ok_and(|h| !h.is_empty()) {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            Some(endpoint)
        }
    };
    if let Some(endpoint) = &webtransport
        && endpoint.info().cert_hash.is_empty()
        && args.self_signed.is_some()
    {
        error!("the WebTransport identity wasn't ready; is the port free?");
        return std::process::ExitCode::from(1);
    }

    let session = Arc::new(with_broker(Session::new(source, webtransport, origins)));
    let listener = match tokio::net::TcpListener::bind(args.listen).await {
        Ok(l) => l,
        Err(e) => {
            error!(addr = %args.listen, error = %e, "couldn't listen");
            return std::process::ExitCode::from(1);
        }
    };
    let http = tokio::spawn(ws::serve(listener, ws::router(session)));
    let result = tokio::select! {
        r = http => r.map_err(|e| e.to_string()).and_then(|r| r.map_err(|e| e.to_string())),
        r = async { match wt_task { Some(t) => t.await.map_err(|e| e.to_string()).and_then(|r| r.map_err(|e| e.to_string())), None => std::future::pending().await } } => r,
        _ = tokio::signal::ctrl_c() => Ok(()),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            error!(error = %e, "gateway stopped");
            std::process::ExitCode::from(1)
        }
    }
}
