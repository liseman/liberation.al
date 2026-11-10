//! The WebTransport transport end to end with a self-signed identity: the client trusts the certificate by its
//! hash (what a browser does with serverCertificateHashes), control messages are length-prefixed on the client's
//! bidirectional stream, and each frame arrives as its own unidirectional stream.

use std::sync::Arc;

use continuum_core::StreamId;
use continuum_gateway::cert::CertSource;
use continuum_gateway::control::Reply;
use continuum_gateway::header::{PayloadType, split};
use continuum_gateway::session::WebTransportEndpoint;
use continuum_gateway::{ChunkSource, OriginPolicy, Session, wt};
use continuum_tier::{ChunkFormat, ChunkWriter};
use std::sync::RwLock;
use wtransport::tls::{Sha256Digest, Sha256DigestFmt};
use wtransport::{ClientConfig, Endpoint};

async fn read_control(recv: &mut wtransport::RecvStream) -> String {
    let mut len = [0u8; 4];
    recv.read_exact(&mut len).await.expect("len");
    let mut body = vec![0u8; u32::from_le_bytes(len) as usize];
    recv.read_exact(&mut body).await.expect("body");
    String::from_utf8(body).expect("utf8")
}

/// Where a test's chunk `name` of `stream` created at `created_ns` goes (`continuum_tier::layout`), its hour made.
fn in_hour(dir: &std::path::Path, stream: u32, created_ns: u64, name: &str) -> std::path::PathBuf {
    let hour = continuum_tier::layout::hour_dir(dir, StreamId(stream), created_ns);
    continuum_tier::layout::make_hour_dir(dir, &hour).expect("hour");
    hour.join(name)
}

#[tokio::test]
async fn a_range_streams_over_webtransport_against_a_self_signed_certificate() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut w = ChunkWriter::create(&in_hour(dir.path(), 2, 1, "2-1-1.cnk"), StreamId(2), ChunkFormat::Passthrough, 1)
        .expect("create");
    for i in 0..3u32 {
        let t = 5_000_000_000 + u64::from(i) * 33_000_000;
        w.append(t, t, i, &[0xff, 0xd8, i as u8]).expect("append");
    }
    w.close().expect("close");

    // Bind a UDP port to find a free one, release it, and give it to the server.
    let probe = std::net::UdpSocket::bind("127.0.0.1:0").expect("udp");
    let bind = probe.local_addr().expect("addr");
    drop(probe);
    let hash = Arc::new(RwLock::new(String::new()));
    let endpoint = WebTransportEndpoint { url: format!("https://127.0.0.1:{}", bind.port()), hash: Arc::clone(&hash) };
    let session =
        Arc::new(Session::new(Arc::new(ChunkSource::new(dir.path())), Some(endpoint), OriginPolicy::default()));
    let source = CertSource::SelfSigned { names: vec!["localhost".to_owned(), "127.0.0.1".to_owned()] };
    let server = tokio::spawn(wt::run(bind, source, Arc::clone(&hash), session));
    let mut hex = String::new();
    for _ in 0..200 {
        hex = hash.read().expect("lock").clone();
        if !hex.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(hex.len(), 64, "the hash is published once the identity exists");

    let digest = Sha256Digest::from_str_fmt(&hex, Sha256DigestFmt::BytesArray).or_else(|_| {
        let dotted: Vec<String> = hex.as_bytes().chunks(2).map(|c| String::from_utf8_lossy(c).into_owned()).collect();
        Sha256Digest::from_str_fmt(dotted.join(":"), Sha256DigestFmt::DottedHex)
    });
    let digest = digest.expect("a digest from the hex the page gets");
    let client =
        Endpoint::client(ClientConfig::builder().with_bind_default().with_server_certificate_hashes([digest]).build())
            .expect("client");
    let conn = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        client.connect(format!("https://127.0.0.1:{}/", bind.port())),
    )
    .await
    .expect("connect in time")
    .expect("connect");
    let (mut send, mut recv) = conn.open_bi().await.expect("open").await.expect("bi");
    let hello = read_control(&mut recv).await;
    assert!(
        matches!(serde_json::from_str::<Reply>(&hello), Ok(Reply::Hello { transport, .. }) if transport == "webtransport"),
        "{hello}"
    );

    let req = r#"{"type":"range","stream":2,"start_us":0,"end_us":6000000000}"#;
    send.write_all(&(req.len() as u32).to_le_bytes()).await.expect("len");
    send.write_all(req.as_bytes()).await.expect("req");
    let mut frames = Vec::new();
    for _ in 0..3 {
        let mut uni = conn.accept_uni().await.expect("uni");
        let mut bytes = Vec::new();
        let mut buf = [0u8; 4096];
        while let Some(k) = uni.read(&mut buf).await.expect("read") {
            bytes.extend_from_slice(&buf[..k]);
        }
        frames.push(bytes);
    }
    let done = read_control(&mut recv).await;
    assert!(matches!(serde_json::from_str::<Reply>(&done), Ok(Reply::Done { frames: 3, .. })), "{done}");
    let mut times: Vec<u64> = frames
        .iter()
        .map(|f| {
            let (h, body) = split(f).expect("header");
            assert_eq!((h.stream_id, h.payload_type), (2, PayloadType::Jpeg));
            assert_eq!(body.len(), 3);
            h.capture_us
        })
        .collect();
    times.sort_unstable();
    assert_eq!(times, vec![5_000_000, 5_033_000, 5_066_000]);
    conn.close(wtransport::VarInt::from_u32(0), b"done");
    server.abort();
}
