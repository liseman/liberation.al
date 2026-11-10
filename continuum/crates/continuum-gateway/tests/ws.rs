//! The WebSocket transport end to end: a chunk on disk, the gateway on an ephemeral port, a tokio-tungstenite
//! client asking for a range, frames with the 16 byte header coming back in order, then `done`.

use std::sync::Arc;

use continuum_core::StreamId;
use continuum_gateway::control::Reply;
use continuum_gateway::header::{PayloadType, split};
use continuum_gateway::{ChunkSource, OriginPolicy, Session, ws};
use continuum_tier::{ChunkFormat, ChunkWriter};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

/// Where a test's chunk `name` of `stream` created at `created_ns` goes (`continuum_tier::layout`), its hour made.
fn in_hour(dir: &std::path::Path, stream: u32, created_ns: u64, name: &str) -> std::path::PathBuf {
    let hour = continuum_tier::layout::hour_dir(dir, StreamId(stream), created_ns);
    continuum_tier::layout::make_hour_dir(dir, &hour).expect("hour");
    hour.join(name)
}

fn jpeg(i: u8) -> Vec<u8> {
    vec![0xff, 0xd8, 0xff, 0xe0, i, 0xff, 0xd9]
}

#[tokio::test]
async fn a_range_streams_over_websockets_with_headers_and_done() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut w = ChunkWriter::create(&in_hour(dir.path(), 4, 1, "4-1-1.cnk"), StreamId(4), ChunkFormat::Passthrough, 1)
        .expect("create");
    for i in 0..5u32 {
        let t = 2_000_000_000 + u64::from(i) * 40_000_000;
        w.append(t, t, i, &jpeg(i as u8)).expect("append");
    }
    w.close().expect("close");
    let session = Arc::new(Session::new(Arc::new(ChunkSource::new(dir.path())), None, OriginPolicy::default()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(ws::serve(listener, ws::router(session)));

    let body = reqwest_free_get(addr, "/healthz").await;
    assert!(body.ends_with("ok"), "{body}");
    let streams = reqwest_free_get(addr, "/streams").await;
    assert!(streams.contains(r#""id":4"#) && streams.contains(r#""end_us":2160001"#), "{streams}");
    let wt = reqwest_free_get(addr, "/wt").await;
    assert!(wt.starts_with("HTTP/1.1 200") && wt.ends_with("null"), "{wt}"); // off, without an error status

    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws")).await.expect("connect");
    let hello = socket.next().await.expect("hello").expect("ok");
    let Message::Text(hello) = hello else { panic!("hello is text") };
    assert!(
        matches!(serde_json::from_str::<Reply>(&hello), Ok(Reply::Hello { transport, webtransport: None, .. }) if transport == "websocket"),
        "{hello}"
    );

    socket
        .send(Message::Text(r#"{"type":"range","stream":4,"start_us":2040000,"end_us":2200000}"#.into()))
        .await
        .expect("send");
    let mut frames = Vec::new();
    let done = loop {
        match socket.next().await.expect("message").expect("ok") {
            Message::Binary(b) => frames.push(b.to_vec()),
            Message::Text(t) => break t,
            other => panic!("unexpected {other:?}"),
        }
    };
    let Ok(Reply::Done { frames: n, skipped: 0, next_us: None, codecs, .. }) = serde_json::from_str::<Reply>(&done)
    else {
        panic!("{done}")
    };
    assert_eq!((n, codecs), (4, vec!["jpeg".to_owned()]));
    assert_eq!(frames.len(), 4, "a frame per record");
    for (k, frame) in frames.iter().enumerate() {
        let (fh, fb) = split(frame).expect("frame header");
        assert_eq!((fh.stream_id, fh.payload_type, fh.keyframe), (4, PayloadType::Jpeg, true));
        assert_eq!(fh.capture_us, 2_040_000 + k as u64 * 40_000);
        assert_eq!(fb, jpeg(k as u8 + 1));
    }

    socket.send(Message::Text(r#"{"type":"ping"}"#.into())).await.expect("send");
    assert!(
        matches!(socket.next().await.expect("pong").expect("ok"), Message::Text(t) if t.as_str() == r#"{"type":"pong"}"#)
    );
    socket.send(Message::Text("junk".into())).await.expect("send");
    assert!(matches!(socket.next().await.expect("error").expect("ok"), Message::Text(t) if t.contains("bad request")));
    socket.close(None).await.expect("close");

    // A page from another site can't open a session; the dashboard on the same host can.
    let mut evil =
        tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(format!("ws://{addr}/ws"))
            .expect("req");
    evil.headers_mut().insert("Origin", "http://evil.example".parse().expect("header"));
    let err = tokio_tungstenite::connect_async(evil).await.expect_err("refused");
    assert!(err.to_string().contains("403"), "{err}");
    let mut dash =
        tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(format!("ws://{addr}/ws"))
            .expect("req");
    dash.headers_mut().insert("Origin", format!("http://{}:8765", addr.ip()).parse().expect("header"));
    // DNS rebinding: Origin and Host agree on a name that isn't the gateway's.
    let mut rebound =
        tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(format!("ws://{addr}/ws"))
            .expect("req");
    rebound.headers_mut().insert("Origin", "http://evil.example".parse().expect("header"));
    rebound.headers_mut().insert("Host", "evil.example".parse().expect("header"));
    let err = tokio_tungstenite::connect_async(rebound).await.expect_err("refused");
    assert!(err.to_string().contains("403"), "{err}");
    let (mut ok, _) = tokio_tungstenite::connect_async(dash).await.expect("same host");
    assert!(matches!(ok.next().await.expect("hello").expect("ok"), Message::Text(_)));
    ok.close(None).await.expect("close");
    let cors = reqwest_free_get_with(addr, "/streams", &[("Origin", "http://evil.example")]).await;
    assert!(!cors.to_ascii_lowercase().contains("access-control-allow-origin"), "{cors}");
    let cors = reqwest_free_get_with(addr, "/streams", &[("Origin", &format!("http://{}:8765", addr.ip()))]).await;
    assert!(cors.to_ascii_lowercase().contains("access-control-allow-origin"), "{cors}");
    server.abort();
}

/// A one-line HTTP GET without pulling in an HTTP client crate: the status line, headers, and the body.
async fn reqwest_free_get(addr: std::net::SocketAddr, path: &str) -> String {
    reqwest_free_get_with(addr, path, &[]).await
}

async fn reqwest_free_get_with(addr: std::net::SocketAddr, path: &str, extra: &[(&str, &str)]) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let headers: String = extra.iter().map(|(k, v)| format!("{k}: {v}\r\n")).collect();
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n{headers}\r\n").as_bytes())
        .await
        .expect("write");
    let mut out = String::new();
    s.read_to_string(&mut out).await.expect("read");
    out
}
