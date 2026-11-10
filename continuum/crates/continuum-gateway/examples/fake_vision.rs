//! A broker with a camera and vision records, and the gateway in front of it, so the dashboard's live view can be
//! tried with no camera, capture, or vision worker: `cargo run -p continuum-gateway --example fake_vision -- [port]`.
//!
//! It runs a broker in-process (on a socket in a temporary directory), registers `cam0.video` (MJPEG) and the vision
//! streams `face.detection`, `face.match`, and `pose.body`, and publishes 10 frames a second of a moving square with
//! a face box over it, matched to a known person, and a two-joint skeleton, each record at its frame's capture time, as
//! the vision workers would (docs/VISION-STREAMS.md). The gateway serves WebSockets on 127.0.0.1:`port` (8766 by
//! default, where the page looks). Stop it with Ctrl-C.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use continuum_broker::server::{Config, Server};
use continuum_client::{Client, ClientOptions, StreamSpec, StreamStorage};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, HeaderSpec, StorageLocator, StreamId, Uuid};
use continuum_gateway::{ChunkSource, OriginPolicy, Session, ws};

const W: u16 = 320;
const H: u16 = 180;

/// A frame: a dark picture with a square at (x, y), 40 pixels a side.
fn jpeg(x: usize, y: usize) -> Vec<u8> {
    let mut rgb = vec![30u8; usize::from(W) * usize::from(H) * 3];
    for yy in y..(y + 40).min(usize::from(H)) {
        for xx in x..(x + 40).min(usize::from(W)) {
            let at = (yy * usize::from(W) + xx) * 3;
            rgb[at] = 230;
            rgb[at + 1] = 160;
            rgb[at + 2] = 40;
        }
    }
    let mut out = Vec::new();
    jpeg_encoder::Encoder::new(&mut out, 80).encode(&rgb, W, H, jpeg_encoder::ColorType::Rgb).expect("encode");
    out
}

fn spec(name: &str, kind: fb::StreamKind, format: fb::PayloadFormat, shape: &str) -> StreamSpec {
    StreamSpec {
        name: name.into(),
        kind,
        format,
        shape_name: shape.into(),
        shape_version: 1,
        nominal_rate_hz: 10.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    }
}

fn now_ns() -> u64 {
    u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH).expect("clock").as_nanos()).expect("fits")
}

/// A UUIDv7-shaped id: the time in milliseconds, then a counter, so every payload gets its own.
fn new_id() -> Uuid {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut b = [0u8; 16];
    b[..6].copy_from_slice(&(now_ns() / 1_000_000).to_be_bytes()[2..]);
    b[8..].copy_from_slice(&N.fetch_add(1, std::sync::atomic::Ordering::Relaxed).to_be_bytes());
    b[6] = 0x70;
    b[8] = 0x80 | (b[8] & 0x3f);
    Uuid(b)
}

async fn publish(p: &Client, stream: u32, t: u64, seq: u32, bytes: Vec<u8>) {
    let mut h = HeaderSpec::raw(new_id(), StreamId(stream), t, t);
    h.seq = seq;
    let header = HeaderBuf::build(&h).expect("header");
    if let Err(e) = p.ingest_acked(&header, Some(&StorageLocator::Inline { bytes })).await {
        eprintln!("fake_vision: ingest: {e}");
    }
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::args().nth(1).and_then(|p| p.parse().ok()).unwrap_or(8766);
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("broker.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.expect("broker");
    tokio::spawn(server.run_until(std::future::pending()));

    let p = Client::connect(&socket, ClientOptions::new("fake-vision")).await.expect("connect");
    let text = fb::PayloadFormat::Text;
    let derived = fb::StreamKind::Derived;
    let cam = p
        .register_stream(&spec("cam0.video", fb::StreamKind::Sensor, fb::PayloadFormat::EncodedVideo, "MJPEG"))
        .await
        .expect("register")
        .stream_id;
    let det = p.register_stream(&spec("face.detection", derived, text, "face.detection")).await.expect("r").stream_id;
    let mat = p.register_stream(&spec("face.match", derived, text, "face.match")).await.expect("r").stream_id;
    let pose = p.register_stream(&spec("pose.body", derived, text, "pose.body")).await.expect("r").stream_id;

    let gateway = Client::connect(&socket, ClientOptions::new("continuum-gateway")).await.expect("connect");
    let chunks = dir.path().join("tier");
    std::fs::create_dir_all(&chunks).expect("mkdir");
    let session = Arc::new(
        Session::new(Arc::new(ChunkSource::new(&chunks)), None, OriginPolicy::default())
            .with_broker(socket.clone(), Some(Arc::new(gateway))),
    );
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.expect("bind");
    eprintln!("fake_vision: the gateway is on ws://127.0.0.1:{port}/ws; cam0.video at 10 fps with faces and poses");
    tokio::spawn(ws::serve(listener, ws::router(session)));

    let mut tick = tokio::time::interval(Duration::from_millis(100));
    for i in 0u32.. {
        tick.tick().await;
        let t = now_ns();
        let x = (i as usize * 4) % (usize::from(W) - 40);
        let y = 70;
        publish(&p, cam, t, i, jpeg(x, y)).await;
        let (x1, y1) = (x as f64 / f64::from(W), y as f64 / f64::from(H));
        let (x2, y2) = ((x + 40) as f64 / f64::from(W), (y + 40) as f64 / f64::from(H));
        let face = format!(
            r#"{{"cam":"cam0","frame_seq":{i},"bbox":[{x1},{y1},{x2},{y2}],"det":0.9,"track_id":"fake:cam0:1"}}"#
        );
        publish(&p, det, t, i, face.into_bytes()).await;
        // Every third frame is posed and matched, as a sampling worker would.
        if i % 3 == 0 {
            publish(&p, mat, t, i, br#"{"cam":"cam0","track_id":"fake:cam0:1","name":"Sam","sim":0.7}"#.to_vec()).await;
            let mut joints = vec!["[-1,-1,0]".to_owned(); 26];
            joints[5] = format!("[{x1},{y1},0.9]");
            joints[7] = format!("[{x2},{y2},0.9]");
            let body = format!(
                r#"{{"cam":"cam0","frame_seq":{i},"person":0,"box":[{x1},{y1},{x2},{y2},0.9],"keypoints":[{}],"score":0.9,"skeleton":"halpe26"}}"#,
                joints.join(",")
            );
            publish(&p, pose, t, i, body.into_bytes()).await;
        }
    }
}
