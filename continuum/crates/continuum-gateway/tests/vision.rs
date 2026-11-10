//! The vision workers' records as overlays through the gateway, with the broker in the loop: a range asks for
//! overlay groups and gets its frames; a live view follows a camera and draws the `face.boxes` ring's faces and the
//! `pose.boxes` ring's skeletons for it;
//! and a v1 recording registered in place plays by GOP.

use std::sync::Arc;
use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_client::{Client, ClientOptions, StreamSpec, StreamStorage, WorkerSpec};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, HeaderSpec, StorageLocator, StreamId, Uuid};
use continuum_gateway::control::Reply;
use continuum_gateway::header::{PayloadType, split};
use continuum_gateway::{ChunkSource, OriginPolicy, Session, overlay, ws};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

const T0: u64 = 2_000_000_000;
const FRAME: u64 = 40_000_000;

fn jpeg(i: u8) -> Vec<u8> {
    vec![0xff, 0xd8, 0xff, 0xe0, i, 0xff, 0xd9]
}

fn id(n: u8) -> Uuid {
    let mut b = [0u8; 16];
    b[6] = 0x70;
    b[15] = n;
    Uuid(b)
}

fn camera(name: &str) -> StreamSpec {
    StreamSpec {
        name: name.into(),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::EncodedVideo,
        shape_name: "MJPEG".into(),
        shape_version: 1,
        nominal_rate_hz: 30.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    }
}

/// A vision stream as vision_log registers it: derived, Text, its shape its name.
fn vision(name: &str) -> StreamSpec {
    StreamSpec {
        kind: fb::StreamKind::Derived,
        format: fb::PayloadFormat::Text,
        shape_name: name.into(),
        nominal_rate_hz: 0.0,
        ..camera(name)
    }
}

struct Rig {
    _dir: tempfile::TempDir,
    producer: Client,
    page: tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    serving: tokio::task::JoinHandle<std::io::Result<()>>,
    stop: tokio::sync::oneshot::Sender<()>,
    broker: tokio::task::JoinHandle<()>,
    socket: std::path::PathBuf,
}

async fn rig() -> Rig {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("broker.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.expect("broker");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let producer = Client::connect(&socket, ClientOptions::new("test-producer")).await.expect("connect");
    let gateway = Client::connect(&socket, ClientOptions::new("continuum-gateway")).await.expect("connect");
    let chunks = dir.path().join("tier");
    std::fs::create_dir_all(&chunks).expect("mkdir");
    let session = Arc::new(
        Session::new(Arc::new(ChunkSource::new(&chunks)), None, OriginPolicy::default())
            .with_broker(socket.clone(), Some(Arc::new(gateway))),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let serving = tokio::spawn(ws::serve(listener, ws::router(session)));
    let (mut page, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws")).await.expect("connect");
    let Message::Text(hello) = page.next().await.expect("hello").expect("ok") else { panic!("text") };
    let Ok(Reply::Hello { overlay_groups, .. }) = serde_json::from_str::<Reply>(&hello) else { panic!("{hello}") };
    assert_eq!(overlay_groups, ["faces", "bodies", "poses", "objects"]);
    Rig { _dir: dir, producer, page, serving, stop, broker, socket }
}

impl Rig {
    async fn end(self) {
        drop(self.page);
        self.serving.abort();
        self.producer.close("done").await.expect("close");
        let _ = self.stop.send(());
        self.broker.await.expect("broker task");
    }

    /// Sends a request and collects binary messages until the text reply.
    async fn request(&mut self, json: String) -> (Vec<Vec<u8>>, String) {
        self.page.send(Message::Text(json.into())).await.expect("send");
        let mut binary = Vec::new();
        loop {
            match tokio::time::timeout(Duration::from_secs(10), self.page.next()).await.expect("in time") {
                Some(Ok(Message::Binary(b))) => binary.push(b.to_vec()),
                Some(Ok(Message::Text(t))) => return (binary, t.to_string()),
                other => panic!("unexpected {other:?}"),
            }
        }
    }
}

/// The `face.boxes` ring as see/face_boxes.py registers it: derived Text records of a fixed stride.
const FACE_STRIDE: usize = 2048;

fn face_ring() -> StreamSpec {
    StreamSpec {
        nominal_rate_hz: 64.0,
        storage: StreamStorage::Ring { stride_bytes: FACE_STRIDE as u32, seconds: 60.0 },
        layer: fb::DomainLayer::Layer2Decision,
        ..vision("face.boxes")
    }
}

/// One ring record as see/face_boxes.py writes it: a little-endian u32 length, then the JSON, zero-padded.
fn face_record(json: &str) -> Vec<u8> {
    let mut rec = (json.len() as u32).to_le_bytes().to_vec();
    rec.extend_from_slice(json.as_bytes());
    rec.resize(FACE_STRIDE, 0);
    rec
}

const FACES: &str = r#"[{"bbox":[0.1,0.2,0.3,0.4],"det":0.9,"name":"Joe","sim":0.7,"cluster":null,"track_id":7}]"#;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_range_sends_frames_and_no_face_overlay_since_the_face_ring_is_live_only() {
    let mut rig = rig().await;
    let p = &rig.producer;
    let cam = p.register_stream(&camera("facecam.video")).await.expect("register").stream_id;
    p.register_stream(&face_ring()).await.expect("register");
    for i in 0..3u8 {
        let t = T0 + u64::from(i) * FRAME;
        let h = HeaderBuf::build(&HeaderSpec::raw(id(i + 1), StreamId(cam), t, t)).expect("header");
        p.ingest_acked(&h, Some(&StorageLocator::Inline { bytes: jpeg(i) })).await.expect("ingest");
    }
    let (got, done) = rig
        .request(format!(
            r#"{{"type":"range","stream":{cam},"start_us":{},"end_us":{},"overlays":["faces"]}}"#,
            T0 / 1_000,
            (T0 + 3 * FRAME) / 1_000
        ))
        .await;
    assert!(matches!(serde_json::from_str::<Reply>(&done), Ok(Reply::Done { frames: 3, .. })), "{done}");
    let kinds: Vec<PayloadType> = got.iter().map(|m| split(m).map(|(h, _)| h.payload_type).expect("header")).collect();
    assert_eq!(kinds, [PayloadType::Jpeg; 3], "recorded faces are Review's, from the segment store");
    // An unknown group is an error.
    let (_, err) =
        rig.request(format!(r#"{{"type":"range","stream":{cam},"start_us":1,"end_us":2,"overlays":["auras"]}}"#)).await;
    assert!(err.contains("unknown overlay group"), "{err}");
    rig.end().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_live_view_draws_the_face_rings_boxes_for_its_camera() {
    let mut rig = rig().await;
    let cam = rig.producer.register_stream(&camera("facecam.video")).await.expect("register").stream_id;
    let ring = rig.producer.register_stream(&face_ring()).await.expect("register");
    let lease = ring.lease.expect("a ring lease");
    let mut writer = rig
        .producer
        .arena(&lease.arena)
        .expect("the ring's arena")
        .carved_ring_writer(&lease.carved_ring().expect("a carved ring"))
        .expect("writer");
    let (_, opened) = rig.request(format!(r#"{{"type":"live","stream":{cam},"overlays":["faces"]}}"#)).await;
    let Ok(Reply::Live { overlays, .. }) = serde_json::from_str::<Reply>(&opened) else { panic!("{opened}") };
    let names: Vec<&str> = overlays.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["face.boxes"]);

    let p = &rig.producer;
    let h = HeaderBuf::build(&HeaderSpec::raw(id(1), StreamId(cam), T0, T0)).expect("header");
    p.ingest_acked(&h, Some(&StorageLocator::Inline { bytes: jpeg(0) })).await.expect("ingest");
    // Another camera's faces first, then this camera's: only this camera's are drawn.
    writer.push(&face_record(&format!(r#"{{"cam":"door","start_ns":{T0},"faces":{FACES}}}"#))).expect("push");
    writer.push(&face_record(&format!(r#"{{"cam":"facecam","start_ns":{T0},"faces":{FACES}}}"#))).expect("push");
    let mut got = Vec::new();
    while got.len() < 2 {
        match tokio::time::timeout(Duration::from_secs(5), rig.page.next()).await.expect("in time") {
            Some(Ok(Message::Binary(b))) => got.push(b.to_vec()),
            other => panic!("unexpected {other:?}"),
        }
    }
    // The frame comes from the subscription and the boxes from the ring's own thread: either can arrive first.
    let of = |kind: PayloadType| {
        got.iter().map(|b| split(b).expect("header")).find(|(h, _)| h.payload_type == kind).expect("one of each")
    };
    let (h0, _) = of(PayloadType::Jpeg);
    assert_eq!(h0.capture_us, T0 / 1_000);
    let (h1, b1) = of(PayloadType::Overlay);
    assert_eq!((h1.stream_id, h1.capture_us, h1.live), (cam, T0 / 1_000, true));
    let ov = overlay::decode(b1).expect("overlay");
    assert_eq!(ov.boxes.len(), 1, "the other camera's face isn't drawn");
    assert_eq!((ov.boxes[0].label.as_str(), ov.boxes[0].xyxy), ("Joe", [0.1, 0.2, 0.3, 0.4]));
    rig.end().await;
}

/// The `pose.boxes` ring as see/pose_boxes.py registers it.
const POSE_STRIDE: usize = 8192;

fn pose_ring() -> StreamSpec {
    StreamSpec {
        nominal_rate_hz: 128.0,
        storage: StreamStorage::Ring { stride_bytes: POSE_STRIDE as u32, seconds: 10.0 },
        layer: fb::DomainLayer::Layer2Decision,
        ..vision("pose.boxes")
    }
}

fn pose_record(json: &str) -> Vec<u8> {
    let mut rec = face_record(json);
    rec.resize(POSE_STRIDE, 0);
    rec
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_live_view_draws_the_pose_rings_skeletons_as_they_are_written() {
    let mut rig = rig().await;
    let cam = rig.producer.register_stream(&camera("facecam.video")).await.expect("register").stream_id;
    let ring = rig.producer.register_stream(&pose_ring()).await.expect("register");
    let lease = ring.lease.expect("a ring lease");
    let mut writer = rig
        .producer
        .arena(&lease.arena)
        .expect("the ring's arena")
        .carved_ring_writer(&lease.carved_ring().expect("a carved ring"))
        .expect("writer");
    let (_, opened) = rig.request(format!(r#"{{"type":"live","stream":{cam},"overlays":["poses"]}}"#)).await;
    let Ok(Reply::Live { overlays, .. }) = serde_json::from_str::<Reply>(&opened) else { panic!("{opened}") };
    assert_eq!(overlays.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(), ["pose.boxes"]);

    // A confident pose (six joints placed at 0.9): a skeleton.
    let mut joints = vec!["[0.2,0.15,0.9]".to_owned(); 6];
    joints.extend(std::iter::repeat_n("[-1,-1,0]".to_owned(), 20));
    let joints = joints.join(",");
    let posed = format!(r#"{{"track_id":"7","box":[0.1,0.2,0.3,0.9,0.8],"keypoints":[{joints}]}}"#);
    writer.push(&pose_record(&format!(r#"{{"cam":"facecam","start_ns":{T0},"people":[{posed}]}}"#))).expect("push");
    let got = match tokio::time::timeout(Duration::from_secs(5), rig.page.next()).await.expect("in time") {
        Some(Ok(Message::Binary(b))) => b.to_vec(),
        other => panic!("unexpected {other:?}"),
    };
    let (h, body) = split(&got).expect("overlay");
    assert_eq!((h.stream_id, h.payload_type, h.capture_us, h.live), (cam, PayloadType::Overlay, T0 / 1_000, true));
    let ov = overlay::decode(body).expect("overlay");
    assert_eq!(ov.skeletons.len(), 1);
    assert_eq!(ov.skeletons[0].layout, "halpe26");
    assert!((ov.skeletons[0].joints[0].0 - 0.2).abs() < 1e-6, "{:?}", ov.skeletons[0].joints[0]);
    rig.end().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_live_view_opened_before_the_pose_ring_exists_takes_it_when_it_appears() {
    let mut rig = rig().await;
    let cam = rig.producer.register_stream(&camera("facecam.video")).await.expect("register").stream_id;
    let (_, opened) = rig.request(format!(r#"{{"type":"live","stream":{cam},"overlays":["poses"]}}"#)).await;
    assert!(matches!(serde_json::from_str::<Reply>(&opened), Ok(Reply::Live { .. })), "{opened}");
    // see-actions registers the ring after the page opened its view (a restart): the feed finds it on a re-check.
    let ring = rig.producer.register_stream(&pose_ring()).await.expect("register");
    let lease = ring.lease.expect("a ring lease");
    let mut writer = rig
        .producer
        .arena(&lease.arena)
        .expect("the ring's arena")
        .carved_ring_writer(&lease.carved_ring().expect("a carved ring"))
        .expect("writer");
    let joints = [vec!["[0.2,0.15,0.9]".to_owned(); 6], vec!["[-1,-1,0]".to_owned(); 20]].concat().join(",");
    let posed = format!(r#"{{"track_id":"7","box":[0.1,0.2,0.3,0.9,0.8],"keypoints":[{joints}]}}"#);
    let record = pose_record(&format!(r#"{{"cam":"facecam","start_ns":{T0},"people":[{posed}]}}"#));
    let got = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            writer.push(&record).expect("push"); // a record a second, as frames keep coming
            if let Ok(Some(Ok(Message::Binary(b)))) =
                tokio::time::timeout(Duration::from_secs(1), rig.page.next()).await
            {
                return b.to_vec();
            }
        }
    })
    .await
    .expect("the ring's skeleton within a few re-checks");
    let (h, body) = split(&got).expect("overlay");
    assert_eq!((h.payload_type, h.capture_us), (PayloadType::Overlay, T0 / 1_000));
    assert_eq!(overlay::decode(body).expect("overlay").skeletons.len(), 1);
    rig.end().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_v1_recording_registered_in_place_plays_by_gop_in_decode_order() {
    let mut rig = rig().await;
    // v1's video directory with one recording (2 s at 10 fps, H.264 with B-frames, a keyframe every 10 frames).
    let video = rig._dir.path().join("video");
    std::fs::create_dir_all(video.join("cam0/2026-10-29")).expect("mkdir");
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../continuum-tier/tests/data/v1_bframes.mp4");
    std::fs::copy(fixture, video.join("cam0/2026-10-29/140000.mp4")).expect("copy");
    std::fs::write(
        video.join("index.jsonl"),
        "{\"cam\": \"cam0\", \"file\": \"cam0/2026-10-29/140000.mp4\", \"start\": 2.0, \"fps\": 10.0}\n",
    )
    .expect("index");
    let importer = Client::connect(&rig.socket, ClientOptions::new("v1 importer")).await.expect("connect");
    let worker = WorkerSpec {
        worker_id: "v1-video-import".into(),
        version_hash: "test".into(),
        capabilities: vec![],
        device_index: None,
        keeps_copies: false,
    };
    importer.register_worker(&worker).await.expect("worker");
    let done = continuum_tier::v1video::register_v1_video_index(&importer, "cam0", &video).await.expect("register");
    assert!(matches!(done[0].1, continuum_tier::v1audio::V1Registered::Registered { .. }), "{done:?}");

    let (_, listed) = rig.request(r#"{"type":"streams"}"#.to_owned()).await;
    let Ok(Reply::Streams { streams }) = serde_json::from_str::<Reply>(&listed) else { panic!("{listed}") };
    let stream = streams.iter().find(|s| s.name == "cam0.video_v1").expect("listed").id;
    // From 0.5 s to 1.5 s: both GOPs overlap it, so both are sent whole.
    let (got, done) =
        rig.request(format!(r#"{{"type":"range","stream":{stream},"start_us":2500000,"end_us":3500000}}"#)).await;
    let Ok(Reply::Done { frames, codecs, .. }) = serde_json::from_str::<Reply>(&done) else { panic!("{done}") };
    assert_eq!((frames, codecs), (20, vec!["avc".to_owned()]));
    let heads: Vec<_> = got.iter().map(|m| split(m).expect("header").0).collect();
    assert!(heads.iter().all(|h| h.payload_type == PayloadType::Video));
    assert_eq!(heads.iter().filter(|h| h.keyframe).count(), 2, "one keyframe per GOP");
    assert!(heads[0].keyframe && heads[10].keyframe);
    let mut shown: Vec<u64> = heads.iter().map(|h| h.capture_us).collect();
    assert_ne!(
        shown,
        {
            let mut s = shown.clone();
            s.sort_unstable();
            s
        },
        "decode order, not display order"
    );
    shown.sort_unstable();
    assert_eq!(shown, (0..20).map(|k| 2_000_000 + k * 100_000).collect::<Vec<_>>());
    // A budget smaller than a GOP still sends that GOP whole, and says where the next one starts.
    let (got, done) = rig
        .request(format!(r#"{{"type":"range","stream":{stream},"start_us":2000000,"end_us":4000000,"max_frames":5}}"#))
        .await;
    let Ok(Reply::Done { frames, next_us, .. }) = serde_json::from_str::<Reply>(&done) else { panic!("{done}") };
    assert_eq!((got.len(), frames, next_us), (10, 10, Some(3_000_000)), "{done}");
    importer.close("done").await.expect("close");
    rig.end().await;
}
