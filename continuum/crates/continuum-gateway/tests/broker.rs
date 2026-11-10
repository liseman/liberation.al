//! The gateway with the broker in the loop: `streams` lists the broker's registrations, and a `range` is a `Query`
//! whose items are read from where the broker says (inline bytes here), then served as frames with their times.

use std::sync::Arc;
use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_client::{Client, ClientOptions, StreamSpec, StreamStorage};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, HeaderSpec, StorageLocator, StreamId, Uuid};
use continuum_gateway::control::Reply;
use continuum_gateway::header::{PayloadType, split};
use continuum_gateway::{ChunkSource, OriginPolicy, Session, overlay, ws};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

fn jpeg(i: u8) -> Vec<u8> {
    vec![0xff, 0xd8, 0xff, 0xe0, i, 0xff, 0xd9]
}

fn id(n: u8) -> Uuid {
    let mut b = [0u8; 16];
    b[6] = 0x70;
    b[15] = n;
    Uuid(b)
}

/// Native camera orientation is payload metadata, independent of the device's current settings.
fn camera_header(payload: Uuid, stream: u32, at: u64, flip: bool) -> HeaderBuf {
    let mut b = flatbuffers::FlatBufferBuilder::new();
    let shape = b.create_string("MJPEG");
    let capture = fb::CaptureInfo::create(&mut b, &fb::CaptureInfoArgs { flip, ..Default::default() });
    let header = fb::PayloadHeader::create(
        &mut b,
        &fb::PayloadHeaderArgs {
            id: Some(&payload.into()),
            stream_id: stream,
            start_ns: at,
            end_ns: at,
            format: fb::PayloadFormat::EncodedVideo,
            shape_name: Some(shape),
            capture: Some(capture),
            ..Default::default()
        },
    );
    fb::finish_payload_header_buffer(&mut b, header);
    HeaderBuf::new(b.finished_data().to_vec()).expect("camera header")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn streams_and_ranges_come_from_the_broker_when_it_is_in_the_loop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("broker.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.expect("broker");
    let core = server.core();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));

    // A producer registers cam0 and commits three small JPEG frames inline (a client with no shared memory).
    let producer = Client::connect(&socket, ClientOptions::new("test-camera")).await.expect("connect");
    let cam = producer
        .register_stream(&StreamSpec {
            name: "cam0".into(),
            kind: fb::StreamKind::Sensor,
            format: fb::PayloadFormat::EncodedVideo,
            shape_name: "MJPEG".into(),
            shape_version: 1,
            nominal_rate_hz: 30.0,
            storage: StreamStorage::Slab { class: 0 },
            anchor: None,
            layer: fb::DomainLayer::Layer1Raw,
        })
        .await
        .expect("register")
        .stream_id;
    for i in 0..3u8 {
        let t = 2_000_000_000 + u64::from(i) * 40_000_000;
        let header = camera_header(id(i + 1), cam, t, i == 1);
        producer.ingest_acked(&header, Some(&StorageLocator::Inline { bytes: jpeg(i) })).await.expect("ingest");
    }

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

    let (mut ws_client, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws")).await.expect("connect");
    let Message::Text(hello) = ws_client.next().await.expect("hello").expect("ok") else { panic!("hello is text") };
    assert!(matches!(serde_json::from_str::<Reply>(&hello), Ok(Reply::Hello { broker: true, .. })), "{hello}");

    ws_client.send(Message::Text(r#"{"type":"streams"}"#.into())).await.expect("send");
    let Message::Text(streams) = ws_client.next().await.expect("streams").expect("ok") else { panic!("text") };
    let Ok(Reply::Streams { streams }) = serde_json::from_str::<Reply>(&streams) else { panic!("{streams}") };
    assert_eq!(streams.len(), 1);
    assert_eq!((streams[0].id, streams[0].name.as_str(), streams[0].live), (cam, "cam0", true));

    let range = format!(r#"{{"type":"range","stream":{cam},"start_us":2000000,"end_us":2100000}}"#);
    ws_client.send(Message::Text(range.into())).await.expect("send");
    let mut frames = Vec::new();
    let done = loop {
        match ws_client.next().await.expect("message").expect("ok") {
            Message::Binary(b) => frames.push(b.to_vec()),
            Message::Text(t) => break t,
            other => panic!("unexpected {other:?}"),
        }
    };
    let Ok(Reply::Done { frames: n, skipped: 0, next_us: None, codecs, .. }) = serde_json::from_str::<Reply>(&done)
    else {
        panic!("{done}")
    };
    assert_eq!((n, codecs), (3, vec!["jpeg".to_owned()]));
    for (k, frame) in frames.iter().enumerate() {
        let (h, body) = split(frame).expect("frame header");
        assert_eq!((h.stream_id, h.payload_type, h.keyframe, h.live), (cam, PayloadType::Jpeg, true, false));
        assert_eq!(h.capture_us, 2_000_000 + k as u64 * 40_000);
        assert_eq!(h.flip, k == 1, "range uses each payload's captured orientation");
        assert_eq!(body, jpeg(k as u8));
    }
    ws_client
        .send(Message::Text(format!(r#"{{"type":"frame","stream":{cam},"at_us":2040000}}"#).into()))
        .await
        .expect("frame request");
    let Message::Binary(frame) = ws_client.next().await.expect("frame").expect("ok") else { panic!("binary frame") };
    assert!(split(&frame).expect("header").0.flip, "Find objects' exact-frame probe retains orientation");
    let Message::Text(reply) = ws_client.next().await.expect("reply").expect("ok") else { panic!("frame reply") };
    assert!(matches!(serde_json::from_str::<Reply>(&reply), Ok(Reply::Frame { capture_us: Some(2_040_000), .. })));

    // A payload the browser can't decode lands between the second and third frames (skipped). A range capped at two
    // frames has the broker's page end on it, short of the cap, so without `next_us` the player would take the range
    // as finished; the gateway continues after the last frame it sent, and the next window has the third frame.
    let t = 2_000_000_000 + 45_000_000;
    let header = HeaderBuf::build(&HeaderSpec::raw(id(9), StreamId(cam), t, t)).expect("header");
    let raw = StorageLocator::Inline { bytes: b"not a frame".to_vec() };
    producer.ingest_acked(&header, Some(&raw)).await.expect("ingest");
    let range = format!(r#"{{"type":"range","stream":{cam},"start_us":2000000,"end_us":2100000,"max_frames":2}}"#);
    ws_client.send(Message::Text(range.into())).await.expect("send");
    let done = loop {
        match ws_client.next().await.expect("message").expect("ok") {
            Message::Binary(_) => {}
            Message::Text(t) => break t,
            other => panic!("unexpected {other:?}"),
        }
    };
    let Ok(Reply::Done { frames: 2, skipped: 1, next_us: Some(next), .. }) = serde_json::from_str::<Reply>(&done)
    else {
        panic!("{done}")
    };
    assert_eq!(next, 2_040_001, "right after the last frame sent");
    let range = format!(r#"{{"type":"range","stream":{cam},"start_us":{next},"end_us":2100000}}"#);
    ws_client.send(Message::Text(range.into())).await.expect("send");
    let done = loop {
        match ws_client.next().await.expect("message").expect("ok") {
            Message::Binary(b) => assert_eq!(split(&b).expect("frame").0.capture_us, 2_080_000),
            Message::Text(t) => break t,
            other => panic!("unexpected {other:?}"),
        }
    };
    assert!(
        matches!(serde_json::from_str::<Reply>(&done), Ok(Reply::Done { frames: 1, skipped: 1, next_us: None, .. })),
        "{done}"
    );
    // Payloads that began before the window and reach into it come first in a page and can fill a one-frame page by
    // themselves; the gateway asks for a larger page rather than handing the player a cursor that goes nowhere.
    for i in 0..3u8 {
        let start = 1_900_000_000 + u64::from(i) * 1_000_000;
        let header =
            HeaderBuf::build(&HeaderSpec::raw(id(20 + i), StreamId(cam), start, 2_500_000_000)).expect("header");
        let raw = StorageLocator::Inline { bytes: b"not a frame either".to_vec() };
        producer.ingest_acked(&header, Some(&raw)).await.expect("ingest");
    }
    let range = format!(r#"{{"type":"range","stream":{cam},"start_us":2000000,"end_us":2100000,"max_frames":1}}"#);
    ws_client.send(Message::Text(range.into())).await.expect("send");
    let done = loop {
        match ws_client.next().await.expect("message").expect("ok") {
            Message::Binary(b) => assert_eq!(split(&b).expect("frame").0.capture_us, 2_000_000),
            Message::Text(t) => break t,
            other => panic!("unexpected {other:?}"),
        }
    };
    assert!(
        matches!(serde_json::from_str::<Reply>(&done), Ok(Reply::Done { frames: 1, next_us: Some(2_000_001), .. })),
        "{done}"
    );
    // The window's spans: the frames 40 ms apart and the payloads spanning 1.9 s to 2.5 s are one stretch.
    let spans = format!(r#"{{"type":"spans","stream":{cam},"start_us":1000000,"end_us":3000000}}"#);
    ws_client.send(Message::Text(spans.into())).await.expect("send");
    let Message::Text(reply) = ws_client.next().await.expect("spans").expect("ok") else { panic!("text") };
    let Ok(Reply::Spans { spans, .. }) = serde_json::from_str::<Reply>(&reply) else { panic!("{reply}") };
    assert_eq!(spans, [(1_900_000, 2_500_000)], "{reply}");
    // Asked about a window with a gap in it, it says where each stretch is.
    let spans = format!(r#"{{"type":"spans","stream":{cam},"start_us":1000000,"end_us":3000000,"gap_us":0}}"#);
    ws_client.send(Message::Text(spans.into())).await.expect("send");
    let Message::Text(reply) = ws_client.next().await.expect("spans").expect("ok") else { panic!("text") };
    assert!(matches!(serde_json::from_str::<Reply>(&reply), Ok(Reply::Spans { .. })), "{reply}");
    // Video past the history window ages out: its stretch is no longer playable, and the spans reply says it
    // expired, so a reader doesn't call it unrecorded.
    // The commit clock moves with the writer's clock sample.
    core.upkeep_now().await.expect("upkeep");
    core.sync_writer().await.expect("sync");
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("clock").as_nanos() as u64;
    producer.age_out(&[id(20), id(21), id(22)], now, 0, "video older than the window", "{}").await.expect("age out");
    let spans = format!(r#"{{"type":"spans","stream":{cam},"start_us":1000000,"end_us":3000000,"gap_us":0}}"#);
    ws_client.send(Message::Text(spans.into())).await.expect("send");
    let Message::Text(reply) = ws_client.next().await.expect("spans").expect("ok") else { panic!("text") };
    let Ok(Reply::Spans { expired, .. }) = serde_json::from_str::<Reply>(&reply) else { panic!("{reply}") };
    assert_eq!(expired, [(1_900_000, 2_500_000)], "{reply}");
    // More aged-out runs than the broker answers in one page are all listed.
    let many: Vec<Uuid> = (0..1_001u32)
        .map(|i| {
            let mut b = [0u8; 16];
            b[6] = 0x70;
            b[12..].copy_from_slice(&(100_000 + i).to_be_bytes());
            Uuid(b)
        })
        .collect();
    for (i, payload) in many.iter().enumerate() {
        let start = 2_600_000_000 + i as u64 * 300_000;
        let header =
            HeaderBuf::build(&HeaderSpec::raw(*payload, StreamId(cam), start, start + 100_000)).expect("header");
        producer.ingest_acked(&header, Some(&StorageLocator::Inline { bytes: b"old".to_vec() })).await.expect("ingest");
    }
    core.upkeep_now().await.expect("upkeep");
    core.sync_writer().await.expect("sync");
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("clock").as_nanos() as u64;
    producer.age_out(&many, now, 0, "video older than the window", "{}").await.expect("age out");
    let spans = format!(r#"{{"type":"spans","stream":{cam},"start_us":1000000,"end_us":3000000,"gap_us":0}}"#);
    ws_client.send(Message::Text(spans.into())).await.expect("send");
    let Message::Text(reply) = ws_client.next().await.expect("spans").expect("ok") else { panic!("text") };
    let Ok(Reply::Spans { expired, .. }) = serde_json::from_str::<Reply>(&reply) else { panic!("{reply}") };
    assert_eq!(expired.len(), 1 + 1_001, "every aged-out run, past the broker's page");
    assert_eq!(expired.last(), Some(&(2_600_000 + 1_000 * 300, 2_600_000 + 1_000 * 300 + 100)));

    // A stream the broker doesn't know is an error, not a fallback.
    ws_client
        .send(Message::Text(r#"{"type":"range","stream":99,"start_us":1,"end_us":2}"#.into()))
        .await
        .expect("send");
    let Message::Text(err) = ws_client.next().await.expect("error").expect("ok") else { panic!("text") };
    assert!(
        matches!(serde_json::from_str::<Reply>(&err), Ok(Reply::Error { message }) if message.contains("no stream 99")),
        "{err}"
    );

    drop(ws_client);
    serving.abort();
    producer.close("done").await.expect("close");
    let _ = stop.send(());
    broker.await.expect("broker task");
}

fn spec(name: &str, kind: fb::StreamKind, shape: &str, layer: fb::DomainLayer) -> StreamSpec {
    StreamSpec {
        name: name.into(),
        kind,
        format: fb::PayloadFormat::EncodedVideo,
        shape_name: shape.into(),
        shape_version: 1,
        nominal_rate_hz: 30.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_live_session_follows_a_camera_and_its_overlay_stream_until_stopped() {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("broker.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.expect("broker");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let producer = Client::connect(&socket, ClientOptions::new("test-camera")).await.expect("connect");
    let cam = producer
        .register_stream(&spec("cam0", fb::StreamKind::Sensor, "MJPEG", fb::DomainLayer::Layer1Raw))
        .await
        .expect("register")
        .stream_id;
    let boxes = producer
        .register_stream(&spec("cam0.boxes", fb::StreamKind::Derived, "overlay", fb::DomainLayer::Layer1Raw))
        .await
        .expect("register")
        .stream_id;

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
    page.next().await.expect("hello").expect("ok");
    // No `streams` first: the names the live request needs are listed on the way (and never by blocking the runtime).
    let live = format!(r#"{{"type":"live","stream":{cam},"overlays":["cam0.boxes"]}}"#);
    page.send(Message::Text(live.into())).await.expect("send");
    let Message::Text(opened) = page.next().await.expect("live").expect("ok") else { panic!("text") };
    let Ok(Reply::Live { stream, overlays }) = serde_json::from_str::<Reply>(&opened) else { panic!("{opened}") };
    assert_eq!(stream, cam);
    assert_eq!((overlays[0].name.as_str(), overlays[0].id), ("cam0.boxes", boxes));

    // Frames and an overlay land after the subscription: each reaches the page with its capture time.
    let t0 = 3_000_000_000u64;
    for i in 0..2u8 {
        let t = t0 + u64::from(i) * 40_000_000;
        let header = camera_header(id(10 + i), cam, t, i == 0);
        producer.ingest_acked(&header, Some(&StorageLocator::Inline { bytes: jpeg(i) })).await.expect("ingest");
    }
    let mut ov = HeaderSpec::raw(id(20), StreamId(boxes), t0, t0);
    ov.shape_name = "overlay".into();
    let ov_header = HeaderBuf::build(&ov).expect("header");
    let ov_bytes = overlay::encode(&overlay::Overlay {
        capture_us: t0 / 1_000,
        captions: vec![("an overlay".to_owned(), String::new())],
        ..overlay::Overlay::default()
    });
    producer.ingest_acked(&ov_header, Some(&StorageLocator::Inline { bytes: ov_bytes.clone() })).await.expect("ingest");
    let mut got = Vec::new();
    while got.len() < 3 {
        match tokio::time::timeout(Duration::from_secs(5), page.next()).await.expect("within 5 s").expect("message") {
            Ok(Message::Binary(b)) => got.push(b.to_vec()),
            Ok(other) => panic!("unexpected {other:?}"),
            Err(e) => panic!("{e}"),
        }
    }
    let (h0, b0) = split(&got[0]).expect("frame");
    let (h1, b1) = split(&got[1]).expect("frame");
    let (h2, b2) = split(&got[2]).expect("overlay");
    assert_eq!((h0.stream_id, h0.payload_type, h0.capture_us, b0), (cam, PayloadType::Jpeg, t0 / 1_000, &jpeg(0)[..]));
    assert!(h0.live && h1.live && h2.live, "live frames and overlays carry the live flag");
    assert!(h0.flip && !h1.flip, "live pictures keep their own captured orientation");
    assert!(!h2.flip, "vision overlays already use upright coordinates");
    assert_eq!((h1.payload_type, h1.capture_us, b1), (PayloadType::Jpeg, t0 / 1_000 + 40_000, &jpeg(1)[..]));
    assert_eq!((h2.stream_id, h2.payload_type, h2.capture_us), (boxes, PayloadType::Overlay, t0 / 1_000));
    assert_eq!(overlay::decode(b2).expect("overlay").capture_us, t0 / 1_000);

    page.send(Message::Text(r#"{"type":"stop"}"#.into())).await.expect("send");
    let Message::Text(done) = page.next().await.expect("stopped").expect("ok") else { panic!("text") };
    assert!(
        matches!(
            serde_json::from_str::<Reply>(&done),
            Ok(Reply::Stopped { frames: 3, dropped: 0, skipped: 0, reason: None })
        ),
        "{done}"
    );
    page.send(Message::Text(r#"{"type":"stop"}"#.into())).await.expect("send");
    let Message::Text(err) = page.next().await.expect("error").expect("ok") else { panic!("text") };
    assert!(matches!(serde_json::from_str::<Reply>(&err), Ok(Reply::Error { .. })), "{err}");
    // A frame after the stop reaches nobody: the next reply is the pong.
    let header = HeaderBuf::build(&HeaderSpec::raw(id(30), StreamId(cam), t0 + 1_000_000_000, t0 + 1_000_000_000))
        .expect("header");
    producer.ingest_acked(&header, Some(&StorageLocator::Inline { bytes: jpeg(9) })).await.expect("ingest");
    page.send(Message::Text(r#"{"type":"ping"}"#.into())).await.expect("send");
    let Message::Text(pong) = page.next().await.expect("pong").expect("ok") else { panic!("text") };
    assert!(matches!(serde_json::from_str::<Reply>(&pong), Ok(Reply::Pong)), "{pong}");

    drop(page);
    serving.abort();
    producer.close("done").await.expect("close");
    let _ = stop.send(());
    broker.await.expect("broker task");
}

/// The `camera.objects` columns the gateway reads (segment_log.py declares the rest).
fn objects_schema() -> arrow_schema::SchemaRef {
    use arrow_schema::{DataType, Field, Schema};
    Arc::new(Schema::new(vec![
        Field::new("ts_ns", DataType::Int64, false),
        Field::new("end_ns", DataType::Int64, false),
        Field::new("obj_labels", DataType::Utf8, false),
        Field::new("obj_norm", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
    ]))
}

fn objects_writer(root: &std::path::Path, cam: &str) -> continuum_segments::Writer {
    continuum_segments::Writer::open(root, "camera.objects", cam, objects_schema()).expect("writer")
}

/// One grounding row: a single box labelled `label`.
fn grounding(w: &mut continuum_segments::Writer, at: u64, label: &str) {
    use arrow_array::builder::{Float32Builder, ListBuilder};
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    let mut norm = ListBuilder::new(Float32Builder::new());
    norm.values().append_slice(&[0.1, 0.1, 0.3, 0.4]);
    norm.append(true);
    let at = i64::try_from(at).expect("time");
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![at])),
        Arc::new(Int64Array::from(vec![at])),
        Arc::new(StringArray::from(vec![format!(r#"["{label}"]"#)])),
        Arc::new(norm.finish()),
    ];
    let batch = RecordBatch::try_new(objects_schema(), columns).expect("batch");
    w.append(&batch).expect("append");
    w.sync().expect("sync");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_live_session_shows_the_objects_standing_in_view_then_follows_new_groundings() {
    // camera.objects is in the segment store. A page that starts following shows the objects in view now,
    // from the camera's newest grounding in the last minute, then each grounding written after it opened.
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("broker.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.expect("broker");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let producer = Client::connect(&socket, ClientOptions::new("test-camera")).await.expect("connect");
    let cam = producer
        .register_stream(&spec("cam0.video", fb::StreamKind::Sensor, "MJPEG", fb::DomainLayer::Layer1Raw))
        .await
        .expect("register")
        .stream_id;
    let segments = dir.path().join("segments");
    let now_ns =
        u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()).unwrap();
    // One too old to stand, an older one, the newest for cam0 (20 s ago), and a newer one for another camera.
    let mut cam0 = objects_writer(&segments, "cam0");
    let mut cam1 = objects_writer(&segments, "cam1");
    grounding(&mut cam0, now_ns - 300_000_000_000, "chair");
    grounding(&mut cam0, now_ns - 50_000_000_000, "kettle");
    grounding(&mut cam0, now_ns - 20_000_000_000, "mug");
    grounding(&mut cam1, now_ns - 10_000_000_000, "lamp");

    let gateway = Client::connect(&socket, ClientOptions::new("continuum-gateway")).await.expect("connect");
    let chunks = dir.path().join("tier");
    std::fs::create_dir_all(&chunks).expect("mkdir");
    let session = Arc::new(
        Session::new(Arc::new(ChunkSource::new(&chunks)), None, OriginPolicy::default())
            .with_broker(socket.clone(), Some(Arc::new(gateway)))
            .with_segments(segments.clone()),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let serving = tokio::spawn(ws::serve(listener, ws::router(session)));
    let (mut page, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws")).await.expect("connect");
    page.next().await.expect("hello").expect("ok");
    let live = format!(r#"{{"type":"live","stream":{cam},"overlays":["faces","objects"]}}"#);
    page.send(Message::Text(live.into())).await.expect("send");
    let Message::Text(opened) = page.next().await.expect("live").expect("ok") else { panic!("text") };
    assert!(matches!(serde_json::from_str::<Reply>(&opened), Ok(Reply::Live { .. })), "{opened}");
    let overlay_of = async |page: &mut _| {
        let next = tokio::time::timeout(Duration::from_secs(5), StreamExt::next(page)).await.expect("within 5 s");
        let Some(Ok(Message::Binary(b))) = next else { panic!("an overlay") };
        let (h, body) = split(&b).expect("overlay");
        assert_eq!((h.stream_id, h.payload_type), (cam, PayloadType::Overlay));
        let o = overlay::decode(body).expect("overlay");
        (h.capture_us, o.boxes.iter().map(|b| b.label.clone()).collect::<Vec<_>>())
    };
    assert_eq!(overlay_of(&mut page).await, ((now_ns - 20_000_000_000) / 1_000, vec!["mug".to_owned()]));

    // Written while the page follows: it arrives live; another camera's doesn't.
    tokio::time::sleep(Duration::from_millis(300)).await;
    grounding(&mut cam1, now_ns + 1_000_000_000, "sofa");
    grounding(&mut cam0, now_ns + 2_000_000_000, "cup");
    assert_eq!(overlay_of(&mut page).await, ((now_ns + 2_000_000_000) / 1_000, vec!["cup".to_owned()]));

    serving.abort();
    drop(producer);
    stop.send(()).expect("stop");
    broker.await.expect("broker");
}

fn now_ns() -> u64 {
    u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("clock").as_nanos())
        .expect("ns")
}

/// One connection watches two cameras, one at every frame and one at 1 fps, beside a request with a `req`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_watch_follows_many_cameras_on_one_connection_each_at_its_own_rate() {
    use continuum_gateway::header::FLAG_WATCH;

    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("broker.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.expect("broker");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let producer = Client::connect(&socket, ClientOptions::new("test-camera")).await.expect("connect");
    let register = |name: &'static str| {
        let producer = &producer;
        async move {
            producer
                .register_stream(&spec(name, fb::StreamKind::Sensor, "MJPEG", fb::DomainLayer::Layer1Raw))
                .await
                .expect("register")
                .stream_id
        }
    };
    let (every, slow) = (register("cam0").await, register("cam1").await);
    let ingest = |stream: u32, n: u8, at: u64| {
        let producer = &producer;
        async move {
            let header = camera_header(id(n), stream, at, false);
            producer.ingest_acked(&header, Some(&StorageLocator::Inline { bytes: jpeg(n) })).await.expect("ingest");
        }
    };
    // cam1 was seen a moment ago: its tile paints from that at once.
    let before = now_ns() - 1_000_000_000;
    ingest(slow, 1, before).await;

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
    assert!(hello.contains(r#""features":["watch","req","streams_req"]"#), "{hello}");

    let watch = format!(r#"{{"type":"watch","streams":[{{"id":{every},"max_fps":0}},{slow}],"max_fps":1}}"#);
    page.send(Message::Text(watch.into())).await.expect("send");
    let Message::Text(watching) = page.next().await.expect("watching").expect("ok") else { panic!("text") };
    let Ok(Reply::Watching { streams }) = serde_json::from_str::<Reply>(&watching) else { panic!("{watching}") };
    assert_eq!(streams.iter().map(|s| (s.id, s.max_fps)).collect::<Vec<_>>(), [(every, 0.0), (slow, 1.0)]);

    let mut frames: Vec<(u32, u64, u8)> = Vec::new();
    let mut texts: Vec<serde_json::Value> = Vec::new();
    let read = async |page: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
                      frames: &mut Vec<(u32, u64, u8)>,
                      texts: &mut Vec<serde_json::Value>,
                      for_: Duration| {
        let until = tokio::time::Instant::now() + for_;
        while let Ok(Some(Ok(msg))) = tokio::time::timeout_at(until, page.next()).await {
            match msg {
                Message::Binary(b) => {
                    let (h, payload) = split(&b).expect("frame");
                    assert_ne!(b[5] & FLAG_WATCH, 0, "a watch frame says so");
                    assert!(h.live, "and is live");
                    frames.push((h.stream_id, h.capture_us, payload[4]));
                }
                Message::Text(t) => texts.push(serde_json::from_str(&t).expect("json")),
                other => panic!("unexpected {other:?}"),
            }
        }
    };
    read(&mut page, &mut frames, &mut texts, Duration::from_millis(500)).await;
    assert_eq!(frames, [(slow, before / 1_000, 1)], "cam1's tile at once; cam0 had nothing recent");

    let t0 = now_ns();
    for i in 0..3u8 {
        ingest(every, 10 + i, t0 + u64::from(i) * 10_000_000).await;
        ingest(slow, 20 + i, t0 + u64::from(i) * 10_000_000).await;
    }
    page.send(Message::Text(format!(r#"{{"type":"spans","stream":{every},"start_us":0,"end_us":1,"req":5}}"#).into()))
        .await
        .expect("send");
    frames.clear();
    read(&mut page, &mut frames, &mut texts, Duration::from_millis(1_500)).await;
    let of = |s: u32| frames.iter().filter(|f| f.0 == s).map(|f| f.2).collect::<Vec<_>>();
    assert_eq!(of(every), [10, 11, 12], "every frame of the camera at 0");
    assert_eq!(of(slow), [22], "the camera at 1 fps sends only its newest, a second after its first");
    assert!(texts.iter().any(|t| t["type"] == "spans" && t["req"] == 5), "{texts:?}");

    page.send(Message::Text(r#"{"type":"unwatch"}"#.into())).await.expect("send");
    let Message::Text(done) = page.next().await.expect("unwatched").expect("ok") else { panic!("text") };
    assert!(
        matches!(serde_json::from_str::<Reply>(&done), Ok(Reply::Unwatched { frames: 5, reason: None, .. })),
        "{done}"
    );

    drop(page);
    serving.abort();
    producer.close("done").await.expect("close");
    let _ = stop.send(());
    broker.await.expect("broker task");
}
