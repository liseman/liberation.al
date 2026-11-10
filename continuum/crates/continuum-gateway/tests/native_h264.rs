//! Synthetic native H264 through a local broker: catalog presentation ordering must never become decode ordering.
use std::sync::Arc;
use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_client::{Client, ClientOptions, StreamSpec, StreamStorage};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, StorageLocator, Uuid};
use continuum_gateway::header::split;
use continuum_gateway::session::Outgoing;
use continuum_gateway::{ChunkSource, OriginPolicy, Session};

fn header(stream: u32, index: usize, seq: u32, pts: u64) -> HeaderBuf {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[8] = 0x80;
    id[12..].copy_from_slice(&(index as u32 + 1).to_be_bytes());
    let mut b = flatbuffers::FlatBufferBuilder::new();
    let shape = b.create_string("H264AnnexB");
    let capture =
        fb::CaptureInfo::create(&mut b, &fb::CaptureInfoArgs { device_seq: u64::from(seq), ..Default::default() });
    let h = fb::PayloadHeader::create(
        &mut b,
        &fb::PayloadHeaderArgs {
            id: Some(&Uuid(id).into()),
            stream_id: stream,
            start_ns: pts,
            end_ns: pts,
            seq,
            format: fb::PayloadFormat::EncodedVideo,
            shape_name: Some(shape),
            capture: Some(capture),
            ..Default::default()
        },
    );
    fb::finish_payload_header_buffer(&mut b, h);
    HeaderBuf::new(b.finished_data().to_vec()).expect("native header")
}

async fn ranges(points: &[(u32, u64, bool)], requests: &[(u64, u64, usize)]) -> Vec<Vec<Outgoing>> {
    ranges_with_archive(points, requests, false).await
}

async fn ranges_with_archive(
    points: &[(u32, u64, bool)],
    requests: &[(u64, u64, usize)],
    archived: bool,
) -> Vec<Vec<Outgoing>> {
    requests_with_archive(points, requests, archived, None, None).await
}

async fn requests_with_archive(
    points: &[(u32, u64, bool)],
    requests: &[(u64, u64, usize)],
    archived: bool,
    frame_at: Option<u64>,
    after_us: Option<u64>,
) -> Vec<Vec<Outgoing>> {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("broker.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.expect("server");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let producer = Client::connect(&socket, ClientOptions::new("synthetic-native-camera")).await.expect("producer");
    let stream = producer
        .register_stream(&StreamSpec {
            name: "native-test-camera".into(),
            kind: fb::StreamKind::Sensor,
            format: fb::PayloadFormat::EncodedVideo,
            shape_name: "H264AnnexB".into(),
            shape_version: 1,
            nominal_rate_hz: 30.0,
            storage: StreamStorage::Slab { class: 0 },
            anchor: None,
            layer: fb::DomainLayer::Layer1Raw,
        })
        .await
        .expect("register")
        .stream_id;
    for (index, &(seq, pts, key)) in points.iter().enumerate() {
        let bytes = if key { vec![0, 0, 1, 0x67, 0x4d, 0, 0, 1, 0x65, 0x88] } else { vec![0, 0, 1, 0x41, 0x88] };
        producer
            .ingest_acked(&header(stream, index, seq, pts), Some(&StorageLocator::Inline { bytes }))
            .await
            .expect("ingest");
    }
    if archived {
        use continuum_client::{ChunkFrame, NewChunk, WorkerSpec};
        use continuum_core::{HeaderSpec, StorageTier, StreamId};
        use continuum_tier::{ChunkFormat, ChunkWriter, gop};
        producer
            .register_worker(&WorkerSpec {
                worker_id: "synthetic-archive".into(),
                version_hash: "test".into(),
                capabilities: vec![],
                device_index: None,
                keeps_copies: true,
            })
            .await
            .unwrap();
        let path = dir.path().join("archived.cnk");
        // A raw camera's Tier 1: one access unit per record, in decode order.
        let mut writer = ChunkWriter::create(&path, StreamId(stream), ChunkFormat::H264Au, 0).unwrap();
        let mut logical = Vec::new();
        for (index, &(seq, pts, key)) in points.iter().enumerate() {
            let unit = gop::Unit {
                start_ns: pts,
                end_ns: pts,
                key,
                bytes: if key { vec![0, 0, 1, 0x67, 0x4d, 0, 0, 1, 0x65, 0x88] } else { vec![0, 0, 1, 0x41, 0x88] },
            };
            let record = writer.append(pts, pts, seq, &gop::encode(&[unit]).unwrap()).unwrap();
            logical.push((
                index,
                ChunkFrame { start_ns: pts, offset: record.offset, length: u64::from(record.len), cut: false },
            ));
        }
        writer.close().unwrap();
        logical.sort_by_key(|(_, frame)| frame.start_ns);
        let start = points.iter().map(|(_, pts, _)| *pts).min().unwrap();
        let end = points.iter().map(|(_, pts, _)| *pts).max().unwrap() + 1;
        let mut spec = HeaderSpec::raw(Uuid([99; 16]), StreamId(stream), start, end);
        spec.shape_name = "H264AnnexB".into();
        spec.format = fb::PayloadFormat::EncodedVideo;
        let absorbed: Vec<_> =
            logical.iter().map(|(index, _)| header(stream, *index, points[*index].0, points[*index].1).id()).collect();
        producer
            .commit_chunk(
                &NewChunk {
                    header: HeaderBuf::build(&spec).unwrap(),
                    location: StorageLocator::Chunk {
                        tier: StorageTier::Tier1Warm,
                        uri: path.to_string_lossy().into_owned(),
                        offset: 0,
                        length: std::fs::metadata(&path).unwrap().len(),
                    },
                    frames: logical.into_iter().map(|(_, frame)| frame).collect(),
                },
                &absorbed,
            )
            .await
            .unwrap();
    }
    let client = Client::connect(&socket, ClientOptions::new("test-gateway")).await.expect("gateway");
    let session = Arc::new(
        Session::new(Arc::new(ChunkSource::new(&dir.path().join("chunks"))), None, OriginPolicy::default())
            .with_broker(socket, Some(Arc::new(client))),
    );
    let mut output = Vec::new();
    let after = after_us.map_or_else(|| "null".to_owned(), |us| us.to_string());
    for &(start_us, end_us, max_frames) in requests {
        let after = after.clone();
        let session = Arc::clone(&session);
        output.push(tokio::task::spawn_blocking(move ||session.handle(&format!(r#"{{"type":"range","stream":{stream},"start_us":{start_us},"end_us":{end_us},"max_frames":{max_frames},"after_us":{after}}}"#))).await.expect("range"));
    }
    if let Some(at_us) = frame_at {
        let session = Arc::clone(&session);
        output.push(
            tokio::task::spawn_blocking(move || {
                session.handle(&format!(r#"{{"type":"frame","stream":{stream},"at_us":{at_us},"media_context":true}}"#))
            })
            .await
            .unwrap(),
        );
    }
    let _ = stop.send(());
    task.await.expect("broker task");
    output
}

fn times(messages: &[Outgoing]) -> Vec<u64> {
    messages
        .iter()
        .filter_map(|m| match m {
            Outgoing::Binary(b) => Some(split(b).expect("frame").0.capture_us),
            _ => None,
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tier0_b_frame_seek_keeps_prefix_tail_decode_order_and_whole_gop_pages() {
    let points = [
        (1, 100_000_000, true),
        (2, 300_000_000, false),
        (3, 200_000_000, false),
        (4, 400_000_000, true),
        (5, 600_000_000, false),
        (6, 500_000_000, false),
        (7, 700_000_000, true),
    ];
    let out = ranges(&points, &[(200_000, 201_000, 1), (0, 650_000, 1), (400_000, 650_000, 1)]).await;
    assert_eq!(times(&out[0]), [100_000, 300_000, 200_000]);
    assert_eq!(times(&out[1]), [100_000, 300_000, 200_000]);
    assert!(matches!(out[1].last(),Some(Outgoing::Text(t)) if t.contains("\"next_us\":400000")));
    assert_eq!(times(&out[2]), [400_000, 600_000, 500_000]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tier0_missing_prefix_tail_or_reference_gap_is_an_error_not_successful_empty() {
    let no_prefix = ranges(&[(2, 200_000, false), (3, 300_000, true)], &[(200, 201, 100)]).await;
    assert!(matches!(no_prefix[0].first(),Some(Outgoing::Text(t)) if t.contains("unavailable")));
    let no_tail = ranges(&[(1, 100_000, true), (2, 200_000, false)], &[(200, 201, 100)]).await;
    assert!(matches!(no_tail[0].first(),Some(Outgoing::Text(t)) if t.contains("complete GOP tail")));
    let gap = ranges(&[(1, 100_000, true), (3, 200_000, false), (4, 300_000, true)], &[(200, 201, 100)]).await;
    assert!(matches!(gap[0].first(),Some(Outgoing::Text(t)) if t.contains("unavailable")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tier0_context_cap_never_certifies_a_partial_requested_window() {
    let points: Vec<_> = (1..=4097u32).map(|seq| (seq, u64::from(seq) * 1000, true)).collect();
    let out = ranges(&points, &[(1, 5000, 600)]).await;
    assert!(times(&out[0]).is_empty());
    assert!(matches!(out[0].first(),Some(Outgoing::Text(t)) if t.contains("bounded catalog context")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn forward_sequence_gap_before_requested_idr_does_not_poison_healthy_new_gop() {
    let out =
        ranges(&[(1, 100_000, true), (5, 200_000, true), (6, 250_000, false), (7, 300_000, true)], &[(200, 251, 100)])
            .await;
    assert_eq!(times(&out[0]), [200, 250]);
    assert!(matches!(out[0].last(),Some(Outgoing::Text(t)) if t.contains("\"type\":\"done\"")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn archived_native_session_interior_seek_matches_original_tier0_decode_chain() {
    let points = [
        (1, 100_000_000, true),
        (2, 300_000_000, false),
        (3, 200_000_000, false),
        (4, 400_000_000, true),
        (5, 600_000_000, false),
        (6, 500_000_000, false),
        (7, 700_000_000, true),
    ];
    let requests = [(200_000, 201_000, 1), (0, 650_000, 1), (400_000, 650_000, 1)];
    let hot = ranges(&points, &requests).await;
    let chunked = ranges_with_archive(&points, &requests, true).await;
    for (original, archived) in hot.iter().zip(&chunked) {
        assert_eq!(
            times(archived),
            times(original),
            "archival must preserve session routing and complete reference chains"
        );
        assert!(matches!(archived.last(), Some(Outgoing::Text(text)) if text.contains("\"type\":\"done\"")));
    }
}

fn independent_idr() -> Vec<u8> {
    // In-band parameter headers plus an actual IDR. The feature-gated test also uses a real encoded picture.
    vec![0, 0, 1, 0x67, 0x42, 0xe0, 0x1f, 0x80, 0, 0, 1, 0x68, 0xce, 0x80, 0, 0, 1, 0x65, 0x88]
}

async fn native_snapshot(points: &[(u64, Option<Vec<u8>>)], at_us: u64) -> (Vec<Outgoing>, Vec<Outgoing>) {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let producer = Client::connect(&socket, ClientOptions::new("snapshot-fixture")).await.unwrap();
    let stream = producer
        .register_stream(&StreamSpec {
            name: "snapshot-camera".into(),
            kind: fb::StreamKind::Sensor,
            format: fb::PayloadFormat::EncodedVideo,
            shape_name: "H264AnnexB".into(),
            shape_version: 1,
            nominal_rate_hz: 30.0,
            storage: StreamStorage::Slab { class: 0 },
            anchor: None,
            layer: fb::DomainLayer::Layer1Raw,
        })
        .await
        .unwrap()
        .stream_id;
    for (index, (pts, bytes)) in points.iter().enumerate() {
        let mut builder = flatbuffers::FlatBufferBuilder::new();
        let id = header(stream, index, index as u32, *pts).id();
        let shape = builder.create_string("H264AnnexB");
        let capture = fb::CaptureInfo::create(
            &mut builder,
            &fb::CaptureInfoArgs { width: 64, height: 96, flip: true, device_seq: index as u64, ..Default::default() },
        );
        let h = fb::PayloadHeader::create(
            &mut builder,
            &fb::PayloadHeaderArgs {
                id: Some(&id.into()),
                stream_id: stream,
                start_ns: *pts,
                end_ns: *pts,
                seq: index as u32,
                format: fb::PayloadFormat::EncodedVideo,
                shape_name: Some(shape),
                capture: Some(capture),
                ..Default::default()
            },
        );
        fb::finish_payload_header_buffer(&mut builder, h);
        let locator = bytes.as_ref().map(|bytes| StorageLocator::Inline { bytes: bytes.clone() });
        producer
            .ingest_acked(&HeaderBuf::new(builder.finished_data().to_vec()).unwrap(), locator.as_ref())
            .await
            .unwrap();
    }
    let client = Client::connect(&socket, ClientOptions::new("snapshot-gateway")).await.unwrap();
    let session = Arc::new(
        Session::new(Arc::new(ChunkSource::new(&dir.path().join("chunks"))), None, OriginPolicy::default())
            .with_broker(socket, Some(Arc::new(client))),
    );
    let s = Arc::clone(&session);
    let frame = tokio::task::spawn_blocking(move || {
        s.handle(&format!(r#"{{"type":"frame","stream":{stream},"at_us":{at_us},"media_context":true}}"#))
    })
    .await
    .unwrap();
    let range = tokio::task::spawn_blocking(move || {
        session.handle(&format!(
            r#"{{"type":"range","stream":{stream},"start_us":0,"end_us":{},"max_frames":100}}"#,
            at_us + 1
        ))
    })
    .await
    .unwrap();
    let _ = stop.send(());
    task.await.unwrap();
    (frame, range)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_native_snapshot_reads_an_independent_idr_without_the_live_gop_tail() {
    let key = independent_idr();
    let (frame, range) = native_snapshot(
        &[
            (100_000_000, Some(key.clone())),
            (400_000_000, Some(vec![0, 0, 1, 0x41, 0x88])),
            (300_000_000, Some(key.clone())),
            (200_000_000, Some(vec![0, 0, 1, 0x41, 0x88])),
        ],
        450_000,
    )
    .await;
    assert_eq!(
        times(&frame),
        [300_000],
        "latest actual IDR at/before the requested time, independent of decode ordering"
    );
    let picture =
        frame.iter().find_map(|m| if let Outgoing::Binary(b) = m { Some(split(b).unwrap()) } else { None }).unwrap();
    assert!(picture.0.keyframe && picture.0.flip);
    assert_eq!(picture.1, key);
    let context = frame
        .iter()
        .filter_map(
            |m| if let Outgoing::Text(t) = m { serde_json::from_str::<serde_json::Value>(t).ok() } else { None },
        )
        .find(|v| v["type"] == "media")
        .unwrap();
    assert_eq!(context["native_geometry"], serde_json::json!({"width":64,"height":96}));
    assert_eq!(context["profile_status"], "absent");
    assert!(matches!(frame.last(),Some(Outgoing::Text(t)) if t.contains("\"capture_us\":300000")));
    assert!(
        matches!(range.first(),Some(Outgoing::Text(t)) if t.contains("complete GOP tail")),
        "recorded range still needs the complete predictive chain"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_native_snapshot_never_invents_a_picture_for_missing_bytes_or_missing_idr() {
    let (missing, _) =
        native_snapshot(&[(100_000_000, Some(independent_idr())), (200_000_000, Some(Vec::new()))], 250_000).await;
    assert!(times(&missing).is_empty());
    assert!(matches!(missing.first(),Some(Outgoing::Text(t)) if t.contains("unavailable")));
    let (delta, _) = native_snapshot(&[(200_000_000, Some(vec![0, 0, 1, 0x41, 0x88]))], 250_000).await;
    assert!(times(&delta).is_empty());
    assert!(matches!(delta.last(),Some(Outgoing::Text(t)) if t.contains("\"capture_us\":null")));
}

#[cfg(feature = "ffmpeg")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_native_snapshot_keyframe_decodes_alone_without_an_unfinished_delta_tail() {
    use continuum_tier::h264::{H264Encoder, H264Settings};
    use continuum_tier::video::H264StreamDecoder;
    let mut encoder =
        H264Encoder::open_with(continuum_tier::h264::SOFTWARE_H264_ENCODERS, 64, 96, H264Settings::default()).unwrap();
    let picture = continuum_tier::hydrate::rgb_to_nv12(&continuum_tier::hydrate::Picture {
        width: 64,
        height: 96,
        rgb: vec![77; 64 * 96 * 3],
    });
    let mut units = encoder.encode(&picture, 100_000_000, 133_333_333).unwrap();
    units.extend(encoder.finish().unwrap());
    let key = units.into_iter().find(|unit| unit.key).unwrap().bytes;
    let (frame, _) =
        native_snapshot(&[(100_000_000, Some(key)), (200_000_000, Some(vec![0, 0, 1, 0x41, 0x88]))], 250_000).await;
    let bytes =
        frame.iter().find_map(|m| if let Outgoing::Binary(b) = m { Some(split(b).unwrap().1) } else { None }).unwrap();
    let mut decoder = H264StreamDecoder::new().unwrap();
    let mut pictures = decoder.push(bytes, 100_000).unwrap();
    pictures.extend(decoder.finish().unwrap());
    assert_eq!(pictures.len(), 1);
    assert_eq!((pictures[0].width, pictures[0].height, pictures[0].tag), (64, 96, 100_000));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_native_snapshot_keeps_the_existing_archive_reader_for_chunk_candidates() {
    let out =
        requests_with_archive(&[(1, 100_000_000, true), (2, 200_000_000, false)], &[], true, Some(250_000), None).await;
    assert_eq!(times(&out[0]), [100_000]);
    assert!(matches!(out[0].last(),Some(Outgoing::Text(t)) if t.contains("\"capture_us\":100000")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn follow_on_ranges_suppress_completed_gops_preserving_b_order_and_cap_cursor() {
    let points = [
        (1, 100_000_000, true),
        (2, 300_000_000, false),
        (3, 200_000_000, false),
        (4, 400_000_000, true),
        (5, 600_000_000, false),
        (6, 500_000_000, false),
        (7, 700_000_000, true),
    ];
    for archived in [false, true] {
        let out = requests_with_archive(
            &points,
            &[(250_000, 650_000, 600), (250_000, 650_000, 1), (400_000, 650_000, 1)],
            archived,
            None,
            Some(300_000),
        )
        .await;
        assert_eq!(times(&out[0]), [400_000, 600_000, 500_000]);
        assert!(times(&out[1]).is_empty(), "the bounded first page contains only a completed GOP");
        assert!(matches!(out[1].last(), Some(Outgoing::Text(t)) if t.contains("\"next_us\":400000")));
        assert_eq!(times(&out[2]), [400_000, 600_000, 500_000]);
        let seek = ranges_with_archive(&points, &[(250_000, 650_000, 600)], archived).await;
        assert_eq!(times(&seek[0]), [100_000, 300_000, 200_000, 400_000, 600_000, 500_000]);
    }
}
