//! A scripted broker keeps subscription backlogs independent from numbered request replies.
mod common;

use continuum_client::{Client, ClientOptions, Start, SubscribeSpec};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, HeaderSpec, StreamId, Uuid};
use continuum_wire::{read_envelope, read_frame, write_frame};
use flatbuffers::FlatBufferBuilder;
use std::time::Duration;
use tokio::sync::oneshot;

fn id(n: u64) -> Uuid {
    let mut bytes = [0; 16];
    bytes[..6].copy_from_slice(&n.to_be_bytes()[2..]);
    bytes[6] = 0x70;
    bytes[8] = 0x80;
    Uuid(bytes)
}

fn subscribed(request: u64, subscription: u64) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let value =
        fb::Subscribed::create(&mut b, &fb::SubscribedArgs { subscription_id: subscription, ..Default::default() });
    let envelope = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs {
            request_id: request,
            payload_type: fb::ControlPayload::Subscribed,
            payload: Some(value.as_union_value()),
        },
    );
    b.finish(envelope, None);
    b.finished_data().to_vec()
}

fn event(subscription: u64, position: u64) -> Vec<u8> {
    let header = HeaderBuf::build(&HeaderSpec::raw(id(position), StreamId(1), position, position + 1)).unwrap();
    let mut b = FlatBufferBuilder::new();
    let header = b.create_vector(header.as_bytes());
    let item = fb::Item::create(
        &mut b,
        &fb::ItemArgs { header_bytes: Some(header), commit_seq: position, ..Default::default() },
    );
    let value = fb::StreamEvent::create(
        &mut b,
        &fb::StreamEventArgs { subscription_id: subscription, position, item: Some(item) },
    );
    let envelope = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs {
            request_id: 0,
            payload_type: fb::ControlPayload::StreamEvent,
            payload: Some(value.as_union_value()),
        },
    );
    b.finish(envelope, None);
    b.finished_data().to_vec()
}

fn ingested(request: u64, id: Uuid) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let id = fb::Uuid::new(&id.0);
    let value =
        fb::Ingested::create(&mut b, &fb::IngestedArgs { stream_id: 2, payload_id: Some(&id), commit_seq: 9000 });
    let envelope = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs {
            request_id: request,
            payload_type: fb::ControlPayload::Ingested,
            payload: Some(value.as_union_value()),
        },
    );
    b.finish(envelope, None);
    b.finished_data().to_vec()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_subscription_fails_pending_ingest_closed_instead_of_timing_out() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");

    let (sent, backlog) = oneshot::channel();
    let (end, ended) = oneshot::channel();
    let output = id(9000);
    let script = common::scripted(&socket, common::welcome(&[]), vec![], move |mut stream| async move {
        let mut bytes = Vec::new();
        read_frame(&mut stream, &mut bytes).await.unwrap();
        let env = read_envelope(&bytes).unwrap();
        let subscription = env.payload_as_subscribe().unwrap().subscription_id();
        write_frame(&mut stream, &subscribed(env.request_id(), subscription)).await.unwrap();
        for position in 1..=4096 {
            write_frame(&mut stream, &event(subscription, position)).await.unwrap();
        }
        sent.send(()).unwrap();
        read_frame(&mut stream, &mut bytes).await.unwrap();
        let request = read_envelope(&bytes).unwrap();
        assert!(request.payload_as_ingest_notice().is_some());
        let request_id = request.request_id();
        let _ = write_frame(&mut stream, &event(subscription, 4097)).await;
        // The original deadlock times out even when this reply is written. The fixed client closes promptly.
        let _ = write_frame(&mut stream, &ingested(request_id, output)).await;
        let _ = ended.await;
    })
    .await;
    let mut options = ClientOptions::new("subscription-and-writer");
    options.heartbeat_interval = Duration::from_secs(60);
    options.request_timeout = Duration::from_secs(2);
    let client = Client::connect(&socket, options).await.unwrap();
    let _subscription = client.subscribe(&SubscribeSpec::new(["input"], Start::Time(0))).await.unwrap();
    backlog.await.unwrap();
    let header = HeaderBuf::build(&HeaderSpec::raw(output, StreamId(2), 1, 2)).unwrap();
    let started = std::time::Instant::now();
    let result = client.ingest_acked(&header, None).await;
    let elapsed = started.elapsed();
    let _ = end.send(());
    script.await.unwrap();
    assert!(matches!(result, Err(continuum_client::Error::Closed)), "{result:?}");
    assert!(elapsed < Duration::from_secs(1), "failed after {elapsed:?}, not promptly");
    let diagnostic = client.next_event().await.unwrap();
    assert!(diagnostic.envelope().payload_as_error().unwrap().message().unwrap().contains("bounded queue"));
    assert_eq!(client.pending_requests(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backpressure_reader_keeps_heartbeating_one_way_while_its_queue_is_full() {
    const EVENTS: u64 = 4100; // past the queue, so the reader parks on the next event
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let (sent, backlog) = oneshot::channel();
    let (beats_tx, mut beats) = tokio::sync::mpsc::unbounded_channel();
    let (end, ended) = oneshot::channel::<()>();
    let script = common::scripted(&socket, common::welcome(&[]), vec![], move |mut stream| async move {
        let mut bytes = Vec::new();
        read_frame(&mut stream, &mut bytes).await.unwrap();
        let env = read_envelope(&bytes).unwrap();
        let subscription = env.payload_as_subscribe().unwrap().subscription_id();
        write_frame(&mut stream, &subscribed(env.request_id(), subscription)).await.unwrap();
        for position in 1..=EVENTS {
            write_frame(&mut stream, &event(subscription, position)).await.unwrap();
        }
        sent.send(()).unwrap();
        // Never answer: a heartbeat that waited for its reply would hold the next one back for the request timeout.
        while let Some((id, kind, _, _)) = common::read_request(&mut stream).await {
            if kind == fb::ControlPayload::Heartbeat && beats_tx.send(id).is_err() {
                break;
            }
            if kind == fb::ControlPayload::Goodbye {
                break;
            }
        }
        let _ = ended.await;
    })
    .await;
    let mut options = ClientOptions::new("dedicated-reader");
    options.subscription_overflow = continuum_client::SubscriptionOverflow::Backpressure;
    options.heartbeat_interval = Duration::from_millis(50);
    options.request_timeout = Duration::from_secs(10);
    let client = Client::connect(&socket, options).await.unwrap();
    let mut subscription = client.subscribe(&SubscribeSpec::new(["input"], Start::Time(0))).await.unwrap();
    backlog.await.unwrap();
    let mut ids = Vec::new();
    for _ in 0..3 {
        let id = tokio::time::timeout(Duration::from_secs(2), beats.recv())
            .await
            .expect("heartbeats keep their cadence behind a full queue")
            .unwrap();
        ids.push(id);
    }
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "{ids:?}");
    assert_eq!(client.pending_requests(), 0, "a one-way heartbeat leaves no waiter");
    // Backpressure lost nothing: every event arrives in order once the consumer reads again.
    for expected in 1..=EVENTS {
        match tokio::time::timeout(Duration::from_secs(5), subscription.next()).await.unwrap() {
            Some(continuum_client::SubEvent::Event { position, .. }) => assert_eq!(position, expected),
            other => panic!("expected event {expected}, got {other:?}"),
        }
    }
    drop(beats);
    let _ = end.send(());
    drop(subscription);
    drop(client);
    script.await.unwrap();
}
