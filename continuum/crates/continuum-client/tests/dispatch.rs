//! Dispatched jobs and purge notices against a stand-in broker: a `DispatchJob` decodes into a typed event, its
//! output lease commits like a quoted one, `reject_job` sends `JobRejected`, a `PurgeNotice` decodes and `ack_purge`
//! sends `PurgeAck`, and a malformed job is reported rather than dropped.

mod common;

use std::time::Duration;

use common::*;
use continuum_client::{BrokerEvent, Client, ClientOptions, DispatchJob, Item, PurgedSpan};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, HeaderSpec, StorageLocator, StreamId, Uuid};
use flatbuffers::FlatBufferBuilder;
use tokio::sync::mpsc;

fn v7(n: u8) -> Uuid {
    let mut id = [n; 16];
    id[6] = 0x70 | (n & 0x0f);
    id[8] = 0x80 | (n & 0x3f);
    Uuid(id)
}

fn header(n: u8) -> HeaderBuf {
    HeaderBuf::build(&HeaderSpec::raw(v7(n), StreamId(3), 100, 200)).unwrap()
}

fn envelope(
    request_id: u64,
    f: impl FnOnce(
        &mut FlatBufferBuilder<'static>,
    ) -> (fb::ControlPayload, flatbuffers::WIPOffset<flatbuffers::UnionWIPOffset>),
) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let (kind, payload) = f(&mut b);
    let env = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs { request_id, payload_type: kind, payload: Some(payload) },
    );
    b.finish(env, None);
    b.finished_data().to_vec()
}

/// A `DispatchJob` with two inputs (one inline, one header only) and output lease 91; `junk` puts bytes that aren't
/// a header in the second input.
fn dispatch(junk: bool) -> Vec<u8> {
    envelope(0, |b| {
        let job = fb::Uuid::new(&v7(1).0);
        let promises = [fb::Uuid::new(&v7(2).0), fb::Uuid::new(&v7(3).0)];
        let promises = b.create_vector(&promises);
        let capability = b.create_string("pose");
        let first = {
            let h = b.create_vector(header(4).as_bytes());
            let l = StorageLocator::Inline { bytes: b"frame".to_vec() }.build_into(b);
            fb::Item::create(b, &fb::ItemArgs { locator: Some(l), header_bytes: Some(h), commit_seq: 0 })
        };
        let second = {
            let bytes = if junk { b"not a header".to_vec() } else { header(5).as_bytes().to_vec() };
            let h = b.create_vector(&bytes);
            fb::Item::create(b, &fb::ItemArgs { locator: None, header_bytes: Some(h), commit_seq: 0 })
        };
        let inputs = b.create_vector(&[first, second]);
        let arena = b.create_string("/cnt.test.1");
        let slot = fb::ShmLocation::create(
            b,
            &fb::ShmLocationArgs {
                arena: Some(arena),
                arena_generation: 1,
                slab_class: 0,
                slot: 2,
                slot_generation: 4,
                offset: 32768,
                length: 16384,
            },
        );
        let slots = b.create_vector(&[slot]);
        let output =
            fb::OutputLease::create(b, &fb::OutputLeaseArgs { lease_id: 91, slots: Some(slots), expires_at_ns: 5_000 });
        let range = fb::TimeRange::new(1_000, 2_000);
        let j = fb::DispatchJob::create(
            b,
            &fb::DispatchJobArgs {
                job_id: Some(&job),
                promise_ids: Some(promises),
                capability: Some(capability),
                inputs: Some(inputs),
                output_range: Some(&range),
                band: fb::PriorityBand::P2Nearline,
                deadline_ns: 9_000,
                output: Some(output),
                capability_id: 6,
            },
        );
        (fb::ControlPayload::DispatchJob, j.as_union_value())
    })
}

/// A one-page notice: purge 7 of payloads 8 and 9, which removed stream 3's 100 to 200.
fn purge_notice() -> Vec<u8> {
    purge_page(7, &[8, 9], &[(3, 100, 200)], false)
}

/// A page of purge `purge`'s notice: these payload ids, these spans (stream, start, end), and whether more follow.
fn purge_page(purge: u8, payloads: &[u8], spans: &[(u32, u64, u64)], more: bool) -> Vec<u8> {
    envelope(0, |b| {
        let purge = fb::Uuid::new(&v7(purge).0);
        let ids: Vec<_> = payloads.iter().map(|&n| fb::Uuid::new(&v7(n).0)).collect();
        let ids = b.create_vector(&ids);
        let spans: Vec<_> = spans
            .iter()
            .map(|&(stream_id, start, end)| {
                let range = fb::TimeRange::new(start, end);
                fb::PurgedSpan::create(b, &fb::PurgedSpanArgs { stream_id, range: Some(&range) })
            })
            .collect();
        let spans = b.create_vector(&spans);
        let p = fb::PurgeNotice::create(
            b,
            &fb::PurgeNoticeArgs { purge_id: Some(&purge), payload_ids: Some(ids), spans: Some(spans), more },
        );
        (fb::ControlPayload::PurgeNotice, p.as_union_value())
    })
}

fn released(request_id: u64, lease_id: u64) -> Vec<u8> {
    envelope(request_id, |b| {
        let r = fb::LeaseReleased::create(b, &fb::LeaseReleasedArgs { lease_id, output_seqs: None });
        (fb::ControlPayload::LeaseReleased, r.as_union_value())
    })
}

/// What the stand-in heard, other than heartbeats.
#[derive(Debug, PartialEq, Eq)]
enum Heard {
    Commit(u64),
    Rejected(Uuid, String),
    PurgeAck(Uuid),
    Other(fb::ControlPayload),
}

/// A stand-in that pushes `pushes` after Welcome, answers a commit with LeaseReleased, and reports what it hears.
async fn stand(socket: &std::path::Path, pushes: Vec<Vec<u8>>) -> mpsc::UnboundedReceiver<Heard> {
    let (tx, rx) = mpsc::unbounded_channel();
    scripted(socket, welcome(&[]), vec![], move |mut s| async move {
        for p in &pushes {
            continuum_wire::write_frame(&mut s, p).await.unwrap();
        }
        let mut buf = Vec::new();
        while continuum_wire::read_frame(&mut s, &mut buf).await.is_ok() {
            let env = continuum_wire::read_envelope(&buf).unwrap();
            let heard = if let Some(c) = env.payload_as_commit_lease() {
                let r = released(env.request_id(), c.lease_id());
                continuum_wire::write_frame(&mut s, &r).await.unwrap();
                Heard::Commit(c.lease_id())
            } else if let Some(j) = env.payload_as_job_rejected() {
                Heard::Rejected(Uuid::from(j.job_id().unwrap()), j.reason().unwrap_or_default().to_owned())
            } else if let Some(a) = env.payload_as_purge_ack() {
                Heard::PurgeAck(Uuid::from(a.purge_id().unwrap()))
            } else if env.payload_type() == fb::ControlPayload::Heartbeat {
                continue;
            } else {
                Heard::Other(env.payload_type())
            };
            let _ = tx.send(heard);
        }
    })
    .await;
    rx
}

async fn next(rx: &mut mpsc::UnboundedReceiver<Heard>) -> Heard {
    tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.unwrap().unwrap()
}

async fn event(c: &Client) -> BrokerEvent {
    tokio::time::timeout(Duration::from_secs(5), c.next_event()).await.unwrap().unwrap().event()
}

#[tokio::test]
async fn a_dispatched_job_decodes_and_its_lease_commits() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let mut heard = stand(&socket, vec![dispatch(false)]).await;
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let BrokerEvent::DispatchJob(mut job) = event(&c).await else { panic!("expected a DispatchJob") };
    let DispatchJob { job_id, promise_ids, capability, capability_id, inputs, output_range, band, deadline_ns, lease } =
        *job.clone();
    assert_eq!(job_id, v7(1));
    assert_eq!(promise_ids, vec![v7(2), v7(3)]);
    assert_eq!((capability.as_str(), capability_id), ("pose", 6));
    assert_eq!(
        inputs,
        vec![
            Item { header: header(4), locator: Some(StorageLocator::Inline { bytes: b"frame".to_vec() }) },
            Item { header: header(5), locator: None },
        ]
    );
    assert_eq!((output_range, band, deadline_ns), ((1_000, 2_000), fb::PriorityBand::P2Nearline, 9_000));
    let lease = lease.unwrap();
    assert_eq!((lease.lease_id, lease.expires_at_ns, lease.slots.len()), (91, 5_000, 1));
    assert_eq!((lease.slots[0].slot, lease.slots[0].slot_generation, lease.slots[0].offset), (2, 4, 32768));

    let work = c.job_lease(&mut job).unwrap();
    assert_eq!((work.id(), work.expires_at_ns(), work.slots().len()), (91, 5_000, 1));
    assert!(matches!(c.job_lease(&mut job), Err(continuum_client::Error::Protocol(_))), "one handle per lease");
    work.commit(&[], None, None).await.unwrap();
    assert_eq!(next(&mut heard).await, Heard::Commit(91));
}

#[tokio::test]
async fn a_job_can_be_turned_down_with_a_reason() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let mut heard = stand(&socket, vec![dispatch(false)]).await;
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let BrokerEvent::DispatchJob(job) = event(&c).await else { panic!("expected a DispatchJob") };
    c.reject_job(job.job_id, "the model isn't loaded").unwrap();
    assert_eq!(next(&mut heard).await, Heard::Rejected(v7(1), "the model isn't loaded".to_owned()));
}

#[tokio::test]
async fn a_purge_notice_decodes_and_is_acknowledged() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let mut heard = stand(&socket, vec![purge_notice()]).await;
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let BrokerEvent::PurgeNotice { purge_id, payload_ids, spans } = event(&c).await else {
        panic!("expected a PurgeNotice")
    };
    assert_eq!((purge_id, payload_ids), (v7(7), vec![v7(8), v7(9)]));
    assert_eq!(spans, vec![PurgedSpan { stream_id: 3, range: (100, 200) }]);
    c.ack_purge(purge_id).unwrap();
    assert_eq!(next(&mut heard).await, Heard::PurgeAck(v7(7)));
}

#[tokio::test]
async fn a_paged_purge_arrives_once_with_every_page_and_another_purge_passes_between() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    // Purge 20 in three pages (spans on the first), with purge 30 (one page) between its pages.
    let pages = vec![
        purge_page(20, &[21, 22], &[(3, 100, 200), (4, 0, 50)], true),
        purge_page(30, &[31], &[(5, 10, 20)], false),
        purge_page(20, &[23], &[], true),
        purge_page(20, &[24, 25], &[], false),
    ];
    let mut heard = stand(&socket, pages).await;
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let BrokerEvent::PurgeNotice { purge_id, payload_ids, .. } = event(&c).await else { panic!("expected purge 30") };
    assert_eq!((purge_id, payload_ids), (v7(30), vec![v7(31)]), "a one-page purge isn't held behind another's pages");
    let BrokerEvent::PurgeNotice { purge_id, payload_ids, spans } = event(&c).await else { panic!("expected 20") };
    assert_eq!(purge_id, v7(20));
    assert_eq!(payload_ids, [21, 22, 23, 24, 25].map(v7).to_vec(), "every page's ids, in order");
    assert_eq!(
        spans,
        vec![PurgedSpan { stream_id: 3, range: (100, 200) }, PurgedSpan { stream_id: 4, range: (0, 50) }]
    );
    let quiet = tokio::time::timeout(Duration::from_millis(100), c.next_event()).await;
    assert!(quiet.is_err(), "nothing else: the pages came as one notice");
    c.ack_purge(v7(30)).unwrap();
    c.ack_purge(v7(20)).unwrap();
    assert_eq!(next(&mut heard).await, Heard::PurgeAck(v7(30)));
    assert_eq!(next(&mut heard).await, Heard::PurgeAck(v7(20)));
}

#[tokio::test]
async fn a_job_with_a_bad_input_is_reported_not_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let _heard = stand(&socket, vec![dispatch(true)]).await;
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let BrokerEvent::Undecodable { kind, reason, job_id, lease_id } = event(&c).await else {
        panic!("expected Undecodable")
    };
    assert_eq!(kind, fb::ControlPayload::DispatchJob);
    assert!(reason.contains("input 1"), "{reason}");
    assert_eq!((job_id, lease_id), (Some(v7(1)), Some(91)), "named, so it can still be rejected");
}

#[tokio::test]
async fn a_job_lease_can_be_made_off_the_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let _heard = stand(&socket, vec![dispatch(false)]).await;
    let c = std::sync::Arc::new(Client::connect(&socket, ClientOptions::new("t")).await.unwrap());
    let BrokerEvent::DispatchJob(mut job) = event(&c).await else { panic!("expected a DispatchJob") };
    // A Python or C caller has no runtime: building the lease and starting its keep-alive must still work.
    let c2 = std::sync::Arc::clone(&c);
    std::thread::spawn(move || {
        let mut lease = c2.job_lease(&mut job).unwrap();
        lease.keep_alive(Duration::from_secs(60), Duration::from_secs(30));
        lease.release();
    })
    .join()
    .unwrap();
}

/// The envelopes the Python tests decode (continuum-py/tests/fixtures), kept equal to what this file builds. Run with
/// `CONTINUUM_BLESS=1` to rewrite them after changing a builder.
#[test]
fn the_python_fixtures_are_current() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../continuum-py/tests/fixtures");
    let bless = std::env::var_os("CONTINUUM_BLESS").is_some();
    for (name, bytes) in [
        ("dispatch_job.bin", dispatch(false)),
        ("dispatch_job_bad_input.bin", dispatch(true)),
        ("purge_notice.bin", purge_notice()),
    ] {
        let path = dir.join(name);
        if bless {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&path, &bytes).unwrap();
        }
        let on_disk = std::fs::read(&path).unwrap_or_default();
        assert!(on_disk == bytes, "{} is stale: rerun with CONTINUUM_BLESS=1", path.display());
    }
}
