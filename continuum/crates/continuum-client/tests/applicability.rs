//! An old broker cannot silently accept missing recording history as an empty window.
mod common;

use common::scripted;
use continuum_client::{Client, ClientOptions, Error};
use continuum_core::StreamId;
use continuum_core::schema::continuum as fb;
use continuum_wire::{read_envelope, read_frame};
use flatbuffers::FlatBufferBuilder;

#[tokio::test]
async fn a_pre28_broker_never_receives_an_applicability_request() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("old.sock");
    let mut b = FlatBufferBuilder::new();
    let schema =
        fb::SchemaVersion::create(&mut b, &fb::SchemaVersionArgs { major: continuum_core::SCHEMA_MAJOR, minor: 27 });
    let session = fb::Uuid([7; 16]);
    let reply = fb::Welcome::create(
        &mut b,
        &fb::WelcomeArgs { schema: Some(schema), session_id: Some(&session), ..Default::default() },
    );
    let env = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs {
            request_id: 0,
            payload_type: fb::ControlPayload::Welcome,
            payload: Some(reply.as_union_value()),
        },
    );
    b.finish(env, None);
    let server = scripted(&socket, b.finished_data().to_vec(), vec![], |mut stream| async move {
        let mut bytes = Vec::new();
        read_frame(&mut stream, &mut bytes).await.unwrap();
        assert_eq!(read_envelope(&bytes).unwrap().payload_type(), fb::ControlPayload::Goodbye);
    })
    .await;
    let mut options = ClientOptions::new("old-broker-proof");
    options.heartbeat_interval = std::time::Duration::from_secs(3600);
    let client = Client::connect(&socket, options).await.unwrap();
    assert!(matches!(
        client.recording_page(StreamId(1), "test.audio", (10, 20), None, None, 1, 1, 1024).await,
        Err(Error::Protocol(_))
    ));
    client.close("done").await.unwrap();
    server.await.unwrap();
}
