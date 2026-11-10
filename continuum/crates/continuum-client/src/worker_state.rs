//! A durable worker's checkpoint as one keyed row in the broker's catalog: [`Client::save_worker_state`],
//! [`Client::load_worker_state`], and [`Client::forget_worker_state`]. They need broker schema minor
//! [`WORKER_STATE_SCHEMA_MINOR`].

use continuum_core::schema::continuum as fb;
use flatbuffers::FlatBufferBuilder;

use crate::messages::finish;
use crate::{Client, Error, unexpected};

/// The broker schema minor that keeps worker state.
pub const WORKER_STATE_SCHEMA_MINOR: u16 = 32;

/// One worker's checkpoint: where its input cursor stood, when it was made, each input stream's name and the
/// invalidation revision it was read under (sorted, unique, 1 to 64), and the worker's encoded state (at most 8 MiB).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerState {
    /// The worker.
    pub owner: String,
    /// Which of its checkpoint formats and settings this is.
    pub contract: String,
    /// The input position the state includes.
    pub position: u64,
    /// When it was made (UTC ns).
    pub created_ns: u64,
    /// Each input stream and the invalidation revision the state was read under.
    pub inputs: Vec<(String, u64)>,
    /// The worker's encoded state.
    pub body: Vec<u8>,
}

impl Client {
    fn needs_worker_state(&self) -> Result<(), Error> {
        let minor = self.broker_schema().1;
        if minor < WORKER_STATE_SCHEMA_MINOR {
            return Err(Error::Protocol(format!(
                "the broker speaks schema minor {minor}; worker state needs {WORKER_STATE_SCHEMA_MINOR}"
            )));
        }
        Ok(())
    }

    /// Stores a worker's checkpoint in place of its last, only if every input's invalidation revision still equals
    /// the one given (checked in the broker's write that stores it). A purge that later advances one of them deletes
    /// it.
    ///
    /// # Errors
    ///
    /// A Conflict broker error when an input's revision moved or it isn't registered; BadRequest when out of bounds;
    /// a Protocol error, sent nothing, for a broker below [`WORKER_STATE_SCHEMA_MINOR`]; and as any request.
    pub async fn save_worker_state(&self, state: &WorkerState) -> Result<(), Error> {
        self.needs_worker_state()?;
        let reply = self
            .shared
            .request(|id| {
                let mut b = FlatBufferBuilder::new();
                let owner = b.create_string(&state.owner);
                let contract = b.create_string(&state.contract);
                let inputs: Vec<_> = state
                    .inputs
                    .iter()
                    .map(|(stream, revision)| {
                        let stream = b.create_string(stream);
                        fb::WorkerStateInput::create(
                            &mut b,
                            &fb::WorkerStateInputArgs { stream: Some(stream), revision: *revision },
                        )
                    })
                    .collect();
                let inputs = b.create_vector(&inputs);
                let body = b.create_vector(&state.body);
                let r = fb::SaveWorkerState::create(
                    &mut b,
                    &fb::SaveWorkerStateArgs {
                        owner: Some(owner),
                        contract: Some(contract),
                        position: state.position,
                        created_ns: state.created_ns,
                        inputs: Some(inputs),
                        body: Some(body),
                    },
                );
                finish(&mut b, id, fb::ControlPayload::SaveWorkerState, r.as_union_value())
            })
            .await?;
        reply.envelope().payload_as_worker_state_saved().ok_or_else(|| unexpected("WorkerStateSaved", &reply))?;
        Ok(())
    }

    /// A worker's checkpoint, if it has one whose inputs' revisions all still equal the stored ones.
    ///
    /// # Errors
    ///
    /// A Protocol error, sent nothing, for a broker below [`WORKER_STATE_SCHEMA_MINOR`]; and as any request.
    pub async fn load_worker_state(&self, owner: &str, contract: &str) -> Result<Option<WorkerState>, Error> {
        self.needs_worker_state()?;
        let reply = self
            .shared
            .request(|id| {
                let mut b = FlatBufferBuilder::new();
                let (o, c) = (b.create_string(owner), b.create_string(contract));
                let r =
                    fb::LoadWorkerState::create(&mut b, &fb::LoadWorkerStateArgs { owner: Some(o), contract: Some(c) });
                finish(&mut b, id, fb::ControlPayload::LoadWorkerState, r.as_union_value())
            })
            .await?;
        let r =
            reply.envelope().payload_as_worker_state_loaded().ok_or_else(|| unexpected("WorkerStateLoaded", &reply))?;
        if !r.found() {
            return Ok(None);
        }
        let inputs = r
            .inputs()
            .map(|v| v.iter().map(|i| (i.stream().unwrap_or_default().to_owned(), i.revision())).collect())
            .unwrap_or_default();
        Ok(Some(WorkerState {
            owner: owner.to_owned(),
            contract: contract.to_owned(),
            position: r.position(),
            created_ns: r.created_ns(),
            inputs,
            body: r.body().map(|b| b.bytes().to_vec()).unwrap_or_default(),
        }))
    }

    /// Deletes a worker's checkpoint; whether it had one.
    ///
    /// # Errors
    ///
    /// A Protocol error, sent nothing, for a broker below [`WORKER_STATE_SCHEMA_MINOR`]; and as any request.
    pub async fn forget_worker_state(&self, owner: &str, contract: &str) -> Result<bool, Error> {
        self.needs_worker_state()?;
        let reply = self
            .shared
            .request(|id| {
                let mut b = FlatBufferBuilder::new();
                let (o, c) = (b.create_string(owner), b.create_string(contract));
                let r = fb::ForgetWorkerState::create(
                    &mut b,
                    &fb::ForgetWorkerStateArgs { owner: Some(o), contract: Some(c) },
                );
                finish(&mut b, id, fb::ControlPayload::ForgetWorkerState, r.as_union_value())
            })
            .await?;
        let r = reply
            .envelope()
            .payload_as_worker_state_forgotten()
            .ok_or_else(|| unexpected("WorkerStateForgotten", &reply))?;
        Ok(r.removed())
    }
}
