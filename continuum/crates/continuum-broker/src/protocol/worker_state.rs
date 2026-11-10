//! The worker state messages (minor 32, #1547): `SaveWorkerState`, `LoadWorkerState`, and `ForgetWorkerState`, a
//! durable worker's checkpoint as one keyed catalog row (continuum-catalog's `worker_state`), read into owned asks
//! and answered from what the catalog said.

use continuum_catalog::WorkerState;
use continuum_core::schema::continuum as fb;
use flatbuffers::FlatBufferBuilder;

use super::finish;

/// One of the three requests, decoded.
#[derive(Debug)]
pub enum Ask {
    /// Store this state in place of the worker's last.
    Save(WorkerState),
    /// Read the worker's state.
    Load {
        /// The worker.
        owner: String,
        /// Its checkpoint contract.
        contract: String,
    },
    /// Delete the worker's state.
    Forget {
        /// The worker.
        owner: String,
        /// Its checkpoint contract.
        contract: String,
    },
}

/// What the catalog answered.
#[derive(Debug)]
pub enum Done {
    /// Stored.
    Saved,
    /// The state, if the worker has one that still holds.
    Loaded(Option<WorkerState>),
    /// Whether it had one.
    Forgotten(bool),
}

fn inputs(
    v: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<fb::WorkerStateInput<'_>>>>,
) -> Option<Vec<(String, u64)>> {
    v.map(|v| v.iter().map(|i| i.stream().map(|s| (s.to_owned(), i.revision()))).collect::<Option<Vec<_>>>())
        .unwrap_or(Some(Vec::new()))
}

/// A `SaveWorkerState`, decoded; None when its owner, contract, or an input's stream is missing (bounds are the
/// catalog's to check).
pub fn save(m: &fb::SaveWorkerState<'_>) -> Option<Ask> {
    Some(Ask::Save(WorkerState {
        owner: m.owner()?.to_owned(),
        contract: m.contract()?.to_owned(),
        position: m.position(),
        created_ns: m.created_ns(),
        inputs: inputs(m.inputs())?,
        body: m.body().map(|b| b.bytes().to_vec()).unwrap_or_default(),
    }))
}

/// A `LoadWorkerState`, decoded.
pub fn load(m: &fb::LoadWorkerState<'_>) -> Option<Ask> {
    Some(Ask::Load { owner: m.owner()?.to_owned(), contract: m.contract()?.to_owned() })
}

/// A `ForgetWorkerState`, decoded.
pub fn forget(m: &fb::ForgetWorkerState<'_>) -> Option<Ask> {
    Some(Ask::Forget { owner: m.owner()?.to_owned(), contract: m.contract()?.to_owned() })
}

/// The reply to one of the three.
pub fn reply(b: &mut FlatBufferBuilder<'static>, request_id: u64, done: &Done) -> Vec<u8> {
    b.reset();
    match done {
        Done::Saved => {
            let r = fb::WorkerStateSaved::create(b, &fb::WorkerStateSavedArgs {});
            finish(b, request_id, fb::ControlPayload::WorkerStateSaved, r.as_union_value()).to_vec()
        }
        Done::Forgotten(removed) => {
            let r = fb::WorkerStateForgotten::create(b, &fb::WorkerStateForgottenArgs { removed: *removed });
            finish(b, request_id, fb::ControlPayload::WorkerStateForgotten, r.as_union_value()).to_vec()
        }
        Done::Loaded(None) => {
            let r = fb::WorkerStateLoaded::create(b, &fb::WorkerStateLoadedArgs { found: false, ..Default::default() });
            finish(b, request_id, fb::ControlPayload::WorkerStateLoaded, r.as_union_value()).to_vec()
        }
        Done::Loaded(Some(state)) => {
            let body = b.create_vector(&state.body);
            let inputs: Vec<_> = state
                .inputs
                .iter()
                .map(|(stream, revision)| {
                    let stream = b.create_string(stream);
                    fb::WorkerStateInput::create(
                        b,
                        &fb::WorkerStateInputArgs { stream: Some(stream), revision: *revision },
                    )
                })
                .collect();
            let inputs = b.create_vector(&inputs);
            let r = fb::WorkerStateLoaded::create(
                b,
                &fb::WorkerStateLoadedArgs {
                    found: true,
                    position: state.position,
                    created_ns: state.created_ns,
                    inputs: Some(inputs),
                    body: Some(body),
                },
            );
            finish(b, request_id, fb::ControlPayload::WorkerStateLoaded, r.as_union_value()).to_vec()
        }
    }
}
